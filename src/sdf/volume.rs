use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use algebrix::{IVec3, Vec3};

use crate::{JobDispatch, PhysicsError, Sequential};

use super::compute::EditSpan;
use super::jobs::{Rebuilt, in_parallel};
use super::{BrickKey, BrickMesh, Edit, EditId, SdfDesc};
use super::{field, nets};

/// Bytes of kept samples a volume holds by default. Past it the bricks edited longest ago let go
/// of theirs, and build from their edits when next edited.
const CACHE_BYTES: usize = 256 << 20;

/// Lattice steps an edit reaches past its shape. There the shape's distance is at least two
/// steps, so cutting the edit off never moves a surface.
const EDIT_APRON: f32 = 2.0;
/// Lattice steps at the bottom no edit can open, so nothing digs out through the floor.
const FLOOR_STEPS: f32 = 2.0;

#[derive(Debug, Clone, Copy)]
pub(super) struct EditEntry {
    pub(super) edit: Edit,
    /// Lattice samples it reaches, inclusive. Whether a sample feels the edit depends on the
    /// sample alone, never on which brick asks.
    pub(super) low: IVec3,
    pub(super) high: IVec3,
}

#[derive(Debug, Clone, Default)]
pub(super) struct Brick {
    revision: u64,
    /// Samples inside material among the ones this brick owns.
    inside: u64,
    mesh: BrickMesh,
    /// Edits that reach it, oldest first.
    edits: Vec<EditId>,
    /// Samples after the base and every edit, before the floor. Kept while the brick has edits,
    /// so the next edit runs alone on top.
    pub(super) values: Option<Arc<Vec<f32>>>,
}

/// Ground as a signed distance field: a base and ordered edits, sampled in sparse bricks. Only
/// bricks near the surface hold triangles. An edit samples again only the bricks it reaches,
/// and each of those only runs the edits that reach it.
#[derive(Debug, Clone)]
pub struct SdfVolume {
    pub(super) desc: SdfDesc,
    pub(super) edits: BTreeMap<EditId, EditEntry>,
    pub(super) next_edit: u32,
    pub(super) bricks: HashMap<BrickKey, Brick>,
    revision: u64,
    cache_bytes: usize,
}

impl SdfVolume {
    /// # Errors
    /// Fails on a bad size, base or material.
    pub fn new(desc: SdfDesc) -> Result<Self, PhysicsError> {
        Self::new_with(desc, &Sequential)
    }

    /// [`Self::new`] with bricks built on the host's workers. The result does not depend on them.
    ///
    /// # Errors
    /// Fails on a bad size, base or material, or a dispatcher that skips jobs.
    pub fn new_with(desc: SdfDesc, dispatch: &dyn JobDispatch) -> Result<Self, PhysicsError> {
        desc.validate()?;
        let mut volume = Self {
            desc,
            edits: BTreeMap::new(),
            next_edit: 0,
            bricks: HashMap::new(),
            revision: 1,
            cache_bytes: CACHE_BYTES,
        };
        let [bx, by, bz] = desc.bricks;
        let mut keys = Vec::with_capacity((bx * by * bz) as usize);
        for z in 0..bz {
            for y in 0..by {
                for x in 0..bx {
                    keys.push(BrickKey([x, y, z]));
                }
            }
        }
        let built = in_parallel(dispatch, &keys, |key| volume.build(*key, &[]))?;
        for item in built {
            if item.inside > 0 || !item.mesh.is_empty() {
                volume.bricks.insert(
                    item.key,
                    Brick {
                        revision: volume.revision,
                        inside: item.inside,
                        mesh: item.mesh,
                        edits: Vec::new(),
                        values: None,
                    },
                );
            }
        }
        Ok(volume)
    }

    pub fn desc(&self) -> &SdfDesc {
        &self.desc
    }

    /// Goes up by one with each committed change.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn edit_count(&self) -> usize {
        self.edits.len()
    }

    /// Most bytes of kept samples to hold. The result of an edit does not depend on it, only how
    /// long the next edit of a brick takes.
    pub fn set_cache_limit(&mut self, bytes: usize) {
        self.cache_bytes = bytes;
        self.trim_cache();
    }

    fn trim_cache(&mut self) {
        let each = self.side().pow(3) * size_of::<f32>();
        let allowed = self.cache_bytes / each;
        let mut kept: Vec<(u64, BrickKey)> = self
            .bricks
            .iter()
            .filter(|(_, brick)| brick.values.is_some())
            .map(|(key, brick)| (brick.revision, *key))
            .collect();
        if kept.len() <= allowed {
            return;
        }
        kept.sort_unstable();
        let extra = kept.len() - allowed;
        for (_, key) in &kept[..extra] {
            if let Some(brick) = self.bricks.get_mut(key) {
                brick.values = None;
            }
        }
    }

    /// Bricks keeping samples for the next edit, and the bytes they hold.
    pub fn cached_bricks(&self) -> (usize, usize) {
        let kept = self
            .bricks
            .values()
            .filter_map(|brick| brick.values.as_ref());
        kept.fold((0, 0), |(count, bytes), values| {
            (count + 1, bytes + values.len() * size_of::<f32>())
        })
    }

    /// Material volume in cubic meters, one lattice sample is one cell of it.
    pub fn material_volume(&self) -> f64 {
        let cell = f64::from(self.desc.voxel).powi(3);
        self.bricks
            .values()
            .map(|brick| brick.inside as f64)
            .sum::<f64>()
            * cell
    }

    /// Bricks with triangles: key, low corner from the volume origin, revision, mesh.
    pub fn surface_bricks(&self) -> impl Iterator<Item = (BrickKey, Vec3, u64, &BrickMesh)> {
        self.bricks
            .iter()
            .filter(|(_, brick)| !brick.mesh.is_empty())
            .map(|(key, brick)| (*key, self.brick_corner(*key), brick.revision, &brick.mesh))
    }

    pub fn brick_mesh(&self, key: BrickKey) -> Option<&BrickMesh> {
        self.bricks.get(&key).map(|brick| &brick.mesh)
    }

    /// Distance to the surface at a point from the volume origin, negative inside material.
    /// Exact on lattice samples, where the bricks read it.
    pub fn distance(&self, at: Vec3) -> f32 {
        let step = self.desc.voxel;
        let (height, slope) = self.desc.base.column(at.x, at.z, step * 0.5);
        let mut distance =
            self.desc
                .base
                .carve(at, height, field::base_distance(height, slope, at.y));
        let sample = IVec3::new(
            (at.x / step).round() as i32,
            (at.y / step).round() as i32,
            (at.z / step).round() as i32,
        );
        for entry in self.edits.values() {
            if inside(sample, entry.low, entry.high) {
                distance = entry.edit.op.apply(distance, entry.edit.shape.distance(at));
            }
        }
        distance
            .min(at.y - FLOOR_STEPS * step)
            .max(self.outside(at))
    }

    /// [`Self::distance`] from the nearest sample of the brick there, so its cost does not grow
    /// with the edits. Exact on lattice samples.
    pub(super) fn quick_distance(&self, at: Vec3) -> f32 {
        let step = self.desc.voxel;
        let sample = IVec3::new(
            (at.x / step).round() as i32,
            (at.y / step).round() as i32,
            (at.z / step).round() as i32,
        );
        let n = self.cells();
        let [bx, by, bz] = self.desc.bricks.map(|count| count as i32);
        let brick = |g: i32, count: i32| g.div_euclid(n).clamp(0, count - 1) as u32;
        let key = BrickKey([
            brick(sample.x, bx),
            brick(sample.y, by),
            brick(sample.z, bz),
        ]);
        let point = Vec3::new(sample.x as f32, sample.y as f32, sample.z as f32) * step;
        let (low, high) = self.brick_samples(key);
        let brick = self.bricks.get(&key);
        if brick.is_some_and(|brick| brick.values.is_none() && !brick.edits.is_empty()) {
            // its samples were let go, the edits are the only record of it
            return self.distance(point);
        }
        let raw = match brick.and_then(|brick| brick.values.as_ref()) {
            Some(values) if inside(sample, low, high) => {
                let side = self.side();
                let local = sample - low;
                values[local.x as usize + side * (local.y as usize + side * local.z as usize)]
            }
            _ => {
                let (height, slope) = self.desc.base.column(point.x, point.z, step * 0.5);
                let base = field::base_distance(height, slope, point.y);
                self.desc.base.carve(point, height, base)
            }
        };
        raw.min(point.y - FLOOR_STEPS * step)
            .max(self.outside(point))
    }

    /// Above zero outside the volume, so the ground closes into a solid with no open edge.
    fn outside(&self, at: Vec3) -> f32 {
        let [bx, by, bz] = self.desc.bricks;
        let size = Vec3::new(bx as f32, by as f32, bz as f32)
            * (self.desc.brick_cells as f32 * self.desc.voxel);
        (-at).max(at - size).max_element()
    }

    fn cells(&self) -> i32 {
        self.desc.brick_cells as i32
    }

    pub(super) fn brick_corner(&self, key: BrickKey) -> Vec3 {
        let [x, y, z] = key.0;
        Vec3::new(x as f32, y as f32, z as f32) * (self.desc.brick_cells as f32 * self.desc.voxel)
    }

    /// Lattice samples a brick reads, inclusive: its own and one more on each side.
    pub(super) fn brick_samples(&self, key: BrickKey) -> (IVec3, IVec3) {
        let n = self.cells();
        let low = IVec3::new(key.0[0] as i32, key.0[1] as i32, key.0[2] as i32) * n;
        (low - IVec3::ONE, low + IVec3::splat(n))
    }

    pub(super) fn entry(&self, edit: Edit) -> EditEntry {
        let (low, high) = edit.shape.bounds();
        let step = self.desc.voxel;
        let reach = Vec3::splat(EDIT_APRON * step);
        let lattice = |at: Vec3, round: fn(f32) -> f32| {
            let scaled = at / step;
            let to_int = |value: f32| round(value).clamp(-1.0e9, 1.0e9) as i32;
            IVec3::new(to_int(scaled.x), to_int(scaled.y), to_int(scaled.z))
        };
        EditEntry {
            edit,
            low: lattice(low - reach, f32::floor),
            high: lattice(high + reach, f32::ceil),
        }
    }

    /// Bricks whose samples an edit reaches.
    pub(super) fn reached(&self, entry: &EditEntry) -> Vec<BrickKey> {
        let n = self.cells();
        let [bx, by, bz] = self.desc.bricks.map(|count| count as i32);
        // brick b reads samples n b - 1 to n b + n
        let first = |low: i32| (low - n).div_euclid(n).max(0);
        let last = |high: i32, count: i32| (high + 1).div_euclid(n).min(count - 1);
        let mut keys = Vec::new();
        for z in first(entry.low.z)..=last(entry.high.z, bz) {
            for y in first(entry.low.y)..=last(entry.high.y, by) {
                for x in first(entry.low.x)..=last(entry.high.x, bx) {
                    let key = BrickKey([x as u32, y as u32, z as u32]);
                    let (low, high) = self.brick_samples(key);
                    if overlaps(low, high, entry.low, entry.high) {
                        keys.push(key);
                    }
                }
            }
        }
        keys
    }

    /// The brick's edits oldest first, with `pending` after them or `skip` left out.
    pub(super) fn entries_of(
        &self,
        key: BrickKey,
        pending: Option<&EditEntry>,
        skip: Option<EditId>,
    ) -> Vec<EditEntry> {
        let listed = self
            .bricks
            .get(&key)
            .map_or(&[][..], |brick| &brick.edits[..]);
        listed
            .iter()
            .filter(|id| Some(**id) != skip)
            .filter_map(|id| self.edits.get(id).copied())
            .chain(pending.copied())
            .collect()
    }

    /// The brick's edits with `id` where it stood in the order, now as `moved` or gone.
    pub(super) fn entries_moved(
        &self,
        key: BrickKey,
        id: EditId,
        moved: Option<&EditEntry>,
    ) -> Vec<EditEntry> {
        let listed = self
            .bricks
            .get(&key)
            .map_or(&[][..], |brick| &brick.edits[..]);
        let mut ids: Vec<EditId> = listed
            .iter()
            .copied()
            .filter(|other| *other != id)
            .collect();
        let at = ids.partition_point(|other| *other < id);
        let mut entries: Vec<EditEntry> = ids
            .drain(..at)
            .filter_map(|other| self.edits.get(&other).copied())
            .collect();
        entries.extend(moved.copied());
        entries.extend(
            ids.iter()
                .filter_map(|other| self.edits.get(other).copied()),
        );
        entries
    }

    /// Samples and meshes one brick from scratch.
    pub(super) fn build(&self, key: BrickKey, edits: &[EditEntry]) -> Rebuilt {
        let (low, high) = self.brick_samples(key);
        if edits.is_empty()
            && let Some(solid) = self.uniform(low, high)
        {
            let inside = if solid {
                u64::from(self.desc.brick_cells).pow(3)
            } else {
                0
            };
            return Rebuilt {
                key,
                inside,
                mesh: BrickMesh::default(),
                values: None,
                collider: None,
                added: Vec::new(),
                closed: None,
            };
        }
        let mut values = self.base_samples(key);
        for entry in edits {
            self.apply(key, &mut values, entry);
        }
        self.finish(key, values)
    }

    /// A brick's samples of the base ground alone, before any edit.
    pub(super) fn base_samples(&self, key: BrickKey) -> Vec<f32> {
        let (low, _) = self.brick_samples(key);
        let side = self.side();
        let step = self.desc.voxel;
        let mut values = vec![0.0_f32; side * side * side];
        for z in 0..side {
            for x in 0..side {
                let gx = (low.x + x as i32) as f32 * step;
                let gz = (low.z + z as i32) as f32 * step;
                let (height, slope) = self.desc.base.column(gx, gz, step * 0.5);
                for y in 0..side {
                    let gy = (low.y + y as i32) as f32 * step;
                    let p = Vec3::new(gx, gy, gz);
                    values[x + side * (y + side * z)] =
                        self.desc
                            .base
                            .carve(p, height, field::base_distance(height, slope, gy));
                }
            }
        }
        values
    }

    /// Whether a brick with no edits is known solid or empty without sampling.
    pub(super) fn known(&self, key: BrickKey) -> bool {
        let (low, high) = self.brick_samples(key);
        self.uniform(low, high).is_some()
    }

    /// What each edit reaches of a brick's samples, those that reach any.
    pub(super) fn spans(&self, key: BrickKey, edits: &[EditEntry]) -> Vec<EditSpan> {
        let (low, high) = self.brick_samples(key);
        edits
            .iter()
            .filter_map(|entry| {
                let (from, to) = (entry.low.max(low), entry.high.min(high));
                (from.x <= to.x && from.y <= to.y && from.z <= to.z).then_some(EditSpan {
                    edit: entry.edit,
                    low: from,
                    high: to,
                })
            })
            .collect()
    }

    pub(super) fn sample_side(&self) -> usize {
        self.side()
    }

    fn side(&self) -> usize {
        (self.cells() + 2) as usize
    }

    pub(super) fn apply(&self, key: BrickKey, values: &mut [f32], entry: &EditEntry) {
        let (low, high) = self.brick_samples(key);
        let side = self.side();
        let step = self.desc.voxel;
        let from = entry.low.max(low);
        let to = entry.high.min(high);
        for gz in from.z..=to.z {
            for gy in from.y..=to.y {
                for gx in from.x..=to.x {
                    let p = Vec3::new(gx as f32, gy as f32, gz as f32) * step;
                    let slot = (gx - low.x) as usize
                        + side * ((gy - low.y) as usize + side * (gz - low.z) as usize);
                    values[slot] = entry
                        .edit
                        .op
                        .apply(values[slot], entry.edit.shape.distance(p));
                }
            }
        }
    }

    /// Floor, material count and triangles from a brick's samples.
    pub(super) fn finish(&self, key: BrickKey, values: Vec<f32>) -> Rebuilt {
        let (low, _) = self.brick_samples(key);
        let side = self.side();
        let step = self.desc.voxel;
        let floor = FLOOR_STEPS * step;
        let mut floored = values.clone();
        for z in 0..side {
            for y in 0..side {
                let gy = (low.y + y as i32) as f32 * step;
                for x in 0..side {
                    let at = Vec3::new(
                        (low.x + x as i32) as f32,
                        (low.y + y as i32) as f32,
                        (low.z + z as i32) as f32,
                    ) * step;
                    let slot = &mut floored[x + side * (y + side * z)];
                    *slot = slot.min(gy - floor).max(self.outside(at));
                }
            }
        }
        let mut inside = 0;
        for z in 1..side - 1 {
            for y in 1..side - 1 {
                for x in 1..side - 1 {
                    inside += u64::from(floored[x + side * (y + side * z)] < 0.0);
                }
            }
        }
        Rebuilt {
            key,
            inside,
            mesh: nets::surface_nets(&floored, side, step),
            values: Some(Arc::new(values)),
            collider: None,
            added: Vec::new(),
            closed: None,
        }
    }

    /// `Some(true)` when the ground is solid at every sample in range, `Some(false)` when empty at
    /// all, known from the height bounds with no sampling. A brick on the volume's edge has the
    /// closing surface to make, and one at the bottom has the floor.
    fn uniform(&self, low: IVec3, high: IVec3) -> Option<bool> {
        let step = self.desc.voxel;
        let n = self.cells();
        let [bx, by, bz] = self.desc.bricks.map(|count| count as i32 * n);
        let (bottom, top) = (low.y as f32 * step, high.y as f32 * step);
        if bottom <= FLOOR_STEPS * step {
            return None;
        }
        let on_edge =
            low.x <= 0 || low.y <= 0 || low.z <= 0 || high.x >= bx || high.y >= by || high.z >= bz;
        let range = self.desc.base.range_over(
            (low.x as f32 * step, low.z as f32 * step),
            (high.x as f32 * step, high.z as f32 * step),
            step,
        );
        match range {
            None => Some(false),
            Some((_, highest)) if bottom > highest => Some(false),
            // caves may open anywhere under the ground, a brick there is sampled
            Some((lowest, _)) if top < lowest && !on_edge && !self.desc.base.has_caves() => {
                Some(true)
            }
            Some(_) => None,
        }
    }

    /// A copy that knows only the lattice, for building bricks away from the volume.
    pub(super) fn lattice(&self) -> Self {
        Self {
            desc: self.desc,
            edits: BTreeMap::new(),
            next_edit: 0,
            bricks: HashMap::new(),
            revision: 0,
            cache_bytes: 0,
        }
    }

    pub(super) fn commit(&mut self, rebuilt: Vec<Rebuilt>) -> (u64, i64) {
        self.revision += 1;
        let mut change = 0_i64;
        for item in rebuilt {
            let brick = self.bricks.entry(item.key).or_default();
            change += item.inside as i64 - brick.inside as i64;
            brick.inside = item.inside;
            brick.mesh = item.mesh;
            brick.revision = self.revision;
            brick.edits.retain(|other| Some(*other) != item.closed);
            brick.edits.extend(item.added);
            brick.edits.sort_unstable();
            brick.values = if brick.edits.is_empty() {
                None
            } else {
                item.values
            };
        }
        self.bricks.retain(|_, brick| {
            brick.inside > 0 || !brick.mesh.is_empty() || !brick.edits.is_empty()
        });
        self.trim_cache();
        (self.revision, change)
    }
}

fn inside(at: IVec3, low: IVec3, high: IVec3) -> bool {
    at.max(low) == at && at.min(high) == at
}

fn overlaps(a_low: IVec3, a_high: IVec3, b_low: IVec3, b_high: IVec3) -> bool {
    (a_high.min(b_high) - a_low.max(b_low)).min_element() >= 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Material, Planet, PlanetDesc, PlanetGround, level_for};
    use algebrix::DVec3;

    #[test]
    fn a_planet_brick_called_solid_or_empty_without_sampling_is_so_at_every_sample() {
        let planet = Planet::new(PlanetDesc::earth(3)).unwrap();
        let anchor = planet.surface(DVec3::new(0.3, 0.8, 0.5));
        let level = level_for(&planet, 2.5);
        let ground = PlanetGround::new(planet, anchor, level).unwrap();
        // away from the anchor too, where the columns lean
        for at in [DVec3::ZERO, DVec3::new(700.0, 0.0, -400.0)] {
            let desc = ground
                .sdf_desc(ground.ground_under(at), 12.0, (0.5, 8), Material::default())
                .unwrap();
            let volume = SdfVolume::new(desc).unwrap();
            let step = desc.voxel;
            let (mut skipped, mut sampled) = (0, 0);
            let [bx, by, bz] = desc.bricks;
            for z in 0..bz {
                for y in 0..by {
                    for x in 0..bx {
                        let key = BrickKey([x, y, z]);
                        let (low, high) = volume.brick_samples(key);
                        let Some(solid) = volume.uniform(low, high) else {
                            sampled += 1;
                            continue;
                        };
                        skipped += 1;
                        for gz in low.z..=high.z {
                            for gy in low.y..=high.y {
                                for gx in low.x..=high.x {
                                    let p = Vec3::new(gx as f32, gy as f32, gz as f32) * step;
                                    let (height, slope) = desc.base.column(p.x, p.z, step * 0.5);
                                    let d = field::base_distance(height, slope, p.y);
                                    assert_eq!(d < 0.0, solid, "{key:?} at {p:?}: {d}");
                                }
                            }
                        }
                    }
                }
            }
            // bricks well above or below the ground are left unsampled
            assert!(skipped > 0 && sampled > 0, "{skipped} {sampled}");
        }
    }
}
