mod field;
mod nets;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};

use algebrix::{DVec3, IVec3, Vec3};

use crate::collision::trimesh::TriMesh;

pub use field::{Base, Edit, EditOp, EditShape};
pub use nets::BrickMesh;

use crate::{
    BodyHandle, BodyKind, CastHit, CollisionFilter, GridPosition, JobDispatch, Material,
    PhysicsError, PhysicsWorld, QueryFilter, Ray, RigidBodyDesc, Sequential, Shape,
};

const MAX_BRICKS: u64 = 1 << 20;
/// A body counts as swallowed when its center is nearer the surface than this share of its radius.
const EJECT_DEPTH: f32 = 0.25;
/// Room left between an ejected body and the surface it lands on, meters.
const EJECT_GAP: f32 = 0.02;
/// Layer bit brick bodies carry besides the default one, so a query can ask for the ground alone.
/// Bodies a host puts on this bit look like ground to [`SdfTerrain::raycast`].
pub const GROUND_LAYER: u32 = 1 << 31;
/// Most edits one job takes, so a long queue still commits in steps.
const MAX_BATCH: usize = 64;
/// Lattice steps an edit reaches past its shape. There the shape's distance is at least two
/// steps, so cutting the edit off never moves a surface.
const EDIT_APRON: f32 = 2.0;
/// Lattice steps at the bottom no edit can open, so nothing digs out through the floor.
const FLOOR_STEPS: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfDesc {
    /// The volume's low corner.
    pub origin: GridPosition,
    /// Lattice step in meters.
    pub voxel: f32,
    /// Cells along each side of a brick.
    pub brick_cells: u32,
    /// Bricks along x, y and z.
    pub bricks: [u32; 3],
    pub base: Base,
    /// Friction and restitution of the ground, and the density removed mass is counted with.
    pub material: Material,
}

impl SdfDesc {
    fn validate(&self) -> Result<(), PhysicsError> {
        let count = self.bricks.iter().map(|n| u64::from(*n)).product::<u64>();
        let valid = self.voxel.is_finite()
            && self.voxel > 0.0
            && (4..=32).contains(&self.brick_cells)
            && self.bricks.iter().all(|n| *n > 0)
            && count <= MAX_BRICKS;
        if !valid {
            return Err(PhysicsError::InvalidShape);
        }
        self.material.validate()?;
        self.base.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EditId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BrickKey(pub [u32; 3]);

#[derive(Debug, Clone, Copy)]
struct EditEntry {
    edit: Edit,
    /// Lattice samples it reaches, inclusive. Whether a sample feels the edit depends on the
    /// sample alone, never on which brick asks.
    low: IVec3,
    high: IVec3,
}

#[derive(Debug, Clone, Default)]
struct Brick {
    revision: u64,
    /// Samples inside material among the ones this brick owns.
    inside: u64,
    mesh: BrickMesh,
    /// Edits that reach it, oldest first.
    edits: Vec<EditId>,
    /// Samples after the base and every edit, before the floor. Kept while the brick has edits,
    /// so the next edit runs alone on top.
    values: Option<Arc<Vec<f32>>>,
}

/// What an edit or removal changed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditReport {
    /// The last edit committed.
    pub edit: EditId,
    /// How many edits this commit holds, more than one when they waited in a queue.
    pub edits: usize,
    pub revision: u64,
    pub bricks: usize,
    /// Material volume after minus before, cubic meters.
    pub volume_change: f64,
    /// Mass an excavation took out, zero for other edits.
    pub removed_mass: f64,
    /// Bodies the new material swallowed and that were moved up to its surface.
    pub ejected: usize,
}

/// A brick built again but not yet in the volume.
#[derive(Debug)]
struct Rebuilt {
    key: BrickKey,
    inside: u64,
    mesh: BrickMesh,
    values: Option<Arc<Vec<f32>>>,
    collider: Option<TriMesh>,
    added: Vec<EditId>,
    closed: Option<EditId>,
}

/// One brick's work: start from its kept samples and run `edits`, or from scratch with `edits`
/// being every edit that reaches it.
#[derive(Debug)]
struct JobBrick {
    key: BrickKey,
    kept: Option<Arc<Vec<f32>>>,
    edits: Vec<EditEntry>,
    added: Vec<EditId>,
    closed: Option<EditId>,
}

/// Building a batch of bricks, cut off from the terrain and the world, so it can run on any
/// thread while the world steps. Comes from [`SdfTerrain::next_job`].
#[derive(Debug)]
pub struct BrickJob {
    lattice: SdfVolume,
    bricks: Vec<JobBrick>,
    token: u64,
}

impl BrickJob {
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// Samples, meshes and readies colliders for every brick. The result does not depend on
    /// `dispatch`.
    ///
    /// # Errors
    /// Fails on triangles a collider will not take, or a dispatcher that skips jobs. Then call
    /// [`SdfTerrain::abandon`].
    pub fn run(self, dispatch: &dyn JobDispatch) -> Result<BuiltBricks, PhysicsError> {
        let lattice = &self.lattice;
        let built = in_parallel(dispatch, &self.bricks, |brick| {
            let mut item = match &brick.kept {
                Some(kept) => {
                    let mut values = Vec::clone(kept);
                    for entry in &brick.edits {
                        lattice.apply(brick.key, &mut values, entry);
                    }
                    lattice.finish(brick.key, values)
                }
                None => lattice.build(brick.key, &brick.edits),
            };
            item.added.clone_from(&brick.added);
            item.closed = brick.closed;
            if !item.mesh.is_empty() {
                item.collider = Some(TriMesh::from_triangles(
                    item.mesh.vertices.clone(),
                    item.mesh.triangles.clone(),
                )?);
            }
            Ok(item)
        })?;
        let rebuilt = built
            .into_iter()
            .collect::<Result<Vec<_>, PhysicsError>>()?;
        Ok(BuiltBricks {
            rebuilt,
            token: self.token,
        })
    }
}

/// A finished [`BrickJob`], for [`SdfTerrain::commit`].
#[derive(Debug)]
pub struct BuiltBricks {
    rebuilt: Vec<Rebuilt>,
    token: u64,
}

/// Work waiting its turn, in the order it was asked for.
#[derive(Debug, Clone, Copy)]
enum Queued {
    Edit(EditId, EditEntry),
    Close(EditId, EditEntry),
}

/// Ground as a signed distance field: a base and ordered edits, sampled in sparse bricks. Only
/// bricks near the surface hold triangles. An edit samples again only the bricks it reaches,
/// and each of those only runs the edits that reach it.
#[derive(Debug, Clone)]
pub struct SdfVolume {
    desc: SdfDesc,
    edits: BTreeMap<EditId, EditEntry>,
    next_edit: u32,
    bricks: HashMap<BrickKey, Brick>,
    revision: u64,
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
        let mut distance = field::base_distance(height, slope, at.y);
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

    fn brick_corner(&self, key: BrickKey) -> Vec3 {
        let [x, y, z] = key.0;
        Vec3::new(x as f32, y as f32, z as f32) * (self.desc.brick_cells as f32 * self.desc.voxel)
    }

    /// Lattice samples a brick reads, inclusive: its own and one more on each side.
    fn brick_samples(&self, key: BrickKey) -> (IVec3, IVec3) {
        let n = self.cells();
        let low = IVec3::new(key.0[0] as i32, key.0[1] as i32, key.0[2] as i32) * n;
        (low - IVec3::ONE, low + IVec3::splat(n))
    }

    fn entry(&self, edit: Edit) -> EditEntry {
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
    fn reached(&self, entry: &EditEntry) -> Vec<BrickKey> {
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
    fn entries_of(
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

    /// Samples and meshes one brick from scratch.
    fn build(&self, key: BrickKey, edits: &[EditEntry]) -> Rebuilt {
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
                    values[x + side * (y + side * z)] = field::base_distance(height, slope, gy);
                }
            }
        }
        for entry in edits {
            self.apply(key, &mut values, entry);
        }
        self.finish(key, values)
    }

    fn side(&self) -> usize {
        (self.cells() + 2) as usize
    }

    fn apply(&self, key: BrickKey, values: &mut [f32], entry: &EditEntry) {
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
    fn finish(&self, key: BrickKey, values: Vec<f32>) -> Rebuilt {
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
        match self.desc.base.height_range() {
            None => Some(false),
            Some((_, highest)) if bottom > highest => Some(false),
            Some((lowest, _)) if top < lowest && !on_edge => Some(true),
            Some(_) => None,
        }
    }

    /// A copy that knows only the lattice, for building bricks away from the volume.
    fn lattice(&self) -> Self {
        Self {
            desc: self.desc,
            edits: BTreeMap::new(),
            next_edit: 0,
            bricks: HashMap::new(),
            revision: 0,
        }
    }

    fn commit(&mut self, rebuilt: Vec<Rebuilt>) -> (u64, i64) {
        self.revision += 1;
        let mut change = 0_i64;
        for item in rebuilt {
            let brick = self.bricks.entry(item.key).or_default();
            change += item.inside as i64 - brick.inside as i64;
            brick.inside = item.inside;
            brick.mesh = item.mesh;
            brick.revision = self.revision;
            brick.edits.extend(item.added);
            brick.edits.retain(|other| Some(*other) != item.closed);
            brick.values = if brick.edits.is_empty() {
                None
            } else {
                item.values
            };
        }
        self.bricks.retain(|_, brick| {
            brick.inside > 0 || !brick.mesh.is_empty() || !brick.edits.is_empty()
        });
        (self.revision, change)
    }
}

/// `work` for each item on the dispatcher's workers, results in item order.
fn in_parallel<T: Sync, R: Send>(
    dispatch: &dyn JobDispatch,
    items: &[T],
    work: impl Fn(&T) -> R + Sync,
) -> Result<Vec<R>, PhysicsError> {
    let slots: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    dispatch.run(items.len(), &|index| {
        let result = work(&items[index]);
        *slots[index].lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
    });
    slots
        .into_iter()
        .map(|slot| slot.into_inner().unwrap_or_else(PoisonError::into_inner))
        .collect::<Option<Vec<R>>>()
        .ok_or(PhysicsError::DispatchIncomplete)
}

fn inside(at: IVec3, low: IVec3, high: IVec3) -> bool {
    at.max(low) == at && at.min(high) == at
}

fn overlaps(a_low: IVec3, a_high: IVec3, b_low: IVec3, b_high: IVec3) -> bool {
    (a_high.min(b_high) - a_low.max(b_low)).min_element() >= 0
}

/// An [`SdfVolume`] in a world: each surface brick is a fixed body with its triangles. Changes
/// land between steps, every brick of one edit at once, so a step never sees half a hole.
#[derive(Debug)]
pub struct SdfTerrain {
    volume: SdfVolume,
    bodies: HashMap<BrickKey, (BodyHandle, Shape)>,
    queue: VecDeque<Queued>,
    /// What the job handed out holds, until it commits or is abandoned.
    flying: Vec<Queued>,
    token: u64,
}

impl SdfTerrain {
    /// # Errors
    /// Fails on a bad description or a brick the world will not take. Nothing stays behind.
    pub fn new(world: &mut PhysicsWorld, desc: SdfDesc) -> Result<Self, PhysicsError> {
        Self::new_with(world, desc, &Sequential)
    }

    /// [`Self::new`] with bricks built on the host's workers.
    ///
    /// # Errors
    /// See [`Self::new`], and a dispatcher that skips jobs.
    pub fn new_with(
        world: &mut PhysicsWorld,
        desc: SdfDesc,
        dispatch: &dyn JobDispatch,
    ) -> Result<Self, PhysicsError> {
        let mut terrain = Self {
            volume: SdfVolume::new_with(desc, dispatch)?,
            bodies: HashMap::new(),
            queue: VecDeque::new(),
            flying: Vec::new(),
            token: 0,
        };
        let mut keys: Vec<BrickKey> = terrain
            .volume
            .surface_bricks()
            .map(|(key, ..)| key)
            .collect();
        keys.sort_unstable();
        for key in keys {
            let mesh = terrain.volume.brick_mesh(key).cloned().unwrap_or_default();
            match terrain.collider(world, key, &mesh) {
                Ok(Some(body)) => {
                    terrain.bodies.insert(key, body);
                }
                Ok(None) => {}
                Err(error) => {
                    terrain.remove(world);
                    return Err(error);
                }
            }
        }
        Ok(terrain)
    }

    pub fn volume(&self) -> &SdfVolume {
        &self.volume
    }

    /// The first ground along `ray`, whatever other bodies are in the way.
    ///
    /// # Errors
    /// Fails on a bad ray.
    pub fn raycast(
        &self,
        world: &PhysicsWorld,
        ray: &Ray,
    ) -> Result<Option<CastHit>, PhysicsError> {
        world.raycast(
            ray,
            &QueryFilter {
                mask: GROUND_LAYER,
                ..QueryFilter::default()
            },
        )
    }

    pub fn body_of(&self, key: BrickKey) -> Option<BodyHandle> {
        self.bodies.get(&key).map(|(body, _)| *body)
    }

    pub fn bodies(&self) -> impl Iterator<Item = BodyHandle> + '_ {
        self.bodies.values().map(|(body, _)| *body)
    }

    fn collider(
        &self,
        world: &mut PhysicsWorld,
        key: BrickKey,
        mesh: &BrickMesh,
    ) -> Result<Option<(BodyHandle, Shape)>, PhysicsError> {
        if mesh.is_empty() {
            return Ok(None);
        }
        let shape = world.add_mesh(&mesh.vertices, &mesh.triangles)?;
        self.place(world, key, shape).map(Some)
    }

    /// A fixed body for a brick's mesh. On failure the mesh is gone too.
    fn place(
        &self,
        world: &mut PhysicsWorld,
        key: BrickKey,
        shape: Shape,
    ) -> Result<(BodyHandle, Shape), PhysicsError> {
        let corner = DVec3::from_vec3(self.volume.brick_corner(key));
        let placed = self
            .volume
            .desc
            .origin
            .shifted(corner)
            .ok_or(PhysicsError::OutOfRange)
            .and_then(|position| {
                world.spawn(
                    RigidBodyDesc::fixed(shape)
                        .position(position)
                        .material(self.volume.desc.material)
                        .filter(CollisionFilter {
                            layers: CollisionFilter::default().layers | GROUND_LAYER,
                            mask: u32::MAX,
                        }),
                )
            });
        match placed {
            Ok(body) => Ok((body, shape)),
            Err(error) => {
                world.remove_mesh(shape)?;
                Err(error)
            }
        }
    }

    /// Asks for an edit. It lands when a job holding it commits, so it can be built away from
    /// the step. Edits land in the order they were asked for.
    ///
    /// # Errors
    /// Fails on a bad shape.
    pub fn queue_edit(&mut self, edit: Edit) -> Result<EditId, PhysicsError> {
        edit.shape.validate()?;
        let id = EditId(self.volume.next_edit);
        self.volume.next_edit += 1;
        let entry = self.volume.entry(edit);
        self.queue.push_back(Queued::Edit(id, entry));
        Ok(id)
    }

    /// Asks for a tunnel to be taken out, the ground it cut comes back.
    ///
    /// # Errors
    /// Fails when `id` is not a tunnel here, or it is already on its way out.
    pub fn queue_close_tunnel(&mut self, id: EditId) -> Result<(), PhysicsError> {
        let waiting = || self.queue.iter().chain(&self.flying);
        if waiting().any(|item| matches!(item, Queued::Close(other, _) if *other == id)) {
            return Err(PhysicsError::InvalidHandle);
        }
        let asked = waiting().find_map(|item| match item {
            Queued::Edit(other, entry) if *other == id => Some(*entry),
            _ => None,
        });
        let entry = asked
            .or_else(|| self.volume.edits.get(&id).copied())
            .filter(|entry| entry.edit.op == EditOp::Tunnel)
            .ok_or(PhysicsError::InvalidHandle)?;
        self.queue.push_back(Queued::Close(id, entry));
        Ok(())
    }

    /// Edits asked for and not landed yet.
    pub fn pending(&self) -> usize {
        self.queue.len() + self.flying.len()
    }

    /// Whether a job is out and its result not committed.
    pub fn job_out(&self) -> bool {
        !self.flying.is_empty()
    }

    /// The next batch to build, `None` with one still out or nothing asked for. A batch is
    /// edits of one kind in a row, so what it did to the ground can be told apart.
    pub fn next_job(&mut self) -> Option<BrickJob> {
        if !self.flying.is_empty() {
            return None;
        }
        let first = self.queue.pop_front()?;
        self.flying.push(first);
        if let Queued::Edit(_, entry) = first {
            while self.flying.len() < MAX_BATCH {
                match self.queue.front() {
                    Some(Queued::Edit(_, next)) if next.edit.op == entry.edit.op => {}
                    _ => break,
                }
                if let Some(next) = self.queue.pop_front() {
                    self.flying.push(next);
                }
            }
        }
        let mut bricks: BTreeMap<BrickKey, JobBrick> = BTreeMap::new();
        for item in &self.flying {
            match *item {
                Queued::Edit(id, entry) => {
                    for key in self.volume.reached(&entry) {
                        let brick = bricks.entry(key).or_insert_with(|| {
                            let kept = self
                                .volume
                                .bricks
                                .get(&key)
                                .and_then(|brick| brick.values.clone());
                            let edits = if kept.is_some() {
                                Vec::new()
                            } else {
                                self.volume.entries_of(key, None, None)
                            };
                            JobBrick {
                                key,
                                kept,
                                edits,
                                added: Vec::new(),
                                closed: None,
                            }
                        });
                        brick.edits.push(entry);
                        brick.added.push(id);
                    }
                }
                Queued::Close(id, entry) => {
                    for key in self.volume.reached(&entry) {
                        bricks.insert(
                            key,
                            JobBrick {
                                key,
                                kept: None,
                                edits: self.volume.entries_of(key, None, Some(id)),
                                added: Vec::new(),
                                closed: Some(id),
                            },
                        );
                    }
                }
            }
        }
        self.token += 1;
        Some(BrickJob {
            lattice: self.volume.lattice(),
            bricks: bricks.into_values().collect(),
            token: self.token,
        })
    }

    /// Drops the edits of the job that is out, its result is refused from now on.
    pub fn abandon(&mut self) {
        self.flying.clear();
        self.token += 1;
    }

    /// Puts a finished job in the world, every brick at once. Call it between steps.
    ///
    /// # Errors
    /// Fails on a result of a job abandoned or already committed, or a brick the world will not
    /// take. Then nothing changed.
    pub fn commit(
        &mut self,
        world: &mut PhysicsWorld,
        built: BuiltBricks,
    ) -> Result<EditReport, PhysicsError> {
        if built.token != self.token || self.flying.is_empty() {
            return Err(PhysicsError::InvalidHandle);
        }
        let mut rebuilt = built.rebuilt;
        let mut fresh = Vec::new();
        for item in &mut rebuilt {
            let Some(collider) = item.collider.take() else {
                continue;
            };
            let placed = world
                .push_mesh(collider)
                .and_then(|shape| self.place(world, item.key, shape));
            match placed {
                Ok(body) => fresh.push((item.key, body)),
                Err(error) => {
                    let bodies: Vec<_> = fresh.iter().map(|(_, body)| *body).collect();
                    release(world, &bodies);
                    return Err(error);
                }
            }
        }
        let keys: Vec<BrickKey> = rebuilt.iter().map(|item| item.key).collect();
        let old: Vec<_> = keys
            .iter()
            .filter_map(|key| self.bodies.remove(key))
            .collect();
        release(world, &old);
        self.bodies.extend(fresh);
        let (revision, change) = self.volume.commit(rebuilt);
        let volume_change = change as f64 * f64::from(self.volume.desc.voxel).powi(3);
        let density = f64::from(self.volume.desc.material.density);
        let mut report = EditReport {
            edit: EditId(0),
            edits: self.flying.len(),
            revision,
            bricks: keys.len(),
            volume_change,
            removed_mass: 0.0,
            ejected: 0,
        };
        for item in std::mem::take(&mut self.flying) {
            match item {
                Queued::Edit(id, entry) => {
                    self.volume.edits.insert(id, entry);
                    report.edit = id;
                    match entry.edit.op {
                        EditOp::Excavate => {
                            report.removed_mass = (-volume_change).max(0.0) * density;
                        }
                        EditOp::Add => report.ejected += self.eject(world, &entry)?,
                        EditOp::Tunnel => {}
                    }
                }
                Queued::Close(id, entry) => {
                    self.volume.edits.remove(&id);
                    report.edit = id;
                    report.ejected += self.eject(world, &entry)?;
                }
            }
        }
        Ok(report)
    }

    /// Queue, build and commit in one call. Only with nothing else waiting.
    fn run_now(
        &mut self,
        world: &mut PhysicsWorld,
        dispatch: &dyn JobDispatch,
    ) -> Result<EditReport, PhysicsError> {
        let Some(job) = self.next_job() else {
            return Err(PhysicsError::TerrainBusy);
        };
        let done = job
            .run(dispatch)
            .and_then(|built| self.commit(world, built));
        if done.is_err() {
            self.abandon();
        }
        done
    }

    /// Applies one edit to the volume and the world.
    ///
    /// # Errors
    /// Fails on a bad shape, with other edits waiting, or when the world will not take a new
    /// brick. Then nothing changed.
    pub fn edit(
        &mut self,
        world: &mut PhysicsWorld,
        edit: Edit,
    ) -> Result<EditReport, PhysicsError> {
        self.edit_with(world, edit, &Sequential)
    }

    /// [`Self::edit`] with its bricks built on the host's workers. The result does not depend
    /// on them.
    ///
    /// # Errors
    /// See [`Self::edit`], and a dispatcher that skips jobs.
    pub fn edit_with(
        &mut self,
        world: &mut PhysicsWorld,
        edit: Edit,
        dispatch: &dyn JobDispatch,
    ) -> Result<EditReport, PhysicsError> {
        if self.pending() > 0 {
            return Err(PhysicsError::TerrainBusy);
        }
        self.queue_edit(edit)?;
        self.run_now(world, dispatch)
    }

    /// Takes a tunnel out again, the ground it cut comes back.
    ///
    /// # Errors
    /// Fails when `id` is not a tunnel here, with other edits waiting, or the world will not
    /// take a brick.
    pub fn remove_tunnel(
        &mut self,
        world: &mut PhysicsWorld,
        id: EditId,
    ) -> Result<EditReport, PhysicsError> {
        self.remove_tunnel_with(world, id, &Sequential)
    }

    /// [`Self::remove_tunnel`] with its bricks built on the host's workers.
    ///
    /// # Errors
    /// See [`Self::remove_tunnel`], and a dispatcher that skips jobs.
    pub fn remove_tunnel_with(
        &mut self,
        world: &mut PhysicsWorld,
        id: EditId,
        dispatch: &dyn JobDispatch,
    ) -> Result<EditReport, PhysicsError> {
        if self.pending() > 0 {
            return Err(PhysicsError::TerrainBusy);
        }
        self.queue_close_tunnel(id)?;
        self.run_now(world, dispatch)
    }

    /// Moves dynamic bodies that new material swallowed straight up to the surface.
    /// Triangles only push from the front, so a body left under them would fall out of the world.
    fn eject(&self, world: &mut PhysicsWorld, entry: &EditEntry) -> Result<usize, PhysicsError> {
        let step = self.volume.desc.voxel;
        let origin = self.volume.desc.origin;
        let (low, high) = (
            Vec3::new(entry.low.x as f32, entry.low.y as f32, entry.low.z as f32) * step,
            Vec3::new(
                entry.high.x as f32,
                entry.high.y as f32,
                entry.high.z as f32,
            ) * step,
        );
        let mut buried = Vec::new();
        for body in world.body_handles() {
            let state = world.body_state(body)?;
            if state.kind != BodyKind::Dynamic {
                continue;
            }
            let at = state.position.offset_from(origin).as_vec3();
            if at.max(low) != at || at.min(high) != at {
                continue;
            }
            let shape = world.body_shape(body)?;
            let radius = world.bodies.assets.bounding_radius(&shape);
            // bodies at rest sit half a radius or more above the surface, so this leaves them be
            if self.volume.distance(at) < radius * EJECT_DEPTH {
                buried.push((body, at, radius));
            }
        }
        // lowest first, so a column of bodies lands in the order it stood
        buried.sort_by(|a, b| a.1.y.total_cmp(&b.1.y));
        let mut placed: Vec<(Vec3, f32)> = Vec::with_capacity(buried.len());
        let top = self
            .volume
            .brick_corner(BrickKey(self.volume.desc.bricks))
            .y;
        for (body, mut at, radius) in buried {
            let clear = |at: Vec3, placed: &[(Vec3, f32)]| {
                self.volume.distance(at) >= radius + EJECT_GAP
                    && placed
                        .iter()
                        .all(|(other, reach)| (at - *other).length() >= radius + reach)
            };
            while at.y < top && !clear(at, &placed) {
                at.y += step;
            }
            placed.push((at, radius));
            let position = origin
                .shifted(DVec3::from_vec3(at))
                .ok_or(PhysicsError::OutOfRange)?;
            world.teleport(body, position)?;
            world.set_velocity(body, Vec3::ZERO, Vec3::ZERO)?;
        }
        Ok(placed.len())
    }

    /// Takes every brick body out of the world.
    pub fn remove(self, world: &mut PhysicsWorld) {
        let bodies: Vec<_> = self.bodies.into_values().collect();
        release(world, &bodies);
    }
}

fn release(world: &mut PhysicsWorld, bodies: &[(BodyHandle, Shape)]) {
    for (body, shape) in bodies {
        // both are ours and alive, so these only fail if someone else removed them first
        let _ = world.despawn(*body);
        let _ = world.remove_mesh(*shape);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use algebrix::Quat;

    use super::*;
    use crate::PhysicsSettings;

    fn hills() -> Base {
        Base::Hills {
            seed: 11,
            height: 6.0,
            amplitude: 2.5,
            wavelength: 14.0,
            octaves: 3,
        }
    }

    fn desc(base: Base) -> SdfDesc {
        SdfDesc {
            origin: GridPosition::ORIGIN,
            voxel: 0.5,
            brick_cells: 8,
            bricks: [6, 4, 6],
            base,
            material: Material {
                density: 2000.0,
                ..Material::default()
            },
        }
    }

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    fn edits() -> Vec<Edit> {
        vec![
            Edit {
                shape: EditShape::Sphere {
                    center: Vec3::new(6.0, 6.0, 6.0),
                    radius: 2.5,
                },
                op: EditOp::Excavate,
            },
            Edit {
                shape: EditShape::Capsule {
                    a: Vec3::new(1.0, 4.5, 12.0),
                    b: Vec3::new(20.0, 5.0, 12.5),
                    radius: 1.3,
                },
                op: EditOp::Tunnel,
            },
            Edit {
                shape: EditShape::Box {
                    center: Vec3::new(14.0, 7.0, 4.0),
                    half_extents: Vec3::new(2.0, 1.5, 1.0),
                    rotation: Quat::from_axis_angle(Vec3::Y, 0.6),
                },
                op: EditOp::Add,
            },
            Edit {
                shape: EditShape::Sphere {
                    center: Vec3::new(4.0, 7.9, 4.0),
                    radius: 1.7,
                },
                op: EditOp::Excavate,
            },
        ]
    }

    /// Every brick sampled straight from `distance`, with no brick lists or caching.
    fn reference(volume: &SdfVolume, key: BrickKey) -> BrickMesh {
        let (low, high) = volume.brick_samples(key);
        let side = (high.x - low.x + 1) as usize;
        let step = volume.desc.voxel;
        let mut values = Vec::with_capacity(side * side * side);
        for z in low.z..=high.z {
            for y in low.y..=high.y {
                for x in low.x..=high.x {
                    values.push(volume.distance(Vec3::new(x as f32, y as f32, z as f32) * step));
                }
            }
        }
        nets::surface_nets(&values, side, step)
    }

    fn all_keys(volume: &SdfVolume) -> Vec<BrickKey> {
        let [bx, by, bz] = volume.desc.bricks;
        let mut keys = Vec::new();
        for z in 0..bz {
            for y in 0..by {
                for x in 0..bx {
                    keys.push(BrickKey([x, y, z]));
                }
            }
        }
        keys
    }

    fn assert_matches_reference(volume: &SdfVolume) {
        for key in all_keys(volume) {
            let stored = volume.brick_mesh(key).cloned().unwrap_or_default();
            assert_eq!(stored, reference(volume, key), "{key:?}");
        }
    }

    #[test]
    fn edited_bricks_match_sampling_everything_again() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        assert_matches_reference(terrain.volume());
        for edit in edits() {
            let report = terrain.edit(&mut world, edit).unwrap();
            assert!(report.bricks > 0);
            assert_matches_reference(terrain.volume());
        }
    }

    #[test]
    fn the_surface_has_no_gaps_between_bricks() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(Base::Empty)).unwrap();
        // a closed blob across many bricks
        for edit in [
            Edit {
                shape: EditShape::Sphere {
                    center: Vec3::splat(8.0),
                    radius: 5.3,
                },
                op: EditOp::Add,
            },
            Edit {
                shape: EditShape::Capsule {
                    a: Vec3::new(3.0, 8.0, 8.0),
                    b: Vec3::new(13.0, 9.0, 7.5),
                    radius: 1.6,
                },
                op: EditOp::Excavate,
            },
        ] {
            terrain.edit(&mut world, edit).unwrap();
        }
        let key = |p: Vec3| {
            (
                (p.x * 1e3).round() as i64,
                (p.y * 1e3).round() as i64,
                (p.z * 1e3).round() as i64,
            )
        };
        let mut edges: HashMap<_, i32> = HashMap::new();
        let mut bricks = 0;
        for (_, corner, _, mesh) in terrain.volume().surface_bricks() {
            bricks += 1;
            for triangle in &mesh.triangles {
                let p = triangle.map(|i| key(mesh.vertices[i as usize] + corner));
                for (a, b) in [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])] {
                    let sign = if a < b { 1 } else { -1 };
                    *edges.entry((a.min(b), a.max(b))).or_default() += sign;
                }
            }
        }
        assert!(bricks > 8, "{bricks}");
        let open = edges.values().filter(|count| **count != 0).count();
        assert_eq!(open, 0, "{open} open edges of {}", edges.len());
    }

    #[test]
    fn excavation_counts_the_volume_and_mass_it_takes() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 12.0 })).unwrap();
        let before = terrain.volume().material_volume();
        let radius = 3.0_f64;
        let report = terrain
            .edit(
                &mut world,
                Edit {
                    shape: EditShape::Sphere {
                        center: Vec3::new(12.0, 6.0, 12.0),
                        radius: radius as f32,
                    },
                    op: EditOp::Excavate,
                },
            )
            .unwrap();
        let ball = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        assert!(
            (-report.volume_change / ball - 1.0).abs() < 0.05,
            "{report:?} {ball}"
        );
        assert!((report.removed_mass - -report.volume_change * 2000.0).abs() < 1e-6);
        let after = terrain.volume().material_volume();
        assert!((before - after + report.volume_change).abs() < 1e-6);

        let added = terrain
            .edit(
                &mut world,
                Edit {
                    shape: EditShape::Sphere {
                        center: Vec3::new(12.0, 6.0, 12.0),
                        radius: 1.0,
                    },
                    op: EditOp::Add,
                },
            )
            .unwrap();
        assert!(added.volume_change > 0.0);
        assert_eq!(added.removed_mass, 0.0);
    }

    #[test]
    fn removing_a_tunnel_gives_back_the_same_ground() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        let meshes = |terrain: &SdfTerrain| {
            let mut all: Vec<_> = terrain
                .volume()
                .surface_bricks()
                .map(|(key, _, _, mesh)| (key, mesh.clone()))
                .collect();
            all.sort_by_key(|(key, _)| *key);
            all
        };
        let start = meshes(&terrain);
        let volume = terrain.volume().material_volume();
        let tunnel = terrain.edit(&mut world, edits()[1]).unwrap();
        assert!(tunnel.volume_change < 0.0);
        assert_eq!(tunnel.removed_mass, 0.0);
        assert_ne!(meshes(&terrain), start);
        let back = terrain.remove_tunnel(&mut world, tunnel.edit).unwrap();
        assert!((back.volume_change + tunnel.volume_change).abs() < 1e-9);
        assert_eq!(meshes(&terrain), start);
        assert_eq!(terrain.volume().material_volume(), volume);
        assert_eq!(
            terrain.remove_tunnel(&mut world, tunnel.edit).map(|_| ()),
            Err(PhysicsError::InvalidHandle)
        );
        let dug = terrain.edit(&mut world, edits()[0]).unwrap();
        assert_eq!(
            terrain.remove_tunnel(&mut world, dug.edit).map(|_| ()),
            Err(PhysicsError::InvalidHandle)
        );
    }

    #[test]
    fn digging_one_spot_many_times_matches_a_full_rebuild() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        for i in 0..25 {
            let edit = Edit {
                shape: EditShape::Sphere {
                    center: Vec3::new(9.0 + (i % 5) as f32 * 0.4, 6.5 - i as f32 * 0.15, 9.0),
                    radius: 1.5 + (i % 3) as f32 * 0.3,
                },
                op: if i % 4 == 3 {
                    EditOp::Add
                } else {
                    EditOp::Excavate
                },
            };
            terrain.edit(&mut world, edit).unwrap();
        }
        assert_matches_reference(terrain.volume());
        let (bricks, bytes) = terrain.volume().cached_bricks();
        assert!(
            bricks > 0 && bytes == bricks * 10 * 10 * 10 * 4,
            "{bricks} {bytes}"
        );
        let tunnel = terrain.edit(&mut world, edits()[1]).unwrap();
        terrain.remove_tunnel(&mut world, tunnel.edit).unwrap();
        assert_matches_reference(terrain.volume());
    }

    /// Runs jobs backwards on scoped threads, so order or sharing bugs show.
    struct Threads;

    impl JobDispatch for Threads {
        fn parallelism(&self) -> usize {
            4
        }

        fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
            std::thread::scope(|scope| {
                for worker in 0..4 {
                    scope.spawn(move || {
                        for index in (0..jobs).rev().filter(|index| index % 4 == worker) {
                            job(index);
                        }
                    });
                }
            });
        }
    }

    struct Lazy;

    impl JobDispatch for Lazy {
        fn parallelism(&self) -> usize {
            1
        }

        fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
            for index in 1..jobs {
                job(index);
            }
        }
    }

    #[test]
    fn workers_give_the_same_ground_as_one_thread() {
        let (mut alone, mut shared) = (world(), world());
        let mut one = SdfTerrain::new(&mut alone, desc(hills())).unwrap();
        let mut many = SdfTerrain::new_with(&mut shared, desc(hills()), &Threads).unwrap();
        for edit in edits() {
            let a = one.edit(&mut alone, edit).unwrap();
            let b = many.edit_with(&mut shared, edit, &Threads).unwrap();
            assert_eq!(a, b);
        }
        let tunnel = EditId(1);
        one.remove_tunnel(&mut alone, tunnel).unwrap();
        many.remove_tunnel_with(&mut shared, tunnel, &Threads)
            .unwrap();
        for key in all_keys(one.volume()) {
            assert_eq!(one.volume().brick_mesh(key), many.volume().brick_mesh(key));
        }
        assert_matches_reference(many.volume());

        let revision = many.volume().revision();
        assert_eq!(
            many.edit_with(&mut shared, edits()[0], &Lazy).map(|_| ()),
            Err(PhysicsError::DispatchIncomplete)
        );
        assert_eq!(many.volume().revision(), revision);
        assert!(SdfVolume::new_with(desc(hills()), &Lazy).is_err());
    }

    #[test]
    fn a_ground_ray_ignores_bodies_in_the_way_and_they_still_land() {
        let mut world = world();
        let terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 8.0 })).unwrap();
        // two overlapping balls right on the ray
        let mut balls = Vec::new();
        for y in [10.0, 10.5] {
            balls.push(
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::sphere(0.5))
                            .position(GridPosition::from_local(Vec3::new(12.0, y, 12.0)).unwrap()),
                    )
                    .unwrap(),
            );
        }
        let ray = Ray {
            origin: GridPosition::from_local(Vec3::new(12.0, 20.0, 12.0)).unwrap(),
            direction: Vec3::NEG_Y,
            max_distance: 50.0,
        };
        let hit = terrain.raycast(&world, &ray).unwrap().unwrap();
        assert!(terrain.bodies().any(|body| body == hit.body));
        assert!((hit.distance - 12.0).abs() < 0.01, "{}", hit.distance);
        let up = Ray {
            direction: Vec3::Y,
            ..ray
        };
        assert_eq!(terrain.raycast(&world, &up).unwrap(), None);
        // the bit changes nothing for collisions
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        let y = world.body_state(balls[0]).unwrap().position.to_dvec3().y;
        assert!((y - 8.5).abs() < 0.1, "{y}");
    }

    fn ball_at(world: &mut PhysicsWorld, at: Vec3) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.4))
                    .position(GridPosition::from_local(at).unwrap()),
            )
            .unwrap()
    }

    fn height(world: &PhysicsWorld, body: BodyHandle) -> f64 {
        world.body_state(body).unwrap().position.to_dvec3().y
    }

    #[test]
    fn filling_over_bodies_lifts_them_out_and_they_stay_up() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 8.0 })).unwrap();
        let first = ball_at(&mut world, Vec3::new(12.0, 8.5, 12.0));
        let second = ball_at(&mut world, Vec3::new(12.0, 9.4, 12.0));
        let away = ball_at(&mut world, Vec3::new(20.0, 8.5, 20.0));
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        let report = terrain
            .edit(
                &mut world,
                Edit {
                    shape: EditShape::Sphere {
                        center: Vec3::new(12.0, 9.0, 12.0),
                        radius: 2.5,
                    },
                    op: EditOp::Add,
                },
            )
            .unwrap();
        assert_eq!(report.ejected, 2);
        // on top of the lump, not inside it or inside each other
        let (a, b) = (height(&world, first), height(&world, second));
        assert!(a > 11.5 && b > 11.5, "{a} {b}");
        assert!((a - b).abs() >= 0.79, "{a} {b}");
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        for body in [first, second, away] {
            let y = height(&world, body);
            assert!(y > 8.0, "{y}");
            let at = world
                .body_state(body)
                .unwrap()
                .position
                .to_dvec3()
                .as_vec3();
            assert!(terrain.volume().distance(at) > -0.1, "{at:?}");
        }
    }

    #[test]
    fn closing_a_tunnel_lifts_what_was_inside_it() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 8.0 })).unwrap();
        let tunnel = terrain
            .edit(
                &mut world,
                Edit {
                    shape: EditShape::Capsule {
                        a: Vec3::new(4.0, 4.0, 12.0),
                        b: Vec3::new(20.0, 4.0, 12.0),
                        radius: 1.5,
                    },
                    op: EditOp::Tunnel,
                },
            )
            .unwrap();
        assert_eq!(tunnel.ejected, 0);
        let inside = ball_at(&mut world, Vec3::new(12.0, 3.5, 12.0));
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        let closed = terrain.remove_tunnel(&mut world, tunnel.edit).unwrap();
        assert_eq!(closed.ejected, 1);
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        assert!(
            (height(&world, inside) - 8.4).abs() < 0.1,
            "{}",
            height(&world, inside)
        );
    }

    #[test]
    fn the_ground_is_closed_all_around() {
        let mut world = world();
        let terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        assert_matches_reference(terrain.volume());
        let key = |p: Vec3| {
            (
                (p.x * 1e3).round() as i64,
                (p.y * 1e3).round() as i64,
                (p.z * 1e3).round() as i64,
            )
        };
        let mut edges: HashMap<_, i32> = HashMap::new();
        for (_, corner, _, mesh) in terrain.volume().surface_bricks() {
            for triangle in &mesh.triangles {
                let p = triangle.map(|i| key(mesh.vertices[i as usize] + corner));
                for (a, b) in [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])] {
                    let sign = if a < b { 1 } else { -1 };
                    *edges.entry((a.min(b), a.max(b))).or_default() += sign;
                }
            }
        }
        let open = edges.values().filter(|count| **count != 0).count();
        assert_eq!(open, 0, "{open} open edges of {}", edges.len());
    }

    fn dig_at(x: f32, z: f32) -> Edit {
        Edit {
            shape: EditShape::Sphere {
                center: Vec3::new(x, 6.0, z),
                radius: 2.0,
            },
            op: EditOp::Excavate,
        }
    }

    #[test]
    fn queued_edits_match_edits_made_one_by_one() {
        let (mut alone, mut queued) = (world(), world());
        let mut one = SdfTerrain::new(&mut alone, desc(hills())).unwrap();
        let mut batch = SdfTerrain::new(&mut queued, desc(hills())).unwrap();
        for edit in edits() {
            one.edit(&mut alone, edit).unwrap();
            batch.queue_edit(edit).unwrap();
        }
        for i in 0..6 {
            let edit = dig_at(5.0 + i as f32 * 1.5, 9.0);
            one.edit(&mut alone, edit).unwrap();
            batch.queue_edit(edit).unwrap();
        }
        let mut commits = 0;
        while let Some(job) = batch.next_job() {
            assert!(batch.job_out());
            assert!(batch.next_job().is_none());
            let built = job.run(&Threads).unwrap();
            batch.commit(&mut queued, built).unwrap();
            commits += 1;
        }
        assert_eq!(batch.pending(), 0);
        // the last digs share one job
        assert!(commits < edits().len() + 6, "{commits}");
        for key in all_keys(one.volume()) {
            assert_eq!(
                one.volume().brick_mesh(key),
                batch.volume().brick_mesh(key),
                "{key:?}"
            );
        }
        assert_matches_reference(batch.volume());
        assert_eq!(
            one.volume().material_volume(),
            batch.volume().material_volume()
        );
    }

    #[test]
    fn a_step_never_sees_an_edit_before_it_commits() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 8.0 })).unwrap();
        let ball = ball_at(&mut world, Vec3::new(12.0, 8.5, 12.0));
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        let revision = terrain.volume().revision();
        terrain
            .queue_edit(Edit {
                shape: EditShape::Box {
                    center: Vec3::new(12.0, 6.0, 12.0),
                    half_extents: Vec3::new(1.5, 2.5, 1.5),
                    rotation: Quat::IDENTITY,
                },
                op: EditOp::Excavate,
            })
            .unwrap();
        let job = terrain.next_job().unwrap();
        // the world goes on while the job is out
        let built = job.run(&Threads).unwrap();
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        assert_eq!(terrain.volume().revision(), revision);
        assert!(
            (height(&world, ball) - 8.4).abs() < 0.05,
            "{}",
            height(&world, ball)
        );
        let report = terrain.commit(&mut world, built).unwrap();
        assert_eq!(report.edits, 1);
        assert!(report.removed_mass > 0.0);
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        assert!(
            (height(&world, ball) - 3.9).abs() < 0.1,
            "{}",
            height(&world, ball)
        );
    }

    #[test]
    fn stale_results_and_busy_calls_are_refused() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        terrain.queue_edit(dig_at(6.0, 6.0)).unwrap();
        assert_eq!(
            terrain.edit(&mut world, dig_at(9.0, 9.0)).map(|_| ()),
            Err(PhysicsError::TerrainBusy)
        );
        let built = terrain.next_job().unwrap().run(&Sequential).unwrap();
        terrain.abandon();
        let revision = terrain.volume().revision();
        assert_eq!(
            terrain.commit(&mut world, built).map(|_| ()),
            Err(PhysicsError::InvalidHandle)
        );
        assert_eq!(terrain.volume().revision(), revision);
        assert_eq!(terrain.pending(), 0);
        assert_eq!(terrain.volume().edit_count(), 0);
        assert_matches_reference(terrain.volume());

        // a tunnel can be closed while it still waits, and only once
        let tunnel = terrain.queue_edit(edits()[1]).unwrap();
        terrain.queue_close_tunnel(tunnel).unwrap();
        assert_eq!(
            terrain.queue_close_tunnel(tunnel),
            Err(PhysicsError::InvalidHandle)
        );
        while let Some(job) = terrain.next_job() {
            let built = job.run(&Sequential).unwrap();
            terrain.commit(&mut world, built).unwrap();
        }
        assert_eq!(terrain.volume().edit_count(), 0);
        assert_matches_reference(terrain.volume());
    }

    #[test]
    fn a_failed_build_can_be_dropped_and_work_goes_on() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        terrain.queue_edit(dig_at(6.0, 6.0)).unwrap();
        let job = terrain.next_job().unwrap();
        assert_eq!(
            job.run(&Lazy).map(|_| ()),
            Err(PhysicsError::DispatchIncomplete)
        );
        terrain.abandon();
        terrain.edit(&mut world, dig_at(9.0, 9.0)).unwrap();
        assert_eq!(terrain.volume().edit_count(), 1);
        assert_matches_reference(terrain.volume());
    }

    #[test]
    fn nothing_digs_through_the_floor() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 8.0 })).unwrap();
        let shaft = Edit {
            shape: EditShape::Capsule {
                a: Vec3::new(12.0, -5.0, 12.0),
                b: Vec3::new(12.0, 20.0, 12.0),
                radius: 2.0,
            },
            op: EditOp::Excavate,
        };
        terrain.edit(&mut world, shaft).unwrap();
        let volume = terrain.volume();
        assert!(volume.distance(Vec3::new(12.0, 0.5, 12.0)) < 0.0);
        assert!(volume.distance(Vec3::new(12.0, 3.0, 12.0)) > 0.0);
        assert_matches_reference(volume);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.4))
                    .position(GridPosition::from_local(Vec3::new(12.0, 6.0, 12.0)).unwrap()),
            )
            .unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        let y = world.body_state(ball).unwrap().position.to_dvec3().y;
        assert!((y - 1.4).abs() < 0.1, "{y}");
    }

    #[test]
    fn repeated_edits_do_not_grow_the_world() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        let bodies = world.body_count();
        let mut slots = 0;
        for round in 0..30 {
            let tunnel = terrain.edit(&mut world, edits()[1]).unwrap();
            terrain.remove_tunnel(&mut world, tunnel.edit).unwrap();
            assert_eq!(world.body_count(), bodies);
            if round == 0 {
                slots = world.bodies.assets.meshes.len();
            }
        }
        assert_eq!(world.bodies.assets.meshes.len(), slots);
        assert_eq!(terrain.volume().edit_count(), 0);
        terrain.remove(&mut world);
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn a_ball_rests_on_the_ground_and_falls_into_a_new_hole() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 8.0 })).unwrap();
        let start = Vec3::new(12.0, 9.0, 12.0);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.4))
                    .position(GridPosition::from_local(start).unwrap()),
            )
            .unwrap();
        let height = |world: &PhysicsWorld| world.body_state(ball).unwrap().position.to_dvec3().y;
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        assert!((height(&world) - 8.4).abs() < 0.05, "{}", height(&world));
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        terrain
            .edit(
                &mut world,
                Edit {
                    shape: EditShape::Box {
                        center: Vec3::new(12.0, 6.0, 12.0),
                        half_extents: Vec3::new(1.5, 2.5, 1.5),
                        rotation: Quat::IDENTITY,
                    },
                    op: EditOp::Excavate,
                },
            )
            .unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        assert!((height(&world) - 3.9).abs() < 0.1, "{}", height(&world));
    }

    #[test]
    fn bad_input_changes_nothing() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        let revision = terrain.volume().revision();
        let bad = Edit {
            shape: EditShape::Sphere {
                center: Vec3::ZERO,
                radius: f32::NAN,
            },
            op: EditOp::Excavate,
        };
        assert_eq!(
            terrain.edit(&mut world, bad).map(|_| ()),
            Err(PhysicsError::InvalidShape)
        );
        assert_eq!(terrain.volume().revision(), revision);
        assert_eq!(terrain.volume().edit_count(), 0);
        let far = Edit {
            shape: EditShape::Sphere {
                center: Vec3::splat(-500.0),
                radius: 1.0,
            },
            op: EditOp::Excavate,
        };
        assert_eq!(terrain.edit(&mut world, far).unwrap().bricks, 0);
        for broken in [
            SdfDesc {
                voxel: 0.0,
                ..desc(hills())
            },
            SdfDesc {
                brick_cells: 2,
                ..desc(hills())
            },
            SdfDesc {
                bricks: [0, 1, 1],
                ..desc(hills())
            },
            SdfDesc {
                bricks: [2048, 2048, 2048],
                ..desc(hills())
            },
        ] {
            assert!(SdfTerrain::new(&mut world, broken).is_err());
        }
    }
}
