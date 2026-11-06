use algebrix::Vec3;

use super::boundary::BoundaryId;
use super::{FluidWorld, Kernel};
use crate::error::PhysicsError;
use crate::sdf::SdfVolume;

/// Most lattice nodes one terrain region may sample, 64 MB of distances.
const MAX_NODES: usize = 1 << 24;
/// Samples of the damping table between half a spacing under a surface and the kernel's reach.
const MAP_SAMPLES: usize = 256;

/// Ground near one particle: how far and which way its surface is, the density the ground adds
/// there and how that density changes across space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Ground {
    pub(super) distance: f32,
    pub(super) normal: Vec3,
    pub(super) density: f32,
    pub(super) gradient: Vec3,
}

/// A terrain over one region, as it stood when sampled: the signed distance at nodes half a
/// spacing apart. A distance read between nodes rounds a concave crease, at a whole spacing that
/// put water resting in it a fifth too dense.
///
/// The ground counts as frozen water on the water's lattice carried with the particle, each site
/// as solid as its distance says, summed at the particle itself. At rest half a spacing off the
/// ground that is what box walls give, on flat ground, along edges and in corners. Frozen water
/// on a fixed grid sat at odd offsets from the water and threw it about, a density read from the
/// distance alone missed the second wall in a corner, and any density summed at the nodes read
/// wrong between them in corners.
#[derive(Debug, Clone)]
pub(super) struct TerrainField {
    pub(super) owner: u32,
    low: Vec3,
    nodes: (usize, usize, usize),
    step: f32,
    distances: Vec<f32>,
}

impl TerrainField {
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (y * self.nodes.2 + z) * self.nodes.0 + x
    }

    /// Trilinear value of `list` at `point`, and its change along each axis. `None` outside.
    fn read(&self, list: &[f32], point: Vec3) -> Option<(f32, Vec3)> {
        let local = (point - self.low) / self.step;
        let inside = |value: f32, nodes: usize| value >= 0.0 && value <= (nodes - 1) as f32;
        if !(inside(local.x, self.nodes.0)
            && inside(local.y, self.nodes.1)
            && inside(local.z, self.nodes.2))
        {
            return None;
        }
        let base = |value: f32, nodes: usize| (value.floor() as usize).min(nodes - 2);
        let (x, y, z) = (
            base(local.x, self.nodes.0),
            base(local.y, self.nodes.1),
            base(local.z, self.nodes.2),
        );
        let (fx, fy, fz) = (local.x - x as f32, local.y - y as f32, local.z - z as f32);
        let corner = |dx: usize, dy: usize, dz: usize| list[self.index(x + dx, y + dy, z + dz)];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let face = |dy: usize, dz: usize| {
            let (a, b) = (corner(0, dy, dz), corner(1, dy, dz));
            (lerp(a, b, fx), b - a)
        };
        let (v00, g00) = face(0, 0);
        let (v10, g10) = face(1, 0);
        let (v01, g01) = face(0, 1);
        let (v11, g11) = face(1, 1);
        let (v0, v1) = (lerp(v00, v10, fy), lerp(v01, v11, fy));
        let value = lerp(v0, v1, fz);
        let change = Vec3::new(
            lerp(lerp(g00, g10, fy), lerp(g01, g11, fy), fz),
            lerp(v10 - v00, v11 - v01, fz),
            v1 - v0,
        ) / self.step;
        Some((value, change))
    }

    /// Distance and the direction away from the terrain. `None` outside the region.
    pub(super) fn sample(&self, point: Vec3) -> Option<(f32, Vec3)> {
        let (distance, change) = self.read(&self.distances, point)?;
        Some((distance, change.normalize_or(Vec3::Y)))
    }

    fn ground(&self, point: Vec3, sites: &GroundSites) -> Option<Ground> {
        let (distance, change) = self.read(&self.distances, point)?;
        let (mut density, mut gradient) = (0.0, Vec3::ZERO);
        if distance < sites.reach {
            let open = distance - sites.spacing / 2.0;
            for (offset, weight, far) in &sites.offsets {
                // a distance moves no faster than the point, this site can not be in the ground
                if *far <= open {
                    continue;
                }
                let Some((there, toward)) = self.read(&self.distances, point + *offset) else {
                    continue;
                };
                let share = solid(there, sites.spacing);
                density += weight * share;
                if share > 0.0 && share < 1.0 {
                    gradient -= toward * (weight / sites.spacing);
                }
            }
        }
        Some(Ground {
            distance,
            normal: change.normalize_or(Vec3::Y),
            density,
            gradient,
        })
    }
}

/// What ground does to a particle, worked out once: the water's lattice around it within the
/// kernel with the density each site adds, and the weight of damping into flat ground at a
/// distance from its surface.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct GroundSites {
    spacing: f32,
    /// Past this distance from the surface no site is solid.
    reach: f32,
    /// Site, density it adds, how far from the particle.
    offsets: Vec<(Vec3, f32, f32)>,
    low: f32,
    step: f32,
    damping: Vec<f32>,
}

impl GroundSites {
    pub(super) fn new(kernel: &Kernel, mass: f32, spacing: f32, radius: f32) -> Self {
        let reach = (radius / spacing).ceil() as i32;
        let mut offsets = Vec::new();
        for z in -reach..=reach {
            for y in -reach..=reach {
                for x in -reach..=reach {
                    let offset = Vec3::new(x as f32, y as f32, z as f32) * spacing;
                    let far = offset.length();
                    if far < radius {
                        offsets.push((offset, mass * kernel.value(far), far));
                    }
                }
            }
        }
        let low = -spacing / 2.0;
        let step = (radius - low) / (MAP_SAMPLES - 1) as f32;
        let soften = 0.01 * radius * radius;
        let across_reach = reach + 1;
        let mut damping = Vec::with_capacity(MAP_SAMPLES);
        for sample in 0..MAP_SAMPLES {
            let distance = low + step * sample as f32;
            let mut weight = 0.0;
            let mut layer = 0;
            loop {
                let depth = distance + spacing * (layer as f32 + 0.5);
                if depth >= radius {
                    break;
                }
                for a in -across_reach..=across_reach {
                    for b in -across_reach..=across_reach {
                        let across = ((a * a + b * b) as f32) * spacing * spacing;
                        let squared = across + depth * depth;
                        let far = squared.sqrt();
                        if far >= radius || far <= 0.0 {
                            continue;
                        }
                        weight +=
                            mass * depth * depth * -kernel.slope(far) / (far * (squared + soften));
                    }
                }
                layer += 1;
            }
            damping.push(weight);
        }
        Self {
            spacing,
            reach: radius + spacing / 2.0,
            offsets,
            low,
            step,
            damping,
        }
    }

    /// Damping weight at `distance` from a surface. Zero out of reach.
    pub(super) fn damping(&self, distance: f32) -> f32 {
        let at = (distance.max(self.low) - self.low) / self.step;
        let below = at.floor() as usize;
        if below + 1 >= MAP_SAMPLES {
            return 0.0;
        }
        let part = at - below as f32;
        self.damping[below] + (self.damping[below + 1] - self.damping[below]) * part
    }
}

/// Solid share of a cell `size` across whose center is `distance` from the surface.
fn solid(distance: f32, size: f32) -> f32 {
    (0.5 - distance / size).clamp(0.0, 1.0)
}

impl FluidWorld {
    /// Lets water rest on and flow over a terrain inside the box from `low` to `high`, in the
    /// water's coordinates. `origin` is where the terrain's volume origin sits in them. The ground
    /// counts as frozen water under its surface, and water closer than half a spacing to the
    /// surface is put back there. Keep the box a kernel reach clear of the water on every side.
    /// The terrain is sampled once: an edit after this is not seen until it is added again.
    ///
    /// # Errors
    /// Fails on a non finite or inside out box, or one with too many nodes.
    pub fn add_terrain(
        &mut self,
        volume: &SdfVolume,
        origin: Vec3,
        low: Vec3,
        high: Vec3,
    ) -> Result<BoundaryId, PhysicsError> {
        if !(low.is_finite() && high.is_finite() && origin.is_finite()) {
            return Err(PhysicsError::NonFinite("terrain box"));
        }
        if high.x <= low.x || high.y <= low.y || high.z <= low.z {
            return Err(PhysicsError::OutOfRange);
        }
        let step = self.settings.spacing / 2.0;
        let count = |extent: f32| (extent / step).ceil().min(1e7) as usize + 1;
        let size = high - low;
        let nodes = (count(size.x), count(size.y), count(size.z));
        let total = nodes.0.saturating_mul(nodes.1).saturating_mul(nodes.2);
        if total > MAX_NODES {
            return Err(PhysicsError::FluidFull);
        }
        let mut distances = Vec::with_capacity(total);
        for y in 0..nodes.1 {
            for z in 0..nodes.2 {
                for x in 0..nodes.0 {
                    let at = low + Vec3::new(x as f32, y as f32, z as f32) * step;
                    distances.push(volume.distance(at - origin));
                }
            }
        }
        let mut field = TerrainField {
            owner: 0,
            low,
            nodes,
            step,
            distances,
        };
        let id = self.add_wall_points(Vec::new())?;
        field.owner = id.raw();
        self.terrains.push(field);
        Ok(id)
    }

    /// Ground near `at`, from each terrain region that holds it.
    pub(super) fn ground_near(&self, at: Vec3) -> impl Iterator<Item = Ground> + '_ {
        self.terrains
            .iter()
            .filter_map(move |field| field.ground(at, &self.ground_sites))
    }
}

/// Puts a particle that came closer to a terrain than `rest` back there. True when it got under
/// the surface.
pub(super) fn terrain_contact(
    terrains: &[TerrainField],
    rest: f32,
    at: &mut Vec3,
    velocity: &mut Vec3,
) -> bool {
    let mut under = false;
    for field in terrains {
        let Some((distance, normal)) = field.sample(*at) else {
            continue;
        };
        if distance < rest {
            under |= distance < 0.0;
            *at += normal * (rest - distance);
            let into = velocity.dot(normal);
            if into < 0.0 {
                *velocity -= normal * into;
            }
        }
    }
    under
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::Sequential;
    use crate::fluid::FluidSettings;
    use crate::material::Material;
    use crate::position::GridPosition;
    use crate::sdf::{Base, Edit, EditOp, EditShape, SdfDesc, SdfTerrain};
    use crate::world::{PhysicsSettings, PhysicsWorld};

    const TICK: f32 = 1.0 / 60.0;
    const LOW: Vec3 = Vec3::new(-0.4, -0.7, -0.5);
    const HIGH: Vec3 = Vec3::new(2.2, 0.3, 0.5);

    /// Flat ground at y = 0 with a trench along x from -0.1 to 1.9, its bed at y = -0.4, 0.6 wide.
    fn trench() -> (SdfVolume, Vec3) {
        let origin = Vec3::new(-0.5, -1.0, -1.0);
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let desc = SdfDesc {
            origin: GridPosition::from_local(origin).unwrap(),
            voxel: 0.1,
            brick_cells: 8,
            bricks: [4, 2, 3],
            base: Base::Plane { height: 1.0 },
            material: Material::default(),
        };
        let mut terrain = SdfTerrain::new(&mut world, desc).unwrap();
        let edit = Edit {
            shape: EditShape::Box {
                center: Vec3::new(1.4, 0.8, 1.0),
                half_extents: Vec3::new(1.0, 0.2, 0.3),
                rotation: algebrix::Quat::IDENTITY,
            },
            op: EditOp::Excavate,
        };
        terrain.edit(&mut world, edit).unwrap();
        (terrain.volume().clone(), origin)
    }

    fn in_trench() -> (FluidWorld, SdfVolume, Vec3) {
        let (volume, origin) = trench();
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
        fluid.add_terrain(&volume, origin, LOW, HIGH).unwrap();
        (fluid, volume, origin)
    }

    #[test]
    fn the_field_matches_the_terrain_between_nodes() {
        let (fluid, volume, origin) = in_trench();
        let field = &fluid.terrains[0];
        for at in [
            Vec3::new(0.5, -0.3, 0.0),
            Vec3::new(1.03, 0.07, 0.41),
            Vec3::new(0.2, -0.5, 0.2),
        ] {
            let (distance, _) = field.sample(at).unwrap();
            let truth = volume.distance(at - origin);
            assert!((distance - truth).abs() < 0.02, "{at:?} {distance} {truth}");
        }
        // on the trench bed the way out is up, on its side wall it is across
        let (_, up) = field.sample(Vec3::new(1.0, -0.38, 0.0)).unwrap();
        assert!(up.y > 0.9, "{up:?}");
        let (_, across) = field.sample(Vec3::new(1.0, -0.2, 0.28)).unwrap();
        assert!(across.z < -0.9, "{across:?}");
        assert!(field.sample(Vec3::new(9.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn ground_gives_what_a_box_wall_gives() {
        // a box wall is frozen water on the water's lattice, the trench bed seen from a particle
        // resting on it must give about the same
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
        fluid
            .add_box_container(Vec3::ZERO, Vec3::splat(0.5))
            .unwrap();
        let at = Vec3::new(0.225, 0.025, 0.225);
        let reach = fluid.settings.radius().powi(2);
        let wall: f32 = fluid
            .walls
            .positions
            .iter()
            .map(|point| (*point - at).length_squared())
            .filter(|squared| *squared < reach)
            .map(|squared| fluid.mass * fluid.kernel.value(squared.sqrt()))
            .sum();
        let (trenched, _, _) = in_trench();
        let bed = Vec3::new(1.0, -0.375, 0.0);
        let ground = trenched.ground_near(bed).next().unwrap();
        assert!(
            (ground.density - wall).abs() / wall < 0.05,
            "{} {wall}",
            ground.density
        );
        // it pushes up, away from the bed
        assert!(ground.gradient.y < 0.0 && ground.gradient.x.abs() < ground.gradient.y.abs() * 0.1);
        // in the corner of bed and side there is more of it
        let corner = trenched
            .ground_near(Vec3::new(1.0, -0.375, 0.275))
            .next()
            .unwrap();
        assert!(
            corner.density > ground.density * 1.2,
            "{} {}",
            corner.density,
            ground.density
        );
    }

    #[test]
    fn water_set_down_in_a_trench_does_not_jump() {
        let (mut fluid, _, _) = in_trench();
        // flush against the trench's end, bed and both sides
        fluid
            .fill_box(Vec3::new(-0.1, -0.4, -0.3), Vec3::new(0.5, -0.1, 0.3))
            .unwrap();
        let report = fluid.step(TICK, &Sequential).unwrap();
        assert!(report.max_speed < 0.5, "{report:?}");
        assert!(report.max_compression < 0.05, "{report:?}");
    }

    #[test]
    fn water_rests_in_a_trench_and_does_not_sink_into_it() {
        let (mut fluid, _, _) = in_trench();
        fluid
            .fill_box(Vec3::new(0.0, -0.4, -0.3), Vec3::new(0.6, -0.1, 0.3))
            .unwrap();
        let count = fluid.len();
        let mut held = 0;
        for _ in 0..150 {
            held += fluid.step(TICK, &Sequential).unwrap().held_back;
        }
        assert_eq!(fluid.len(), count);
        assert_eq!(held, 0, "water got under the ground");
        let deepest = fluid
            .positions()
            .iter()
            .fold(1.0_f32, |most, at| most.min(at.y));
        assert!(deepest > -0.4, "{deepest}");
        // it spread along the trench, and stayed between its walls
        let furthest = fluid
            .positions()
            .iter()
            .fold(0.0_f32, |most, at| most.max(at.x));
        assert!(furthest > 1.0, "{furthest}");
        // the walls come from a 10 cm voxel field, true to about half a voxel
        assert!(fluid.positions().iter().all(|at| at.z.abs() < 0.35));
    }

    #[test]
    fn a_terrain_region_is_checked_and_can_go_away() {
        let (volume, origin) = trench();
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
        assert!(
            fluid
                .add_terrain(&volume, origin, Vec3::ONE, Vec3::ZERO)
                .is_err()
        );
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(
            fluid
                .add_terrain(&volume, nan, Vec3::ZERO, Vec3::ONE)
                .is_err()
        );
        assert!(
            fluid
                .add_terrain(&volume, origin, Vec3::ZERO, Vec3::splat(1e6))
                .is_err()
        );
        let id = fluid
            .add_terrain(&volume, origin, Vec3::ZERO, Vec3::ONE)
            .unwrap();
        assert!(fluid.remove_boundary(id));
        assert!(fluid.terrains.is_empty());
    }
}
