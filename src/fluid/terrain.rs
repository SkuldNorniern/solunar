use algebrix::Vec3;

use super::FluidWorld;
use super::boundary::BoundaryId;
use crate::error::PhysicsError;
use crate::sdf::SdfVolume;

/// Most lattice nodes one terrain region may sample, 64 MB of distances.
const MAX_NODES: usize = 1 << 24;

/// Signed distance to a terrain on the water's lattice, over one region, as the terrain stood
/// when sampled. Water reads it between nodes, the terrain itself is not touched per substep.
#[derive(Debug, Clone)]
pub(super) struct TerrainField {
    pub(super) owner: u32,
    low: Vec3,
    nodes: (usize, usize, usize),
    step: f32,
    distances: Vec<f32>,
}

impl TerrainField {
    fn at(&self, x: usize, y: usize, z: usize) -> f32 {
        self.distances[(y * self.nodes.2 + z) * self.nodes.0 + x]
    }

    /// Distance and the direction away from the terrain, between the nodes. `None` outside the
    /// region.
    pub(super) fn sample(&self, point: Vec3) -> Option<(f32, Vec3)> {
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
        let corner = |dx: usize, dy: usize, dz: usize| self.at(x + dx, y + dy, z + dz);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        // trilinear value, and its slope along each axis
        let along_x = |dy, dz| (corner(0, dy, dz), corner(1, dy, dz));
        let face = |dy: usize, dz: usize| {
            let (a, b) = along_x(dy, dz);
            (lerp(a, b, fx), b - a)
        };
        let (v00, g00) = face(0, 0);
        let (v10, g10) = face(1, 0);
        let (v01, g01) = face(0, 1);
        let (v11, g11) = face(1, 1);
        let (v0, v1) = (lerp(v00, v10, fy), lerp(v01, v11, fy));
        let distance = lerp(v0, v1, fz);
        let gx = lerp(lerp(g00, g10, fy), lerp(g01, g11, fy), fz);
        let gy = lerp(v10 - v00, v11 - v01, fz);
        let gz = v1 - v0;
        let normal = Vec3::new(gx, gy, gz).normalize_or(Vec3::Y);
        Some((distance, normal))
    }
}

impl FluidWorld {
    /// Lets water rest on and flow over a terrain inside the box from `low` to `high`, in the
    /// water's coordinates. `origin` is where the terrain's volume origin sits in them. Frozen
    /// water fills the ground under the surface as deep as the kernel reaches, and water closer
    /// than half a spacing to the surface is put back there. The terrain is sampled once: an edit
    /// after this is not seen until it is added again.
    ///
    /// # Errors
    /// Fails on a non finite or inside out box, or one with too many nodes or wall points.
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
        let step = self.settings.spacing;
        let count = |extent: f32| (extent / step).ceil().min(1e7) as usize + 1;
        let size = high - low;
        let nodes = (count(size.x), count(size.y), count(size.z));
        let total = nodes.0.saturating_mul(nodes.1).saturating_mul(nodes.2);
        if total > MAX_NODES {
            return Err(PhysicsError::FluidFull);
        }
        let reach = self.settings.radius();
        let mut distances = Vec::with_capacity(total);
        let mut points = Vec::new();
        for y in 0..nodes.1 {
            for z in 0..nodes.2 {
                for x in 0..nodes.0 {
                    let at = low + Vec3::new(x as f32, y as f32, z as f32) * step;
                    let distance = volume.distance(at - origin);
                    distances.push(distance);
                    if distance < 0.0 && distance >= -reach {
                        points.push(at);
                    }
                }
            }
        }
        let room = self
            .settings
            .max_particles
            .saturating_sub(self.wall_positions().len());
        if points.len() > room {
            return Err(PhysicsError::FluidFull);
        }
        let id = self.add_wall_points(points)?;
        self.terrains.push(TerrainField {
            owner: id.raw(),
            low,
            nodes,
            step,
            distances,
        });
        Ok(id)
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

    #[test]
    fn the_field_matches_the_terrain_between_nodes() {
        let (volume, origin) = trench();
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
        fluid
            .add_terrain(
                &volume,
                origin,
                Vec3::new(-0.4, -0.7, -0.5),
                Vec3::new(2.2, 0.3, 0.5),
            )
            .unwrap();
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
        assert!(!fluid.wall_positions().is_empty());
    }

    #[test]
    fn water_rests_in_a_trench_and_does_not_sink_into_it() {
        let (volume, origin) = trench();
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
        fluid
            .add_terrain(
                &volume,
                origin,
                Vec3::new(-0.4, -0.7, -0.5),
                Vec3::new(2.2, 0.3, 0.5),
            )
            .unwrap();
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
        assert!(fluid.wall_positions().is_empty());
    }
}
