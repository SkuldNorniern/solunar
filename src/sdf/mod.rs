mod compute;
mod field;
mod jobs;
mod nets;
mod terrain;
mod volume;

use crate::{GridPosition, Material, PhysicsError};

pub use compute::{BrickSamples, CpuEdits, EditSpan, SdfCompute};
pub use field::{Base, Edit, EditOp, EditShape};
pub use jobs::{BrickJob, BuiltBricks};
pub use nets::BrickMesh;
pub use terrain::{PreparedTerrain, SdfTerrain};
pub use volume::SdfVolume;

const MAX_BRICKS: u64 = 1 << 20;

/// Layer bit brick bodies carry besides the default one, so a query can ask for the ground alone.
/// Bodies a host puts on this bit look like ground to [`SdfTerrain::raycast`].
pub const GROUND_LAYER: u32 = 1 << 31;

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
    pub(crate) fn validate(&self) -> Result<(), PhysicsError> {
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

#[cfg(test)]
mod tests {
    use crate::dispatch::testing::{SkipsJobs, Threads};
    use crate::world::testing::world;
    use std::collections::HashMap;

    use algebrix::Quat;

    use super::*;
    use crate::{BodyHandle, PhysicsWorld, Ray, RigidBodyDesc, Sequential, Shape};
    use algebrix::Vec3;

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

    #[test]
    fn workers_give_the_same_ground_as_one_thread() {
        let (mut alone, mut shared) = (world(), world());
        let mut one = SdfTerrain::new(&mut alone, desc(hills())).unwrap();
        let mut many = SdfTerrain::new_with(&mut shared, desc(hills()), &Threads(4)).unwrap();
        for edit in edits() {
            let a = one.edit(&mut alone, edit).unwrap();
            let b = many.edit_with(&mut shared, edit, &Threads(4)).unwrap();
            assert_eq!(a, b);
        }
        let tunnel = EditId(1);
        one.remove_tunnel(&mut alone, tunnel).unwrap();
        many.remove_tunnel_with(&mut shared, tunnel, &Threads(4))
            .unwrap();
        for key in all_keys(one.volume()) {
            assert_eq!(one.volume().brick_mesh(key), many.volume().brick_mesh(key));
        }
        assert_matches_reference(many.volume());

        let revision = many.volume().revision();
        assert_eq!(
            many.edit_with(
                &mut shared,
                edits()[0],
                &SkipsJobs {
                    parallelism: 1,
                    skip: |index| index == 0
                }
            )
            .map(|_| ()),
            Err(PhysicsError::DispatchIncomplete)
        );
        assert_eq!(many.volume().revision(), revision);
        assert!(
            SdfVolume::new_with(
                desc(hills()),
                &SkipsJobs {
                    parallelism: 1,
                    skip: |index| index == 0
                }
            )
            .is_err()
        );
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
            let built = job.run(&Threads(4)).unwrap();
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

    /// A device that writes nonsense and then turns the work down.
    struct Lost;

    impl SdfCompute for Lost {
        fn apply(&self, bricks: &mut [BrickSamples<'_>]) -> bool {
            for brick in bricks.iter_mut() {
                brick.values.fill(f32::NAN);
            }
            false
        }
    }

    #[test]
    fn edits_applied_by_a_compute_give_the_same_terrain() {
        let run = |compute: Option<&dyn SdfCompute>| {
            let mut world = world();
            let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
            for edit in edits() {
                terrain.queue_edit(edit).unwrap();
            }
            for i in 0..6 {
                terrain
                    .queue_edit(dig_at(5.0 + i as f32 * 1.5, 9.0))
                    .unwrap();
            }
            while let Some(job) = terrain.next_job() {
                let built = match compute {
                    Some(compute) => job.run_with(&Threads(4), compute),
                    None => job.run(&Threads(4)),
                }
                .unwrap();
                terrain.commit(&mut world, built).unwrap();
            }
            terrain
        };
        let plain = run(None);
        for compute in [&CpuEdits as &dyn SdfCompute, &Lost] {
            let other = run(Some(compute));
            for key in all_keys(plain.volume()) {
                assert_eq!(
                    plain.volume().brick_mesh(key),
                    other.volume().brick_mesh(key),
                    "{key:?}"
                );
            }
            assert_eq!(
                plain.volume().material_volume(),
                other.volume().material_volume()
            );
            assert_matches_reference(other.volume());
        }
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
        let built = job.run(&Threads(4)).unwrap();
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
            job.run(&SkipsJobs {
                parallelism: 1,
                skip: |index| index == 0
            })
            .map(|_| ()),
            Err(PhysicsError::DispatchIncomplete)
        );
        terrain.abandon();
        terrain.edit(&mut world, dig_at(9.0, 9.0)).unwrap();
        assert_eq!(terrain.volume().edit_count(), 1);
        assert_matches_reference(terrain.volume());
    }

    #[test]
    fn quick_distance_agrees_with_distance_on_samples() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        for edit in edits() {
            terrain.edit(&mut world, edit).unwrap();
        }
        let volume = terrain.volume();
        let step = volume.desc.voxel;
        for z in (0..96).step_by(5) {
            for y in (0..64).step_by(3) {
                for x in (0..96).step_by(5) {
                    let at = Vec3::new(x as f32, y as f32, z as f32) * step;
                    let (full, quick) = (volume.distance(at), volume.quick_distance(at));
                    assert_eq!(full < 0.0, quick < 0.0, "{at:?} {full} {quick}");
                    // kept samples may skip an edit far away, near a surface they are exact
                    if full.abs() < step {
                        assert_eq!(full, quick, "{at:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn rescue_lifts_what_got_through_and_leaves_the_rest() {
        let mut world = world();
        let terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 8.0 })).unwrap();
        let resting = ball_at(&mut world, Vec3::new(4.0, 8.5, 4.0));
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        assert_eq!(terrain.rescue(&mut world).unwrap(), 0);
        let inside = ball_at(&mut world, Vec3::new(12.0, 5.0, 12.0));
        let under = ball_at(&mut world, Vec3::new(14.0, -3.0, 14.0));
        // off the side of the block is not ours
        let beside = ball_at(&mut world, Vec3::new(-5.0, 3.0, 12.0));
        let rest_y = height(&world, resting);
        assert_eq!(terrain.rescue(&mut world).unwrap(), 2);
        assert!(height(&world, inside) > 8.3, "{}", height(&world, inside));
        assert!(height(&world, under) > 8.3, "{}", height(&world, under));
        assert!((height(&world, beside) - 3.0).abs() < 1e-3);
        assert!((height(&world, resting) - rest_y).abs() < 1e-3);
        for _ in 0..120 {
            world.step_fixed().unwrap();
            terrain.rescue(&mut world).unwrap();
        }
        for body in [inside, under, resting] {
            assert!(
                (height(&world, body) - 8.4).abs() < 0.1,
                "{}",
                height(&world, body)
            );
        }
    }

    fn hole_at(x: f32) -> EditShape {
        EditShape::Sphere {
            center: Vec3::new(x, 6.0, 12.0),
            radius: 2.5,
        }
    }

    fn drain(terrain: &mut SdfTerrain, world: &mut PhysicsWorld) -> usize {
        let mut jobs = 0;
        while let Some(job) = terrain.next_job() {
            let built = job.run(&Threads(4)).unwrap();
            terrain.commit(world, built).unwrap();
            jobs += 1;
        }
        jobs
    }

    #[test]
    fn a_moved_tunnel_is_like_one_made_there_and_the_ground_closes_behind() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 12.0 })).unwrap();
        let start = terrain.volume().material_volume();
        // an edit before and after it, so its place in the order matters
        terrain.edit(&mut world, dig_at(6.0, 12.0)).unwrap();
        let tunnel = terrain
            .edit(
                &mut world,
                Edit {
                    shape: hole_at(4.0),
                    op: EditOp::Tunnel,
                },
            )
            .unwrap()
            .edit;
        terrain
            .edit(
                &mut world,
                Edit {
                    shape: EditShape::Sphere {
                        center: Vec3::new(8.0, 6.0, 12.0),
                        radius: 1.5,
                    },
                    op: EditOp::Add,
                },
            )
            .unwrap();
        for x in [6.0, 9.0, 14.0, 20.0] {
            terrain.queue_move_tunnel(tunnel, hole_at(x)).unwrap();
            drain(&mut terrain, &mut world);
            assert_matches_reference(terrain.volume());
        }
        assert!(terrain.volume().distance(Vec3::new(20.0, 6.0, 12.0)) > 0.0);
        // far behind it the rock is back
        assert!(terrain.volume().distance(Vec3::new(14.0, 6.0, 12.0)) < 0.0);
        terrain.remove_tunnel(&mut world, tunnel).unwrap();
        assert_matches_reference(terrain.volume());
        assert!(terrain.volume().material_volume() < start);
    }

    #[test]
    fn moves_faster_than_jobs_collapse_into_the_last_one() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        let tunnel = terrain
            .edit(
                &mut world,
                Edit {
                    shape: hole_at(4.0),
                    op: EditOp::Tunnel,
                },
            )
            .unwrap()
            .edit;
        for i in 0..50 {
            terrain
                .queue_move_tunnel(tunnel, hole_at(4.0 + i as f32 * 0.3))
                .unwrap();
        }
        assert_eq!(terrain.pending(), 1);
        assert_eq!(drain(&mut terrain, &mut world), 1);
        assert_matches_reference(terrain.volume());
        assert!(
            terrain
                .volume()
                .distance(Vec3::new(4.0 + 49.0 * 0.3, 6.0, 12.0))
                > 0.0
        );
        assert!(terrain.volume().distance(Vec3::new(4.0, 6.0, 12.0)) < 0.0);

        // one waiting behind a job out stays apart from what the job holds
        terrain.queue_move_tunnel(tunnel, hole_at(6.0)).unwrap();
        let out = terrain.next_job().unwrap();
        terrain.queue_move_tunnel(tunnel, hole_at(8.0)).unwrap();
        terrain.queue_move_tunnel(tunnel, hole_at(10.0)).unwrap();
        assert_eq!(terrain.pending(), 2);
        terrain
            .commit(&mut world, out.run(&Threads(4)).unwrap())
            .unwrap();
        drain(&mut terrain, &mut world);
        assert!(terrain.volume().distance(Vec3::new(10.0, 6.0, 12.0)) > 0.0);
        assert_matches_reference(terrain.volume());
    }

    #[test]
    fn only_open_tunnels_move() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(hills())).unwrap();
        let dug = terrain.edit(&mut world, dig_at(6.0, 6.0)).unwrap().edit;
        assert_eq!(
            terrain.queue_move_tunnel(dug, hole_at(9.0)),
            Err(PhysicsError::InvalidHandle)
        );
        let tunnel = terrain
            .queue_edit(Edit {
                shape: hole_at(4.0),
                op: EditOp::Tunnel,
            })
            .unwrap();
        // it can move while it still waits to land
        terrain.queue_move_tunnel(tunnel, hole_at(9.0)).unwrap();
        terrain.queue_close_tunnel(tunnel).unwrap();
        assert_eq!(
            terrain.queue_move_tunnel(tunnel, hole_at(12.0)),
            Err(PhysicsError::InvalidHandle)
        );
        assert_eq!(
            terrain.queue_close_tunnel(tunnel),
            Err(PhysicsError::InvalidHandle)
        );
        let bad = EditShape::Sphere {
            center: Vec3::ZERO,
            radius: f32::NAN,
        };
        assert_eq!(
            terrain.queue_move_tunnel(tunnel, bad),
            Err(PhysicsError::InvalidShape)
        );
        drain(&mut terrain, &mut world);
        assert_eq!(terrain.volume().edit_count(), 1);
        assert_matches_reference(terrain.volume());
    }

    #[test]
    fn a_passing_hole_drops_a_body_and_closing_lifts_it_back() {
        let mut world = world();
        let mut terrain = SdfTerrain::new(&mut world, desc(Base::Plane { height: 8.0 })).unwrap();
        let ball = ball_at(&mut world, Vec3::new(12.0, 8.5, 12.0));
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        let far = EditShape::Sphere {
            center: Vec3::new(4.0, 7.0, 12.0),
            radius: 2.5,
        };
        let tunnel = terrain
            .edit(
                &mut world,
                Edit {
                    shape: far,
                    op: EditOp::Tunnel,
                },
            )
            .unwrap()
            .edit;
        assert!((height(&world, ball) - 8.4).abs() < 0.05);
        let under = EditShape::Sphere {
            center: Vec3::new(12.0, 7.0, 12.0),
            radius: 2.5,
        };
        terrain.queue_move_tunnel(tunnel, under).unwrap();
        drain(&mut terrain, &mut world);
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        assert!(height(&world, ball) < 6.0, "{}", height(&world, ball));
        terrain.queue_move_tunnel(tunnel, far).unwrap();
        drain(&mut terrain, &mut world);
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        assert!(
            (height(&world, ball) - 8.4).abs() < 0.1,
            "{}",
            height(&world, ball)
        );
    }

    #[test]
    fn a_small_cache_only_costs_time() {
        let (mut big, mut small) = (world(), world());
        let mut plenty = SdfTerrain::new(&mut big, desc(hills())).unwrap();
        let mut tight = SdfTerrain::new(&mut small, desc(hills())).unwrap();
        // room for two bricks of 8 cells
        tight.set_cache_limit(2 * 10 * 10 * 10 * 4);
        for i in 0..30 {
            let edit = dig_at(3.0 + (i % 10) as f32 * 2.0, 3.0 + (i % 7) as f32 * 2.5);
            plenty.edit(&mut big, edit).unwrap();
            tight.edit(&mut small, edit).unwrap();
            let (count, bytes) = tight.volume().cached_bricks();
            assert!(count <= 2 && bytes <= 8000, "{count} {bytes}");
        }
        for key in all_keys(plenty.volume()) {
            assert_eq!(
                plenty.volume().brick_mesh(key),
                tight.volume().brick_mesh(key),
                "{key:?}"
            );
        }
        assert_matches_reference(tight.volume());
        // bricks with edits and no samples still answer right
        let step = tight.volume().desc.voxel;
        for x in (0..96).step_by(3) {
            let at = Vec3::new(x as f32, 12.0, 12.0) * step;
            let (full, quick) = (
                tight.volume().distance(at),
                tight.volume().quick_distance(at),
            );
            assert_eq!(full < 0.0, quick < 0.0, "{at:?} {full} {quick}");
        }
        tight.set_cache_limit(0);
        assert_eq!(tight.volume().cached_bricks(), (0, 0));
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
    fn a_tall_stack_of_heavy_blocks_stands_still_on_flat_ground() {
        let mut world = world();
        let desc = SdfDesc {
            origin: GridPosition::from_local(Vec3::new(-8.0, -4.0, -8.0)).unwrap(),
            voxel: 0.5,
            brick_cells: 16,
            bricks: [2, 1, 2],
            base: Base::Plane { height: 4.0 },
            material: Material::default(),
        };
        SdfTerrain::new(&mut world, desc).unwrap();
        // concrete blocks 2 by 1 by 3 m, eight high, about 115 t on one footprint over many
        // triangles
        let half = Vec3::new(1.0, 0.5, 1.5);
        let concrete = Material {
            density: 2400.0,
            friction: 1.0,
            restitution: 0.0,
            ..Material::default()
        };
        let placed: Vec<(BodyHandle, Vec3)> = (0..8)
            .map(|row| {
                let at = Vec3::new(4.0, 0.5 + row as f32, 4.0);
                let desc = RigidBodyDesc::dynamic(Shape::cuboid(half))
                    .material(concrete)
                    .local_position(at);
                (world.spawn(desc).unwrap(), at)
            })
            .collect();
        for _ in 0..180 {
            world.step_fixed().unwrap();
        }
        for (body, at) in placed {
            let moved = world.body_state(body).unwrap().position.local - at;
            assert!(
                Vec3::new(moved.x, 0.0, moved.z).length() < 0.01,
                "{moved:?}"
            );
            assert!(moved.y > -0.01, "{moved:?}");
        }
    }

    #[test]
    fn a_fine_voxel_edit_meshes() {
        let mut world = world();
        let desc = SdfDesc {
            origin: GridPosition::from_local(Vec3::new(-1.0, -1.0, -2.0)).unwrap(),
            voxel: 0.05,
            brick_cells: 16,
            bricks: [14, 4, 6],
            base: Base::Hills {
                seed: 7,
                height: 2.0,
                amplitude: 0.2,
                wavelength: 5.0,
                octaves: 3,
            },
            material: Material::default(),
        };
        let mut terrain = SdfTerrain::new(&mut world, desc).unwrap();
        let channel = Edit {
            shape: EditShape::Box {
                center: Vec3::new(2.8, 1.9, 2.0),
                half_extents: Vec3::new(1.8, 0.7, 0.35),
                rotation: Quat::IDENTITY,
            },
            op: EditOp::Excavate,
        };
        terrain.edit(&mut world, channel).unwrap();
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
