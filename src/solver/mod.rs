mod contacts;
mod islands;
mod rows;
mod stages;

use algebrix::Vec3;

use crate::body::Bodies;
use crate::collision::find_pairs;
use crate::joint::{
    JOINT_IMPULSES, Joints, add_joint_loads, build_rows, capture_impulses, joint_turning,
};
use crate::material::Material;
use crate::world::Integrated;
use crate::{BodyKind, JobDispatch, PhysicsError, Shape};

pub(crate) use contacts::{
    CachedManifold, CachedPoint, ContactScratch, FinishScratch, finish_contacts,
};
use contacts::{build_manifolds, refresh_manifolds};
use islands::{NONE, find_islands, group};
use stages::{Shared, color_constraints, factor_islands, plan_stages, run_stages, solve_block};

/// Contacts exist this far apart, so a corner lifting a hair keeps its constraint.
pub(crate) const CONTACT_MARGIN: f32 = 0.02;

/// Past this much travel in one step a pair counts as fast, and contacts reach as far as it moves.
const FAST_MOTION: f32 = 0.1;

/// Constraints that find no free color among these go in one more, solved in order.
pub(super) const COLORS: usize = 64;
/// Islands with fewer constraints keep the plain in order sweep, one island per job. Colors
/// carry load down a stack slower, a 16 box stack fell, so only big islands get them.
pub(super) const COLORED_ISLAND: usize = if cfg!(miri) { 8 } else { 1024 };

/// The part of a body the iterations change. Positions and turns stay put during them.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Motion {
    pub(crate) linear: Vec3,
    pub(crate) angular: Vec3,
    pub(crate) recovery_linear: Vec3,
    pub(crate) recovery_angular: Vec3,
    pub(crate) touched: bool,
}

/// The two bodies of one constraint. A dynamic one is the real body, a static or kinematic one
/// is a copy, the iterations never move it.
pub(crate) struct Pair<'a> {
    pub(crate) a: &'a mut Motion,
    pub(crate) b: &'a mut Motion,
}

/// How far the surface of a body can travel in one step, turning included.
fn step_motion(bodies: &Bodies, dense: usize, dt: f32) -> f32 {
    if bodies.kind[dense] == BodyKind::Static {
        return 0.0;
    }
    let shape = &bodies.shape[dense];
    let reach = if matches!(shape, Shape::TriMesh(_)) {
        0.0
    } else {
        bodies.assets.bounding_radius(shape)
    };
    (bodies.linear[dense].length() + bodies.angular[dense].length() * reach) * dt
}

/// Contacts of a pair exist this far apart. The bounds cover at least this much.
pub(crate) fn pair_margin(bodies: &Bodies, a: usize, b: usize, dt: f32) -> f32 {
    let motion = step_motion(bodies, a, dt) + step_motion(bodies, b, dt);
    if motion > FAST_MOTION {
        CONTACT_MARGIN + motion
    } else {
        CONTACT_MARGIN
    }
}

/// Straight travel of B against A this step. `None` when turning matters, a straight sweep
/// could not rule the contact out then.
pub(super) fn straight_motion(bodies: &Bodies, a: usize, b: usize, dt: f32) -> Option<Vec3> {
    let turning = |dense: usize| {
        let shape = &bodies.shape[dense];
        if matches!(shape, Shape::TriMesh(_)) {
            0.0
        } else {
            bodies.angular[dense].length() * bodies.assets.bounding_radius(shape) * dt
        }
    };
    (turning(a) + turning(b) <= CONTACT_MARGIN).then(|| (bodies.linear[b] - bodies.linear[a]) * dt)
}

/// With `every`, all overlapping pairs. Otherwise only pairs with a moving body, the rest stay
/// still, and `add_woken_pairs` fills in what wakes.
pub(crate) fn find_broad_pairs(
    bodies: &Bodies,
    scratch: &mut ContactScratch,
    every: bool,
    (dispatch, dt): (&dyn JobDispatch, f32),
) {
    let count = bodies.len();
    scratch.bounds.truncate(count);
    scratch.posed.truncate(count);
    for dense in 0..count {
        let pose = (
            bodies.position(dense),
            bodies.rotation[dense],
            bodies.shape[dense],
            CONTACT_MARGIN + 2.0 * step_motion(bodies, dense, dt),
        );
        if scratch.posed.get(dense) == Some(&pose) {
            continue;
        }
        let bound = bodies.assets.bounds(&pose.2, pose.0, pose.1, pose.3);
        if dense < scratch.posed.len() {
            scratch.bounds[dense] = bound;
            scratch.posed[dense] = pose;
        } else {
            scratch.bounds.push(bound);
            scratch.posed.push(pose);
        }
    }
    if every {
        find_pairs(
            &scratch.bounds,
            &mut scratch.order,
            &mut scratch.pairs,
            dispatch,
        );
    } else {
        let moving = |dense: usize| bodies.kind[dense] != BodyKind::Static && !bodies.asleep[dense];
        scratch
            .moving
            .find(&scratch.bounds, moving, &mut scratch.pairs, dispatch);
    }
}

/// Pairs for sleepers that `awake` wakes this step.
pub(crate) fn add_woken_pairs(bodies: &Bodies, awake: &[bool], scratch: &mut ContactScratch) {
    let woken = (0..bodies.len()).filter(|dense| bodies.asleep[*dense] && awake[*dense]);
    scratch
        .moving
        .add_woken(&scratch.bounds, woken, &mut scratch.pairs);
}

/// Most a body in contact turns in one substep, radians. Box and hull contact points are kept
/// on the body and carried round with it, past this they swing off where the bodies meet: a bar
/// spinning at 600 rad/s into a wall came off at 16,000 rad/s with four substeps.
const MOST_TURN: f32 = 0.4;
/// Most substeps a step takes for turning, past this a step is as good as it gets.
const MOST_SUBSTEPS: u32 = 64;
/// A body whose far edge moves less than this in one of the settings' substeps is left to them:
/// a shard spinning fast hardly moves its points, and counting it cost a fracture scene half
/// again its step.
const SMALL_SWING: f32 = 0.05;

/// Substeps this step needs so no dynamic body in contact turns more than [`MOST_TURN`] in one.
/// Balls are left out, their point is the one facing the other body whatever their turn. Bodies
/// turning in the open cost nothing.
pub(crate) fn turning_substeps(
    bodies: &Bodies,
    state: &[Integrated],
    scratch: &ContactScratch,
    (dt, substeps): (f32, u32),
) -> u32 {
    let part = dt / substeps.max(1) as f32;
    let mut fastest = 0.0_f32;
    for manifold in &scratch.manifolds {
        for dense in [manifold.a, manifold.b] {
            let shape = &bodies.shape[dense];
            if bodies.kind[dense] != BodyKind::Dynamic || matches!(shape, Shape::Sphere { .. }) {
                continue;
            }
            let spin = state[dense].angular.length();
            if spin * bodies.assets.bounding_radius(shape) * part > SMALL_SWING {
                fastest = fastest.max(spin);
            }
        }
    }
    ((fastest * dt / MOST_TURN).ceil() as u32).min(MOST_SUBSTEPS)
}

/// Finds contacts once per step and loads the working joint impulses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_contacts(
    bodies: &Bodies,
    awake: &[bool],
    joints: &Joints,
    state: &[Integrated],
    cache: &[CachedManifold],
    scratch: &mut ContactScratch,
    dispatch: &dyn JobDispatch,
    dt: f32,
) -> Result<(), PhysicsError> {
    build_manifolds(bodies, awake, joints, state, cache, scratch, dispatch, dt)?;
    find_islands(bodies, awake, joints, scratch);
    scratch.joint_impulses.clear();
    scratch
        .joint_impulses
        .extend(joints.slots.iter().map(|slot| {
            slot.state
                .map_or([0.0; JOINT_IMPULSES], |live| live.impulses)
        }));
    scratch.joint_loads.clear();
    scratch
        .joint_loads
        .resize(joints.slots.len(), (Vec3::ZERO, Vec3::ZERO));
    Ok(())
}

/// One substep of velocities. Joints rebuild from the moved bodies, contacts refresh their gaps.
pub(crate) fn solve_substep(
    bodies: &Bodies,
    awake: &[bool],
    joints: &Joints,
    state: &mut [Integrated],
    scratch: &mut ContactScratch,
    (substep, last): (u32, bool),
    (dispatch, dt): (&dyn JobDispatch, f32),
) {
    if substep == 0 {
        joint_turning(joints, bodies, dt, &mut scratch.joint_turning);
    }
    build_rows(
        joints,
        &scratch.joint_impulses,
        bodies,
        (state, &scratch.joint_turning),
        awake,
        &mut scratch.joint_points,
        &mut scratch.joint_rows,
        dt,
    );
    let islands = &mut scratch.islands;
    let count = islands.count();
    let of = &islands.of;
    let pair = |(a, b): (usize, usize)| if of[a] != NONE { of[a] } else { of[b] };
    group(
        &mut scratch.joint_points,
        &mut islands.spare_points,
        count,
        |point| pair(point.bodies()),
        &mut islands.point_ends,
    );
    group(
        &mut scratch.joint_rows,
        &mut islands.spare_rows,
        count,
        |row| pair(row.bodies()),
        &mut islands.row_ends,
    );
    refresh_manifolds(bodies, state, scratch, (substep == 0, last), (dispatch, dt));
    color_constraints(bodies, scratch);
    factor_islands(bodies, scratch);
    scratch.motions.clear();
    scratch.motions.extend(state.iter().map(|body| Motion {
        linear: body.linear,
        angular: body.angular,
        recovery_linear: body.recovery_linear,
        recovery_angular: body.recovery_angular,
        touched: body.touched,
    }));
    let shared = (
        Shared::new(&mut scratch.motions),
        Shared::new(&mut scratch.joint_points),
        Shared::new(&mut scratch.joint_rows),
        Shared::new(&mut scratch.manifolds),
        Shared::new(&mut scratch.points),
    );
    plan_stages(&mut scratch.colors, substep);
    let colors = &scratch.colors;
    let direct = (
        scratch.factors.as_slice(),
        scratch.shapes.as_slice(),
        scratch.direct_nodes.as_slice(),
    );
    run_stages(&colors.stages, dispatch, &|stage, block| {
        solve_block(bodies, colors, direct, &shared, stage, block, substep);
    });
    for (body, motion) in state.iter_mut().zip(&scratch.motions) {
        body.linear = motion.linear;
        body.angular = motion.angular;
        body.recovery_linear = motion.recovery_linear;
        body.recovery_angular = motion.recovery_angular;
        body.touched = motion.touched;
    }
    capture_impulses(
        &scratch.joint_points,
        &scratch.joint_rows,
        &mut scratch.joint_impulses,
    );
    add_joint_loads(
        &scratch.joint_points,
        &scratch.joint_rows,
        &mut scratch.joint_loads,
    );
    for (frame, point) in scratch.frames.iter_mut().zip(&scratch.points) {
        frame.pushed += point.impulse.x.max(0.0);
    }
}

/// The material a contact on this part of the body touches: a compound part's own when it has
/// one, the body's otherwise.
pub(crate) fn part_material(bodies: &Bodies, dense: usize, part: u32) -> Material {
    let own = bodies.material[dense];
    let Shape::Compound(id) = bodies.shape[dense] else {
        return own;
    };
    bodies
        .assets
        .compound(id)
        .and_then(|compound| compound.children.get(part as usize))
        .and_then(|child| child.surface)
        .unwrap_or(own)
}

#[cfg(test)]
mod tests {
    use crate::dispatch::testing::SkipsJobs;
    use crate::world::testing::{no_gravity, world};
    use algebrix::IVec3;

    use crate::{
        BodyHandle, CollisionFilter, FrameDesc, FrameId, GridPosition, JobDispatch, Material,
        PhysicsSettings, PhysicsWorld, Quat, RigidBodyDesc, Sequential, Shape, Vec3,
    };

    /// `threads` workers taking jobs off one counter.
    struct Workers(usize);

    impl JobDispatch for Workers {
        fn parallelism(&self) -> usize {
            self.0
        }

        fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..self.0 {
                    scope.spawn(|| {
                        loop {
                            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if index >= jobs {
                                break;
                            }
                            job(index);
                        }
                    });
                }
            });
        }
    }

    /// A pyramid with `base` boxes on the bottom row, all one island.
    fn pyramid(base: usize) -> (PhysicsWorld, BodyHandle, Vec3) {
        let mut world = world();
        ground(&mut world);
        let mut top = None;
        for row in 0..base {
            for column in 0..base - row {
                let x = column as f32 * 1.05 - (base - row) as f32 * 0.525;
                let at = Vec3::new(x, 0.5 + row as f32 * 1.0, 0.0);
                let body = world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5))).local_position(at),
                    )
                    .unwrap();
                top = Some((body, at));
            }
        }
        let (body, at) = top.unwrap();
        (world, body, at)
    }

    /// Top surface at y = 0.
    fn ground(world: &mut PhysicsWorld) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(500.0, 1.0, 500.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap()
    }

    fn run(world: &mut PhysicsWorld, seconds: f32) {
        let steps = (seconds / world.settings().fixed_timestep) as usize;
        for _ in 0..steps {
            world.step_fixed().unwrap();
        }
    }

    fn y(world: &PhysicsWorld, body: BodyHandle) -> f32 {
        world.body_state(body).unwrap().position.local.y
    }

    fn space() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap()
    }

    /// The furthest along x `mover` got over `steps` steps.
    fn furthest(world: &mut PhysicsWorld, mover: crate::BodyHandle, steps: usize) -> f64 {
        let mut most = f64::MIN;
        for _ in 0..steps {
            world.step_fixed().unwrap();
            most = most.max(world.body_state(mover).unwrap().position.to_dvec3().x);
        }
        most
    }

    fn plate(world: &mut PhysicsWorld, half: Vec3, at: GridPosition) {
        world
            .spawn(RigidBodyDesc::fixed(Shape::cuboid(half)).position(at))
            .unwrap();
    }

    #[test]
    fn a_ball_at_1000_m_s_stops_on_a_25_cm_plate() {
        let mut world = space();
        let at = GridPosition::from_local(Vec3::new(10.0, 0.0, 0.0)).unwrap();
        plate(&mut world, Vec3::new(0.125, 2.0, 2.0), at);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.05))
                    .density(7800.0)
                    .local_position(Vec3::new(-20.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(1000.0, 0.0, 0.0)),
            )
            .unwrap();
        // the face less the radius, never past it
        assert!(furthest(&mut world, ball, 30) <= 9.875 - 0.05 + 1e-3);
    }

    #[test]
    fn a_hit_across_a_cell_edge_is_caught() {
        let mut world = space();
        let at = GridPosition::from_dvec3(algebrix::DVec3::new(1024.0, 0.0, 0.0)).unwrap();
        plate(&mut world, Vec3::new(0.125, 2.0, 2.0), at);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.05))
                    .position(
                        GridPosition::from_dvec3(algebrix::DVec3::new(1000.0, 0.0, 0.0)).unwrap(),
                    )
                    .linear_velocity(Vec3::new(800.0, 0.0, 0.0)),
            )
            .unwrap();
        assert!(furthest(&mut world, ball, 30) <= 1023.875 - 0.05 + 1e-3);
    }

    #[test]
    fn two_boxes_at_300_m_s_meet_head_on() {
        let mut world = space();
        let shot = |world: &mut PhysicsWorld, x: f32, speed: f32| {
            world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                        .local_position(Vec3::new(x, 0.0, 0.0))
                        .linear_velocity(Vec3::new(speed, 0.0, 0.0)),
                )
                .unwrap()
        };
        let (a, b) = (
            shot(&mut world, -30.0, 300.0),
            shot(&mut world, 30.0, -300.0),
        );
        for _ in 0..20 {
            world.step_fixed().unwrap();
            let x = |body| world.body_state(body).unwrap().position.local.x;
            assert!(x(b) - x(a) >= 1.0 - 0.01, "{} {}", x(a), x(b));
        }
    }

    #[test]
    fn a_glancing_ball_or_capsule_slides_along_a_thin_plate() {
        // 10 degrees in at 500 m/s on a 5 cm plate, a slide that spins it up to 2000 rad/s
        for shape in [Shape::sphere(0.05), Shape::capsule(0.05, 0.1)] {
            let mut world = space();
            let at = GridPosition::from_local(Vec3::new(10.0, 0.0, 0.0)).unwrap();
            plate(&mut world, Vec3::new(0.025, 20.0, 20.0), at);
            let way = Vec3::new(
                10.0_f32.to_radians().sin(),
                0.0,
                10.0_f32.to_radians().cos(),
            );
            let body = world
                .spawn(
                    RigidBodyDesc::dynamic(shape)
                        .local_position(Vec3::new(9.0, 0.0, -10.0))
                        .linear_velocity(way * 500.0),
                )
                .unwrap();
            // into the face no more than the 5 mm the solver lets things rest in. it went 8 to
            // 11 cm in and out the back, its contact point carried round by its spin
            let deepest = furthest(&mut world, body, 4);
            assert!(deepest <= 9.975 - 0.05 + 5e-3, "{shape:?} {deepest}");
            let state = world.body_state(body).unwrap();
            assert!(state.linear_velocity.z > 300.0, "{shape:?} {state:?}");
            assert!(state.linear_velocity.x.abs() < 1.0, "{shape:?} {state:?}");
            assert!(
                state.angular_velocity.length() > 1000.0,
                "{shape:?} {state:?}"
            );
        }
    }

    #[test]
    fn a_bar_spinning_into_a_wall_never_comes_off_with_more_energy() {
        // with four substeps a bar at 600 rad/s came off at 16,000: its contact points, carried
        // round with it, swung over a radian a substep away from where it met the wall
        for spin in [50.0_f32, 300.0, 600.0, 1500.0] {
            let mut world = space();
            plate(
                &mut world,
                Vec3::new(0.02, 3.0, 3.0),
                GridPosition::from_local(Vec3::new(0.8, 0.0, 0.0)).unwrap(),
            );
            let mut desc = RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(1.0, 0.03, 0.03)))
                .rotation(Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2))
                .angular_velocity(Vec3::new(0.0, 0.0, spin));
            desc.can_sleep = false;
            let bar = world.spawn(desc).unwrap();
            let energy = |world: &PhysicsWorld| {
                let state = world.body_state(bar).unwrap();
                let (mass, inertia) = (7.2_f32, Vec3::new(0.00432, 2.4, 2.4));
                let local = state
                    .rotation
                    .conjugate()
                    .rotate_vec3(state.angular_velocity);
                0.5 * mass * state.linear_velocity.length_squared()
                    + 0.5 * (inertia * local).dot(local)
            };
            let start = energy(&world);
            let mut most = 0_u32;
            for _ in 0..60 {
                world.step_fixed().unwrap();
                most = most.max(world.last_substeps());
                let state = world.body_state(bar).unwrap();
                for side in [-1.0_f32, 1.0] {
                    let tip = state.position.to_dvec3().as_vec3()
                        + state.rotation.rotate_vec3(Vec3::X * side);
                    assert!(tip.x < 0.82 + 0.05 || tip.y.abs() > 2.9, "{spin} through");
                }
            }
            let end = energy(&world);
            assert!(
                end <= start * 1.01,
                "{spin}: {start} to {end}, {most} substeps"
            );
        }
    }

    #[test]
    fn a_plate_spinning_at_50_rad_s_hits_what_is_in_its_sweep() {
        let mut world = space();
        let mut desc = RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(1.0, 0.5, 0.025)))
            .density(5000.0)
            .angular_velocity(Vec3::new(0.0, 50.0, 0.0));
        desc.can_sleep = false;
        world.spawn(desc).unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.05))
                    .local_position(Vec3::new(0.7, 0.0, 0.0)),
            )
            .unwrap();
        let mut fastest = 0.0_f32;
        for _ in 0..30 {
            world.step_fixed().unwrap();
            fastest = fastest.max(world.body_state(ball).unwrap().linear_velocity.length());
        }
        // the plate's edge passes at 35 m/s there, the ball goes off with about that
        assert!(fastest > 20.0, "{fastest}");
        let gone = world.body_state(ball).unwrap().position.to_dvec3().length();
        assert!(gone > 1.0, "{gone}");
    }

    #[test]
    fn gap_inside_the_margin_does_not_stop_a_slow_fall_early() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .local_position(Vec3::new(0.0, 0.51, 0.0)),
            )
            .unwrap();
        world.step_fixed().unwrap();
        // one step of gravity still fits in the gap, so it falls freely
        let dt = world.settings().fixed_timestep;
        let speed = world.body_state(ball).unwrap().linear_velocity.y;
        assert!((speed + 9.81 * dt).abs() < 1e-4, "{speed}");
        run(&mut world, 1.0);
        assert!((y(&world, ball) - 0.5).abs() < 0.01);
    }

    #[test]
    fn fast_body_in_the_margin_is_stopped_and_bounces_once_it_arrives() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .material(Material {
                        restitution: 1.0,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(0.0, 0.51, 0.0))
                    .linear_velocity(Vec3::new(0.0, -10.0, 0.0)),
            )
            .unwrap();
        world.step_fixed().unwrap();
        let state = world.body_state(ball).unwrap();
        assert!(state.linear_velocity.y > 9.0, "{state:?}");
        assert!(state.position.local.y > 0.5, "{state:?}");
    }

    #[test]
    fn sphere_comes_to_rest_on_the_ground() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 2.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 3.0);
        assert!((y(&world, ball) - 0.5).abs() < 0.02, "{}", y(&world, ball));
        let speed = world.body_state(ball).unwrap().linear_velocity.length();
        assert!(speed < 0.05, "{speed}");
    }

    #[test]
    fn restitution_sets_the_rebound_height() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .material(Material {
                        restitution: 0.5,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(0.0, 5.5, 0.0)),
            )
            .unwrap();
        let mut bounced = false;
        let mut peak = 0.0_f32;
        for _ in 0..240 {
            world.step_fixed().unwrap();
            let velocity = world.body_state(ball).unwrap().linear_velocity.y;
            bounced |= velocity > 0.0;
            if bounced {
                peak = peak.max(y(&world, ball));
            }
        }
        // 5 m of fall at e = 0.5 comes back 1.25 m
        assert!((peak - 1.75).abs() < 0.2, "{peak}");
    }

    #[test]
    fn no_restitution_does_not_bounce() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 5.5, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.1);
        let mut peak = 0.0_f32;
        for _ in 0..120 {
            world.step_fixed().unwrap();
            peak = peak.max(y(&world, ball));
        }
        assert!(peak < 0.6, "{peak}");
    }

    #[test]
    fn friction_stops_a_sliding_box_at_the_right_distance() {
        let mut world = world();
        ground(&mut world);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(0.0, 0.5, 0.0))
                    .linear_velocity(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 3.0);
        let state = world.body_state(cube).unwrap();
        assert!(state.linear_velocity.length() < 0.05);
        // v^2 / (2 mu g) with mu = 0.5
        let expected = 25.0 / 9.81;
        assert!(
            (state.position.local.x - expected).abs() < expected * 0.15,
            "{}",
            state.position.local.x
        );
    }

    #[test]
    fn no_friction_keeps_sliding() {
        let mut world = world();
        ground(&mut world);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .material(Material {
                        friction: 0.0,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(0.0, 0.5, 0.0))
                    .linear_velocity(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        let velocity = world.body_state(cube).unwrap().linear_velocity.x;
        assert!((velocity - 5.0).abs() < 0.01, "{velocity}");
    }

    #[test]
    fn box_stack_stays_up() {
        let mut world = world();
        ground(&mut world);
        let mut boxes = Vec::new();
        for level in 0..5 {
            boxes.push(
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap(),
            );
        }
        run(&mut world, 5.0);
        for (level, body) in boxes.iter().enumerate() {
            let state = world.body_state(*body).unwrap();
            let expected = 0.5 + level as f32;
            assert!(
                (state.position.local.y - expected).abs() < 0.05,
                "level {level}: {}",
                state.position.local.y
            );
            assert!(state.position.local.x.abs() < 0.02);
            assert!(state.position.local.z.abs() < 0.02);
            assert!(state.linear_velocity.length() < 0.1);
        }
    }

    fn wall(world: &mut PhysicsWorld, x: f32) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(0.05, 5.0, 5.0)))
                    .local_position(Vec3::new(x, 0.0, 0.0)),
            )
            .unwrap()
    }

    #[test]
    fn fast_bodies_do_not_pass_a_thin_wall() {
        for speed in [50.0, 200.0, 1000.0, 5000.0] {
            for shape in [
                Shape::sphere(0.25),
                Shape::cuboid(Vec3::splat(0.25)),
                Shape::capsule(0.1, 0.5),
            ] {
                let mut world = no_gravity();
                wall(&mut world, 3.0);
                let bullet = world
                    .spawn(
                        RigidBodyDesc::dynamic(shape)
                            .local_position(Vec3::new(-20.0, 0.1, 0.2))
                            .linear_velocity(Vec3::new(speed, 0.0, 0.0)),
                    )
                    .unwrap();
                run(&mut world, 1.0);
                let at = world.body_state(bullet).unwrap().position.local.x;
                assert!(at < 2.95, "{shape:?} at {speed} m/s ended at {at}");
            }
        }
    }

    #[test]
    fn thin_bullet_stops_on_thin_plate_at_the_surface() {
        let mut world = no_gravity();
        wall(&mut world, 3.0);
        let bullet = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.02))
                    .local_position(Vec3::new(-10.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(800.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 0.5);
        let state = world.body_state(bullet).unwrap();
        assert!((state.position.local.x - 2.93).abs() < 0.01, "{state:?}");
        assert!(state.linear_velocity.x.abs() < 1.0, "{state:?}");
    }

    #[test]
    fn fast_ball_keeps_its_restitution() {
        let mut world = no_gravity();
        wall(&mut world, 3.0);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .material(Material {
                        restitution: 0.5,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(-20.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(200.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 0.5);
        let speed = world.body_state(ball).unwrap().linear_velocity.x;
        assert!((speed + 100.0).abs() < 2.0, "{speed}");
    }

    #[test]
    fn fast_ball_flying_past_a_corner_is_not_touched() {
        let mut world = no_gravity();
        world
            .spawn(RigidBodyDesc::fixed(Shape::cuboid(Vec3::splat(1.0))))
            .unwrap();
        // the path closes in on the box and leaves it 0.3 m clear of the corner
        let start = Vec3::new(-6.0, 2.3 - 0.5, 0.0);
        let velocity = Vec3::new(100.0, 100.0 * 0.5 / 6.0, 0.0);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.05))
                    .local_position(start)
                    .linear_velocity(velocity),
            )
            .unwrap();
        run(&mut world, 0.5);
        let state = world.body_state(ball).unwrap();
        assert_eq!(state.linear_velocity, velocity, "{state:?}");
    }

    #[test]
    fn fast_ball_hits_a_body_and_momentum_stays() {
        let mut world = no_gravity();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.1))
                    .density(1000.0)
                    .local_position(Vec3::new(-30.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(300.0, 0.0, 0.0)),
            )
            .unwrap();
        let target = world
            .spawn(RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(0.05, 1.0, 1.0))).density(1000.0))
            .unwrap();
        run(&mut world, 0.5);
        let mass_ball = 1000.0 * 4.0 / 3.0 * std::f32::consts::PI * 0.001;
        let mass_target = 1000.0 * 0.1 * 2.0 * 2.0;
        let total = world.body_state(ball).unwrap().linear_velocity.x * mass_ball
            + world.body_state(target).unwrap().linear_velocity.x * mass_target;
        let expected = 300.0 * mass_ball;
        assert!(
            (total - expected).abs() < expected * 1e-2,
            "{total} {expected}"
        );
        assert!(world.body_state(target).unwrap().linear_velocity.x > 0.1);
    }

    #[test]
    fn fast_spinning_bar_does_not_pass_a_ball() {
        let mut world = no_gravity();
        world
            .spawn(
                RigidBodyDesc::kinematic(Shape::cuboid(Vec3::new(4.0, 0.02, 0.2)))
                    .angular_velocity(Vec3::new(0.0, 0.0, 200.0)),
            )
            .unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.1)).local_position(Vec3::new(3.0, 0.5, 0.0)),
            )
            .unwrap();
        run(&mut world, 0.2);
        let speed = world.body_state(ball).unwrap().linear_velocity.length();
        assert!(speed > 1.0, "{speed}");
    }

    #[test]
    fn fast_hit_wakes_a_sleeping_stack_cleanly() {
        let mut world = world();
        ground(&mut world);
        let boxes: Vec<_> = (0..3)
            .map(|level| {
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap()
            })
            .collect();
        run(&mut world, 2.0);
        assert!(world.body_state(boxes[0]).unwrap().asleep);
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.2))
                    .density(5000.0)
                    .local_position(Vec3::new(-20.0, 1.5, 0.0))
                    .linear_velocity(Vec3::new(300.0, 0.0, 0.0)),
            )
            .unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
            for body in &boxes {
                let state = world.body_state(*body).unwrap();
                if state.asleep {
                    assert_eq!(state.linear_velocity, Vec3::ZERO, "{state:?}");
                }
            }
        }
        assert!(world.body_state(boxes[1]).unwrap().linear_velocity.x > 0.1);
    }

    #[test]
    fn fast_drop_stays_on_terrain() {
        let mut world = world();
        let heights = vec![0.0; 25];
        let shape = world.add_heightfield(5, 5, 4.0, 4.0, &heights).unwrap();
        world
            .spawn(RigidBodyDesc::fixed(shape).local_position(Vec3::new(-8.0, 0.0, -8.0)))
            .unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .local_position(Vec3::new(0.3, 40.0, 0.2))
                    .linear_velocity(Vec3::new(0.0, -500.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        let y = world.body_state(ball).unwrap().position.local.y;
        assert!((y - 0.25).abs() < 0.02, "{y}");
    }

    #[test]
    fn small_pile_on_threads_matches_one_thread() {
        // under miri the thresholds are tiny, so this takes the colored and threaded path
        let run = |dispatch: &dyn JobDispatch| {
            let (mut world, _, _) = pyramid(4);
            let ball = world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::sphere(0.3))
                        .local_position(Vec3::new(0.0, 6.0, 0.0)),
                )
                .unwrap();
            let post = world
                .spawn(RigidBodyDesc::fixed(Shape::sphere(0.1)))
                .unwrap();
            let joint = world
                .joint_at(
                    crate::JointDesc::ball(post, ball),
                    GridPosition::from_local(Vec3::new(0.0, 7.0, 0.0)).unwrap(),
                    Quat::IDENTITY,
                )
                .unwrap();
            world.add_joint(joint).unwrap();
            for _ in 0..if cfg!(miri) { 2 } else { 30 } {
                world.step_with(dispatch).unwrap();
            }
            world.state_hash()
        };
        assert_eq!(run(&Workers(3)), run(&Sequential));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_dispatcher_skipping_contact_jobs_fails_and_leaves_state() {
        // under the size where integration dispatches, so contact finding is what skips
        let (mut world, _, _) = pyramid(40);
        assert!(world.body_count() < 1024);
        world.step_fixed().unwrap();
        let before = world.state_hash();
        assert_eq!(
            world.step_with(&SkipsJobs {
                parallelism: 2,
                skip: |_| true
            }),
            Err(crate::PhysicsError::DispatchIncomplete)
        );
        assert_eq!(world.state_hash(), before);
        world.step_fixed().unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn big_pyramid_stands_when_colored() {
        let (mut world, top, start) = pyramid(40);
        world.step_fixed().unwrap();
        let constraints = world.island_stats().largest_constraints;
        assert!(constraints >= super::COLORED_ISLAND, "{constraints}");
        for _ in 0..300 {
            world.step_with(&Workers(4)).unwrap();
        }
        let now = world.body_state(top).unwrap().position.local;
        assert!((now - start).length() < 0.1, "{start:?} {now:?}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn colored_solve_is_the_same_on_any_thread_count() {
        let run = |dispatch: &dyn JobDispatch| {
            let (mut world, _, _) = pyramid(40);
            for _ in 0..60 {
                world.step_with(dispatch).unwrap();
            }
            world.state_hash()
        };
        let one = run(&Sequential);
        assert_eq!(run(&Workers(4)), one);
        assert_eq!(run(&Workers(16)), one);
    }

    #[test]
    fn sixteen_box_stack_stands() {
        let mut world = world();
        ground(&mut world);
        let boxes: Vec<_> = (0..16)
            .map(|level| {
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap()
            })
            .collect();
        run(&mut world, 10.0);
        for (level, body) in boxes.iter().enumerate() {
            let at = world.body_state(*body).unwrap().position.local;
            assert!((at.y - (0.5 + level as f32)).abs() < 0.05, "{level} {at:?}");
            assert!(at.x.hypot(at.z) < 0.1, "{level} {at:?}");
        }
    }

    #[test]
    fn bad_substep_counts_fail() {
        for substeps in [0, 65] {
            let settings = PhysicsSettings {
                substeps,
                ..PhysicsSettings::default()
            };
            assert!(PhysicsWorld::new(settings).is_err());
        }
    }

    #[test]
    fn equal_spheres_swap_velocity_when_elastic() {
        let mut world = no_gravity();
        let elastic = Material {
            restitution: 1.0,
            ..Material::default()
        };
        let sphere = |x: f32, speed: f32| {
            RigidBodyDesc::dynamic(Shape::sphere(0.5))
                .material(elastic)
                .local_position(Vec3::new(x, 0.0, 0.0))
                .linear_velocity(Vec3::new(speed, 0.0, 0.0))
        };
        let left = world.spawn(sphere(-2.0, 2.0)).unwrap();
        let right = world.spawn(sphere(2.0, -2.0)).unwrap();
        run(&mut world, 2.0);
        let left = world.body_state(left).unwrap().linear_velocity.x;
        let right = world.body_state(right).unwrap().linear_velocity.x;
        assert!((left + 2.0).abs() < 0.1, "{left}");
        assert!((right - 2.0).abs() < 0.1, "{right}");
        assert!((left + right).abs() < 1e-3);
    }

    #[test]
    fn light_body_hitting_heavy_one_conserves_momentum() {
        let mut world = no_gravity();
        let light = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .density(1.0)
                    .local_position(Vec3::new(-2.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(4.0, 0.0, 0.0)),
            )
            .unwrap();
        let heavy = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.5)).density(9.0))
            .unwrap();
        run(&mut world, 1.0);
        let left = world.body_state(light).unwrap().linear_velocity.x;
        let right = world.body_state(heavy).unwrap().linear_velocity.x;
        // masses 1 to 9, no bounce, both end up at 4 / 10
        assert!((left + 9.0 * right - 4.0).abs() < 1e-3, "{left} {right}");
        assert!((right - 0.4).abs() < 0.05, "{right}");
    }

    #[test]
    fn tilted_box_settles_flat() {
        let mut world = world();
        ground(&mut world);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(0.0, 1.5, 0.0))
                    .rotation(Quat::from_axis_angle(
                        Vec3::new(1.0, 0.0, 1.0).normalize(),
                        0.5,
                    )),
            )
            .unwrap();
        run(&mut world, 5.0);
        let state = world.body_state(cube).unwrap();
        assert!((state.position.local.y - 0.5).abs() < 0.02);
        assert!(state.angular_velocity.length() < 0.1);
        let up = state.rotation.rotate_vec3(Vec3::Y);
        let aligned = up.x.abs().max(up.y.abs()).max(up.z.abs());
        assert!(aligned > 0.999, "{up:?}");
    }

    #[test]
    fn embedded_body_recovers_without_exploding() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 0.1, 0.0)),
            )
            .unwrap();
        let mut fastest = 0.0_f32;
        for _ in 0..180 {
            world.step_fixed().unwrap();
            fastest = fastest.max(world.body_state(ball).unwrap().linear_velocity.length());
        }
        assert!(fastest <= 3.5, "{fastest}");
        assert!((y(&world, ball) - 0.5).abs() < 0.02);
    }

    #[test]
    fn filtered_pair_passes_through() {
        let mut world = world();
        ground(&mut world);
        let ghost = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .filter(CollisionFilter { layers: 2, mask: 2 })
                    .local_position(Vec3::new(0.0, 1.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        assert!(y(&world, ghost) < -1.0);
    }

    #[test]
    fn different_frames_do_not_collide() {
        let mut world = world();
        ground(&mut world);
        let other = world.add_frame(FrameDesc::new(FrameId::ROOT)).unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .frame(other)
                    .local_position(Vec3::new(0.0, 1.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        assert!(y(&world, ball) < -1.0);
    }

    #[test]
    fn contact_works_far_from_origin_and_across_cells() {
        let mut world = world();
        let cell = IVec3::new(2_000_000, -700_000, 1_000_000);
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(2000.0, 1.0, 10.0))).position(
                    GridPosition {
                        cell,
                        local: Vec3::new(0.0, -1.0, 0.0),
                    },
                ),
            )
            .unwrap();
        let next_cell = IVec3::new(cell.x + 1, cell.y, cell.z);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).position(GridPosition {
                    cell: next_cell,
                    local: Vec3::new(0.0, 1.0, 0.0),
                }),
            )
            .unwrap();
        run(&mut world, 3.0);
        let state = world.body_state(ball).unwrap();
        assert_eq!(state.position.cell, next_cell);
        assert!((state.position.local.y - 0.5).abs() < 0.02);
    }

    #[test]
    fn snapshot_in_mid_contact_continues_exactly() {
        let mut world = world();
        ground(&mut world);
        for level in 0..4 {
            world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                        .local_position(Vec3::new(0.05 * level as f32, 0.6 + level as f32, 0.0)),
                )
                .unwrap();
        }
        run(&mut world, 0.5);
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        run(&mut world, 1.0);
        run(&mut loaded, 1.0);
        assert_eq!(loaded.state_hash(), world.state_hash());
    }

    #[test]
    fn removing_a_body_mid_stack_keeps_the_rest_working() {
        let mut world = world();
        ground(&mut world);
        let mut boxes = Vec::new();
        for level in 0..3 {
            boxes.push(
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap(),
            );
        }
        run(&mut world, 1.0);
        world.despawn(boxes[0]).unwrap();
        run(&mut world, 2.0);
        assert!(
            (y(&world, boxes[1]) - 0.5).abs() < 0.05,
            "{}",
            y(&world, boxes[1])
        );
        assert!(
            (y(&world, boxes[2]) - 1.5).abs() < 0.05,
            "{}",
            y(&world, boxes[2])
        );
    }
}
