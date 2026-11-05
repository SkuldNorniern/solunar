use algebrix::Vec3;

use crate::body::Bodies;
use crate::joint::Joints;
use crate::solver::CONTACT_MARGIN;
use crate::solver::CachedManifold;
use crate::{BodyHandle, BodyKind};

const SLEEP_TIME: f32 = 0.5;
const SLEEP_LINEAR_SQUARED: f32 = 0.05 * 0.05;
const SLEEP_ANGULAR_SQUARED: f32 = 0.05 * 0.05;

#[derive(Debug, Clone, Default)]
pub(crate) struct SleepScratch {
    awake: Vec<bool>,
    parent: Vec<u32>,
    flagged: Vec<bool>,
    supported: Vec<bool>,
}

impl SleepScratch {
    /// Bodies that run this step, by dense index.
    pub(crate) fn awake(&self) -> &[bool] {
        &self.awake
    }

    fn reset_islands(&mut self, count: usize) {
        self.parent.clear();
        self.parent.extend(0..count as u32);
    }

    fn find(&mut self, dense: usize) -> usize {
        let mut at = dense;
        while self.parent[at] as usize != at {
            let up = self.parent[at] as usize;
            self.parent[at] = self.parent[up];
            at = up;
        }
        at
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        self.parent[a.max(b)] = a.min(b) as u32;
    }
}

fn is_moving(bodies: &Bodies, awake: &[bool], dense: usize) -> bool {
    match bodies.kind[dense] {
        BodyKind::Dynamic => awake[dense],
        BodyKind::Kinematic => {
            bodies.linear[dense] != Vec3::ZERO || bodies.angular[dense] != Vec3::ZERO
        }
        BodyKind::Static => false,
    }
}

/// Fills `awake`. Anything moving that touches a sleeper wakes its whole sleeping island.
/// With `recheck`, a static or kinematic body newly touching a sleeper counts as well.
pub(crate) fn plan_wakes(
    bodies: &Bodies,
    cache: &[CachedManifold],
    joints: &Joints,
    pairs: &[(u32, u32)],
    recheck: bool,
    scratch: &mut SleepScratch,
) {
    scratch.awake.clear();
    scratch
        .awake
        .extend(bodies.asleep.iter().map(|asleep| !*asleep));
    if !bodies.asleep.contains(&true) {
        return;
    }
    let count = bodies.len();
    scratch.reset_islands(count);
    scratch.flagged.clear();
    scratch.flagged.resize(count, false);

    for entry in cache {
        let (Ok(a), Ok(b)) = (bodies.dense_index(entry.a), bodies.dense_index(entry.b)) else {
            continue;
        };
        if bodies.asleep[a] && bodies.asleep[b] {
            scratch.union(a, b);
        } else if bodies.asleep[a] && is_moving(bodies, &scratch.awake, b) {
            scratch.flagged[a] = true;
        } else if bodies.asleep[b] && is_moving(bodies, &scratch.awake, a) {
            scratch.flagged[b] = true;
        }
    }

    for (a, b, _) in joints.links(bodies) {
        if bodies.asleep[a] && bodies.asleep[b] {
            scratch.union(a, b);
        } else if bodies.asleep[a] && is_moving(bodies, &scratch.awake, b) {
            scratch.flagged[a] = true;
        } else if bodies.asleep[b] && is_moving(bodies, &scratch.awake, a) {
            scratch.flagged[b] = true;
        }
    }

    for &(first, second) in pairs {
        let (first, second) = (first as usize, second as usize);
        let (sleeper, other) = match (bodies.asleep[first], bodies.asleep[second]) {
            (true, false) => (first, second),
            (false, true) => (second, first),
            _ => continue,
        };
        let wakes = is_moving(bodies, &scratch.awake, other)
            || (recheck
                && bodies.kind[other] != BodyKind::Dynamic
                && !is_cached(cache, bodies.handle_at(sleeper), bodies.handle_at(other)));
        if wakes
            && bodies.frame[sleeper] == bodies.frame[other]
            && bodies.filter[sleeper].allows(bodies.filter[other])
            && !joints.excludes(bodies.handle_at(sleeper), bodies.handle_at(other))
            && bodies.assets.any_contact(
                (&bodies.shape[sleeper], bodies.rotation[sleeper]),
                (&bodies.shape[other], bodies.rotation[other]),
                bodies
                    .position(other)
                    .offset_from(bodies.position(sleeper))
                    .as_vec3(),
                CONTACT_MARGIN,
            )
        {
            scratch.flagged[sleeper] = true;
        }
    }

    for dense in 0..count {
        if scratch.flagged[dense] {
            let root = scratch.find(dense);
            scratch.flagged[root] = true;
        }
    }
    for dense in 0..count {
        if bodies.asleep[dense] {
            let root = scratch.find(dense);
            scratch.awake[dense] = scratch.flagged[root];
        }
    }
}

fn is_cached(cache: &[CachedManifold], a: BodyHandle, b: BodyHandle) -> bool {
    let key = (a.min(b), a.max(b));
    let at = cache.partition_point(|entry| (entry.a, entry.b) < key);
    cache.get(at).is_some_and(|entry| (entry.a, entry.b) == key)
}

/// Runs on committed state. An island sleeps when every body stayed slow long enough.
/// It needs support from a still static or kinematic body, a free drift keeps its momentum.
/// A joint counts as a link and as support, a running motor keeps its island awake.
pub(crate) fn update_sleep(
    bodies: &mut Bodies,
    joints: &Joints,
    touching: impl Iterator<Item = (usize, usize)> + Clone,
    scratch: &mut SleepScratch,
    dt: f32,
) {
    let count = bodies.len();
    scratch.reset_islands(count);
    for (a, b) in touching.clone() {
        if bodies.kind[a] == BodyKind::Dynamic && bodies.kind[b] == BodyKind::Dynamic {
            scratch.union(a, b);
        }
    }
    for (a, b, _) in joints.links(bodies) {
        if bodies.kind[a] == BodyKind::Dynamic && bodies.kind[b] == BodyKind::Dynamic {
            scratch.union(a, b);
        }
    }
    // `flagged` marks islands that must stay awake from here on
    scratch.flagged.clear();
    scratch.flagged.resize(count, false);
    scratch.supported.clear();
    scratch.supported.resize(count, false);
    let links: Vec<(usize, usize, bool)> = joints.links(bodies).collect();
    let contacts = touching.map(|(a, b)| (a, b, false));
    for (a, b, running) in contacts.chain(links) {
        if running {
            let body = if bodies.kind[a] == BodyKind::Dynamic {
                a
            } else {
                b
            };
            let root = scratch.find(body);
            scratch.flagged[root] = true;
        }
        let (body, other) = match (bodies.kind[a], bodies.kind[b]) {
            (BodyKind::Dynamic, BodyKind::Dynamic) => continue,
            (BodyKind::Dynamic, _) => (a, b),
            _ => (b, a),
        };
        let root = scratch.find(body);
        if is_moving(bodies, &scratch.awake, other) {
            scratch.flagged[root] = true;
        } else {
            scratch.supported[root] = true;
        }
    }

    for dense in 0..count {
        if bodies.kind[dense] != BodyKind::Dynamic || bodies.asleep[dense] {
            continue;
        }
        let slow = bodies.can_sleep[dense]
            && bodies.linear[dense].length_squared() < SLEEP_LINEAR_SQUARED
            && bodies.angular[dense].length_squared() < SLEEP_ANGULAR_SQUARED;
        bodies.still_time[dense] = if slow {
            (bodies.still_time[dense] + dt).min(SLEEP_TIME)
        } else {
            0.0
        };
        if bodies.still_time[dense] < SLEEP_TIME {
            let root = scratch.find(dense);
            scratch.flagged[root] = true;
        }
    }

    for dense in 0..count {
        if bodies.kind[dense] != BodyKind::Dynamic || bodies.asleep[dense] {
            continue;
        }
        let root = scratch.find(dense);
        if scratch.supported[root] && !scratch.flagged[root] {
            bodies.asleep[dense] = true;
            bodies.still_time[dense] = 0.0;
            bodies.linear[dense] = Vec3::ZERO;
            bodies.angular[dense] = Vec3::ZERO;
            bodies.momentum[dense] = Vec3::ZERO;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        BodyHandle, Command, CommandBuffer, JobDispatch, PhysicsError, PhysicsSettings,
        PhysicsWorld, RigidBodyDesc, Shape, Vec3,
    };

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    fn ground(world: &mut PhysicsWorld) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(50.0, 1.0, 50.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap()
    }

    fn cube() -> RigidBodyDesc {
        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
    }

    fn stack(world: &mut PhysicsWorld, height: usize) -> Vec<BodyHandle> {
        (0..height)
            .map(|level| {
                world
                    .spawn(cube().local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)))
                    .unwrap()
            })
            .collect()
    }

    fn run(world: &mut PhysicsWorld, seconds: f32) {
        let steps = (seconds / world.settings().fixed_timestep) as usize;
        for _ in 0..steps {
            world.step_fixed().unwrap();
        }
    }

    fn asleep(world: &PhysicsWorld, body: BodyHandle) -> bool {
        world.body_state(body).unwrap().asleep
    }

    fn sleeping_stack(height: usize) -> (PhysicsWorld, BodyHandle, Vec<BodyHandle>) {
        let mut world = world();
        let floor = ground(&mut world);
        let boxes = stack(&mut world, height);
        run(&mut world, 3.0);
        for body in &boxes {
            assert!(asleep(&world, *body));
        }
        (world, floor, boxes)
    }

    #[test]
    fn tall_stack_settles_and_sleeps() {
        let mut world = world();
        ground(&mut world);
        let boxes = stack(&mut world, 8);
        run(&mut world, 6.0);
        for (level, body) in boxes.iter().enumerate() {
            let state = world.body_state(*body).unwrap();
            assert!(state.asleep, "level {level}");
            assert!((state.position.local.y - (0.5 + level as f32)).abs() < 0.05);
            assert!(state.position.local.x.abs() < 0.05, "{state:?}");
            assert!(state.position.local.z.abs() < 0.05, "{state:?}");
        }
    }

    #[test]
    fn tall_stack_does_not_gain_speed_while_settling() {
        let mut world = world();
        ground(&mut world);
        let boxes: Vec<_> = (0..8)
            .map(|level| {
                world
                    .spawn(cube().can_sleep(false).local_position(Vec3::new(
                        0.0,
                        0.5 + level as f32,
                        0.0,
                    )))
                    .unwrap()
            })
            .collect();
        run(&mut world, 3.0);
        for _ in 0..420 {
            world.step_fixed().unwrap();
            for body in &boxes {
                let speed = world.body_state(*body).unwrap().linear_velocity.length();
                assert!(speed < 0.05, "{speed}");
            }
        }
    }

    #[test]
    fn stack_falls_asleep_and_stops_exactly() {
        let (mut world, _, boxes) = sleeping_stack(5);
        let before: Vec<_> = boxes
            .iter()
            .map(|body| world.body_state(*body).unwrap())
            .collect();
        for (level, state) in before.iter().enumerate() {
            assert!((state.position.local.y - (0.5 + level as f32)).abs() < 0.05);
            assert_eq!(state.linear_velocity, Vec3::ZERO);
            assert_eq!(state.angular_velocity, Vec3::ZERO);
        }
        let cache = world.contact_cache.clone();
        run(&mut world, 2.0);
        for (body, old) in boxes.iter().zip(&before) {
            assert_eq!(world.body_state(*body).unwrap(), *old);
        }
        assert_eq!(world.contact_cache, cache);
    }

    #[test]
    fn free_drift_never_sleeps_and_keeps_its_speed() {
        let mut world = PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap();
        let body = world
            .spawn(cube().linear_velocity(Vec3::new(0.01, 0.0, 0.0)))
            .unwrap();
        let other = world
            .spawn(
                cube()
                    .local_position(Vec3::new(0.0, 0.99, 0.0))
                    .linear_velocity(Vec3::new(0.01, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 5.0);
        assert!(!asleep(&world, body));
        assert!(!asleep(&world, other));
        let state = world.body_state(body).unwrap();
        assert_eq!(state.linear_velocity.x, 0.01);
    }

    #[test]
    fn impulse_force_torque_velocity_and_wake_all_wake_it() {
        type Action<'a> = &'a dyn Fn(&mut PhysicsWorld, BodyHandle);
        let actions: [Action; 5] = [
            &|world, body| world.apply_impulse(body, Vec3::Y).unwrap(),
            &|world, body| world.apply_force(body, Vec3::Y).unwrap(),
            &|world, body| world.apply_torque(body, Vec3::Y).unwrap(),
            &|world, body| world.set_velocity(body, Vec3::Y, Vec3::ZERO).unwrap(),
            &|world, body| world.wake(body).unwrap(),
        ];
        for action in actions {
            let (mut world, _, boxes) = sleeping_stack(1);
            action(&mut world, boxes[0]);
            assert!(!asleep(&world, boxes[0]));
        }
    }

    #[test]
    fn rejected_input_leaves_it_asleep() {
        let (mut world, _, boxes) = sleeping_stack(1);
        let bad = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(world.apply_force(boxes[0], bad).is_err());
        assert!(world.apply_impulse(boxes[0], bad).is_err());
        assert!(asleep(&world, boxes[0]));
    }

    #[test]
    fn wake_command_works_and_reports_stale_handles() {
        let (mut world, _, boxes) = sleeping_stack(2);
        let mut buffer = CommandBuffer::new();
        buffer.push(Command::Wake(boxes[1]));
        assert!(world.apply_commands(&mut buffer).rejected.is_empty());
        assert!(!asleep(&world, boxes[1]));
        world.despawn(boxes[1]).unwrap();
        assert_eq!(world.wake(boxes[1]), Err(PhysicsError::InvalidHandle));
    }

    #[test]
    fn falling_ball_wakes_the_whole_stack_at_once() {
        let (mut world, floor, boxes) = sleeping_stack(4);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .density(5000.0)
                    .local_position(Vec3::new(0.0, 6.0, 0.0)),
            )
            .unwrap();
        let mut steps = 0;
        while asleep(&world, boxes[3]) {
            world.step_fixed().unwrap();
            steps += 1;
            assert!(steps < 240);
            if !asleep(&world, boxes[3]) {
                assert!(!asleep(&world, boxes[0]));
            }
        }
        assert!(!asleep(&world, ball));
        // a stack that really woke carries the hit down to the floor, a sleeper would act like a wall
        let load = |world: &PhysicsWorld| {
            world
                .contact_cache
                .iter()
                .filter(|entry| entry.a == floor)
                .flat_map(|entry| entry.points[..entry.count].iter())
                .map(|point| point.impulse.x)
                .sum::<f32>()
        };
        run(&mut world, 1.0);
        // four boxes of 1000 kg and the ball, per substep
        let part = world.settings().fixed_timestep / world.settings().substeps as f32;
        let ball_mass = 5000.0 * 4.0 / 3.0 * std::f32::consts::PI * 0.125;
        let expected = (4000.0 + ball_mass) * 9.81 * part;
        let measured = load(&world);
        assert!(
            (measured - expected).abs() < expected * 0.02,
            "{measured} {expected}"
        );
    }

    #[test]
    fn woken_stack_goes_back_to_sleep() {
        let (mut world, _, boxes) = sleeping_stack(3);
        world.wake(boxes[1]).unwrap();
        world.step_fixed().unwrap();
        for body in &boxes {
            assert!(!asleep(&world, *body));
        }
        run(&mut world, 3.0);
        for (level, body) in boxes.iter().enumerate() {
            assert!(asleep(&world, *body));
            let y = world.body_state(*body).unwrap().position.local.y;
            assert!((y - (0.5 + level as f32)).abs() < 0.05, "{y}");
        }
    }

    #[test]
    fn warm_start_survives_sleep() {
        let (mut world, _, boxes) = sleeping_stack(3);
        world.wake(boxes[0]).unwrap();
        world.step_fixed().unwrap();
        for body in &boxes {
            let speed = world.body_state(*body).unwrap().linear_velocity.length();
            assert!(speed < 0.05, "{speed}");
        }
    }

    #[test]
    fn removing_the_support_drops_what_was_on_it() {
        let (mut world, floor, boxes) = sleeping_stack(2);
        world.despawn(floor).unwrap();
        assert!(!asleep(&world, boxes[0]));
        run(&mut world, 0.5);
        assert!(world.body_state(boxes[1]).unwrap().position.local.y < 0.5);
    }

    #[test]
    fn removing_a_box_mid_stack_drops_the_ones_above() {
        let (mut world, _, boxes) = sleeping_stack(3);
        world.despawn(boxes[1]).unwrap();
        run(&mut world, 1.0);
        let top = world.body_state(boxes[2]).unwrap().position.local.y;
        assert!((top - 1.5).abs() < 0.05, "{top}");
        assert!((world.body_state(boxes[0]).unwrap().position.local.y - 0.5).abs() < 0.05);
    }

    #[test]
    fn moving_kinematic_wakes_its_rider_and_keeps_it_awake() {
        let mut world = world();
        let platform = world
            .spawn(
                RigidBodyDesc::kinematic(Shape::cuboid(Vec3::new(5.0, 0.5, 5.0)))
                    .local_position(Vec3::new(0.0, -0.5, 0.0)),
            )
            .unwrap();
        let rider = world
            .spawn(cube().local_position(Vec3::new(0.0, 0.5, 0.0)))
            .unwrap();
        run(&mut world, 2.0);
        assert!(asleep(&world, rider));
        world
            .set_velocity(platform, Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO)
            .unwrap();
        world.step_fixed().unwrap();
        assert!(!asleep(&world, rider));
        run(&mut world, 2.0);
        let y = world.body_state(rider).unwrap().position.local.y;
        assert!((y + 1.5).abs() < 0.3, "{y}");
    }

    #[test]
    fn opting_out_never_sleeps() {
        let mut world = world();
        ground(&mut world);
        let body = world
            .spawn(
                cube()
                    .can_sleep(false)
                    .local_position(Vec3::new(0.0, 0.5, 0.0)),
            )
            .unwrap();
        run(&mut world, 3.0);
        assert!(!asleep(&world, body));
    }

    #[test]
    fn opting_out_keeps_its_whole_island_awake() {
        let mut world = world();
        ground(&mut world);
        let below = world
            .spawn(cube().local_position(Vec3::new(0.0, 0.5, 0.0)))
            .unwrap();
        world
            .spawn(
                cube()
                    .can_sleep(false)
                    .local_position(Vec3::new(0.0, 1.5, 0.0)),
            )
            .unwrap();
        run(&mut world, 3.0);
        assert!(!asleep(&world, below));
    }

    #[test]
    fn new_static_wakes_only_what_it_touches() {
        let (mut world, _, boxes) = sleeping_stack(1);
        let far = world
            .spawn(cube().local_position(Vec3::new(10.0, 0.5, 0.0)))
            .unwrap();
        run(&mut world, 2.0);
        assert!(asleep(&world, far));
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::sphere(0.25)).local_position(Vec3::new(10.0, 1.0, 0.0)),
            )
            .unwrap();
        world.step_fixed().unwrap();
        assert!(!asleep(&world, far));
        assert!(asleep(&world, boxes[0]));
    }

    #[test]
    fn snapshot_while_asleep_continues_exactly() {
        let (mut world, _, boxes) = sleeping_stack(3);
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        assert_eq!(loaded.contact_cache, world.contact_cache);
        run(&mut world, 0.5);
        run(&mut loaded, 0.5);
        assert_eq!(loaded.state_hash(), world.state_hash());
        world.wake(boxes[2]).unwrap();
        loaded.wake(boxes[2]).unwrap();
        run(&mut world, 1.0);
        run(&mut loaded, 1.0);
        assert_eq!(loaded.state_hash(), world.state_hash());
    }

    struct SkipsJobs;

    impl JobDispatch for SkipsJobs {
        fn parallelism(&self) -> usize {
            2
        }

        fn run(&self, _jobs: usize, _job: &(dyn Fn(usize) + Sync)) {}
    }

    #[test]
    fn failed_step_does_not_wake_anything() {
        let (mut world, _, boxes) = sleeping_stack(2);
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 2.4, 0.0)),
            )
            .unwrap();
        assert_eq!(
            world.step_with(&SkipsJobs),
            Err(PhysicsError::DispatchIncomplete)
        );
        assert!(asleep(&world, boxes[0]));
        world.step_fixed().unwrap();
        assert!(!asleep(&world, boxes[0]));
    }
}
