use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use algebrix::{Quat, Vec3};

use crate::body::{Bodies, apply_diagonal, momentum_from_velocity};
use crate::gravity::StepGravity;
use crate::profile::PhaseTimes;
use crate::sleep::{is_slow, plan_wakes, update_sleep};
use crate::solver::{
    CachedManifold, add_woken_pairs, find_broad_pairs, finish_contacts, prepare_contacts,
    solve_substep,
};
use crate::{
    BodyHandle, BodyKind, ContactEvent, FrameId, GridPosition, JobDispatch, JointHandle,
    PhysicsError, Sequential,
};

use super::{Advance, Integrated, Pairs, PhysicsWorld, finite};

/// Integrating a body is tens of nanoseconds, a dispatch several microseconds.
const MIN_BODIES_PER_JOB: usize = 1024;

impl PhysicsWorld {
    /// Adds real time and runs due steps, up to the step limit. Time past the limit stays queued.
    ///
    /// # Errors
    /// Stops at the first failed step, that step leaves state unchanged.
    pub fn advance(&mut self, real_seconds: f64) -> Result<Advance, PhysicsError> {
        if !real_seconds.is_finite() || real_seconds < 0.0 {
            return Err(PhysicsError::NonFinite("advance time"));
        }
        self.accumulator += real_seconds;
        let timestep = f64::from(self.settings.fixed_timestep);
        // 1/60 as f32 is a hair over 1/60 as f64, and the gap grows with queued steps, so exact
        // multiples of the step would come up one short
        let due = timestep * (1.0 - 1e-3);
        let mut steps = 0;
        while self.accumulator >= due && steps < self.settings.max_steps_per_advance {
            self.step_fixed()?;
            self.accumulator = (self.accumulator - timestep).max(0.0);
            steps += 1;
        }
        Ok(Advance {
            steps,
            alpha: (self.accumulator / timestep).min(1.0) as f32,
            overloaded: self.accumulator >= due,
        })
    }

    /// # Errors
    /// See [`Self::step_with`].
    pub fn step_fixed(&mut self) -> Result<(), PhysicsError> {
        self.step_with(&Sequential)
    }

    /// Runs one step on the host dispatcher. Result does not depend on its worker count.
    ///
    /// # Errors
    /// Fails if a body goes non finite or out of range, or the dispatcher skips a job.
    /// State stays at the last step when this fails.
    pub fn step_with(&mut self, dispatch: &dyn JobDispatch) -> Result<(), PhysicsError> {
        self.step_with_hooks(dispatch, &mut [])
    }

    /// [`Self::step_with`], with `hooks` run inside every substep before its solve, in order.
    /// Coupled solvers that must push bodies and be pushed back within the step go here.
    ///
    /// # Errors
    /// As [`Self::step_with`], and when a hook fails. State stays at the last step then too,
    /// whatever the hook did to its own state.
    pub fn step_with_hooks(
        &mut self,
        dispatch: &dyn JobDispatch,
        hooks: &mut [&mut dyn crate::substep::SubstepHook],
    ) -> Result<(), PhysicsError> {
        let started = Instant::now();
        self.failed_body = None;
        self.gravity
            .prepare(self.settings.gravity, &self.frames, &self.gravity_sources)?;
        self.remember_velocities();
        self.remember_structure_motion();
        self.apply_fields();
        self.apply_aero()?;
        self.joints.refresh_exclusions();
        let dt = self.settings.fixed_timestep;
        let mut phases = PhaseTimes::default();
        let mut mark = started;
        let mut lap = |into: &mut Duration| {
            let now = Instant::now();
            *into += now - mark;
            mark = now;
        };
        find_broad_pairs(
            &self.bodies,
            &mut self.contact_scratch,
            self.recheck_sleepers,
            (dispatch, dt),
        );
        plan_wakes(
            &self.bodies,
            &self.contact_cache,
            &self.joints,
            self.contact_scratch.pairs(),
            self.recheck_sleepers,
            &mut self.sleep,
            dt,
        );
        if !self.recheck_sleepers {
            add_woken_pairs(&self.bodies, self.sleep.awake(), &mut self.contact_scratch);
        }
        let count = self.bodies.len();
        self.scratch.clear();
        self.scratch
            .extend((0..count).map(|dense| starting_state(&self.bodies, dense)));

        // a small world integrates on this thread, a big one in a job per worker
        let chunk_size = if count < MIN_BODIES_PER_JOB {
            count.max(1)
        } else {
            count.div_ceil(dispatch.parallelism().max(1))
        };
        let bodies = &self.bodies;
        let gravity = &self.gravity;
        let awake = self.sleep.awake();
        let substeps = self.settings.substeps;
        let part = dt / substeps as f32;
        let mut next_cache = std::mem::take(&mut self.spare_cache);
        next_cache.clear();
        next_cache.extend_from_slice(&self.contact_cache);
        lap(&mut phases.broadphase);
        prepare_contacts(
            bodies,
            awake,
            &self.joints,
            &self.scratch,
            &self.contact_cache,
            &mut self.contact_scratch,
            dispatch,
            dt,
        )?;
        lap(&mut phases.narrowphase);
        let mut woken = Vec::new();
        // outside forces were worked out at the start of the step, so they go in whole before the
        // first move. spread over substeps they would lag the motion and add energy
        run_jobs(&mut self.scratch, chunk_size, dispatch, &|dense, slot| {
            integrate_velocity(bodies, gravity, awake[dense], dense, slot, (part, dt));
        })?;
        lap(&mut phases.integrate);
        for substep in 0..substeps {
            if !hooks.is_empty() {
                let mut view = crate::substep::SubstepBodies {
                    bodies,
                    state: &mut self.scratch,
                    awake,
                    woken: &mut woken,
                };
                for hook in hooks.iter_mut() {
                    hook.before_solve(&mut view, part, dispatch)?;
                }
            }
            solve_substep(
                bodies,
                awake,
                &self.joints,
                &mut self.scratch,
                &mut self.contact_scratch,
                (substep, substep + 1 == substeps),
                (dispatch, part),
            );
            lap(&mut phases.solve);
            // this substep's move and the next one's velocity in one pass, body by body as
            // before, one dispatch fewer
            let next = substep + 1 < substeps;
            run_jobs(&mut self.scratch, chunk_size, dispatch, &|dense, slot| {
                integrate_position(bodies, awake[dense], dense, slot, part);
                if next {
                    integrate_velocity(bodies, gravity, awake[dense], dense, slot, (part, 0.0));
                }
            })?;
            lap(&mut phases.integrate);
        }
        finish_contacts(
            bodies,
            awake,
            &mut next_cache,
            &self.contact_scratch,
            &mut self.finish_scratch,
        );

        self.commit()?;
        contact_changes(
            &self.contact_cache,
            &next_cache,
            &mut self.touched,
            &mut self.contact_events,
        );
        self.spare_cache = std::mem::replace(&mut self.contact_cache, next_cache);
        let (points, rows, impulses) = self.contact_scratch.joint_parts();
        let loads = self.contact_scratch.joint_loads();
        self.joints.store(impulses, (points, rows), loads, dt);
        self.break_joints();
        update_sleep(
            &mut self.bodies,
            &self.joints,
            self.contact_scratch.touching(),
            &mut self.sleep,
            dt,
        );
        for dense in woken {
            self.bodies.wake(dense);
        }
        self.recheck_sleepers = false;
        self.tick += 1;
        self.frames
            .advance_to(self.tick as f64 * f64::from(self.settings.fixed_timestep));
        self.change_frames();
        self.update_sensors()?;
        self.update_breakables()?;
        self.update_structures()?;
        lap(&mut phases.finish);
        self.phases = phases;
        self.times.record(started.elapsed());
        Ok(())
    }

    /// Takes out every joint the last step pulled or twisted past its limit. Between steps,
    /// never inside a solve.
    fn break_joints(&mut self) {
        self.broken_joints.clear();
        let over: Vec<JointHandle> = self
            .joints
            .handles()
            .filter(|joint| {
                self.joints.get(*joint).is_ok_and(|state| {
                    state.force.length() > state.desc.break_force
                        || state.torque.length() > state.desc.break_torque
                })
            })
            .collect();
        for joint in over {
            if self.remove_joint(joint).is_ok() {
                self.broken_joints.push(joint);
            }
        }
    }

    /// Bodies gone out of their frame's reach go to its parent, ones come into a child's reach
    /// go in. Between steps, in body order. A body with joints stays, as does one whose move
    /// fails, it is tried again next step.
    fn change_frames(&mut self) {
        self.frame_changes.clear();
        let frames = &self.frames;
        let reaching: Vec<FrameId> = (1..frames.len() as u32)
            .map(FrameId)
            .filter(|frame| frames.get(*frame).is_ok_and(|desc| desc.reach > 0.0))
            .collect();
        if reaching.is_empty() {
            return;
        }
        let mut moves = Vec::new();
        for dense in 0..self.bodies.len() {
            if self.bodies.kind[dense] != BodyKind::Dynamic || self.bodies.asleep[dense] {
                continue;
            }
            let frame = self.bodies.frame[dense];
            let at = self.bodies.position(dense).to_dvec3();
            let Ok(desc) = frames.get(frame) else {
                continue;
            };
            let target = if desc.reach > 0.0 && at.length() > desc.reach * 1.1 {
                Some(desc.parent)
            } else {
                reaching
                    .iter()
                    .filter_map(|child| {
                        let inner = frames.get(*child).ok()?;
                        let distance = (at - inner.translation).length();
                        (inner.parent == frame && distance < inner.reach * 0.9)
                            .then_some((distance, *child))
                    })
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, child)| child)
            };
            if let Some(target) = target {
                moves.push((self.bodies.handle_at(dense), frame, target));
            }
        }
        for (body, from, to) in moves {
            if self.transfer_to_frame(body, to).is_ok() {
                self.frame_changes.push((body, from, to));
            }
        }
    }

    fn commit(&mut self) -> Result<(), PhysicsError> {
        for (dense, result) in self.scratch.iter().enumerate() {
            let checked = if result.out_of_range {
                Err(PhysicsError::OutOfRange)
            } else if !result.rotation.is_finite() {
                Err(PhysicsError::NonFinite("rotation"))
            } else {
                finite(result.linear, "velocity")
                    .and_then(|()| finite(result.angular, "velocity"))
                    .and_then(|()| finite(result.momentum, "momentum"))
            };
            if let Err(error) = checked {
                self.failed_body = Some(self.bodies.handle_at(dense));
                return Err(error);
            }
        }
        // the query tree follows what moved, a sleeping world keeps it as it is
        let moved: Vec<u32> = self
            .scratch
            .iter()
            .enumerate()
            .filter(|(dense, result)| {
                result.position.cell != self.bodies.cell[*dense]
                    || result.position.local != self.bodies.local[*dense]
                    || result.rotation != self.bodies.rotation[*dense]
            })
            .map(|(dense, _)| dense as u32)
            .collect();
        for (dense, result) in self.scratch.iter().enumerate() {
            self.bodies.cell[dense] = result.position.cell;
            self.bodies.local[dense] = result.position.local;
            self.bodies.rotation[dense] = result.rotation;
            self.bodies.linear[dense] = result.linear;
            self.bodies.angular[dense] = result.angular;
            self.bodies.momentum[dense] = result.momentum;
            self.bodies.force[dense] = Vec3::ZERO;
            self.bodies.torque[dense] = Vec3::ZERO;
            self.bodies.asleep[dense] = !self.sleep.awake()[dense];
            // recovery moves a body with no velocity to show for it, that is not resting
            if !is_slow(result.recovery_linear, result.recovery_angular) {
                self.bodies.still_time[dense] = 0.0;
            }
        }
        if !moved.is_empty() {
            let bodies = &mut self.bodies;
            let mut tree = std::mem::take(&mut bodies.tree);
            tree.refit(bodies, &moved);
            bodies.tree = tree;
        }
        Ok(())
    }
}

/// Runs `work` for every body on the host. Fails if the host skipped a job.
fn run_jobs(
    state: &mut [Integrated],
    chunk_size: usize,
    dispatch: &dyn JobDispatch,
    work: &(dyn Fn(usize, &mut Integrated) + Sync),
) -> Result<(), PhysicsError> {
    if state.len() <= chunk_size {
        for (index, slot) in state.iter_mut().enumerate() {
            work(index, slot);
        }
        return Ok(());
    }
    let chunks: Vec<Mutex<(&mut [Integrated], bool)>> = state
        .chunks_mut(chunk_size)
        .map(|chunk| Mutex::new((chunk, false)))
        .collect();
    dispatch.run(chunks.len(), &|job| {
        let Some(chunk) = chunks.get(job) else {
            return;
        };
        let mut guard = chunk.lock().unwrap_or_else(PoisonError::into_inner);
        let (out, done) = &mut *guard;
        let base = job * chunk_size;
        for (offset, slot) in out.iter_mut().enumerate() {
            work(base + offset, slot);
        }
        *done = true;
    });
    let all_done = chunks
        .iter()
        .all(|chunk| chunk.lock().unwrap_or_else(PoisonError::into_inner).1);
    if all_done {
        Ok(())
    } else {
        Err(PhysicsError::DispatchIncomplete)
    }
}

fn starting_state(bodies: &Bodies, dense: usize) -> Integrated {
    Integrated {
        position: bodies.position(dense),
        rotation: bodies.rotation[dense],
        linear: bodies.linear[dense],
        angular: bodies.angular[dense],
        momentum: bodies.momentum[dense],
        ..Integrated::default()
    }
}

fn integrate_velocity(
    bodies: &Bodies,
    gravity: &StepGravity,
    awake: bool,
    dense: usize,
    slot: &mut Integrated,
    (dt, push): (f32, f32),
) {
    slot.recovery_linear = Vec3::ZERO;
    slot.recovery_angular = Vec3::ZERO;
    if bodies.kind[dense] != BodyKind::Dynamic || !awake {
        return;
    }
    let frame = bodies.frame[dense];
    let spin = gravity.spin(frame);
    let pull = gravity.without_coriolis(frame, slot.position);
    slot.momentum += bodies.torque[dense] * push;
    slot.linear += bodies.force[dense] * (bodies.inverse_mass[dense] * push);
    if spin == Vec3::ZERO {
        slot.linear += pull * dt;
    } else {
        // half the pull, coriolis as the exact turn it is, the other half. pushed once, a body
        // still in the root drifted 4.6 cm in 4 s seen from a frame turning at 0.7 rad/s, now 1.6
        let rate = spin.length();
        let turn = Quat::from_axis_angle(spin / rate, -2.0 * rate * dt);
        slot.linear = turn.rotate_vec3(slot.linear + pull * (0.5 * dt)) + pull * (0.5 * dt);
    }
    slot.angular = apply_diagonal(slot.rotation, bodies.inverse_inertia[dense], slot.momentum);
}

fn move_slot(slot: &mut Integrated, velocity: Vec3, dt: f32) {
    let moved = GridPosition {
        cell: slot.position.cell,
        local: slot.position.local + velocity * dt,
    }
    .rebased();
    match moved {
        Some(position) => slot.position = position,
        None => slot.out_of_range = true,
    }
}

fn integrate_position(bodies: &Bodies, awake: bool, dense: usize, slot: &mut Integrated, dt: f32) {
    if !awake {
        return;
    }
    match bodies.kind[dense] {
        BodyKind::Static => {}
        BodyKind::Kinematic => {
            move_slot(slot, slot.linear, dt);
            slot.rotation = spin(slot.rotation, slot.angular, dt);
        }
        BodyKind::Dynamic => {
            let inverse_inertia = bodies.inverse_inertia[dense];
            if slot.touched {
                slot.momentum = momentum_from_velocity(
                    BodyKind::Dynamic,
                    slot.rotation,
                    inverse_inertia,
                    slot.angular,
                );
            }
            move_slot(slot, slot.linear + slot.recovery_linear, dt);
            slot.rotation = tumble(slot.rotation, inverse_inertia, slot.momentum, dt);
            if slot.recovery_angular != Vec3::ZERO {
                slot.rotation = spin(slot.rotation, slot.recovery_angular, dt);
            }
            slot.angular = apply_diagonal(slot.rotation, inverse_inertia, slot.momentum);
        }
    }
}

fn spin(rotation: Quat, angular: Vec3, dt: f32) -> Quat {
    (Quat::from_scaled_axis(angular * dt) * rotation).normalize()
}

/// Symmetric Lie-Poisson splitting (Dullweber, Reich, Leimkuhler 1997).
fn tumble(rotation: Quat, inverse_inertia: Vec3, momentum: Vec3, dt: f32) -> Quat {
    // equal axes have no gyroscopic term, it spins about a fixed axis
    if inverse_inertia.x == inverse_inertia.y && inverse_inertia.y == inverse_inertia.z {
        return spin(rotation, momentum * inverse_inertia.x, dt);
    }
    let half = dt * 0.5;
    let mut next = rotation;
    let mut body = rotation.conjugate().rotate_vec3(momentum);
    for (axis, inverse, time) in [
        (Vec3::X, inverse_inertia.x, half),
        (Vec3::Y, inverse_inertia.y, half),
        (Vec3::Z, inverse_inertia.z, dt),
        (Vec3::Y, inverse_inertia.y, half),
        (Vec3::X, inverse_inertia.x, half),
    ] {
        let turn = Quat::from_scaled_axis(axis * (body.dot(axis) * inverse * time));
        next = next * turn;
        body = turn.conjugate().rotate_vec3(body);
    }
    next.normalize()
}

/// Body pairs with a loaded point, lower handle first, sorted, once each.
fn touching(cache: &[CachedManifold], out: &mut Vec<(BodyHandle, BodyHandle)>) {
    out.clear();
    for entry in cache {
        let loaded = entry.points[..entry.count]
            .iter()
            .any(|point| point.impulse.x > 0.0);
        if loaded && out.last() != Some(&(entry.a, entry.b)) {
            out.push((entry.a, entry.b));
        }
    }
}

fn contact_changes(
    old: &[CachedManifold],
    new: &[CachedManifold],
    (before, after): &mut (Pairs, Pairs),
    out: &mut Vec<ContactEvent>,
) {
    touching(old, before);
    touching(new, after);
    out.clear();
    let (mut i, mut j) = (0, 0);
    while i < before.len() || j < after.len() {
        match (before.get(i), after.get(j)) {
            (Some(was), Some(now)) if was == now => {
                i += 1;
                j += 1;
            }
            (Some(was), Some(now)) if was < now => {
                out.push(ContactEvent::Ended(was.0, was.1));
                i += 1;
            }
            (Some(was), None) => {
                out.push(ContactEvent::Ended(was.0, was.1));
                i += 1;
            }
            (_, Some(now)) => {
                out.push(ContactEvent::Began(now.0, now.1));
                j += 1;
            }
            (None, None) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use crate::dispatch::testing::SkipsJobs;
    use crate::{Command, CommandBuffer, IslandStats, JointDesc, RigidBodyDesc, Shape};

    use super::super::testing::{no_gravity, world};
    use super::super::tests::cube;
    use super::*;

    fn angular_momentum(world: &PhysicsWorld, body: BodyHandle) -> Vec3 {
        let dense = world.bodies.dense_index(body).unwrap();
        apply_diagonal(
            world.bodies.rotation[dense],
            world.bodies.inverse_inertia[dense].recip(),
            world.bodies.angular[dense],
        )
    }

    struct Threads(usize);

    impl JobDispatch for Threads {
        fn parallelism(&self) -> usize {
            self.0
        }

        fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
            thread::scope(|scope| {
                for index in 0..jobs {
                    scope.spawn(move || job(index));
                }
            });
        }
    }

    fn island_scene() -> PhysicsWorld {
        let mut world = world();
        let ground = world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(30.0, 1.0, 30.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
        for x in [-6.0_f32, 0.0, 6.0] {
            for i in 0..5 {
                world
                    .spawn(cube().local_position(Vec3::new(x, 0.51 + i as f32 * 1.01, 0.0)))
                    .unwrap();
            }
        }
        let post = world
            .spawn(
                RigidBodyDesc::fixed(Shape::sphere(0.1)).local_position(Vec3::new(0.0, 8.0, 6.0)),
            )
            .unwrap();
        let mut last = post;
        for i in 1..=4 {
            let at = Vec3::new(i as f32 * 0.8, 8.0, 6.0);
            let ball = world
                .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.2)).local_position(at))
                .unwrap();
            let anchor = GridPosition::from_local(at - Vec3::X * 0.4).unwrap();
            let joint = world
                .joint_at(JointDesc::ball(last, ball), anchor, Quat::IDENTITY)
                .unwrap();
            world.add_joint(joint).unwrap();
            last = ball;
        }
        let door = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(0.5, 1.0, 0.05)))
                    .local_position(Vec3::new(-5.5, 1.2, 6.0))
                    .angular_velocity(Vec3::Y * 2.0),
            )
            .unwrap();
        let hinge = world
            .joint_at(
                JointDesc::hinge(ground, door),
                GridPosition::from_local(Vec3::new(-6.0, 1.2, 6.0)).unwrap(),
                Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            )
            .unwrap();
        world.add_joint(hinge).unwrap();
        world
    }

    #[test]
    fn islands_split_at_static_bodies_and_join_by_joints() {
        let mut world = island_scene();
        assert_eq!(world.island_stats(), IslandStats::default());
        world.step_fixed().unwrap();
        // three stacks on the ground, the chain on its post, the door on its hinge
        let stats = world.island_stats();
        assert_eq!(stats.count, 5, "{stats:?}");
        assert_eq!(stats.largest_bodies, 5);
        // four contacts and their manifolds per stack, chain links count as joints
        assert!(stats.largest_constraints >= 5, "{stats:?}");
    }

    #[test]
    fn sleeping_stacks_leave_the_islands() {
        let mut world = island_scene();
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        let stats = world.island_stats();
        assert!(stats.count < 5, "{stats:?}");
        assert_eq!(
            world.sleeping_body_count() + count_awake_dynamic(&world),
            15 + 5
        );
    }

    fn count_awake_dynamic(world: &PhysicsWorld) -> usize {
        world
            .body_handles()
            .filter(|body| {
                let state = world.body_state(*body).unwrap();
                state.kind == BodyKind::Dynamic && !state.asleep
            })
            .count()
    }

    #[test]
    fn a_free_ball_spins_about_a_fixed_axis_exactly() {
        let mut world = no_gravity();
        let axis = Vec3::new(1.0, 2.0, -0.5).normalize();
        let ball = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.5)).angular_velocity(axis * 3.0))
            .unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        let state = world.body_state(ball).unwrap();
        let wanted = Quat::from_scaled_axis(axis * 3.0);
        assert!(state.rotation.angle_between(wanted) < 1e-4, "{state:?}");
        assert!((state.angular_velocity - axis * 3.0).length() < 1e-5);
    }

    #[test]
    fn contact_events_begin_on_landing_end_on_leaving_and_survive_sleep() {
        let mut world = world();
        let ground = world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(10.0, 1.0, 10.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
        let ball = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::Y * 3.0))
            .unwrap();
        let mut seen = Vec::new();
        for _ in 0..240 {
            world.step_fixed().unwrap();
            seen.extend_from_slice(world.contact_events());
        }
        let pair = (ground.min(ball), ground.max(ball));
        assert_eq!(seen, [ContactEvent::Began(pair.0, pair.1)]);
        assert!(world.body_state(ball).unwrap().asleep);
        world.set_velocity(ball, Vec3::Y * 8.0, Vec3::ZERO).unwrap();
        let mut after = Vec::new();
        for _ in 0..10 {
            world.step_fixed().unwrap();
            after.extend_from_slice(world.contact_events());
        }
        assert_eq!(after, [ContactEvent::Ended(pair.0, pair.1)]);
    }

    #[test]
    fn a_kinematic_target_is_reached_in_one_step() {
        let mut world = world();
        let platform = world
            .spawn(RigidBodyDesc::kinematic(Shape::cuboid(Vec3::ONE)))
            .unwrap();
        let target = GridPosition::from_local(Vec3::new(2.0, 1.0, -0.5)).unwrap();
        let turn = Quat::from_rotation_y(1.2) * Quat::from_rotation_x(0.3);
        let mut buffer = CommandBuffer::new();
        buffer.push(Command::SetKinematicTarget(platform, target, turn));
        assert!(world.apply_commands(&mut buffer).rejected.is_empty());
        world.step_fixed().unwrap();
        let state = world.body_state(platform).unwrap();
        assert!(
            (state.position.local - target.local).length() < 1e-4,
            "{state:?}"
        );
        assert!(state.rotation.angle_between(turn) < 1e-4);
        let dynamic = world.spawn(cube()).unwrap();
        assert_eq!(
            world.set_kinematic_target(dynamic, target, turn),
            Err(PhysicsError::NotKinematic)
        );
        let nan = GridPosition::from_local(Vec3::ZERO).map(|mut at| {
            at.local.x = f32::NAN;
            at
        });
        assert!(
            world
                .set_kinematic_target(platform, nan.unwrap(), turn)
                .is_err()
        );
    }

    #[test]
    fn free_fall_matches_semi_implicit_euler() {
        let mut world = world();
        let body = world
            .spawn(cube().local_position(Vec3::new(0.0, 100.0, 0.0)))
            .unwrap();
        let dt = world.settings().fixed_timestep;
        let steps = 120;
        for _ in 0..steps {
            world.step_fixed().unwrap();
        }
        let state = world.body_state(body).unwrap();
        let parts = world.settings().substeps as f32;
        let n = steps as f32 * parts;
        let part = dt / parts;
        let expected_velocity = -9.81 * part * n;
        let expected_y = 100.0 - 9.81 * part * part * n * (n + 1.0) / 2.0;
        assert!((state.linear_velocity.y - expected_velocity).abs() < 1e-3);
        assert!((state.position.local.y - expected_y).abs() < 1e-2);
        assert_eq!(world.tick(), steps);
    }

    #[test]
    fn free_fall_converges_on_the_analytic_answer() {
        let mut coarse = world();
        let body = coarse.spawn(cube()).unwrap();
        let steps = 240;
        for _ in 0..steps {
            coarse.step_fixed().unwrap();
        }
        let time = steps as f32 * coarse.settings().fixed_timestep;
        let analytic = -0.5 * 9.81 * time * time;
        let error = (coarse.body_state(body).unwrap().position.local.y - analytic).abs();
        assert!(error < 0.5 * 9.81 * time * coarse.settings().fixed_timestep + 1e-3);
    }

    #[test]
    fn a_spring_force_from_outside_keeps_its_energy() {
        // a pull toward the origin read once per step, like any game force
        let mut world = no_gravity();
        let body = world
            .spawn(cube().density(1.0).local_position(Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        let stiffness = 40.0;
        let mut farthest = 0.0_f32;
        for _ in 0..1200 {
            let at = world.body_state(body).unwrap().position.local;
            world.apply_force(body, at * -stiffness).unwrap();
            world.step_fixed().unwrap();
            farthest = farthest.max(world.body_state(body).unwrap().position.local.length());
        }
        assert!(farthest < 1.02, "{farthest}");
    }

    #[test]
    fn torque_free_tumbling_keeps_angular_momentum() {
        let mut world = no_gravity();
        let body = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(0.2, 0.7, 1.5)))
                    .angular_velocity(Vec3::new(1.0, 8.0, 0.5)),
            )
            .unwrap();
        let energy = |world: &PhysicsWorld| {
            let angular = world.body_state(body).unwrap().angular_velocity;
            0.5 * angular.dot(angular_momentum(world, body))
        };
        let start = angular_momentum(&world, body);
        let start_energy = energy(&world);
        let stored = world.bodies.momentum.clone();
        let mut worst_energy = 0.0_f32;
        for _ in 0..3600 {
            world.step_fixed().unwrap();
            worst_energy = worst_energy.max((energy(&world) - start_energy).abs() / start_energy);
        }
        assert_eq!(world.bodies.momentum, stored);
        let end = angular_momentum(&world, body);
        assert!((end - start).length() / start.length() < 1e-4);
        assert!(worst_energy < 1e-2, "energy drift {worst_energy}");
        let rotation = world.body_state(body).unwrap().rotation;
        assert!((rotation.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn blowup_is_reported_and_state_stays() {
        let mut world = no_gravity();
        let good = world.spawn(cube()).unwrap();
        let bad = world
            .spawn(
                cube()
                    .density(1e-30)
                    .local_position(Vec3::new(1.0, 0.0, 0.0)),
            )
            .unwrap();
        world.apply_force(bad, Vec3::splat(f32::MAX)).unwrap();
        let before = world.body_state(good).unwrap();
        assert!(world.step_fixed().is_err());
        assert_eq!(world.body_state(good).unwrap(), before);
        assert_eq!(world.tick(), 0);
    }

    #[test]
    fn whole_seconds_of_real_time_give_whole_steps() {
        let mut world = world();
        let mut steps = 0;
        for _ in 0..180 {
            steps += world.advance(1.0 / 60.0).unwrap().steps;
        }
        assert_eq!(steps, 180);
        let mut world = super::super::testing::world();
        assert_eq!(world.advance(3.0).unwrap().steps, 8);
        for _ in 0..30 {
            world.advance(0.0).unwrap();
        }
        assert_eq!(world.tick(), 180);
    }

    #[test]
    fn advance_runs_due_steps_and_reports_overload() {
        let mut world = world();
        let dt = f64::from(world.settings().fixed_timestep);
        let on_time = world.advance(dt * 2.5).unwrap();
        assert_eq!(on_time.steps, 2);
        assert!(!on_time.overloaded);
        assert!((on_time.alpha - 0.5).abs() < 1e-3);
        let late = world.advance(dt * 100.0).unwrap();
        assert_eq!(late.steps, world.settings().max_steps_per_advance);
        assert!(late.overloaded);
        assert_eq!(world.tick(), 10);
        assert!(world.advance(f64::NAN).is_err());
        assert!(world.advance(-1.0).is_err());
    }

    #[test]
    fn worker_count_does_not_change_the_result() {
        let build = || {
            let mut world = world();
            for index in 0..2000 {
                let x = index as f32 * 0.25;
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(0.3, 0.6, 0.9)))
                            .local_position(Vec3::new(x, 0.0, -x))
                            .angular_velocity(Vec3::new(x.sin(), x.cos(), 1.0)),
                    )
                    .unwrap();
            }
            world
        };
        let mut sequential = build();
        let mut threaded = build();
        for _ in 0..30 {
            sequential.step_fixed().unwrap();
            threaded.step_with(&Threads(4)).unwrap();
        }
        assert_eq!(sequential.bodies, threaded.bodies);
    }

    #[test]
    fn dispatcher_skipping_jobs_fails_and_leaves_state() {
        let mut world = world();
        let body = world.spawn(cube()).unwrap();
        // enough bodies that a step hands work to the dispatcher at all
        for i in 0..1100 {
            let at = Vec3::new(1000.0 + (i % 40) as f32 * 3.0, 0.0, (i / 40) as f32 * 3.0);
            world
                .spawn(RigidBodyDesc::fixed(Shape::sphere(0.1)).local_position(at))
                .unwrap();
        }
        let before = world.body_state(body).unwrap();
        assert_eq!(
            world.step_with(&SkipsJobs {
                parallelism: 2,
                skip: |_| true
            }),
            Err(PhysicsError::DispatchIncomplete)
        );
        assert_eq!(world.body_state(body).unwrap(), before);
        assert_eq!(world.tick(), 0);
    }

    #[test]
    fn step_times_are_recorded() {
        let mut world = world();
        assert_eq!(world.step_report(), None);
        world.step_fixed().unwrap();
        assert_eq!(world.step_report().unwrap().samples, 1);
    }
}
