use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use algebrix::{DQuat, DVec3, Quat, Vec3};

use crate::body::{Bodies, apply_diagonal, momentum_from_velocity, validate_desc};
use crate::joint::JointKind;
use crate::profile::StepTimes;
use crate::sleep::{SleepScratch, is_slow, plan_wakes, update_sleep};
use crate::solver::{
    CachedManifold, ContactScratch, find_broad_pairs, finish_contacts, prepare_contacts,
    solve_substep,
};
use crate::{
    ApplyReport, BodyHandle, BodyKind, BodyState, Command, CommandBuffer, FrameDesc, FrameId,
    FrameState, Frames, GridPosition, JobDispatch, JointDesc, JointDof, JointHandle, JointMotor,
    JointReaction, Joints, PhysicsError, RigidBodyDesc, Sequential, StepReport,
};

const MIN_BODIES_PER_JOB: usize = 256;
const MAX_SUBSTEPS: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsSettings {
    pub gravity: Vec3,
    pub fixed_timestep: f32,
    pub max_steps_per_advance: u32,
    /// Solver passes per step. More costs time and keeps joints tighter with less energy loss.
    pub substeps: u32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            fixed_timestep: 1.0 / 60.0,
            max_steps_per_advance: 8,
            substeps: 4,
        }
    }
}

impl PhysicsSettings {
    pub(crate) fn validate(&self) -> Result<(), PhysicsError> {
        if !self.gravity.is_finite()
            || !self.fixed_timestep.is_finite()
            || self.fixed_timestep <= 0.0
            || self.max_steps_per_advance == 0
            || self.substeps == 0
            || self.substeps > MAX_SUBSTEPS
        {
            return Err(PhysicsError::InvalidSettings);
        }
        Ok(())
    }
}

/// `alpha` blends render state between the last two steps. `overloaded` means time is left over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Advance {
    pub steps: u32,
    pub alpha: f32,
    pub overloaded: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Integrated {
    pub(crate) position: GridPosition,
    pub(crate) out_of_range: bool,
    pub(crate) rotation: Quat,
    pub(crate) linear: Vec3,
    pub(crate) angular: Vec3,
    pub(crate) momentum: Vec3,
    pub(crate) recovery_linear: Vec3,
    pub(crate) recovery_angular: Vec3,
    pub(crate) touched: bool,
}

impl Default for Integrated {
    fn default() -> Self {
        Self {
            position: GridPosition::ORIGIN,
            out_of_range: false,
            rotation: Quat::IDENTITY,
            linear: Vec3::ZERO,
            angular: Vec3::ZERO,
            momentum: Vec3::ZERO,
            recovery_linear: Vec3::ZERO,
            recovery_angular: Vec3::ZERO,
            touched: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhysicsWorld {
    pub(crate) settings: PhysicsSettings,
    pub(crate) frames: Frames,
    pub(crate) bodies: Bodies,
    pub(crate) joints: Joints,
    pub(crate) tick: u64,
    pub(crate) accumulator: f64,
    pub(crate) contact_cache: Vec<CachedManifold>,
    /// A static or kinematic body appeared, so sleepers must check for new contact.
    pub(crate) recheck_sleepers: bool,
    contact_scratch: ContactScratch,
    sleep: SleepScratch,
    scratch: Vec<Integrated>,
    frame_gravity: Vec<Vec3>,
    times: StepTimes,
}

impl PhysicsWorld {
    /// # Errors
    /// Fails on a bad timestep, gravity or step limit.
    pub fn new(settings: PhysicsSettings) -> Result<Self, PhysicsError> {
        settings.validate()?;
        Ok(Self::from_parts(
            settings,
            Frames::new(),
            Bodies::default(),
            0,
            0.0,
        ))
    }

    pub(crate) fn from_parts(
        settings: PhysicsSettings,
        frames: Frames,
        bodies: Bodies,
        tick: u64,
        accumulator: f64,
    ) -> Self {
        Self {
            settings,
            frames,
            bodies,
            joints: Joints::default(),
            tick,
            accumulator,
            contact_cache: Vec::new(),
            recheck_sleepers: false,
            contact_scratch: ContactScratch::default(),
            sleep: SleepScratch::default(),
            scratch: Vec::new(),
            frame_gravity: Vec::new(),
            times: StepTimes::default(),
        }
    }

    pub fn settings(&self) -> &PhysicsSettings {
        &self.settings
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    pub fn frames(&self) -> &Frames {
        &self.frames
    }

    pub fn step_report(&self) -> Option<StepReport> {
        self.times.report()
    }

    /// # Errors
    /// Fails on an unknown parent, a too deep chain or non finite values.
    pub fn add_frame(&mut self, desc: FrameDesc) -> Result<FrameId, PhysicsError> {
        self.frames.add(desc)
    }

    /// # Errors
    /// Fails on a bad desc or an unknown frame.
    pub fn spawn(&mut self, desc: RigidBodyDesc) -> Result<BodyHandle, PhysicsError> {
        validate_desc(&desc, &self.bodies.assets)?;
        if !self.frames.contains(desc.frame) {
            return Err(PhysicsError::InvalidFrame);
        }
        let position = desc.position.rebased().ok_or(PhysicsError::OutOfRange)?;
        let body = self.bodies.insert(&RigidBodyDesc { position, ..desc })?;
        self.recheck_sleepers |= desc.kind != BodyKind::Dynamic;
        Ok(body)
    }

    /// # Errors
    /// Fails on a stale handle.
    pub fn despawn(&mut self, body: BodyHandle) -> Result<(), PhysicsError> {
        self.bodies.remove(body)?;
        self.wake_contacts_of(body);
        let attached: Vec<JointHandle> = self.joints.touching(body).collect();
        for joint in attached {
            self.remove_joint(joint)?;
        }
        Ok(())
    }

    pub fn joints(&self) -> &Joints {
        &self.joints
    }

    /// Bodies must exist, share a frame, and at least one must be dynamic.
    ///
    /// # Errors
    /// Fails on a stale body or a bad setup.
    pub fn add_joint(&mut self, desc: JointDesc) -> Result<JointHandle, PhysicsError> {
        desc.validate()?;
        let a = self.bodies.dense_index(desc.a)?;
        let b = self.bodies.dense_index(desc.b)?;
        if self.bodies.frame[a] != self.bodies.frame[b]
            || (self.bodies.kind[a] != BodyKind::Dynamic
                && self.bodies.kind[b] != BodyKind::Dynamic)
        {
            return Err(PhysicsError::InvalidJoint);
        }
        let joint = self.joints.insert(desc)?;
        self.bodies.wake(a);
        self.bodies.wake(b);
        Ok(joint)
    }

    /// Wakes both bodies, the joint may have been holding them.
    ///
    /// # Errors
    /// Fails on a stale handle.
    pub fn remove_joint(&mut self, joint: JointHandle) -> Result<(), PhysicsError> {
        let removed = self.joints.remove(joint)?;
        for body in [removed.desc.a, removed.desc.b] {
            if let Ok(dense) = self.bodies.dense_index(body) {
                self.bodies.wake(dense);
            }
        }
        Ok(())
    }

    /// Force and torque of the last step on body B.
    ///
    /// # Errors
    /// Fails on a stale handle.
    pub fn joint_reaction(&self, joint: JointHandle) -> Result<JointReaction, PhysicsError> {
        let state = self.joints.get(joint)?;
        Ok(JointReaction {
            force: state.force,
            torque: state.torque,
        })
    }

    /// Sets or clears one axis motor, and wakes both bodies.
    ///
    /// # Errors
    /// Fails on a stale handle, a distance joint or a bad motor.
    pub fn set_joint_motor(
        &mut self,
        joint: JointHandle,
        dof: JointDof,
        motor: Option<JointMotor>,
    ) -> Result<(), PhysicsError> {
        let state = self.joints.get(joint)?;
        if !matches!(state.desc.kind, JointKind::Axes(_)) {
            return Err(PhysicsError::InvalidJoint);
        }
        let mut desc = state.desc;
        if let JointKind::Axes(axes) = &mut desc.kind {
            axes[dof.index()].motor = motor;
        }
        desc.validate()?;
        let state = self.joints.get_mut(joint)?;
        state.desc = desc;
        state.impulses[dof.index() + crate::joint::MOTOR_SLOT] = 0.0;
        for body in [desc.a, desc.b] {
            let dense = self.bodies.dense_index(body)?;
            self.bodies.wake(dense);
        }
        Ok(())
    }

    /// Fills anchors and frames so the joint sits at `at`, turned by `rotation`, as things stand now.
    ///
    /// # Errors
    /// Fails on a stale body.
    pub fn joint_at(
        &self,
        desc: JointDesc,
        at: GridPosition,
        rotation: Quat,
    ) -> Result<JointDesc, PhysicsError> {
        let a = self.bodies.dense_index(desc.a)?;
        let b = self.bodies.dense_index(desc.b)?;
        let local = |dense: usize| {
            let inverse = self.bodies.rotation[dense].conjugate();
            let offset = at.offset_from(self.bodies.position(dense)).as_vec3();
            (inverse.rotate_vec3(offset), inverse * rotation)
        };
        let ((anchor_a, frame_a), (anchor_b, frame_b)) = (local(a), local(b));
        Ok(JointDesc {
            anchor_a,
            anchor_b,
            frame_a,
            frame_b,
            ..desc
        })
    }

    /// # Errors
    /// Fails on a stale handle.
    pub fn wake(&mut self, body: BodyHandle) -> Result<(), PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        self.bodies.wake(dense);
        Ok(())
    }

    /// Whatever rested on `body` loses it, so it can not stay asleep.
    fn wake_contacts_of(&mut self, body: BodyHandle) {
        for entry in &self.contact_cache {
            let other = if entry.a == body {
                entry.b
            } else if entry.b == body {
                entry.a
            } else {
                continue;
            };
            if let Ok(dense) = self.bodies.dense_index(other) {
                self.bodies.wake(dense);
            }
        }
    }

    /// # Errors
    /// Fails on a stale handle.
    pub fn body_state(&self, body: BodyHandle) -> Result<BodyState, PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        Ok(BodyState {
            kind: self.bodies.kind[dense],
            frame: self.bodies.frame[dense],
            position: GridPosition {
                cell: self.bodies.cell[dense],
                local: self.bodies.local[dense],
            },
            rotation: self.bodies.rotation[dense],
            linear_velocity: self.bodies.linear[dense],
            angular_velocity: self.bodies.angular[dense],
            asleep: self.bodies.asleep[dense],
        })
    }

    /// # Errors
    /// Fails on a stale handle, a non dynamic body or a non finite force.
    pub fn apply_force(&mut self, body: BodyHandle, force: Vec3) -> Result<(), PhysicsError> {
        let dense = self.dynamic_index(body)?;
        finite(force, "force")?;
        let next = self.bodies.force[dense] + force;
        finite(next, "force")?;
        self.bodies.force[dense] = next;
        self.bodies.wake(dense);
        Ok(())
    }

    /// # Errors
    /// Fails on a stale handle, a non dynamic body or a non finite torque.
    pub fn apply_torque(&mut self, body: BodyHandle, torque: Vec3) -> Result<(), PhysicsError> {
        let dense = self.dynamic_index(body)?;
        finite(torque, "torque")?;
        let next = self.bodies.torque[dense] + torque;
        finite(next, "torque")?;
        self.bodies.torque[dense] = next;
        self.bodies.wake(dense);
        Ok(())
    }

    /// # Errors
    /// Fails on a stale handle, a non dynamic body or a non finite impulse.
    pub fn apply_impulse(&mut self, body: BodyHandle, impulse: Vec3) -> Result<(), PhysicsError> {
        let dense = self.dynamic_index(body)?;
        finite(impulse, "impulse")?;
        let next = self.bodies.linear[dense] + impulse * self.bodies.inverse_mass[dense];
        finite(next, "velocity")?;
        self.bodies.linear[dense] = next;
        self.bodies.wake(dense);
        Ok(())
    }

    /// Impulse at `offset` from the center of mass, in frame axes.
    ///
    /// # Errors
    /// Fails on a stale handle, a non dynamic body or non finite input.
    pub fn apply_impulse_at(
        &mut self,
        body: BodyHandle,
        impulse: Vec3,
        offset: Vec3,
    ) -> Result<(), PhysicsError> {
        let dense = self.dynamic_index(body)?;
        finite(impulse, "impulse")?;
        finite(offset, "offset")?;
        let linear = self.bodies.linear[dense] + impulse * self.bodies.inverse_mass[dense];
        let momentum = self.bodies.momentum[dense] + offset.cross(impulse);
        let angular = apply_diagonal(
            self.bodies.rotation[dense],
            self.bodies.inverse_inertia[dense],
            momentum,
        );
        finite(linear, "velocity")?;
        finite(momentum, "momentum")?;
        finite(angular, "velocity")?;
        self.bodies.linear[dense] = linear;
        self.bodies.momentum[dense] = momentum;
        self.bodies.angular[dense] = angular;
        self.bodies.wake(dense);
        Ok(())
    }

    /// # Errors
    /// Fails on a stale handle, a static body or non finite input.
    pub fn set_velocity(
        &mut self,
        body: BodyHandle,
        linear: Vec3,
        angular: Vec3,
    ) -> Result<(), PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        if self.bodies.kind[dense] == BodyKind::Static {
            return Err(PhysicsError::NotDynamic);
        }
        finite(linear, "velocity")?;
        finite(angular, "velocity")?;
        let momentum = momentum_from_velocity(
            self.bodies.kind[dense],
            self.bodies.rotation[dense],
            self.bodies.inverse_inertia[dense],
            angular,
        );
        finite(momentum, "momentum")?;
        self.bodies.linear[dense] = linear;
        self.bodies.angular[dense] = angular;
        self.bodies.momentum[dense] = momentum;
        self.bodies.wake(dense);
        Ok(())
    }

    /// Moves a body to another frame, keeping its motion in the root frame.
    ///
    /// # Errors
    /// Fails on a stale handle, an unknown frame, a body with joints, or a result out of range.
    pub fn transfer_to_frame(
        &mut self,
        body: BodyHandle,
        frame: FrameId,
    ) -> Result<(), PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        if self.joints.touching(body).next().is_some() {
            return Err(PhysicsError::BodyHasJoints);
        }
        let from = self.bodies.frame[dense];
        let position = GridPosition {
            cell: self.bodies.cell[dense],
            local: self.bodies.local[dense],
        };
        let moved = self.frames.transfer(
            from,
            frame,
            FrameState {
                position: position.to_dvec3(),
                rotation: DQuat::from_quat(self.bodies.rotation[dense]),
                linear_velocity: DVec3::from_vec3(self.bodies.linear[dense]),
                angular_velocity: DVec3::from_vec3(self.bodies.angular[dense]),
            },
        )?;
        let next = GridPosition::from_dvec3(moved.position).ok_or(PhysicsError::OutOfRange)?;
        let linear = moved.linear_velocity.as_vec3();
        let angular = moved.angular_velocity.as_vec3();
        let rotation = moved.rotation.as_quat().normalize();
        finite(linear, "velocity")?;
        finite(angular, "velocity")?;
        let momentum = momentum_from_velocity(
            self.bodies.kind[dense],
            rotation,
            self.bodies.inverse_inertia[dense],
            angular,
        );
        finite(momentum, "momentum")?;
        self.bodies.cell[dense] = next.cell;
        self.bodies.local[dense] = next.local;
        self.bodies.rotation[dense] = rotation;
        self.bodies.linear[dense] = linear;
        self.bodies.angular[dense] = angular;
        self.bodies.momentum[dense] = momentum;
        self.bodies.frame[dense] = frame;
        self.bodies.wake(dense);
        self.recheck_sleepers |= self.bodies.kind[dense] != BodyKind::Dynamic;
        self.wake_contacts_of(body);
        Ok(())
    }

    /// Applies and clears the buffer. A bad command is reported, the rest still run.
    pub fn apply_commands(&mut self, buffer: &mut CommandBuffer) -> ApplyReport {
        let mut report = ApplyReport::default();
        for (index, command) in buffer.commands.drain(..).enumerate() {
            let result = match command {
                Command::CreateBody(desc) => self.spawn(desc).map(|body| {
                    report.created.push(body);
                }),
                Command::DestroyBody(body) => self.despawn(body),
                Command::ApplyForce(body, force) => self.apply_force(body, force),
                Command::ApplyTorque(body, torque) => self.apply_torque(body, torque),
                Command::ApplyImpulse(body, impulse) => self.apply_impulse(body, impulse),
                Command::ApplyImpulseAt(body, impulse, offset) => {
                    self.apply_impulse_at(body, impulse, offset)
                }
                Command::SetVelocity(body, linear, angular) => {
                    self.set_velocity(body, linear, angular)
                }
                Command::TransferToFrame(body, frame) => self.transfer_to_frame(body, frame),
                Command::Wake(body) => self.wake(body),
                Command::CreateJoint(desc) => self.add_joint(desc).map(|joint| {
                    report.created_joints.push(joint);
                }),
                Command::DestroyJoint(joint) => self.remove_joint(joint),
                Command::SetJointMotor(joint, dof, motor) => {
                    self.set_joint_motor(joint, dof, motor)
                }
            };
            if let Err(error) = result {
                report.rejected.push((index, error));
            }
        }
        report
    }

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
        let mut steps = 0;
        while self.accumulator >= timestep && steps < self.settings.max_steps_per_advance {
            self.step_fixed()?;
            self.accumulator -= timestep;
            steps += 1;
        }
        Ok(Advance {
            steps,
            alpha: (self.accumulator / timestep).min(1.0) as f32,
            overloaded: self.accumulator >= timestep,
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
        let started = Instant::now();
        self.prepare_frame_gravity()?;
        self.joints.refresh_exclusions();
        find_broad_pairs(&self.bodies, &mut self.contact_scratch);
        plan_wakes(
            &self.bodies,
            &self.contact_cache,
            &self.joints,
            self.contact_scratch.pairs(),
            self.recheck_sleepers,
            &mut self.sleep,
        );
        let count = self.bodies.len();
        self.scratch.clear();
        self.scratch
            .extend((0..count).map(|dense| starting_state(&self.bodies, dense)));

        let chunk_size = count
            .div_ceil(dispatch.parallelism().max(1))
            .max(MIN_BODIES_PER_JOB);
        let bodies = &self.bodies;
        let frame_gravity = &self.frame_gravity;
        let awake = self.sleep.awake();
        let dt = self.settings.fixed_timestep;
        let substeps = self.settings.substeps;
        let part = dt / substeps as f32;
        let mut next_cache = self.contact_cache.clone();
        prepare_contacts(
            bodies,
            awake,
            &self.joints,
            &self.scratch,
            &self.contact_cache,
            &mut self.contact_scratch,
            dt,
        );
        for substep in 0..substeps {
            run_jobs(&mut self.scratch, chunk_size, dispatch, &|dense, slot| {
                integrate_velocity(bodies, frame_gravity, awake[dense], dense, slot, part);
            })?;
            solve_substep(
                bodies,
                awake,
                &self.joints,
                &mut self.scratch,
                &mut self.contact_scratch,
                substep,
                part,
            );
            run_jobs(&mut self.scratch, chunk_size, dispatch, &|dense, slot| {
                integrate_position(bodies, awake[dense], dense, slot, part);
            })?;
        }
        finish_contacts(bodies, awake, &mut next_cache, &self.contact_scratch);

        self.commit()?;
        self.contact_cache = next_cache;
        let (points, rows, impulses) = self.contact_scratch.joint_parts();
        self.joints.store(impulses, points, rows, part);
        update_sleep(
            &mut self.bodies,
            &self.joints,
            self.contact_scratch.touching(),
            &mut self.sleep,
            dt,
        );
        self.recheck_sleepers = false;
        self.tick += 1;
        self.times.record(started.elapsed());
        Ok(())
    }

    fn prepare_frame_gravity(&mut self) -> Result<(), PhysicsError> {
        self.frame_gravity.clear();
        let gravity = DVec3::from_vec3(self.settings.gravity);
        for index in 0..self.frames.len() {
            let rotation = self.frames.root_rotation(FrameId(index as u32))?;
            self.frame_gravity
                .push(rotation.conjugate().mul_vec3(gravity).as_vec3());
        }
        Ok(())
    }

    fn commit(&mut self) -> Result<(), PhysicsError> {
        for result in &self.scratch {
            if result.out_of_range {
                return Err(PhysicsError::OutOfRange);
            }
            finite(result.linear, "velocity")?;
            finite(result.angular, "velocity")?;
            finite(result.momentum, "momentum")?;
            if !result.rotation.is_finite() {
                return Err(PhysicsError::NonFinite("rotation"));
            }
        }
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
        Ok(())
    }

    fn dynamic_index(&self, body: BodyHandle) -> Result<usize, PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        if self.bodies.kind[dense] == BodyKind::Dynamic {
            Ok(dense)
        } else {
            Err(PhysicsError::NotDynamic)
        }
    }
}

/// Runs `work` for every body on the host. Fails if the host skipped a job.
fn run_jobs(
    state: &mut [Integrated],
    chunk_size: usize,
    dispatch: &dyn JobDispatch,
    work: &(dyn Fn(usize, &mut Integrated) + Sync),
) -> Result<(), PhysicsError> {
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
    frame_gravity: &[Vec3],
    awake: bool,
    dense: usize,
    slot: &mut Integrated,
    dt: f32,
) {
    slot.recovery_linear = Vec3::ZERO;
    slot.recovery_angular = Vec3::ZERO;
    if bodies.kind[dense] != BodyKind::Dynamic || !awake {
        return;
    }
    let gravity = frame_gravity[bodies.frame[dense].0 as usize];
    slot.momentum += bodies.torque[dense] * dt;
    slot.linear += (gravity + bodies.force[dense] * bodies.inverse_mass[dense]) * dt;
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

fn finite(value: Vec3, what: &'static str) -> Result<(), PhysicsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PhysicsError::NonFinite(what))
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use algebrix::IVec3;

    use super::*;
    use crate::{CELL_SIZE, Shape};

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    fn no_gravity() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap()
    }

    fn cube() -> RigidBodyDesc {
        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
    }

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

    struct SkipsJobs;

    impl JobDispatch for SkipsJobs {
        fn parallelism(&self) -> usize {
            2
        }

        fn run(&self, _jobs: usize, _job: &(dyn Fn(usize) + Sync)) {}
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
    fn force_and_torque_last_one_step() {
        let mut world = no_gravity();
        let body = world.spawn(cube().density(1.0)).unwrap();
        world.apply_force(body, Vec3::new(60.0, 0.0, 0.0)).unwrap();
        world.step_fixed().unwrap();
        let after_one = world.body_state(body).unwrap().linear_velocity.x;
        assert!((after_one - 1.0).abs() < 1e-4);
        world.step_fixed().unwrap();
        assert_eq!(world.body_state(body).unwrap().linear_velocity.x, after_one);
    }

    #[test]
    fn impulse_changes_velocity_by_impulse_over_mass() {
        let mut world = no_gravity();
        let body = world.spawn(cube().density(4.0)).unwrap();
        world.apply_impulse(body, Vec3::new(8.0, 0.0, 0.0)).unwrap();
        let state = world.body_state(body).unwrap();
        assert!((state.linear_velocity.x - 2.0).abs() < 1e-5);
        assert_eq!(state.angular_velocity, Vec3::ZERO);
    }

    #[test]
    fn off_center_impulse_makes_it_rotate() {
        let mut world = no_gravity();
        let body = world.spawn(cube().density(6.0)).unwrap();
        world
            .apply_impulse_at(body, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.5, 0.0, 0.0))
            .unwrap();
        // cube of mass 6 and side 1 has inertia 1
        let state = world.body_state(body).unwrap();
        assert!((state.angular_velocity.z - 0.5).abs() < 1e-5);
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        let (axis, angle) = world.body_state(body).unwrap().rotation.to_axis_angle();
        assert!((angle - 0.5).abs() < 1e-3);
        assert!((axis.z - 1.0).abs() < 1e-3);
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
    fn static_never_moves_and_kinematic_ignores_gravity() {
        let mut world = world();
        let fixed = world
            .spawn(RigidBodyDesc::fixed(Shape::sphere(1.0)))
            .unwrap();
        let moving = world
            .spawn(
                RigidBodyDesc::kinematic(Shape::sphere(1.0))
                    .linear_velocity(Vec3::new(6.0, 0.0, 0.0)),
            )
            .unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        assert_eq!(
            world.body_state(fixed).unwrap().position,
            GridPosition::ORIGIN
        );
        let state = world.body_state(moving).unwrap();
        assert!((state.position.local.x - 6.0).abs() < 1e-3);
        assert_eq!(state.position.local.y, 0.0);
        assert_eq!(
            world.apply_force(fixed, Vec3::X),
            Err(PhysicsError::NotDynamic)
        );
        assert_eq!(
            world.apply_force(moving, Vec3::X),
            Err(PhysicsError::NotDynamic)
        );
    }

    #[test]
    fn far_from_origin_moves_same_as_at_origin() {
        let mut world = world();
        let near = world.spawn(cube()).unwrap();
        let far = world
            .spawn(cube().position(GridPosition {
                cell: IVec3::new(1_000_000, -500_000, 2_000_000),
                local: Vec3::ZERO,
            }))
            .unwrap();
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        let near = world.body_state(near).unwrap();
        let far = world.body_state(far).unwrap();
        assert_eq!(near.linear_velocity, far.linear_velocity);
        assert_eq!(near.position.local, far.position.local);
    }

    #[test]
    fn crossing_cells_keeps_motion_going() {
        let mut world = no_gravity();
        let body = world
            .spawn(cube().linear_velocity(Vec3::new(CELL_SIZE * 2.0, 0.0, 0.0)))
            .unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        let position = world.body_state(body).unwrap().position;
        assert_eq!(position.cell.x, 2);
        assert!(position.local.x.abs() < 1.0);
    }

    #[test]
    fn gravity_points_down_in_root_inside_a_rotated_frame() {
        let mut world = world();
        let flipped = world
            .add_frame(FrameDesc {
                rotation: DQuat::from_axis_angle(DVec3::Z, std::f64::consts::PI),
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        let body = world.spawn(cube().frame(flipped)).unwrap();
        world.step_fixed().unwrap();
        assert!(world.body_state(body).unwrap().linear_velocity.y > 0.0);
    }

    #[test]
    fn transfer_keeps_root_motion() {
        let mut world = no_gravity();
        let moving = world
            .add_frame(FrameDesc {
                translation: DVec3::new(100.0, 0.0, 0.0),
                linear_velocity: DVec3::new(0.0, 5.0, 0.0),
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        let body = world
            .spawn(
                cube()
                    .frame(moving)
                    .local_position(Vec3::new(1.0, 2.0, 3.0)),
            )
            .unwrap();
        world.transfer_to_frame(body, FrameId::ROOT).unwrap();
        let state = world.body_state(body).unwrap();
        assert_eq!(state.frame, FrameId::ROOT);
        assert_eq!(state.position.local, Vec3::new(101.0, 2.0, 3.0));
        assert_eq!(state.linear_velocity, Vec3::new(0.0, 5.0, 0.0));
        assert_eq!(
            world.transfer_to_frame(body, FrameId(9)),
            Err(PhysicsError::InvalidFrame)
        );
    }

    #[test]
    fn commands_apply_in_order_and_report_bad_ones() {
        let mut world = no_gravity();
        let existing = world.spawn(cube()).unwrap();
        let stale = world.spawn(cube()).unwrap();
        world.despawn(stale).unwrap();
        let mut buffer = CommandBuffer::new();
        buffer.push(Command::CreateBody(cube()));
        buffer.push(Command::ApplyImpulse(stale, Vec3::X));
        buffer.push(Command::ApplyImpulse(
            existing,
            Vec3::new(f32::NAN, 0.0, 0.0),
        ));
        buffer.push(Command::SetVelocity(existing, Vec3::X, Vec3::ZERO));
        buffer.push(Command::DestroyBody(stale));
        let report = world.apply_commands(&mut buffer);
        assert!(buffer.is_empty());
        assert_eq!(report.created.len(), 1);
        assert_eq!(
            report.rejected,
            vec![
                (1, PhysicsError::InvalidHandle),
                (2, PhysicsError::NonFinite("impulse")),
                (4, PhysicsError::InvalidHandle),
            ]
        );
        assert_eq!(world.body_state(existing).unwrap().linear_velocity, Vec3::X);
    }

    #[test]
    fn stale_handle_and_bad_input_fail_safe() {
        let mut world = world();
        let body = world.spawn(cube()).unwrap();
        world.despawn(body).unwrap();
        assert_eq!(world.body_state(body), Err(PhysicsError::InvalidHandle));
        assert_eq!(world.despawn(body), Err(PhysicsError::InvalidHandle));
        assert_eq!(
            world.spawn(cube().frame(FrameId(5))),
            Err(PhysicsError::InvalidFrame)
        );
        assert!(
            PhysicsWorld::new(PhysicsSettings {
                fixed_timestep: 0.0,
                ..PhysicsSettings::default()
            })
            .is_err()
        );
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
    fn stacked_forces_can_not_overflow() {
        let mut world = no_gravity();
        let body = world.spawn(cube()).unwrap();
        world.apply_force(body, Vec3::splat(f32::MAX)).unwrap();
        assert_eq!(
            world.apply_force(body, Vec3::splat(f32::MAX)),
            Err(PhysicsError::NonFinite("force"))
        );
        world.apply_torque(body, Vec3::splat(f32::MAX)).unwrap();
        assert_eq!(
            world.apply_torque(body, Vec3::splat(f32::MAX)),
            Err(PhysicsError::NonFinite("torque"))
        );
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
        let before = world.body_state(body).unwrap();
        assert_eq!(
            world.step_with(&SkipsJobs),
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
