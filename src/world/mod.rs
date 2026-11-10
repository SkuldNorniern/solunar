mod step;

use algebrix::{DQuat, DVec3, Quat, Vec3};

use crate::body::{Bodies, apply_diagonal, momentum_from_velocity, validate_desc};
use crate::field::Fields;
use crate::gravity::{GravityId, GravitySource, MAX_GRAVITY_SOURCES, StepGravity};
use crate::joint::JointKind;
use crate::profile::{PhaseTimes, StepTimes};
use crate::sensor::Sensors;
use crate::sleep::SleepScratch;
use crate::solver::{CachedManifold, ContactScratch};
use crate::{
    ApplyReport, BodyHandle, BodyKind, BodyState, Command, CommandBuffer, ContactEvent, FrameDesc,
    FrameId, FrameState, Frames, GridPosition, JointDesc, JointDof, JointHandle, JointMotor,
    JointReaction, Joints, PhysicsError, RigidBodyDesc, StepReport,
};

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

/// Body pairs, lower handle first.
type Pairs = Vec<(BodyHandle, BodyHandle)>;

#[derive(Debug, Clone)]
pub struct PhysicsWorld {
    pub(crate) settings: PhysicsSettings,
    pub(crate) frames: Frames,
    pub(crate) bodies: Bodies,
    pub(crate) joints: Joints,
    pub(crate) tick: u64,
    pub(crate) accumulator: f64,
    pub(crate) contact_cache: Vec<CachedManifold>,
    /// Last step's cache, kept for its memory: the next one is built in it.
    spare_cache: Vec<CachedManifold>,
    finish_scratch: crate::solver::FinishScratch,
    /// Touching pairs before and after a step, for the contact events.
    touched: (Pairs, Pairs),
    contact_events: Vec<ContactEvent>,
    pub(crate) sensors: Sensors,
    pub(crate) breakables: crate::fracture::Breakables,
    pub(crate) structures: crate::structure::Structures,
    pub(crate) atmospheres: crate::atmosphere::Atmospheres,
    pub(crate) gas: crate::gas::Gas,
    pub(crate) heat: crate::heat::Heat,
    broken_joints: Vec<JointHandle>,
    frame_changes: Vec<(BodyHandle, FrameId, FrameId)>,
    pub(crate) fields: Fields,
    failed_body: Option<BodyHandle>,
    /// A static or kinematic body appeared, so sleepers must check for new contact.
    pub(crate) recheck_sleepers: bool,
    pub(crate) contact_scratch: ContactScratch,
    sleep: SleepScratch,
    scratch: Vec<Integrated>,
    pub(crate) gravity: StepGravity,
    pub(crate) gravity_sources: Vec<Option<GravitySource>>,
    times: StepTimes,
    phases: PhaseTimes,
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
        mut frames: Frames,
        bodies: Bodies,
        tick: u64,
        accumulator: f64,
    ) -> Self {
        frames.advance_to(tick as f64 * f64::from(settings.fixed_timestep));
        Self {
            settings,
            frames,
            bodies,
            joints: Joints::default(),
            tick,
            accumulator,
            contact_cache: Vec::new(),
            spare_cache: Vec::new(),
            finish_scratch: crate::solver::FinishScratch::default(),
            touched: (Vec::new(), Vec::new()),
            contact_events: Vec::new(),
            sensors: Sensors::default(),
            breakables: crate::fracture::Breakables::default(),
            structures: crate::structure::Structures::default(),
            atmospheres: crate::atmosphere::Atmospheres::default(),
            gas: crate::gas::Gas::default(),
            heat: crate::heat::Heat::default(),
            broken_joints: Vec::new(),
            frame_changes: Vec::new(),
            fields: Fields::default(),
            failed_body: None,
            recheck_sleepers: false,
            contact_scratch: ContactScratch::default(),
            sleep: SleepScratch::default(),
            scratch: Vec::new(),
            gravity: StepGravity::default(),
            gravity_sources: Vec::new(),
            times: StepTimes::default(),
            phases: PhaseTimes::default(),
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

    pub fn phase_times(&self) -> PhaseTimes {
        self.phases
    }

    /// The body that made the last step fail, when one did. A failed step keeps failing until
    /// this body is fixed or removed, there can be more after it.
    pub fn failed_body(&self) -> Option<BodyHandle> {
        self.failed_body
    }

    /// Pairs that started or stopped pushing on each other in the last step, lower handle
    /// first, sorted. An `Ended` pair can name a body removed since.
    pub fn contact_events(&self) -> &[ContactEvent] {
        &self.contact_events
    }

    /// # Errors
    /// Fails on an unknown parent, a too deep chain or non finite values.
    pub fn add_frame(&mut self, desc: FrameDesc) -> Result<FrameId, PhysicsError> {
        self.frames.add(desc)
    }

    /// # Errors
    /// Fails on an unknown frame, a non finite center, mass or radius, or too many sources.
    pub fn add_gravity(&mut self, source: GravitySource) -> Result<GravityId, PhysicsError> {
        if !self.frames.contains(source.frame) || !source.is_valid() {
            return Err(PhysicsError::InvalidSettings);
        }
        let free = self.gravity_sources.iter().position(Option::is_none);
        let index = match free {
            Some(index) => index,
            None if self.gravity_sources.len() < MAX_GRAVITY_SOURCES => {
                self.gravity_sources.push(None);
                self.gravity_sources.len() - 1
            }
            None => return Err(PhysicsError::OutOfRange),
        };
        self.gravity_sources[index] = Some(source);
        Ok(GravityId(index as u32))
    }

    /// Returns false for a source that was not there.
    pub fn remove_gravity(&mut self, id: GravityId) -> bool {
        self.gravity_sources
            .get_mut(id.0 as usize)
            .and_then(Option::take)
            .is_some()
    }

    pub fn gravity_source(&self, id: GravityId) -> Option<&GravitySource> {
        self.gravity_sources.get(id.0 as usize)?.as_ref()
    }

    /// What a body in `frame` at `position` moving at `velocity` falls with, in that frame's axes:
    /// uniform gravity, the sources and the frame's own motion, now.
    ///
    /// # Errors
    /// Fails on an unknown frame.
    pub fn gravity_at(
        &self,
        frame: FrameId,
        position: GridPosition,
        velocity: Vec3,
    ) -> Result<Vec3, PhysicsError> {
        self.frames.get(frame)?;
        let mut now = StepGravity::default();
        now.prepare(self.settings.gravity, &self.frames, &self.gravity_sources)?;
        Ok(now.at(frame, position, velocity))
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
        let shape = self.bodies.shape[self.bodies.dense_index(body)?];
        self.bodies.remove(body)?;
        self.free_owned(shape);
        self.breakables.forget(body);
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

    /// Joints that broke at the end of the last step, their handles no longer good.
    pub fn broken_joints(&self) -> &[JointHandle] {
        &self.broken_joints
    }

    /// Bodies the last step moved between frames for their reach: body, from, to.
    pub fn frame_changes(&self) -> &[(BodyHandle, FrameId, FrameId)] {
        &self.frame_changes
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

    /// Sets how a body's surface moves, in its own frame, as a belt. What touches it wakes.
    ///
    /// # Errors
    /// Fails on a stale handle or a non finite velocity.
    pub fn set_surface_velocity(
        &mut self,
        body: BodyHandle,
        velocity: Vec3,
    ) -> Result<(), PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        finite(velocity, "velocity")?;
        self.bodies.surface[dense] = velocity;
        self.bodies.wake(dense);
        self.wake_contacts_of(body);
        Ok(())
    }

    /// Moves a body without any physics in between. It wakes, and so does what touched it.
    ///
    /// # Errors
    /// Fails on a stale handle or a position out of range.
    pub fn teleport(
        &mut self,
        body: BodyHandle,
        position: GridPosition,
    ) -> Result<(), PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        let position = position.rebased().ok_or(PhysicsError::OutOfRange)?;
        self.bodies.tree.clear();
        self.bodies.cell[dense] = position.cell;
        self.bodies.local[dense] = position.local;
        self.bodies.wake(dense);
        self.recheck_sleepers |= self.bodies.kind[dense] != BodyKind::Dynamic;
        self.wake_contacts_of(body);
        Ok(())
    }

    /// Whatever rested on `body` loses it, so it can not stay asleep.
    pub(crate) fn wake_contacts_of(&mut self, body: BodyHandle) {
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

    /// Mass an impulse along `direction` at `offset` from the center of mass meets, turning
    /// included. Infinite for static and kinematic bodies.
    ///
    /// # Errors
    /// Fails on a stale handle or non finite input.
    pub fn mass_at(
        &self,
        body: BodyHandle,
        offset: Vec3,
        direction: Vec3,
    ) -> Result<f32, PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        finite(offset, "offset")?;
        finite(direction, "direction")?;
        if self.bodies.kind[dense] != BodyKind::Dynamic {
            return Ok(f32::INFINITY);
        }
        let arm = offset.cross(direction);
        let turn = apply_diagonal(
            self.bodies.rotation[dense],
            self.bodies.inverse_inertia[dense],
            arm,
        );
        let inverse = self.bodies.inverse_mass[dense] * direction.length_squared() + arm.dot(turn);
        Ok(if inverse > 0.0 {
            1.0 / inverse
        } else {
            f32::INFINITY
        })
    }

    /// Sets a kinematic body's velocity so the next step ends at `position` and `rotation`, in
    /// its frame. It keeps that velocity after, set a new target or a velocity each step.
    ///
    /// # Errors
    /// Fails on a stale handle, a body that is not kinematic, bad input or a target too far.
    pub fn set_kinematic_target(
        &mut self,
        body: BodyHandle,
        position: GridPosition,
        rotation: Quat,
    ) -> Result<(), PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        if self.bodies.kind[dense] != BodyKind::Kinematic {
            return Err(PhysicsError::NotKinematic);
        }
        finite(position.local, "position")?;
        if !rotation.is_finite() || rotation.length_squared() < 1e-12 {
            return Err(PhysicsError::NonFinite("rotation"));
        }
        let dt = self.settings.fixed_timestep;
        let travel = position.offset_from(self.bodies.position(dense)).as_vec3();
        let turn =
            (rotation.normalize() * self.bodies.rotation[dense].conjugate()).to_scaled_axis();
        self.set_velocity(body, travel / dt, turn / dt)
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
        self.bodies.tree.clear();
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
                Command::SetKinematicTarget(body, position, rotation) => {
                    self.set_kinematic_target(body, position, rotation)
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

    fn dynamic_index(&self, body: BodyHandle) -> Result<usize, PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        if self.bodies.kind[dense] == BodyKind::Dynamic {
            Ok(dense)
        } else {
            Err(PhysicsError::NotDynamic)
        }
    }
}

fn finite(value: Vec3, what: &'static str) -> Result<(), PhysicsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PhysicsError::NonFinite(what))
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::{RigidBodyDesc, Shape};

    pub(crate) fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    pub(crate) fn no_gravity() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap()
    }

    pub(crate) fn ground(world: &mut PhysicsWorld) {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(50.0, 1.0, 50.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use algebrix::IVec3;

    use super::*;
    use crate::world::testing::{no_gravity, world};
    use crate::{CELL_SIZE, Shape};

    pub(super) fn cube() -> RigidBodyDesc {
        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
    }

    #[test]
    fn an_impulse_of_mass_at_changes_the_point_speed_by_one() {
        let mut world = no_gravity();
        let body = world
            .spawn(RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(
                0.5, 1.0, 0.25,
            ))))
            .unwrap();
        let wall = world
            .spawn(RigidBodyDesc::fixed(Shape::sphere(1.0)).local_position(Vec3::X * 10.0))
            .unwrap();
        let offset = Vec3::new(0.3, 0.8, -0.1);
        let direction = Vec3::new(1.0, 0.2, 0.4).normalize();
        let mass = world.mass_at(body, offset, direction).unwrap();
        world
            .apply_impulse_at(body, direction * mass, offset)
            .unwrap();
        let state = world.body_state(body).unwrap();
        let speed = (state.linear_velocity + state.angular_velocity.cross(offset)).dot(direction);
        assert!((speed - 1.0).abs() < 1e-5, "{speed}");
        assert_eq!(world.mass_at(wall, offset, direction), Ok(f32::INFINITY));
        assert!(
            world
                .mass_at(body, Vec3::splat(f32::NAN), direction)
                .is_err()
        );
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
}
