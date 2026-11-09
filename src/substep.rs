use algebrix::{Quat, Vec3};

use crate::body::{Bodies, apply_diagonal};
use crate::world::Integrated;
use crate::{BodyHandle, BodyKind, GridPosition, JobDispatch, PhysicsError};

/// Something stepped with the rigid bodies inside each substep, as water that pushes them and
/// is pushed back. Called after the substep's gravity and forces, before its solve, so contacts
/// and the hook's pushes are solved together. Hooks run in the order given.
pub trait SubstepHook {
    /// # Errors
    /// An error fails the whole step, no body state of it is kept.
    fn before_solve(
        &mut self,
        bodies: &mut SubstepBodies<'_>,
        dt: f32,
        dispatch: &dyn JobDispatch,
    ) -> Result<(), PhysicsError>;
}

/// Where a body is and how fast inside a substep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubstepPose {
    pub position: GridPosition,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

/// The bodies as a substep has them, for a [`SubstepHook`] to read and push.
pub struct SubstepBodies<'a> {
    pub(crate) bodies: &'a Bodies,
    pub(crate) state: &'a mut [crate::world::Integrated],
    pub(crate) awake: &'a [bool],
    /// Sleepers a hook pushed, woken once the step is done.
    pub(crate) woken: &'a mut Vec<usize>,
}

impl SubstepBodies<'_> {
    pub fn pose(&self, body: BodyHandle) -> Option<SubstepPose> {
        let dense = self.bodies.dense_index(body).ok()?;
        let slot: &Integrated = &self.state[dense];
        Some(SubstepPose {
            position: slot.position,
            rotation: slot.rotation,
            linear_velocity: slot.linear,
            angular_velocity: slot.angular,
        })
    }

    /// Whether pushes move it, false for a fixed or kinematic body or a stale handle.
    pub fn is_dynamic(&self, body: BodyHandle) -> bool {
        self.bodies
            .dense_index(body)
            .is_ok_and(|dense| self.bodies.kind[dense] == BodyKind::Dynamic)
    }

    /// Adds an impulse through the middle and an angular impulse, now, to a dynamic body. A
    /// sleeping one is not moved this step, it wakes once the step is done.
    ///
    /// # Errors
    /// Fails on a stale handle or non finite input.
    pub fn push(
        &mut self,
        body: BodyHandle,
        impulse: Vec3,
        angular: Vec3,
    ) -> Result<(), PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        if !impulse.is_finite() || !angular.is_finite() {
            return Err(PhysicsError::NonFinite("substep push"));
        }
        if self.bodies.kind[dense] != BodyKind::Dynamic {
            return Ok(());
        }
        if !self.awake[dense] {
            if impulse != Vec3::ZERO || angular != Vec3::ZERO {
                self.woken.push(dense);
            }
            return Ok(());
        }
        let slot = &mut self.state[dense];
        slot.linear += impulse * self.bodies.inverse_mass[dense];
        slot.momentum += angular;
        slot.angular = apply_diagonal(
            slot.rotation,
            self.bodies.inverse_inertia[dense],
            slot.momentum,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhysicsSettings, PhysicsWorld, RigidBodyDesc, Sequential, Shape};

    /// Holds one body up against gravity, or fails when asked to.
    struct Lift {
        body: BodyHandle,
        fail: bool,
    }

    impl SubstepHook for Lift {
        fn before_solve(
            &mut self,
            bodies: &mut SubstepBodies<'_>,
            dt: f32,
            _: &dyn JobDispatch,
        ) -> Result<(), PhysicsError> {
            if self.fail {
                return Err(PhysicsError::InvalidSettings);
            }
            // 1 kg ball
            bodies.push(self.body, Vec3::new(0.0, 9.81 * dt, 0.0), Vec3::ZERO)
        }
    }

    fn ball(world: &mut PhysicsWorld) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .density(1.0 / (4.0 / 3.0 * std::f32::consts::PI * 0.125))
                    .local_position(Vec3::new(0.0, 5.0, 0.0)),
            )
            .unwrap()
    }

    #[test]
    fn a_hook_pushes_inside_every_substep_and_a_failure_keeps_the_last_step() {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let body = ball(&mut world);
        let mut lift = Lift { body, fail: false };
        for _ in 0..60 {
            world
                .step_with_hooks(&Sequential, &mut [&mut lift])
                .unwrap();
        }
        let state = world.body_state(body).unwrap();
        assert!((state.position.to_dvec3().y - 5.0).abs() < 1e-3);
        assert!(state.linear_velocity.length() < 1e-3);
        let saved = world.save_snapshot();
        lift.fail = true;
        assert!(
            world
                .step_with_hooks(&Sequential, &mut [&mut lift])
                .is_err()
        );
        assert_eq!(world.save_snapshot(), saved);
    }
}
