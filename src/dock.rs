use algebrix::{Quat, Vec3};

use crate::body::BodyHandle;
use crate::error::PhysicsError;
use crate::joint::{JointAxis, JointDesc, JointHandle, JointKind, JointMotor};
use crate::world::PhysicsWorld;

/// Where two bodies meet to dock: a point in the body's axes and how the port faces, its +x
/// pointing out. Two ports dock facing each other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DockPort {
    pub anchor: Vec3,
    pub facing: Quat,
}

/// How far two ports are from docked: apart, turned off facing each other, closing or opening,
/// turning against each other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DockFit {
    pub distance: f32,
    /// Radians off facing each other.
    pub angle: f32,
    pub speed: f32,
    pub spin: f32,
}

impl DockFit {
    /// Within `limits` on all four.
    pub fn within(&self, limits: &DockFit) -> bool {
        self.distance <= limits.distance
            && self.angle <= limits.angle
            && self.speed <= limits.speed
            && self.spin <= limits.spin
    }
}

/// A port's frame turned to face back along its +x: b's frame in a joint, so ports face.
fn facing_back(facing: Quat) -> Quat {
    (facing * Quat::from_rotation_y(std::f32::consts::PI)).normalize()
}

impl PhysicsWorld {
    /// How far `a`'s and `b`'s ports are from docked, now.
    ///
    /// # Errors
    /// Fails on a stale handle.
    pub fn docking_fit(
        &self,
        (a, port_a): (BodyHandle, DockPort),
        (b, port_b): (BodyHandle, DockPort),
    ) -> Result<DockFit, PhysicsError> {
        let (one, two) = (self.body_state(a)?, self.body_state(b)?);
        if one.frame != two.frame {
            return Err(PhysicsError::InvalidFrame);
        }
        let arm_a = one.rotation.rotate_vec3(port_a.anchor);
        let arm_b = two.rotation.rotate_vec3(port_b.anchor);
        let at_b = two.position.offset_from(one.position).as_vec3() + arm_b;
        let (frame_a, frame_b) = (
            one.rotation * port_a.facing,
            two.rotation * facing_back(port_b.facing),
        );
        let off = (frame_a.conjugate() * frame_b).normalize();
        let angle = 2.0 * off.w.abs().clamp(0.0, 1.0).acos();
        let velocity = |state: &crate::BodyState, arm: Vec3| {
            state.linear_velocity + state.angular_velocity.cross(arm)
        };
        Ok(DockFit {
            distance: (at_b - arm_a).length(),
            angle,
            speed: (velocity(&two, arm_b) - velocity(&one, arm_a)).length(),
            spin: (two.angular_velocity - one.angular_velocity).length(),
        })
    }

    /// Draws `b`'s port onto `a`'s, turned to face it, no harder than `force` newtons and
    /// newton meters and no faster than `speed` m/s and rad/s: a soft dock, held but giving.
    /// Its pull is between the two, so together they keep their momentum.
    ///
    /// # Errors
    /// As [`Self::add_joint`].
    pub fn dock_soft(
        &mut self,
        (a, port_a): (BodyHandle, DockPort),
        (b, port_b): (BodyHandle, DockPort),
        force: f32,
        speed: f32,
    ) -> Result<JointHandle, PhysicsError> {
        let pull = JointAxis::FREE.motor(JointMotor::Position {
            target: 0.0,
            max_speed: speed,
            max_force: force,
        });
        let desc = JointDesc::new(a, b, JointKind::Axes([pull; 6]))
            .anchors(port_a.anchor, port_b.anchor)
            .frames(port_a.facing, facing_back(port_b.facing));
        self.add_joint(desc)
    }

    /// Locks `a` and `b` together as they are, a hard dock, when their ports fit within
    /// `limits`. Locked where they are, not where the ports would meet, so it pulls on
    /// nothing and no speed jumps. `strength` newtons and newton meters break it, infinite
    /// holds. Undocking is [`Self::remove_joint`], they go on as they move.
    ///
    /// # Errors
    /// Fails when the ports do not fit, as [`Self::docking_fit`], or as [`Self::add_joint`].
    pub fn dock_hard(
        &mut self,
        (a, port_a): (BodyHandle, DockPort),
        (b, port_b): (BodyHandle, DockPort),
        limits: &DockFit,
        strength: f32,
    ) -> Result<JointHandle, PhysicsError> {
        if !self.docking_fit((a, port_a), (b, port_b))?.within(limits) {
            return Err(PhysicsError::OutOfRange);
        }
        let (one, two) = (self.body_state(a)?, self.body_state(b)?);
        // a's port as it is now, in b's axes, point and frame
        let point = one.position.offset_from(two.position).as_vec3()
            + one.rotation.rotate_vec3(port_a.anchor);
        let back = two.rotation.conjugate();
        let anchor_b = back.rotate_vec3(point);
        let frame_b = (back * one.rotation * port_a.facing).normalize();
        let desc = JointDesc::fixed(a, b)
            .anchors(port_a.anchor, anchor_b)
            .frames(port_a.facing, frame_b)
            .breaks_at(strength, strength);
        self.add_joint(desc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::testing::no_gravity;
    use crate::{RigidBodyDesc, Shape};

    fn ship(world: &mut PhysicsWorld, x: f32, velocity: Vec3, turn: Quat) -> BodyHandle {
        let mut desc = RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(1.0, 0.5, 0.5)))
            .local_position(Vec3::new(x, 0.0, 0.0))
            .rotation(turn)
            .linear_velocity(velocity);
        desc.can_sleep = false;
        world.spawn(desc).unwrap()
    }

    /// Out of a's +x end, and out of b's -x end.
    fn ports() -> (DockPort, DockPort) {
        (
            DockPort {
                anchor: Vec3::new(1.0, 0.0, 0.0),
                facing: Quat::IDENTITY,
            },
            DockPort {
                anchor: Vec3::new(-1.0, 0.0, 0.0),
                facing: Quat::from_rotation_y(std::f32::consts::PI),
            },
        )
    }

    fn momentum(world: &PhysicsWorld, bodies: &[BodyHandle]) -> Vec3 {
        bodies.iter().fold(Vec3::ZERO, |sum, body| {
            let dense = world.bodies.dense_index(*body).unwrap();
            sum + world.bodies.linear[dense] / world.bodies.inverse_mass[dense]
        })
    }

    #[test]
    fn a_soft_dock_draws_the_ports_together_then_a_hard_one_locks_without_a_jolt() {
        let mut world = no_gravity();
        let (port_a, port_b) = ports();
        let a = ship(&mut world, 0.0, Vec3::new(0.2, 0.0, 0.0), Quat::IDENTITY);
        // a meter off, turned 4 degrees and drifting in
        let b = ship(
            &mut world,
            3.0,
            Vec3::new(-0.1, 0.05, 0.0),
            Quat::from_rotation_z(4.0_f32.to_radians()),
        );
        let before = momentum(&world, &[a, b]);
        let soft = world
            .dock_soft((a, port_a), (b, port_b), 2_000.0, 0.5)
            .unwrap();
        let limits = DockFit {
            distance: 0.005,
            angle: 0.5_f32.to_radians(),
            speed: 0.01,
            spin: 0.01,
        };
        // too far to lock yet
        assert!(
            world
                .dock_hard((a, port_a), (b, port_b), &limits, f32::INFINITY)
                .is_err()
        );
        let mut fit = world.docking_fit((a, port_a), (b, port_b)).unwrap();
        for _ in 0..600 {
            world.step_fixed().unwrap();
            fit = world.docking_fit((a, port_a), (b, port_b)).unwrap();
            if fit.within(&limits) {
                break;
            }
        }
        assert!(fit.within(&limits), "{fit:?}");
        let pulled = momentum(&world, &[a, b]);
        assert!(
            (pulled - before).length() < 1e-3 * before.length().max(1.0),
            "{before:?} {pulled:?}"
        );
        world.remove_joint(soft).unwrap();
        let speed = |world: &PhysicsWorld, body| world.body_state(body).unwrap().linear_velocity;
        let (was_a, was_b) = (speed(&world, a), speed(&world, b));
        let hard = world
            .dock_hard((a, port_a), (b, port_b), &limits, f32::INFINITY)
            .unwrap();
        world.step_fixed().unwrap();
        // locked where they were: no jump in speed
        assert!(
            (speed(&world, a) - was_a).length() < 0.02,
            "{:?} {was_a:?}",
            speed(&world, a)
        );
        assert!((speed(&world, b) - was_b).length() < 0.02);
        // docked they go as one
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        let joined = world.docking_fit((a, port_a), (b, port_b)).unwrap();
        assert!(joined.speed < 1e-3 && joined.spin < 1e-3, "{joined:?}");
        // undocked they go on, momentum as it was
        world.remove_joint(hard).unwrap();
        world.step_fixed().unwrap();
        let after = momentum(&world, &[a, b]);
        assert!(
            (after - before).length() < 1e-2 * before.length().max(1.0),
            "{before:?} {after:?}"
        );
    }

    #[test]
    fn a_hard_dock_breaks_at_its_strength_and_ports_in_other_frames_do_not_fit() {
        let mut world = no_gravity();
        let (port_a, port_b) = ports();
        let a = ship(&mut world, 0.0, Vec3::ZERO, Quat::IDENTITY);
        let b = ship(&mut world, 2.0, Vec3::ZERO, Quat::IDENTITY);
        let loose = DockFit {
            distance: 0.01,
            angle: 0.01,
            speed: 0.01,
            spin: 0.01,
        };
        let joint = world
            .dock_hard((a, port_a), (b, port_b), &loose, 5_000.0)
            .unwrap();
        // pulled apart past 5 kN it lets go
        let mut broke = false;
        for _ in 0..5 {
            world.apply_force(b, Vec3::new(30_000.0, 0.0, 0.0)).unwrap();
            world.step_fixed().unwrap();
            broke |= world.broken_joints().contains(&joint);
        }
        assert!(broke);
        let frame = world
            .add_frame(crate::FrameDesc::new(crate::FrameId::ROOT))
            .unwrap();
        let mut desc = RigidBodyDesc::dynamic(Shape::sphere(0.5));
        desc.frame = frame;
        let other = world.spawn(desc).unwrap();
        assert!(world.docking_fit((a, port_a), (other, port_b)).is_err());
    }
}
