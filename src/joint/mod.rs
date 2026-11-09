use algebrix::{Quat, Vec3};

use crate::{BodyHandle, PhysicsError};

mod rows;
mod store;

pub(crate) use rows::{
    JointPoint, JointRow, MOTOR_SLOT, add_joint_loads, build_rows, capture_impulses, solve_point,
    solve_point_recovery, solve_row, solve_row_recovery, warm_start_point, warm_start_row,
};
pub use store::Joints;
pub(crate) use store::{JointSlot, JointState};

pub(crate) const JOINT_IMPULSES: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JointHandle {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

impl JointHandle {
    /// Slot and generation, for save data. Kept with a world's snapshot they mean the same thing
    /// again after it loads.
    pub fn to_raw(self) -> (u32, u32) {
        (self.index, self.generation)
    }

    /// From [`Self::to_raw`]. One that names no live slot is only stale: every call that takes it
    /// answers with an error.
    pub fn from_raw((index, generation): (u32, u32)) -> Self {
        Self { index, generation }
    }
}

/// Axes are the joint frame of body A. Linear is in meters, angular in radians.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointDof {
    LinearX,
    LinearY,
    LinearZ,
    AngularX,
    AngularY,
    AngularZ,
}

impl JointDof {
    pub(crate) fn index(self) -> usize {
        match self {
            Self::LinearX => 0,
            Self::LinearY => 1,
            Self::LinearZ => 2,
            Self::AngularX => 3,
            Self::AngularY => 4,
            Self::AngularZ => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointLimit {
    Free,
    Locked,
    Range { min: f32, max: f32 },
}

/// Forces are newtons or newton meters, for a linear or an angular axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointMotor {
    Velocity {
        speed: f32,
        max_force: f32,
    },
    Position {
        target: f32,
        max_speed: f32,
        max_force: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointAxis {
    pub limit: JointLimit,
    pub motor: Option<JointMotor>,
}

impl JointAxis {
    pub const FREE: Self = Self {
        limit: JointLimit::Free,
        motor: None,
    };
    pub const LOCKED: Self = Self {
        limit: JointLimit::Locked,
        motor: None,
    };

    pub fn range(min: f32, max: f32) -> Self {
        Self {
            limit: JointLimit::Range { min, max },
            motor: None,
        }
    }

    pub fn motor(mut self, motor: JointMotor) -> Self {
        self.motor = Some(motor);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointKind {
    /// Linear x y z, then angular x y z.
    Axes([JointAxis; 6]),
    /// Keeps the anchors `min` to `max` apart.
    Distance { min: f32, max: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointDesc {
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub frame_a: Quat,
    pub frame_b: Quat,
    pub kind: JointKind,
    pub collide_connected: bool,
    /// Newtons on body B past which the joint breaks at the end of a step. Infinite holds.
    pub break_force: f32,
    /// Newton meters, the same for turning.
    pub break_torque: f32,
}

impl JointDesc {
    pub fn new(a: BodyHandle, b: BodyHandle, kind: JointKind) -> Self {
        Self {
            a,
            b,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            frame_a: Quat::IDENTITY,
            frame_b: Quat::IDENTITY,
            kind,
            collide_connected: false,
            break_force: f32::INFINITY,
            break_torque: f32::INFINITY,
        }
    }

    /// Breaks once a step pulls harder than `force` or twists harder than `torque`.
    pub fn breaks_at(mut self, force: f32, torque: f32) -> Self {
        self.break_force = force;
        self.break_torque = torque;
        self
    }

    pub fn fixed(a: BodyHandle, b: BodyHandle) -> Self {
        Self::new(a, b, JointKind::Axes([JointAxis::LOCKED; 6]))
    }

    pub fn ball(a: BodyHandle, b: BodyHandle) -> Self {
        let (l, f) = (JointAxis::LOCKED, JointAxis::FREE);
        Self::new(a, b, JointKind::Axes([l, l, l, f, f, f]))
    }

    /// Turns around the X axis of the joint frame.
    pub fn hinge(a: BodyHandle, b: BodyHandle) -> Self {
        let (l, f) = (JointAxis::LOCKED, JointAxis::FREE);
        Self::new(a, b, JointKind::Axes([l, l, l, f, l, l]))
    }

    /// Slides along the X axis of the joint frame.
    pub fn slider(a: BodyHandle, b: BodyHandle) -> Self {
        let (l, f) = (JointAxis::LOCKED, JointAxis::FREE);
        Self::new(a, b, JointKind::Axes([f, l, l, l, l, l]))
    }

    pub fn distance(a: BodyHandle, b: BodyHandle, min: f32, max: f32) -> Self {
        Self::new(a, b, JointKind::Distance { min, max })
    }

    pub fn anchors(mut self, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        self.anchor_a = anchor_a;
        self.anchor_b = anchor_b;
        self
    }

    pub fn frames(mut self, frame_a: Quat, frame_b: Quat) -> Self {
        self.frame_a = frame_a;
        self.frame_b = frame_b;
        self
    }

    pub fn collide_connected(mut self, collide: bool) -> Self {
        self.collide_connected = collide;
        self
    }

    /// Does nothing on a distance joint, it has no axes.
    pub fn axis(mut self, dof: JointDof, axis: JointAxis) -> Self {
        if let JointKind::Axes(axes) = &mut self.kind {
            axes[dof.index()] = axis;
        }
        self
    }

    pub(crate) fn validate(&self) -> Result<(), PhysicsError> {
        let unit = |quat: Quat| {
            let squared = quat.length_squared();
            squared.is_finite() && squared >= 1e-12
        };
        let valid = self.a != self.b
            && self.anchor_a.is_finite()
            && self.anchor_b.is_finite()
            && unit(self.frame_a)
            && unit(self.frame_b)
            && match self.kind {
                JointKind::Axes(axes) => axes
                    .iter()
                    .enumerate()
                    .all(|(index, axis)| axis_is_valid(axis, index >= 3)),
                JointKind::Distance { min, max } => {
                    min.is_finite() && max.is_finite() && min >= 0.0 && min <= max
                }
            }
            // infinite holds for good, nan or zero or less never held
            && self.break_force > 0.0
            && self.break_torque > 0.0;
        if valid {
            Ok(())
        } else {
            Err(PhysicsError::InvalidJoint)
        }
    }
}

fn axis_is_valid(axis: &JointAxis, angular: bool) -> bool {
    let limit = match axis.limit {
        JointLimit::Range { min, max } => {
            let in_turn = !angular || (min >= -std::f32::consts::PI && max <= std::f32::consts::PI);
            min.is_finite() && max.is_finite() && min <= max && in_turn
        }
        _ => true,
    };
    limit && axis.motor.is_none_or(|motor| motor_is_valid(&motor))
}

fn motor_is_valid(motor: &JointMotor) -> bool {
    match *motor {
        JointMotor::Velocity { speed, max_force } => {
            speed.is_finite() && max_force.is_finite() && max_force >= 0.0
        }
        JointMotor::Position {
            target,
            max_speed,
            max_force,
        } => {
            target.is_finite()
                && max_speed.is_finite()
                && max_speed >= 0.0
                && max_force.is_finite()
                && max_force >= 0.0
        }
    }
}

/// Last step's load on body B, at its anchor. Force is along the frame axes of the world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointReaction {
    pub force: Vec3,
    pub torque: Vec3,
}

#[cfg(test)]
mod tests {
    use crate::dispatch::testing::SkipsJobs;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use super::*;
    use crate::world::testing::{no_gravity, world};
    use crate::{
        Command, CommandBuffer, FrameDesc, FrameId, GridPosition, PhysicsSettings, PhysicsWorld,
        RigidBodyDesc, Shape,
    };

    fn fixed_at(world: &mut PhysicsWorld, at: Vec3) -> BodyHandle {
        world
            .spawn(RigidBodyDesc::fixed(Shape::sphere(0.05)).local_position(at))
            .unwrap()
    }

    fn ball_at(world: &mut PhysicsWorld, at: Vec3) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .can_sleep(false)
                    .local_position(at),
            )
            .unwrap()
    }

    fn block_at(world: &mut PhysicsWorld, at: Vec3) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .can_sleep(false)
                    .density(6.0)
                    .local_position(at),
            )
            .unwrap()
    }

    fn run(world: &mut PhysicsWorld, seconds: f32) {
        let steps = (seconds / world.settings().fixed_timestep) as usize;
        for _ in 0..steps {
            world.step_fixed().unwrap();
        }
    }

    fn position(world: &PhysicsWorld, body: BodyHandle) -> Vec3 {
        world.body_state(body).unwrap().position.local
    }

    #[test]
    fn ball_joint_pendulum_has_the_analytic_period() {
        let mut world = world();
        let pivot = fixed_at(&mut world, Vec3::ZERO);
        let length = 2.0_f32;
        let angle = 0.1_f32;
        let bob = ball_at(
            &mut world,
            Vec3::new(length * angle.sin(), -length * angle.cos(), 0.0),
        );
        world
            .add_joint(JointDesc::ball(pivot, bob).anchors(
                Vec3::ZERO,
                Vec3::new(-length * angle.sin(), length * angle.cos(), 0.0),
            ))
            .unwrap();
        // solid sphere radius 0.25 on an arm of 2
        let period = 2.0 * PI * ((0.4 * 0.25 * 0.25 + length * length) / (9.81 * length)).sqrt();
        let dt = world.settings().fixed_timestep;
        let (mut crossings, mut first, mut last) = (0, 0.0, 0.0);
        let mut previous = position(&world, bob).x;
        for step in 1..=1200 {
            world.step_fixed().unwrap();
            let x = position(&world, bob).x;
            if previous > 0.0 && x <= 0.0 {
                let time = (step as f32 - 1.0 + previous / (previous - x)) * dt;
                if crossings == 0 {
                    first = time;
                }
                last = time;
                crossings += 1;
            }
            previous = x;
        }
        assert!(crossings >= 5, "{crossings}");
        let measured = (last - first) / (crossings - 1) as f32;
        assert!(
            (measured - period).abs() < period * 0.01,
            "{measured} {period}"
        );
    }

    fn swing_peaks(dt: f32, seconds: f32) -> (Vec<f32>, f32) {
        let settings = PhysicsSettings {
            fixed_timestep: dt,
            ..PhysicsSettings::default()
        };
        let mut world = PhysicsWorld::new(settings).unwrap();
        let pivot = fixed_at(&mut world, Vec3::ZERO);
        let bob = ball_at(&mut world, Vec3::new(2.0, 0.0, 0.0));
        let joint = world
            .add_joint(JointDesc::ball(pivot, bob).anchors(Vec3::ZERO, Vec3::new(-2.0, 0.0, 0.0)))
            .unwrap();
        let (mut peaks, mut last, mut rising, mut worst) = (Vec::new(), 0.0_f32, false, 0.0_f32);
        for _ in 0..(seconds / dt) as usize {
            world.step_fixed().unwrap();
            let y = position(&world, bob).y;
            if rising && y < last {
                peaks.push(last);
            }
            rising = y > last;
            last = y;
            worst = worst.max(anchor_gap(&world, joint));
        }
        (peaks, worst)
    }

    #[test]
    fn pendulum_never_gains_energy() {
        let (peaks, worst) = swing_peaks(1.0 / 60.0, 30.0);
        assert!(peaks.len() > 10);
        assert!(peaks[0] < 0.0, "{peaks:?}");
        for pair in peaks.windows(2) {
            assert!(pair[1] <= pair[0], "{peaks:?}");
        }
        assert!(worst < 0.002, "{worst}");
    }

    #[test]
    fn pendulum_energy_loss_shrinks_with_the_timestep() {
        let (coarse, _) = swing_peaks(1.0 / 60.0, 4.0);
        let (fine, _) = swing_peaks(1.0 / 240.0, 4.0);
        // first order in dt, so a quarter of the step loses about a quarter of the height
        let ratio = coarse[0] / fine[0];
        assert!(
            ratio > 3.0 && ratio < 5.0,
            "{} {} {ratio}",
            coarse[0],
            fine[0]
        );
    }

    fn anchor_gap(world: &PhysicsWorld, joint: JointHandle) -> f32 {
        let desc = world.joints().get(joint).unwrap().desc;
        let at = |body: BodyHandle, anchor: Vec3| {
            let state = world.body_state(body).unwrap();
            state.position.local + state.rotation.rotate_vec3(anchor)
        };
        (at(desc.b, desc.anchor_b) - at(desc.a, desc.anchor_a)).length()
    }

    #[test]
    fn hinge_stays_on_its_axis() {
        let mut world = world();
        let pivot = fixed_at(&mut world, Vec3::ZERO);
        let door = block_at(&mut world, Vec3::new(0.0, 0.0, 1.0));
        world
            .add_joint(JointDesc::hinge(pivot, door).anchors(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0)))
            .unwrap();
        world
            .set_velocity(door, Vec3::ZERO, Vec3::new(0.5, 3.0, -2.0))
            .unwrap();
        for _ in 0..600 {
            world.step_fixed().unwrap();
            let state = world.body_state(door).unwrap();
            // the x axis of the block stays the hinge axis, its center stays in the yz plane
            let axis = state.rotation.rotate_vec3(Vec3::X);
            assert!((axis - Vec3::X).length() < 0.05, "{axis:?}");
            assert!(state.position.local.x.abs() < 0.03, "{state:?}");
            assert!(
                (state.position.local.length() - 1.0).abs() < 0.03,
                "{state:?}"
            );
        }
    }

    #[test]
    fn hinge_rests_at_the_range_limit() {
        let mut world = world();
        let pivot = fixed_at(&mut world, Vec3::ZERO);
        // arm along +z swings down around x, toward positive angles
        let door = block_at(&mut world, Vec3::new(0.0, 0.0, 1.0));
        let limit = FRAC_PI_4;
        world
            .add_joint(
                JointDesc::hinge(pivot, door)
                    .anchors(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0))
                    .axis(JointDof::AngularX, JointAxis::range(-limit, limit)),
            )
            .unwrap();
        run(&mut world, 4.0);
        let state = world.body_state(door).unwrap();
        let expected = Vec3::new(0.0, -limit.sin(), limit.cos());
        assert!(
            (state.position.local - expected).length() < 0.02,
            "{:?} {expected:?}",
            state.position.local
        );
        assert!(state.angular_velocity.length() < 0.05);
    }

    #[test]
    fn velocity_motor_reaches_its_speed_and_keeps_it() {
        let mut world = no_gravity();
        let pivot = fixed_at(&mut world, Vec3::ZERO);
        let wheel = block_at(&mut world, Vec3::ZERO);
        world
            .add_joint(JointDesc::hinge(pivot, wheel).axis(
                JointDof::AngularX,
                JointAxis::FREE.motor(JointMotor::Velocity {
                    speed: 4.0,
                    max_force: 1000.0,
                }),
            ))
            .unwrap();
        run(&mut world, 2.0);
        let spin = world.body_state(wheel).unwrap().angular_velocity;
        assert!((spin.x - 4.0).abs() < 1e-3, "{spin:?}");
    }

    #[test]
    fn weak_motor_accelerates_by_torque_over_inertia() {
        let mut world = no_gravity();
        let pivot = fixed_at(&mut world, Vec3::ZERO);
        // density 6, side 1, inertia 1
        let wheel = block_at(&mut world, Vec3::ZERO);
        world
            .add_joint(JointDesc::hinge(pivot, wheel).axis(
                JointDof::AngularX,
                JointAxis::FREE.motor(JointMotor::Velocity {
                    speed: 100.0,
                    max_force: 2.0,
                }),
            ))
            .unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        let spin = world.body_state(wheel).unwrap().angular_velocity.x;
        assert!((spin - 2.0).abs() < 1e-3, "{spin}");
    }

    #[test]
    fn position_motor_reaches_and_holds_against_gravity() {
        let mut world = world();
        let pivot = fixed_at(&mut world, Vec3::ZERO);
        let arm = block_at(&mut world, Vec3::new(0.0, 0.0, 1.0));
        let target = -0.5_f32;
        world
            .add_joint(
                JointDesc::hinge(pivot, arm)
                    .anchors(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0))
                    .axis(
                        JointDof::AngularX,
                        JointAxis::FREE.motor(JointMotor::Position {
                            target,
                            max_speed: 2.0,
                            max_force: 1000.0,
                        }),
                    ),
            )
            .unwrap();
        run(&mut world, 3.0);
        let at = position(&world, arm);
        let angle = (-at.y).atan2(at.z);
        assert!((angle - target).abs() < 0.01, "{angle}");
    }

    #[test]
    fn slider_moves_only_along_its_axis_and_stops_at_the_range() {
        let mut world = world();
        let rail = fixed_at(&mut world, Vec3::ZERO);
        let cart = block_at(&mut world, Vec3::ZERO);
        let tilt = Quat::from_rotation_z(-FRAC_PI_4);
        world
            .add_joint(
                JointDesc::slider(rail, cart)
                    .frames(tilt, tilt)
                    .axis(JointDof::LinearX, JointAxis::range(-1.0, 1.0)),
            )
            .unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
            let state = world.body_state(cart).unwrap();
            let at = state.position.local;
            // the rail runs down the x = -y line
            assert!((at.x + at.y).abs() < 0.01, "{at:?}");
            assert!(state.rotation.to_array()[3] > 0.999, "{state:?}");
        }
        let at = position(&world, cart);
        let along = at.dot(tilt.rotate_vec3(Vec3::X));
        assert!((along - 1.0).abs() < 0.02, "{along}");
    }

    #[test]
    fn slider_motor_lifts_a_load_it_can_lift_and_not_one_it_can_not() {
        let build = |max_force: f32| {
            let mut world = world();
            let rail = fixed_at(&mut world, Vec3::ZERO);
            // density 6, side 1, weight is 6 g
            let cart = block_at(&mut world, Vec3::ZERO);
            let up = Quat::from_rotation_z(FRAC_PI_2);
            world
                .add_joint(JointDesc::slider(rail, cart).frames(up, up).axis(
                    JointDof::LinearX,
                    JointAxis::FREE.motor(JointMotor::Velocity {
                        speed: 1.0,
                        max_force,
                    }),
                ))
                .unwrap();
            for _ in 0..60 {
                world.step_fixed().unwrap();
            }
            world.body_state(cart).unwrap().linear_velocity.y
        };
        assert!((build(500.0) - 1.0).abs() < 1e-3);
        let falling = build(30.0);
        // net (30 - 58.86) / 6 downward for one second
        assert!(
            (falling - (30.0 - 6.0 * 9.81) / 6.0).abs() < 1e-3,
            "{falling}"
        );
    }

    #[test]
    fn fixed_joint_moves_as_one_body() {
        let mut world = no_gravity();
        let a = block_at(&mut world, Vec3::ZERO);
        let b = block_at(&mut world, Vec3::new(2.0, 0.0, 0.0));
        world
            .add_joint(
                JointDesc::fixed(a, b).anchors(Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
            )
            .unwrap();
        world.apply_impulse(a, Vec3::new(0.0, 12.0, 0.0)).unwrap();
        let momentum = |world: &PhysicsWorld| {
            world.body_state(a).unwrap().linear_velocity
                + world.body_state(b).unwrap().linear_velocity
        };
        run(&mut world, 3.0);
        // both have mass 6, so the velocities add up to 12 / 6
        assert!((momentum(&world) - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-3);
        let (one, two) = (world.body_state(a).unwrap(), world.body_state(b).unwrap());
        let gap = two.position.local - one.position.local;
        assert!((gap.length() - 2.0).abs() < 0.02, "{gap:?}");
        let relative = one.rotation.conjugate() * two.rotation;
        assert!(relative.to_array()[3].abs() > 0.999, "{relative:?}");
    }

    #[test]
    fn rope_only_pulls() {
        let mut world = world();
        let hook = fixed_at(&mut world, Vec3::ZERO);
        let weight = ball_at(&mut world, Vec3::new(0.0, -1.0, 0.0));
        world
            .add_joint(JointDesc::distance(hook, weight, 0.0, 3.0))
            .unwrap();
        let dt = world.settings().fixed_timestep;
        world.step_fixed().unwrap();
        // slack, so it falls freely for now
        let speed = world.body_state(weight).unwrap().linear_velocity.y;
        assert!((speed + 9.81 * dt).abs() < 1e-4, "{speed}");
        run(&mut world, 3.0);
        let at = position(&world, weight);
        assert!((at.length() - 3.0).abs() < 0.02, "{at:?}");
    }

    #[test]
    fn rigid_rod_also_pushes() {
        let mut world = world();
        let base = fixed_at(&mut world, Vec3::ZERO);
        let top = ball_at(&mut world, Vec3::new(0.0, 2.0, 0.0));
        world
            .add_joint(JointDesc::distance(base, top, 2.0, 2.0))
            .unwrap();
        run(&mut world, 0.5);
        assert!((position(&world, top).length() - 2.0).abs() < 0.01);
    }

    #[test]
    fn reaction_force_holds_up_the_weight() {
        let mut world = world();
        let hook = fixed_at(&mut world, Vec3::ZERO);
        let weight = block_at(&mut world, Vec3::new(0.0, -1.0, 0.0));
        let joint = world
            .add_joint(JointDesc::fixed(hook, weight).anchors(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .unwrap();
        run(&mut world, 1.0);
        let reaction = world.joint_reaction(joint).unwrap();
        assert!((reaction.force.y - 6.0 * 9.81).abs() < 0.1, "{reaction:?}");
        assert!(reaction.force.x.abs() < 0.1, "{reaction:?}");
    }

    /// Links of mass `4/3 pi 0.25^3 1000`, one meter long, hanging from the origin along `dir`.
    fn chain(world: &mut PhysicsWorld, count: usize, dir: Vec3) -> (BodyHandle, Vec<JointHandle>) {
        let hook = fixed_at(world, Vec3::ZERO);
        let (mut previous, mut joints) = (hook, Vec::new());
        for index in 0..count {
            let link = ball_at(world, dir * (index as f32 + 0.5));
            let anchor_a = if index == 0 { Vec3::ZERO } else { dir * 0.5 };
            joints.push(
                world
                    .add_joint(JointDesc::ball(previous, link).anchors(anchor_a, dir * -0.5))
                    .unwrap(),
            );
            previous = link;
        }
        (previous, joints)
    }

    #[test]
    fn chain_holds_together_when_swinging() {
        let mut world = world();
        let (_, joints) = chain(&mut world, 10, Vec3::X);
        let (mut worst, mut fastest) = (0.0_f32, 0.0_f32);
        for _ in 0..600 {
            world.step_fixed().unwrap();
            for joint in &joints {
                worst = worst.max(anchor_gap(&world, *joint));
                let body = world.joints().get(*joint).unwrap().desc.b;
                let speed = world.body_state(body).unwrap().linear_velocity.length();
                fastest = fastest.max(speed);
            }
        }
        // measured 0.036 while the tip whips past 15 m/s
        assert!(fastest > 15.0, "{fastest}");
        assert!(worst < 0.06, "{worst}");
    }

    /// A 10 kg ball on a ball joint under a fixed point, rated to `limit` newtons.
    fn hung(limit: f32) -> (PhysicsWorld, JointHandle, BodyHandle) {
        let mut world = world();
        let top = fixed_at(&mut world, Vec3::new(0.0, 5.0, 0.0));
        let weight = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.2))
                    .density(10.0 / (4.0 / 3.0 * PI * 0.008))
                    .can_sleep(false)
                    .local_position(Vec3::new(0.0, 4.0, 0.0)),
            )
            .unwrap();
        let joint = world
            .add_joint(
                JointDesc::ball(top, weight)
                    .anchors(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))
                    .breaks_at(limit, f32::INFINITY),
            )
            .unwrap();
        (world, joint, weight)
    }

    #[test]
    fn a_joint_feels_an_outside_force_that_comes_in_at_the_first_substep() {
        // two equal free boxes locked together, one pulled: the lock carries half the pull
        let mut world = PhysicsWorld::new(crate::PhysicsSettings {
            gravity: Vec3::ZERO,
            ..crate::PhysicsSettings::default()
        })
        .unwrap();
        let box_at = |world: &mut PhysicsWorld, x: f32| {
            world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                        .can_sleep(false)
                        .local_position(Vec3::new(x, 0.0, 0.0)),
                )
                .unwrap()
        };
        let (a, b) = (box_at(&mut world, 0.0), box_at(&mut world, 1.0));
        let joint = world
            .add_joint(
                JointDesc::fixed(a, b).anchors(Vec3::new(0.5, 0.0, 0.0), Vec3::new(-0.5, 0.0, 0.0)),
            )
            .unwrap();
        for _ in 0..3 {
            world.apply_force(b, Vec3::new(1000.0, 0.0, 0.0)).unwrap();
            world.step_fixed().unwrap();
        }
        let force = world.joint_reaction(joint).unwrap().force;
        assert!((force.x.abs() - 500.0).abs() < 5.0, "{force:?}");
    }

    #[test]
    fn a_joint_holds_under_its_limit_and_breaks_past_it() {
        // 10 kg pulls about 98 n
        let (mut world, joint, weight) = hung(150.0);
        for _ in 0..120 {
            world.step_fixed().unwrap();
            assert!(world.broken_joints().is_empty());
        }
        assert!(world.joint_reaction(joint).is_ok());
        let held = world.body_state(weight).unwrap().position.to_dvec3().y;
        assert!((held - 4.0).abs() < 0.05, "{held}");
        let (mut world, joint, weight) = hung(60.0);
        world.step_fixed().unwrap();
        assert_eq!(world.broken_joints(), [joint]);
        assert!(world.joint_reaction(joint).is_err());
        for _ in 0..60 {
            world.step_fixed().unwrap();
            assert!(world.broken_joints().is_empty());
        }
        let fell = world.body_state(weight).unwrap().position.to_dvec3().y;
        assert!(fell < 0.0, "{fell}");
        // a limit of zero or nan never held, refused
        let (mut world, _, _) = hung(150.0);
        let (a, b) = (
            world.body_handles().next().unwrap(),
            world.body_handles().nth(1).unwrap(),
        );
        for bad in [0.0, -1.0, f32::NAN] {
            assert!(
                world
                    .add_joint(JointDesc::ball(a, b).breaks_at(bad, 1.0))
                    .is_err()
            );
        }
    }

    #[test]
    fn a_joints_limits_survive_a_snapshot() {
        let (world, joint, _) = hung(150.0);
        let loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        let desc = loaded.joints.get(joint).unwrap().desc;
        assert_eq!(
            (desc.break_force, desc.break_torque),
            (150.0, f32::INFINITY)
        );
    }

    fn hang_load(ratio: f32) -> (f32, f32) {
        let mut world = world();
        let down = -Vec3::Y;
        let (last, mut joints) = chain(&mut world, 8, down);
        let load = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .density(1000.0 * ratio)
                    .can_sleep(false)
                    .local_position(Vec3::new(0.0, -8.5, 0.0)),
            )
            .unwrap();
        joints.push(
            world
                .add_joint(JointDesc::ball(last, load).anchors(down * 0.5, down * -0.5))
                .unwrap(),
        );
        let mut worst = 0.0_f32;
        for _ in 0..600 {
            world.step_fixed().unwrap();
            for joint in &joints {
                worst = worst.max(anchor_gap(&world, *joint));
            }
        }
        (worst, -8.5 - position(&world, load).y)
    }

    // the direct solve holds a straight hang exactly, sweeps alone let it sag
    #[test]
    fn light_chain_holds_a_load_of_a_hundred_links() {
        let (worst, sag) = hang_load(100.0);
        assert!(worst < 1e-3, "{worst}");
        assert!(sag < 1e-3, "{sag}");
    }

    #[test]
    fn light_chain_holds_a_thousand_times_load() {
        let (worst, sag) = hang_load(1000.0);
        assert!(worst < 1e-3, "{worst}");
        assert!(sag < 1e-3, "{sag}");
    }

    /// A net of balls held at two corners falls and swings. Its weight goes through the links
    /// beside the corners, hundreds to one, where sweeps alone stretched them 70 cm.
    #[test]
    fn swinging_net_held_at_two_corners_keeps_its_links() {
        let worst = swing_net(8, 180);
        assert!(worst < 0.02, "{worst}");
    }

    /// Past the size where islands are colored, the joints still get their direct solve.
    #[test]
    fn a_net_too_big_for_one_job_keeps_its_links_too() {
        // 0.19 measured, sweeps alone 0.54
        let worst = swing_net(24, 60);
        assert!(worst < 0.25, "{worst}");
    }

    /// The widest a joint of a `side` by `side` net opens over `steps`.
    fn swing_net(side: usize, steps: usize) -> f32 {
        let mut world = world();
        let at = |x: usize, z: usize| Vec3::new(x as f32 * 0.5, 10.0, z as f32 * 0.5);
        let mut grid = Vec::new();
        for z in 0..side {
            for x in 0..side {
                grid.push(
                    world
                        .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.1)).local_position(at(x, z)))
                        .unwrap(),
                );
            }
        }
        let mut joints = Vec::new();
        let mut ball = |world: &mut PhysicsWorld, a, b, point: Vec3| {
            let anchor = GridPosition::from_local(point).unwrap();
            let desc = world
                .joint_at(JointDesc::ball(a, b), anchor, Quat::IDENTITY)
                .unwrap();
            joints.push(world.add_joint(desc).unwrap());
        };
        for z in 0..side {
            for x in 0..side {
                let here = grid[z * side + x];
                if x + 1 < side {
                    let mid = (at(x, z) + at(x + 1, z)) * 0.5;
                    ball(&mut world, here, grid[z * side + x + 1], mid);
                }
                if z + 1 < side {
                    let mid = (at(x, z) + at(x, z + 1)) * 0.5;
                    ball(&mut world, here, grid[(z + 1) * side + x], mid);
                }
            }
        }
        for x in [0, side - 1] {
            let post = world
                .spawn(RigidBodyDesc::fixed(Shape::sphere(0.05)).local_position(at(x, 0)))
                .unwrap();
            ball(&mut world, post, grid[x], at(x, 0));
        }
        let mut worst = 0.0_f32;
        for _ in 0..steps {
            world.step_fixed().unwrap();
            for joint in &joints {
                worst = worst.max(anchor_gap(&world, *joint));
            }
        }
        worst
    }

    #[test]
    fn jointed_pair_does_not_collide_unless_asked() {
        let run_pair = |collide: bool| {
            let mut world = no_gravity();
            let a = ball_at(&mut world, Vec3::ZERO);
            let b = ball_at(&mut world, Vec3::new(0.3, 0.0, 0.0));
            world
                .add_joint(JointDesc::distance(a, b, 0.0, 100.0).collide_connected(collide))
                .unwrap();
            run(&mut world, 1.0);
            (position(&world, b) - position(&world, a)).length()
        };
        assert!((run_pair(false) - 0.3).abs() < 1e-3);
        assert!(run_pair(true) >= 0.49);
    }

    #[test]
    fn despawn_removes_its_joints_and_handles_go_stale() {
        let mut world = world();
        let hook = fixed_at(&mut world, Vec3::ZERO);
        let weight = ball_at(&mut world, Vec3::new(0.0, -1.0, 0.0));
        let joint = world
            .add_joint(JointDesc::ball(hook, weight).anchors(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .unwrap();
        assert_eq!(world.joints().len(), 1);
        run(&mut world, 0.5);
        world.despawn(hook).unwrap();
        assert!(world.joints().is_empty());
        assert_eq!(
            world.joint_reaction(joint),
            Err(PhysicsError::InvalidHandle)
        );
        assert_eq!(world.remove_joint(joint), Err(PhysicsError::InvalidHandle));
        let before = position(&world, weight).y;
        run(&mut world, 0.5);
        assert!(position(&world, weight).y < before - 1.0);
    }

    #[test]
    fn removing_a_joint_frees_the_body() {
        let mut world = world();
        let hook = fixed_at(&mut world, Vec3::ZERO);
        let weight = ball_at(&mut world, Vec3::new(0.0, -1.0, 0.0));
        let joint = world
            .add_joint(JointDesc::ball(hook, weight).anchors(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .unwrap();
        run(&mut world, 0.5);
        world.remove_joint(joint).unwrap();
        let next = world.add_joint(JointDesc::ball(hook, weight)).unwrap();
        assert_eq!(next.index, joint.index);
        assert_ne!(next.generation, joint.generation);
    }

    #[test]
    fn bad_joints_fail() {
        let mut world = world();
        let fixed = fixed_at(&mut world, Vec3::ZERO);
        let other_fixed = fixed_at(&mut world, Vec3::X);
        let body = ball_at(&mut world, Vec3::Y);
        let frame = world.add_frame(FrameDesc::new(FrameId::ROOT)).unwrap();
        let elsewhere = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(1.0)).frame(frame))
            .unwrap();
        let gone = ball_at(&mut world, Vec3::Z);
        world.despawn(gone).unwrap();
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        for bad in [
            JointDesc::ball(body, body),
            JointDesc::ball(fixed, other_fixed),
            JointDesc::ball(fixed, elsewhere),
            JointDesc::ball(fixed, body).anchors(nan, Vec3::ZERO),
            JointDesc::ball(fixed, body).frames(Quat::new(0.0, 0.0, 0.0, 0.0), Quat::IDENTITY),
            JointDesc::distance(fixed, body, 2.0, 1.0),
            JointDesc::distance(fixed, body, -1.0, 1.0),
            JointDesc::hinge(fixed, body).axis(JointDof::AngularX, JointAxis::range(1.0, -1.0)),
            JointDesc::hinge(fixed, body).axis(JointDof::AngularX, JointAxis::range(-4.0, 1.0)),
            JointDesc::hinge(fixed, body).axis(
                JointDof::AngularX,
                JointAxis::FREE.motor(JointMotor::Velocity {
                    speed: 1.0,
                    max_force: -1.0,
                }),
            ),
        ] {
            assert_eq!(
                world.add_joint(bad),
                Err(PhysicsError::InvalidJoint),
                "{bad:?}"
            );
        }
        assert_eq!(
            world.add_joint(JointDesc::ball(gone, body)),
            Err(PhysicsError::InvalidHandle)
        );
        assert!(world.joints().is_empty());
    }

    #[test]
    fn jointed_body_can_not_change_frame() {
        let mut world = world();
        let hook = fixed_at(&mut world, Vec3::ZERO);
        let weight = ball_at(&mut world, Vec3::new(0.0, -1.0, 0.0));
        world.add_joint(JointDesc::ball(hook, weight)).unwrap();
        let frame = world.add_frame(FrameDesc::new(FrameId::ROOT)).unwrap();
        assert_eq!(
            world.transfer_to_frame(weight, frame),
            Err(PhysicsError::BodyHasJoints)
        );
        assert_eq!(world.body_state(weight).unwrap().frame, FrameId::ROOT);
    }

    #[test]
    fn motor_change_applies_and_is_checked() {
        let mut world = no_gravity();
        let pivot = fixed_at(&mut world, Vec3::ZERO);
        let wheel = block_at(&mut world, Vec3::ZERO);
        let hinge = world.add_joint(JointDesc::hinge(pivot, wheel)).unwrap();
        let motor = JointMotor::Velocity {
            speed: 3.0,
            max_force: 100.0,
        };
        world
            .set_joint_motor(hinge, JointDof::AngularX, Some(motor))
            .unwrap();
        run(&mut world, 1.0);
        assert!((world.body_state(wheel).unwrap().angular_velocity.x - 3.0).abs() < 1e-3);
        let bad = JointMotor::Velocity {
            speed: f32::NAN,
            max_force: 1.0,
        };
        assert_eq!(
            world.set_joint_motor(hinge, JointDof::AngularX, Some(bad)),
            Err(PhysicsError::InvalidJoint)
        );
        let rope = world
            .add_joint(JointDesc::distance(pivot, wheel, 0.0, 5.0))
            .unwrap();
        assert_eq!(
            world.set_joint_motor(rope, JointDof::LinearX, Some(motor)),
            Err(PhysicsError::InvalidJoint)
        );
        world
            .set_joint_motor(hinge, JointDof::AngularX, None)
            .unwrap();
        world.step_fixed().unwrap();
    }

    #[test]
    fn commands_create_and_destroy_joints() {
        let mut world = world();
        let hook = fixed_at(&mut world, Vec3::ZERO);
        let weight = ball_at(&mut world, Vec3::new(0.0, -1.0, 0.0));
        let mut buffer = CommandBuffer::new();
        buffer.push(Command::CreateJoint(JointDesc::ball(hook, weight)));
        buffer.push(Command::CreateJoint(JointDesc::ball(hook, hook)));
        let report = world.apply_commands(&mut buffer);
        assert_eq!(report.created_joints.len(), 1);
        assert_eq!(report.rejected, vec![(1, PhysicsError::InvalidJoint)]);
        buffer.push(Command::SetJointMotor(
            report.created_joints[0],
            JointDof::AngularX,
            None,
        ));
        buffer.push(Command::DestroyJoint(report.created_joints[0]));
        assert!(world.apply_commands(&mut buffer).rejected.is_empty());
        assert!(world.joints().is_empty());
    }

    #[test]
    fn world_anchor_helper_finds_the_same_point_on_both_bodies() {
        let mut world = world();
        let turn = Quat::from_rotation_z(0.7);
        let a = world
            .spawn(RigidBodyDesc::fixed(Shape::sphere(0.1)).rotation(turn))
            .unwrap();
        let b = ball_at(&mut world, Vec3::new(3.0, 1.0, 0.0));
        let at = GridPosition::from_local(Vec3::new(1.0, 1.0, 0.0)).unwrap();
        let desc = world
            .joint_at(JointDesc::fixed(a, b), at, Quat::from_rotation_x(0.3))
            .unwrap();
        let joint = world.add_joint(desc).unwrap();
        // already lined up, so nothing to fix but the weight
        world.step_fixed().unwrap();
        let moved = (position(&world, b) - Vec3::new(3.0, 1.0, 0.0)).length();
        assert!(moved < 0.01, "{moved}");
        assert!(world.joint_reaction(joint).is_ok());
    }

    #[test]
    fn pendulum_sleeps_and_a_touch_wakes_the_whole_link() {
        let mut world = world();
        let hook = fixed_at(&mut world, Vec3::new(0.0, 5.0, 0.0));
        let upper = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .local_position(Vec3::new(0.0, 4.0, 0.0)),
            )
            .unwrap();
        let lower = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .local_position(Vec3::new(0.0, 3.0, 0.0)),
            )
            .unwrap();
        world
            .add_joint(JointDesc::ball(hook, upper).anchors(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .unwrap();
        world
            .add_joint(
                JointDesc::ball(upper, lower)
                    .anchors(Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 0.5, 0.0)),
            )
            .unwrap();
        run(&mut world, 2.0);
        assert!(world.body_state(upper).unwrap().asleep);
        assert!(world.body_state(lower).unwrap().asleep);
        world.wake(lower).unwrap();
        world.step_fixed().unwrap();
        assert!(!world.body_state(upper).unwrap().asleep);
    }

    #[test]
    fn linked_bodies_with_nothing_holding_them_never_sleep() {
        let mut world = no_gravity();
        let a = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.25)))
            .unwrap();
        let b = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .local_position(Vec3::new(1.0, 0.0, 0.0)),
            )
            .unwrap();
        world
            .add_joint(JointDesc::distance(a, b, 1.0, 1.0))
            .unwrap();
        run(&mut world, 3.0);
        assert!(!world.body_state(a).unwrap().asleep);
    }

    #[test]
    fn running_motor_keeps_it_awake_and_setting_a_motor_wakes_it() {
        let mut world = no_gravity();
        let pivot = fixed_at(&mut world, Vec3::ZERO);
        let wheel = world
            .spawn(RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5))))
            .unwrap();
        let hinge = world.add_joint(JointDesc::hinge(pivot, wheel)).unwrap();
        run(&mut world, 2.0);
        assert!(world.body_state(wheel).unwrap().asleep);
        // too slow to count as moving, the motor must hold it awake anyway
        let motor = JointMotor::Velocity {
            speed: 0.01,
            max_force: 100_000.0,
        };
        world
            .set_joint_motor(hinge, JointDof::AngularX, Some(motor))
            .unwrap();
        assert!(!world.body_state(wheel).unwrap().asleep);
        run(&mut world, 2.0);
        assert!(!world.body_state(wheel).unwrap().asleep);
    }

    #[test]
    fn snapshot_with_joints_continues_exactly() {
        let mut world = world();
        let hook = fixed_at(&mut world, Vec3::ZERO);
        let arm = block_at(&mut world, Vec3::new(1.0, 0.0, 0.0));
        let tip = ball_at(&mut world, Vec3::new(2.0, 0.0, 0.0));
        world
            .add_joint(
                JointDesc::hinge(hook, arm)
                    .frames(
                        Quat::from_rotation_y(FRAC_PI_2),
                        Quat::from_rotation_y(FRAC_PI_2),
                    )
                    .anchors(Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0))
                    .axis(
                        JointDof::AngularX,
                        JointAxis::range(-1.0, 1.0).motor(JointMotor::Position {
                            target: 0.25,
                            max_speed: 1.0,
                            max_force: 20.0,
                        }),
                    ),
            )
            .unwrap();
        world
            .add_joint(
                JointDesc::distance(arm, tip, 0.0, 1.0)
                    .anchors(Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO),
            )
            .unwrap();
        run(&mut world, 0.5);
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        run(&mut world, 1.0);
        run(&mut loaded, 1.0);
        assert_eq!(loaded.state_hash(), world.state_hash());
    }

    #[test]
    fn failed_step_keeps_the_joint_impulses() {
        let mut world = world();
        let hook = fixed_at(&mut world, Vec3::ZERO);
        let weight = ball_at(&mut world, Vec3::new(0.0, -1.0, 0.0));
        let joint = world
            .add_joint(JointDesc::ball(hook, weight).anchors(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .unwrap();
        run(&mut world, 0.5);
        // enough bodies that a step hands work to the dispatcher at all
        for i in 0..1100 {
            let at = Vec3::new(1000.0 + (i % 40) as f32 * 3.0, 0.0, (i / 40) as f32 * 3.0);
            world
                .spawn(RigidBodyDesc::fixed(Shape::sphere(0.1)).local_position(at))
                .unwrap();
        }
        let before = world.joint_reaction(joint).unwrap();
        let bytes = world.save_snapshot();
        assert!(
            world
                .step_with(&SkipsJobs {
                    parallelism: 2,
                    skip: |_| true
                })
                .is_err()
        );
        assert_eq!(world.joint_reaction(joint).unwrap(), before);
        assert_eq!(world.save_snapshot(), bytes);
    }
}
