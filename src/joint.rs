use algebrix::{Quat, Vec3};

use crate::body::{Bodies, apply_diagonal};
use crate::direct::{DirectNode, DirectRow};
use crate::solver::Pair;
use crate::world::Integrated;
use crate::{BodyHandle, BodyKind, PhysicsError};

pub(crate) const JOINT_IMPULSES: usize = 12;
pub(crate) const MOTOR_SLOT: usize = 6;
const RECOVERY_RATE: f32 = 0.2;
const MAX_RECOVERY_SPEED: f32 = 3.0;
const FREE: u32 = u32::MAX;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JointState {
    pub(crate) desc: JointDesc,
    pub(crate) impulses: [f32; JOINT_IMPULSES],
    pub(crate) force: Vec3,
    pub(crate) torque: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JointSlot {
    pub(crate) generation: u32,
    pub(crate) state: Option<JointState>,
}

#[derive(Debug, Clone, Default)]
pub struct Joints {
    pub(crate) slots: Vec<JointSlot>,
    pub(crate) free: Vec<u32>,
    excluded: Vec<(BodyHandle, BodyHandle)>,
    excluded_stale: bool,
}

impl Joints {
    pub fn len(&self) -> usize {
        self.live().count()
    }

    pub fn is_empty(&self) -> bool {
        self.live().next().is_none()
    }

    pub(crate) fn live(&self) -> impl Iterator<Item = (usize, &JointState)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.state.as_ref().map(|state| (index, state)))
    }

    pub(crate) fn get(&self, handle: JointHandle) -> Result<&JointState, PhysicsError> {
        match self.slots.get(handle.index as usize) {
            Some(JointSlot {
                generation,
                state: Some(state),
            }) if *generation == handle.generation => Ok(state),
            _ => Err(PhysicsError::InvalidHandle),
        }
    }

    pub(crate) fn get_mut(&mut self, handle: JointHandle) -> Result<&mut JointState, PhysicsError> {
        match self.slots.get_mut(handle.index as usize) {
            Some(JointSlot {
                generation,
                state: Some(state),
            }) if *generation == handle.generation => Ok(state),
            _ => Err(PhysicsError::InvalidHandle),
        }
    }

    pub(crate) fn insert(&mut self, desc: JointDesc) -> Result<JointHandle, PhysicsError> {
        let state = JointState {
            desc: JointDesc {
                frame_a: desc.frame_a.normalize(),
                frame_b: desc.frame_b.normalize(),
                ..desc
            },
            impulses: [0.0; JOINT_IMPULSES],
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
        };
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                let index = u32::try_from(self.slots.len())
                    .ok()
                    .filter(|index| *index != FREE)
                    .ok_or(PhysicsError::OutOfRange)?;
                self.slots.push(JointSlot {
                    generation: 0,
                    state: None,
                });
                index
            }
        };
        let slot = &mut self.slots[index as usize];
        slot.state = Some(state);
        self.excluded_stale = true;
        Ok(JointHandle {
            index,
            generation: slot.generation,
        })
    }

    pub(crate) fn remove(&mut self, handle: JointHandle) -> Result<JointState, PhysicsError> {
        let state = *self.get(handle)?;
        let slot = &mut self.slots[handle.index as usize];
        slot.state = None;
        // a slot at the last generation is retired, so old handles can never match again
        if slot.generation < u32::MAX {
            slot.generation += 1;
            self.free.push(handle.index);
        }
        self.excluded_stale = true;
        Ok(state)
    }

    pub(crate) fn mark_loaded(&mut self) {
        self.excluded_stale = true;
    }

    pub(crate) fn handles(&self) -> impl Iterator<Item = JointHandle> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.state.as_ref()?;
            Some(JointHandle {
                index: index as u32,
                generation: slot.generation,
            })
        })
    }

    pub(crate) fn touching(&self, body: BodyHandle) -> impl Iterator<Item = JointHandle> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(move |(index, slot)| {
                let state = slot.state.as_ref()?;
                (state.desc.a == body || state.desc.b == body).then_some(JointHandle {
                    index: index as u32,
                    generation: slot.generation,
                })
            })
    }

    pub(crate) fn refresh_exclusions(&mut self) {
        if !self.excluded_stale {
            return;
        }
        let mut excluded = std::mem::take(&mut self.excluded);
        excluded.clear();
        excluded.extend(self.live().filter_map(|(_, state)| {
            let desc = &state.desc;
            (!desc.collide_connected).then_some((desc.a.min(desc.b), desc.a.max(desc.b)))
        }));
        excluded.sort_unstable();
        excluded.dedup();
        self.excluded = excluded;
        self.excluded_stale = false;
    }

    pub(crate) fn excludes(&self, a: BodyHandle, b: BodyHandle) -> bool {
        !self.excluded.is_empty() && self.excluded.binary_search(&(a.min(b), a.max(b))).is_ok()
    }

    /// Dense pairs linked by a joint, and whether a motor keeps it running.
    pub(crate) fn links<'a>(
        &'a self,
        bodies: &'a Bodies,
    ) -> impl Iterator<Item = (usize, usize, bool)> + 'a {
        self.live().filter_map(move |(_, state)| {
            let a = bodies.dense_index(state.desc.a).ok()?;
            let b = bodies.dense_index(state.desc.b).ok()?;
            let running = match state.desc.kind {
                JointKind::Axes(axes) => axes.iter().any(|axis| {
                    matches!(axis.motor, Some(JointMotor::Velocity { speed, .. }) if speed != 0.0)
                }),
                JointKind::Distance { .. } => false,
            };
            Some((a, b, running))
        })
    }

    /// Runs after the step commits, so a failed step keeps the old impulses.
    pub(crate) fn store(
        &mut self,
        impulses: &[[f32; JOINT_IMPULSES]],
        points: &[JointPoint],
        rows: &[JointRow],
        dt: f32,
    ) {
        for (slot, working) in self.slots.iter_mut().zip(impulses) {
            if let Some(state) = slot.state.as_mut() {
                state.impulses = *working;
            }
        }
        let touched = points
            .iter()
            .map(|point| point.joint)
            .chain(rows.iter().map(|row| row.joint));
        for joint in touched {
            if let Some(state) = self.slots[joint as usize].state.as_mut() {
                state.force = Vec3::ZERO;
                state.torque = Vec3::ZERO;
            }
        }
        for point in points {
            let Some(state) = self.slots[point.joint as usize].state.as_mut() else {
                continue;
            };
            state.force += point.impulse * dt.recip();
        }
        for row in rows {
            let Some(state) = self.slots[row.joint as usize].state.as_mut() else {
                continue;
            };
            if row.linear == Vec3::ZERO {
                state.torque += row.ang_b * (row.impulse / dt);
            } else {
                state.force += row.linear * (row.impulse / dt);
            }
        }
    }
}

/// Keeps this substep's impulses for the next one to start from.
impl JointPoint {
    pub(crate) fn bodies(&self) -> (usize, usize) {
        (self.a, self.b)
    }
}

impl JointRow {
    pub(crate) fn bodies(&self) -> (usize, usize) {
        (self.a, self.b)
    }

    /// A lock with no bound either way, for the direct solve.
    pub(crate) fn is_equality(&self) -> bool {
        self.low == f32::NEG_INFINITY
            && self.high == f32::INFINITY
            && self.recovery_low == f32::NEG_INFINITY
            && self.recovery_high == f32::INFINITY
    }

    pub(crate) fn direct_node(&self) -> DirectNode {
        let row = DirectRow {
            lin_a: -self.linear,
            ang_a: -self.ang_a,
            lin_b: self.linear,
            ang_b: self.ang_b,
            move_lin_a: -self.linear * self.inverse_a,
            move_ang_a: -self.turn_a,
            move_lin_b: self.linear * self.inverse_b,
            move_ang_b: self.turn_b,
            target: self.target,
            recovery: self.recovery,
        };
        DirectNode {
            a: self.a,
            b: self.b,
            count: 1,
            rows: [row, DirectRow::default(), DirectRow::default()],
        }
    }

    pub(crate) fn add_direct(&mut self, impulse: f32, recovery: bool) {
        if recovery {
            self.recovery_impulse += impulse;
        } else {
            self.impulse += impulse;
        }
    }
}

impl JointPoint {
    /// Its three world axis rows, for the direct solve.
    pub(crate) fn direct_node(&self) -> DirectNode {
        let mut rows = [DirectRow::default(); 3];
        for ((row, axis), recovery) in rows
            .iter_mut()
            .zip([Vec3::X, Vec3::Y, Vec3::Z])
            .zip(self.recovery.to_array())
        {
            let (turn_a, turn_b) = (self.arm_a.cross(axis), self.arm_b.cross(axis));
            *row = DirectRow {
                lin_a: -axis,
                ang_a: -turn_a,
                lin_b: axis,
                ang_b: turn_b,
                move_lin_a: -axis * self.inverse_a,
                move_ang_a: -matrix_times(&self.inertia_a, turn_a),
                move_lin_b: axis * self.inverse_b,
                move_ang_b: matrix_times(&self.inertia_b, turn_b),
                target: 0.0,
                recovery,
            };
        }
        DirectNode {
            a: self.a,
            b: self.b,
            count: 3,
            rows,
        }
    }

    pub(crate) fn add_direct(&mut self, impulse: Vec3, recovery: bool) {
        if recovery {
            self.recovery_impulse += impulse;
        } else {
            self.impulse += impulse;
        }
    }
}

pub(crate) fn capture_impulses(
    points: &[JointPoint],
    rows: &[JointRow],
    impulses: &mut [[f32; JOINT_IMPULSES]],
) {
    for point in points {
        for (slot, axis) in impulses[point.joint as usize].iter_mut().zip(&point.axes) {
            *slot = point.impulse.dot(*axis);
        }
    }
    for row in rows {
        impulses[row.joint as usize][row.slot] = row.impulse;
    }
}

/// All three linear axes locked, solved at once. Scalar rows fight each other on a long arm.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JointPoint {
    joint: u32,
    a: usize,
    b: usize,
    arm_a: Vec3,
    arm_b: Vec3,
    axes: [Vec3; 3],
    inverse_mass: [Vec3; 3],
    inverse_a: f32,
    inverse_b: f32,
    /// World inverse inertia of each body as columns, at this substep's turn.
    inertia_a: [Vec3; 3],
    inertia_b: [Vec3; 3],
    impulse: Vec3,
    recovery: Vec3,
    recovery_impulse: Vec3,
}

fn matrix_times(columns: &[Vec3; 3], value: Vec3) -> Vec3 {
    columns[0] * value.x + columns[1] * value.y + columns[2] * value.z
}

/// Columns of the inverse, or `None` when the matrix has no real inverse.
fn inverted(columns: &[Vec3; 3]) -> Option<[Vec3; 3]> {
    let [x, y, z] = *columns;
    let (yz, zx, xy) = (y.cross(z), z.cross(x), x.cross(y));
    let determinant = x.dot(yz);
    if !determinant.is_finite() || determinant.abs() < 1e-20 {
        return None;
    }
    let scale = determinant.recip();
    // rows of the inverse are the crosses, so read them out by column
    Some([
        Vec3::new(yz.x, zx.x, xy.x) * scale,
        Vec3::new(yz.y, zx.y, xy.y) * scale,
        Vec3::new(yz.z, zx.z, xy.z) * scale,
    ])
}

#[allow(clippy::too_many_arguments)]
fn build_point(
    bodies: &Bodies,
    poses: &[Integrated],
    impulses: &[f32; JOINT_IMPULSES],
    joint: u32,
    pair: (usize, usize),
    arms: (Vec3, Vec3),
    axes: [Vec3; 3],
    recovery: Vec3,
) -> JointPoint {
    let (a, b) = pair;
    let (arm_a, arm_b) = arms;
    let (inverse_a, inverse_b) = (bodies.inverse_mass[a], bodies.inverse_mass[b]);
    let column = |unit: Vec3| {
        let part = |dense: usize, arm: Vec3| {
            apply_diagonal(
                poses[dense].rotation,
                bodies.inverse_inertia[dense],
                arm.cross(unit),
            )
            .cross(arm)
        };
        unit * (inverse_a + inverse_b) + part(a, arm_a) + part(b, arm_b)
    };
    let matrix = [column(Vec3::X), column(Vec3::Y), column(Vec3::Z)];
    let inertia = |dense: usize| {
        let turn =
            |unit: Vec3| apply_diagonal(poses[dense].rotation, bodies.inverse_inertia[dense], unit);
        [turn(Vec3::X), turn(Vec3::Y), turn(Vec3::Z)]
    };
    JointPoint {
        joint,
        a,
        b,
        arm_a,
        arm_b,
        axes,
        inverse_mass: inverted(&matrix).unwrap_or([Vec3::ZERO; 3]),
        inverse_a,
        inverse_b,
        inertia_a: inertia(a),
        inertia_b: inertia(b),
        impulse: axes[0] * impulses[0] + axes[1] * impulses[1] + axes[2] * impulses[2],
        recovery,
        recovery_impulse: Vec3::ZERO,
    }
}

fn point_recovery(gap: Vec3, dt: f32) -> Vec3 {
    let wanted = gap * (-RECOVERY_RATE / dt);
    let speed = wanted.length();
    if speed > MAX_RECOVERY_SPEED {
        wanted * (MAX_RECOVERY_SPEED / speed)
    } else {
        wanted
    }
}

pub(crate) fn warm_start_point(pair: &mut Pair, point: &JointPoint) {
    let impulse = point.impulse;
    let body = &mut *pair.a;
    body.linear -= impulse * point.inverse_a;
    body.angular -= matrix_times(&point.inertia_a, point.arm_a.cross(impulse));
    body.touched = true;
    let body = &mut *pair.b;
    body.linear += impulse * point.inverse_b;
    body.angular += matrix_times(&point.inertia_b, point.arm_b.cross(impulse));
    body.touched = true;
}

pub(crate) fn solve_point(pair: &mut Pair, point: &mut JointPoint) {
    let (a, b) = (&*pair.a, &*pair.b);
    let speed = b.linear + b.angular.cross(point.arm_b) - a.linear - a.angular.cross(point.arm_a);
    let change = matrix_times(&point.inverse_mass, -speed);
    point.impulse += change;
    let single = JointPoint {
        impulse: change,
        ..*point
    };
    warm_start_point(pair, &single);
}

pub(crate) fn solve_point_recovery(pair: &mut Pair, point: &mut JointPoint) {
    let (a, b) = (&*pair.a, &*pair.b);
    let speed = b.recovery_linear + b.recovery_angular.cross(point.arm_b)
        - a.recovery_linear
        - a.recovery_angular.cross(point.arm_a);
    let change = matrix_times(&point.inverse_mass, point.recovery - speed);
    point.recovery_impulse += change;
    let body = &mut *pair.a;
    body.recovery_linear -= change * point.inverse_a;
    body.recovery_angular -= matrix_times(&point.inertia_a, point.arm_a.cross(change));
    let body = &mut *pair.b;
    body.recovery_linear += change * point.inverse_b;
    body.recovery_angular += matrix_times(&point.inertia_b, point.arm_b.cross(change));
}

/// One scalar constraint. Speed along it is `linear . (vb - va) + ang_b . wb - ang_a . wa`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JointRow {
    joint: u32,
    slot: usize,
    a: usize,
    b: usize,
    linear: Vec3,
    ang_a: Vec3,
    ang_b: Vec3,
    turn_a: Vec3,
    turn_b: Vec3,
    inverse_a: f32,
    inverse_b: f32,
    mass: f32,
    target: f32,
    low: f32,
    high: f32,
    impulse: f32,
    recovery: f32,
    recovery_low: f32,
    recovery_high: f32,
    recovery_impulse: f32,
}

/// What a row asks for, before the body data goes in.
#[derive(Clone, Copy)]
struct RowGoal {
    slot: usize,
    target: f32,
    low: f32,
    high: f32,
    recovery: f32,
    recovery_low: f32,
    recovery_high: f32,
}

struct RowBuilder<'a> {
    bodies: &'a Bodies,
    poses: &'a [Integrated],
    impulses: &'a [f32; JOINT_IMPULSES],
    joint: u32,
    a: usize,
    b: usize,
    rows: &'a mut Vec<JointRow>,
}

impl RowBuilder<'_> {
    fn push(&mut self, linear: Vec3, ang_a: Vec3, ang_b: Vec3, goal: RowGoal) {
        let (a, b) = (self.a, self.b);
        let bodies = self.bodies;
        let turn_a = apply_diagonal(self.poses[a].rotation, bodies.inverse_inertia[a], ang_a);
        let turn_b = apply_diagonal(self.poses[b].rotation, bodies.inverse_inertia[b], ang_b);
        let (inverse_a, inverse_b) = (bodies.inverse_mass[a], bodies.inverse_mass[b]);
        let inverse = linear.length_squared() * (inverse_a + inverse_b)
            + ang_a.dot(turn_a)
            + ang_b.dot(turn_b);
        let mass = if inverse > 1e-12 {
            inverse.recip()
        } else {
            0.0
        };
        self.rows.push(JointRow {
            joint: self.joint,
            slot: goal.slot,
            a,
            b,
            linear,
            ang_a,
            ang_b,
            turn_a,
            turn_b,
            inverse_a,
            inverse_b,
            mass,
            target: goal.target,
            low: goal.low,
            high: goal.high,
            // not `clamp`, a limit may have changed side and old values come from outside
            impulse: self.impulses[goal.slot].min(goal.high).max(goal.low),
            recovery: goal.recovery,
            recovery_low: goal.recovery_low,
            recovery_high: goal.recovery_high,
            recovery_impulse: 0.0,
        });
    }

    /// Motor first, so a limit has the last word over it.
    fn axis(&mut self, index: usize, axis: &JointAxis, value: f32, parts: [Vec3; 3], dt: f32) {
        let [linear, ang_a, ang_b] = parts;
        if let Some(goal) = motor_goal(axis.motor, index, value, dt) {
            self.push(linear, ang_a, ang_b, goal);
        }
        if let Some(goal) = limit_goal(axis.limit, index, value, dt) {
            self.push(linear, ang_a, ang_b, goal);
        }
    }
}

fn motor_goal(motor: Option<JointMotor>, index: usize, value: f32, dt: f32) -> Option<RowGoal> {
    let (target, max_force) = match motor? {
        JointMotor::Velocity { speed, max_force } => (speed, max_force),
        JointMotor::Position {
            target,
            max_speed,
            max_force,
        } => {
            let wanted = (target - value) * RECOVERY_RATE / dt;
            (wanted.clamp(-max_speed, max_speed), max_force)
        }
    };
    let reach = max_force * dt;
    Some(RowGoal {
        slot: MOTOR_SLOT + index,
        target,
        low: -reach,
        high: reach,
        recovery: 0.0,
        recovery_low: 0.0,
        recovery_high: 0.0,
    })
}

fn limit_goal(limit: JointLimit, slot: usize, value: f32, dt: f32) -> Option<RowGoal> {
    let locked_at = |at: f32| {
        let wanted = (at - value) * RECOVERY_RATE / dt;
        RowGoal {
            slot,
            target: 0.0,
            low: f32::NEG_INFINITY,
            high: f32::INFINITY,
            recovery: wanted.clamp(-MAX_RECOVERY_SPEED, MAX_RECOVERY_SPEED),
            recovery_low: f32::NEG_INFINITY,
            recovery_high: f32::INFINITY,
        }
    };
    match limit {
        JointLimit::Free => None,
        JointLimit::Locked => Some(locked_at(0.0)),
        JointLimit::Range { min, max } if min == max => Some(locked_at(min)),
        JointLimit::Range { min, max } => Some(if value - min <= max - value {
            // inside the range it may still close the gap in one step
            let error = min - value;
            RowGoal {
                slot,
                target: (error / dt).min(0.0),
                low: 0.0,
                high: f32::INFINITY,
                recovery: (error * RECOVERY_RATE / dt).clamp(0.0, MAX_RECOVERY_SPEED),
                recovery_low: 0.0,
                recovery_high: f32::INFINITY,
            }
        } else {
            let error = max - value;
            RowGoal {
                slot,
                target: (error / dt).max(0.0),
                low: f32::NEG_INFINITY,
                high: 0.0,
                recovery: (error * RECOVERY_RATE / dt).clamp(-MAX_RECOVERY_SPEED, 0.0),
                recovery_low: f32::NEG_INFINITY,
                recovery_high: 0.0,
            }
        }),
    }
}

/// Angle of each axis of `relative`, the turn of B's frame in A's frame.
/// With one open axis it splits off the twist, so a hinge far around still reads its lock right.
/// Other setups read each axis on its own, which is only close for small swing.
fn relative_angles(relative: Quat, axes: &[JointAxis; 6]) -> [f32; 3] {
    let [x, y, z, w] = relative.to_array();
    let (x, y, z, w) = if w < 0.0 {
        (-x, -y, -z, -w)
    } else {
        (x, y, z, w)
    };
    let parts = [x, y, z];
    let mut open = [false; 3];
    for (out, axis) in open.iter_mut().zip(&axes[3..]) {
        *out = axis.limit != JointLimit::Locked;
    }
    let open_count = open.iter().filter(|value| **value).count();
    let single = open.iter().position(|value| *value);
    if let (1, Some(axis)) = (open_count, single) {
        let twist = parts[axis].atan2(w);
        let length = (parts[axis] * parts[axis] + w * w).sqrt();
        if length > 1e-6 {
            let (twist_v, twist_w) = (parts[axis] / length, w / length);
            let mut angle = [0.0; 3];
            // swing = relative * conjugate(twist)
            let next = (axis + 1) % 3;
            let last = (axis + 2) % 3;
            angle[next] = 2.0 * (parts[next] * twist_w - parts[last] * twist_v);
            angle[last] = 2.0 * (parts[last] * twist_w + parts[next] * twist_v);
            angle[axis] = 2.0 * twist;
            return angle;
        }
    }
    [2.0 * x.atan2(w), 2.0 * y.atan2(w), 2.0 * z.atan2(w)]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_rows(
    joints: &Joints,
    impulses: &[[f32; JOINT_IMPULSES]],
    bodies: &Bodies,
    poses: &[Integrated],
    awake: &[bool],
    points: &mut Vec<JointPoint>,
    rows: &mut Vec<JointRow>,
    dt: f32,
) {
    points.clear();
    rows.clear();
    let active = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic && awake[dense];
    for (index, state) in joints.live() {
        let desc = &state.desc;
        let (Ok(a), Ok(b)) = (bodies.dense_index(desc.a), bodies.dense_index(desc.b)) else {
            continue;
        };
        if !active(a) && !active(b) {
            continue;
        }
        let (rotation_a, rotation_b) = (poses[a].rotation, poses[b].rotation);
        let arm_a = rotation_a.rotate_vec3(desc.anchor_a);
        let arm_b = rotation_b.rotate_vec3(desc.anchor_b);
        let center_b = poses[b].position.offset_from(poses[a].position).as_vec3();
        let gap = center_b + arm_b - arm_a;
        let mut builder = RowBuilder {
            bodies,
            poses,
            impulses: &impulses[index],
            joint: index as u32,
            a,
            b,
            rows,
        };
        match desc.kind {
            JointKind::Distance { min, max } => {
                let length = gap.length();
                let along = if length > 1e-6 {
                    gap * length.recip()
                } else {
                    Vec3::Y
                };
                let parts = [along, arm_a.cross(along), arm_b.cross(along)];
                let axis = JointAxis::range(min, max);
                builder.axis(0, &axis, length, parts, dt);
            }
            JointKind::Axes(axes) => {
                let world_a = rotation_a * desc.frame_a;
                let world_b = rotation_b * desc.frame_b;
                let directions = [
                    world_a.rotate_vec3(Vec3::X),
                    world_a.rotate_vec3(Vec3::Y),
                    world_a.rotate_vec3(Vec3::Z),
                ];
                let point_joint = axes[..3]
                    .iter()
                    .all(|axis| axis.limit == JointLimit::Locked && axis.motor.is_none());
                if point_joint {
                    points.push(build_point(
                        bodies,
                        poses,
                        &impulses[index],
                        index as u32,
                        (a, b),
                        (arm_a, arm_b),
                        directions,
                        point_recovery(gap, dt),
                    ));
                } else {
                    // the point on A sits where B's anchor is, that keeps sliders exact when far out
                    let reach_a = arm_a + gap;
                    for (index, direction) in directions.iter().enumerate() {
                        let parts = [
                            *direction,
                            reach_a.cross(*direction),
                            arm_b.cross(*direction),
                        ];
                        builder.axis(index, &axes[index], gap.dot(*direction), parts, dt);
                    }
                }
                let angles = relative_angles(world_a.conjugate() * world_b, &axes);
                for (index, direction) in directions.iter().enumerate() {
                    let parts = [Vec3::ZERO, *direction, *direction];
                    builder.axis(3 + index, &axes[3 + index], angles[index], parts, dt);
                }
            }
        }
    }
}

pub(crate) fn warm_start_row(pair: &mut Pair, row: &JointRow) {
    apply_row(pair, row, row.impulse);
}

pub(crate) fn solve_row(pair: &mut Pair, row: &mut JointRow) {
    let (a, b) = (&*pair.a, &*pair.b);
    let speed =
        row.linear.dot(b.linear - a.linear) + row.ang_b.dot(b.angular) - row.ang_a.dot(a.angular);
    let next = (row.impulse + (row.target - speed) * row.mass)
        .min(row.high)
        .max(row.low);
    let change = next - row.impulse;
    row.impulse = next;
    apply_row(pair, row, change);
}

pub(crate) fn solve_row_recovery(pair: &mut Pair, row: &mut JointRow) {
    let (a, b) = (&*pair.a, &*pair.b);
    let speed = row.linear.dot(b.recovery_linear - a.recovery_linear)
        + row.ang_b.dot(b.recovery_angular)
        - row.ang_a.dot(a.recovery_angular);
    let next = (row.recovery_impulse + (row.recovery - speed) * row.mass)
        .min(row.recovery_high)
        .max(row.recovery_low);
    let change = next - row.recovery_impulse;
    row.recovery_impulse = next;
    let body = &mut *pair.a;
    body.recovery_linear -= row.linear * (change * row.inverse_a);
    body.recovery_angular -= row.turn_a * change;
    let body = &mut *pair.b;
    body.recovery_linear += row.linear * (change * row.inverse_b);
    body.recovery_angular += row.turn_b * change;
}

fn apply_row(pair: &mut Pair, row: &JointRow, impulse: f32) {
    let body = &mut *pair.a;
    body.linear -= row.linear * (impulse * row.inverse_a);
    body.angular -= row.turn_a * impulse;
    body.touched = true;
    let body = &mut *pair.b;
    body.linear += row.linear * (impulse * row.inverse_b);
    body.angular += row.turn_b * impulse;
    body.touched = true;
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use super::*;
    use crate::{
        Command, CommandBuffer, FrameDesc, FrameId, GridPosition, PhysicsSettings, PhysicsWorld,
        RigidBodyDesc, Shape,
    };

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

    struct SkipsJobs;

    impl crate::JobDispatch for SkipsJobs {
        fn parallelism(&self) -> usize {
            2
        }

        fn run(&self, _jobs: usize, _job: &(dyn Fn(usize) + Sync)) {}
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
        assert!(world.step_with(&SkipsJobs).is_err());
        assert_eq!(world.joint_reaction(joint).unwrap(), before);
        assert_eq!(world.save_snapshot(), bytes);
    }
}
