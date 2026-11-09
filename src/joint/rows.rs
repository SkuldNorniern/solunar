use algebrix::{Quat, Vec3};

use crate::BodyKind;
use crate::body::{Bodies, apply_diagonal};
use crate::direct::{DirectNode, DirectRow};
use crate::solver::Pair;
use crate::world::Integrated;

use super::store::Joints;
use super::{JOINT_IMPULSES, JointAxis, JointKind, JointLimit, JointMotor};

pub(crate) const MOTOR_SLOT: usize = 6;
const RECOVERY_RATE: f32 = 0.2;
const MAX_RECOVERY_SPEED: f32 = 3.0;

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
    pub(super) joint: u32,
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
    pub(super) impulse: Vec3,
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
    pub(super) joint: u32,
    slot: usize,
    a: usize,
    b: usize,
    pub(super) linear: Vec3,
    ang_a: Vec3,
    pub(super) ang_b: Vec3,
    turn_a: Vec3,
    turn_b: Vec3,
    inverse_a: f32,
    inverse_b: f32,
    mass: f32,
    target: f32,
    low: f32,
    high: f32,
    pub(super) impulse: f32,
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
