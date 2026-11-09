use algebrix::Vec3;

use crate::BodyKind;
use crate::body::Bodies;
use crate::world::Integrated;

use super::Pair;
use super::contacts::{NormalBlock, SolverManifold, SolverPoint};

/// Two points still apart and closing faster than this, m/s, are solved as one block. Resting
/// pairs keep the iterations, a block there cost a pile 10% and changed nothing.
const BLOCK_SPEED: f32 = 1.0;

/// A point slipping slower than this, m/s, counts as held still and gets static friction.
const SLIP_SPEED: f32 = 0.05;

pub(super) fn warm_start(pair: &mut Pair, manifold: &SolverManifold, points: &mut [SolverPoint]) {
    for k in 0..3 {
        turn(pair, manifold, k, manifold.resist_impulse[k]);
    }
    for point in points.iter() {
        for k in 0..3 {
            push(
                pair,
                (&manifold.axes, manifold.inverse_mass),
                point,
                k,
                point.impulse[k],
            );
        }
    }
    // with last substep's impulses back in, a held point barely moves, a sliding one still does.
    // decided per point and not per row, a point's rows share the load of the whole body
    for point in points.iter_mut() {
        let (one, two) = (
            slip_along(pair, manifold, point, 1),
            slip_along(pair, manifold, point, 2),
        );
        point.sliding = one * one + two * two > SLIP_SPEED * SLIP_SPEED;
    }
}

/// Angular impulse `amount` about the manifold's axis `k`, A one way and B the other.
fn turn(pair: &mut Pair, manifold: &SolverManifold, k: usize, amount: f32) {
    if amount == 0.0 {
        return;
    }
    let [turn_a, turn_b] = manifold.resist_turn[k];
    pair.a.angular -= turn_a * amount;
    pair.b.angular += turn_b * amount;
    pair.a.touched = true;
    pair.b.touched = true;
}

/// Spin and rolling held back up to their levers times the load the points carry now.
fn resist(pair: &mut Pair, manifold: &mut SolverManifold, load: f32) {
    for k in 0..3 {
        let lever = manifold.resist[k];
        if lever <= 0.0 {
            continue;
        }
        let axis = manifold.axes[k];
        let speed = (pair.b.angular - pair.a.angular).dot(axis);
        let limit = (lever * load).max(0.0);
        let accumulated = manifold.resist_impulse[k];
        // not `clamp`, it panics on a nan limit
        let next = (accumulated - speed * manifold.resist_mass[k])
            .min(limit)
            .max(-limit);
        manifold.resist_impulse[k] = next;
        turn(pair, manifold, k, next - accumulated);
    }
}

/// How fast the surfaces slide over each other along tangent `k`, belts counted.
fn slip_along(pair: &Pair, manifold: &SolverManifold, point: &SolverPoint, k: usize) -> f32 {
    speed_along(pair, &manifold.axes, point, k) + manifold.surface.dot(manifold.axes[k])
}

pub(super) fn solve_manifold(
    pair: &mut Pair,
    manifold: &mut SolverManifold,
    points: &mut [SolverPoint],
) {
    let (axes, inverse) = (&manifold.axes, manifold.inverse_mass);
    // friction first, so the normal impulse has the last word on penetration
    if let (Some(block), [one, two]) = (manifold.block, &mut *points)
        && one.target.min(two.target) < -BLOCK_SPEED
    {
        solve_friction(pair, manifold, one);
        solve_friction(pair, manifold, two);
        solve_block(pair, (axes, inverse), &block, [one, two]);
    } else {
        for point in points.iter_mut() {
            solve_friction(pair, manifold, point);
            let speed = speed_along(pair, axes, point, 0);
            let next = (point.impulse.x + (point.target - speed) * point.mass.x).max(0.0);
            let change = next - point.impulse.x;
            point.impulse.x = next;
            push(pair, (axes, inverse), point, 0, change);
        }
    }
    let load: f32 = points.iter().map(|point| point.impulse.x).sum();
    resist(pair, manifold, load);
}

fn solve_friction(pair: &mut Pair, manifold: &SolverManifold, point: &mut SolverPoint) {
    let (axes, inverse) = (&manifold.axes, manifold.inverse_mass);
    let (limit, hold) = (
        (manifold.friction * point.impulse.x).max(0.0),
        (manifold.static_friction * point.impulse.x).max(0.0),
    );
    for k in 1..3 {
        let speed = slip_along(pair, manifold, point, k);
        let accumulated = point.impulse[k];
        let bound = if point.sliding { limit } else { hold };
        // not `clamp`, it panics on a nan limit
        let next = (accumulated - speed * point.mass[k]).min(bound).max(-bound);
        point.impulse[k] = next;
        push(pair, (axes, inverse), point, k, next - accumulated);
    }
}

/// Both normal impulses at once: the totals that leave each point at its target speed or
/// pushing nothing, tried as both pushing, either alone, then neither (the block solver of
/// Box2D).
fn solve_block(
    pair: &mut Pair,
    parts: (&[Vec3; 3], [f32; 2]),
    block: &NormalBlock,
    [one, two]: [&mut SolverPoint; 2],
) {
    let [k11, k12, k22] = block.k;
    let [i11, i12, i22] = block.inverse;
    let (old1, old2) = (one.impulse.x, two.impulse.x);
    // speed over target each point would have with no impulse at all
    let b1 = speed_along(pair, parts.0, one, 0) - one.target - (k11 * old1 + k12 * old2);
    let b2 = speed_along(pair, parts.0, two, 0) - two.target - (k12 * old1 + k22 * old2);
    let both = (-(i11 * b1 + i12 * b2), -(i12 * b1 + i22 * b2));
    let first = -b1 / k11;
    let second = -b2 / k22;
    let (x1, x2) = if both.0 >= 0.0 && both.1 >= 0.0 {
        both
    } else if first >= 0.0 && k12 * first + b2 >= 0.0 {
        (first, 0.0)
    } else if second >= 0.0 && k12 * second + b1 >= 0.0 {
        (0.0, second)
    } else if b1 >= 0.0 && b2 >= 0.0 {
        (0.0, 0.0)
    } else {
        // only rounding gets here, the old impulses stay
        (old1, old2)
    };
    one.impulse.x = x1;
    two.impulse.x = x2;
    push(pair, parts, one, 0, x1 - old1);
    push(pair, parts, two, 0, x2 - old2);
}

/// Split impulse. Pushes overlapping bodies apart through position only, so it adds no energy.
pub(super) fn solve_recovery(
    pair: &mut Pair,
    manifold: &SolverManifold,
    points: &mut [SolverPoint],
) {
    let normal = manifold.axes[0];
    let [inverse_a, inverse_b] = manifold.inverse_mass;
    for point in points {
        let (a, b) = (&mut *pair.a, &mut *pair.b);
        let speed = (b.recovery_linear - a.recovery_linear).dot(normal)
            + b.recovery_angular.dot(point.arm_b[0])
            - a.recovery_angular.dot(point.arm_a[0]);
        let next = (point.recovery_impulse + (point.recovery - speed) * point.mass.x).max(0.0);
        let change = next - point.recovery_impulse;
        point.recovery_impulse = next;
        if inverse_a > 0.0 {
            a.recovery_linear -= normal * (change * inverse_a);
            a.recovery_angular -= point.turn_a[0] * change;
        }
        if inverse_b > 0.0 {
            b.recovery_linear += normal * (change * inverse_b);
            b.recovery_angular += point.turn_b[0] * change;
        }
    }
}

/// Speed of B against A at the point along axis `k`.
fn speed_along(pair: &Pair, axes: &[Vec3; 3], point: &SolverPoint, k: usize) -> f32 {
    let (a, b) = (&*pair.a, &*pair.b);
    (b.linear - a.linear).dot(axes[k]) + b.angular.dot(point.arm_b[k])
        - a.angular.dot(point.arm_a[k])
}

/// `impulse` along axis `k` on B, its opposite on A.
fn push(
    pair: &mut Pair,
    (axes, [inverse_a, inverse_b]): (&[Vec3; 3], [f32; 2]),
    point: &SolverPoint,
    k: usize,
    impulse: f32,
) {
    let axis = axes[k];
    if inverse_a > 0.0 {
        let body = &mut *pair.a;
        body.linear -= axis * (impulse * inverse_a);
        body.angular -= point.turn_a[k] * impulse;
        body.touched = true;
    }
    if inverse_b > 0.0 {
        let body = &mut *pair.b;
        body.linear += axis * (impulse * inverse_b);
        body.angular += point.turn_b[k] * impulse;
        body.touched = true;
    }
}

pub(super) fn dynamic_inverse_mass(bodies: &Bodies, dense: usize) -> f32 {
    if bodies.kind[dense] == BodyKind::Dynamic {
        bodies.inverse_mass[dense]
    } else {
        0.0
    }
}

pub(super) fn relative_velocity(
    state: &[Integrated],
    a: usize,
    b: usize,
    offset_a: Vec3,
    offset_b: Vec3,
) -> Vec3 {
    let at_a = state[a].linear + state[a].angular.cross(offset_a);
    let at_b = state[b].linear + state[b].angular.cross(offset_b);
    at_b - at_a
}

pub(super) fn tangent_of(normal: Vec3) -> Vec3 {
    let helper = if normal.x.abs() < 0.57 {
        Vec3::X
    } else {
        Vec3::Y
    };
    normal.cross(helper).normalize()
}
