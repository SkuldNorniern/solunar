use std::sync::{Mutex, PoisonError};

use algebrix::{Quat, Vec3};

use crate::body::{Bodies, apply_diagonal};
use crate::collision::{Bounds, MAX_CONTACTS, MovingPairs, RawManifold};
use crate::direct::{DirectNode, Factor, Symbolic};
use crate::joint::{JOINT_IMPULSES, JointPoint, JointRow, Joints};
use crate::material::Contact;
use crate::world::Integrated;
use crate::{BodyHandle, BodyKind, GridPosition, JobDispatch, PhysicsError, Shape};

use super::islands::Islands;
use super::rows::{dynamic_inverse_mass, relative_velocity, tangent_of};
use super::stages::Colors;
use super::{Motion, pair_margin, part_material, straight_motion};

/// Overlap is pushed out by a spring this stiff, hertz, and this damped, as a ratio of
/// critical (box2d's soft contact). Pushed through position alone it never pushed back on
/// the speed, and a pile of 400 rocks crept and slumped for a minute.
const CONTACT_HERTZ: f32 = 60.0;
const CONTACT_DAMPING: f32 = 10.0;
const MAX_PUSH_SPEED: f32 = 3.0;
/// Overlap deeper than this, meters, goes out through position only, as before the soft rows:
/// pushed out through speed a spawned overlap or a fast hit came out with more energy.
const SOFT_REACH: f32 = 0.01;
const RECOVERY_RATE: f32 = 0.2;
const MAX_RECOVERY_SPEED: f32 = 3.0;
const MATCH_DISTANCE_SQUARED: f32 = 0.05 * 0.05;
/// A pair whose bodies moved less than this against each other since its contacts were found,
/// meters squared, keeps them rather than finding them again.
/// Radians a substep a manifold's bodies may turn together and keep their arms from the first
/// substep.
const FAST_TURN: f32 = 0.2;
const REUSE_DISTANCE_SQUARED: f32 = 0.0002 * 0.0002;
/// The same for turning: the cosine of half of 0.1 degree.
const REUSE_HALF_TURN_COS: f32 = 0.999_999_6;
/// Normals this close keep their friction when a manifold's points move on.
const SAME_NORMAL_COS: f32 = 0.9999;

/// Pairs in one job of contact finding. Fixed, so the output does not depend on the thread count.
const PAIR_CHUNK: usize = 128;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CachedPoint {
    /// Contact in body A axes, finds the same point next step.
    pub(crate) local_a: Vec3,
    /// The same on B, in its axes.
    pub(crate) local_b: Vec3,
    pub(crate) impulse: Vec3,
    /// Normal impulse over the whole step, every substep's added, newton seconds. A hit is
    /// mostly taken in the first substep, the last one alone says little about it.
    pub(crate) pushed: f32,
}

/// `pushed` is what the last step did, the next step never reads it, so snapshots leave it
/// out and equal state ignores it.
impl PartialEq for CachedPoint {
    fn eq(&self, other: &Self) -> bool {
        self.local_a == other.local_a
            && self.local_b == other.local_b
            && self.impulse == other.impulse
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CachedManifold {
    pub(crate) a: BodyHandle,
    pub(crate) b: BodyHandle,
    pub(crate) part_a: u32,
    pub(crate) part_b: u32,
    pub(crate) points: [CachedPoint; MAX_CONTACTS],
    pub(crate) count: usize,
    /// The normal in A axes, zero when these contacts may not be kept, as from an old snapshot.
    pub(crate) normal: Vec3,
    /// Where B was in A's axes, center and turn, when these contacts were found.
    pub(crate) pose: (Vec3, Quat),
}

impl CachedManifold {
    fn keeps(&self, (center, turn): (Vec3, Quat)) -> bool {
        self.normal != Vec3::ZERO
            && (center - self.pose.0).length_squared() < REUSE_DISTANCE_SQUARED
            && turn.dot(self.pose.1).abs() > REUSE_HALF_TURN_COS
    }
}

/// `arm_a[k]` is `offset_a × axes[k]`, `turn_a[k]` the spin of A per unit impulse along it.
/// Both are worked out once a substep, the iterations only scale and add them.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SolverPoint {
    pub(super) arm_a: [Vec3; 3],
    pub(super) arm_b: [Vec3; 3],
    pub(super) turn_a: [Vec3; 3],
    pub(super) turn_b: [Vec3; 3],
    pub(super) mass: Vec3,
    pub(super) target: f32,
    /// Soft rows keep this share of their mass and give up this share of what they took, rigid
    /// ones 1 and 0.
    pub(super) mass_scale: f32,
    pub(super) impulse_scale: f32,
    /// What the relax pass holds the normal speed to, worked out after the bodies moved.
    pub(super) relax_target: f32,
    /// Bouncing this substep, the relax keeps the bounce.
    pub(super) bouncing: bool,
    /// Position only push speed for the overlap past `SOFT_REACH`, and what it took so far.
    pub(super) recovery: f32,
    pub(super) recovery_impulse: f32,
    pub(super) impulse: Vec3,
    /// Slipping faster than `SLIP_SPEED` when the substep began, static friction no longer holds.
    pub(super) sliding: bool,
}

/// What a contact point keeps that the iterations never read, apart so a pass reads less.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct PointFrame {
    local_a: Vec3,
    local_b: Vec3,
    bounce: f32,
    pub(super) pushed: f32,
}

#[derive(Clone, Copy)]
pub(super) struct SolverManifold {
    pub(super) a: usize,
    pub(super) b: usize,
    handle_a: BodyHandle,
    handle_b: BodyHandle,
    part_a: u32,
    part_b: u32,
    pub(super) axes: [Vec3; 3],
    pub(super) friction: f32,
    /// The most a point takes before it slides, then `friction` holds it.
    pub(super) static_friction: f32,
    /// B's surface speed over A's, world, from belts. Friction carries the bodies to it.
    pub(super) surface: Vec3,
    /// Spin about the normal and rolling about the two tangents, resisted up to these levers
    /// times the manifold's load. Zero skips the row.
    pub(super) resist: [f32; 3],
    /// Per angular row: spin of A and of B per unit impulse, its mass and what it took so far.
    pub(super) resist_turn: [[Vec3; 2]; 3],
    pub(super) resist_mass: [f32; 3],
    pub(super) resist_impulse: [f32; 3],
    /// Two points' normal rows solved together, none when they are too near the same.
    pub(super) block: Option<NormalBlock>,
    /// Zero for bodies that are not dynamic, they are never pushed.
    pub(super) inverse_mass: [f32; 2],
    /// Its points are `first..first + count` in the scratch point lists.
    pub(super) first: usize,
    pub(super) count: usize,
    normal_a: Vec3,
    pose: (Vec3, Quat),
}

/// What one job of contact finding made. `first` in its manifolds is local to it.
#[derive(Debug, Clone, Default)]
struct FoundContacts {
    manifolds: Vec<SolverManifold>,
    points: Vec<SolverPoint>,
    frames: Vec<PointFrame>,
    done: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ContactScratch {
    pub(super) bounds: Vec<Bounds>,
    /// What each bound was made from, a body that did not change keeps it.
    pub(super) posed: Vec<(GridPosition, Quat, Shape, f32)>,
    pub(super) order: Vec<u32>,
    pub(super) pairs: Vec<(u32, u32)>,
    pub(super) moving: MovingPairs,
    pub(super) manifolds: Vec<SolverManifold>,
    pub(super) points: Vec<SolverPoint>,
    pub(super) frames: Vec<PointFrame>,
    found: Vec<FoundContacts>,
    pub(super) joint_points: Vec<JointPoint>,
    pub(super) joint_rows: Vec<JointRow>,
    pub(super) joint_impulses: Vec<[f32; JOINT_IMPULSES]>,
    /// Each joint's force and torque impulse on B, over the substeps so far.
    pub(super) joint_loads: Vec<(Vec3, Vec3)>,
    /// Inverse inertia as the joints see it this step.
    pub(super) joint_turning: Vec<Vec3>,
    pub(super) islands: Islands,
    pub(super) colors: Colors,
    pub(super) motions: Vec<Motion>,
    /// One factor per serial island, its nodes beside `colors.direct`. Shapes are kept between
    /// substeps and steps while an island joins the same bodies.
    pub(super) factors: Vec<Factor>,
    pub(super) shapes: Vec<Symbolic>,
    pub(super) direct_nodes: Vec<DirectNode>,
    /// Manifolds and points of each refresh block, worked out before the stages run.
    pub(super) refresh_ranges: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)>,
}

impl ContactScratch {
    pub(crate) fn islands(&self) -> &Islands {
        &self.islands
    }
}

impl ContactScratch {
    pub(crate) fn pairs(&self) -> &[(u32, u32)] {
        &self.pairs
    }

    pub(crate) fn joint_parts(&self) -> (&[JointPoint], &[JointRow], &[[f32; JOINT_IMPULSES]]) {
        (&self.joint_points, &self.joint_rows, &self.joint_impulses)
    }

    pub(crate) fn joint_loads(&self) -> &[(Vec3, Vec3)] {
        &self.joint_loads
    }

    /// Dense pairs that made contact in the last solve.
    pub(crate) fn touching(&self) -> impl Iterator<Item = (usize, usize)> + Clone + '_ {
        self.manifolds
            .iter()
            .map(|manifold| (manifold.a, manifold.b))
    }
}

impl std::fmt::Debug for SolverManifold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverManifold")
            .field("a", &self.a)
            .field("b", &self.b)
            .field("count", &self.count)
            .finish()
    }
}

/// Bodies moved since the contacts were found, so gaps follow them. Arms and masses are worked
/// out in the first substep and kept for the step, as box2d does: redone every substep they were
/// a sixth of the 2000 body pile's step, the bodies barely turn in a step.
/// Manifolds refreshed in one job.
const REFRESH_CHUNK: usize = 256;

/// Blocks of manifolds refreshed as the first stage of a solve, `REFRESH_CHUNK` each. Points lie
/// in manifold order, one range after another, so a block's points are one range too. Out of
/// that order the whole refresh is one block.
pub(super) fn refresh_blocks(manifolds: &[SolverManifold]) -> u32 {
    let in_a_row = manifolds.first().is_none_or(|first| first.first == 0)
        && manifolds
            .windows(2)
            .all(|pair| pair[0].first + pair[0].count == pair[1].first);
    if manifolds.is_empty() {
        0
    } else if in_a_row {
        manifolds.len().div_ceil(REFRESH_CHUNK) as u32
    } else {
        1
    }
}

/// The manifolds and points of refresh block `block` out of `blocks`.
pub(super) fn refresh_range(
    manifolds: &[SolverManifold],
    (block, blocks): (usize, u32),
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    if blocks == 1 {
        let points = manifolds
            .iter()
            .map(|manifold| manifold.first + manifold.count)
            .max()
            .unwrap_or(0);
        return (0..manifolds.len(), 0..points);
    }
    let start = block * REFRESH_CHUNK;
    let end = (start + REFRESH_CHUNK).min(manifolds.len());
    let first = manifolds[start].first;
    let last = &manifolds[end - 1];
    (start..end, first..last.first + last.count)
}

/// Refreshes one block: `manifolds` with their `points` and `frames` starting at `base`.
pub(super) fn refresh_block(
    bodies: &Bodies,
    state: &[Integrated],
    (manifolds, points, frames): (&mut [SolverManifold], &mut [SolverPoint], &[PointFrame]),
    base: usize,
    flags: (bool, bool, bool),
    dt: f32,
) {
    for manifold in manifolds.iter_mut() {
        let range = manifold.first - base..manifold.first - base + manifold.count;
        let pair = (&mut points[range.clone()], &frames[range]);
        refresh_manifold(bodies, state, manifold, pair, flags, dt);
    }
}

fn refresh_manifold(
    bodies: &Bodies,
    state: &[Integrated],
    manifold: &mut SolverManifold,
    (points, frames): (&mut [SolverPoint], &[PointFrame]),
    (first, last, relax): (bool, bool, bool),
    dt: f32,
) {
    let (a, b) = (&state[manifold.a], &state[manifold.b]);
    // against something that never moves twice as stiff, as box2d does
    let fixed = manifold.inverse_mass[0] == 0.0 || manifold.inverse_mass[1] == 0.0;
    let soft = soft_contact(dt, if fixed { 2.0 } else { 1.0 });
    // a body turning fast moves its arms a lot in a substep, kept from the first one a capsule
    // rolling at 1700 rad/s tumbled up to 38,000 and flew off
    let first = first || (a.angular.length() + b.angular.length()) * dt > FAST_TURN;
    let center_b = b.position.offset_from(a.position).as_vec3();
    let normal = manifold.axes[0];
    let [inverse_a, inverse_b] = manifold.inverse_mass;
    let response = |dense: usize, rotation, arm: Vec3| {
        if bodies.kind[dense] == BodyKind::Dynamic {
            apply_diagonal(rotation, bodies.inverse_inertia[dense], arm)
        } else {
            Vec3::ZERO
        }
    };
    for k in 0..3 {
        if !first || manifold.resist[k] <= 0.0 {
            continue;
        }
        let axis = manifold.axes[k];
        let (turn_a, turn_b) = (
            response(manifold.a, a.rotation, axis),
            response(manifold.b, b.rotation, axis),
        );
        let inverse = axis.dot(turn_a) + axis.dot(turn_b);
        manifold.resist_turn[k] = [turn_a, turn_b];
        manifold.resist_mass[k] = if inverse > 1e-12 {
            inverse.recip()
        } else {
            0.0
        };
    }
    let (shape_a, shape_b) = (bodies.shape[manifold.a], bodies.shape[manifold.b]);
    for (point, frame) in points.iter_mut().zip(frames) {
        let offset_a = round_point(&shape_a, a.rotation, frame.local_a, normal);
        let offset_b = round_point(&shape_b, b.rotation, frame.local_b, -normal);
        if first {
            for (k, axis) in manifold.axes.iter().enumerate() {
                point.arm_a[k] = offset_a.cross(*axis);
                point.arm_b[k] = offset_b.cross(*axis);
                point.turn_a[k] = response(manifold.a, a.rotation, point.arm_a[k]);
                point.turn_b[k] = response(manifold.b, b.rotation, point.arm_b[k]);
                let inverse = inverse_a
                    + inverse_b
                    + point.arm_a[k].dot(point.turn_a[k])
                    + point.arm_b[k].dot(point.turn_b[k]);
                point.mass[k] = if inverse > 1e-12 {
                    inverse.recip()
                } else {
                    0.0
                };
            }
        }
        let gap = (center_b + offset_b - offset_a).dot(normal);
        if relax {
            // after the move: only what the relax holds to, the bounce kept as it was
            if !point.bouncing {
                point.relax_target = if gap > 0.0 { -gap / dt } else { 0.0 };
            }
            continue;
        }
        // still apart it may close the gap, and only arriving can bounce. arriving in the last
        // substep would clamp the speed the next step's bounce is worked out from, so it bounces now
        let speed =
            relative_velocity(state, manifold.a, manifold.b, offset_a, offset_b).dot(normal);
        let arrives = -speed * dt >= gap;
        point.bouncing = frame.bounce > 0.0 && (gap <= 0.0 || (last && arrives));
        (point.target, point.mass_scale, point.impulse_scale) = if point.bouncing {
            (frame.bounce, 1.0, 0.0)
        } else if gap > 0.0 {
            (-gap / dt, 1.0, 0.0)
        } else {
            (
                (soft.0 * (-gap).min(SOFT_REACH)).min(MAX_PUSH_SPEED),
                soft.1,
                soft.2,
            )
        };
        point.recovery =
            (RECOVERY_RATE / dt * (-gap - SOFT_REACH).max(0.0)).min(MAX_RECOVERY_SPEED);
        point.recovery_impulse = 0.0;
        point.relax_target = if point.bouncing {
            point.target
        } else if gap > 0.0 {
            -gap / dt
        } else {
            0.0
        };
    }
    if first {
        manifold.block = match &*points {
            [one, two] => NormalBlock::of(one, two, inverse_a + inverse_b),
            _ => None,
        };
    }
}

/// Push out speed per meter of overlap, mass scale and impulse scale of a soft contact row for
/// a substep of `dt` (Catto, Solver2D, 2024). Never stiffer than a quarter of the substep rate,
/// half of it against a fixed body.
fn soft_contact(dt: f32, stiffer: f32) -> (f32, f32, f32) {
    let hertz = CONTACT_HERTZ.min(0.25 / dt) * stiffer;
    let omega = 2.0 * std::f32::consts::PI * hertz;
    let a1 = 2.0 * CONTACT_DAMPING + dt * omega;
    let a2 = dt * omega * a1;
    let a3 = 1.0 / (1.0 + a2);
    (omega / a1, a2 * a3, a3)
}

/// Two points' normal rows as one 2 by 2 system, `k` and its inverse. Solved one after the
/// other, a push at one end of a long light body mostly turns it, and a capsule's other end ran
/// through a wall at 5000 m/s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct NormalBlock {
    pub(super) k: [f32; 3],
    pub(super) inverse: [f32; 3],
}

/// Past this the two rows are near the same, and the block is left to the iterations.
const MOST_BLOCK_CONDITION: f32 = 1000.0;

impl NormalBlock {
    fn of(one: &SolverPoint, two: &SolverPoint, linear: f32) -> Option<Self> {
        if one.mass.x <= 0.0 || two.mass.x <= 0.0 {
            return None;
        }
        let k11 = one.mass.x.recip();
        let k22 = two.mass.x.recip();
        let k12 = linear + one.arm_a[0].dot(two.turn_a[0]) + one.arm_b[0].dot(two.turn_b[0]);
        let determinant = k11 * k22 - k12 * k12;
        if !(determinant.is_finite() && k11 * k11 < MOST_BLOCK_CONDITION * determinant) {
            return None;
        }
        let over = determinant.recip();
        Some(Self {
            k: [k11, k12, k22],
            inverse: [k22 * over, -k12 * over, k11 * over],
        })
    }
}

/// Where a contact point on a body is from its center, in world axes. On a ball the point facing
/// `toward` is the same however it has turned, so it is taken that way: carried round with a ball
/// spun up by a fast slide, 1700 rad/s, the point swung away from the wall between substeps and
/// the gap read as open while the ball went through a 5 cm plate. A capsule is a ball swept
/// along its axis, so its point is the one over the same place on the axis: rolling along a
/// plate at 100 rad/s it went through the same way.
fn round_point(shape: &Shape, rotation: Quat, local: Vec3, toward: Vec3) -> Vec3 {
    match *shape {
        Shape::Sphere { radius } => toward * radius,
        Shape::Capsule {
            radius,
            half_height,
        } => {
            let core = Vec3::new(0.0, local.y.clamp(-half_height, half_height), 0.0);
            rotation.rotate_vec3(core) + toward * radius
        }
        _ => rotation.rotate_vec3(local),
    }
}

/// `cache` goes in old and comes out new.
/// Pairs with nothing awake were skipped, and keep their entry for the wake up.
/// Keys and a spare list for [`finish_contacts`], kept between steps so it allocates nothing.
#[derive(Debug, Clone, Default)]
pub(crate) struct FinishScratch {
    keys: Vec<((BodyHandle, BodyHandle, u32, u32), u32)>,
    merged: Vec<CachedManifold>,
}

/// The next contact cache: sleeping pairs kept, this step's contacts in, all in key order. The
/// kept entries are in order already, this step's are sorted by key alone and merged in, so no
/// manifold is moved more than once.
pub(crate) fn finish_contacts(
    bodies: &Bodies,
    awake: &[bool],
    cache: &mut Vec<CachedManifold>,
    scratch: &ContactScratch,
    spare: &mut FinishScratch,
) {
    let active = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic && awake[dense];
    cache.retain(
        |entry| match (bodies.dense_index(entry.a), bodies.dense_index(entry.b)) {
            (Ok(a), Ok(b)) => !active(a) && !active(b),
            _ => false,
        },
    );
    let key = |entry: &CachedManifold| (entry.a, entry.b, entry.part_a, entry.part_b);
    spare.keys.clear();
    spare.keys.extend(
        scratch
            .manifolds
            .iter()
            .enumerate()
            .map(|(index, manifold)| {
                let key = (
                    manifold.handle_a,
                    manifold.handle_b,
                    manifold.part_a,
                    manifold.part_b,
                );
                (key, index as u32)
            }),
    );
    spare.keys.sort_unstable();
    let built = |index: usize| {
        let manifold = &scratch.manifolds[index];
        let mut points = [CachedPoint::default(); MAX_CONTACTS];
        let range = manifold.first..manifold.first + manifold.count;
        let solved = scratch.points[range.clone()]
            .iter()
            .zip(&scratch.frames[range]);
        for (out, (point, frame)) in points.iter_mut().zip(solved) {
            *out = CachedPoint {
                local_a: frame.local_a,
                local_b: frame.local_b,
                impulse: point.impulse,
                pushed: frame.pushed,
            };
        }
        CachedManifold {
            a: manifold.handle_a,
            b: manifold.handle_b,
            part_a: manifold.part_a,
            part_b: manifold.part_b,
            points,
            count: manifold.count,
            normal: manifold.normal_a,
            pose: manifold.pose,
        }
    };
    let merged = &mut spare.merged;
    merged.clear();
    merged.reserve(cache.len() + spare.keys.len());
    let mut kept = cache.iter().peekable();
    for &(wanted, index) in &spare.keys {
        while let Some(entry) = kept.next_if(|entry| key(entry) < wanted) {
            merged.push(*entry);
        }
        merged.push(built(index as usize));
    }
    merged.extend(kept.copied());
    std::mem::swap(cache, merged);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_manifolds(
    bodies: &Bodies,
    awake: &[bool],
    joints: &Joints,
    state: &[Integrated],
    cache: &[CachedManifold],
    scratch: &mut ContactScratch,
    dispatch: &dyn JobDispatch,
    dt: f32,
) -> Result<(), PhysicsError> {
    let chunks = scratch.pairs.len().div_ceil(PAIR_CHUNK);
    scratch.found.resize_with(chunks, FoundContacts::default);
    let find = |at: usize, out: &mut FoundContacts| {
        out.manifolds.clear();
        out.points.clear();
        out.frames.clear();
        let start = at * PAIR_CHUNK;
        let pairs = &scratch.pairs[start..(start + PAIR_CHUNK).min(scratch.pairs.len())];
        find_contacts(bodies, awake, joints, state, cache, pairs, out, dt);
        out.done = true;
    };
    if chunks < 2 || dispatch.parallelism() < 2 {
        for (at, out) in scratch.found.iter_mut().enumerate() {
            find(at, out);
        }
    } else {
        let slots: Vec<Mutex<&mut FoundContacts>> =
            scratch.found.iter_mut().map(Mutex::new).collect();
        dispatch.run(chunks, &|at| {
            if let Some(slot) = slots.get(at) {
                let mut out = slot.lock().unwrap_or_else(PoisonError::into_inner);
                find(at, &mut out);
            }
        });
    }
    scratch.manifolds.clear();
    scratch.points.clear();
    scratch.frames.clear();
    let mut all_done = true;
    for out in &mut scratch.found {
        all_done &= std::mem::take(&mut out.done);
        let base = scratch.points.len();
        scratch
            .manifolds
            .extend(out.manifolds.iter().map(|manifold| SolverManifold {
                first: manifold.first + base,
                ..*manifold
            }));
        scratch.points.extend_from_slice(&out.points);
        scratch.frames.extend_from_slice(&out.frames);
    }
    if !all_done {
        return Err(PhysicsError::DispatchIncomplete);
    }
    scratch.manifolds.sort_unstable_by_key(|manifold| {
        (
            manifold.handle_a,
            manifold.handle_b,
            manifold.part_a,
            manifold.part_b,
        )
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn find_contacts(
    bodies: &Bodies,
    awake: &[bool],
    joints: &Joints,
    state: &[Integrated],
    cache: &[CachedManifold],
    pairs: &[(u32, u32)],
    out: &mut FoundContacts,
    dt: f32,
) {
    let active = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic && awake[dense];
    for &(first, second) in pairs {
        let (mut a, mut b) = (first as usize, second as usize);
        if !active(a) && !active(b) {
            continue;
        }
        if bodies.frame[a] != bodies.frame[b]
            || !bodies.filter[a].allows(bodies.filter[b])
            || joints.excludes(bodies.handle_at(a), bodies.handle_at(b))
        {
            continue;
        }
        // the lower handle is always A, dense order can change but the cache must still match
        if bodies.handle_at(a) > bodies.handle_at(b) {
            std::mem::swap(&mut a, &mut b);
        }
        let center_b = bodies.position(b).offset_from(bodies.position(a)).as_vec3();
        let (manifolds, points, frames) = (&mut out.manifolds, &mut out.points, &mut out.frames);
        let pose = relative_pose(bodies, a, b, center_b);
        let kept = kept_contacts(bodies, cache, (a, b), pose);
        if !kept.is_empty() {
            for old in kept {
                manifolds.push(kept_manifold(
                    bodies,
                    state,
                    old,
                    (a, b),
                    center_b,
                    (&mut *points, &mut *frames),
                    dt,
                ));
            }
            continue;
        }
        bodies.assets.for_each_contact(
            (&bodies.shape[a], bodies.rotation[a]),
            (&bodies.shape[b], bodies.rotation[b]),
            center_b,
            pair_margin(bodies, a, b, dt),
            straight_motion(bodies, a, b, dt),
            |part_a, part_b, raw, at_a| {
                manifolds.push(contact_manifold(
                    bodies,
                    state,
                    cache,
                    (a, b),
                    (part_a, part_b),
                    raw,
                    at_a,
                    center_b,
                    (&mut *points, &mut *frames),
                    dt,
                ));
            },
        );
    }
}

/// Where B is in A's axes.
fn relative_pose(bodies: &Bodies, a: usize, b: usize, center_b: Vec3) -> (Vec3, Quat) {
    let back = bodies.rotation[a].conjugate();
    (back.rotate_vec3(center_b), back * bodies.rotation[b])
}

/// The cached contacts of pair `a`, `b` when all of them may be kept as they are: the bodies
/// barely moved against each other since they were found. Empty otherwise, or for a mesh, which
/// a terrain edit changes under a still body.
fn kept_contacts<'c>(
    bodies: &Bodies,
    cache: &'c [CachedManifold],
    (a, b): (usize, usize),
    pose: (Vec3, Quat),
) -> &'c [CachedManifold] {
    if matches!(bodies.shape[a], Shape::TriMesh(_)) || matches!(bodies.shape[b], Shape::TriMesh(_))
    {
        return &[];
    }
    let key = (bodies.handle_at(a), bodies.handle_at(b));
    let start = cache.partition_point(|entry| (entry.a, entry.b) < key);
    let end = start + cache[start..].partition_point(|entry| (entry.a, entry.b) == key);
    let found = &cache[start..end];
    if found.iter().all(|entry| entry.keeps(pose)) {
        found
    } else {
        &[]
    }
}

/// A cached manifold again, its points where they were on each body. Only the bounce is new.
fn kept_manifold(
    bodies: &Bodies,
    state: &[Integrated],
    old: &CachedManifold,
    (a, b): (usize, usize),
    center_b: Vec3,
    (points, frames): (&mut Vec<SolverPoint>, &mut Vec<PointFrame>),
    dt: f32,
) -> SolverManifold {
    let normal = bodies.rotation[a].rotate_vec3(old.normal);
    let (manifold, contact) = new_manifold(
        bodies,
        (a, b),
        (old.part_a, old.part_b),
        normal,
        (points.len(), old.count),
    );
    for cached in &old.points[..old.count] {
        let offset_a = bodies.rotation[a].rotate_vec3(cached.local_a);
        let offset_b = bodies.rotation[b].rotate_vec3(cached.local_b);
        let depth = (offset_a - offset_b - center_b).dot(normal);
        points.push(SolverPoint {
            impulse: cached.impulse,
            ..SolverPoint::default()
        });
        frames.push(PointFrame {
            local_a: cached.local_a,
            local_b: cached.local_b,
            bounce: bounce(
                state,
                (a, b),
                (offset_a, offset_b),
                normal,
                depth,
                &contact,
                dt,
            ),
            pushed: 0.0,
        });
    }
    SolverManifold {
        normal_a: old.normal,
        pose: old.pose,
        ..manifold
    }
}

/// What a point bounces back at, if it reaches the other body this step fast enough.
fn bounce(
    state: &[Integrated],
    (a, b): (usize, usize),
    (offset_a, offset_b): (Vec3, Vec3),
    normal: Vec3,
    depth: f32,
    contact: &Contact,
    dt: f32,
) -> f32 {
    let approach = relative_velocity(state, a, b, offset_a, offset_b).dot(normal);
    let reaches = -approach * dt >= -depth;
    if reaches && approach < -contact.bounce_threshold {
        -contact.restitution * approach
    } else {
        0.0
    }
}

/// A manifold's materials and axes, before its points.
fn new_manifold(
    bodies: &Bodies,
    (a, b): (usize, usize),
    (part_a, part_b): (u32, u32),
    normal: Vec3,
    (first, count): (usize, usize),
) -> (SolverManifold, Contact) {
    let tangent = tangent_of(normal);
    let axes = [normal, tangent, normal.cross(tangent)];
    let contact = Contact::of(
        &part_material(bodies, a, part_a),
        &part_material(bodies, b, part_b),
    );
    let manifold = SolverManifold {
        a,
        b,
        handle_a: bodies.handle_at(a),
        handle_b: bodies.handle_at(b),
        part_a,
        part_b,
        axes,
        friction: contact.friction,
        static_friction: contact.static_friction,
        surface: bodies.rotation[b].rotate_vec3(bodies.surface[b])
            - bodies.rotation[a].rotate_vec3(bodies.surface[a]),
        resist: [contact.spin, contact.rolling, contact.rolling],
        resist_turn: [[Vec3::ZERO; 2]; 3],
        resist_mass: [0.0; 3],
        resist_impulse: [0.0; 3],
        block: None,
        inverse_mass: [
            dynamic_inverse_mass(bodies, a),
            dynamic_inverse_mass(bodies, b),
        ],
        first,
        count,
        normal_a: bodies.rotation[a].conjugate().rotate_vec3(normal),
        pose: (Vec3::ZERO, Quat::IDENTITY),
    };
    (manifold, contact)
}

#[allow(clippy::too_many_arguments)]
fn contact_manifold(
    bodies: &Bodies,
    state: &[Integrated],
    cache: &[CachedManifold],
    (a, b): (usize, usize),
    (part_a, part_b): (u32, u32),
    raw: &RawManifold,
    at_a: Vec3,
    center_b: Vec3,
    (points, frames): (&mut Vec<SolverPoint>, &mut Vec<PointFrame>),
    dt: f32,
) -> SolverManifold {
    let (handle_a, handle_b) = (bodies.handle_at(a), bodies.handle_at(b));
    let old = cache
        .binary_search_by_key(&(handle_a, handle_b, part_a, part_b), |entry| {
            (entry.a, entry.b, entry.part_a, entry.part_b)
        })
        .ok()
        .map(|at| &cache[at]);

    let normal = raw.normal;
    let normal_a = bodies.rotation[a].conjugate().rotate_vec3(normal);
    let (manifold, contact) = new_manifold(
        bodies,
        (a, b),
        (part_a, part_b),
        normal,
        (points.len(), raw.count),
    );
    let manifold = SolverManifold {
        pose: relative_pose(bodies, a, b, center_b),
        ..manifold
    };
    let inverse_rotation_a = bodies.rotation[a].conjugate();
    let inverse_rotation_b = bodies.rotation[b].conjugate();
    let mut claimed = [false; MAX_CONTACTS];
    for point in &raw.points[..raw.count] {
        // raw points sit between the surfaces, each side anchors on its own. an anchor out in the
        // gap would swing on a long lever as soon as the body turned
        let middle = point.position + at_a;
        let offset_a = middle + normal * (point.depth * 0.5);
        let offset_b = middle - normal * (point.depth * 0.5) - center_b;
        let local_a = inverse_rotation_a.rotate_vec3(offset_a);
        let local_b = inverse_rotation_b.rotate_vec3(offset_b);
        let bounce = bounce(
            state,
            (a, b),
            (offset_a, offset_b),
            normal,
            point.depth,
            &contact,
            dt,
        );
        points.push(SolverPoint {
            impulse: matching_impulse(old, local_a, &mut claimed),
            ..SolverPoint::default()
        });
        frames.push(PointFrame {
            local_a,
            local_b,
            bounce,
            pushed: 0.0,
        });
    }
    if let Some(old) = old {
        spread_unclaimed(old, (&claimed, normal_a), &mut points[manifold.first..]);
    }
    manifold
}

/// The load of old points that found no new one goes to the new points that found no old one.
/// A piece on a mesh sees its points jump from one corner to the next as it settles, starting
/// those from nothing lost what held it up and the heap above sank a little every step.
fn spread_unclaimed(
    old: &CachedManifold,
    (claimed, normal_a): (&[bool; MAX_CONTACTS], Vec3),
    points: &mut [SolverPoint],
) {
    let mut left: Vec3 = old.points[..old.count]
        .iter()
        .zip(claimed)
        .filter(|(_, claimed)| !**claimed)
        .map(|(point, _)| point.impulse)
        .fold(Vec3::ZERO, |sum, impulse| sum + impulse);
    let fresh = points
        .iter()
        .filter(|point| point.impulse == Vec3::ZERO)
        .count();
    if left.x <= 0.0 || fresh == 0 {
        return;
    }
    // friction only along the same tangents, a turned normal starts it over
    if old.normal == Vec3::ZERO || old.normal.dot(normal_a) < SAME_NORMAL_COS {
        left.y = 0.0;
        left.z = 0.0;
    }
    let share = left / fresh as f32;
    for point in points
        .iter_mut()
        .filter(|point| point.impulse == Vec3::ZERO)
    {
        point.impulse = share;
    }
}

/// Each old point goes to one new point, two sharing it would double its load.
fn matching_impulse(
    old: Option<&CachedManifold>,
    local_a: Vec3,
    claimed: &mut [bool; MAX_CONTACTS],
) -> Vec3 {
    let Some(old) = old else {
        return Vec3::ZERO;
    };
    let mut best = None;
    let mut closest = MATCH_DISTANCE_SQUARED;
    for (index, point) in old.points[..old.count].iter().enumerate() {
        let distance = (point.local_a - local_a).length_squared();
        if distance < closest && !claimed[index] {
            closest = distance;
            best = Some(index);
        }
    }
    best.map_or(Vec3::ZERO, |index| {
        claimed[index] = true;
        old.points[index].impulse
    })
}
