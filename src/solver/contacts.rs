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

const RECOVERY_RATE: f32 = 0.2;
const PENETRATION_SLOP: f32 = 0.005;
const MAX_RECOVERY_SPEED: f32 = 3.0;
const MATCH_DISTANCE_SQUARED: f32 = 0.05 * 0.05;
/// A pair whose bodies moved less than this against each other since its contacts were found,
/// meters squared, keeps them rather than finding them again.
const REUSE_DISTANCE_SQUARED: f32 = 0.0002 * 0.0002;
/// The same for turning: the cosine of half of 0.1 degree.
const REUSE_HALF_TURN_COS: f32 = 0.999_999_6;

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
    pub(super) islands: Islands,
    pub(super) colors: Colors,
    pub(super) motions: Vec<Motion>,
    /// One factor per serial island, its nodes beside `colors.direct`. Shapes are kept between
    /// substeps and steps while an island joins the same bodies.
    pub(super) factors: Vec<Factor>,
    pub(super) shapes: Vec<Symbolic>,
    pub(super) direct_nodes: Vec<DirectNode>,
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

/// Each manifold's refresh reads bodies and writes only its own points, so chunks of them run as
/// jobs side by side, the same as one after another. A chunk a dispatcher skipped runs here after.
pub(super) fn refresh_manifolds(
    bodies: &Bodies,
    state: &[Integrated],
    scratch: &mut ContactScratch,
    (first, last): (bool, bool),
    (dispatch, dt): (&dyn JobDispatch, f32),
) {
    let manifolds = &mut scratch.manifolds;
    // points lie in manifold order, one range after another, so chunks split them cleanly
    let in_a_row = manifolds.first().is_none_or(|first| first.first == 0)
        && manifolds
            .windows(2)
            .all(|pair| pair[0].first + pair[0].count == pair[1].first);
    if manifolds.len() <= REFRESH_CHUNK || dispatch.parallelism() <= 1 || !in_a_row {
        for manifold in manifolds.iter_mut() {
            let range = manifold.first..manifold.first + manifold.count;
            let points = (&mut scratch.points[range.clone()], &scratch.frames[range]);
            refresh_manifold(bodies, state, manifold, points, (first, last), dt);
        }
        return;
    }
    type Chunk<'a> = (
        &'a mut [SolverManifold],
        &'a mut [SolverPoint],
        &'a [PointFrame],
        bool,
    );
    let mut chunks: Vec<Mutex<Chunk<'_>>> = Vec::new();
    let (mut rest, mut points, mut frames) = (
        manifolds.as_mut_slice(),
        scratch.points.as_mut_slice(),
        scratch.frames.as_slice(),
    );
    while !rest.is_empty() {
        let (head, tail) = rest.split_at_mut(REFRESH_CHUNK.min(rest.len()));
        let used = head
            .last()
            .map_or(0, |last| last.first + last.count - head[0].first);
        let (head_points, tail_points) = points.split_at_mut(used);
        let (head_frames, tail_frames) = frames.split_at(used);
        chunks.push(Mutex::new((head, head_points, head_frames, false)));
        (rest, points, frames) = (tail, tail_points, tail_frames);
    }
    let refresh = |chunk: &Mutex<Chunk<'_>>| {
        let mut guard = chunk.lock().unwrap_or_else(PoisonError::into_inner);
        let (manifolds, points, frames, done) = &mut *guard;
        if *done {
            return;
        }
        let base = manifolds.first().map_or(0, |first| first.first);
        for manifold in manifolds.iter_mut() {
            let range = manifold.first - base..manifold.first - base + manifold.count;
            let pair = (&mut points[range.clone()], &frames[range]);
            refresh_manifold(bodies, state, manifold, pair, (first, last), dt);
        }
        *done = true;
    };
    dispatch.run(chunks.len(), &|job| {
        if let Some(chunk) = chunks.get(job) {
            refresh(chunk);
        }
    });
    for chunk in &chunks {
        refresh(chunk);
    }
}

fn refresh_manifold(
    bodies: &Bodies,
    state: &[Integrated],
    manifold: &mut SolverManifold,
    (points, frames): (&mut [SolverPoint], &[PointFrame]),
    (first, last): (bool, bool),
    dt: f32,
) {
    let (a, b) = (&state[manifold.a], &state[manifold.b]);
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
    for (point, frame) in points.iter_mut().zip(frames) {
        let offset_a = a.rotation.rotate_vec3(frame.local_a);
        let offset_b = b.rotation.rotate_vec3(frame.local_b);
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
        // still apart it may close the gap, and only arriving can bounce. arriving in the last
        // substep would clamp the speed the next step's bounce is worked out from, so it bounces now
        let speed =
            relative_velocity(state, manifold.a, manifold.b, offset_a, offset_b).dot(normal);
        let arrives = -speed * dt >= gap;
        point.target = if gap > 0.0 && !(last && arrives && frame.bounce > 0.0) {
            -gap / dt
        } else {
            frame.bounce
        };
        point.recovery =
            (RECOVERY_RATE / dt * (-gap - PENETRATION_SLOP).max(0.0)).min(MAX_RECOVERY_SPEED);
        point.recovery_impulse = 0.0;
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
    manifold
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
