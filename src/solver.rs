use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use algebrix::{Quat, Vec3};

use crate::body::{Bodies, apply_diagonal};
use crate::collision::{Bounds, MAX_CONTACTS, MovingPairs, RawManifold, find_pairs};
use crate::direct::{DirectNode, Factor, Symbolic};
use crate::joint::{
    JOINT_IMPULSES, JointPoint, JointRow, Joints, build_rows, capture_impulses, solve_point,
    solve_point_recovery, solve_row, solve_row_recovery, warm_start_point, warm_start_row,
};
use crate::material::{Contact, Material};
use crate::world::Integrated;
use crate::{BodyHandle, BodyKind, GridPosition, JobDispatch, PhysicsError, Shape};

/// Contacts exist this far apart, so a corner lifting a hair keeps its constraint.
pub(crate) const CONTACT_MARGIN: f32 = 0.02;
/// Past this much travel in one step a pair counts as fast, and contacts reach as far as it moves.
const FAST_MOTION: f32 = 0.1;
/// Per substep. Odd on purpose, 2 and 4 left stacks jittering far more than 3.
const VELOCITY_ITERATIONS: usize = 3;
const RECOVERY_ITERATIONS: usize = 1;
const RECOVERY_RATE: f32 = 0.2;
const PENETRATION_SLOP: f32 = 0.005;
const MAX_RECOVERY_SPEED: f32 = 3.0;
const MATCH_DISTANCE_SQUARED: f32 = 0.05 * 0.05;
/// A point slipping slower than this, m/s, counts as held still and gets static friction.
const SLIP_SPEED: f32 = 0.05;
/// Constraints in one block of work. A block is the smallest piece a thread takes.
const COLOR_CHUNK: usize = if cfg!(miri) { 2 } else { 64 };
/// Below this many blocks in a substep the solve stays on the calling thread.
const PARALLEL_BLOCKS: usize = if cfg!(miri) { 2 } else { 8 };
/// Pairs in one job of contact finding. Fixed, so the output does not depend on the thread count.
const PAIR_CHUNK: usize = 128;
/// Spins before a waiting thread yields.
const SPINS: u32 = 64;
/// Colors smaller than this, next to each other, go in one block, one color after another.
/// A barrier per tiny color cost more than solving them on one thread.
const SMALL_COLOR: usize = 2 * COLOR_CHUNK;
/// A stage this many blocks wide keeps one more thread busy.
const BLOCKS_PER_WORKER: usize = if cfg!(miri) { 1 } else { 4 };
/// Constraints that find no free color among these go in one more, solved in order.
const COLORS: usize = 64;
/// Islands with fewer constraints keep the plain in order sweep, one island per job. Colors
/// carry load down a stack slower, a 16 box stack fell, so only big islands get them.
const COLORED_ISLAND: usize = if cfg!(miri) { 8 } else { 1024 };

/// The part of a body the iterations change. Positions and turns stay put during them.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Motion {
    pub(crate) linear: Vec3,
    pub(crate) angular: Vec3,
    pub(crate) recovery_linear: Vec3,
    pub(crate) recovery_angular: Vec3,
    pub(crate) touched: bool,
}

/// The two bodies of one constraint. A dynamic one is the real body, a static or kinematic one
/// is a copy, the iterations never move it.
pub(crate) struct Pair<'a> {
    pub(crate) a: &'a mut Motion,
    pub(crate) b: &'a mut Motion,
}

/// A constraint in solve order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Item {
    Point(u32),
    Row(u32),
    Manifold(u32),
}

/// Constraints put in colors, so no two in a color share a dynamic body. Colors go one after
/// another, a color can go on many threads and still give what one thread does.
#[derive(Debug, Clone, Default)]
struct Colors {
    used: Vec<u64>,
    color: Vec<u8>,
    order: Vec<Item>,
    items: Vec<Item>,
    ends: Vec<u32>,
    /// Small islands, each solved in order by one job.
    serial: Vec<Item>,
    serial_ends: Vec<u32>,
    /// Their equality joint rows, solved directly, ends beside `serial_ends`.
    direct: Vec<Item>,
    direct_ends: Vec<u32>,
    /// Equality joint rows of colored islands that have any, one direct solve each.
    colored_direct: Vec<Item>,
    colored_direct_ends: Vec<u32>,
    /// Islands grouped into blocks of about a chunk of constraints.
    runs: Vec<u32>,
    stages: Vec<Stage>,
}

/// Blocks that may run at once. Every block of a stage is done before the next one starts.
#[derive(Debug, Clone, Copy)]
enum Stage {
    Islands {
        blocks: u32,
    },
    /// The direct solve of each colored island with equality joints, one island a block.
    Direct {
        pass: Pass,
        blocks: u32,
    },
    /// Colors `first` to `last`. More than one is a single block.
    Colors {
        first: u32,
        last: u32,
        pass: Pass,
        reverse: bool,
        blocks: u32,
    },
}

impl Stage {
    fn blocks(self) -> usize {
        match self {
            Self::Islands { blocks }
            | Self::Direct { blocks, .. }
            | Self::Colors { blocks, .. } => blocks as usize,
        }
    }
}

/// Lets the jobs of one color reach their own constraints and bodies at once.
struct Shared<'a, T> {
    ptr: *mut T,
    len: usize,
    marker: PhantomData<&'a mut [T]>,
}

// SAFETY: jobs only use indexes no other job of the same color touches, see `solve_color`.
unsafe impl<T: Send> Sync for Shared<'_, T> {}

impl<'a, T: Copy> Shared<'a, T> {
    fn new(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            marker: PhantomData,
        }
    }

    /// # Safety
    /// No other job may write `index` while this runs.
    unsafe fn read(&self, index: usize) -> T {
        assert!(index < self.len);
        // SAFETY: in bounds, and the caller keeps writers of `index` out.
        unsafe { self.ptr.add(index).read() }
    }

    /// # Safety
    /// No other job may use `start..start + len` while the slice lives.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        assert!(start <= self.len && len <= self.len - start);
        // SAFETY: in bounds, and the caller gives the range to one job only.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }

    /// # Safety
    /// No other job may use `index` while the reference lives.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self, index: usize) -> &mut T {
        assert!(index < self.len);
        // SAFETY: in bounds, and the caller gives each index to one job only.
        unsafe { &mut *self.ptr.add(index) }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CachedPoint {
    /// Contact in body A axes, finds the same point next step.
    pub(crate) local_a: Vec3,
    pub(crate) impulse: Vec3,
    /// Normal impulse over the whole step, every substep's added, newton seconds. A hit is
    /// mostly taken in the first substep, the last one alone says little about it.
    pub(crate) pushed: f32,
}

/// `pushed` is what the last step did, the next step never reads it, so snapshots leave it
/// out and equal state ignores it.
impl PartialEq for CachedPoint {
    fn eq(&self, other: &Self) -> bool {
        self.local_a == other.local_a && self.impulse == other.impulse
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
}

/// `arm_a[k]` is `offset_a × axes[k]`, `turn_a[k]` the spin of A per unit impulse along it.
/// Both are worked out once a substep, the iterations only scale and add them.
#[derive(Debug, Clone, Copy, Default)]
struct SolverPoint {
    arm_a: [Vec3; 3],
    arm_b: [Vec3; 3],
    turn_a: [Vec3; 3],
    turn_b: [Vec3; 3],
    mass: Vec3,
    target: f32,
    recovery: f32,
    recovery_impulse: f32,
    impulse: Vec3,
    /// Slipping faster than `SLIP_SPEED` when the substep began, static friction no longer holds.
    sliding: bool,
}

/// What a contact point keeps that the iterations never read, apart so a pass reads less.
#[derive(Debug, Clone, Copy, Default)]
struct PointFrame {
    local_a: Vec3,
    local_b: Vec3,
    bounce: f32,
    pushed: f32,
}

#[derive(Clone, Copy)]
struct SolverManifold {
    a: usize,
    b: usize,
    handle_a: BodyHandle,
    handle_b: BodyHandle,
    part_a: u32,
    part_b: u32,
    axes: [Vec3; 3],
    friction: f32,
    /// The most a point takes before it slides, then `friction` holds it.
    static_friction: f32,
    /// B's surface speed over A's, world, from belts. Friction carries the bodies to it.
    surface: Vec3,
    /// Spin about the normal and rolling about the two tangents, resisted up to these levers
    /// times the manifold's load. Zero skips the row.
    resist: [f32; 3],
    /// Per angular row: spin of A and of B per unit impulse, its mass and what it took so far.
    resist_turn: [[Vec3; 2]; 3],
    resist_mass: [f32; 3],
    resist_impulse: [f32; 3],
    /// Zero for bodies that are not dynamic, they are never pushed.
    inverse_mass: [f32; 2],
    /// Its points are `first..first + count` in the scratch point lists.
    first: usize,
    count: usize,
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
    bounds: Vec<Bounds>,
    /// What each bound was made from, a body that did not change keeps it.
    posed: Vec<(GridPosition, Quat, Shape, f32)>,
    order: Vec<u32>,
    pairs: Vec<(u32, u32)>,
    moving: MovingPairs,
    manifolds: Vec<SolverManifold>,
    points: Vec<SolverPoint>,
    frames: Vec<PointFrame>,
    found: Vec<FoundContacts>,
    joint_points: Vec<JointPoint>,
    joint_rows: Vec<JointRow>,
    joint_impulses: Vec<[f32; JOINT_IMPULSES]>,
    islands: Islands,
    colors: Colors,
    motions: Vec<Motion>,
    /// One factor per serial island, its nodes beside `colors.direct`. Shapes are kept between
    /// substeps and steps while an island joins the same bodies.
    factors: Vec<Factor>,
    shapes: Vec<Symbolic>,
    direct_nodes: Vec<DirectNode>,
}

/// Awake dynamic bodies joined by contacts and joints. Static, kinematic and sleeping bodies
/// never join two islands. Each island's constraints sit together, in the order they had.
#[derive(Debug, Clone, Default)]
pub(crate) struct Islands {
    parent: Vec<u32>,
    /// Island of each dense body, `NONE` when it is not awake and dynamic.
    of: Vec<u32>,
    bodies: Vec<u32>,
    manifold_ends: Vec<u32>,
    point_ends: Vec<u32>,
    row_ends: Vec<u32>,
    spare_manifolds: Vec<SolverManifold>,
    spare_contacts: Vec<SolverPoint>,
    spare_frames: Vec<PointFrame>,
    spare_points: Vec<JointPoint>,
    spare_rows: Vec<JointRow>,
}

const NONE: u32 = u32::MAX;

impl Islands {
    fn find(&mut self, mut dense: usize) -> usize {
        while self.parent[dense] as usize != dense {
            let up = self.parent[dense] as usize;
            self.parent[dense] = self.parent[up];
            dense = up;
        }
        dense
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        // the lower index stays the root, so the result does not depend on link order
        if a != b {
            let (low, high) = (a.min(b), a.max(b));
            self.parent[high] = low as u32;
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.bodies.len()
    }

    /// Bodies and constraints in each island, in island order. Constraints count joint rows
    /// from the last substep.
    pub(crate) fn sizes(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.count()).map(|island| {
            let span = |ends: &[u32]| {
                let start = if island == 0 { 0 } else { ends[island - 1] };
                (ends[island] - start) as usize
            };
            let constraints =
                span(&self.manifold_ends) + span(&self.point_ends) + span(&self.row_ends);
            (self.bodies[island] as usize, constraints)
        })
    }

    fn range(ends: &[u32], island: usize) -> std::ops::Range<usize> {
        let start = if island == 0 { 0 } else { ends[island - 1] };
        start as usize..ends[island] as usize
    }
}

/// Puts items of island 0 first, then 1 and on, keeping their order inside an island.
fn group<T: Copy>(
    items: &mut [T],
    spare: &mut Vec<T>,
    count: usize,
    island: impl Fn(&T) -> u32,
    ends: &mut Vec<u32>,
) {
    ends.clear();
    ends.resize(count, 0);
    for item in items.iter() {
        ends[island(item) as usize] += 1;
    }
    let mut next = 0;
    for end in ends.iter_mut() {
        let size = *end;
        *end = next;
        next += size;
    }
    spare.clear();
    spare.extend_from_slice(items);
    for item in spare.iter() {
        let slot = &mut ends[island(item) as usize];
        items[*slot as usize] = *item;
        *slot += 1;
    }
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

/// How far the surface of a body can travel in one step, turning included.
fn step_motion(bodies: &Bodies, dense: usize, dt: f32) -> f32 {
    if bodies.kind[dense] == BodyKind::Static {
        return 0.0;
    }
    let shape = &bodies.shape[dense];
    let reach = if matches!(shape, Shape::TriMesh(_)) {
        0.0
    } else {
        bodies.assets.bounding_radius(shape)
    };
    (bodies.linear[dense].length() + bodies.angular[dense].length() * reach) * dt
}

/// Contacts of a pair exist this far apart. The bounds cover at least this much.
pub(crate) fn pair_margin(bodies: &Bodies, a: usize, b: usize, dt: f32) -> f32 {
    let motion = step_motion(bodies, a, dt) + step_motion(bodies, b, dt);
    if motion > FAST_MOTION {
        CONTACT_MARGIN + motion
    } else {
        CONTACT_MARGIN
    }
}

/// Straight travel of B against A this step. `None` when turning matters, a straight sweep
/// could not rule the contact out then.
fn straight_motion(bodies: &Bodies, a: usize, b: usize, dt: f32) -> Option<Vec3> {
    let turning = |dense: usize| {
        let shape = &bodies.shape[dense];
        if matches!(shape, Shape::TriMesh(_)) {
            0.0
        } else {
            bodies.angular[dense].length() * bodies.assets.bounding_radius(shape) * dt
        }
    };
    (turning(a) + turning(b) <= CONTACT_MARGIN).then(|| (bodies.linear[b] - bodies.linear[a]) * dt)
}

/// With `every`, all overlapping pairs. Otherwise only pairs with a moving body, the rest stay
/// still, and `add_woken_pairs` fills in what wakes.
pub(crate) fn find_broad_pairs(
    bodies: &Bodies,
    scratch: &mut ContactScratch,
    every: bool,
    dt: f32,
) {
    let count = bodies.len();
    scratch.bounds.truncate(count);
    scratch.posed.truncate(count);
    for dense in 0..count {
        let pose = (
            bodies.position(dense),
            bodies.rotation[dense],
            bodies.shape[dense],
            CONTACT_MARGIN + 2.0 * step_motion(bodies, dense, dt),
        );
        if scratch.posed.get(dense) == Some(&pose) {
            continue;
        }
        let bound = bodies.assets.bounds(&pose.2, pose.0, pose.1, pose.3);
        if dense < scratch.posed.len() {
            scratch.bounds[dense] = bound;
            scratch.posed[dense] = pose;
        } else {
            scratch.bounds.push(bound);
            scratch.posed.push(pose);
        }
    }
    if every {
        find_pairs(&scratch.bounds, &mut scratch.order, &mut scratch.pairs);
    } else {
        let moving = |dense: usize| bodies.kind[dense] != BodyKind::Static && !bodies.asleep[dense];
        scratch
            .moving
            .find(&scratch.bounds, moving, &mut scratch.pairs);
    }
}

/// Pairs for sleepers that `awake` wakes this step.
pub(crate) fn add_woken_pairs(bodies: &Bodies, awake: &[bool], scratch: &mut ContactScratch) {
    let woken = (0..bodies.len()).filter(|dense| bodies.asleep[*dense] && awake[*dense]);
    scratch
        .moving
        .add_woken(&scratch.bounds, woken, &mut scratch.pairs);
}

/// Finds contacts once per step and loads the working joint impulses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_contacts(
    bodies: &Bodies,
    awake: &[bool],
    joints: &Joints,
    state: &[Integrated],
    cache: &[CachedManifold],
    scratch: &mut ContactScratch,
    dispatch: &dyn JobDispatch,
    dt: f32,
) -> Result<(), PhysicsError> {
    build_manifolds(bodies, awake, joints, state, cache, scratch, dispatch, dt)?;
    find_islands(bodies, awake, joints, scratch);
    scratch.joint_impulses.clear();
    scratch
        .joint_impulses
        .extend(joints.slots.iter().map(|slot| {
            slot.state
                .map_or([0.0; JOINT_IMPULSES], |live| live.impulses)
        }));
    Ok(())
}

fn find_islands(bodies: &Bodies, awake: &[bool], joints: &Joints, scratch: &mut ContactScratch) {
    let active = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic && awake[dense];
    let islands = &mut scratch.islands;
    let count = bodies.len();
    islands.parent.clear();
    islands.parent.extend(0..count as u32);
    for manifold in &scratch.manifolds {
        if active(manifold.a) && active(manifold.b) {
            islands.union(manifold.a, manifold.b);
        }
    }
    for (_, state) in joints.live() {
        let (Ok(a), Ok(b)) = (
            bodies.dense_index(state.desc.a),
            bodies.dense_index(state.desc.b),
        ) else {
            continue;
        };
        if active(a) && active(b) {
            islands.union(a, b);
        }
    }
    islands.of.clear();
    islands.of.resize(count, NONE);
    islands.bodies.clear();
    for dense in 0..count {
        if !active(dense) {
            continue;
        }
        let root = islands.find(dense);
        if islands.of[root] == NONE {
            islands.of[root] = islands.bodies.len() as u32;
            islands.bodies.push(0);
        }
        let island = islands.of[root];
        islands.of[dense] = island;
        islands.bodies[island as usize] += 1;
    }
    let islands = &mut scratch.islands;
    let of = &islands.of;
    group(
        &mut scratch.manifolds,
        &mut islands.spare_manifolds,
        islands.bodies.len(),
        |manifold| {
            if of[manifold.a] != NONE {
                of[manifold.a]
            } else {
                of[manifold.b]
            }
        },
        &mut islands.manifold_ends,
    );
    order_big_islands(bodies, scratch);
}

/// Islands that will surely be colored get their contacts in color order, and the points
/// rewritten to match, so a pass reads them front to back. Joints are left out of these
/// colors, the substep coloring adds them. Smaller islands keep their order and results.
fn order_big_islands(bodies: &Bodies, scratch: &mut ContactScratch) {
    let dynamic = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic;
    let islands = &mut scratch.islands;
    let used = &mut scratch.colors.used;
    used.clear();
    used.resize(bodies.len(), 0);
    let mut moved = false;
    for island in 0..islands.count() {
        let range = Islands::range(&islands.manifold_ends, island);
        if range.len() < COLORED_ISLAND {
            continue;
        }
        moved = true;
        let manifolds = &mut scratch.manifolds[range];
        let mut keys: Vec<(u32, u32)> = Vec::with_capacity(manifolds.len());
        for (index, manifold) in manifolds.iter().enumerate() {
            let mask = |dense: usize| if dynamic(dense) { used[dense] } else { 0 };
            let color = (!(mask(manifold.a) | mask(manifold.b)))
                .trailing_zeros()
                .min(COLORS as u32);
            if (color as usize) < COLORS {
                for dense in [manifold.a, manifold.b] {
                    if dynamic(dense) {
                        used[dense] |= 1 << color;
                    }
                }
            }
            keys.push((color, index as u32));
        }
        keys.sort_unstable();
        islands.spare_manifolds.clear();
        islands.spare_manifolds.extend_from_slice(manifolds);
        for (slot, (_, from)) in manifolds.iter_mut().zip(&keys) {
            *slot = islands.spare_manifolds[*from as usize];
        }
    }
    if !moved {
        return;
    }
    let (points, frames) = (
        &mut scratch.islands.spare_contacts,
        &mut scratch.islands.spare_frames,
    );
    points.clear();
    frames.clear();
    for manifold in &mut scratch.manifolds {
        let range = manifold.first..manifold.first + manifold.count;
        manifold.first = points.len();
        points.extend_from_slice(&scratch.points[range.clone()]);
        frames.extend_from_slice(&scratch.frames[range]);
    }
    std::mem::swap(&mut scratch.points, points);
    std::mem::swap(&mut scratch.frames, frames);
}

/// One substep of velocities. Joints rebuild from the moved bodies, contacts refresh their gaps.
pub(crate) fn solve_substep(
    bodies: &Bodies,
    awake: &[bool],
    joints: &Joints,
    state: &mut [Integrated],
    scratch: &mut ContactScratch,
    (substep, last): (u32, bool),
    (dispatch, dt): (&dyn JobDispatch, f32),
) {
    build_rows(
        joints,
        &scratch.joint_impulses,
        bodies,
        state,
        awake,
        &mut scratch.joint_points,
        &mut scratch.joint_rows,
        dt,
    );
    let islands = &mut scratch.islands;
    let count = islands.count();
    let of = &islands.of;
    let pair = |(a, b): (usize, usize)| if of[a] != NONE { of[a] } else { of[b] };
    group(
        &mut scratch.joint_points,
        &mut islands.spare_points,
        count,
        |point| pair(point.bodies()),
        &mut islands.point_ends,
    );
    group(
        &mut scratch.joint_rows,
        &mut islands.spare_rows,
        count,
        |row| pair(row.bodies()),
        &mut islands.row_ends,
    );
    for manifold in &mut scratch.manifolds {
        let range = manifold.first..manifold.first + manifold.count;
        let points = (&mut scratch.points[range.clone()], &scratch.frames[range]);
        refresh_manifold(bodies, state, manifold, points, last, dt);
    }
    color_constraints(bodies, scratch);
    factor_islands(bodies, scratch);
    scratch.motions.clear();
    scratch.motions.extend(state.iter().map(|body| Motion {
        linear: body.linear,
        angular: body.angular,
        recovery_linear: body.recovery_linear,
        recovery_angular: body.recovery_angular,
        touched: body.touched,
    }));
    let shared = (
        Shared::new(&mut scratch.motions),
        Shared::new(&mut scratch.joint_points),
        Shared::new(&mut scratch.joint_rows),
        Shared::new(&mut scratch.manifolds),
        Shared::new(&mut scratch.points),
    );
    plan_stages(&mut scratch.colors, substep);
    let colors = &scratch.colors;
    let direct = (
        scratch.factors.as_slice(),
        scratch.shapes.as_slice(),
        scratch.direct_nodes.as_slice(),
    );
    run_stages(&colors.stages, dispatch, &|stage, block| {
        solve_block(bodies, colors, direct, &shared, stage, block, substep);
    });
    for (body, motion) in state.iter_mut().zip(&scratch.motions) {
        body.linear = motion.linear;
        body.angular = motion.angular;
        body.recovery_linear = motion.recovery_linear;
        body.recovery_angular = motion.recovery_angular;
        body.touched = motion.touched;
    }
    capture_impulses(
        &scratch.joint_points,
        &scratch.joint_rows,
        &mut scratch.joint_impulses,
    );
    for (frame, point) in scratch.frames.iter_mut().zip(&scratch.points) {
        frame.pushed += point.impulse.x.max(0.0);
    }
}

impl Colors {
    fn count(&self) -> usize {
        self.ends.len()
    }

    fn of(&self, color: usize) -> &[Item] {
        let start = if color == 0 { 0 } else { self.ends[color - 1] };
        &self.items[start as usize..self.ends[color] as usize]
    }
}

/// Greedy, in island order: each constraint takes the lowest color neither of its dynamic
/// bodies has. Static and kinematic bodies are only read, any number share them.
fn color_constraints(bodies: &Bodies, scratch: &mut ContactScratch) {
    let dynamic = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic;
    let colors = &mut scratch.colors;
    colors.used.clear();
    colors.used.resize(bodies.len(), 0);
    colors.color.clear();
    colors.items.clear();
    colors.order.clear();
    colors.serial.clear();
    colors.serial_ends.clear();
    colors.direct.clear();
    colors.direct_ends.clear();
    colors.colored_direct.clear();
    colors.colored_direct_ends.clear();
    let islands = &scratch.islands;
    for island in 0..islands.count() {
        let points = Islands::range(&islands.point_ends, island);
        let rows = Islands::range(&islands.row_ends, island);
        let manifolds = Islands::range(&islands.manifold_ends, island);
        let size = points.len() + rows.len() + manifolds.len();
        if size < COLORED_ISLAND {
            // equality joint rows go to the direct solve, the rest stay in order
            colors
                .direct
                .extend(points.map(|index| Item::Point(index as u32)));
            for index in rows {
                let item = Item::Row(index as u32);
                if scratch.joint_rows[index].is_equality() {
                    colors.direct.push(item);
                } else {
                    colors.serial.push(item);
                }
            }
            colors
                .serial
                .extend(manifolds.map(|index| Item::Manifold(index as u32)));
            colors.serial_ends.push(colors.serial.len() as u32);
            colors.direct_ends.push(colors.direct.len() as u32);
            continue;
        }
        // the same split for a colored island, its joints get a direct stage of their own
        let held = colors.colored_direct.len();
        colors
            .colored_direct
            .extend(points.map(|index| Item::Point(index as u32)));
        for index in rows {
            let item = Item::Row(index as u32);
            if scratch.joint_rows[index].is_equality() {
                colors.colored_direct.push(item);
            } else {
                colors.order.push(item);
            }
        }
        if colors.colored_direct.len() > held {
            colors
                .colored_direct_ends
                .push(colors.colored_direct.len() as u32);
        }
        colors
            .order
            .extend(manifolds.map(|index| Item::Manifold(index as u32)));
    }
    colors.runs.clear();
    colors.runs.push(0);
    let mut start = 0;
    for (index, end) in colors.serial_ends.iter().enumerate() {
        if (end - start) as usize >= COLOR_CHUNK {
            colors.runs.push(index as u32 + 1);
            start = *end;
        }
    }
    if *colors.runs.last().unwrap_or(&0) as usize != colors.serial_ends.len() {
        colors.runs.push(colors.serial_ends.len() as u32);
    }
    let order = std::mem::take(&mut colors.order);
    let mut counts = [0_u32; COLORS + 1];
    for item in &order {
        let (a, b) = match *item {
            Item::Point(index) => scratch.joint_points[index as usize].bodies(),
            Item::Row(index) => scratch.joint_rows[index as usize].bodies(),
            Item::Manifold(index) => {
                let manifold = &scratch.manifolds[index as usize];
                (manifold.a, manifold.b)
            }
        };
        let mask = |dense: usize| {
            if dynamic(dense) {
                colors.used[dense]
            } else {
                0
            }
        };
        let taken = mask(a) | mask(b);
        let color = (!taken).trailing_zeros() as usize;
        if color < COLORS {
            let bit = 1_u64 << color;
            for dense in [a, b] {
                if dynamic(dense) {
                    colors.used[dense] |= bit;
                }
            }
        }
        let color = color.min(COLORS);
        colors.color.push(color as u8);
        counts[color] += 1;
    }
    let last = counts
        .iter()
        .rposition(|count| *count > 0)
        .map_or(0, |at| at + 1);
    colors.ends.clear();
    let mut next = 0;
    for count in &counts[..last] {
        next += count;
        colors.ends.push(next);
    }
    let mut slot: Vec<u32> = std::iter::once(0)
        .chain(colors.ends.iter().copied())
        .take(last)
        .collect();
    colors.items.resize(order.len(), Item::Point(0));
    for (item, color) in order.iter().zip(&colors.color) {
        let at = &mut slot[*color as usize];
        colors.items[*at as usize] = *item;
        *at += 1;
    }
    colors.order = order;
}

/// The direct solve's nodes and factor for each serial island's equality rows.
fn factor_islands(bodies: &Bodies, scratch: &mut ContactScratch) {
    let dynamic = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic;
    let colors = &scratch.colors;
    scratch.direct_nodes.clear();
    for item in colors.direct.iter().chain(&colors.colored_direct) {
        scratch.direct_nodes.push(match *item {
            Item::Point(index) => scratch.joint_points[index as usize].direct_node(),
            Item::Row(index) => scratch.joint_rows[index as usize].direct_node(),
            Item::Manifold(_) => continue,
        });
    }
    // serial islands first, then colored ones, their nodes after the serial ones
    let serial = colors.direct.len() as u32;
    let ends: Vec<u32> = colors
        .direct_ends
        .iter()
        .copied()
        .chain(colors.colored_direct_ends.iter().map(|end| end + serial))
        .collect();
    let islands = ends.len();
    scratch.factors.resize_with(islands, Factor::default);
    scratch.shapes.resize_with(islands, Symbolic::default);
    let mut start = 0;
    for (island, end) in ends.iter().enumerate() {
        let nodes = &scratch.direct_nodes[start..*end as usize];
        start = *end as usize;
        if nodes.is_empty() {
            continue;
        }
        let shape = &mut scratch.shapes[island];
        if !shape.fits(nodes, &dynamic) {
            *shape = Symbolic::new(nodes, &dynamic);
        }
        scratch.factors[island].refactor(shape, nodes);
    }
}

/// One direct solve of an island's equality rows: the impulses that make every row's speed
/// what it asks, given every other impulse so far.
///
/// # Safety
/// The caller owns the island's constraints and dynamic bodies for this call.
unsafe fn solve_direct(
    bodies: &Bodies,
    items: &[Item],
    nodes: &[DirectNode],
    (factor, shape): (&Factor, &Symbolic),
    shared: &SharedParts<'_>,
    recovery: bool,
    x: &mut Vec<[f64; 3]>,
) {
    if nodes.is_empty() {
        return;
    }
    let (states, points, rows, _, _) = shared;
    let dynamic = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic;
    let speeds = |motion: &Motion| {
        if recovery {
            (motion.recovery_linear, motion.recovery_angular)
        } else {
            (motion.linear, motion.angular)
        }
    };
    x.clear();
    x.reserve(nodes.len());
    for node in nodes {
        // SAFETY: the island's bodies are this job's, others are only read.
        let (a, b) = unsafe { (states.read(node.a), states.read(node.b)) };
        let ((va, wa), (vb, wb)) = (speeds(&a), speeds(&b));
        let mut part = [0.0; 3];
        for (value, row) in part.iter_mut().zip(&node.rows[..node.count]) {
            let speed =
                row.lin_a.dot(va) + row.ang_a.dot(wa) + row.lin_b.dot(vb) + row.ang_b.dot(wb);
            let goal = if recovery { row.recovery } else { row.target };
            *value = f64::from(goal) - f64::from(speed);
        }
        x.push(part);
    }
    factor.solve(shape, x);
    for ((node, item), part) in nodes.iter().zip(items).zip(x.iter()) {
        let impulse = &part[..node.count];
        for (dense, side_a) in [(node.a, true), (node.b, false)] {
            if !dynamic(dense) {
                continue;
            }
            // SAFETY: as above, a dynamic body of the island.
            let motion = unsafe { states.get(dense) };
            let (mut linear, mut angular) = (Vec3::ZERO, Vec3::ZERO);
            for (row, value) in node.rows.iter().zip(impulse) {
                let value = *value as f32;
                let (move_linear, move_angular) = if side_a {
                    (row.move_lin_a, row.move_ang_a)
                } else {
                    (row.move_lin_b, row.move_ang_b)
                };
                linear += move_linear * value;
                angular += move_angular * value;
            }
            if recovery {
                motion.recovery_linear += linear;
                motion.recovery_angular += angular;
            } else {
                motion.linear += linear;
                motion.angular += angular;
                motion.touched = true;
            }
        }
        match *item {
            // SAFETY: the island's constraints are this job's.
            Item::Point(index) => unsafe { points.get(index as usize) }.add_direct(
                Vec3::new(impulse[0] as f32, impulse[1] as f32, impulse[2] as f32),
                recovery,
            ),
            Item::Row(index) => {
                unsafe { rows.get(index as usize) }.add_direct(impulse[0] as f32, recovery);
            }
            Item::Manifold(_) => {}
        }
    }
}

/// Small islands in one stage, then each pass of the colored solve, color by color.
fn plan_stages(colors: &mut Colors, substep: u32) {
    let mut stages = std::mem::take(&mut colors.stages);
    stages.clear();
    if colors.runs.len() > 1 {
        stages.push(Stage::Islands {
            blocks: colors.runs.len() as u32 - 1,
        });
    }
    // runs of colors: a big one alone, small neighbours together in one block
    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    for color in 0..colors.count() {
        let size = colors.of(color).len();
        let small = color == COLORS || size < SMALL_COLOR;
        match groups.last_mut() {
            Some((_, last, blocks)) if small && *blocks == 1 && *last + 1 == color => {
                *last = color;
            }
            _ if size == 0 => {}
            _ => {
                let blocks = if small { 1 } else { size.div_ceil(COLOR_CHUNK) };
                groups.push((color, color, blocks));
            }
        }
    }
    let direct = colors.colored_direct_ends.len() as u32;
    let mut sweep = |pass, reverse: bool| {
        let mut push = |&(first, last, blocks): &(usize, usize, usize)| {
            stages.push(Stage::Colors {
                first: first as u32,
                last: last as u32,
                pass,
                reverse,
                blocks: blocks as u32,
            });
        };
        if reverse {
            groups.iter().rev().for_each(&mut push);
        } else {
            groups.iter().for_each(&mut push);
        }
        // the joints after the rest, so they have the last word
        if direct > 0 {
            stages.push(Stage::Direct {
                pass,
                blocks: direct,
            });
        }
    };
    sweep(Pass::WarmStart, false);
    // sweeping both ways keeps a stack from favoring one end, one way it rocks and never settles
    for iteration in 0..VELOCITY_ITERATIONS {
        sweep(
            Pass::Velocity,
            !(substep as usize + iteration).is_multiple_of(2),
        );
    }
    for _ in 0..RECOVERY_ITERATIONS {
        sweep(Pass::Recovery, false);
    }
    colors.stages = stages;
}

/// One block of a stage. Islands go start to end: warm start, sweeps both ways, recovery.
fn solve_block(
    bodies: &Bodies,
    colors: &Colors,
    (factors, shapes, nodes): (&[Factor], &[Symbolic], &[DirectNode]),
    shared: &SharedParts<'_>,
    stage: Stage,
    block: usize,
    substep: u32,
) {
    // SAFETY: a block owns its constraints and their dynamic bodies for its stage. Islands share
    // no dynamic body, and inside one color no two constraints do. The overflow color is one
    // block. Static and kinematic bodies are only read, nobody writes them during the solve.
    let go = |item: &Item, pass| unsafe { solve_item(bodies, *item, shared, pass) };
    // right hand sides of direct solves, one buffer for the block
    let mut rhs = Vec::new();
    match stage {
        Stage::Islands { .. } => {
            let (first, last) = (colors.runs[block] as usize, colors.runs[block + 1] as usize);
            for index in first..last {
                let start = if index == 0 {
                    0
                } else {
                    colors.serial_ends[index - 1]
                };
                let items = &colors.serial[start as usize..colors.serial_ends[index] as usize];
                let from = if index == 0 {
                    0
                } else {
                    colors.direct_ends[index - 1] as usize
                };
                let to = colors.direct_ends[index] as usize;
                let exact = &colors.direct[from..to];
                let factor = (&factors[index], &shapes[index]);
                // SAFETY: as above, the island is this block's.
                let mut direct = |recovery| unsafe {
                    solve_direct(
                        bodies,
                        exact,
                        &nodes[from..to],
                        factor,
                        shared,
                        recovery,
                        &mut rhs,
                    );
                };
                exact.iter().for_each(|item| go(item, Pass::WarmStart));
                items.iter().for_each(|item| go(item, Pass::WarmStart));
                // the rest by sweeps, then the joints exactly, so they have the last word
                // repeated direct solves refine each other, the factor drops redundant
                // directions and so is not quite exact
                for iteration in 0..VELOCITY_ITERATIONS {
                    if (substep as usize + iteration).is_multiple_of(2) {
                        items.iter().for_each(|item| go(item, Pass::Velocity));
                    } else {
                        items.iter().rev().for_each(|item| go(item, Pass::Velocity));
                    }
                    direct(false);
                }
                for _ in 0..RECOVERY_ITERATIONS {
                    items.iter().for_each(|item| go(item, Pass::Recovery));
                    direct(true);
                }
            }
        }
        Stage::Direct { pass, .. } => {
            let ends = &colors.colored_direct_ends;
            let from = if block == 0 {
                0
            } else {
                ends[block - 1] as usize
            };
            let to = ends[block] as usize;
            let items = &colors.colored_direct[from..to];
            // colored islands' nodes and factors come after the serial ones
            let (offset, island) = (colors.direct.len(), colors.direct_ends.len() + block);
            let nodes = &nodes[offset + from..offset + to];
            let factor = (&factors[island], &shapes[island]);
            match pass {
                Pass::WarmStart => items.iter().for_each(|item| go(item, pass)),
                // SAFETY: as above, a block owns its island.
                Pass::Velocity => unsafe {
                    solve_direct(bodies, items, nodes, factor, shared, false, &mut rhs);
                },
                Pass::Recovery => unsafe {
                    solve_direct(bodies, items, nodes, factor, shared, true, &mut rhs);
                },
            }
        }
        Stage::Colors {
            first,
            last,
            pass,
            reverse,
            ..
        } => {
            let run = |items: &[Item]| {
                if reverse {
                    items.iter().rev().for_each(|item| go(item, pass));
                } else {
                    items.iter().for_each(|item| go(item, pass));
                }
            };
            if first != last {
                let range = first as usize..=last as usize;
                if reverse {
                    range.rev().for_each(|color| run(colors.of(color)));
                } else {
                    range.for_each(|color| run(colors.of(color)));
                }
                return;
            }
            let items = colors.of(first as usize);
            if first as usize == COLORS {
                run(items);
            } else {
                let start = block * COLOR_CHUNK;
                run(&items[start..(start + COLOR_CHUNK).min(items.len())]);
            }
        }
    }
}

/// Runs every block of every stage, in stage order. One dispatch for all of them: job 0 takes
/// blocks too and moves on to the next stage once every block of this one is done, the other
/// jobs take blocks while there are some. Job 0 never waits for another job to start, so a
/// dispatcher that runs jobs one after another still finishes, job 0 just does all of it.
fn run_stages(stages: &[Stage], dispatch: &dyn JobDispatch, run: &(dyn Fn(Stage, usize) + Sync)) {
    let total: usize = stages.iter().map(|stage| stage.blocks()).sum();
    let widest = stages.iter().map(|stage| stage.blocks()).max().unwrap_or(0);
    let workers = dispatch
        .parallelism()
        .min(widest.div_ceil(BLOCKS_PER_WORKER));
    if workers < 2 || total < PARALLEL_BLOCKS {
        for stage in stages {
            (0..stage.blocks()).for_each(|block| run(*stage, block));
        }
        return;
    }
    let current = AtomicUsize::new(0);
    let claimed: Vec<AtomicU32> = stages.iter().map(|_| AtomicU32::new(0)).collect();
    let done: Vec<AtomicU32> = stages.iter().map(|_| AtomicU32::new(0)).collect();
    let failure: Mutex<Option<Box<dyn Any + Send>>> = Mutex::new(None);
    // a block that panics still counts as done, so nobody waits on it forever
    let take = |index: usize| -> bool {
        let stage = stages[index];
        let block = claimed[index].fetch_add(1, Ordering::Relaxed) as usize;
        if block >= stage.blocks() {
            return false;
        }
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| run(stage, block))) {
            let mut slot = failure.lock().unwrap_or_else(|poison| poison.into_inner());
            slot.get_or_insert(panic);
        }
        done[index].fetch_add(1, Ordering::Release);
        true
    };
    dispatch.run(workers, &|job| {
        if job == 0 {
            for (index, stage) in stages.iter().enumerate() {
                while take(index) {}
                let mut spins = 0;
                while (done[index].load(Ordering::Acquire) as usize) < stage.blocks() {
                    wait(&mut spins);
                }
                current.store(index + 1, Ordering::Release);
            }
            return;
        }
        let mut index = current.load(Ordering::Acquire);
        let mut spins = 0;
        while index < stages.len() {
            if take(index) {
                spins = 0;
                continue;
            }
            wait(&mut spins);
            index = current.load(Ordering::Acquire);
        }
    });
    if let Some(panic) = failure
        .into_inner()
        .unwrap_or_else(|poison| poison.into_inner())
    {
        resume_unwind(panic);
    }
}

fn wait(spins: &mut u32) {
    if *spins < SPINS {
        *spins += 1;
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pass {
    WarmStart,
    Velocity,
    Recovery,
}

type SharedParts<'a> = (
    Shared<'a, Motion>,
    Shared<'a, JointPoint>,
    Shared<'a, JointRow>,
    Shared<'a, SolverManifold>,
    Shared<'a, SolverPoint>,
);

/// # Safety
/// No other running job may use this constraint or its dynamic bodies.
unsafe fn solve_item(bodies: &Bodies, item: Item, shared: &SharedParts<'_>, pass: Pass) {
    let (states, points, rows, manifolds, contacts) = shared;
    let (a, b) = match item {
        // SAFETY: the caller gives this constraint to this job only.
        Item::Point(index) => unsafe { points.get(index as usize) }.bodies(),
        Item::Row(index) => unsafe { rows.get(index as usize) }.bodies(),
        Item::Manifold(index) => {
            let manifold = unsafe { manifolds.get(index as usize) };
            (manifold.a, manifold.b)
        }
    };
    let dynamic = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic;
    // SAFETY: a dynamic body of this constraint belongs to this job alone. Others are only
    // read, nobody writes them during the solve.
    let (mut copy_a, mut copy_b) = unsafe { (states.read(a), states.read(b)) };
    let mut pair = Pair {
        a: if dynamic(a) {
            unsafe { states.get(a) }
        } else {
            &mut copy_a
        },
        b: if dynamic(b) {
            unsafe { states.get(b) }
        } else {
            &mut copy_b
        },
    };
    match item {
        Item::Point(index) => {
            // SAFETY: as above.
            let point = unsafe { points.get(index as usize) };
            match pass {
                Pass::WarmStart => warm_start_point(&mut pair, point),
                Pass::Velocity => solve_point(&mut pair, point),
                Pass::Recovery => solve_point_recovery(&mut pair, point),
            }
        }
        Item::Row(index) => {
            let row = unsafe { rows.get(index as usize) };
            match pass {
                Pass::WarmStart => warm_start_row(&mut pair, row),
                Pass::Velocity => solve_row(&mut pair, row),
                Pass::Recovery => solve_row_recovery(&mut pair, row),
            }
        }
        Item::Manifold(index) => {
            let manifold = unsafe { manifolds.get(index as usize) };
            // SAFETY: a manifold's points are its own, and the manifold is this job's.
            let points = unsafe { contacts.slice(manifold.first, manifold.count) };
            match pass {
                Pass::WarmStart => warm_start(&mut pair, manifold, points),
                Pass::Velocity => solve_manifold(&mut pair, manifold, points),
                Pass::Recovery => solve_recovery(&mut pair, manifold, points),
            }
        }
    }
}

/// Bodies moved since the contacts were found, so gaps and arms follow them.
fn refresh_manifold(
    bodies: &Bodies,
    state: &[Integrated],
    manifold: &mut SolverManifold,
    (points, frames): (&mut [SolverPoint], &[PointFrame]),
    last: bool,
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
        if manifold.resist[k] <= 0.0 {
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
        let mut mass = [0.0; 3];
        for (k, axis) in manifold.axes.iter().enumerate() {
            point.arm_a[k] = offset_a.cross(*axis);
            point.arm_b[k] = offset_b.cross(*axis);
            point.turn_a[k] = response(manifold.a, a.rotation, point.arm_a[k]);
            point.turn_b[k] = response(manifold.b, b.rotation, point.arm_b[k]);
            let inverse = inverse_a
                + inverse_b
                + point.arm_a[k].dot(point.turn_a[k])
                + point.arm_b[k].dot(point.turn_b[k]);
            mass[k] = if inverse > 1e-12 {
                inverse.recip()
            } else {
                0.0
            };
        }
        point.mass = Vec3::new(mass[0], mass[1], mass[2]);
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
pub(crate) fn finish_contacts(
    bodies: &Bodies,
    awake: &[bool],
    cache: &mut Vec<CachedManifold>,
    scratch: &ContactScratch,
) {
    let active = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic && awake[dense];
    cache.retain(
        |entry| match (bodies.dense_index(entry.a), bodies.dense_index(entry.b)) {
            (Ok(a), Ok(b)) => !active(a) && !active(b),
            _ => false,
        },
    );
    for manifold in &scratch.manifolds {
        let mut points = [CachedPoint::default(); MAX_CONTACTS];
        let range = manifold.first..manifold.first + manifold.count;
        let solved = scratch.points[range.clone()]
            .iter()
            .zip(&scratch.frames[range]);
        for (out, (point, frame)) in points.iter_mut().zip(solved) {
            *out = CachedPoint {
                local_a: frame.local_a,
                impulse: point.impulse,
                pushed: frame.pushed,
            };
        }
        cache.push(CachedManifold {
            a: manifold.handle_a,
            b: manifold.handle_b,
            part_a: manifold.part_a,
            part_b: manifold.part_b,
            points,
            count: manifold.count,
        });
    }
    cache.sort_unstable_by_key(|entry| (entry.a, entry.b, entry.part_a, entry.part_b));
}

#[allow(clippy::too_many_arguments)]
fn build_manifolds(
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
    let tangent = tangent_of(normal);
    let axes = [normal, tangent, normal.cross(tangent)];
    let contact = Contact::of(
        &part_material(bodies, a, part_a),
        &part_material(bodies, b, part_b),
    );
    let manifold = SolverManifold {
        a,
        b,
        handle_a,
        handle_b,
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
        first: points.len(),
        count: raw.count,
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
        let approach = relative_velocity(state, a, b, offset_a, offset_b).dot(normal);
        let reaches = -approach * dt >= -point.depth;
        let bounce = if reaches && approach < -contact.bounce_threshold {
            -contact.restitution * approach
        } else {
            0.0
        };
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

/// The material a contact on this part of the body touches: a compound part's own when it has
/// one, the body's otherwise.
pub(crate) fn part_material(bodies: &Bodies, dense: usize, part: u32) -> Material {
    let own = bodies.material[dense];
    let Shape::Compound(id) = bodies.shape[dense] else {
        return own;
    };
    bodies
        .assets
        .compound(id)
        .and_then(|compound| compound.children.get(part as usize))
        .and_then(|child| child.surface)
        .unwrap_or(own)
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

fn warm_start(pair: &mut Pair, manifold: &SolverManifold, points: &mut [SolverPoint]) {
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

fn solve_manifold(pair: &mut Pair, manifold: &mut SolverManifold, points: &mut [SolverPoint]) {
    let (axes, inverse) = (&manifold.axes, manifold.inverse_mass);
    let (friction, held) = (manifold.friction, manifold.static_friction);
    for point in points.iter_mut() {
        // friction first, so the normal impulse has the last word on penetration
        let (limit, hold) = (
            (friction * point.impulse.x).max(0.0),
            (held * point.impulse.x).max(0.0),
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
        let speed = speed_along(pair, axes, point, 0);
        let next = (point.impulse.x + (point.target - speed) * point.mass.x).max(0.0);
        let change = next - point.impulse.x;
        point.impulse.x = next;
        push(pair, (axes, inverse), point, 0, change);
    }
    let load: f32 = points.iter().map(|point| point.impulse.x).sum();
    resist(pair, manifold, load);
}

/// Split impulse. Pushes overlapping bodies apart through position only, so it adds no energy.
fn solve_recovery(pair: &mut Pair, manifold: &SolverManifold, points: &mut [SolverPoint]) {
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

fn dynamic_inverse_mass(bodies: &Bodies, dense: usize) -> f32 {
    if bodies.kind[dense] == BodyKind::Dynamic {
        bodies.inverse_mass[dense]
    } else {
        0.0
    }
}

fn relative_velocity(
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

fn tangent_of(normal: Vec3) -> Vec3 {
    let helper = if normal.x.abs() < 0.57 {
        Vec3::X
    } else {
        Vec3::Y
    };
    normal.cross(helper).normalize()
}

#[cfg(test)]
mod tests {
    use algebrix::IVec3;

    use super::{Islands, group};

    use crate::{
        BodyHandle, CollisionFilter, FrameDesc, FrameId, GridPosition, JobDispatch, Material,
        PhysicsSettings, PhysicsWorld, Quat, RigidBodyDesc, Sequential, Shape, Vec3,
    };

    /// `threads` workers taking jobs off one counter.
    struct Workers(usize);

    impl JobDispatch for Workers {
        fn parallelism(&self) -> usize {
            self.0
        }

        fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..self.0 {
                    scope.spawn(|| {
                        loop {
                            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if index >= jobs {
                                break;
                            }
                            job(index);
                        }
                    });
                }
            });
        }
    }

    /// A pyramid with `base` boxes on the bottom row, all one island.
    fn pyramid(base: usize) -> (PhysicsWorld, BodyHandle, Vec3) {
        let mut world = world();
        ground(&mut world);
        let mut top = None;
        for row in 0..base {
            for column in 0..base - row {
                let x = column as f32 * 1.05 - (base - row) as f32 * 0.525;
                let at = Vec3::new(x, 0.5 + row as f32 * 1.0, 0.0);
                let body = world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5))).local_position(at),
                    )
                    .unwrap();
                top = Some((body, at));
            }
        }
        let (body, at) = top.unwrap();
        (world, body, at)
    }

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

    /// Top surface at y = 0.
    fn ground(world: &mut PhysicsWorld) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(500.0, 1.0, 500.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap()
    }

    fn run(world: &mut PhysicsWorld, seconds: f32) {
        let steps = (seconds / world.settings().fixed_timestep) as usize;
        for _ in 0..steps {
            world.step_fixed().unwrap();
        }
    }

    fn y(world: &PhysicsWorld, body: BodyHandle) -> f32 {
        world.body_state(body).unwrap().position.local.y
    }

    #[test]
    fn grouping_keeps_order_inside_each_island() {
        let mut items = vec![(1, 'a'), (0, 'b'), (2, 'c'), (0, 'd'), (1, 'e')];
        let (mut spare, mut ends) = (Vec::new(), Vec::new());
        group(&mut items, &mut spare, 3, |item| item.0, &mut ends);
        let order: String = items.iter().map(|item| item.1).collect();
        assert_eq!(order, "bdaec");
        assert_eq!(ends, [2, 4, 5]);
        assert_eq!(Islands::range(&ends, 1), 2..4);
    }

    #[test]
    fn gap_inside_the_margin_does_not_stop_a_slow_fall_early() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .local_position(Vec3::new(0.0, 0.51, 0.0)),
            )
            .unwrap();
        world.step_fixed().unwrap();
        // one step of gravity still fits in the gap, so it falls freely
        let dt = world.settings().fixed_timestep;
        let speed = world.body_state(ball).unwrap().linear_velocity.y;
        assert!((speed + 9.81 * dt).abs() < 1e-4, "{speed}");
        run(&mut world, 1.0);
        assert!((y(&world, ball) - 0.5).abs() < 0.01);
    }

    #[test]
    fn fast_body_in_the_margin_is_stopped_and_bounces_once_it_arrives() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .material(Material {
                        restitution: 1.0,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(0.0, 0.51, 0.0))
                    .linear_velocity(Vec3::new(0.0, -10.0, 0.0)),
            )
            .unwrap();
        world.step_fixed().unwrap();
        let state = world.body_state(ball).unwrap();
        assert!(state.linear_velocity.y > 9.0, "{state:?}");
        assert!(state.position.local.y > 0.5, "{state:?}");
    }

    #[test]
    fn sphere_comes_to_rest_on_the_ground() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 2.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 3.0);
        assert!((y(&world, ball) - 0.5).abs() < 0.02, "{}", y(&world, ball));
        let speed = world.body_state(ball).unwrap().linear_velocity.length();
        assert!(speed < 0.05, "{speed}");
    }

    #[test]
    fn restitution_sets_the_rebound_height() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .material(Material {
                        restitution: 0.5,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(0.0, 5.5, 0.0)),
            )
            .unwrap();
        let mut bounced = false;
        let mut peak = 0.0_f32;
        for _ in 0..240 {
            world.step_fixed().unwrap();
            let velocity = world.body_state(ball).unwrap().linear_velocity.y;
            bounced |= velocity > 0.0;
            if bounced {
                peak = peak.max(y(&world, ball));
            }
        }
        // 5 m of fall at e = 0.5 comes back 1.25 m
        assert!((peak - 1.75).abs() < 0.2, "{peak}");
    }

    #[test]
    fn no_restitution_does_not_bounce() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 5.5, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.1);
        let mut peak = 0.0_f32;
        for _ in 0..120 {
            world.step_fixed().unwrap();
            peak = peak.max(y(&world, ball));
        }
        assert!(peak < 0.6, "{peak}");
    }

    #[test]
    fn friction_stops_a_sliding_box_at_the_right_distance() {
        let mut world = world();
        ground(&mut world);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(0.0, 0.5, 0.0))
                    .linear_velocity(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 3.0);
        let state = world.body_state(cube).unwrap();
        assert!(state.linear_velocity.length() < 0.05);
        // v^2 / (2 mu g) with mu = 0.5
        let expected = 25.0 / 9.81;
        assert!(
            (state.position.local.x - expected).abs() < expected * 0.15,
            "{}",
            state.position.local.x
        );
    }

    #[test]
    fn no_friction_keeps_sliding() {
        let mut world = world();
        ground(&mut world);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .material(Material {
                        friction: 0.0,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(0.0, 0.5, 0.0))
                    .linear_velocity(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        let velocity = world.body_state(cube).unwrap().linear_velocity.x;
        assert!((velocity - 5.0).abs() < 0.01, "{velocity}");
    }

    #[test]
    fn box_stack_stays_up() {
        let mut world = world();
        ground(&mut world);
        let mut boxes = Vec::new();
        for level in 0..5 {
            boxes.push(
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap(),
            );
        }
        run(&mut world, 5.0);
        for (level, body) in boxes.iter().enumerate() {
            let state = world.body_state(*body).unwrap();
            let expected = 0.5 + level as f32;
            assert!(
                (state.position.local.y - expected).abs() < 0.05,
                "level {level}: {}",
                state.position.local.y
            );
            assert!(state.position.local.x.abs() < 0.02);
            assert!(state.position.local.z.abs() < 0.02);
            assert!(state.linear_velocity.length() < 0.1);
        }
    }

    fn wall(world: &mut PhysicsWorld, x: f32) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(0.05, 5.0, 5.0)))
                    .local_position(Vec3::new(x, 0.0, 0.0)),
            )
            .unwrap()
    }

    #[test]
    fn fast_bodies_do_not_pass_a_thin_wall() {
        for speed in [50.0, 200.0, 1000.0, 5000.0] {
            for shape in [
                Shape::sphere(0.25),
                Shape::cuboid(Vec3::splat(0.25)),
                Shape::capsule(0.1, 0.5),
            ] {
                let mut world = no_gravity();
                wall(&mut world, 3.0);
                let bullet = world
                    .spawn(
                        RigidBodyDesc::dynamic(shape)
                            .local_position(Vec3::new(-20.0, 0.1, 0.2))
                            .linear_velocity(Vec3::new(speed, 0.0, 0.0)),
                    )
                    .unwrap();
                run(&mut world, 1.0);
                let at = world.body_state(bullet).unwrap().position.local.x;
                assert!(at < 2.95, "{shape:?} at {speed} m/s ended at {at}");
            }
        }
    }

    #[test]
    fn thin_bullet_stops_on_thin_plate_at_the_surface() {
        let mut world = no_gravity();
        wall(&mut world, 3.0);
        let bullet = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.02))
                    .local_position(Vec3::new(-10.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(800.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 0.5);
        let state = world.body_state(bullet).unwrap();
        assert!((state.position.local.x - 2.93).abs() < 0.01, "{state:?}");
        assert!(state.linear_velocity.x.abs() < 1.0, "{state:?}");
    }

    #[test]
    fn fast_ball_keeps_its_restitution() {
        let mut world = no_gravity();
        wall(&mut world, 3.0);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .material(Material {
                        restitution: 0.5,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(-20.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(200.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 0.5);
        let speed = world.body_state(ball).unwrap().linear_velocity.x;
        assert!((speed + 100.0).abs() < 2.0, "{speed}");
    }

    #[test]
    fn fast_ball_flying_past_a_corner_is_not_touched() {
        let mut world = no_gravity();
        world
            .spawn(RigidBodyDesc::fixed(Shape::cuboid(Vec3::splat(1.0))))
            .unwrap();
        // the path closes in on the box and leaves it 0.3 m clear of the corner
        let start = Vec3::new(-6.0, 2.3 - 0.5, 0.0);
        let velocity = Vec3::new(100.0, 100.0 * 0.5 / 6.0, 0.0);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.05))
                    .local_position(start)
                    .linear_velocity(velocity),
            )
            .unwrap();
        run(&mut world, 0.5);
        let state = world.body_state(ball).unwrap();
        assert_eq!(state.linear_velocity, velocity, "{state:?}");
    }

    #[test]
    fn fast_ball_hits_a_body_and_momentum_stays() {
        let mut world = no_gravity();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.1))
                    .density(1000.0)
                    .local_position(Vec3::new(-30.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(300.0, 0.0, 0.0)),
            )
            .unwrap();
        let target = world
            .spawn(RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(0.05, 1.0, 1.0))).density(1000.0))
            .unwrap();
        run(&mut world, 0.5);
        let mass_ball = 1000.0 * 4.0 / 3.0 * std::f32::consts::PI * 0.001;
        let mass_target = 1000.0 * 0.1 * 2.0 * 2.0;
        let total = world.body_state(ball).unwrap().linear_velocity.x * mass_ball
            + world.body_state(target).unwrap().linear_velocity.x * mass_target;
        let expected = 300.0 * mass_ball;
        assert!(
            (total - expected).abs() < expected * 1e-2,
            "{total} {expected}"
        );
        assert!(world.body_state(target).unwrap().linear_velocity.x > 0.1);
    }

    #[test]
    fn fast_spinning_bar_does_not_pass_a_ball() {
        let mut world = no_gravity();
        world
            .spawn(
                RigidBodyDesc::kinematic(Shape::cuboid(Vec3::new(4.0, 0.02, 0.2)))
                    .angular_velocity(Vec3::new(0.0, 0.0, 200.0)),
            )
            .unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.1)).local_position(Vec3::new(3.0, 0.5, 0.0)),
            )
            .unwrap();
        run(&mut world, 0.2);
        let speed = world.body_state(ball).unwrap().linear_velocity.length();
        assert!(speed > 1.0, "{speed}");
    }

    #[test]
    fn fast_hit_wakes_a_sleeping_stack_cleanly() {
        let mut world = world();
        ground(&mut world);
        let boxes: Vec<_> = (0..3)
            .map(|level| {
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap()
            })
            .collect();
        run(&mut world, 2.0);
        assert!(world.body_state(boxes[0]).unwrap().asleep);
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.2))
                    .density(5000.0)
                    .local_position(Vec3::new(-20.0, 1.5, 0.0))
                    .linear_velocity(Vec3::new(300.0, 0.0, 0.0)),
            )
            .unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
            for body in &boxes {
                let state = world.body_state(*body).unwrap();
                if state.asleep {
                    assert_eq!(state.linear_velocity, Vec3::ZERO, "{state:?}");
                }
            }
        }
        assert!(world.body_state(boxes[1]).unwrap().linear_velocity.x > 0.1);
    }

    #[test]
    fn fast_drop_stays_on_terrain() {
        let mut world = world();
        let heights = vec![0.0; 25];
        let shape = world.add_heightfield(5, 5, 4.0, 4.0, &heights).unwrap();
        world
            .spawn(RigidBodyDesc::fixed(shape).local_position(Vec3::new(-8.0, 0.0, -8.0)))
            .unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .local_position(Vec3::new(0.3, 40.0, 0.2))
                    .linear_velocity(Vec3::new(0.0, -500.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        let y = world.body_state(ball).unwrap().position.local.y;
        assert!((y - 0.25).abs() < 0.02, "{y}");
    }

    #[test]
    fn small_pile_on_threads_matches_one_thread() {
        // under miri the thresholds are tiny, so this takes the colored and threaded path
        let run = |dispatch: &dyn JobDispatch| {
            let (mut world, _, _) = pyramid(4);
            let ball = world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::sphere(0.3))
                        .local_position(Vec3::new(0.0, 6.0, 0.0)),
                )
                .unwrap();
            let post = world
                .spawn(RigidBodyDesc::fixed(Shape::sphere(0.1)))
                .unwrap();
            let joint = world
                .joint_at(
                    crate::JointDesc::ball(post, ball),
                    GridPosition::from_local(Vec3::new(0.0, 7.0, 0.0)).unwrap(),
                    Quat::IDENTITY,
                )
                .unwrap();
            world.add_joint(joint).unwrap();
            for _ in 0..if cfg!(miri) { 2 } else { 30 } {
                world.step_with(dispatch).unwrap();
            }
            world.state_hash()
        };
        assert_eq!(run(&Workers(3)), run(&Sequential));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_dispatcher_skipping_contact_jobs_fails_and_leaves_state() {
        struct Skips;
        impl JobDispatch for Skips {
            fn parallelism(&self) -> usize {
                2
            }

            fn run(&self, _jobs: usize, _job: &(dyn Fn(usize) + Sync)) {}
        }
        // under the size where integration dispatches, so contact finding is what skips
        let (mut world, _, _) = pyramid(40);
        assert!(world.body_count() < 1024);
        world.step_fixed().unwrap();
        let before = world.state_hash();
        assert_eq!(
            world.step_with(&Skips),
            Err(crate::PhysicsError::DispatchIncomplete)
        );
        assert_eq!(world.state_hash(), before);
        world.step_fixed().unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn big_pyramid_stands_when_colored() {
        let (mut world, top, start) = pyramid(40);
        world.step_fixed().unwrap();
        let constraints = world.island_stats().largest_constraints;
        assert!(constraints >= super::COLORED_ISLAND, "{constraints}");
        for _ in 0..300 {
            world.step_with(&Workers(4)).unwrap();
        }
        let now = world.body_state(top).unwrap().position.local;
        assert!((now - start).length() < 0.1, "{start:?} {now:?}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn colored_solve_is_the_same_on_any_thread_count() {
        let run = |dispatch: &dyn JobDispatch| {
            let (mut world, _, _) = pyramid(40);
            for _ in 0..60 {
                world.step_with(dispatch).unwrap();
            }
            world.state_hash()
        };
        let one = run(&Sequential);
        assert_eq!(run(&Workers(4)), one);
        assert_eq!(run(&Workers(16)), one);
    }

    #[test]
    fn sixteen_box_stack_stands() {
        let mut world = world();
        ground(&mut world);
        let boxes: Vec<_> = (0..16)
            .map(|level| {
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap()
            })
            .collect();
        run(&mut world, 10.0);
        for (level, body) in boxes.iter().enumerate() {
            let at = world.body_state(*body).unwrap().position.local;
            assert!((at.y - (0.5 + level as f32)).abs() < 0.05, "{level} {at:?}");
            assert!(at.x.hypot(at.z) < 0.1, "{level} {at:?}");
        }
    }

    #[test]
    fn bad_substep_counts_fail() {
        for substeps in [0, 65] {
            let settings = PhysicsSettings {
                substeps,
                ..PhysicsSettings::default()
            };
            assert!(PhysicsWorld::new(settings).is_err());
        }
    }

    #[test]
    fn equal_spheres_swap_velocity_when_elastic() {
        let mut world = no_gravity();
        let elastic = Material {
            restitution: 1.0,
            ..Material::default()
        };
        let sphere = |x: f32, speed: f32| {
            RigidBodyDesc::dynamic(Shape::sphere(0.5))
                .material(elastic)
                .local_position(Vec3::new(x, 0.0, 0.0))
                .linear_velocity(Vec3::new(speed, 0.0, 0.0))
        };
        let left = world.spawn(sphere(-2.0, 2.0)).unwrap();
        let right = world.spawn(sphere(2.0, -2.0)).unwrap();
        run(&mut world, 2.0);
        let left = world.body_state(left).unwrap().linear_velocity.x;
        let right = world.body_state(right).unwrap().linear_velocity.x;
        assert!((left + 2.0).abs() < 0.1, "{left}");
        assert!((right - 2.0).abs() < 0.1, "{right}");
        assert!((left + right).abs() < 1e-3);
    }

    #[test]
    fn light_body_hitting_heavy_one_conserves_momentum() {
        let mut world = no_gravity();
        let light = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .density(1.0)
                    .local_position(Vec3::new(-2.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(4.0, 0.0, 0.0)),
            )
            .unwrap();
        let heavy = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.5)).density(9.0))
            .unwrap();
        run(&mut world, 1.0);
        let left = world.body_state(light).unwrap().linear_velocity.x;
        let right = world.body_state(heavy).unwrap().linear_velocity.x;
        // masses 1 to 9, no bounce, both end up at 4 / 10
        assert!((left + 9.0 * right - 4.0).abs() < 1e-3, "{left} {right}");
        assert!((right - 0.4).abs() < 0.05, "{right}");
    }

    #[test]
    fn tilted_box_settles_flat() {
        let mut world = world();
        ground(&mut world);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(0.0, 1.5, 0.0))
                    .rotation(Quat::from_axis_angle(
                        Vec3::new(1.0, 0.0, 1.0).normalize(),
                        0.5,
                    )),
            )
            .unwrap();
        run(&mut world, 5.0);
        let state = world.body_state(cube).unwrap();
        assert!((state.position.local.y - 0.5).abs() < 0.02);
        assert!(state.angular_velocity.length() < 0.1);
        let up = state.rotation.rotate_vec3(Vec3::Y);
        let aligned = up.x.abs().max(up.y.abs()).max(up.z.abs());
        assert!(aligned > 0.999, "{up:?}");
    }

    #[test]
    fn embedded_body_recovers_without_exploding() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 0.1, 0.0)),
            )
            .unwrap();
        let mut fastest = 0.0_f32;
        for _ in 0..180 {
            world.step_fixed().unwrap();
            fastest = fastest.max(world.body_state(ball).unwrap().linear_velocity.length());
        }
        assert!(fastest <= 3.5, "{fastest}");
        assert!((y(&world, ball) - 0.5).abs() < 0.02);
    }

    #[test]
    fn filtered_pair_passes_through() {
        let mut world = world();
        ground(&mut world);
        let ghost = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .filter(CollisionFilter { layers: 2, mask: 2 })
                    .local_position(Vec3::new(0.0, 1.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        assert!(y(&world, ghost) < -1.0);
    }

    #[test]
    fn different_frames_do_not_collide() {
        let mut world = world();
        ground(&mut world);
        let other = world.add_frame(FrameDesc::new(FrameId::ROOT)).unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .frame(other)
                    .local_position(Vec3::new(0.0, 1.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        assert!(y(&world, ball) < -1.0);
    }

    #[test]
    fn contact_works_far_from_origin_and_across_cells() {
        let mut world = world();
        let cell = IVec3::new(2_000_000, -700_000, 1_000_000);
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(2000.0, 1.0, 10.0))).position(
                    GridPosition {
                        cell,
                        local: Vec3::new(0.0, -1.0, 0.0),
                    },
                ),
            )
            .unwrap();
        let next_cell = IVec3::new(cell.x + 1, cell.y, cell.z);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).position(GridPosition {
                    cell: next_cell,
                    local: Vec3::new(0.0, 1.0, 0.0),
                }),
            )
            .unwrap();
        run(&mut world, 3.0);
        let state = world.body_state(ball).unwrap();
        assert_eq!(state.position.cell, next_cell);
        assert!((state.position.local.y - 0.5).abs() < 0.02);
    }

    #[test]
    fn snapshot_in_mid_contact_continues_exactly() {
        let mut world = world();
        ground(&mut world);
        for level in 0..4 {
            world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                        .local_position(Vec3::new(0.05 * level as f32, 0.6 + level as f32, 0.0)),
                )
                .unwrap();
        }
        run(&mut world, 0.5);
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        run(&mut world, 1.0);
        run(&mut loaded, 1.0);
        assert_eq!(loaded.state_hash(), world.state_hash());
    }

    #[test]
    fn removing_a_body_mid_stack_keeps_the_rest_working() {
        let mut world = world();
        ground(&mut world);
        let mut boxes = Vec::new();
        for level in 0..3 {
            boxes.push(
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap(),
            );
        }
        run(&mut world, 1.0);
        world.despawn(boxes[0]).unwrap();
        run(&mut world, 2.0);
        assert!(
            (y(&world, boxes[1]) - 0.5).abs() < 0.05,
            "{}",
            y(&world, boxes[1])
        );
        assert!(
            (y(&world, boxes[2]) - 1.5).abs() < 0.05,
            "{}",
            y(&world, boxes[2])
        );
    }
}
