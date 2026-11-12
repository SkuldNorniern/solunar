use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use algebrix::Vec3;

use crate::body::Bodies;
use crate::direct::{DirectNode, Factor, Symbolic};
use crate::joint::{
    JointPoint, JointRow, solve_point, solve_point_recovery, solve_row, solve_row_recovery,
    warm_start_point, warm_start_row,
};
use crate::{BodyKind, JobDispatch};

use super::contacts::{ContactScratch, SolverManifold, SolverPoint};
use super::islands::Islands;
use super::rows::{solve_manifold, solve_recovery, warm_start};
use super::{COLORED_ISLAND, COLORS, Motion, Pair};

/// Per substep. Odd on purpose, 2 and 4 left stacks jittering far more than 3.
const VELOCITY_ITERATIONS: usize = 3;
const RECOVERY_ITERATIONS: usize = 1;
const RELAX_ITERATIONS: usize = 1;

/// Constraints in one block of work. A block is the smallest piece a thread takes.
const COLOR_CHUNK: usize = if cfg!(miri) { 2 } else { 64 };
/// Below this many blocks in a substep the solve stays on the calling thread.
const PARALLEL_BLOCKS: usize = if cfg!(miri) { 2 } else { 8 };

/// Spins before a waiting thread yields.
const SPINS: u32 = 64;

/// Colors smaller than this, next to each other, go in one block, one color after another.
/// A barrier per tiny color cost more than solving them on one thread.
const SMALL_COLOR: usize = 2 * COLOR_CHUNK;
/// A stage this many blocks wide keeps one more thread busy.
const BLOCKS_PER_WORKER: usize = if cfg!(miri) { 1 } else { 4 };

/// A constraint in solve order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Item {
    Point(u32),
    Row(u32),
    Manifold(u32),
}

/// Constraints put in colors, so no two in a color share a dynamic body. Colors go one after
/// another, a color can go on many threads and still give what one thread does.
#[derive(Debug, Clone, Default)]
pub(super) struct Colors {
    pub(super) used: Vec<u64>,
    color: Vec<u8>,
    pub(super) order: Vec<Item>,
    items: Vec<Item>,
    pub(super) ends: Vec<u32>,
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
    pub(super) stages: Vec<Stage>,
}

/// Blocks that may run at once. Every block of a stage is done before the next one starts.
#[derive(Debug, Clone, Copy)]
pub(super) enum Stage {
    Islands {
        blocks: u32,
        relax: bool,
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
            Self::Islands { blocks, .. }
            | Self::Direct { blocks, .. }
            | Self::Colors { blocks, .. } => blocks as usize,
        }
    }
}

/// Lets the jobs of one color reach their own constraints and bodies at once.
pub(super) struct Shared<'a, T> {
    ptr: *mut T,
    len: usize,
    marker: PhantomData<&'a mut [T]>,
}

// SAFETY: jobs only use indexes no other job of the same color touches, see `solve_color`.
unsafe impl<T: Send> Sync for Shared<'_, T> {}

impl<'a, T: Copy> Shared<'a, T> {
    pub(super) fn new(slice: &'a mut [T]) -> Self {
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
pub(super) fn color_constraints(bodies: &Bodies, scratch: &mut ContactScratch) {
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
pub(super) fn factor_islands(bodies: &Bodies, scratch: &mut ContactScratch) {
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
/// With `relax` only the relax sweeps, run after the bodies moved.
pub(super) fn plan_stages(colors: &mut Colors, substep: u32, relax: bool) {
    let mut stages = std::mem::take(&mut colors.stages);
    stages.clear();
    if colors.runs.len() > 1 {
        stages.push(Stage::Islands {
            blocks: colors.runs.len() as u32 - 1,
            relax,
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
        if direct > 0 && pass != Pass::Relax {
            stages.push(Stage::Direct {
                pass,
                blocks: direct,
            });
        }
    };
    if relax {
        for _ in 0..RELAX_ITERATIONS {
            sweep(Pass::Relax, false);
        }
        colors.stages = stages;
        return;
    }
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
pub(super) fn solve_block(
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
        Stage::Islands { relax, .. } => {
            let (first, last) = (colors.runs[block] as usize, colors.runs[block + 1] as usize);
            for index in first..last {
                let start = if index == 0 {
                    0
                } else {
                    colors.serial_ends[index - 1]
                };
                let items = &colors.serial[start as usize..colors.serial_ends[index] as usize];
                if relax {
                    for _ in 0..RELAX_ITERATIONS {
                        items.iter().for_each(|item| go(item, Pass::Relax));
                    }
                    continue;
                }
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
                Pass::Relax => {}
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
pub(super) fn run_stages(
    stages: &[Stage],
    dispatch: &dyn JobDispatch,
    run: &(dyn Fn(Stage, usize) + Sync),
) {
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
pub(super) enum Pass {
    WarmStart,
    Velocity,
    Recovery,
    /// Contacts again after the move, without their push out.
    Relax,
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
                Pass::Relax => {}
            }
        }
        Item::Row(index) => {
            let row = unsafe { rows.get(index as usize) };
            match pass {
                Pass::WarmStart => warm_start_row(&mut pair, row),
                Pass::Velocity => solve_row(&mut pair, row),
                Pass::Recovery => solve_row_recovery(&mut pair, row),
                Pass::Relax => {}
            }
        }
        Item::Manifold(index) => {
            let manifold = unsafe { manifolds.get(index as usize) };
            // SAFETY: a manifold's points are its own, and the manifold is this job's.
            let points = unsafe { contacts.slice(manifold.first, manifold.count) };
            match pass {
                Pass::WarmStart => warm_start(&mut pair, manifold, points),
                Pass::Velocity => solve_manifold(&mut pair, manifold, points, false),
                Pass::Relax => solve_manifold(&mut pair, manifold, points, true),
                Pass::Recovery => solve_recovery(&mut pair, manifold, points),
            }
        }
    }
}
