//! Fixed rigid body scenes, timed per step.
//! `cargo run --release --example bench [--threads N] [pile ...]`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use solunar::{
    Base, BodyKind, CompoundChild, Edit, EditOp, EditShape, FluidSettings, FluidSolver, FluidWorld,
    GridPosition, JobDispatch, JointDesc, JointHandle, Material, PhaseTimes, PhysicsError,
    PhysicsSettings, PhysicsWorld, Quat, QueryFilter, Ray, RigidBodyDesc, SdfDesc, SdfTerrain,
    Sequential, ShallowSettings, ShallowWater, Shape, ShapeCast, Vec3,
};

type Result<T> = std::result::Result<T, PhysicsError>;

struct Outcome {
    bodies: usize,
    times: Vec<Duration>,
    unit: &'static str,
    check: String,
}

type Suite = fn() -> Result<Outcome>;

const SUITES: [(&str, Suite); 23] = [
    ("pile", pile),
    ("sleepers", sleepers),
    ("sleep-rays", sleep_rays),
    ("stacks", stacks),
    ("chains", chains),
    ("net", net),
    ("compound", compound),
    ("queries", queries),
    ("ray-batch", ray_batch),
    ("blocked-batch", blocked_batch),
    ("fan-nearest", fan_nearest),
    ("fan-blocked", fan_blocked),
    ("terrain-fan", terrain_fan),
    ("terrain-casts", terrain_casts),
    ("bullets", bullets),
    ("tunnel", tunnel),
    ("commit", tunnel_commit),
    ("moving", moving_tunnel),
    ("dam", dam_small),
    ("dam-fine", dam_medium),
    ("dam-dfsph", dam_dfsph),
    ("shallow", shallow),
    ("sensors", sensors),
];

/// A rayon pool as the host's job runner, standing in for a game engine's.
struct Pool(rayon::ThreadPool);

impl JobDispatch for Pool {
    fn parallelism(&self) -> usize {
        self.0.current_num_threads()
    }

    fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
        self.0.scope(|scope| {
            for index in 0..jobs {
                scope.spawn(move |_| job(index));
            }
        });
    }
}

/// The system allocator, counting calls and bytes, so a step's allocations can be read.
struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);

// SAFETY: every call goes straight to `System` with the same arguments, only counters are added.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: as the caller promised to us.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: as the caller promised to us.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED.fetch_add(new_size as u64, Ordering::Relaxed);
        // SAFETY: as the caller promised to us.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as the caller promised to us.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// Allocations over the measured steps: calls, bytes and the most calls in one step.
#[derive(Clone, Copy, Default)]
struct Allocs {
    calls: u64,
    bytes: u64,
    most: u64,
}

thread_local! {
    static ALLOCS: std::cell::Cell<Allocs> = const { std::cell::Cell::new(Allocs { calls: 0, bytes: 0, most: 0 }) };
}

thread_local! {
    static DISPATCH: std::cell::RefCell<Option<Pool>> = const { std::cell::RefCell::new(None) };
}

thread_local! {
    static PHASES: std::cell::RefCell<(PhaseTimes, u32)> =
        std::cell::RefCell::new((PhaseTimes::default(), 0));
}

fn main() {
    let mut wanted: Vec<String> = std::env::args().skip(1).collect();
    let mut threads = 1;
    if let Some(at) = wanted.iter().position(|arg| arg == "--threads") {
        threads = wanted
            .get(at + 1)
            .and_then(|count| count.parse().ok())
            .unwrap_or(1);
        wanted.drain(at..(at + 2).min(wanted.len()));
    }
    if threads > 1 {
        match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            Ok(pool) => DISPATCH.with(|dispatch| *dispatch.borrow_mut() = Some(Pool(pool))),
            Err(error) => {
                eprintln!("no thread pool: {error}");
                return;
            }
        }
    }
    println!(
        "solunar {} {} {}, {} threads seen, {threads} used",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |count| count.get()),
    );
    println!("| suite | bodies | samples | p50 | p95 | p99 | max | check |");
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    let mut splits = Vec::new();
    for (name, suite) in SUITES {
        if !wanted.is_empty() && !wanted.iter().any(|want| want == name) {
            continue;
        }
        PHASES.with(|phases| *phases.borrow_mut() = (PhaseTimes::default(), 0));
        ALLOCS.with(|allocs| allocs.set(Allocs::default()));
        match suite() {
            Ok(outcome) => print_row(name, &outcome),
            Err(error) => println!("| {name} | failed: {error} | | | | | | |"),
        }
        splits.push((
            name,
            PHASES.with(|phases| *phases.borrow()),
            ALLOCS.with(std::cell::Cell::get),
        ));
    }
    println!();
    println!("mean ms per measured step, step is the sum");
    println!();
    println!("| suite | broadphase | narrowphase | solve | integrate | finish | step |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for (name, (sum, count), _) in &splits {
        let (sum, count) = (*sum, *count);
        if count == 0 {
            continue;
        }
        let mean = |time: Duration| format!("{:.3}", time.as_secs_f64() * 1e3 / f64::from(count));
        println!(
            "| {name} | {} | {} | {} | {} | {} | {} |",
            mean(sum.broadphase),
            mean(sum.narrowphase),
            mean(sum.solve),
            mean(sum.integrate),
            mean(sum.finish),
            mean(sum.broadphase + sum.narrowphase + sum.solve + sum.integrate + sum.finish),
        );
    }
    println!();
    println!("allocations per measured step, every thread");
    println!();
    println!("| suite | mean calls | most calls | mean kib |");
    println!("| --- | --- | --- | --- |");
    for (name, (_, count), allocs) in &splits {
        if *count == 0 {
            continue;
        }
        let steps = f64::from(*count);
        println!(
            "| {name} | {:.1} | {} | {:.1} |",
            allocs.calls as f64 / steps,
            allocs.most,
            allocs.bytes as f64 / steps / 1024.0,
        );
    }
}

fn dam_small() -> Result<Outcome> {
    dam(0.05)
}

fn dam_medium() -> Result<Outcome> {
    dam(0.03)
}

fn dam_dfsph() -> Result<Outcome> {
    dam_with(0.03, FluidSolver::Dfsph)
}

/// A water column 0.6 m tall and wide in a 2.4 m tank, let go at once. Timed per 60 hz tick, which
/// holds as many fluid substeps as the flow needs.
fn dam(spacing: f32) -> Result<Outcome> {
    dam_with(spacing, FluidSolver::Wcsph)
}

fn dam_with(spacing: f32, solver: FluidSolver) -> Result<Outcome> {
    let mut fluid = FluidWorld::new(FluidSettings {
        spacing,
        solver,
        ..FluidSettings::default()
    })?;
    fluid.add_box_container(Vec3::ZERO, Vec3::new(2.4, 1.0, 0.3))?;
    fluid.fill_box(Vec3::ZERO, Vec3::new(0.6, 0.6, 0.3))?;
    let mut times = Vec::new();
    let mut substeps = 0;
    let mut held = 0;
    let mut squeeze = 0.0_f32;
    // two seconds, the wave hits the far wall and comes back
    for _ in 0..120 {
        let started = Instant::now();
        let report = DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => fluid.step(1.0 / 60.0, pool),
            None => fluid.step(1.0 / 60.0, &Sequential),
        })?;
        times.push(started.elapsed());
        substeps += report.substeps;
        held += report.held_back;
        squeeze = squeeze.max(report.max_compression);
    }
    Ok(Outcome {
        bodies: fluid.len(),
        times,
        unit: "ms",
        check: format!(
            "{:.1} substeps a tick, most squeezed {:.1}%, {held} held back, hash {:016x}",
            f64::from(substeps) / 120.0,
            squeeze * 100.0,
            fluid.state_hash()
        ),
    })
}

/// A reservoir let go down a 100 by 25 m valley on 0.25 m cells, 40k of them, timed per 60 hz
/// tick.
fn shallow() -> Result<Outcome> {
    let size = (400, 100);
    let mut bed = Vec::with_capacity(size.0 * size.1);
    for z in 0..size.1 {
        for x in 0..size.0 {
            // falls along x, rises to both banks
            let across = (z as f32 - 50.0) / 50.0;
            bed.push(-0.02 * x as f32 + 3.0 * across * across);
        }
    }
    let mut water = ShallowWater::new(ShallowSettings::default(), Vec3::ZERO, size, bed)?;
    water.fill((0, 0), (99, 99), 2.0)?;
    let mut times = Vec::new();
    let mut substeps = 0;
    for _ in 0..120 {
        let started = Instant::now();
        let report = DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => water.step(1.0 / 60.0, pool),
            None => water.step(1.0 / 60.0, &Sequential),
        })?;
        times.push(started.elapsed());
        substeps += report.substeps;
    }
    Ok(Outcome {
        bodies: size.0 * size.1,
        times,
        unit: "ms",
        check: format!(
            "{:.1} substeps a tick, {:.1} m3 held, hash {:016x}",
            f64::from(substeps) / 120.0,
            water.volume(),
            water.state_hash()
        ),
    })
}

fn print_row(name: &str, outcome: &Outcome) {
    let mut sorted = outcome.times.clone();
    sorted.sort_unstable();
    let at = |percent: usize| {
        let rank = (sorted.len() * percent).div_ceil(100).max(1);
        sorted.get(rank - 1).copied().unwrap_or_default()
    };
    let show = |time: Duration| match outcome.unit {
        "us" => format!("{:.2} us", time.as_secs_f64() * 1e6),
        _ => format!("{:.3} ms", time.as_secs_f64() * 1e3),
    };
    println!(
        "| {name} | {} | {} | {} | {} | {} | {} | {} |",
        outcome.bodies,
        sorted.len(),
        show(at(50)),
        show(at(95)),
        show(at(99)),
        show(sorted.last().copied().unwrap_or_default()),
        outcome.check,
    );
}

fn world() -> Result<PhysicsWorld> {
    PhysicsWorld::new(PhysicsSettings::default())
}

fn ground(world: &mut PhysicsWorld, half: f32) -> Result<()> {
    world.spawn(
        RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(half, 1.0, half)))
            .local_position(Vec3::new(0.0, -1.0, 0.0)),
    )?;
    Ok(())
}

fn steps(world: &mut PhysicsWorld, count: usize) -> Result<Vec<Duration>> {
    let mut times = Vec::with_capacity(count);
    for _ in 0..count {
        let (calls, bytes) = (
            ALLOCATIONS.load(Ordering::Relaxed),
            ALLOCATED.load(Ordering::Relaxed),
        );
        let started = Instant::now();
        DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => world.step_with(pool),
            None => world.step_with(&Sequential),
        })?;
        times.push(started.elapsed());
        let calls = ALLOCATIONS.load(Ordering::Relaxed) - calls;
        let bytes = ALLOCATED.load(Ordering::Relaxed) - bytes;
        ALLOCS.with(|allocs| {
            let mut sum = allocs.get();
            sum.calls += calls;
            sum.bytes += bytes;
            sum.most = sum.most.max(calls);
            allocs.set(sum);
        });
        let step = world.phase_times();
        PHASES.with(|phases| {
            let (sum, count) = &mut *phases.borrow_mut();
            sum.broadphase += step.broadphase;
            sum.narrowphase += step.narrowphase;
            sum.solve += step.solve;
            sum.integrate += step.integrate;
            sum.finish += step.finish;
            *count += 1;
        });
    }
    Ok(times)
}

fn max_speed(world: &PhysicsWorld) -> Result<f32> {
    let mut fastest = 0.0_f32;
    for body in world.body_handles() {
        let state = world.body_state(body)?;
        if state.kind == BodyKind::Dynamic {
            fastest = fastest.max(state.linear_velocity.length());
        }
    }
    Ok(fastest)
}

fn max_gap(world: &PhysicsWorld, joints: &[JointHandle]) -> Result<f32> {
    let mut widest = 0.0_f32;
    for joint in joints {
        let (a, b) = world.joint_anchors(*joint)?;
        widest = widest.max(b.offset_from(a).length() as f32);
    }
    Ok(widest)
}

fn ball_joint(
    world: &mut PhysicsWorld,
    a: solunar::BodyHandle,
    b: solunar::BodyHandle,
    at: Vec3,
) -> Result<JointHandle> {
    let anchor = GridPosition::from_local(at).ok_or(PhysicsError::OutOfRange)?;
    let desc = world.joint_at(JointDesc::ball(a, b), anchor, Quat::IDENTITY)?;
    world.add_joint(desc)
}

/// 2,000 spheres, boxes and capsules falling into a pile, all awake.
fn pile() -> Result<Outcome> {
    let mut world = world()?;
    ground(&mut world, 60.0)?;
    let shapes = [
        Shape::sphere(0.4),
        Shape::cuboid(Vec3::splat(0.4)),
        Shape::capsule(0.3, 0.3),
    ];
    for i in 0..2000 {
        let (x, z, y) = (i % 20, (i / 20) % 10, i / 200);
        let at = Vec3::new(
            x as f32 * 1.2 - 12.0,
            1.0 + y as f32 * 1.2,
            z as f32 * 1.2 - 6.0,
        );
        world.spawn(RigidBodyDesc::dynamic(shapes[i % 3]).local_position(at))?;
    }
    let times = steps(&mut world, 300)?;
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!(
            "{} awake, fastest {:.2} m/s, {} islands, largest {}",
            world.body_count() - 1 - world.sleeping_body_count(),
            max_speed(&world)?,
            world.island_stats().count,
            world.island_stats().largest_bodies
        ),
    })
}

/// 10,000 boxes resting apart and asleep, so the step is mostly the sensors: 400 spheres of
/// 2 m over the field, each reporting what is inside.
fn sensors() -> Result<Outcome> {
    let mut world = world()?;
    ground(&mut world, 200.0)?;
    for i in 0..10_000 {
        let (x, z) = ((i % 100) as f32, (i / 100) as f32);
        let at = Vec3::new(x * 1.5 - 75.0, 0.41, z * 1.5 - 75.0);
        world.spawn(RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.4))).local_position(at))?;
    }
    let mut sensors = Vec::new();
    for i in 0..400 {
        let (x, z) = ((i % 20) as f32, (i / 20) as f32);
        let at = GridPosition::from_local(Vec3::new(x * 7.5 - 72.0, 0.5, z * 7.5 - 72.0))
            .ok_or(PhysicsError::OutOfRange)?;
        sensors.push(world.add_sensor(solunar::SensorDesc::new(Shape::sphere(2.0), at))?);
    }
    let times = steps(&mut world, 120)?;
    let held: usize = sensors
        .iter()
        .map(|sensor| world.sensor_bodies(*sensor).len())
        .sum();
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("400 sensors, {held} held"),
    })
}

/// 10,000 boxes resting apart, asleep, and 100 balls dropped among them.
fn sleepers() -> Result<Outcome> {
    let mut world = world()?;
    ground(&mut world, 200.0)?;
    for i in 0..10_000 {
        let (x, z) = ((i % 100) as f32, (i / 100) as f32);
        let at = Vec3::new(x * 1.5 - 75.0, 0.41, z * 1.5 - 75.0);
        world.spawn(RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.4))).local_position(at))?;
    }
    for _ in 0..600 {
        world.step_fixed()?;
        if world.sleeping_body_count() == 10_000 {
            break;
        }
    }
    for i in 0..100 {
        let (x, z) = ((i % 10) as f32, (i / 10) as f32);
        let at = Vec3::new(x * 15.0 - 75.75, 5.0, z * 15.0 - 75.75);
        world.spawn(RigidBodyDesc::dynamic(Shape::sphere(0.2)).local_position(at))?;
    }
    let times = steps(&mut world, 300)?;
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{} asleep", world.sleeping_body_count()),
    })
}

/// The `sleepers` field once all of it sleeps, then frames of one step and 100 rays down onto
/// it, like a listener's or a light's checks each frame. Times are whole frames.
fn sleep_rays() -> Result<Outcome> {
    let mut world = world()?;
    ground(&mut world, 200.0)?;
    for i in 0..10_000 {
        let (x, z) = ((i % 100) as f32, (i / 100) as f32);
        let at = Vec3::new(x * 1.5 - 75.0, 0.41, z * 1.5 - 75.0);
        world.spawn(RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.4))).local_position(at))?;
    }
    for _ in 0..600 {
        world.step_fixed()?;
        if world.sleeping_body_count() == 10_000 {
            break;
        }
    }
    let rays: Vec<Ray> = (0..100)
        .map(|i| {
            let (x, z) = ((i % 10) as f32, (i / 10) as f32);
            let at = Vec3::new(x * 15.0 - 75.0, 5.0, z * 15.0 - 75.0);
            Ok(Ray {
                origin: GridPosition::from_local(at).ok_or(PhysicsError::OutOfRange)?,
                direction: Vec3::NEG_Y,
                max_distance: 10.0,
            })
        })
        .collect::<Result<_>>()?;
    let filter = QueryFilter::default();
    let mut hits = 0;
    let mut times = Vec::with_capacity(300);
    for _ in 0..300 {
        let started = Instant::now();
        world.step_fixed()?;
        hits = 0;
        for ray in &rays {
            hits += usize::from(world.raycast(ray, &filter)?.is_some());
        }
        times.push(started.elapsed());
    }
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{hits} of 100 hit, {} asleep", world.sleeping_body_count()),
    })
}

/// 30 stacks of 10 boxes.
fn stacks() -> Result<Outcome> {
    let mut world = world()?;
    ground(&mut world, 60.0)?;
    let mut tops = Vec::new();
    for stack in 0..30 {
        let (x, z) = (
            (stack % 6) as f32 * 2.5 - 6.25,
            (stack / 6) as f32 * 2.5 - 5.0,
        );
        for level in 0..10 {
            let at = Vec3::new(x, 0.5 + level as f32 * 1.01, z);
            let body = world.spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5))).local_position(at),
            )?;
            if level == 9 {
                tops.push((body, at));
            }
        }
    }
    let times = steps(&mut world, 600)?;
    let mut fallen = 0;
    for (body, start) in tops {
        let now = world.body_state(body)?.position.local;
        if (now - start).length() > 0.3 {
            fallen += 1;
        }
    }
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{fallen} of 30 fell, fastest {:.3} m/s", max_speed(&world)?),
    })
}

/// 20 chains of 15 balls, let go level so they swing.
fn chains() -> Result<Outcome> {
    let mut world = world()?;
    let mut joints = Vec::new();
    for chain in 0..20 {
        let z = chain as f32 * 1.5;
        let mut last = world.spawn(
            RigidBodyDesc::fixed(Shape::sphere(0.1)).local_position(Vec3::new(0.0, 20.0, z)),
        )?;
        for link in 1..=15 {
            let at = Vec3::new(link as f32 * 0.5, 20.0, z);
            let ball =
                world.spawn(RigidBodyDesc::dynamic(Shape::sphere(0.15)).local_position(at))?;
            joints.push(ball_joint(&mut world, last, ball, at - Vec3::X * 0.25)?);
            last = ball;
        }
    }
    let times = steps(&mut world, 600)?;
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!(
            "widest joint gap {:.2} cm",
            max_gap(&world, &joints)? * 100.0
        ),
    })
}

/// A 20 by 20 net of balls, each tied to its right and lower neighbour, held at two corners.
fn net() -> Result<Outcome> {
    let mut world = world()?;
    let mut grid = Vec::new();
    let spacing = 0.5;
    let at = |x: usize, z: usize| Vec3::new(x as f32 * spacing, 10.0, z as f32 * spacing);
    for z in 0..20 {
        for x in 0..20 {
            grid.push(
                world.spawn(RigidBodyDesc::dynamic(Shape::sphere(0.1)).local_position(at(x, z)))?,
            );
        }
    }
    let mut joints = Vec::new();
    for z in 0..20 {
        for x in 0..20 {
            let here = grid[z * 20 + x];
            if x + 1 < 20 {
                let mid = (at(x, z) + at(x + 1, z)) * 0.5;
                joints.push(ball_joint(&mut world, here, grid[z * 20 + x + 1], mid)?);
            }
            if z + 1 < 20 {
                let mid = (at(x, z) + at(x, z + 1)) * 0.5;
                joints.push(ball_joint(&mut world, here, grid[(z + 1) * 20 + x], mid)?);
            }
        }
    }
    for x in [0, 19] {
        let post =
            world.spawn(RigidBodyDesc::fixed(Shape::sphere(0.05)).local_position(at(x, 0)))?;
        joints.push(ball_joint(&mut world, post, grid[x], at(x, 0))?);
    }
    let times = steps(&mut world, 300)?;
    let islands = world.island_stats();
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!(
            "{} joints, largest island {}, widest gap {:.2} cm",
            joints.len(),
            islands.largest_bodies,
            max_gap(&world, &joints)? * 100.0
        ),
    })
}

/// A compound of 2,000 small boxes dropped on the ground, then 50 boxes onto it.
fn compound() -> Result<Outcome> {
    let mut world = world()?;
    ground(&mut world, 60.0)?;
    let mut parts = Vec::new();
    for i in 0..2000 {
        let (x, y, z) = ((i % 20) as f32, ((i / 20) % 10) as f32, (i / 200) as f32);
        parts.push(CompoundChild::new(
            Shape::cuboid(Vec3::splat(0.1)),
            Vec3::new(x * 0.2, y * 0.2, z * 0.2),
        ));
    }
    let info = world.add_compound(&parts)?;
    let (center, turn) = info.placed(Vec3::new(-2.0, 0.5, -1.0), Quat::IDENTITY);
    world.spawn(
        RigidBodyDesc::dynamic(info.shape)
            .local_position(center)
            .rotation(turn),
    )?;
    for i in 0..50 {
        let at = Vec3::new((i % 10) as f32 * 0.4 - 2.0, 4.0 + (i / 10) as f32, 0.0);
        world.spawn(RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.15))).local_position(at))?;
    }
    let times = steps(&mut world, 300)?;
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("2000 parts, fastest {:.3} m/s", max_speed(&world)?),
    })
}

/// The falling pile at rest, then 10,000 rays and 1,000 sphere casts, timed one by one.
/// The `queries` scene and 10,000 rays down into it.
fn ray_scene() -> Result<(PhysicsWorld, Vec<Ray>)> {
    let mut world = world()?;
    ground(&mut world, 60.0)?;
    let shapes = [
        Shape::sphere(0.4),
        Shape::cuboid(Vec3::splat(0.4)),
        Shape::capsule(0.3, 0.3),
    ];
    for i in 0..2000 {
        let (x, z, y) = (i % 20, (i / 20) % 10, i / 200);
        let at = Vec3::new(
            x as f32 * 1.2 - 12.0,
            1.0 + y as f32 * 1.2,
            z as f32 * 1.2 - 6.0,
        );
        world.spawn(RigidBodyDesc::dynamic(shapes[i % 3]).local_position(at))?;
    }
    steps(&mut world, 300)?;
    let mut seed = 0x9e37_79b9_u32;
    let mut random = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1 << 24) as f32
    };
    let mut rays = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let origin = Vec3::new(random() * 30.0 - 15.0, 15.0, random() * 16.0 - 8.0);
        let target = Vec3::new(random() * 24.0 - 12.0, 0.0, random() * 12.0 - 6.0);
        rays.push(Ray {
            origin: GridPosition::from_local(origin).ok_or(PhysicsError::OutOfRange)?,
            direction: (target - origin).normalize(),
            max_distance: 40.0,
        });
    }
    Ok((world, rays))
}

/// The `queries` scene, 10,000 rays asked as one batch on the bench's workers, 100 times.
fn ray_batch() -> Result<Outcome> {
    let (world, rays) = ray_scene()?;
    let filter = QueryFilter::default();
    let mut out = Vec::new();
    let mut times = Vec::with_capacity(100);
    for _ in 0..100 {
        let started = Instant::now();
        DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => world.raycast_batch(&rays, &filter, pool, &mut out),
            None => world.raycast_batch(&rays, &filter, &Sequential, &mut out),
        });
        times.push(started.elapsed());
    }
    let hits = out.iter().filter(|hit| matches!(hit, Ok(Some(_)))).count();
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{hits} of 10000 hit, per batch of 10000"),
    })
}

/// The `queries` scene and 10,000 rays out from one point over it, spread evenly over the
/// sphere: like a listener or a light, about half go down into the pile, half up into nothing.
fn fan_scene() -> Result<(PhysicsWorld, Vec<Ray>)> {
    let (world, _) = ray_scene()?;
    Ok((world, fan(Vec3::new(0.0, 20.0, 0.0), 40.0)?))
}

/// 10,000 rays out from `from`, spread evenly over the sphere.
fn fan(from: Vec3, length: f32) -> Result<Vec<Ray>> {
    let origin = GridPosition::from_local(from).ok_or(PhysicsError::OutOfRange)?;
    let count = 10_000;
    let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let rays = (0..count)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
            let around = (1.0 - y * y).sqrt();
            let turn = golden * i as f32;
            Ray {
                origin,
                direction: Vec3::new(around * turn.cos(), y, around * turn.sin()),
                max_distance: length,
            }
        })
        .collect();
    Ok(rays)
}

/// The tunnel hills alone.
fn hills() -> Result<PhysicsWorld> {
    let mut world = world()?;
    SdfTerrain::new(
        &mut world,
        SdfDesc {
            origin: GridPosition::ORIGIN,
            voxel: 0.5,
            brick_cells: 16,
            bricks: [16, 6, 16],
            base: Base::Hills {
                seed: 5,
                height: 24.0,
                amplitude: 3.0,
                wavelength: 30.0,
                octaves: 4,
            },
            material: Material::default(),
        },
    )?;
    Ok(world)
}

/// 1,000 casts of a 0.3 m sphere along the `terrain-fan` directions, every tenth ray.
fn terrain_casts() -> Result<Outcome> {
    let world = hills()?;
    let casts: Vec<ShapeCast> = fan(Vec3::new(64.0, 33.0, 64.0), 60.0)?
        .into_iter()
        .step_by(10)
        .map(|ray| ShapeCast {
            shape: Shape::sphere(0.3),
            rotation: Quat::IDENTITY,
            origin: ray.origin,
            direction: ray.direction,
            max_distance: ray.max_distance,
        })
        .collect();
    let filter = QueryFilter::default();
    let mut out = Vec::new();
    let mut times = Vec::with_capacity(100);
    for _ in 0..100 {
        let started = Instant::now();
        DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => world.shape_cast_batch(&casts, &filter, pool, &mut out),
            None => world.shape_cast_batch(&casts, &filter, &Sequential, &mut out),
        });
        times.push(started.elapsed());
    }
    let hits = out.iter().filter(|hit| matches!(hit, Ok(Some(_)))).count();
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{hits} of 1000 hit, per batch of 1000"),
    })
}

/// The fan nearest hits from 6 m over the middle of the tunnel hills, 60 m long: the down half
/// lands on terrain triangles, the up half runs to its end.
fn terrain_fan() -> Result<Outcome> {
    let world = hills()?;
    let rays = fan(Vec3::new(64.0, 33.0, 64.0), 60.0)?;
    let filter = QueryFilter::default();
    let mut out = Vec::new();
    let mut times = Vec::with_capacity(100);
    for _ in 0..100 {
        let started = Instant::now();
        DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => world.raycast_batch(&rays, &filter, pool, &mut out),
            None => world.raycast_batch(&rays, &filter, &Sequential, &mut out),
        });
        times.push(started.elapsed());
    }
    let hits = out.iter().filter(|hit| matches!(hit, Ok(Some(_)))).count();
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{hits} of 10000 hit, per batch of 10000"),
    })
}

fn fan_nearest() -> Result<Outcome> {
    let (world, rays) = fan_scene()?;
    let filter = QueryFilter::default();
    let mut out = Vec::new();
    let mut times = Vec::with_capacity(100);
    for _ in 0..100 {
        let started = Instant::now();
        DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => world.raycast_batch(&rays, &filter, pool, &mut out),
            None => world.raycast_batch(&rays, &filter, &Sequential, &mut out),
        });
        times.push(started.elapsed());
    }
    let hits = out.iter().filter(|hit| matches!(hit, Ok(Some(_)))).count();
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{hits} of 10000 hit, per batch of 10000"),
    })
}

fn fan_blocked() -> Result<Outcome> {
    let (world, rays) = fan_scene()?;
    let filter = QueryFilter::default();
    let mut out = Vec::new();
    let mut times = Vec::with_capacity(100);
    for _ in 0..100 {
        let started = Instant::now();
        DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => world.ray_blocked_batch(&rays, &filter, pool, &mut out),
            None => world.ray_blocked_batch(&rays, &filter, &Sequential, &mut out),
        });
        times.push(started.elapsed());
    }
    let blocked = out.iter().filter(|hit| matches!(hit, Ok(true))).count();
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{blocked} of 10000 blocked, per batch of 10000"),
    })
}

/// The same rays as `ray-batch`, asked only whether something is in the way.
fn blocked_batch() -> Result<Outcome> {
    let (world, rays) = ray_scene()?;
    let filter = QueryFilter::default();
    let mut out = Vec::new();
    let mut times = Vec::with_capacity(100);
    for _ in 0..100 {
        let started = Instant::now();
        DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => world.ray_blocked_batch(&rays, &filter, pool, &mut out),
            None => world.ray_blocked_batch(&rays, &filter, &Sequential, &mut out),
        });
        times.push(started.elapsed());
    }
    let blocked = out.iter().filter(|hit| matches!(hit, Ok(true))).count();
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{blocked} of 10000 blocked, per batch of 10000"),
    })
}

fn queries() -> Result<Outcome> {
    let mut world = world()?;
    ground(&mut world, 60.0)?;
    let shapes = [
        Shape::sphere(0.4),
        Shape::cuboid(Vec3::splat(0.4)),
        Shape::capsule(0.3, 0.3),
    ];
    for i in 0..2000 {
        let (x, z, y) = (i % 20, (i / 20) % 10, i / 200);
        let at = Vec3::new(
            x as f32 * 1.2 - 12.0,
            1.0 + y as f32 * 1.2,
            z as f32 * 1.2 - 6.0,
        );
        world.spawn(RigidBodyDesc::dynamic(shapes[i % 3]).local_position(at))?;
    }
    steps(&mut world, 300)?;
    let mut seed = 0x9e37_79b9_u32;
    let mut random = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1 << 24) as f32
    };
    let filter = QueryFilter::default();
    let mut times = Vec::with_capacity(11_000);
    let mut hits = 0;
    for i in 0..11_000 {
        let origin = Vec3::new(random() * 30.0 - 15.0, 15.0, random() * 16.0 - 8.0);
        let target = Vec3::new(random() * 24.0 - 12.0, 0.0, random() * 12.0 - 6.0);
        let direction = (target - origin).normalize();
        let origin = GridPosition::from_local(origin).ok_or(PhysicsError::OutOfRange)?;
        let started = Instant::now();
        let hit = if i < 10_000 {
            world.raycast(
                &Ray {
                    origin,
                    direction,
                    max_distance: 40.0,
                },
                &filter,
            )?
        } else {
            world.shape_cast(
                &ShapeCast {
                    shape: Shape::sphere(0.2),
                    rotation: Quat::IDENTITY,
                    origin,
                    direction,
                    max_distance: 40.0,
                },
                &filter,
            )?
        };
        times.push(started.elapsed());
        hits += usize::from(hit.is_some());
    }
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "us",
        check: format!("{hits} of 11000 hit, per query"),
    })
}

/// 200 small spheres at 300 m/s into a 1 cm plate.
fn bullets() -> Result<Outcome> {
    let mut world = PhysicsWorld::new(PhysicsSettings {
        gravity: Vec3::ZERO,
        ..PhysicsSettings::default()
    })?;
    world.spawn(
        RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(0.005, 5.0, 5.0)))
            .local_position(Vec3::new(10.0, 0.0, 0.0)),
    )?;
    let mut shots = Vec::new();
    for i in 0..200 {
        let (y, z) = ((i % 20) as f32 * 0.4 - 4.0, (i / 20) as f32 * 0.8 - 4.0);
        let start = -(i as f32) * 0.37;
        shots.push(
            world.spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.02))
                    .local_position(Vec3::new(start, y, z))
                    .linear_velocity(Vec3::new(300.0, 0.0, 0.0)),
            )?,
        );
    }
    let times = steps(&mut world, 120)?;
    let mut through = 0;
    for shot in shots {
        if world.body_state(shot)?.position.local.x > 10.0 {
            through += 1;
        }
    }
    Ok(Outcome {
        bodies: world.body_count(),
        times,
        unit: "ms",
        check: format!("{through} of 200 went through"),
    })
}

/// A tunnel bored a little further each step under boxes lying on hills. Times are one edit
/// each, from asking to the bricks being in the world.
fn tunnel() -> Result<Outcome> {
    let mut world = world()?;
    let started = Instant::now();
    let mut terrain = SdfTerrain::new(
        &mut world,
        SdfDesc {
            origin: GridPosition::ORIGIN,
            voxel: 0.5,
            brick_cells: 16,
            bricks: [16, 6, 16],
            base: Base::Hills {
                seed: 5,
                height: 24.0,
                amplitude: 3.0,
                wavelength: 30.0,
                octaves: 4,
            },
            material: Material::default(),
        },
    )?;
    let built = started.elapsed();
    let mut boxes = Vec::new();
    for i in 0..200 {
        let x = 10.0 + (i % 50) as f32 * 2.2;
        let z = 60.0 + (i / 50) as f32 * 2.5;
        boxes.push(
            world.spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.4)))
                    .local_position(Vec3::new(x, 31.0, z)),
            )?,
        );
    }
    steps(&mut world, 240)?;
    let mut edits = Vec::new();
    let mut step_times = Vec::new();
    for i in 0..400 {
        let x = 4.0 + i as f32 * 0.3;
        let edit = Edit {
            shape: EditShape::Capsule {
                a: Vec3::new(x - 0.3, 22.0, 63.0),
                b: Vec3::new(x, 22.0, 63.0),
                radius: 2.0,
            },
            op: EditOp::Excavate,
        };
        let started = Instant::now();
        DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
            Some(pool) => terrain.edit_with(&mut world, edit, pool),
            None => terrain.edit(&mut world, edit),
        })?;
        edits.push(started.elapsed());
        step_times.extend(steps(&mut world, 1)?);
    }
    let mut fell = 0;
    for body in &boxes {
        if world.body_state(*body)?.position.to_dvec3().y < 0.0 {
            fell += 1;
        }
    }
    step_times.sort_unstable();
    let step_p99 = step_times[step_times.len() * 99 / 100];
    Ok(Outcome {
        bodies: world.body_count(),
        times: edits,
        unit: "ms",
        check: format!(
            "build {:.0} ms, step p99 {:.2} ms, {fell} fell out, {:.0} m3 removed",
            built.as_secs_f64() * 1e3,
            step_p99.as_secs_f64() * 1e3,
            removed(&terrain),
        ),
    })
}

/// The same bore built as a job. Times are the commit alone, what stays between steps when the
/// build runs elsewhere.
fn tunnel_commit() -> Result<Outcome> {
    let mut world = world()?;
    let mut terrain = SdfTerrain::new(
        &mut world,
        SdfDesc {
            origin: GridPosition::ORIGIN,
            voxel: 0.5,
            brick_cells: 16,
            bricks: [16, 6, 16],
            base: Base::Hills {
                seed: 5,
                height: 24.0,
                amplitude: 3.0,
                wavelength: 30.0,
                octaves: 4,
            },
            material: Material::default(),
        },
    )?;
    let mut commits = Vec::new();
    let mut builds = Vec::new();
    for i in 0..400 {
        let x = 4.0 + i as f32 * 0.3;
        terrain.queue_edit(Edit {
            shape: EditShape::Capsule {
                a: Vec3::new(x - 0.3, 22.0, 63.0),
                b: Vec3::new(x, 22.0, 63.0),
                radius: 2.0,
            },
            op: EditOp::Excavate,
        })?;
        let Some(job) = terrain.next_job() else {
            continue;
        };
        let started = Instant::now();
        let built = job.run(&Sequential)?;
        builds.push(started.elapsed());
        let started = Instant::now();
        terrain.commit(&mut world, built)?;
        commits.push(started.elapsed());
    }
    builds.sort_unstable();
    let build_p50 = builds[builds.len() / 2];
    Ok(Outcome {
        bodies: world.body_count(),
        times: commits,
        unit: "ms",
        check: format!(
            "build on one thread p50 {:.2} ms",
            build_p50.as_secs_f64() * 1e3
        ),
    })
}

/// A hole travelling under boxes lying on hills, the ground closing behind it. Times are one
/// move each, build and commit, as often as a step.
fn moving_tunnel() -> Result<Outcome> {
    let mut world = world()?;
    let mut terrain = SdfTerrain::new(
        &mut world,
        SdfDesc {
            origin: GridPosition::ORIGIN,
            voxel: 0.5,
            brick_cells: 16,
            bricks: [16, 6, 16],
            base: Base::Hills {
                seed: 5,
                height: 24.0,
                amplitude: 3.0,
                wavelength: 30.0,
                octaves: 4,
            },
            material: Material::default(),
        },
    )?;
    let mut boxes = Vec::new();
    for i in 0..200 {
        let x = 10.0 + (i % 50) as f32 * 2.2;
        let z = 60.0 + (i / 50) as f32 * 2.5;
        boxes.push(
            world.spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.4)))
                    .local_position(Vec3::new(x, 31.0, z)),
            )?,
        );
    }
    steps(&mut world, 240)?;
    let place = |x: f32| EditShape::Sphere {
        center: Vec3::new(x, 22.0, 63.0),
        radius: 4.0,
    };
    let tunnel = terrain
        .edit(
            &mut world,
            Edit {
                shape: place(4.0),
                op: EditOp::Tunnel,
            },
        )?
        .edit;
    let mut moves = Vec::new();
    let mut rescued = 0;
    for i in 0..400 {
        let started = Instant::now();
        terrain.queue_move_tunnel(tunnel, place(4.0 + i as f32 * 0.3))?;
        if let Some(job) = terrain.next_job() {
            let built = DISPATCH.with(|dispatch| match dispatch.borrow().as_ref() {
                Some(pool) => job.run(pool),
                None => job.run(&Sequential),
            })?;
            terrain.commit(&mut world, built)?;
        }
        moves.push(started.elapsed());
        steps(&mut world, 1)?;
        rescued += terrain.rescue(&mut world)?;
    }
    let mut fell = 0;
    for body in &boxes {
        if world.body_state(*body)?.position.to_dvec3().y < 0.0 {
            fell += 1;
        }
    }
    Ok(Outcome {
        bodies: world.body_count(),
        times: moves,
        unit: "ms",
        check: format!(
            "{fell} fell out, {rescued} rescued, {} edits kept",
            terrain.volume().edit_count()
        ),
    })
}

fn removed(terrain: &SdfTerrain) -> f64 {
    let desc = terrain.volume().desc();
    let full = solunar::SdfVolume::new(*desc).map_or(0.0, |fresh| fresh.material_volume());
    full - terrain.volume().material_volume()
}
