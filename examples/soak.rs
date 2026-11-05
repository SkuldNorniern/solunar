//! Steps a world with bodies coming and going for a long time and prints whether anything creeps.
//! `cargo run --release --example soak [seconds] [threads]`.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use solunar::{
    BodyHandle, JobDispatch, PhysicsError, PhysicsSettings, PhysicsWorld, RigidBodyDesc,
    Sequential, Shape, Vec3,
};

const CHURN_PER_STEP: usize = 2;
const LIVE: usize = 1500;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let seconds: u64 = args.next().map_or(Ok(60), |text| text.parse())?;
    let threads: usize = args.next().map_or(Ok(1), |text| text.parse())?;
    let pool = if threads > 1 {
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        Some(Pool(built))
    } else {
        None
    };
    let dispatch: &dyn JobDispatch = match &pool {
        Some(pool) => pool,
        None => &Sequential,
    };
    println!("{seconds} s on {threads} threads, {LIVE} bodies, {CHURN_PER_STEP} replaced a step");
    println!("| at s | steps | bodies | asleep | contacts | islands | p50 ms | p99 ms | max ms |");
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    soak(seconds, dispatch)?;
    Ok(())
}

fn soak(seconds: u64, dispatch: &dyn JobDispatch) -> Result<(), PhysicsError> {
    let mut world = PhysicsWorld::new(PhysicsSettings::default())?;
    world.spawn(
        RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(40.0, 1.0, 40.0)))
            .local_position(Vec3::new(0.0, -1.0, 0.0)),
    )?;
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut live: VecDeque<BodyHandle> = VecDeque::new();
    let started = Instant::now();
    let mut times: Vec<Duration> = Vec::new();
    let mut next_report = Duration::from_secs(seconds.div_ceil(12).max(1));
    let report_every = next_report;
    let mut steps = 0_u64;
    let mut contacts = Vec::new();
    while started.elapsed() < Duration::from_secs(seconds) {
        let wanted = if live.len() < LIVE { 8 } else { CHURN_PER_STEP };
        for _ in 0..wanted {
            let mut next = |low: f32, high: f32| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                low + (seed % 10_000) as f32 / 10_000.0 * (high - low)
            };
            let at = Vec3::new(next(-25.0, 25.0), next(4.0, 20.0), next(-25.0, 25.0));
            let shape = match (next(0.0, 3.0)) as u32 {
                0 => Shape::sphere(next(0.15, 0.5)),
                1 => Shape::cuboid(Vec3::splat(next(0.15, 0.5))),
                _ => Shape::capsule(next(0.1, 0.3), next(0.1, 0.4)),
            };
            live.push_back(world.spawn(RigidBodyDesc::dynamic(shape).local_position(at))?);
            if live.len() > LIVE
                && let Some(oldest) = live.pop_front()
            {
                world.despawn(oldest)?;
            }
        }
        let step = Instant::now();
        world.step_with(dispatch)?;
        times.push(step.elapsed());
        steps += 1;
        if started.elapsed() >= next_report {
            next_report += report_every;
            times.sort_unstable();
            let at = |percent: usize| {
                let rank = (times.len() * percent).div_ceil(100).max(1);
                times[rank - 1].as_secs_f64() * 1e3
            };
            world.contacts(&mut contacts);
            println!(
                "| {} | {steps} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2} |",
                started.elapsed().as_secs(),
                world.body_count(),
                world.sleeping_body_count(),
                contacts.len(),
                world.island_stats().count,
                at(50),
                at(99),
                at(100),
            );
            times.clear();
        }
    }
    Ok(())
}
