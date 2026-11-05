//! Fixed rigid body suites rb-01 to rb-08. `cargo run --release --example bench [rb-01 ...]`.

use std::time::{Duration, Instant};

use solunar::{
    BodyKind, CompoundChild, GridPosition, JointDesc, JointHandle, PhysicsError, PhysicsSettings,
    PhysicsWorld, Quat, QueryFilter, Ray, RigidBodyDesc, Shape, ShapeCast, Vec3,
};

type Result<T> = std::result::Result<T, PhysicsError>;

struct Outcome {
    bodies: usize,
    times: Vec<Duration>,
    unit: &'static str,
    check: String,
}

const SUITES: [(&str, fn() -> Result<Outcome>); 8] = [
    ("rb-01", rb01),
    ("rb-02", rb02),
    ("rb-03", rb03),
    ("rb-04", rb04),
    ("rb-05", rb05),
    ("rb-06", rb06),
    ("rb-07", rb07),
    ("rb-08", rb08),
];

fn main() {
    let wanted: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "solunar {} {} {}, {} threads seen, sequential solver",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |count| count.get()),
    );
    println!("| suite | bodies | samples | p50 | p95 | p99 | max | check |");
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    for (name, suite) in SUITES {
        if !wanted.is_empty() && !wanted.iter().any(|want| want == name) {
            continue;
        }
        match suite() {
            Ok(outcome) => print_row(name, &outcome),
            Err(error) => println!("| {name} | failed: {error} | | | | | | |"),
        }
    }
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
        let started = Instant::now();
        world.step_fixed()?;
        times.push(started.elapsed());
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
fn rb01() -> Result<Outcome> {
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
            "{} awake, fastest {:.2} m/s",
            world.body_count() - 1 - world.sleeping_body_count(),
            max_speed(&world)?
        ),
    })
}

/// 10,000 boxes resting apart, asleep, and 100 balls dropped among them.
fn rb02() -> Result<Outcome> {
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

/// 30 stacks of 10 boxes.
fn rb03() -> Result<Outcome> {
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
fn rb04() -> Result<Outcome> {
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
fn rb05() -> Result<Outcome> {
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
fn rb06() -> Result<Outcome> {
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

/// The rb-01 pile at rest, then 10,000 rays and 1,000 sphere casts, timed one by one.
fn rb07() -> Result<Outcome> {
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
fn rb08() -> Result<Outcome> {
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
