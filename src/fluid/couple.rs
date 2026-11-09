use algebrix::Vec3;

use super::{CellKey, FluidSolver, FluidWorld, Grid};
use crate::body::BodyHandle;
use crate::dispatch::JobDispatch;
use crate::error::PhysicsError;
use crate::shape::Shape;
use crate::snapshot::{Reader, SnapshotError, Writer};
use crate::substep::{SubstepBodies, SubstepHook, SubstepPose};
use crate::world::PhysicsWorld;

use super::step::FluidReport;

/// A body the water and the rigid world share. Its points are frozen water on its own lattice,
/// as deep under its surface as the kernel reaches, in body axes about its center.
#[derive(Debug, Clone)]
pub(super) struct Coupled {
    body: BodyHandle,
    points: Vec<Vec3>,
    /// Where the body was at the start of this step and how fast it went, for torque.
    pub(super) center: Vec3,
    velocity: Vec3,
    /// What the water did to it this step: impulse and angular impulse about `center`.
    pub(super) impulse: (Vec3, Vec3),
    /// How far the center has gone this step.
    moved: Vec3,
}

/// The points of every coupled body for this step. Through the water's substeps they move on
/// at their body's speed from where the step began, so the water sees a body sweep through. Held
/// still, a fast body jumped a third of a meter into the water each tick and the pressure blew
/// bricks apart. The exchange of pushes still happens once a step.
#[derive(Debug, Clone, Default)]
pub(super) struct Movers {
    /// Where each point was when the step began.
    start: Vec<Vec3>,
    pub(super) positions: Vec<Vec3>,
    pub(super) velocities: Vec<Vec3>,
    /// Index into the coupled bodies.
    pub(super) owners: Vec<u32>,
    pub(super) grid: Grid,
}

impl Coupled {
    pub(super) fn save(&self, out: &mut Writer) {
        out.body(self.body);
        out.len(self.points.len());
        for point in &self.points {
            out.vec3(*point);
        }
    }

    pub(super) fn load(reader: &mut Reader<'_>, max: usize) -> Result<Self, SnapshotError> {
        let body = reader.handle()?;
        let count = reader.count(12, max)?;
        let mut points = Vec::with_capacity(count);
        for _ in 0..count {
            let point = reader.vec3()?;
            if !point.is_finite() {
                return Err(SnapshotError::Invalid("fluid body point"));
            }
            points.push(point);
        }
        if points.is_empty() {
            return Err(SnapshotError::Invalid("fluid body"));
        }
        Ok(Self {
            body,
            points,
            center: Vec3::ZERO,
            velocity: Vec3::ZERO,
            impulse: (Vec3::ZERO, Vec3::ZERO),
            moved: Vec3::ZERO,
        })
    }

    pub(super) fn body(&self) -> BodyHandle {
        self.body
    }

    pub(super) fn point_count(&self) -> usize {
        self.points.len()
    }
}

/// The water as a [`SubstepHook`]: each rigid substep it steps the substep's time with the
/// bodies where the substep has them, and pushes them before that substep's solve, so the
/// tick's lag of [`FluidWorld::step_with_bodies`] goes. Not ready: a body held only by water
/// spins more this way, see the tests.
pub struct FluidHook<'a> {
    fluid: &'a mut FluidWorld,
    /// What the water did over the rigid step so far, substeps added, highs kept.
    pub report: FluidReport,
}

impl FluidWorld {
    /// Passes this water to [`PhysicsWorld::step_with_hooks`].
    pub fn hook(&mut self) -> FluidHook<'_> {
        FluidHook {
            fluid: self,
            report: FluidReport::default(),
        }
    }
}

impl SubstepHook for FluidHook<'_> {
    fn before_solve(
        &mut self,
        bodies: &mut SubstepBodies<'_>,
        dt: f32,
        dispatch: &dyn JobDispatch,
    ) -> Result<(), PhysicsError> {
        let fluid = &mut *self.fluid;
        if fluid.settings.solver != FluidSolver::Wcsph && !fluid.coupled.is_empty() {
            return Err(PhysicsError::InvalidSettings);
        }
        fluid.place_movers_at(|body| bodies.pose(body));
        let stepped = fluid.step(dt, dispatch);
        fluid.movers = Movers::default();
        let step = stepped?;
        let total = &mut self.report;
        total.substeps += step.substeps;
        total.capped |= step.capped;
        total.max_compression = total.max_compression.max(step.max_compression);
        total.max_speed = total.max_speed.max(step.max_speed);
        total.held_back += step.held_back;
        total.iterations += step.iterations;
        total.emitted += step.emitted;
        total.drained += step.drained;
        total.lost += step.lost;
        total.sound_speed = total.sound_speed.max(step.sound_speed);
        for known in &mut fluid.coupled {
            let (impulse, angular) = std::mem::take(&mut known.impulse);
            bodies.push(known.body, impulse, angular)?;
        }
        Ok(())
    }
}

impl FluidWorld {
    /// Lets water and `body` push each other in [`Self::step_with_bodies`]. The body keeps the
    /// shape given here.
    ///
    /// # Errors
    /// Fails on a shape other than a box, ball or capsule, one too small to hold a point, or too
    /// many points.
    pub fn couple(&mut self, body: BodyHandle, shape: &Shape) -> Result<(), PhysicsError> {
        let settings = &self.settings;
        let points = shape_points(
            shape,
            settings.spacing,
            settings.radius(),
            settings.max_particles,
        )?;
        if points.is_empty() {
            return Err(PhysicsError::InvalidShape);
        }
        let room = self
            .settings
            .max_particles
            .saturating_sub(self.coupled_points());
        if points.len() > room {
            return Err(PhysicsError::FluidFull);
        }
        self.coupled.retain(|known| known.body != body);
        self.coupled.push(Coupled {
            body,
            points,
            center: Vec3::ZERO,
            velocity: Vec3::ZERO,
            impulse: (Vec3::ZERO, Vec3::ZERO),
            moved: Vec3::ZERO,
        });
        self.coupled.sort_by_key(|known| known.body);
        Ok(())
    }

    /// Returns false for a body that was not coupled.
    pub fn uncouple(&mut self, body: BodyHandle) -> bool {
        let before = self.coupled.len();
        self.coupled.retain(|known| known.body != body);
        self.coupled.len() != before
    }

    fn coupled_points(&self) -> usize {
        self.coupled.iter().map(|known| known.points.len()).sum()
    }

    /// Advances the water `dt` seconds with the coupled bodies standing where `world` has them,
    /// then hands each dynamic body the push the water gave it, as force and torque for the
    /// world's next step. Call it right before that step. Bodies gone from the world are dropped.
    ///
    /// # Errors
    /// Fails like [`Self::step`], or for the divergence free solver, which does not couple yet.
    pub fn step_with_bodies(
        &mut self,
        dt: f32,
        world: &mut PhysicsWorld,
        dispatch: &dyn JobDispatch,
    ) -> Result<FluidReport, PhysicsError> {
        if self.settings.solver != FluidSolver::Wcsph && !self.coupled.is_empty() {
            return Err(PhysicsError::InvalidSettings);
        }
        self.place_movers(world);
        let report = self.step(dt, dispatch);
        self.movers = Movers::default();
        let report = report?;
        if dt > 0.0 {
            for known in &mut self.coupled {
                let (impulse, angular) = std::mem::take(&mut known.impulse);
                // a fixed or kinematic body pushes water but is not pushed back
                if world.apply_force(known.body, impulse / dt).is_ok() {
                    world.apply_torque(known.body, angular / dt)?;
                }
            }
        }
        Ok(report)
    }

    fn place_movers(&mut self, world: &PhysicsWorld) {
        self.place_movers_at(|body| {
            let state = world.body_state(body).ok()?;
            Some(SubstepPose {
                position: state.position,
                rotation: state.rotation,
                linear_velocity: state.linear_velocity,
                angular_velocity: state.angular_velocity,
            })
        });
    }

    /// Body points where `pose` says their bodies are. A body it does not know is uncoupled.
    fn place_movers_at(&mut self, pose: impl Fn(BodyHandle) -> Option<SubstepPose>) {
        self.coupled.retain(|known| pose(known.body).is_some());
        let mut placed: Vec<(CellKey, u32, Vec3, Vec3)> = Vec::new();
        let radius = self.settings.radius();
        for (owner, known) in self.coupled.iter_mut().enumerate() {
            let Some(state) = pose(known.body) else {
                continue;
            };
            let center = state.position.to_dvec3().as_vec3();
            known.center = center;
            known.velocity = state.linear_velocity;
            known.impulse = (Vec3::ZERO, Vec3::ZERO);
            for local in &known.points {
                let arm = state.rotation.rotate_vec3(*local);
                let at = center + arm;
                let velocity = state.linear_velocity + state.angular_velocity.cross(arm);
                let cell = at / radius;
                let key = (
                    cell.x.floor() as i32,
                    cell.y.floor() as i32,
                    cell.z.floor() as i32,
                );
                placed.push((key, owner as u32, at, velocity));
            }
        }
        self.movers = Movers {
            start: placed.iter().map(|item| item.2).collect(),
            positions: placed.iter().map(|item| item.2).collect(),
            velocities: placed.iter().map(|item| item.3).collect(),
            owners: placed.iter().map(|item| item.1).collect(),
            grid: Grid::default(),
        };
        self.carry_movers(0.0);
    }

    /// Moves body points on to `elapsed` seconds into the step, and sorts them by cell again.
    pub(super) fn carry_movers(&mut self, elapsed: f32) {
        let radius = self.settings.radius();
        let movers = &mut self.movers;
        let mut order: Vec<(CellKey, u32, usize)> = (0..movers.start.len())
            .map(|index| {
                let at = movers.start[index] + movers.velocities[index] * elapsed;
                movers.positions[index] = at;
                let cell = at / radius;
                let key = (
                    cell.x.floor() as i32,
                    cell.y.floor() as i32,
                    cell.z.floor() as i32,
                );
                (key, movers.owners[index], index)
            })
            .collect();
        // owner and place in the body's list break ties, so the order never depends on threads
        order.sort_unstable();
        let pick = |list: &[Vec3]| order.iter().map(|item| list[item.2]).collect::<Vec<Vec3>>();
        movers.start = pick(&movers.start);
        movers.positions = pick(&movers.positions);
        movers.velocities = pick(&movers.velocities);
        movers.owners = order.iter().map(|item| item.1).collect();
        movers.grid = Grid::build(order.iter().map(|item| item.0));
        for known in &mut self.coupled {
            known.moved = known.velocity * elapsed;
        }
    }

    /// The fastest body point, for the substep length.
    pub(super) fn fastest_mover(&self) -> f32 {
        self.movers
            .velocities
            .iter()
            .fold(0.0_f32, |most, velocity| most.max(velocity.length()))
    }

    /// Pressure and damping between particle `index` and the body points near it. Adds what the
    /// particle feels to `push`, and what each body feels to `bodies`.
    pub(super) fn mover_push(
        &self,
        index: usize,
        mirrored: f32,
        damping: f32,
        push: &mut Vec3,
        bodies: &mut [(Vec3, Vec3)],
    ) {
        let near = self.mover_neighbors.of(index);
        if near.is_empty() {
            return;
        }
        let at = self.positions[index];
        let velocity = self.velocities[index];
        let density = self.densities[index];
        let radius = self.settings.radius();
        let soften = 0.01 * radius * radius;
        for &point in near {
            let point = point as usize;
            let offset = at - self.movers.positions[point];
            let relative = velocity - self.movers.velocities[point];
            let toward = relative.dot(offset);
            let damp = if toward < 0.0 {
                -damping * toward / (offset.length_squared() + soften) / density
            } else {
                0.0
            };
            let part = -self.kernel.gradient(offset) * (self.mass * (mirrored + damp));
            *push += part;
            // the body gets the opposite of what this particle got
            let force = -part * self.mass;
            let owner = self.movers.owners[point] as usize;
            let known = &self.coupled[owner];
            let arm = self.movers.positions[point] - (known.center + known.moved);
            bodies[owner].0 += force;
            bodies[owner].1 += arm.cross(force);
        }
    }
}

/// Lattice points `spacing` apart inside the shape, no deeper than `reach`, centered on it. Fails
/// past `limit` points or on a shape it can not fill.
pub(super) fn shape_points(
    shape: &Shape,
    spacing: f32,
    reach: f32,
    limit: usize,
) -> Result<Vec<Vec3>, PhysicsError> {
    let (half, depth): (Vec3, Box<dyn Fn(Vec3) -> f32>) = match *shape {
        Shape::Cuboid { half_extents } => (
            half_extents,
            Box::new(move |at: Vec3| {
                let room = half_extents - at.abs();
                room.x.min(room.y).min(room.z)
            }),
        ),
        Shape::Sphere { radius } => (
            Vec3::splat(radius),
            Box::new(move |at: Vec3| radius - at.length()),
        ),
        Shape::Capsule {
            radius,
            half_height,
        } => (
            Vec3::new(radius, half_height + radius, radius),
            Box::new(move |at: Vec3| {
                let y = at.y.clamp(-half_height, half_height);
                radius - (at - Vec3::new(0.0, y, 0.0)).length()
            }),
        ),
        _ => return Err(PhysicsError::InvalidShape),
    };
    if !half.is_finite() || half.min_element() <= 0.0 {
        return Err(PhysicsError::InvalidShape);
    }
    let count = |extent: f32| ((2.0 * extent / spacing).floor() as i64).clamp(0, 1 << 12);
    let cells = (count(half.x), count(half.y), count(half.z));
    let total = cells.0 * cells.1 * cells.2;
    if total > limit as i64 {
        return Err(PhysicsError::FluidFull);
    }
    let start = |cells: i64| -(cells as f32 - 1.0) * spacing / 2.0;
    let origin = Vec3::new(start(cells.0), start(cells.1), start(cells.2));
    let mut points = Vec::new();
    for y in 0..cells.1 {
        for z in 0..cells.2 {
            for x in 0..cells.0 {
                let at = origin + Vec3::new(x as f32, y as f32, z as f32) * spacing;
                let inside = depth(at);
                if inside >= 0.0 && inside <= reach {
                    points.push(at);
                }
            }
        }
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBodyDesc;
    use crate::dispatch::Sequential;
    use crate::fluid::FluidSettings;
    use crate::world::PhysicsSettings;

    const TICK: f32 = 1.0 / 60.0;

    fn pool() -> (FluidWorld, PhysicsWorld) {
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
        let size = Vec3::new(1.0, 1.2, 1.0);
        fluid.add_box_container(Vec3::ZERO, size).unwrap();
        fluid
            .fill_box(Vec3::ZERO, Vec3::new(1.0, 0.6, 1.0))
            .unwrap();
        let world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        (fluid, world)
    }

    fn drop_box(fluid: &mut FluidWorld, world: &mut PhysicsWorld, density: f32) -> BodyHandle {
        let shape = Shape::cuboid(Vec3::splat(0.15));
        let desc = RigidBodyDesc::dynamic(shape)
            .density(density)
            .local_position(Vec3::new(0.5, 0.9, 0.5));
        let body = world.spawn(desc).unwrap();
        fluid.couple(body, &shape).unwrap();
        body
    }

    #[test]
    fn a_ball_faster_than_the_water_sound_speed_does_not_blow_it_apart() {
        let (mut fluid, mut world) = pool();
        // open above, so the splash may fly, taken out past the domain
        fluid
            .set_domain(Some((Vec3::splat(-5.0), Vec3::splat(6.0))))
            .unwrap();
        let shape = Shape::sphere(0.1);
        let speed = 24.0;
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(shape)
                    .density(7800.0)
                    .local_position(Vec3::new(0.5, 0.75, 0.5))
                    .linear_velocity(Vec3::new(0.0, -speed, 0.0)),
            )
            .unwrap();
        fluid.couple(ball, &shape).unwrap();
        assert!(FluidSettings::default().sound_speed < speed);
        let (mut fastest, mut squeeze, mut louder) = (0.0_f32, 0.0_f32, 0.0_f32);
        for _ in 0..30 {
            let report = fluid
                .step_with_bodies(TICK, &mut world, &Sequential)
                .unwrap();
            world.step_fixed().unwrap();
            fastest = fastest.max(report.max_speed);
            squeeze = squeeze.max(report.max_compression);
            louder = louder.max(report.sound_speed);
        }
        // measured before the sound speed followed the hit: 0.6 squeeze, water at 2.75 times the ball
        assert!(louder >= 4.0 * speed, "{louder}");
        assert!(squeeze < 0.2, "{squeeze}");
        assert!(fastest < 2.0 * speed, "{fastest}");
    }

    #[test]
    fn a_body_added_inside_the_water_pushes_it_aside_without_throwing_it() {
        let (mut fluid, mut world) = pool();
        // half under the surface at 0.6 m, at rest, as a cannon fired from the surface would be
        let shape = Shape::sphere(0.2);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(shape)
                    .density(7800.0)
                    .local_position(Vec3::new(0.5, 0.6, 0.5)),
            )
            .unwrap();
        fluid.couple(ball, &shape).unwrap();
        let (mut fastest, mut last) = (0.0_f32, 0.0_f32);
        for _ in 0..90 {
            let report = fluid
                .step_with_bodies(TICK, &mut world, &Sequential)
                .unwrap();
            world.step_fixed().unwrap();
            fastest = fastest.max(report.max_speed);
            last = report.max_speed;
        }
        // a burst as the ball's half volume of water leaves it at once, then calm. measured
        // 710 m/s before pressure stopped growing past 10% over rest, 17 while fast water raised
        // its own sound speed, 9.4 now
        assert!(fastest < 12.0, "{fastest}");
        assert!(last < 2.0, "{last}");
    }

    /// A brick dropped into a 1 m deep pool, and its fastest motion in the fifth second, settled
    /// on the floor by then.
    fn brick_under_water(hooked: bool) -> (f32, f32, f32) {
        let mut fluid = FluidWorld::new(FluidSettings {
            spacing: 0.1,
            ..FluidSettings::default()
        })
        .unwrap();
        fluid
            .add_box_container(Vec3::ZERO, Vec3::new(3.0, 2.0, 2.0))
            .unwrap();
        fluid
            .fill_box(Vec3::ZERO, Vec3::new(3.0, 1.0, 2.0))
            .unwrap();
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(5.0, 0.5, 5.0)))
                    .local_position(Vec3::new(1.5, -0.5, 1.0)),
            )
            .unwrap();
        let shape = Shape::cuboid(Vec3::new(0.2, 0.15, 0.35));
        let brick = world
            .spawn(
                RigidBodyDesc::dynamic(shape)
                    .density(2400.0)
                    .local_position(Vec3::new(1.5, 1.3, 1.0)),
            )
            .unwrap();
        fluid.couple(brick, &shape).unwrap();
        let (mut speed, mut spin, mut highest) = (0.0_f32, 0.0_f32, 0.0_f32);
        for tick in 0..300 {
            if hooked {
                let mut hook = fluid.hook();
                world
                    .step_with_hooks(&Sequential, &mut [&mut hook])
                    .unwrap();
            } else {
                fluid
                    .step_with_bodies(TICK, &mut world, &Sequential)
                    .unwrap();
                world.step_fixed().unwrap();
            }
            let state = world.body_state(brick).unwrap();
            if tick >= 240 {
                speed = speed.max(state.linear_velocity.length());
                spin = spin.max(state.angular_velocity.length());
                highest = highest.max(state.position.to_dvec3().y as f32 - 0.15);
            }
        }
        (speed, spin, highest)
    }

    #[test]
    fn pushing_inside_the_substeps_takes_out_the_tick_lag_bounce() {
        // measured in the fifth second, speed m/s, spin rad/s, off the floor m: once a tick
        // 0.21, 0.05, 0.05. inside the substeps 0.07, 1.6, 0.05. the lag's bounce goes, the half
        // spacing layer of water under the brick stays and the spin it floats with grows
        let (speed, _, gap) = brick_under_water(true);
        let (lagged, _, lagged_gap) = brick_under_water(false);
        assert!(speed < lagged / 2.0, "{speed} against {lagged}");
        assert!(gap < 0.06 && lagged_gap < 0.06, "{gap} {lagged_gap}");
    }

    #[test]
    fn a_floating_box_and_its_water_step_on_the_same_after_snapshots() {
        let (mut fluid, mut world) = pool();
        drop_box(&mut fluid, &mut world, 500.0);
        run(&mut fluid, &mut world, 20);
        let mut loaded_fluid = FluidWorld::load_snapshot(&fluid.save_snapshot()).unwrap();
        let mut loaded_world = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        run(&mut fluid, &mut world, 30);
        run(&mut loaded_fluid, &mut loaded_world, 30);
        assert_eq!(loaded_fluid.state_hash(), fluid.state_hash());
        assert_eq!(loaded_world.state_hash(), world.state_hash());
    }

    fn run(fluid: &mut FluidWorld, world: &mut PhysicsWorld, ticks: usize) -> f32 {
        for _ in 0..ticks {
            fluid.step_with_bodies(TICK, world, &Sequential).unwrap();
            world.step_fixed().unwrap();
        }

        fluid
            .positions()
            .iter()
            .fold(0.0_f32, |most, at| most.max(at.y))
    }

    fn height(world: &PhysicsWorld, body: BodyHandle) -> f32 {
        world.body_state(body).unwrap().position.local.y
    }

    #[test]
    fn a_light_box_floats_and_a_heavy_one_sinks() {
        let (mut light_water, mut light_world) = pool();
        let light = drop_box(&mut light_water, &mut light_world, 400.0);
        run(&mut light_water, &mut light_world, 180);
        let (mut heavy_water, mut heavy_world) = pool();
        let heavy = drop_box(&mut heavy_water, &mut heavy_world, 3000.0);
        run(&mut heavy_water, &mut heavy_world, 180);
        let (floats, sinks) = (height(&light_world, light), height(&heavy_world, heavy));
        // the water's surface is near 0.6
        assert!(floats > 0.45, "{floats}");
        assert!(sinks < 0.3, "{sinks}");
        // still in the water, not thrown out
        assert!(floats < 0.9, "{floats}");
        assert_eq!(light_water.len(), 20 * 12 * 20);
    }

    #[test]
    fn a_falling_box_lifts_the_water_and_is_slowed_by_it() {
        let (mut fluid, mut world) = pool();
        let still = run(&mut fluid, &mut world, 30);
        let body = drop_box(&mut fluid, &mut world, 3000.0);
        let mut slowest = 0.0_f32;
        let mut free = 0.0_f32;
        for tick in 0..40 {
            fluid
                .step_with_bodies(TICK, &mut world, &Sequential)
                .unwrap();
            world.step_fixed().unwrap();
            let speed = -world.body_state(body).unwrap().linear_velocity.y;
            free = 9.81 * TICK * (tick + 1) as f32;
            slowest = slowest.max(speed);
        }
        // never as fast as falling through air for the same time
        assert!(slowest < free * 0.8, "{slowest} {free}");
        let top = fluid
            .positions()
            .iter()
            .fold(0.0_f32, |most, at| most.max(at.y));
        assert!(top > still + 0.01, "{top} {still}");
    }

    #[test]
    fn coupling_is_the_same_on_any_thread_count() {
        struct Threads;
        impl JobDispatch for Threads {
            fn parallelism(&self) -> usize {
                4
            }
            fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
                std::thread::scope(|scope| {
                    for worker in 0..4 {
                        scope.spawn(move || {
                            for index in (0..jobs).rev().filter(|index| index % 4 == worker) {
                                job(index);
                            }
                        });
                    }
                });
            }
        }
        let go = |dispatch: &dyn JobDispatch| {
            let (mut fluid, mut world) = pool();
            let body = drop_box(&mut fluid, &mut world, 800.0);
            for _ in 0..20 {
                fluid.step_with_bodies(TICK, &mut world, dispatch).unwrap();
                world.step_fixed().unwrap();
            }
            (
                fluid.state_hash(),
                world.body_state(body).unwrap().position.local.y.to_bits(),
            )
        };
        assert_eq!(go(&Sequential), go(&Threads));
    }

    #[test]
    fn shapes_sample_and_bad_ones_are_refused() {
        let (mut fluid, mut world) = pool();
        let body = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.2)))
            .unwrap();
        for shape in [
            Shape::sphere(0.2),
            Shape::capsule(0.1, 0.2),
            Shape::cuboid(Vec3::new(0.3, 0.1, 0.1)),
        ] {
            fluid.couple(body, &shape).unwrap();
            assert_eq!(fluid.coupled.len(), 1);
        }
        assert!(fluid.couple(body, &Shape::sphere(0.001)).is_err());
        assert!(fluid.couple(body, &Shape::sphere(-1.0)).is_err());
        assert!(fluid.uncouple(body));
        assert!(!fluid.uncouple(body));
        // a body gone from the world is dropped
        fluid.couple(body, &Shape::sphere(0.2)).unwrap();
        world.despawn(body).unwrap();
        fluid
            .step_with_bodies(TICK, &mut world, &Sequential)
            .unwrap();
        assert!(fluid.coupled.is_empty());
    }
}
