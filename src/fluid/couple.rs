use algebrix::Vec3;

use super::{CellKey, FluidSolver, FluidWorld, Grid};
use crate::body::BodyHandle;
use crate::dispatch::JobDispatch;
use crate::error::PhysicsError;
use crate::shape::Shape;
use crate::world::PhysicsWorld;

use super::step::FluidReport;

/// A body the water and the rigid world share. Its points are frozen water on its own lattice,
/// as deep under its surface as the kernel reaches, in body axes about its center.
#[derive(Debug, Clone)]
pub(super) struct Coupled {
    body: BodyHandle,
    points: Vec<Vec3>,
    /// Where the body was this step, for torque.
    pub(super) center: Vec3,
    /// What the water did to it this step: impulse and angular impulse about `center`.
    pub(super) impulse: (Vec3, Vec3),
}

/// The points of every coupled body, placed for this step. Bodies hold still while the water
/// takes its substeps, the exchange happens once a step.
#[derive(Debug, Clone, Default)]
pub(super) struct Movers {
    pub(super) positions: Vec<Vec3>,
    pub(super) velocities: Vec<Vec3>,
    /// Index into the coupled bodies.
    pub(super) owners: Vec<u32>,
    pub(super) grid: Grid,
}

impl FluidWorld {
    /// Lets water and `body` push each other in [`Self::step_with_bodies`]. The body keeps the
    /// shape given here.
    ///
    /// # Errors
    /// Fails on a shape other than a box, ball or capsule, one too small to hold a point, or too
    /// many points.
    pub fn couple(&mut self, body: BodyHandle, shape: &Shape) -> Result<(), PhysicsError> {
        let points = self.sample(shape)?;
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
            impulse: (Vec3::ZERO, Vec3::ZERO),
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

    /// Lattice points inside the shape, no deeper than the kernel reaches, centered on it.
    fn sample(&self, shape: &Shape) -> Result<Vec<Vec3>, PhysicsError> {
        let spacing = self.settings.spacing;
        let reach = self.settings.radius();
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
        if total > self.settings.max_particles as i64 {
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
        self.coupled
            .retain(|known| world.body_state(known.body).is_ok());
        let mut placed: Vec<(CellKey, u32, Vec3, Vec3)> = Vec::new();
        let radius = self.settings.radius();
        for (owner, known) in self.coupled.iter_mut().enumerate() {
            let Ok(state) = world.body_state(known.body) else {
                continue;
            };
            let center = state.position.to_dvec3().as_vec3();
            known.center = center;
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
        // stable on equal keys, so points keep the order they were listed in
        placed.sort_by_key(|item| (item.0, item.1));
        self.movers = Movers {
            positions: placed.iter().map(|item| item.2).collect(),
            velocities: placed.iter().map(|item| item.3).collect(),
            owners: placed.iter().map(|item| item.1).collect(),
            grid: Grid::build(placed.iter().map(|item| item.0)),
        };
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
            let arm = self.movers.positions[point] - self.coupled[owner].center;
            bodies[owner].0 += force;
            bodies[owner].1 += arm.cross(force);
        }
    }
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
