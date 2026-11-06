use algebrix::Vec3;

use super::terrain::terrain_contact;
use super::{FluidSolver, FluidWorld, in_chunks};
use crate::dispatch::JobDispatch;
use crate::error::PhysicsError;

/// Exponent of the Tait equation of state for water.
const TAIT: i32 = 7;
/// Artificial viscosity against walls, on motion into them only. The front of a wave has few
/// neighbors, so its density and pressure read low even at a wall. This is what stops it there.
const WALL_DAMPING: f32 = 1.0;
/// Explicit damping over a substep is stable up to about this share of `radius^2 / dt`, walls
/// and particles among themselves. Long substeps get as much as they can take, more turned still
/// water into a slosh that grew. Short ones never reach it.
const STABLE_WALL_DAMPING: f32 = 0.25;
const STABLE_VISCOSITY: f32 = 0.05;

/// What one [`FluidWorld::step`] did.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FluidReport {
    pub substeps: u32,
    /// The step wanted more substeps than allowed and took bigger ones. Results are less stable.
    pub capped: bool,
    /// Most squeezed particle at the end, as a share of rest density.
    pub max_compression: f32,
    pub max_speed: f32,
    /// Particles that got past a wall's face and were put back. Should stay zero. Ones that only
    /// came closer than their rest distance, half a spacing, are stopped there and not counted:
    /// a sheet of water one particle thick has too little pressure to stand off a floor.
    pub held_back: u32,
    /// Pressure solve iterations, all substeps together. Zero for the weakly compressible solver.
    pub iterations: u32,
    /// Particles emitters added and drains took after the step.
    pub emitted: u32,
    pub drained: u32,
}

impl FluidWorld {
    /// Advances `dt` seconds in as many substeps as the flow needs, with the solver the settings
    /// name.
    ///
    /// # Errors
    /// Fails on a non finite or negative `dt`, or if the dispatcher skipped a job. Particles are
    /// left as the last finished substep put them.
    pub fn step(
        &mut self,
        dt: f32,
        dispatch: &dyn JobDispatch,
    ) -> Result<FluidReport, PhysicsError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(PhysicsError::NonFinite("fluid step"));
        }
        let mut report = FluidReport::default();
        if dt == 0.0 {
            return Ok(report);
        }
        if !self.is_empty() {
            match self.settings.solver {
                FluidSolver::Wcsph => self.wcsph(dt, dispatch, &mut report)?,
                FluidSolver::Dfsph => self.dfsph(dt, dispatch, &mut report)?,
            }
        }
        let flowed = self.run_flows();
        if let Ok((emitted, drained)) = flowed {
            report.emitted = emitted;
            report.drained = drained;
        }
        flowed?;
        report.max_speed = self.fastest();
        let rest = self.settings.rest_density;
        report.max_compression = self
            .densities
            .iter()
            .fold(0.0_f32, |most, density| most.max(density / rest - 1.0));
        Ok(report)
    }

    pub(super) fn fastest(&self) -> f32 {
        self.velocities
            .iter()
            .fold(0.0_f32, |most, velocity| most.max(velocity.length()))
    }

    fn wcsph(
        &mut self,
        dt: f32,
        dispatch: &dyn JobDispatch,
        report: &mut FluidReport,
    ) -> Result<(), PhysicsError> {
        // body points move through the substeps too, none may jump further than water may
        let fastest = self.fastest().max(self.fastest_mover());
        let largest =
            self.settings.courant * self.settings.radius() / (self.settings.sound_speed + fastest);
        let wanted = (dt / largest).ceil().max(1.0);
        let substeps = wanted.min(self.settings.max_substeps as f32) as u32;
        report.capped = wanted > self.settings.max_substeps as f32;
        let small = dt / substeps as f32;
        for index in 0..substeps {
            if !self.coupled.is_empty() {
                self.carry_movers(small * index as f32);
            }
            report.held_back += self.wcsph_substep(small, dispatch)?;
        }
        report.substeps = substeps;
        Ok(())
    }

    fn wcsph_substep(&mut self, dt: f32, dispatch: &dyn JobDispatch) -> Result<u32, PhysicsError> {
        self.update_neighbors(dispatch)?;
        self.compute_densities(dispatch)?;
        let settings = self.settings;
        let rest = settings.rest_density;
        let stiffness = rest * settings.sound_speed.powi(2) / TAIT as f32;
        // no pull at the free surface, sph would clump particles there
        let pressures: Vec<f32> = self
            .densities
            .iter()
            .map(|density| (stiffness * ((density / rest).powi(TAIT) - 1.0)).max(0.0))
            .collect();
        let world = &*self;
        let mass = self.mass;
        let bodies = self.coupled.len();
        let radius = settings.radius();
        let damping = (WALL_DAMPING * settings.sound_speed * radius)
            .min(STABLE_WALL_DAMPING * radius * radius / dt);
        let parts = in_chunks(dispatch, self.len(), |range| {
            let mut felt = vec![(Vec3::ZERO, Vec3::ZERO); bodies];
            let pushes = range
                .map(|index| {
                    let at = world.positions[index];
                    let density = world.densities[index];
                    let own = pressures[index] / (density * density);
                    let mut push = Vec3::ZERO;
                    for &other in world.neighbors.of(index) {
                        let other = other as usize;
                        if other == index {
                            continue;
                        }
                        let their = world.densities[other];
                        let pressure = own + pressures[other] / (their * their);
                        push -=
                            world.kernel.gradient(at - world.positions[other]) * (mass * pressure);
                    }
                    // a wall pushes back with this particle's own pressure, as water would
                    let mirrored = own + pressures[index] / (rest * rest);
                    for &wall in world.wall_neighbors.of(index) {
                        let offset = at - world.walls.positions[wall as usize];
                        push -= world.kernel.gradient(offset) * (mass * mirrored);
                    }
                    world.mover_push(index, mirrored, damping, &mut push, &mut felt);
                    push + world.viscous(index, dt) + settings.gravity
                })
                .collect::<Vec<Vec3>>();
            (pushes, felt)
        })?;
        let mut velocities = self.velocities.iter_mut();
        for (pushes, felt) in parts {
            // pushes first: zip asks its left side first, and must not take a velocity past the chunk
            for (push, velocity) in pushes.into_iter().zip(velocities.by_ref()) {
                *velocity += push * dt;
            }
            for (known, (force, torque)) in self.coupled.iter_mut().zip(felt) {
                known.impulse.0 += force * dt;
                known.impulse.1 += torque * dt;
            }
        }
        self.advance(dt)
    }

    /// Artificial viscosity among particles, and against walls on motion into them only. Against
    /// walls on all motion it was friction that held the dam break front 40% back, with none the
    /// front of a wave got through.
    pub(super) fn viscous(&self, index: usize, dt: f32) -> Vec3 {
        let settings = &self.settings;
        let radius = settings.radius();
        let soften = 0.01 * radius * radius;
        let stable = radius * radius / dt;
        let viscosity =
            (settings.viscosity * settings.sound_speed * radius).min(STABLE_VISCOSITY * stable);
        let at = self.positions[index];
        let velocity = self.velocities[index];
        let density = self.densities[index];
        let mut push = Vec3::ZERO;
        for &other in self.neighbors.of(index) {
            let other = other as usize;
            if other == index {
                continue;
            }
            let offset = at - self.positions[other];
            let toward = (velocity - self.velocities[other]).dot(offset);
            if toward < 0.0 {
                let squeeze = toward / (offset.length_squared() + soften);
                let damp = -viscosity * squeeze * 2.0 / (density + self.densities[other]);
                push -= self.kernel.gradient(offset) * (self.mass * damp);
            }
        }
        let walls = self.wall_neighbors.of(index);
        let away = walls
            .iter()
            .map(|&wall| {
                -self
                    .kernel
                    .gradient(at - self.walls.positions[wall as usize])
            })
            .fold(Vec3::ZERO, |sum, part| sum + part);
        let normal = away.normalize_or(Vec3::ZERO);
        let into = normal * velocity.dot(normal).min(0.0);
        let wall_viscosity =
            (WALL_DAMPING * settings.sound_speed * radius).min(STABLE_WALL_DAMPING * stable);
        for &wall in walls {
            let offset = at - self.walls.positions[wall as usize];
            let toward = into.dot(offset);
            if toward < 0.0 {
                let squeeze = toward / (offset.length_squared() + soften);
                let damp = -wall_viscosity * squeeze / density;
                push -= self.kernel.gradient(offset) * (self.mass * damp);
            }
        }
        push
    }

    /// Moves particles by their velocity. Ones that end past a container wall are put back and
    /// counted.
    pub(super) fn advance(&mut self, dt: f32) -> Result<u32, PhysicsError> {
        let mut held = 0;
        let bounds = self.walls.bounds;
        let rest = Vec3::splat(self.settings.spacing / 2.0);
        for (at, velocity) in self.positions.iter_mut().zip(self.velocities.iter_mut()) {
            let mut next = *at + *velocity * dt;
            if let Some((low, high)) = bounds {
                let inside = next.clamp(low + rest, high - rest);
                if inside != next {
                    if next.clamp(low, high) != next {
                        held += 1;
                    }
                    // the part of the motion into the wall stops
                    if inside.x != next.x {
                        velocity.x = 0.0;
                    }
                    if inside.y != next.y {
                        velocity.y = 0.0;
                    }
                    if inside.z != next.z {
                        velocity.z = 0.0;
                    }
                    next = inside;
                }
            }
            if terrain_contact(&self.terrains, rest.x, &mut next, velocity) {
                held += 1;
            }
            if !next.is_finite() || !velocity.is_finite() {
                return Err(PhysicsError::NonFinite("fluid particle"));
            }
            *at = next;
        }
        Ok(held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::Sequential;
    use crate::fluid::{FluidSettings, FluidSolver};

    const TICK: f32 = 1.0 / 60.0;

    struct Threads(usize);

    impl JobDispatch for Threads {
        fn parallelism(&self) -> usize {
            self.0
        }

        fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
            let workers = self.0;
            std::thread::scope(|scope| {
                for worker in 0..workers {
                    scope.spawn(move || {
                        for index in (0..jobs).rev().filter(|index| index % workers == worker) {
                            job(index);
                        }
                    });
                }
            });
        }
    }

    fn tank(water: Vec3, size: Vec3) -> FluidWorld {
        tank_with(FluidSolver::default(), water, size)
    }

    fn tank_with(solver: FluidSolver, water: Vec3, size: Vec3) -> FluidWorld {
        let mut fluid = FluidWorld::new(FluidSettings {
            solver,
            ..FluidSettings::default()
        })
        .unwrap();
        fluid.add_box_container(Vec3::ZERO, size).unwrap();
        fluid.fill_box(Vec3::ZERO, water).unwrap();
        fluid
    }

    fn inside(fluid: &FluidWorld, size: Vec3) -> bool {
        fluid.positions().iter().all(|at| {
            at.x >= 0.0
                && at.y >= 0.0
                && at.z >= 0.0
                && at.x <= size.x
                && at.y <= size.y
                && at.z <= size.z
        })
    }

    #[test]
    fn still_water_stays_still_and_presses_by_its_depth() {
        let size = Vec3::new(0.5, 0.6, 0.5);
        let depth = 0.4;
        let mut fluid = tank(Vec3::new(0.5, depth, 0.5), size);
        let count = fluid.len();
        let mut held = 0;
        let mut last = FluidReport::default();
        for _ in 0..180 {
            last = fluid.step(TICK, &Sequential).unwrap();
            held += last.held_back;
            assert!(!last.capped);
        }
        assert_eq!(fluid.len(), count);
        assert!(inside(&fluid, size));
        assert_eq!(held, 0, "water got through the walls");
        let mean = fluid.velocities().iter().map(|v| v.length()).sum::<f32>() / count as f32;
        assert!(last.max_speed < 0.2 && mean < 0.03, "{last:?} {mean}");
        // the bottom squeezes by the tait law under the water's weight: 1 + p / b to the 1/7
        let settings = fluid.settings();
        let stiffness = settings.rest_density * settings.sound_speed.powi(2) / 7.0;
        let bottom = settings.rest_density * 9.81 * depth;
        let expected = (1.0 + bottom / stiffness).powf(1.0 / 7.0) - 1.0;
        let squeeze = last.max_compression;
        assert!(
            squeeze > expected * 0.5 && squeeze < expected * 2.0,
            "{squeeze} {expected}"
        );
        // the surface settles about where it started
        let top = fluid
            .positions()
            .iter()
            .fold(0.0_f32, |most, at| most.max(at.y));
        assert!((top - (depth - 0.025)).abs() < 0.05, "{top}");
    }

    #[test]
    fn a_column_collapses_but_never_outruns_the_ideal_front() {
        let size = Vec3::new(1.6, 0.6, 0.2);
        let height = 0.4;
        let mut fluid = tank(Vec3::new(0.4, height, 0.2), size);
        let front = |fluid: &FluidWorld| {
            fluid
                .positions()
                .iter()
                .fold(0.0_f32, |most, at| most.max(at.x))
        };
        let start = front(&fluid);
        let mut held = 0;
        // the dry bed front of an ideal dam break moves at 2 sqrt(g h)
        let ideal = 2.0 * (9.81_f32 * height).sqrt();
        let mut time = 0.0;
        let mut last = start;
        for _ in 0..24 {
            let report = fluid.step(TICK, &Sequential).unwrap();
            held += report.held_back;
            time += TICK;
            let now = front(&fluid);
            assert!(
                now - start <= ideal * time + fluid.settings().spacing,
                "{time} {now}"
            );
            assert!(now >= last - 0.01, "the front went back");
            last = now;
        }
        assert!(last - start > 0.4, "{last}");
        assert!(inside(&fluid, size));
        assert_eq!(held, 0, "water got through the walls");
    }

    /// Martin and Moyce 1952, figure 3, a column twice as tall as wide (n^2 = 2, a = 2.25 in):
    /// t sqrt(2 g / a) against front distance over a. As digitized in pysph's `db_exp_data.py`.
    const MARTIN_MOYCE: [(f32, f32); 10] = [
        (0.832, 1.217),
        (1.219, 1.474),
        (1.997, 2.292),
        (2.547, 2.995),
        (3.345, 4.134),
        (4.034, 4.944),
        (4.418, 5.881),
        (5.091, 6.980),
        (5.685, 7.945),
        (6.306, 8.966),
    ];

    #[test]
    fn the_dam_break_front_follows_martin_and_moyce() {
        let width = 0.3;
        let spacing = 0.3 / 18.0;
        let mut fluid = FluidWorld::new(FluidSettings {
            spacing,
            ..FluidSettings::default()
        })
        .unwrap();
        // long enough that the front never reaches the far wall
        let size = Vec3::new(3.2, 0.8, 0.1);
        fluid.add_box_container(Vec3::ZERO, size).unwrap();
        fluid
            .fill_box(Vec3::ZERO, Vec3::new(width, 2.0 * width, 0.1))
            .unwrap();
        let count = fluid.len();
        let scale = (2.0 * 9.81 / width).sqrt();
        let tick = 1.0 / 240.0;
        let mut time = 0.0_f32;
        let mut errors = Vec::new();
        let mut held = 0;
        for (when, measured) in MARTIN_MOYCE {
            while time * scale < when {
                held += fluid.step(tick, &Sequential).unwrap().held_back;
                time += tick;
            }
            // particle centers sit half a spacing in from the water's edge
            let edge = fluid
                .positions()
                .iter()
                .fold(0.0_f32, |most, at| most.max(at.x));
            let front = (edge + spacing / 2.0) / width;
            let error = (front - measured) / measured;
            // ahead by up to 15% early, about 4% late, never behind
            assert!(
                error < 0.16 && error > -0.03,
                "at {when}: {front} against {measured}"
            );
            errors.push(error);
        }
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        assert!(mean < 0.10, "{errors:?}");
        assert_eq!(fluid.len(), count);
        assert_eq!(held, 0);
        assert!(inside(&fluid, size));
    }

    #[test]
    fn a_gate_holds_until_it_is_taken_away() {
        let size = Vec3::new(1.2, 0.6, 0.2);
        let mut fluid = tank(Vec3::new(0.4, 0.4, 0.2), size);
        let gate = fluid
            .add_wall(Vec3::new(0.4, 0.0, 0.0), Vec3::new(0.5, 0.6, 0.2))
            .unwrap();
        let front = |fluid: &FluidWorld| {
            fluid
                .positions()
                .iter()
                .fold(0.0_f32, |most, at| most.max(at.x))
        };
        for _ in 0..30 {
            fluid.step(TICK, &Sequential).unwrap();
        }
        assert!(front(&fluid) < 0.45, "{}", front(&fluid));
        assert!(fluid.remove_boundary(gate));
        assert!(!fluid.remove_boundary(gate));
        for _ in 0..20 {
            fluid.step(TICK, &Sequential).unwrap();
        }
        assert!(front(&fluid) > 0.7, "{}", front(&fluid));
    }

    #[test]
    fn any_thread_count_ends_on_the_same_state() {
        for solver in [FluidSolver::Wcsph, FluidSolver::Dfsph] {
            let run = |dispatch: &dyn JobDispatch| {
                let water = Vec3::new(0.3, 0.3, 0.2);
                let mut fluid = tank_with(solver, water, Vec3::new(0.9, 0.5, 0.2));
                for _ in 0..12 {
                    fluid.step(TICK, dispatch).unwrap();
                }
                fluid.state_hash()
            };
            let one = run(&Sequential);
            assert_eq!(run(&Threads(4)), one, "{solver:?}");
            assert_eq!(run(&Threads(3)), one, "{solver:?}");
            assert_eq!(run(&Sequential), one, "{solver:?}");
        }
    }

    #[test]
    fn bad_steps_are_refused_and_a_cap_is_reported() {
        let mut fluid = tank(Vec3::splat(0.2), Vec3::splat(0.3));
        assert!(fluid.step(-1.0, &Sequential).is_err());
        assert!(fluid.step(f32::NAN, &Sequential).is_err());
        assert_eq!(fluid.step(0.0, &Sequential).unwrap().substeps, 0);

        let mut tight = FluidWorld::new(FluidSettings {
            max_substeps: 2,
            ..FluidSettings::default()
        })
        .unwrap();
        tight.fill_box(Vec3::ZERO, Vec3::splat(0.1)).unwrap();
        let report = tight.step(TICK, &Sequential).unwrap();
        assert!(report.capped && report.substeps == 2);
        for settings in [
            FluidSettings {
                sound_speed: 0.0,
                ..FluidSettings::default()
            },
            FluidSettings {
                courant: 1.5,
                ..FluidSettings::default()
            },
            FluidSettings {
                viscosity: -1.0,
                ..FluidSettings::default()
            },
        ] {
            assert!(FluidWorld::new(settings).is_err());
        }
    }
}
