use algebrix::Vec3;

use super::{FluidWorld, in_chunks};
use crate::dispatch::JobDispatch;
use crate::error::PhysicsError;

/// Exponent of the Tait equation of state for water.
const TAIT: i32 = 7;

/// What one [`FluidWorld::step`] did.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FluidReport {
    pub substeps: u32,
    /// The step wanted more substeps than allowed and took bigger ones. Results are less stable.
    pub capped: bool,
    /// Most squeezed particle at the end, as a share of rest density.
    pub max_compression: f32,
    pub max_speed: f32,
    /// Particles that got past a wall and were put back inside. Should stay zero.
    pub held_back: u32,
}

impl FluidWorld {
    /// Advances `dt` seconds in as many substeps as the flow needs, weakly compressible sph.
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
        if dt == 0.0 || self.is_empty() {
            return Ok(report);
        }
        let fastest = self.fastest();
        let largest =
            self.settings.courant * self.settings.radius() / (self.settings.sound_speed + fastest);
        let wanted = (dt / largest).ceil().max(1.0);
        let substeps = wanted.min(self.settings.max_substeps as f32) as u32;
        report.capped = wanted > self.settings.max_substeps as f32;
        let small = dt / substeps as f32;
        for _ in 0..substeps {
            report.held_back += self.substep(small, dispatch)?;
        }
        report.substeps = substeps;
        report.max_speed = self.fastest();
        let rest = self.settings.rest_density;
        report.max_compression = self
            .densities
            .iter()
            .fold(0.0_f32, |most, density| most.max(density / rest - 1.0));
        Ok(report)
    }

    fn fastest(&self) -> f32 {
        self.velocities
            .iter()
            .fold(0.0_f32, |most, velocity| most.max(velocity.length()))
    }

    fn substep(&mut self, dt: f32, dispatch: &dyn JobDispatch) -> Result<u32, PhysicsError> {
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
        let radius = settings.radius();
        let viscosity = settings.viscosity * settings.sound_speed * radius;
        let parts = in_chunks(dispatch, self.len(), |range| {
            range
                .map(|index| {
                    let at = world.positions[index];
                    let velocity = world.velocities[index];
                    let density = world.densities[index];
                    let own = pressures[index] / (density * density);
                    let mut push = Vec3::ZERO;
                    for &other in world.neighbors.of(index) {
                        let other = other as usize;
                        if other == index {
                            continue;
                        }
                        let offset = at - world.positions[other];
                        let gradient = world.kernel.gradient(offset);
                        let their = world.densities[other];
                        let pressure = own + pressures[other] / (their * their);
                        let toward = (velocity - world.velocities[other]).dot(offset);
                        let damp = if toward < 0.0 {
                            let squeeze =
                                toward / (offset.length_squared() + 0.01 * radius * radius);
                            -viscosity * squeeze * 2.0 / (density + their)
                        } else {
                            0.0
                        };
                        push -= gradient * (mass * (pressure + damp));
                    }
                    // a wall pushes back with this particle's own pressure, as water would
                    let mirrored = own + pressures[index] / (rest * rest);
                    for &wall in world.wall_neighbors.of(index) {
                        let offset = at - world.walls.positions[wall as usize];
                        let gradient = world.kernel.gradient(offset);
                        let share = mass;
                        let toward = velocity.dot(offset);
                        let damp = if toward < 0.0 {
                            let squeeze =
                                toward / (offset.length_squared() + 0.01 * radius * radius);
                            -viscosity * squeeze / density
                        } else {
                            0.0
                        };
                        push -= gradient * (share * (mirrored + damp));
                    }
                    push + settings.gravity
                })
                .collect::<Vec<Vec3>>()
        })?;
        let pushes: Vec<Vec3> = parts.into_iter().flatten().collect();

        let mut held = 0;
        for (index, push) in pushes.into_iter().enumerate() {
            let velocity = self.velocities[index] + push * dt;
            let mut at = self.positions[index] + velocity * dt;
            let mut velocity = velocity;
            if let Some((low, high)) = self.walls.bounds {
                let inside = at.clamp(low, high);
                if inside != at {
                    held += 1;
                    // the part of the motion into the wall stops
                    for (axis, out) in [
                        (0, inside.x != at.x),
                        (1, inside.y != at.y),
                        (2, inside.z != at.z),
                    ] {
                        if out {
                            match axis {
                                0 => velocity.x = 0.0,
                                1 => velocity.y = 0.0,
                                _ => velocity.z = 0.0,
                            }
                        }
                    }
                    at = inside;
                }
            }
            if !at.is_finite() || !velocity.is_finite() {
                return Err(PhysicsError::NonFinite("fluid particle"));
            }
            self.positions[index] = at;
            self.velocities[index] = velocity;
        }
        Ok(held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::Sequential;
    use crate::fluid::FluidSettings;

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
        let mut fluid = FluidWorld::new(FluidSettings::default()).unwrap();
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
        let run = |dispatch: &dyn JobDispatch| {
            let mut fluid = tank(Vec3::new(0.3, 0.3, 0.2), Vec3::new(0.9, 0.5, 0.2));
            for _ in 0..12 {
                fluid.step(TICK, dispatch).unwrap();
            }
            fluid.state_hash()
        };
        let one = run(&Sequential);
        assert_eq!(run(&Threads(4)), one);
        assert_eq!(run(&Threads(3)), one);
        assert_eq!(run(&Sequential), one);
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
