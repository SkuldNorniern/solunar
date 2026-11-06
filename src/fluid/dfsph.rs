use algebrix::Vec3;

use super::step::FluidReport;
use super::{FluidWorld, in_chunks};
use crate::dispatch::JobDispatch;
use crate::error::PhysicsError;

/// Average squeeze a density solve stops at, as a share of rest density.
const DENSITY_TOLERANCE: f32 = 5e-4;
/// Average density change over a substep a divergence solve stops at, same share.
const DIVERGENCE_TOLERANCE: f32 = 1e-3;
const MIN_ITERATIONS: u32 = 2;
const MAX_ITERATIONS: u32 = 100;
/// Fewer neighbors than this is the free surface, it may spread without the divergence solve
/// pulling it back.
const SURFACE_NEIGHBORS: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Solve {
    Divergence,
    Density,
}

/// Kernel gradient of every listed pair, in list order, worked out once a substep: positions
/// do not move while the solves run, only velocities do.
struct Gradients {
    fluid: Vec<Vec3>,
    walls: Vec<Vec3>,
}

impl Gradients {
    fn fluid(&self, world: &FluidWorld, index: usize) -> &[Vec3] {
        let lists = &world.neighbors;
        &self.fluid[lists.start[index] as usize..lists.start[index + 1] as usize]
    }

    fn walls(&self, world: &FluidWorld, index: usize) -> &[Vec3] {
        let lists = &world.wall_neighbors;
        &self.walls[lists.start[index] as usize..lists.start[index + 1] as usize]
    }
}

impl FluidWorld {
    /// Divergence free sph: pressure from two small solves each substep instead of an equation
    /// of state, so the step is held by how fast water moves, not by a sound speed.
    pub(super) fn dfsph(
        &mut self,
        dt: f32,
        dispatch: &dyn JobDispatch,
        report: &mut FluidReport,
    ) -> Result<(), PhysicsError> {
        // ground pushes through the weakly compressible solver's pressure only
        if !self.terrains.is_empty() {
            return Err(PhysicsError::InvalidSettings);
        }
        // a particle may move this share of a spacing in a substep
        let spacing = self.settings.spacing;
        let courant = self.settings.courant;
        let pull = self.settings.gravity.length();
        let mut left = dt;
        while left > 0.0 {
            let mut small = courant * spacing / self.fastest().max(1e-6);
            if pull > 0.0 {
                // from rest, gravity alone may carry a particle that far in this time
                small = small.min((2.0 * courant * spacing / pull).sqrt());
            }
            if report.substeps + 1 >= self.settings.max_substeps && small < left {
                report.capped = true;
                small = left;
            }
            let small = small.min(left);
            report.iterations += self.dfsph_substep(small, dispatch, &mut report.held_back)?;
            report.substeps += 1;
            left -= small;
        }
        Ok(())
    }

    fn dfsph_substep(
        &mut self,
        dt: f32,
        dispatch: &dyn JobDispatch,
        held: &mut u32,
    ) -> Result<u32, PhysicsError> {
        self.update_neighbors(dispatch)?;
        self.compute_densities(dispatch)?;
        let gradients = self.gradients(dispatch)?;
        let factors = self.factors(&gradients, dispatch)?;
        let mut iterations = self.solve(Solve::Divergence, dt, &factors, &gradients, dispatch)?;
        let world = &*self;
        let gravity = self.settings.gravity;
        let pushes = in_chunks(dispatch, self.len(), |range| {
            range
                .map(|index| world.viscous(index, dt) + gravity)
                .collect::<Vec<Vec3>>()
        })?;
        for (velocity, push) in self.velocities.iter_mut().zip(pushes.into_iter().flatten()) {
            *velocity += push * dt;
        }
        iterations += self.solve(Solve::Density, dt, &factors, &gradients, dispatch)?;
        *held += self.advance(dt)?;
        Ok(iterations)
    }

    /// For each particle, one over how much its own pressure changes its density rate: the
    /// square of the summed kernel gradient, walls in it, plus each neighbor's alone. Zero for a
    /// lone particle.
    fn gradients(&self, dispatch: &dyn JobDispatch) -> Result<Gradients, PhysicsError> {
        let parts = in_chunks(dispatch, self.len(), |range| {
            let mut fluid = Vec::new();
            let mut walls = Vec::new();
            for index in range {
                let at = self.positions[index];
                fluid.extend(
                    self.neighbors
                        .of(index)
                        .iter()
                        .map(|&other| self.kernel.gradient(at - self.positions[other as usize])),
                );
                walls.extend(self.wall_neighbors.of(index).iter().map(|&wall| {
                    self.kernel
                        .gradient(at - self.walls.positions[wall as usize])
                }));
            }
            (fluid, walls)
        })?;
        let mut gradients = Gradients {
            fluid: Vec::with_capacity(self.neighbors.items.len()),
            walls: Vec::with_capacity(self.wall_neighbors.items.len()),
        };
        for (fluid, walls) in parts {
            gradients.fluid.extend(fluid);
            gradients.walls.extend(walls);
        }
        Ok(gradients)
    }

    fn factors(
        &self,
        gradients: &Gradients,
        dispatch: &dyn JobDispatch,
    ) -> Result<Vec<f32>, PhysicsError> {
        let parts = in_chunks(dispatch, self.len(), |range| {
            range
                .map(|index| {
                    let mut sum = Vec3::ZERO;
                    let mut squares = 0.0;
                    // its own entry has no gradient
                    for gradient in gradients.fluid(self, index) {
                        let part = *gradient * self.mass;
                        sum += part;
                        squares += part.length_squared();
                    }
                    for gradient in gradients.walls(self, index) {
                        sum += *gradient * self.mass;
                    }
                    let total = sum.length_squared() + squares;
                    if total > 0.0 { 1.0 / total } else { 0.0 }
                })
                .collect::<Vec<f32>>()
        })?;
        Ok(parts.into_iter().flatten().collect())
    }

    /// How fast the density sum is changing, walls standing still.
    fn density_rate(&self, index: usize, gradients: &Gradients) -> f32 {
        let velocity = self.velocities[index];
        let mut rate = 0.0;
        let pairs = self
            .neighbors
            .of(index)
            .iter()
            .zip(gradients.fluid(self, index));
        for (&other, gradient) in pairs {
            rate += (velocity - self.velocities[other as usize]).dot(*gradient);
        }
        for gradient in gradients.walls(self, index) {
            rate += velocity.dot(*gradient);
        }
        rate * self.mass
    }

    /// Velocity change from each particle's stiffness, walls standing still.
    fn push_apart(
        &mut self,
        stiffness: &[f32],
        dt: f32,
        gradients: &Gradients,
        dispatch: &dyn JobDispatch,
    ) -> Result<(), PhysicsError> {
        let world = &*self;
        let changes = in_chunks(dispatch, self.len(), |range| {
            range
                .map(|index| {
                    let own = stiffness[index];
                    let mut change = Vec3::ZERO;
                    let pairs = world.neighbors.of(index).iter();
                    for (&other, gradient) in pairs.zip(gradients.fluid(world, index)) {
                        change -= *gradient * (world.mass * (own + stiffness[other as usize]));
                    }
                    for gradient in gradients.walls(world, index) {
                        change -= *gradient * (world.mass * own);
                    }
                    change * dt
                })
                .collect::<Vec<Vec3>>()
        })?;
        for (velocity, change) in self
            .velocities
            .iter_mut()
            .zip(changes.into_iter().flatten())
        {
            *velocity += change;
        }
        Ok(())
    }

    /// Jacobi iterations on velocities until the average error is under tolerance. Each sums per
    /// particle in list order, so the count and the result do not depend on threads. Starting from
    /// half of last substep's pressure, as dfsph papers do, pushed water apart where the solve can
    /// not pull it back, and a dam break front ran twice as fast.
    fn solve(
        &mut self,
        solve: Solve,
        dt: f32,
        factors: &[f32],
        gradients: &Gradients,
        dispatch: &dyn JobDispatch,
    ) -> Result<u32, PhysicsError> {
        let rest = self.settings.rest_density;
        let count = self.len();
        let tolerance = match solve {
            Solve::Divergence => DIVERGENCE_TOLERANCE,
            Solve::Density => DENSITY_TOLERANCE,
        };
        let mut iterations = 0;
        loop {
            let world = &*self;
            let parts = in_chunks(dispatch, count, |range| {
                range
                    .map(|index| {
                        let rate = world.density_rate(index, gradients);
                        let (target, error) = match solve {
                            Solve::Density => {
                                let over = (world.densities[index] + dt * rate - rest).max(0.0);
                                (over / (dt * dt), over / rest)
                            }
                            Solve::Divergence => {
                                let crowd = world.neighbors.of(index).len()
                                    + world.wall_neighbors.of(index).len();
                                let rate = if crowd < SURFACE_NEIGHBORS {
                                    0.0
                                } else {
                                    rate.max(0.0)
                                };
                                (rate / dt, dt * rate / rest)
                            }
                        };
                        (target * factors[index], error)
                    })
                    .collect::<Vec<(f32, f32)>>()
            })?;
            let (stiffness, errors): (Vec<f32>, Vec<f32>) = parts.into_iter().flatten().unzip();
            // over the particles under load only: averaged over all, a few squeezed ones at the bottom
            // of a deep column looked settled and the column sank onto the floor
            let loaded = errors.iter().filter(|error| **error > 0.0).count().max(1);
            let average = errors.iter().sum::<f32>() / loaded as f32;
            let settled = iterations >= MIN_ITERATIONS && average <= tolerance;
            if settled || iterations >= MAX_ITERATIONS {
                return Ok(iterations);
            }
            self.push_apart(&stiffness, dt, gradients, dispatch)?;
            iterations += 1;
        }
    }
}
