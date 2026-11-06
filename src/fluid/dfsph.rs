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
        let factors = self.factors(dispatch)?;
        let mut iterations = self.solve(Solve::Divergence, dt, &factors, dispatch)?;
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
        iterations += self.solve(Solve::Density, dt, &factors, dispatch)?;
        *held += self.advance(dt)?;
        Ok(iterations)
    }

    /// For each particle, one over how much its own pressure changes its density rate: the
    /// square of the summed kernel gradient, walls in it, plus each neighbor's alone. Zero for a
    /// lone particle.
    fn factors(&self, dispatch: &dyn JobDispatch) -> Result<Vec<f32>, PhysicsError> {
        let parts = in_chunks(dispatch, self.len(), |range| {
            range
                .map(|index| {
                    let at = self.positions[index];
                    let mut sum = Vec3::ZERO;
                    let mut squares = 0.0;
                    for &other in self.neighbors.of(index) {
                        if other as usize == index {
                            continue;
                        }
                        let part =
                            self.kernel.gradient(at - self.positions[other as usize]) * self.mass;
                        sum += part;
                        squares += part.length_squared();
                    }
                    for &wall in self.wall_neighbors.of(index) {
                        sum += self
                            .kernel
                            .gradient(at - self.walls.positions[wall as usize])
                            * self.mass;
                    }
                    let total = sum.length_squared() + squares;
                    if total > 0.0 { 1.0 / total } else { 0.0 }
                })
                .collect::<Vec<f32>>()
        })?;
        Ok(parts.into_iter().flatten().collect())
    }

    /// How fast the density sum is changing, walls standing still.
    fn density_rate(&self, index: usize) -> f32 {
        let at = self.positions[index];
        let velocity = self.velocities[index];
        let mut rate = 0.0;
        for &other in self.neighbors.of(index) {
            let other = other as usize;
            let gradient = self.kernel.gradient(at - self.positions[other]);
            rate += (velocity - self.velocities[other]).dot(gradient);
        }
        for &wall in self.wall_neighbors.of(index) {
            let gradient = self
                .kernel
                .gradient(at - self.walls.positions[wall as usize]);
            rate += velocity.dot(gradient);
        }
        rate * self.mass
    }

    /// Velocity change from each particle's stiffness, walls standing still.
    fn push_apart(
        &mut self,
        stiffness: &[f32],
        dt: f32,
        dispatch: &dyn JobDispatch,
    ) -> Result<(), PhysicsError> {
        let world = &*self;
        let changes = in_chunks(dispatch, self.len(), |range| {
            range
                .map(|index| {
                    let at = world.positions[index];
                    let own = stiffness[index];
                    let mut change = Vec3::ZERO;
                    for &other in world.neighbors.of(index) {
                        let other = other as usize;
                        if other == index {
                            continue;
                        }
                        let gradient = world.kernel.gradient(at - world.positions[other]);
                        change -= gradient * (world.mass * (own + stiffness[other]));
                    }
                    for &wall in world.wall_neighbors.of(index) {
                        let gradient = world
                            .kernel
                            .gradient(at - world.walls.positions[wall as usize]);
                        change -= gradient * (world.mass * own);
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
                        let rate = world.density_rate(index);
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
            self.push_apart(&stiffness, dt, dispatch)?;
            iterations += 1;
        }
    }
}
