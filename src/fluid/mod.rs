use std::f32::consts::PI;
use std::ops::Range;
use std::sync::{Mutex, PoisonError};

use algebrix::Vec3;

use crate::dispatch::JobDispatch;
use crate::error::PhysicsError;
use crate::snapshot::fnv1a;

mod boundary;
mod couple;
mod dfsph;
mod flow;
mod handoff;
mod save;
mod shallow;
mod step;
mod terrain;

pub use boundary::BoundaryId;
use boundary::Walls;
pub use couple::FluidHook;
use couple::{Coupled, Movers};
pub use flow::FlowId;
use flow::Flows;
pub use handoff::{Handoff, HandoffReport, hand_off};
pub use save::{FLUID_SNAPSHOT_VERSION, MAX_FLUID_SNAPSHOT_BYTES};
pub use shallow::{Erosion, ShallowReport, ShallowSettings, ShallowWater};
pub use step::FluidReport;
use terrain::{GroundSites, TerrainField};

/// Particles per job. Fixed, so the split of work and every sum in it never depend on how many
/// threads run it.
const CHUNK: usize = 512;
const SUPPORT_RANGE: (f32, f32) = (1.5, 4.0);
/// A dense cell table is kept while its box holds at most this many cells per occupied one, plus
/// the slack. Water spread thin falls back to searching.
const DENSE_PER_CELL: i64 = 64;
const DENSE_SLACK: i64 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FluidSolver {
    /// Weakly compressible: pressure from an equation of state. Checked against martin and moyce,
    /// its step is held by `sound_speed`.
    #[default]
    Wcsph,
    /// Divergence free: pressure from small solves each substep, steps held by the flow speed.
    /// Not ready: still water keeps sloshing and squeezes more than the reference, its bottom layer
    /// reaches the floor plane, hard impacts squeeze it more and the dam break front runs ahead.
    Dfsph,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidSettings {
    pub solver: FluidSolver,
    /// Distance between particles at rest, in meters.
    pub spacing: f32,
    /// Kilograms per cubic meter.
    pub rest_density: f32,
    /// Kernel reach as a multiple of `spacing`.
    pub support: f32,
    pub max_particles: usize,
    /// Meters a second. For the weakly compressible solver the equation of state is stiff enough
    /// that the fluid squeezes about `(fastest flow / sound_speed)^2`, so ten times the fastest
    /// flow keeps it near one percent, higher costs more substeps. Both solvers scale viscosity by
    /// it.
    pub sound_speed: f32,
    /// Artificial viscosity. Below about 0.1 still water on a cubic lattice keeps churning as the
    /// lattice settles into a denser packing. It also slows fast flow a little.
    pub viscosity: f32,
    pub gravity: Vec3,
    /// Share of the kernel radius a particle may move in a substep.
    pub courant: f32,
    /// Most substeps one step may take. A step that needs more is reported, not cut short.
    pub max_substeps: u32,
}

impl Default for FluidSettings {
    fn default() -> Self {
        Self {
            solver: FluidSolver::default(),
            spacing: 0.05,
            rest_density: 1000.0,
            support: 2.0,
            max_particles: 1 << 22,
            sound_speed: 20.0,
            viscosity: 0.15,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            courant: 0.4,
            max_substeps: 2000,
        }
    }
}

impl FluidSettings {
    fn validate(&self) -> Result<(), PhysicsError> {
        let positive = |value: f32| value.is_finite() && value > 0.0;
        let support = (SUPPORT_RANGE.0..=SUPPORT_RANGE.1).contains(&self.support);
        if positive(self.spacing)
            && positive(self.rest_density)
            && support
            && self.max_particles > 0
            && positive(self.sound_speed)
            && self.viscosity.is_finite()
            && self.viscosity >= 0.0
            && self.gravity.is_finite()
            && self.courant > 0.0
            && self.courant <= 1.0
            && self.max_substeps > 0
        {
            Ok(())
        } else {
            Err(PhysicsError::InvalidSettings)
        }
    }

    /// How far a particle feels its neighbors.
    pub fn radius(&self) -> f32 {
        self.spacing * self.support
    }
}

/// Wendland C2 in 3D: smooth, zero past `radius`, no clumping of particles under pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kernel {
    radius: f32,
    scale: f32,
}

impl Kernel {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            scale: 21.0 / (2.0 * PI * radius.powi(3)),
        }
    }

    pub fn value(&self, distance: f32) -> f32 {
        let q = distance / self.radius;
        if q >= 1.0 {
            return 0.0;
        }
        self.scale * (1.0 - q).powi(4) * (1.0 + 4.0 * q)
    }

    /// Change of the value with distance.
    pub fn slope(&self, distance: f32) -> f32 {
        let q = distance / self.radius;
        if q >= 1.0 {
            return 0.0;
        }
        self.scale * -20.0 * q * (1.0 - q).powi(3) / self.radius
    }

    /// Gradient at `offset` from the particle, pointing away from it.
    pub fn gradient(&self, offset: Vec3) -> Vec3 {
        let distance = offset.length();
        if distance <= 0.0 {
            return Vec3::ZERO;
        }
        offset * (self.slope(distance) / distance)
    }
}

type CellKey = (i32, i32, i32);

/// Cells in key order, each with its range in an array sorted the same way.
#[derive(Debug, Clone, Default)]
struct Grid {
    cells: Vec<(CellKey, Range<u32>)>,
    /// Slot in `cells` for every cell of the box they span, `u32::MAX` for empty ones. Left
    /// empty when the box is much bigger than the cells in it, lookups then search `cells`.
    dense: Vec<u32>,
    low: CellKey,
    span: (i64, i64, i64),
}

impl Grid {
    /// `keys` must already be sorted.
    fn build(keys: impl Iterator<Item = CellKey>) -> Self {
        let mut cells: Vec<(CellKey, Range<u32>)> = Vec::new();
        for (index, cell) in keys.enumerate() {
            let index = index as u32;
            match cells.last_mut() {
                Some((last, range)) if *last == cell => range.end = index + 1,
                _ => cells.push((cell, index..index + 1)),
            }
        }
        let mut grid = Self {
            cells,
            ..Self::default()
        };
        grid.index();
        grid
    }

    fn index(&mut self) {
        let Some(first) = self.cells.first() else {
            return;
        };
        let (mut low, mut high) = (first.0, first.0);
        for (cell, _) in &self.cells {
            low = (low.0.min(cell.0), low.1.min(cell.1), low.2.min(cell.2));
            high = (high.0.max(cell.0), high.1.max(cell.1), high.2.max(cell.2));
        }
        let side = |a: i32, b: i32| i64::from(b) - i64::from(a) + 1;
        let span = (
            side(low.0, high.0),
            side(low.1, high.1),
            side(low.2, high.2),
        );
        let volume = span.0.saturating_mul(span.1).saturating_mul(span.2);
        if volume > DENSE_PER_CELL * self.cells.len() as i64 + DENSE_SLACK {
            return;
        }
        let mut dense = vec![u32::MAX; volume as usize];
        for (slot, (cell, _)) in self.cells.iter().enumerate() {
            let at = (i64::from(cell.0 - low.0) * span.1 + i64::from(cell.1 - low.1)) * span.2
                + i64::from(cell.2 - low.2);
            dense[at as usize] = slot as u32;
        }
        self.dense = dense;
        self.low = low;
        self.span = span;
    }

    fn range(&self, key: CellKey) -> Option<Range<u32>> {
        let slot = if self.dense.is_empty() {
            self.cells.binary_search_by_key(&key, |cell| cell.0).ok()?
        } else {
            let offset = |value: i32, low: i32, span: i64| {
                let at = i64::from(value) - i64::from(low);
                (0..span).contains(&at).then_some(at)
            };
            let x = offset(key.0, self.low.0, self.span.0)?;
            let y = offset(key.1, self.low.1, self.span.1)?;
            let z = offset(key.2, self.low.2, self.span.2)?;
            let slot = self.dense[((x * self.span.1 + y) * self.span.2 + z) as usize];
            if slot == u32::MAX {
                return None;
            }
            slot as usize
        };
        Some(self.cells[slot].1.clone())
    }

    /// Ranges of the 27 cells around `cell` that hold anything, in a fixed order.
    fn around(&self, (cx, cy, cz): CellKey, out: &mut Vec<Range<u32>>) {
        out.clear();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(range) = self.range((cx + dx, cy + dy, cz + dz)) {
                        out.push(range);
                    }
                }
            }
        }
    }
}

/// Indexes in `ranges` within `reach` of `at`, in their order.
fn near(positions: &[Vec3], ranges: &[Range<u32>], at: Vec3, reach: f32, out: &mut Vec<u32>) {
    for range in ranges {
        // every candidate is written and only the near ones kept, a branch on each guessed wrong
        // too often
        let mut kept = out.len();
        out.resize(kept + range.len(), 0);
        for other in range.clone() {
            out[kept] = other;
            kept += usize::from((positions[other as usize] - at).length_squared() < reach);
        }
        out.truncate(kept);
    }
}

/// Neighbor lists of every particle, one after another.
#[derive(Debug, Clone, Default)]
struct Lists {
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Lists {
    fn of(&self, index: usize) -> &[u32] {
        let (Some(&start), Some(&end)) = (self.start.get(index), self.start.get(index + 1)) else {
            return &[];
        };
        &self.items[start as usize..end as usize]
    }

    fn join(parts: impl Iterator<Item = (Vec<u32>, Vec<u32>)>) -> Self {
        let mut lists = Self {
            start: vec![0],
            items: Vec::new(),
        };
        for (counts, found) in parts {
            for count in counts {
                let last = lists.start.last().copied().unwrap_or(0);
                lists.start.push(last + count);
            }
            lists.items.extend(found);
        }
        lists
    }
}

/// Particles as arrays per field, kept sorted by grid cell so neighbors sit close in memory.
#[derive(Debug, Clone)]
pub struct FluidWorld {
    settings: FluidSettings,
    kernel: Kernel,
    mass: f32,
    ids: Vec<u32>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    densities: Vec<f32>,
    /// How the ground's density changes at each particle, from the last density pass, empty
    /// without terrains. Summed once there, it is the dearest part of the ground.
    ground_slopes: Vec<Vec3>,
    next_id: u32,
    grid: Grid,
    neighbors: Lists,
    /// Walls near each particle, indexes into `walls`.
    wall_neighbors: Lists,
    walls: Walls,
    coupled: Vec<Coupled>,
    movers: Movers,
    /// Body points near each particle, indexes into `movers`.
    mover_neighbors: Lists,
    flows: Flows,
    terrains: Vec<TerrainField>,
    ground_sites: GroundSites,
    /// Water outside this box after a step is taken out, so a splash flung far away does not
    /// fly on for ever, setting every step's substeps by its speed.
    domain: Option<(Vec3, Vec3)>,
}

impl FluidWorld {
    /// # Errors
    /// Fails on settings out of range.
    pub fn new(settings: FluidSettings) -> Result<Self, PhysicsError> {
        settings.validate()?;
        let kernel = Kernel::new(settings.radius());
        let mass = rest_mass(&settings);
        Ok(Self {
            settings,
            kernel,
            mass,
            ground_sites: GroundSites::new(&kernel, mass, settings.spacing, settings.radius()),
            ids: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            densities: Vec::new(),
            ground_slopes: Vec::new(),
            next_id: 0,
            grid: Grid::default(),
            neighbors: Lists::default(),
            wall_neighbors: Lists::default(),
            walls: Walls::default(),
            coupled: Vec::new(),
            movers: Movers::default(),
            mover_neighbors: Lists::default(),
            flows: Flows::default(),
            terrains: Vec::new(),
            domain: None,
        })
    }

    pub fn settings(&self) -> &FluidSettings {
        &self.settings
    }

    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    pub fn particle_mass(&self) -> f32 {
        self.mass
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Stable across steps. The arrays below are reordered, ids say which particle is which.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// From the last [`Self::compute_densities`].
    pub fn densities(&self) -> &[f32] {
        &self.densities
    }

    /// # Errors
    /// Fails on a non finite value or a full world.
    pub fn add_particle(&mut self, position: Vec3, velocity: Vec3) -> Result<u32, PhysicsError> {
        if !position.is_finite() || !velocity.is_finite() {
            return Err(PhysicsError::NonFinite("fluid particle"));
        }
        if self.len() >= self.settings.max_particles || self.next_id == u32::MAX {
            return Err(PhysicsError::FluidFull);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.ids.push(id);
        self.positions.push(position);
        self.velocities.push(velocity);
        self.densities.push(0.0);
        Ok(id)
    }

    /// Water that leaves this box is taken out after each step, counted in
    /// [`FluidReport::lost`]. `None` keeps all water wherever it goes.
    ///
    /// # Errors
    /// Fails on a non finite or inside out box.
    pub fn set_domain(&mut self, domain: Option<(Vec3, Vec3)>) -> Result<(), PhysicsError> {
        if let Some((low, high)) = domain {
            if !low.is_finite() || !high.is_finite() {
                return Err(PhysicsError::NonFinite("fluid domain"));
            }
            if high.x <= low.x || high.y <= low.y || high.z <= low.z {
                return Err(PhysicsError::OutOfRange);
            }
        }
        self.domain = domain;
        Ok(())
    }

    pub fn domain(&self) -> Option<(Vec3, Vec3)> {
        self.domain
    }

    /// Takes out water outside the domain. Returns how much.
    pub(super) fn keep_in_domain(&mut self) -> u32 {
        let Some((low, high)) = self.domain else {
            return 0;
        };
        let keep: Vec<bool> = self
            .positions
            .iter()
            .map(|at| at.clamp(low, high) == *at)
            .collect();
        let lost = keep.iter().filter(|kept| !**kept).count();
        if lost > 0 {
            self.keep_only(&keep);
        }
        lost as u32
    }

    /// Drops every particle whose `keep` is false, the rest stay in order.
    pub(super) fn keep_only(&mut self, keep: &[bool]) {
        let mut index = 0;
        self.ids.retain(|_| {
            index += 1;
            keep[index - 1]
        });
        for list in [&mut self.positions, &mut self.velocities] {
            let mut index = 0;
            list.retain(|_| {
                index += 1;
                keep[index - 1]
            });
        }
        let mut index = 0;
        self.densities.retain(|_| {
            index += 1;
            keep[index - 1]
        });
    }

    /// Fills the box from `low` to `high` with particles on the rest lattice, half a spacing in
    /// from each face. Returns how many.
    ///
    /// # Errors
    /// Fails on a non finite or inside out box, or when they would not all fit. Then none are added.
    pub fn fill_box(&mut self, low: Vec3, high: Vec3) -> Result<usize, PhysicsError> {
        if !low.is_finite() || !high.is_finite() {
            return Err(PhysicsError::NonFinite("fluid box"));
        }
        if high.x < low.x || high.y < low.y || high.z < low.z {
            return Err(PhysicsError::OutOfRange);
        }
        let spacing = self.settings.spacing;
        let count = |extent: f32| (extent / spacing).floor().min(u32::MAX as f32) as usize;
        let size = high - low;
        let (nx, ny, nz) = (count(size.x), count(size.y), count(size.z));
        let total = nx.saturating_mul(ny).saturating_mul(nz);
        let room = self.settings.max_particles.saturating_sub(self.len());
        let ids_left = (u32::MAX - self.next_id) as usize;
        if total > room || total > ids_left {
            return Err(PhysicsError::FluidFull);
        }
        self.ids.reserve(total);
        self.positions.reserve(total);
        self.velocities.reserve(total);
        self.densities.reserve(total);
        let start = low + Vec3::splat(spacing * 0.5);
        for y in 0..ny {
            for z in 0..nz {
                for x in 0..nx {
                    let at = start + Vec3::new(x as f32, y as f32, z as f32) * spacing;
                    self.add_particle(at, Vec3::ZERO)?;
                }
            }
        }
        Ok(total)
    }

    fn cell_of(&self, position: Vec3) -> CellKey {
        let cell = position / self.settings.radius();
        (
            cell.x.floor() as i32,
            cell.y.floor() as i32,
            cell.z.floor() as i32,
        )
    }

    /// Sorts particles by cell then id, and lists each one's neighbors within the kernel, itself
    /// included, in a fixed order.
    ///
    /// # Errors
    /// Fails if the dispatcher skipped a job.
    pub fn update_neighbors(&mut self, dispatch: &dyn JobDispatch) -> Result<(), PhysicsError> {
        let mut order: Vec<(CellKey, u32, u32)> = (0..self.len())
            .map(|index| {
                (
                    self.cell_of(self.positions[index]),
                    self.ids[index],
                    index as u32,
                )
            })
            .collect();
        order.sort_unstable();
        let pick = |index: &(CellKey, u32, u32)| index.2 as usize;
        self.ids = order.iter().map(|item| self.ids[pick(item)]).collect();
        self.positions = order
            .iter()
            .map(|item| self.positions[pick(item)])
            .collect();
        self.velocities = order
            .iter()
            .map(|item| self.velocities[pick(item)])
            .collect();
        self.densities = order
            .iter()
            .map(|item| self.densities[pick(item)])
            .collect();

        self.grid = Grid::build(order.iter().map(|item| item.0));

        let reach = self.settings.radius().powi(2);
        let world = &*self;
        let lists = in_chunks(dispatch, self.len(), |range| {
            let mut fluid = (Vec::with_capacity(range.len()), Vec::new());
            let mut walls = (Vec::with_capacity(range.len()), Vec::new());
            let mut movers = (Vec::with_capacity(range.len()), Vec::new());
            // particles are sorted by cell, the cells around are looked up once per cell
            let mut last = None;
            let mut around = (Vec::new(), Vec::new(), Vec::new());
            for index in range {
                let at = world.positions[index];
                let cell = world.cell_of(at);
                if last != Some(cell) {
                    last = Some(cell);
                    world.grid.around(cell, &mut around.0);
                    world.walls.grid.around(cell, &mut around.1);
                    world.movers.grid.around(cell, &mut around.2);
                }
                let before = (fluid.1.len(), walls.1.len(), movers.1.len());
                near(&world.positions, &around.0, at, reach, &mut fluid.1);
                near(&world.walls.positions, &around.1, at, reach, &mut walls.1);
                near(&world.movers.positions, &around.2, at, reach, &mut movers.1);
                fluid.0.push((fluid.1.len() - before.0) as u32);
                walls.0.push((walls.1.len() - before.1) as u32);
                movers.0.push((movers.1.len() - before.2) as u32);
            }
            (fluid, walls, movers)
        })?;
        let mut fluid = Vec::with_capacity(lists.len());
        let mut walls = Vec::with_capacity(lists.len());
        let mut movers = Vec::with_capacity(lists.len());
        for (a, b, c) in lists {
            fluid.push(a);
            walls.push(b);
            movers.push(c);
        }
        self.neighbors = Lists::join(fluid.into_iter());
        self.wall_neighbors = Lists::join(walls.into_iter());
        self.mover_neighbors = Lists::join(movers.into_iter());
        Ok(())
    }

    /// Indexes into the arrays, from the last [`Self::update_neighbors`]. Empty when it has not
    /// run since particles were added.
    pub fn neighbors_of(&self, index: usize) -> &[u32] {
        self.neighbors.of(index)
    }

    /// Mass weighted by the kernel over each particle's neighbors, walls taking the place of the
    /// water that is not there.
    ///
    /// # Errors
    /// Fails if the dispatcher skipped a job, or neighbors are stale.
    pub fn compute_densities(&mut self, dispatch: &dyn JobDispatch) -> Result<(), PhysicsError> {
        if self.neighbors.start.len() != self.len() + 1 {
            return Err(PhysicsError::OutOfRange);
        }
        let mass = self.mass;
        let world = &*self;
        let parts = in_chunks(dispatch, self.len(), |range| {
            range
                .map(|index| {
                    let at = world.positions[index];
                    let sum: f32 = world
                        .neighbors_of(index)
                        .iter()
                        .map(|&other| {
                            let distance = (world.positions[other as usize] - at).length();
                            world.kernel.value(distance)
                        })
                        .sum();
                    let walls: f32 = world
                        .wall_neighbors
                        .of(index)
                        .iter()
                        .map(|&wall| {
                            let distance = (world.walls.positions[wall as usize] - at).length();
                            world.kernel.value(distance)
                        })
                        .sum();
                    let movers: f32 = world
                        .mover_neighbors
                        .of(index)
                        .iter()
                        .map(|&point| {
                            let distance = (world.movers.positions[point as usize] - at).length();
                            world.kernel.value(distance)
                        })
                        .sum();
                    let (ground, slope) = world.ground_near(at).fold(
                        (0.0, Vec3::ZERO),
                        |(density, slope), ground| {
                            (density + ground.density, slope + ground.gradient)
                        },
                    );
                    (mass * (sum + walls + movers) + ground, slope)
                })
                .collect::<Vec<(f32, Vec3)>>()
        })?;
        let (densities, slopes): (Vec<f32>, Vec<Vec3>) = parts.into_iter().flatten().unzip();
        self.densities = densities;
        self.ground_slopes = if self.terrains.is_empty() {
            Vec::new()
        } else {
            slopes
        };
        Ok(())
    }

    /// Hash of ids, positions and velocities in id order, for replay checks.
    pub fn state_hash(&self) -> u64 {
        let mut order: Vec<usize> = (0..self.len()).collect();
        order.sort_unstable_by_key(|&index| self.ids[index]);
        let mut bytes = Vec::with_capacity(self.len() * 28);
        for index in order {
            bytes.extend_from_slice(&self.ids[index].to_le_bytes());
            for vector in [self.positions[index], self.velocities[index]] {
                for value in [vector.x, vector.y, vector.z] {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        fnv1a(&bytes)
    }
}

/// The mass that makes a full rest lattice sum to exactly the rest density. One lattice cell's
/// worth of water would be a few percent off, the kernel sum on a lattice is not exactly one.
fn rest_mass(settings: &FluidSettings) -> f32 {
    let kernel = Kernel::new(settings.radius());
    let reach = settings.support.ceil() as i32;
    let mut sum = 0.0;
    for z in -reach..=reach {
        for y in -reach..=reach {
            for x in -reach..=reach {
                let offset = Vec3::new(x as f32, y as f32, z as f32) * settings.spacing;
                sum += kernel.value(offset.length());
            }
        }
    }
    settings.rest_density / sum
}

/// `work` for each `CHUNK` of `0..count` on the dispatcher, results in chunk order.
fn in_chunks<R: Send>(
    dispatch: &dyn JobDispatch,
    count: usize,
    work: impl Fn(Range<usize>) -> R + Sync,
) -> Result<Vec<R>, PhysicsError> {
    in_chunks_of(dispatch, count, CHUNK, work)
}

/// [`in_chunks`] with `chunk` items a job.
fn in_chunks_of<R: Send>(
    dispatch: &dyn JobDispatch,
    count: usize,
    chunk: usize,
    work: impl Fn(Range<usize>) -> R + Sync,
) -> Result<Vec<R>, PhysicsError> {
    let chunk = chunk.max(1);
    let jobs = count.div_ceil(chunk);
    let slots: Vec<Mutex<Option<R>>> = (0..jobs).map(|_| Mutex::new(None)).collect();
    dispatch.run(jobs, &|job| {
        let range = job * chunk..((job + 1) * chunk).min(count);
        let result = work(range);
        *slots[job].lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
    });
    slots
        .into_iter()
        .map(|slot| slot.into_inner().unwrap_or_else(PoisonError::into_inner))
        .collect::<Option<Vec<R>>>()
        .ok_or(PhysicsError::DispatchIncomplete)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::Sequential;
    use crate::dispatch::testing::{SkipsJobs, Threads};

    fn world() -> FluidWorld {
        FluidWorld::new(FluidSettings::default()).unwrap()
    }

    /// A lattice block with every particle nudged a little, so distances are not all equal.
    fn shaken(count: usize) -> FluidWorld {
        let mut fluid = world();
        let mut seed = 0x9E37_79B9_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32 - 0.5) * 0.02
        };
        let side = (count as f32).cbrt().ceil() as usize;
        for index in 0..count {
            let lattice = Vec3::new(
                (index % side) as f32,
                (index / side % side) as f32,
                (index / side / side) as f32,
            ) * 0.05;
            fluid
                .add_particle(lattice + Vec3::new(next(), next(), next()), Vec3::ZERO)
                .unwrap();
        }
        fluid
    }

    #[test]
    fn the_kernel_holds_unit_mass_and_its_slope_matches() {
        let kernel = Kernel::new(0.1);
        // integrate over the ball by shells
        let steps = 4000;
        let dr = 0.1 / steps as f32;
        let total: f32 = (0..steps)
            .map(|step| {
                let r = (step as f32 + 0.5) * dr;
                4.0 * PI * r * r * kernel.value(r) * dr
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-3, "{total}");
        assert_eq!(kernel.value(0.1), 0.0);
        assert_eq!(kernel.value(0.5), 0.0);
        for r in [0.01, 0.03, 0.05, 0.08, 0.099] {
            let h = 1e-4;
            let numeric = (kernel.value(r + h) - kernel.value(r - h)) / (2.0 * h);
            let slope = kernel.slope(r);
            assert!((numeric - slope).abs() < 1e-2 * slope.abs().max(1.0), "{r}");
            assert!(slope <= 0.0);
        }
        let gradient = kernel.gradient(Vec3::new(0.03, 0.0, 0.0));
        assert!(gradient.x < 0.0 && gradient.y == 0.0);
        assert_eq!(kernel.gradient(Vec3::ZERO), Vec3::ZERO);
    }

    #[test]
    fn the_dense_table_and_the_search_find_the_same_cells() {
        let keys: Vec<CellKey> = vec![(-3, 0, 2), (-3, 0, 2), (-1, 4, 0), (0, 0, 0), (5, -2, 1)];
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        let dense = Grid::build(sorted.iter().copied());
        assert!(!dense.dense.is_empty());
        let mut search = dense.clone();
        search.dense.clear();
        for x in -5..7 {
            for y in -4..6 {
                for z in -2..4 {
                    assert_eq!(
                        dense.range((x, y, z)),
                        search.range((x, y, z)),
                        "{x} {y} {z}"
                    );
                }
            }
        }
        // spread far apart, no table
        let far = Grid::build([(-100_000, 0, 0), (100_000, 0, 0)].into_iter());
        assert!(far.dense.is_empty());
        assert_eq!(far.range((100_000, 0, 0)), Some(1..2));
        assert_eq!(far.range((i32::MAX, i32::MIN, 0)), None);
    }

    #[test]
    fn neighbors_are_exactly_the_brute_force_ones() {
        let mut fluid = shaken(900);
        fluid.update_neighbors(&Sequential).unwrap();
        let reach = fluid.settings.radius().powi(2);
        for index in 0..fluid.len() {
            let at = fluid.positions[index];
            let mut brute: Vec<u32> = (0..fluid.len() as u32)
                .filter(|&other| (fluid.positions[other as usize] - at).length_squared() < reach)
                .collect();
            let mut found = fluid.neighbors_of(index).to_vec();
            brute.sort_unstable();
            found.sort_unstable();
            assert_eq!(found, brute, "{index}");
        }
    }

    #[test]
    fn any_thread_count_gives_the_same_lists_densities_and_hash() {
        let run = |dispatch: &dyn JobDispatch| {
            let mut fluid = shaken(3000);
            fluid.update_neighbors(dispatch).unwrap();
            fluid.compute_densities(dispatch).unwrap();
            let bits: Vec<u32> = fluid.densities.iter().map(|d| d.to_bits()).collect();
            (
                fluid.neighbors.items.clone(),
                bits,
                fluid.ids.clone(),
                fluid.state_hash(),
            )
        };
        let one = run(&Sequential);
        assert_eq!(run(&Threads(4)), one);
        assert_eq!(run(&Threads(7)), one);
    }

    #[test]
    fn a_lattice_inside_is_at_rest_density() {
        let mut fluid = world();
        let count = fluid.fill_box(Vec3::ZERO, Vec3::splat(0.6)).unwrap();
        assert_eq!(count, 12 * 12 * 12);
        fluid.update_neighbors(&Sequential).unwrap();
        fluid.compute_densities(&Sequential).unwrap();
        let rest = fluid.settings.rest_density;
        let inside = |at: Vec3| {
            let margin = fluid.settings.radius();
            [at.x, at.y, at.z]
                .iter()
                .all(|&value| value > margin && value < 0.6 - margin)
        };
        let mut checked = 0;
        for (index, &at) in fluid.positions.iter().enumerate() {
            if inside(at) {
                let error = (fluid.densities[index] - rest).abs() / rest;
                assert!(error < 1e-4, "{error}");
                checked += 1;
            } else {
                // the edge is missing neighbors
                assert!(fluid.densities[index] <= rest * (1.0 + 1e-4));
            }
        }
        assert!(checked > 100);
        // close to a lattice cell's worth of water, but not equal to it
        let cell = rest * fluid.settings.spacing.powi(3);
        assert!((fluid.particle_mass() / cell - 1.0).abs() < 0.1);
    }

    #[test]
    fn ids_follow_particles_through_the_sort() {
        let mut fluid = shaken(700);
        let before: Vec<(u32, Vec3)> = fluid
            .ids
            .iter()
            .copied()
            .zip(fluid.positions.clone())
            .collect();
        let hash = fluid.state_hash();
        fluid.update_neighbors(&Sequential).unwrap();
        for (id, at) in before {
            let index = fluid.ids.iter().position(|&known| known == id).unwrap();
            assert_eq!(fluid.positions[index], at);
        }
        assert_eq!(fluid.state_hash(), hash);
    }

    #[test]
    fn bad_input_is_refused_and_changes_nothing() {
        for settings in [
            FluidSettings {
                spacing: 0.0,
                ..FluidSettings::default()
            },
            FluidSettings {
                rest_density: f32::NAN,
                ..FluidSettings::default()
            },
            FluidSettings {
                support: 9.0,
                ..FluidSettings::default()
            },
            FluidSettings {
                max_particles: 0,
                ..FluidSettings::default()
            },
        ] {
            assert_eq!(
                FluidWorld::new(settings).err(),
                Some(PhysicsError::InvalidSettings)
            );
        }
        let mut fluid = FluidWorld::new(FluidSettings {
            max_particles: 100,
            ..FluidSettings::default()
        })
        .unwrap();
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(fluid.add_particle(nan, Vec3::ZERO).is_err());
        assert!(fluid.add_particle(Vec3::ZERO, nan).is_err());
        assert!(fluid.fill_box(Vec3::ONE, Vec3::ZERO).is_err());
        assert!(fluid.fill_box(Vec3::ZERO, nan).is_err());
        assert_eq!(
            fluid.fill_box(Vec3::ZERO, Vec3::splat(1.0)),
            Err(PhysicsError::FluidFull)
        );
        // a huge box is refused before anything is made for it
        assert_eq!(
            fluid.fill_box(Vec3::ZERO, Vec3::splat(1e30)),
            Err(PhysicsError::FluidFull)
        );
        assert!(fluid.is_empty());
        assert_eq!(fluid.fill_box(Vec3::ZERO, Vec3::splat(0.2)), Ok(64));
        assert_eq!(
            fluid.compute_densities(&Sequential),
            Err(PhysicsError::OutOfRange)
        );
        assert_eq!(
            fluid.update_neighbors(&SkipsJobs {
                parallelism: 1,
                skip: |index| index == 0
            }),
            Err(PhysicsError::DispatchIncomplete)
        );
    }
}
