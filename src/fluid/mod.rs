use std::f32::consts::PI;
use std::ops::Range;
use std::sync::{Mutex, PoisonError};

use algebrix::Vec3;

use crate::dispatch::JobDispatch;
use crate::error::PhysicsError;
use crate::snapshot::fnv1a;

/// Particles per job. Fixed, so the split of work and every sum in it never depend on how many
/// threads run it.
const CHUNK: usize = 512;
const SUPPORT_RANGE: (f32, f32) = (1.5, 4.0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidSettings {
    /// Distance between particles at rest, in meters.
    pub spacing: f32,
    /// Kilograms per cubic meter.
    pub rest_density: f32,
    /// Kernel reach as a multiple of `spacing`.
    pub support: f32,
    pub max_particles: usize,
}

impl Default for FluidSettings {
    fn default() -> Self {
        Self {
            spacing: 0.05,
            rest_density: 1000.0,
            support: 2.0,
            max_particles: 1 << 22,
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
    next_id: u32,
    cells: Vec<(CellKey, Range<u32>)>,
    neighbor_start: Vec<u32>,
    neighbors: Vec<u32>,
}

impl FluidWorld {
    /// # Errors
    /// Fails on settings out of range.
    pub fn new(settings: FluidSettings) -> Result<Self, PhysicsError> {
        settings.validate()?;
        Ok(Self {
            settings,
            kernel: Kernel::new(settings.radius()),
            mass: rest_mass(&settings),
            ids: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            densities: Vec::new(),
            next_id: 0,
            cells: Vec::new(),
            neighbor_start: Vec::new(),
            neighbors: Vec::new(),
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

        self.cells.clear();
        for (index, (cell, _, _)) in order.iter().enumerate() {
            match self.cells.last_mut() {
                Some((last, range)) if last == cell => range.end = index as u32 + 1,
                _ => self.cells.push((*cell, index as u32..index as u32 + 1)),
            }
        }

        let reach = self.settings.radius().powi(2);
        let world = &*self;
        let lists = in_chunks(dispatch, self.len(), |range| {
            let mut counts = Vec::with_capacity(range.len());
            let mut found = Vec::new();
            for index in range {
                let at = world.positions[index];
                let (cx, cy, cz) = world.cell_of(at);
                let before = found.len();
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let key = (cx + dx, cy + dy, cz + dz);
                            let Ok(slot) = world.cells.binary_search_by_key(&key, |cell| cell.0)
                            else {
                                continue;
                            };
                            for other in world.cells[slot].1.clone() {
                                let offset = world.positions[other as usize] - at;
                                if offset.length_squared() < reach {
                                    found.push(other);
                                }
                            }
                        }
                    }
                }
                counts.push((found.len() - before) as u32);
            }
            (counts, found)
        })?;
        self.neighbor_start.clear();
        self.neighbors.clear();
        self.neighbor_start.push(0);
        for (counts, found) in lists {
            for count in counts {
                let last = self.neighbor_start.last().copied().unwrap_or(0);
                self.neighbor_start.push(last + count);
            }
            self.neighbors.extend(found);
        }
        Ok(())
    }

    /// Indexes into the arrays, from the last [`Self::update_neighbors`]. Empty when it has not
    /// run since particles were added.
    pub fn neighbors_of(&self, index: usize) -> &[u32] {
        let (Some(&start), Some(&end)) = (
            self.neighbor_start.get(index),
            self.neighbor_start.get(index + 1),
        ) else {
            return &[];
        };
        &self.neighbors[start as usize..end as usize]
    }

    /// Mass weighted by the kernel over each particle's neighbors.
    ///
    /// # Errors
    /// Fails if the dispatcher skipped a job, or neighbors are stale.
    pub fn compute_densities(&mut self, dispatch: &dyn JobDispatch) -> Result<(), PhysicsError> {
        if self.neighbor_start.len() != self.len() + 1 {
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
                    mass * sum
                })
                .collect::<Vec<f32>>()
        })?;
        self.densities = parts.into_iter().flatten().collect();
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
    let jobs = count.div_ceil(CHUNK);
    let slots: Vec<Mutex<Option<R>>> = (0..jobs).map(|_| Mutex::new(None)).collect();
    dispatch.run(jobs, &|job| {
        let range = job * CHUNK..((job + 1) * CHUNK).min(count);
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

    struct Skips;

    impl JobDispatch for Skips {
        fn parallelism(&self) -> usize {
            1
        }

        fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
            for index in 1..jobs {
                job(index);
            }
        }
    }

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
                fluid.neighbors.clone(),
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
            fluid.update_neighbors(&Skips),
            Err(PhysicsError::DispatchIncomplete)
        );
    }
}
