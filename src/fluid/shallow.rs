use algebrix::Vec3;

use super::in_chunks;
use crate::dispatch::JobDispatch;
use crate::error::PhysicsError;
use crate::sdf::SdfVolume;
use crate::snapshot::fnv1a;

/// Most cells one shallow water grid may hold.
const MAX_CELLS: usize = 1 << 24;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShallowSettings {
    /// Width of a square cell.
    pub cell: f32,
    /// Downward pull, positive.
    pub gravity: f32,
    /// Share of a cell the fastest wave may cross in a substep. Over a quarter can push depth
    /// below zero.
    pub courant: f32,
    pub max_substeps: u32,
    /// Shallower than this a cell counts as dry and holds no flow.
    pub dry: f32,
}

impl Default for ShallowSettings {
    fn default() -> Self {
        Self {
            cell: 0.25,
            gravity: 9.81,
            courant: 0.25,
            max_substeps: 64,
            dry: 1e-4,
        }
    }
}

impl ShallowSettings {
    fn validate(&self) -> Result<(), PhysicsError> {
        let valid = self.cell.is_finite()
            && self.cell > 0.0
            && self.gravity.is_finite()
            && self.gravity > 0.0
            && self.courant > 0.0
            && self.courant <= 0.25
            && self.max_substeps > 0
            && self.dry.is_finite()
            && self.dry > 0.0;
        if valid {
            Ok(())
        } else {
            Err(PhysicsError::InvalidSettings)
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ShallowReport {
    pub substeps: u32,
    /// The flow wanted more substeps than allowed, the step went further than is stable.
    pub capped: bool,
    pub max_speed: f32,
}

/// Water as columns over a bed on a grid in x and z: depth and flow per cell. For rivers,
/// reservoirs, floods and seas, where water is wide and shallow next to its size. Cost goes with
/// cells, not with how much water there is.
///
/// Finite volumes with hydrostatic reconstruction and hll fluxes (audusse and others, 2004): still
/// water over any bed stays still, water is only moved between cells so it is kept exactly, and
/// cells run dry and wet again without going below zero. Edges of the grid are walls.
#[derive(Debug, Clone)]
pub struct ShallowWater {
    settings: ShallowSettings,
    /// Corner of cell (0, 0), its y is unused.
    origin: Vec3,
    size: (usize, usize),
    bed: Vec<f32>,
    depth: Vec<f32>,
    /// Depth times velocity, along x and along z.
    flow_x: Vec<f32>,
    flow_z: Vec<f32>,
}

/// Depth, flow along the face's normal and across it, on one side of a face.
#[derive(Debug, Clone, Copy)]
struct Side {
    depth: f32,
    along: f32,
    across: f32,
}

impl ShallowWater {
    /// A grid of `size.0` by `size.1` cells from `origin` in x and z, dry, over `bed` heights
    /// listed x fastest.
    ///
    /// # Errors
    /// Fails on bad settings, an empty or too big grid, a bed of the wrong length or a non
    /// finite value.
    pub fn new(
        settings: ShallowSettings,
        origin: Vec3,
        size: (usize, usize),
        bed: Vec<f32>,
    ) -> Result<Self, PhysicsError> {
        settings.validate()?;
        let count = size.0.saturating_mul(size.1);
        if size.0 == 0 || size.1 == 0 || count > MAX_CELLS || bed.len() != count {
            return Err(PhysicsError::OutOfRange);
        }
        if !origin.is_finite() || bed.iter().any(|height| !height.is_finite()) {
            return Err(PhysicsError::NonFinite("shallow water bed"));
        }
        Ok(Self {
            settings,
            origin,
            size,
            bed,
            depth: vec![0.0; count],
            flow_x: vec![0.0; count],
            flow_z: vec![0.0; count],
        })
    }

    /// [`Self::new`] with the bed read from terrain: at each cell's center, the first surface
    /// going down from `top`, `bottom` where none is found. `volume_origin` is where the
    /// volume's origin sits. Overhangs are not seen, only the top surface.
    ///
    /// # Errors
    /// As [`Self::new`], and on a non finite or upside down height range.
    pub fn over_terrain(
        settings: ShallowSettings,
        origin: Vec3,
        size: (usize, usize),
        volume: &SdfVolume,
        volume_origin: Vec3,
        (top, bottom): (f32, f32),
    ) -> Result<Self, PhysicsError> {
        if !(top.is_finite() && bottom.is_finite() && volume_origin.is_finite()) || top <= bottom {
            return Err(PhysicsError::OutOfRange);
        }
        settings.validate()?;
        let count = size.0.saturating_mul(size.1);
        if count > MAX_CELLS {
            return Err(PhysicsError::OutOfRange);
        }
        let mut bed = Vec::with_capacity(count);
        for z in 0..size.1 {
            for x in 0..size.0 {
                let center =
                    origin + Vec3::new(x as f32 + 0.5, 0.0, z as f32 + 0.5) * settings.cell;
                let distance =
                    |y: f32| volume.distance(Vec3::new(center.x, y, center.z) - volume_origin);
                bed.push(surface_below(distance, top, bottom, settings.cell * 1e-3));
            }
        }
        Self::new(settings, origin, size, bed)
    }

    pub fn settings(&self) -> &ShallowSettings {
        &self.settings
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Bed heights, x fastest.
    pub fn bed(&self) -> &[f32] {
        &self.bed
    }

    /// Water depths, x fastest.
    pub fn depths(&self) -> &[f32] {
        &self.depth
    }

    fn index(&self, x: usize, z: usize) -> usize {
        z * self.size.0 + x
    }

    /// The cell holding `at` in x and z.
    pub fn cell_at(&self, at: Vec3) -> Option<(usize, usize)> {
        let local = (at - self.origin) / self.settings.cell;
        let inside = |value: f32, cells: usize| value >= 0.0 && value < cells as f32;
        (inside(local.x, self.size.0) && inside(local.z, self.size.1))
            .then_some((local.x as usize, local.z as usize))
    }

    /// Water surface height at a cell, the bed where it is dry.
    pub fn surface(&self, (x, z): (usize, usize)) -> f32 {
        let index = self.index(x, z);
        self.bed[index] + self.depth[index]
    }

    /// Water velocity in a cell, zero where it is dry.
    pub fn velocity(&self, (x, z): (usize, usize)) -> Vec3 {
        let index = self.index(x, z);
        let depth = self.depth[index];
        if depth < self.settings.dry {
            return Vec3::ZERO;
        }
        Vec3::new(self.flow_x[index] / depth, 0.0, self.flow_z[index] / depth)
    }

    /// Raises the water in cells from `low` to `high` (inclusive) to `level` where the bed is
    /// lower, still. Returns the volume added.
    ///
    /// # Errors
    /// Fails on a non finite level or cells outside the grid.
    pub fn fill(
        &mut self,
        low: (usize, usize),
        high: (usize, usize),
        level: f32,
    ) -> Result<f64, PhysicsError> {
        if !level.is_finite() {
            return Err(PhysicsError::NonFinite("shallow water level"));
        }
        if high.0 >= self.size.0 || high.1 >= self.size.1 || low.0 > high.0 || low.1 > high.1 {
            return Err(PhysicsError::OutOfRange);
        }
        let area = f64::from(self.settings.cell).powi(2);
        let mut added = 0.0;
        for z in low.1..=high.1 {
            for x in low.0..=high.0 {
                let index = self.index(x, z);
                let wanted = (level - self.bed[index]).max(0.0);
                if wanted > self.depth[index] {
                    added += f64::from(wanted - self.depth[index]) * area;
                    self.depth[index] = wanted;
                    self.flow_x[index] = 0.0;
                    self.flow_z[index] = 0.0;
                }
            }
        }
        Ok(added)
    }

    /// Water held, cubic meters.
    pub fn volume(&self) -> f64 {
        let area = f64::from(self.settings.cell).powi(2);
        self.depth
            .iter()
            .map(|depth| f64::from(*depth))
            .sum::<f64>()
            * area
    }

    fn fastest_wave(&self) -> f32 {
        let gravity = self.settings.gravity;
        (0..self.depth.len())
            .filter(|&index| self.depth[index] >= self.settings.dry)
            .map(|index| {
                let depth = self.depth[index];
                let speed = (self.flow_x[index].abs()).max(self.flow_z[index].abs()) / depth;
                speed + (gravity * depth).sqrt()
            })
            .fold(0.0, f32::max)
    }

    /// Advances `dt` seconds in as many substeps as the fastest wave needs.
    ///
    /// # Errors
    /// Fails on a non finite or negative `dt`, or if the dispatcher skipped a job.
    pub fn step(
        &mut self,
        dt: f32,
        dispatch: &dyn JobDispatch,
    ) -> Result<ShallowReport, PhysicsError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(PhysicsError::NonFinite("shallow water step"));
        }
        let mut report = ShallowReport::default();
        let mut left = dt;
        while left > 0.0 {
            let wave = self.fastest_wave();
            let mut small = if wave > 0.0 {
                self.settings.courant * self.settings.cell / wave
            } else {
                left
            };
            if report.substeps + 1 >= self.settings.max_substeps && small < left {
                report.capped = true;
                small = left;
            }
            let small = small.min(left);
            self.substep(small, dispatch)?;
            report.substeps += 1;
            left -= small;
        }
        report.max_speed = (0..self.depth.len())
            .map(|index| {
                let (x, z) = (index % self.size.0, index / self.size.0);
                self.velocity((x, z)).length()
            })
            .fold(0.0, f32::max);
        Ok(report)
    }

    fn side(&self, index: usize, along_x: bool, bed: f32) -> Side {
        let depth = self.depth[index];
        let (along, across) = if depth < self.settings.dry {
            (0.0, 0.0)
        } else if along_x {
            (self.flow_x[index] / depth, self.flow_z[index] / depth)
        } else {
            (self.flow_z[index] / depth, self.flow_x[index] / depth)
        };
        // hydrostatic reconstruction: depth over the higher of the two beds
        Side {
            depth: (depth + self.bed[index] - bed).max(0.0),
            along,
            across,
        }
    }

    /// Flux through the face between cells `a` and `b`, and the pressure each side adds back
    /// for its own bed: mass, momentum along, momentum across.
    fn face(&self, a: usize, b: usize, along_x: bool) -> ([f32; 3], f32, f32) {
        let bed = self.bed[a].max(self.bed[b]);
        let left = self.side(a, along_x, bed);
        let right = self.side(b, along_x, bed);
        let gravity = self.settings.gravity;
        let flux = hll(left, right, gravity);
        let half = gravity * 0.5;
        let own =
            |index: usize, side: Side| half * (self.depth[index].powi(2) - side.depth.powi(2));
        (flux, own(a, left), own(b, right))
    }

    fn substep(&mut self, dt: f32, dispatch: &dyn JobDispatch) -> Result<(), PhysicsError> {
        let (nx, nz) = self.size;
        let half = self.settings.gravity * 0.5;
        let world = &*self;
        // faces x: between (x, z) and (x + 1, z), (nx - 1) per row; z: between (x, z) and (x, z + 1)
        let faces_x = in_chunks(dispatch, nz, |rows| {
            let mut out = Vec::with_capacity(rows.len() * nx.saturating_sub(1));
            for z in rows {
                for x in 0..nx - 1 {
                    out.push(world.face(world.index(x, z), world.index(x + 1, z), true));
                }
            }
            out
        })?
        .concat();
        let faces_z = in_chunks(dispatch, nz.saturating_sub(1), |rows| {
            let mut out = Vec::with_capacity(rows.len() * nx);
            for z in rows {
                for x in 0..nx {
                    out.push(world.face(world.index(x, z), world.index(x, z + 1), false));
                }
            }
            out
        })?
        .concat();
        let scale = dt / self.settings.cell;
        let dry = self.settings.dry;
        let next = in_chunks(dispatch, nz, |rows| {
            let mut out = Vec::with_capacity(rows.len() * nx);
            for z in rows {
                for x in 0..nx {
                    let index = world.index(x, z);
                    let depth = world.depth[index];
                    let wall = half * depth * depth;
                    let (mut mass, mut along_x, mut along_z) = (0.0, 0.0, 0.0);
                    // x faces: flux leaves through the right face, comes in through the left
                    let (right, left) = (
                        (x + 1 < nx).then(|| faces_x[z * (nx - 1) + x]),
                        (x > 0).then(|| faces_x[z * (nx - 1) + x - 1]),
                    );
                    match right {
                        Some((flux, own, _)) => {
                            mass -= flux[0];
                            along_x -= flux[1] + own;
                            along_z -= flux[2];
                        }
                        None => along_x -= wall,
                    }
                    match left {
                        Some((flux, _, own)) => {
                            mass += flux[0];
                            along_x += flux[1] + own;
                            along_z += flux[2];
                        }
                        None => along_x += wall,
                    }
                    let (up, down) = (
                        (z + 1 < nz).then(|| faces_z[z * nx + x]),
                        (z > 0).then(|| faces_z[(z - 1) * nx + x]),
                    );
                    match up {
                        Some((flux, own, _)) => {
                            mass -= flux[0];
                            along_z -= flux[1] + own;
                            along_x -= flux[2];
                        }
                        None => along_z -= wall,
                    }
                    match down {
                        Some((flux, _, own)) => {
                            mass += flux[0];
                            along_z += flux[1] + own;
                            along_x += flux[2];
                        }
                        None => along_z += wall,
                    }
                    let depth = (depth + scale * mass).max(0.0);
                    let (flow_x, flow_z) = if depth < dry {
                        (0.0, 0.0)
                    } else {
                        (
                            world.flow_x[index] + scale * along_x,
                            world.flow_z[index] + scale * along_z,
                        )
                    };
                    out.push((depth, flow_x, flow_z));
                }
            }
            out
        })?;
        for (index, (depth, flow_x, flow_z)) in next.into_iter().flatten().enumerate() {
            if !(depth.is_finite() && flow_x.is_finite() && flow_z.is_finite()) {
                return Err(PhysicsError::NonFinite("shallow water cell"));
            }
            self.depth[index] = depth;
            self.flow_x[index] = flow_x;
            self.flow_z[index] = flow_z;
        }
        Ok(())
    }

    /// Bits of every cell's state, x fastest.
    pub fn state_hash(&self) -> u64 {
        let mut bytes = Vec::with_capacity(self.depth.len() * 12);
        for list in [&self.depth, &self.flow_x, &self.flow_z] {
            for value in list {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        fnv1a(&bytes)
    }
}

/// Hll flux of mass, momentum along and across the face normal.
fn hll(left: Side, right: Side, gravity: f32) -> [f32; 3] {
    if left.depth <= 0.0 && right.depth <= 0.0 {
        return [0.0; 3];
    }
    let physical = |side: Side| {
        let flow = side.depth * side.along;
        [
            flow,
            flow * side.along + 0.5 * gravity * side.depth * side.depth,
            flow * side.across,
        ]
    };
    let (wave_left, wave_right) = (
        (gravity * left.depth).sqrt(),
        (gravity * right.depth).sqrt(),
    );
    // a dry side moves at the wet side's front speed
    let slow = if left.depth <= 0.0 {
        right.along - 2.0 * wave_right
    } else {
        (left.along - wave_left).min(right.along - wave_right)
    };
    let fast = if right.depth <= 0.0 {
        left.along + 2.0 * wave_left
    } else {
        (left.along + wave_left).max(right.along + wave_right)
    };
    if slow >= 0.0 {
        return physical(left);
    }
    if fast <= 0.0 {
        return physical(right);
    }
    let (a, b) = (physical(left), physical(right));
    let state = |side: Side| {
        [
            side.depth,
            side.depth * side.along,
            side.depth * side.across,
        ]
    };
    let (qa, qb) = (state(left), state(right));
    let mut flux = [0.0; 3];
    for part in 0..3 {
        flux[part] =
            (fast * a[part] - slow * b[part] + slow * fast * (qb[part] - qa[part])) / (fast - slow);
    }
    flux
}

/// First height going down from `top` where `distance` turns negative, to within `close`.
fn surface_below(distance: impl Fn(f32) -> f32, top: f32, bottom: f32, close: f32) -> f32 {
    let mut above = top;
    let mut here = distance(above);
    if here <= 0.0 {
        return top;
    }
    // a distance never overshoots the surface, so step down by it
    while above > bottom {
        let next = (above - here.max(close)).max(bottom);
        let there = distance(next);
        if there <= 0.0 {
            let mut below = next;
            while above - below > close {
                let middle = 0.5 * (above + below);
                if distance(middle) > 0.0 {
                    above = middle;
                } else {
                    below = middle;
                }
            }
            return 0.5 * (above + below);
        }
        above = next;
        here = there;
    }
    bottom
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

    const TICK: f32 = 1.0 / 60.0;

    fn bumpy(size: (usize, usize)) -> ShallowWater {
        let settings = ShallowSettings::default();
        let mut bed = Vec::new();
        for z in 0..size.1 {
            for x in 0..size.0 {
                bed.push(0.3 * (x as f32 * 0.7).sin() * (z as f32 * 0.4).cos());
            }
        }
        ShallowWater::new(settings, Vec3::ZERO, size, bed).unwrap()
    }

    #[test]
    fn still_water_over_a_bumpy_bed_stays_still() {
        let mut water = bumpy((24, 16));
        water.fill((0, 0), (23, 15), 1.0).unwrap();
        for _ in 0..120 {
            water.step(TICK, &Sequential).unwrap();
        }
        let report = water.step(TICK, &Sequential).unwrap();
        assert!(report.max_speed < 1e-4, "{report:?}");
        for z in 0..16 {
            for x in 0..24 {
                assert!((water.surface((x, z)) - 1.0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn water_is_kept_and_runs_dry_without_going_below_zero() {
        let mut water = bumpy((32, 12));
        // a column on one side runs out over the dry rest
        let start = water.fill((0, 0), (9, 11), 0.8).unwrap();
        for _ in 0..240 {
            water.step(TICK, &Sequential).unwrap();
        }
        let now = water.volume();
        assert!((now - start).abs() / start < 1e-4, "{now} {start}");
        assert!(water.depths().iter().all(|depth| *depth >= 0.0));
    }

    #[test]
    fn a_dam_break_on_a_dry_bed_runs_as_fast_as_ritter_says() {
        // flat dry bed, one meter of water behind x = 10 m, let go
        let settings = ShallowSettings {
            cell: 0.05,
            ..ShallowSettings::default()
        };
        let size = (800, 1);
        let mut water = ShallowWater::new(settings, Vec3::ZERO, size, vec![0.0; 800]).unwrap();
        water.fill((0, 0), (199, 0), 1.0).unwrap();
        let time = 1.0;
        for _ in 0..60 {
            water.step(time / 60.0, &Sequential).unwrap();
        }
        // ritter: the front runs at twice the wave speed of the still water, the middle stays at
        // four ninths of the depth
        let front = (0..800).rev().find(|&x| water.depths()[x] > 1e-3).unwrap() as f32 * 0.05;
        let ideal = 10.0 + 2.0 * (9.81_f32).sqrt() * time;
        assert!(
            front < ideal + 0.2 && front > ideal * 0.85,
            "{front} {ideal}"
        );
        let middle = water.depths()[200];
        assert!((middle - 4.0 / 9.0).abs() < 0.03, "{middle}");
    }

    #[test]
    fn any_thread_count_ends_on_the_same_state() {
        let run = |dispatch: &dyn JobDispatch| {
            let mut water = bumpy((40, 30));
            water.fill((0, 0), (12, 29), 0.9).unwrap();
            for _ in 0..30 {
                water.step(TICK, dispatch).unwrap();
            }
            water.state_hash()
        };
        let one = run(&Sequential);
        assert_eq!(run(&Threads(4)), one);
        assert_eq!(run(&Threads(7)), one);
    }

    #[test]
    fn bad_input_is_refused() {
        let settings = ShallowSettings::default();
        assert!(ShallowWater::new(settings, Vec3::ZERO, (2, 2), vec![0.0; 3]).is_err());
        assert!(ShallowWater::new(settings, Vec3::ZERO, (0, 2), Vec::new()).is_err());
        assert!(ShallowWater::new(settings, Vec3::ZERO, (1, 1), vec![f32::NAN]).is_err());
        let fast = ShallowSettings {
            courant: 0.6,
            ..settings
        };
        assert!(ShallowWater::new(fast, Vec3::ZERO, (1, 1), vec![0.0]).is_err());
        let mut water = ShallowWater::new(settings, Vec3::ZERO, (2, 2), vec![0.0; 4]).unwrap();
        assert!(water.step(f32::NAN, &Sequential).is_err());
        assert!(water.fill((0, 0), (2, 0), 1.0).is_err());
        assert!(water.fill((0, 0), (1, 1), f32::INFINITY).is_err());
    }

    #[test]
    fn the_bed_is_read_from_terrain() {
        use crate::material::Material;
        use crate::position::GridPosition;
        use crate::sdf::{Base, SdfDesc};
        let desc = SdfDesc {
            origin: GridPosition::from_local(Vec3::ZERO).unwrap(),
            voxel: 0.5,
            brick_cells: 8,
            bricks: [2, 2, 2],
            base: Base::Plane { height: 1.3 },
            material: Material::default(),
        };
        let volume = SdfVolume::new(desc).unwrap();
        let water = ShallowWater::over_terrain(
            ShallowSettings::default(),
            Vec3::ONE,
            (4, 4),
            &volume,
            Vec3::ZERO,
            (6.0, -1.0),
        )
        .unwrap();
        for height in water.bed() {
            assert!((height - 1.3).abs() < 1e-2, "{height}");
        }
    }
}
