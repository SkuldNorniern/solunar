use algebrix::Vec3;

use super::couple::shape_points;
use super::in_chunks_of;
use crate::body::BodyHandle;
use crate::dispatch::JobDispatch;
use crate::error::PhysicsError;
use crate::sdf::SdfVolume;
use crate::shape::Shape;
use crate::snapshot::fnv1a;
use crate::world::PhysicsWorld;

/// Most cells one shallow water grid may hold.
const MAX_CELLS: usize = 1 << 24;
/// About this many cells to a job, a job is whole rows.
const CELLS_PER_JOB: usize = 4096;
/// Most points one held body is sampled with.
const MAX_BODY_POINTS: usize = 1 << 16;
/// Drag of a body point against the flow, as for a blunt body.
const DRAG: f32 = 1.0;

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
    /// Of the water, for what it does to bodies.
    pub density: f32,
}

impl Default for ShallowSettings {
    fn default() -> Self {
        Self {
            cell: 0.25,
            gravity: 9.81,
            courant: 0.25,
            max_substeps: 64,
            dry: 1e-4,
            density: 1000.0,
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
            && self.dry > 0.0
            && self.density.is_finite()
            && self.density > 0.0;
        if valid {
            Ok(())
        } else {
            Err(PhysicsError::InvalidSettings)
        }
    }
}

/// How a loose bed washes away under flowing water: past a shear it can take it goes at a rate
/// set by how much past, and a wet bank steeper than it can stand slumps. What washes off leaves
/// the water, it is not carried and laid down again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Erosion {
    /// Shear the bed takes before it starts to wash away, pascals.
    pub critical: f32,
    /// Meters a second it goes per pascal past `critical`.
    pub rate: f32,
    /// Share of the water's speed squared, times its density, that pulls on the bed.
    pub friction: f32,
    /// Steepest a bank beside water stands, rise over run.
    pub repose: f32,
}

impl Default for Erosion {
    /// Packed earth fill.
    fn default() -> Self {
        Self {
            critical: 5.0,
            rate: 2e-4,
            friction: 0.003,
            repose: 0.7,
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
    /// The bed raised to the top of any body standing on it, this step. Water over a low body
    /// runs over it, a tall one holds it back, one that falls in lifts the water.
    floor: Vec<f32>,
    /// Lowest the bed washes down to, the bed itself where it does not wash at all.
    hard: Vec<f32>,
    held: Vec<Held>,
}

/// A body the water pushes and is held back by, as points through its volume.
#[derive(Debug, Clone)]
struct Held {
    body: BodyHandle,
    points: Vec<Vec3>,
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
            floor: bed.clone(),
            hard: bed.clone(),
            bed,
            depth: vec![0.0; count],
            flow_x: vec![0.0; count],
            flow_z: vec![0.0; count],
            held: Vec::new(),
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

    /// The bed with bodies standing on it, x fastest. The water sits on this.
    pub fn floors(&self) -> &[f32] {
        &self.floor
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
        self.floor[index] + self.depth[index]
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
                let wanted = (level - self.floor[index]).max(0.0);
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

    fn rows_per_job(&self) -> usize {
        (CELLS_PER_JOB / self.size.0).max(1)
    }

    /// The fastest wave and the fastest flow over all wet cells.
    fn fastest(&self, dispatch: &dyn JobDispatch) -> Result<(f32, f32), PhysicsError> {
        let (nx, gravity, dry) = (self.size.0, self.settings.gravity, self.settings.dry);
        let parts = in_chunks_of(dispatch, self.size.1, self.rows_per_job(), |rows| {
            let mut most = (0.0_f32, 0.0_f32);
            for index in rows.start * nx..rows.end * nx {
                let depth = self.depth[index];
                if depth < dry {
                    continue;
                }
                let (x, z) = (self.flow_x[index] / depth, self.flow_z[index] / depth);
                let wave = x.abs().max(z.abs()) + (gravity * depth).sqrt();
                most = (most.0.max(wave), most.1.max((x * x + z * z).sqrt()));
            }
            most
        })?;
        Ok(parts
            .into_iter()
            .fold((0.0, 0.0), |a, b| (a.0.max(b.0), a.1.max(b.1))))
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
            let (wave, _) = self.fastest(dispatch)?;
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
        report.max_speed = self.fastest(dispatch)?.1;
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
            depth: (depth + self.floor[index] - bed).max(0.0),
            along,
            across,
        }
    }

    /// Flux through the face between cells `a` and `b`, and the pressure each side adds back
    /// for its own bed: mass, momentum along, momentum across.
    fn face(&self, a: usize, b: usize, along_x: bool) -> ([f32; 3], f32, f32) {
        let bed = self.floor[a].max(self.floor[b]);
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
        let scale = dt / self.settings.cell;
        let dry = self.settings.dry;
        let world = &*self;
        // each job works out the faces its rows touch, the z faces on its edges twice, and its
        // rows' next state: nothing is shared or copied between jobs
        let next = in_chunks_of(dispatch, nz, self.rows_per_job(), |rows| {
            // x faces between (x, z) and (x + 1, z), nx - 1 a row
            let mut faces_x = Vec::with_capacity(rows.len() * nx.saturating_sub(1));
            for z in rows.clone() {
                for x in 0..nx - 1 {
                    faces_x.push(world.face(world.index(x, z), world.index(x + 1, z), true));
                }
            }
            // z faces between (x, z) and (x, z + 1), from the row before the first
            let first = rows.start.saturating_sub(1);
            let last = rows.end.min(nz - 1);
            let mut faces_z = Vec::with_capacity((last - first) * nx);
            for z in first..last {
                for x in 0..nx {
                    faces_z.push(world.face(world.index(x, z), world.index(x, z + 1), false));
                }
            }
            let cells = rows.len() * nx;
            let mut out = (
                Vec::with_capacity(cells),
                Vec::with_capacity(cells),
                Vec::with_capacity(cells),
            );
            for z in rows.clone() {
                let row_x = (z - rows.start) * (nx - 1);
                for x in 0..nx {
                    let index = world.index(x, z);
                    let depth = world.depth[index];
                    let wall = half * depth * depth;
                    let (mut mass, mut along_x, mut along_z) = (0.0, 0.0, 0.0);
                    // x faces: flux leaves through the right face, comes in through the left
                    let (right, left) = (
                        (x + 1 < nx).then(|| faces_x[row_x + x]),
                        (x > 0).then(|| faces_x[row_x + x - 1]),
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
                        (z + 1 < nz).then(|| faces_z[(z - first) * nx + x]),
                        (z > 0).then(|| faces_z[(z - 1 - first) * nx + x]),
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
                    out.0.push(depth);
                    out.1.push(flow_x);
                    out.2.push(flow_z);
                }
            }
            out
        })?;
        let mut at = 0;
        for (depth, flow_x, flow_z) in next {
            let end = at + depth.len();
            let finite = |list: &[f32]| list.iter().all(|value| value.is_finite());
            if !(finite(&depth) && finite(&flow_x) && finite(&flow_z)) {
                return Err(PhysicsError::NonFinite("shallow water cell"));
            }
            self.depth[at..end].copy_from_slice(&depth);
            self.flow_x[at..end].copy_from_slice(&flow_x);
            self.flow_z[at..end].copy_from_slice(&flow_z);
            at = end;
        }
        Ok(())
    }

    /// Sets the bed of a cell, the water on it keeps its depth.
    ///
    /// # Errors
    /// Fails on a non finite height.
    pub fn set_bed(&mut self, (x, z): (usize, usize), height: f32) -> Result<(), PhysicsError> {
        if !height.is_finite() {
            return Err(PhysicsError::NonFinite("shallow water bed"));
        }
        let index = self.index(x, z);
        let loose = self.hard[index] < self.bed[index];
        self.bed[index] = height;
        self.floor[index] = height;
        self.hard[index] = if loose {
            self.hard[index].min(height)
        } else {
            height
        };
        Ok(())
    }

    /// Lets the bed of a cell wash down as far as `lowest`.
    ///
    /// # Errors
    /// Fails on a non finite height.
    pub fn make_loose(&mut self, (x, z): (usize, usize), lowest: f32) -> Result<(), PhysicsError> {
        if !lowest.is_finite() {
            return Err(PhysicsError::NonFinite("shallow water bed"));
        }
        let index = self.index(x, z);
        self.hard[index] = lowest.min(self.bed[index]);
        Ok(())
    }

    /// Washes the loose bed away for `dt` seconds of the flow as it is. Returns the volume of
    /// bed taken, cubic meters. Bodies standing on the bed shield it.
    ///
    /// # Errors
    /// Fails on a non finite or negative `dt` or bad settings.
    pub fn erode(&mut self, dt: f32, erosion: &Erosion) -> Result<f64, PhysicsError> {
        let valid = [
            erosion.critical,
            erosion.rate,
            erosion.friction,
            erosion.repose,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
            && erosion.repose > 0.0;
        if !valid {
            return Err(PhysicsError::InvalidSettings);
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(PhysicsError::NonFinite("erosion step"));
        }
        let (nx, nz) = self.size;
        let cell = self.settings.cell;
        let dry = self.settings.dry;
        let before = self.bed.clone();
        let loose = |index: usize| self.hard[index] < before[index];
        let mut bed = before.clone();
        // scour under fast water
        for index in 0..bed.len() {
            let depth = self.depth[index];
            if depth < dry || !loose(index) || self.floor[index] > before[index] {
                continue;
            }
            let speed_squared =
                (self.flow_x[index].powi(2) + self.flow_z[index].powi(2)) / (depth * depth);
            let shear = self.settings.density * erosion.friction * speed_squared;
            if shear > erosion.critical {
                let cut = erosion.rate * (shear - erosion.critical) * dt;
                bed[index] = (bed[index] - cut).max(self.hard[index]);
            }
        }
        // a loose bank over wet ground steeper than it stands slumps to that slope, the high
        // side only, so a breach opens wider
        let drop = erosion.repose * cell;
        for z in 0..nz {
            for x in 0..nx {
                let index = self.index(x, z);
                if !loose(index) || self.floor[index] > before[index] {
                    continue;
                }
                let around = [
                    (x + 1 < nx).then(|| index + 1),
                    (x > 0).then(|| index - 1),
                    (z + 1 < nz).then(|| index + nx),
                    (z > 0).then(|| index - nx),
                ];
                for near in around.into_iter().flatten() {
                    if self.depth[near] < dry {
                        continue;
                    }
                    let lowest = before[near] + drop;
                    if bed[index] > lowest {
                        bed[index] = lowest.max(self.hard[index]);
                    }
                }
            }
        }
        let mut taken = 0.0;
        for (index, height) in bed.into_iter().enumerate() {
            let cut = before[index] - height;
            if cut > 0.0 {
                taken += f64::from(cut);
                self.bed[index] = height;
                if self.floor[index] <= before[index] {
                    self.floor[index] = height;
                }
            }
        }
        Ok(taken * f64::from(cell * cell))
    }

    /// Lets `body` hold the water back where it stands on the bed, and be pushed by it: the
    /// water's pressure on the faces it blocks, lift where it is under the surface, drag of the
    /// flow. The water does not feel the drag back. Coupling a body again replaces it.
    ///
    /// # Errors
    /// Fails on a shape that can not be sampled.
    pub fn couple(&mut self, body: BodyHandle, shape: &Shape) -> Result<(), PhysicsError> {
        let spacing = self.settings.cell / 2.0;
        let points = shape_points(shape, spacing, f32::INFINITY, MAX_BODY_POINTS)?;
        if points.is_empty() {
            return Err(PhysicsError::InvalidShape);
        }
        self.held.retain(|held| held.body != body);
        self.held.push(Held { body, points });
        self.held.sort_by_key(|held| held.body);
        Ok(())
    }

    /// Returns false for a body that was not coupled.
    pub fn uncouple(&mut self, body: BodyHandle) -> bool {
        let before = self.held.len();
        self.held.retain(|held| held.body != body);
        self.held.len() != before
    }

    /// Advances the water `dt` seconds held back by the coupled bodies where `world` has them,
    /// then hands each dynamic body the water's push as force and torque for the world's next
    /// step. Call it right before that step. Bodies gone from the world are dropped.
    ///
    /// # Errors
    /// Fails like [`Self::step`].
    pub fn step_with_bodies(
        &mut self,
        dt: f32,
        world: &mut PhysicsWorld,
        dispatch: &dyn JobDispatch,
    ) -> Result<ShallowReport, PhysicsError> {
        self.held.retain(|held| world.body_state(held.body).is_ok());
        let spacing = self.settings.cell / 2.0;
        // each body's points in the world, with their velocity
        let mut placed = Vec::with_capacity(self.held.len());
        let mut standing = vec![false; self.depth.len()];
        // body points by the cell they are in, cell then owner then order
        let mut in_cells: Vec<(usize, u32, Vec3)> = Vec::new();
        for (owner, held) in self.held.iter().enumerate() {
            let state = world.body_state(held.body)?;
            let center = state.position.to_dvec3().as_vec3();
            let mut points = Vec::with_capacity(held.points.len());
            for local in &held.points {
                let arm = state.rotation.rotate_vec3(*local);
                let at = center + arm;
                let velocity = state.linear_velocity + state.angular_velocity.cross(arm);
                if let Some(cell) = self.cell_at(at) {
                    let index = self.index(cell.0, cell.1);
                    // standing on the bed: nothing gets under it
                    if at.y <= self.bed[index] + spacing {
                        standing[index] = true;
                    }
                    in_cells.push((index, owner as u32, at));
                }
                points.push((at, velocity));
            }
            placed.push((center, points));
        }
        in_cells.sort_by_key(|item| (item.0, item.1));
        // the floor rises through body points stacked up from the bed with no gap, a body in the
        // air over the cell lifted the water up to it
        let before = std::mem::replace(&mut self.floor, self.bed.clone());
        let mut stacked: Vec<(usize, f32)> = in_cells
            .iter()
            .filter(|item| standing[item.0])
            .map(|item| (item.0, item.2.y))
            .collect();
        stacked.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        for (index, y) in stacked {
            let floor = &mut self.floor[index];
            if y - 2.0 * spacing <= *floor {
                *floor = floor.max(y + spacing / 2.0);
            }
        }
        self.displace(&before);
        let report = self.step(dt, dispatch)?;
        if dt <= 0.0 {
            return Ok(report);
        }
        let settings = self.settings;
        let weight = settings.density * settings.gravity;
        let mut pushes = vec![(Vec3::ZERO, Vec3::ZERO); self.held.len()];
        // pressure on each face a body blocks, from the water beside it, shared by the points in
        // the cell beside at their own depth: a dam's bottom block takes the bottom's push, not
        // the whole column's
        let layer = (settings.cell / spacing).powi(2);
        let (nx, nz) = self.size;
        for z in 0..nz {
            for x in 0..nx {
                let index = self.index(x, z);
                let depth = self.depth[index];
                if depth < settings.dry {
                    continue;
                }
                let floor = self.floor[index];
                let surface = floor + depth;
                let sides = [
                    (x + 1 < nx, (x + 1, z), Vec3::X),
                    (x > 0, (x.wrapping_sub(1), z), -Vec3::X),
                    (z + 1 < nz, (x, z + 1), Vec3::Z),
                    (z > 0, (x, z.wrapping_sub(1)), -Vec3::Z),
                ];
                for (inside, (bx, bz), toward) in sides {
                    if !inside {
                        continue;
                    }
                    let beside = self.index(bx, bz);
                    if !standing[beside] {
                        continue;
                    }
                    let start = in_cells.partition_point(|item| item.0 < beside);
                    let end = in_cells.partition_point(|item| item.0 <= beside);
                    for &(_, owner, at) in &in_cells[start..end] {
                        if at.y >= surface || at.y < floor {
                            continue;
                        }
                        let pressure = weight * (surface - at.y);
                        let force = toward * (pressure * spacing * settings.cell / layer);
                        let (center, _) = &placed[owner as usize];
                        let push = &mut pushes[owner as usize];
                        push.0 += force;
                        push.1 += (at - *center).cross(force);
                    }
                }
            }
        }
        // water over a body standing on the bed weighs on its highest point there
        let mut start = 0;
        while start < in_cells.len() {
            let index = in_cells[start].0;
            let end = start + in_cells[start..].partition_point(|item| item.0 == index);
            let depth = self.depth[index];
            if standing[index] && depth >= settings.dry {
                let top = in_cells[start..end]
                    .iter()
                    .copied()
                    .reduce(|best, item| if item.2.y > best.2.y { item } else { best });
                if let Some((_, owner, at)) = top {
                    let force = Vec3::Y * -(weight * depth * settings.cell * settings.cell);
                    let (center, _) = &placed[owner as usize];
                    let push = &mut pushes[owner as usize];
                    push.0 += force;
                    push.1 += (at - *center).cross(force);
                }
            }
            start = end;
        }
        // lift on each point under the surface, drag on the body as a whole: its points summed
        // one by one made a body's drag area grow as its volume over the point spacing, and a
        // block thrown through the water stopped dead
        let volume = spacing.powi(3);
        for (owner, (center, points)) in placed.iter().enumerate() {
            let (mut under, mut middle, mut past) = (0_u32, Vec3::ZERO, Vec3::ZERO);
            let mut waterline = 0_u32;
            for (at, velocity) in points {
                let Some(cell) = self.cell_at(*at) else {
                    continue;
                };
                let index = self.index(cell.0, cell.1);
                // inside a raised floor is inside the body, not in water
                let floor = self.floor[index];
                let surface = floor + self.depth[index];
                if at.y >= surface || at.y < floor {
                    continue;
                }
                under += 1;
                waterline += u32::from(at.y >= surface - spacing);
                middle += *at;
                past += self.velocity(cell) - *velocity;
            }
            if under == 0 {
                continue;
            }
            let share = 1.0 / under as f32;
            let (middle, past) = (middle * share, past * share);
            let wetted = (under as f32 * volume).powf(2.0 / 3.0);
            // a body bobbing in water sends waves off, the columns do not get them from it, so
            // this stands in: damping on its waterplane at the speed of such waves
            let plane = waterline as f32 * spacing * spacing;
            let waves = settings.density * plane * (settings.gravity * plane.sqrt()).sqrt();
            let force = Vec3::Y * (weight * volume * under as f32 + waves * past.y)
                + past * (0.5 * DRAG * settings.density * wetted * past.length());
            let push = &mut pushes[owner];
            push.0 += force;
            push.1 += (middle - *center).cross(force);
        }
        for (held, (force, torque)) in self.held.iter().zip(pushes) {
            // a fixed or kinematic body holds the water but is not pushed
            if world.apply_force(held.body, force).is_ok() {
                world.apply_torque(held.body, torque)?;
            }
        }
        Ok(report)
    }

    /// Where a floor rose under water, the water it took the place of goes to the lower cells
    /// around, kept. Carried up on the floor instead, it was lifted over the top of a dam whose
    /// blocks shifted a little and ran down its back.
    fn displace(&mut self, before: &[f32]) {
        let (nx, nz) = self.size;
        let mut added = vec![0.0_f32; self.depth.len()];
        for z in 0..nz {
            for x in 0..nx {
                let index = self.index(x, z);
                let rise = self.floor[index] - before[index];
                let depth = self.depth[index];
                if rise <= 0.0 || depth <= 0.0 {
                    continue;
                }
                let lower: Vec<usize> = [
                    (x + 1 < nx).then(|| index + 1),
                    (x > 0).then(|| index - 1),
                    (z + 1 < nz).then(|| index + nx),
                    (z > 0).then(|| index - nx),
                ]
                .into_iter()
                .flatten()
                .filter(|&near| self.floor[near] < self.floor[index])
                .collect();
                if lower.is_empty() {
                    continue;
                }
                let moved = rise.min(depth);
                let kept = depth - moved;
                let scale = kept / depth;
                self.depth[index] = kept;
                self.flow_x[index] *= scale;
                self.flow_z[index] *= scale;
                let share = moved / lower.len() as f32;
                for near in lower {
                    added[near] += share;
                }
            }
        }
        for (depth, more) in self.depth.iter_mut().zip(added) {
            *depth += more;
        }
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
    use crate::body::RigidBodyDesc;
    use crate::dispatch::Sequential;
    use crate::world::PhysicsSettings;

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

    fn channel() -> (ShallowWater, PhysicsWorld) {
        // 30 by 8 cells of 0.25 m on a flat bed at y = 0
        let water = ShallowWater::new(
            ShallowSettings::default(),
            Vec3::ZERO,
            (30, 8),
            vec![0.0; 240],
        )
        .unwrap();
        (
            water,
            PhysicsWorld::new(PhysicsSettings::default()).unwrap(),
        )
    }

    /// A box across the channel at x = 3.75 m, standing on the bed.
    fn wall(world: &mut PhysicsWorld, desc: fn(Shape) -> RigidBodyDesc) -> (BodyHandle, Shape) {
        let shape = Shape::cuboid(Vec3::new(0.25, 0.75, 1.0));
        let body = world
            .spawn(desc(shape).local_position(Vec3::new(3.75, 0.75, 1.0)))
            .unwrap();
        (body, shape)
    }

    /// A loose ridge across the channel at x = 3.75 m, its top at 1.2 m, rock at 0.2 m, with a
    /// notch 0.2 m deep in its middle and a lake behind it to 1.3 m.
    fn ridge() -> ShallowWater {
        let (mut water, _) = channel();
        for z in 0..8 {
            for x in 13..18 {
                let notch = if (3..5).contains(&z) { 0.2 } else { 0.0 };
                water.set_bed((x, z), 1.2 - notch).unwrap();
                water.make_loose((x, z), 0.2).unwrap();
            }
        }
        water.fill((0, 0), (12, 7), 1.3).unwrap();
        water
    }

    #[test]
    fn still_water_does_not_wash_a_bank_away() {
        let (mut water, _) = channel();
        for z in 0..8 {
            water.set_bed((15, z), 1.0).unwrap();
            water.make_loose((15, z), 0.0).unwrap();
        }
        water.fill((0, 0), (14, 7), 0.5).unwrap();
        let mut taken = 0.0;
        for _ in 0..120 {
            water.step(TICK, &Sequential).unwrap();
            taken += water.erode(TICK, &Erosion::default()).unwrap();
        }
        // only the slump of the bank to its slope, then nothing
        let slope = Erosion::default().repose * water.settings.cell;
        assert!(
            taken < f64::from(8.0 * (1.0 - slope) * 0.0625) + 1e-3,
            "{taken}"
        );
    }

    #[test]
    fn water_through_a_notch_cuts_it_deeper_and_wider() {
        let mut water = ridge();
        let start = water.volume();
        let erosion = Erosion {
            rate: 1e-3,
            ..Erosion::default()
        };
        for _ in 0..600 {
            water.step(TICK, &Sequential).unwrap();
            water.erode(TICK, &erosion).unwrap();
        }
        let crest = |water: &ShallowWater, z: usize| {
            (13..18)
                .map(|x| water.bed()[water.index(x, z)])
                .fold(f32::MAX, f32::min)
        };
        // the notch went down, and the ridge beside it too
        assert!(crest(&water, 3) < 0.9, "{}", crest(&water, 3));
        assert!(crest(&water, 2) < 1.1, "{}", crest(&water, 2));
        assert!(water.bed().iter().all(|height| *height >= 0.0));
        for z in 0..8 {
            for x in 13..18 {
                assert!(water.bed()[water.index(x, z)] >= 0.2 - 1e-6);
            }
        }
        // water is not lost to the bed going
        assert!((water.volume() - start).abs() / start < 1e-4);
    }

    #[test]
    fn a_fixed_wall_holds_the_water_back() {
        let (mut water, mut world) = channel();
        let (body, shape) = wall(&mut world, RigidBodyDesc::fixed);
        water.couple(body, &shape).unwrap();
        let start = water.fill((0, 0), (13, 7), 1.0).unwrap();
        for _ in 0..120 {
            water
                .step_with_bodies(TICK, &mut world, &Sequential)
                .unwrap();
            world.step_fixed().unwrap();
        }
        for z in 0..8 {
            for x in 16..30 {
                assert!(water.depths()[water.index(x, z)] < 1e-3, "{x} {z}");
            }
        }
        assert!((water.volume() - start).abs() / start < 1e-4);
    }

    #[test]
    fn water_runs_over_a_low_sill() {
        let (mut water, mut world) = channel();
        let shape = Shape::cuboid(Vec3::new(0.25, 0.25, 1.0));
        let desc = RigidBodyDesc::fixed(shape).local_position(Vec3::new(3.75, 0.25, 1.0));
        let body = world.spawn(desc).unwrap();
        water.couple(body, &shape).unwrap();
        let start = water.fill((0, 0), (13, 7), 1.0).unwrap();
        for _ in 0..120 {
            water
                .step_with_bodies(TICK, &mut world, &Sequential)
                .unwrap();
            world.step_fixed().unwrap();
        }
        // over the sill and on down the channel, the water on it counted
        assert!(water.depths()[water.index(25, 4)] > 0.05);
        assert!((water.volume() - start).abs() / start < 1e-4);
        let sill = water.index(15, 4);
        assert!(
            (water.floors()[sill] - 0.5).abs() < 0.07,
            "{}",
            water.floors()[sill]
        );
    }

    #[test]
    fn a_block_set_in_still_water_pushes_it_aside() {
        let (mut water, mut world) = channel();
        let start = water.fill((0, 0), (29, 7), 1.0).unwrap();
        let shape = Shape::cuboid(Vec3::new(0.25, 0.25, 1.0));
        let desc = RigidBodyDesc::fixed(shape).local_position(Vec3::new(3.75, 0.25, 1.0));
        let body = world.spawn(desc).unwrap();
        water.couple(body, &shape).unwrap();
        water
            .step_with_bodies(TICK, &mut world, &Sequential)
            .unwrap();
        // the surface over the block stays near where it was, not half a meter up
        let sill = water.index(15, 4);
        assert!(water.surface((15, 4)) < 1.1, "{}", water.surface((15, 4)));
        assert!(water.depths()[sill] < 0.6);
        assert!((water.volume() - start).abs() / start < 1e-4);
    }

    #[test]
    fn water_pushes_a_dam_block_downstream() {
        let (mut water, mut world) = channel();
        let (body, shape) = wall(&mut world, |shape| {
            RigidBodyDesc::dynamic(shape).density(2400.0)
        });
        water.couple(body, &shape).unwrap();
        water.fill((0, 0), (13, 7), 1.2).unwrap();
        water
            .step_with_bodies(TICK, &mut world, &Sequential)
            .unwrap();
        world.step_fixed().unwrap();
        let velocity = world.body_state(body).unwrap().linear_velocity;
        // 1.2 m of water on a 2 m wide face pushes 1000 * 9.81 * 1.2^2 / 2 * 2 = 14.1 kN, on
        // 3600 kg that is 0.065 m/s in a tick
        let ideal = 14126.0 / 3600.0 * TICK;
        assert!(
            (velocity.x - ideal).abs() < ideal * 0.15,
            "{velocity:?} {ideal}"
        );
        assert!(velocity.z.abs() < velocity.x * 0.01, "{velocity:?}");
    }

    #[test]
    fn a_block_thrown_through_water_is_slowed_by_its_front_only() {
        let (mut water, mut world) = channel();
        water.fill((0, 0), (29, 7), 3.0).unwrap();
        let shape = Shape::cuboid(Vec3::splat(0.5));
        let desc = RigidBodyDesc::dynamic(shape)
            .density(2400.0)
            .local_position(Vec3::new(1.0, 1.5, 1.0))
            .linear_velocity(Vec3::new(10.0, 0.0, 0.0));
        let body = world.spawn(desc).unwrap();
        water.couple(body, &shape).unwrap();
        for _ in 0..15 {
            water
                .step_with_bodies(TICK, &mut world, &Sequential)
                .unwrap();
            world.step_fixed().unwrap();
        }
        // half of 1000 * 1 m2 * v^2 on 2400 kg: 10 m/s falls to about 6.5 in a quarter second
        let speed = world.body_state(body).unwrap().linear_velocity.x;
        assert!(speed > 5.5 && speed < 8.0, "{speed}");
    }

    #[test]
    fn a_light_box_floats_half_under() {
        let (mut water, mut world) = channel();
        water.fill((0, 0), (29, 7), 2.0).unwrap();
        let shape = Shape::cuboid(Vec3::splat(0.4));
        let desc = RigidBodyDesc::dynamic(shape)
            .density(500.0)
            .local_position(Vec3::new(3.75, 2.6, 1.0));
        let body = world.spawn(desc).unwrap();
        water.couple(body, &shape).unwrap();
        for _ in 0..300 {
            water
                .step_with_bodies(TICK, &mut world, &Sequential)
                .unwrap();
            world.step_fixed().unwrap();
        }
        let state = world.body_state(body).unwrap();
        let height = state.position.local.y;
        assert!((height - 2.0).abs() < 0.1, "{height}");
        assert!(state.linear_velocity.length() < 0.2, "{state:?}");
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
