use algebrix::Vec3;

use super::{CellKey, FluidWorld, Grid};
use crate::error::PhysicsError;
use crate::snapshot::{Reader, SnapshotError, Writer};

/// Cell, owner and position bits, then where it was: a total order for the walls.
type WallKey = (CellKey, u32, (u32, u32, u32), usize);

/// One wall or container, so it can be taken away again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BoundaryId(u32);

impl BoundaryId {
    pub(super) fn raw(self) -> u32 {
        self.0
    }
}

/// Solid walls as frozen water: particles on the fluid's lattice with its particle mass, as deep
/// as the kernel reaches. Still water next to a wall then sums to rest density, corners too. Walls
/// that overlap count twice.
#[derive(Debug, Clone, Default)]
pub(super) struct Walls {
    pub(super) positions: Vec<Vec3>,
    owners: Vec<u32>,
    pub(super) grid: Grid,
    next: u32,
    /// Containers' insides, together. Particles that still get past a wall are held here and
    /// counted.
    pub(super) bounds: Option<(Vec3, Vec3)>,
    containers: Vec<(u32, Vec3, Vec3)>,
}

impl FluidWorld {
    /// A closed box with its inside from `low` to `high`. Fill it from `low` so the water sits on
    /// the walls' lattice.
    ///
    /// # Errors
    /// Fails on a non finite or inside out box, or more wall particles than the fluid may hold.
    pub fn add_box_container(&mut self, low: Vec3, high: Vec3) -> Result<BoundaryId, PhysicsError> {
        check_box(low, high)?;
        let cells = self.cells_in(high - low);
        let layers = self.settings.support.ceil() as i64;
        let span = |count: i64| -layers..count + layers;
        let shell = (cells.0 + 2 * layers) * (cells.1 + 2 * layers) * (cells.2 + 2 * layers)
            - cells.0 * cells.1 * cells.2;
        self.check_room(shell)?;
        let spacing = self.settings.spacing;
        let mut points = Vec::with_capacity(shell as usize);
        for y in span(cells.1) {
            for z in span(cells.2) {
                for x in span(cells.0) {
                    let inside = (0..cells.0).contains(&x)
                        && (0..cells.1).contains(&y)
                        && (0..cells.2).contains(&z);
                    if !inside {
                        let at = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                        points.push(low + at * spacing);
                    }
                }
            }
        }
        let id = self.add_wall_points(points)?;
        self.walls.containers.push((id.0, low, high));
        self.walls.bounds = union(&self.walls.containers);
        Ok(id)
    }

    /// A solid block from `low` to `high`, such as a gate to take away later.
    ///
    /// # Errors
    /// Fails on a non finite or inside out box, or too many wall particles.
    pub fn add_wall(&mut self, low: Vec3, high: Vec3) -> Result<BoundaryId, PhysicsError> {
        check_box(low, high)?;
        let cells = self.cells_in(high - low);
        let cells = (cells.0.max(1), cells.1.max(1), cells.2.max(1));
        let count = cells.0 * cells.1 * cells.2;
        self.check_room(count)?;
        let spacing = self.settings.spacing;
        let mut points = Vec::with_capacity(count as usize);
        for y in 0..cells.1 {
            for z in 0..cells.2 {
                for x in 0..cells.0 {
                    let at = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    points.push(low + at * spacing);
                }
            }
        }
        self.add_wall_points(points)
    }

    /// Returns false for an id that is not there.
    pub fn remove_boundary(&mut self, id: BoundaryId) -> bool {
        let before = self.walls.positions.len();
        let owners = &self.walls.owners;
        let mut index = 0;
        self.walls.positions.retain(|_| {
            index += 1;
            owners[index - 1] != id.0
        });
        self.walls.owners.retain(|owner| *owner != id.0);
        let had_container = self.walls.containers.iter().any(|item| item.0 == id.0);
        self.walls.containers.retain(|item| item.0 != id.0);
        self.walls.bounds = union(&self.walls.containers);
        let terrains = self.terrains.len();
        self.terrains.retain(|field| field.owner != id.0);
        let gone = self.walls.positions.len() != before
            || had_container
            || self.terrains.len() != terrains;
        if gone {
            self.settle_walls();
        }
        gone
    }

    /// Wall particles, for drawing or checks.
    pub fn wall_positions(&self) -> &[Vec3] {
        &self.walls.positions
    }

    /// Whole lattice cells along each side, for an extent known to be finite and not negative.
    fn cells_in(&self, extent: Vec3) -> (i64, i64, i64) {
        let count = |length: f32| (length / self.settings.spacing).round().min(1e9) as i64;
        (count(extent.x), count(extent.y), count(extent.z))
    }

    fn check_room(&self, count: i64) -> Result<(), PhysicsError> {
        let room = self
            .settings
            .max_particles
            .saturating_sub(self.walls.positions.len());
        if count < 0 || count as u64 > room as u64 || self.walls.next == u32::MAX {
            return Err(PhysicsError::FluidFull);
        }
        Ok(())
    }

    pub(super) fn add_wall_points(
        &mut self,
        points: Vec<Vec3>,
    ) -> Result<BoundaryId, PhysicsError> {
        let id = self.walls.next;
        self.walls.next += 1;
        self.walls
            .owners
            .extend(std::iter::repeat_n(id, points.len()));
        self.walls.positions.extend(points);
        self.settle_walls();
        Ok(BoundaryId(id))
    }

    /// Sorts the walls by cell. Walls do not move, so this only runs when one is added or taken
    /// away.
    pub(super) fn settle_walls(&mut self) {
        let walls = &mut self.walls;
        let radius = self.settings.radius();
        let cell = |at: Vec3| -> CellKey {
            let cell = at / radius;
            (
                cell.x.floor() as i32,
                cell.y.floor() as i32,
                cell.z.floor() as i32,
            )
        };
        let bits = |at: Vec3| (at.x.to_bits(), at.y.to_bits(), at.z.to_bits());
        // owner and position break ties, so the order never depends on how walls were added
        let mut order: Vec<WallKey> = walls
            .positions
            .iter()
            .enumerate()
            .map(|(index, at)| (cell(*at), walls.owners[index], bits(*at), index))
            .collect();
        order.sort_unstable();
        walls.positions = order.iter().map(|item| walls.positions[item.3]).collect();
        walls.owners = order.iter().map(|item| item.1).collect();
        walls.grid = Grid::build(order.iter().map(|item| item.0));
    }
}

impl Walls {
    pub(super) fn save(&self, out: &mut Writer) {
        out.len(self.positions.len());
        for (at, owner) in self.positions.iter().zip(&self.owners) {
            out.vec3(*at);
            out.u32(*owner);
        }
        out.u32(self.next);
        out.len(self.containers.len());
        for (id, low, high) in &self.containers {
            out.u32(*id);
            out.vec3(*low);
            out.vec3(*high);
        }
    }

    pub(super) fn next(&self) -> u32 {
        self.next
    }

    /// Without its grid, the world settles the walls once it has its settings.
    pub(super) fn load(reader: &mut Reader<'_>, max: usize) -> Result<Self, SnapshotError> {
        let count = reader.count(16, max)?;
        let mut walls = Self::default();
        for _ in 0..count {
            let at = reader.vec3()?;
            if !at.is_finite() {
                return Err(SnapshotError::Invalid("fluid wall"));
            }
            walls.positions.push(at);
            walls.owners.push(reader.u32()?);
        }
        walls.next = reader.u32()?;
        let containers = reader.count(28, max)?;
        for _ in 0..containers {
            let (id, low, high) = (reader.u32()?, reader.vec3()?, reader.vec3()?);
            if check_box(low, high).is_err() || id >= walls.next {
                return Err(SnapshotError::Invalid("fluid container"));
            }
            walls.containers.push((id, low, high));
        }
        if walls.owners.iter().any(|owner| *owner >= walls.next) {
            return Err(SnapshotError::Invalid("fluid wall owner"));
        }
        walls.bounds = union(&walls.containers);
        Ok(walls)
    }
}

fn check_box(low: Vec3, high: Vec3) -> Result<(), PhysicsError> {
    if !low.is_finite() || !high.is_finite() {
        return Err(PhysicsError::NonFinite("fluid box"));
    }
    if high.x <= low.x || high.y <= low.y || high.z <= low.z {
        return Err(PhysicsError::OutOfRange);
    }
    Ok(())
}

fn union(containers: &[(u32, Vec3, Vec3)]) -> Option<(Vec3, Vec3)> {
    containers
        .iter()
        .fold(None, |sum, &(_, low, high)| match sum {
            None => Some((low, high)),
            Some((a, b)) => Some((a.min(low), b.max(high))),
        })
}
