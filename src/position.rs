use algebrix::{DVec3, IVec3, Vec3};

pub const CELL_SIZE: f32 = 1024.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPosition {
    pub cell: IVec3,
    pub local: Vec3,
}

impl GridPosition {
    pub const ORIGIN: Self = Self {
        cell: IVec3::new(0, 0, 0),
        local: Vec3::ZERO,
    };

    pub fn from_local(local: Vec3) -> Option<Self> {
        Self {
            cell: IVec3::new(0, 0, 0),
            local,
        }
        .rebased()
    }

    pub fn from_dvec3(position: DVec3) -> Option<Self> {
        let size = f64::from(CELL_SIZE);
        let x = cell_index(position.x / size)?;
        let y = cell_index(position.y / size)?;
        let z = cell_index(position.z / size)?;
        let local = Vec3::new(
            (position.x - f64::from(x) * size) as f32,
            (position.y - f64::from(y) * size) as f32,
            (position.z - f64::from(z) * size) as f32,
        );
        Some(Self {
            cell: IVec3::new(x, y, z),
            local,
        })
    }

    pub fn to_dvec3(self) -> DVec3 {
        let size = f64::from(CELL_SIZE);
        DVec3::new(
            f64::from(self.cell.x) * size + f64::from(self.local.x),
            f64::from(self.cell.y) * size + f64::from(self.local.y),
            f64::from(self.cell.z) * size + f64::from(self.local.z),
        )
    }

    /// Difference of cells first, so it stays exact far from the origin.
    pub fn offset_from(self, from: Self) -> DVec3 {
        let size = f64::from(CELL_SIZE);
        let axis = |cell_to: i32, cell_from: i32, local_to: f32, local_from: f32| {
            (f64::from(cell_to) - f64::from(cell_from)) * size + f64::from(local_to)
                - f64::from(local_from)
        };
        DVec3::new(
            axis(self.cell.x, from.cell.x, self.local.x, from.local.x),
            axis(self.cell.y, from.cell.y, self.local.y, from.local.y),
            axis(self.cell.z, from.cell.z, self.local.z, from.local.z),
        )
    }

    /// `None` if the cell overflows.
    pub fn shifted(self, offset: DVec3) -> Option<Self> {
        let moved = Self::from_dvec3(DVec3::from_vec3(self.local) + offset)?;
        Some(Self {
            cell: IVec3::new(
                self.cell.x.checked_add(moved.cell.x)?,
                self.cell.y.checked_add(moved.cell.y)?,
                self.cell.z.checked_add(moved.cell.z)?,
            ),
            local: moved.local,
        })
    }

    /// Moves whole cells out of `local`. `None` if not finite or the cell overflows.
    pub fn rebased(self) -> Option<Self> {
        let (x, local_x) = rebase_axis(self.cell.x, self.local.x)?;
        let (y, local_y) = rebase_axis(self.cell.y, self.local.y)?;
        let (z, local_z) = rebase_axis(self.cell.z, self.local.z)?;
        Some(Self {
            cell: IVec3::new(x, y, z),
            local: Vec3::new(local_x, local_y, local_z),
        })
    }
}

fn cell_index(cells: f64) -> Option<i32> {
    if !cells.is_finite() {
        return None;
    }
    let rounded = cells.round();
    (rounded >= f64::from(i32::MIN) && rounded <= f64::from(i32::MAX)).then_some(rounded as i32)
}

fn rebase_axis(cell: i32, local: f32) -> Option<(i32, f32)> {
    if !local.is_finite() {
        return None;
    }
    let shift = (local / CELL_SIZE).round();
    if shift == 0.0 {
        return Some((cell, local));
    }
    let moved = cell_index(f64::from(cell) + f64::from(shift))?;
    Some((moved, local - shift * CELL_SIZE))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub(crate) fn at(x: f32, y: f32, z: f32) -> GridPosition {
        GridPosition::from_local(Vec3::new(x, y, z)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebase_keeps_world_position() {
        let position = GridPosition {
            cell: IVec3::new(3, -2, 0),
            local: Vec3::new(1500.0, -700.0, 12.5),
        };
        let rebased = position.rebased().unwrap();
        assert_eq!(rebased.cell, IVec3::new(4, -3, 0));
        assert_eq!(rebased.local, Vec3::new(476.0, 324.0, 12.5));
        assert_eq!(rebased.to_dvec3(), position.to_dvec3());
    }

    #[test]
    fn local_stays_inside_half_a_cell() {
        let half = CELL_SIZE / 2.0;
        let position = GridPosition::from_local(Vec3::new(50_000.0, -50_000.0, 511.0)).unwrap();
        assert!(position.local.x.abs() <= half);
        assert!(position.local.y.abs() <= half);
        assert_eq!(position.local.z, 511.0);
    }

    #[test]
    fn far_positions_round_trip_exact() {
        let far = DVec3::new(1_000_000_000.25, -6_371_000.5, 42.125);
        let position = GridPosition::from_dvec3(far).unwrap();
        assert_eq!(position.to_dvec3(), far);
    }

    #[test]
    fn bad_values_fail() {
        assert!(GridPosition::from_local(Vec3::new(f32::NAN, 0.0, 0.0)).is_none());
        assert!(GridPosition::from_local(Vec3::new(f32::INFINITY, 0.0, 0.0)).is_none());
        assert!(GridPosition::from_dvec3(DVec3::new(f64::NAN, 0.0, 0.0)).is_none());
        assert!(GridPosition::from_dvec3(DVec3::new(1e300, 0.0, 0.0)).is_none());
        let edge = GridPosition {
            cell: IVec3::new(i32::MAX, 0, 0),
            local: Vec3::new(CELL_SIZE, 0.0, 0.0),
        };
        assert!(edge.rebased().is_none());
    }
}
