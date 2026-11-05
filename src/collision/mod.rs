mod broadphase;
pub(crate) mod convex;
mod gjk;
mod narrowphase;
mod query;
pub(crate) mod trimesh;

pub(crate) use broadphase::{Bounds, MovingPairs, find_pairs};
pub(crate) use gjk::Hull;
pub(crate) use narrowphase::{MAX_CONTACTS, RawManifold, contacts};
pub(crate) use query::sweep;
pub use query::{CastHit, PointHit, QueryFilter, Ray, ShapeCast};

/// A pair collides when each body's `layers` hits the other's `mask`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollisionFilter {
    pub layers: u32,
    pub mask: u32,
}

impl Default for CollisionFilter {
    fn default() -> Self {
        Self {
            layers: 1,
            mask: u32::MAX,
        }
    }
}

impl CollisionFilter {
    pub fn allows(self, other: Self) -> bool {
        self.layers & other.mask != 0 && other.layers & self.mask != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_sides_must_agree() {
        let a = CollisionFilter {
            layers: 0b01,
            mask: 0b10,
        };
        let b = CollisionFilter {
            layers: 0b10,
            mask: 0b01,
        };
        let one_way = CollisionFilter {
            layers: 0b10,
            mask: 0b10,
        };
        assert!(a.allows(b));
        assert!(b.allows(a));
        assert!(!a.allows(one_way));
        assert!(!a.allows(a));
    }
}
