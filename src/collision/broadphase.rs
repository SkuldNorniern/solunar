use algebrix::{DVec3, Quat, Vec3};

use crate::{GridPosition, Shape};

/// World bounds in f64, so far cells do not lose precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) min: DVec3,
    pub(crate) max: DVec3,
}

impl Bounds {
    pub(crate) fn of(shape: &Shape, position: GridPosition, rotation: Quat, margin: f32) -> Self {
        let extent = DVec3::from_vec3(shape_extent(shape, rotation) + Vec3::splat(margin * 0.5));
        let center = position.to_dvec3();
        Self {
            min: center - extent,
            max: center + extent,
        }
    }

    fn overlaps_yz(&self, other: &Self) -> bool {
        self.min.y <= other.max.y
            && other.min.y <= self.max.y
            && self.min.z <= other.max.z
            && other.min.z <= self.max.z
    }
}

fn shape_extent(shape: &Shape, rotation: Quat) -> Vec3 {
    match *shape {
        Shape::Sphere { radius } => Vec3::splat(radius),
        Shape::Capsule {
            radius,
            half_height,
        } => rotation.rotate_vec3(Vec3::Y * half_height).abs() + Vec3::splat(radius),
        Shape::Cuboid { half_extents } => {
            rotation.rotate_vec3(Vec3::X * half_extents.x).abs()
                + rotation.rotate_vec3(Vec3::Y * half_extents.y).abs()
                + rotation.rotate_vec3(Vec3::Z * half_extents.z).abs()
        }
        Shape::Compound(_) | Shape::TriMesh(_) => Vec3::splat(f32::MAX),
    }
}

/// Sort and sweep on x. Fills `pairs` with overlapping index pairs, low index first, sorted.
pub(crate) fn find_pairs(bounds: &[Bounds], order: &mut Vec<u32>, pairs: &mut Vec<(u32, u32)>) {
    order.clear();
    order.extend(0..bounds.len() as u32);
    order.sort_unstable_by(|a, b| {
        bounds[*a as usize]
            .min
            .x
            .total_cmp(&bounds[*b as usize].min.x)
            .then(a.cmp(b))
    });
    pairs.clear();
    for (at, a) in order.iter().enumerate() {
        let first = &bounds[*a as usize];
        for b in &order[at + 1..] {
            let second = &bounds[*b as usize];
            if second.min.x > first.max.x {
                break;
            }
            if first.overlaps_yz(second) {
                pairs.push(((*a).min(*b), (*a).max(*b)));
            }
        }
    }
    pairs.sort_unstable();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f64, y: f64, z: f64, half: f64) -> Bounds {
        let center = DVec3::new(x, y, z);
        Bounds {
            min: center - DVec3::splat(half),
            max: center + DVec3::splat(half),
        }
    }

    fn brute_force(bounds: &[Bounds]) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        for a in 0..bounds.len() {
            for b in a + 1..bounds.len() {
                let (l, r) = (&bounds[a], &bounds[b]);
                if l.min.x <= r.max.x && r.min.x <= l.max.x && l.overlaps_yz(r) {
                    pairs.push((a as u32, b as u32));
                }
            }
        }
        pairs
    }

    #[test]
    fn matches_brute_force_on_random_boxes() {
        let mut state = 0x1234_5678_9abc_def1_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 10_000) as f64 / 100.0
        };
        for count in [0, 1, 2, 50, 400] {
            let bounds: Vec<Bounds> = (0..count)
                .map(|_| at(next(), next(), next(), next() / 20.0))
                .collect();
            let (mut order, mut pairs) = (Vec::new(), Vec::new());
            find_pairs(&bounds, &mut order, &mut pairs);
            assert_eq!(pairs, brute_force(&bounds));
        }
    }

    #[test]
    fn touching_counts_and_gaps_do_not() {
        let bounds = [
            at(0.0, 0.0, 0.0, 1.0),
            at(2.0, 0.0, 0.0, 1.0),
            at(9.0, 0.0, 0.0, 1.0),
        ];
        let (mut order, mut pairs) = (Vec::new(), Vec::new());
        find_pairs(&bounds, &mut order, &mut pairs);
        assert_eq!(pairs, [(0, 1)]);
    }

    #[test]
    fn rotated_cuboid_bounds_grow() {
        let shape = Shape::cuboid(Vec3::new(1.0, 1.0, 1.0));
        let turned = Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_4);
        let bounds = Bounds::of(&shape, GridPosition::ORIGIN, turned, 0.0);
        assert!((bounds.max.x - 2.0_f64.sqrt()).abs() < 1e-5);
        assert!((bounds.max.z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn capsule_bounds_follow_its_axis() {
        let shape = Shape::capsule(0.5, 2.0);
        let upright = Bounds::of(&shape, GridPosition::ORIGIN, Quat::IDENTITY, 0.0);
        assert!((upright.max.y - 2.5).abs() < 1e-5);
        assert!((upright.max.x - 0.5).abs() < 1e-5);
    }
}
