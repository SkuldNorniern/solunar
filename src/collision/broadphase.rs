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

    fn overlaps(&self, other: &Self) -> bool {
        self.min.x <= other.max.x && other.min.x <= self.max.x && self.overlaps_yz(other)
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
        Shape::Compound(_) | Shape::TriMesh(_) | Shape::Convex(_) => Vec3::splat(f32::MAX),
    }
}

/// Passive bodies wider than this on x are checked against every mover, not swept.
const WIDE: f64 = 16.0;

/// Sort and sweep on x. Fills `pairs` with overlapping index pairs, low index first, sorted.
pub(crate) fn find_pairs(bounds: &[Bounds], order: &mut Vec<u32>, pairs: &mut Vec<(u32, u32)>) {
    order.clear();
    order.extend(0..bounds.len() as u32);
    pairs.clear();
    sweep(bounds, order, pairs);
    pairs.sort_unstable();
}

fn by_min_x(bounds: &[Bounds], a: u32, b: u32) -> std::cmp::Ordering {
    bounds[a as usize]
        .min
        .x
        .total_cmp(&bounds[b as usize].min.x)
        .then(a.cmp(&b))
}

fn sort_by_min_x(bounds: &[Bounds], order: &mut [u32]) {
    order.sort_unstable_by(|a, b| by_min_x(bounds, *a, *b));
}

/// Pairs among `order`, which it sorts. Adds to `pairs` unsorted.
fn sweep(bounds: &[Bounds], order: &mut [u32], pairs: &mut Vec<(u32, u32)>) {
    sort_by_min_x(bounds, order);
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
}

/// Pairs that have at least one mover. Static and sleeping bodies never pair with each other,
/// they stay still, so a field of sleepers costs a sort and not a sweep.
#[derive(Debug, Clone, Default)]
pub(crate) struct MovingPairs {
    movers: Vec<u32>,
    /// Sorted by min x. Kept between steps, still things stay sorted.
    narrow: Vec<u32>,
    is_narrow: Vec<bool>,
    wide: Vec<u32>,
    reach: f64,
}

impl MovingPairs {
    /// Fills `pairs`, sorted, low index first.
    pub(crate) fn find(
        &mut self,
        bounds: &[Bounds],
        moving: impl Fn(usize) -> bool,
        pairs: &mut Vec<(u32, u32)>,
    ) {
        self.movers.clear();
        self.wide.clear();
        self.reach = 0.0;
        self.is_narrow.clear();
        self.is_narrow.resize(bounds.len(), false);
        let mut narrow = 0;
        for (index, bound) in bounds.iter().enumerate() {
            let width = bound.max.x - bound.min.x;
            if moving(index) {
                self.movers.push(index as u32);
            } else if width.is_finite() && width <= WIDE {
                self.is_narrow[index] = true;
                narrow += 1;
                self.reach = self.reach.max(width);
            } else {
                self.wide.push(index as u32);
            }
        }
        let same = self.narrow.len() == narrow
            && self
                .narrow
                .iter()
                .all(|index| self.is_narrow.get(*index as usize) == Some(&true));
        if !same {
            self.narrow.clear();
            self.narrow
                .extend((0..bounds.len() as u32).filter(|index| self.is_narrow[*index as usize]));
        }
        if !self
            .narrow
            .is_sorted_by(|a, b| by_min_x(bounds, *a, *b).is_le())
        {
            sort_by_min_x(bounds, &mut self.narrow);
        }
        pairs.clear();
        sweep(bounds, &mut self.movers, pairs);
        for at in 0..self.movers.len() {
            self.against_passive(bounds, self.movers[at] as usize, pairs);
        }
        pairs.sort_unstable();
    }

    /// Adds pairs of bodies that stopped being passive this step, with everything passive.
    pub(crate) fn add_woken(
        &self,
        bounds: &[Bounds],
        woken: impl Iterator<Item = usize>,
        pairs: &mut Vec<(u32, u32)>,
    ) {
        let before = pairs.len();
        for index in woken {
            self.against_passive(bounds, index, pairs);
        }
        if pairs.len() > before {
            pairs.sort_unstable();
            pairs.dedup();
        }
    }

    fn against_passive(&self, bounds: &[Bounds], index: usize, pairs: &mut Vec<(u32, u32)>) {
        let bound = &bounds[index];
        let pair = |other: u32| ((index as u32).min(other), (index as u32).max(other));
        let start = self
            .narrow
            .partition_point(|other| bounds[*other as usize].min.x < bound.min.x - self.reach);
        for &other in &self.narrow[start..] {
            let second = &bounds[other as usize];
            if second.min.x > bound.max.x {
                break;
            }
            if other as usize != index && bound.overlaps(second) {
                pairs.push(pair(other));
            }
        }
        for &other in &self.wide {
            if other as usize != index && bound.overlaps(&bounds[other as usize]) {
                pairs.push(pair(other));
            }
        }
    }
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
    fn moving_pairs_are_the_brute_force_pairs_with_a_mover() {
        let mut state = 0x9abc_def1_2345_6789_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 10_000) as f64 / 100.0
        };
        for count in [0, 1, 2, 50, 400] {
            let mut bounds: Vec<Bounds> = (0..count)
                .map(|_| at(next(), next(), next(), next() / 20.0))
                .collect();
            if count > 2 {
                bounds[1] = at(50.0, 50.0, 50.0, 40.0);
                bounds[2].min.x = f64::NEG_INFINITY;
                bounds[2].max.x = f64::INFINITY;
            }
            let moving = |index: usize| index % 7 == 3;
            let woken = |index: usize| index % 11 == 5;
            let mut search = MovingPairs::default();
            let mut pairs = Vec::new();
            search.find(&bounds, moving, &mut pairs);
            let mut wanted: Vec<(u32, u32)> = brute_force(&bounds)
                .into_iter()
                .filter(|(a, b)| moving(*a as usize) || moving(*b as usize))
                .collect();
            assert_eq!(pairs, wanted);
            search.add_woken(
                &bounds,
                (0..count).filter(|i| woken(*i) && !moving(*i)),
                &mut pairs,
            );
            wanted = brute_force(&bounds)
                .into_iter()
                .filter(|(a, b)| {
                    let live = |i: u32| moving(i as usize) || woken(i as usize);
                    live(*a) || live(*b)
                })
                .collect();
            assert_eq!(pairs, wanted);
        }
    }

    #[test]
    fn kept_passive_order_matches_a_fresh_search() {
        let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 10_000) as f64 / 100.0
        };
        let mut bounds: Vec<Bounds> = (0..300)
            .map(|_| at(next(), next(), next(), next() / 20.0))
            .collect();
        let mut kept = MovingPairs::default();
        let (mut pairs, mut fresh_pairs) = (Vec::new(), Vec::new());
        for step in 0..40_usize {
            let moving = |index: usize| (index + step / 5).is_multiple_of(9);
            // a passive body moves now and then, a body goes now and then
            if step % 7 == 3 {
                let index = (next() * 2.0) as usize % bounds.len();
                bounds[index] = at(next(), next(), next(), 1.0);
            }
            if step % 11 == 6 {
                bounds.swap_remove((next() * 2.0) as usize % bounds.len());
            }
            kept.find(&bounds, moving, &mut pairs);
            MovingPairs::default().find(&bounds, moving, &mut fresh_pairs);
            assert_eq!(pairs, fresh_pairs, "step {step}");
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
