use crate::BodyKind;
use crate::body::Bodies;
use crate::joint::{JointPoint, JointRow, Joints};

use super::contacts::{ContactScratch, PointFrame, SolverManifold, SolverPoint};
use super::{COLORED_ISLAND, COLORS};

/// Awake dynamic bodies joined by contacts and joints. Static, kinematic and sleeping bodies
/// never join two islands. Each island's constraints sit together, in the order they had.
#[derive(Debug, Clone, Default)]
pub(crate) struct Islands {
    parent: Vec<u32>,
    /// Island of each dense body, `NONE` when it is not awake and dynamic.
    pub(super) of: Vec<u32>,
    pub(super) bodies: Vec<u32>,
    pub(super) manifold_ends: Vec<u32>,
    pub(super) point_ends: Vec<u32>,
    pub(super) row_ends: Vec<u32>,
    spare_manifolds: Vec<SolverManifold>,
    spare_contacts: Vec<SolverPoint>,
    spare_frames: Vec<PointFrame>,
    pub(super) spare_points: Vec<JointPoint>,
    pub(super) spare_rows: Vec<JointRow>,
}

pub(super) const NONE: u32 = u32::MAX;

impl Islands {
    fn find(&mut self, mut dense: usize) -> usize {
        while self.parent[dense] as usize != dense {
            let up = self.parent[dense] as usize;
            self.parent[dense] = self.parent[up];
            dense = up;
        }
        dense
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        // the lower index stays the root, so the result does not depend on link order
        if a != b {
            let (low, high) = (a.min(b), a.max(b));
            self.parent[high] = low as u32;
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.bodies.len()
    }

    /// Bodies and constraints in each island, in island order. Constraints count joint rows
    /// from the last substep.
    pub(crate) fn sizes(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.count()).map(|island| {
            let span = |ends: &[u32]| {
                let start = if island == 0 { 0 } else { ends[island - 1] };
                (ends[island] - start) as usize
            };
            let constraints =
                span(&self.manifold_ends) + span(&self.point_ends) + span(&self.row_ends);
            (self.bodies[island] as usize, constraints)
        })
    }

    pub(super) fn range(ends: &[u32], island: usize) -> std::ops::Range<usize> {
        let start = if island == 0 { 0 } else { ends[island - 1] };
        start as usize..ends[island] as usize
    }
}

/// Puts items of island 0 first, then 1 and on, keeping their order inside an island.
pub(super) fn group<T: Copy>(
    items: &mut [T],
    spare: &mut Vec<T>,
    count: usize,
    island: impl Fn(&T) -> u32,
    ends: &mut Vec<u32>,
) {
    ends.clear();
    ends.resize(count, 0);
    for item in items.iter() {
        ends[island(item) as usize] += 1;
    }
    let mut next = 0;
    for end in ends.iter_mut() {
        let size = *end;
        *end = next;
        next += size;
    }
    spare.clear();
    spare.extend_from_slice(items);
    for item in spare.iter() {
        let slot = &mut ends[island(item) as usize];
        items[*slot as usize] = *item;
        *slot += 1;
    }
}

pub(super) fn find_islands(
    bodies: &Bodies,
    awake: &[bool],
    joints: &Joints,
    scratch: &mut ContactScratch,
) {
    let active = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic && awake[dense];
    let islands = &mut scratch.islands;
    let count = bodies.len();
    islands.parent.clear();
    islands.parent.extend(0..count as u32);
    for manifold in &scratch.manifolds {
        if active(manifold.a) && active(manifold.b) {
            islands.union(manifold.a, manifold.b);
        }
    }
    for (_, state) in joints.live() {
        let (Ok(a), Ok(b)) = (
            bodies.dense_index(state.desc.a),
            bodies.dense_index(state.desc.b),
        ) else {
            continue;
        };
        if active(a) && active(b) {
            islands.union(a, b);
        }
    }
    islands.of.clear();
    islands.of.resize(count, NONE);
    islands.bodies.clear();
    for dense in 0..count {
        if !active(dense) {
            continue;
        }
        let root = islands.find(dense);
        if islands.of[root] == NONE {
            islands.of[root] = islands.bodies.len() as u32;
            islands.bodies.push(0);
        }
        let island = islands.of[root];
        islands.of[dense] = island;
        islands.bodies[island as usize] += 1;
    }
    let islands = &mut scratch.islands;
    let of = &islands.of;
    group(
        &mut scratch.manifolds,
        &mut islands.spare_manifolds,
        islands.bodies.len(),
        |manifold| {
            if of[manifold.a] != NONE {
                of[manifold.a]
            } else {
                of[manifold.b]
            }
        },
        &mut islands.manifold_ends,
    );
    order_big_islands(bodies, scratch);
}

/// Islands that will surely be colored get their contacts in color order, and the points
/// rewritten to match, so a pass reads them front to back. Joints are left out of these
/// colors, the substep coloring adds them. Smaller islands keep their order and results.
fn order_big_islands(bodies: &Bodies, scratch: &mut ContactScratch) {
    let dynamic = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic;
    let islands = &mut scratch.islands;
    let used = &mut scratch.colors.used;
    used.clear();
    used.resize(bodies.len(), 0);
    let mut moved = false;
    for island in 0..islands.count() {
        let range = Islands::range(&islands.manifold_ends, island);
        if range.len() < COLORED_ISLAND {
            continue;
        }
        moved = true;
        let manifolds = &mut scratch.manifolds[range];
        let mut keys: Vec<(u32, u32)> = Vec::with_capacity(manifolds.len());
        for (index, manifold) in manifolds.iter().enumerate() {
            let mask = |dense: usize| if dynamic(dense) { used[dense] } else { 0 };
            let color = (!(mask(manifold.a) | mask(manifold.b)))
                .trailing_zeros()
                .min(COLORS as u32);
            if (color as usize) < COLORS {
                for dense in [manifold.a, manifold.b] {
                    if dynamic(dense) {
                        used[dense] |= 1 << color;
                    }
                }
            }
            keys.push((color, index as u32));
        }
        keys.sort_unstable();
        islands.spare_manifolds.clear();
        islands.spare_manifolds.extend_from_slice(manifolds);
        for (slot, (_, from)) in manifolds.iter_mut().zip(&keys) {
            *slot = islands.spare_manifolds[*from as usize];
        }
    }
    if !moved {
        return;
    }
    let (points, frames) = (
        &mut scratch.islands.spare_contacts,
        &mut scratch.islands.spare_frames,
    );
    points.clear();
    frames.clear();
    for manifold in &mut scratch.manifolds {
        let range = manifold.first..manifold.first + manifold.count;
        manifold.first = points.len();
        points.extend_from_slice(&scratch.points[range.clone()]);
        frames.extend_from_slice(&scratch.frames[range]);
    }
    std::mem::swap(&mut scratch.points, points);
    std::mem::swap(&mut scratch.frames, frames);
}

#[cfg(test)]
mod tests {
    use super::{Islands, group};

    #[test]
    fn grouping_keeps_order_inside_each_island() {
        let mut items = vec![(1, 'a'), (0, 'b'), (2, 'c'), (0, 'd'), (1, 'e')];
        let (mut spare, mut ends) = (Vec::new(), Vec::new());
        group(&mut items, &mut spare, 3, |item| item.0, &mut ends);
        let order: String = items.iter().map(|item| item.1).collect();
        assert_eq!(order, "bdaec");
        assert_eq!(ends, [2, 4, 5]);
        assert_eq!(Islands::range(&ends, 1), 2..4);
    }
}
