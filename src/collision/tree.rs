use std::sync::OnceLock;

use algebrix::DVec3;

use crate::body::Bodies;
use crate::position::GridPosition;

const LEAF: usize = 4;

type Bounds = ([f64; 3], [f64; 3]);

fn axes(v: DVec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

#[derive(Debug, Clone, Copy)]
struct Node {
    min: [f64; 3],
    max: [f64; 3],
    /// A leaf holds `items[first..first + count]`, an inner node has `count` 0 and its children
    /// at `first` and `first + 1`.
    first: u32,
    count: u32,
}

/// Bounding sphere boxes of all bodies, so queries visit only what they can reach.
#[derive(Debug, Clone)]
pub(crate) struct QueryTree {
    /// Boxes are offsets from here, so far cells keep their precision.
    origin: GridPosition,
    nodes: Vec<Node>,
    items: Vec<u32>,
    /// Each item's box, in the order of `items`, so a leaf tests its bodies before visiting.
    boxes: Vec<Bounds>,
}

/// Pending nodes of a walk, in place on the call stack. A median split tree of `u32` items is
/// at most 32 deep, a walk holds at most two nodes a level, so the heap is never reached in
/// practice. It still spills there rather than lose a node.
pub(crate) struct Pending<T: Copy + Default> {
    held: [T; 64],
    len: usize,
    spilled: Vec<T>,
}

impl<T: Copy + Default> Pending<T> {
    pub(crate) fn new(first: T) -> Self {
        let mut held = [T::default(); 64];
        held[0] = first;
        Self {
            held,
            len: 1,
            spilled: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, item: T) {
        if self.len < self.held.len() {
            self.held[self.len] = item;
            self.len += 1;
        } else {
            self.spilled.push(item);
        }
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        if let Some(item) = self.spilled.pop() {
            return Some(item);
        }
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.held[self.len])
    }
}

/// Derived from the bodies, never part of what two worlds compare as.
#[derive(Debug, Default)]
pub(crate) struct TreeCache(OnceLock<QueryTree>);

impl Clone for TreeCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for TreeCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl TreeCache {
    pub(crate) fn get(&self, bodies: &Bodies) -> &QueryTree {
        self.0.get_or_init(|| QueryTree::build(bodies, LEAF))
    }

    /// One leaf over every body, the same as scanning them all.
    #[cfg(test)]
    pub(crate) fn flat(bodies: &Bodies) -> Self {
        Self(OnceLock::from(QueryTree::build(bodies, usize::MAX)))
    }

    /// Bodies moved, came or went.
    pub(crate) fn clear(&mut self) {
        self.0 = OnceLock::new();
    }
}

impl QueryTree {
    fn build(bodies: &Bodies, leaf: usize) -> Self {
        let count = bodies.len();
        let origin = if count == 0 {
            GridPosition::ORIGIN
        } else {
            bodies.position(0)
        };
        // each body's own turned box, far tighter than its bounding sphere for a floor or a wall
        let base = origin.to_dvec3();
        let boxes: Vec<Bounds> = (0..count)
            .map(|dense| {
                let bound = bodies.assets.bounds(
                    &bodies.shape[dense],
                    bodies.position(dense),
                    bodies.rotation[dense],
                    0.0,
                );
                (axes(bound.min - base), axes(bound.max - base))
            })
            .collect();
        let mut tree = Self {
            origin,
            nodes: Vec::with_capacity(count / 2 + 1),
            items: (0..count as u32).collect(),
            boxes: Vec::new(),
        };
        if count > 0 {
            tree.nodes.push(Node {
                min: [0.0; 3],
                max: [0.0; 3],
                first: 0,
                count: 0,
            });
            tree.split(0, 0, count, leaf, &boxes);
        }
        tree.boxes = tree
            .items
            .iter()
            .map(|item| boxes[*item as usize])
            .collect();
        tree
    }

    /// Fills node `at` over `items[start..end]`, split at the median of the longest axis.
    fn split(&mut self, at: usize, start: usize, end: usize, leaf: usize, boxes: &[Bounds]) {
        let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for item in &self.items[start..end] {
            let (low, high) = boxes[*item as usize];
            for axis in 0..3 {
                min[axis] = min[axis].min(low[axis]);
                max[axis] = max[axis].max(high[axis]);
            }
        }
        if end - start <= leaf {
            self.nodes[at] = Node {
                min,
                max,
                first: start as u32,
                count: (end - start) as u32,
            };
            return;
        }
        let size = [0, 1, 2].map(|axis| max[axis] - min[axis]);
        let axis = if size[0] >= size[1] && size[0] >= size[2] {
            0
        } else if size[1] >= size[2] {
            1
        } else {
            2
        };
        let middle = start + (end - start) / 2;
        let center = |item: &u32| {
            let (low, high) = boxes[*item as usize];
            low[axis] + high[axis]
        };
        // ties go to the lower index, the same bodies give the same tree
        self.items[start..end].select_nth_unstable_by(middle - start, |a, b| {
            center(a).total_cmp(&center(b)).then(a.cmp(b))
        });
        let left = self.nodes.len();
        self.nodes.push(self.nodes[at]);
        self.nodes.push(self.nodes[at]);
        self.nodes[at] = Node {
            min,
            max,
            first: left as u32,
            count: 0,
        };
        self.split(left, start, middle, leaf, boxes);
        self.split(left + 1, middle, end, leaf, boxes);
    }

    pub(crate) fn local(&self, at: GridPosition) -> DVec3 {
        at.offset_from(self.origin)
    }

    /// Every body whose box, grown by `grow`, the path from `start` along unit `direction`
    /// enters before `limit()`. `limit` is read again as it shrinks. `visit` returns true to end
    /// the walk.
    pub(crate) fn along(
        &self,
        start: DVec3,
        direction: DVec3,
        grow: f64,
        limit: impl Fn() -> f64,
        mut visit: impl FnMut(usize) -> bool,
    ) {
        if self.nodes.is_empty() {
            return;
        }
        let (from, toward) = (axes(start), axes(direction));
        // a component this small runs along its slab, checked by position instead
        let flat = toward.map(|component| component.abs() < 1e-300);
        let inverse = toward.map(f64::recip);
        let enter = |min: &[f64; 3], max: &[f64; 3]| {
            let (mut near, mut far) = (0.0_f64, f64::INFINITY);
            for axis in 0..3 {
                let (low, high) = (min[axis] - grow, max[axis] + grow);
                if flat[axis] {
                    if from[axis] < low || from[axis] > high {
                        return None;
                    }
                    continue;
                }
                let a = (low - from[axis]) * inverse[axis];
                let b = (high - from[axis]) * inverse[axis];
                near = near.max(a.min(b));
                far = far.min(a.max(b));
            }
            (near <= far).then_some(near)
        };
        let node_enter = |node: &Node| enter(&node.min, &node.max);
        let Some(root) = node_enter(&self.nodes[0]) else {
            return;
        };
        let mut stack = Pending::new((0_usize, root));
        while let Some((at, near)) = stack.pop() {
            // some room past the limit, the exact tests decide
            if near > limit() + 1e-6 {
                continue;
            }
            let node = self.nodes[at];
            if node.count > 0 {
                let range = node.first as usize..(node.first + node.count) as usize;
                for (item, (min, max)) in self.items[range.clone()].iter().zip(&self.boxes[range]) {
                    let reached = enter(min, max).is_some_and(|near| near <= limit() + 1e-6);
                    if reached && visit(*item as usize) {
                        return;
                    }
                }
                continue;
            }
            let (a, b) = (node.first as usize, node.first as usize + 1);
            // the nearer child goes on top
            match (node_enter(&self.nodes[a]), node_enter(&self.nodes[b])) {
                (Some(ta), Some(tb)) if ta <= tb => {
                    stack.push((b, tb));
                    stack.push((a, ta));
                }
                (Some(ta), Some(tb)) => {
                    stack.push((a, ta));
                    stack.push((b, tb));
                }
                (Some(ta), None) => stack.push((a, ta)),
                (None, Some(tb)) => stack.push((b, tb)),
                (None, None) => {}
            }
        }
    }

    /// Every body whose box comes within `radius` of `center` on each axis.
    pub(crate) fn within(&self, center: DVec3, radius: f64, mut visit: impl FnMut(usize)) {
        if self.nodes.is_empty() {
            return;
        }
        let center = axes(center);
        let mut stack = Pending::new(0_usize);
        while let Some(at) = stack.pop() {
            let node = self.nodes[at];
            let apart = (0..3).any(|axis| {
                node.min[axis] > center[axis] + radius || node.max[axis] < center[axis] - radius
            });
            if apart {
                continue;
            }
            if node.count > 0 {
                let range = node.first as usize..(node.first + node.count) as usize;
                for (item, (min, max)) in self.items[range.clone()].iter().zip(&self.boxes[range]) {
                    let apart = (0..3).any(|axis| {
                        min[axis] > center[axis] + radius || max[axis] < center[axis] - radius
                    });
                    if !apart {
                        visit(*item as usize);
                    }
                }
            } else {
                stack.push(node.first as usize + 1);
                stack.push(node.first as usize);
            }
        }
    }
}
