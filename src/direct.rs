//! Exact solve of an island's equality joint rows: a sparse block LDLᵀ of J M⁻¹ Jᵀ in f64, in
//! minimum degree order. Gauss Seidel can not carry a load ratio of hundreds to one across a
//! joint, this can.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

use algebrix::Vec3;

/// One row of an equality constraint. Speed along it is `lin_a . va + ang_a . wa + lin_b . vb +
/// ang_b . wb`, an impulse `x` along it moves A by `x * move_*_a` and B by `x * move_*_b`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DirectRow {
    pub(crate) lin_a: Vec3,
    pub(crate) ang_a: Vec3,
    pub(crate) lin_b: Vec3,
    pub(crate) ang_b: Vec3,
    pub(crate) move_lin_a: Vec3,
    pub(crate) move_ang_a: Vec3,
    pub(crate) move_lin_b: Vec3,
    pub(crate) move_ang_b: Vec3,
    pub(crate) target: f32,
    pub(crate) recovery: f32,
}

/// A joint point (three rows) or one locked row, between two bodies.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectNode {
    pub(crate) a: usize,
    pub(crate) b: usize,
    pub(crate) count: usize,
    pub(crate) rows: [DirectRow; 3],
}

impl DirectRow {
    fn coupling(&self, other: &Self, side: Side, other_side: Side) -> f64 {
        let (lin, ang) = match side {
            Side::A => (self.lin_a, self.ang_a),
            Side::B => (self.lin_b, self.ang_b),
        };
        let (move_lin, move_ang) = match other_side {
            Side::A => (other.move_lin_a, other.move_ang_a),
            Side::B => (other.move_lin_b, other.move_ang_b),
        };
        f64::from(lin.dot(move_lin)) + f64::from(ang.dot(move_ang))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    A,
    B,
}

type Block = [f64; 9];

/// Relative to the diagonal, keeps a redundant set of rows solvable, a very stiff spring.
const REGULARIZE: f64 = 1e-9;
/// A pivot this small against its row's own diagonal is a direction other rows already hold.
const PIVOT_FLOOR: f64 = 1e-6;

/// The shape of one island's factor: elimination order, fill and where every block lives. It
/// depends only on which bodies the nodes join, so it is kept while that stays.
#[derive(Debug, Clone, Default)]
pub(crate) struct Symbolic {
    /// Each node's two bodies and rows, `usize::MAX` for a body that never moves.
    key: Vec<(usize, usize, usize)>,
    /// Nodes in elimination order.
    order: Vec<u32>,
    blocks: usize,
    diagonal: Vec<u32>,
    /// Coupling of two nodes through a shared body: block, low node, high node, their sides.
    assemble: Vec<(u32, u32, u32, Side, Side)>,
    /// Per eliminated node, in order: a later node, the block between them, and whether it is
    /// stored the other way round (pivot rows first).
    columns: Vec<(u32, u32, bool)>,
    column_ends: Vec<u32>,
    /// Per eliminated node: two column entries `i <= j` and the block of their nodes.
    updates: Vec<(u16, u16, u32)>,
    update_ends: Vec<u32>,
}

impl Symbolic {
    /// Whether `nodes` join the same bodies the same way as when this was made.
    pub(crate) fn fits(&self, nodes: &[DirectNode], dynamic: &dyn Fn(usize) -> bool) -> bool {
        self.key.len() == nodes.len()
            && self
                .key
                .iter()
                .zip(nodes)
                .all(|(key, node)| *key == node_key(node, dynamic))
    }

    /// Orders `nodes` by minimum degree, ties to the lower index, and lays out every block the
    /// factor will have.
    pub(crate) fn new(nodes: &[DirectNode], dynamic: &dyn Fn(usize) -> bool) -> Self {
        let count = nodes.len();
        let key: Vec<_> = nodes.iter().map(|node| node_key(node, dynamic)).collect();

        let mut on_body: Vec<(usize, u32, Side)> = Vec::with_capacity(count * 2);
        for (index, (a, b, _)) in key.iter().enumerate() {
            if *a != usize::MAX {
                on_body.push((*a, index as u32, Side::A));
            }
            if *b != usize::MAX {
                on_body.push((*b, index as u32, Side::B));
            }
        }
        on_body.sort_unstable_by_key(|(body, index, _)| (*body, *index));

        let mut slots: HashMap<(u32, u32), u32> = HashMap::with_capacity(count * 4);
        let mut slot = |low: u32, high: u32| -> u32 {
            let next = slots.len() as u32;
            *slots.entry((low, high)).or_insert(next)
        };
        let diagonal: Vec<u32> = (0..count as u32).map(|node| slot(node, node)).collect();
        let mut adjacent: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); count];
        let mut assemble = Vec::new();
        let mut start = 0;
        while start < on_body.len() {
            let body = on_body[start].0;
            let end = on_body[start..]
                .iter()
                .position(|(other, _, _)| *other != body)
                .map_or(on_body.len(), |at| start + at);
            let group = &on_body[start..end];
            for (i, &(_, p, side_p)) in group.iter().enumerate() {
                for &(_, q, side_q) in &group[i..] {
                    let (low, high, side_low, side_high) = if p <= q {
                        (p, q, side_p, side_q)
                    } else {
                        (q, p, side_q, side_p)
                    };
                    assemble.push((slot(low, high), low, high, side_low, side_high));
                    if p != q {
                        adjacent[p as usize].insert(q);
                        adjacent[q as usize].insert(p);
                    }
                }
            }
            start = end;
        }

        let mut heap: BinaryHeap<Reverse<(usize, u32)>> = adjacent
            .iter()
            .enumerate()
            .map(|(index, set)| Reverse((set.len(), index as u32)))
            .collect();
        let mut done = vec![false; count];
        let mut order = Vec::with_capacity(count);
        let mut columns = Vec::new();
        let mut column_ends = Vec::with_capacity(count);
        let mut updates = Vec::new();
        let mut update_ends = Vec::with_capacity(count);
        while let Some(Reverse((degree, node))) = heap.pop() {
            let at = node as usize;
            if done[at] || degree != adjacent[at].len() {
                continue;
            }
            done[at] = true;
            order.push(node);
            let neighbours: Vec<u32> = std::mem::take(&mut adjacent[at]).into_iter().collect();
            for &q in &neighbours {
                let set = &mut adjacent[q as usize];
                set.remove(&node);
                for &r in &neighbours {
                    if r != q {
                        set.insert(r);
                    }
                }
                heap.push(Reverse((set.len(), q)));
            }
            for &q in &neighbours {
                let (low, high) = (node.min(q), node.max(q));
                columns.push((q, slot(low, high), node < q));
            }
            // neighbours are sorted, so i <= j means the lower node first
            for i in 0..neighbours.len() {
                for j in i..neighbours.len() {
                    let block = slot(neighbours[i], neighbours[j]);
                    updates.push((i as u16, j as u16, block));
                }
            }
            column_ends.push(columns.len() as u32);
            update_ends.push(updates.len() as u32);
        }
        Self {
            key,
            order,
            blocks: slots.len(),
            diagonal,
            assemble,
            columns,
            column_ends,
            updates,
            update_ends,
        }
    }

    fn range(ends: &[u32], at: usize) -> std::ops::Range<usize> {
        let start = if at == 0 { 0 } else { ends[at - 1] as usize };
        start..ends[at] as usize
    }
}

fn node_key(node: &DirectNode, dynamic: &dyn Fn(usize) -> bool) -> (usize, usize, usize) {
    let body = |dense: usize| if dynamic(dense) { dense } else { usize::MAX };
    (body(node.a), body(node.b), node.count)
}

/// The numbers of one island's factor. Built once a substep, solved once per pass. Every node
/// is 3 wide, a node with fewer rows has zeros there and its spare pivots drop.
#[derive(Debug, Clone, Default)]
pub(crate) struct Factor {
    blocks: Vec<Block>,
    inverse_diagonal: Vec<Block>,
    /// `L` beside `Symbolic::columns`: the later node's rows by the pivot's.
    lower: Vec<Block>,
    column_blocks: Vec<Block>,
    floors: Vec<f64>,
}

impl Factor {
    /// Factors `J M⁻¹ Jᵀ` of `nodes` in the shape `symbolic` gives. The buffers are reused.
    pub(crate) fn refactor(&mut self, symbolic: &Symbolic, nodes: &[DirectNode]) {
        self.blocks.clear();
        self.blocks.resize(symbolic.blocks, [0.0; 9]);
        for &(slot, low, high, side_low, side_high) in &symbolic.assemble {
            let (node_low, node_high) = (&nodes[low as usize], &nodes[high as usize]);
            let block = &mut self.blocks[slot as usize];
            for r in 0..node_low.count {
                for c in 0..node_high.count {
                    block[r * 3 + c] +=
                        node_low.rows[r].coupling(&node_high.rows[c], side_low, side_high);
                }
            }
        }
        self.floors.clear();
        for (node, slot) in nodes.iter().zip(&symbolic.diagonal) {
            let block = &mut self.blocks[*slot as usize];
            let scale = (0..node.count)
                .map(|r| block[r * 3 + r].abs())
                .fold(0.0, f64::max);
            self.floors.push(PIVOT_FLOOR * scale);
            for r in 0..node.count {
                block[r * 3 + r] += REGULARIZE * scale.max(1e-12);
            }
        }

        self.inverse_diagonal.clear();
        self.inverse_diagonal.resize(nodes.len(), [0.0; 9]);
        self.lower.clear();
        self.lower.resize(symbolic.columns.len(), [0.0; 9]);
        for (at, &p) in symbolic.order.iter().enumerate() {
            let diagonal = self.blocks[symbolic.diagonal[p as usize] as usize];
            let inverse = invert(&diagonal, self.floors[p as usize]);
            self.inverse_diagonal[p as usize] = inverse;
            let columns = Symbolic::range(&symbolic.column_ends, at);
            // the pivot's column blocks as later node by pivot, kept for the updates
            self.column_blocks.clear();
            for (lower, &(_, slot, stored_pivot_first)) in self.lower[columns.clone()]
                .iter_mut()
                .zip(&symbolic.columns[columns.clone()])
            {
                let stored = &self.blocks[slot as usize];
                let a_qp = if stored_pivot_first {
                    transpose(stored)
                } else {
                    *stored
                };
                *lower = multiply(&a_qp, &inverse);
                self.column_blocks.push(a_qp);
            }
            let lower = &self.lower[columns];
            for &(i, j, slot) in &symbolic.updates[Symbolic::range(&symbolic.update_ends, at)] {
                // A_qr -= L_qp A_pr, A_pr being A_rp turned
                subtract_times_turned(
                    &mut self.blocks[slot as usize],
                    &lower[i as usize],
                    &self.column_blocks[j as usize],
                );
            }
        }
    }

    /// Solves in place: `x` comes in as the right hand side, one part per node.
    pub(crate) fn solve(&self, symbolic: &Symbolic, x: &mut [[f64; 3]]) {
        for (at, &p) in symbolic.order.iter().enumerate() {
            let value = x[p as usize];
            let columns = Symbolic::range(&symbolic.column_ends, at);
            for (l_qp, &(q, _, _)) in self.lower[columns.clone()]
                .iter()
                .zip(&symbolic.columns[columns])
            {
                let change = times(l_qp, value);
                let part = &mut x[q as usize];
                for r in 0..3 {
                    part[r] -= change[r];
                }
            }
        }
        for (part, inverse) in x.iter_mut().zip(&self.inverse_diagonal) {
            *part = times(inverse, *part);
        }
        for (at, &p) in symbolic.order.iter().enumerate().rev() {
            let columns = Symbolic::range(&symbolic.column_ends, at);
            let mut part = x[p as usize];
            for (l_qp, &(q, _, _)) in self.lower[columns.clone()]
                .iter()
                .zip(&symbolic.columns[columns])
            {
                let value = x[q as usize];
                for c in 0..3 {
                    part[c] -= l_qp[c] * value[0] + l_qp[3 + c] * value[1] + l_qp[6 + c] * value[2];
                }
            }
            x[p as usize] = part;
        }
    }
}

fn times(block: &Block, value: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| {
        block[r * 3] * value[0] + block[r * 3 + 1] * value[1] + block[r * 3 + 2] * value[2]
    })
}

fn transpose(block: &Block) -> Block {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[c * 3 + r] = block[r * 3 + c];
        }
    }
    out
}

/// `into -= l * aᵀ`.
#[inline]
fn subtract_times_turned(into: &mut Block, l: &Block, a: &Block) {
    for r in 0..3 {
        let (l0, l1, l2) = (l[r * 3], l[r * 3 + 1], l[r * 3 + 2]);
        for c in 0..3 {
            into[r * 3 + c] -= l0 * a[c * 3] + l1 * a[c * 3 + 1] + l2 * a[c * 3 + 2];
        }
    }
}

fn multiply(a: &Block, b: &Block) -> Block {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }
    out
}

/// Inverse by its own LDLᵀ. A pivot under `floor` is a redundant direction: it gets no impulse
/// instead of a huge one, so the result is a pseudo inverse there.
fn invert(block: &Block, floor: f64) -> Block {
    let dim = 3;
    let mut lower = [0.0_f64; 9];
    let mut pivots = [0.0_f64; 3];
    for k in 0..dim {
        let mut d = block[k * 3 + k];
        for j in 0..k {
            d -= lower[k * 3 + j] * lower[k * 3 + j] * pivots[j];
        }
        // NaN is dropped too
        if d.is_nan() || d <= floor {
            continue;
        }
        pivots[k] = d;
        lower[k * 3 + k] = 1.0;
        for i in k + 1..dim {
            let mut value = block[i * 3 + k];
            for j in 0..k {
                value -= lower[i * 3 + j] * lower[k * 3 + j] * pivots[j];
            }
            lower[i * 3 + k] = value / d;
        }
    }
    // columns of the inverse: solve L D Lᵀ x = e, dropped pivots give zero
    let mut out = [0.0; 9];
    for column in 0..dim {
        let mut x = [0.0_f64; 3];
        x[column] = 1.0;
        for i in 0..dim {
            for j in 0..i {
                x[i] -= lower[i * 3 + j] * x[j];
            }
        }
        for i in 0..dim {
            x[i] = if pivots[i] > 0.0 {
                x[i] / pivots[i]
            } else {
                0.0
            };
        }
        for i in (0..dim).rev() {
            for j in i + 1..dim {
                x[i] -= lower[j * 3 + i] * x[j];
            }
        }
        for row in 0..dim {
            out[row * 3 + column] = x[row];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row pulling two bodies apart along x, unit masses and no turning.
    fn pull(a: usize, b: usize, target: f32) -> DirectNode {
        let row = DirectRow {
            lin_a: -Vec3::X,
            lin_b: Vec3::X,
            move_lin_a: -Vec3::X,
            move_lin_b: Vec3::X,
            target,
            ..DirectRow::default()
        };
        DirectNode {
            a,
            b,
            count: 1,
            rows: [row, DirectRow::default(), DirectRow::default()],
        }
    }

    #[test]
    fn a_chain_of_rows_solves_exactly_against_the_plain_matrix() {
        // body 0 is held, 1 to 4 hang on in a line
        let nodes: Vec<DirectNode> = (0..4).map(|at| pull(at, at + 1, 1.0 + at as f32)).collect();
        let dynamic = |body: usize| body != 0;
        let symbolic = Symbolic::new(&nodes, &dynamic);
        let mut factor = Factor::default();
        factor.refactor(&symbolic, &nodes);
        let mut parts: Vec<[f64; 3]> = nodes
            .iter()
            .map(|node| [f64::from(node.rows[0].target), 0.0, 0.0])
            .collect();
        factor.solve(&symbolic, &mut parts);
        let x: Vec<f64> = parts.iter().map(|part| part[0]).collect();
        // J M⁻¹ Jᵀ of the chain: 1 on the first diagonal, 2 on the others, -1 beside them
        let matrix = [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 2.0],
        ];
        for (row, want) in matrix.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            let got: f64 = row.iter().zip(&x).map(|(m, v)| m * v).sum();
            assert!((got - want).abs() < 1e-6, "{got} {want} {x:?}");
        }
    }

    #[test]
    fn three_row_points_couple_through_a_shared_body() {
        // inverse masses: body 0 is 2, body 1 is 0.5, body 2 is 1
        let inverse = [2.0, 0.5, 1.0];
        let point = |a: usize, b: usize| {
            let mut node = DirectNode {
                a,
                b,
                count: 3,
                rows: [DirectRow::default(); 3],
            };
            for (row, axis) in node.rows.iter_mut().zip([Vec3::X, Vec3::Y, Vec3::Z]) {
                *row = DirectRow {
                    lin_a: -axis,
                    lin_b: axis,
                    move_lin_a: -axis * inverse[a],
                    move_lin_b: axis * inverse[b],
                    target: 1.0,
                    ..DirectRow::default()
                };
            }
            node
        };
        let nodes = [point(0, 1), point(1, 2)];
        let symbolic = Symbolic::new(&nodes, &|_| true);
        assert!(symbolic.fits(&nodes, &|_| true));
        assert!(!symbolic.fits(&nodes[..1], &|_| true));
        let mut factor = Factor::default();
        factor.refactor(&symbolic, &nodes);
        let mut parts = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        factor.solve(&symbolic, &mut parts);
        let x: Vec<f64> = parts.concat();
        let want = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // per axis: [[2.5, -0.5], [-0.5, 1.5]]
        for axis in 0..3 {
            let (u, v) = (x[axis], x[3 + axis]);
            assert!((2.5 * u - 0.5 * v - want[axis]).abs() < 1e-6, "{x:?}");
            assert!((-0.5 * u + 1.5 * v - want[3 + axis]).abs() < 1e-6, "{x:?}");
        }
    }
}
