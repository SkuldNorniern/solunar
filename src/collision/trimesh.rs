use algebrix::{DVec3, Quat, Vec3};

use super::convex::Poly;
use super::gjk::{Hull, distance};
use super::narrowphase::RawManifold;
use crate::{PhysicsError, Shape};

pub(crate) const MAX_TRIANGLES: usize = 1 << 28;
const LEAF_TRIANGLES: usize = 4;
const CONVEX_EDGE_EPSILON: f32 = 1e-4;
const FEATURE_EPSILON: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Aabb {
    pub(crate) min: Vec3,
    pub(crate) max: Vec3,
}

impl Aabb {
    pub(crate) fn around(center: Vec3, reach: f32) -> Self {
        Self {
            min: center - Vec3::splat(reach),
            max: center + Vec3::splat(reach),
        }
    }

    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn overlaps(self, other: Self) -> bool {
        self.min.x <= other.max.x
            && other.min.x <= self.max.x
            && self.min.y <= other.max.y
            && other.min.y <= self.max.y
            && self.min.z <= other.max.z
            && other.min.z <= self.max.z
    }

    fn of_triangle(triangle: &[Vec3; 3]) -> Self {
        Self {
            min: triangle[0].min(triangle[1]).min(triangle[2]),
            max: triangle[0].max(triangle[1]).max(triangle[2]),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BvhNode {
    bounds: Aabb,
    /// Leaf when `count > 0`, `first` indexes `order`. Otherwise the right child, left is next.
    first: u32,
    count: u32,
}

#[derive(Debug, Clone, PartialEq)]
enum Lookup {
    Grid {
        columns: u32,
        rows: u32,
        spacing_x: f32,
        spacing_z: f32,
    },
    Bvh {
        nodes: Vec<BvhNode>,
        order: Vec<u32>,
    },
}

/// Static triangles. Front faces wind counter clockwise, normals point out of the surface.
/// `convex` has one bit per triangle edge: set when a body may be pushed off that edge at a
/// slant. Flat and concave inner edges stay clear, so nothing catches on the seams.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TriMesh {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) triangles: Vec<[u32; 3]>,
    convex: Vec<u8>,
    lookup: Lookup,
    pub(crate) bounds: Aabb,
    pub(crate) radius: f32,
    pub(crate) grid: Option<(u32, u32, f32, f32)>,
}

impl TriMesh {
    /// # Errors
    /// Fails on non finite vertices, bad indexes or triangles with no area.
    pub(crate) fn from_triangles(
        vertices: Vec<Vec3>,
        triangles: Vec<[u32; 3]>,
    ) -> Result<Self, PhysicsError> {
        validate_mesh(&vertices, &triangles)?;
        let (nodes, order) = build_bvh(&vertices, &triangles);
        Self::finish(vertices, triangles, Lookup::Bvh { nodes, order }, None)
    }

    /// Heights go row by row, x across, z down. The corner of height 0 sits at the local origin.
    ///
    /// # Errors
    /// Fails on a bad size, spacing or height.
    pub(crate) fn from_heights(
        columns: u32,
        rows: u32,
        spacing_x: f32,
        spacing_z: f32,
        heights: &[f32],
    ) -> Result<Self, PhysicsError> {
        let count = (columns as usize).checked_mul(rows as usize);
        let cells = (columns.max(1) as usize - 1).saturating_mul(rows.max(1) as usize - 1);
        if columns < 2
            || rows < 2
            || count != Some(heights.len())
            || cells.saturating_mul(2) > MAX_TRIANGLES
            || !spacing_x.is_finite()
            || !spacing_z.is_finite()
            || spacing_x <= 0.0
            || spacing_z <= 0.0
            || heights.iter().any(|height| !height.is_finite())
        {
            return Err(PhysicsError::InvalidShape);
        }
        let at = |x: u32, z: u32| z * columns + x;
        let mut vertices = Vec::with_capacity(heights.len());
        for z in 0..rows {
            for x in 0..columns {
                vertices.push(Vec3::new(
                    x as f32 * spacing_x,
                    heights[at(x, z) as usize],
                    z as f32 * spacing_z,
                ));
            }
        }
        let mut triangles = Vec::with_capacity(cells * 2);
        for z in 0..rows - 1 {
            for x in 0..columns - 1 {
                triangles.push([at(x, z), at(x, z + 1), at(x + 1, z)]);
                triangles.push([at(x + 1, z), at(x, z + 1), at(x + 1, z + 1)]);
            }
        }
        let lookup = Lookup::Grid {
            columns,
            rows,
            spacing_x,
            spacing_z,
        };
        Self::finish(
            vertices,
            triangles,
            lookup,
            Some((columns, rows, spacing_x, spacing_z)),
        )
    }

    fn finish(
        vertices: Vec<Vec3>,
        triangles: Vec<[u32; 3]>,
        lookup: Lookup,
        grid: Option<(u32, u32, f32, f32)>,
    ) -> Result<Self, PhysicsError> {
        let mut bounds = Aabb {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        };
        let mut radius = 0.0_f32;
        for vertex in &vertices {
            bounds.min = bounds.min.min(*vertex);
            bounds.max = bounds.max.max(*vertex);
            radius = radius.max(vertex.length());
        }
        if !radius.is_finite() {
            return Err(PhysicsError::InvalidShape);
        }
        let convex = convex_edges(&vertices, &triangles);
        Ok(Self {
            vertices,
            triangles,
            convex,
            lookup,
            bounds,
            radius,
            grid,
        })
    }

    pub(crate) fn triangle(&self, index: u32) -> [Vec3; 3] {
        let [a, b, c] = self.triangles[index as usize];
        [
            self.vertices[a as usize],
            self.vertices[b as usize],
            self.vertices[c as usize],
        ]
    }

    /// Calls `visit` with every triangle whose box touches `area`. Order depends only on the mesh.
    pub(crate) fn for_each_near(&self, area: Aabb, mut visit: impl FnMut(u32)) {
        if !self.bounds.overlaps(area) {
            return;
        }
        match &self.lookup {
            Lookup::Grid {
                columns,
                rows,
                spacing_x,
                spacing_z,
            } => {
                let span = |low: f32, high: f32, size: f32, cells: u32| {
                    let first = (low / size).floor().max(0.0) as u32;
                    let last = ((high / size).floor().max(0.0) as u32).min(cells - 1);
                    (first.min(cells - 1), last)
                };
                let (x0, x1) = span(area.min.x, area.max.x, *spacing_x, columns - 1);
                let (z0, z1) = span(area.min.z, area.max.z, *spacing_z, rows - 1);
                for z in z0..=z1 {
                    for x in x0..=x1 {
                        let cell = (z * (columns - 1) + x) * 2;
                        for index in [cell, cell + 1] {
                            if Aabb::of_triangle(&self.triangle(index)).overlaps(area) {
                                visit(index);
                            }
                        }
                    }
                }
            }
            Lookup::Bvh { nodes, order } => {
                let mut stack = crate::collision::tree::Pending::new(0_u32);
                while let Some(at) = stack.pop() {
                    let node = &nodes[at as usize];
                    if !node.bounds.overlaps(area) {
                        continue;
                    }
                    if node.count > 0 {
                        let range = node.first as usize..(node.first + node.count) as usize;
                        for index in &order[range] {
                            if Aabb::of_triangle(&self.triangle(*index)).overlaps(area) {
                                visit(*index);
                            }
                        }
                    } else {
                        stack.push(node.first);
                        stack.push(at + 1);
                    }
                }
            }
        }
    }
}

impl TriMesh {
    /// Nearest front face along unit `direction` from `origin`, in mesh axes, before `max`:
    /// time, face normal and triangle. Equal times go to the lower triangle. With `first`, any
    /// hit will do.
    pub(crate) fn ray(
        &self,
        origin: DVec3,
        direction: DVec3,
        max: f64,
        first: bool,
    ) -> Option<(f64, DVec3, u32)> {
        let (enter, leave) = slab(&self.bounds, origin, direction, max)?;
        let mut best: Option<(f64, DVec3, u32)> = None;
        let test = |index: u32, best: &mut Option<(f64, DVec3, u32)>| {
            let limit = best.map_or(max, |found| found.0);
            if let Some((time, normal)) =
                ray_triangle(origin, direction, &self.triangle(index), limit)
                && best.is_none_or(|found| (time, index) < (found.0, found.2))
            {
                *best = Some((time, normal, index));
            }
        };
        match &self.lookup {
            Lookup::Grid {
                columns,
                rows,
                spacing_x,
                spacing_z,
            } => {
                let (cells_x, cells_z) = (i64::from(*columns) - 1, i64::from(*rows) - 1);
                let (size_x, size_z) = (f64::from(*spacing_x), f64::from(*spacing_z));
                let at = origin + direction * enter;
                let mut x = ((at.x / size_x).floor() as i64).clamp(0, cells_x - 1);
                let mut z = ((at.z / size_z).floor() as i64).clamp(0, cells_z - 1);
                // time to the next cell edge on an axis, and the time one cell takes
                let walk = |cell: i64, size: f64, from: f64, along: f64| {
                    if along > 0.0 {
                        (((cell + 1) as f64 * size - from) / along, size / along, 1)
                    } else if along < 0.0 {
                        ((cell as f64 * size - from) / along, -size / along, -1)
                    } else {
                        (f64::INFINITY, f64::INFINITY, 0)
                    }
                };
                let (mut next_x, step_x, by_x) = walk(x, size_x, origin.x, direction.x);
                let (mut next_z, step_z, by_z) = walk(z, size_z, origin.z, direction.z);
                loop {
                    let cell = ((z * cells_x + x) * 2) as u32;
                    test(cell, &mut best);
                    test(cell + 1, &mut best);
                    let exit = next_x.min(next_z);
                    // a later cell only holds later times
                    let shortest = best.map_or(leave, |found| found.0.min(leave));
                    if exit > shortest || (first && best.is_some()) {
                        break;
                    }
                    if next_x < next_z {
                        x += by_x;
                        next_x += step_x;
                    } else {
                        z += by_z;
                        next_z += step_z;
                    }
                    if !(0..cells_x).contains(&x) || !(0..cells_z).contains(&z) {
                        break;
                    }
                }
            }
            Lookup::Bvh { nodes, order } => {
                let mut stack = crate::collision::tree::Pending::new((0_u32, enter));
                while let Some((at, near)) = stack.pop() {
                    if best.is_some_and(|found| near > found.0) {
                        continue;
                    }
                    let node = &nodes[at as usize];
                    if node.count > 0 {
                        let range = node.first as usize..(node.first + node.count) as usize;
                        for index in &order[range] {
                            test(*index, &mut best);
                        }
                        if first && best.is_some() {
                            break;
                        }
                        continue;
                    }
                    let limit = best.map_or(max, |found| found.0);
                    let (a, b) = (at + 1, node.first);
                    let near_a = slab(&nodes[a as usize].bounds, origin, direction, limit);
                    let near_b = slab(&nodes[b as usize].bounds, origin, direction, limit);
                    // the nearer child goes on top
                    match (near_a, near_b) {
                        (Some((ta, _)), Some((tb, _))) if ta <= tb => {
                            stack.push((b, tb));
                            stack.push((a, ta));
                        }
                        (Some((ta, _)), Some((tb, _))) => {
                            stack.push((a, ta));
                            stack.push((b, tb));
                        }
                        (Some((ta, _)), None) => stack.push((a, ta)),
                        (None, Some((tb, _))) => stack.push((b, tb)),
                        (None, None) => {}
                    }
                }
            }
        }
        best
    }
}

/// Where a ray is inside a box, between 0 and `max`, with some room for rounding.
fn slab(bounds: &Aabb, origin: DVec3, direction: DVec3, max: f64) -> Option<(f64, f64)> {
    let span = bounds.max - bounds.min;
    let room = 1e-5 * (1.0 + f64::from(span.x.max(span.y).max(span.z)));
    let (min, top) = (bounds.min.to_array(), bounds.max.to_array());
    let (origin, direction) = (
        [origin.x, origin.y, origin.z],
        [direction.x, direction.y, direction.z],
    );
    let (mut near, mut far) = (0.0_f64, max);
    for axis in 0..3 {
        let (low, high) = (f64::from(min[axis]) - room, f64::from(top[axis]) + room);
        let (from, along) = (origin[axis], direction[axis]);
        if along.abs() < 1e-300 {
            if from < low || from > high {
                return None;
            }
            continue;
        }
        let inverse = along.recip();
        let (a, b) = ((low - from) * inverse, (high - from) * inverse);
        near = near.max(a.min(b));
        far = far.min(a.max(b));
        if near > far {
            return None;
        }
    }
    Some((near, far))
}

/// Smallest edge cross product a mesh triangle may have, twice its area.
pub(crate) const MIN_TRIANGLE_CROSS: f32 = 1e-8;

fn validate_mesh(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Result<(), PhysicsError> {
    if triangles.is_empty()
        || triangles.len() > MAX_TRIANGLES
        || vertices.iter().any(|vertex| !vertex.is_finite())
    {
        return Err(PhysicsError::InvalidShape);
    }
    for triangle in triangles {
        let mut corners = [Vec3::ZERO; 3];
        for (corner, index) in corners.iter_mut().zip(triangle) {
            *corner = *vertices
                .get(*index as usize)
                .ok_or(PhysicsError::InvalidShape)?;
        }
        let area = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
        if area.length_squared() <= MIN_TRIANGLE_CROSS * MIN_TRIANGLE_CROSS {
            return Err(PhysicsError::InvalidShape);
        }
    }
    Ok(())
}

/// Shared edges by sorted vertex pair. An edge with no one partner counts as convex.
fn convex_edges(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Vec<u8> {
    let mut edges: Vec<(u32, u32, u32, u8)> = Vec::with_capacity(triangles.len() * 3);
    for (index, triangle) in triangles.iter().enumerate() {
        for edge in 0..3 {
            let (p, q) = (triangle[edge], triangle[(edge + 1) % 3]);
            edges.push((p.min(q), p.max(q), index as u32, edge as u8));
        }
    }
    edges.sort_unstable();
    let mut flags = vec![0b111_u8; triangles.len()];
    let corners = |index: u32| {
        let [a, b, c] = triangles[index as usize];
        [
            vertices[a as usize],
            vertices[b as usize],
            vertices[c as usize],
        ]
    };
    let mut at = 0;
    while at < edges.len() {
        let mut end = at + 1;
        while end < edges.len() && (edges[end].0, edges[end].1) == (edges[at].0, edges[at].1) {
            end += 1;
        }
        if end - at == 2 {
            for (mine, theirs) in [(edges[at], edges[at + 1]), (edges[at + 1], edges[at])] {
                let triangle = corners(mine.2);
                let other = corners(theirs.2);
                let normal = (triangle[1] - triangle[0])
                    .cross(triangle[2] - triangle[0])
                    .normalize();
                let opposite = other[(theirs.3 as usize + 2) % 3];
                if (opposite - triangle[0]).dot(normal) >= -CONVEX_EDGE_EPSILON {
                    flags[mine.2 as usize] &= !(1 << mine.3);
                }
            }
        }
        at = end;
    }
    flags
}

fn build_bvh(vertices: &[Vec3], triangles: &[[u32; 3]]) -> (Vec<BvhNode>, Vec<u32>) {
    let boxes: Vec<Aabb> = triangles
        .iter()
        .map(|[a, b, c]| {
            Aabb::of_triangle(&[
                vertices[*a as usize],
                vertices[*b as usize],
                vertices[*c as usize],
            ])
        })
        .collect();
    let mut order: Vec<u32> = (0..triangles.len() as u32).collect();
    let mut nodes = Vec::with_capacity(triangles.len());
    build_node(&boxes, &mut order, 0, &mut nodes);
    (nodes, order)
}

fn build_node(boxes: &[Aabb], order: &mut [u32], offset: u32, nodes: &mut Vec<BvhNode>) -> u32 {
    let mut bounds = boxes[order[0] as usize];
    for index in order.iter() {
        bounds = bounds.union(boxes[*index as usize]);
    }
    let at = nodes.len() as u32;
    if order.len() <= LEAF_TRIANGLES {
        nodes.push(BvhNode {
            bounds,
            first: offset,
            count: order.len() as u32,
        });
        return at;
    }
    nodes.push(BvhNode {
        bounds,
        first: 0,
        count: 0,
    });
    let size = bounds.max - bounds.min;
    let axis = if size.x >= size.y && size.x >= size.z {
        0
    } else if size.y >= size.z {
        1
    } else {
        2
    };
    let key = |index: u32| {
        let center = boxes[index as usize].min + boxes[index as usize].max;
        [center.x, center.y, center.z][axis]
    };
    order.sort_by(|a, b| key(*a).total_cmp(&key(*b)).then(a.cmp(b)));
    let middle = order.len() / 2;
    let (left, right) = order.split_at_mut(middle);
    build_node(boxes, left, offset, nodes);
    let right_at = build_node(boxes, right, offset + middle as u32, nodes);
    nodes[at as usize].first = right_at;
    at
}

/// Closest point on a triangle and the feature it falls on: 0 inside, then edge bits.
fn closest_on_triangle(point: Vec3, triangle: &[Vec3; 3]) -> (Vec3, u8) {
    let [a, b, c] = *triangle;
    let (ab, ac, ap) = (b - a, c - a, point - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, 0b101);
    }
    let bp = point - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, 0b011);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (a + ab * (d1 / (d1 - d3)), 0b001);
    }
    let cp = point - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, 0b110);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (a + ac * (d2 / (d2 - d6)), 0b100);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))), 0b010);
    }
    let scale = (va + vb + vc).recip();
    (a + ab * (vb * scale) + ac * (vc * scale), 0)
}

/// Contact of a shape against one triangle. Positions are in mesh axes from the mesh origin,
/// the normal points from the triangle to the shape, and never against the face normal.
pub(crate) fn triangle_contact(
    mesh: &TriMesh,
    index: u32,
    shape: &Shape,
    rotation: Quat,
    center: Vec3,
    margin: f32,
) -> Option<RawManifold> {
    let triangle = mesh.triangle(index);
    let convex = mesh.convex[index as usize];
    match *shape {
        Shape::Sphere { radius } => {
            rounded_contact(&triangle, convex, center, center, radius, margin)
        }
        Shape::Capsule {
            radius,
            half_height,
        } => {
            let half = rotation.rotate_vec3(Vec3::Y * half_height);
            rounded_contact(
                &triangle,
                convex,
                center - half,
                center + half,
                radius,
                margin,
            )
        }
        Shape::Cuboid { half_extents } => poly_triangle(
            mesh,
            index,
            &Poly::of_box(half_extents, rotation, center),
            margin,
        ),
        Shape::Compound(_) | Shape::TriMesh(_) | Shape::Convex(_) => None,
    }
}

fn rounded_contact(
    triangle: &[Vec3; 3],
    convex: u8,
    start: Vec3,
    end: Vec3,
    radius: f32,
    margin: f32,
) -> Option<RawManifold> {
    let face = (triangle[1] - triangle[0])
        .cross(triangle[2] - triangle[0])
        .normalize();
    let core = if start == end {
        Hull::Point(DVec3::from_vec3(start))
    } else {
        Hull::Segment(DVec3::from_vec3(start), DVec3::from_vec3(end))
    };
    let near = distance(
        &core,
        &Hull::Triangle(
            DVec3::from_vec3(triangle[0]),
            DVec3::from_vec3(triangle[1]),
            DVec3::from_vec3(triangle[2]),
        ),
    );
    let reach = radius + margin;
    let through = near.direction == DVec3::ZERO;
    if !through && near.distance >= f64::from(reach) {
        return None;
    }
    let on_triangle = near.on_b.as_vec3();
    let on_core = on_triangle + near.direction.as_vec3() * near.distance as f32;
    let behind = (on_core - triangle[0]).dot(face) < 0.0;
    let (_, feature) = closest_on_triangle(on_core, triangle);
    // a slant only makes sense off a real outer edge, from in front of the face
    let slanted = !through && !behind && feature != 0 && feature & convex != 0;
    let normal = if slanted {
        near.direction.as_vec3().normalize()
    } else {
        face
    };
    let mut result = RawManifold::new(normal);
    let mut add = |point: Vec3| {
        let (closest, part) = closest_on_triangle(point, triangle);
        // the far end of a capsule only counts while it is over this triangle
        if part != 0 && !slanted && point != on_core {
            return;
        }
        let gap = (point - closest).dot(normal);
        let depth = radius - gap;
        if depth > -margin {
            result.push(closest + normal * (gap - radius + depth * 0.5), depth);
        }
    };
    add(on_core);
    if start != end {
        for tip in [start, end] {
            if (tip - on_core).length_squared() > 1e-6 && !slanted {
                add(tip);
            }
        }
    }
    result.filled()
}

/// A box or convex hull against a triangle, in mesh axes.
pub(crate) fn poly_triangle(
    mesh: &TriMesh,
    index: u32,
    poly: &Poly<'_>,
    margin: f32,
) -> Option<RawManifold> {
    let triangle = mesh.triangle(index);
    let convex = mesh.convex[index as usize];
    let center = poly.center;
    let face = (triangle[1] - triangle[0])
        .cross(triangle[2] - triangle[0])
        .normalize();
    let edges = [
        triangle[1] - triangle[0],
        triangle[2] - triangle[1],
        triangle[0] - triangle[2],
    ];
    // gap along the face normal is one sided, something buried is pushed out the front
    let face_gap = (center - triangle[0]).dot(face) - poly.reach(-face);
    if face_gap > margin {
        return None;
    }
    let mut best_gap = face_gap;
    let mut best_axis = face;
    let mut is_face = true;
    let mut try_axis = |axis: Vec3, allowed: bool| -> bool {
        let length = axis.length();
        if length < 1e-4 {
            return true;
        }
        let mut axis = axis * length.recip();
        if axis.dot(center - triangle[0]) < 0.0 {
            axis = -axis;
        }
        let from_center = |point: Vec3| (point - center).dot(axis);
        let highest = from_center(triangle[0])
            .max(from_center(triangle[1]))
            .max(from_center(triangle[2]));
        let gap = -highest - poly.reach(-axis);
        if gap > margin {
            return false;
        }
        if allowed && gap > best_gap + 1e-3 {
            best_gap = gap;
            best_axis = axis;
            is_face = false;
        }
        true
    };
    let any_edge = convex != 0;
    for (normal, _) in poly.planes() {
        if !try_axis(*normal, any_edge) {
            return None;
        }
    }
    for (at, edge) in edges.iter().enumerate() {
        let open = convex & (1 << at) != 0;
        for axis in poly.edges() {
            if !try_axis(edge.cross(*axis), open) {
                return None;
            }
        }
    }
    let normal = best_axis;
    let mut result = RawManifold::new(normal);
    if is_face {
        // corners over the triangle, and triangle corners under the shape
        let mut kept = [(Vec3::ZERO, 0.0_f32); 16];
        let mut count = 0;
        for corner in poly.vertices() {
            let gap = (*corner - triangle[0]).dot(face);
            let (closest, part) = closest_on_triangle(*corner, &triangle);
            if part == 0 && gap < margin && (center - *corner).dot(face) > 0.0 && count < kept.len()
            {
                kept[count] = (closest + face * (gap * 0.5), -gap);
                count += 1;
            }
        }
        for corner in &triangle {
            if let Some(gap) = poly.surface_above(*corner, face)
                && gap < margin
                && count < kept.len()
            {
                kept[count] = (*corner + face * (gap * 0.5), -gap);
                count += 1;
            }
        }
        result.reduce_from(&kept[..count]);
    }
    if result.count == 0 {
        // edge against edge, or no corner over the triangle: one point between the closest parts
        let on_triangle = {
            let mut best = triangle[0];
            let mut most = f32::NEG_INFINITY;
            for corner in &triangle {
                let value = corner.dot(normal);
                if value > most {
                    most = value;
                    best = *corner;
                }
            }
            best
        };
        let (closest, _) = closest_on_triangle(center, &triangle);
        let point = if is_face { closest } else { on_triangle };
        result.push(point + normal * (best_gap * 0.5), -best_gap);
    }
    result.filled()
}

/// Front faces only. Returns the time along `direction`, and the face normal.
pub(crate) fn ray_triangle(
    origin: DVec3,
    direction: DVec3,
    triangle: &[Vec3; 3],
    max: f64,
) -> Option<(f64, DVec3)> {
    let a = DVec3::from_vec3(triangle[0]);
    let ab = DVec3::from_vec3(triangle[1]) - a;
    let ac = DVec3::from_vec3(triangle[2]) - a;
    let normal = ab.cross(ac);
    let facing = direction.dot(normal);
    if facing >= -1e-14 {
        return None;
    }
    let to_origin = origin - a;
    let time = -to_origin.dot(normal) / facing;
    if time < 0.0 || time > max {
        return None;
    }
    let hit = to_origin + direction * time;
    let (d00, d01, d11) = (ab.dot(ab), ab.dot(ac), ac.dot(ac));
    let (d20, d21) = (hit.dot(ab), hit.dot(ac));
    let scale = (d00 * d11 - d01 * d01).recip();
    let v = (d11 * d20 - d01 * d21) * scale;
    let w = (d00 * d21 - d01 * d20) * scale;
    if v < 0.0 || w < 0.0 || v + w > 1.0 {
        return None;
    }
    Some((time, normal.normalize()))
}

/// Closest surface point and outward direction for a point query. Never negative.
pub(crate) fn point_triangle(point: Vec3, triangle: &[Vec3; 3]) -> (f32, Vec3) {
    let (closest, _) = closest_on_triangle(point, triangle);
    let gap = point - closest;
    let length = gap.length();
    if length > FEATURE_EPSILON {
        (length, gap * length.recip())
    } else {
        let face = (triangle[1] - triangle[0])
            .cross(triangle[2] - triangle[0])
            .normalize();
        (0.0, face)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GridPosition, Material, PhysicsSettings, PhysicsWorld, QueryFilter, Ray, RigidBodyDesc,
        ShapeCast,
    };

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    fn at(x: f32, y: f32, z: f32) -> GridPosition {
        GridPosition::from_local(Vec3::new(x, y, z)).unwrap()
    }

    fn near(a: f32, b: f32, tolerance: f32) {
        assert!((a - b).abs() <= tolerance, "{a} != {b}");
    }

    /// 21 by 21 posts, 1 m apart, centered on the origin.
    fn flat(world: &mut PhysicsWorld, height: f32) -> Shape {
        let heights = vec![height; 21 * 21];
        let shape = world.add_heightfield(21, 21, 1.0, 1.0, &heights).unwrap();
        world
            .spawn(RigidBodyDesc::fixed(shape).local_position(Vec3::new(-10.0, 0.0, -10.0)))
            .unwrap();
        shape
    }

    fn slick() -> Material {
        Material {
            friction: 0.0,
            ..Material::default()
        }
    }

    #[test]
    fn mesh_rays_answer_like_every_triangle_tried() {
        let (columns, rows) = (17, 13);
        let heights: Vec<f32> = (0..columns * rows)
            .map(|i| ((i % columns) as f32 * 0.9).sin() * 1.5 + ((i / columns) as f32 * 0.6).cos())
            .collect();
        let grid = TriMesh::from_heights(columns, rows, 0.7, 1.3, &heights).unwrap();
        let tree = TriMesh::from_triangles(grid.vertices.clone(), grid.triangles.clone()).unwrap();
        let mut seed = 7_u64;
        let mut random = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut hits = 0;
        for i in 0..3000 {
            let origin = DVec3::new(
                random() * 16.0 - 2.0,
                random() * 8.0 - 3.0,
                random() * 20.0 - 2.0,
            );
            let direction = match i % 5 {
                0 => DVec3::new(0.0, -1.0, 0.0),
                1 => DVec3::new(random() - 0.5, 0.0, 0.0).normalize(),
                _ => DVec3::new(random() - 0.5, random() - 0.5, random() - 0.5).normalize(),
            };
            let max = random() * 30.0;
            let mut wanted: Option<(f64, u32)> = None;
            for index in 0..grid.triangles.len() as u32 {
                let limit = wanted.map_or(max, |found| found.0);
                if let Some((time, _)) =
                    ray_triangle(origin, direction, &grid.triangle(index), limit)
                    && wanted.is_none_or(|found| (time, index) < found)
                {
                    wanted = Some((time, index));
                }
            }
            hits += usize::from(wanted.is_some());
            for mesh in [&grid, &tree] {
                let found = mesh
                    .ray(origin, direction, max, false)
                    .map(|(time, _, index)| (time, index));
                assert_eq!(found, wanted, "ray {i}");
                let any = mesh.ray(origin, direction, max, true);
                assert_eq!(any.is_some(), wanted.is_some(), "ray {i}");
            }
        }
        assert!(hits > 500, "{hits} hits");
    }

    #[test]
    fn grid_lookup_finds_the_right_triangles() {
        let heights = vec![0.0; 16];
        let mesh = TriMesh::from_heights(4, 4, 1.0, 1.0, &heights).unwrap();
        assert_eq!(mesh.triangles.len(), 18);
        let mut found = Vec::new();
        mesh.for_each_near(Aabb::around(Vec3::new(1.5, 0.0, 1.5), 0.1), |i| {
            found.push(i)
        });
        assert_eq!(found, [8, 9]);
        found.clear();
        mesh.for_each_near(Aabb::around(Vec3::new(50.0, 0.0, 1.5), 0.1), |i| {
            found.push(i)
        });
        assert!(found.is_empty());
        found.clear();
        mesh.for_each_near(Aabb::around(Vec3::new(-0.05, 0.0, 0.5), 0.1), |i| {
            found.push(i)
        });
        assert_eq!(found, [0, 1]);
    }

    #[test]
    fn bvh_finds_the_same_triangles_as_checking_all() {
        let mut state = 0x4242_1234_dead_0001_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 100.0 - 10.0
        };
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for _ in 0..500 {
            let base = Vec3::new(next(), next(), next());
            let start = vertices.len() as u32;
            vertices.push(base);
            vertices.push(base + Vec3::new(1.0, next() * 0.05, 0.0));
            vertices.push(base + Vec3::new(0.0, next() * 0.05, 1.0));
            triangles.push([start, start + 2, start + 1]);
        }
        let mesh = TriMesh::from_triangles(vertices, triangles).unwrap();
        for _ in 0..300 {
            let area = Aabb::around(Vec3::new(next(), next(), next()), next().abs() * 0.3);
            let mut found = Vec::new();
            mesh.for_each_near(area, |index| found.push(index));
            found.sort_unstable();
            let brute: Vec<u32> = (0..500)
                .filter(|index| Aabb::of_triangle(&mesh.triangle(*index)).overlaps(area))
                .collect();
            assert_eq!(found, brute);
        }
    }

    #[test]
    fn flat_inner_edges_are_clear_and_outer_ones_set() {
        let heights = vec![0.0; 9];
        let mesh = TriMesh::from_heights(3, 3, 1.0, 1.0, &heights).unwrap();
        // the shared cell diagonal of the first cell is flat, its outer sides have no partner
        assert_eq!(mesh.convex[0] & 0b010, 0);
        assert_ne!(mesh.convex[0] & 0b101, 0);
        let ridge = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let peak = TriMesh::from_heights(3, 3, 1.0, 1.0, &ridge).unwrap();
        let valley = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let dip = TriMesh::from_heights(3, 3, 1.0, 1.0, &valley).unwrap();
        // the middle column edge between cells: a ridge can push at a slant, a valley can not
        let shared = |mesh: &TriMesh, triangle: u32| {
            let corners = mesh.triangles[triangle as usize];
            (0..3)
                .find(|edge| {
                    let (p, q) = (corners[*edge], corners[(*edge + 1) % 3]);
                    p.min(q) == 1 && p.max(q) == 4
                })
                .map(|edge| mesh.convex[triangle as usize] & (1 << edge))
        };
        assert_ne!(shared(&peak, 1), Some(0));
        assert_eq!(shared(&dip, 1), Some(0));
    }

    #[test]
    fn sphere_rests_on_a_heightfield() {
        let mut world = world();
        flat(&mut world, 2.0);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.3, 4.0, 0.7)),
            )
            .unwrap();
        for _ in 0..180 {
            world.step_fixed().unwrap();
        }
        near(world.body_state(ball).unwrap().position.local.y, 2.5, 0.01);
    }

    #[test]
    fn box_and_capsule_rest_on_a_heightfield() {
        let mut world = world();
        flat(&mut world, 0.0);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(0.2, 1.0, 0.1)),
            )
            .unwrap();
        let lying = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let capsule = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::capsule(0.25, 1.0))
                    .rotation(lying)
                    .local_position(Vec3::new(-4.3, 1.0, 2.2)),
            )
            .unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        let cube = world.body_state(cube).unwrap();
        near(cube.position.local.y, 0.5, 0.02);
        assert!(cube.rotation.rotate_vec3(Vec3::Y).y > 0.999, "{cube:?}");
        let capsule = world.body_state(capsule).unwrap();
        near(capsule.position.local.y, 0.25, 0.02);
        near(capsule.rotation.rotate_vec3(Vec3::Y).y, 0.0, 0.02);
    }

    #[test]
    fn slick_sphere_crosses_the_seams_without_a_bump() {
        let mut world = world();
        flat(&mut world, 0.0);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .material(slick())
                    .local_position(Vec3::new(-9.0, 0.5, 0.2))
                    .linear_velocity(Vec3::new(5.0, 0.0, 1.0)),
            )
            .unwrap();
        for _ in 0..180 {
            world.step_fixed().unwrap();
            let state = world.body_state(ball).unwrap();
            near(state.linear_velocity.x, 5.0, 1e-3);
            near(state.linear_velocity.z, 1.0, 1e-3);
            near(state.position.local.y, 0.5, 2e-3);
            assert!(state.linear_velocity.y.abs() < 0.05, "{state:?}");
        }
    }

    #[test]
    fn slick_box_crosses_the_seams_without_a_bump() {
        let mut world = world();
        flat(&mut world, 0.0);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(0.5, 0.25, 0.5)))
                    .material(slick())
                    .local_position(Vec3::new(-9.0, 0.25, 0.3))
                    .linear_velocity(Vec3::new(5.0, 0.0, 1.0)),
            )
            .unwrap();
        for _ in 0..180 {
            world.step_fixed().unwrap();
            let state = world.body_state(cube).unwrap();
            near(state.linear_velocity.x, 5.0, 0.02);
            near(state.linear_velocity.z, 1.0, 0.02);
            near(state.position.local.y, 0.25, 5e-3);
            // a corner crossing a seam loses its warm start, measured about 0.2
            assert!(state.angular_velocity.length() < 0.5, "{state:?}");
        }
    }

    fn ramp_heights(columns: u32, rows: u32, slope: f32) -> Vec<f32> {
        let mut heights = Vec::new();
        for _ in 0..rows {
            for x in 0..columns {
                heights.push(-(x as f32) * slope);
            }
        }
        heights
    }

    #[test]
    fn box_slides_down_a_slope_at_the_analytic_rate() {
        let mut world = world();
        let slope = 0.5_f32;
        let heights = ramp_heights(61, 11, slope);
        let shape = world.add_heightfield(61, 11, 1.0, 1.0, &heights).unwrap();
        world.spawn(RigidBodyDesc::fixed(shape)).unwrap();
        let angle = slope.atan();
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.25)))
                    .rotation(Quat::from_rotation_z(-angle))
                    .local_position(Vec3::new(
                        2.0 + 0.25 * angle.sin(),
                        -2.0 * slope + 0.25 * angle.cos(),
                        5.0,
                    )),
            )
            .unwrap();
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        let start = world.body_state(cube).unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        let end = world.body_state(cube).unwrap();
        // a = g (sin - mu cos), mu = 0.5 for two default materials
        let wanted = 9.81 * (angle.sin() - 0.5 * angle.cos());
        let gained = (end.linear_velocity - start.linear_velocity).length();
        assert!(
            (gained - wanted).abs() < wanted * 0.1 + 0.05,
            "{gained} {wanted}"
        );
    }

    #[test]
    fn steep_bumps_stop_a_ball_rolling_into_them() {
        let mut world = world();
        let mut heights = vec![0.0; 21 * 21];
        for z in 0..21 {
            heights[z * 21 + 12] = 3.0;
        }
        let shape = world.add_heightfield(21, 21, 1.0, 1.0, &heights).unwrap();
        world
            .spawn(RigidBodyDesc::fixed(shape).local_position(Vec3::new(-10.0, 0.0, -10.0)))
            .unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.4))
                    .material(slick())
                    .local_position(Vec3::new(-5.0, 0.4, 0.1))
                    .linear_velocity(Vec3::new(3.0, 0.0, 0.0)),
            )
            .unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        // ridge starts at x = 1, 3 m high, not reachable with 0.46 m of kinetic height
        assert!(world.body_state(ball).unwrap().position.local.x < 2.0);
    }

    #[test]
    fn something_under_the_surface_comes_back_out_the_top() {
        let mut world = world();
        flat(&mut world, 0.0);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .local_position(Vec3::new(0.2, -0.3, 0.1)),
            )
            .unwrap();
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(4.2, -0.3, 2.1)),
            )
            .unwrap();
        for _ in 0..180 {
            world.step_fixed().unwrap();
        }
        near(world.body_state(ball).unwrap().position.local.y, 0.5, 0.02);
        let cube = world.body_state(cube).unwrap();
        near(cube.position.local.y, 0.5, 0.02);
        assert!(
            cube.rotation.rotate_vec3(Vec3::Y).y.abs() > 0.999,
            "{cube:?}"
        );
    }

    #[test]
    fn a_mesh_works_like_the_same_heightfield() {
        let mut vertices = Vec::new();
        for z in 0..5 {
            for x in 0..5 {
                vertices.push(Vec3::new(x as f32 * 2.0 - 4.0, 0.0, z as f32 * 2.0 - 4.0));
            }
        }
        let mut triangles = Vec::new();
        for z in 0..4_u32 {
            for x in 0..4_u32 {
                let at = |x: u32, z: u32| z * 5 + x;
                triangles.push([at(x, z), at(x, z + 1), at(x + 1, z)]);
                triangles.push([at(x + 1, z), at(x, z + 1), at(x + 1, z + 1)]);
            }
        }
        let mut world = world();
        let shape = world.add_mesh(&vertices, &triangles).unwrap();
        world.spawn(RigidBodyDesc::fixed(shape)).unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.3, 2.0, 0.4)),
            )
            .unwrap();
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.25)))
                    .local_position(Vec3::new(-2.3, 2.0, 1.4)),
            )
            .unwrap();
        for _ in 0..180 {
            world.step_fixed().unwrap();
        }
        near(world.body_state(ball).unwrap().position.local.y, 0.5, 0.01);
        near(world.body_state(cube).unwrap().position.local.y, 0.25, 0.02);
    }

    #[test]
    fn turned_mesh_body_works_too() {
        let vertices = [
            Vec3::new(-20.0, 0.0, -20.0),
            Vec3::new(20.0, 0.0, -20.0),
            Vec3::new(-20.0, 0.0, 20.0),
        ];
        let mut world = world();
        let shape = world.add_mesh(&vertices, &[[0, 2, 1]]).unwrap();
        world
            .spawn(
                RigidBodyDesc::fixed(shape)
                    .rotation(Quat::from_rotation_y(0.7))
                    .local_position(Vec3::new(0.0, 1.0, 0.0)),
            )
            .unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 3.0, 0.0)),
            )
            .unwrap();
        for _ in 0..180 {
            world.step_fixed().unwrap();
        }
        near(world.body_state(ball).unwrap().position.local.y, 1.5, 0.01);
    }

    #[test]
    fn compound_lands_on_a_heightfield_and_sleeps() {
        let mut world = world();
        flat(&mut world, 0.0);
        let info = world
            .add_compound(&[
                crate::CompoundChild::new(
                    Shape::cuboid(Vec3::splat(0.25)),
                    Vec3::new(-0.75, 0.0, 0.0),
                ),
                crate::CompoundChild::new(
                    Shape::cuboid(Vec3::splat(0.25)),
                    Vec3::new(0.75, 0.0, 0.0),
                ),
            ])
            .unwrap();
        let (place, turn) = info.placed(Vec3::new(0.1, 1.0, 0.2), Quat::IDENTITY);
        let body = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        for _ in 0..300 {
            world.step_fixed().unwrap();
        }
        let state = world.body_state(body).unwrap();
        near(state.position.local.y, 0.25, 0.02);
        assert!(state.asleep, "{state:?}");
    }

    #[test]
    fn dynamic_triangle_bodies_are_refused() {
        let mut world = world();
        let heights = vec![0.0; 4];
        let shape = world.add_heightfield(2, 2, 1.0, 1.0, &heights).unwrap();
        assert_eq!(
            world.spawn(RigidBodyDesc::dynamic(shape)),
            Err(PhysicsError::InvalidShape)
        );
        assert!(world.spawn(RigidBodyDesc::kinematic(shape)).is_ok());
    }

    #[test]
    fn bad_terrain_and_meshes_fail() {
        let mut world = world();
        let nan = f32::NAN;
        for (columns, rows, x, z, heights) in [
            (1, 2, 1.0, 1.0, vec![0.0; 2]),
            (2, 2, 1.0, 1.0, vec![0.0; 3]),
            (2, 2, 0.0, 1.0, vec![0.0; 4]),
            (2, 2, 1.0, nan, vec![0.0; 4]),
            (2, 2, 1.0, 1.0, vec![0.0, nan, 0.0, 0.0]),
            (u32::MAX, u32::MAX, 1.0, 1.0, vec![0.0; 4]),
        ] {
            assert_eq!(
                world.add_heightfield(columns, rows, x, z, &heights),
                Err(PhysicsError::InvalidShape)
            );
        }
        let points = [Vec3::ZERO, Vec3::X, Vec3::Z];
        assert!(world.add_mesh(&points, &[]).is_err());
        assert!(world.add_mesh(&points, &[[0, 1, 9]]).is_err());
        assert!(world.add_mesh(&points, &[[0, 1, 1]]).is_err());
        assert!(
            world
                .add_mesh(&[Vec3::ZERO, Vec3::X, Vec3::splat(nan)], &[[0, 1, 2]])
                .is_err()
        );
    }

    fn sloped(world: &mut PhysicsWorld) -> crate::BodyHandle {
        let heights = ramp_heights(41, 41, -0.25);
        let shape = world.add_heightfield(41, 41, 1.0, 1.0, &heights).unwrap();
        world
            .spawn(RigidBodyDesc::fixed(shape).local_position(Vec3::new(-20.0, 0.0, -20.0)))
            .unwrap()
    }

    #[test]
    fn ray_hits_terrain_at_the_exact_height() {
        let mut world = world();
        let body = sloped(&mut world);
        let filter = QueryFilter::default();
        let ray = Ray {
            origin: at(4.0, 50.0, 3.3),
            direction: -Vec3::Y,
            max_distance: 100.0,
        };
        let hit = world.raycast(&ray, &filter).unwrap().unwrap();
        assert_eq!(hit.body, body);
        // height at x = 4 is (4 + 20) * 0.25
        near(hit.distance, 44.0, 1e-3);
        let expected = Vec3::new(-0.25, 1.0, 0.0).normalize();
        assert!((hit.normal - expected).length() < 1e-3, "{:?}", hit.normal);
        let upward = Ray {
            origin: at(4.0, -50.0, 3.3),
            direction: Vec3::Y,
            max_distance: 100.0,
        };
        assert_eq!(world.raycast(&upward, &filter).unwrap(), None);
        let off = Ray {
            origin: at(50.0, 50.0, 3.3),
            direction: -Vec3::Y,
            max_distance: 100.0,
        };
        assert_eq!(world.raycast(&off, &filter).unwrap(), None);
    }

    #[test]
    fn shape_cast_stops_on_terrain() {
        let mut world = world();
        flat(&mut world, 1.0);
        let filter = QueryFilter::default();
        let cast = |shape: Shape| ShapeCast {
            shape,
            rotation: Quat::IDENTITY,
            origin: at(0.3, 10.0, 0.4),
            direction: -Vec3::Y,
            max_distance: 100.0,
        };
        let ball = world
            .shape_cast(&cast(Shape::sphere(0.5)), &filter)
            .unwrap()
            .unwrap();
        near(ball.distance, 8.5, 1e-3);
        near(ball.normal.y, 1.0, 1e-2);
        let cube = world
            .shape_cast(&cast(Shape::cuboid(Vec3::splat(0.5))), &filter)
            .unwrap()
            .unwrap();
        near(cube.distance, 8.5, 1e-3);
        near(cube.point.local.y, 1.0, 1e-2);
    }

    #[test]
    fn overlap_and_nearest_point_see_terrain() {
        let mut world = world();
        flat(&mut world, 1.0);
        let filter = QueryFilter::default();
        let mut out = Vec::new();
        world
            .overlap(
                &Shape::sphere(0.5),
                at(0.3, 1.2, 0.4),
                Quat::IDENTITY,
                &filter,
                &mut out,
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        world
            .overlap(
                &Shape::sphere(0.5),
                at(0.3, 3.0, 0.4),
                Quat::IDENTITY,
                &filter,
                &mut out,
            )
            .unwrap();
        assert!(out.is_empty());
        let hit = world
            .nearest_point(at(0.3, 4.0, 0.4), 10.0, &filter)
            .unwrap()
            .unwrap();
        near(hit.distance, 3.0, 1e-4);
        near(hit.point.local.y, 1.0, 1e-4);
        near(hit.normal.y, 1.0, 1e-4);
        assert_eq!(
            world
                .nearest_point(at(0.3, 40.0, 0.4), 1.0, &filter)
                .unwrap(),
            None
        );
    }

    #[test]
    fn snapshot_with_terrain_and_a_mesh_continues_exactly() {
        let mut world = world();
        flat(&mut world, 0.0);
        let mesh = world
            .add_mesh(&[Vec3::ZERO, Vec3::X, Vec3::Z], &[[0, 2, 1]])
            .unwrap();
        world
            .spawn(RigidBodyDesc::fixed(mesh).local_position(Vec3::new(5.0, 5.0, 5.0)))
            .unwrap();
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(0.2, 1.0, 0.3)),
            )
            .unwrap();
        for _ in 0..20 {
            world.step_fixed().unwrap();
        }
        let bytes = world.save_snapshot();
        let mut loaded = PhysicsWorld::load_snapshot(&bytes).unwrap();
        assert_eq!(loaded.save_snapshot(), bytes);
        assert_eq!(loaded.bodies, world.bodies);
        for _ in 0..60 {
            world.step_fixed().unwrap();
            loaded.step_fixed().unwrap();
        }
        assert_eq!(loaded.state_hash(), world.state_hash());
    }

    #[test]
    fn random_contacts_stay_finite_and_point_out_of_the_face() {
        let mut state = 0x7654_3210_cafe_0001_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 1000.0 - 1.0
        };
        let heights: Vec<f32> = (0..25).map(|_| next() * 0.5).collect();
        let mesh = TriMesh::from_heights(5, 5, 1.0, 1.0, &heights).unwrap();
        let shapes = [
            Shape::sphere(0.4),
            Shape::capsule(0.2, 0.5),
            Shape::cuboid(Vec3::new(0.3, 0.2, 0.5)),
        ];
        let mut seen = 0;
        for _ in 0..20_000 {
            let shape = shapes[(next().abs() * 2.99) as usize];
            let rotation = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let center = Vec3::new(next() * 2.0 + 2.0, next(), next() * 2.0 + 2.0);
            let index = ((next().abs() * 31.9) as u32).min(31);
            if let Some(hit) = triangle_contact(&mesh, index, &shape, rotation, center, 0.02) {
                seen += 1;
                assert!(hit.count >= 1 && hit.count <= crate::collision::MAX_CONTACTS);
                assert!((hit.normal.length() - 1.0).abs() < 1e-3, "{hit:?}");
                for point in &hit.points[..hit.count] {
                    assert!(point.position.is_finite(), "{hit:?}");
                    assert!(point.depth.is_finite() && point.depth > -0.1, "{hit:?}");
                }
            }
        }
        assert!(seen > 500, "{seen}");
    }
}
