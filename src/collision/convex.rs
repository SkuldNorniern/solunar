use algebrix::{DQuat, DVec3, Quat, Vec3};

use super::gjk::{Hull, distance};
use super::narrowphase::RawManifold;
use crate::{MassProperties, PhysicsError};

pub(crate) const MAX_CONVEX_POINTS: usize = 64;
const MAX_FACES: usize = 2 * MAX_CONVEX_POINTS;
const MAX_EDGES: usize = 3 * MAX_CONVEX_POINTS;
const MAX_POLYGON: usize = 32;
const EDGE_BIAS: f32 = 1e-3;

/// A convex shape, stored centered on its mass in its principal axes. Faces are counter clockwise
/// seen from outside, flattened so a box can share the same view with no allocation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConvexHull {
    /// The points it was built from, a reload rebuilds the same hull from these.
    pub(crate) source: Vec<Vec3>,
    pub(crate) vertices: Vec<Vec3>,
    planes: Vec<(Vec3, f32)>,
    face_starts: Vec<u32>,
    face_corners: Vec<u32>,
    edges: Vec<Vec3>,
    pub(crate) unit: MassProperties,
    pub(crate) radius: f32,
    pub(crate) box_center: Vec3,
    pub(crate) box_half: Vec3,
}

struct RawHull {
    vertices: Vec<Vec3>,
    planes: Vec<(Vec3, f32)>,
    face_starts: Vec<u32>,
    face_corners: Vec<u32>,
    edges: Vec<Vec3>,
}

/// Every plane through three points with all points on one side is a face.
/// Quartic in the point count, the cap keeps it small and it only runs when a shape is added.
fn build_hull(points: &[Vec3]) -> Result<RawHull, PhysicsError> {
    if points.len() < 4 || points.len() > MAX_CONVEX_POINTS {
        return Err(PhysicsError::InvalidShape);
    }
    let mut size = 0.0_f32;
    for point in points {
        if !point.is_finite() {
            return Err(PhysicsError::InvalidShape);
        }
        size = size.max(point.length());
    }
    let tolerance = (size * 1e-5).max(1e-6);
    let mut unique: Vec<Vec3> = Vec::with_capacity(points.len());
    for point in points {
        if !unique
            .iter()
            .any(|known| (*known - *point).length() <= tolerance)
        {
            unique.push(*point);
        }
    }
    let mut planes: Vec<(Vec3, f32)> = Vec::new();
    for i in 0..unique.len() {
        for j in i + 1..unique.len() {
            for k in j + 1..unique.len() {
                let crossed = (unique[j] - unique[i]).cross(unique[k] - unique[i]);
                let length = crossed.length();
                if length <= tolerance * tolerance {
                    continue;
                }
                let normal = crossed * length.recip();
                let offset = normal.dot(unique[i]);
                let (mut above, mut below) = (false, false);
                for point in &unique {
                    let side = normal.dot(*point) - offset;
                    above |= side > tolerance;
                    below |= side < -tolerance;
                }
                if above && below {
                    continue;
                }
                let plane = if above {
                    (-normal, -offset)
                } else {
                    (normal, offset)
                };
                let known = planes.iter().any(|(other, at)| {
                    other.dot(plane.0) > 1.0 - 1e-5 && (at - plane.1).abs() <= tolerance * 4.0
                });
                if !known {
                    planes.push(plane);
                }
            }
        }
    }
    if planes.len() < 4 {
        return Err(PhysicsError::InvalidShape);
    }
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut face_starts = vec![0];
    let mut face_corners: Vec<u32> = Vec::new();
    let mut edges: Vec<Vec3> = Vec::new();
    for (normal, offset) in &planes {
        let mut corners: Vec<Vec3> = unique
            .iter()
            .filter(|point| (normal.dot(**point) - offset).abs() <= tolerance * 4.0)
            .copied()
            .collect();
        let middle = corners.iter().fold(Vec3::ZERO, |sum, point| sum + *point)
            * (corners.len() as f32).recip();
        let across = (corners[0] - middle).normalize();
        let up = normal.cross(across);
        corners.sort_by(|a, b| {
            let angle = |point: &Vec3| {
                (*point - middle)
                    .dot(up)
                    .atan2((*point - middle).dot(across))
            };
            angle(a).total_cmp(&angle(b))
        });
        // points on an edge between two corners are not corners
        let count = corners.len();
        let kept: Vec<Vec3> = (0..count)
            .filter(|at| {
                let (before, here, after) = (
                    corners[(at + count - 1) % count],
                    corners[*at],
                    corners[(at + 1) % count],
                );
                (here - before)
                    .normalize()
                    .cross((after - here).normalize())
                    .dot(*normal)
                    > 1e-4
            })
            .map(|at| corners[at])
            .collect();
        for (at, corner) in kept.iter().enumerate() {
            let index = match vertices
                .iter()
                .position(|known| (*known - *corner).length() <= tolerance)
            {
                Some(index) => index,
                None => {
                    vertices.push(*corner);
                    vertices.len() - 1
                }
            };
            face_corners.push(index as u32);
            let along = (kept[(at + 1) % kept.len()] - *corner).normalize();
            if !edges
                .iter()
                .any(|known| known.dot(along).abs() > 1.0 - 1e-5)
            {
                edges.push(along);
            }
        }
        face_starts.push(face_corners.len() as u32);
    }
    Ok(RawHull {
        vertices,
        planes,
        face_starts,
        face_corners,
        edges,
    })
}

fn add_outer(matrix: &mut [[f64; 3]; 3], scale: f64, axis: DVec3) {
    let parts = [axis.x, axis.y, axis.z];
    for (row, left) in matrix.iter_mut().zip(parts) {
        for (cell, right) in row.iter_mut().zip(parts) {
            *cell += scale * left * right;
        }
    }
}

/// Volume, center of mass and inertia about it at density 1, summed over tetrahedra from the
/// origin. Signed volumes cancel what lies outside.
fn mass_of(hull: &RawHull) -> (f64, DVec3, [[f64; 3]; 3]) {
    let mut volume = 0.0;
    let mut weighted = DVec3::ZERO;
    let mut moment = [[0.0; 3]; 3];
    for face in hull.face_starts.windows(2) {
        let corners = &hull.face_corners[face[0] as usize..face[1] as usize];
        let at = |index: u32| DVec3::from_vec3(hull.vertices[index as usize]);
        let first = at(corners[0]);
        for pair in corners[1..].windows(2) {
            let (second, third) = (at(pair[0]), at(pair[1]));
            let det = first.dot(second.cross(third));
            volume += det / 6.0;
            weighted += (first + second + third) * (det / 24.0);
            let scale = det / 120.0;
            add_outer(&mut moment, scale, first);
            add_outer(&mut moment, scale, second);
            add_outer(&mut moment, scale, third);
            add_outer(&mut moment, scale, first + second + third);
        }
    }
    let center = weighted / volume;
    add_outer(&mut moment, -volume, center);
    let trace = moment[0][0] + moment[1][1] + moment[2][2];
    let mut inertia = [[0.0; 3]; 3];
    for (row, values) in inertia.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            let diagonal = if row == column { trace } else { 0.0 };
            *value = diagonal - moment[row][column];
        }
    }
    (volume, center, inertia)
}

impl ConvexHull {
    /// Hull of raw points, plus its center and inertia in their axes.
    ///
    /// # Errors
    /// Fails on too few or too many points, non finite ones or no volume.
    pub(crate) fn measure(points: &[Vec3]) -> Result<(f64, DVec3, [[f64; 3]; 3]), PhysicsError> {
        let hull = build_hull(points)?;
        let measured = mass_of(&hull);
        if !measured.0.is_finite() || measured.0 <= 1e-12 {
            return Err(PhysicsError::InvalidShape);
        }
        Ok(measured)
    }

    /// `points` must already be centered in principal axes. Snapshot loading comes through here.
    ///
    /// # Errors
    /// Fails like [`Self::measure`], and on a bad `unit`.
    pub(crate) fn from_rebased(
        points: &[Vec3],
        unit: MassProperties,
    ) -> Result<Self, PhysicsError> {
        let hull = build_hull(points)?;
        if !unit.mass.is_finite()
            || unit.mass <= 0.0
            || !unit.inertia.is_finite()
            || unit.inertia.min_element() <= 0.0
        {
            return Err(PhysicsError::InvalidShape);
        }
        let mut low = Vec3::splat(f32::INFINITY);
        let mut high = Vec3::splat(f32::NEG_INFINITY);
        let mut radius = 0.0_f32;
        for vertex in &hull.vertices {
            low = low.min(*vertex);
            high = high.max(*vertex);
            radius = radius.max(vertex.length());
        }
        Ok(Self {
            source: points.to_vec(),
            vertices: hull.vertices,
            planes: hull.planes,
            face_starts: hull.face_starts,
            face_corners: hull.face_corners,
            edges: hull.edges,
            unit,
            radius,
            box_center: (low + high) * 0.5,
            box_half: (high - low) * 0.5,
        })
    }

    /// Each outline edge once, in hull axes.
    pub(crate) fn outline(&self, out: &mut Vec<(Vec3, Vec3)>) {
        out.clear();
        for face in self.face_starts.windows(2) {
            let corners = &self.face_corners[face[0] as usize..face[1] as usize];
            for (at, first) in corners.iter().enumerate() {
                let second = corners[(at + 1) % corners.len()];
                // every edge is on two faces, keep the one where it runs low to high
                if *first < second {
                    out.push((
                        self.vertices[*first as usize],
                        self.vertices[second as usize],
                    ));
                }
            }
        }
    }

    pub(crate) fn hull(&self, rotation: Quat, center: DVec3) -> Hull<'_> {
        Hull::Points {
            vertices: &self.vertices,
            rotation: DQuat::from_quat(rotation),
            center,
        }
    }

    /// Ray in hull axes. A start inside gives time 0 and a normal against the ray.
    pub(crate) fn ray(&self, origin: DVec3, direction: DVec3, max: f64) -> Option<(f64, DVec3)> {
        let (mut enter, mut leave) = (0.0_f64, max);
        let mut normal = -direction;
        for (plane, offset) in &self.planes {
            let plane = DVec3::from_vec3(*plane);
            let room = f64::from(*offset) - plane.dot(origin);
            let along = plane.dot(direction);
            if along.abs() < 1e-14 {
                if room < 0.0 {
                    return None;
                }
            } else if along < 0.0 {
                let time = room / along;
                if time > enter {
                    enter = time;
                    normal = plane;
                }
            } else {
                leave = leave.min(room / along);
            }
            if enter > leave {
                return None;
            }
        }
        Some((enter, normal))
    }

    /// Signed distance from a point in hull axes and the outward normal.
    pub(crate) fn point_distance(&self, point: DVec3) -> (f64, DVec3) {
        let near = distance(&Hull::Point(point), &self.hull(Quat::IDENTITY, DVec3::ZERO));
        if near.direction != DVec3::ZERO {
            return (near.distance, near.direction);
        }
        let mut gap = f64::NEG_INFINITY;
        let mut normal = DVec3::Y;
        for (plane, offset) in &self.planes {
            let plane = DVec3::from_vec3(*plane);
            let side = plane.dot(point) - f64::from(*offset);
            if side > gap {
                gap = side;
                normal = plane;
            }
        }
        (gap, normal)
    }
}

/// A polytope in world axes, on the stack so a pair allocates nothing.
pub(crate) struct Poly<'a> {
    vertices: [Vec3; MAX_CONVEX_POINTS],
    vertex_count: usize,
    planes: [(Vec3, f32); MAX_FACES],
    plane_count: usize,
    edges: [Vec3; MAX_EDGES],
    edge_count: usize,
    face_starts: &'a [u32],
    face_corners: &'a [u32],
    pub(crate) center: Vec3,
}

const BOX_STARTS: [u32; 7] = [0, 4, 8, 12, 16, 20, 24];
const BOX_CORNERS: [u32; 24] = [
    1, 3, 7, 5, // +x
    0, 4, 6, 2, // -x
    2, 6, 7, 3, // +y
    0, 1, 5, 4, // -y
    4, 5, 7, 6, // +z
    0, 2, 3, 1, // -z
];

impl Poly<'static> {
    pub(crate) fn of_box(half: Vec3, rotation: Quat, center: Vec3) -> Self {
        let axes = [
            rotation.rotate_vec3(Vec3::X),
            rotation.rotate_vec3(Vec3::Y),
            rotation.rotate_vec3(Vec3::Z),
        ];
        let mut poly = Self::empty(&BOX_STARTS, &BOX_CORNERS, center);
        for index in 0..8 {
            let side = |bit: usize| {
                if index >> bit & 1 == 1 { 1.0 } else { -1.0 }
            };
            poly.vertices[index] = center
                + axes[0] * (side(0) * half.x)
                + axes[1] * (side(1) * half.y)
                + axes[2] * (side(2) * half.z);
        }
        poly.vertex_count = 8;
        let extents = [half.x, half.y, half.z];
        for (axis_index, axis) in axes.iter().enumerate() {
            let middle = axis.dot(center);
            poly.planes[axis_index * 2] = (*axis, middle + extents[axis_index]);
            poly.planes[axis_index * 2 + 1] = (-*axis, extents[axis_index] - middle);
            poly.edges[axis_index] = *axis;
        }
        poly.plane_count = 6;
        poly.edge_count = 3;
        poly
    }
}

impl<'a> Poly<'a> {
    fn empty(face_starts: &'a [u32], face_corners: &'a [u32], center: Vec3) -> Self {
        Self {
            vertices: [Vec3::ZERO; MAX_CONVEX_POINTS],
            vertex_count: 0,
            planes: [(Vec3::ZERO, 0.0); MAX_FACES],
            plane_count: 0,
            edges: [Vec3::ZERO; MAX_EDGES],
            edge_count: 0,
            face_starts,
            face_corners,
            center,
        }
    }

    pub(crate) fn of_hull(hull: &'a ConvexHull, rotation: Quat, center: Vec3) -> Self {
        let mut poly = Self::empty(&hull.face_starts, &hull.face_corners, center);
        poly.vertex_count = hull.vertices.len().min(MAX_CONVEX_POINTS);
        for (out, vertex) in poly.vertices.iter_mut().zip(&hull.vertices) {
            *out = center + rotation.rotate_vec3(*vertex);
        }
        poly.plane_count = hull.planes.len().min(MAX_FACES);
        for (out, (normal, offset)) in poly.planes.iter_mut().zip(&hull.planes) {
            let normal = rotation.rotate_vec3(*normal);
            *out = (normal, *offset + normal.dot(center));
        }
        poly.edge_count = hull.edges.len().min(MAX_EDGES);
        for (out, edge) in poly.edges.iter_mut().zip(&hull.edges) {
            *out = rotation.rotate_vec3(*edge);
        }
        poly
    }

    pub(crate) fn vertices(&self) -> &[Vec3] {
        &self.vertices[..self.vertex_count]
    }

    pub(crate) fn planes(&self) -> &[(Vec3, f32)] {
        &self.planes[..self.plane_count]
    }

    pub(crate) fn edges(&self) -> &[Vec3] {
        &self.edges[..self.edge_count]
    }

    fn face(&self, index: usize) -> &[u32] {
        &self.face_corners[self.face_starts[index] as usize..self.face_starts[index + 1] as usize]
    }

    /// Furthest reach from the center along `direction`.
    pub(crate) fn reach(&self, direction: Vec3) -> f32 {
        self.vertices()
            .iter()
            .map(|vertex| (*vertex - self.center).dot(direction))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    fn lowest_along(&self, direction: Vec3) -> f32 {
        self.vertices()
            .iter()
            .map(|vertex| vertex.dot(direction))
            .fold(f32::INFINITY, f32::min)
    }

    fn support(&self, direction: Vec3) -> Vec3 {
        let mut best = self.center;
        let mut most = f32::NEG_INFINITY;
        for vertex in self.vertices() {
            let value = vertex.dot(direction);
            if value > most {
                most = value;
                best = *vertex;
            }
        }
        best
    }

    /// Height of the surface over `point`, walking along `up`. Negative when `point` is inside.
    /// `None` when that line misses.
    pub(crate) fn surface_above(&self, point: Vec3, up: Vec3) -> Option<f32> {
        let (mut enter, mut leave) = (f32::NEG_INFINITY, f32::INFINITY);
        for (normal, offset) in self.planes() {
            let room = *offset - normal.dot(point);
            let along = normal.dot(up);
            if along.abs() < 1e-6 {
                if room < 0.0 {
                    return None;
                }
            } else if along < 0.0 {
                enter = enter.max(room / along);
            } else {
                leave = leave.min(room / along);
            }
        }
        (enter <= leave && leave >= 0.0).then_some(enter)
    }
}

/// Keeps the side where `point . axis <= limit`.
fn clip(polygon: &mut [Vec3; MAX_POLYGON], count: usize, axis: Vec3, limit: f32) -> usize {
    let input = *polygon;
    let mut out = 0;
    for i in 0..count {
        let current = input[i];
        let next = input[(i + 1) % count];
        let current_gap = current.dot(axis) - limit;
        let next_gap = next.dot(axis) - limit;
        if current_gap <= 0.0 && out < MAX_POLYGON {
            polygon[out] = current;
            out += 1;
        }
        if (current_gap < 0.0) != (next_gap < 0.0) && current_gap != next_gap && out < MAX_POLYGON {
            polygon[out] = current + (next - current) * (current_gap / (current_gap - next_gap));
            out += 1;
        }
    }
    out
}

/// Clips the incident face of `other` to reference face `index`. `flip` when reference is B.
fn face_contact(
    reference: &Poly<'_>,
    other: &Poly<'_>,
    index: usize,
    flip: bool,
    margin: f32,
) -> Option<RawManifold> {
    let (outward, offset) = reference.planes()[index];
    let mut incident = 0;
    let mut lowest = f32::INFINITY;
    for (at, (normal, _)) in other.planes().iter().enumerate() {
        let facing = normal.dot(outward);
        if facing < lowest {
            lowest = facing;
            incident = at;
        }
    }
    let mut polygon = [Vec3::ZERO; MAX_POLYGON];
    let mut count = 0;
    for corner in other.face(incident) {
        if count < MAX_POLYGON {
            polygon[count] = other.vertices()[*corner as usize];
            count += 1;
        }
    }
    let corners = reference.face(index);
    for (at, corner) in corners.iter().enumerate() {
        let start = reference.vertices()[*corner as usize];
        let end = reference.vertices()[corners[(at + 1) % corners.len()] as usize];
        let side = (end - start).cross(outward);
        let length = side.length();
        if length < 1e-7 {
            continue;
        }
        let side = side * length.recip();
        count = clip(&mut polygon, count, side, side.dot(start));
        if count == 0 {
            return None;
        }
    }
    let mut kept = [(Vec3::ZERO, 0.0_f32); 16];
    let mut kept_count = 0;
    for point in &polygon[..count] {
        let depth = offset - point.dot(outward);
        if depth >= -margin && kept_count < kept.len() {
            kept[kept_count] = (*point + outward * (depth * 0.5), depth);
            kept_count += 1;
        }
    }
    let normal = if flip { -outward } else { outward };
    let mut result = RawManifold::new(normal);
    result.reduce_from(&kept[..kept_count]);
    result.filled()
}

/// Separating axes over both sets of faces and every pair of edges. Normal points from A to B.
pub(crate) fn poly_poly(a: &Poly<'_>, b: &Poly<'_>, margin: f32) -> Option<RawManifold> {
    let mut best_face = f32::NEG_INFINITY;
    let mut face_of_a = true;
    let mut face_index = 0;
    for (index, (normal, offset)) in a.planes().iter().enumerate() {
        let gap = b.lowest_along(*normal) - *offset;
        if gap > margin {
            return None;
        }
        if gap > best_face {
            best_face = gap;
            face_index = index;
        }
    }
    for (index, (normal, offset)) in b.planes().iter().enumerate() {
        let gap = a.lowest_along(*normal) - *offset;
        if gap > margin {
            return None;
        }
        if gap > best_face + EDGE_BIAS {
            best_face = gap;
            face_index = index;
            face_of_a = false;
        }
    }
    let between = b.center - a.center;
    let mut best_edge = best_face;
    let mut edge_normal = None;
    for edge_a in a.edges() {
        for edge_b in b.edges() {
            let crossed = edge_a.cross(*edge_b);
            let length = crossed.length();
            if length < 1e-3 {
                continue;
            }
            let mut axis = crossed * length.recip();
            if axis.dot(between) < 0.0 {
                axis = -axis;
            }
            let gap = b.lowest_along(axis) + a.lowest_along(-axis);
            if gap > margin {
                return None;
            }
            // faces win ties, they give full manifolds
            if gap > best_edge + EDGE_BIAS {
                best_edge = gap;
                edge_normal = Some(axis);
            }
        }
    }
    if let Some(normal) = edge_normal {
        let mut result = RawManifold::new(normal);
        let on_a = a.support(normal);
        let on_b = b.support(-normal);
        result.push((on_a + on_b) * 0.5, -best_edge);
        return Some(result);
    }
    if face_of_a {
        face_contact(a, b, face_index, false, margin)
    } else {
        face_contact(b, a, face_index, true, margin)
    }
}

/// A sphere or capsule, as the hull's segment and radius, against a convex shape.
/// Normal points from the poly to the rounded shape, positions are as given.
pub(crate) fn poly_rounded(
    poly_hull: &Hull<'_>,
    poly: &Poly<'_>,
    (start, end): (Vec3, Vec3),
    radius: f32,
    margin: f32,
) -> Option<RawManifold> {
    let core = if start == end {
        Hull::Point(DVec3::from_vec3(start))
    } else {
        Hull::Segment(DVec3::from_vec3(start), DVec3::from_vec3(end))
    };
    let near = distance(&core, poly_hull);
    let reach = radius + margin;
    if near.direction != DVec3::ZERO {
        if near.distance >= f64::from(reach) {
            return None;
        }
        let normal = near.direction.as_vec3().normalize();
        let on_poly = near.on_b.as_vec3();
        let gap = near.distance as f32;
        let mut result = RawManifold::new(normal);
        let depth = radius - gap;
        result.push(on_poly + normal * (depth * -0.5), depth);
        if start != end {
            for tip in [start, end] {
                let tip_near = distance(&Hull::Point(DVec3::from_vec3(tip)), poly_hull);
                if tip_near.direction == DVec3::ZERO {
                    continue;
                }
                let same_way = tip_near.direction.as_vec3().dot(normal) > 0.9;
                let tip_gap = (tip - tip_near.on_b.as_vec3()).dot(normal);
                let tip_depth = radius - tip_gap;
                let apart = (tip_near.on_b.as_vec3() - on_poly).length_squared() > 1e-6;
                if same_way && apart && tip_depth > -margin {
                    result.push(
                        tip_near.on_b.as_vec3() + normal * (tip_depth * -0.5),
                        tip_depth,
                    );
                }
            }
        }
        return Some(result);
    }
    // the core is inside, push out the nearest face
    let middle = (start + end) * 0.5;
    let mut gap = f32::NEG_INFINITY;
    let mut normal = Vec3::Y;
    for (plane, offset) in poly.planes() {
        let side = plane.dot(middle) - *offset;
        if side > gap {
            gap = side;
            normal = *plane;
        }
    }
    let depth = radius - gap;
    let mut result = RawManifold::new(normal);
    result.push(middle - normal * (gap + depth * 0.5), depth);
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConvexInfo, GridPosition, PhysicsSettings, PhysicsWorld, QueryFilter, Ray, RigidBodyDesc,
        Shape, ShapeCast,
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

    fn cube_points(half: f32) -> Vec<Vec3> {
        let mut points = Vec::new();
        for x in [-half, half] {
            for y in [-half, half] {
                for z in [-half, half] {
                    points.push(Vec3::new(x, y, z));
                }
            }
        }
        points
    }

    fn tetra() -> Vec<Vec3> {
        vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z]
    }

    fn octa(half: f32) -> Vec<Vec3> {
        vec![
            Vec3::X * half,
            -Vec3::X * half,
            Vec3::Y * half,
            -Vec3::Y * half,
            Vec3::Z * half,
            -Vec3::Z * half,
        ]
    }

    fn ground(world: &mut PhysicsWorld) {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(50.0, 1.0, 50.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
    }

    fn drop_on(world: &mut PhysicsWorld, info: &ConvexInfo, x: f32, y: f32) -> crate::BodyHandle {
        let (place, turn) = info.placed(Vec3::new(x, y, 0.0), Quat::IDENTITY);
        world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap()
    }

    #[test]
    fn cube_hull_has_six_faces_and_three_edges() {
        let hull = build_hull(&cube_points(1.0)).unwrap();
        assert_eq!(hull.planes.len(), 6);
        assert_eq!(hull.vertices.len(), 8);
        assert_eq!(hull.edges.len(), 3);
        assert_eq!(hull.face_corners.len(), 24);
    }

    #[test]
    fn points_inside_and_on_edges_are_dropped() {
        let mut points = cube_points(1.0);
        points.push(Vec3::ZERO);
        points.push(Vec3::new(0.2, 0.1, -0.3));
        points.push(Vec3::new(1.0, 1.0, 0.0));
        points.push(Vec3::new(1.0, 0.3, -0.2));
        points.push(Vec3::new(1.0, 1.0, 1.0));
        let hull = build_hull(&points).unwrap();
        assert_eq!(hull.vertices.len(), 8);
        assert_eq!(hull.planes.len(), 6);
    }

    #[test]
    fn faces_wind_counter_clockwise_from_outside() {
        for points in [cube_points(1.0), tetra(), octa(1.5)] {
            let hull = build_hull(&points).unwrap();
            for (index, face) in hull.face_starts.windows(2).enumerate() {
                let corners = &hull.face_corners[face[0] as usize..face[1] as usize];
                let p = |at: usize| hull.vertices[corners[at] as usize];
                let normal = (p(1) - p(0)).cross(p(2) - p(0));
                assert!(normal.dot(hull.planes[index].0) > 0.0, "face {index}");
            }
        }
    }

    #[test]
    fn mass_matches_textbook_shapes() {
        let (volume, center, inertia) = ConvexHull::measure(&cube_points(1.0)).unwrap();
        assert!((volume - 8.0).abs() < 1e-5);
        assert!(center.length() < 1e-5);
        // m (a^2 + b^2) / 12 with side 2
        assert!(
            (inertia[0][0] - 8.0 * 8.0 / 12.0).abs() < 1e-4,
            "{inertia:?}"
        );
        assert!(inertia[0][1].abs() < 1e-4);
        // tetrahedron with unit legs: volume 1/6, center at 1/4
        let (volume, center, inertia) = ConvexHull::measure(&tetra()).unwrap();
        assert!((volume - 1.0 / 6.0).abs() < 1e-6);
        assert!((center.x - 0.25).abs() < 1e-6);
        // about the center, from the known 1/60 and 1/120 terms
        let expected = 1.0 / 60.0 * 2.0 - 2.0 * (1.0 / 6.0) * 0.0625;
        assert!((inertia[0][0] - expected).abs() < 1e-6, "{inertia:?}");
    }

    #[test]
    fn bad_points_fail() {
        let mut world = world();
        let nan = f32::NAN;
        let flat = vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::X + Vec3::Z];
        let line = vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0, Vec3::X * 3.0];
        let same = vec![Vec3::ZERO; 8];
        let lots = vec![Vec3::ONE; MAX_CONVEX_POINTS + 1];
        for points in [
            vec![],
            vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            flat,
            line,
            same,
            lots,
            vec![Vec3::new(nan, 0.0, 0.0), Vec3::X, Vec3::Y, Vec3::Z],
        ] {
            assert_eq!(
                world.add_convex(&points),
                Err(PhysicsError::InvalidShape),
                "{points:?}"
            );
        }
    }

    #[test]
    fn hull_cube_matches_a_real_box_on_mass_and_rest() {
        let mut world = world();
        ground(&mut world);
        let info = world.add_convex(&cube_points(0.5)).unwrap();
        let hull_body = drop_on(&mut world, &info, -3.0, 0.5);
        let plain = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(3.0, 0.5, 0.0)),
            )
            .unwrap();
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        let hull_state = world.body_state(hull_body).unwrap();
        let plain_state = world.body_state(plain).unwrap();
        near(
            hull_state.position.local.y,
            plain_state.position.local.y,
            0.01,
        );
        assert!(hull_state.asleep);
    }

    #[test]
    fn hull_cube_slides_like_a_real_box() {
        let mut world = world();
        ground(&mut world);
        let info = world.add_convex(&cube_points(0.5)).unwrap();
        let (place, turn) = info.placed(Vec3::new(0.0, 0.5, -5.0), Quat::IDENTITY);
        let hull_body = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(place)
                    .rotation(turn)
                    .linear_velocity(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        let expected = 25.0 / 9.81;
        let state = world.body_state(hull_body).unwrap();
        assert!(
            (state.position.local.x - expected).abs() < expected * 0.15,
            "{state:?}"
        );
    }

    #[test]
    fn tetra_tips_onto_a_face_and_rests_at_its_height() {
        let mut world = world();
        ground(&mut world);
        let info = world.add_convex(&tetra()).unwrap();
        let body = drop_on(&mut world, &info, 0.0, 0.5);
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        let state = world.body_state(body).unwrap();
        // leg faces leave the center 1/4 up, the slanted face a quarter of its height
        let low = 0.25;
        let slanted = 0.25 / 3.0_f32.sqrt();
        let y = state.position.local.y;
        assert!(
            (y - low).abs() < 0.02 || (y - slanted).abs() < 0.02,
            "{state:?}"
        );
        assert!(state.linear_velocity.length() < 0.1, "{state:?}");
    }

    #[test]
    fn stack_of_hulls_stays_up() {
        let mut world = world();
        ground(&mut world);
        let info = world.add_convex(&cube_points(0.5)).unwrap();
        let boxes: Vec<_> = (0..5)
            .map(|level| drop_on(&mut world, &info, 0.0, 0.5 + level as f32))
            .collect();
        for _ in 0..300 {
            world.step_fixed().unwrap();
        }
        for (level, body) in boxes.iter().enumerate() {
            let at = world.body_state(*body).unwrap().position.local;
            near(at.y, 0.5 + level as f32, 0.05);
            assert!(at.x.hypot(at.z) < 0.05, "{level} {at:?}");
        }
    }

    #[test]
    fn hull_meets_sphere_capsule_box_and_itself() {
        let mut world = world();
        let info = world.add_convex(&cube_points(1.0)).unwrap();
        let (place, turn) = info.placed(Vec3::ZERO, Quat::IDENTITY);
        world
            .spawn(
                RigidBodyDesc::fixed(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .local_position(Vec3::new(0.2, 3.0, 0.1)),
            )
            .unwrap();
        for _ in 0..180 {
            world.step_fixed().unwrap();
        }
        near(world.body_state(ball).unwrap().position.local.y, 1.25, 0.01);

        let mut world = self::world();
        let info = world.add_convex(&cube_points(1.0)).unwrap();
        let (place, turn) = info.placed(Vec3::ZERO, Quat::IDENTITY);
        world
            .spawn(
                RigidBodyDesc::fixed(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        let lying = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let capsule = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::capsule(0.1, 0.3))
                    .rotation(lying)
                    .local_position(Vec3::new(0.0, 1.5, 0.0)),
            )
            .unwrap();
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.2)))
                    .local_position(Vec3::new(0.0, 1.5, 0.7)),
            )
            .unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        let capsule = world.body_state(capsule).unwrap();
        near(capsule.position.local.y, 1.1, 0.02);
        near(capsule.rotation.rotate_vec3(Vec3::Y).y, 0.0, 0.02);
        near(world.body_state(cube).unwrap().position.local.y, 1.2, 0.02);
    }

    #[test]
    fn octahedron_balanced_on_a_box_falls_over() {
        let mut world = world();
        ground(&mut world);
        let info = world.add_convex(&octa(1.0)).unwrap();
        // on its point, tipped a little off every symmetry plane, a clean tilt about z lands on an
        // edge and balances there
        let tilt = Quat::from_axis_angle(Vec3::new(0.3, 0.0, 1.0).normalize(), 0.05);
        let (place, turn) = info.placed(Vec3::new(0.0, 1.05, 0.0), tilt);
        let body = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        let state = world.body_state(body).unwrap();
        // rests on a face, which is 1 / sqrt(3) from the center
        near(state.position.local.y, 1.0 / 3.0_f32.sqrt(), 0.03);
    }

    #[test]
    fn hull_rests_on_terrain() {
        let mut world = world();
        let heights = vec![0.0; 25];
        let terrain = world.add_heightfield(5, 5, 2.0, 2.0, &heights).unwrap();
        world
            .spawn(RigidBodyDesc::fixed(terrain).local_position(Vec3::new(-4.0, 0.0, -4.0)))
            .unwrap();
        let info = world.add_convex(&cube_points(0.5)).unwrap();
        let body = drop_on(&mut world, &info, 0.3, 1.0);
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        near(world.body_state(body).unwrap().position.local.y, 0.5, 0.02);
    }

    #[test]
    fn fast_hull_does_not_pass_a_thin_wall() {
        let mut world = PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(0.05, 5.0, 5.0)))
                    .local_position(Vec3::new(3.0, 0.0, 0.0)),
            )
            .unwrap();
        let info = world.add_convex(&octa(0.25)).unwrap();
        let (place, turn) = info.placed(Vec3::new(-20.0, 0.0, 0.0), Quat::IDENTITY);
        let body = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(place)
                    .rotation(turn)
                    .linear_velocity(Vec3::new(1000.0, 0.0, 0.0)),
            )
            .unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        let x = world.body_state(body).unwrap().position.local.x;
        assert!(x < 2.95, "{x}");
    }

    #[test]
    fn queries_see_a_hull() {
        let mut world = world();
        let info = world.add_convex(&octa(1.0)).unwrap();
        let (place, turn) = info.placed(Vec3::ZERO, Quat::IDENTITY);
        let body = world
            .spawn(
                RigidBodyDesc::fixed(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        let filter = QueryFilter::default();
        let ray = Ray {
            origin: at(0.0, 5.0, 0.0),
            direction: -Vec3::Y,
            max_distance: 100.0,
        };
        let hit = world.raycast(&ray, &filter).unwrap().unwrap();
        assert_eq!(hit.body, body);
        near(hit.distance, 4.0, 1e-3);
        let beside = Ray {
            origin: at(0.8, 5.0, 0.8),
            ..ray
        };
        assert_eq!(world.raycast(&beside, &filter).unwrap(), None);
        let inside = Ray {
            origin: at(0.1, 0.1, 0.0),
            ..ray
        };
        assert_eq!(
            world.raycast(&inside, &filter).unwrap().unwrap().distance,
            0.0
        );

        let cast = ShapeCast {
            shape: Shape::sphere(0.25),
            rotation: Quat::IDENTITY,
            origin: at(0.0, 5.0, 0.0),
            direction: -Vec3::Y,
            max_distance: 100.0,
        };
        let swept = world.shape_cast(&cast, &filter).unwrap().unwrap();
        // above the tip, the sphere reaches it at a height of 1 + 0.25
        near(swept.distance, 3.75, 2e-3);
        let hull_cast = ShapeCast {
            shape: info.shape,
            rotation: info.axes,
            ..cast
        };
        let swept = world.shape_cast(&hull_cast, &filter).unwrap().unwrap();
        // tip against tip, 5 - 1 - 1
        near(swept.distance, 3.0, 1e-2);

        let mut out = Vec::new();
        world
            .overlap(
                &Shape::sphere(0.25),
                at(0.5, 0.5, 0.0),
                Quat::IDENTITY,
                &filter,
                &mut out,
            )
            .unwrap();
        assert_eq!(out, [body]);
        world
            .overlap(
                &Shape::sphere(0.25),
                at(1.0, 1.0, 0.0),
                Quat::IDENTITY,
                &filter,
                &mut out,
            )
            .unwrap();
        assert!(out.is_empty());

        let outside = world
            .nearest_point(at(0.0, 3.0, 0.0), 100.0, &filter)
            .unwrap()
            .unwrap();
        near(outside.distance, 2.0, 1e-3);
        let within = world
            .nearest_point(at(0.0, 0.0, 0.0), 100.0, &filter)
            .unwrap()
            .unwrap();
        near(within.distance, -1.0 / 3.0_f32.sqrt(), 1e-3);
    }

    #[test]
    fn sat_agrees_with_gjk_on_random_poses() {
        let mut state = 0xabcd_ef01_2345_6789_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 1000.0 - 1.0
        };
        let a = ConvexHull::from_rebased(
            &octa(0.5),
            MassProperties {
                mass: 1.0,
                inertia: Vec3::ONE,
            },
        )
        .unwrap();
        let b = ConvexHull::from_rebased(
            &cube_points(0.4),
            MassProperties {
                mass: 1.0,
                inertia: Vec3::ONE,
            },
        )
        .unwrap();
        let (mut touching, mut apart) = (0, 0);
        for _ in 0..10_000 {
            let turn_a = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let turn_b = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let center = Vec3::new(next(), next(), next()) * 1.5;
            let separation = distance(
                &a.hull(turn_a, DVec3::ZERO),
                &b.hull(turn_b, DVec3::from_vec3(center)),
            );
            let poly_a = Poly::of_hull(&a, turn_a, Vec3::ZERO);
            let poly_b = Poly::of_hull(&b, turn_b, center);
            let hit = poly_poly(&poly_a, &poly_b, 0.0);
            if separation.distance > 1e-3 {
                assert!(hit.is_none(), "{center:?} {}", separation.distance);
                apart += 1;
            } else if separation.direction == DVec3::ZERO {
                let hit = hit.unwrap_or_else(|| panic!("{center:?} missed"));
                assert!((hit.normal.length() - 1.0).abs() < 1e-3, "{hit:?}");
                touching += 1;
            }
        }
        assert!(touching > 500 && apart > 500, "{touching} {apart}");
    }

    #[test]
    fn snapshot_with_a_hull_continues_exactly() {
        let mut world = world();
        ground(&mut world);
        let info = world.add_convex(&tetra()).unwrap();
        drop_on(&mut world, &info, 0.0, 1.0);
        for _ in 0..20 {
            world.step_fixed().unwrap();
        }
        let bytes = world.save_snapshot();
        let mut loaded = PhysicsWorld::load_snapshot(&bytes).unwrap();
        assert_eq!(loaded.save_snapshot(), bytes);
        for _ in 0..60 {
            world.step_fixed().unwrap();
            loaded.step_fixed().unwrap();
        }
        assert_eq!(loaded.state_hash(), world.state_hash());
    }
}
