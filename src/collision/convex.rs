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
    pub(crate) planes: Vec<(Vec3, f32)>,
    face_starts: Vec<u32>,
    face_corners: Vec<u32>,
    edges: Vec<Vec3>,
    edge_faces: Vec<EdgeFaces>,
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
    edge_faces: Vec<EdgeFaces>,
}

/// An edge's two corners and the two faces it joins.
pub(crate) type EdgeFaces = [u32; 4];

/// Each edge once with its two faces, empty when one is not on exactly two, as on a hull with a
/// sliver face. The separating axis test then falls back to every pair of edge directions.
fn pair_edges(face_starts: &[u32], face_corners: &[u32]) -> Vec<EdgeFaces> {
    let mut sides: Vec<(u32, u32, u32)> = Vec::new();
    for (face, range) in face_starts.windows(2).enumerate() {
        let corners = &face_corners[range[0] as usize..range[1] as usize];
        for (at, first) in corners.iter().enumerate() {
            let second = corners[(at + 1) % corners.len()];
            sides.push(((*first).min(second), (*first).max(second), face as u32));
        }
    }
    sides.sort_unstable();
    let mut paired = Vec::with_capacity(sides.len() / 2);
    for pair in sides.chunks(2) {
        match pair {
            [one, two] if one.0 == two.0 && one.1 == two.1 => {
                paired.push([one.0, one.1, one.2, two.2]);
            }
            _ => return Vec::new(),
        }
    }
    paired
}

/// The faces, each a normal and its corners in order round it, counter clockwise from outside,
/// as indexes into `points`. A face is the points within the band of a supporting plane; two
/// planes facing the same way whose points share three or more are one face a hair apart and
/// become one, so no sliver face overlaps its neighbour and none is lost.
fn face_rings(points: &[Vec3], planes: &[(Vec3, f32)], tolerance: f32) -> Vec<(Vec3, Vec<usize>)> {
    let band = tolerance * 4.0;
    let mut groups: Vec<(Vec3, Vec<usize>)> = planes
        .iter()
        .filter_map(|(normal, offset)| {
            let set: Vec<usize> = (0..points.len())
                .filter(|i| (normal.dot(points[*i]) - offset).abs() <= band)
                .collect();
            (set.len() >= 3).then_some((*normal, set))
        })
        .collect();
    let shared = |a: &[usize], b: &[usize]| a.iter().filter(|i| b.binary_search(i).is_ok()).count();
    let mut merged = true;
    while merged {
        merged = false;
        'outer: for a in 0..groups.len() {
            for b in a + 1..groups.len() {
                // three points on one edge are no plane: the same face also faces the same way
                if groups[a].0.dot(groups[b].0) > 1.0 - 1e-3
                    && shared(&groups[a].1, &groups[b].1) >= 3
                {
                    let (keep, gone) = if groups[a].1.len() >= groups[b].1.len() {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    let mut set = groups[gone].1.clone();
                    set.extend_from_slice(&groups[keep].1);
                    set.sort_unstable();
                    set.dedup();
                    groups[keep].1 = set;
                    groups.remove(gone);
                    merged = true;
                    break 'outer;
                }
            }
        }
    }
    groups
        .into_iter()
        .filter_map(|(normal, set)| {
            let ring = convex_ring(points, &set, normal);
            (ring.len() >= 3).then_some((normal, ring))
        })
        .collect()
}

/// The corners of `set` as a convex ring in the plane of `normal`, counter clockwise seen from
/// outside, points along a side left out.
fn convex_ring(points: &[Vec3], set: &[usize], normal: Vec3) -> Vec<usize> {
    let across = if normal.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let u = normal.cross(across).normalize();
    let v = normal.cross(u);
    let mut flat: Vec<(f32, f32, usize)> = set
        .iter()
        .map(|i| (points[*i].dot(u), points[*i].dot(v), *i))
        .collect();
    flat.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let turn = |o: &(f32, f32, usize), a: &(f32, f32, usize), b: &(f32, f32, usize)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    // the size of the face sets what counts as a turn, so points along a side drop out
    let size = flat
        .iter()
        .fold(0.0_f32, |most, p| most.max(p.0.abs()).max(p.1.abs()))
        .max(1e-6);
    let straight = size * size * 1e-6;
    let mut hull: Vec<(f32, f32, usize)> = Vec::with_capacity(flat.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let order: Vec<&(f32, f32, usize)> = if pass == 0 {
            flat.iter().collect()
        } else {
            flat.iter().rev().collect()
        };
        for point in order {
            while hull.len() >= start + 2
                && turn(&hull[hull.len() - 2], &hull[hull.len() - 1], point) <= straight
            {
                hull.pop();
            }
            hull.push(*point);
        }
        hull.pop();
    }
    hull.into_iter().map(|p| p.2).collect()
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
    let mut vertex_of = vec![u32::MAX; unique.len()];
    let mut face_starts = vec![0];
    let mut face_corners: Vec<u32> = Vec::new();
    let mut edges: Vec<Vec3> = Vec::new();
    let mut faces: Vec<(Vec3, f32)> = Vec::with_capacity(planes.len());
    for (normal, ring) in face_rings(&unique, &planes, tolerance) {
        let offset = ring
            .iter()
            .map(|i| normal.dot(unique[*i]))
            .fold(f32::NEG_INFINITY, f32::max);
        faces.push((normal, offset));
        for (at, &point) in ring.iter().enumerate() {
            if vertex_of[point] == u32::MAX {
                vertex_of[point] = vertices.len() as u32;
                vertices.push(unique[point]);
            }
            face_corners.push(vertex_of[point]);
            let along = (unique[ring[(at + 1) % ring.len()]] - unique[point]).normalize();
            if !edges
                .iter()
                .any(|known| known.dot(along).abs() > 1.0 - 1e-5)
            {
                edges.push(along);
            }
        }
        face_starts.push(face_corners.len() as u32);
    }
    if faces.len() < 4 {
        return Err(PhysicsError::InvalidShape);
    }
    let planes = faces;
    let edge_faces = pair_edges(&face_starts, &face_corners);
    Ok(RawHull {
        vertices,
        planes,
        face_starts,
        face_corners,
        edges,
        edge_faces,
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
            edge_faces: hull.edge_faces,
            unit,
            radius,
            box_center: (low + high) * 0.5,
            box_half: (high - low) * 0.5,
        })
    }

    /// Each outline edge once, in hull axes.
    /// Each face as a fan of triangles.
    pub(crate) fn triangles(&self, out: &mut Vec<[Vec3; 3]>) {
        out.clear();
        for face in self.face_starts.windows(2) {
            let corners = &self.face_corners[face[0] as usize..face[1] as usize];
            let Some((first, rest)) = corners.split_first() else {
                continue;
            };
            for pair in rest.windows(2) {
                out.push([
                    self.vertices[*first as usize],
                    self.vertices[pair[0] as usize],
                    self.vertices[pair[1] as usize],
                ]);
            }
        }
    }

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
    edge_faces: &'a [EdgeFaces],
    pub(crate) center: Vec3,
}

/// Corners by bits x 1, y 2, z 4, faces +x -x +y -y +z -z.
const BOX_EDGE_FACES: [EdgeFaces; 12] = [
    [0, 1, 3, 5],
    [2, 3, 2, 5],
    [4, 5, 3, 4],
    [6, 7, 2, 4],
    [0, 2, 1, 5],
    [1, 3, 0, 5],
    [4, 6, 1, 4],
    [5, 7, 0, 4],
    [0, 4, 1, 3],
    [1, 5, 0, 3],
    [2, 6, 1, 2],
    [3, 7, 0, 2],
];

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
        let mut poly = Self::empty(&BOX_STARTS, &BOX_CORNERS, &BOX_EDGE_FACES, center);
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
    fn empty(
        face_starts: &'a [u32],
        face_corners: &'a [u32],
        edge_faces: &'a [EdgeFaces],
        center: Vec3,
    ) -> Self {
        Self {
            vertices: [Vec3::ZERO; MAX_CONVEX_POINTS],
            vertex_count: 0,
            planes: [(Vec3::ZERO, 0.0); MAX_FACES],
            plane_count: 0,
            edges: [Vec3::ZERO; MAX_EDGES],
            edge_count: 0,
            face_starts,
            face_corners,
            edge_faces,
            center,
        }
    }

    pub(crate) fn of_hull(hull: &'a ConvexHull, rotation: Quat, center: Vec3) -> Self {
        let mut poly = Self::empty(
            &hull.face_starts,
            &hull.face_corners,
            &hull.edge_faces,
            center,
        );
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
    let along = reference.vertices()[corners[1 % corners.len()] as usize]
        - reference.vertices()[corners[0] as usize];
    result.reduce_from(&kept[..kept_count], along);
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
    // touching or in each other, no axis gives more than 0, and an edge axis must beat the best
    // face by the bias: none can, so the edges are not tried. apart by less than the margin it
    // stays a speculative face contact. that is pieces packed face to face, 256 shards of a broken
    // cube spent 20 of their 21 us a pair in the edges
    let (best_edge, edge_normal) = if best_face > -EDGE_BIAS {
        (best_face, None)
    } else if a.edge_faces.is_empty() || b.edge_faces.is_empty() {
        every_edge_pair(a, b, best_face, margin)?
    } else {
        minkowski_edges(a, b, best_face, margin)?
    };
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

/// The edge pair axes of two hulls, only for pairs whose faces' normals make a face of their
/// Minkowski difference (their arcs on the unit sphere cross), each measured from the edges' own
/// corners: any other pair's axis is never the deepest. The best gap past `best_face` and its
/// axis, or none when an axis separates them by more than `margin`.
fn minkowski_edges(
    a: &Poly<'_>,
    b: &Poly<'_>,
    best_face: f32,
    margin: f32,
) -> Option<(f32, Option<Vec3>)> {
    let mut best_edge = best_face;
    let mut edge_normal = None;
    // b's arcs once, not once for each of a's edges: b's normals turned round, so its faces meet
    // a's in the difference. two dots throw most pairs out, the other two only after
    let mut arcs = [(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO); MAX_EDGES];
    let count = b.edge_faces.len().min(MAX_EDGES);
    for (arc, edge_b) in arcs.iter_mut().zip(b.edge_faces) {
        let face_b = -b.planes[edge_b[2] as usize].0;
        let other_b = -b.planes[edge_b[3] as usize].0;
        *arc = (face_b, other_b, other_b.cross(face_b));
    }
    for edge_a in a.edge_faces {
        let start_a = a.vertices[edge_a[0] as usize];
        let along_a = a.vertices[edge_a[1] as usize] - start_a;
        let face_a = a.planes[edge_a[2] as usize].0;
        let other_a = a.planes[edge_a[3] as usize].0;
        let across_a = other_a.cross(face_a);
        for (edge_b, (face_b, other_b, across_b)) in b.edge_faces.iter().zip(&arcs[..count]) {
            let (cba, dba) = (face_b.dot(across_a), other_b.dot(across_a));
            if cba * dba >= 0.0 {
                continue;
            }
            let (adc, bdc) = (face_a.dot(*across_b), other_a.dot(*across_b));
            if adc * bdc >= 0.0 || cba * bdc <= 0.0 {
                continue;
            }
            let start_b = b.vertices[edge_b[0] as usize];
            let along_b = b.vertices[edge_b[1] as usize] - start_b;
            let crossed = along_a.cross(along_b);
            let length = crossed.length();
            if length < 1e-3 * along_a.length() * along_b.length() {
                continue;
            }
            let mut axis = crossed * length.recip();
            if axis.dot(start_a - a.center) < 0.0 {
                axis = -axis;
            }
            let gap = axis.dot(start_b - start_a);
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
    Some((best_edge, edge_normal))
}

/// Every pair of edge directions, each axis measured over all corners.
fn every_edge_pair(
    a: &Poly<'_>,
    b: &Poly<'_>,
    best_face: f32,
    margin: f32,
) -> Option<(f32, Option<Vec3>)> {
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
    Some((best_edge, edge_normal))
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

    #[test]
    fn a_rock_heap_on_hills_goes_to_sleep() {
        use crate::{Base, SdfDesc, SdfTerrain};
        let mut world = world();
        SdfTerrain::new(
            &mut world,
            SdfDesc {
                origin: crate::GridPosition::ORIGIN,
                voxel: 0.5,
                brick_cells: 16,
                bricks: [1, 1, 1],
                base: Base::Hills {
                    seed: 5,
                    height: 2.0,
                    amplitude: 1.0,
                    wavelength: 6.0,
                    octaves: 2,
                },
                material: crate::Material::default(),
            },
        )
        .unwrap();
        let pattern = crate::voronoi_box(Vec3::splat(1.0), 40, 9).unwrap();
        let mut pieces = Vec::new();
        for piece in &pattern.pieces {
            let middle =
                piece.points.iter().fold(Vec3::ZERO, |s, p| s + *p) / piece.points.len() as f32;
            let local: Vec<Vec3> = piece.points.iter().map(|p| (*p - middle) * 0.9).collect();
            let info = world.add_convex(&local).unwrap();
            let (place, turn) =
                info.placed(Vec3::new(4.0, 5.0, 4.0) + middle * 1.1, Quat::IDENTITY);
            pieces.push(
                world
                    .spawn(
                        RigidBodyDesc::dynamic(info.shape)
                            .density(2700.0)
                            .local_position(place)
                            .rotation(turn),
                    )
                    .unwrap(),
            );
        }
        // it slumped and rocked for over a minute: points jumping on the mesh dropped their load,
        // and small pieces rocking in place counted as turning
        for _ in 0..60 * 30 {
            world.step_fixed().unwrap();
        }
        let awake = pieces
            .iter()
            .filter(|p| !world.bodies.asleep[world.bodies.dense_index(**p).unwrap()])
            .count();
        assert_eq!(awake, 0);
    }

    #[test]
    fn voronoi_pieces_close_into_hulls() {
        let (mut open, mut total) = (0, 0);
        for seed in 0..8u64 {
            let pattern = crate::voronoi_box(Vec3::splat(2.0), 200, seed).unwrap();
            for piece in &pattern.pieces {
                let middle =
                    piece.points.iter().fold(Vec3::ZERO, |s, p| s + *p) / piece.points.len() as f32;
                let local: Vec<Vec3> = piece.points.iter().map(|p| *p - middle).collect();
                let Ok(hull) = build_hull(&local) else {
                    continue;
                };
                total += 1;
                if hull.edge_faces.is_empty() {
                    open += 1;
                }
            }
        }
        assert!(total > 1000);
        assert!(open * 50 < total, "{open} of {total} open");
    }

    #[test]
    fn near_flat_point_sets_give_a_hull_or_an_error_never_a_panic() {
        let mut seed = 0x2545_f491_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32 - 0.5
        };
        for _ in 0..20_000 {
            let count = 4 + (next().abs() * 40.0) as usize;
            let flat = 10.0_f32.powf(-1.0 - 6.0 * (next() + 0.5));
            let points: Vec<Vec3> = (0..count)
                .map(|_| {
                    let mut p = Vec3::new(next(), next(), next() * flat);
                    // some points pushed onto a face, a hair off it
                    if next() > 0.2 {
                        p.x = 0.5 + next() * flat;
                    }
                    p
                })
                .collect();
            if let Ok((volume, ..)) = ConvexHull::measure(&points) {
                assert!(volume > 0.0);
            }
        }
    }
    use crate::position::testing::at;
    use crate::world::testing::{ground, world};
    use crate::{
        ConvexInfo, PhysicsSettings, PhysicsWorld, QueryFilter, Ray, RigidBodyDesc, Shape,
        ShapeCast,
    };

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
    fn edge_pairs_on_the_gauss_map_give_what_every_pair_gives() {
        let unit = MassProperties {
            mass: 1.0,
            inertia: Vec3::ONE,
        };
        let pattern = crate::voronoi_box(Vec3::new(0.4, 0.3, 0.05), 40, 9).unwrap();
        let hulls: Vec<ConvexHull> = pattern
            .pieces
            .iter()
            .map(|piece| {
                let points: Vec<Vec3> = piece.points.iter().map(|p| *p - piece.center).collect();
                ConvexHull::from_rebased(&points, unit).unwrap()
            })
            .collect();
        assert!(hulls.iter().all(|hull| !hull.edge_faces.is_empty()));
        let mut state = 0x1234_5678_9abc_def1_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 1000.0 - 1.0
        };
        let deepest = |hit: &RawManifold| {
            hit.points[..hit.count]
                .iter()
                .fold(f32::NEG_INFINITY, |most, point| most.max(point.depth))
        };
        let (mut touching, mut apart) = (0, 0);
        for round in 0..20_000 {
            let a = &hulls[round % hulls.len()];
            let b = &hulls[(round * 7 + 3) % hulls.len()];
            let turn_a = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let turn_b = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let center = Vec3::new(next(), next(), next()) * 0.25;
            let poly_a = Poly::of_hull(a, turn_a, Vec3::ZERO);
            let poly_b = Poly::of_hull(b, turn_b, center);
            let mut every_a = Poly::of_hull(a, turn_a, Vec3::ZERO);
            let mut every_b = Poly::of_hull(b, turn_b, center);
            every_a.edge_faces = &[];
            every_b.edge_faces = &[];
            let pruned = poly_poly(&poly_a, &poly_b, 0.01);
            let every = poly_poly(&every_a, &every_b, 0.01);
            match (pruned, every) {
                (None, None) => apart += 1,
                (Some(pruned), Some(every)) => {
                    // every pair turns each axis toward b's center, which deep in can point it
                    // the wrong way and miss the shallowest. pruned turns it out of a's edge
                    let (one, two) = (deepest(&pruned), deepest(&every));
                    assert!(one <= two + 2e-3, "{round}: {one} {two}");
                    if two < 0.02 {
                        assert!((one - two).abs() < 2e-3, "{round}: {one} {two}");
                    }
                    touching += 1;
                }
                // apart, the gap over the faces of their difference can be less than over
                // other edge pairs, so this keeps a speculative point every pair dropped
                (Some(pruned), None) if deepest(&pruned) < 0.0 => apart += 1,
                (one, two) => panic!("{round}: {one:?} against {two:?}"),
            }
        }
        assert!(touching > 2000 && apart > 2000, "{touching} {apart}");
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
