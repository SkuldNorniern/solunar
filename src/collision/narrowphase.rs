use algebrix::{Quat, Vec3};

use crate::Shape;

pub(crate) const MAX_CONTACTS: usize = 4;
const MAX_CLIP_POINTS: usize = 16;
const PARALLEL_COS: f32 = 0.995;
const EDGE_BIAS: f32 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RawPoint {
    /// Relative to the center of body A.
    pub(crate) position: Vec3,
    pub(crate) depth: f32,
}

/// `normal` points from A to B.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RawManifold {
    pub(crate) normal: Vec3,
    pub(crate) points: [RawPoint; MAX_CONTACTS],
    pub(crate) count: usize,
}

impl RawManifold {
    fn new(normal: Vec3) -> Self {
        Self {
            normal,
            points: [RawPoint {
                position: Vec3::ZERO,
                depth: 0.0,
            }; MAX_CONTACTS],
            count: 0,
        }
    }

    fn push(&mut self, position: Vec3, depth: f32) {
        if self.count < MAX_CONTACTS {
            self.points[self.count] = RawPoint { position, depth };
            self.count += 1;
        }
    }

    fn filled(self) -> Option<Self> {
        (self.count > 0).then_some(self)
    }
}

/// `center_b` is the center of B relative to the center of A. Reports real overlap only.
pub(crate) fn collide(
    shape_a: &Shape,
    rotation_a: Quat,
    shape_b: &Shape,
    rotation_b: Quat,
    center_b: Vec3,
) -> Option<RawManifold> {
    contacts(shape_a, rotation_a, shape_b, rotation_b, center_b, 0.0)
}

/// Also reports points up to `margin` apart, those have a negative depth.
pub(crate) fn contacts(
    shape_a: &Shape,
    rotation_a: Quat,
    shape_b: &Shape,
    rotation_b: Quat,
    center_b: Vec3,
    margin: f32,
) -> Option<RawManifold> {
    match (
        rounded_segment(shape_a, rotation_a),
        rounded_segment(shape_b, rotation_b),
    ) {
        (Some((half_a, radius_a)), Some((half_b, radius_b))) => {
            segment_segment(half_a, radius_a, half_b, radius_b, center_b, margin)
        }
        (None, Some((half_b, radius_b))) => {
            let extents = half_extents(shape_a)?;
            box_segment(extents, rotation_a, center_b, half_b, radius_b, margin)
        }
        (Some((half_a, radius_a)), None) => {
            let extents = half_extents(shape_b)?;
            let flipped = box_segment(extents, rotation_b, -center_b, half_a, radius_a, margin)?;
            let mut result = RawManifold::new(-flipped.normal);
            for point in &flipped.points[..flipped.count] {
                result.push(point.position + center_b, point.depth);
            }
            Some(result)
        }
        (None, None) => box_box(
            half_extents(shape_a)?,
            rotation_a,
            half_extents(shape_b)?,
            rotation_b,
            center_b,
            margin,
        ),
    }
}

/// Spheres and capsules are both a segment with a radius.
fn rounded_segment(shape: &Shape, rotation: Quat) -> Option<(Vec3, f32)> {
    match *shape {
        Shape::Sphere { radius } => Some((Vec3::ZERO, radius)),
        Shape::Capsule {
            radius,
            half_height,
        } => Some((rotation.rotate_vec3(Vec3::Y * half_height), radius)),
        Shape::Cuboid { .. } => None,
    }
}

fn half_extents(shape: &Shape) -> Option<Vec3> {
    match *shape {
        Shape::Cuboid { half_extents } => Some(half_extents),
        _ => None,
    }
}

/// Closest points of segments `a0 a1` and `b0 b1` (Ericson, Real-Time Collision Detection).
fn closest_on_segments(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> (Vec3, Vec3) {
    let along_a = a1 - a0;
    let along_b = b1 - b0;
    let between = a0 - b0;
    let len_a = along_a.length_squared();
    let len_b = along_b.length_squared();
    let f = along_b.dot(between);
    let (s, t) = if len_a <= 1e-12 && len_b <= 1e-12 {
        (0.0, 0.0)
    } else if len_a <= 1e-12 {
        (0.0, (f / len_b).clamp(0.0, 1.0))
    } else {
        let c = along_a.dot(between);
        if len_b <= 1e-12 {
            ((-c / len_a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = along_a.dot(along_b);
            let denom = len_a * len_b - b * b;
            let mut s = if denom > 1e-12 {
                ((b * f - c * len_b) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / len_b;
            if t < 0.0 {
                t = 0.0;
                s = (-c / len_a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / len_a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    (a0 + along_a * s, b0 + along_b * t)
}

fn segment_segment(
    half_a: Vec3,
    radius_a: f32,
    half_b: Vec3,
    radius_b: f32,
    center_b: Vec3,
    margin: f32,
) -> Option<RawManifold> {
    let (on_a, on_b) = closest_on_segments(-half_a, half_a, center_b - half_b, center_b + half_b);
    let gap = on_b - on_a;
    let distance = gap.length();
    let radii = radius_a + radius_b;
    if distance >= radii + margin {
        return None;
    }
    let normal = if distance > 1e-6 {
        gap * distance.recip()
    } else {
        fallback_normal(half_a, half_b, center_b)
    };
    let mut result = RawManifold::new(normal);

    let (len_a, len_b) = (half_a.length(), half_b.length());
    if len_a > 1e-6 && len_b > 1e-6 {
        let (axis_a, axis_b) = (half_a * len_a.recip(), half_b * len_b.recip());
        if axis_a.dot(axis_b).abs() > PARALLEL_COS {
            let along = center_b.dot(axis_a);
            let reach = len_b * axis_a.dot(axis_b).abs();
            let low = (-len_a).max(along - reach);
            let high = len_a.min(along + reach);
            if high - low > 1e-3 {
                for s in [low, high] {
                    let point_a = axis_a * s;
                    let point_b = center_b + axis_b * (point_a - center_b).dot(axis_b);
                    let depth = radii - (point_b - point_a).dot(normal);
                    if depth > -margin {
                        result.push(point_a + normal * (radius_a - depth * 0.5), depth);
                    }
                }
                if result.count > 0 {
                    return Some(result);
                }
            }
        }
    }
    let depth = radii - distance;
    result.push(on_a + normal * (radius_a - depth * 0.5), depth);
    Some(result)
}

fn fallback_normal(half_a: Vec3, half_b: Vec3, center_b: Vec3) -> Vec3 {
    let crossed = half_a.cross(half_b);
    if crossed.length_squared() > 1e-10 {
        return crossed.normalize();
    }
    if center_b.length_squared() > 1e-10 {
        return center_b.normalize();
    }
    Vec3::Y
}

/// Signed distance from `point` to a box at the origin, and the outward normal. Box axes.
fn box_distance(extents: Vec3, point: Vec3) -> (f32, Vec3) {
    let clamped = point.clamp(-extents, extents);
    let outside = point - clamped;
    let length = outside.length();
    if length > 1e-6 {
        return (length, outside * length.recip());
    }
    let room = extents - point.abs();
    let sign = |value: f32| if value < 0.0 { -1.0 } else { 1.0 };
    if room.x <= room.y && room.x <= room.z {
        (-room.x, Vec3::X * sign(point.x))
    } else if room.y <= room.z {
        (-room.y, Vec3::Y * sign(point.y))
    } else {
        (-room.z, Vec3::Z * sign(point.z))
    }
}

/// Positions are relative to the box center, normal points from box to segment.
fn box_segment(
    extents: Vec3,
    rotation: Quat,
    center: Vec3,
    half: Vec3,
    radius: f32,
    margin: f32,
) -> Option<RawManifold> {
    let inverse = rotation.conjugate();
    let center = inverse.rotate_vec3(center);
    let half = inverse.rotate_vec3(half);
    let at = |t: f32| center + half * t;
    let distance_at = |t: f32| box_distance(extents, at(t)).0;

    // signed distance to a convex shape is convex along a segment
    let (mut low, mut high) = (-1.0_f32, 1.0_f32);
    if half.length_squared() > 1e-12 {
        for _ in 0..24 {
            let third = (high - low) / 3.0;
            if distance_at(low + third) <= distance_at(high - third) {
                high -= third;
            } else {
                low += third;
            }
        }
    }
    let best = (low + high) * 0.5;
    let (distance, normal) = box_distance(extents, at(best));
    if distance >= radius + margin {
        return None;
    }

    let mut result = RawManifold::new(rotation.rotate_vec3(normal));
    let add = |result: &mut RawManifold, t: f32| {
        let point = at(t);
        let (distance, other) = box_distance(extents, point);
        if distance < radius + margin && other.dot(normal) > 0.9 {
            let contact = point - other * ((distance + radius) * 0.5);
            result.push(rotation.rotate_vec3(contact), radius - distance);
        }
    };
    let is_segment = half.length_squared() > 1e-12;
    if is_segment {
        add(&mut result, -1.0);
        add(&mut result, 1.0);
    }
    if !is_segment || best.abs() < 0.999 {
        add(&mut result, best);
    }
    if result.count == 0 {
        add(&mut result, best);
    }
    result.filled()
}

#[derive(Clone, Copy)]
struct Oriented {
    axes: [Vec3; 3],
    extents: [f32; 3],
    center: Vec3,
}

impl Oriented {
    fn new(extents: Vec3, rotation: Quat, center: Vec3) -> Self {
        Self {
            axes: [
                rotation.rotate_vec3(Vec3::X),
                rotation.rotate_vec3(Vec3::Y),
                rotation.rotate_vec3(Vec3::Z),
            ],
            extents: [extents.x, extents.y, extents.z],
            center,
        }
    }

    fn reach(&self, direction: Vec3) -> f32 {
        (0..3)
            .map(|i| self.extents[i] * self.axes[i].dot(direction).abs())
            .sum()
    }

    /// Corners of the face around axis `index` on side `sign`.
    fn face(&self, index: usize, sign: f32) -> [Vec3; 4] {
        let (u, v) = ((index + 1) % 3, (index + 2) % 3);
        let middle = self.center + self.axes[index] * (sign * self.extents[index]);
        let along_u = self.axes[u] * self.extents[u];
        let along_v = self.axes[v] * self.extents[v];
        [
            middle + along_u + along_v,
            middle - along_u + along_v,
            middle - along_u - along_v,
            middle + along_u - along_v,
        ]
    }

    /// Edge along axis `index` that is furthest in `direction`.
    fn support_edge(&self, index: usize, direction: Vec3) -> (Vec3, Vec3) {
        let mut point = self.center;
        for other in 0..3 {
            if other != index {
                let sign = if self.axes[other].dot(direction) >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                point += self.axes[other] * (sign * self.extents[other]);
            }
        }
        let along = self.axes[index] * self.extents[index];
        (point - along, point + along)
    }
}

#[derive(Clone, Copy)]
enum Axis {
    FaceA(usize),
    FaceB(usize),
    Edge(usize, usize),
}

fn box_box(
    extents_a: Vec3,
    rotation_a: Quat,
    extents_b: Vec3,
    rotation_b: Quat,
    center_b: Vec3,
    margin: f32,
) -> Option<RawManifold> {
    let a = Oriented::new(extents_a, rotation_a, Vec3::ZERO);
    let b = Oriented::new(extents_b, rotation_b, center_b);

    let separation = |axis: Vec3| center_b.dot(axis).abs() - a.reach(axis) - b.reach(axis);
    let mut best_face = f32::NEG_INFINITY;
    let mut best = Axis::FaceA(0);
    let mut best_normal = a.axes[0];
    for i in 0..6 {
        let (axis, kind) = if i < 3 {
            (a.axes[i], Axis::FaceA(i))
        } else {
            (b.axes[i - 3], Axis::FaceB(i - 3))
        };
        let gap = separation(axis);
        if gap > margin {
            return None;
        }
        if gap > best_face {
            best_face = gap;
            best = kind;
            best_normal = axis;
        }
    }
    let mut best_gap = best_face;
    for i in 0..3 {
        for j in 0..3 {
            let crossed = a.axes[i].cross(b.axes[j]);
            let length = crossed.length();
            if length < 1e-3 {
                continue;
            }
            let axis = crossed * length.recip();
            let gap = separation(axis);
            if gap > margin {
                return None;
            }
            // faces win ties, they give full manifolds
            if gap > best_gap + EDGE_BIAS {
                best_gap = gap;
                best = Axis::Edge(i, j);
                best_normal = axis;
            }
        }
    }
    let normal = if center_b.dot(best_normal) < 0.0 {
        -best_normal
    } else {
        best_normal
    };

    match best {
        Axis::FaceA(index) => Some(face_contact(&a, &b, index, normal, normal, margin)),
        Axis::FaceB(index) => Some(face_contact(&b, &a, index, -normal, normal, margin)),
        Axis::Edge(i, j) => {
            let (a0, a1) = a.support_edge(i, normal);
            let (b0, b1) = b.support_edge(j, -normal);
            let (on_a, on_b) = closest_on_segments(a0, a1, b0, b1);
            let mut result = RawManifold::new(normal);
            result.push((on_a + on_b) * 0.5, -best_gap);
            Some(result)
        }
    }
    .and_then(RawManifold::filled)
}

/// Clips the incident face of `other` against reference face `index` of `reference`.
fn face_contact(
    reference: &Oriented,
    other: &Oriented,
    index: usize,
    outward: Vec3,
    normal: Vec3,
    margin: f32,
) -> RawManifold {
    let plane_at = reference.center.dot(outward) + reference.extents[index];

    let mut incident = 0;
    let mut most = -1.0_f32;
    for i in 0..3 {
        let facing = other.axes[i].dot(outward).abs();
        if facing > most {
            most = facing;
            incident = i;
        }
    }
    let incident_sign = if other.axes[incident].dot(outward) > 0.0 {
        -1.0
    } else {
        1.0
    };

    let mut polygon = [Vec3::ZERO; MAX_CLIP_POINTS];
    let mut count = 4;
    polygon[..4].copy_from_slice(&other.face(incident, incident_sign));
    for side in [(index + 1) % 3, (index + 2) % 3] {
        let axis = reference.axes[side];
        let middle = reference.center.dot(axis);
        let limit = reference.extents[side];
        count = clip(&mut polygon, count, axis, middle + limit);
        count = clip(&mut polygon, count, -axis, limit - middle);
    }

    let mut result = RawManifold::new(normal);
    let mut kept = [RawPoint {
        position: Vec3::ZERO,
        depth: 0.0,
    }; MAX_CLIP_POINTS];
    let mut kept_count = 0;
    for point in &polygon[..count] {
        let depth = plane_at - point.dot(outward);
        if depth >= -margin {
            kept[kept_count] = RawPoint {
                position: *point + outward * (depth * 0.5),
                depth,
            };
            kept_count += 1;
        }
    }
    reduce(&kept[..kept_count], &mut result);
    result
}

/// Keeps the side where `point . axis <= limit`.
fn clip(polygon: &mut [Vec3; MAX_CLIP_POINTS], count: usize, axis: Vec3, limit: f32) -> usize {
    let input = *polygon;
    let mut out = 0;
    for i in 0..count {
        let current = input[i];
        let next = input[(i + 1) % count];
        let current_gap = current.dot(axis) - limit;
        let next_gap = next.dot(axis) - limit;
        if current_gap <= 0.0 && out < MAX_CLIP_POINTS {
            polygon[out] = current;
            out += 1;
        }
        if (current_gap < 0.0) != (next_gap < 0.0)
            && current_gap != next_gap
            && out < MAX_CLIP_POINTS
        {
            polygon[out] = current + (next - current) * (current_gap / (current_gap - next_gap));
            out += 1;
        }
    }
    out
}

/// Deepest point, the one furthest from it, then the largest triangles on each side.
fn reduce(points: &[RawPoint], result: &mut RawManifold) {
    if points.len() <= MAX_CONTACTS {
        for point in points {
            result.push(point.position, point.depth);
        }
        return;
    }
    let pick = |score: &dyn Fn(&RawPoint) -> f32| {
        let mut best = points[0];
        let mut most = f32::NEG_INFINITY;
        for point in points {
            let value = score(point);
            if value > most {
                most = value;
                best = *point;
            }
        }
        best
    };
    let first = pick(&|point| point.depth);
    let second = pick(&|point| (point.position - first.position).length_squared());
    let span = second.position - first.position;
    let area = |point: &RawPoint| {
        span.cross(point.position - first.position)
            .dot(result.normal)
    };
    let third = pick(&|point| area(point));
    let fourth = pick(&|point| -area(point));
    for point in [first, second, third, fourth] {
        result.push(point.position, point.depth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-4, "{a} != {b}");
    }

    fn near_vec(a: Vec3, b: Vec3) {
        assert!((a - b).length() < 1e-4, "{a:?} != {b:?}");
    }

    fn identity(a: &Shape, b: &Shape, center: Vec3) -> Option<RawManifold> {
        collide(a, Quat::IDENTITY, b, Quat::IDENTITY, center)
    }

    #[test]
    fn spheres_touch_by_radius() {
        let a = Shape::sphere(1.0);
        let b = Shape::sphere(0.5);
        assert!(identity(&a, &b, Vec3::new(2.0, 0.0, 0.0)).is_none());
        let hit = identity(&a, &b, Vec3::new(1.25, 0.0, 0.0)).unwrap();
        assert_eq!(hit.count, 1);
        near_vec(hit.normal, Vec3::X);
        near(hit.points[0].depth, 0.25);
        near(hit.points[0].position.x, 0.875);
    }

    #[test]
    fn same_center_spheres_still_get_a_normal() {
        let hit = identity(&Shape::sphere(1.0), &Shape::sphere(1.0), Vec3::ZERO).unwrap();
        near(hit.normal.length(), 1.0);
        near(hit.points[0].depth, 2.0);
    }

    #[test]
    fn crossed_capsules_touch_once() {
        let a = Shape::capsule(0.5, 2.0);
        let b = Shape::capsule(0.5, 2.0);
        let hit = collide(
            &a,
            Quat::IDENTITY,
            &b,
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            Vec3::new(0.75, 0.0, 0.0),
        )
        .unwrap();
        assert_eq!(hit.count, 1);
        near_vec(hit.normal, Vec3::X);
        near(hit.points[0].depth, 0.25);
    }

    #[test]
    fn parallel_capsules_touch_along_a_span() {
        let capsule = Shape::capsule(0.5, 2.0);
        let hit = identity(&capsule, &capsule, Vec3::new(0.75, 1.0, 0.0)).unwrap();
        assert_eq!(hit.count, 2);
        near_vec(hit.normal, Vec3::X);
        near(hit.points[0].depth, 0.25);
        near(hit.points[1].depth, 0.25);
        near(
            (hit.points[0].position - hit.points[1].position).length(),
            3.0,
        );
    }

    #[test]
    fn sphere_on_box_face_and_corner() {
        let ground = Shape::cuboid(Vec3::new(5.0, 1.0, 5.0));
        let ball = Shape::sphere(1.0);
        let hit = identity(&ground, &ball, Vec3::new(0.0, 1.75, 0.0)).unwrap();
        near_vec(hit.normal, Vec3::Y);
        near(hit.points[0].depth, 0.25);
        assert!(identity(&ground, &ball, Vec3::new(0.0, 2.5, 0.0)).is_none());

        let corner = 5.0 + 0.5 / 3.0_f32.sqrt();
        let unit = Shape::cuboid(Vec3::splat(5.0));
        let hit = identity(&unit, &ball, Vec3::splat(corner)).unwrap();
        near_vec(hit.normal, Vec3::splat(1.0 / 3.0_f32.sqrt()));
        near(hit.points[0].depth, 0.5);
    }

    #[test]
    fn sphere_inside_box_is_pushed_out_the_nearest_face() {
        let hit = identity(
            &Shape::cuboid(Vec3::new(5.0, 1.0, 5.0)),
            &Shape::sphere(0.25),
            Vec3::new(0.0, 0.5, 0.0),
        )
        .unwrap();
        near_vec(hit.normal, Vec3::Y);
        near(hit.points[0].depth, 0.75);
    }

    #[test]
    fn capsule_lying_on_box_has_two_points() {
        let ground = Shape::cuboid(Vec3::new(5.0, 1.0, 5.0));
        let capsule = Shape::capsule(0.5, 1.0);
        let lying = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let hit = collide(
            &ground,
            Quat::IDENTITY,
            &capsule,
            lying,
            Vec3::new(0.0, 1.25, 0.0),
        )
        .unwrap();
        assert_eq!(hit.count, 2);
        near_vec(hit.normal, Vec3::Y);
        near(hit.points[0].depth, 0.25);
        near(hit.points[1].depth, 0.25);
    }

    #[test]
    fn standing_capsule_on_box_has_one_point() {
        let ground = Shape::cuboid(Vec3::new(5.0, 1.0, 5.0));
        let capsule = Shape::capsule(0.5, 1.0);
        let hit = identity(&ground, &capsule, Vec3::new(0.0, 2.25, 0.0)).unwrap();
        assert_eq!(hit.count, 1);
        near(hit.points[0].depth, 0.25);
    }

    #[test]
    fn stacked_boxes_have_four_points() {
        let shape = Shape::cuboid(Vec3::splat(0.5));
        let hit = identity(&shape, &shape, Vec3::new(0.0, 0.9, 0.0)).unwrap();
        assert_eq!(hit.count, 4);
        near_vec(hit.normal, Vec3::Y);
        for point in &hit.points {
            near(point.depth, 0.1);
            near(point.position.y, 0.45);
        }
        assert!(identity(&shape, &shape, Vec3::new(0.0, 1.1, 0.0)).is_none());
    }

    #[test]
    fn offset_boxes_clip_to_the_overlap() {
        let shape = Shape::cuboid(Vec3::splat(0.5));
        let hit = identity(&shape, &shape, Vec3::new(0.5, 0.9, 0.0)).unwrap();
        assert_eq!(hit.count, 4);
        for point in &hit.points {
            assert!(point.position.x >= -1e-4 && point.position.x <= 0.5 + 1e-4);
        }
    }

    #[test]
    fn box_turned_on_its_corner_touches_once() {
        let ground = Shape::cuboid(Vec3::new(5.0, 1.0, 5.0));
        let cube = Shape::cuboid(Vec3::splat(0.5));
        let turn = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4) * Quat::from_rotation_x(0.6);
        let reach = Oriented::new(Vec3::splat(0.5), turn, Vec3::ZERO).reach(Vec3::Y);
        let hit = collide(
            &ground,
            Quat::IDENTITY,
            &cube,
            turn,
            Vec3::new(0.0, 1.0 + reach - 0.05, 0.0),
        )
        .unwrap();
        near_vec(hit.normal, Vec3::Y);
        assert_eq!(hit.count, 1);
        near(hit.points[0].depth, 0.05);
    }

    #[test]
    fn crossed_boxes_meet_edge_to_edge() {
        let long = Shape::cuboid(Vec3::new(2.0, 0.2, 0.2));
        let quarter = std::f32::consts::FRAC_PI_4;
        let a = Quat::from_rotation_x(quarter);
        let b = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2) * Quat::from_rotation_x(quarter);
        let reach = 0.2 * 2.0_f32.sqrt();
        let hit = collide(&long, a, &long, b, Vec3::new(0.0, 2.0 * reach - 0.02, 0.0)).unwrap();
        assert_eq!(hit.count, 1);
        near_vec(hit.normal, Vec3::Y);
        near(hit.points[0].depth, 0.02);
    }

    #[test]
    fn swapping_bodies_flips_the_normal() {
        let shapes = [
            Shape::sphere(0.7),
            Shape::capsule(0.4, 0.8),
            Shape::cuboid(Vec3::new(0.6, 0.5, 0.9)),
        ];
        let rotation_a = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
        let rotation_b = Quat::from_axis_angle(Vec3::new(-2.0, 1.0, 0.5).normalize(), 1.9);
        let center = Vec3::new(0.3, 0.7, -0.2);
        for a in &shapes {
            for b in &shapes {
                let one = collide(a, rotation_a, b, rotation_b, center).unwrap();
                let two = collide(b, rotation_b, a, rotation_a, -center).unwrap();
                assert!(
                    (one.normal + two.normal).length() < 1e-2,
                    "{a:?} {b:?} {:?} {:?}",
                    one.normal,
                    two.normal
                );
                near(one.normal.length(), 1.0);
            }
        }
    }

    #[test]
    fn random_poses_stay_finite_and_point_toward_b() {
        let mut state = 0xdead_beef_cafe_f00d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 1000.0 - 1.0
        };
        let shapes = [
            Shape::sphere(0.5),
            Shape::capsule(0.3, 0.7),
            Shape::cuboid(Vec3::new(0.5, 0.2, 0.8)),
        ];
        for _ in 0..20_000 {
            let a = shapes[(next().abs() * 2.99) as usize];
            let b = shapes[(next().abs() * 2.99) as usize];
            let rotation_a = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let rotation_b = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let center = Vec3::new(next(), next(), next()) * 1.5;
            if let Some(hit) = collide(&a, rotation_a, &b, rotation_b, center) {
                assert!(hit.count >= 1 && hit.count <= MAX_CONTACTS);
                assert!(hit.normal.is_finite());
                assert!((hit.normal.length() - 1.0).abs() < 1e-3);
                for point in &hit.points[..hit.count] {
                    assert!(point.position.is_finite());
                    assert!(point.depth.is_finite() && point.depth >= 0.0);
                }
            }
        }
    }
}
