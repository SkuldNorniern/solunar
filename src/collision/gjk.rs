use algebrix::{DQuat, DVec3, Quat};

use crate::Shape;

const MAX_ITERATIONS: usize = 48;
const OVERLAP_SQUARED: f64 = 1e-20;
const RELATIVE_TOLERANCE: f64 = 1e-10;

/// Shape without its radius. Spheres are a point, capsules a segment.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Hull {
    Point(DVec3),
    Segment(DVec3, DVec3),
    Box {
        center: DVec3,
        axes: [DVec3; 3],
        extents: [f64; 3],
    },
}

impl Hull {
    /// Returns the hull and the radius around it.
    pub(crate) fn of(shape: &Shape, rotation: Quat, center: DVec3) -> (Self, f64) {
        let rotation = DQuat::from_quat(rotation);
        match *shape {
            Shape::Sphere { radius } => (Self::Point(center), f64::from(radius)),
            Shape::Capsule {
                radius,
                half_height,
            } => {
                let half = rotation.mul_vec3(DVec3::Y * f64::from(half_height));
                (
                    Self::Segment(center - half, center + half),
                    f64::from(radius),
                )
            }
            Shape::Cuboid { half_extents } => (
                Self::Box {
                    center,
                    axes: [
                        rotation.mul_vec3(DVec3::X),
                        rotation.mul_vec3(DVec3::Y),
                        rotation.mul_vec3(DVec3::Z),
                    ],
                    extents: [
                        f64::from(half_extents.x),
                        f64::from(half_extents.y),
                        f64::from(half_extents.z),
                    ],
                },
                0.0,
            ),
        }
    }

    pub(crate) fn moved(self, offset: DVec3) -> Self {
        match self {
            Self::Point(point) => Self::Point(point + offset),
            Self::Segment(start, end) => Self::Segment(start + offset, end + offset),
            Self::Box {
                center,
                axes,
                extents,
            } => Self::Box {
                center: center + offset,
                axes,
                extents,
            },
        }
    }

    fn support(&self, direction: DVec3) -> DVec3 {
        match *self {
            Self::Point(point) => point,
            Self::Segment(start, end) => {
                if end.dot(direction) > start.dot(direction) {
                    end
                } else {
                    start
                }
            }
            Self::Box {
                center,
                axes,
                extents,
            } => {
                let mut point = center;
                for (axis, extent) in axes.iter().zip(extents) {
                    if axis.dot(direction) >= 0.0 {
                        point += *axis * extent;
                    } else {
                        point -= *axis * extent;
                    }
                }
                point
            }
        }
    }

    fn middle(&self) -> DVec3 {
        match *self {
            Self::Point(point) => point,
            Self::Segment(start, end) => (start + end) * 0.5,
            Self::Box { center, .. } => center,
        }
    }
}

/// `direction` is the unit vector from B to A, zero when the hulls overlap.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Separation {
    pub(crate) distance: f64,
    pub(crate) direction: DVec3,
    pub(crate) on_b: DVec3,
}

#[derive(Clone, Copy)]
struct Vertex {
    difference: DVec3,
    on_b: DVec3,
}

/// Part of a simplex closest to the origin, with its barycentric weights.
#[derive(Clone, Copy)]
struct Reduced {
    vertices: [Vertex; 4],
    weights: [f64; 4],
    count: usize,
}

impl Reduced {
    fn of(vertices: &[Vertex], weights: &[f64]) -> Self {
        let empty = Vertex {
            difference: DVec3::ZERO,
            on_b: DVec3::ZERO,
        };
        let mut result = Self {
            vertices: [empty; 4],
            weights: [0.0; 4],
            count: vertices.len(),
        };
        result.vertices[..vertices.len()].copy_from_slice(vertices);
        result.weights[..weights.len()].copy_from_slice(weights);
        result
    }

    fn sum(&self, read: impl Fn(&Vertex) -> DVec3) -> DVec3 {
        let mut total = DVec3::ZERO;
        for (vertex, weight) in self.vertices.iter().zip(&self.weights).take(self.count) {
            total += read(vertex) * *weight;
        }
        total
    }

    fn closest(&self) -> DVec3 {
        self.sum(|vertex| vertex.difference)
    }
}

/// Distance between two convex hulls (van den Bergen, Gilbert Johnson Keerthi).
pub(crate) fn distance(a: &Hull, b: &Hull) -> Separation {
    let vertex = |direction: DVec3| {
        let on_b = b.support(-direction);
        Vertex {
            difference: a.support(direction) - on_b,
            on_b,
        }
    };
    let start = a.middle() - b.middle();
    let first = vertex(if start.length_squared() > 1e-20 {
        start
    } else {
        DVec3::X
    });
    let mut simplex = Reduced::of(&[first], &[1.0]);
    let mut closest = simplex.closest();
    for _ in 0..MAX_ITERATIONS {
        let squared = closest.length_squared();
        if squared < OVERLAP_SQUARED {
            break;
        }
        let next = vertex(-closest);
        if squared - closest.dot(next.difference) <= RELATIVE_TOLERANCE * squared {
            break;
        }
        let known = simplex.vertices[..simplex.count]
            .iter()
            .any(|vertex| vertex.difference == next.difference);
        if known {
            break;
        }
        let mut grown = simplex.vertices;
        grown[simplex.count] = next;
        let reduced = reduce(&grown[..=simplex.count]);
        let candidate = reduced.closest();
        // rounding can stop the distance from shrinking, the last simplex is the answer then
        if candidate.length_squared() >= squared {
            break;
        }
        simplex = reduced;
        closest = candidate;
    }
    let squared = closest.length_squared();
    if squared < OVERLAP_SQUARED {
        return Separation {
            distance: 0.0,
            direction: DVec3::ZERO,
            on_b: simplex.sum(|vertex| vertex.on_b),
        };
    }
    let length = squared.sqrt();
    Separation {
        distance: length,
        direction: closest / length,
        on_b: simplex.sum(|vertex| vertex.on_b),
    }
}

fn reduce(vertices: &[Vertex]) -> Reduced {
    match *vertices {
        [a] => Reduced::of(&[a], &[1.0]),
        [a, b] => reduce_segment(a, b),
        [a, b, c] => reduce_triangle(a, b, c),
        [a, b, c, d] => reduce_tetrahedron(a, b, c, d),
        _ => Reduced::of(&[], &[]),
    }
}

fn reduce_segment(a: Vertex, b: Vertex) -> Reduced {
    let along = b.difference - a.difference;
    let squared = along.length_squared();
    if squared <= 0.0 {
        return Reduced::of(&[a], &[1.0]);
    }
    let t = -a.difference.dot(along) / squared;
    if t <= 0.0 {
        Reduced::of(&[a], &[1.0])
    } else if t >= 1.0 {
        Reduced::of(&[b], &[1.0])
    } else {
        Reduced::of(&[a, b], &[1.0 - t, t])
    }
}

/// Closest point on a triangle (Ericson, Real-Time Collision Detection 5.1.5).
fn reduce_triangle(a: Vertex, b: Vertex, c: Vertex) -> Reduced {
    let ab = b.difference - a.difference;
    let ac = c.difference - a.difference;
    let d1 = ab.dot(-a.difference);
    let d2 = ac.dot(-a.difference);
    if d1 <= 0.0 && d2 <= 0.0 {
        return Reduced::of(&[a], &[1.0]);
    }
    let d3 = ab.dot(-b.difference);
    let d4 = ac.dot(-b.difference);
    if d3 >= 0.0 && d4 <= d3 {
        return Reduced::of(&[b], &[1.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return Reduced::of(&[a, b], &[1.0 - t, t]);
    }
    let d5 = ab.dot(-c.difference);
    let d6 = ac.dot(-c.difference);
    if d6 >= 0.0 && d5 <= d6 {
        return Reduced::of(&[c], &[1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return Reduced::of(&[a, c], &[1.0 - t, t]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return Reduced::of(&[b, c], &[1.0 - t, t]);
    }
    let total = va + vb + vc;
    if total <= 0.0 {
        return nearest_of([
            reduce_segment(a, b),
            reduce_segment(a, c),
            reduce_segment(b, c),
        ]);
    }
    let (v, w) = (vb / total, vc / total);
    Reduced::of(&[a, b, c], &[1.0 - v - w, v, w])
}

fn reduce_tetrahedron(a: Vertex, b: Vertex, c: Vertex, d: Vertex) -> Reduced {
    let mut best: Option<Reduced> = None;
    for (p, q, r, opposite) in [(a, b, c, d), (a, c, d, b), (a, d, b, c), (b, d, c, a)] {
        let normal = (q.difference - p.difference).cross(r.difference - p.difference);
        let origin_side = normal.dot(-p.difference);
        let other_side = normal.dot(opposite.difference - p.difference);
        if origin_side * other_side > 0.0 {
            continue;
        }
        let candidate = reduce_triangle(p, q, r);
        best = Some(match best {
            Some(old) if old.closest().length_squared() <= candidate.closest().length_squared() => {
                old
            }
            _ => candidate,
        });
    }
    // the origin behind no face is inside, so the hulls overlap
    best.unwrap_or_else(|| {
        let mut inside = Reduced::of(&[a, b, c, d], &[0.25; 4]);
        inside.weights = [0.0; 4];
        inside
    })
}

fn nearest_of(options: [Reduced; 3]) -> Reduced {
    let [mut best, second, third] = options;
    for option in [second, third] {
        if option.closest().length_squared() < best.closest().length_squared() {
            best = option;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use algebrix::Vec3;

    use super::*;
    use crate::collision::narrowphase::collide;

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }

    /// Surface gap between two shapes, negative never: overlap reads as zero.
    fn gap(a: &Shape, rotation_a: Quat, b: &Shape, rotation_b: Quat, center_b: Vec3) -> f64 {
        let (hull_a, radius_a) = Hull::of(a, rotation_a, DVec3::ZERO);
        let (hull_b, radius_b) = Hull::of(b, rotation_b, DVec3::from_vec3(center_b));
        (distance(&hull_a, &hull_b).distance - radius_a - radius_b).max(0.0)
    }

    #[test]
    fn point_to_point() {
        let result = distance(
            &Hull::Point(DVec3::ZERO),
            &Hull::Point(DVec3::new(3.0, 4.0, 0.0)),
        );
        near(result.distance, 5.0);
        near(result.direction.x, -0.6);
        near(result.direction.y, -0.8);
        near(result.on_b.x, 3.0);
    }

    #[test]
    fn point_to_segment_interior() {
        let result = distance(
            &Hull::Point(DVec3::new(0.0, 2.0, 0.0)),
            &Hull::Segment(DVec3::new(-1.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
        );
        near(result.distance, 2.0);
        near(result.on_b.x, 0.0);
        near(result.direction.y, 1.0);
    }

    #[test]
    fn boxes_face_to_face_edge_and_corner() {
        let cube = Shape::cuboid(Vec3::splat(1.0));
        let rest = Quat::IDENTITY;
        near(gap(&cube, rest, &cube, rest, Vec3::new(5.0, 0.0, 0.0)), 3.0);
        let edge = gap(&cube, rest, &cube, rest, Vec3::new(5.0, 5.0, 0.0));
        near(edge, (3.0_f64 * 3.0 * 2.0).sqrt());
        let corner = gap(&cube, rest, &cube, rest, Vec3::new(5.0, 5.0, 5.0));
        near(corner, (3.0_f64 * 3.0 * 3.0).sqrt());
    }

    #[test]
    fn turned_box_reaches_by_its_diagonal() {
        let cube = Shape::cuboid(Vec3::splat(1.0));
        let turned = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let result = gap(
            &cube,
            Quat::IDENTITY,
            &cube,
            turned,
            Vec3::new(5.0, 0.0, 0.0),
        );
        assert!((result - (4.0 - 2.0_f64.sqrt())).abs() < 1e-5, "{result}");
    }

    #[test]
    fn overlap_reads_zero() {
        let cube = Shape::cuboid(Vec3::splat(1.0));
        let result = gap(
            &cube,
            Quat::IDENTITY,
            &cube,
            Quat::IDENTITY,
            Vec3::new(1.0, 0.5, 0.0),
        );
        near(result, 0.0);
    }

    #[test]
    fn big_box_stays_accurate() {
        let ground = Shape::cuboid(Vec3::new(500.0, 1.0, 500.0));
        let cube = Shape::cuboid(Vec3::splat(0.5));
        let result = gap(
            &ground,
            Quat::IDENTITY,
            &cube,
            Quat::IDENTITY,
            Vec3::new(123.0, 1.501, -77.0),
        );
        assert!((result - 0.001).abs() < 1e-5, "{result}");
    }

    #[test]
    fn agrees_with_the_narrowphase_on_random_poses() {
        let mut state = 0x0bad_f00d_1337_c0de_u64;
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
        let (mut touching, mut apart) = (0, 0);
        for _ in 0..20_000 {
            let a = shapes[(next().abs() * 2.99) as usize];
            let b = shapes[(next().abs() * 2.99) as usize];
            let rotation_a = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let rotation_b = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let center = Vec3::new(next(), next(), next()) * 2.5;
            let forward = gap(&a, rotation_a, &b, rotation_b, center);
            let backward = gap(&b, rotation_b, &a, rotation_a, -center);
            assert!((forward - backward).abs() < 1e-4, "{a:?} {b:?} {center:?}");
            let hit = collide(&a, rotation_a, &b, rotation_b, center).is_some();
            if forward > 1e-3 {
                assert!(!hit, "{a:?} {b:?} {center:?} gap {forward}");
                apart += 1;
            } else if forward == 0.0 {
                assert!(hit, "{a:?} {b:?} {center:?}");
                touching += 1;
            }
        }
        assert!(touching > 1000 && apart > 1000, "{touching} {apart}");
    }
}
