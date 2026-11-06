use algebrix::{DQuat, DVec3, Quat, Vec3};

use super::gjk::{Hull, distance};
use super::trimesh::{Aabb, point_triangle, ray_triangle};
use crate::{BodyHandle, FrameId, GridPosition, PhysicsError, PhysicsWorld, Shape};

const MAX_CAST_ITERATIONS: usize = 64;
const CAST_SKIN: f64 = 1e-4;
const PARALLEL_SQUARED: f64 = 1e-14;

/// Queries see one frame. `mask` is tested against each body's layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryFilter {
    pub frame: FrameId,
    pub mask: u32,
    pub exclude: Option<BodyHandle>,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            frame: FrameId::ROOT,
            mask: u32::MAX,
            exclude: None,
        }
    }
}

/// `direction` does not need to be unit length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: GridPosition,
    pub direction: Vec3,
    pub max_distance: f32,
}

/// Sweeps `shape` along `direction` without turning it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeCast {
    pub shape: Shape,
    pub rotation: Quat,
    pub origin: GridPosition,
    pub direction: Vec3,
    pub max_distance: f32,
}

/// `normal` faces the caster. Starting inside gives distance 0 and the normal against the motion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CastHit {
    pub body: BodyHandle,
    pub distance: f32,
    pub point: GridPosition,
    pub normal: Vec3,
}

/// `distance` is negative inside the body. `point` is on its surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointHit {
    pub body: BodyHandle,
    pub distance: f32,
    pub point: GridPosition,
    pub normal: Vec3,
}

impl PhysicsWorld {
    /// Nearest hit, the lower handle wins a tie.
    ///
    /// # Errors
    /// Fails on an unknown frame, non finite input or a zero direction.
    pub fn raycast(
        &self,
        ray: &Ray,
        filter: &QueryFilter,
    ) -> Result<Option<CastHit>, PhysicsError> {
        let direction = self.check_cast(ray.origin, ray.direction, ray.max_distance, filter)?;
        let max = f64::from(ray.max_distance);
        let mut best: Option<(f64, BodyHandle, DVec3)> = None;
        for dense in 0..self.bodies.len() {
            if !self.visible(dense, filter) {
                continue;
            }
            let offset = ray.origin.offset_from(self.bodies.position(dense));
            let limit = best.map_or(max, |found| found.0);
            let assets = &self.bodies.assets;
            let reach = f64::from(assets.bounding_radius(&self.bodies.shape[dense]));
            if !sphere_near_path(offset, direction, limit, reach) {
                continue;
            }
            let handle = self.bodies.handle_at(dense);
            let body = (&self.bodies.shape[dense], self.bodies.rotation[dense]);
            if let Some(mesh) = assets.mesh(body.0) {
                let rotation = DQuat::from_quat(body.1);
                let inverse = rotation.conjugate();
                let start = inverse.mul_vec3(offset);
                let along = inverse.mul_vec3(direction);
                let tip = (start + along * limit).as_vec3();
                let area = Aabb::around(start.as_vec3(), 0.0).union(Aabb::around(tip, 0.0));
                mesh.for_each_near(area, |index| {
                    let limit = best.map_or(max, |found| found.0);
                    let triangle = mesh.triangle(index);
                    if let Some((time, normal)) = ray_triangle(start, along, &triangle, limit)
                        && best.is_none_or(|found| (time, handle) < (found.0, found.1))
                    {
                        best = Some((time, handle, rotation.mul_vec3(normal)));
                    }
                });
                continue;
            }
            assets.for_each_part(body.0, body.1, |_, part, at, turn| {
                let limit = best.map_or(max, |found| found.0);
                let rotation = DQuat::from_quat(turn);
                let inverse = rotation.conjugate();
                let start = inverse.mul_vec3(offset - DVec3::from_vec3(at));
                let along = inverse.mul_vec3(direction);
                let found = match assets.convex(part) {
                    Some(hull) => hull.ray(start, along, limit),
                    None => ray_shape(part, start, along, limit),
                };
                let Some((time, normal)) = found else {
                    return;
                };
                let normal = rotation.mul_vec3(normal);
                if best.is_none_or(|found| (time, handle) < (found.0, found.1)) {
                    best = Some((time, handle, normal));
                }
            });
        }
        best.map(|(time, body, normal)| {
            let point = shifted(ray.origin, direction * time)?;
            Ok(CastHit {
                body,
                distance: time as f32,
                point,
                normal: normal.as_vec3().normalize(),
            })
        })
        .transpose()
    }

    /// Nearest hit along the sweep, the lower handle wins a tie.
    ///
    /// # Errors
    /// Fails on a bad shape, an unknown frame, non finite input or a zero direction.
    pub fn shape_cast(
        &self,
        cast: &ShapeCast,
        filter: &QueryFilter,
    ) -> Result<Option<CastHit>, PhysicsError> {
        check_query_shape(&cast.shape)?;
        let rotation = checked_rotation(cast.rotation)?;
        let rotation_in = rotation;
        let direction = self.check_cast(cast.origin, cast.direction, cast.max_distance, filter)?;
        self.bodies.assets.validate(&cast.shape)?;
        let (moving, radius) = self
            .bodies
            .assets
            .hull_of(&cast.shape, rotation, DVec3::ZERO);
        let own_reach = f64::from(self.bodies.assets.bounding_radius(&cast.shape));
        let max = f64::from(cast.max_distance);
        let mut best: Option<(f64, BodyHandle, DVec3, DVec3)> = None;
        for dense in 0..self.bodies.len() {
            if !self.visible(dense, filter) {
                continue;
            }
            let center = self.bodies.position(dense).offset_from(cast.origin);
            let limit = best.map_or(max, |found| found.0);
            let assets = &self.bodies.assets;
            let reach = own_reach + f64::from(assets.bounding_radius(&self.bodies.shape[dense]));
            if !sphere_near_path(-center, direction, limit, reach) {
                continue;
            }
            let handle = self.bodies.handle_at(dense);
            let body = (&self.bodies.shape[dense], self.bodies.rotation[dense]);
            if let Some(mesh) = assets.mesh(body.0) {
                let rotation = DQuat::from_quat(body.1);
                let inverse = rotation.conjugate();
                let start = inverse.mul_vec3(-center);
                let along = inverse.mul_vec3(direction);
                let (local, local_radius) =
                    assets.hull_of(&cast.shape, body.1.conjugate() * rotation_in, start);
                let tip = (start + along * limit).as_vec3();
                let area = Aabb::around(start.as_vec3(), own_reach as f32)
                    .union(Aabb::around(tip, own_reach as f32));
                mesh.for_each_near(area, |index| {
                    let limit = best.map_or(max, |found| found.0);
                    let [a, b, c] = mesh.triangle(index);
                    let triangle = Hull::Triangle(
                        DVec3::from_vec3(a),
                        DVec3::from_vec3(b),
                        DVec3::from_vec3(c),
                    );
                    if let Some(hit) = sweep(&local, local_radius, &triangle, 0.0, along, limit)
                        && best.is_none_or(|found| (hit.0, handle) < (found.0, found.1))
                    {
                        let normal = rotation.mul_vec3(hit.1);
                        let point = rotation.mul_vec3(hit.2) + center;
                        best = Some((hit.0, handle, normal, point));
                    }
                });
                continue;
            }
            assets.for_each_part(body.0, body.1, |_, part, at, turn| {
                let limit = best.map_or(max, |found| found.0);
                let (target, target_radius) =
                    assets.hull_of(part, turn, center + DVec3::from_vec3(at));
                let Some(hit) = sweep(&moving, radius, &target, target_radius, direction, limit)
                else {
                    return;
                };
                if best.is_none_or(|found| (hit.0, handle) < (found.0, found.1)) {
                    best = Some((hit.0, handle, hit.1, hit.2));
                }
            });
        }
        best.map(|(time, body, normal, point)| {
            Ok(CastHit {
                body,
                distance: time as f32,
                point: shifted(cast.origin, point)?,
                normal: normal.as_vec3().normalize(),
            })
        })
        .transpose()
    }

    /// Fills `out` with every body that collides with the shape, sorted by handle.
    ///
    /// # Errors
    /// Fails on a bad shape, an unknown frame or non finite input.
    pub fn overlap(
        &self,
        shape: &Shape,
        position: GridPosition,
        rotation: Quat,
        filter: &QueryFilter,
        out: &mut Vec<BodyHandle>,
    ) -> Result<(), PhysicsError> {
        out.clear();
        check_query_shape(shape)?;
        let rotation = checked_rotation(rotation)?;
        let position = checked_position(position)?;
        self.check_frame(filter)?;
        self.bodies.assets.validate(shape)?;
        let own_reach = f64::from(self.bodies.assets.bounding_radius(shape));
        for dense in 0..self.bodies.len() {
            if !self.visible(dense, filter) {
                continue;
            }
            let center = self.bodies.position(dense).offset_from(position);
            let assets = &self.bodies.assets;
            let reach = own_reach + f64::from(assets.bounding_radius(&self.bodies.shape[dense]));
            if center.length_squared() > reach * reach {
                continue;
            }
            let touching = assets.overlapping(
                (shape, rotation),
                (&self.bodies.shape[dense], self.bodies.rotation[dense]),
                center.as_vec3(),
            );
            if touching {
                out.push(self.bodies.handle_at(dense));
            }
        }
        out.sort_unstable();
        Ok(())
    }

    /// Closest surface within `max_distance`. A point inside a body always counts.
    ///
    /// # Errors
    /// Fails on an unknown frame, non finite input or a negative distance.
    pub fn nearest_point(
        &self,
        position: GridPosition,
        max_distance: f32,
        filter: &QueryFilter,
    ) -> Result<Option<PointHit>, PhysicsError> {
        let position = checked_position(position)?;
        self.check_frame(filter)?;
        check_max_distance(max_distance)?;
        let mut best: Option<(f64, BodyHandle, DVec3)> = None;
        for dense in 0..self.bodies.len() {
            if !self.visible(dense, filter) {
                continue;
            }
            let offset = position.offset_from(self.bodies.position(dense));
            let limit = best.map_or(f64::from(max_distance), |found| found.0);
            let assets = &self.bodies.assets;
            let reach = f64::from(assets.bounding_radius(&self.bodies.shape[dense]));
            if offset.length() - reach > limit {
                continue;
            }
            let handle = self.bodies.handle_at(dense);
            let body = (&self.bodies.shape[dense], self.bodies.rotation[dense]);
            if let Some(mesh) = assets.mesh(body.0) {
                let rotation = DQuat::from_quat(body.1);
                let local = rotation.conjugate().mul_vec3(offset).as_vec3();
                let area = Aabb::around(local, limit as f32);
                mesh.for_each_near(area, |index| {
                    let limit = best.map_or(f64::from(max_distance), |found| found.0);
                    let (gap, normal) = point_triangle(local, &mesh.triangle(index));
                    let gap = f64::from(gap);
                    if gap <= limit && best.is_none_or(|found| (gap, handle) < (found.0, found.1)) {
                        let normal = rotation.mul_vec3(DVec3::from_vec3(normal));
                        best = Some((gap, handle, normal));
                    }
                });
                continue;
            }
            assets.for_each_part(body.0, body.1, |_, part, at, turn| {
                let limit = best.map_or(f64::from(max_distance), |found| found.0);
                let rotation = DQuat::from_quat(turn);
                let local = rotation.conjugate().mul_vec3(offset - DVec3::from_vec3(at));
                let (gap, normal) = match assets.convex(part) {
                    Some(hull) => hull.point_distance(local),
                    None => surface_distance(part, local),
                };
                if gap > limit {
                    return;
                }
                let normal = rotation.mul_vec3(normal);
                if best.is_none_or(|found| (gap, handle) < (found.0, found.1)) {
                    best = Some((gap, handle, normal));
                }
            });
        }
        best.map(|(gap, body, normal)| {
            Ok(PointHit {
                body,
                distance: gap as f32,
                point: shifted(position, normal * -gap)?,
                normal: normal.as_vec3().normalize(),
            })
        })
        .transpose()
    }

    /// The overlap to push out of first: unit normal toward `shape` and depth. `None` when free.
    pub(crate) fn deepest_overlap(
        &self,
        shape: &Shape,
        position: GridPosition,
        filter: &QueryFilter,
    ) -> Result<Option<(Vec3, f32)>, PhysicsError> {
        check_query_shape(shape)?;
        let position = checked_position(position)?;
        self.check_frame(filter)?;
        let own_reach = f64::from(shape.bounding_radius());
        let mut deepest: Option<(Vec3, f32)> = None;
        for dense in 0..self.bodies.len() {
            if !self.visible(dense, filter) {
                continue;
            }
            let assets = &self.bodies.assets;
            let center = position.offset_from(self.bodies.position(dense));
            let reach = own_reach + f64::from(assets.bounding_radius(&self.bodies.shape[dense]));
            if center.length_squared() > reach * reach {
                continue;
            }
            assets.for_each_contact(
                (&self.bodies.shape[dense], self.bodies.rotation[dense]),
                (shape, Quat::IDENTITY),
                center.as_vec3(),
                0.0,
                None,
                |_, _, raw, _| {
                    for point in &raw.points[..raw.count] {
                        if point.depth > 0.0 && deepest.is_none_or(|found| point.depth > found.1) {
                            deepest = Some((raw.normal, point.depth));
                        }
                    }
                },
            );
        }
        Ok(deepest)
    }

    fn visible(&self, dense: usize, filter: &QueryFilter) -> bool {
        self.bodies.frame[dense] == filter.frame
            && self.bodies.filter[dense].layers & filter.mask != 0
            && filter.exclude != Some(self.bodies.handle_at(dense))
    }

    fn check_frame(&self, filter: &QueryFilter) -> Result<(), PhysicsError> {
        if self.frames.contains(filter.frame) {
            Ok(())
        } else {
            Err(PhysicsError::InvalidFrame)
        }
    }

    fn check_cast(
        &self,
        origin: GridPosition,
        direction: Vec3,
        max_distance: f32,
        filter: &QueryFilter,
    ) -> Result<DVec3, PhysicsError> {
        self.check_frame(filter)?;
        checked_position(origin)?;
        check_max_distance(max_distance)?;
        if !direction.is_finite() {
            return Err(PhysicsError::NonFinite("direction"));
        }
        let wide = DVec3::from_vec3(direction);
        if wide.length_squared() < 1e-24 {
            return Err(PhysicsError::NonFinite("direction"));
        }
        Ok(wide.normalize())
    }
}

fn check_query_shape(shape: &Shape) -> Result<(), PhysicsError> {
    if matches!(shape, Shape::Compound(_) | Shape::TriMesh(_)) {
        return Err(PhysicsError::InvalidShape);
    }
    shape.validate()
}

fn check_max_distance(max_distance: f32) -> Result<(), PhysicsError> {
    if max_distance.is_finite() && max_distance >= 0.0 {
        Ok(())
    } else {
        Err(PhysicsError::NonFinite("max distance"))
    }
}

fn checked_position(position: GridPosition) -> Result<GridPosition, PhysicsError> {
    if !position.local.is_finite() {
        return Err(PhysicsError::NonFinite("position"));
    }
    position.rebased().ok_or(PhysicsError::OutOfRange)
}

fn checked_rotation(rotation: Quat) -> Result<Quat, PhysicsError> {
    let squared = rotation.length_squared();
    if !squared.is_finite() || squared < 1e-12 {
        return Err(PhysicsError::NonFinite("rotation"));
    }
    Ok(rotation.normalize())
}

fn shifted(origin: GridPosition, offset: DVec3) -> Result<GridPosition, PhysicsError> {
    origin.shifted(offset).ok_or(PhysicsError::OutOfRange)
}

/// False when a sphere of `reach` around the origin can not touch the path from `start`.
fn sphere_near_path(start: DVec3, direction: DVec3, limit: f64, reach: f64) -> bool {
    let along = (-start.dot(direction)).min(limit).max(0.0);
    (start + direction * along).length_squared() <= reach * reach
}

/// Ray in shape axes. Returns time and normal, also in shape axes.
fn ray_shape(shape: &Shape, origin: DVec3, direction: DVec3, max: f64) -> Option<(f64, DVec3)> {
    match *shape {
        Shape::Sphere { radius } => ray_capsule(origin, direction, 0.0, f64::from(radius), max),
        Shape::Capsule {
            radius,
            half_height,
        } => ray_capsule(
            origin,
            direction,
            f64::from(half_height),
            f64::from(radius),
            max,
        ),
        Shape::Cuboid { half_extents } => ray_box(origin, direction, half_extents, max),
        Shape::Compound(_) | Shape::TriMesh(_) | Shape::Convex(_) => None,
    }
}

fn ray_capsule(
    origin: DVec3,
    direction: DVec3,
    half: f64,
    radius: f64,
    max: f64,
) -> Option<(f64, DVec3)> {
    let core = |y: f64| DVec3::new(0.0, y.min(half).max(-half), 0.0);
    if (origin - core(origin.y)).length_squared() <= radius * radius {
        return Some((0.0, -direction));
    }
    let mut time = f64::INFINITY;
    let flat = direction.x * direction.x + direction.z * direction.z;
    if flat > PARALLEL_SQUARED {
        let b = origin.x * direction.x + origin.z * direction.z;
        let c = origin.x * origin.x + origin.z * origin.z - radius * radius;
        let root = b * b - flat * c;
        if root >= 0.0 {
            let t = (-b - root.sqrt()) / flat;
            if t >= 0.0 && (origin.y + direction.y * t).abs() <= half {
                time = t;
            }
        }
    }
    for y in [-half, half] {
        let from_cap = origin - DVec3::new(0.0, y, 0.0);
        let b = from_cap.dot(direction);
        let root = b * b - (from_cap.length_squared() - radius * radius);
        if root >= 0.0 {
            let t = -b - root.sqrt();
            if t >= 0.0 && t < time {
                time = t;
            }
        }
    }
    if time > max {
        return None;
    }
    let point = origin + direction * time;
    Some((time, (point - core(point.y)).normalize()))
}

fn ray_box(origin: DVec3, direction: DVec3, extents: Vec3, max: f64) -> Option<(f64, DVec3)> {
    let axes = [DVec3::X, DVec3::Y, DVec3::Z];
    let origins = [origin.x, origin.y, origin.z];
    let directions = [direction.x, direction.y, direction.z];
    let reaches = [
        f64::from(extents.x),
        f64::from(extents.y),
        f64::from(extents.z),
    ];
    if (0..3).all(|i| origins[i].abs() <= reaches[i]) {
        return Some((0.0, -direction));
    }
    let (mut enter, mut leave) = (0.0_f64, max);
    let mut normal = -direction;
    for i in 0..3 {
        if directions[i].abs() < 1e-14 {
            if origins[i].abs() > reaches[i] {
                return None;
            }
            continue;
        }
        let side = if directions[i] > 0.0 { -1.0 } else { 1.0 };
        let near = (side * reaches[i] - origins[i]) / directions[i];
        let far = (-side * reaches[i] - origins[i]) / directions[i];
        if near > enter {
            enter = near;
            normal = axes[i] * side;
        }
        leave = leave.min(far);
        if enter > leave {
            return None;
        }
    }
    Some((enter, normal))
}

/// Conservative advancement. Hulls can close no faster than the speed along their separation.
/// Returns time, normal facing the mover, and the contact relative to the cast origin.
pub(crate) fn sweep(
    moving: &Hull,
    moving_radius: f64,
    target: &Hull,
    target_radius: f64,
    direction: DVec3,
    max: f64,
) -> Option<(f64, DVec3, DVec3)> {
    let radii = moving_radius + target_radius;
    let mut time = 0.0;
    let mut last: Option<(DVec3, DVec3)> = None;
    for _ in 0..MAX_CAST_ITERATIONS {
        let separation = distance(&moving.moved(direction * time), target);
        let gap = separation.distance - radii;
        if separation.direction == DVec3::ZERO || gap <= CAST_SKIN {
            return Some(match last {
                Some((normal, point)) if separation.direction == DVec3::ZERO => {
                    (time, normal, point)
                }
                _ if separation.direction == DVec3::ZERO || gap <= 0.0 && time == 0.0 => {
                    (time, -direction, direction * time)
                }
                _ => (
                    time,
                    separation.direction,
                    separation.on_b + separation.direction * target_radius,
                ),
            });
        }
        let closing = -direction.dot(separation.direction);
        if closing <= 1e-12 {
            return None;
        }
        last = Some((
            separation.direction,
            separation.on_b + separation.direction * target_radius,
        ));
        time += gap / closing;
        if time > max {
            return None;
        }
    }
    // out of iterations still closing, reporting the hit is the safe side
    last.map(|(normal, point)| (time, normal, point))
}

/// Signed distance to the surface and the outward normal, both in shape axes.
fn surface_distance(shape: &Shape, point: DVec3) -> (f64, DVec3) {
    match *shape {
        Shape::Sphere { radius } => rounded_distance(point, 0.0, f64::from(radius)),
        Shape::Capsule {
            radius,
            half_height,
        } => rounded_distance(point, f64::from(half_height), f64::from(radius)),
        Shape::Cuboid { half_extents } => box_distance(point, half_extents),
        Shape::Compound(_) | Shape::TriMesh(_) | Shape::Convex(_) => (f64::INFINITY, DVec3::Y),
    }
}

fn rounded_distance(point: DVec3, half: f64, radius: f64) -> (f64, DVec3) {
    let gap = point - DVec3::new(0.0, point.y.min(half).max(-half), 0.0);
    let length = gap.length();
    if length > 1e-12 {
        (length - radius, gap / length)
    } else if half > 0.0 {
        (-radius, DVec3::X)
    } else {
        (-radius, DVec3::Y)
    }
}

fn box_distance(point: DVec3, extents: Vec3) -> (f64, DVec3) {
    let reaches = [
        f64::from(extents.x),
        f64::from(extents.y),
        f64::from(extents.z),
    ];
    let values = [point.x, point.y, point.z];
    let axes = [DVec3::X, DVec3::Y, DVec3::Z];
    let mut outside = DVec3::ZERO;
    for i in 0..3 {
        let beyond = values[i].abs() - reaches[i];
        if beyond > 0.0 {
            outside += axes[i] * beyond.copysign(values[i]);
        }
    }
    let length = outside.length();
    if length > 0.0 {
        return (length, outside / length);
    }
    let mut nearest = 0;
    for i in 1..3 {
        if reaches[i] - values[i].abs() < reaches[nearest] - values[nearest].abs() {
            nearest = i;
        }
    }
    let side = if values[nearest] < 0.0 { -1.0 } else { 1.0 };
    (
        values[nearest].abs() - reaches[nearest],
        axes[nearest] * side,
    )
}

#[cfg(test)]
mod tests {
    use algebrix::IVec3;

    use super::*;
    use crate::{CollisionFilter, FrameDesc, PhysicsSettings, RigidBodyDesc};

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    fn at(x: f32, y: f32, z: f32) -> GridPosition {
        GridPosition::from_local(Vec3::new(x, y, z)).unwrap()
    }

    fn fixed(world: &mut PhysicsWorld, shape: Shape, x: f32, y: f32, z: f32) -> BodyHandle {
        world
            .spawn(RigidBodyDesc::fixed(shape).local_position(Vec3::new(x, y, z)))
            .unwrap()
    }

    fn ray(origin: GridPosition, direction: Vec3) -> Ray {
        Ray {
            origin,
            direction,
            max_distance: 1000.0,
        }
    }

    fn cast(shape: Shape, origin: GridPosition, direction: Vec3) -> ShapeCast {
        ShapeCast {
            shape,
            rotation: Quat::IDENTITY,
            origin,
            direction,
            max_distance: 1000.0,
        }
    }

    fn near(a: f32, b: f32, tolerance: f32) {
        assert!((a - b).abs() <= tolerance, "{a} != {b}");
    }

    fn near_vec(a: Vec3, b: Vec3) {
        assert!((a - b).length() < 1e-3, "{a:?} != {b:?}");
    }

    #[test]
    fn ray_hits_a_sphere_at_the_analytic_distance() {
        let mut world = world();
        let ball = fixed(&mut world, Shape::sphere(2.0), 10.0, 0.0, 0.0);
        let hit = world
            .raycast(&ray(at(0.0, 0.0, 0.0), Vec3::X), &QueryFilter::default())
            .unwrap()
            .unwrap();
        assert_eq!(hit.body, ball);
        near(hit.distance, 8.0, 1e-5);
        near_vec(hit.normal, -Vec3::X);
        near_vec(hit.point.local, Vec3::new(8.0, 0.0, 0.0));
    }

    #[test]
    fn ray_off_center_uses_the_chord() {
        let mut world = world();
        fixed(&mut world, Shape::sphere(5.0), 0.0, 0.0, 0.0);
        let hit = world
            .raycast(&ray(at(-10.0, 3.0, 0.0), Vec3::X), &QueryFilter::default())
            .unwrap()
            .unwrap();
        near(hit.distance, 6.0, 1e-5);
        near_vec(hit.normal, Vec3::new(-0.8, 0.6, 0.0));
    }

    #[test]
    fn ray_misses_and_respects_max_distance() {
        let mut world = world();
        fixed(&mut world, Shape::sphere(1.0), 10.0, 0.0, 0.0);
        let filter = QueryFilter::default();
        let world = world;
        let beside = ray(at(0.0, 2.0, 0.0), Vec3::X);
        assert_eq!(world.raycast(&beside, &filter).unwrap(), None);
        let behind = ray(at(0.0, 0.0, 0.0), -Vec3::X);
        assert_eq!(world.raycast(&behind, &filter).unwrap(), None);
        let short = Ray {
            max_distance: 8.5,
            ..ray(at(0.0, 0.0, 0.0), Vec3::X)
        };
        assert_eq!(world.raycast(&short, &filter).unwrap(), None);
        let enough = Ray {
            max_distance: 9.5,
            ..short
        };
        assert!(world.raycast(&enough, &filter).unwrap().is_some());
    }

    #[test]
    fn ray_hits_capsule_side_and_cap() {
        let mut world = world();
        fixed(&mut world, Shape::capsule(1.0, 2.0), 0.0, 0.0, 0.0);
        let filter = QueryFilter::default();
        let side = world
            .raycast(&ray(at(-5.0, 1.5, 0.0), Vec3::X), &filter)
            .unwrap()
            .unwrap();
        near(side.distance, 4.0, 1e-5);
        near_vec(side.normal, -Vec3::X);
        let top = world
            .raycast(&ray(at(0.0, 10.0, 0.0), -Vec3::Y), &filter)
            .unwrap()
            .unwrap();
        near(top.distance, 7.0, 1e-5);
        near_vec(top.normal, Vec3::Y);
        let cap_edge = world
            .raycast(&ray(at(-5.0, 2.6, 0.0), Vec3::X), &filter)
            .unwrap()
            .unwrap();
        near(cap_edge.distance, 4.2, 1e-5);
        near_vec(cap_edge.normal, Vec3::new(-0.8, 0.6, 0.0));
    }

    #[test]
    fn ray_hits_turned_box_on_its_corner() {
        let mut world = world();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::splat(1.0)))
                    .rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4)),
            )
            .unwrap();
        let hit = world
            .raycast(&ray(at(-5.0, 0.0, 0.0), Vec3::X), &QueryFilter::default())
            .unwrap()
            .unwrap();
        near(hit.distance, 5.0 - 2.0_f32.sqrt(), 1e-5);
    }

    #[test]
    fn ray_starting_inside_reports_zero() {
        let mut world = world();
        fixed(&mut world, Shape::cuboid(Vec3::splat(2.0)), 0.0, 0.0, 0.0);
        let hit = world
            .raycast(&ray(at(0.5, 0.0, 0.0), Vec3::Y), &QueryFilter::default())
            .unwrap()
            .unwrap();
        assert_eq!(hit.distance, 0.0);
        near_vec(hit.normal, -Vec3::Y);
    }

    #[test]
    fn nearest_body_and_lower_handle_win() {
        let mut world = world();
        let far = fixed(&mut world, Shape::sphere(1.0), 20.0, 0.0, 0.0);
        let close = fixed(&mut world, Shape::sphere(1.0), 10.0, 0.0, 0.0);
        let twin = fixed(&mut world, Shape::sphere(1.0), 10.0, 0.0, 0.0);
        let filter = QueryFilter::default();
        let origin = ray(at(0.0, 0.0, 0.0), Vec3::X);
        assert_eq!(
            world.raycast(&origin, &filter).unwrap().unwrap().body,
            close
        );
        world.despawn(close).unwrap();
        assert_eq!(world.raycast(&origin, &filter).unwrap().unwrap().body, twin);
        world.despawn(twin).unwrap();
        assert_eq!(world.raycast(&origin, &filter).unwrap().unwrap().body, far);
    }

    #[test]
    fn filter_mask_exclude_and_frame_hide_bodies() {
        let mut world = world();
        let other = world.add_frame(FrameDesc::new(FrameId::ROOT)).unwrap();
        let layered = world
            .spawn(
                RigidBodyDesc::fixed(Shape::sphere(1.0))
                    .filter(CollisionFilter {
                        layers: 0b10,
                        mask: 0,
                    })
                    .local_position(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();
        let beyond = world
            .spawn(
                RigidBodyDesc::fixed(Shape::sphere(1.0)).local_position(Vec3::new(10.0, 0.0, 0.0)),
            )
            .unwrap();
        let in_frame = world
            .spawn(
                RigidBodyDesc::fixed(Shape::sphere(1.0))
                    .frame(other)
                    .local_position(Vec3::new(15.0, 0.0, 0.0)),
            )
            .unwrap();
        let line = ray(at(0.0, 0.0, 0.0), Vec3::X);
        let find = |filter: QueryFilter| world.raycast(&line, &filter).unwrap().unwrap().body;
        assert_eq!(find(QueryFilter::default()), layered);
        let low = QueryFilter {
            mask: 0b01,
            ..QueryFilter::default()
        };
        assert_eq!(find(low), beyond);
        let skip = QueryFilter {
            exclude: Some(layered),
            ..QueryFilter::default()
        };
        assert_eq!(find(skip), beyond);
        let framed = QueryFilter {
            frame: other,
            ..QueryFilter::default()
        };
        assert_eq!(find(framed), in_frame);
    }

    #[test]
    fn bad_input_fails() {
        let world = world();
        let filter = QueryFilter::default();
        for bad in [
            ray(at(0.0, 0.0, 0.0), Vec3::ZERO),
            ray(at(0.0, 0.0, 0.0), Vec3::new(f32::NAN, 0.0, 0.0)),
            ray(
                GridPosition {
                    cell: IVec3::new(0, 0, 0),
                    local: Vec3::new(f32::NAN, 0.0, 0.0),
                },
                Vec3::X,
            ),
            Ray {
                max_distance: -1.0,
                ..ray(at(0.0, 0.0, 0.0), Vec3::X)
            },
            Ray {
                max_distance: f32::NAN,
                ..ray(at(0.0, 0.0, 0.0), Vec3::X)
            },
        ] {
            assert!(world.raycast(&bad, &filter).is_err(), "{bad:?}");
        }
        let unknown = QueryFilter {
            frame: FrameId(9),
            ..QueryFilter::default()
        };
        assert_eq!(
            world.raycast(&ray(at(0.0, 0.0, 0.0), Vec3::X), &unknown),
            Err(PhysicsError::InvalidFrame)
        );
        let bad_shape = cast(Shape::sphere(-1.0), at(0.0, 0.0, 0.0), Vec3::X);
        assert_eq!(
            world.shape_cast(&bad_shape, &filter),
            Err(PhysicsError::InvalidShape)
        );
        assert!(
            world
                .nearest_point(at(0.0, 0.0, 0.0), -1.0, &filter)
                .is_err()
        );
    }

    #[test]
    fn queries_work_far_from_the_origin_and_across_cells() {
        let mut world = world();
        let cell = IVec3::new(2_000_000, -700_000, 1_000_000);
        let origin = GridPosition {
            cell,
            local: Vec3::new(500.0, 0.0, 0.0),
        };
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::sphere(1.0)).position(GridPosition {
                    cell: IVec3::new(cell.x + 1, cell.y, cell.z),
                    local: Vec3::ZERO,
                }),
            )
            .unwrap();
        let filter = QueryFilter::default();
        let hit = world
            .raycast(&ray(origin, Vec3::X), &filter)
            .unwrap()
            .unwrap();
        // 1024 - 500 - 1
        assert_eq!(hit.distance, 523.0);
        assert_eq!(hit.point.cell.x, cell.x + 1);
        assert_eq!(hit.point.local.x, -1.0);
        let swept = world
            .shape_cast(&cast(Shape::sphere(0.5), origin, Vec3::X), &filter)
            .unwrap()
            .unwrap();
        near(swept.distance, 522.5, 1e-3);
        let nearest = world
            .nearest_point(origin, 1000.0, &filter)
            .unwrap()
            .unwrap();
        near(nearest.distance, 523.0, 1e-5);
    }

    #[test]
    fn sphere_cast_matches_the_inflated_shape() {
        let mut world = world();
        let cube = fixed(&mut world, Shape::cuboid(Vec3::splat(1.0)), 10.0, 0.0, 0.0);
        let hit = world
            .shape_cast(
                &cast(Shape::sphere(0.5), at(0.0, 0.0, 0.0), Vec3::X),
                &QueryFilter::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(hit.body, cube);
        near(hit.distance, 8.5, 2e-4);
        near_vec(hit.normal, -Vec3::X);
        near_vec(hit.point.local, Vec3::new(9.0, 0.0, 0.0));
    }

    #[test]
    fn box_cast_lands_on_the_ground() {
        let mut world = world();
        fixed(
            &mut world,
            Shape::cuboid(Vec3::new(500.0, 1.0, 500.0)),
            0.0,
            -1.0,
            0.0,
        );
        let hit = world
            .shape_cast(
                &cast(
                    Shape::cuboid(Vec3::splat(0.5)),
                    at(3.0, 10.0, -4.0),
                    -Vec3::Y,
                ),
                &QueryFilter::default(),
            )
            .unwrap()
            .unwrap();
        near(hit.distance, 9.5, 2e-4);
        near_vec(hit.normal, Vec3::Y);
        near(hit.point.local.y, 0.0, 1e-3);
    }

    #[test]
    fn capsule_cast_clears_a_gap_it_fits_through() {
        let mut world = world();
        fixed(&mut world, Shape::sphere(1.0), 10.0, 3.0, 0.0);
        fixed(&mut world, Shape::sphere(1.0), 10.0, -3.0, 0.0);
        let filter = QueryFilter::default();
        let capsule = Shape::capsule(0.5, 1.0);
        let fits = cast(capsule, at(0.0, 0.0, 0.0), Vec3::X);
        assert_eq!(world.shape_cast(&fits, &filter).unwrap(), None);
        let wide = cast(Shape::capsule(1.5, 1.0), at(0.0, 0.0, 0.0), Vec3::X);
        assert!(world.shape_cast(&wide, &filter).unwrap().is_some());
    }

    #[test]
    fn cast_starting_inside_reports_zero() {
        let mut world = world();
        fixed(&mut world, Shape::sphere(2.0), 0.0, 0.0, 0.0);
        let hit = world
            .shape_cast(
                &cast(Shape::sphere(0.5), at(1.0, 0.0, 0.0), Vec3::X),
                &QueryFilter::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(hit.distance, 0.0);
        near_vec(hit.normal, -Vec3::X);
    }

    #[test]
    fn cast_moving_away_misses() {
        let mut world = world();
        fixed(&mut world, Shape::sphere(1.0), 0.0, 0.0, 0.0);
        let away = cast(Shape::sphere(0.5), at(5.0, 0.0, 0.0), Vec3::X);
        assert_eq!(
            world.shape_cast(&away, &QueryFilter::default()).unwrap(),
            None
        );
    }

    #[test]
    fn random_casts_stop_right_at_first_contact() {
        let mut state = 0x7777_1234_abcd_0001_u64;
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
        let filter = QueryFilter::default();
        let mut hits = 0;
        for _ in 0..4000 {
            let target = shapes[(next().abs() * 2.99) as usize];
            let mover = shapes[(next().abs() * 2.99) as usize];
            let target_rotation = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let rotation = Quat::new(next(), next(), next(), next() + 2.0).normalize();
            let mut world = world();
            let center = Vec3::new(next(), next(), next()) * 5.0;
            world
                .spawn(
                    RigidBodyDesc::fixed(target)
                        .rotation(target_rotation)
                        .local_position(center),
                )
                .unwrap();
            let start = Vec3::new(next(), next(), next()) * 8.0;
            let direction = (center - start).normalize() + Vec3::new(next(), next(), next()) * 0.2;
            let sweep = ShapeCast {
                shape: mover,
                rotation,
                origin: at(start.x, start.y, start.z),
                direction,
                max_distance: 30.0,
            };
            let unit = direction.normalize();
            let touches = |distance: f32| {
                let place = start + unit * distance;
                let mut out = Vec::new();
                world
                    .overlap(
                        &mover,
                        at(place.x, place.y, place.z),
                        rotation,
                        &filter,
                        &mut out,
                    )
                    .unwrap();
                !out.is_empty()
            };
            match world.shape_cast(&sweep, &filter).unwrap() {
                Some(hit) if hit.distance > 0.0 => {
                    hits += 1;
                    assert!(!touches(hit.distance - 2e-3), "{sweep:?} {hit:?}");
                    // a graze can leave again right after, so check the gap, not a later overlap
                    let place = DVec3::from_vec3(start + unit * hit.distance);
                    let (a, radius_a) = Hull::of(&mover, rotation, place);
                    let (b, radius_b) =
                        Hull::of(&target, target_rotation, DVec3::from_vec3(center));
                    let gap = distance(&a, &b).distance - radius_a - radius_b;
                    assert!(gap.abs() < 1e-3, "{sweep:?} {hit:?} gap {gap}");
                    assert!((hit.normal.length() - 1.0).abs() < 1e-3);
                }
                Some(_) => assert!(touches(0.0), "{sweep:?}"),
                None => {
                    for step in 0..=120 {
                        assert!(!touches(step as f32 * 0.25), "{sweep:?} at {step}");
                    }
                }
            }
        }
        assert!(hits > 500, "{hits}");
    }

    #[test]
    fn a_near_miss_is_not_an_overlap() {
        let mut world = world();
        let floor = Shape::cuboid(Vec3::new(5.0, 0.5, 5.0));
        let body = RigidBodyDesc::fixed(floor).local_position(Vec3::new(0.0, -0.5, 0.0));
        world.spawn(body).unwrap();
        let shape = Shape::cuboid(Vec3::splat(1.0));
        let mut out = Vec::new();
        let filter = QueryFilter::default();
        // a centimeter up is inside the narrowphase's look ahead margin, still not touching
        world
            .overlap(
                &shape,
                at(0.0, 1.01, 0.0),
                Quat::IDENTITY,
                &filter,
                &mut out,
            )
            .unwrap();
        assert!(out.is_empty());
        world
            .overlap(
                &shape,
                at(0.0, 0.99, 0.0),
                Quat::IDENTITY,
                &filter,
                &mut out,
            )
            .unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn overlap_lists_touching_bodies_in_handle_order() {
        let mut world = world();
        let a = fixed(&mut world, Shape::sphere(1.0), 0.0, 0.0, 0.0);
        let b = fixed(&mut world, Shape::cuboid(Vec3::splat(1.0)), 2.5, 0.0, 0.0);
        fixed(&mut world, Shape::sphere(1.0), 9.0, 0.0, 0.0);
        world.despawn(a).unwrap();
        let c = fixed(&mut world, Shape::sphere(1.0), 0.0, 0.0, 0.0);
        let mut out = vec![c];
        world
            .overlap(
                &Shape::sphere(1.0),
                at(1.0, 0.0, 0.0),
                Quat::IDENTITY,
                &QueryFilter::default(),
                &mut out,
            )
            .unwrap();
        let mut expected = vec![b, c];
        expected.sort_unstable();
        assert_eq!(out, expected);
    }

    #[test]
    fn nearest_point_outside_and_inside() {
        let mut world = world();
        let cube = fixed(&mut world, Shape::cuboid(Vec3::splat(1.0)), 0.0, 0.0, 0.0);
        let filter = QueryFilter::default();
        let outside = world
            .nearest_point(at(4.0, 5.0, 0.0), 100.0, &filter)
            .unwrap()
            .unwrap();
        assert_eq!(outside.body, cube);
        near(outside.distance, 5.0, 1e-5);
        near_vec(outside.normal, Vec3::new(0.6, 0.8, 0.0));
        near_vec(outside.point.local, Vec3::new(1.0, 1.0, 0.0));
        let inside = world
            .nearest_point(at(0.75, 0.0, 0.0), 0.0, &filter)
            .unwrap()
            .unwrap();
        near(inside.distance, -0.25, 1e-5);
        near_vec(inside.normal, Vec3::X);
        near_vec(inside.point.local, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(
            world
                .nearest_point(at(4.0, 5.0, 0.0), 4.0, &filter)
                .unwrap(),
            None
        );
    }

    #[test]
    fn nearest_point_on_capsule_and_at_its_center() {
        let mut world = world();
        fixed(&mut world, Shape::capsule(1.0, 2.0), 0.0, 0.0, 0.0);
        let filter = QueryFilter::default();
        let side = world
            .nearest_point(at(3.0, 1.0, 0.0), 100.0, &filter)
            .unwrap()
            .unwrap();
        near(side.distance, 2.0, 1e-5);
        let top = world
            .nearest_point(at(0.0, 5.0, 0.0), 100.0, &filter)
            .unwrap()
            .unwrap();
        near(top.distance, 2.0, 1e-5);
        let center = world
            .nearest_point(at(0.0, 0.0, 0.0), 100.0, &filter)
            .unwrap()
            .unwrap();
        near(center.distance, -1.0, 1e-5);
        near(center.normal.length(), 1.0, 1e-5);
    }

    #[test]
    fn rays_agree_with_nearest_point_on_random_poses() {
        let mut state = 0x1357_9bdf_0246_8ace_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 1000.0 - 1.0
        };
        let shapes = [
            Shape::sphere(1.0),
            Shape::capsule(0.5, 1.0),
            Shape::cuboid(Vec3::new(1.0, 0.5, 0.75)),
        ];
        let filter = QueryFilter::default();
        let mut hits = 0;
        for _ in 0..5000 {
            let shape = shapes[(next().abs() * 2.99) as usize];
            let mut world = world();
            world
                .spawn(
                    RigidBodyDesc::fixed(shape)
                        .rotation(Quat::new(next(), next(), next(), next() + 2.0).normalize()),
                )
                .unwrap();
            let origin = Vec3::new(next(), next(), next()) * 6.0;
            let line = ray(
                at(origin.x, origin.y, origin.z),
                -origin + Vec3::new(next(), next(), next()) * 3.0,
            );
            if let Some(hit) = world
                .raycast(&line, &filter)
                .unwrap()
                .filter(|hit| hit.distance > 0.0)
            {
                hits += 1;
                let on_surface = world
                    .nearest_point(hit.point, 100.0, &filter)
                    .unwrap()
                    .unwrap();
                assert!(on_surface.distance.abs() < 1e-4, "{line:?} {on_surface:?}");
                if hit.distance > 0.0 {
                    assert!(hit.normal.dot(line.direction) <= 1e-4, "{line:?} {hit:?}");
                }
            }
        }
        assert!(hits > 500, "{hits}");
    }
}
