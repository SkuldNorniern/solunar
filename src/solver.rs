use algebrix::Vec3;

use crate::body::{Bodies, apply_diagonal};
use crate::collision::{Bounds, MAX_CONTACTS, contacts, find_pairs};
use crate::world::Integrated;
use crate::{BodyHandle, BodyKind};

/// Contacts exist this far apart, so a corner lifting a hair keeps its constraint.
pub(crate) const CONTACT_MARGIN: f32 = 0.02;
const VELOCITY_ITERATIONS: usize = 10;
const RECOVERY_ITERATIONS: usize = 10;
const RECOVERY_RATE: f32 = 0.2;
const PENETRATION_SLOP: f32 = 0.005;
const MAX_RECOVERY_SPEED: f32 = 3.0;
const RESTITUTION_THRESHOLD: f32 = 1.0;
const MATCH_DISTANCE_SQUARED: f32 = 0.05 * 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct CachedPoint {
    /// Contact in body A axes, finds the same point next step.
    pub(crate) local_a: Vec3,
    pub(crate) impulse: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CachedManifold {
    pub(crate) a: BodyHandle,
    pub(crate) b: BodyHandle,
    pub(crate) points: [CachedPoint; MAX_CONTACTS],
    pub(crate) count: usize,
}

#[derive(Clone, Copy, Default)]
struct SolverPoint {
    offset_a: Vec3,
    offset_b: Vec3,
    mass: Vec3,
    target: f32,
    recovery: f32,
    recovery_impulse: f32,
    impulse: Vec3,
    local_a: Vec3,
}

#[derive(Clone, Copy)]
struct SolverManifold {
    a: usize,
    b: usize,
    handle_a: BodyHandle,
    handle_b: BodyHandle,
    axes: [Vec3; 3],
    friction: f32,
    points: [SolverPoint; MAX_CONTACTS],
    count: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ContactScratch {
    bounds: Vec<Bounds>,
    order: Vec<u32>,
    pairs: Vec<(u32, u32)>,
    manifolds: Vec<SolverManifold>,
}

impl ContactScratch {
    pub(crate) fn pairs(&self) -> &[(u32, u32)] {
        &self.pairs
    }

    /// Dense pairs that made contact in the last solve.
    pub(crate) fn touching(&self) -> impl Iterator<Item = (usize, usize)> + Clone + '_ {
        self.manifolds
            .iter()
            .map(|manifold| (manifold.a, manifold.b))
    }
}

impl std::fmt::Debug for SolverManifold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolverManifold")
            .field("a", &self.a)
            .field("b", &self.b)
            .field("count", &self.count)
            .finish()
    }
}

pub(crate) fn find_broad_pairs(bodies: &Bodies, scratch: &mut ContactScratch) {
    scratch.bounds.clear();
    for dense in 0..bodies.len() {
        scratch.bounds.push(Bounds::of(
            &bodies.shape[dense],
            bodies.position(dense),
            bodies.rotation[dense],
            CONTACT_MARGIN,
        ));
    }
    find_pairs(&scratch.bounds, &mut scratch.order, &mut scratch.pairs);
}

/// Finds contacts and solves velocities in `state`. `cache` goes in old and comes out new.
/// Pairs with nothing awake are skipped, and keep their cache entry for the wake up.
pub(crate) fn solve_contacts(
    bodies: &Bodies,
    awake: &[bool],
    state: &mut [Integrated],
    cache: &mut Vec<CachedManifold>,
    scratch: &mut ContactScratch,
    dt: f32,
) {
    build_manifolds(bodies, awake, state, cache, scratch, dt);
    for manifold in &scratch.manifolds {
        warm_start(bodies, state, manifold);
    }
    // sweeping both ways keeps a stack from favoring one end, one way it rocks and never settles
    for iteration in 0..VELOCITY_ITERATIONS {
        if iteration % 2 == 0 {
            for manifold in &mut scratch.manifolds {
                solve_manifold(bodies, state, manifold);
            }
        } else {
            for manifold in scratch.manifolds.iter_mut().rev() {
                solve_manifold(bodies, state, manifold);
            }
        }
    }
    for _ in 0..RECOVERY_ITERATIONS {
        for manifold in &mut scratch.manifolds {
            solve_recovery(bodies, state, manifold);
        }
    }
    let active = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic && awake[dense];
    cache.retain(
        |entry| match (bodies.dense_index(entry.a), bodies.dense_index(entry.b)) {
            (Ok(a), Ok(b)) => !active(a) && !active(b),
            _ => false,
        },
    );
    for manifold in &scratch.manifolds {
        let mut points = [CachedPoint::default(); MAX_CONTACTS];
        for (out, point) in points.iter_mut().zip(&manifold.points[..manifold.count]) {
            *out = CachedPoint {
                local_a: point.local_a,
                impulse: point.impulse,
            };
        }
        cache.push(CachedManifold {
            a: manifold.handle_a,
            b: manifold.handle_b,
            points,
            count: manifold.count,
        });
    }
    cache.sort_unstable_by_key(|entry| (entry.a, entry.b));
}

fn build_manifolds(
    bodies: &Bodies,
    awake: &[bool],
    state: &[Integrated],
    cache: &[CachedManifold],
    scratch: &mut ContactScratch,
    dt: f32,
) {
    let active = |dense: usize| bodies.kind[dense] == BodyKind::Dynamic && awake[dense];
    scratch.manifolds.clear();
    for &(first, second) in &scratch.pairs {
        let (mut a, mut b) = (first as usize, second as usize);
        if !active(a) && !active(b) {
            continue;
        }
        if bodies.frame[a] != bodies.frame[b] || !bodies.filter[a].allows(bodies.filter[b]) {
            continue;
        }
        // the lower handle is always A, dense order can change but the cache must still match
        if bodies.handle_at(a) > bodies.handle_at(b) {
            std::mem::swap(&mut a, &mut b);
        }
        let center_b = bodies.position(b).offset_from(bodies.position(a)).as_vec3();
        let Some(raw) = contacts(
            &bodies.shape[a],
            bodies.rotation[a],
            &bodies.shape[b],
            bodies.rotation[b],
            center_b,
            CONTACT_MARGIN,
        ) else {
            continue;
        };
        let (handle_a, handle_b) = (bodies.handle_at(a), bodies.handle_at(b));
        let old = cache
            .binary_search_by_key(&(handle_a, handle_b), |entry| (entry.a, entry.b))
            .ok()
            .map(|at| &cache[at]);

        let normal = raw.normal;
        let tangent = tangent_of(normal);
        let axes = [normal, tangent, normal.cross(tangent)];
        let (material_a, material_b) = (&bodies.material[a], &bodies.material[b]);
        let restitution = material_a.restitution.max(material_b.restitution);
        let mut manifold = SolverManifold {
            a,
            b,
            handle_a,
            handle_b,
            axes,
            friction: (material_a.friction * material_b.friction).sqrt(),
            points: [SolverPoint::default(); MAX_CONTACTS],
            count: raw.count,
        };
        let inverse_rotation_a = bodies.rotation[a].conjugate();
        let mut claimed = [false; MAX_CONTACTS];
        for (out, point) in manifold.points.iter_mut().zip(&raw.points[..raw.count]) {
            let offset_a = point.position;
            let offset_b = point.position - center_b;
            let local_a = inverse_rotation_a.rotate_vec3(offset_a);
            let mut mass = Vec3::ZERO;
            for (index, axis) in axes.iter().enumerate() {
                let inverse = effective_inverse_mass(bodies, a, b, offset_a, offset_b, *axis);
                let value = if inverse > 1e-12 {
                    inverse.recip()
                } else {
                    0.0
                };
                match index {
                    0 => mass.x = value,
                    1 => mass.y = value,
                    _ => mass.z = value,
                }
            }
            let approach = relative_velocity(state, a, b, offset_a, offset_b).dot(normal);
            let recovery = (RECOVERY_RATE / dt * (point.depth - PENETRATION_SLOP).max(0.0))
                .min(MAX_RECOVERY_SPEED);
            // still apart it may close the gap this step, and only reaching it can bounce
            let gap = -point.depth;
            let target = if gap > 0.0 && -approach * dt < gap {
                -gap / dt
            } else if approach < -RESTITUTION_THRESHOLD {
                -restitution * approach
            } else {
                -gap.max(0.0) / dt
            };
            *out = SolverPoint {
                offset_a,
                offset_b,
                mass,
                target,
                recovery,
                recovery_impulse: 0.0,
                impulse: matching_impulse(old, local_a, &mut claimed),
                local_a,
            };
        }
        scratch.manifolds.push(manifold);
    }
    scratch
        .manifolds
        .sort_unstable_by_key(|manifold| (manifold.handle_a, manifold.handle_b));
}

/// Each old point goes to one new point, two sharing it would double its load.
fn matching_impulse(
    old: Option<&CachedManifold>,
    local_a: Vec3,
    claimed: &mut [bool; MAX_CONTACTS],
) -> Vec3 {
    let Some(old) = old else {
        return Vec3::ZERO;
    };
    let mut best = None;
    let mut closest = MATCH_DISTANCE_SQUARED;
    for (index, point) in old.points[..old.count].iter().enumerate() {
        let distance = (point.local_a - local_a).length_squared();
        if distance < closest && !claimed[index] {
            closest = distance;
            best = Some(index);
        }
    }
    best.map_or(Vec3::ZERO, |index| {
        claimed[index] = true;
        old.points[index].impulse
    })
}

fn warm_start(bodies: &Bodies, state: &mut [Integrated], manifold: &SolverManifold) {
    for point in &manifold.points[..manifold.count] {
        let impulse = manifold.axes[0] * point.impulse.x
            + manifold.axes[1] * point.impulse.y
            + manifold.axes[2] * point.impulse.z;
        apply_pair(bodies, state, manifold, point, impulse);
    }
}

fn solve_manifold(bodies: &Bodies, state: &mut [Integrated], manifold: &mut SolverManifold) {
    let manifold_copy = *manifold;
    for point in &mut manifold.points[..manifold_copy.count] {
        // friction first, so the normal impulse has the last word on penetration
        let limit = (manifold_copy.friction * point.impulse.x).max(0.0);
        for (axis, mass, accumulated) in [
            (manifold_copy.axes[1], point.mass.y, &mut point.impulse.y),
            (manifold_copy.axes[2], point.mass.z, &mut point.impulse.z),
        ] {
            let speed = relative_velocity(
                state,
                manifold_copy.a,
                manifold_copy.b,
                point.offset_a,
                point.offset_b,
            )
            .dot(axis);
            // not `clamp`, it panics on a nan limit
            let next = (*accumulated - speed * mass).min(limit).max(-limit);
            let change = next - *accumulated;
            *accumulated = next;
            apply_offsets(
                bodies,
                state,
                &manifold_copy,
                point.offset_a,
                point.offset_b,
                axis * change,
            );
        }

        let normal = manifold_copy.axes[0];
        let speed = relative_velocity(
            state,
            manifold_copy.a,
            manifold_copy.b,
            point.offset_a,
            point.offset_b,
        )
        .dot(normal);
        let next = (point.impulse.x + (point.target - speed) * point.mass.x).max(0.0);
        let change = next - point.impulse.x;
        point.impulse.x = next;
        apply_offsets(
            bodies,
            state,
            &manifold_copy,
            point.offset_a,
            point.offset_b,
            normal * change,
        );
    }
}

/// Split impulse. Pushes overlapping bodies apart through position only, so it adds no energy.
fn solve_recovery(bodies: &Bodies, state: &mut [Integrated], manifold: &mut SolverManifold) {
    let (a, b) = (manifold.a, manifold.b);
    let normal = manifold.axes[0];
    for point in &mut manifold.points[..manifold.count] {
        let at_a = state[a].recovery_linear + state[a].recovery_angular.cross(point.offset_a);
        let at_b = state[b].recovery_linear + state[b].recovery_angular.cross(point.offset_b);
        let speed = (at_b - at_a).dot(normal);
        let next = (point.recovery_impulse + (point.recovery - speed) * point.mass.x).max(0.0);
        let change = normal * (next - point.recovery_impulse);
        point.recovery_impulse = next;
        apply_recovery(bodies, state, a, point.offset_a, -change);
        apply_recovery(bodies, state, b, point.offset_b, change);
    }
}

fn apply_recovery(
    bodies: &Bodies,
    state: &mut [Integrated],
    dense: usize,
    offset: Vec3,
    impulse: Vec3,
) {
    if bodies.kind[dense] != BodyKind::Dynamic {
        return;
    }
    let body = &mut state[dense];
    body.recovery_linear += impulse * bodies.inverse_mass[dense];
    body.recovery_angular += apply_diagonal(
        bodies.rotation[dense],
        bodies.inverse_inertia[dense],
        offset.cross(impulse),
    );
}

fn apply_pair(
    bodies: &Bodies,
    state: &mut [Integrated],
    manifold: &SolverManifold,
    point: &SolverPoint,
    impulse: Vec3,
) {
    apply_offsets(
        bodies,
        state,
        manifold,
        point.offset_a,
        point.offset_b,
        impulse,
    );
}

/// `impulse` acts on B, its opposite on A.
fn apply_offsets(
    bodies: &Bodies,
    state: &mut [Integrated],
    manifold: &SolverManifold,
    offset_a: Vec3,
    offset_b: Vec3,
    impulse: Vec3,
) {
    apply_one(bodies, state, manifold.a, offset_a, -impulse);
    apply_one(bodies, state, manifold.b, offset_b, impulse);
}

fn apply_one(bodies: &Bodies, state: &mut [Integrated], dense: usize, offset: Vec3, impulse: Vec3) {
    if bodies.kind[dense] != BodyKind::Dynamic {
        return;
    }
    let body = &mut state[dense];
    body.linear += impulse * bodies.inverse_mass[dense];
    body.angular += apply_diagonal(
        bodies.rotation[dense],
        bodies.inverse_inertia[dense],
        offset.cross(impulse),
    );
    body.touched = true;
}

fn relative_velocity(
    state: &[Integrated],
    a: usize,
    b: usize,
    offset_a: Vec3,
    offset_b: Vec3,
) -> Vec3 {
    let at_a = state[a].linear + state[a].angular.cross(offset_a);
    let at_b = state[b].linear + state[b].angular.cross(offset_b);
    at_b - at_a
}

fn effective_inverse_mass(
    bodies: &Bodies,
    a: usize,
    b: usize,
    offset_a: Vec3,
    offset_b: Vec3,
    axis: Vec3,
) -> f32 {
    let part = |dense: usize, offset: Vec3| {
        let arm = offset.cross(axis);
        bodies.inverse_mass[dense]
            + arm.dot(apply_diagonal(
                bodies.rotation[dense],
                bodies.inverse_inertia[dense],
                arm,
            ))
    };
    part(a, offset_a) + part(b, offset_b)
}

fn tangent_of(normal: Vec3) -> Vec3 {
    let helper = if normal.x.abs() < 0.57 {
        Vec3::X
    } else {
        Vec3::Y
    };
    normal.cross(helper).normalize()
}

#[cfg(test)]
mod tests {
    use algebrix::IVec3;

    use crate::{
        BodyHandle, CollisionFilter, FrameDesc, FrameId, GridPosition, Material, PhysicsSettings,
        PhysicsWorld, Quat, RigidBodyDesc, Shape, Vec3,
    };

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    fn no_gravity() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap()
    }

    /// Top surface at y = 0.
    fn ground(world: &mut PhysicsWorld) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(500.0, 1.0, 500.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap()
    }

    fn run(world: &mut PhysicsWorld, seconds: f32) {
        let steps = (seconds / world.settings().fixed_timestep) as usize;
        for _ in 0..steps {
            world.step_fixed().unwrap();
        }
    }

    fn y(world: &PhysicsWorld, body: BodyHandle) -> f32 {
        world.body_state(body).unwrap().position.local.y
    }

    #[test]
    fn gap_inside_the_margin_does_not_stop_a_slow_fall_early() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .local_position(Vec3::new(0.0, 0.51, 0.0)),
            )
            .unwrap();
        world.step_fixed().unwrap();
        // one step of gravity still fits in the gap, so it falls freely
        let dt = world.settings().fixed_timestep;
        let speed = world.body_state(ball).unwrap().linear_velocity.y;
        assert!((speed + 9.81 * dt).abs() < 1e-4, "{speed}");
        run(&mut world, 1.0);
        assert!((y(&world, ball) - 0.5).abs() < 0.01);
    }

    #[test]
    fn fast_body_in_the_margin_is_stopped_and_bounces_once_it_arrives() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .material(Material {
                        restitution: 1.0,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(0.0, 0.51, 0.0))
                    .linear_velocity(Vec3::new(0.0, -10.0, 0.0)),
            )
            .unwrap();
        world.step_fixed().unwrap();
        let state = world.body_state(ball).unwrap();
        assert!(state.linear_velocity.y > 9.0, "{state:?}");
        assert!(state.position.local.y > 0.5, "{state:?}");
    }

    #[test]
    fn sphere_comes_to_rest_on_the_ground() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 2.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 3.0);
        assert!((y(&world, ball) - 0.5).abs() < 0.02, "{}", y(&world, ball));
        let speed = world.body_state(ball).unwrap().linear_velocity.length();
        assert!(speed < 0.05, "{speed}");
    }

    #[test]
    fn restitution_sets_the_rebound_height() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .material(Material {
                        restitution: 0.5,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(0.0, 5.5, 0.0)),
            )
            .unwrap();
        let mut bounced = false;
        let mut peak = 0.0_f32;
        for _ in 0..240 {
            world.step_fixed().unwrap();
            let velocity = world.body_state(ball).unwrap().linear_velocity.y;
            bounced |= velocity > 0.0;
            if bounced {
                peak = peak.max(y(&world, ball));
            }
        }
        // 5 m of fall at e = 0.5 comes back 1.25 m
        assert!((peak - 1.75).abs() < 0.2, "{peak}");
    }

    #[test]
    fn no_restitution_does_not_bounce() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 5.5, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.1);
        let mut peak = 0.0_f32;
        for _ in 0..120 {
            world.step_fixed().unwrap();
            peak = peak.max(y(&world, ball));
        }
        assert!(peak < 0.6, "{peak}");
    }

    #[test]
    fn friction_stops_a_sliding_box_at_the_right_distance() {
        let mut world = world();
        ground(&mut world);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(0.0, 0.5, 0.0))
                    .linear_velocity(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 3.0);
        let state = world.body_state(cube).unwrap();
        assert!(state.linear_velocity.length() < 0.05);
        // v^2 / (2 mu g) with mu = 0.5
        let expected = 25.0 / 9.81;
        assert!(
            (state.position.local.x - expected).abs() < expected * 0.15,
            "{}",
            state.position.local.x
        );
    }

    #[test]
    fn no_friction_keeps_sliding() {
        let mut world = world();
        ground(&mut world);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .material(Material {
                        friction: 0.0,
                        ..Material::default()
                    })
                    .local_position(Vec3::new(0.0, 0.5, 0.0))
                    .linear_velocity(Vec3::new(5.0, 0.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        let velocity = world.body_state(cube).unwrap().linear_velocity.x;
        assert!((velocity - 5.0).abs() < 0.01, "{velocity}");
    }

    #[test]
    fn box_stack_stays_up() {
        let mut world = world();
        ground(&mut world);
        let mut boxes = Vec::new();
        for level in 0..5 {
            boxes.push(
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap(),
            );
        }
        run(&mut world, 5.0);
        for (level, body) in boxes.iter().enumerate() {
            let state = world.body_state(*body).unwrap();
            let expected = 0.5 + level as f32;
            assert!(
                (state.position.local.y - expected).abs() < 0.05,
                "level {level}: {}",
                state.position.local.y
            );
            assert!(state.position.local.x.abs() < 0.02);
            assert!(state.position.local.z.abs() < 0.02);
            assert!(state.linear_velocity.length() < 0.1);
        }
    }

    #[test]
    fn equal_spheres_swap_velocity_when_elastic() {
        let mut world = no_gravity();
        let elastic = Material {
            restitution: 1.0,
            ..Material::default()
        };
        let sphere = |x: f32, speed: f32| {
            RigidBodyDesc::dynamic(Shape::sphere(0.5))
                .material(elastic)
                .local_position(Vec3::new(x, 0.0, 0.0))
                .linear_velocity(Vec3::new(speed, 0.0, 0.0))
        };
        let left = world.spawn(sphere(-2.0, 2.0)).unwrap();
        let right = world.spawn(sphere(2.0, -2.0)).unwrap();
        run(&mut world, 2.0);
        let left = world.body_state(left).unwrap().linear_velocity.x;
        let right = world.body_state(right).unwrap().linear_velocity.x;
        assert!((left + 2.0).abs() < 0.1, "{left}");
        assert!((right - 2.0).abs() < 0.1, "{right}");
        assert!((left + right).abs() < 1e-3);
    }

    #[test]
    fn light_body_hitting_heavy_one_conserves_momentum() {
        let mut world = no_gravity();
        let light = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .density(1.0)
                    .local_position(Vec3::new(-2.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(4.0, 0.0, 0.0)),
            )
            .unwrap();
        let heavy = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.5)).density(9.0))
            .unwrap();
        run(&mut world, 1.0);
        let left = world.body_state(light).unwrap().linear_velocity.x;
        let right = world.body_state(heavy).unwrap().linear_velocity.x;
        // masses 1 to 9, no bounce, both end up at 4 / 10
        assert!((left + 9.0 * right - 4.0).abs() < 1e-3, "{left} {right}");
        assert!((right - 0.4).abs() < 0.05, "{right}");
    }

    #[test]
    fn tilted_box_settles_flat() {
        let mut world = world();
        ground(&mut world);
        let cube = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::new(0.0, 1.5, 0.0))
                    .rotation(Quat::from_axis_angle(
                        Vec3::new(1.0, 0.0, 1.0).normalize(),
                        0.5,
                    )),
            )
            .unwrap();
        run(&mut world, 5.0);
        let state = world.body_state(cube).unwrap();
        assert!((state.position.local.y - 0.5).abs() < 0.02);
        assert!(state.angular_velocity.length() < 0.1);
        let up = state.rotation.rotate_vec3(Vec3::Y);
        let aligned = up.x.abs().max(up.y.abs()).max(up.z.abs());
        assert!(aligned > 0.999, "{up:?}");
    }

    #[test]
    fn embedded_body_recovers_without_exploding() {
        let mut world = world();
        ground(&mut world);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).local_position(Vec3::new(0.0, 0.1, 0.0)),
            )
            .unwrap();
        let mut fastest = 0.0_f32;
        for _ in 0..180 {
            world.step_fixed().unwrap();
            fastest = fastest.max(world.body_state(ball).unwrap().linear_velocity.length());
        }
        assert!(fastest <= 3.5, "{fastest}");
        assert!((y(&world, ball) - 0.5).abs() < 0.02);
    }

    #[test]
    fn filtered_pair_passes_through() {
        let mut world = world();
        ground(&mut world);
        let ghost = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .filter(CollisionFilter { layers: 2, mask: 2 })
                    .local_position(Vec3::new(0.0, 1.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        assert!(y(&world, ghost) < -1.0);
    }

    #[test]
    fn different_frames_do_not_collide() {
        let mut world = world();
        ground(&mut world);
        let other = world.add_frame(FrameDesc::new(FrameId::ROOT)).unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .frame(other)
                    .local_position(Vec3::new(0.0, 1.0, 0.0)),
            )
            .unwrap();
        run(&mut world, 1.0);
        assert!(y(&world, ball) < -1.0);
    }

    #[test]
    fn contact_works_far_from_origin_and_across_cells() {
        let mut world = world();
        let cell = IVec3::new(2_000_000, -700_000, 1_000_000);
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(2000.0, 1.0, 10.0))).position(
                    GridPosition {
                        cell,
                        local: Vec3::new(0.0, -1.0, 0.0),
                    },
                ),
            )
            .unwrap();
        let next_cell = IVec3::new(cell.x + 1, cell.y, cell.z);
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5)).position(GridPosition {
                    cell: next_cell,
                    local: Vec3::new(0.0, 1.0, 0.0),
                }),
            )
            .unwrap();
        run(&mut world, 3.0);
        let state = world.body_state(ball).unwrap();
        assert_eq!(state.position.cell, next_cell);
        assert!((state.position.local.y - 0.5).abs() < 0.02);
    }

    #[test]
    fn snapshot_in_mid_contact_continues_exactly() {
        let mut world = world();
        ground(&mut world);
        for level in 0..4 {
            world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                        .local_position(Vec3::new(0.05 * level as f32, 0.6 + level as f32, 0.0)),
                )
                .unwrap();
        }
        run(&mut world, 0.5);
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        run(&mut world, 1.0);
        run(&mut loaded, 1.0);
        assert_eq!(loaded.state_hash(), world.state_hash());
    }

    #[test]
    fn removing_a_body_mid_stack_keeps_the_rest_working() {
        let mut world = world();
        ground(&mut world);
        let mut boxes = Vec::new();
        for level in 0..3 {
            boxes.push(
                world
                    .spawn(
                        RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                            .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                    )
                    .unwrap(),
            );
        }
        run(&mut world, 1.0);
        world.despawn(boxes[0]).unwrap();
        run(&mut world, 2.0);
        assert!(
            (y(&world, boxes[1]) - 0.5).abs() < 0.05,
            "{}",
            y(&world, boxes[1])
        );
        assert!(
            (y(&world, boxes[2]) - 1.5).abs() < 0.05,
            "{}",
            y(&world, boxes[2])
        );
    }
}
