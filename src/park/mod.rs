use algebrix::{DQuat, DVec3};

use crate::body::{BodyHandle, BodyKind, RigidBodyDesc};
use crate::compound::CompoundChild;
use crate::error::PhysicsError;
use crate::frame::{FrameId, FrameState};
use crate::gravity::GravityId;
use crate::orbit::Coast;
use crate::position::GridPosition;
use crate::shape::Shape;
use crate::world::PhysicsWorld;

/// A body taken out of the world, coasting round a gravity source until it is put back: a
/// satellite nobody is near, worked out as an orbit rather than stepped. The host owns it and
/// decides when it goes out and comes back. Its turning is kept as it was, torque free.
#[derive(Debug, Clone, PartialEq)]
pub struct Parked {
    desc: RigidBodyDesc,
    /// A compound the world made for the body, made again when it comes back.
    children: Option<Vec<CompoundChild>>,
    source: GravityId,
    /// Round the source's center, in the root, from `since`.
    coast: Coast,
    since: f64,
    /// The body's turn in the root and how it turns, at `since`.
    rotation: DQuat,
    spin: DVec3,
}

impl Parked {
    pub fn source(&self) -> GravityId {
        self.source
    }

    /// Its coast round the source's center in root axes, time 0 when it was parked.
    pub fn coast(&self) -> &Coast {
        &self.coast
    }

    /// World seconds it was parked at.
    pub fn since(&self) -> f64 {
        self.since
    }
}

impl PhysicsWorld {
    /// Takes a dynamic `body` out of the world onto a coast round `source`, from where it is
    /// and how it moves now. Between steps.
    ///
    /// # Errors
    /// Fails on a stale handle, an unknown source, a body that is not dynamic, has joints or
    /// that a breakable owns.
    pub fn park(&mut self, body: BodyHandle, source: GravityId) -> Result<Parked, PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        if self.bodies.kind[dense] != BodyKind::Dynamic {
            return Err(PhysicsError::NotDynamic);
        }
        if self.joints.touching(body).next().is_some() {
            return Err(PhysicsError::BodyHasJoints);
        }
        if self.breakables.holds(body) || self.structures.holds(body) {
            return Err(PhysicsError::ShapeInUse);
        }
        let pull = *self
            .gravity_source(source)
            .ok_or(PhysicsError::InvalidSettings)?;
        let time = self.frames.time();
        let state = self.body_state(body)?;
        let root = self.frames.to_root(
            state.frame,
            FrameState {
                position: state.position.to_dvec3(),
                rotation: DQuat::from_quat(state.rotation),
                linear_velocity: DVec3::from_vec3(state.linear_velocity),
                angular_velocity: DVec3::from_vec3(state.angular_velocity),
            },
        )?;
        let (center, moving) = self.source_in_root(pull.frame, pull.center, time);
        let coast = Coast::new(
            pull.mu,
            root.position - center,
            root.linear_velocity - moving,
        )
        .ok_or(PhysicsError::OutOfRange)?;
        let shape = self.bodies.shape[dense];
        let children = match shape {
            Shape::Compound(id)
                if self
                    .bodies
                    .assets
                    .compound(id)
                    .is_some_and(|compound| compound.life == crate::compound::Life::Engine) =>
            {
                Some(self.compound_children(body)?)
            }
            _ => None,
        };
        let mut desc =
            RigidBodyDesc::new(BodyKind::Dynamic, shape).material(self.bodies.material[dense]);
        desc.frame = state.frame;
        desc.filter = self.bodies.filter[dense];
        desc.can_sleep = self.bodies.can_sleep[dense];
        desc.surface_velocity = self.bodies.surface[dense];
        self.despawn(body)?;
        Ok(Parked {
            desc,
            children,
            source,
            coast,
            since: time,
            rotation: root.rotation,
            spin: root.angular_velocity,
        })
    }

    /// Puts `parked` back where its coast has taken it by now, in the frame it left. A new
    /// handle. Between steps.
    ///
    /// # Errors
    /// Fails when its source is gone, or the world refuses the body.
    pub fn unpark(&mut self, parked: &Parked) -> Result<BodyHandle, PhysicsError> {
        let pull = *self
            .gravity_source(parked.source)
            .ok_or(PhysicsError::InvalidSettings)?;
        let time = self.frames.time();
        let elapsed = time - parked.since;
        let (position, velocity) = parked.coast.state_at(elapsed);
        let (center, moving) = self.source_in_root(pull.frame, pull.center, time);
        let rate = parked.spin.length();
        let rotation = if rate > 0.0 {
            (DQuat::from_axis_angle(parked.spin / rate, rate * elapsed) * parked.rotation)
                .normalize()
        } else {
            parked.rotation
        };
        let local = self.frames.from_root(
            parked.desc.frame,
            FrameState {
                position: center + position,
                rotation,
                linear_velocity: moving + velocity,
                angular_velocity: parked.spin,
            },
        )?;
        let mut desc = parked.desc;
        let mut at = local.position;
        let mut turn = local.rotation.as_quat().normalize();
        if let Some(children) = &parked.children {
            let info = self.add_owned_compound(children)?;
            desc.shape = info.shape;
            at += DVec3::from_vec3(turn.rotate_vec3(info.center));
            turn = (turn * info.axes).normalize();
        }
        desc.position = GridPosition::from_dvec3(at).ok_or(PhysicsError::OutOfRange)?;
        desc.rotation = turn;
        desc.linear_velocity = local.linear_velocity.as_vec3();
        desc.angular_velocity = local.angular_velocity.as_vec3();
        let made = self.spawn(desc);
        if made.is_err() {
            self.free_owned(desc.shape);
        }
        made
    }

    /// Where a point standing in `frame` is in the root at `time`, and how fast it goes there.
    fn source_in_root(&self, frame: FrameId, center: DVec3, time: f64) -> (DVec3, DVec3) {
        let origin = self.frames.origin_at(frame, time);
        let arm = origin.rotation.mul_vec3(center);
        (
            origin.position + arm,
            origin.linear_velocity + origin.angular_velocity.cross(arm),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compound::CompoundChild;
    use crate::frame::{FrameDesc, FrameMotion};
    use crate::gravity::GravitySource;
    use crate::world::testing::no_gravity;
    use algebrix::Vec3;

    /// A small planet, a minute a turn at 100 m, and a ball going round it.
    fn orbiting(world: &mut PhysicsWorld, frame: FrameId) -> (GravityId, BodyHandle, f64) {
        let (radius, period) = (100.0_f64, 60.0_f64);
        let mu = (std::f64::consts::TAU / period).powi(2) * radius.powi(3);
        let source = world
            .add_gravity(GravitySource::point(frame, DVec3::ZERO, mu))
            .unwrap();
        let speed = (mu / radius).sqrt();
        let mut desc = RigidBodyDesc::dynamic(Shape::sphere(0.5))
            .local_position(Vec3::new(radius as f32, 0.0, 0.0))
            .linear_velocity(Vec3::new(0.0, 0.0, -speed as f32))
            .angular_velocity(Vec3::new(0.0, 0.4, 0.0));
        desc.frame = frame;
        desc.can_sleep = false;
        (source, world.spawn(desc).unwrap(), period)
    }

    fn at(world: &PhysicsWorld, body: BodyHandle) -> DVec3 {
        world.body_state(body).unwrap().position.to_dvec3()
    }

    #[test]
    fn parked_and_put_back_it_is_where_staying_would_have_taken_it() {
        let mut stayed = no_gravity();
        let (_, kept, period) = orbiting(&mut stayed, FrameId::ROOT);
        let mut parked_world = no_gravity();
        let (source, body, _) = orbiting(&mut parked_world, FrameId::ROOT);
        let steps = (period * 60.0 * 0.4) as usize;
        let parked = parked_world.park(body, source).unwrap();
        assert!(parked_world.body_state(body).is_err());
        for _ in 0..steps {
            stayed.step_fixed().unwrap();
            parked_world.step_fixed().unwrap();
        }
        let back = parked_world.unpark(&parked).unwrap();
        // the coast is exact, the stepped one has the integrator's error: well under a meter
        let gap = (at(&parked_world, back) - at(&stayed, kept)).length();
        assert!(gap < 0.5, "{gap}");
        assert!((at(&parked_world, back).length() - 100.0).abs() < 0.01);
        let spin = parked_world.body_state(back).unwrap().angular_velocity;
        assert!((spin - Vec3::new(0.0, 0.4, 0.0)).length() < 1e-5);
        // and it steps on from there
        for _ in 0..60 {
            parked_world.step_fixed().unwrap();
        }
        assert!((at(&parked_world, back).length() - 100.0).abs() < 0.1);
    }

    #[test]
    fn a_coast_round_a_turning_planet_frame_is_not_bent_by_the_turning() {
        let mut world = no_gravity();
        let planet = world
            .add_frame(FrameDesc {
                motion: FrameMotion {
                    spin: DVec3::new(0.0, 0.5, 0.0),
                    orbit: None,
                },
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        let (source, body, period) = orbiting(&mut world, planet);
        // its speed was given in the turning frame: the turning adds 50 m/s in the root, and it
        // leaves the planet on a hyperbola
        let parked = world.park(body, source).unwrap();
        assert!((parked.coast().velocity.length() - (50.0 + 10.47)).abs() < 0.1);
        assert!(parked.coast().period().is_none());
        for _ in 0..(period * 60.0 * 0.25) as usize {
            world.step_fixed().unwrap();
        }
        let back = world.unpark(&parked).unwrap();
        let state = world.body_state(back).unwrap();
        assert_eq!(state.frame, planet);
        let (expected, _) = parked
            .coast()
            .state_at(world.frames().time() - parked.since());
        let root = world
            .frames()
            .to_root(
                planet,
                FrameState {
                    position: state.position.to_dvec3(),
                    rotation: DQuat::IDENTITY,
                    linear_velocity: DVec3::ZERO,
                    angular_velocity: DVec3::ZERO,
                },
            )
            .unwrap();
        assert!(
            (root.position - expected).length() < 1e-3,
            "{:?} {expected:?}",
            root.position
        );
    }

    #[test]
    fn a_parked_assembly_comes_back_whole_and_bad_parks_are_refused() {
        let mut world = no_gravity();
        let (source, ball, _) = orbiting(&mut world, FrameId::ROOT);
        let parts: Vec<CompoundChild> = (0..3)
            .map(|index| {
                CompoundChild::new(
                    Shape::cuboid(Vec3::splat(0.5)),
                    Vec3::new(index as f32, 0.0, 0.0),
                )
            })
            .collect();
        let info = world.add_compound(&parts).unwrap();
        let ship = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(Vec3::new(0.0, 50.0, 0.0))
                    .linear_velocity(Vec3::new(0.0, 0.0, 3.0)),
            )
            .unwrap();
        // an edited one is the world's own, it is made again on the way back
        let children = world.compound_children(ship).unwrap();
        world.reshape_compound(ship, &children[..2]).unwrap();
        let mass = |world: &PhysicsWorld, body| {
            world.bodies.inverse_mass[world.bodies.dense_index(body).unwrap()].recip()
        };
        let before = mass(&world, ship);
        let parked = world.park(ship, source).unwrap();
        let back = world.unpark(&parked).unwrap();
        assert!((mass(&world, back) - before).abs() < 1e-3);
        assert_eq!(world.compound_children(back).unwrap().len(), 2);
        let fixed = world
            .spawn(
                RigidBodyDesc::fixed(Shape::sphere(1.0)).local_position(Vec3::new(9.0, 0.0, 0.0)),
            )
            .unwrap();
        assert!(world.park(fixed, source).is_err());
        assert!(world.park(ball, GravityId(99)).is_err());
        world.remove_gravity(source);
        assert!(world.unpark(&parked).is_err());
    }
}
