use algebrix::{DQuat, DVec3, Quat, Vec3};

use crate::body::{BodyHandle, BodyKind, RigidBodyDesc};
use crate::error::PhysicsError;
use crate::frame::{FrameId, FrameState};
use crate::gravity::GravityId;
use crate::orbit::Coast;
use crate::position::GridPosition;
use crate::snapshot::{Reader, SnapshotError, Writer};
use crate::structure::{Link, MAX_LINKS, StructureHandle};
use crate::world::PhysicsWorld;

mod group;
mod stored;

pub use group::{PARKED_GROUP_VERSION, ParkedGroup, UnparkedGroup};
use stored::{Geometry, load_geometry, save_geometry};

const MAGIC: [u8; 4] = *b"SLPK";
const STAND_IN: crate::Shape = crate::Shape::Sphere { radius: 1.0 };
/// Version of [`Parked::to_bytes`].
pub const PARKED_VERSION: u32 = 1;

/// A body taken out of the world, kept small until it is wanted again: a satellite nobody is
/// near coasting round a planet, a ship left at a far station held where it is. It keeps its
/// shape as it is, not as an id, so it goes back into any world with the same frames and
/// gravity, and its structure with the strain and damage it had. The host owns it, keeps it as
/// bytes if it likes, and decides when it comes back.
#[derive(Debug, Clone, PartialEq)]
pub struct Parked {
    desc: RigidBodyDesc,
    geometry: Geometry,
    motion: Motion,
    links: Option<Vec<Link>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Motion {
    /// Where it was in its frame and how it moved, as it was.
    Held,
    /// Round a source's center in the root from world time `since`, turning torque free.
    Coasting {
        source: GravityId,
        coast: Coast,
        since: f64,
        rotation: DQuat,
        spin: DVec3,
    },
}

/// A parked body back in the world: its new handle, and its structure's if it had one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unparked {
    pub body: BodyHandle,
    pub structure: Option<StructureHandle>,
}

impl Parked {
    /// The coast it is on, none for one held where it is.
    pub fn coast(&self) -> Option<&Coast> {
        match &self.motion {
            Motion::Coasting { coast, .. } => Some(coast),
            Motion::Held => None,
        }
    }

    /// World seconds it started coasting at, none for one held.
    pub fn since(&self) -> Option<f64> {
        match self.motion {
            Motion::Coasting { since, .. } => Some(since),
            Motion::Held => None,
        }
    }

    pub fn source(&self) -> Option<GravityId> {
        match self.motion {
            Motion::Coasting { source, .. } => Some(source),
            Motion::Held => None,
        }
    }

    /// The links it carries, numbered by its parts.
    pub fn links(&self) -> Option<&[Link]> {
        self.links.as_deref()
    }

    /// As bytes for keeping, with a version and a checksum.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Writer::with_capacity(256);
        out.bytes(&MAGIC);
        out.u32(PARKED_VERSION);
        let desc = &self.desc;
        out.u32(desc.frame.index());
        out.material(&desc.material);
        out.u32(desc.filter.layers);
        out.u32(desc.filter.mask);
        out.u8(u8::from(desc.can_sleep));
        out.vec3(desc.surface_velocity);
        out.position(desc.position);
        out.quat(desc.rotation);
        out.vec3(desc.linear_velocity);
        out.vec3(desc.angular_velocity);
        save_geometry(&mut out, &self.geometry);
        match self.motion {
            Motion::Held => out.u8(0),
            Motion::Coasting {
                source,
                coast,
                since,
                rotation,
                spin,
            } => {
                out.u8(1);
                out.u32(source.0);
                out.f64(coast.mu);
                out.dvec3(coast.position);
                out.dvec3(coast.velocity);
                out.f64(since);
                for value in [rotation.x, rotation.y, rotation.z, rotation.w] {
                    out.f64(value);
                }
                out.dvec3(spin);
            }
        }
        match &self.links {
            None => out.u8(0),
            Some(links) => {
                out.u8(1);
                out.len(links.len());
                for link in links {
                    out.u32(link.a);
                    out.u32(link.b);
                    for value in [link.strength, link.load, link.strain, link.damage] {
                        out.f32(value);
                    }
                    out.u8(u8::from(link.broken));
                }
            }
        }
        let sum = crate::snapshot::fnv1a(&out.data);
        out.u64(sum);
        out.data
    }

    /// # Errors
    /// Fails on bytes that are short, corrupt, of another version or out of range.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        let bad = SnapshotError::Invalid("parked");
        if bytes.len() < 16 || bytes[..4] != MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        let (body, sum) = bytes.split_at(bytes.len() - 8);
        if crate::snapshot::fnv1a(body) != u64::from_le_bytes(sum.try_into().map_err(|_| bad)?) {
            return Err(SnapshotError::BadChecksum);
        }
        let mut reader = Reader {
            data: body,
            at: 4,
            version: 0,
        };
        let version = reader.u32()?;
        if version != PARKED_VERSION {
            return Err(SnapshotError::UnsupportedVersion(version));
        }
        reader.version = crate::snapshot::SNAPSHOT_VERSION;
        let frame = FrameId(reader.u32()?);
        let material = reader.material()?;
        let filter = crate::CollisionFilter {
            layers: reader.u32()?,
            mask: reader.u32()?,
        };
        let can_sleep = match reader.u8()? {
            0 => false,
            1 => true,
            _ => return Err(bad),
        };
        let surface_velocity = reader.vec3()?;
        let position = reader.position()?;
        let rotation = reader.quat()?;
        if (rotation.length() - 1.0).abs() >= 1e-3 || rotation.length().is_nan() {
            return Err(bad);
        }
        let (linear_velocity, angular_velocity) = (reader.vec3()?, reader.vec3()?);
        let geometry = load_geometry(&mut reader)?;
        let motion = match reader.u8()? {
            0 => Motion::Held,
            1 => {
                let source = GravityId(reader.u32()?);
                let mu = reader.f64()?;
                let (start, speed) = (reader.dvec3()?, reader.dvec3()?);
                let since = reader.f64()?;
                let turn = DQuat::new(reader.f64()?, reader.f64()?, reader.f64()?, reader.f64()?);
                let spin = reader.dvec3()?;
                let coast = Coast::new(mu, start, speed).ok_or(bad)?;
                let finite = |v: DVec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
                let length =
                    (turn.x * turn.x + turn.y * turn.y + turn.z * turn.z + turn.w * turn.w).sqrt();
                if !since.is_finite()
                    || !finite(spin)
                    || (length - 1.0).abs() >= 1e-3
                    || length.is_nan()
                {
                    return Err(bad);
                }
                Motion::Coasting {
                    source,
                    coast,
                    since,
                    rotation: turn,
                    spin,
                }
            }
            _ => return Err(bad),
        };
        let links = match reader.u8()? {
            0 => None,
            1 => {
                let count = reader.count(25, MAX_LINKS)?;
                let mut links = Vec::with_capacity(count);
                for _ in 0..count {
                    let (a, b) = (reader.u32()?, reader.u32()?);
                    let (strength, load, strain, damage) =
                        (reader.f32()?, reader.f32()?, reader.f32()?, reader.f32()?);
                    let broken = match reader.u8()? {
                        0 => false,
                        1 => true,
                        _ => return Err(bad),
                    };
                    let positive = strength.is_finite() && strength > 0.0;
                    if a == b
                        || !positive
                        || !load.is_finite()
                        || !strain.is_finite()
                        || !(0.0..=1.0).contains(&damage)
                    {
                        return Err(bad);
                    }
                    links.push(Link {
                        a,
                        b,
                        strength,
                        load,
                        strain,
                        damage,
                        broken,
                    });
                }
                Some(links)
            }
            _ => return Err(bad),
        };
        if reader.at != reader.data.len()
            || !linear_velocity.is_finite()
            || !angular_velocity.is_finite()
            || !surface_velocity.is_finite()
        {
            return Err(bad);
        }
        let mut desc = RigidBodyDesc::new(BodyKind::Dynamic, STAND_IN)
            .material(material)
            .position(position)
            .rotation(rotation)
            .linear_velocity(linear_velocity)
            .angular_velocity(angular_velocity);
        desc.frame = frame;
        desc.filter = filter;
        desc.can_sleep = can_sleep;
        desc.surface_velocity = surface_velocity;
        Ok(Self {
            desc,
            geometry,
            motion,
            links,
        })
    }
}

impl PhysicsWorld {
    /// Takes a dynamic `body` out of the world onto a coast round `source`, from where it is
    /// and how it moves now. Between steps.
    ///
    /// # Errors
    /// As [`Self::hold`], and on an unknown source.
    pub fn park(&mut self, body: BodyHandle, source: GravityId) -> Result<Parked, PhysicsError> {
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
        self.take_out(
            body,
            Motion::Coasting {
                source,
                coast,
                since: time,
                rotation: root.rotation,
                spin: root.angular_velocity,
            },
        )
    }

    /// Takes a dynamic `body` out of the world, held where it is in its frame and moving as it
    /// moves: a ship at a station, a wreck on a moon. It comes back just so, however long it was
    /// away. Between steps.
    ///
    /// # Errors
    /// Fails on a stale handle, a body that is not dynamic, has joints, is a mesh or a
    /// breakable owns.
    pub fn hold(&mut self, body: BodyHandle) -> Result<Parked, PhysicsError> {
        self.take_out(body, Motion::Held)
    }

    fn take_out(&mut self, body: BodyHandle, motion: Motion) -> Result<Parked, PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        if self.bodies.kind[dense] != BodyKind::Dynamic {
            return Err(PhysicsError::NotDynamic);
        }
        if self.joints.touching(body).next().is_some() {
            return Err(PhysicsError::BodyHasJoints);
        }
        if self.breakables.holds(body) {
            return Err(PhysicsError::ShapeInUse);
        }
        let shape = self.bodies.shape[dense];
        let geometry = self.store_geometry(&shape)?;
        let state = self.body_state(body)?;
        // the shape lives in the geometry, the desc's is a stand in until it is built again
        let mut desc = RigidBodyDesc::new(BodyKind::Dynamic, STAND_IN)
            .material(self.bodies.material[dense])
            .position(state.position)
            .rotation(state.rotation)
            .linear_velocity(state.linear_velocity)
            .angular_velocity(state.angular_velocity);
        desc.frame = state.frame;
        desc.filter = self.bodies.filter[dense];
        desc.can_sleep = self.bodies.can_sleep[dense];
        desc.surface_velocity = self.bodies.surface[dense];
        let links = self.structures.take(body);
        self.despawn(body)?;
        Ok(Parked {
            desc,
            geometry,
            motion,
            links,
        })
    }

    /// Puts `parked` back: where its coast has taken it by now, or where it was held, in the
    /// frame it left, with its structure. A new handle. Between steps.
    ///
    /// # Errors
    /// Fails when its source or frame is gone, or the world refuses the body or its parts.
    pub fn unpark(&mut self, parked: &Parked) -> Result<Unparked, PhysicsError> {
        self.put_back(parked).map(|(unparked, ..)| unparked)
    }

    /// [`Self::unpark`], and where the body's center and axes now sit against those it had,
    /// for what is fixed to it.
    fn put_back(&mut self, parked: &Parked) -> Result<(Unparked, Vec3, Quat), PhysicsError> {
        let mut desc = parked.desc;
        if let Motion::Coasting {
            source,
            coast,
            since,
            rotation,
            spin,
        } = parked.motion
        {
            let pull = *self
                .gravity_source(source)
                .ok_or(PhysicsError::InvalidSettings)?;
            let time = self.frames.time();
            let elapsed = time - since;
            let (position, velocity) = coast.state_at(elapsed);
            let (center, moving) = self.source_in_root(pull.frame, pull.center, time);
            let rate = spin.length();
            let turned = if rate > 0.0 {
                (DQuat::from_axis_angle(spin / rate, rate * elapsed) * rotation).normalize()
            } else {
                rotation
            };
            let local = self.frames.from_root(
                desc.frame,
                FrameState {
                    position: center + position,
                    rotation: turned,
                    linear_velocity: moving + velocity,
                    angular_velocity: spin,
                },
            )?;
            desc.position =
                GridPosition::from_dvec3(local.position).ok_or(PhysicsError::OutOfRange)?;
            desc.rotation = local.rotation.as_quat().normalize();
            desc.linear_velocity = local.linear_velocity.as_vec3();
            desc.angular_velocity = local.angular_velocity.as_vec3();
        }
        self.frames.get(desc.frame)?;
        let (shape, center, axes) = self.restore_geometry(&parked.geometry)?;
        desc.shape = shape;
        desc.position = desc
            .position
            .shifted(DVec3::from_vec3(desc.rotation.rotate_vec3(center)))
            .ok_or(PhysicsError::OutOfRange)?;
        desc.rotation = (desc.rotation * axes).normalize();
        let body = match self.spawn(desc) {
            Ok(body) => body,
            Err(error) => {
                self.free_owned(shape);
                return Err(error);
            }
        };
        let structure = match &parked.links {
            Some(links) => Some(self.restore_structure(body, links.clone())?),
            None => None,
        };
        Ok((Unparked { body, structure }, center, axes))
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
    use crate::{RigidBodyDesc, Shape};
    use algebrix::{Quat, Vec3};

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
        let back = parked_world.unpark(&parked).unwrap().body;
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
        assert!((parked.coast().unwrap().velocity.length() - (50.0 + 10.47)).abs() < 0.1);
        assert!(parked.coast().unwrap().period().is_none());
        for _ in 0..(period * 60.0 * 0.25) as usize {
            world.step_fixed().unwrap();
        }
        let back = world.unpark(&parked).unwrap().body;
        let state = world.body_state(back).unwrap();
        assert_eq!(state.frame, planet);
        let (expected, _) = parked
            .coast()
            .unwrap()
            .state_at(world.frames().time() - parked.since().unwrap());
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
        let back = world.unpark(&parked).unwrap().body;
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

    #[test]
    fn a_held_ship_comes_back_where_it_was_with_its_structure() {
        let mut world = no_gravity();
        let parts: Vec<CompoundChild> = (0..3)
            .map(|index| {
                CompoundChild::new(
                    Shape::cuboid(Vec3::splat(0.5)),
                    Vec3::new(index as f32, 0.0, 0.0),
                )
            })
            .collect();
        let info = world.add_compound(&parts).unwrap();
        let (center, turn) = info.placed(Vec3::new(3.0, 2.0, 1.0), Quat::from_rotation_y(0.3));
        let ship = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(center)
                    .rotation(turn),
            )
            .unwrap();
        let link = |a, b| crate::LinkDesc {
            a,
            b,
            strength: 1.0e5,
        };
        let structure = world
            .add_structure(ship, &[link(0, 1), link(1, 2)])
            .unwrap();
        world.break_link(structure, 1).unwrap();
        let before = world.body_state(ship).unwrap();
        let parked = world.hold(ship).unwrap();
        assert!(parked.coast().is_none());
        assert_eq!(parked.links().unwrap().len(), 2);
        assert!(
            world.structure_links(structure).is_err(),
            "its only body left"
        );
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        let back = world.unpark(&parked).unwrap();
        let after = world.body_state(back.body).unwrap();
        assert!((after.position.to_dvec3() - before.position.to_dvec3()).length() < 1e-4);
        assert!((after.rotation.dot(before.rotation)).abs() > 1.0 - 1e-5);
        let links = world.structure_links(back.structure.unwrap()).unwrap();
        assert_eq!(links.iter().filter(|link| link.broken).count(), 1);
        // the next step splits it where its link was broken before it went
        world.step_fixed().unwrap();
        assert_eq!(
            world
                .structure_bodies(back.structure.unwrap())
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn as_bytes_it_goes_into_a_new_world_and_bad_bytes_are_refused() {
        let mut world = no_gravity();
        let (source, ball, _) = orbiting(&mut world, FrameId::ROOT);
        // a hull, which the new world has never seen
        let points = [
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.7, 0.0),
            Vec3::new(0.0, 0.0, 0.6),
            Vec3::new(0.1, -0.4, -0.3),
        ];
        let hull = world.add_convex(&points).unwrap();
        let (place, turn) = hull.placed(Vec3::new(0.0, 100.0, 0.0), Quat::IDENTITY);
        let rock = world
            .spawn(
                RigidBodyDesc::dynamic(hull.shape)
                    .local_position(place)
                    .rotation(turn)
                    .linear_velocity(Vec3::new(10.0, 0.0, 0.0)),
            )
            .unwrap();
        let mass = world.bodies.inverse_mass[world.bodies.dense_index(rock).unwrap()].recip();
        let parked = world.park(rock, source).unwrap();
        let bytes = parked.to_bytes();
        let again = Parked::from_bytes(&bytes).unwrap();
        assert_eq!(again, parked);
        // a new world with the same pull: the rock comes back on its coast
        let mut other = no_gravity();
        let _ = orbiting(&mut other, FrameId::ROOT);
        let back = other.unpark(&again).unwrap().body;
        let dense = other.bodies.dense_index(back).unwrap();
        assert!((other.bodies.inverse_mass[dense].recip() - mass).abs() < 1e-3 * mass);
        let at = other.body_state(back).unwrap().position.to_dvec3();
        assert!((at - DVec3::from_vec3(place)).length() < 1e-3, "{at:?}");
        let _ = ball;
        // broken bytes fail, never panic
        let mut seed = 0x1234_5678_u32;
        for _ in 0..2_000 {
            let mut broken = bytes.clone();
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let at = (seed as usize >> 4) % broken.len();
            broken[at] ^= 1 + (seed as u8 >> 1);
            let _ = Parked::from_bytes(&broken);
        }
        assert!(Parked::from_bytes(&bytes[..bytes.len() - 3]).is_err());
        let mut later = bytes.clone();
        later[4..8].copy_from_slice(&(PARKED_VERSION + 1).to_le_bytes());
        let body = later.len() - 8;
        let sum = crate::snapshot::fnv1a(&later[..body]);
        later[body..].copy_from_slice(&sum.to_le_bytes());
        assert_eq!(
            Parked::from_bytes(&later),
            Err(SnapshotError::UnsupportedVersion(PARKED_VERSION + 1))
        );
    }
}
