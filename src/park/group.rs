use std::collections::BTreeSet;

use crate::body::{BodyHandle, BodyKind};
use crate::error::PhysicsError;
use crate::joint::{JOINT_IMPULSES, JointDesc, JointHandle};
use crate::snapshot::{Reader, SnapshotError, Writer, read_joint_desc};
use crate::world::PhysicsWorld;

use super::{Motion, Parked, Unparked};

const MAGIC: [u8; 4] = *b"SLPG";
pub const PARKED_GROUP_VERSION: u32 = 1;
/// Most bodies, and most joints, one group holds.
const MOST_MEMBERS: usize = 4096;

/// Bodies held together with the joints among them: a ship docked at a station, a crane with
/// its arm. Each comes back where it was, the joints with what they carried.
#[derive(Debug, Clone, PartialEq)]
pub struct ParkedGroup {
    members: Vec<Parked>,
    joints: Vec<StoredJoint>,
}

/// A joint with its ends as places in the group's member list.
#[derive(Debug, Clone, Copy, PartialEq)]
struct StoredJoint {
    desc: JointDesc,
    ends: [u32; 2],
    impulses: [f32; JOINT_IMPULSES],
}

/// A group back in the world: its bodies in the order they were held, and its joints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnparkedGroup {
    pub members: Vec<Unparked>,
    pub joints: Vec<JointHandle>,
}

impl ParkedGroup {
    pub fn members(&self) -> &[Parked] {
        &self.members
    }

    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Writer::with_capacity(64 + self.members.len() * 256);
        out.bytes(&MAGIC);
        out.u32(PARKED_GROUP_VERSION);
        out.len(self.members.len());
        for member in &self.members {
            let bytes = member.to_bytes();
            out.len(bytes.len());
            out.bytes(&bytes);
        }
        out.len(self.joints.len());
        for joint in &self.joints {
            out.u32(joint.ends[0]);
            out.u32(joint.ends[1]);
            out.joint_desc(&joint.desc);
            for impulse in joint.impulses {
                out.f32(impulse);
            }
        }
        let sum = crate::snapshot::fnv1a(&out.data);
        out.u64(sum);
        out.data
    }

    /// # Errors
    /// Fails on bytes that are not a group of this version, cut short, changed, holding a body
    /// [`Parked::from_bytes`] refuses, or a joint that is bad or names no member.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        let bad = SnapshotError::Invalid("parked group");
        if bytes.len() < 24 || bytes[..4] != MAGIC {
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
        if version != PARKED_GROUP_VERSION {
            return Err(SnapshotError::UnsupportedVersion(version));
        }
        reader.version = crate::snapshot::SNAPSHOT_VERSION;
        let count = reader.count(20, MOST_MEMBERS)?;
        if count == 0 {
            return Err(bad);
        }
        let mut members = Vec::with_capacity(count);
        for _ in 0..count {
            let length = reader.count(1, usize::MAX)?;
            let start = reader.at;
            reader.at += length;
            members.push(Parked::from_bytes(&body[start..start + length])?);
        }
        let count = reader.count(8, MOST_MEMBERS)?;
        let mut joints = Vec::with_capacity(count);
        for _ in 0..count {
            let ends = [reader.u32()?, reader.u32()?];
            let desc = read_joint_desc(&mut reader)?;
            let mut impulses = [0.0; JOINT_IMPULSES];
            for impulse in &mut impulses {
                *impulse = reader.f32()?;
            }
            let fits = ends.iter().all(|end| (*end as usize) < members.len());
            if !fits || ends[0] == ends[1] || impulses.iter().any(|value| !value.is_finite()) {
                return Err(bad);
            }
            joints.push(StoredJoint {
                desc,
                ends,
                impulses,
            });
        }
        if reader.at != body.len() {
            return Err(bad);
        }
        Ok(Self { members, joints })
    }
}

impl PhysicsWorld {
    /// Takes `bodies` out of the world together, each held where it is as [`Self::hold`]
    /// does, with the joints among them. Between steps. Nothing changes when it fails.
    ///
    /// # Errors
    /// Fails on no bodies, one named twice, one [`Self::hold`] refuses for itself, or a joint
    /// from one of them to a body left behind.
    pub fn hold_group(&mut self, bodies: &[BodyHandle]) -> Result<ParkedGroup, PhysicsError> {
        if bodies.is_empty() || bodies.len() > MOST_MEMBERS {
            return Err(PhysicsError::OutOfRange);
        }
        let unique: BTreeSet<BodyHandle> = bodies.iter().copied().collect();
        if unique.len() != bodies.len() {
            return Err(PhysicsError::OutOfRange);
        }
        let place = |body: BodyHandle| bodies.iter().position(|member| *member == body);
        let mut found = BTreeSet::new();
        for body in bodies {
            let dense = self.bodies.dense_index(*body)?;
            if self.bodies.kind[dense] != BodyKind::Dynamic {
                return Err(PhysicsError::NotDynamic);
            }
            if self.breakables.holds(*body) {
                return Err(PhysicsError::ShapeInUse);
            }
            self.store_geometry(&self.bodies.shape[dense])?;
            for joint in self.joints.touching(*body) {
                let desc = self.joints.get(joint)?.desc;
                if place(desc.a).is_none() || place(desc.b).is_none() {
                    return Err(PhysicsError::BodyHasJoints);
                }
                found.insert(joint);
            }
        }
        if found.len() > MOST_MEMBERS {
            return Err(PhysicsError::OutOfRange);
        }
        let mut joints = Vec::with_capacity(found.len());
        for joint in found {
            let state = self.joints.get(joint)?;
            let (a, b) = (place(state.desc.a), place(state.desc.b));
            let (Some(a), Some(b)) = (a, b) else {
                return Err(PhysicsError::BodyHasJoints);
            };
            let mut desc = state.desc;
            // the ends stand for places in the group, not bodies of any world
            desc.a = BodyHandle {
                index: a as u32,
                generation: 0,
            };
            desc.b = BodyHandle {
                index: b as u32,
                generation: 0,
            };
            joints.push(StoredJoint {
                desc,
                ends: [a as u32, b as u32],
                impulses: state.impulses,
            });
            self.joints.remove(joint)?;
        }
        let mut members = Vec::with_capacity(bodies.len());
        for body in bodies {
            members.push(self.take_out(*body, Motion::Held)?);
        }
        Ok(ParkedGroup { members, joints })
    }

    /// Puts a group back, each body as [`Self::unpark`] does, then its joints, fixed to the
    /// same points of each body however its shape was built again. Between steps. Nothing
    /// stays behind when it fails.
    ///
    /// # Errors
    /// As [`Self::unpark`] for any member, or when the world refuses a joint.
    pub fn unpark_group(&mut self, group: &ParkedGroup) -> Result<UnparkedGroup, PhysicsError> {
        let mut members = Vec::with_capacity(group.members.len());
        let mut rebased = Vec::with_capacity(group.members.len());
        let mut joints = Vec::with_capacity(group.joints.len());
        let result = (|| {
            for parked in &group.members {
                let (unparked, center, axes) = self.put_back(parked)?;
                members.push(unparked);
                rebased.push((center, axes.conjugate()));
            }
            for joint in &group.joints {
                let [a, b] = joint.ends.map(|end| end as usize);
                let mut desc = joint.desc;
                desc.a = members[a].body;
                desc.b = members[b].body;
                let ((center_a, back_a), (center_b, back_b)) = (rebased[a], rebased[b]);
                desc.anchor_a = back_a.rotate_vec3(desc.anchor_a - center_a);
                desc.anchor_b = back_b.rotate_vec3(desc.anchor_b - center_b);
                desc.frame_a = (back_a * desc.frame_a).normalize();
                desc.frame_b = (back_b * desc.frame_b).normalize();
                let handle = self.add_joint(desc)?;
                self.joints.get_mut(handle)?.impulses = joint.impulses;
                joints.push(handle);
            }
            Ok(())
        })();
        if let Err(error) = result {
            for joint in joints {
                let _ = self.remove_joint(joint);
            }
            for member in &members {
                let _ = self.despawn(member.body);
            }
            return Err(error);
        }
        Ok(UnparkedGroup { members, joints })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compound::CompoundChild;
    use crate::world::testing::no_gravity;
    use crate::{RigidBodyDesc, Shape};
    use algebrix::Vec3;

    /// A hull and a ball held under it by a hinge, swinging, and a box left alone beside them.
    fn crane(world: &mut PhysicsWorld) -> (BodyHandle, BodyHandle, BodyHandle) {
        let hull = world
            .add_convex(&[
                Vec3::new(-1.0, -0.2, -0.5),
                Vec3::new(1.5, -0.2, -0.5),
                Vec3::new(-1.0, 0.3, -0.5),
                Vec3::new(-1.0, -0.2, 0.7),
                Vec3::new(1.2, 0.4, 0.6),
            ])
            .unwrap();
        let mut desc = RigidBodyDesc::dynamic(hull.shape)
            .local_position(Vec3::new(0.0, 5.0, 0.0))
            .angular_velocity(Vec3::new(0.0, 0.3, 0.0));
        desc.can_sleep = false;
        let arm = world.spawn(desc).unwrap();
        let mut desc = RigidBodyDesc::dynamic(Shape::sphere(0.3))
            .local_position(Vec3::new(0.5, 3.0, 0.0))
            .linear_velocity(Vec3::new(0.0, 0.0, 1.0));
        desc.can_sleep = false;
        let ball = world.spawn(desc).unwrap();
        let mut hinge = JointDesc::fixed(arm, ball);
        hinge.anchor_a = world
            .body_state(arm)
            .unwrap()
            .rotation
            .conjugate()
            .rotate_vec3(
                Vec3::new(0.5, 3.0, 0.0)
                    - world.body_state(arm).unwrap().position.to_dvec3().as_vec3(),
            );
        world.add_joint(hinge).unwrap();
        let other = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.2)))
                    .local_position(Vec3::new(9.0, 0.0, 0.0)),
            )
            .unwrap();
        (arm, ball, other)
    }

    #[test]
    fn a_group_held_and_put_back_goes_on_as_if_it_never_left() {
        let mut world = no_gravity();
        let (arm, ball, _) = crane(&mut world);
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        let mut stayed = world.clone();
        let group = world.hold_group(&[arm, ball]).unwrap();
        assert_eq!(group.joint_count(), 1);
        assert_eq!(world.joints().handles().count(), 0);
        let group = ParkedGroup::from_bytes(&group.to_bytes()).unwrap();
        let back = world.unpark_group(&group).unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
            stayed.step_fixed().unwrap();
        }
        for (now, then) in [(back.members[0].body, arm), (back.members[1].body, ball)] {
            let (a, b) = (
                world.body_state(now).unwrap(),
                stayed.body_state(then).unwrap(),
            );
            let gap = (a.position.to_dvec3() - b.position.to_dvec3()).length();
            assert!(gap < 1e-3, "{gap}");
        }
        // the ball still hangs where the joint holds it on the arm
        let (a, b) = (
            world.body_state(back.members[0].body).unwrap(),
            world.body_state(back.members[1].body).unwrap(),
        );
        let (c, d) = (
            stayed.body_state(arm).unwrap(),
            stayed.body_state(ball).unwrap(),
        );
        let held = (b.position.to_dvec3() - a.position.to_dvec3()).length();
        let held_then = (d.position.to_dvec3() - c.position.to_dvec3()).length();
        assert!((held - held_then).abs() < 1e-3, "{held} {held_then}");
    }

    #[test]
    fn a_joint_to_a_body_left_behind_refuses_and_changes_nothing() {
        let mut world = no_gravity();
        let (arm, ball, other) = crane(&mut world);
        let before = world.save_snapshot();
        assert_eq!(world.hold_group(&[arm]), Err(PhysicsError::BodyHasJoints));
        assert_eq!(world.hold_group(&[arm, arm]), Err(PhysicsError::OutOfRange));
        assert_eq!(world.hold_group(&[]), Err(PhysicsError::OutOfRange));
        assert_eq!(world.save_snapshot(), before);
        // one with no joints is a group of one
        let alone = world.hold_group(&[other]).unwrap();
        assert_eq!((alone.members().len(), alone.joint_count()), (1, 0));
        world.unpark_group(&alone).unwrap();
        let _ = ball;
    }

    #[test]
    fn a_compound_rebuilt_elsewhere_keeps_its_joint_on_the_same_point() {
        let mut world = no_gravity();
        let parts = [
            CompoundChild::new(Shape::cuboid(Vec3::splat(0.5)), Vec3::ZERO),
            CompoundChild::new(
                Shape::cuboid(Vec3::new(1.0, 0.25, 0.25)),
                Vec3::new(1.5, 0.0, 0.0),
            ),
        ];
        let info = world.add_compound(&parts).unwrap();
        let ship = world
            .spawn(RigidBodyDesc::dynamic(info.shape).local_position(Vec3::new(0.0, 0.0, 0.0)))
            .unwrap();
        let pod = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.2)).local_position(Vec3::new(3.0, 0.0, 0.0)),
            )
            .unwrap();
        let ship_state = world.body_state(ship).unwrap();
        let mut dock = JointDesc::fixed(ship, pod);
        dock.anchor_a = ship_state
            .rotation
            .conjugate()
            .rotate_vec3(Vec3::new(3.0, 0.0, 0.0) - ship_state.position.to_dvec3().as_vec3());
        world.add_joint(dock).unwrap();
        let group = world.hold_group(&[ship, pod]).unwrap();
        let bytes = group.to_bytes();
        // a fresh world, where the compound is built again
        let mut other = no_gravity();
        let back = other
            .unpark_group(&ParkedGroup::from_bytes(&bytes).unwrap())
            .unwrap();
        let (ship, pod) = (back.members[0].body, back.members[1].body);
        let joint = back.joints[0];
        let ends = other.joint_anchors(joint).unwrap();
        let gap = (ends.0.to_dvec3() - ends.1.to_dvec3()).length();
        assert!(gap < 1e-4, "{gap}");
        let at = other.body_state(pod).unwrap().position.to_dvec3();
        assert!(
            (at - algebrix::DVec3::new(3.0, 0.0, 0.0)).length() < 1e-4,
            "{at:?}"
        );
        let _ = ship;
    }

    #[test]
    fn bad_group_bytes_are_refused_and_never_panic() {
        let mut world = no_gravity();
        let (arm, ball, _) = crane(&mut world);
        let bytes = world.hold_group(&[arm, ball]).unwrap().to_bytes();
        assert!(ParkedGroup::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        assert!(ParkedGroup::from_bytes(b"SLPG").is_err());
        let mut seed = 11_u64;
        for _ in 0..2000 {
            let mut broken = bytes.clone();
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let at = (seed >> 33) as usize % broken.len();
            broken[at] ^= 1 << ((seed >> 20) % 8);
            if at.is_multiple_of(3) {
                broken.truncate(broken.len() - (seed as usize % 7));
            }
            let _ = ParkedGroup::from_bytes(&broken);
        }
    }
}
