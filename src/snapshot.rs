use std::error::Error;
use std::fmt;

use algebrix::{DQuat, DVec3, IVec3, Quat, Vec3};

use crate::body::{Bodies, Slot, inverse_mass_properties, validate_desc};
use crate::collision::MAX_CONTACTS;
use crate::collision::convex::{ConvexHull, MAX_CONVEX_POINTS};
use crate::collision::trimesh::{MAX_TRIANGLES, TriMesh};
use crate::compound::{Compound, MAX_COMPOUND_CHILDREN};
use crate::joint::{JOINT_IMPULSES, JointSlot, JointState};
use crate::solver::{CachedManifold, CachedPoint};
use crate::{
    BodyHandle, BodyKind, CollisionFilter, CompoundChild, CompoundId, ConvexId, FrameDesc, FrameId,
    Frames, GridPosition, JointAxis, JointDesc, JointKind, JointLimit, JointMotor, Joints,
    MassProperties, Material, MeshId, PhysicsSettings, PhysicsWorld, RigidBodyDesc, Shape,
};

const MAGIC: [u8; 4] = *b"SLNR";
pub const SNAPSHOT_VERSION: u32 = 1;

const HEADER_BYTES: usize = 8;
const CHECKSUM_BYTES: usize = 8;
const FRAME_BYTES: usize = 4 + 8 * 13;
const SLOT_BYTES: usize = 8;
const BODY_BYTES: usize =
    4 + 1 + 1 + 4 * 3 + 4 * 3 + 4 * 2 + 4 + 4 * 3 + 4 * 3 + 4 * 4 + 4 * 3 * 5 + 1 + 4;
const FLAG_ASLEEP: u8 = 1;
const FLAG_CAN_SLEEP: u8 = 2;
const MANIFOLD_BYTES: usize = 4 * 4 + 4 * 2 + 1;
const COMPOUND_BYTES: usize = 4 + 4 * 3 + 4;
const CHILD_BYTES: usize = 13 + 4 * 3 + 4 * 4 + 4;
const JOINT_SLOT_BYTES: usize = 4 + 1;
const FREE_SLOT: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotLimits {
    pub max_bytes: usize,
    pub max_frames: usize,
    pub max_slots: usize,
}

impl Default for SnapshotLimits {
    fn default() -> Self {
        Self {
            max_bytes: 1 << 30,
            max_frames: 1 << 16,
            max_slots: 1 << 24,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotError {
    TooLarge,
    Truncated,
    BadMagic,
    UnsupportedVersion(u32),
    BadChecksum,
    Invalid(&'static str),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => f.write_str("snapshot over limit"),
            Self::Truncated => f.write_str("snapshot truncated"),
            Self::BadMagic => f.write_str("not a snapshot"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported snapshot version {version}")
            }
            Self::BadChecksum => f.write_str("snapshot checksum mismatch"),
            Self::Invalid(what) => write!(f, "invalid snapshot: {what}"),
        }
    }
}

impl Error for SnapshotError {}

impl PhysicsWorld {
    /// Little endian on every platform. Holds everything a later step reads.
    pub fn save_snapshot(&self) -> Vec<u8> {
        let mut out = Writer::with_capacity(
            HEADER_BYTES
                + self.frames.len() * FRAME_BYTES
                + self.bodies.slots.len() * (SLOT_BYTES + 4)
                + self.bodies.len() * BODY_BYTES
                + 128,
        );
        out.bytes(&MAGIC);
        out.u32(SNAPSHOT_VERSION);
        out.vec3(self.settings.gravity);
        out.f32(self.settings.fixed_timestep);
        out.u32(self.settings.max_steps_per_advance);
        out.u32(self.settings.substeps);
        out.u64(self.tick);
        out.f64(self.accumulator);
        out.u8(u8::from(self.recheck_sleepers));

        let frames = &self.frames.descs()[1..];
        out.len(frames.len());
        for frame in frames {
            out.u32(frame.parent.index());
            out.dvec3(frame.translation);
            out.f64(frame.rotation.x);
            out.f64(frame.rotation.y);
            out.f64(frame.rotation.z);
            out.f64(frame.rotation.w);
            out.dvec3(frame.linear_velocity);
            out.dvec3(frame.angular_velocity);
        }

        out.len(self.bodies.assets.compounds.len());
        for compound in &self.bodies.assets.compounds {
            out.f32(compound.unit.mass);
            out.vec3(compound.unit.inertia);
            out.len(compound.children.len());
            for child in &compound.children {
                out.shape(&child.shape);
                out.vec3(child.position);
                out.quat(child.rotation);
                out.f32(child.density_scale);
            }
        }

        out.len(self.bodies.assets.convexes.len());
        for hull in &self.bodies.assets.convexes {
            out.f32(hull.unit.mass);
            out.vec3(hull.unit.inertia);
            out.len(hull.source.len());
            for point in &hull.source {
                out.vec3(*point);
            }
        }

        out.len(self.bodies.assets.meshes.len());
        for mesh in &self.bodies.assets.meshes {
            out.mesh(mesh);
        }

        out.len(self.bodies.slots.len());
        for slot in &self.bodies.slots {
            out.u32(slot.generation);
            out.u32(slot.dense);
        }
        out.len(self.bodies.free.len());
        for index in &self.bodies.free {
            out.u32(*index);
        }

        let bodies = &self.bodies;
        out.len(bodies.len());
        for dense in 0..bodies.len() {
            out.u32(bodies.owner[dense]);
            out.u8(kind_tag(bodies.kind[dense]));
            out.shape(&bodies.shape[dense]);
            let material = &bodies.material[dense];
            out.f32(material.density);
            out.f32(material.friction);
            out.f32(material.restitution);
            out.u32(bodies.filter[dense].layers);
            out.u32(bodies.filter[dense].mask);
            out.u32(bodies.frame[dense].index());
            let cell = bodies.cell[dense];
            out.i32(cell.x);
            out.i32(cell.y);
            out.i32(cell.z);
            out.vec3(bodies.local[dense]);
            for value in bodies.rotation[dense].to_array() {
                out.f32(value);
            }
            out.vec3(bodies.linear[dense]);
            out.vec3(bodies.angular[dense]);
            out.vec3(bodies.momentum[dense]);
            out.vec3(bodies.force[dense]);
            out.vec3(bodies.torque[dense]);
            let mut flags = 0;
            if bodies.asleep[dense] {
                flags |= FLAG_ASLEEP;
            }
            if bodies.can_sleep[dense] {
                flags |= FLAG_CAN_SLEEP;
            }
            out.u8(flags);
            out.f32(bodies.still_time[dense]);
        }

        out.len(self.contact_cache.len());
        for manifold in &self.contact_cache {
            out.u32(manifold.a.index);
            out.u32(manifold.a.generation);
            out.u32(manifold.b.index);
            out.u32(manifold.b.generation);
            out.u32(manifold.part_a);
            out.u32(manifold.part_b);
            out.u8(manifold.count as u8);
            for point in &manifold.points[..manifold.count] {
                out.vec3(point.local_a);
                out.vec3(point.impulse);
            }
        }

        out.len(self.joints.slots.len());
        for slot in &self.joints.slots {
            out.u32(slot.generation);
            let Some(state) = &slot.state else {
                out.u8(0);
                continue;
            };
            out.u8(1);
            out.joint(state);
        }
        out.len(self.joints.free.len());
        for index in &self.joints.free {
            out.u32(*index);
        }

        let checksum = fnv1a(&out.data);
        out.u64(checksum);
        out.data
    }

    /// Hash of the full simulation state, for replay checks.
    pub fn state_hash(&self) -> u64 {
        fnv1a(&self.save_snapshot())
    }

    /// # Errors
    /// See [`Self::load_snapshot_with_limits`].
    pub fn load_snapshot(data: &[u8]) -> Result<Self, SnapshotError> {
        Self::load_snapshot_with_limits(data, &SnapshotLimits::default())
    }

    /// Input is untrusted. Bad data returns an error, never panics.
    ///
    /// # Errors
    /// Fails on oversized, truncated, corrupt or inconsistent data, or an unknown version.
    pub fn load_snapshot_with_limits(
        data: &[u8],
        limits: &SnapshotLimits,
    ) -> Result<Self, SnapshotError> {
        if data.len() > limits.max_bytes {
            return Err(SnapshotError::TooLarge);
        }
        if data.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(SnapshotError::Truncated);
        }
        if data[..4] != MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        let (body, stored) = data.split_at(data.len() - CHECKSUM_BYTES);
        let mut reader = Reader {
            data: body,
            at: MAGIC.len(),
        };
        let version = reader.u32()?;
        if version != SNAPSHOT_VERSION {
            return Err(SnapshotError::UnsupportedVersion(version));
        }
        let mut checksum = Reader {
            data: stored,
            at: 0,
        };
        if checksum.u64()? != fnv1a(body) {
            return Err(SnapshotError::BadChecksum);
        }

        let settings = PhysicsSettings {
            gravity: reader.vec3()?,
            fixed_timestep: reader.f32()?,
            max_steps_per_advance: reader.u32()?,
            substeps: reader.u32()?,
        };
        settings
            .validate()
            .map_err(|_| SnapshotError::Invalid("settings"))?;
        let tick = reader.u64()?;
        let accumulator = reader.f64()?;
        if !accumulator.is_finite() || accumulator < 0.0 {
            return Err(SnapshotError::Invalid("accumulator"));
        }
        let recheck_sleepers = match reader.u8()? {
            0 => false,
            1 => true,
            _ => return Err(SnapshotError::Invalid("recheck flag")),
        };

        let frames = read_frames(&mut reader, limits)?;
        let bodies = read_bodies(&mut reader, &frames, limits)?;
        let contact_cache = read_contacts(&mut reader, limits)?;
        let joints = read_joints(&mut reader, &bodies, limits)?;
        if reader.at != reader.data.len() {
            return Err(SnapshotError::Invalid("trailing data"));
        }
        let mut world = Self::from_parts(settings, frames, bodies, tick, accumulator);
        world.contact_cache = contact_cache;
        world.joints = joints;
        world.recheck_sleepers = recheck_sleepers;
        Ok(world)
    }
}

fn read_frames(reader: &mut Reader<'_>, limits: &SnapshotLimits) -> Result<Frames, SnapshotError> {
    let count = reader.count(FRAME_BYTES, limits.max_frames)?;
    let mut frames = Frames::new();
    for _ in 0..count {
        let parent = FrameId(reader.u32()?);
        let translation = reader.dvec3()?;
        let rotation = DQuat::new(reader.f64()?, reader.f64()?, reader.f64()?, reader.f64()?);
        let linear_velocity = reader.dvec3()?;
        let angular_velocity = reader.dvec3()?;
        frames
            .restore(FrameDesc {
                parent,
                translation,
                rotation,
                linear_velocity,
                angular_velocity,
            })
            .map_err(|_| SnapshotError::Invalid("frame"))?;
    }
    Ok(frames)
}

fn read_bodies(
    reader: &mut Reader<'_>,
    frames: &Frames,
    limits: &SnapshotLimits,
) -> Result<Bodies, SnapshotError> {
    let mut bodies = Bodies::default();
    let compound_count = reader.count(COMPOUND_BYTES, limits.max_slots)?;
    bodies.assets.compounds.reserve(compound_count);
    for _ in 0..compound_count {
        bodies.assets.compounds.push(read_compound(reader)?);
    }
    let convex_count = reader.count(4 + 12 + 4, limits.max_slots)?;
    bodies.assets.convexes.reserve(convex_count);
    for _ in 0..convex_count {
        bodies.assets.convexes.push(read_convex(reader)?);
    }
    let mesh_count = reader.count(1, limits.max_slots)?;
    bodies.assets.meshes.reserve(mesh_count);
    for _ in 0..mesh_count {
        bodies.assets.meshes.push(read_mesh(reader, limits)?);
    }
    let slot_count = reader.count(SLOT_BYTES, limits.max_slots)?;
    bodies.slots.reserve(slot_count);
    for _ in 0..slot_count {
        bodies.slots.push(Slot {
            generation: reader.u32()?,
            dense: reader.u32()?,
        });
    }

    let free_count = reader.count(4, slot_count)?;
    let mut in_free_list = vec![false; slot_count];
    bodies.free.reserve(free_count);
    for _ in 0..free_count {
        let index = reader.u32()?;
        let seen = in_free_list
            .get_mut(index as usize)
            .ok_or(SnapshotError::Invalid("free list"))?;
        let slot = bodies.slots[index as usize];
        if *seen || !slot.is_free() || slot.generation == u32::MAX {
            return Err(SnapshotError::Invalid("free list"));
        }
        *seen = true;
        bodies.free.push(index);
    }

    let body_count = reader.count(BODY_BYTES, slot_count)?;
    for dense in 0..body_count {
        read_body(reader, frames, &mut bodies, dense)?;
    }

    let mut used = 0;
    for (index, slot) in bodies.slots.iter().enumerate() {
        if slot.is_free() {
            if slot.generation != u32::MAX && !in_free_list[index] {
                return Err(SnapshotError::Invalid("free list"));
            }
        } else {
            used += 1;
        }
    }
    if used != body_count {
        return Err(SnapshotError::Invalid("slots"));
    }
    Ok(bodies)
}

fn read_body(
    reader: &mut Reader<'_>,
    frames: &Frames,
    bodies: &mut Bodies,
    dense: usize,
) -> Result<(), SnapshotError> {
    let owner = reader.u32()?;
    let slot = bodies
        .slots
        .get(owner as usize)
        .ok_or(SnapshotError::Invalid("body owner"))?;
    if slot.dense == FREE_SLOT || slot.dense as usize != dense {
        return Err(SnapshotError::Invalid("body owner"));
    }
    let kind = kind_from_tag(reader.u8()?)?;
    let shape = reader.shape()?;
    let material = Material {
        density: reader.f32()?,
        friction: reader.f32()?,
        restitution: reader.f32()?,
    };
    let filter = CollisionFilter {
        layers: reader.u32()?,
        mask: reader.u32()?,
    };
    let frame = FrameId(reader.u32()?);
    if !frames.contains(frame) {
        return Err(SnapshotError::Invalid("body frame"));
    }
    let cell = IVec3::new(reader.i32()?, reader.i32()?, reader.i32()?);
    let local = reader.vec3()?;
    let position = GridPosition { cell, local };
    if position.rebased() != Some(position) {
        return Err(SnapshotError::Invalid("body position"));
    }
    let rotation = Quat::new(reader.f32()?, reader.f32()?, reader.f32()?, reader.f32()?);
    let error = (rotation.length_squared() - 1.0).abs();
    if error.is_nan() || error >= 1e-3 {
        return Err(SnapshotError::Invalid("body rotation"));
    }
    let linear = reader.vec3()?;
    let angular = reader.vec3()?;
    let momentum = reader.vec3()?;
    let force = reader.vec3()?;
    let torque = reader.vec3()?;
    if !momentum.is_finite() || !force.is_finite() || !torque.is_finite() {
        return Err(SnapshotError::Invalid("body state"));
    }
    let flags = reader.u8()?;
    if flags & !(FLAG_ASLEEP | FLAG_CAN_SLEEP) != 0 {
        return Err(SnapshotError::Invalid("body flags"));
    }
    let asleep = flags & FLAG_ASLEEP != 0;
    let can_sleep = flags & FLAG_CAN_SLEEP != 0;
    let still_time = reader.f32()?;
    if !still_time.is_finite() || still_time < 0.0 {
        return Err(SnapshotError::Invalid("body still time"));
    }
    // a sleeper is skipped by the step, so it must have nothing left to apply
    let at_rest = [linear, angular, momentum, force, torque]
        .iter()
        .all(|value| *value == Vec3::ZERO);
    if asleep && (kind != BodyKind::Dynamic || !can_sleep || !at_rest) {
        return Err(SnapshotError::Invalid("sleeping body"));
    }

    let desc = RigidBodyDesc {
        kind,
        shape,
        material,
        filter,
        frame,
        position,
        rotation,
        linear_velocity: linear,
        angular_velocity: angular,
        can_sleep,
    };
    validate_desc(&desc, &bodies.assets).map_err(|_| SnapshotError::Invalid("body"))?;
    let (inverse_mass, inverse_inertia) = inverse_mass_properties(kind, &desc, &bodies.assets);

    bodies.owner.push(owner);
    bodies.cell.push(cell);
    bodies.local.push(local);
    bodies.rotation.push(rotation);
    bodies.linear.push(linear);
    bodies.angular.push(angular);
    bodies.momentum.push(momentum);
    bodies.force.push(force);
    bodies.torque.push(torque);
    bodies.inverse_mass.push(inverse_mass);
    bodies.inverse_inertia.push(inverse_inertia);
    bodies.kind.push(kind);
    bodies.frame.push(frame);
    bodies.shape.push(shape);
    bodies.material.push(material);
    bodies.filter.push(filter);
    bodies.asleep.push(asleep);
    bodies.can_sleep.push(can_sleep);
    bodies.still_time.push(still_time);
    Ok(())
}

fn read_compound(reader: &mut Reader<'_>) -> Result<Compound, SnapshotError> {
    let unit = MassProperties {
        mass: reader.f32()?,
        inertia: reader.vec3()?,
    };
    let count = reader.count(CHILD_BYTES, MAX_COMPOUND_CHILDREN)?;
    let mut children = Vec::with_capacity(count);
    for _ in 0..count {
        children.push(CompoundChild {
            shape: reader.shape()?,
            position: reader.vec3()?,
            rotation: reader.quat()?,
            density_scale: reader.f32()?,
        });
    }
    Compound::from_rebased(children, unit).map_err(|_| SnapshotError::Invalid("compound"))
}

fn read_convex(reader: &mut Reader<'_>) -> Result<ConvexHull, SnapshotError> {
    let unit = MassProperties {
        mass: reader.f32()?,
        inertia: reader.vec3()?,
    };
    let count = reader.count(12, MAX_CONVEX_POINTS)?;
    let mut points = Vec::with_capacity(count);
    for _ in 0..count {
        points.push(reader.vec3()?);
    }
    ConvexHull::from_rebased(&points, unit).map_err(|_| SnapshotError::Invalid("convex"))
}

fn read_mesh(reader: &mut Reader<'_>, limits: &SnapshotLimits) -> Result<TriMesh, SnapshotError> {
    let bad = |_| SnapshotError::Invalid("mesh");
    match reader.u8()? {
        0 => {
            let vertex_count = reader.count(12, MAX_TRIANGLES)?;
            let mut vertices = Vec::with_capacity(vertex_count);
            for _ in 0..vertex_count {
                vertices.push(reader.vec3()?);
            }
            let count = reader.count(12, MAX_TRIANGLES)?;
            let mut triangles = Vec::with_capacity(count);
            for _ in 0..count {
                triangles.push([reader.u32()?, reader.u32()?, reader.u32()?]);
            }
            TriMesh::from_triangles(vertices, triangles).map_err(bad)
        }
        1 => {
            let (columns, rows) = (reader.u32()?, reader.u32()?);
            let (spacing_x, spacing_z) = (reader.f32()?, reader.f32()?);
            let count = reader.count(4, limits.max_bytes)?;
            let mut heights = Vec::with_capacity(count);
            for _ in 0..count {
                heights.push(reader.f32()?);
            }
            TriMesh::from_heights(columns, rows, spacing_x, spacing_z, &heights).map_err(bad)
        }
        _ => Err(SnapshotError::Invalid("mesh kind")),
    }
}

fn read_contacts(
    reader: &mut Reader<'_>,
    limits: &SnapshotLimits,
) -> Result<Vec<CachedManifold>, SnapshotError> {
    let count = reader.count(MANIFOLD_BYTES, limits.max_slots)?;
    let mut cache: Vec<CachedManifold> = Vec::with_capacity(count);
    for _ in 0..count {
        let a = BodyHandle {
            index: reader.u32()?,
            generation: reader.u32()?,
        };
        let b = BodyHandle {
            index: reader.u32()?,
            generation: reader.u32()?,
        };
        let (part_a, part_b) = (reader.u32()?, reader.u32()?);
        let points_count = reader.u8()? as usize;
        if points_count == 0 || points_count > MAX_CONTACTS {
            return Err(SnapshotError::Invalid("contact count"));
        }
        // the solver finds entries by binary search
        let key = (a, b, part_a, part_b);
        if cache
            .last()
            .is_some_and(|last| (last.a, last.b, last.part_a, last.part_b) >= key)
        {
            return Err(SnapshotError::Invalid("contact order"));
        }
        let mut points = [CachedPoint::default(); MAX_CONTACTS];
        for point in &mut points[..points_count] {
            point.local_a = reader.vec3()?;
            point.impulse = reader.vec3()?;
            // the normal impulse only ever pushes apart
            if !point.local_a.is_finite() || !point.impulse.is_finite() || point.impulse.x < 0.0 {
                return Err(SnapshotError::Invalid("contact"));
            }
        }
        cache.push(CachedManifold {
            a,
            b,
            part_a,
            part_b,
            points,
            count: points_count,
        });
    }
    Ok(cache)
}

fn read_joints(
    reader: &mut Reader<'_>,
    bodies: &Bodies,
    limits: &SnapshotLimits,
) -> Result<Joints, SnapshotError> {
    let mut joints = Joints::default();
    let slot_count = reader.count(JOINT_SLOT_BYTES, limits.max_slots)?;
    joints.slots.reserve(slot_count);
    for _ in 0..slot_count {
        let generation = reader.u32()?;
        let state = match reader.u8()? {
            0 => None,
            1 => Some(read_joint(reader, bodies)?),
            _ => return Err(SnapshotError::Invalid("joint slot")),
        };
        joints.slots.push(JointSlot { generation, state });
    }
    let free_count = reader.count(4, slot_count)?;
    let mut in_free_list = vec![false; slot_count];
    joints.free.reserve(free_count);
    for _ in 0..free_count {
        let index = reader.u32()?;
        let seen = in_free_list
            .get_mut(index as usize)
            .ok_or(SnapshotError::Invalid("joint free list"))?;
        let slot = &joints.slots[index as usize];
        if *seen || slot.state.is_some() || slot.generation == u32::MAX {
            return Err(SnapshotError::Invalid("joint free list"));
        }
        *seen = true;
        joints.free.push(index);
    }
    for (slot, in_list) in joints.slots.iter().zip(&in_free_list) {
        if slot.state.is_none() && slot.generation != u32::MAX && !in_list {
            return Err(SnapshotError::Invalid("joint free list"));
        }
    }
    joints.mark_loaded();
    Ok(joints)
}

fn read_joint(reader: &mut Reader<'_>, bodies: &Bodies) -> Result<JointState, SnapshotError> {
    let a = reader.handle()?;
    let b = reader.handle()?;
    let anchor_a = reader.vec3()?;
    let anchor_b = reader.vec3()?;
    let frame_a = reader.unit_quat()?;
    let frame_b = reader.unit_quat()?;
    let collide_connected = match reader.u8()? {
        0 => false,
        1 => true,
        _ => return Err(SnapshotError::Invalid("joint flag")),
    };
    let kind = match reader.u8()? {
        0 => {
            let mut axes = [JointAxis::FREE; 6];
            for axis in &mut axes {
                *axis = reader.joint_axis()?;
            }
            JointKind::Axes(axes)
        }
        1 => JointKind::Distance {
            min: reader.f32()?,
            max: reader.f32()?,
        },
        _ => return Err(SnapshotError::Invalid("joint kind")),
    };
    let desc = JointDesc {
        a,
        b,
        anchor_a,
        anchor_b,
        frame_a,
        frame_b,
        kind,
        collide_connected,
    };
    desc.validate()
        .map_err(|_| SnapshotError::Invalid("joint"))?;
    let (Ok(dense_a), Ok(dense_b)) = (bodies.dense_index(a), bodies.dense_index(b)) else {
        return Err(SnapshotError::Invalid("joint body"));
    };
    if bodies.frame[dense_a] != bodies.frame[dense_b]
        || (bodies.kind[dense_a] != BodyKind::Dynamic && bodies.kind[dense_b] != BodyKind::Dynamic)
    {
        return Err(SnapshotError::Invalid("joint body"));
    }
    let mut impulses = [0.0; JOINT_IMPULSES];
    for impulse in &mut impulses {
        *impulse = reader.f32()?;
        if !impulse.is_finite() {
            return Err(SnapshotError::Invalid("joint impulse"));
        }
    }
    let force = reader.vec3()?;
    let torque = reader.vec3()?;
    if !force.is_finite() || !torque.is_finite() {
        return Err(SnapshotError::Invalid("joint reaction"));
    }
    Ok(JointState {
        desc,
        impulses,
        force,
        torque,
    })
}

fn kind_tag(kind: BodyKind) -> u8 {
    match kind {
        BodyKind::Static => 0,
        BodyKind::Kinematic => 1,
        BodyKind::Dynamic => 2,
    }
}

fn kind_from_tag(tag: u8) -> Result<BodyKind, SnapshotError> {
    match tag {
        0 => Ok(BodyKind::Static),
        1 => Ok(BodyKind::Kinematic),
        2 => Ok(BodyKind::Dynamic),
        _ => Err(SnapshotError::Invalid("body kind")),
    }
}

fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct Writer {
    data: Vec<u8>,
}

impl Writer {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    fn bytes(&mut self, value: &[u8]) {
        self.data.extend_from_slice(value);
    }

    fn u8(&mut self, value: u8) {
        self.data.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn i32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    // storage indexes are u32, so every length fits
    fn len(&mut self, value: usize) {
        self.u32(value as u32);
    }

    fn vec3(&mut self, value: Vec3) {
        self.f32(value.x);
        self.f32(value.y);
        self.f32(value.z);
    }

    fn dvec3(&mut self, value: DVec3) {
        self.f64(value.x);
        self.f64(value.y);
        self.f64(value.z);
    }

    fn quat(&mut self, value: Quat) {
        for part in value.to_array() {
            self.f32(part);
        }
    }

    fn joint(&mut self, state: &JointState) {
        let desc = &state.desc;
        for handle in [desc.a, desc.b] {
            self.u32(handle.index);
            self.u32(handle.generation);
        }
        self.vec3(desc.anchor_a);
        self.vec3(desc.anchor_b);
        self.quat(desc.frame_a);
        self.quat(desc.frame_b);
        self.u8(u8::from(desc.collide_connected));
        match desc.kind {
            JointKind::Axes(axes) => {
                self.u8(0);
                for axis in &axes {
                    self.joint_axis(axis);
                }
            }
            JointKind::Distance { min, max } => {
                self.u8(1);
                self.f32(min);
                self.f32(max);
            }
        }
        for impulse in state.impulses {
            self.f32(impulse);
        }
        self.vec3(state.force);
        self.vec3(state.torque);
    }

    fn joint_axis(&mut self, axis: &JointAxis) {
        let (tag, min, max) = match axis.limit {
            JointLimit::Free => (0, 0.0, 0.0),
            JointLimit::Locked => (1, 0.0, 0.0),
            JointLimit::Range { min, max } => (2, min, max),
        };
        self.u8(tag);
        self.f32(min);
        self.f32(max);
        let (tag, values) = match axis.motor {
            None => (0, [0.0; 3]),
            Some(JointMotor::Velocity { speed, max_force }) => (1, [speed, max_force, 0.0]),
            Some(JointMotor::Position {
                target,
                max_speed,
                max_force,
            }) => (2, [target, max_speed, max_force]),
        };
        self.u8(tag);
        for value in values {
            self.f32(value);
        }
    }

    fn mesh(&mut self, mesh: &TriMesh) {
        if let Some((columns, rows, spacing_x, spacing_z)) = mesh.grid {
            self.u8(1);
            self.u32(columns);
            self.u32(rows);
            self.f32(spacing_x);
            self.f32(spacing_z);
            self.len(mesh.vertices.len());
            for vertex in &mesh.vertices {
                self.f32(vertex.y);
            }
        } else {
            self.u8(0);
            self.len(mesh.vertices.len());
            for vertex in &mesh.vertices {
                self.vec3(*vertex);
            }
            self.len(mesh.triangles.len());
            for triangle in &mesh.triangles {
                for index in triangle {
                    self.u32(*index);
                }
            }
        }
    }

    fn shape(&mut self, shape: &Shape) {
        let (tag, values) = match *shape {
            Shape::Sphere { radius } => (0, [radius, 0.0, 0.0]),
            Shape::Cuboid { half_extents } => (1, [half_extents.x, half_extents.y, half_extents.z]),
            Shape::Capsule {
                radius,
                half_height,
            } => (2, [radius, half_height, 0.0]),
            Shape::Compound(id) => (3, [f32::from_bits(id.0), 0.0, 0.0]),
            Shape::TriMesh(id) => (4, [f32::from_bits(id.0), 0.0, 0.0]),
            Shape::Convex(id) => (5, [f32::from_bits(id.0), 0.0, 0.0]),
        };
        self.u8(tag);
        for value in values {
            self.f32(value);
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], SnapshotError> {
        let end = self.at.checked_add(N).ok_or(SnapshotError::Truncated)?;
        let bytes = self
            .data
            .get(self.at..end)
            .ok_or(SnapshotError::Truncated)?;
        self.at = end;
        bytes.try_into().map_err(|_| SnapshotError::Truncated)
    }

    fn u8(&mut self) -> Result<u8, SnapshotError> {
        Ok(self.take::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn i32(&mut self) -> Result<i32, SnapshotError> {
        Ok(i32::from_le_bytes(self.take()?))
    }

    fn u64(&mut self) -> Result<u64, SnapshotError> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn f32(&mut self) -> Result<f32, SnapshotError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64, SnapshotError> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// Reads a count, checks it against `max` and the bytes left before anything allocates.
    fn count(&mut self, record_bytes: usize, max: usize) -> Result<usize, SnapshotError> {
        let count = self.u32()? as usize;
        if count > max {
            return Err(SnapshotError::Invalid("count over limit"));
        }
        let left = self.data.len() - self.at;
        if count
            .checked_mul(record_bytes)
            .is_none_or(|bytes| bytes > left)
        {
            return Err(SnapshotError::Truncated);
        }
        Ok(count)
    }

    fn vec3(&mut self) -> Result<Vec3, SnapshotError> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }

    fn dvec3(&mut self) -> Result<DVec3, SnapshotError> {
        Ok(DVec3::new(self.f64()?, self.f64()?, self.f64()?))
    }

    fn handle(&mut self) -> Result<BodyHandle, SnapshotError> {
        Ok(BodyHandle {
            index: self.u32()?,
            generation: self.u32()?,
        })
    }

    fn quat(&mut self) -> Result<Quat, SnapshotError> {
        Ok(Quat::new(
            self.f32()?,
            self.f32()?,
            self.f32()?,
            self.f32()?,
        ))
    }

    fn unit_quat(&mut self) -> Result<Quat, SnapshotError> {
        let value = self.quat()?;
        let error = (value.length_squared() - 1.0).abs();
        if error.is_nan() || error >= 1e-3 {
            return Err(SnapshotError::Invalid("joint frame"));
        }
        Ok(value)
    }

    fn joint_axis(&mut self) -> Result<JointAxis, SnapshotError> {
        let tag = self.u8()?;
        let (min, max) = (self.f32()?, self.f32()?);
        let limit = match tag {
            0 => JointLimit::Free,
            1 => JointLimit::Locked,
            2 => JointLimit::Range { min, max },
            _ => return Err(SnapshotError::Invalid("joint limit")),
        };
        let tag = self.u8()?;
        let values = [self.f32()?, self.f32()?, self.f32()?];
        let motor = match tag {
            0 => None,
            1 => Some(JointMotor::Velocity {
                speed: values[0],
                max_force: values[1],
            }),
            2 => Some(JointMotor::Position {
                target: values[0],
                max_speed: values[1],
                max_force: values[2],
            }),
            _ => return Err(SnapshotError::Invalid("joint motor")),
        };
        Ok(JointAxis { limit, motor })
    }

    fn shape(&mut self) -> Result<Shape, SnapshotError> {
        let tag = self.u8()?;
        let values = [self.f32()?, self.f32()?, self.f32()?];
        match tag {
            0 => Ok(Shape::sphere(values[0])),
            1 => Ok(Shape::cuboid(Vec3::new(values[0], values[1], values[2]))),
            2 => Ok(Shape::capsule(values[0], values[1])),
            3 => Ok(Shape::Compound(CompoundId(values[0].to_bits()))),
            4 => Ok(Shape::TriMesh(MeshId(values[0].to_bits()))),
            5 => Ok(Shape::Convex(ConvexId(values[0].to_bits()))),
            _ => Err(SnapshotError::Invalid("shape kind")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BodyHandle, JointDof, PhysicsError};

    fn sample() -> (PhysicsWorld, BodyHandle, BodyHandle) {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let frame = world
            .add_frame(FrameDesc {
                translation: DVec3::new(1.0e7, -2.5, 3.0),
                rotation: DQuat::new(0.0, 0.6, 0.0, 0.8),
                linear_velocity: DVec3::new(0.0, 1.0, 0.0),
                angular_velocity: DVec3::new(0.0, 0.1, 0.0),
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        let ground = world
            .spawn(RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(
                10.0, 1.0, 10.0,
            ))))
            .unwrap();
        let removed = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(1.0)))
            .unwrap();
        let spinner = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::capsule(0.5, 2.0))
                    .frame(frame)
                    .position(GridPosition {
                        cell: IVec3::new(5_000, -1, 7),
                        local: Vec3::new(1.0, 2.0, 3.0),
                    })
                    .angular_velocity(Vec3::new(1.0, 2.0, 3.0)),
            )
            .unwrap();
        world
            .spawn(
                RigidBodyDesc::kinematic(Shape::sphere(0.25))
                    .linear_velocity(Vec3::new(1.0, 0.0, 0.0)),
            )
            .unwrap();
        let swinging = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .local_position(Vec3::new(40.0, 0.0, 0.0)),
            )
            .unwrap();
        let old_joint = world
            .add_joint(JointDesc::distance(ground, swinging, 1.0, 50.0))
            .unwrap();
        world.remove_joint(old_joint).unwrap();
        world
            .add_joint(
                JointDesc::hinge(ground, swinging)
                    .anchors(Vec3::new(30.0, 0.0, 0.0), Vec3::new(-10.0, 0.0, 0.0))
                    .axis(
                        JointDof::AngularX,
                        JointAxis::range(-1.0, 1.0).motor(JointMotor::Velocity {
                            speed: 0.5,
                            max_force: 10.0,
                        }),
                    ),
            )
            .unwrap();
        world
            .add_joint(JointDesc::distance(ground, swinging, 1.0, 50.0))
            .unwrap();
        world.despawn(removed).unwrap();
        world
            .apply_force(spinner, Vec3::new(1.0, 2.0, 3.0))
            .unwrap();
        (world, spinner, removed)
    }

    /// Rewrites the checksum, so a test reaches the checks behind it.
    fn resealed(mut data: Vec<u8>) -> Vec<u8> {
        data.truncate(data.len() - CHECKSUM_BYTES);
        let checksum = fnv1a(&data);
        data.extend_from_slice(&checksum.to_le_bytes());
        data
    }

    #[test]
    fn round_trip_is_byte_exact() {
        let (mut world, _, _) = sample();
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        world.advance(0.001).unwrap();
        let bytes = world.save_snapshot();
        let loaded = PhysicsWorld::load_snapshot(&bytes).unwrap();
        assert_eq!(loaded.save_snapshot(), bytes);
        assert_eq!(loaded.tick(), world.tick());
        assert_eq!(loaded.bodies, world.bodies);
        assert_eq!(loaded.frames, world.frames);
    }

    #[test]
    fn loaded_world_continues_exactly_like_the_original() {
        let (mut world, spinner, _) = sample();
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        for _ in 0..300 {
            world
                .apply_torque(spinner, Vec3::new(0.5, 0.0, 0.0))
                .unwrap();
            loaded
                .apply_torque(spinner, Vec3::new(0.5, 0.0, 0.0))
                .unwrap();
            world.step_fixed().unwrap();
            loaded.step_fixed().unwrap();
        }
        assert_eq!(loaded.state_hash(), world.state_hash());
    }

    #[test]
    fn handles_survive_and_stale_ones_stay_stale() {
        let (world, spinner, removed) = sample();
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        assert_eq!(
            loaded.body_state(spinner).unwrap(),
            world.body_state(spinner).unwrap()
        );
        assert_eq!(loaded.body_state(removed), Err(PhysicsError::InvalidHandle));
        let next = loaded
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(1.0)))
            .unwrap();
        assert_eq!(next.index, removed.index);
        assert_ne!(next.generation, removed.generation);
    }

    #[test]
    fn same_run_gives_same_hash() {
        let run = || {
            let (mut world, spinner, _) = sample();
            for step in 0..600 {
                if step % 7 == 0 {
                    world
                        .apply_torque(spinner, Vec3::new(1.0, -2.0, 0.5))
                        .unwrap();
                }
                world.step_fixed().unwrap();
            }
            world.state_hash()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn header_is_magic_and_version() {
        let (world, _, _) = sample();
        let bytes = world.save_snapshot();
        assert_eq!(&bytes[..4], b"SLNR");
        assert_eq!(bytes[4..8], SNAPSHOT_VERSION.to_le_bytes());
    }

    #[test]
    fn every_truncation_fails() {
        let bytes = sample().0.save_snapshot();
        for len in 0..bytes.len() {
            assert!(
                PhysicsWorld::load_snapshot(&bytes[..len]).is_err(),
                "len {len}"
            );
        }
    }

    #[test]
    fn every_flipped_byte_fails() {
        let bytes = sample().0.save_snapshot();
        for at in 0..bytes.len() {
            let mut bad = bytes.clone();
            bad[at] ^= 0x40;
            assert!(PhysicsWorld::load_snapshot(&bad).is_err(), "byte {at}");
        }
    }

    #[test]
    fn random_corruption_past_the_checksum_never_panics() {
        let bytes = sample().0.save_snapshot();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let mut bad = bytes.clone();
            for _ in 0..=(next() % 4) {
                let at = (next() as usize) % (bad.len() - CHECKSUM_BYTES);
                bad[at] = next() as u8;
            }
            if let Ok(mut world) = PhysicsWorld::load_snapshot(&resealed(bad)) {
                // whatever loads must stay usable
                let saved = world.save_snapshot();
                assert!(PhysicsWorld::load_snapshot(&saved).is_ok());
                let _ = world.step_fixed();
            }
        }
    }

    fn sample_in_contact() -> PhysicsWorld {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(50.0, 1.0, 50.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
        for level in 0..3 {
            world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                        .local_position(Vec3::new(0.0, 0.5 + level as f32, 0.0)),
                )
                .unwrap();
        }
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        assert!(!world.contact_cache.is_empty());
        world
    }

    #[test]
    fn contact_cache_round_trips() {
        let world = sample_in_contact();
        let loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        assert_eq!(loaded.contact_cache, world.contact_cache);
    }

    #[test]
    fn corrupt_contact_cache_never_panics() {
        let bytes = sample_in_contact().save_snapshot();
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let tail = 4 * (MANIFOLD_BYTES + 24 * MAX_CONTACTS);
        for _ in 0..20_000 {
            let mut bad = bytes.clone();
            let body_end = bad.len() - CHECKSUM_BYTES;
            for _ in 0..=(next() % 3) {
                let at = body_end - 1 - (next() as usize) % tail.min(body_end);
                bad[at] = next() as u8;
            }
            if let Ok(mut world) = PhysicsWorld::load_snapshot(&resealed(bad)) {
                let _ = world.step_fixed();
            }
        }
    }

    #[test]
    fn asleep_body_with_motion_fails() {
        let mut world = sample_in_contact();
        for _ in 0..180 {
            world.step_fixed().unwrap();
        }
        assert!(world.bodies.asleep.contains(&true));
        let loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        assert_eq!(loaded.bodies, world.bodies);
        let dense = world
            .bodies
            .asleep
            .iter()
            .position(|asleep| *asleep)
            .unwrap();
        world.bodies.linear[dense] = Vec3::X;
        assert_eq!(
            PhysicsWorld::load_snapshot(&world.save_snapshot()).err(),
            Some(SnapshotError::Invalid("sleeping body"))
        );
    }

    #[test]
    fn asleep_static_body_fails() {
        let mut world = sample_in_contact();
        world.bodies.asleep[0] = true;
        assert_eq!(
            PhysicsWorld::load_snapshot(&world.save_snapshot()).err(),
            Some(SnapshotError::Invalid("sleeping body"))
        );
    }

    #[test]
    fn unsorted_contact_cache_fails() {
        let mut world = sample_in_contact();
        world.contact_cache.reverse();
        assert_eq!(
            PhysicsWorld::load_snapshot(&world.save_snapshot()).err(),
            Some(SnapshotError::Invalid("contact order"))
        );
    }

    #[test]
    fn appended_data_fails() {
        let mut bytes = sample().0.save_snapshot();
        bytes.insert(bytes.len() - CHECKSUM_BYTES, 0);
        assert_eq!(
            PhysicsWorld::load_snapshot(&resealed(bytes)).err(),
            Some(SnapshotError::Invalid("trailing data"))
        );
    }

    #[test]
    fn wrong_header_fails() {
        let bytes = sample().0.save_snapshot();
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert_eq!(
            PhysicsWorld::load_snapshot(&magic).err(),
            Some(SnapshotError::BadMagic)
        );
        let mut version = bytes;
        version[4] = 99;
        assert_eq!(
            PhysicsWorld::load_snapshot(&resealed(version)).err(),
            Some(SnapshotError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn huge_counts_fail_before_allocating() {
        let bytes = sample().0.save_snapshot();
        // gravity, timestep, step limit, substeps, tick, accumulator, recheck flag
        let frame_count_at = HEADER_BYTES + 12 + 4 + 4 + 4 + 8 + 8 + 1;
        let mut bad = bytes;
        bad[frame_count_at..frame_count_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(PhysicsWorld::load_snapshot(&resealed(bad)).is_err());
    }

    #[test]
    fn limits_apply() {
        let bytes = sample().0.save_snapshot();
        let tiny = SnapshotLimits {
            max_bytes: 16,
            ..SnapshotLimits::default()
        };
        assert_eq!(
            PhysicsWorld::load_snapshot_with_limits(&bytes, &tiny).err(),
            Some(SnapshotError::TooLarge)
        );
        let no_frames = SnapshotLimits {
            max_frames: 0,
            ..SnapshotLimits::default()
        };
        assert!(PhysicsWorld::load_snapshot_with_limits(&bytes, &no_frames).is_err());
        let no_slots = SnapshotLimits {
            max_slots: 1,
            ..SnapshotLimits::default()
        };
        assert!(PhysicsWorld::load_snapshot_with_limits(&bytes, &no_slots).is_err());
    }
}
