use std::error::Error;
use std::fmt;

use algebrix::{DQuat, DVec3, IVec3, Quat, Vec3};

use crate::body::{Bodies, Slot, inverse_mass_properties, validate_desc};
use crate::collision::MAX_CONTACTS;
use crate::collision::convex::{ConvexHull, MAX_CONVEX_POINTS};
use crate::collision::trimesh::{MAX_TRIANGLES, TriMesh};
use crate::compound::{Compound, MAX_COMPOUND_CHILDREN, MeshSlot};
use crate::joint::{JOINT_IMPULSES, JointSlot, JointState};
use crate::solver::{CachedManifold, CachedPoint};
use crate::{
    BodyHandle, BodyKind, CollisionFilter, CompoundChild, CompoundId, ConvexId, FrameDesc, FrameId,
    Frames, GridPosition, JointAxis, JointDesc, JointKind, JointLimit, JointMotor, Joints,
    MassProperties, Material, MeshId, PhysicsSettings, PhysicsWorld, RigidBodyDesc, Shape,
};

const MAGIC: [u8; 4] = *b"SLNR";
pub const SNAPSHOT_VERSION: u32 = 4;
/// Older versions still read, what they lack takes its default.
const OLDEST_VERSION: u32 = 2;

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
                out.u8(u8::from(child.surface.is_some()));
                out.material(&child.surface.unwrap_or_default());
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
        for slot in &self.bodies.assets.meshes {
            out.u32(slot.generation);
            match &slot.mesh {
                Some(mesh) => {
                    out.u8(1);
                    out.mesh(mesh);
                }
                None => out.u8(0),
            }
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
            out.f32(material.static_friction);
            out.f32(material.bounce_threshold);
            out.u8(material.friction_combine.tag());
            out.u8(material.restitution_combine.tag());
            out.f32(material.rolling_resistance);
            out.f32(material.spin_resistance);
            out.vec3(bodies.surface[dense]);
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

        // version 3: sensors with what they hold, so loading starts no false enter events
        out.len(self.sensors.slots.len());
        for (slot, generation) in self.sensors.slots.iter().zip(&self.sensors.generations) {
            out.u32(*generation);
            let Some(sensor) = slot else {
                out.u8(0);
                continue;
            };
            out.u8(1);
            let desc = &sensor.desc;
            out.shape(&desc.shape);
            out.position(desc.position);
            out.quat(desc.rotation);
            out.u32(desc.filter.frame.index());
            out.u32(desc.filter.mask);
            out.u8(u8::from(desc.filter.exclude.is_some()));
            out.body(desc.filter.exclude.unwrap_or(BodyHandle {
                index: 0,
                generation: 0,
            }));
            out.len(sensor.inside.len());
            for body in &sensor.inside {
                out.body(*body);
            }
        }
        out.len(self.fields.slots.len());
        for (slot, generation) in self.fields.slots.iter().zip(&self.fields.generations) {
            out.u32(*generation);
            let Some(desc) = slot else {
                out.u8(0);
                continue;
            };
            out.u8(1);
            out.field(desc);
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
            version: 0,
        };
        let version = reader.u32()?;
        if !(OLDEST_VERSION..=SNAPSHOT_VERSION).contains(&version) {
            return Err(SnapshotError::UnsupportedVersion(version));
        }
        reader.version = version;
        let mut checksum = Reader {
            data: stored,
            at: 0,
            version,
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
        let (sensors, fields) = if reader.version >= 3 {
            (
                read_sensors(&mut reader, limits)?,
                read_fields(&mut reader, limits)?,
            )
        } else {
            Default::default()
        };
        if reader.at != reader.data.len() {
            return Err(SnapshotError::Invalid("trailing data"));
        }
        let mut world = Self::from_parts(settings, frames, bodies, tick, accumulator);
        world.contact_cache = contact_cache;
        world.joints = joints;
        world.recheck_sleepers = recheck_sleepers;
        world.sensors = sensors;
        world.fields = fields;
        // shapes and frames are the loaded world's, check the sensors against them
        for sensor in world.sensors.slots.iter().flatten() {
            let desc = &sensor.desc;
            if world.bodies.assets.validate(&desc.shape).is_err()
                || !world.frames.contains(desc.filter.frame)
            {
                return Err(SnapshotError::Invalid("sensor"));
            }
        }
        if world
            .fields
            .slots
            .iter()
            .flatten()
            .any(|desc| !world.frames.contains(desc.frame))
        {
            return Err(SnapshotError::Invalid("field"));
        }
        Ok(world)
    }
}

fn read_sensors(
    reader: &mut Reader<'_>,
    limits: &SnapshotLimits,
) -> Result<crate::sensor::Sensors, SnapshotError> {
    let mut sensors = crate::sensor::Sensors::default();
    let count = reader.count(5, limits.max_slots)?;
    for _ in 0..count {
        sensors.generations.push(reader.u32()?);
        let slot = match reader.u8()? {
            0 => None,
            1 => {
                let shape = reader.shape()?;
                let position = reader.position()?;
                let rotation = reader.quat()?;
                let error = (rotation.length_squared() - 1.0).abs();
                if !error.is_finite() || error > 1e-3 {
                    return Err(SnapshotError::Invalid("sensor rotation"));
                }
                let frame = FrameId(reader.u32()?);
                let mask = reader.u32()?;
                let excluded = reader.u8()?;
                let excluded_body = reader.handle()?;
                let exclude = match excluded {
                    0 => None,
                    1 => Some(excluded_body),
                    _ => return Err(SnapshotError::Invalid("sensor filter")),
                };
                let held = reader.count(8, limits.max_slots)?;
                let mut inside = Vec::with_capacity(held);
                for _ in 0..held {
                    inside.push(reader.handle()?);
                }
                if !inside.windows(2).all(|pair| pair[0] < pair[1]) {
                    return Err(SnapshotError::Invalid("sensor bodies"));
                }
                Some(crate::sensor::Sensor {
                    desc: crate::sensor::SensorDesc {
                        shape,
                        position,
                        rotation,
                        filter: crate::QueryFilter {
                            frame,
                            mask,
                            exclude,
                        },
                    },
                    inside,
                })
            }
            _ => return Err(SnapshotError::Invalid("sensor slot")),
        };
        sensors.slots.push(slot);
    }
    Ok(sensors)
}

fn read_fields(
    reader: &mut Reader<'_>,
    limits: &SnapshotLimits,
) -> Result<crate::field::Fields, SnapshotError> {
    let mut fields = crate::field::Fields::default();
    let count = reader.count(5, limits.max_slots)?;
    for _ in 0..count {
        fields.generations.push(reader.u32()?);
        let slot = match reader.u8()? {
            0 => None,
            1 => {
                let desc = reader.field()?;
                desc.validate()
                    .map_err(|_| SnapshotError::Invalid("field"))?;
                Some(desc)
            }
            _ => return Err(SnapshotError::Invalid("field slot")),
        };
        fields.slots.push(slot);
    }
    Ok(fields)
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
    let mut compounds = Vec::with_capacity(compound_count);
    for _ in 0..compound_count {
        compounds.push(read_compound(reader)?);
    }
    let convex_count = reader.count(4 + 12 + 4, limits.max_slots)?;
    bodies.assets.convexes.reserve(convex_count);
    for _ in 0..convex_count {
        bodies.assets.convexes.push(read_convex(reader)?);
    }
    // compounds come first in the data, their hull parts after
    bodies.assets.compounds.reserve(compound_count);
    for (children, unit) in compounds {
        let compound = Compound::from_rebased(children, unit, &bodies.assets.convexes)
            .map_err(|_| SnapshotError::Invalid("compound"))?;
        bodies.assets.compounds.push(compound);
    }
    let mesh_count = reader.count(5, limits.max_slots)?;
    bodies.assets.meshes.reserve(mesh_count);
    for index in 0..mesh_count {
        let generation = reader.u32()?;
        let mesh = match reader.u8()? {
            0 => {
                bodies.assets.free_meshes.push(index as u32);
                None
            }
            1 => Some(read_mesh(reader, limits)?),
            _ => return Err(SnapshotError::Invalid("mesh slot")),
        };
        bodies.assets.meshes.push(MeshSlot { generation, mesh });
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
    let (density, friction, restitution) = (reader.f32()?, reader.f32()?, reader.f32()?);
    let mut surface = Vec3::ZERO;
    let mut material = Material {
        density,
        friction,
        restitution,
        ..Material::default()
    };
    if reader.version >= 3 {
        material.static_friction = reader.f32()?;
        material.bounce_threshold = reader.f32()?;
        let combine =
            |tag| crate::Combine::from_tag(tag).ok_or(SnapshotError::Invalid("material combine"));
        material.friction_combine = combine(reader.u8()?)?;
        material.restitution_combine = combine(reader.u8()?)?;
        material.rolling_resistance = reader.f32()?;
        material.spin_resistance = reader.f32()?;
        surface = reader.vec3()?;
    }
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
        surface_velocity: surface,
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
    bodies.surface.push(surface);
    bodies.filter.push(filter);
    bodies.asleep.push(asleep);
    bodies.can_sleep.push(can_sleep);
    bodies.still_time.push(still_time);
    Ok(())
}

fn read_compound(
    reader: &mut Reader<'_>,
) -> Result<(Vec<CompoundChild>, MassProperties), SnapshotError> {
    let unit = MassProperties {
        mass: reader.f32()?,
        inertia: reader.vec3()?,
    };
    let count = reader.count(CHILD_BYTES, MAX_COMPOUND_CHILDREN)?;
    let mut children = Vec::with_capacity(count);
    for _ in 0..count {
        let (shape, position, rotation, density_scale) = (
            reader.shape()?,
            reader.vec3()?,
            reader.quat()?,
            reader.f32()?,
        );
        let surface = if reader.version >= 3 {
            let has = reader.u8()?;
            let material = reader.material()?;
            match has {
                0 => None,
                1 => Some(material),
                _ => return Err(SnapshotError::Invalid("compound surface")),
            }
        } else {
            None
        };
        children.push(CompoundChild {
            shape,
            position,
            rotation,
            density_scale,
            surface,
        });
    }
    Ok((children, unit))
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

pub(crate) fn fnv1a(data: &[u8]) -> u64 {
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

    fn material(&mut self, material: &Material) {
        self.f32(material.density);
        self.f32(material.friction);
        self.f32(material.restitution);
        self.f32(material.static_friction);
        self.f32(material.bounce_threshold);
        self.u8(material.friction_combine.tag());
        self.u8(material.restitution_combine.tag());
        self.f32(material.rolling_resistance);
        self.f32(material.spin_resistance);
    }

    fn position(&mut self, value: GridPosition) {
        self.i32(value.cell.x);
        self.i32(value.cell.y);
        self.i32(value.cell.z);
        self.vec3(value.local);
    }

    fn body(&mut self, handle: BodyHandle) {
        self.u32(handle.index);
        self.u32(handle.generation);
    }

    fn field(&mut self, desc: &crate::field::FieldDesc) {
        use crate::field::{FieldRegion, ForceField};
        match desc.field {
            ForceField::Wind { velocity, drag } => {
                self.u8(0);
                self.vec3(velocity);
                self.f32(drag);
            }
            ForceField::Drag { linear, angular } => {
                self.u8(1);
                self.f32(linear);
                self.f32(angular);
            }
            ForceField::Fluid {
                level,
                density,
                drag,
            } => {
                self.u8(2);
                self.f32(level);
                self.f32(density);
                self.f32(drag);
            }
            ForceField::Attractor {
                at,
                strength,
                min_distance,
            } => {
                self.u8(3);
                self.position(at);
                self.f32(strength);
                self.f32(min_distance);
            }
        }
        match desc.region {
            FieldRegion::Everywhere => self.u8(0),
            FieldRegion::Sphere { center, radius } => {
                self.u8(1);
                self.position(center);
                self.f32(radius);
            }
            FieldRegion::Box {
                center,
                half_extents,
                rotation,
            } => {
                self.u8(2);
                self.position(center);
                self.vec3(half_extents);
                self.quat(rotation);
            }
        }
        self.u32(desc.filter.layers);
        self.u32(desc.filter.mask);
        // version 4: the field's frame
        self.u32(desc.frame.0);
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
            Shape::TriMesh(id) => (
                4,
                [f32::from_bits(id.index), f32::from_bits(id.generation), 0.0],
            ),
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
    /// The snapshot's format version, what it holds depends on it.
    version: u32,
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

    fn material(&mut self) -> Result<Material, SnapshotError> {
        let combine =
            |tag| crate::Combine::from_tag(tag).ok_or(SnapshotError::Invalid("material combine"));
        let material = Material {
            density: self.f32()?,
            friction: self.f32()?,
            restitution: self.f32()?,
            static_friction: self.f32()?,
            bounce_threshold: self.f32()?,
            friction_combine: combine(self.u8()?)?,
            restitution_combine: combine(self.u8()?)?,
            rolling_resistance: self.f32()?,
            spin_resistance: self.f32()?,
        };
        material
            .validate()
            .map_err(|_| SnapshotError::Invalid("material"))?;
        Ok(material)
    }

    fn position(&mut self) -> Result<GridPosition, SnapshotError> {
        let cell = IVec3::new(self.i32()?, self.i32()?, self.i32()?);
        let position = GridPosition {
            cell,
            local: self.vec3()?,
        };
        if position.rebased() != Some(position) {
            return Err(SnapshotError::Invalid("position"));
        }
        Ok(position)
    }

    fn field(&mut self) -> Result<crate::field::FieldDesc, SnapshotError> {
        use crate::field::{FieldDesc, FieldRegion, ForceField};
        let field = match self.u8()? {
            0 => ForceField::Wind {
                velocity: self.vec3()?,
                drag: self.f32()?,
            },
            1 => ForceField::Drag {
                linear: self.f32()?,
                angular: self.f32()?,
            },
            2 => ForceField::Fluid {
                level: self.f32()?,
                density: self.f32()?,
                drag: self.f32()?,
            },
            3 => ForceField::Attractor {
                at: self.position()?,
                strength: self.f32()?,
                min_distance: self.f32()?,
            },
            _ => return Err(SnapshotError::Invalid("field kind")),
        };
        let region = match self.u8()? {
            0 => FieldRegion::Everywhere,
            1 => FieldRegion::Sphere {
                center: self.position()?,
                radius: self.f32()?,
            },
            2 => FieldRegion::Box {
                center: self.position()?,
                half_extents: self.vec3()?,
                rotation: self.quat()?,
            },
            _ => return Err(SnapshotError::Invalid("field region")),
        };
        let filter = CollisionFilter {
            layers: self.u32()?,
            mask: self.u32()?,
        };
        let frame = if self.version >= 4 {
            crate::FrameId(self.u32()?)
        } else {
            crate::FrameId::ROOT
        };
        Ok(FieldDesc {
            field,
            region,
            filter,
            frame,
        })
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
            4 => Ok(Shape::TriMesh(MeshId {
                index: values[0].to_bits(),
                generation: values[1].to_bits(),
            })),
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
    fn a_version_2_snapshot_still_loads_with_material_defaults() {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let material = Material {
            density: 1234.5,
            friction: 0.375,
            restitution: 0.25,
            ..Material::default()
        };
        let body = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.5)).material(material))
            .unwrap();
        let mut bytes = world.save_snapshot();
        // version 2 had density, friction and restitution, then the filter: cut what 3 added
        let mut marker = 1234.5_f32.to_le_bytes().to_vec();
        marker.extend(0.375_f32.to_le_bytes());
        marker.extend(0.25_f32.to_le_bytes());
        let at = bytes
            .windows(marker.len())
            .position(|window| window == marker.as_slice())
            .unwrap();
        bytes.drain(at + marker.len()..at + marker.len() + 4 * 2 + 2 + 4 * 2 + 4 * 3);
        // and the sensor and field counts at the end, both empty
        let end = bytes.len() - CHECKSUM_BYTES;
        bytes.drain(end - 2 * 4..end);
        bytes[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let loaded = PhysicsWorld::load_snapshot(&resealed(bytes)).unwrap();
        let read = loaded.bodies.material[loaded.bodies.dense_index(body).unwrap()];
        assert_eq!(read, material);
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

    mod scripted {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use algebrix::{Quat, Vec3};

        use crate::{
            BodyHandle, BodyKind, CompoundChild, GridPosition, JobDispatch, JointDesc, JointDof,
            JointMotor, PhysicsError, PhysicsSettings, PhysicsWorld, QueryFilter, Ray,
            RigidBodyDesc, Sequential, Shape,
        };

        struct Workers(usize);

        impl JobDispatch for Workers {
            fn parallelism(&self) -> usize {
                self.0
            }

            fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
                let next = AtomicUsize::new(0);
                std::thread::scope(|scope| {
                    for _ in 0..self.0 {
                        scope.spawn(|| {
                            loop {
                                let index = next.fetch_add(1, Ordering::Relaxed);
                                if index >= jobs {
                                    break;
                                }
                                job(index);
                            }
                        });
                    }
                });
            }
        }

        struct Dice(u64);

        impl Dice {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }

            fn float(&mut self, low: f32, high: f32) -> f32 {
                low + (self.next() % 10_000) as f32 / 10_000.0 * (high - low)
            }

            fn spot(&mut self, reach: f32, high: f32) -> Vec3 {
                Vec3::new(
                    self.float(-reach, reach),
                    self.float(1.0, high),
                    self.float(-reach, reach),
                )
            }
        }

        /// One of everything the solver handles. `pyramid` wide boxes make a big colored island.
        fn scene(pyramid: usize) -> PhysicsWorld {
            let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
            let floor = RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(200.0, 1.0, 200.0)))
                .local_position(Vec3::new(0.0, -1.0, 0.0));
            let floor = world.spawn(floor).unwrap();
            for row in 0..pyramid {
                for column in 0..pyramid - row {
                    let x = column as f32 * 1.05 - (pyramid - row) as f32 * 0.525;
                    let at = Vec3::new(x, 0.5 + row as f32, -30.0);
                    let body = RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)));
                    world.spawn(body.local_position(at)).unwrap();
                }
            }
            let mut last = world
                .spawn(
                    RigidBodyDesc::fixed(Shape::sphere(0.1))
                        .local_position(Vec3::new(10.0, 8.0, 0.0)),
                )
                .unwrap();
            for link in 1..6 {
                let at = Vec3::new(10.0 + link as f32 * 0.6, 8.0, 0.0);
                let ball = RigidBodyDesc::dynamic(Shape::sphere(0.15)).local_position(at);
                let ball = world.spawn(ball).unwrap();
                let anchor = GridPosition::from_local(at - Vec3::X * 0.3).unwrap();
                let joint = world
                    .joint_at(JointDesc::ball(last, ball), anchor, Quat::IDENTITY)
                    .unwrap();
                world.add_joint(joint).unwrap();
                last = ball;
            }
            let wheel = RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(0.1, 0.5, 0.5)))
                .local_position(Vec3::new(-10.0, 2.0, 0.0));
            let wheel = world.spawn(wheel).unwrap();
            let anchor = GridPosition::from_local(Vec3::new(-10.0, 2.0, 0.0)).unwrap();
            let hinge = world
                .joint_at(JointDesc::hinge(floor, wheel), anchor, Quat::IDENTITY)
                .unwrap();
            let hinge = world.add_joint(hinge).unwrap();
            let motor = JointMotor::Velocity {
                speed: 2.0,
                max_force: 500.0,
            };
            world
                .set_joint_motor(hinge, JointDof::AngularX, Some(motor))
                .unwrap();
            let parts = [
                CompoundChild::new(Shape::cuboid(Vec3::splat(0.3)), Vec3::ZERO),
                CompoundChild::new(Shape::sphere(0.3), Vec3::new(0.6, 0.0, 0.0)),
            ];
            let compound = world.add_compound(&parts).unwrap();
            let (at, turn) = compound.placed(Vec3::new(0.0, 3.0, 8.0), Quat::IDENTITY);
            let body = RigidBodyDesc::dynamic(compound.shape).local_position(at);
            world.spawn(body.rotation(turn)).unwrap();
            let hull = world
                .add_convex(&[
                    Vec3::new(0.5, 0.0, 0.0),
                    Vec3::new(-0.5, 0.0, 0.0),
                    Vec3::new(0.0, 0.7, 0.0),
                    Vec3::new(0.0, 0.0, 0.5),
                    Vec3::new(0.0, 0.0, -0.5),
                ])
                .unwrap();
            let (at, turn) = hull.placed(Vec3::new(3.0, 3.0, 8.0), Quat::IDENTITY);
            let body = RigidBodyDesc::dynamic(hull.shape).local_position(at);
            world.spawn(body.rotation(turn)).unwrap();
            let mut heights = Vec::new();
            for z in 0..9 {
                for x in 0..9 {
                    heights.push(((x + z) % 3) as f32 * 0.1);
                }
            }
            let terrain = world.add_heightfield(9, 9, 1.0, 1.0, &heights).unwrap();
            let body = RigidBodyDesc::fixed(terrain).local_position(Vec3::new(12.0, 0.01, 8.0));
            world.spawn(body).unwrap();
            let mover = RigidBodyDesc::kinematic(Shape::cuboid(Vec3::new(1.0, 0.1, 1.0)))
                .local_position(Vec3::new(-4.0, 0.2, 8.0))
                .linear_velocity(Vec3::new(0.3, 0.0, 0.0));
            world.spawn(mover).unwrap();
            world
        }

        /// The same outside changes from the same dice, whatever ran before.
        fn script(world: &mut PhysicsWorld, dice: &mut Dice, thrown: &mut Vec<BodyHandle>) {
            match dice.next() % 12 {
                0 => {
                    let shape = match dice.next() % 3 {
                        0 => Shape::sphere(0.3),
                        1 => Shape::cuboid(Vec3::splat(0.3)),
                        _ => Shape::capsule(0.2, 0.3),
                    };
                    let at = dice.spot(12.0, 8.0);
                    let body = RigidBodyDesc::dynamic(shape).local_position(at);
                    thrown.push(world.spawn(body).unwrap());
                }
                1 if thrown.len() > 8 => {
                    world.despawn(thrown.remove(0)).unwrap();
                }
                2 => {
                    if let Some(body) = thrown.last() {
                        let push = Vec3::new(dice.float(-200.0, 200.0), 300.0, 0.0);
                        world.apply_impulse(*body, push).unwrap();
                    }
                }
                _ => {}
            }
        }

        fn run(pyramid: usize, steps: usize, dispatch: &dyn JobDispatch) -> Vec<u64> {
            let mut world = scene(pyramid);
            let mut dice = Dice(0x1234_5678_9abc_def1);
            let mut thrown = Vec::new();
            let mut hashes = Vec::with_capacity(steps);
            for _ in 0..steps {
                script(&mut world, &mut dice, &mut thrown);
                world.step_with(dispatch).unwrap();
                hashes.push(world.state_hash());
            }
            hashes
        }

        #[test]
        fn a_replay_gives_the_same_state_every_step_on_any_thread_count() {
            let first = run(4, 240, &Sequential);
            assert_eq!(run(4, 240, &Sequential), first);
            assert_eq!(run(4, 240, &Workers(4)), first);
        }

        #[test]
        #[cfg_attr(miri, ignore)]
        fn a_replay_of_a_big_colored_island_matches_on_threads() {
            let first = run(40, 60, &Sequential);
            assert_eq!(run(40, 60, &Workers(4)), first);
        }

        #[test]
        fn a_snapshot_mid_run_continues_exactly() {
            let mut world = scene(4);
            let mut dice = Dice(0x0fed_cba9_8765_4321);
            let mut thrown = Vec::new();
            for _ in 0..200 {
                script(&mut world, &mut dice, &mut thrown);
                world.step_fixed().unwrap();
            }
            let bytes = world.save_snapshot();
            let mut loaded = PhysicsWorld::load_snapshot(&bytes).unwrap();
            assert_eq!(loaded.state_hash(), world.state_hash());
            assert_eq!(loaded.save_snapshot(), bytes);
            let mut other = Dice(dice.0);
            let mut other_thrown = thrown.clone();
            for step in 0..200 {
                script(&mut world, &mut dice, &mut thrown);
                script(&mut loaded, &mut other, &mut other_thrown);
                world.step_fixed().unwrap();
                loaded.step_fixed().unwrap();
                assert_eq!(loaded.state_hash(), world.state_hash(), "step {step}");
            }
        }

        #[test]
        #[cfg_attr(miri, ignore)]
        fn huge_speeds_reach_the_solver_and_a_failed_step_is_recoverable() {
            for speed in [1e4_f32, 1e8, 1e15, 1e30] {
                let mut world = scene(4);
                let mut dice = Dice(0x5555_aaaa_1234_9876);
                let mut thrown = Vec::new();
                for _ in 0..30 {
                    script(&mut world, &mut dice, &mut thrown);
                    world.step_fixed().unwrap();
                }
                for _ in 0..6 {
                    let body = RigidBodyDesc::dynamic(Shape::sphere(0.3))
                        .local_position(dice.spot(10.0, 6.0));
                    let body = world.spawn(body).unwrap();
                    let aim = Vec3::new(dice.float(-1.0, 1.0), -1.0, dice.float(-1.0, 1.0));
                    world
                        .set_velocity(body, aim * speed, Vec3::new(speed, 0.0, 0.0))
                        .unwrap();
                    thrown.push(body);
                }
                let mut failed = false;
                for _ in 0..30 {
                    let before = world.save_snapshot();
                    if world.step_fixed().is_err() {
                        failed = true;
                        assert_eq!(world.save_snapshot(), before, "speed {speed}");
                        break;
                    }
                }
                // a failed step names no body, so the game removes what it threw
                for body in thrown {
                    if world.body_state(body).is_ok() {
                        world.despawn(body).unwrap();
                    }
                }
                // what it hit can have picked the speed up too, the world says which
                let mut removed = 0;
                for _ in 0..30 {
                    while let Err(error) = world.step_fixed() {
                        let body = world
                            .failed_body()
                            .unwrap_or_else(|| panic!("speed {speed}: {error} names no body"));
                        world.despawn(body).unwrap();
                        removed += 1;
                        assert!(removed < 200, "speed {speed}");
                    }
                }
                assert!(failed || removed == 0);
                for body in world.body_handles().collect::<Vec<_>>() {
                    let state = world.body_state(body).unwrap();
                    assert!(
                        state.linear_velocity.is_finite(),
                        "speed {speed}: {state:?}"
                    );
                }
            }
        }

        fn wild(dice: &mut Dice) -> f32 {
            match dice.next() % 7 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => -1e30,
                3 => 1e30,
                4 => 0.0,
                _ => dice.float(-1e4, 1e4),
            }
        }

        #[test]
        #[cfg_attr(miri, ignore)]
        fn churn_with_wild_input_never_panics_and_leaves_nothing() {
            let mut world = scene(8);
            let mut dice = Dice(0xdead_beef_cafe_f00d);
            let mut live: Vec<BodyHandle> = Vec::new();
            let mut gone: Vec<BodyHandle> = Vec::new();
            let mut most_contacts = 0;
            let filter = QueryFilter::default();
            for _ in 0..1500 {
                for _ in 0..3 {
                    match dice.next() % 10 {
                        0..=2 if live.len() < 300 => {
                            let at = dice.spot(30.0, 20.0);
                            let shape = match dice.next() % 3 {
                                0 => Shape::sphere(dice.float(0.05, 0.8)),
                                1 => Shape::cuboid(Vec3::splat(dice.float(0.05, 0.8))),
                                _ => Shape::capsule(dice.float(0.05, 0.4), dice.float(0.0, 0.8)),
                            };
                            let body = RigidBodyDesc::dynamic(shape).local_position(at);
                            live.push(world.spawn(body).unwrap());
                        }
                        3 if !live.is_empty() => {
                            let at = dice.next() as usize % live.len();
                            let body = live.swap_remove(at);
                            world.despawn(body).unwrap();
                            gone.push(body);
                        }
                        4 if !gone.is_empty() => {
                            // a stale handle fails, it never reaches another body
                            let body = gone[dice.next() as usize % gone.len()];
                            assert!(world.apply_impulse(body, Vec3::X).is_err());
                            assert!(world.body_state(body).is_err());
                        }
                        5 | 6 if !live.is_empty() => {
                            // bad numbers are turned down, good ones go in. only a sane range is
                            // applied by force, a real game clamps too
                            let body = live[dice.next() as usize % live.len()];
                            let value = Vec3::new(wild(&mut dice), wild(&mut dice), 0.0);
                            let result = world.apply_impulse(body, value);
                            assert_eq!(result.is_ok(), value.is_finite(), "{value:?}");
                            if result.is_ok() {
                                let calm = Vec3::new(0.0, 0.0, 0.0);
                                world.set_velocity(body, calm, calm).unwrap();
                            }
                        }
                        7 => {
                            let ray = Ray {
                                origin: GridPosition::from_local(dice.spot(30.0, 25.0)).unwrap(),
                                direction: Vec3::new(wild(&mut dice), -1.0, 0.0),
                                max_distance: 50.0,
                            };
                            // only that it answers, bad directions fail
                            let _ = world.raycast(&ray, &filter);
                        }
                        _ => {}
                    }
                }
                world.step_fixed().unwrap();
                let mut contacts = Vec::new();
                world.contacts(&mut contacts);
                most_contacts = most_contacts.max(contacts.len());
                assert!(contacts.iter().all(|contact| contact.load.is_finite()));
            }
            for body in world.body_handles().collect::<Vec<_>>() {
                let state = world.body_state(body).unwrap();
                assert!(state.position.local.is_finite(), "{state:?}");
                assert!(state.linear_velocity.is_finite(), "{state:?}");
                if state.kind == BodyKind::Dynamic {
                    assert!(state.rotation.is_finite(), "{state:?}");
                }
            }
            assert!(most_contacts > 0);
            for body in world.body_handles().collect::<Vec<_>>() {
                world.despawn(body).unwrap();
            }
            world.step_fixed().unwrap();
            let mut contacts = Vec::new();
            world.contacts(&mut contacts);
            assert_eq!(world.body_count(), 0);
            assert_eq!(world.joint_handles().count(), 0);
            assert!(contacts.is_empty());
            assert_eq!(world.island_stats().count, 0);
            let empty_error: Result<(), PhysicsError> = world.step_fixed();
            assert!(empty_error.is_ok());
        }
    }

    #[test]
    fn sensors_and_fields_come_back_from_a_snapshot_and_run_the_same() {
        use crate::{FieldDesc, FieldRegion, ForceField, SensorDesc};
        let (mut world, _, _) = sample();
        let zone = world
            .add_sensor(SensorDesc::new(
                Shape::cuboid(Vec3::splat(3.0)),
                GridPosition::from_local(Vec3::new(0.0, 1.0, 0.0)).unwrap(),
            ))
            .unwrap();
        let wind = world
            .add_field(
                FieldDesc::new(ForceField::Wind {
                    velocity: Vec3::new(3.0, 0.0, 0.0),
                    drag: 2.0,
                })
                .region(FieldRegion::Box {
                    center: GridPosition::from_local(Vec3::new(0.0, 2.0, 0.0)).unwrap(),
                    half_extents: Vec3::splat(4.0),
                    rotation: Quat::from_rotation_y(0.3),
                }),
            )
            .unwrap();
        for _ in 0..20 {
            world.step_fixed().unwrap();
        }
        let bytes = world.save_snapshot();
        let mut loaded = PhysicsWorld::load_snapshot(&bytes).unwrap();
        assert_eq!(loaded.save_snapshot(), bytes);
        assert_eq!(loaded.sensor_bodies(zone), world.sensor_bodies(zone));
        for _ in 0..30 {
            world.step_fixed().unwrap();
            loaded.step_fixed().unwrap();
            assert_eq!(loaded.sensor_events(), world.sensor_events());
        }
        assert_eq!(loaded.state_hash(), world.state_hash());
        // the handles still work on the loaded world
        assert!(loaded.remove_field(wind));
        // a field slot with a bad kind is refused, not a panic
        let mut wind_bytes = Vec::new();
        for number in [3.0_f32, 0.0, 0.0, 2.0] {
            wind_bytes.extend(number.to_le_bytes());
        }
        let at = bytes
            .windows(wind_bytes.len())
            .rposition(|window| window == wind_bytes.as_slice())
            .unwrap();
        let mut bad = bytes.clone();
        bad[at - 1] = 9;
        assert!(PhysicsWorld::load_snapshot(&resealed(bad)).is_err());
    }

    #[test]
    fn raw_handles_name_the_same_bodies_after_a_load_and_bad_ones_are_stale() {
        let (world, _, _) = sample();
        let handles: Vec<_> = world.body_handles().map(BodyHandle::to_raw).collect();
        let loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        for raw in &handles {
            let body = BodyHandle::from_raw(*raw);
            assert_eq!(
                loaded.body_state(body).unwrap().position,
                world.body_state(body).unwrap().position
            );
        }
        assert!(
            loaded
                .body_state(BodyHandle::from_raw((u32::MAX, 7)))
                .is_err()
        );
        let (index, generation) = handles[0];
        assert!(
            loaded
                .body_state(BodyHandle::from_raw((index, generation + 1)))
                .is_err()
        );
    }
}
