use std::error::Error;
use std::fmt;

use algebrix::{DQuat, DVec3, IVec3, Quat, Vec3};

use crate::body::{Bodies, Slot, inverse_mass_properties, validate_desc};
use crate::{
    BodyKind, FrameDesc, FrameId, Frames, GridPosition, Material, PhysicsSettings, PhysicsWorld,
    RigidBodyDesc, Shape,
};

const MAGIC: [u8; 4] = *b"SLNR";
pub const SNAPSHOT_VERSION: u32 = 1;

const HEADER_BYTES: usize = 8;
const CHECKSUM_BYTES: usize = 8;
const FRAME_BYTES: usize = 4 + 8 * 13;
const SLOT_BYTES: usize = 8;
const BODY_BYTES: usize = 4 + 1 + 1 + 4 * 3 + 4 * 3 + 4 + 4 * 3 + 4 * 3 + 4 * 4 + 4 * 3 * 5;
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
        out.u64(self.tick);
        out.f64(self.accumulator);

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
        };
        settings
            .validate()
            .map_err(|_| SnapshotError::Invalid("settings"))?;
        let tick = reader.u64()?;
        let accumulator = reader.f64()?;
        if !accumulator.is_finite() || accumulator < 0.0 {
            return Err(SnapshotError::Invalid("accumulator"));
        }

        let frames = read_frames(&mut reader, limits)?;
        let bodies = read_bodies(&mut reader, &frames, limits)?;
        if reader.at != reader.data.len() {
            return Err(SnapshotError::Invalid("trailing data"));
        }
        Ok(Self::from_parts(
            settings,
            frames,
            bodies,
            tick,
            accumulator,
        ))
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

    let desc = RigidBodyDesc {
        kind,
        shape,
        material,
        frame,
        position,
        rotation,
        linear_velocity: linear,
        angular_velocity: angular,
    };
    validate_desc(&desc).map_err(|_| SnapshotError::Invalid("body"))?;
    let (inverse_mass, inverse_inertia) = inverse_mass_properties(kind, &desc);

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
    Ok(())
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

    fn shape(&mut self, shape: &Shape) {
        let (tag, values) = match *shape {
            Shape::Sphere { radius } => (0, [radius, 0.0, 0.0]),
            Shape::Cuboid { half_extents } => (1, [half_extents.x, half_extents.y, half_extents.z]),
            Shape::Capsule {
                radius,
                half_height,
            } => (2, [radius, half_height, 0.0]),
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

    fn shape(&mut self) -> Result<Shape, SnapshotError> {
        let tag = self.u8()?;
        let values = [self.f32()?, self.f32()?, self.f32()?];
        match tag {
            0 => Ok(Shape::sphere(values[0])),
            1 => Ok(Shape::cuboid(Vec3::new(values[0], values[1], values[2]))),
            2 => Ok(Shape::capsule(values[0], values[1])),
            _ => Err(SnapshotError::Invalid("shape kind")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BodyHandle, PhysicsError};

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
        world
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
    fn format_is_stable() {
        let (world, _, _) = sample();
        let bytes = world.save_snapshot();
        assert_eq!(&bytes[..4], b"SLNR");
        assert_eq!(bytes[4..8], SNAPSHOT_VERSION.to_le_bytes());
        assert_eq!(
            fnv1a(&bytes),
            GOLDEN_HASH,
            "format changed, bump the version"
        );
    }

    const GOLDEN_HASH: u64 = 11172332768420601478;

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
        // gravity, timestep, step limit, tick, accumulator
        let frame_count_at = HEADER_BYTES + 12 + 4 + 4 + 8 + 8;
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
