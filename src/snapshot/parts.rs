use algebrix::{DQuat, IVec3, Quat, Vec3};

use super::{
    BODY_BYTES, CHILD_BYTES, COMPOUND_BYTES, FLAG_ASLEEP, FLAG_CAN_SLEEP, FRAME_BYTES, FREE_SLOT,
    JOINT_SLOT_BYTES, MANIFOLD_BYTES, Reader, SLOT_BYTES, SnapshotError, SnapshotLimits, Writer,
};
use crate::body::{Bodies, Slot, inverse_mass_properties, validate_desc};
use crate::collision::MAX_CONTACTS;
use crate::collision::convex::{ConvexHull, MAX_CONVEX_POINTS};
use crate::collision::trimesh::{MAX_TRIANGLES, TriMesh};
use crate::compound::{Compound, MAX_COMPOUND_CHILDREN, MeshSlot};
use crate::joint::{JOINT_IMPULSES, JointSlot, JointState};
use crate::solver::{CachedManifold, CachedPoint};
use crate::{
    BodyHandle, BodyKind, CollisionFilter, CompoundChild, CompoundId, ConvexId, FrameDesc, FrameId,
    FrameMotion, Frames, GravitySource, GridPosition, JointAxis, JointDesc, JointKind, JointLimit,
    JointMotor, Joints, MAX_GRAVITY_SOURCES, MassProperties, Material, MeshId, Orbit,
    RigidBodyDesc, Shape,
};

pub(super) fn read_sensors(
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

pub(super) fn read_fields(
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

pub(super) fn read_frames(
    reader: &mut Reader<'_>,
    limits: &SnapshotLimits,
) -> Result<Frames, SnapshotError> {
    let count = reader.count(FRAME_BYTES, limits.max_frames)?;
    let mut frames = Frames::new();
    for _ in 0..count {
        let parent = FrameId(reader.u32()?);
        let translation = reader.dvec3()?;
        let rotation = DQuat::new(reader.f64()?, reader.f64()?, reader.f64()?, reader.f64()?);
        let linear_velocity = reader.dvec3()?;
        let angular_velocity = reader.dvec3()?;
        let reach = if reader.version >= 10 {
            reader.f64()?
        } else {
            0.0
        };
        let motion = if reader.version >= 9 {
            let spin = reader.dvec3()?;
            let orbit = match reader.u8()? {
                0 => None,
                1 => Some(Orbit {
                    mu: reader.f64()?,
                    semi_major_axis: reader.f64()?,
                    eccentricity: reader.f64()?,
                    inclination: reader.f64()?,
                    ascending_node: reader.f64()?,
                    periapsis: reader.f64()?,
                    mean_anomaly: reader.f64()?,
                }),
                _ => return Err(SnapshotError::Invalid("frame orbit")),
            };
            FrameMotion { spin, orbit }
        } else {
            FrameMotion::default()
        };
        frames
            .restore(FrameDesc {
                parent,
                translation,
                rotation,
                linear_velocity,
                angular_velocity,
                motion,
                reach,
            })
            .map_err(|_| SnapshotError::Invalid("frame"))?;
    }
    Ok(frames)
}

pub(super) fn read_gravity(
    reader: &mut Reader<'_>,
    frames: &Frames,
) -> Result<Vec<Option<GravitySource>>, SnapshotError> {
    let count = reader.count(1, MAX_GRAVITY_SOURCES)?;
    let mut sources = Vec::with_capacity(count);
    for _ in 0..count {
        let source = match reader.u8()? {
            0 => None,
            1 => {
                let source = GravitySource {
                    frame: FrameId(reader.u32()?),
                    center: reader.dvec3()?,
                    mu: reader.f64()?,
                    radius: reader.f64()?,
                };
                if !frames.contains(source.frame) || !source.is_valid() {
                    return Err(SnapshotError::Invalid("gravity source"));
                }
                Some(source)
            }
            _ => return Err(SnapshotError::Invalid("gravity source")),
        };
        sources.push(source);
    }
    Ok(sources)
}

pub(super) fn read_bodies(
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

pub(super) fn read_contacts(
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
            if reader.version >= 8 {
                point.local_b = reader.vec3()?;
            }
            // the normal impulse only ever pushes apart
            if !point.local_a.is_finite()
                || !point.local_b.is_finite()
                || !point.impulse.is_finite()
                || point.impulse.x < 0.0
            {
                return Err(SnapshotError::Invalid("contact"));
            }
        }
        // older snapshots kept no normal, those contacts are found again
        let (normal, pose) = if reader.version >= 8 {
            let normal = reader.vec3()?;
            let pose = (reader.vec3()?, reader.quat()?);
            let unit = |length: f32| (length - 1.0).abs() < 1e-3;
            if !pose.0.is_finite()
                || !unit(pose.1.length())
                || !(normal == Vec3::ZERO || unit(normal.length()))
            {
                return Err(SnapshotError::Invalid("contact"));
            }
            (normal, pose)
        } else {
            (Vec3::ZERO, (Vec3::ZERO, Quat::IDENTITY))
        };
        cache.push(CachedManifold {
            a,
            b,
            part_a,
            part_b,
            points,
            count: points_count,
            normal,
            pose,
        });
    }
    Ok(cache)
}

pub(super) fn read_joints(
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
    let (break_force, break_torque) = if reader.version >= 5 {
        (reader.f32()?, reader.f32()?)
    } else {
        (f32::INFINITY, f32::INFINITY)
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
        break_force,
        break_torque,
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

pub(super) fn kind_tag(kind: BodyKind) -> u8 {
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

impl Writer {
    pub(crate) fn material(&mut self, material: &Material) {
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

    pub(super) fn field(&mut self, desc: &crate::field::FieldDesc) {
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

    pub(super) fn joint(&mut self, state: &JointState) {
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
        self.f32(desc.break_force);
        self.f32(desc.break_torque);
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

    pub(super) fn mesh(&mut self, mesh: &TriMesh) {
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

    pub(crate) fn shape(&mut self, shape: &Shape) {
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

impl Reader<'_> {
    pub(crate) fn material(&mut self) -> Result<Material, SnapshotError> {
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

    pub(crate) fn shape(&mut self) -> Result<Shape, SnapshotError> {
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
