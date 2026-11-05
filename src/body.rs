use algebrix::{IVec3, Quat, Vec3};

use crate::{
    CollisionFilter, FrameId, GridPosition, MassProperties, Material, PhysicsError, Shape,
};

const FREE: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyHandle {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyKind {
    Static,
    Kinematic,
    Dynamic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBodyDesc {
    pub kind: BodyKind,
    pub shape: Shape,
    pub material: Material,
    pub filter: CollisionFilter,
    pub frame: FrameId,
    pub position: GridPosition,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

impl RigidBodyDesc {
    pub fn new(kind: BodyKind, shape: Shape) -> Self {
        Self {
            kind,
            shape,
            material: Material::default(),
            filter: CollisionFilter::default(),
            frame: FrameId::ROOT,
            position: GridPosition::ORIGIN,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
        }
    }

    pub fn dynamic(shape: Shape) -> Self {
        Self::new(BodyKind::Dynamic, shape)
    }

    pub fn kinematic(shape: Shape) -> Self {
        Self::new(BodyKind::Kinematic, shape)
    }

    pub fn fixed(shape: Shape) -> Self {
        Self::new(BodyKind::Static, shape)
    }

    pub fn material(mut self, material: Material) -> Self {
        self.material = material;
        self
    }

    pub fn filter(mut self, filter: CollisionFilter) -> Self {
        self.filter = filter;
        self
    }

    pub fn density(mut self, density: f32) -> Self {
        self.material.density = density;
        self
    }

    pub fn frame(mut self, frame: FrameId) -> Self {
        self.frame = frame;
        self
    }

    pub fn position(mut self, position: GridPosition) -> Self {
        self.position = position;
        self
    }

    pub fn local_position(mut self, local: Vec3) -> Self {
        self.position.local = local;
        self
    }

    pub fn rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn linear_velocity(mut self, velocity: Vec3) -> Self {
        self.linear_velocity = velocity;
        self
    }

    pub fn angular_velocity(mut self, velocity: Vec3) -> Self {
        self.angular_velocity = velocity;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyState {
    pub kind: BodyKind,
    pub frame: FrameId,
    pub position: GridPosition,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Slot {
    pub(crate) generation: u32,
    pub(crate) dense: u32,
}

impl Slot {
    pub(crate) fn is_free(self) -> bool {
        self.dense == FREE
    }
}

/// Parallel arrays in dense order, slots map handles to dense indexes.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Bodies {
    pub(crate) slots: Vec<Slot>,
    pub(crate) free: Vec<u32>,
    pub(crate) owner: Vec<u32>,
    pub(crate) cell: Vec<IVec3>,
    pub(crate) local: Vec<Vec3>,
    pub(crate) rotation: Vec<Quat>,
    pub(crate) linear: Vec<Vec3>,
    pub(crate) angular: Vec<Vec3>,
    pub(crate) momentum: Vec<Vec3>,
    pub(crate) force: Vec<Vec3>,
    pub(crate) torque: Vec<Vec3>,
    pub(crate) inverse_mass: Vec<f32>,
    pub(crate) inverse_inertia: Vec<Vec3>,
    pub(crate) kind: Vec<BodyKind>,
    pub(crate) frame: Vec<FrameId>,
    pub(crate) shape: Vec<Shape>,
    pub(crate) material: Vec<Material>,
    pub(crate) filter: Vec<CollisionFilter>,
}

impl Bodies {
    pub(crate) fn len(&self) -> usize {
        self.owner.len()
    }

    pub(crate) fn dense_index(&self, handle: BodyHandle) -> Result<usize, PhysicsError> {
        match self.slots.get(handle.index as usize) {
            Some(slot) if slot.generation == handle.generation && !slot.is_free() => {
                Ok(slot.dense as usize)
            }
            _ => Err(PhysicsError::InvalidHandle),
        }
    }

    pub(crate) fn handle_at(&self, dense: usize) -> BodyHandle {
        let index = self.owner[dense];
        BodyHandle {
            index,
            generation: self.slots[index as usize].generation,
        }
    }

    pub(crate) fn position(&self, dense: usize) -> GridPosition {
        GridPosition {
            cell: self.cell[dense],
            local: self.local[dense],
        }
    }

    /// Desc must be validated by the caller.
    pub(crate) fn insert(&mut self, desc: &RigidBodyDesc) -> Result<BodyHandle, PhysicsError> {
        let dense = u32::try_from(self.len())
            .ok()
            .filter(|dense| *dense != FREE)
            .ok_or(PhysicsError::OutOfRange)?;
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                let index = u32::try_from(self.slots.len())
                    .ok()
                    .filter(|index| *index != FREE)
                    .ok_or(PhysicsError::OutOfRange)?;
                self.slots.push(Slot {
                    generation: 0,
                    dense: FREE,
                });
                index
            }
        };
        let slot = &mut self.slots[index as usize];
        slot.dense = dense;
        let generation = slot.generation;

        let (inverse_mass, inverse_inertia) = inverse_mass_properties(desc.kind, desc);
        let rotation = desc.rotation.normalize();
        self.owner.push(index);
        self.cell.push(desc.position.cell);
        self.local.push(desc.position.local);
        self.rotation.push(rotation);
        self.linear.push(desc.linear_velocity);
        self.angular.push(desc.angular_velocity);
        self.momentum.push(momentum_from_velocity(
            desc.kind,
            rotation,
            inverse_inertia,
            desc.angular_velocity,
        ));
        self.force.push(Vec3::ZERO);
        self.torque.push(Vec3::ZERO);
        self.inverse_mass.push(inverse_mass);
        self.inverse_inertia.push(inverse_inertia);
        self.kind.push(desc.kind);
        self.frame.push(desc.frame);
        self.shape.push(desc.shape);
        self.material.push(desc.material);
        self.filter.push(desc.filter);
        Ok(BodyHandle { index, generation })
    }

    pub(crate) fn remove(&mut self, handle: BodyHandle) -> Result<(), PhysicsError> {
        let dense = self.dense_index(handle)?;
        self.owner.swap_remove(dense);
        self.cell.swap_remove(dense);
        self.local.swap_remove(dense);
        self.rotation.swap_remove(dense);
        self.linear.swap_remove(dense);
        self.angular.swap_remove(dense);
        self.momentum.swap_remove(dense);
        self.force.swap_remove(dense);
        self.torque.swap_remove(dense);
        self.inverse_mass.swap_remove(dense);
        self.inverse_inertia.swap_remove(dense);
        self.kind.swap_remove(dense);
        self.frame.swap_remove(dense);
        self.shape.swap_remove(dense);
        self.material.swap_remove(dense);
        self.filter.swap_remove(dense);
        if let Some(moved) = self.owner.get(dense) {
            self.slots[*moved as usize].dense = dense as u32;
        }
        let slot = &mut self.slots[handle.index as usize];
        slot.dense = FREE;
        // a slot at the last generation is retired, so old handles can never match again
        if slot.generation < u32::MAX {
            slot.generation += 1;
            self.free.push(handle.index);
        }
        Ok(())
    }
}

pub(crate) fn inverse_mass_properties(kind: BodyKind, desc: &RigidBodyDesc) -> (f32, Vec3) {
    if kind != BodyKind::Dynamic {
        return (0.0, Vec3::ZERO);
    }
    let MassProperties { mass, inertia } = desc.shape.mass_properties(desc.material.density);
    (mass.recip(), inertia.recip())
}

/// `R * diag(values) * R^T * vector`
pub(crate) fn apply_diagonal(rotation: Quat, values: Vec3, vector: Vec3) -> Vec3 {
    let body = rotation.conjugate().rotate_vec3(vector);
    rotation.rotate_vec3(Vec3::new(
        body.x * values.x,
        body.y * values.y,
        body.z * values.z,
    ))
}

pub(crate) fn momentum_from_velocity(
    kind: BodyKind,
    rotation: Quat,
    inverse_inertia: Vec3,
    angular: Vec3,
) -> Vec3 {
    if kind == BodyKind::Dynamic {
        apply_diagonal(rotation, inverse_inertia.recip(), angular)
    } else {
        Vec3::ZERO
    }
}

/// # Errors
/// Fails on bad shape, material, or non finite state.
pub(crate) fn validate_desc(desc: &RigidBodyDesc) -> Result<(), PhysicsError> {
    desc.shape.validate()?;
    desc.material.validate()?;
    if desc.kind == BodyKind::Dynamic {
        let (inverse_mass, inverse_inertia) = inverse_mass_properties(desc.kind, desc);
        if !inverse_mass.is_finite()
            || inverse_mass <= 0.0
            || !inverse_inertia.is_finite()
            || inverse_inertia.min_element() <= 0.0
        {
            return Err(PhysicsError::InvalidMaterial);
        }
    }
    if !desc.position.local.is_finite() {
        return Err(PhysicsError::NonFinite("position"));
    }
    if !desc.linear_velocity.is_finite() || !desc.angular_velocity.is_finite() {
        return Err(PhysicsError::NonFinite("velocity"));
    }
    let length_squared = desc.rotation.length_squared();
    if !length_squared.is_finite() || length_squared < 1e-12 {
        return Err(PhysicsError::NonFinite("rotation"));
    }
    let (_, inverse_inertia) = inverse_mass_properties(desc.kind, desc);
    let momentum = momentum_from_velocity(
        desc.kind,
        desc.rotation.normalize(),
        inverse_inertia,
        desc.angular_velocity,
    );
    if !momentum.is_finite() {
        return Err(PhysicsError::NonFinite("momentum"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc() -> RigidBodyDesc {
        RigidBodyDesc::dynamic(Shape::sphere(1.0))
    }

    #[test]
    fn insert_and_lookup() {
        let mut bodies = Bodies::default();
        let a = bodies.insert(&desc()).unwrap();
        let b = bodies.insert(&desc()).unwrap();
        assert_eq!(bodies.dense_index(a), Ok(0));
        assert_eq!(bodies.dense_index(b), Ok(1));
    }

    #[test]
    fn removed_handle_goes_stale_and_slot_is_reused() {
        let mut bodies = Bodies::default();
        let a = bodies.insert(&desc()).unwrap();
        let b = bodies.insert(&desc()).unwrap();
        bodies.remove(a).unwrap();
        assert_eq!(bodies.dense_index(a), Err(PhysicsError::InvalidHandle));
        assert_eq!(bodies.remove(a), Err(PhysicsError::InvalidHandle));
        assert_eq!(bodies.dense_index(b), Ok(0));
        let c = bodies.insert(&desc()).unwrap();
        assert_eq!(c.index, a.index);
        assert_ne!(c.generation, a.generation);
        assert_eq!(bodies.dense_index(a), Err(PhysicsError::InvalidHandle));
        assert_eq!(bodies.dense_index(c), Ok(1));
    }

    #[test]
    fn last_generation_retires_the_slot() {
        let mut bodies = Bodies::default();
        let a = bodies.insert(&desc()).unwrap();
        bodies.slots[a.index as usize].generation = u32::MAX;
        let old = BodyHandle {
            index: a.index,
            generation: u32::MAX,
        };
        bodies.remove(old).unwrap();
        assert!(bodies.free.is_empty());
        let next = bodies.insert(&desc()).unwrap();
        assert_ne!(next.index, a.index);
    }

    #[test]
    fn bad_desc_fails() {
        let mut bad = desc();
        bad.linear_velocity = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(validate_desc(&bad).is_err());
        let mut zero = desc();
        zero.rotation = Quat::new(0.0, 0.0, 0.0, 0.0);
        assert!(validate_desc(&zero).is_err());
        let tiny = desc().density(f32::from_bits(1));
        assert!(validate_desc(&tiny).is_err());
        assert_eq!(validate_desc(&desc()), Ok(()));
    }

    #[test]
    fn only_dynamic_has_mass() {
        let fixed = RigidBodyDesc::fixed(Shape::sphere(1.0));
        assert_eq!(
            inverse_mass_properties(fixed.kind, &fixed),
            (0.0, Vec3::ZERO)
        );
        let (inverse_mass, _) = inverse_mass_properties(BodyKind::Dynamic, &desc());
        assert!(inverse_mass > 0.0);
    }
}
