use std::f32::consts::PI;

use algebrix::{Quat, Vec3};

use crate::body::BodyKind;
use crate::collision::CollisionFilter;
use crate::error::PhysicsError;
use crate::position::GridPosition;
use crate::shape::Shape;
use crate::world::PhysicsWorld;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldHandle {
    index: u32,
    generation: u32,
}

/// A force on bodies, worked out once at the start of each step from where they are and how
/// they move then. A host changes one between steps for gusts or tides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ForceField {
    /// Air moving at `velocity`: force `drag * frontal area * (air - body velocity)`.
    Wind { velocity: Vec3, drag: f32 },
    /// Force `-linear * velocity` and torque `-angular * spin`.
    Drag { linear: f32, angular: f32 },
    /// Fluid below `level`, measured against gravity from the field's frame origin: lift of the
    /// volume under it, and `drag` times the share under it against the body's velocity.
    Fluid { level: f32, density: f32, drag: f32 },
    /// Pull toward `at`, `strength * mass / distance^2`, the distance no less than
    /// `min_distance`.
    Attractor {
        at: GridPosition,
        strength: f32,
        min_distance: f32,
    },
}

/// Where a field acts, by a body's center.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldRegion {
    Everywhere,
    Sphere {
        center: GridPosition,
        radius: f32,
    },
    Box {
        center: GridPosition,
        half_extents: Vec3,
        rotation: Quat,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldDesc {
    pub field: ForceField,
    pub region: FieldRegion,
    /// Bodies whose layers meet `mask` feel it.
    pub filter: CollisionFilter,
}

impl FieldDesc {
    pub fn new(field: ForceField) -> Self {
        Self {
            field,
            region: FieldRegion::Everywhere,
            filter: CollisionFilter::default(),
        }
    }

    pub fn region(mut self, region: FieldRegion) -> Self {
        self.region = region;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), PhysicsError> {
        let finite = |value: f32| value.is_finite();
        let positive = |value: f32| value.is_finite() && value >= 0.0;
        let field = match self.field {
            ForceField::Wind { velocity, drag } => velocity.is_finite() && positive(drag),
            ForceField::Drag { linear, angular } => positive(linear) && positive(angular),
            ForceField::Fluid {
                level,
                density,
                drag,
            } => finite(level) && positive(density) && positive(drag),
            ForceField::Attractor {
                at,
                strength,
                min_distance,
            } => {
                at.local.is_finite()
                    && finite(strength)
                    && min_distance.is_finite()
                    && min_distance > 0.0
            }
        };
        let region = match self.region {
            FieldRegion::Everywhere => true,
            FieldRegion::Sphere { center, radius } => center.local.is_finite() && positive(radius),
            FieldRegion::Box {
                center,
                half_extents,
                rotation,
            } => {
                center.local.is_finite()
                    && half_extents.is_finite()
                    && half_extents.min_element() >= 0.0
                    && rotation.is_finite()
                    && rotation.length_squared() > 1e-12
            }
        };
        if field && region {
            Ok(())
        } else {
            Err(PhysicsError::NonFinite("force field"))
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Fields {
    pub(crate) slots: Vec<Option<FieldDesc>>,
    pub(crate) generations: Vec<u32>,
}

impl Fields {
    fn index(&self, handle: FieldHandle) -> Option<usize> {
        let index = handle.index as usize;
        (self.generations.get(index) == Some(&handle.generation)
            && self.slots.get(index).is_some_and(Option::is_some))
        .then_some(index)
    }
}

/// Area a body shows the wind: a sphere its disc, a box and a capsule the mean of their
/// projected areas (a face for a cube), anything else the disc of a sphere of its volume.
fn frontal_area(shape: &Shape, volume: f32) -> f32 {
    match *shape {
        Shape::Sphere { radius } => PI * radius * radius,
        Shape::Cuboid { half_extents: h } => 4.0 * (h.x * h.y + h.y * h.z + h.x * h.z) / 3.0,
        Shape::Capsule {
            radius,
            half_height,
        } => 4.0 * radius * half_height + PI * radius * radius,
        _ => {
            let radius = equal_radius(volume);
            PI * radius * radius
        }
    }
}

fn equal_radius(volume: f32) -> f32 {
    (volume * 3.0 / (4.0 * PI)).cbrt()
}

/// Volume under a level the body's center is `height_above_level` above, and its share. A sphere
/// uses the exact cap, a box and a capsule the share of their height, anything else the cap of a
/// sphere of its volume.
fn submerged(shape: &Shape, volume: f32, height_above_level: f32) -> (f32, f32) {
    let cap = |radius: f32| {
        let depth = (radius - height_above_level).clamp(0.0, 2.0 * radius);
        let under = PI * depth * depth * (3.0 * radius - depth) / 3.0;
        let whole = 4.0 / 3.0 * PI * radius * radius * radius;
        (under, if whole > 0.0 { under / whole } else { 0.0 })
    };
    let slab = |half: f32| {
        let share = if half > 0.0 {
            ((half - height_above_level) / (2.0 * half)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (share * volume, share)
    };
    match *shape {
        Shape::Sphere { radius } => cap(radius),
        Shape::Cuboid { half_extents } => slab(half_extents.y),
        Shape::Capsule {
            radius,
            half_height,
        } => slab(half_height + radius),
        _ => cap(equal_radius(volume)),
    }
}

impl PhysicsWorld {
    /// Its force starts with the next step. Bodies in its region wake.
    ///
    /// # Errors
    /// Fails on a non finite or negative setting.
    pub fn add_field(&mut self, desc: FieldDesc) -> Result<FieldHandle, PhysicsError> {
        desc.validate()?;
        let fields = &mut self.fields;
        let index = match fields.slots.iter().position(Option::is_none) {
            Some(index) => index,
            None => {
                fields.slots.push(None);
                fields.generations.push(0);
                fields.slots.len() - 1
            }
        };
        fields.generations[index] = fields.generations[index].wrapping_add(1).max(1);
        fields.slots[index] = Some(desc);
        let handle = FieldHandle {
            index: index as u32,
            generation: fields.generations[index],
        };
        self.wake_in(&desc.region);
        Ok(handle)
    }

    /// For gusts, tides and moving pulls, between steps. Bodies in the old and new region wake.
    ///
    /// # Errors
    /// Fails on a stale handle or a bad setting, then nothing changed.
    pub fn set_field(&mut self, handle: FieldHandle, desc: FieldDesc) -> Result<(), PhysicsError> {
        let index = self
            .fields
            .index(handle)
            .ok_or(PhysicsError::InvalidHandle)?;
        desc.validate()?;
        let old = self.fields.slots[index].replace(desc);
        if let Some(old) = old {
            self.wake_in(&old.region);
        }
        self.wake_in(&desc.region);
        Ok(())
    }

    /// False for a stale handle.
    pub fn remove_field(&mut self, handle: FieldHandle) -> bool {
        let Some(index) = self.fields.index(handle) else {
            return false;
        };
        self.fields.slots[index] = None;
        true
    }

    fn wake_in(&mut self, region: &FieldRegion) {
        for dense in 0..self.bodies.len() {
            if self.bodies.kind[dense] == BodyKind::Dynamic
                && inside(region, self.bodies.position(dense))
            {
                self.bodies.wake(dense);
            }
        }
    }

    /// Adds every field's force and torque to the awake dynamic bodies, before the step.
    pub(crate) fn apply_fields(&mut self) {
        if self.fields.slots.iter().all(Option::is_none) {
            return;
        }
        let up = (-self.settings.gravity).normalize_or(Vec3::Y);
        let gravity = self.settings.gravity.length();
        for dense in 0..self.bodies.len() {
            let bodies = &self.bodies;
            if bodies.kind[dense] != BodyKind::Dynamic || bodies.asleep[dense] {
                continue;
            }
            let inverse_mass = bodies.inverse_mass[dense];
            if inverse_mass <= 0.0 {
                continue;
            }
            let mass = inverse_mass.recip();
            let volume = mass / bodies.material[dense].density;
            let position = bodies.position(dense);
            let (velocity, spin) = (bodies.linear[dense], bodies.angular[dense]);
            let shape = &bodies.shape[dense];
            let (mut force, mut torque) = (Vec3::ZERO, Vec3::ZERO);
            for desc in self.fields.slots.iter().flatten() {
                if bodies.filter[dense].layers & desc.filter.mask == 0
                    || !inside(&desc.region, position)
                {
                    continue;
                }
                match desc.field {
                    ForceField::Wind {
                        velocity: air,
                        drag,
                    } => {
                        force += (air - velocity) * (drag * frontal_area(shape, volume));
                    }
                    ForceField::Drag { linear, angular } => {
                        force -= velocity * linear;
                        torque -= spin * angular;
                    }
                    ForceField::Fluid {
                        level,
                        density,
                        drag,
                    } => {
                        // the level is a height against gravity in the frame's own space
                        let height = position.to_dvec3().as_vec3().dot(up) - level;
                        let (under, share) = submerged(shape, volume, height);
                        force += up * (density * gravity * under) - velocity * (drag * share);
                    }
                    ForceField::Attractor {
                        at,
                        strength,
                        min_distance,
                    } => {
                        let offset = at.offset_from(position).as_vec3();
                        let distance = offset.length().max(min_distance);
                        force += offset.normalize_or(Vec3::ZERO)
                            * (strength * mass / (distance * distance));
                    }
                }
            }
            if !(force.is_finite() && torque.is_finite()) {
                continue;
            }
            let bodies = &mut self.bodies;
            bodies.force[dense] += force;
            bodies.torque[dense] += torque;
        }
    }
}

fn inside(region: &FieldRegion, at: GridPosition) -> bool {
    match *region {
        FieldRegion::Everywhere => true,
        FieldRegion::Sphere { center, radius } => {
            at.offset_from(center).length_squared() <= f64::from(radius) * f64::from(radius)
        }
        FieldRegion::Box {
            center,
            half_extents,
            rotation,
        } => {
            let local = rotation
                .normalize()
                .conjugate()
                .rotate_vec3(at.offset_from(center).as_vec3())
                .abs();
            local.x <= half_extents.x && local.y <= half_extents.y && local.z <= half_extents.z
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBodyDesc;
    use crate::world::PhysicsSettings;

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    fn at(x: f32, y: f32, z: f32) -> GridPosition {
        GridPosition::from_local(Vec3::new(x, y, z)).unwrap()
    }

    #[test]
    fn a_ball_in_still_water_floats_at_the_depth_its_density_says() {
        let mut world = world();
        // a 500 kg/m3 ball in water floats half under, its center on the level
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .density(500.0)
                    .local_position(Vec3::new(0.0, 2.0, 0.0)),
            )
            .unwrap();
        world
            .add_field(FieldDesc::new(ForceField::Fluid {
                level: 0.0,
                density: 1000.0,
                drag: 400.0,
            }))
            .unwrap();
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        let y = world.body_state(ball).unwrap().position.local.y;
        assert!(y.abs() < 0.03, "{y}");
    }

    #[test]
    fn wind_drag_and_a_pull_act_only_where_their_region_and_filter_say() {
        let mut world = PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap();
        let spawn = |world: &mut PhysicsWorld, x: f32| {
            world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::sphere(0.5))
                        // light, so the wind carries it within a second
                        .density(1.0)
                        .local_position(Vec3::new(x, 0.0, 0.0)),
                )
                .unwrap()
        };
        let (inside_ball, outside_ball) = (spawn(&mut world, 0.0), spawn(&mut world, 20.0));
        let wind = FieldDesc::new(ForceField::Wind {
            velocity: Vec3::new(0.0, 0.0, 5.0),
            drag: 1.0,
        })
        .region(FieldRegion::Sphere {
            center: at(0.0, 0.0, 0.0),
            radius: 5.0,
        });
        let gust = world.add_field(wind).unwrap();
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        // carried toward the wind's speed, never past it, inside the region only
        let carried = world.body_state(inside_ball).unwrap().linear_velocity.z;
        assert!(carried > 4.0 && carried <= 5.0, "{carried}");
        assert_eq!(
            world.body_state(outside_ball).unwrap().linear_velocity,
            Vec3::ZERO
        );
        // drag brings it to rest once the wind is gone
        assert!(world.remove_field(gust));
        assert!(!world.remove_field(gust));
        world
            .add_field(FieldDesc::new(ForceField::Drag {
                linear: 50.0,
                angular: 1.0,
            }))
            .unwrap();
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        assert!(
            world
                .body_state(inside_ball)
                .unwrap()
                .linear_velocity
                .length()
                < 0.01
        );
        // a pull on a body its filter leaves out does nothing
        let mut pull = FieldDesc::new(ForceField::Attractor {
            at: at(20.0, 10.0, 0.0),
            strength: 100.0,
            min_distance: 1.0,
        });
        pull.filter.mask = 0;
        let held = world.add_field(pull).unwrap();
        world.step_fixed().unwrap();
        assert_eq!(
            world.body_state(outside_ball).unwrap().linear_velocity,
            Vec3::ZERO
        );
        pull.filter.mask = u32::MAX;
        world.set_field(held, pull).unwrap();
        world.step_fixed().unwrap();
        assert!(world.body_state(outside_ball).unwrap().linear_velocity.y > 0.0);
    }

    #[test]
    fn bad_fields_are_refused() {
        let mut world = world();
        for field in [
            ForceField::Wind {
                velocity: Vec3::new(f32::NAN, 0.0, 0.0),
                drag: 1.0,
            },
            ForceField::Drag {
                linear: -1.0,
                angular: 0.0,
            },
            ForceField::Attractor {
                at: at(0.0, 0.0, 0.0),
                strength: 1.0,
                min_distance: 0.0,
            },
        ] {
            assert!(world.add_field(FieldDesc::new(field)).is_err());
        }
        let ok = world
            .add_field(FieldDesc::new(ForceField::Drag {
                linear: 1.0,
                angular: 1.0,
            }))
            .unwrap();
        let bad = FieldDesc::new(ForceField::Drag {
            linear: 1.0,
            angular: 1.0,
        })
        .region(FieldRegion::Sphere {
            center: at(0.0, 0.0, 0.0),
            radius: -1.0,
        });
        assert!(world.set_field(ok, bad).is_err());
    }
}
