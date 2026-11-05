use std::f32::consts::PI;

use algebrix::Vec3;

use crate::{CompoundId, PhysicsError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    Sphere {
        radius: f32,
    },
    Cuboid {
        half_extents: Vec3,
    },
    Capsule {
        radius: f32,
        half_height: f32,
    },
    /// Parts registered with PhysicsWorld::add_compound.
    Compound(CompoundId),
}

/// Mass and diagonal inertia of a shape in its own axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassProperties {
    pub mass: f32,
    pub inertia: Vec3,
}

impl Shape {
    pub fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    pub fn cuboid(half_extents: Vec3) -> Self {
        Self::Cuboid { half_extents }
    }

    /// Capsule along local Y. `half_height` is the cylinder part only.
    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self::Capsule {
            radius,
            half_height,
        }
    }

    /// # Errors
    /// Fails on zero, negative or non finite sizes.
    pub fn validate(&self) -> Result<(), PhysicsError> {
        let valid = match *self {
            Self::Sphere { radius } => positive(radius),
            Self::Cuboid { half_extents } => {
                positive(half_extents.x) && positive(half_extents.y) && positive(half_extents.z)
            }
            Self::Capsule {
                radius,
                half_height,
            } => positive(radius) && half_height.is_finite() && half_height >= 0.0,
            Self::Compound(_) => true,
        };
        if valid {
            Ok(())
        } else {
            Err(PhysicsError::InvalidShape)
        }
    }

    /// Radius of a sphere around the center that holds the whole shape.
    pub fn bounding_radius(&self) -> f32 {
        match *self {
            Self::Sphere { radius } => radius,
            Self::Cuboid { half_extents } => half_extents.length(),
            Self::Capsule {
                radius,
                half_height,
            } => radius + half_height,
            // only the world knows it, this errs on the safe side
            Self::Compound(_) => f32::INFINITY,
        }
    }

    pub fn volume(&self) -> f32 {
        match *self {
            Self::Sphere { radius } => 4.0 / 3.0 * PI * radius * radius * radius,
            Self::Cuboid { half_extents } => 8.0 * half_extents.x * half_extents.y * half_extents.z,
            Self::Capsule {
                radius,
                half_height,
            } => {
                let r2 = radius * radius;
                PI * r2 * 2.0 * half_height + 4.0 / 3.0 * PI * r2 * radius
            }
            Self::Compound(_) => 0.0,
        }
    }

    pub fn mass_properties(&self, density: f32) -> MassProperties {
        let mass = density * self.volume();
        let inertia = match *self {
            Self::Sphere { radius } => Vec3::splat(0.4 * mass * radius * radius),
            Self::Cuboid { half_extents } => {
                let x2 = half_extents.x * half_extents.x;
                let y2 = half_extents.y * half_extents.y;
                let z2 = half_extents.z * half_extents.z;
                Vec3::new(y2 + z2, x2 + z2, x2 + y2) * (mass / 3.0)
            }
            Self::Capsule {
                radius,
                half_height,
            } => {
                let r2 = radius * radius;
                let height = 2.0 * half_height;
                let cylinder = density * PI * r2 * height;
                let caps = density * 4.0 / 3.0 * PI * r2 * radius;
                let axial = cylinder * r2 / 2.0 + caps * 0.4 * r2;
                let side = cylinder * (height * height / 12.0 + r2 / 4.0)
                    + caps * (0.4 * r2 + height * height / 4.0 + 3.0 * height * radius / 8.0);
                Vec3::new(side, axial, side)
            }
            Self::Compound(_) => Vec3::ZERO,
        };
        MassProperties { mass, inertia }
    }
}

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f32, b: f32) {
        assert!((a - b).abs() <= b.abs() * 1e-5, "{a} != {b}");
    }

    #[test]
    fn sphere_mass_and_inertia() {
        let props = Shape::sphere(2.0).mass_properties(3.0);
        let mass = 3.0 * 4.0 / 3.0 * PI * 8.0;
        near(props.mass, mass);
        near(props.inertia.x, 0.4 * mass * 4.0);
        assert_eq!(props.inertia.x, props.inertia.z);
    }

    #[test]
    fn cuboid_inertia_matches_textbook() {
        let props = Shape::cuboid(Vec3::new(1.0, 2.0, 3.0)).mass_properties(1.0);
        let mass = 2.0 * 4.0 * 6.0;
        near(props.mass, mass);
        near(props.inertia.x, mass * (16.0 + 36.0) / 12.0);
        near(props.inertia.y, mass * (4.0 + 36.0) / 12.0);
        near(props.inertia.z, mass * (4.0 + 16.0) / 12.0);
    }

    #[test]
    fn capsule_with_no_cylinder_is_a_sphere() {
        let capsule = Shape::capsule(1.5, 0.0).mass_properties(2.0);
        let sphere = Shape::sphere(1.5).mass_properties(2.0);
        near(capsule.mass, sphere.mass);
        near(capsule.inertia.x, sphere.inertia.x);
        near(capsule.inertia.y, sphere.inertia.y);
    }

    #[test]
    fn long_capsule_is_easier_to_spin_on_its_axis() {
        let props = Shape::capsule(0.5, 3.0).mass_properties(1.0);
        assert!(props.inertia.y < props.inertia.x);
        assert_eq!(props.inertia.x, props.inertia.z);
    }

    #[test]
    fn bad_sizes_fail() {
        for shape in [
            Shape::sphere(0.0),
            Shape::sphere(-1.0),
            Shape::sphere(f32::NAN),
            Shape::cuboid(Vec3::new(1.0, 0.0, 1.0)),
            Shape::cuboid(Vec3::new(1.0, 1.0, f32::INFINITY)),
            Shape::capsule(0.0, 1.0),
            Shape::capsule(1.0, -1.0),
        ] {
            assert_eq!(shape.validate(), Err(PhysicsError::InvalidShape));
        }
    }
}
