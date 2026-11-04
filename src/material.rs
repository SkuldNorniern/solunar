use crate::PhysicsError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            density: 1000.0,
            friction: 0.5,
            restitution: 0.0,
        }
    }
}

impl Material {
    /// # Errors
    /// Fails on non positive density, negative friction or restitution outside 0 to 1.
    pub fn validate(&self) -> Result<(), PhysicsError> {
        let valid = self.density.is_finite()
            && self.density > 0.0
            && self.friction.is_finite()
            && self.friction >= 0.0
            && (0.0..=1.0).contains(&self.restitution);
        if valid {
            Ok(())
        } else {
            Err(PhysicsError::InvalidMaterial)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_valid() {
        assert_eq!(Material::default().validate(), Ok(()));
    }

    #[test]
    fn bad_values_fail() {
        let base = Material::default();
        for material in [
            Material {
                density: 0.0,
                ..base
            },
            Material {
                density: f32::NAN,
                ..base
            },
            Material {
                friction: -0.1,
                ..base
            },
            Material {
                restitution: 1.5,
                ..base
            },
            Material {
                restitution: f32::NAN,
                ..base
            },
        ] {
            assert_eq!(material.validate(), Err(PhysicsError::InvalidMaterial));
        }
    }
}
