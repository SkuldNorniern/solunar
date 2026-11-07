use crate::PhysicsError;

/// How two bodies' values make one for their contact. When the two ask for different rules, the
/// one later in this list wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Combine {
    GeometricMean,
    Average,
    Min,
    Multiply,
    Max,
}

impl Combine {
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Self::GeometricMean => (a * b).sqrt(),
            Self::Average => 0.5 * (a + b),
            Self::Min => a.min(b),
            Self::Multiply => a * b,
            Self::Max => a.max(b),
        }
    }

    pub(crate) fn tag(self) -> u8 {
        self as u8
    }

    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        Some(match tag {
            0 => Self::GeometricMean,
            1 => Self::Average,
            2 => Self::Min,
            3 => Self::Multiply,
            4 => Self::Max,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub density: f32,
    /// Friction while sliding.
    pub friction: f32,
    /// Friction while held still, the most a contact takes before it starts to slide. Only counts
    /// above `friction`, so zero means it holds and slides alike.
    pub static_friction: f32,
    pub restitution: f32,
    /// Approach speed under which nothing bounces, m/s. Keeps resting contacts from jittering.
    pub bounce_threshold: f32,
    pub friction_combine: Combine,
    pub restitution_combine: Combine,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            density: 1000.0,
            friction: 0.5,
            static_friction: 0.0,
            restitution: 0.0,
            bounce_threshold: 1.0,
            friction_combine: Combine::GeometricMean,
            restitution_combine: Combine::Max,
        }
    }
}

/// What two materials give their contact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Contact {
    pub(crate) friction: f32,
    pub(crate) static_friction: f32,
    pub(crate) restitution: f32,
    pub(crate) bounce_threshold: f32,
}

impl Contact {
    pub(crate) fn of(a: &Material, b: &Material) -> Self {
        let friction = a.friction_combine.max(b.friction_combine);
        let restitution = a.restitution_combine.max(b.restitution_combine);
        Self {
            friction: friction.apply(a.friction, b.friction),
            static_friction: friction.apply(
                a.static_friction.max(a.friction),
                b.static_friction.max(b.friction),
            ),
            restitution: restitution.apply(a.restitution, b.restitution),
            bounce_threshold: a.bounce_threshold.max(b.bounce_threshold),
        }
    }
}

impl Material {
    /// # Errors
    /// Fails on non positive density, negative friction, restitution outside 0 to 1, or a
    /// negative bounce threshold.
    pub fn validate(&self) -> Result<(), PhysicsError> {
        let positive = |value: f32| value.is_finite() && value >= 0.0;
        let valid = self.density.is_finite()
            && self.density > 0.0
            && positive(self.friction)
            && positive(self.static_friction)
            && positive(self.bounce_threshold)
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
            Material {
                static_friction: -1.0,
                ..base
            },
            Material {
                bounce_threshold: f32::INFINITY,
                ..base
            },
        ] {
            assert_eq!(material.validate(), Err(PhysicsError::InvalidMaterial));
        }
    }

    #[test]
    fn combine_rules_and_the_later_rule_wins() {
        assert_eq!(Combine::GeometricMean.apply(0.25, 1.0), 0.5);
        assert_eq!(Combine::Average.apply(0.25, 1.0), 0.625);
        assert_eq!(Combine::Min.apply(0.25, 1.0), 0.25);
        assert_eq!(Combine::Multiply.apply(0.25, 1.0), 0.25);
        assert_eq!(Combine::Max.apply(0.25, 1.0), 1.0);
        for rule in [
            Combine::GeometricMean,
            Combine::Average,
            Combine::Min,
            Combine::Multiply,
            Combine::Max,
        ] {
            assert_eq!(Combine::from_tag(rule.tag()), Some(rule));
        }
        assert_eq!(Combine::from_tag(9), None);
        // ice asks for the least, rubber for the most: max is later, rubber wins
        let ice = Material {
            friction: 0.05,
            friction_combine: Combine::Min,
            ..Material::default()
        };
        let rubber = Material {
            friction: 0.9,
            friction_combine: Combine::Max,
            ..Material::default()
        };
        assert_eq!(Contact::of(&ice, &rubber).friction, 0.9);
        assert_eq!(Contact::of(&ice, &Material::default()).friction, 0.05);
        // defaults keep what was there before: geometric mean friction, max restitution
        let (a, b) = (
            Material::default(),
            Material {
                restitution: 0.6,
                ..Material::default()
            },
        );
        let contact = Contact::of(&a, &b);
        assert_eq!(
            (
                contact.friction,
                contact.static_friction,
                contact.restitution
            ),
            (0.5, 0.5, 0.6)
        );
    }

    /// How far a box slides down a 1 in 2 slope, tan 0.5, in two seconds.
    fn slide(material: Material) -> f32 {
        use crate::{
            GridPosition, PhysicsSettings, PhysicsWorld, Quat, RigidBodyDesc, Shape, Vec3,
        };
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let tilt = Quat::from_rotation_z(0.5_f32.atan());
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(10.0, 0.5, 4.0)))
                    .rotation(tilt)
                    .material(material),
            )
            .unwrap();
        let up = tilt.rotate_vec3(Vec3::Y);
        let start = up * 0.75;
        let block = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.25)))
                    .rotation(tilt)
                    .local_position(start)
                    .material(material),
            )
            .unwrap();
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        let at = world.body_state(block).unwrap().position;
        let origin = GridPosition::from_local(start).unwrap();
        at.offset_from(origin).length() as f32
    }

    #[test]
    fn static_friction_holds_on_a_slope_its_sliding_friction_can_not() {
        let sliding = Material {
            friction: 0.3,
            ..Material::default()
        };
        let held = Material {
            static_friction: 0.7,
            ..sliding
        };
        let (moved, stayed) = (slide(sliding), slide(held));
        assert!(moved > 1.0, "{moved}");
        assert!(stayed < 0.02, "{stayed}");
    }

    /// Highest a ball dropped from 0.3 m gets after it first lands.
    fn rebound(material: Material) -> f32 {
        use crate::{PhysicsSettings, PhysicsWorld, RigidBodyDesc, Shape, Vec3};
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(5.0, 0.5, 5.0)))
                    .local_position(Vec3::new(0.0, -0.5, 0.0))
                    .material(material),
            )
            .unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.1))
                    .local_position(Vec3::new(0.0, 0.4, 0.0))
                    .material(material),
            )
            .unwrap();
        let (mut landed, mut highest) = (false, 0.0_f32);
        for _ in 0..90 {
            world.step_fixed().unwrap();
            let state = world.body_state(ball).unwrap();
            let height = state.position.local.y;
            landed |= height < 0.12;
            if landed {
                highest = highest.max(height);
            }
        }
        highest
    }

    #[test]
    fn a_slow_landing_bounces_only_under_its_threshold() {
        // lands at about 2.4 m/s
        let bouncy = Material {
            restitution: 0.8,
            ..Material::default()
        };
        let calm = Material {
            bounce_threshold: 3.0,
            ..bouncy
        };
        let (up, still) = (rebound(bouncy), rebound(calm));
        assert!(up > 0.25, "{up}");
        assert!(still < 0.11, "{still}");
    }
}
