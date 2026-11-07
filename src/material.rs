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
    /// Lever, meters: the turning a contact resists while rolling is this times its load.
    pub rolling_resistance: f32,
    /// Lever, meters: the same for spinning about the contact normal.
    pub spin_resistance: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// What two materials give their contact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Contact {
    pub(crate) friction: f32,
    pub(crate) static_friction: f32,
    pub(crate) restitution: f32,
    pub(crate) bounce_threshold: f32,
    pub(crate) rolling: f32,
    pub(crate) spin: f32,
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
            // they resist like friction does, so they combine by its rule
            rolling: friction.apply(a.rolling_resistance, b.rolling_resistance),
            spin: friction.apply(a.spin_resistance, b.spin_resistance),
        }
    }
}

impl Material {
    /// The default, usable in a `const`: `Material { density: 2400.0, ..Material::DEFAULT }`.
    pub const DEFAULT: Self = Self {
        density: 1000.0,
        friction: 0.5,
        static_friction: 0.0,
        restitution: 0.0,
        bounce_threshold: 1.0,
        friction_combine: Combine::GeometricMean,
        restitution_combine: Combine::Max,
        rolling_resistance: 0.0,
        spin_resistance: 0.0,
    };

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
            && positive(self.rolling_resistance)
            && positive(self.spin_resistance)
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

    #[test]
    fn a_belt_carries_what_lands_on_it_along_its_own_axis() {
        use crate::{PhysicsSettings, PhysicsWorld, Quat, RigidBodyDesc, Shape, Vec3};
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        // turned a quarter about y, the belt's own +x runs along world -z
        let belt = world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(10.0, 0.25, 10.0)))
                    .rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
                    .surface_velocity(Vec3::new(2.0, 0.0, 0.0)),
            )
            .unwrap();
        let crate_ = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.25)))
                    .local_position(Vec3::new(0.0, 0.6, 0.0)),
            )
            .unwrap();
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        let speed = world.body_state(crate_).unwrap().linear_velocity;
        assert!(
            (speed.z + 2.0).abs() < 0.05 && speed.x.abs() < 0.05,
            "{speed:?}"
        );
        // the belt itself never moved
        assert_eq!(world.body_state(belt).unwrap().linear_velocity, Vec3::ZERO);
        // a snapshot keeps it running, stopping it lets friction stop the crate
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        loaded.step_fixed().unwrap();
        let carried = loaded.body_state(crate_).unwrap().linear_velocity;
        assert!((carried.z + 2.0).abs() < 0.05, "{carried:?}");
        loaded.set_surface_velocity(belt, Vec3::ZERO).unwrap();
        for _ in 0..120 {
            loaded.step_fixed().unwrap();
        }
        assert!(loaded.body_state(crate_).unwrap().linear_velocity.length() < 0.01);
        assert!(
            loaded
                .set_surface_velocity(belt, Vec3::new(f32::NAN, 0.0, 0.0))
                .is_err()
        );
    }

    /// A ball rolled at 2 m/s on a floor of `material`: how far it gets in 4 s.
    fn roll(material: Material) -> f32 {
        use crate::{PhysicsSettings, PhysicsWorld, RigidBodyDesc, Shape, Vec3};
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(50.0, 0.5, 5.0)))
                    .local_position(Vec3::new(0.0, -0.5, 0.0))
                    .material(material),
            )
            .unwrap();
        let mut desc = RigidBodyDesc::dynamic(Shape::sphere(0.25))
            .local_position(Vec3::new(-40.0, 0.25, 0.0))
            .material(material);
        desc.linear_velocity = Vec3::new(2.0, 0.0, 0.0);
        desc.angular_velocity = Vec3::new(0.0, 0.0, -8.0);
        let ball = world.spawn(desc).unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        world.body_state(ball).unwrap().position.local.x + 40.0
    }

    /// A ball spun about y at 10 rad/s resting on a floor: its spin after 2 s.
    fn spin(material: Material) -> f32 {
        use crate::{PhysicsSettings, PhysicsWorld, RigidBodyDesc, Shape, Vec3};
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(5.0, 0.5, 5.0)))
                    .local_position(Vec3::new(0.0, -0.5, 0.0))
                    .material(material),
            )
            .unwrap();
        let mut desc = RigidBodyDesc::dynamic(Shape::sphere(0.25))
            .local_position(Vec3::new(0.0, 0.25, 0.0))
            .material(material);
        desc.angular_velocity = Vec3::new(0.0, 10.0, 0.0);
        desc.can_sleep = false;
        let ball = world.spawn(desc).unwrap();
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        world.body_state(ball).unwrap().angular_velocity.y
    }

    #[test]
    fn rolling_and_spin_resistance_slow_a_ball_and_nothing_else_does() {
        let free = Material::default();
        let rolling = Material {
            rolling_resistance: 0.02,
            ..free
        };
        let spinning = Material {
            spin_resistance: 0.02,
            ..free
        };
        let (far, near) = (roll(free), roll(rolling));
        // free it keeps rolling, 2 m/s for 4 s. a lever l on a rolling ball of radius r slows it
        // by l g / (1.4 r), 0.56 m/s^2 here, so it stops after v^2 / 2a = 3.57 m
        assert!(far > 7.5, "{far}");
        assert!((near - 3.57).abs() < 0.4, "{near}");
        let (kept, lost) = (spin(free), spin(spinning));
        assert!(kept > 9.0, "{kept}");
        assert!(lost.abs() < 0.5, "{lost}");
    }
}
