use algebrix::{Quat, Vec3};

use crate::PhysicsError;

/// Signed distance where no base and no edit has put anything. Far, but finite so crossings
/// still interpolate.
pub(crate) const FAR: f32 = 1.0e6;

/// What the volume holds before any edit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Base {
    Empty,
    Plane {
        height: f32,
    },
    /// Rolling ground from value noise, `octaves` layers each half the size and height.
    Hills {
        seed: u32,
        height: f32,
        amplitude: f32,
        wavelength: f32,
        octaves: u32,
    },
}

impl Base {
    pub(crate) fn validate(&self) -> Result<(), PhysicsError> {
        let valid = match *self {
            Self::Empty => true,
            Self::Plane { height } => height.is_finite(),
            Self::Hills {
                height,
                amplitude,
                wavelength,
                octaves,
                ..
            } => {
                height.is_finite()
                    && amplitude.is_finite()
                    && amplitude >= 0.0
                    && wavelength.is_finite()
                    && wavelength > 0.0
                    && (1..=8).contains(&octaves)
            }
        };
        if valid {
            Ok(())
        } else {
            Err(PhysicsError::InvalidShape)
        }
    }

    /// Lowest and highest ground, `None` with no ground.
    pub(crate) fn height_range(&self) -> Option<(f32, f32)> {
        match *self {
            Self::Empty => None,
            Self::Plane { height } => Some((height, height)),
            Self::Hills {
                height,
                amplitude,
                octaves,
                ..
            } => {
                let reach = amplitude * (2.0 - 0.5_f32.powi(octaves as i32 - 1));
                Some((height - reach, height + reach))
            }
        }
    }

    /// Ground height and the factor that turns height above it into distance, for one column.
    pub(crate) fn column(&self, x: f32, z: f32, step: f32) -> (f32, f32) {
        match *self {
            Self::Empty => (f32::NEG_INFINITY, 1.0),
            Self::Plane { height } => (height, 1.0),
            Self::Hills { .. } => {
                let h = self.hills(x, z);
                let dx = (self.hills(x + step, z) - self.hills(x - step, z)) / (2.0 * step);
                let dz = (self.hills(x, z + step) - self.hills(x, z - step)) / (2.0 * step);
                (h, 1.0 / (1.0 + dx * dx + dz * dz).sqrt())
            }
        }
    }

    fn hills(&self, x: f32, z: f32) -> f32 {
        let Self::Hills {
            seed,
            height,
            amplitude,
            wavelength,
            octaves,
        } = *self
        else {
            return 0.0;
        };
        let mut sum = 0.0;
        let mut size = amplitude;
        let mut frequency = 1.0 / wavelength;
        for octave in 0..octaves {
            sum += size * value_noise(x * frequency, z * frequency, seed.wrapping_add(octave));
            size *= 0.5;
            frequency *= 2.0;
        }
        height + sum
    }
}

pub(crate) fn base_distance(height: f32, slope: f32, y: f32) -> f32 {
    if height == f32::NEG_INFINITY {
        FAR
    } else {
        ((y - height) * slope).clamp(-FAR, FAR)
    }
}

fn hash(x: i32, z: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (z as u32).wrapping_mul(0xd816_3841)
        ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    (h >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// In -1 to 1, smooth, the same on every machine.
fn value_noise(x: f32, z: f32, seed: u32) -> f32 {
    let (fx, fz) = (x.floor(), z.floor());
    let (ix, iz) = (fx as i32, fz as i32);
    let (u, v) = (fade(x - fx), fade(z - fz));
    let a = hash(ix, iz, seed);
    let b = hash(ix + 1, iz, seed);
    let c = hash(ix, iz + 1, seed);
    let d = hash(ix + 1, iz + 1, seed);
    let top = a + (b - a) * u;
    let bottom = c + (d - c) * u;
    top + (bottom - top) * v
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditShape {
    Sphere {
        center: Vec3,
        radius: f32,
    },
    Box {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
    },
    /// Round ended segment from `a` to `b`, a tunnel or a drill hole.
    Capsule {
        a: Vec3,
        b: Vec3,
        radius: f32,
    },
}

/// The order edits come in is the order they apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditOp {
    /// Puts material in, for good.
    Add,
    /// Takes material out for good and counts the mass removed.
    Excavate,
    /// A hole by rule, no material is taken. Removing the edit closes it again.
    Tunnel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edit {
    pub shape: EditShape,
    pub op: EditOp,
}

impl EditShape {
    pub(crate) fn validate(&self) -> Result<(), PhysicsError> {
        let valid = match *self {
            Self::Sphere { center, radius } => {
                center.is_finite() && radius.is_finite() && radius > 0.0
            }
            Self::Box {
                center,
                half_extents,
                rotation,
            } => {
                center.is_finite()
                    && half_extents.is_finite()
                    && half_extents.min_element() > 0.0
                    && rotation.is_finite()
                    && rotation.length_squared() > 1e-12
            }
            Self::Capsule { a, b, radius } => {
                a.is_finite() && b.is_finite() && radius.is_finite() && radius > 0.0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(PhysicsError::InvalidShape)
        }
    }

    pub(crate) fn distance(&self, p: Vec3) -> f32 {
        match *self {
            Self::Sphere { center, radius } => (p - center).length() - radius,
            Self::Box {
                center,
                half_extents,
                rotation,
            } => {
                let q = rotation
                    .normalize()
                    .conjugate()
                    .rotate_vec3(p - center)
                    .abs()
                    - half_extents;
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
            }
            Self::Capsule { a, b, radius } => {
                let along = b - a;
                let squared = along.length_squared();
                let t = if squared > 0.0 {
                    ((p - a).dot(along) / squared).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (p - (a + along * t)).length() - radius
            }
        }
    }

    /// Box around the shape.
    pub(crate) fn bounds(&self) -> (Vec3, Vec3) {
        match *self {
            Self::Sphere { center, radius } => {
                (center - Vec3::splat(radius), center + Vec3::splat(radius))
            }
            Self::Box {
                center,
                half_extents,
                rotation,
            } => {
                let rotation = rotation.normalize();
                let extent = rotation.rotate_vec3(Vec3::X * half_extents.x).abs()
                    + rotation.rotate_vec3(Vec3::Y * half_extents.y).abs()
                    + rotation.rotate_vec3(Vec3::Z * half_extents.z).abs();
                (center - extent, center + extent)
            }
            Self::Capsule { a, b, radius } => (
                a.min(b) - Vec3::splat(radius),
                a.max(b) + Vec3::splat(radius),
            ),
        }
    }
}

impl EditOp {
    pub(crate) fn apply(self, distance: f32, shape: f32) -> f32 {
        match self {
            Self::Add => distance.min(shape),
            Self::Excavate | Self::Tunnel => distance.max(-shape),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_stays_in_range_and_repeats() {
        for i in 0..2000 {
            let (x, z) = (i as f32 * 0.173 - 50.0, i as f32 * 0.311 - 80.0);
            let value = value_noise(x, z, 7);
            assert!((-1.0..=1.0).contains(&value), "{value}");
            assert_eq!(value, value_noise(x, z, 7));
        }
        assert_ne!(value_noise(0.5, 0.5, 1), value_noise(0.5, 0.5, 2));
    }

    #[test]
    fn hills_stay_inside_their_range() {
        let base = Base::Hills {
            seed: 3,
            height: 2.0,
            amplitude: 5.0,
            wavelength: 20.0,
            octaves: 4,
        };
        let (low, high) = base.height_range().unwrap_or((0.0, 0.0));
        for i in 0..5000 {
            let (h, slope) = base.column(i as f32 * 0.7, i as f32 * -0.37, 0.25);
            assert!((low..=high).contains(&h), "{h} outside {low} {high}");
            assert!(slope > 0.0 && slope <= 1.0);
        }
    }

    #[test]
    fn shape_distances_are_right_at_known_points() {
        let sphere = EditShape::Sphere {
            center: Vec3::ONE,
            radius: 2.0,
        };
        assert!((sphere.distance(Vec3::ONE) + 2.0).abs() < 1e-6);
        assert!((sphere.distance(Vec3::new(4.0, 1.0, 1.0)) - 1.0).abs() < 1e-6);

        let turned = EditShape::Box {
            center: Vec3::ZERO,
            half_extents: Vec3::new(2.0, 1.0, 1.0),
            rotation: Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2),
        };
        assert!((turned.distance(Vec3::new(0.0, 0.0, 3.0)) - 1.0).abs() < 1e-5);
        assert!((turned.distance(Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-5);
        assert!((turned.distance(Vec3::ZERO) + 1.0).abs() < 1e-5);

        let capsule = EditShape::Capsule {
            a: Vec3::ZERO,
            b: Vec3::X * 4.0,
            radius: 1.0,
        };
        assert!((capsule.distance(Vec3::new(2.0, 3.0, 0.0)) - 2.0).abs() < 1e-6);
        assert!((capsule.distance(Vec3::new(-2.0, 0.0, 0.0)) - 1.0).abs() < 1e-6);

        for shape in [sphere, turned, capsule] {
            let (low, high) = shape.bounds();
            for i in 0..200 {
                let p = Vec3::new(
                    (i as f32 * 0.37).sin() * 6.0,
                    (i as f32 * 0.71).cos() * 6.0,
                    (i as f32 * 1.13).sin() * 6.0,
                );
                if shape.distance(p) < 0.0 {
                    assert!(p.max(low) == p && p.min(high) == p, "{p:?}");
                }
            }
        }
    }

    #[test]
    fn bad_shapes_fail() {
        let bad = [
            EditShape::Sphere {
                center: Vec3::ZERO,
                radius: 0.0,
            },
            EditShape::Sphere {
                center: Vec3::splat(f32::NAN),
                radius: 1.0,
            },
            EditShape::Box {
                center: Vec3::ZERO,
                half_extents: Vec3::new(1.0, 0.0, 1.0),
                rotation: Quat::IDENTITY,
            },
            EditShape::Box {
                center: Vec3::ZERO,
                half_extents: Vec3::ONE,
                rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            },
            EditShape::Capsule {
                a: Vec3::ZERO,
                b: Vec3::X,
                radius: -1.0,
            },
        ];
        for shape in bad {
            assert_eq!(
                shape.validate(),
                Err(PhysicsError::InvalidShape),
                "{shape:?}"
            );
        }
        let bad_bases = [
            Base::Plane { height: f32::NAN },
            Base::Hills {
                seed: 0,
                height: 0.0,
                amplitude: 1.0,
                wavelength: 0.0,
                octaves: 2,
            },
            Base::Hills {
                seed: 0,
                height: 0.0,
                amplitude: 1.0,
                wavelength: 4.0,
                octaves: 9,
            },
        ];
        for base in bad_bases {
            assert!(base.validate().is_err(), "{base:?}");
        }
    }
}
