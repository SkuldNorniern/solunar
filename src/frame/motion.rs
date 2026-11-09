use std::f64::consts::{PI, TAU};

use algebrix::{DQuat, DVec3};

use super::FrameDesc;

/// How a frame moves in its parent over the world's time, worked out from the time each step
/// rather than stepped, so it never drifts. None of it set, the frame stays where it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMotion {
    /// Turning about this axis in parent axes, at its length in radians a second, from the
    /// desc's rotation at time 0.
    pub spin: DVec3,
    /// Going round the parent's origin, plus the desc's translation.
    pub orbit: Option<Orbit>,
}

/// A kepler orbit. Its plane is the parent's xz turned by the angles here, it goes round the
/// parent's y the right handed way, from x toward minus z.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orbit {
    /// The parent's mass times the gravitational constant, m^3/s^2.
    pub mu: f64,
    pub semi_major_axis: f64,
    /// 0 for a circle, under 1.
    pub eccentricity: f64,
    /// Tilt of the plane, radians, about the line to the ascending node.
    pub inclination: f64,
    /// Where the plane crosses going up, radians about the parent's y from its x.
    pub ascending_node: f64,
    /// From the ascending node to the closest point, radians in the plane.
    pub periapsis: f64,
    /// Share of a turn gone at time 0 as mean anomaly, radians.
    pub mean_anomaly: f64,
}

impl Orbit {
    /// A circle of `radius` in the parent's xz plane, starting on its x axis.
    pub fn circular(mu: f64, radius: f64) -> Self {
        Self {
            mu,
            semi_major_axis: radius,
            eccentricity: 0.0,
            inclination: 0.0,
            ascending_node: 0.0,
            periapsis: 0.0,
            mean_anomaly: 0.0,
        }
    }

    /// Seconds for one turn.
    pub fn period(&self) -> f64 {
        TAU * (self.semi_major_axis.powi(3) / self.mu).sqrt()
    }

    pub(crate) fn is_valid(&self) -> bool {
        let angles = [
            self.inclination,
            self.ascending_node,
            self.periapsis,
            self.mean_anomaly,
        ];
        self.mu.is_finite()
            && self.mu > 0.0
            && self.semi_major_axis.is_finite()
            && self.semi_major_axis > 0.0
            && (0.0..1.0).contains(&self.eccentricity)
            && angles.iter().all(|angle| angle.is_finite())
    }

    /// Position and velocity at `time` from the parent's origin, in parent axes.
    pub fn state_at(&self, time: f64) -> (DVec3, DVec3) {
        let (a, e) = (self.semi_major_axis, self.eccentricity);
        let mean_motion = (self.mu / (a * a * a)).sqrt();
        let mean = (self.mean_anomaly + mean_motion * time).rem_euclid(TAU);
        let mean = if mean > PI { mean - TAU } else { mean };
        // newton on kepler's equation, from the mean anomaly or from pi when very eccentric
        let mut eccentric = if e < 0.8 { mean } else { PI.copysign(mean) };
        for _ in 0..50 {
            let step = (eccentric - e * eccentric.sin() - mean) / (1.0 - e * eccentric.cos());
            eccentric -= step;
            if step.abs() < 1e-15 {
                break;
            }
        }
        let (sin, cos) = eccentric.sin_cos();
        let squeeze = (1.0 - e * e).sqrt();
        let distance = a * (1.0 - e * cos);
        let speed = (self.mu * a).sqrt() / distance;
        let position = DVec3::new(a * (cos - e), 0.0, -a * squeeze * sin);
        let velocity = DVec3::new(-speed * sin, 0.0, -speed * squeeze * cos);
        let turn = DQuat::from_axis_angle(DVec3::Y, self.ascending_node)
            * DQuat::from_axis_angle(DVec3::X, self.inclination)
            * DQuat::from_axis_angle(DVec3::Y, self.periapsis);
        (turn.mul_vec3(position), turn.mul_vec3(velocity))
    }
}

impl Default for FrameMotion {
    fn default() -> Self {
        Self {
            spin: DVec3::ZERO,
            orbit: None,
        }
    }
}

impl FrameMotion {
    pub(crate) fn is_valid(&self) -> bool {
        let spin = self.spin;
        spin.x.is_finite()
            && spin.y.is_finite()
            && spin.z.is_finite()
            && self.orbit.is_none_or(|orbit| orbit.is_valid())
    }

    pub(crate) fn is_still(&self) -> bool {
        self.spin == DVec3::ZERO && self.orbit.is_none()
    }

    /// `base` at `time`.
    pub(crate) fn moved(&self, base: &FrameDesc, time: f64) -> FrameDesc {
        let mut desc = *base;
        let rate = self.spin.length();
        if rate > 0.0 {
            let turn = DQuat::from_axis_angle(self.spin / rate, rate * time);
            desc.rotation = (turn * base.rotation).normalize();
            desc.angular_velocity = self.spin;
        }
        if let Some(orbit) = self.orbit {
            let (position, velocity) = orbit.state_at(time);
            desc.translation = base.translation + position;
            desc.linear_velocity = base.linear_velocity + velocity;
        }
        desc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn earth_moon() -> Orbit {
        Orbit {
            mu: 3.986e14,
            semi_major_axis: 3.844e8,
            eccentricity: 0.0549,
            inclination: 0.09,
            ascending_node: 1.2,
            periapsis: 0.4,
            mean_anomaly: 2.0,
        }
    }

    #[test]
    fn an_orbit_closes_after_one_period() {
        for orbit in [Orbit::circular(3.986e14, 7.0e6), earth_moon()] {
            let (start, speed) = orbit.state_at(0.0);
            let (end, end_speed) = orbit.state_at(orbit.period());
            let size = orbit.semi_major_axis;
            assert!((end - start).length() < 1e-9 * size, "{start:?} {end:?}");
            assert!((end_speed - speed).length() < 1e-9 * speed.length());
        }
    }

    #[test]
    fn energy_and_angular_momentum_are_the_same_all_round() {
        let orbit = earth_moon();
        let energy = |time: f64| {
            let (position, velocity) = orbit.state_at(time);
            0.5 * velocity.length_squared() - orbit.mu / position.length()
        };
        let momentum = |time: f64| {
            let (position, velocity) = orbit.state_at(time);
            position.cross(velocity)
        };
        let (first, turn) = (energy(0.0), momentum(0.0));
        assert!((first + orbit.mu / (2.0 * orbit.semi_major_axis)).abs() < 1e-9 * first.abs());
        for step in 1..64 {
            let time = orbit.period() * f64::from(step) / 64.0;
            assert!((energy(time) - first).abs() < 1e-9 * first.abs());
            assert!((momentum(time) - turn).length() < 1e-9 * turn.length());
        }
    }

    #[test]
    fn velocity_is_how_position_changes() {
        let orbit = earth_moon();
        for time in [0.0, 1.0e5, 7.3e5, 2.0e6] {
            let h = 1.0;
            let (before, _) = orbit.state_at(time - h);
            let (after, _) = orbit.state_at(time + h);
            let (_, velocity) = orbit.state_at(time);
            let slope = (after - before) / (2.0 * h);
            assert!((slope - velocity).length() < 1e-6 * velocity.length());
        }
    }

    #[test]
    fn a_flat_circle_goes_round_y_from_x_toward_minus_z() {
        let orbit = Orbit::circular(1.0, 1.0);
        let (start, velocity) = orbit.state_at(0.0);
        assert!((start - DVec3::X).length() < 1e-12);
        assert!((velocity - DVec3::new(0.0, 0.0, -1.0)).length() < 1e-12);
        assert!(start.cross(velocity).y > 0.0);
    }

    #[test]
    fn bad_orbits_are_refused() {
        let good = Orbit::circular(1.0, 1.0);
        for bad in [
            Orbit { mu: 0.0, ..good },
            Orbit {
                semi_major_axis: -1.0,
                ..good
            },
            Orbit {
                eccentricity: 1.0,
                ..good
            },
            Orbit {
                inclination: f64::NAN,
                ..good
            },
        ] {
            assert!(!bad.is_valid());
        }
        assert!(good.is_valid());
    }
}
