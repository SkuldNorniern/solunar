use std::f64::consts::{PI, TAU};

use algebrix::{DQuat, DVec3};

use crate::PhysicsError;

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

    /// The elements of the closed orbit through `position` at `velocity` round a body of `mu`,
    /// its mean anomaly at time 0 where it is now. None for an open one, see [`Coast`].
    pub fn from_state(mu: f64, position: DVec3, velocity: DVec3) -> Option<Self> {
        let coast = Coast::new(mu, position, velocity)?;
        if coast.energy() >= 0.0 {
            return None;
        }
        // the textbook's axes, z up: ours with y up and z the other way round
        let r = DVec3::new(position.x, -position.z, position.y);
        let v = DVec3::new(velocity.x, -velocity.z, velocity.y);
        let distance = r.length();
        let h = r.cross(v);
        let node = DVec3::Z.cross(h);
        let e_vector = (r * (v.length_squared() - mu / distance) - v * r.dot(v)) / mu;
        let e = e_vector.length();
        let semi_major_axis = -mu / (2.0 * coast.energy());
        let inclination = (h.z / h.length()).clamp(-1.0, 1.0).acos();
        // in the reference plane the node is taken on x, on a circle the periapsis on the node:
        // angles that are not defined are zero
        let flat = node.length() < 1e-12 * h.length();
        let round = e < 1e-12;
        let node_line = if flat { DVec3::X } else { node.normalize() };
        let ascending_node = if flat {
            0.0
        } else {
            node_line.y.atan2(node_line.x).rem_euclid(TAU)
        };
        // a quarter turn on from the node line, with the motion
        let ahead = h.normalize().cross(node_line);
        let periapsis = if round {
            0.0
        } else {
            e_vector
                .dot(ahead)
                .atan2(e_vector.dot(node_line))
                .rem_euclid(TAU)
        };
        let reference = if round { node_line } else { e_vector / e };
        let side = h.normalize().cross(reference);
        let true_anomaly = r.dot(side).atan2(r.dot(reference));
        let half = true_anomaly / 2.0;
        let eccentric = 2.0 * ((1.0 - e).sqrt() * half.sin()).atan2((1.0 + e).sqrt() * half.cos());
        let mean_anomaly = (eccentric - e * eccentric.sin()).rem_euclid(TAU);
        Some(Self {
            mu,
            semi_major_axis,
            eccentricity: e,
            inclination,
            ascending_node,
            periapsis,
            mean_anomaly,
        })
    }

    /// The same path as a [`Coast`] from where it is at time 0.
    pub fn coast(&self) -> Coast {
        let (position, velocity) = self.state_at(0.0);
        Coast {
            mu: self.mu,
            position,
            velocity,
        }
    }
}

/// A body coasting round one of `mu` from `position` at `velocity` at time 0: an ellipse, a
/// parabola or a hyperbola, by universal variables, so none is a special case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coast {
    pub mu: f64,
    pub position: DVec3,
    pub velocity: DVec3,
}

impl Coast {
    /// None for a mass, position or velocity that is not finite, or a start at the center.
    pub fn new(mu: f64, position: DVec3, velocity: DVec3) -> Option<Self> {
        let finite = |v: DVec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        let valid = mu.is_finite()
            && mu > 0.0
            && finite(position)
            && finite(velocity)
            && position.length() > 0.0;
        valid.then_some(Self {
            mu,
            position,
            velocity,
        })
    }

    /// Energy per kilogram, J/kg: under zero it comes back, zero or more it leaves.
    pub fn energy(&self) -> f64 {
        0.5 * self.velocity.length_squared() - self.mu / self.position.length()
    }

    /// Seconds a turn, none for an open path.
    pub fn period(&self) -> Option<f64> {
        let energy = self.energy();
        (energy < 0.0).then(|| {
            let a = -self.mu / (2.0 * energy);
            TAU * (a * a * a / self.mu).sqrt()
        })
    }

    /// Closest and furthest from the center, the furthest none for an open path.
    pub fn apses(&self) -> (f64, Option<f64>) {
        let (r, v) = (self.position, self.velocity);
        let h = r.cross(v).length();
        let p = h * h / self.mu;
        let distance = r.length();
        let e = ((r * (v.length_squared() - self.mu / distance) - v * r.dot(v)) / self.mu).length();
        (p / (1.0 + e), (e < 1.0).then(|| p / (1.0 - e)))
    }

    /// Position and velocity `time` seconds on, back for less than zero.
    pub fn state_at(&self, time: f64) -> (DVec3, DVec3) {
        let (r0, v0) = (self.position, self.velocity);
        let root = self.mu.sqrt();
        let distance = r0.length();
        let radial = r0.dot(v0) / distance;
        let alpha = 2.0 / distance - v0.length_squared() / self.mu;
        // whole turns change nothing, and newton starts better near the start
        let time = match self.period() {
            Some(period) => time - (time / period).round() * period,
            None => time,
        };
        let mut chi = first_guess(self, (distance, alpha), time);
        for _ in 0..200 {
            let z = alpha * chi * chi;
            let (c, s) = stumpff(z);
            let value = distance * radial / root * chi * chi * c
                + (1.0 - alpha * distance) * chi * chi * chi * s
                + distance * chi
                - root * time;
            let slope = distance * radial / root * chi * (1.0 - z * s)
                + (1.0 - alpha * distance) * chi * chi * c
                + distance;
            let step = value / slope;
            chi -= step;
            if step.abs() <= 1e-14 * chi.abs().max(1.0) {
                break;
            }
        }
        let z = alpha * chi * chi;
        let (c, s) = stumpff(z);
        let f = 1.0 - chi * chi / distance * c;
        let g = time - chi * chi * chi * s / root;
        let position = r0 * f + v0 * g;
        let far = position.length();
        let f_rate = root / (far * distance) * (alpha * chi * chi * chi * s - chi);
        let g_rate = 1.0 - chi * chi / far * c;
        (position, r0 * f_rate + v0 * g_rate)
    }

    /// An instant burn of `change` m/s `time` seconds on: the coast after it, its time 0 at the
    /// burn.
    pub fn burn(&self, time: f64, change: DVec3) -> Self {
        let (position, velocity) = self.state_at(time);
        Self {
            position,
            velocity: velocity + change,
            ..*self
        }
    }
}

/// Where newton starts on the universal anomaly, after vallado: from the mean motion on an
/// ellipse, from barker's equation near a parabola, from the log of the distance on a hyperbola.
/// The ellipse's guess on a hyperbola went out past what cosh holds.
fn first_guess(coast: &Coast, (distance, alpha): (f64, f64), time: f64) -> f64 {
    let (mu, r0, v0) = (coast.mu, coast.position, coast.velocity);
    let root = mu.sqrt();
    if alpha > 1e-12 / distance {
        return root * alpha * time;
    }
    if alpha.abs() <= 1e-12 / distance {
        let h = r0.cross(v0).length();
        let p = h * h / mu;
        let s = 0.5 * (1.0 / (3.0 * (mu / (p * p * p)).sqrt() * time)).atan();
        let w = s.tan().cbrt().atan();
        return p.sqrt() * 2.0 / (2.0 * w).tan();
    }
    let a = 1.0 / alpha;
    let sign = time.signum();
    let inside = -2.0 * mu * alpha * time
        / (r0.dot(v0) + sign * (-mu * a).sqrt() * (1.0 - distance * alpha));
    if inside > 0.0 && inside.is_finite() {
        sign * (-a).sqrt() * inside.ln()
    } else {
        root * time / distance
    }
}

/// Stumpff's c and s, by their series near zero where the closed forms lose digits.
fn stumpff(z: f64) -> (f64, f64) {
    if z.abs() < 1e-3 {
        let c = 0.5 - z / 24.0 + z * z / 720.0 - z * z * z / 40_320.0;
        let s = 1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z * z * z / 362_880.0;
        (c, s)
    } else if z > 0.0 {
        let root = z.sqrt();
        ((1.0 - root.cos()) / z, (root - root.sin()) / (root * z))
    } else {
        let root = (-z).sqrt();
        ((root.cosh() - 1.0) / -z, (root.sinh() - root) / (root * -z))
    }
}

/// Dormand and Prince's 5(4) pair.
const DP_A: [[f64; 6]; 6] = [
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];
const DP_C: [f64; 6] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
const DP_FIFTH: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];
const DP_FOURTH: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];

/// Most steps [`propagate`] takes before it gives up.
const MAX_PROPAGATION_STEPS: usize = 1_000_000;

/// Position and velocity at `to` from `start` at `from`, under `acceleration` of time, position
/// and velocity, by dormand and prince with each step's error under `tolerance` meters. For
/// thrust and for more than one pull at once, where no coast holds.
///
/// # Errors
/// Fails on a bad tolerance or time, a path going non finite, or one needing more than a
/// million steps.
pub fn propagate(
    acceleration: impl Fn(f64, DVec3, DVec3) -> DVec3,
    start: (DVec3, DVec3),
    (from, to): (f64, f64),
    tolerance: f64,
) -> Result<(DVec3, DVec3), PhysicsError> {
    if !tolerance.is_finite() || tolerance <= 0.0 || !from.is_finite() || !to.is_finite() {
        return Err(PhysicsError::InvalidSettings);
    }
    let finite = |v: DVec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let (mut position, mut velocity) = start;
    let mut time = from;
    let direction = (to - from).signum();
    let mut step = (to - from).abs().clamp(1e-9, 1.0);
    for _ in 0..MAX_PROPAGATION_STEPS {
        let left = (to - time).abs();
        if left <= 1e-12 * to.abs().max(1.0) {
            return Ok((position, velocity));
        }
        let h = step.min(left) * direction;
        let mut moves = [DVec3::ZERO; 7];
        let mut pulls = [DVec3::ZERO; 7];
        moves[0] = velocity;
        pulls[0] = acceleration(time, position, velocity);
        for stage in 0..6 {
            let (mut x, mut v) = (position, velocity);
            for (k, weight) in DP_A[stage].iter().enumerate().take(stage + 1) {
                x += moves[k] * (h * weight);
                v += pulls[k] * (h * weight);
            }
            moves[stage + 1] = v;
            pulls[stage + 1] = acceleration(time + DP_C[stage] * h, x, v);
        }
        let (mut x5, mut v5, mut x4, mut v4) = (position, velocity, position, velocity);
        for k in 0..7 {
            x5 += moves[k] * (h * DP_FIFTH[k]);
            v5 += pulls[k] * (h * DP_FIFTH[k]);
            x4 += moves[k] * (h * DP_FOURTH[k]);
            v4 += pulls[k] * (h * DP_FOURTH[k]);
        }
        if !finite(x5) || !finite(v5) {
            return Err(PhysicsError::NonFinite("propagation"));
        }
        let error = (x5 - x4).length().max((v5 - v4).length() * h.abs());
        if error <= tolerance {
            time += h;
            position = x5;
            velocity = v5;
        }
        let grow = if error > 0.0 {
            0.9 * (tolerance / error).powf(0.2)
        } else {
            5.0
        };
        step = h.abs() * grow.clamp(0.2, 5.0);
    }
    Err(PhysicsError::OutOfRange)
}

/// `steps` steps of `dt` of kick, drift, kick leapfrog under a pull of position alone. It is
/// symplectic: a coast's energy swings but does not drift, over thousands of turns.
pub fn leapfrog(
    pull: impl Fn(DVec3) -> DVec3,
    (mut position, mut velocity): (DVec3, DVec3),
    dt: f64,
    steps: usize,
) -> (DVec3, DVec3) {
    let mut acceleration = pull(position);
    for _ in 0..steps {
        velocity += acceleration * (0.5 * dt);
        position += velocity * dt;
        acceleration = pull(position);
        velocity += acceleration * (0.5 * dt);
    }
    (position, velocity)
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

    fn close(a: DVec3, b: DVec3, scale: f64) -> bool {
        (a - b).length() <= 1e-9 * scale
    }

    #[test]
    fn a_coast_follows_the_same_ellipse_as_its_elements() {
        let orbit = earth_moon();
        let coast = orbit.coast();
        for share in [0.0, 0.13, 0.5, 0.77, 1.0, 3.4, -0.6] {
            let time = orbit.period() * share;
            let (position, velocity) = orbit.state_at(time);
            let (along, speed) = coast.state_at(time);
            assert!(close(along, position, orbit.semi_major_axis), "{share}");
            assert!(close(speed, velocity, velocity.length()), "{share}");
        }
        assert!((coast.period().unwrap() - orbit.period()).abs() < 1e-6);
    }

    #[test]
    fn elements_come_back_from_their_state() {
        let tilted = earth_moon();
        let flat = Orbit::circular(3.986e14, 7.0e6);
        let polar = Orbit {
            inclination: std::f64::consts::FRAC_PI_2,
            eccentricity: 0.3,
            ascending_node: 4.0,
            periapsis: 5.5,
            mean_anomaly: 0.2,
            ..Orbit::circular(3.986e14, 2.0e7)
        };
        for orbit in [tilted, flat, polar] {
            let (position, velocity) = orbit.state_at(0.0);
            let back = Orbit::from_state(orbit.mu, position, velocity).unwrap();
            // the same path whatever the angles came out as
            for share in [0.0, 0.3, 0.71] {
                let time = orbit.period() * share;
                let (a, _) = orbit.state_at(time);
                let (b, _) = back.state_at(time);
                assert!(
                    close(a, b, orbit.semi_major_axis * 10.0),
                    "{orbit:?} {back:?}"
                );
            }
            assert!((back.eccentricity - orbit.eccentricity).abs() < 1e-9);
            assert!((back.inclination - orbit.inclination).abs() < 1e-9);
        }
        let (position, velocity) = earth_moon().state_at(0.0);
        assert!(Orbit::from_state(earth_moon().mu, position, velocity * 2.0).is_none());
    }

    #[test]
    fn open_paths_keep_their_energy_and_turn() {
        let mu = 3.986e14;
        let r = DVec3::new(7.0e6, 0.0, 0.0);
        let escape = (2.0 * mu / r.length()).sqrt();
        for speed in [escape, escape * 1.4] {
            let coast = Coast::new(mu, r, DVec3::new(0.0, 0.3, -1.0).normalize() * speed).unwrap();
            // at escape speed rounding puts the energy either side of zero
            assert!(speed == escape || coast.period().is_none());
            let turn = coast.position.cross(coast.velocity);
            for time in [600.0, 3.0e4, 4.0e5, -2.0e3] {
                let (position, velocity) = coast.state_at(time);
                let moved = Coast::new(mu, position, velocity).unwrap();
                let energy = coast.energy().abs().max(1.0);
                assert!((moved.energy() - coast.energy()).abs() < 1e-6 * energy.max(mu / 7.0e6));
                assert!((position.cross(velocity) - turn).length() < 1e-9 * turn.length());
            }
            // on the way out it only gets further
            let (near, _) = coast.state_at(1.0e4);
            let (far, _) = coast.state_at(1.0e5);
            assert!(far.length() > near.length());
        }
    }

    #[test]
    fn a_hohmann_burn_reaches_the_higher_orbit() {
        let mu = 3.986e14;
        let (low, high) = (7.0e6, 4.2e7);
        let start = Orbit::circular(mu, low).coast();
        let speed = (mu / low).sqrt();
        let change = speed * ((2.0 * high / (low + high)).sqrt() - 1.0);
        let along = start.velocity.normalize();
        let transfer = start.burn(0.0, along * change);
        let (near, far) = transfer.apses();
        assert!((near - low).abs() < 1e-6 * low, "{near}");
        assert!((far.unwrap() - high).abs() < 1e-6 * high, "{far:?}");
        // half a turn on it is at the top
        let half = transfer.period().unwrap() / 2.0;
        let (top, _) = transfer.state_at(half);
        assert!((top.length() - high).abs() < 1e-6 * high);
    }

    #[test]
    fn the_propagators_follow_a_coast() {
        let orbit = earth_moon();
        let coast = Orbit {
            semi_major_axis: 2.0e7,
            ..orbit
        }
        .coast();
        let mu = coast.mu;
        let pull = |at: DVec3| -at * (mu / at.length().powi(3));
        let period = coast.period().unwrap();
        let start = (coast.position, coast.velocity);
        let (position, _) =
            propagate(|_, at, _| pull(at), start, (0.0, period * 0.37), 1e-3).unwrap();
        let (exact, _) = coast.state_at(period * 0.37);
        assert!(
            (position - exact).length() < 1.0,
            "{}",
            (position - exact).length()
        );
        // a hundred turns of leapfrog: the energy swings, it does not drift
        let steps_per_turn = 2000;
        let dt = period / steps_per_turn as f64;
        let mut state = start;
        let (mut lowest, mut highest) = (f64::MAX, f64::MIN);
        for _ in 0..100 {
            state = leapfrog(pull, state, dt, steps_per_turn);
            let energy = Coast::new(mu, state.0, state.1).unwrap().energy();
            lowest = lowest.min(energy);
            highest = highest.max(energy);
        }
        let first = coast.energy();
        assert!(
            ((highest - lowest) / first).abs() < 1e-3,
            "{lowest} {highest}"
        );
        assert!(((highest - first) / first).abs() < 1e-3);
        assert!(propagate(|_, at, _| pull(at), start, (0.0, 1.0), 0.0).is_err());
    }
}
