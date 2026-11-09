use algebrix::{DQuat, DVec3, Vec3};

use crate::frame::{FrameId, Frames};
use crate::{GridPosition, PhysicsError};

/// Most gravity sources a world holds.
pub const MAX_GRAVITY_SOURCES: usize = 1024;

/// A body that pulls everything toward it, standing still in `frame` at `center`. Bodies feel it
/// where they are, on top of the world's uniform gravity, which a planet's world sets to zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravitySource {
    pub frame: FrameId,
    pub center: DVec3,
    /// Mass times the gravitational constant, m^3/s^2.
    pub mu: f64,
    /// Zero pulls as a point. A planet of this radius pulls as a point outside it and as a
    /// uniform ball inside it, falling to nothing at its middle.
    pub radius: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GravityId(pub(crate) u32);

impl GravitySource {
    pub fn point(frame: FrameId, center: DVec3, mu: f64) -> Self {
        Self {
            frame,
            center,
            mu,
            radius: 0.0,
        }
    }

    pub(crate) fn is_valid(&self) -> bool {
        let center = self.center;
        center.x.is_finite()
            && center.y.is_finite()
            && center.z.is_finite()
            && self.mu.is_finite()
            && self.mu > 0.0
            && self.radius.is_finite()
            && self.radius >= 0.0
    }

    /// Pull at `offset` from the center, any axes.
    pub fn pull(&self, offset: DVec3) -> DVec3 {
        let distance = offset.length();
        if distance <= 0.0 {
            return DVec3::ZERO;
        }
        let reach = distance.max(self.radius);
        -offset * (self.mu / (reach * reach * reach))
    }
}

/// What a frame's bodies feel from the step's gravity and from the frame's own motion.
#[derive(Debug, Clone, Copy)]
struct FrameGravity {
    uniform: Vec3,
    origin: DVec3,
    rotation: DQuat,
    /// The frame's motion against the root, in its axes: how its origin speeds up, how fast it
    /// turns and how its turning speeds up. All zero for a still frame.
    moving: bool,
    acceleration: DVec3,
    spin: DVec3,
    spin_change: DVec3,
}

/// Gravity for one step: the uniform part, the sources where they are, and each frame's motion.
#[derive(Debug, Clone, Default)]
pub(crate) struct StepGravity {
    frames: Vec<FrameGravity>,
    sources: Vec<GravitySource>,
    /// Source centers in the root.
    centers: Vec<DVec3>,
}

/// Seconds either side of a step's time its frames' motion is read at, for how it changes.
const SLOPE_TIME: f64 = 1e-3;

impl StepGravity {
    pub(crate) fn prepare(
        &mut self,
        uniform: Vec3,
        frames: &Frames,
        sources: &[Option<GravitySource>],
    ) -> Result<(), PhysicsError> {
        let time = frames.time();
        let uniform = DVec3::from_vec3(uniform);
        self.frames.clear();
        for index in 0..frames.len() {
            let frame = FrameId(index as u32);
            let rotation = frames.root_rotation(frame)?;
            let origin = frames.origin_at(frame, time);
            let back = rotation.conjugate();
            let moving = frames.moves(frame);
            let (acceleration, spin, spin_change) = if moving {
                let before = frames.origin_at(frame, time - SLOPE_TIME);
                let after = frames.origin_at(frame, time + SLOPE_TIME);
                let slope = 0.5 / SLOPE_TIME;
                (
                    back.mul_vec3((after.linear_velocity - before.linear_velocity) * slope),
                    back.mul_vec3(origin.angular_velocity),
                    back.mul_vec3((after.angular_velocity - before.angular_velocity) * slope),
                )
            } else {
                (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO)
            };
            self.frames.push(FrameGravity {
                uniform: back.mul_vec3(uniform).as_vec3(),
                origin: origin.position,
                rotation,
                moving,
                acceleration,
                spin,
                spin_change,
            });
        }
        self.sources.clear();
        self.centers.clear();
        for source in sources.iter().flatten() {
            let frame = &self.frames[source.frame.0 as usize];
            self.centers
                .push(frame.origin + frame.rotation.mul_vec3(source.center));
            self.sources.push(*source);
        }
        Ok(())
    }

    /// The acceleration a body in `frame` at `position` moving at `velocity` feels, in that
    /// frame's axes. A moving frame adds what its motion does: minus its origin's acceleration,
    /// coriolis, centrifugal and the turning speeding up.
    pub(crate) fn at(&self, frame: FrameId, position: GridPosition, velocity: Vec3) -> Vec3 {
        let coriolis = self.frames[frame.0 as usize]
            .spin
            .cross(DVec3::from_vec3(velocity))
            * 2.0;
        self.without_coriolis(frame, position) - coriolis.as_vec3()
    }

    /// How fast `frame` turns against the root, in its axes, zero for one that does not.
    pub(crate) fn spin(&self, frame: FrameId) -> Vec3 {
        self.frames[frame.0 as usize].spin.as_vec3()
    }

    /// [`Self::at`] but coriolis, which only turns a velocity: the integrator turns it exactly
    /// with [`Self::spin`], as a push it gains speed and runs ahead.
    pub(crate) fn without_coriolis(&self, frame: FrameId, position: GridPosition) -> Vec3 {
        let frame = &self.frames[frame.0 as usize];
        if self.sources.is_empty() && !frame.moving {
            return frame.uniform;
        }
        let local = position.to_dvec3();
        let mut pull = DVec3::ZERO;
        if !self.sources.is_empty() {
            let root = frame.origin + frame.rotation.mul_vec3(local);
            for (source, center) in self.sources.iter().zip(&self.centers) {
                pull += source.pull(root - *center);
            }
            pull = frame.rotation.conjugate().mul_vec3(pull);
        }
        if frame.moving {
            let spin = frame.spin;
            pull -=
                frame.acceleration + spin.cross(spin.cross(local)) + frame.spin_change.cross(local);
        }
        frame.uniform + pull.as_vec3()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_planet_pulls_as_a_point_outside_and_less_inside() {
        let planet = GravitySource {
            radius: 6.371e6,
            ..GravitySource::point(FrameId::ROOT, DVec3::ZERO, 3.986e14)
        };
        let surface = planet.pull(DVec3::new(0.0, 6.371e6, 0.0));
        assert!((surface.y + 9.82).abs() < 0.01, "{surface:?}");
        let far = planet.pull(DVec3::new(0.0, 2.0 * 6.371e6, 0.0));
        assert!((far.y - surface.y / 4.0).abs() < 1e-9);
        let half = planet.pull(DVec3::new(0.0, 0.5 * 6.371e6, 0.0));
        assert!((half.y - surface.y / 2.0).abs() < 1e-9);
        assert_eq!(planet.pull(DVec3::ZERO), DVec3::ZERO);
    }

    use crate::frame::{FrameDesc, FrameMotion, FrameState, Orbit};
    use crate::world::testing::no_gravity;
    use crate::{BodyHandle, PhysicsWorld, RigidBodyDesc, Shape};

    fn ball(world: &mut PhysicsWorld, frame: FrameId, at: Vec3, velocity: Vec3) -> BodyHandle {
        let mut desc = RigidBodyDesc::dynamic(Shape::sphere(0.5))
            .local_position(at)
            .linear_velocity(velocity);
        desc.frame = frame;
        desc.can_sleep = false;
        world.spawn(desc).unwrap()
    }

    fn in_root(world: &PhysicsWorld, body: BodyHandle) -> FrameState {
        let state = world.body_state(body).unwrap();
        world
            .frames()
            .to_root(
                state.frame,
                FrameState {
                    position: state.position.to_dvec3(),
                    rotation: DQuat::from_quat(state.rotation),
                    linear_velocity: DVec3::from_vec3(state.linear_velocity),
                    angular_velocity: DVec3::from_vec3(state.angular_velocity),
                },
            )
            .unwrap()
    }

    #[test]
    fn a_ball_on_a_circular_orbit_stays_on_it() {
        let mut world = no_gravity();
        // one turn a minute at 100 m
        let (radius, period) = (100.0_f64, 60.0_f64);
        let mu = (std::f64::consts::TAU / period).powi(2) * radius.powi(3);
        world
            .add_gravity(GravitySource::point(FrameId::ROOT, DVec3::ZERO, mu))
            .unwrap();
        let speed = (mu / radius).sqrt();
        let body = ball(
            &mut world,
            FrameId::ROOT,
            Vec3::new(radius as f32, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -speed as f32),
        );
        let (mut nearest, mut farthest) = (f64::MAX, 0.0_f64);
        for _ in 0..(10.0 * period * 60.0) as usize {
            world.step_fixed().unwrap();
            let distance = in_root(&world, body).position.length();
            nearest = nearest.min(distance);
            farthest = farthest.max(distance);
        }
        assert!(farthest - nearest < 0.01 * radius, "{nearest} {farthest}");
    }

    #[test]
    fn a_ball_still_in_the_root_stays_still_seen_from_a_spinning_frame() {
        let mut world = no_gravity();
        let spin = DVec3::new(0.0, 0.7, 0.0);
        let frame = world
            .add_frame(FrameDesc {
                motion: FrameMotion { spin, orbit: None },
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        // still in the root means going round backwards in the frame
        let at = DVec3::new(4.0, 1.0, 0.0);
        let body = ball(&mut world, frame, at.as_vec3(), (-spin.cross(at)).as_vec3());
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        let root = in_root(&world, body);
        // 1.6 cm off at 0.7 rad/s, a planet turns ten thousand times slower
        assert!((root.position - at).length() < 0.02, "{:?}", root.position);
        assert!(
            root.linear_velocity.length() < 0.02,
            "{:?}",
            root.linear_velocity
        );
    }

    #[test]
    fn a_ball_in_a_frame_orbiting_a_source_falls_with_it() {
        let mut world = no_gravity();
        let (radius, period) = (1000.0_f64, 120.0_f64);
        let mu = (std::f64::consts::TAU / period).powi(2) * radius.powi(3);
        world
            .add_gravity(GravitySource::point(FrameId::ROOT, DVec3::ZERO, mu))
            .unwrap();
        let moon = world
            .add_frame(FrameDesc {
                motion: FrameMotion {
                    spin: DVec3::ZERO,
                    orbit: Some(Orbit::circular(mu, radius)),
                },
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        let body = ball(&mut world, moon, Vec3::ZERO, Vec3::ZERO);
        for _ in 0..(period * 60.0) as usize {
            world.step_fixed().unwrap();
        }
        // what is left is the tide over a meter, nothing of the 2.7 m/s^2 the frame falls with
        let state = world.body_state(body).unwrap();
        assert!(
            state.position.to_dvec3().length() < 0.05,
            "{:?}",
            state.position
        );
        let distance = world.frames().get(moon).unwrap().translation.length();
        assert!((distance - radius).abs() < 1e-6 * radius);
    }

    #[test]
    fn a_ball_crossing_a_frame_goes_in_and_out_at_its_reach() {
        let mut world = no_gravity();
        let station = world
            .add_frame(FrameDesc {
                translation: DVec3::new(100.0, 0.0, 0.0),
                rotation: DQuat::from_axis_angle(DVec3::Y, 0.7),
                reach: 10.0,
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        let body = ball(
            &mut world,
            FrameId::ROOT,
            Vec3::new(80.0, 0.0, 0.0),
            Vec3::X * 10.0,
        );
        let mut changes = Vec::new();
        for step in 1..=240 {
            world.step_fixed().unwrap();
            for change in world.frame_changes() {
                let x = in_root(&world, body).position.x;
                changes.push((change.1, change.2, x));
            }
            // its path in the root goes on as if no frame were there, to f32 rounding over 900
            // moves
            let expected = 80.0 + 10.0 * f64::from(step) / 60.0;
            let got = in_root(&world, body).position;
            assert!((got.x - expected).abs() < 5e-3, "{step} {got:?} {expected}");
        }
        assert_eq!(changes.len(), 2, "{changes:?}");
        assert_eq!((changes[0].0, changes[0].1), (FrameId::ROOT, station));
        assert!(changes[0].2 > 91.0 && changes[0].2 < 91.3, "{changes:?}");
        assert_eq!((changes[1].0, changes[1].1), (station, FrameId::ROOT));
        assert!(changes[1].2 > 111.0 && changes[1].2 < 111.3, "{changes:?}");
    }

    #[test]
    fn moving_frames_and_sources_continue_exactly_after_a_snapshot() {
        let mut world = no_gravity();
        let mu = 4.0e6;
        world
            .add_gravity(GravitySource {
                radius: 20.0,
                ..GravitySource::point(FrameId::ROOT, DVec3::ZERO, mu)
            })
            .unwrap();
        let moon = world
            .add_frame(FrameDesc {
                motion: FrameMotion {
                    spin: DVec3::new(0.0, 0.0, 0.3),
                    orbit: Some(Orbit {
                        eccentricity: 0.2,
                        inclination: 0.3,
                        ..Orbit::circular(mu, 300.0)
                    }),
                },
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        ball(&mut world, moon, Vec3::new(2.0, 0.0, 0.0), Vec3::Y);
        ball(
            &mut world,
            FrameId::ROOT,
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::X * 5.0,
        );
        for _ in 0..90 {
            world.step_fixed().unwrap();
        }
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        assert_eq!(loaded.frames(), world.frames());
        for _ in 0..90 {
            world.step_fixed().unwrap();
            loaded.step_fixed().unwrap();
        }
        assert_eq!(loaded.state_hash(), world.state_hash());
        assert_eq!(loaded.save_snapshot(), world.save_snapshot());
    }

    #[test]
    fn bad_sources_are_refused() {
        let good = GravitySource::point(FrameId::ROOT, DVec3::ZERO, 1.0);
        assert!(good.is_valid());
        assert!(!GravitySource { mu: 0.0, ..good }.is_valid());
        assert!(
            !GravitySource {
                radius: -1.0,
                ..good
            }
            .is_valid()
        );
        assert!(
            !GravitySource {
                center: DVec3::new(f64::NAN, 0.0, 0.0),
                ..good
            }
            .is_valid()
        );
    }
}
