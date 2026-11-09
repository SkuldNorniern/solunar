use algebrix::{DVec3, Quat, Vec3};

use crate::body::{BodyHandle, BodyKind};
use crate::error::PhysicsError;
use crate::frame::{FrameId, FrameState};
use crate::position::GridPosition;
use crate::snapshot::{Reader, SnapshotError, Writer};
use crate::world::PhysicsWorld;

/// Gas constant of dry air, J/(kg K), and its ratio of heats.
const AIR_GAS: f32 = 287.05;
const AIR_HEATS: f32 = 1.4;
/// Sutton and graves' constant for air, SI, for heating at a blunt nose.
const SUTTON_GRAVES: f32 = 1.7415e-4;
/// Most atmospheres and aero bodies a world holds.
pub const MAX_ATMOSPHERES: usize = 64;
pub const MAX_AERO_BODIES: usize = 1 << 16;
/// Most lifting surfaces on one body.
pub const MAX_SURFACES: usize = 64;
/// Angle of attack past which a surface stalls, its lift falling away.
const STALL: f32 = 15.0 * std::f32::consts::PI / 180.0;

/// Air round a planet: thinning by `scale_height` going up, cooling by `lapse` to `floor`
/// degrees, turning with `frame` as the planet does. Centered on `center` in that frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereDesc {
    pub frame: FrameId,
    pub center: DVec3,
    /// Meters to the ground the heights count from.
    pub radius: f64,
    /// Kg/m^3 at the ground.
    pub density: f32,
    /// Meters over which the density falls to 1/e.
    pub scale_height: f32,
    /// Kelvin at the ground, falling `lapse` K/m up to `floor` K.
    pub temperature: f32,
    pub lapse: f32,
    pub floor: f32,
    /// Meters over the ground past which there is no air at all.
    pub top: f32,
    /// Wind in `frame` axes, the same everywhere.
    pub wind: Vec3,
}

impl AtmosphereDesc {
    /// The earth's, roughly: 1.225 kg/m^3 at sea level, 8.5 km scale height, 288 K falling
    /// 6.5 K a km to 217 K, nothing past 150 km.
    pub fn earth(frame: FrameId, center: DVec3, radius: f64) -> Self {
        Self {
            frame,
            center,
            radius,
            density: 1.225,
            scale_height: 8_500.0,
            temperature: 288.15,
            lapse: 0.0065,
            floor: 216.65,
            top: 150_000.0,
            wind: Vec3::ZERO,
        }
    }

    fn is_valid(&self) -> bool {
        let positive = |value: f32| value.is_finite() && value > 0.0;
        self.center.x.is_finite()
            && self.center.y.is_finite()
            && self.center.z.is_finite()
            && self.radius.is_finite()
            && self.radius >= 0.0
            && self.density.is_finite()
            && self.density >= 0.0
            && positive(self.scale_height)
            && positive(self.temperature)
            && self.lapse.is_finite()
            && self.lapse >= 0.0
            && positive(self.floor)
            && positive(self.top)
            && self.wind.is_finite()
    }

    /// The air `height` meters over the ground, still air.
    pub fn at_height(&self, height: f32) -> (f32, f32, f32) {
        if height > self.top {
            return (0.0, 0.0, self.floor);
        }
        let height = height.max(0.0);
        let density = self.density * (-height / self.scale_height).exp();
        let temperature = (self.temperature - self.lapse * height).max(self.floor);
        (density, density * AIR_GAS * temperature, temperature)
    }
}

/// The air at a point: kg/m^3, pascals, kelvin, its motion in the asking frame's axes, and how
/// fast sound goes in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Air {
    pub density: f32,
    pub pressure: f32,
    pub temperature: f32,
    pub wind: Vec3,
    pub sound: f32,
}

impl Air {
    pub const NONE: Self = Self {
        density: 0.0,
        pressure: 0.0,
        temperature: 0.0,
        wind: Vec3::ZERO,
        sound: 0.0,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AtmosphereId(pub(crate) u32);

/// A flat lifting surface on a body, a wing or a fin: where it is and its normal in the body's
/// axes, its area. Its lift is 2 pi per radian of attack up to the stall.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    pub position: Vec3,
    pub normal: Vec3,
    pub area: f32,
}

/// How a body meets the air: its drag coefficient over `area` square meters, the radius of its
/// nose for heating, its lifting surfaces.
#[derive(Debug, Clone, PartialEq)]
pub struct AeroDesc {
    pub drag: f32,
    pub area: f32,
    pub nose: f32,
    pub surfaces: Vec<Surface>,
}

impl AeroDesc {
    fn is_valid(&self) -> bool {
        let positive = |value: f32| value.is_finite() && value > 0.0;
        self.drag.is_finite()
            && self.drag >= 0.0
            && self.area.is_finite()
            && self.area >= 0.0
            && positive(self.nose)
            && self.surfaces.len() <= MAX_SURFACES
            && self.surfaces.iter().all(|surface| {
                surface.position.is_finite()
                    && surface.normal.is_finite()
                    && surface.normal.length() > 1e-6
                    && surface.area.is_finite()
                    && surface.area >= 0.0
            })
    }
}

/// What the air did to a body in the last step.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AeroReport {
    /// Half density times speed squared through the air, pascals.
    pub dynamic_pressure: f32,
    pub mach: f32,
    /// Watts a square meter at its nose, sutton and graves.
    pub heating: f32,
    /// Newtons on it, in its frame's axes, drag and lift.
    pub force: Vec3,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Atmospheres {
    slots: Vec<Option<AtmosphereDesc>>,
    /// Bodies the air pushes, by handle, with the last step's report.
    bodies: Vec<(BodyHandle, AeroDesc, AeroReport)>,
}

impl Atmospheres {
    pub(crate) fn save(&self, out: &mut Writer) {
        out.len(self.slots.len());
        for slot in &self.slots {
            let Some(air) = slot else {
                out.u8(0);
                continue;
            };
            out.u8(1);
            out.u32(air.frame.index());
            out.dvec3(air.center);
            out.f64(air.radius);
            for value in [
                air.density,
                air.scale_height,
                air.temperature,
                air.lapse,
                air.floor,
                air.top,
            ] {
                out.f32(value);
            }
            out.vec3(air.wind);
        }
        out.len(self.bodies.len());
        for (body, desc, _) in &self.bodies {
            out.body(*body);
            out.f32(desc.drag);
            out.f32(desc.area);
            out.f32(desc.nose);
            out.len(desc.surfaces.len());
            for surface in &desc.surfaces {
                out.vec3(surface.position);
                out.vec3(surface.normal);
                out.f32(surface.area);
            }
        }
    }

    pub(crate) fn load(
        reader: &mut Reader<'_>,
        frames: &crate::frame::Frames,
    ) -> Result<Self, SnapshotError> {
        let bad = SnapshotError::Invalid("atmosphere");
        let mut air = Self::default();
        let count = reader.count(1, MAX_ATMOSPHERES)?;
        for _ in 0..count {
            let slot = match reader.u8()? {
                0 => None,
                1 => {
                    let desc = AtmosphereDesc {
                        frame: FrameId(reader.u32()?),
                        center: reader.dvec3()?,
                        radius: reader.f64()?,
                        density: reader.f32()?,
                        scale_height: reader.f32()?,
                        temperature: reader.f32()?,
                        lapse: reader.f32()?,
                        floor: reader.f32()?,
                        top: reader.f32()?,
                        wind: reader.vec3()?,
                    };
                    if !desc.is_valid() || !frames.contains(desc.frame) {
                        return Err(bad);
                    }
                    Some(desc)
                }
                _ => return Err(bad),
            };
            air.slots.push(slot);
        }
        let count = reader.count(24, MAX_AERO_BODIES)?;
        for _ in 0..count {
            let body = reader.handle()?;
            let (drag, area, nose) = (reader.f32()?, reader.f32()?, reader.f32()?);
            let surfaces = reader.count(28, MAX_SURFACES)?;
            let mut desc = AeroDesc {
                drag,
                area,
                nose,
                surfaces: Vec::with_capacity(surfaces),
            };
            for _ in 0..surfaces {
                desc.surfaces.push(Surface {
                    position: reader.vec3()?,
                    normal: reader.vec3()?,
                    area: reader.f32()?,
                });
            }
            if !desc.is_valid() || air.bodies.last().is_some_and(|last| last.0 >= body) {
                return Err(bad);
            }
            air.bodies.push((body, desc, AeroReport::default()));
        }
        Ok(air)
    }
}

impl PhysicsWorld {
    /// # Errors
    /// Fails on an unknown frame, values out of range or too many atmospheres.
    pub fn add_atmosphere(&mut self, desc: AtmosphereDesc) -> Result<AtmosphereId, PhysicsError> {
        if !self.frames.contains(desc.frame) || !desc.is_valid() {
            return Err(PhysicsError::InvalidSettings);
        }
        let slots = &mut self.atmospheres.slots;
        let index = match slots.iter().position(Option::is_none) {
            Some(index) => index,
            None if slots.len() < MAX_ATMOSPHERES => {
                slots.push(None);
                slots.len() - 1
            }
            None => return Err(PhysicsError::OutOfRange),
        };
        slots[index] = Some(desc);
        Ok(AtmosphereId(index as u32))
    }

    /// Returns false for one that was not there.
    pub fn remove_atmosphere(&mut self, id: AtmosphereId) -> bool {
        self.atmospheres
            .slots
            .get_mut(id.0 as usize)
            .and_then(Option::take)
            .is_some()
    }

    /// Lets the air push `body` from the next step on, or stops it with `None`.
    ///
    /// # Errors
    /// Fails on a stale handle or a bad desc.
    pub fn set_aero(
        &mut self,
        body: BodyHandle,
        desc: Option<AeroDesc>,
    ) -> Result<(), PhysicsError> {
        self.bodies.dense_index(body)?;
        let list = &mut self.atmospheres.bodies;
        let at = list.binary_search_by_key(&body, |entry| entry.0);
        match (desc, at) {
            (Some(desc), _) if !desc.is_valid() => Err(PhysicsError::InvalidShape),
            (Some(desc), Ok(at)) => {
                list[at].1 = desc;
                Ok(())
            }
            (Some(desc), Err(at)) if list.len() < MAX_AERO_BODIES => {
                list.insert(at, (body, desc, AeroReport::default()));
                Ok(())
            }
            (Some(_), Err(_)) => Err(PhysicsError::OutOfRange),
            (None, Ok(at)) => {
                list.remove(at);
                Ok(())
            }
            (None, Err(_)) => Ok(()),
        }
    }

    /// What the air did to `body` in the last step, none for a body without aero.
    pub fn aero_report(&self, body: BodyHandle) -> Option<AeroReport> {
        let list = &self.atmospheres.bodies;
        let at = list.binary_search_by_key(&body, |entry| entry.0).ok()?;
        Some(list[at].2)
    }

    /// The air at `position` in `frame`, its wind in that frame's axes. The densest atmosphere
    /// there, still air for none.
    ///
    /// # Errors
    /// Fails on an unknown frame.
    pub fn air_at(&self, frame: FrameId, position: GridPosition) -> Result<Air, PhysicsError> {
        self.frames.get(frame)?;
        let mut best = Air::NONE;
        for desc in self.atmospheres.slots.iter().flatten() {
            // the point in the atmosphere's frame, still there
            let there = self.frames.transfer(
                frame,
                desc.frame,
                FrameState {
                    position: position.to_dvec3(),
                    rotation: algebrix::DQuat::IDENTITY,
                    linear_velocity: DVec3::ZERO,
                    angular_velocity: DVec3::ZERO,
                },
            )?;
            let height = ((there.position - desc.center).length() - desc.radius) as f32;
            let (density, pressure, temperature) = desc.at_height(height);
            if density <= best.density {
                continue;
            }
            // air still in its frame, as moving in the asking one
            let back = self.frames.transfer(
                desc.frame,
                frame,
                FrameState {
                    position: there.position,
                    rotation: algebrix::DQuat::IDENTITY,
                    linear_velocity: DVec3::from_vec3(desc.wind),
                    angular_velocity: DVec3::ZERO,
                },
            )?;
            best = Air {
                density,
                pressure,
                temperature,
                wind: back.linear_velocity.as_vec3(),
                sound: (AIR_HEATS * AIR_GAS * temperature).sqrt(),
            };
        }
        Ok(best)
    }

    /// Drag, lift and heating on each aero body as the step starts, put on as force and torque
    /// for it. Asleep bodies are left.
    pub(crate) fn apply_aero(&mut self) -> Result<(), PhysicsError> {
        if self.atmospheres.bodies.is_empty() || self.atmospheres.slots.iter().all(Option::is_none)
        {
            return Ok(());
        }
        let mut list = std::mem::take(&mut self.atmospheres.bodies);
        list.retain(|entry| self.bodies.dense_index(entry.0).is_ok());
        for (body, desc, report) in &mut list {
            *report = AeroReport::default();
            let dense = self.bodies.dense_index(*body)?;
            if self.bodies.kind[dense] != BodyKind::Dynamic || self.bodies.asleep[dense] {
                continue;
            }
            let air = self.air_at(self.bodies.frame[dense], self.bodies.position(dense))?;
            if air.density <= 0.0 {
                continue;
            }
            let (force, torque, airflow) = aero_force(
                desc,
                &air,
                self.bodies.rotation[dense],
                (self.bodies.linear[dense], self.bodies.angular[dense]),
            );
            let speed = airflow.length();
            *report = AeroReport {
                dynamic_pressure: 0.5 * air.density * speed * speed,
                mach: speed / air.sound.max(1e-3),
                heating: SUTTON_GRAVES * (air.density / desc.nose).sqrt() * speed.powi(3),
                force,
            };
            self.bodies.force[dense] += force;
            self.bodies.torque[dense] += torque;
        }
        self.atmospheres.bodies = list;
        Ok(())
    }
}

/// Drag coefficient over how fast it goes against sound: up by half through the sound barrier,
/// a little over 1 past it, as blunt bodies do, flat below.
fn transonic(mach: f32) -> f32 {
    if mach < 0.8 {
        1.0
    } else if mach < 1.2 {
        1.0 + 0.5 * (mach - 0.8) / 0.4
    } else {
        1.5 - 0.3 * ((mach - 1.2) / 2.0).min(1.0)
    }
}

/// Force, torque about the center and the air's speed past it, for a body moving at `linear`
/// turning at `angular`.
fn aero_force(
    desc: &AeroDesc,
    air: &Air,
    rotation: Quat,
    (linear, angular): (Vec3, Vec3),
) -> (Vec3, Vec3, Vec3) {
    let past = linear - air.wind;
    let speed = past.length();
    if speed <= 0.0 {
        return (Vec3::ZERO, Vec3::ZERO, past);
    }
    let mach = speed / air.sound.max(1e-3);
    let pressure = 0.5 * air.density * speed * speed;
    let mut force = -past / speed * (pressure * desc.drag * transonic(mach) * desc.area);
    let mut torque = Vec3::ZERO;
    for surface in &desc.surfaces {
        let arm = rotation.rotate_vec3(surface.position);
        let normal = rotation.rotate_vec3(surface.normal).normalize();
        let flow = linear + angular.cross(arm) - air.wind;
        let flow_speed = flow.length();
        if flow_speed <= 0.0 {
            continue;
        }
        let along = flow / flow_speed;
        // attack: how far the flow comes onto the face, positive from its back
        let attack = (-along.dot(normal)).clamp(-1.0, 1.0).asin();
        let coefficient = if attack.abs() <= STALL {
            std::f32::consts::TAU * attack
        } else {
            // past the stall the lift falls back toward a flat plate's
            std::f32::consts::TAU * STALL * attack.signum() * (STALL / attack.abs())
        };
        // lift across the flow in the plane of flow and normal
        let side = normal - along * normal.dot(along);
        let across = side.length();
        if across < 1e-6 {
            continue;
        }
        let up = side / across;
        let lift = up * (0.5 * air.density * flow_speed * flow_speed * surface.area * coefficient);
        force += lift;
        torque += arm.cross(lift);
    }
    (force, torque, past)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity::GravitySource;
    use crate::world::testing::no_gravity;
    use crate::{PhysicsSettings, RigidBodyDesc, Shape};

    fn ball_aero(radius: f32) -> AeroDesc {
        AeroDesc {
            drag: 0.47,
            area: std::f32::consts::PI * radius * radius,
            nose: radius,
            surfaces: Vec::new(),
        }
    }

    #[test]
    fn earth_air_thins_and_cools_going_up() {
        let air = AtmosphereDesc::earth(FrameId::ROOT, DVec3::ZERO, 6.371e6);
        let (sea, pressure, warm) = air.at_height(0.0);
        assert!((sea - 1.225).abs() < 1e-6);
        assert!((pressure - 101_325.0).abs() < 200.0, "{pressure}");
        assert!((warm - 288.15).abs() < 1e-3);
        let (high, _, cold) = air.at_height(8_500.0);
        assert!((high - 1.225 / std::f32::consts::E).abs() < 1e-3);
        assert!((cold - (288.15 - 55.25)).abs() < 1e-2);
        assert_eq!(air.at_height(200_000.0).0, 0.0);
        assert!((air.at_height(30_000.0).2 - 216.65).abs() < 1e-3);
    }

    #[test]
    fn a_falling_ball_settles_at_its_terminal_speed() {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        // a planet far under so the air is near sea level all the way
        world
            .add_atmosphere(AtmosphereDesc::earth(
                FrameId::ROOT,
                DVec3::new(0.0, -6.371e6, 0.0),
                6.371e6,
            ))
            .unwrap();
        let radius = 0.1;
        let mut desc = RigidBodyDesc::dynamic(Shape::sphere(radius))
            .density(500.0)
            .local_position(Vec3::new(0.0, 1000.0, 0.0));
        desc.can_sleep = false;
        let ball = world.spawn(desc).unwrap();
        world.set_aero(ball, Some(ball_aero(radius))).unwrap();
        for _ in 0..60 * 30 {
            world.step_fixed().unwrap();
        }
        let mass = 500.0 * 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3);
        let state = world.body_state(ball).unwrap();
        let air = world.air_at(FrameId::ROOT, state.position).unwrap();
        let terminal = (2.0 * mass * 9.81 / (air.density * 0.47 * ball_aero(radius).area)).sqrt();
        let speed = -state.linear_velocity.y;
        assert!(
            (speed - terminal).abs() < 0.01 * terminal,
            "{speed} {terminal}"
        );
        let report = world.aero_report(ball).unwrap();
        assert!(report.mach < 0.3 && report.heating > 0.0);
    }

    #[test]
    fn a_plate_at_five_degrees_lifts_two_pi_alpha() {
        let mut world = no_gravity();
        world
            .add_atmosphere(AtmosphereDesc::earth(
                FrameId::ROOT,
                DVec3::new(0.0, -6.371e6, 0.0),
                6.371e6,
            ))
            .unwrap();
        let mut desc = RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(1.0, 0.05, 1.0)))
            .linear_velocity(Vec3::new(100.0, 0.0, 0.0));
        desc.can_sleep = false;
        let plate = world.spawn(desc).unwrap();
        // the flow 5 degrees up into its underside
        let alpha = 5.0_f32.to_radians();
        let wing = AeroDesc {
            drag: 0.0,
            area: 0.0,
            nose: 1.0,
            surfaces: vec![Surface {
                position: Vec3::ZERO,
                normal: Vec3::new(-alpha.sin(), alpha.cos(), 0.0),
                area: 4.0,
            }],
        };
        world.set_aero(plate, Some(wing)).unwrap();
        world.step_fixed().unwrap();
        let report = world.aero_report(plate).unwrap();
        let expected = 0.5 * 1.225 * 100.0 * 100.0 * 4.0 * std::f32::consts::TAU * alpha;
        assert!(
            (report.force.y - expected * alpha.cos()).abs() < 0.02 * expected,
            "{report:?} {expected}"
        );
        assert!(report.force.y > 0.0);
    }

    #[test]
    fn an_entry_peaks_in_deceleration_as_allen_and_eggers_say() {
        // a dense capsule into an earth sized atmosphere at 7 km/s, 30 degrees down. its worst
        // deceleration is v^2 sin(g) / (2 e H) whatever its mass, gravity small beside it:
        // 530 m/s^2, here 564, the path bending down under gravity
        let radius = 6.371e6;
        let mut world = no_gravity();
        world
            .add_gravity(GravitySource {
                radius,
                ..GravitySource::point(FrameId::ROOT, DVec3::ZERO, 3.986e14)
            })
            .unwrap();
        world
            .add_atmosphere(AtmosphereDesc {
                lapse: 0.0,
                ..AtmosphereDesc::earth(FrameId::ROOT, DVec3::ZERO, radius)
            })
            .unwrap();
        let (speed, path) = (7_000.0_f32, 30.0_f32.to_radians());
        let start = radius + 120_000.0;
        let mut desc = RigidBodyDesc::dynamic(Shape::sphere(1.0))
            .density(3000.0)
            .position(GridPosition::from_dvec3(DVec3::new(0.0, start, 0.0)).unwrap())
            .linear_velocity(Vec3::new(speed * path.cos(), -speed * path.sin(), 0.0));
        desc.can_sleep = false;
        let capsule = world.spawn(desc).unwrap();
        world.set_aero(capsule, Some(ball_aero(1.0))).unwrap();
        let mut worst = 0.0_f32;
        let mut before = world.body_state(capsule).unwrap().linear_velocity;
        for _ in 0..60 * 120 {
            world.step_fixed().unwrap();
            let state = world.body_state(capsule).unwrap();
            if state.position.to_dvec3().length() < radius + 1_000.0 {
                break;
            }
            let pull = world
                .gravity_at(FrameId::ROOT, state.position, state.linear_velocity)
                .unwrap();
            worst = worst.max(((state.linear_velocity - before) * 60.0 - pull).length());
            before = state.linear_velocity;
        }
        let expected = speed * speed * path.sin() / (2.0 * std::f32::consts::E * 8_500.0);
        assert!(
            (worst - expected).abs() < 0.1 * expected,
            "{worst} {expected}"
        );
    }

    #[test]
    fn air_and_aero_bodies_continue_exactly_after_a_snapshot_and_bad_ones_fail() {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        world
            .add_atmosphere(AtmosphereDesc {
                scale_height: 7_777.5,
                wind: Vec3::new(3.0, 0.0, 1.0),
                ..AtmosphereDesc::earth(FrameId::ROOT, DVec3::new(0.0, -6.371e6, 0.0), 6.371e6)
            })
            .unwrap();
        let mut desc =
            RigidBodyDesc::dynamic(Shape::sphere(0.2)).local_position(Vec3::new(0.0, 50.0, 0.0));
        desc.can_sleep = false;
        let ball = world.spawn(desc).unwrap();
        world.set_aero(ball, Some(ball_aero(0.2))).unwrap();
        for _ in 0..20 {
            world.step_fixed().unwrap();
        }
        let bytes = world.save_snapshot();
        let mut loaded = PhysicsWorld::load_snapshot(&bytes).unwrap();
        for _ in 0..40 {
            world.step_fixed().unwrap();
            loaded.step_fixed().unwrap();
        }
        assert_eq!(loaded.save_snapshot(), world.save_snapshot());
        let marker = 7_777.5_f32.to_le_bytes();
        let at = bytes
            .windows(4)
            .position(|window| window == marker)
            .unwrap();
        for bad in [0.0_f32, -1.0, f32::NAN] {
            let mut broken = bytes.clone();
            broken[at..at + 4].copy_from_slice(&bad.to_le_bytes());
            let body = broken.len() - 8;
            let sum = crate::snapshot::testing::checksum(&broken[..body]);
            broken[body..].copy_from_slice(&sum.to_le_bytes());
            assert!(PhysicsWorld::load_snapshot(&broken).is_err());
        }
    }

    #[test]
    fn bad_air_is_refused() {
        let mut world = no_gravity();
        let earth = AtmosphereDesc::earth(FrameId::ROOT, DVec3::ZERO, 1.0);
        assert!(
            world
                .add_atmosphere(AtmosphereDesc {
                    scale_height: 0.0,
                    ..earth
                })
                .is_err()
        );
        assert!(
            world
                .add_atmosphere(AtmosphereDesc {
                    frame: FrameId(9),
                    ..earth
                })
                .is_err()
        );
        let ball = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(1.0)))
            .unwrap();
        assert!(
            world
                .set_aero(
                    ball,
                    Some(AeroDesc {
                        nose: 0.0,
                        ..ball_aero(1.0)
                    })
                )
                .is_err()
        );
        assert!(world.set_aero(ball, Some(ball_aero(1.0))).is_ok());
        assert!(world.aero_report(ball).is_some());
        world.set_aero(ball, None).unwrap();
        assert!(world.aero_report(ball).is_none());
    }
}
