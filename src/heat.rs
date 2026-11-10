use crate::body::BodyHandle;
use crate::error::PhysicsError;
use crate::snapshot::{Reader, SnapshotError, Writer};
use crate::world::PhysicsWorld;

pub const MAX_HEATED_BODIES: usize = 1 << 16;
const STEFAN_BOLTZMANN: f32 = 5.670_374e-8;
/// Kelvin of empty space, what a hull with no air round it radiates against.
const SPACE: f32 = 2.7;
/// Kelvin a hull starts at.
const START: f32 = 293.15;
const NEWTON_STEPS: usize = 30;

/// A body's hull as one lump of heat: what it holds per kelvin, where the air heats it, what
/// it radiates from, and how fast it trades heat with the gas rooms inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatDesc {
    /// Joules a kelvin, its mass times its specific heat.
    pub capacity: f32,
    /// Square meters taking the heating the air puts on its nose, see
    /// [`AeroReport::heating`](crate::AeroReport::heating).
    pub heated_area: f32,
    pub radiating_area: f32,
    /// 0 to 1.
    pub emissivity: f32,
    /// Watts a kelvin between it and each gas room in its body.
    pub gas_exchange: f32,
}

impl HeatDesc {
    fn is_valid(&self) -> bool {
        let amount = |value: f32| value.is_finite() && value >= 0.0;
        self.capacity.is_finite()
            && self.capacity > 0.0
            && amount(self.heated_area)
            && amount(self.radiating_area)
            && (0.0..=1.0).contains(&self.emissivity)
            && amount(self.gas_exchange)
    }
}

/// A hull's temperature and the heat that went through it in the last step, watts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatState {
    pub temperature: f32,
    /// From the air at its nose.
    pub absorbed: f32,
    /// Out as light, less what came back from round it.
    pub radiated: f32,
    /// Into its gas rooms, below zero when they warm it.
    pub to_gas: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Heat {
    bodies: Vec<(BodyHandle, HeatDesc, HeatState)>,
}

impl Heat {
    pub(crate) fn save(&self, out: &mut Writer) {
        out.len(self.bodies.len());
        for (body, desc, state) in &self.bodies {
            out.body(*body);
            for value in [
                desc.capacity,
                desc.heated_area,
                desc.radiating_area,
                desc.emissivity,
                desc.gas_exchange,
                state.temperature,
                state.absorbed,
                state.radiated,
                state.to_gas,
            ] {
                out.f32(value);
            }
        }
    }

    pub(crate) fn load(reader: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        let bad = SnapshotError::Invalid("heat");
        let count = reader.count(44, MAX_HEATED_BODIES)?;
        let mut bodies: Vec<(BodyHandle, HeatDesc, HeatState)> = Vec::with_capacity(count);
        for _ in 0..count {
            let body = reader.handle()?;
            let desc = HeatDesc {
                capacity: reader.f32()?,
                heated_area: reader.f32()?,
                radiating_area: reader.f32()?,
                emissivity: reader.f32()?,
                gas_exchange: reader.f32()?,
            };
            let state = HeatState {
                temperature: reader.f32()?,
                absorbed: reader.f32()?,
                radiated: reader.f32()?,
                to_gas: reader.f32()?,
            };
            let numbers = [state.absorbed, state.radiated, state.to_gas];
            if !desc.is_valid()
                || !(state.temperature.is_finite() && state.temperature >= 0.0)
                || numbers.iter().any(|value| !value.is_finite())
                || bodies.last().is_some_and(|last| last.0 >= body)
            {
                return Err(bad);
            }
            bodies.push((body, desc, state));
        }
        Ok(Self { bodies })
    }
}

impl PhysicsWorld {
    /// Gives `body` a hull that heats and cools from the next step on, at 293 K when new, or
    /// takes it away with `None`. A body that had one keeps its temperature.
    ///
    /// # Errors
    /// Fails on a stale handle, a bad desc or too many heated bodies.
    pub fn set_heat(
        &mut self,
        body: BodyHandle,
        desc: Option<HeatDesc>,
    ) -> Result<(), PhysicsError> {
        self.bodies.dense_index(body)?;
        let list = &mut self.heat.bodies;
        let at = list.binary_search_by_key(&body, |entry| entry.0);
        match (desc, at) {
            (Some(desc), _) if !desc.is_valid() => Err(PhysicsError::InvalidShape),
            (Some(desc), Ok(at)) => {
                list[at].1 = desc;
                Ok(())
            }
            (Some(desc), Err(at)) if list.len() < MAX_HEATED_BODIES => {
                let state = HeatState {
                    temperature: START,
                    absorbed: 0.0,
                    radiated: 0.0,
                    to_gas: 0.0,
                };
                list.insert(at, (body, desc, state));
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

    /// Sets a heated body's hull to `kelvin`.
    ///
    /// # Errors
    /// Fails on a body with no hull from [`Self::set_heat`] or a temperature that is not finite
    /// and positive.
    pub fn set_temperature(&mut self, body: BodyHandle, kelvin: f32) -> Result<(), PhysicsError> {
        if !(kelvin.is_finite() && kelvin > 0.0) {
            return Err(PhysicsError::OutOfRange);
        }
        let list = &mut self.heat.bodies;
        let at = list
            .binary_search_by_key(&body, |entry| entry.0)
            .map_err(|_| PhysicsError::InvalidHandle)?;
        list[at].2.temperature = kelvin;
        Ok(())
    }

    pub fn heat(&self, body: BodyHandle) -> Option<HeatState> {
        let list = &self.heat.bodies;
        let at = list.binary_search_by_key(&body, |entry| entry.0).ok()?;
        Some(list[at].2)
    }

    /// Each hull trades heat with its gas rooms, then takes the air's heating and radiates, over
    /// one step. Both are solved so a light hull or a small room can not swing past where they
    /// settle.
    pub(crate) fn step_heat(&mut self, dt: f32) -> Result<(), PhysicsError> {
        if self.heat.bodies.is_empty() {
            return Ok(());
        }
        let mut list = std::mem::take(&mut self.heat.bodies);
        list.retain(|entry| self.bodies.dense_index(entry.0).is_ok());
        for (body, desc, state) in &mut list {
            let dense = self.bodies.dense_index(*body)?;
            let mut hull = state.temperature;
            let mut to_gas = 0.0;
            if desc.gas_exchange > 0.0 {
                for room in self.gas.rooms_in(*body) {
                    let held = room.mass * crate::gas::CV;
                    if held <= 0.0 {
                        continue;
                    }
                    // the two settle toward where their heat evens out, exactly
                    let even =
                        (desc.capacity * hull + held * room.temperature) / (desc.capacity + held);
                    let keep =
                        (-desc.gas_exchange * (desc.capacity.recip() + held.recip()) * dt).exp();
                    let warmed = even + (room.temperature - even) * keep;
                    to_gas += held * (warmed - room.temperature) / dt;
                    room.temperature = warmed;
                    hull = even + (hull - even) * keep;
                }
            }
            let absorbed = self
                .aero_report(*body)
                .map_or(0.0, |report| report.heating * desc.heated_area);
            let air = self.air_at(self.bodies.frame[dense], self.bodies.position(dense))?;
            let round = if air.density > 0.0 {
                air.temperature
            } else {
                SPACE
            };
            let light = STEFAN_BOLTZMANN * desc.emissivity * desc.radiating_area;
            let back = light * round.powi(4);
            // backward euler: C (t - hull) / dt = absorbed + back - light t^4, by newton's steps.
            // the left less the right grows and bends up, so from either side it closes in
            let per = desc.capacity / dt;
            let mut t = hull;
            for _ in 0..NEWTON_STEPS {
                let miss = per * (t - hull) - absorbed - back + light * t.powi(4);
                let slope = per + 4.0 * light * t.powi(3);
                let next = (t - miss / slope).max(0.0);
                let done = (next - t).abs() <= 1e-6 * t.max(1.0);
                t = next;
                if done {
                    break;
                }
            }
            *state = HeatState {
                temperature: t,
                absorbed,
                radiated: light * t.powi(4) - back,
                to_gas,
            };
        }
        self.heat.bodies = list;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gas::CompartmentDesc;
    use crate::world::testing::no_gravity;
    use crate::{RigidBodyDesc, Shape};
    use algebrix::Vec3;

    fn hull(world: &mut PhysicsWorld) -> BodyHandle {
        world
            .spawn(RigidBodyDesc::dynamic(Shape::cuboid(Vec3::ONE)))
            .unwrap()
    }

    const PLATE: HeatDesc = HeatDesc {
        capacity: 50_000.0,
        heated_area: 0.0,
        radiating_area: 4.0,
        emissivity: 0.9,
        gas_exchange: 0.0,
    };

    #[test]
    fn a_hot_hull_in_space_cools_as_radiation_says() {
        let mut world = no_gravity();
        let body = hull(&mut world);
        world.set_heat(body, Some(PLATE)).unwrap();
        world.set_temperature(body, 1500.0).unwrap();
        // with nothing coming back, 1 / T^3 grows as 3 sigma eps A t / C
        let seconds = 60.0;
        for _ in 0..(seconds * 60.0) as usize {
            world.step_fixed().unwrap();
        }
        let rate =
            3.0 * STEFAN_BOLTZMANN * PLATE.emissivity * PLATE.radiating_area / PLATE.capacity;
        let expected = (1500.0_f32.powi(-3) + rate * seconds).powf(-1.0 / 3.0);
        let now = world.heat(body).unwrap();
        assert!(
            (now.temperature - expected).abs() < 0.005 * expected,
            "{} {expected}",
            now.temperature
        );
        assert!(now.radiated > 0.0);
    }

    #[test]
    fn a_light_hull_with_a_huge_step_settles_and_does_not_swing() {
        let mut world = no_gravity();
        let body = hull(&mut world);
        world
            .set_heat(
                body,
                Some(HeatDesc {
                    capacity: 1.0,
                    radiating_area: 100.0,
                    emissivity: 1.0,
                    ..PLATE
                }),
            )
            .unwrap();
        world.set_temperature(body, 3000.0).unwrap();
        let mut last = 3000.0;
        for _ in 0..30 {
            world.step_fixed().unwrap();
            let t = world.heat(body).unwrap().temperature;
            assert!(t.is_finite() && t > 0.0 && t <= last, "{t} {last}");
            last = t;
        }
    }

    #[test]
    fn hull_and_gas_meet_where_their_heat_evens_out() {
        let mut world = no_gravity();
        let body = hull(&mut world);
        let room = world
            .add_compartment(CompartmentDesc::air(20.0, 101_325.0, Some(body)))
            .unwrap();
        world
            .set_heat(
                body,
                Some(HeatDesc {
                    capacity: 10_000.0,
                    emissivity: 0.0,
                    gas_exchange: 500.0,
                    ..PLATE
                }),
            )
            .unwrap();
        world.set_temperature(body, 400.0).unwrap();
        let gas = *world.compartment(room).unwrap();
        let held = gas.mass * crate::gas::CV;
        let before = 10_000.0 * 400.0 + held * gas.temperature;
        for _ in 0..60 * 120 {
            world.step_fixed().unwrap();
        }
        let (hull, air) = (
            world.heat(body).unwrap().temperature,
            world.compartment(room).unwrap().temperature,
        );
        let after = 10_000.0 * hull + held * air;
        assert!((after - before).abs() < 1e-4 * before, "{after} {before}");
        let even = before / (10_000.0 + held);
        assert!(
            (hull - even).abs() < 0.05 && (air - even).abs() < 0.05,
            "{hull} {air} {even}"
        );
        assert!(air > gas.temperature);
    }

    #[test]
    fn a_fast_hull_in_thick_air_heats_and_a_heavier_one_less() {
        let peak = |capacity: f32| {
            let mut world = no_gravity();
            // sea level air all round, 3 km/s
            world
                .add_atmosphere(crate::AtmosphereDesc::earth(
                    crate::FrameId::ROOT,
                    algebrix::DVec3::new(0.0, -6.371e6, 0.0),
                    6.371e6,
                ))
                .unwrap();
            let body = world
                .spawn(
                    RigidBodyDesc::dynamic(Shape::sphere(0.5))
                        .density(8000.0)
                        .linear_velocity(Vec3::X * 3000.0),
                )
                .unwrap();
            world
                .set_aero(
                    body,
                    Some(crate::AeroDesc {
                        drag: 0.0,
                        area: 0.0,
                        nose: 0.5,
                        surfaces: Vec::new(),
                    }),
                )
                .unwrap();
            world
                .set_heat(
                    body,
                    Some(HeatDesc {
                        capacity,
                        heated_area: 0.8,
                        ..PLATE
                    }),
                )
                .unwrap();
            for _ in 0..30 {
                world.step_fixed().unwrap();
            }
            let state = world.heat(body).unwrap();
            assert!(state.absorbed > 1.0e6, "{state:?}");
            state.temperature
        };
        let (light, heavy) = (peak(50_000.0), peak(500_000.0));
        assert!(
            heavy > START + 5.0 && light > heavy + 50.0,
            "{light} {heavy}"
        );
    }

    #[test]
    fn bad_hulls_are_refused_and_heat_continues_after_a_snapshot() {
        let mut world = no_gravity();
        let body = hull(&mut world);
        for bad in [
            HeatDesc {
                capacity: 0.0,
                ..PLATE
            },
            HeatDesc {
                emissivity: 1.5,
                ..PLATE
            },
            HeatDesc {
                radiating_area: f32::NAN,
                ..PLATE
            },
        ] {
            assert!(world.set_heat(body, Some(bad)).is_err());
        }
        assert!(world.set_temperature(body, 400.0).is_err());
        world.set_heat(body, Some(PLATE)).unwrap();
        world.set_temperature(body, 900.0).unwrap();
        assert!(world.set_temperature(body, -1.0).is_err());
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        let mut copy = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        for _ in 0..30 {
            world.step_fixed().unwrap();
            copy.step_fixed().unwrap();
        }
        assert_eq!(world.heat(body), copy.heat(body));
        world.set_heat(body, None).unwrap();
        assert_eq!(world.heat(body), None);
    }
}
