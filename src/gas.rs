use algebrix::Vec3;

use crate::body::{BodyHandle, BodyKind};
use crate::error::PhysicsError;
use crate::snapshot::{Reader, SnapshotError, Writer};
use crate::world::PhysicsWorld;

/// Most compartments and openings a world holds.
pub const MAX_COMPARTMENTS: usize = 1 << 16;
/// Air's gas constant, J/(kg K), and its ratio of heats.
const GAS: f32 = 287.05;
const HEATS: f32 = 1.4;
/// Share of the gas upstream of an opening that may pass in one gas step, so a small room
/// never swings past its neighbour's pressure.
const MOST_PER_STEP: f32 = 0.1;
/// Most gas steps in one physics step.
const MOST_GAS_STEPS: usize = 256;

/// A sealed volume of air: cubic meters, kilograms, kelvin. Gas left behind in a room it flows
/// out of expands and cools, gas flowing in brings its heat, no heat passes through the walls.
/// In a body, gas leaving it to outside pushes that body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompartmentDesc {
    pub volume: f32,
    pub mass: f32,
    pub temperature: f32,
    pub body: Option<BodyHandle>,
}

impl CompartmentDesc {
    /// `volume` cubic meters of air at `pressure` pascals and 293 K.
    pub fn air(volume: f32, pressure: f32, body: Option<BodyHandle>) -> Self {
        let temperature = 293.15;
        Self {
            volume,
            mass: pressure * volume / (GAS * temperature),
            temperature,
            body,
        }
    }

    pub fn pressure(&self) -> f32 {
        self.mass * GAS * self.temperature / self.volume
    }

    fn is_valid(&self) -> bool {
        let positive = |value: f32| value.is_finite() && value > 0.0;
        positive(self.volume)
            && self.mass.is_finite()
            && self.mass >= 0.0
            && positive(self.temperature)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CompartmentId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpeningId(u32);

/// A way between two compartments, or from one to outside: a door, a vent, a breach. `area`
/// square meters, its `discharge` coefficient (0.6 for a sharp hole, 1 for a smooth nozzle),
/// where it is and which way it faces out of `from`, in the body's axes. A closed one passes
/// nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpeningDesc {
    pub from: CompartmentId,
    pub to: Option<CompartmentId>,
    pub area: f32,
    pub discharge: f32,
    pub position: Vec3,
    pub normal: Vec3,
    pub open: bool,
}

impl OpeningDesc {
    fn is_valid(&self) -> bool {
        self.area.is_finite()
            && self.area >= 0.0
            && (0.0..=1.0).contains(&self.discharge)
            && self.position.is_finite()
            && self.normal.is_finite()
            && self.normal.length() > 1e-6
            && self.to != Some(self.from)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Gas {
    compartments: Vec<Option<CompartmentDesc>>,
    openings: Vec<Option<(OpeningDesc, f32)>>,
}

impl Gas {
    fn is_empty(&self) -> bool {
        self.openings.iter().all(Option::is_none)
    }

    pub(crate) fn save(&self, out: &mut Writer) {
        out.len(self.compartments.len());
        for slot in &self.compartments {
            let Some(room) = slot else {
                out.u8(0);
                continue;
            };
            out.u8(1);
            out.f32(room.volume);
            out.f32(room.mass);
            out.f32(room.temperature);
            match room.body {
                None => out.u8(0),
                Some(body) => {
                    out.u8(1);
                    out.body(body);
                }
            }
        }
        out.len(self.openings.len());
        for slot in &self.openings {
            let Some((way, flow)) = slot else {
                out.u8(0);
                continue;
            };
            out.u8(1);
            out.u32(way.from.0);
            out.u32(way.to.map_or(u32::MAX, |to| to.0));
            out.f32(way.area);
            out.f32(way.discharge);
            out.vec3(way.position);
            out.vec3(way.normal);
            out.u8(u8::from(way.open));
            out.f32(*flow);
        }
    }

    pub(crate) fn load(reader: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        let bad = SnapshotError::Invalid("gas");
        let mut gas = Self::default();
        let count = reader.count(1, MAX_COMPARTMENTS)?;
        for _ in 0..count {
            let slot = match reader.u8()? {
                0 => None,
                1 => {
                    let (volume, mass, temperature) = (reader.f32()?, reader.f32()?, reader.f32()?);
                    let body = match reader.u8()? {
                        0 => None,
                        1 => Some(reader.handle()?),
                        _ => return Err(bad),
                    };
                    let room = CompartmentDesc {
                        volume,
                        mass,
                        temperature,
                        body,
                    };
                    if !room.is_valid() {
                        return Err(bad);
                    }
                    Some(room)
                }
                _ => return Err(bad),
            };
            gas.compartments.push(slot);
        }
        let count = reader.count(1, MAX_COMPARTMENTS)?;
        let live = |id: u32, gas: &Self| {
            gas.compartments
                .get(id as usize)
                .is_some_and(Option::is_some)
        };
        for _ in 0..count {
            let slot = match reader.u8()? {
                0 => None,
                1 => {
                    let from = reader.u32()?;
                    let to = reader.u32()?;
                    let way = OpeningDesc {
                        from: CompartmentId(from),
                        to: (to != u32::MAX).then_some(CompartmentId(to)),
                        area: reader.f32()?,
                        discharge: reader.f32()?,
                        position: reader.vec3()?,
                        normal: reader.vec3()?,
                        open: match reader.u8()? {
                            0 => false,
                            1 => true,
                            _ => return Err(bad),
                        },
                    };
                    let flow = reader.f32()?;
                    let ends = live(from, &gas) && way.to.is_none_or(|to| live(to.0, &gas));
                    if !way.is_valid() || !ends || !flow.is_finite() {
                        return Err(bad);
                    }
                    Some((way, flow))
                }
                _ => return Err(bad),
            };
            gas.openings.push(slot);
        }
        Ok(gas)
    }
}

/// One opening's flow in a gas step: kg/s signed from its `from` side, the speed the jet
/// leaves at, its exit pressure over outside, and the most mass it may carry in the step.
#[derive(Debug, Clone, Copy)]
struct Flow {
    opening: usize,
    from: usize,
    to: Option<usize>,
    rate: f32,
    speed: f32,
    over: f32,
    most: f32,
}

/// Kilograms a second through `area` with `discharge` from `up` to `down` pascals, the air
/// upstream at `temperature`: choked past the critical pressure ratio, the subsonic orifice
/// law below it. With the speed it leaves at and the pressure it leaves with.
fn orifice(area: f32, discharge: f32, (up, down): (f32, f32), temperature: f32) -> (f32, f32, f32) {
    if up <= down || up <= 0.0 {
        return (0.0, 0.0, down);
    }
    let critical = (2.0 / (HEATS + 1.0)).powf(HEATS / (HEATS - 1.0));
    let ratio = (down / up).max(critical);
    let flow = discharge
        * area
        * up
        * ((2.0 * HEATS / ((HEATS - 1.0) * GAS * temperature))
            * (ratio.powf(2.0 / HEATS) - ratio.powf((HEATS + 1.0) / HEATS)))
        .max(0.0)
        .sqrt();
    let speed = (2.0 * HEATS / (HEATS - 1.0)
        * GAS
        * temperature
        * (1.0 - ratio.powf((HEATS - 1.0) / HEATS)))
    .max(0.0)
    .sqrt();
    (flow, speed, up * ratio)
}

/// `amount` kilograms leave `room`. What stays behind expands with no heat in or out, so its
/// temperature falls as the mass to the power gamma - 1, taken exactly so a long step does not
/// cool it less than many short ones.
fn give(room: &mut CompartmentDesc, amount: f32) {
    let left = room.mass - amount;
    if left > 0.0 && room.mass > 0.0 {
        room.temperature *= (left / room.mass).powf(HEATS - 1.0);
    }
    room.mass = left.max(0.0);
}

/// `amount` kilograms at `warmth` come into `room`, bringing their enthalpy: a room filled from
/// empty ends gamma times warmer than the gas that filled it.
fn take(room: &mut CompartmentDesc, amount: f32, warmth: f32) {
    let mass = room.mass + amount;
    if mass > 0.0 {
        room.temperature = (room.mass * room.temperature + HEATS * amount * warmth) / mass;
    }
    room.mass = mass;
}

impl PhysicsWorld {
    /// # Errors
    /// Fails on values out of range, a stale body or too many compartments.
    pub fn add_compartment(
        &mut self,
        desc: CompartmentDesc,
    ) -> Result<CompartmentId, PhysicsError> {
        if !desc.is_valid() {
            return Err(PhysicsError::InvalidSettings);
        }
        if let Some(body) = desc.body {
            self.bodies.dense_index(body)?;
        }
        let slots = &mut self.gas.compartments;
        if slots.len() >= MAX_COMPARTMENTS {
            return Err(PhysicsError::OutOfRange);
        }
        slots.push(Some(desc));
        Ok(CompartmentId(slots.len() as u32 - 1))
    }

    /// The compartment as it is now.
    pub fn compartment(&self, id: CompartmentId) -> Option<&CompartmentDesc> {
        self.gas.compartments.get(id.0 as usize)?.as_ref()
    }

    /// # Errors
    /// Fails on an unknown compartment, values out of range or too many openings.
    pub fn add_opening(&mut self, desc: OpeningDesc) -> Result<OpeningId, PhysicsError> {
        let known = |id: CompartmentId| self.compartment(id).is_some();
        if !desc.is_valid() || !known(desc.from) || desc.to.is_some_and(|to| !known(to)) {
            return Err(PhysicsError::InvalidSettings);
        }
        let slots = &mut self.gas.openings;
        if slots.len() >= MAX_COMPARTMENTS {
            return Err(PhysicsError::OutOfRange);
        }
        slots.push(Some((desc, 0.0)));
        Ok(OpeningId(slots.len() as u32 - 1))
    }

    /// Opens or closes `id`, and gives it a new area: a door swinging, a breach growing.
    ///
    /// # Errors
    /// Fails on an unknown opening or a bad area.
    pub fn set_opening(
        &mut self,
        id: OpeningId,
        open: bool,
        area: f32,
    ) -> Result<(), PhysicsError> {
        if !area.is_finite() || area < 0.0 {
            return Err(PhysicsError::InvalidSettings);
        }
        let (way, _) = self
            .gas
            .openings
            .get_mut(id.0 as usize)
            .and_then(Option::as_mut)
            .ok_or(PhysicsError::OutOfRange)?;
        way.open = open;
        way.area = area;
        Ok(())
    }

    /// Kilograms a second through `id` over the last step, from its `from` side.
    pub fn opening_flow(&self, id: OpeningId) -> Option<f32> {
        self.gas
            .openings
            .get(id.0 as usize)?
            .as_ref()
            .map(|slot| slot.1)
    }

    /// The gas over `dt`, in steps short enough that no compartment loses more than a tenth of
    /// its gas in one. Gas leaving a body to outside pushes it, as force for this step.
    pub(crate) fn step_gas(&mut self, dt: f32) -> Result<(), PhysicsError> {
        if self.gas.is_empty() {
            return Ok(());
        }
        // what lies outside each opening to outside: the air there, if any
        let outside: Vec<f32> = self
            .gas
            .openings
            .iter()
            .map(|slot| {
                let Some((way, _)) = slot else {
                    return 0.0;
                };
                if way.to.is_some() {
                    return 0.0;
                }
                let Some(body) =
                    self.gas.compartments[way.from.0 as usize].and_then(|room| room.body)
                else {
                    return 0.0;
                };
                let Ok(dense) = self.bodies.dense_index(body) else {
                    return 0.0;
                };
                self.air_at(self.bodies.frame[dense], self.bodies.position(dense))
                    .map_or(0.0, |air| air.pressure)
            })
            .collect();
        let gas = &mut self.gas;
        let mut pushes: Vec<(BodyHandle, Vec3, Vec3)> = Vec::new();
        let mut moved = vec![0.0_f32; gas.openings.len()];
        let mut left = dt;
        let mut steps = 0;
        while left > 0.0 && steps < MOST_GAS_STEPS {
            steps += 1;
            // each open way's flow now, and the longest step none overshoots in
            let mut flows: Vec<Flow> = Vec::new();
            let mut step = left;
            for (index, slot) in gas.openings.iter().enumerate() {
                let Some((way, _)) = slot else {
                    continue;
                };
                if !way.open || way.area <= 0.0 {
                    continue;
                }
                let (Some(here), there) = (
                    gas.compartments[way.from.0 as usize],
                    way.to.map(|to| gas.compartments[to.0 as usize]),
                ) else {
                    continue;
                };
                let far = match there {
                    Some(Some(room)) => room.pressure(),
                    Some(None) => continue,
                    None => outside[index],
                };
                let near = here.pressure();
                // the flow goes from the higher side, signed from `from`
                let (sign, up, down, temperature, mass) = if near >= far {
                    (1.0, near, far, here.temperature, here.mass)
                } else {
                    match there {
                        Some(Some(room)) => (-1.0, far, near, room.temperature, room.mass),
                        _ => continue,
                    }
                };
                let (flow, speed, exit) = orifice(way.area, way.discharge, (up, down), temperature);
                if flow <= 0.0 {
                    continue;
                }
                step = step.min(MOST_PER_STEP * mass / flow);
                // between two rooms, never more than half of what would even them out: near
                // even the flow goes as the root of the difference and a step swung past it
                let most = match there {
                    Some(Some(room)) => {
                        let give = |side: &CompartmentDesc| GAS * side.temperature / side.volume;
                        0.5 * (up - down) / (give(&here) + give(&room))
                    }
                    _ => f32::INFINITY,
                };
                flows.push(Flow {
                    opening: index,
                    from: way.from.0 as usize,
                    to: way.to.map(|to| to.0 as usize),
                    rate: sign * flow,
                    speed,
                    over: exit - down,
                    most,
                });
            }
            let step = step.max(dt / MOST_GAS_STEPS as f32).min(left);
            for &Flow {
                opening: index,
                from,
                to,
                rate: flow,
                speed,
                over,
                most,
            } in &flows
            {
                let carried = (flow * step).clamp(-most, most);
                let (giver, taker, amount) = if carried >= 0.0 {
                    (Some(from), to, carried)
                } else {
                    (to, Some(from), -carried)
                };
                let warmth = giver
                    .and_then(|giver| gas.compartments[giver])
                    .map(|room| room.temperature);
                if let Some(room) = giver.and_then(|giver| gas.compartments[giver].as_mut()) {
                    give(room, amount);
                }
                if let (Some(room), Some(warmth)) = (
                    taker
                        .filter(|_| to.is_some())
                        .and_then(|taker| gas.compartments[taker].as_mut()),
                    warmth,
                ) {
                    take(room, amount, warmth);
                }
                if to.is_some() {
                    moved[index] += carried;
                    continue;
                }
                moved[index] += carried;
                // out to outside: the jet pushes the body back, against the way it faces
                let Some((way, _)) = gas.openings[index] else {
                    continue;
                };
                if let Some(body) = gas.compartments[from].and_then(|room| room.body) {
                    let thrust = flow.abs() * speed + over * way.area;
                    pushes.push((
                        body,
                        -way.normal.normalize() * (thrust * step),
                        way.position,
                    ));
                }
            }
            left -= step;
            if flows.is_empty() {
                break;
            }
        }
        for (slot, carried) in gas.openings.iter_mut().zip(&moved) {
            if let Some((_, flow)) = slot.as_mut() {
                *flow = carried / dt;
            }
        }
        for (body, impulse, at) in pushes {
            let Ok(dense) = self.bodies.dense_index(body) else {
                continue;
            };
            if self.bodies.kind[dense] != BodyKind::Dynamic {
                continue;
            }
            let rotation = self.bodies.rotation[dense];
            let force = rotation.rotate_vec3(impulse) / dt;
            self.bodies.force[dense] += force;
            self.bodies.torque[dense] += rotation.rotate_vec3(at).cross(force);
            self.bodies.wake(dense);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::testing::no_gravity;
    use crate::{RigidBodyDesc, Shape};

    #[test]
    fn two_rooms_through_a_door_end_at_one_pressure_and_keep_their_air() {
        let mut world = no_gravity();
        let full = world
            .add_compartment(CompartmentDesc::air(30.0, 101_325.0, None))
            .unwrap();
        let empty = world
            .add_compartment(CompartmentDesc::air(10.0, 20_000.0, None))
            .unwrap();
        let door = world
            .add_opening(OpeningDesc {
                from: full,
                to: Some(empty),
                area: 0.5,
                discharge: 0.6,
                position: Vec3::ZERO,
                normal: Vec3::X,
                open: true,
            })
            .unwrap();
        let mass = |world: &PhysicsWorld| {
            world.compartment(full).unwrap().mass + world.compartment(empty).unwrap().mass
        };
        let before = mass(&world);
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        let (a, b) = (
            world.compartment(full).unwrap().pressure(),
            world.compartment(empty).unwrap().pressure(),
        );
        // the energy, p V / (gamma - 1), is kept, so the end pressure is the volume weighted one.
        // the room that gave is colder now and the one that took warmer
        let ends = (30.0 * 101_325.0 + 10.0 * 20_000.0) / 40.0;
        let (giver, taker) = (
            world.compartment(full).unwrap().temperature,
            world.compartment(empty).unwrap().temperature,
        );
        assert!(giver < 293.0 && taker > 294.0, "{giver} {taker}");
        assert!(
            (a - ends).abs() < 0.01 * ends && (b - ends).abs() < 0.01 * ends,
            "{a} {b} {ends}"
        );
        assert!((mass(&world) - before).abs() < 1e-3 * before);
        // closed, nothing more goes
        world.set_opening(door, false, 0.5).unwrap();
        world.step_fixed().unwrap();
        assert_eq!(world.opening_flow(door), Some(0.0));
    }

    #[test]
    fn a_room_filled_from_almost_nothing_ends_gamma_times_warmer() {
        let mut world = no_gravity();
        // a big tank, so its air barely changes while the small room fills
        let tank = world
            .add_compartment(CompartmentDesc::air(1.0e5, 101_325.0, None))
            .unwrap();
        let room = world
            .add_compartment(CompartmentDesc::air(1.0, 10.0, None))
            .unwrap();
        world
            .add_opening(OpeningDesc {
                from: tank,
                to: Some(room),
                area: 0.01,
                discharge: 1.0,
                position: Vec3::ZERO,
                normal: Vec3::X,
                open: true,
            })
            .unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        let filled = world.compartment(room).unwrap();
        assert!((filled.pressure() - 101_325.0).abs() < 0.01 * 101_325.0);
        let hot = HEATS * 293.15;
        assert!(
            (filled.temperature - hot).abs() < 0.01 * hot,
            "{} {hot}",
            filled.temperature
        );
    }

    #[test]
    fn a_breach_to_vacuum_blows_down_as_choked_flow_says() {
        let mut world = no_gravity();
        let room = world
            .add_compartment(CompartmentDesc::air(10.0, 101_325.0, None))
            .unwrap();
        world
            .add_opening(OpeningDesc {
                from: room,
                to: None,
                area: 0.01,
                discharge: 1.0,
                position: Vec3::ZERO,
                normal: Vec3::X,
                open: true,
            })
            .unwrap();
        // choked all the way to vacuum, the gas left expanding with no heat in:
        // (m / m0)^(-(gamma - 1) / 2) = 1 + (gamma - 1) / 2 t / tau,
        // tau = V / (A sqrt(gamma R T0)) / (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1))),
        // p / p0 the mass share to the gamma, T / T0 to gamma - 1
        let temperature = 293.15_f32;
        let choke = (2.0 / (HEATS + 1.0)).powf((HEATS + 1.0) / (2.0 * (HEATS - 1.0)));
        let tau = 10.0 / (0.01 * (HEATS * GAS * temperature).sqrt() * choke);
        let start = world.compartment(room).unwrap().pressure();
        let steps = (tau * 60.0) as usize;
        for _ in 0..steps {
            world.step_fixed().unwrap();
        }
        let after = *world.compartment(room).unwrap();
        let t = steps as f32 / 60.0;
        let share = (1.0 + (HEATS - 1.0) / 2.0 * t / tau).powf(-2.0 / (HEATS - 1.0));
        let expected = start * share.powf(HEATS);
        assert!(
            (after.pressure() - expected).abs() < 0.03 * expected,
            "{} {expected} tau {tau}",
            after.pressure()
        );
        let cold = temperature * share.powf(HEATS - 1.0);
        assert!(
            (after.temperature - cold).abs() < 0.01 * cold,
            "{} {cold}",
            after.temperature
        );
    }

    #[test]
    fn gas_out_of_a_hull_pushes_it_by_what_it_carries_away() {
        let mut world = no_gravity();
        let mut desc = RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(1.0))).density(100.0);
        desc.can_sleep = false;
        let ship = world.spawn(desc).unwrap();
        let tank = world
            .add_compartment(CompartmentDesc::air(5.0, 500_000.0, Some(ship)))
            .unwrap();
        let nozzle = world
            .add_opening(OpeningDesc {
                from: tank,
                to: None,
                area: 0.002,
                discharge: 1.0,
                position: Vec3::new(-1.0, 0.0, 0.0),
                normal: Vec3::NEG_X,
                open: true,
            })
            .unwrap();
        let mut carried = 0.0;
        for _ in 0..60 {
            world.step_fixed().unwrap();
            carried += world.opening_flow(nozzle).unwrap() / 60.0;
        }
        // jetting out the back, the ship goes forward; the push is at least the gas's momentum
        // at sound speed, and more by the pressure at the exit
        let state = world.body_state(ship).unwrap();
        let momentum = state.linear_velocity.x * 800.0;
        let sound = (HEATS * GAS * 293.15 * 2.0 / (HEATS + 1.0)).sqrt();
        assert!(state.linear_velocity.x > 0.0);
        assert!(
            momentum > carried * sound * 0.99 && momentum < carried * sound * 2.5,
            "{momentum} {carried}"
        );
        assert!(state.angular_velocity.length() < 1e-4);
    }

    #[test]
    fn gas_continues_exactly_after_a_snapshot_and_bad_gas_is_refused() {
        let mut world = no_gravity();
        let ship = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(1.0)))
            .unwrap();
        let a = world
            .add_compartment(CompartmentDesc::air(4.0, 90_000.0, Some(ship)))
            .unwrap();
        let b = world
            .add_compartment(CompartmentDesc::air(2.0, 1_000.0, Some(ship)))
            .unwrap();
        let way = |from, to| OpeningDesc {
            from,
            to,
            area: 0.01,
            discharge: 0.6,
            position: Vec3::ZERO,
            normal: Vec3::Y,
            open: true,
        };
        world.add_opening(way(a, Some(b))).unwrap();
        world.add_opening(way(b, None)).unwrap();
        for _ in 0..10 {
            world.step_fixed().unwrap();
        }
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        for _ in 0..30 {
            world.step_fixed().unwrap();
            loaded.step_fixed().unwrap();
        }
        assert_eq!(loaded.save_snapshot(), world.save_snapshot());
        assert!(
            world
                .add_compartment(CompartmentDesc {
                    volume: 0.0,
                    ..CompartmentDesc::air(1.0, 1.0, None)
                })
                .is_err()
        );
        assert!(world.add_opening(way(a, Some(a))).is_err());
        assert!(
            world
                .add_opening(OpeningDesc {
                    discharge: 2.0,
                    ..way(a, None)
                })
                .is_err()
        );
        assert!(world.add_opening(way(CompartmentId(99), None)).is_err());
    }
}
