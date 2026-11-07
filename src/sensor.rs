use algebrix::Quat;

use crate::body::BodyHandle;
use crate::collision::QueryFilter;
use crate::error::PhysicsError;
use crate::position::GridPosition;
use crate::shape::Shape;
use crate::world::PhysicsWorld;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SensorHandle {
    index: u32,
    generation: u32,
}

/// A volume that reports bodies coming and going. Not a body: nothing touches it or is pushed by
/// it.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorDesc {
    pub shape: Shape,
    pub position: GridPosition,
    pub rotation: Quat,
    pub filter: QueryFilter,
}

impl SensorDesc {
    pub fn new(shape: Shape, position: GridPosition) -> Self {
        Self {
            shape,
            position,
            rotation: Quat::IDENTITY,
            filter: QueryFilter::default(),
        }
    }
}

/// A body that started or stopped overlapping a sensor in the last step. A removed body ends
/// its overlap there too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SensorEvent {
    pub sensor: SensorHandle,
    pub body: BodyHandle,
    pub entered: bool,
}

#[derive(Debug, Clone)]
struct Sensor {
    desc: SensorDesc,
    /// Sorted by handle.
    inside: Vec<BodyHandle>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Sensors {
    slots: Vec<Option<Sensor>>,
    generations: Vec<u32>,
    events: Vec<SensorEvent>,
    /// What each slot holds after this step, beside `slots`.
    found: Vec<Vec<BodyHandle>>,
    /// Sweep entries: min x, max x, sensor slot or body dense index, whether it is a sensor.
    entries: Vec<(f64, f64, u32, bool)>,
    /// Entries still open in the sweep: where each ends, and its index.
    active_sensors: Vec<(f64, u32)>,
    active_bodies: Vec<(f64, u32)>,
}

impl Sensors {
    fn get(&self, handle: SensorHandle) -> Option<&Sensor> {
        let index = handle.index as usize;
        if self.generations.get(index) != Some(&handle.generation) {
            return None;
        }
        self.slots.get(index)?.as_ref()
    }

    fn get_mut(&mut self, handle: SensorHandle) -> Option<&mut Sensor> {
        let index = handle.index as usize;
        if self.generations.get(index) != Some(&handle.generation) {
            return None;
        }
        self.slots.get_mut(index)?.as_mut()
    }
}

impl PhysicsWorld {
    /// Its first events come from the next step.
    ///
    /// # Errors
    /// Fails on a bad shape, position, rotation or frame.
    pub fn add_sensor(&mut self, desc: SensorDesc) -> Result<SensorHandle, PhysicsError> {
        let mut found = Vec::new();
        // the same checks a query makes
        self.overlap(
            &desc.shape,
            desc.position,
            desc.rotation,
            &desc.filter,
            &mut found,
        )?;
        let sensors = &mut self.sensors;
        let sensor = Some(Sensor {
            desc,
            inside: Vec::new(),
        });
        let index = match sensors.slots.iter().position(Option::is_none) {
            Some(free) => {
                sensors.slots[free] = sensor;
                free
            }
            None => {
                sensors.slots.push(sensor);
                sensors.generations.push(0);
                sensors.slots.len() - 1
            }
        };
        Ok(SensorHandle {
            index: index as u32,
            generation: sensors.generations[index],
        })
    }

    /// Returns false for a handle that is not there. Gives no exit events.
    pub fn remove_sensor(&mut self, handle: SensorHandle) -> bool {
        if self.sensors.get(handle).is_none() {
            return false;
        }
        let index = handle.index as usize;
        self.sensors.slots[index] = None;
        self.sensors.generations[index] = self.sensors.generations[index].wrapping_add(1);
        true
    }

    /// Moves a sensor. What it holds changes with the next step.
    ///
    /// # Errors
    /// Fails on an unknown handle or non finite input.
    pub fn set_sensor_pose(
        &mut self,
        handle: SensorHandle,
        position: GridPosition,
        rotation: Quat,
    ) -> Result<(), PhysicsError> {
        if !position.local.is_finite() || !rotation.is_finite() {
            return Err(PhysicsError::NonFinite("sensor pose"));
        }
        let sensor = self
            .sensors
            .get_mut(handle)
            .ok_or(PhysicsError::InvalidHandle)?;
        sensor.desc.position = position;
        sensor.desc.rotation = rotation.normalize();
        Ok(())
    }

    /// Bodies overlapping the sensor after the last step, sorted. Empty for an unknown handle.
    pub fn sensor_bodies(&self, handle: SensorHandle) -> &[BodyHandle] {
        self.sensors
            .get(handle)
            .map_or(&[], |sensor| sensor.inside.as_slice())
    }

    /// Enter and leave events of the last step, sorted by sensor, then body.
    pub fn sensor_events(&self) -> &[SensorEvent] {
        &self.sensors.events
    }

    /// What every sensor overlaps now, one sweep along x for all of them: a body is tried only
    /// against sensors its bounds reach, bodies are never tried against each other. The exact
    /// test is the overlap query's.
    fn sweep_sensors(&self, sensors: &mut Sensors) -> Result<(), PhysicsError> {
        let count = sensors.slots.len();
        sensors.found.resize_with(count, Vec::new);
        sensors.found.iter_mut().for_each(Vec::clear);
        let Some(origin) = sensors
            .slots
            .iter()
            .flatten()
            .map(|s| s.desc.position)
            .next()
        else {
            return Ok(());
        };
        for sensor in sensors.slots.iter().flatten() {
            self.check_frame(&sensor.desc.filter)?;
        }
        let assets = &self.bodies.assets;
        let entries = &mut sensors.entries;
        entries.clear();
        for (slot, sensor) in sensors.slots.iter().enumerate() {
            if let Some(sensor) = sensor {
                let x = sensor.desc.position.offset_from(origin).x;
                let reach = f64::from(assets.bounding_radius(&sensor.desc.shape));
                entries.push((x - reach, x + reach, slot as u32, true));
            }
        }
        for dense in 0..self.bodies.len() {
            let x = self.bodies.position(dense).offset_from(origin).x;
            let reach = f64::from(assets.bounding_radius(&self.bodies.shape[dense]));
            entries.push((x - reach, x + reach, dense as u32, false));
        }
        entries
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.3.cmp(&b.3)).then(a.2.cmp(&b.2)));
        let (active_sensors, active_bodies) =
            (&mut sensors.active_sensors, &mut sensors.active_bodies);
        active_sensors.clear();
        active_bodies.clear();
        for &(start, end, index, is_sensor) in entries.iter() {
            // what ended before this starts has met all its partners
            if is_sensor {
                active_bodies.retain(|(until, _)| *until >= start);
                let sensor = sensors.slots[index as usize].as_ref();
                for &(_, dense) in active_bodies.iter() {
                    self.try_pair(sensor, dense as usize, &mut sensors.found[index as usize]);
                }
                active_sensors.push((end, index));
            } else {
                active_sensors.retain(|(until, _)| *until >= start);
                for &(_, slot) in active_sensors.iter() {
                    let sensor = sensors.slots[slot as usize].as_ref();
                    self.try_pair(sensor, index as usize, &mut sensors.found[slot as usize]);
                }
                active_bodies.push((end, index));
            }
        }
        sensors
            .found
            .iter_mut()
            .for_each(|found| found.sort_unstable());
        Ok(())
    }

    /// The overlap query's own test of one sensor against one body.
    fn try_pair(&self, sensor: Option<&Sensor>, dense: usize, found: &mut Vec<BodyHandle>) {
        let Some(sensor) = sensor else {
            return;
        };
        let desc = &sensor.desc;
        if !self.visible(dense, &desc.filter) {
            return;
        }
        let assets = &self.bodies.assets;
        let center = self.bodies.position(dense).offset_from(desc.position);
        let reach = f64::from(assets.bounding_radius(&desc.shape))
            + f64::from(assets.bounding_radius(&self.bodies.shape[dense]));
        if center.length_squared() > reach * reach {
            return;
        }
        if assets.overlapping(
            (&desc.shape, desc.rotation),
            (&self.bodies.shape[dense], self.bodies.rotation[dense]),
            center.as_vec3(),
        ) {
            found.push(self.bodies.handle_at(dense));
        }
    }

    /// Runs after each step, on the bodies where the step left them.
    pub(crate) fn update_sensors(&mut self) -> Result<(), PhysicsError> {
        let mut sensors = std::mem::take(&mut self.sensors);
        sensors.events.clear();
        let result = self.sweep_sensors(&mut sensors);
        if result.is_err() {
            self.sensors = sensors;
            return result;
        }
        for index in 0..sensors.slots.len() {
            let Some(sensor) = sensors.slots[index].as_mut() else {
                continue;
            };
            let handle = SensorHandle {
                index: index as u32,
                generation: sensors.generations[index],
            };
            let (before, now) = (&sensor.inside, &sensors.found[index]);
            let (mut a, mut b) = (0, 0);
            while a < before.len() || b < now.len() {
                let event = match (before.get(a), now.get(b)) {
                    (Some(old), Some(new)) if old == new => {
                        a += 1;
                        b += 1;
                        continue;
                    }
                    (Some(old), Some(new)) if old < new => {
                        a += 1;
                        (*old, false)
                    }
                    (Some(old), None) => {
                        a += 1;
                        (*old, false)
                    }
                    (_, Some(new)) => {
                        b += 1;
                        (*new, true)
                    }
                    (None, None) => break,
                };
                sensors.events.push(SensorEvent {
                    sensor: handle,
                    body: event.0,
                    entered: event.1,
                });
            }
            sensor.inside.clone_from(&sensors.found[index]);
        }
        sensors.events.sort_unstable();
        self.sensors = sensors;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use algebrix::Vec3;

    use super::*;
    use crate::body::RigidBodyDesc;
    use crate::world::PhysicsSettings;

    fn world() -> PhysicsWorld {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let floor = Shape::cuboid(Vec3::new(20.0, 0.5, 20.0));
        world
            .spawn(RigidBodyDesc::fixed(floor).local_position(Vec3::new(0.0, -0.5, 0.0)))
            .unwrap();
        world
    }

    fn at(x: f32, y: f32) -> GridPosition {
        GridPosition::from_local(Vec3::new(x, y, 0.0)).unwrap()
    }

    fn zone(world: &mut PhysicsWorld, x: f32) -> SensorHandle {
        let shape = Shape::cuboid(Vec3::new(1.0, 1.0, 1.0));
        world
            .add_sensor(SensorDesc::new(shape, at(x, 1.2)))
            .unwrap()
    }

    fn ball(world: &mut PhysicsWorld, x: f32, y: f32) -> BodyHandle {
        world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(0.25)).local_position(Vec3::new(x, y, 0.0)))
            .unwrap()
    }

    #[test]
    fn a_falling_ball_enters_and_a_rolling_one_leaves_without_being_pushed() {
        let mut world = world();
        let sensor = zone(&mut world, 0.0);
        let falls = ball(&mut world, 0.0, 6.0);
        let mut seen = Vec::new();
        for _ in 0..120 {
            world.step_fixed().unwrap();
            seen.extend_from_slice(world.sensor_events());
        }
        assert_eq!(
            seen,
            [SensorEvent {
                sensor,
                body: falls,
                entered: true
            }]
        );
        assert!(world.sensor_bodies(sensor).contains(&falls));
        // it fell through the sensor onto the floor, untouched by it
        let state = world.body_state(falls).unwrap();
        assert!((state.position.local.y - 0.25).abs() < 0.02);

        world
            .set_velocity(falls, Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO)
            .unwrap();
        let mut left = Vec::new();
        for _ in 0..60 {
            world.step_fixed().unwrap();
            left.extend_from_slice(world.sensor_events());
        }
        assert_eq!(left.len(), 1);
        assert!(!left[0].entered && left[0].body == falls);
        assert!(world.sensor_bodies(sensor).is_empty());
    }

    #[test]
    fn a_removed_body_leaves_and_events_come_in_order() {
        let mut world = world();
        let (first, second) = (zone(&mut world, 0.0), zone(&mut world, 6.0));
        let bodies = [ball(&mut world, 0.0, 1.0), ball(&mut world, 6.0, 1.0)];
        world.step_fixed().unwrap();
        let entered: Vec<(SensorHandle, BodyHandle)> = world
            .sensor_events()
            .iter()
            .map(|event| (event.sensor, event.body))
            .collect();
        assert_eq!(entered, [(first, bodies[0]), (second, bodies[1])]);
        world.despawn(bodies[1]).unwrap();
        world.step_fixed().unwrap();
        let events = world.sensor_events();
        assert_eq!(events.len(), 1);
        assert_eq!((events[0].sensor, events[0].entered), (second, false));
    }

    #[test]
    fn a_moved_sensor_picks_up_what_is_there_and_old_handles_die() {
        let mut world = world();
        let sensor = zone(&mut world, 0.0);
        let body = ball(&mut world, 10.0, 1.0);
        world.step_fixed().unwrap();
        assert!(world.sensor_events().is_empty());
        world
            .set_sensor_pose(sensor, at(10.0, 1.2), Quat::IDENTITY)
            .unwrap();
        world.step_fixed().unwrap();
        assert_eq!(world.sensor_bodies(sensor), [body]);

        assert!(world.remove_sensor(sensor));
        assert!(!world.remove_sensor(sensor));
        assert!(world.sensor_bodies(sensor).is_empty());
        let again = zone(&mut world, 0.0);
        assert_ne!(again, sensor);
        assert!(
            world
                .set_sensor_pose(sensor, at(0.0, 0.0), Quat::IDENTITY)
                .is_err()
        );
        let nan = GridPosition {
            local: Vec3::new(f32::NAN, 0.0, 0.0),
            ..at(0.0, 0.0)
        };
        assert!(world.set_sensor_pose(again, nan, Quat::IDENTITY).is_err());
        assert!(
            world
                .add_sensor(SensorDesc::new(Shape::sphere(-1.0), at(0.0, 0.0)))
                .is_err()
        );
    }

    #[test]
    fn a_filter_leaves_bodies_out() {
        let mut world = world();
        let shape = Shape::cuboid(Vec3::splat(1.0));
        let mut desc = SensorDesc::new(shape, at(0.0, 1.2));
        desc.filter.mask = 0;
        let sensor = world.add_sensor(desc).unwrap();
        ball(&mut world, 0.0, 1.0);
        world.step_fixed().unwrap();
        assert!(world.sensor_bodies(sensor).is_empty());
        assert!(world.sensor_events().is_empty());
    }

    #[test]
    fn the_one_sweep_finds_what_an_overlap_query_per_sensor_finds() {
        let mut world = world();
        // a scatter of shapes, some well apart, some piled, some outside every sensor
        let mut seed = 7_u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        for _ in 0..300 {
            let at = Vec3::new(
                next() * 80.0 - 40.0,
                next() * 6.0 + 0.3,
                next() * 20.0 - 10.0,
            );
            let shape = match (next() * 3.0) as u32 {
                0 => Shape::sphere(0.2 + next() * 0.6),
                1 => Shape::cuboid(Vec3::splat(0.2 + next() * 0.5)),
                _ => Shape::capsule(0.2, 0.4),
            };
            world
                .spawn(RigidBodyDesc::dynamic(shape).local_position(at))
                .unwrap();
        }
        let mut sensors = Vec::new();
        for _ in 0..40 {
            let at = GridPosition::from_local(Vec3::new(
                next() * 90.0 - 45.0,
                next() * 4.0,
                next() * 20.0 - 10.0,
            ))
            .unwrap();
            let shape = if next() < 0.5 {
                Shape::sphere(1.0 + next() * 3.0)
            } else {
                Shape::cuboid(Vec3::new(1.0 + next() * 4.0, 1.0, 1.0 + next()))
            };
            let mut desc = SensorDesc::new(shape, at);
            desc.rotation = Quat::from_rotation_y(next() * 3.0);
            sensors.push(world.add_sensor(desc).unwrap());
        }
        let mut queried = Vec::new();
        for _ in 0..30 {
            world.step_fixed().unwrap();
            for sensor in &sensors {
                let desc = world.sensors.get(*sensor).unwrap().desc.clone();
                world
                    .overlap(
                        &desc.shape,
                        desc.position,
                        desc.rotation,
                        &desc.filter,
                        &mut queried,
                    )
                    .unwrap();
                assert_eq!(world.sensor_bodies(*sensor), queried.as_slice());
            }
        }
        let held: usize = sensors
            .iter()
            .map(|sensor| world.sensor_bodies(*sensor).len())
            .sum();
        assert!(held > 20, "{held}");
    }
}
