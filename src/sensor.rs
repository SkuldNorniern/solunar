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
    found: Vec<BodyHandle>,
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

    /// Runs after each step, on the bodies where the step left them.
    pub(crate) fn update_sensors(&mut self) -> Result<(), PhysicsError> {
        let mut sensors = std::mem::take(&mut self.sensors);
        sensors.events.clear();
        let mut result = Ok(());
        for index in 0..sensors.slots.len() {
            let Some(sensor) = sensors.slots[index].as_mut() else {
                continue;
            };
            let handle = SensorHandle {
                index: index as u32,
                generation: sensors.generations[index],
            };
            let desc = &sensor.desc;
            if let Err(error) = self.overlap(
                &desc.shape,
                desc.position,
                desc.rotation,
                &desc.filter,
                &mut sensors.found,
            ) {
                result = Err(error);
                break;
            }
            let (before, now) = (&sensor.inside, &sensors.found);
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
            sensor.inside.clone_from(&sensors.found);
        }
        sensors.events.sort_unstable();
        self.sensors = sensors;
        result
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
}
