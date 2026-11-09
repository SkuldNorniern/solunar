use algebrix::{DVec3, Quat, Vec3};

use crate::{BodyHandle, BodyKind, GridPosition, PhysicsError, PhysicsWorld, QueryFilter, Ray};

/// One wheel, in the chassis axes. The chassis drives along +Z with +Y up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelDesc {
    /// Where the suspension starts, at full extension it holds the wheel center `rest_length`
    /// below this.
    pub mount: Vec3,
    pub radius: f32,
    pub rest_length: f32,
    /// Newtons per meter of compression.
    pub stiffness: f32,
    /// Newtons per meter per second.
    pub damping: f32,
    pub friction: f32,
    pub driven: bool,
    pub steered: bool,
}

impl WheelDesc {
    pub fn at(mount: Vec3) -> Self {
        Self {
            mount,
            radius: 0.35,
            rest_length: 0.3,
            stiffness: 30_000.0,
            damping: 3_000.0,
            friction: 1.0,
            driven: false,
            steered: false,
        }
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        let positive = |value: f32| value.is_finite() && value > 0.0;
        let good = |value: f32| value.is_finite() && value >= 0.0;
        if self.mount.is_finite()
            && positive(self.radius)
            && good(self.rest_length)
            && positive(self.stiffness)
            && good(self.damping)
            && good(self.friction)
        {
            Ok(())
        } else {
            Err(PhysicsError::InvalidShape)
        }
    }
}

/// What the driver asks for this step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Controls {
    /// Torque on each driven wheel, newton meters. Negative backs up.
    pub drive: f32,
    /// Most braking torque on each wheel, newton meters.
    pub brake: f32,
    /// Steered wheels turn this far about up, radians, positive to the left.
    pub steer: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelContact {
    pub body: BodyHandle,
    pub point: GridPosition,
    pub normal: Vec3,
    /// Suspension force pushing the chassis up, newtons.
    pub load: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WheelState {
    /// Suspension travel from full extension, meters.
    pub compression: f32,
    /// Spin about the axle, radians, for drawing.
    pub angle: f32,
    pub spin: f32,
    pub steer: f32,
    pub contact: Option<WheelContact>,
}

/// Wheels on a dynamic chassis. Each wheel is a ray down from its mount. `drive` turns the
/// wheels into forces on the chassis before the world steps.
#[derive(Debug, Clone, PartialEq)]
pub struct Vehicle {
    chassis: BodyHandle,
    wheels: Vec<WheelDesc>,
    states: Vec<WheelState>,
    filter: QueryFilter,
}

struct Grip {
    offset: Vec3,
    forward: Vec3,
    side: Vec3,
    load: f32,
}

impl Vehicle {
    /// # Errors
    /// Fails on a stale or non dynamic chassis, no wheels or a bad wheel.
    pub fn new(
        world: &PhysicsWorld,
        chassis: BodyHandle,
        wheels: &[WheelDesc],
    ) -> Result<Self, PhysicsError> {
        if world.body_state(chassis)?.kind != BodyKind::Dynamic {
            return Err(PhysicsError::NotDynamic);
        }
        if wheels.is_empty() {
            return Err(PhysicsError::InvalidShape);
        }
        for wheel in wheels {
            wheel.validate()?;
        }
        Ok(Self {
            chassis,
            wheels: wheels.to_vec(),
            states: vec![WheelState::default(); wheels.len()],
            filter: QueryFilter {
                exclude: Some(chassis),
                ..QueryFilter::default()
            },
        })
    }

    pub fn chassis(&self) -> BodyHandle {
        self.chassis
    }

    pub fn wheels(&self) -> &[WheelDesc] {
        &self.wheels
    }

    pub fn states(&self) -> &[WheelState] {
        &self.states
    }

    /// Only bodies in `mask` hold the wheels up.
    pub fn set_mask(&mut self, mask: u32) {
        self.filter.mask = mask;
    }

    /// Where wheel `index` is drawn: its center and turn. `None` past the last wheel or for a
    /// stale chassis.
    pub fn wheel_pose(&self, world: &PhysicsWorld, index: usize) -> Option<(GridPosition, Quat)> {
        let wheel = self.wheels.get(index)?;
        let state = self.states.get(index)?;
        let body = world.body_state(self.chassis).ok()?;
        let drop = wheel.rest_length - state.compression;
        let center = body.rotation.rotate_vec3(wheel.mount - Vec3::Y * drop);
        let at = body.position.shifted(DVec3::from_vec3(center))?;
        let turn =
            body.rotation * Quat::from_rotation_y(state.steer) * Quat::from_rotation_x(state.angle);
        Some((at, turn))
    }

    /// Casts every wheel, then puts suspension, drive, brake and side grip on the chassis as
    /// forces for the next step. Dynamic ground gets the opposite. Call once before each step.
    ///
    /// # Errors
    /// Fails on non finite controls or a stale chassis.
    pub fn drive(
        &mut self,
        world: &mut PhysicsWorld,
        controls: Controls,
    ) -> Result<(), PhysicsError> {
        if !(controls.drive.is_finite() && controls.brake.is_finite() && controls.steer.is_finite())
        {
            return Err(PhysicsError::NonFinite("controls"));
        }
        let dt = world.settings().fixed_timestep;
        // grip goes in with the first substep and gravity over all of them, so aim the mean slip
        // over the coming step at zero: gravity counts (n + 1) / 2n of a step
        let parts = world.settings().substeps.max(1) as f32;
        let body = world.body_state(self.chassis)?;
        // the chassis' own frame: on a planet that is the ground frame, its pull and turning in
        let felt = world.gravity_at(body.frame, body.position, body.linear_velocity)?;
        let gravity = felt * ((parts + 1.0) / (2.0 * parts));
        let filter = QueryFilter {
            frame: body.frame,
            ..self.filter
        };
        let up = body.rotation.rotate_vec3(Vec3::Y);
        let point_velocity =
            |offset: Vec3| body.linear_velocity + body.angular_velocity.cross(offset);

        let mut grips: Vec<Option<Grip>> = Vec::with_capacity(self.wheels.len());
        for (wheel, state) in self.wheels.iter().zip(&mut self.states) {
            state.steer = if wheel.steered { controls.steer } else { 0.0 };
            let mount = body.rotation.rotate_vec3(wheel.mount);
            let origin = body
                .position
                .shifted(DVec3::from_vec3(mount))
                .ok_or(PhysicsError::OutOfRange)?;
            let reach = wheel.rest_length + wheel.radius;
            let hit = world.raycast(
                &Ray {
                    origin,
                    direction: -up,
                    max_distance: reach,
                },
                &filter,
            )?;
            let Some(hit) = hit else {
                state.compression = 0.0;
                state.contact = None;
                grips.push(None);
                continue;
            };
            let compression = reach - hit.distance;
            let offset = hit.point.offset_from(body.position).as_vec3();
            let ground = world.body_state(hit.body)?;
            let ground_offset = hit.point.offset_from(ground.position).as_vec3();
            let ground_velocity =
                ground.linear_velocity + ground.angular_velocity.cross(ground_offset);
            // forces go in at the start of a step, so a held body reads a sawtooth speed at the
            // end of one. the change in compression does not have it.
            let closing = if state.contact.is_some() {
                (compression - state.compression) / dt
            } else {
                -(point_velocity(offset) - ground_velocity).dot(up)
            };
            let load = (wheel.stiffness * compression + wheel.damping * closing).max(0.0);
            let heading = body.rotation * Quat::from_rotation_y(state.steer);
            let along = heading.rotate_vec3(Vec3::Z);
            let flat = along - hit.normal * along.dot(hit.normal);
            let forward = if flat.length_squared() > 1e-8 {
                flat.normalize()
            } else {
                along
            };
            state.compression = compression;
            state.contact = Some(WheelContact {
                body: hit.body,
                point: hit.point,
                normal: hit.normal,
                load,
            });
            grips.push(Some(Grip {
                offset,
                forward,
                side: hit.normal.cross(forward),
                load,
            }));
        }

        let touching = grips.iter().flatten().count().max(1) as f32;
        let mut force = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        let mut reactions = Vec::new();
        for ((wheel, state), grip) in self.wheels.iter().zip(&mut self.states).zip(&grips) {
            let Some(grip) = grip else {
                state.spin = if wheel.driven {
                    state.spin + controls.drive.signum() * dt
                } else {
                    state.spin * 0.99
                };
                state.angle = (state.angle + state.spin * dt) % std::f32::consts::TAU;
                continue;
            };
            let Some(contact) = state.contact else {
                continue;
            };
            let ground = world.body_state(contact.body)?;
            let ground_offset = contact.point.offset_from(ground.position).as_vec3();
            let relative = point_velocity(grip.offset)
                - (ground.linear_velocity + ground.angular_velocity.cross(ground_offset));
            let rolling = relative.dot(grip.forward);
            let sliding = rolling_on(relative, gravity, grip.side, dt);
            let rolling_next = rolling_on(relative, gravity, grip.forward, dt);
            let mass_along = world.mass_at(self.chassis, Vec3::ZERO, grip.forward)?;
            let mass_side = world.mass_at(self.chassis, Vec3::ZERO, grip.side)?;

            let mut push = 0.0;
            if wheel.driven {
                push += controls.drive / wheel.radius;
            }
            let brake = controls.brake.max(0.0) / wheel.radius;
            if brake > 0.0 {
                // a brake holds against rolling, it never pushes the other way
                let stop = (rolling_next * mass_along / dt / touching).abs();
                push -= rolling_next.signum() * brake.min(stop);
            }
            let side = -sliding * mass_side / dt / touching;
            let mut tire = grip.forward * push + grip.side * side;
            let limit = wheel.friction * grip.load;
            let length = tire.length();
            if length > limit {
                tire *= limit / length;
            }
            let total = contact.normal * grip.load + tire;
            force += total;
            // tire forces act at the height of the center of mass, low contacts tipped it into a
            // roll the springs had to hold every step and it crept downhill
            let level = grip.offset - up * grip.offset.dot(up);
            torque += grip.offset.cross(contact.normal * grip.load) + level.cross(tire);
            if ground.kind == BodyKind::Dynamic {
                reactions.push((contact.body, -total, ground_offset));
            }
            state.spin = rolling / wheel.radius;
            state.angle = (state.angle + state.spin * dt) % std::f32::consts::TAU;
        }
        world.apply_force(self.chassis, force)?;
        world.apply_torque(self.chassis, torque)?;
        for (body, force, offset) in reactions {
            world.apply_force(body, force)?;
            world.apply_torque(body, offset.cross(force))?;
        }
        Ok(())
    }
}

fn rolling_on(relative: Vec3, gravity: Vec3, direction: Vec3, dt: f32) -> f32 {
    (relative + gravity * dt).dot(direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhysicsSettings, RigidBodyDesc, Shape};

    const CHASSIS_HALF: Vec3 = Vec3::new(0.9, 0.3, 2.0);
    const DENSITY: f32 = 1_000.0 / (1.8 * 0.6 * 4.0);

    fn world() -> PhysicsWorld {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(200.0, 1.0, 200.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
        world
    }

    fn wheels() -> Vec<WheelDesc> {
        let mut out = Vec::new();
        for z in [1.4_f32, -1.4] {
            for x in [-0.8_f32, 0.8] {
                out.push(WheelDesc {
                    driven: z < 0.0,
                    steered: z > 0.0,
                    ..WheelDesc::at(Vec3::new(x, -0.3, z))
                });
            }
        }
        out
    }

    /// A 1000 kg car with its chassis center `height` up.
    fn car(world: &mut PhysicsWorld, height: f32, rotation: Quat) -> Vehicle {
        let chassis = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(CHASSIS_HALF))
                    .density(DENSITY)
                    .rotation(rotation)
                    .local_position(Vec3::new(0.0, height, 0.0)),
            )
            .unwrap();
        Vehicle::new(world, chassis, &wheels()).unwrap()
    }

    fn run(world: &mut PhysicsWorld, vehicle: &mut Vehicle, controls: Controls, steps: usize) {
        for _ in 0..steps {
            vehicle.drive(world, controls).unwrap();
            world.step_fixed().unwrap();
        }
    }

    fn chassis(world: &PhysicsWorld, vehicle: &Vehicle) -> crate::BodyState {
        world.body_state(vehicle.chassis()).unwrap()
    }

    #[test]
    fn settles_on_its_springs_at_the_right_height() {
        let mut world = world();
        let mut vehicle = car(&mut world, 1.2, Quat::IDENTITY);
        run(&mut world, &mut vehicle, Controls::default(), 300);
        let state = chassis(&world, &vehicle);
        let sag = 1_000.0 * 9.81 / (4.0 * 30_000.0);
        // wheel bottom on the ground: mount - (rest - sag) - radius = 0
        let wanted = 0.3 + (0.3 - sag) + 0.35;
        assert!(
            (state.position.local.y - wanted).abs() < 5e-3,
            "{} {wanted}",
            state.position.local.y
        );
        assert!(
            state.linear_velocity.length() < 0.07,
            "{:?}",
            state.linear_velocity
        );
        assert!(state.angular_velocity.length() < 0.01);
        for wheel in vehicle.states() {
            let contact = wheel.contact.unwrap();
            assert!(
                (contact.load - 1_000.0 * 9.81 / 4.0).abs() < 60.0,
                "{}",
                contact.load
            );
        }
    }

    #[test]
    fn in_a_turned_frame_it_settles_by_that_frame_and_its_pull() {
        use crate::{FrameDesc, FrameId, GravitySource};
        use algebrix::DQuat;
        let mut world = PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap();
        let frame = world
            .add_frame(FrameDesc {
                rotation: DQuat::from_axis_angle(DVec3::Z, std::f64::consts::FRAC_PI_2),
                ..FrameDesc::new(FrameId::ROOT)
            })
            .unwrap();
        // a planet a thousand kilometers under the frame, 9.81 at its origin
        let radius = 1.0e6;
        world
            .add_gravity(GravitySource::point(
                frame,
                DVec3::new(0.0, -radius, 0.0),
                9.81 * radius * radius,
            ))
            .unwrap();
        let mut ground = RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(200.0, 1.0, 200.0)))
            .local_position(Vec3::new(0.0, -1.0, 0.0));
        ground.frame = frame;
        world.spawn(ground).unwrap();
        let mut desc = RigidBodyDesc::dynamic(Shape::cuboid(CHASSIS_HALF))
            .density(DENSITY)
            .local_position(Vec3::new(0.0, 1.2, 0.0));
        desc.frame = frame;
        let body = world.spawn(desc).unwrap();
        let mut vehicle = Vehicle::new(&world, body, &wheels()).unwrap();
        run(&mut world, &mut vehicle, Controls::default(), 300);
        let state = chassis(&world, &vehicle);
        let sag = 1_000.0 * 9.81 / (4.0 * 30_000.0);
        let wanted = 0.3 + (0.3 - sag) + 0.35;
        assert!(
            (state.position.local.y - wanted).abs() < 5e-3,
            "{} {wanted}",
            state.position.local.y
        );
        assert!(vehicle.states().iter().all(|wheel| wheel.contact.is_some()));
    }

    #[test]
    fn drive_torque_pushes_it_forward_at_force_over_mass() {
        let mut world = world();
        let mut vehicle = car(&mut world, 0.86, Quat::IDENTITY);
        run(&mut world, &mut vehicle, Controls::default(), 120);
        let start = chassis(&world, &vehicle).position.local;
        let drive = Controls {
            drive: 350.0,
            ..Controls::default()
        };
        run(&mut world, &mut vehicle, drive, 120);
        let state = chassis(&world, &vehicle);
        // two wheels, 350 / 0.35 = 1000 n each, 2 m/s² for 2 s
        assert!(
            (state.linear_velocity.z - 4.0).abs() < 0.2,
            "{:?}",
            state.linear_velocity
        );
        assert!(state.linear_velocity.x.abs() < 0.01);
        assert!(state.position.local.z - start.z > 3.5);
        assert!(vehicle.states()[2].spin > 10.0);
    }

    #[test]
    fn brakes_stop_it_and_do_not_back_it_up() {
        let mut world = world();
        let mut vehicle = car(&mut world, 0.86, Quat::IDENTITY);
        run(&mut world, &mut vehicle, Controls::default(), 120);
        let body = vehicle.chassis();
        world
            .set_velocity(body, Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO)
            .unwrap();
        let start = chassis(&world, &vehicle).position.local.z;
        let brake = Controls {
            brake: 1_500.0,
            ..Controls::default()
        };
        run(&mut world, &mut vehicle, brake, 240);
        let state = chassis(&world, &vehicle);
        // a body held up by forces reads a little fall at step ends, so only flat speed counts
        let flat = Vec3::new(state.linear_velocity.x, 0.0, state.linear_velocity.z);
        assert!(flat.length() < 0.05, "{:?}", state.linear_velocity);
        // the friction circle caps it at about 1 g
        let distance = state.position.local.z - start;
        assert!((5.0..12.0).contains(&distance), "{distance}");
    }

    #[test]
    fn steering_left_turns_it_left() {
        let mut world = world();
        let mut vehicle = car(&mut world, 0.86, Quat::IDENTITY);
        run(&mut world, &mut vehicle, Controls::default(), 120);
        let turn = Controls {
            drive: 200.0,
            steer: 0.3,
            ..Controls::default()
        };
        run(&mut world, &mut vehicle, turn, 180);
        let state = chassis(&world, &vehicle);
        let heading = state.rotation.rotate_vec3(Vec3::Z);
        // left of +z is +x
        assert!(heading.x > 0.3, "{heading:?}");
        assert!(
            state.angular_velocity.y > 0.1,
            "{:?}",
            state.angular_velocity
        );
        let up = state.rotation.rotate_vec3(Vec3::Y);
        assert!(up.y > 0.98, "{up:?}");
    }

    #[test]
    fn side_grip_holds_it_across_a_slope() {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let tilt = Quat::from_rotation_z(10.0_f32.to_radians());
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(50.0, 1.0, 50.0)))
                    .rotation(tilt)
                    .local_position(tilt.rotate_vec3(Vec3::new(0.0, -1.0, 0.0))),
            )
            .unwrap();
        let chassis_at = tilt.rotate_vec3(Vec3::new(0.0, 0.9, 0.0));
        let body = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(CHASSIS_HALF))
                    .density(DENSITY)
                    .rotation(tilt)
                    .local_position(chassis_at),
            )
            .unwrap();
        let mut vehicle = Vehicle::new(&world, body, &wheels()).unwrap();
        run(&mut world, &mut vehicle, Controls::default(), 120);
        let settled = chassis(&world, &vehicle).position.local;
        run(&mut world, &mut vehicle, Controls::default(), 300);
        let after = chassis(&world, &vehicle).position.local;
        assert!((after - settled).length() < 0.005, "{settled:?} {after:?}");
    }

    #[test]
    fn falls_free_with_no_ground_and_wheels_hang() {
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let mut vehicle = car(&mut world, 10.0, Quat::IDENTITY);
        run(&mut world, &mut vehicle, Controls::default(), 30);
        let state = chassis(&world, &vehicle);
        assert!((state.linear_velocity.y + 9.81 * 0.5).abs() < 0.05);
        for wheel in vehicle.states() {
            assert_eq!(wheel.contact, None);
            assert_eq!(wheel.compression, 0.0);
        }
    }

    #[test]
    fn dynamic_ground_gets_the_opposite_push() {
        let mut world = PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap();
        // 6 x 0.2 x 8 m at 100 kg/m³, 960 kg
        let plank = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::new(3.0, 0.1, 4.0)))
                    .density(100.0)
                    .local_position(Vec3::ZERO),
            )
            .unwrap();
        // springs start pressed 15 cm
        let mut vehicle = car(&mut world, 0.8, Quat::IDENTITY);
        run(&mut world, &mut vehicle, Controls::default(), 20);
        let car_velocity = chassis(&world, &vehicle).linear_velocity;
        let plank_velocity = world.body_state(plank).unwrap().linear_velocity;
        assert!(car_velocity.y > 0.1, "{car_velocity:?}");
        let momentum = car_velocity * 1_000.0 + plank_velocity * 960.0;
        assert!(momentum.length() < 1.0, "{momentum:?}");
    }

    #[test]
    fn bad_input_fails() {
        let mut world = world();
        let fixed = world
            .spawn(RigidBodyDesc::fixed(Shape::sphere(1.0)))
            .unwrap();
        assert_eq!(
            Vehicle::new(&world, fixed, &wheels()),
            Err(PhysicsError::NotDynamic)
        );
        let mut vehicle = car(&mut world, 1.0, Quat::IDENTITY);
        let chassis = vehicle.chassis();
        assert!(Vehicle::new(&world, chassis, &[]).is_err());
        let bad = WheelDesc {
            radius: 0.0,
            ..WheelDesc::at(Vec3::ZERO)
        };
        assert!(Vehicle::new(&world, chassis, &[bad]).is_err());
        let nan = Controls {
            steer: f32::NAN,
            ..Controls::default()
        };
        assert!(vehicle.drive(&mut world, nan).is_err());
    }
}
