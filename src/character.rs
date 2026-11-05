use algebrix::{DVec3, Quat, Vec3};

use crate::{
    BodyHandle, BodyKind, GridPosition, PhysicsError, PhysicsWorld, QueryFilter, Shape, ShapeCast,
};

const SLIDE_ITERATIONS: usize = 4;
const PUSH_OUT_ROUNDS: usize = 4;

/// A capsule standing on local Y, moved by the game, not by the solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterDesc {
    pub radius: f32,
    pub half_height: f32,
    /// Steepest walkable ground, in radians from flat.
    pub max_slope: f32,
    /// Highest ledge it walks up without a jump.
    pub step_height: f32,
    /// Gap kept to every surface, so the next cast does not start touching.
    pub skin: f32,
    /// Most force `push` puts on one body, in newtons. 0 pushes nothing.
    pub push_force: f32,
    pub filter: QueryFilter,
}

impl Default for CharacterDesc {
    fn default() -> Self {
        Self {
            radius: 0.3,
            half_height: 0.6,
            max_slope: 50.0_f32.to_radians(),
            step_height: 0.35,
            skin: 0.01,
            push_force: 400.0,
            filter: QueryFilter::default(),
        }
    }
}

impl CharacterDesc {
    fn validate(&self) -> Result<(), PhysicsError> {
        let good = |value: f32| value.is_finite() && value >= 0.0;
        if good(self.half_height)
            && good(self.step_height)
            && good(self.push_force)
            && self.radius.is_finite()
            && self.radius > 0.0
            && self.skin.is_finite()
            && self.skin > 0.0
            && self.skin < self.radius
            && self.max_slope.is_finite()
            && (0.0..std::f32::consts::FRAC_PI_2).contains(&self.max_slope)
        {
            Ok(())
        } else {
            Err(PhysicsError::InvalidShape)
        }
    }

    pub fn shape(&self) -> Shape {
        Shape::capsule(self.radius, self.half_height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ground {
    pub body: BodyHandle,
    pub normal: Vec3,
}

/// Where a move ended up and what it ran into. `moved` counts the ride on a platform too.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MoveReport {
    pub moved: Vec3,
    pub hit_wall: bool,
    pub hit_ceiling: bool,
    pub stepped: bool,
}

/// A wall or ceiling the last move ran into. `pushing` is the part of the asked motion going into it,
/// pointing into the body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Touch {
    pub body: BodyHandle,
    pub point: GridPosition,
    pub normal: Vec3,
    pub pushing: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Platform {
    body: BodyHandle,
    /// Center of the character in the body's axes.
    local: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Character {
    desc: CharacterDesc,
    position: GridPosition,
    ground: Option<Ground>,
    platform: Option<Platform>,
    touches: Vec<Touch>,
}

struct Slide {
    at: GridPosition,
    report: MoveReport,
    ground: Option<Ground>,
    touches: Vec<Touch>,
}

impl Character {
    /// `position` is the capsule center.
    ///
    /// # Errors
    /// Fails on a bad size, slope or skin, or a bad position.
    pub fn new(desc: CharacterDesc, position: GridPosition) -> Result<Self, PhysicsError> {
        desc.validate()?;
        if !position.local.is_finite() {
            return Err(PhysicsError::NonFinite("position"));
        }
        let position = position.rebased().ok_or(PhysicsError::OutOfRange)?;
        Ok(Self {
            desc,
            position,
            ground: None,
            platform: None,
            touches: Vec::new(),
        })
    }

    pub fn desc(&self) -> &CharacterDesc {
        &self.desc
    }

    pub fn position(&self) -> GridPosition {
        self.position
    }

    /// Bottom of the capsule.
    pub fn feet(&self) -> GridPosition {
        let down = f64::from(self.desc.half_height + self.desc.radius);
        self.position
            .shifted(DVec3::new(0.0, -down, 0.0))
            .unwrap_or(self.position)
    }

    pub fn ground(&self) -> Option<Ground> {
        self.ground
    }

    pub fn is_grounded(&self) -> bool {
        self.ground.is_some()
    }

    pub fn touches(&self) -> &[Touch] {
        &self.touches
    }

    /// Puts it somewhere else, for spawning and teleports. Ground is found on the next move.
    ///
    /// # Errors
    /// Fails on a bad position.
    pub fn teleport(&mut self, position: GridPosition) -> Result<(), PhysicsError> {
        if !position.local.is_finite() {
            return Err(PhysicsError::NonFinite("position"));
        }
        self.position = position.rebased().ok_or(PhysicsError::OutOfRange)?;
        self.ground = None;
        self.platform = None;
        self.touches.clear();
        Ok(())
    }

    /// Moves by `motion`, usually velocity times the step. Rides along with the body it stands
    /// on, pushes out of overlaps, slides along what it hits, walks up steps, and keeps to the
    /// ground going down slopes. The world is only read.
    ///
    /// # Errors
    /// Fails on non finite motion, an unknown frame or a result out of range.
    pub fn move_by(
        &mut self,
        world: &PhysicsWorld,
        motion: Vec3,
    ) -> Result<MoveReport, PhysicsError> {
        if !motion.is_finite() {
            return Err(PhysicsError::NonFinite("motion"));
        }
        let start = self.position;
        let was_grounded = self.ground.is_some();
        let mut at = shift(start, self.ride(world))?;
        at = self.push_out(world, at)?;
        let mut touches = Vec::new();

        let horizontal = Vec3::new(motion.x, 0.0, motion.z);
        let rising = motion.y.max(0.0);
        let falling = motion.y.min(0.0);

        let mut report = MoveReport::default();
        if rising > 0.0 {
            let up = self.slide(world, at, Vec3::Y * rising)?;
            report.hit_ceiling = up.report.hit_ceiling || up.report.hit_wall;
            at = up.at;
            touches.extend(up.touches);
        }
        let before = at;
        let plain = self.slide(world, at, horizontal)?;
        at = plain.at;
        report.hit_wall = plain.report.hit_wall;
        touches.extend_from_slice(&plain.touches);

        if was_grounded
            && plain.report.hit_wall
            && rising == 0.0
            && let Some(stepped) = self.step_up(world, before, horizontal, &plain)?
        {
            at = stepped;
            report.stepped = true;
        }

        let down = self.slide(world, at, Vec3::Y * falling)?;
        at = down.at;
        let mut ground = down.ground;

        // stay on the ground going down a slope or off a small ledge, not while jumping
        if ground.is_none() && was_grounded && rising == 0.0 {
            let reach = self.desc.step_height + self.desc.skin * 2.0;
            if let Some((distance, found)) = self.ground_below(world, at, reach)? {
                at = shift(at, Vec3::Y * -(distance - self.desc.skin))?;
                ground = Some(found);
            }
        }
        if ground.is_none() {
            ground = self
                .ground_below(world, at, self.desc.skin * 2.0)?
                .map(|(_, found)| found);
        }

        self.platform = ground.and_then(|ground| {
            let state = world.body_state(ground.body).ok()?;
            (state.kind != BodyKind::Static).then(|| Platform {
                body: ground.body,
                local: state
                    .rotation
                    .conjugate()
                    .rotate_vec3(at.offset_from(state.position).as_vec3()),
            })
        });
        self.position = at;
        self.ground = ground;
        self.touches = touches;
        report.moved = at.offset_from(start).as_vec3();
        Ok(report)
    }

    /// Pushes the dynamic bodies the last move ran into, toward the speed it wanted, with at
    /// most `push_force` each for `dt`. Call it between `move_by` and the world step.
    ///
    /// # Errors
    /// Fails on a bad `dt`.
    pub fn push(&self, world: &mut PhysicsWorld, dt: f32) -> Result<(), PhysicsError> {
        if !dt.is_finite() {
            return Err(PhysicsError::NonFinite("dt"));
        }
        if dt <= 0.0 {
            return Err(PhysicsError::InvalidSettings);
        }
        for touch in &self.touches {
            let Ok(state) = world.body_state(touch.body) else {
                continue;
            };
            let length = touch.pushing.length();
            if state.kind != BodyKind::Dynamic || length < 1e-6 {
                continue;
            }
            let direction = touch.pushing / length;
            let offset = touch.point.offset_from(state.position).as_vec3();
            let speed =
                (state.linear_velocity + state.angular_velocity.cross(offset)).dot(direction);
            let short = length / dt - speed;
            if short <= 0.0 {
                continue;
            }
            let mass = world.mass_at(touch.body, offset, direction)?;
            let impulse = (short * mass).min(self.desc.push_force * dt);
            if impulse > 0.0 {
                world.apply_impulse_at(touch.body, direction * impulse, offset)?;
            }
        }
        Ok(())
    }

    /// How far the body it stood on carried it since the last move.
    fn ride(&self, world: &PhysicsWorld) -> Vec3 {
        let Some(platform) = self.platform else {
            return Vec3::ZERO;
        };
        let Ok(state) = world.body_state(platform.body) else {
            return Vec3::ZERO;
        };
        let Some(carried) = state
            .position
            .shifted(DVec3::from_vec3(state.rotation.rotate_vec3(platform.local)))
        else {
            return Vec3::ZERO;
        };
        carried.offset_from(self.position).as_vec3()
    }

    fn walkable(&self, normal: Vec3) -> bool {
        normal.y >= self.desc.max_slope.cos()
    }

    fn cast(
        &self,
        world: &PhysicsWorld,
        at: GridPosition,
        direction: Vec3,
        distance: f32,
    ) -> Result<Option<crate::CastHit>, PhysicsError> {
        world.shape_cast(
            &ShapeCast {
                shape: self.desc.shape(),
                rotation: Quat::IDENTITY,
                origin: at,
                direction,
                max_distance: distance,
            },
            &self.desc.filter,
        )
    }

    fn ground_below(
        &self,
        world: &PhysicsWorld,
        at: GridPosition,
        reach: f32,
    ) -> Result<Option<(f32, Ground)>, PhysicsError> {
        let Some(hit) = self.cast(world, at, -Vec3::Y, reach)? else {
            return Ok(None);
        };
        if !self.walkable(hit.normal) {
            return Ok(None);
        }
        Ok(Some((
            hit.distance,
            Ground {
                body: hit.body,
                normal: hit.normal,
            },
        )))
    }

    /// Moves along `motion`, cutting away the part that runs into what it hits.
    fn slide(
        &self,
        world: &PhysicsWorld,
        mut at: GridPosition,
        mut motion: Vec3,
    ) -> Result<Slide, PhysicsError> {
        let mut report = MoveReport::default();
        let mut ground = None;
        let mut touches = Vec::new();
        let original = motion;
        for _ in 0..SLIDE_ITERATIONS {
            let length = motion.length();
            if length < 1e-6 {
                break;
            }
            let direction = motion / length;
            let Some(hit) = self.cast(world, at, direction, length + self.desc.skin)? else {
                at = shift(at, motion)?;
                break;
            };
            // the skin is kept along the surface normal, a slanted approach needs to stop sooner
            let facing = (-direction.dot(hit.normal)).max(0.1);
            let travel = (hit.distance - self.desc.skin / facing).max(0.0);
            at = shift(at, direction * travel)?;
            let mut normal = hit.normal;
            if self.walkable(normal) {
                ground = Some(Ground {
                    body: hit.body,
                    normal,
                });
            } else if normal.y < -0.3 {
                report.hit_ceiling = true;
            } else {
                // too steep to walk, so it is a wall: never let it lift
                report.hit_wall = true;
                let flat = Vec3::new(normal.x, 0.0, normal.z);
                if flat.length_squared() > 1e-8 {
                    normal = flat.normalize();
                }
            }
            let mut left = direction * (length - travel);
            let walkable = self.walkable(normal);
            // landing on walkable ground ends a fall, sliding on would carry it down the slope
            if walkable && direction.y < -0.7 {
                break;
            }
            let into = left.dot(normal);
            if into < 0.0 {
                left -= normal * into;
                if !walkable {
                    touches.push(Touch {
                        body: hit.body,
                        point: hit.point,
                        normal,
                        pushing: normal * (direction * length).dot(normal),
                    });
                }
            }
            // up a walkable slope it keeps its flat speed, it does not slow down climbing
            let flat_left = Vec3::new(left.x, 0.0, left.z).length();
            let flat_wanted = Vec3::new(direction.x, 0.0, direction.z).length() * (length - travel);
            if walkable && original.y == 0.0 && flat_left > 1e-6 {
                left *= flat_wanted / flat_left;
            }
            // never slide back against the way it was asked to go
            if left.dot(original) <= 0.0 {
                break;
            }
            motion = left;
        }
        Ok(Slide {
            at,
            report,
            ground,
            touches,
        })
    }

    /// Up by the step height, across, then down onto walkable ground. Taken only if it got
    /// further than the plain slide.
    fn step_up(
        &self,
        world: &PhysicsWorld,
        from: GridPosition,
        horizontal: Vec3,
        plain: &Slide,
    ) -> Result<Option<GridPosition>, PhysicsError> {
        let up = self.slide(world, from, Vec3::Y * self.desc.step_height)?;
        let across = self.slide(world, up.at, horizontal)?;
        let lifted = up.at.offset_from(from).y as f32;
        let Some((distance, _)) =
            self.ground_below(world, across.at, lifted + self.desc.skin * 2.0)?
        else {
            return Ok(None);
        };
        let landed = shift(across.at, Vec3::Y * -(distance - self.desc.skin))?;
        let flat = |to: GridPosition| {
            let offset = to.offset_from(from);
            (offset.x * offset.x + offset.z * offset.z).sqrt()
        };
        // it has to end up on something higher, getting a hair further at the same height is
        // only the rounded bottom grazing the ledge
        let rose = landed.offset_from(plain.at).y as f32 > self.desc.skin;
        if rose && flat(landed) > flat(plain.at) + 1e-4 {
            Ok(Some(landed))
        } else {
            Ok(None)
        }
    }

    /// Moves out of anything it overlaps, deepest first.
    fn push_out(
        &self,
        world: &PhysicsWorld,
        mut at: GridPosition,
    ) -> Result<GridPosition, PhysicsError> {
        for _ in 0..PUSH_OUT_ROUNDS {
            let Some((normal, depth)) =
                world.deepest_overlap(&self.desc.shape(), at, &self.desc.filter)?
            else {
                break;
            };
            at = shift(at, normal * (depth + self.desc.skin))?;
        }
        Ok(at)
    }
}

fn shift(at: GridPosition, by: Vec3) -> Result<GridPosition, PhysicsError> {
    at.shifted(DVec3::from_vec3(by))
        .ok_or(PhysicsError::OutOfRange)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CollisionFilter, PhysicsSettings, RigidBodyDesc};

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    fn at(x: f32, y: f32, z: f32) -> GridPosition {
        GridPosition::from_local(Vec3::new(x, y, z)).unwrap()
    }

    fn floor(world: &mut PhysicsWorld) {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(50.0, 1.0, 50.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
    }

    fn block(world: &mut PhysicsWorld, center: Vec3, half: Vec3) -> BodyHandle {
        world
            .spawn(RigidBodyDesc::fixed(Shape::cuboid(half)).local_position(center))
            .unwrap()
    }

    /// Standing height of the default capsule's center, with its skin.
    const STAND: f32 = 0.9 + 0.01;

    fn walker(world: &PhysicsWorld, x: f32, z: f32) -> Character {
        let mut character = Character::new(CharacterDesc::default(), at(x, 3.0, z)).unwrap();
        for _ in 0..120 {
            character.move_by(world, Vec3::new(0.0, -0.2, 0.0)).unwrap();
        }
        character
    }

    fn walk(character: &mut Character, world: &PhysicsWorld, step: Vec3, count: usize) {
        for _ in 0..count {
            // a little gravity each step, like a game would add
            character
                .move_by(world, step + Vec3::new(0.0, -0.05, 0.0))
                .unwrap();
        }
    }

    fn near(a: f32, b: f32, tolerance: f32) {
        assert!((a - b).abs() <= tolerance, "{a} != {b}");
    }

    #[test]
    fn falls_and_stands_on_the_floor() {
        let mut world = world();
        floor(&mut world);
        let character = walker(&world, 0.0, 0.0);
        assert!(character.is_grounded());
        near(character.position().local.y, STAND, 2e-3);
        near(character.feet().local.y, 0.01, 2e-3);
    }

    #[test]
    fn walks_on_flat_ground_at_the_asked_speed() {
        let mut world = world();
        floor(&mut world);
        let mut character = walker(&world, 0.0, 0.0);
        walk(&mut character, &world, Vec3::new(0.1, 0.0, 0.0), 50);
        near(character.position().local.x, 5.0, 1e-3);
        near(character.position().local.y, STAND, 2e-3);
        assert!(character.is_grounded());
    }

    #[test]
    fn a_wall_stops_it_and_it_slides_along() {
        let mut world = world();
        floor(&mut world);
        block(
            &mut world,
            Vec3::new(3.0, 2.0, 0.0),
            Vec3::new(0.5, 2.0, 10.0),
        );
        let mut character = walker(&world, 0.0, 0.0);
        // into the wall at 45 degrees
        walk(&mut character, &world, Vec3::new(0.1, 0.0, 0.1), 60);
        let x = character.position().local.x;
        near(x, 2.5 - 0.3 - 0.01, 2e-3);
        // the part along the wall is kept
        near(character.position().local.z, 6.0, 1e-2);
    }

    #[test]
    fn walks_up_a_low_step_but_not_a_high_one() {
        let run = |height: f32| {
            let mut world = world();
            floor(&mut world);
            block(
                &mut world,
                Vec3::new(5.0, height * 0.5, 0.0),
                Vec3::new(2.0, height * 0.5, 5.0),
            );
            let mut character = walker(&world, 0.0, 0.0);
            walk(&mut character, &world, Vec3::new(0.1, 0.0, 0.0), 60);
            character
        };
        let low = run(0.3);
        near(low.position().local.y, 0.3 + STAND, 3e-3);
        assert!(low.position().local.x > 5.0, "{:?}", low.position());
        assert!(low.is_grounded());
        let high = run(0.6);
        near(high.position().local.y, STAND, 3e-3);
        near(high.position().local.x, 3.0 - 0.3 - 0.01, 3e-3);
    }

    fn slope(world: &mut PhysicsWorld, angle: f32) {
        // a long thin plate turned about z, its top face passes through the origin
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(20.0, 0.5, 5.0)))
                    .rotation(Quat::from_rotation_z(angle))
                    .local_position(Vec3::new(0.5 * angle.sin(), -0.5 * angle.cos(), 0.0)),
            )
            .unwrap();
    }

    #[test]
    fn climbs_a_walkable_slope_and_not_a_steep_one() {
        for (degrees, climbs) in [(30.0_f32, true), (65.0, false)] {
            let mut world = world();
            let angle = degrees.to_radians();
            slope(&mut world, angle);
            let mut character = Character::new(
                CharacterDesc::default(),
                at(-3.0, 3.0 - 3.0 * angle.tan(), 0.0),
            )
            .unwrap();
            for _ in 0..60 {
                character
                    .move_by(&world, Vec3::new(0.0, -0.2, 0.0))
                    .unwrap();
            }
            let start = character.position().local;
            walk(&mut character, &world, Vec3::new(0.05, 0.0, 0.0), 60);
            let end = character.position().local;
            if climbs {
                assert!(end.x - start.x > 2.5, "{degrees}: {start:?} {end:?}");
                assert!(end.y > start.y + 1.4, "{degrees}: {start:?} {end:?}");
                assert!(character.is_grounded());
            } else {
                assert!(end.y < start.y + 0.05, "{degrees}: {start:?} {end:?}");
                assert!(!character.is_grounded() || end.x - start.x < 0.2);
            }
        }
    }

    #[test]
    fn walking_down_a_slope_keeps_it_on_the_ground() {
        let mut world = world();
        let angle = 25.0_f32.to_radians();
        slope(&mut world, angle);
        let mut character = Character::new(
            CharacterDesc::default(),
            at(3.0, 3.0 + 3.0 * angle.tan(), 0.0),
        )
        .unwrap();
        for _ in 0..60 {
            character
                .move_by(&world, Vec3::new(0.0, -0.2, 0.0))
                .unwrap();
        }
        for _ in 0..40 {
            // no gravity at all, only walking downhill
            character
                .move_by(&world, Vec3::new(-0.1, 0.0, 0.0))
                .unwrap();
            assert!(character.is_grounded(), "{:?}", character.position());
        }
    }

    #[test]
    fn standing_on_a_slope_does_not_slide_down() {
        let mut world = world();
        let angle = 30.0_f32.to_radians();
        slope(&mut world, angle);
        let mut character = Character::new(CharacterDesc::default(), at(0.0, 3.0, 0.0)).unwrap();
        for _ in 0..60 {
            character
                .move_by(&world, Vec3::new(0.0, -0.2, 0.0))
                .unwrap();
        }
        let rested = character.position().local;
        for _ in 0..120 {
            character
                .move_by(&world, Vec3::new(0.0, -0.2, 0.0))
                .unwrap();
        }
        let after = character.position().local;
        assert!((after - rested).length() < 1e-3, "{rested:?} {after:?}");
        assert!(character.is_grounded());
    }

    #[test]
    fn jumping_leaves_the_ground_and_lands_again() {
        let mut world = world();
        floor(&mut world);
        let mut character = walker(&world, 0.0, 0.0);
        character.move_by(&world, Vec3::new(0.0, 0.5, 0.0)).unwrap();
        assert!(!character.is_grounded());
        near(character.position().local.y, STAND + 0.5, 1e-3);
        for _ in 0..20 {
            character
                .move_by(&world, Vec3::new(0.0, -0.1, 0.0))
                .unwrap();
        }
        assert!(character.is_grounded());
        near(character.position().local.y, STAND, 2e-3);
    }

    #[test]
    fn a_ceiling_stops_a_jump() {
        let mut world = world();
        floor(&mut world);
        block(
            &mut world,
            Vec3::new(0.0, 2.5, 0.0),
            Vec3::new(3.0, 0.25, 3.0),
        );
        // spawned under the ceiling, the usual spawn height is inside it
        let mut character = Character::new(CharacterDesc::default(), at(0.0, 1.2, 0.0)).unwrap();
        for _ in 0..20 {
            character
                .move_by(&world, Vec3::new(0.0, -0.1, 0.0))
                .unwrap();
        }
        near(character.position().local.y, STAND, 2e-3);
        let report = character.move_by(&world, Vec3::new(0.0, 2.0, 0.0)).unwrap();
        assert!(report.hit_ceiling);
        near(character.position().local.y, 2.25 - 0.9 - 0.01, 2e-3);
    }

    #[test]
    fn a_big_move_does_not_pass_a_thin_wall() {
        let mut world = world();
        floor(&mut world);
        block(
            &mut world,
            Vec3::new(5.0, 2.0, 0.0),
            Vec3::new(0.02, 2.0, 5.0),
        );
        let mut character = walker(&world, 0.0, 0.0);
        character
            .move_by(&world, Vec3::new(50.0, 0.0, 0.0))
            .unwrap();
        assert!(
            character.position().local.x < 4.98,
            "{:?}",
            character.position()
        );
    }

    #[test]
    fn starting_inside_a_box_pushes_it_out() {
        let mut world = world();
        floor(&mut world);
        block(
            &mut world,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.5, 1.0, 0.5),
        );
        let mut character = Character::new(CharacterDesc::default(), at(0.3, 1.0, 0.0)).unwrap();
        character.move_by(&world, Vec3::ZERO).unwrap();
        let position = character.position().local;
        assert!(position.x >= 0.5 + 0.3 - 1e-3, "{position:?}");
    }

    #[test]
    fn walks_over_terrain_and_stays_on_it() {
        let mut world = world();
        let mut heights = Vec::new();
        for z in 0..21 {
            for x in 0..21 {
                heights.push((x as f32 * 0.4).sin() * 0.3 + (z as f32 * 0.3).cos() * 0.2);
            }
        }
        let terrain = world.add_heightfield(21, 21, 1.0, 1.0, &heights).unwrap();
        world
            .spawn(RigidBodyDesc::fixed(terrain).local_position(Vec3::new(-10.0, 0.0, -10.0)))
            .unwrap();
        let mut character = Character::new(CharacterDesc::default(), at(-6.0, 3.0, 0.2)).unwrap();
        for _ in 0..60 {
            character
                .move_by(&world, Vec3::new(0.0, -0.2, 0.0))
                .unwrap();
        }
        let filter = QueryFilter::default();
        for _ in 0..100 {
            character
                .move_by(&world, Vec3::new(0.1, -0.05, 0.0))
                .unwrap();
            let feet = character.feet();
            let down = crate::Ray {
                origin: character.position(),
                direction: -Vec3::Y,
                max_distance: 10.0,
            };
            let surface = world.raycast(&down, &filter).unwrap().unwrap();
            // a capsule touches slopes off center, so its feet can sit a little above the
            // surface right under the middle, never under it
            let gap = feet.local.y - surface.point.local.y;
            assert!(
                (-1e-3..0.15).contains(&gap),
                "{gap} at {:?}",
                character.position()
            );
        }
        assert!(character.is_grounded());
    }

    #[test]
    fn filter_lets_it_walk_through_a_layer() {
        let mut world = world();
        floor(&mut world);
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(0.5, 2.0, 5.0)))
                    .filter(CollisionFilter {
                        layers: 0b10,
                        mask: u32::MAX,
                    })
                    .local_position(Vec3::new(3.0, 2.0, 0.0)),
            )
            .unwrap();
        let mut character = walker(&world, 0.0, 0.0);
        character.desc.filter.mask = 0b01;
        walk(&mut character, &world, Vec3::new(0.1, 0.0, 0.0), 60);
        near(character.position().local.x, 6.0, 1e-3);
    }

    #[test]
    fn bad_input_fails() {
        let world = world();
        let bad = [
            CharacterDesc {
                radius: 0.0,
                ..CharacterDesc::default()
            },
            CharacterDesc {
                skin: 0.5,
                ..CharacterDesc::default()
            },
            CharacterDesc {
                max_slope: 2.0,
                ..CharacterDesc::default()
            },
            CharacterDesc {
                step_height: f32::NAN,
                ..CharacterDesc::default()
            },
        ];
        for desc in bad {
            assert!(Character::new(desc, at(0.0, 0.0, 0.0)).is_err(), "{desc:?}");
        }
        let mut character = Character::new(CharacterDesc::default(), at(0.0, 0.0, 0.0)).unwrap();
        assert!(
            character
                .move_by(&world, Vec3::new(f32::NAN, 0.0, 0.0))
                .is_err()
        );
        assert_eq!(character.position(), at(0.0, 0.0, 0.0));
    }

    fn crate_box(world: &mut PhysicsWorld, x: f32, half: f32, density: f32) -> BodyHandle {
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(half)))
                    .density(density)
                    .local_position(Vec3::new(x, half, 0.0)),
            )
            .unwrap()
    }

    fn walk_and_push(
        character: &mut Character,
        world: &mut PhysicsWorld,
        step: Vec3,
        count: usize,
    ) {
        let dt = 1.0 / 60.0;
        for _ in 0..count {
            character
                .move_by(world, step + Vec3::new(0.0, -0.05, 0.0))
                .unwrap();
            character.push(world, dt).unwrap();
            world.step_fixed().unwrap();
        }
    }

    #[test]
    fn walking_into_a_light_box_pushes_it() {
        let mut world = world();
        floor(&mut world);
        let light = crate_box(&mut world, 2.0, 0.4, 50.0);
        let mut character = walker(&world, 0.0, 0.0);
        walk_and_push(&mut character, &mut world, Vec3::new(0.03, 0.0, 0.0), 180);
        let state = world.body_state(light).unwrap();
        assert!(state.position.local.x > 5.5, "{:?}", state.position);
        // close to the walking speed, friction takes one step of it
        near(state.linear_velocity.x, 1.8, 0.15);
        assert!(character.touches().iter().any(|touch| touch.body == light));
        // it walks behind the box, not through it
        let gap = state.position.local.x - character.position().local.x;
        assert!(gap > 0.4 + 0.3 - 0.05, "{gap}");
        assert!(character.position().local.x > 3.0);
    }

    #[test]
    fn a_heavy_box_or_no_push_force_stays_put() {
        for (half, density, force) in [(0.5, 1000.0, 400.0), (0.4, 50.0, 0.0)] {
            let mut world = world();
            floor(&mut world);
            let block = crate_box(&mut world, 2.0, half, density);
            let mut character = Character::new(
                CharacterDesc {
                    push_force: force,
                    ..CharacterDesc::default()
                },
                at(0.0, STAND, 0.0),
            )
            .unwrap();
            walk_and_push(&mut character, &mut world, Vec3::new(0.03, 0.0, 0.0), 180);
            let x = world.body_state(block).unwrap().position.local.x;
            near(x, 2.0, 0.02);
            near(character.position().local.x, 2.0 - half - 0.3 - 0.01, 0.02);
        }
    }

    #[test]
    fn rides_a_moving_platform() {
        let mut world = world();
        floor(&mut world);
        let platform = world
            .spawn(
                RigidBodyDesc::kinematic(Shape::cuboid(Vec3::new(2.0, 0.25, 2.0)))
                    .local_position(Vec3::new(0.0, 1.0, 0.0))
                    .linear_velocity(Vec3::new(1.0, 0.5, 0.0)),
            )
            .unwrap();
        let mut character = Character::new(CharacterDesc::default(), at(0.0, 3.0, 0.0)).unwrap();
        for _ in 0..30 {
            character
                .move_by(&world, Vec3::new(0.0, -0.2, 0.0))
                .unwrap();
        }
        assert_eq!(character.ground().map(|g| g.body), Some(platform));
        let start = character.position().local;
        let base = world.body_state(platform).unwrap().position.local;
        for _ in 0..60 {
            world.step_fixed().unwrap();
            character
                .move_by(&world, Vec3::new(0.0, -0.05, 0.0))
                .unwrap();
            assert!(character.is_grounded());
        }
        let moved = world.body_state(platform).unwrap().position.local - base;
        let carried = character.position().local - start;
        assert!((carried - moved).length() < 5e-3, "{carried:?} {moved:?}");
    }

    #[test]
    fn turns_with_a_spinning_platform() {
        let mut world = world();
        floor(&mut world);
        world
            .spawn(
                RigidBodyDesc::kinematic(Shape::cuboid(Vec3::new(3.0, 0.25, 3.0)))
                    .local_position(Vec3::new(0.0, 0.25, 0.0))
                    .angular_velocity(Vec3::new(0.0, 1.0, 0.0)),
            )
            .unwrap();
        let mut character = Character::new(CharacterDesc::default(), at(2.0, 2.0, 0.0)).unwrap();
        for _ in 0..30 {
            character
                .move_by(&world, Vec3::new(0.0, -0.2, 0.0))
                .unwrap();
        }
        let height = character.position().local.y;
        // a quarter turn
        let steps = (std::f32::consts::FRAC_PI_2 * 60.0).round() as usize;
        for _ in 0..steps {
            world.step_fixed().unwrap();
            character.move_by(&world, Vec3::ZERO).unwrap();
        }
        let position = character.position().local;
        let angle = steps as f32 / 60.0;
        let wanted = Vec3::new(2.0 * angle.cos(), height, -2.0 * angle.sin());
        assert!(
            (position - wanted).length() < 0.02,
            "{position:?} {wanted:?}"
        );
    }

    #[test]
    fn push_rejects_a_bad_dt() {
        let mut world = world();
        let character = Character::new(CharacterDesc::default(), at(0.0, 0.0, 0.0)).unwrap();
        assert!(character.push(&mut world, 0.0).is_err());
        assert!(character.push(&mut world, f32::NAN).is_err());
    }

    #[test]
    fn moving_reads_the_world_and_changes_nothing() {
        let mut world = world();
        floor(&mut world);
        let before = world.state_hash();
        let mut character = walker(&world, 0.0, 0.0);
        walk(&mut character, &world, Vec3::new(0.1, 0.0, 0.0), 20);
        assert_eq!(world.state_hash(), before);
    }
}
