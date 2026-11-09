use std::collections::{BTreeMap, BTreeSet};

use algebrix::{DQuat, DVec3, Vec3};

use super::{Planet, TileKey, direction, face_of};
use crate::body::{BodyHandle, RigidBodyDesc};
use crate::error::PhysicsError;
use crate::frame::{FrameDesc, FrameId, FrameMotion};
use crate::gravity::{GravityId, GravitySource};
use crate::shape::Shape;
use crate::world::PhysicsWorld;

/// Points across a ground tile, so cells a sixteenth of the tile.
const SAMPLES: usize = 17;

/// A planet's ground where bodies are, as fixed mesh tiles in a world that runs in a frame set
/// at `anchor`, turned so up there is +y. Within a few kilometers of the anchor the ground's
/// curve is small and the world's own gravity down -y holds. Tiles are made around the points
/// asked for and dropped once nothing is near them.
#[derive(Debug, Clone)]
pub struct PlanetGround {
    planet: Planet,
    anchor: DVec3,
    turn: DQuat,
    level: u8,
    tiles: BTreeMap<TileKey, (BodyHandle, Shape)>,
    /// The frame the tiles stand in.
    frame: FrameId,
}

/// The frames and pull [`PlanetGround::on_planet`] made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanetFrames {
    /// Centered on the planet and turning with it.
    pub planet: FrameId,
    /// At the anchor in the planet frame, up there +y: the ground's tiles and bodies on them go
    /// here.
    pub ground: FrameId,
    pub gravity: GravityId,
}

impl PlanetGround {
    /// Ground tiles at `level`, from [`level_for`]. `anchor` is a point from the planet's center,
    /// the world's origin.
    ///
    /// # Errors
    /// Fails on a non finite anchor or a level past the deepest.
    pub fn new(planet: Planet, anchor: DVec3, level: u8) -> Result<Self, PhysicsError> {
        let finite = anchor.x.is_finite() && anchor.y.is_finite() && anchor.z.is_finite();
        if !finite || anchor.length() <= 0.0 || level > super::MAX_LEVEL {
            return Err(PhysicsError::OutOfRange);
        }
        Ok(Self {
            turn: level_at(anchor),
            planet,
            anchor,
            level,
            tiles: BTreeMap::new(),
            frame: FrameId::ROOT,
        })
    }

    /// The ground on a real planet in `world`: a frame at its center turning once a
    /// [`PlanetDesc::day`](super::PlanetDesc::day), its pull there, and a frame at `anchor` in it
    /// with up +y, where the tiles stand. Bodies in that frame fall toward the center however far
    /// they go and feel the turning, so the world's uniform gravity should be zero.
    ///
    /// # Errors
    /// As [`Self::new`], and when the world refuses a frame or the pull.
    pub fn on_planet(
        world: &mut PhysicsWorld,
        planet: Planet,
        anchor: DVec3,
        level: u8,
    ) -> Result<(Self, PlanetFrames), PhysicsError> {
        let mut ground = Self::new(planet, anchor, level)?;
        let desc = *ground.planet.desc();
        let spin = if desc.day > 0.0 {
            DVec3::Y * (std::f64::consts::TAU / desc.day)
        } else {
            DVec3::ZERO
        };
        let center = world.add_frame(FrameDesc {
            motion: FrameMotion { spin, orbit: None },
            ..FrameDesc::new(FrameId::ROOT)
        })?;
        // pulls as a point down to the lowest ground can be, the base sphere less the relief:
        // with the base sphere a valley under it was pulled as inside the ball, 0.06% weak
        let gravity = world.add_gravity(GravitySource {
            radius: desc.radius - desc.relief,
            ..GravitySource::point(center, DVec3::ZERO, desc.mu)
        })?;
        ground.frame = world.add_frame(FrameDesc {
            translation: anchor,
            rotation: ground.turn.conjugate(),
            ..FrameDesc::new(center)
        })?;
        let frames = PlanetFrames {
            planet: center,
            ground: ground.frame,
            gravity,
        };
        Ok((ground, frames))
    }

    /// The frame the tiles stand in, the root for [`Self::new`].
    pub fn frame(&self) -> FrameId {
        self.frame
    }

    pub fn planet(&self) -> &Planet {
        &self.planet
    }

    pub fn anchor(&self) -> DVec3 {
        self.anchor
    }

    /// Turns planet directions into the world's, up at the anchor to +y.
    pub fn turn(&self) -> DQuat {
        self.turn
    }

    /// A point from the planet's center in the world's frame.
    pub fn to_world(&self, from_center: DVec3) -> DVec3 {
        self.turn.mul_vec3(from_center - self.anchor)
    }

    /// A point of the world's frame from the planet's center.
    pub fn to_planet(&self, world: DVec3) -> DVec3 {
        self.anchor + self.turn.conjugate().mul_vec3(world)
    }

    /// The ground straight under or over a point of the world's frame, in that frame.
    pub fn ground_under(&self, world: DVec3) -> DVec3 {
        self.to_world(self.planet.surface(self.to_planet(world)))
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// The bodies of the tiles standing now.
    pub fn bodies(&self) -> impl Iterator<Item = BodyHandle> + '_ {
        self.tiles.values().map(|(body, _)| *body)
    }

    /// Makes the tiles within `reach` of each point of `near`, in the world's frame, and drops
    /// the rest.
    ///
    /// # Errors
    /// Fails on a non finite point or reach, or when the world refuses a tile.
    pub fn update(
        &mut self,
        world: &mut PhysicsWorld,
        near: &[DVec3],
        reach: f64,
    ) -> Result<(), PhysicsError> {
        if !(reach.is_finite() && reach >= 0.0) {
            return Err(PhysicsError::OutOfRange);
        }
        let width = self.tile_width();
        let steps = (reach / (width / 2.0)).ceil() as i64;
        let mut wanted = BTreeSet::new();
        for point in near {
            if !(point.x.is_finite() && point.y.is_finite() && point.z.is_finite()) {
                return Err(PhysicsError::OutOfRange);
            }
            // points over the ground around this one, half a tile apart, each names its tile
            let center = self.to_planet(*point);
            let up = center.normalize();
            let side = up.cross(if up.y.abs() < 0.9 { DVec3::Y } else { DVec3::X });
            let side = side.normalize();
            let ahead = up.cross(side);
            for a in -steps..=steps {
                for b in -steps..=steps {
                    let offset = side * (a as f64 * width / 2.0) + ahead * (b as f64 * width / 2.0);
                    wanted.insert(self.tile_at(center + offset));
                }
            }
        }
        let gone: Vec<TileKey> = self
            .tiles
            .keys()
            .filter(|tile| !wanted.contains(tile))
            .copied()
            .collect();
        for tile in gone {
            if let Some((body, shape)) = self.tiles.remove(&tile) {
                world.despawn(body)?;
                world.remove_mesh(shape)?;
            }
        }
        for tile in wanted {
            if self.tiles.contains_key(&tile) {
                continue;
            }
            let (vertices, triangles, center) = self.tile_mesh(tile);
            let shape = world.add_mesh(&vertices, &triangles)?;
            let mut desc = RigidBodyDesc::fixed(shape).local_position(center.as_vec3());
            desc.frame = self.frame;
            let body = world.spawn(desc);
            match body {
                Ok(body) => {
                    self.tiles.insert(tile, (body, shape));
                }
                Err(error) => {
                    world.remove_mesh(shape)?;
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// How wide a ground tile is, near enough everywhere.
    pub fn tile_width(&self) -> f64 {
        let radius = self.planet.desc().radius;
        radius * std::f64::consts::FRAC_PI_2 / f64::from(1u32 << self.level)
    }

    fn tile_at(&self, from_center: DVec3) -> TileKey {
        let (face, u, v) = face_of(from_center.normalize());
        let count = 1u32 << self.level;
        let cell = |t: f64| (((t + 1.0) / 2.0 * f64::from(count)) as u32).min(count - 1);
        TileKey {
            face,
            level: self.level,
            x: cell(u),
            y: cell(v),
        }
    }

    /// The tile's ground in the world's frame around its middle, wound to face up, and where
    /// its middle sits.
    fn tile_mesh(&self, tile: TileKey) -> (Vec<Vec3>, Vec<[u32; 3]>, DVec3) {
        let ((u0, v0), (u1, v1)) = tile.bounds();
        let steps = (SAMPLES - 1) as f64;
        let mut points = Vec::with_capacity(SAMPLES * SAMPLES);
        for j in 0..SAMPLES {
            for i in 0..SAMPLES {
                let way = direction(
                    tile.face,
                    u0 + (u1 - u0) * i as f64 / steps,
                    v0 + (v1 - v0) * j as f64 / steps,
                );
                points.push(self.to_world(self.planet.surface(way)));
            }
        }
        let center = points[SAMPLES * SAMPLES / 2];
        let vertices: Vec<Vec3> = points.iter().map(|at| (*at - center).as_vec3()).collect();
        let outward = self.turn.mul_vec3(self.to_planet(center).normalize());
        let (a, b, c) = (points[0], points[1], points[SAMPLES + 1]);
        let up = (b - a).cross(c - a).dot(outward) > 0.0;
        let mut triangles = Vec::with_capacity((SAMPLES - 1) * (SAMPLES - 1) * 2);
        for j in 0..SAMPLES - 1 {
            for i in 0..SAMPLES - 1 {
                let corner = |di: usize, dj: usize| ((j + dj) * SAMPLES + i + di) as u32;
                let (p, q, r, t) = (corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1));
                if up {
                    triangles.extend([[p, q, r], [p, r, t]]);
                } else {
                    triangles.extend([[p, r, q], [p, t, r]]);
                }
            }
        }
        (vertices, triangles, center)
    }
}

/// The deepest tile level whose cells are no wider than `cell` meters on `planet`.
pub fn level_for(planet: &Planet, cell: f64) -> u8 {
    let face = planet.desc().radius * std::f64::consts::FRAC_PI_2;
    let mut level = 0;
    while level < super::MAX_LEVEL && face / f64::from(1u32 << level) / (SAMPLES - 1) as f64 > cell
    {
        level += 1;
    }
    level
}

/// Turns straight up at `at`, out from the center, to +y.
pub fn level_at(at: DVec3) -> DQuat {
    let up = at.normalize();
    let axis = up.cross(DVec3::Y);
    let size = axis.length();
    if size < 1e-12 {
        let flip = if up.y > 0.0 {
            0.0
        } else {
            std::f64::consts::PI
        };
        return DQuat::from_axis_angle(DVec3::X, flip);
    }
    DQuat::from_axis_angle(
        axis * size.recip(),
        up.dot(DVec3::Y).clamp(-1.0, 1.0).acos(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::PlanetDesc;
    use crate::world::PhysicsSettings;

    fn ground() -> (PlanetGround, PhysicsWorld) {
        let planet = Planet::new(PlanetDesc::earth(3)).unwrap();
        let anchor = planet.surface(DVec3::new(0.3, 0.8, 0.5));
        let level = level_for(&planet, 2.5);
        let ground = PlanetGround::new(planet, anchor, level).unwrap();
        (
            ground,
            PhysicsWorld::new(PhysicsSettings::default()).unwrap(),
        )
    }

    #[test]
    fn the_anchor_is_the_origin_and_up_there_is_y() {
        let (ground, _) = ground();
        assert!(ground.to_world(ground.anchor()).length() < 1e-6);
        let up = ground.to_world(ground.anchor() * 1.0001);
        assert!(up.normalize().dot(DVec3::Y) > 1.0 - 1e-9, "{up:?}");
        let back = ground.to_planet(DVec3::new(3.0, 4.0, 5.0));
        assert!(ground.to_world(back).distance(DVec3::new(3.0, 4.0, 5.0)) < 1e-6);
    }

    #[test]
    fn a_box_dropped_on_the_planet_rests_on_its_ground() {
        let (mut ground, mut world) = ground();
        let start = DVec3::new(12.0, 0.0, -7.0);
        let under = ground.ground_under(start);
        let drop = under + DVec3::Y * 3.0;
        let body = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(drop.as_vec3()),
            )
            .unwrap();
        for _ in 0..180 {
            let at = world.body_state(body).unwrap().position.to_dvec3();
            ground.update(&mut world, &[at], 20.0).unwrap();
            world.step_fixed().unwrap();
        }
        let at = world.body_state(body).unwrap().position.to_dvec3();
        let height = at.y - ground.ground_under(at).y;
        assert!((height - 0.5).abs() < 0.15, "{height}");
        assert!(ground.tile_count() <= 16, "{}", ground.tile_count());
    }

    /// Earth turning, its pull, and its ground frame at the equator on the x axis.
    fn on_earth() -> (PlanetGround, PlanetFrames, PhysicsWorld) {
        let planet = Planet::new(PlanetDesc::earth(3)).unwrap();
        let anchor = planet.surface(DVec3::X);
        let level = level_for(&planet, 2.5);
        let mut world = PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap();
        let (ground, frames) = PlanetGround::on_planet(&mut world, planet, anchor, level).unwrap();
        (ground, frames, world)
    }

    #[test]
    fn on_a_turning_planet_things_fall_with_its_pull_less_the_turning() {
        let (ground, frames, world) = on_earth();
        let desc = *ground.planet().desc();
        let r = ground.anchor().length();
        let spin = std::f64::consts::TAU / desc.day;
        let felt = world
            .gravity_at(frames.ground, crate::GridPosition::ORIGIN, Vec3::ZERO)
            .unwrap();
        let expected = desc.mu / (r * r) - spin * spin * r;
        assert!(
            (f64::from(-felt.y) - expected).abs() < 1e-4,
            "{felt:?} {expected}"
        );
        assert!(felt.x.abs() < 1e-4 && felt.z.abs() < 1e-4, "{felt:?}");
    }

    #[test]
    fn a_ball_dropped_at_the_equator_lands_east_by_coriolis() {
        let (ground, frames, mut world) = on_earth();
        let height = 100.0_f32;
        let mut desc =
            RigidBodyDesc::dynamic(Shape::sphere(0.1)).local_position(Vec3::new(0.0, height, 0.0));
        desc.frame = frames.ground;
        desc.can_sleep = false;
        let ball = world.spawn(desc).unwrap();
        let mut time = 0.0;
        while world.body_state(ball).unwrap().position.local.y > 0.0 {
            world.step_fixed().unwrap();
            time += 1.0 / 60.0;
        }
        // the planet turns about its y, so east at the anchor on its x is minus z there
        let east = ground.turn().mul_vec3(DVec3::new(0.0, 0.0, -1.0)).as_vec3();
        let landed = world.body_state(ball).unwrap().position.local;
        let spin = std::f64::consts::TAU / ground.planet().desc().day;
        let g = 2.0 * f64::from(height) / (time * time);
        let expected = spin * g * time.powi(3) / 3.0;
        let off = f64::from(landed.dot(east));
        assert!((off - expected).abs() < 0.2 * expected, "{off} {expected}");
        assert!(expected > 0.02);
    }

    #[test]
    fn a_box_rests_on_a_turning_planet() {
        let (mut ground, frames, mut world) = on_earth();
        let drop = ground.ground_under(DVec3::new(12.0, 0.0, -7.0)) + DVec3::Y * 3.0;
        let mut desc =
            RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5))).local_position(drop.as_vec3());
        desc.frame = frames.ground;
        let body = world.spawn(desc).unwrap();
        for _ in 0..240 {
            let at = world.body_state(body).unwrap().position.to_dvec3();
            ground.update(&mut world, &[at], 20.0).unwrap();
            world.step_fixed().unwrap();
        }
        let state = world.body_state(body).unwrap();
        let at = state.position.to_dvec3();
        let height = at.y - ground.ground_under(at).y;
        assert!((height - 0.5).abs() < 0.15, "{height}");
        assert!(
            state.linear_velocity.length() < 0.01,
            "{:?}",
            state.linear_velocity
        );
        assert!(
            ground
                .bodies()
                .all(|tile| world.body_state(tile).unwrap().frame == frames.ground)
        );
    }

    #[test]
    fn tiles_follow_the_points_and_go_when_left() {
        let (mut ground, mut world) = ground();
        ground.update(&mut world, &[DVec3::ZERO], 10.0).unwrap();
        let here = ground.tile_count();
        assert!(here >= 1);
        let far = DVec3::new(5000.0, 0.0, 0.0);
        ground.update(&mut world, &[far], 10.0).unwrap();
        assert!(ground.tile_count() <= here + 4);
        // only the far tiles stand, the first ones went
        assert_eq!(world.body_handles().count(), ground.tile_count());
        assert!(
            ground
                .update(&mut world, &[DVec3::splat(f64::NAN)], 1.0)
                .is_err()
        );
    }

    #[test]
    fn cells_are_as_fine_as_asked() {
        let planet = Planet::new(PlanetDesc::earth(3)).unwrap();
        let level = level_for(&planet, 2.5);
        let ground = PlanetGround::new(planet, DVec3::new(0.0, 6.4e6, 0.0), level).unwrap();
        let cell = ground.tile_width() / 16.0;
        assert!(cell <= 2.5 && cell > 1.2, "{cell}");
    }
}
