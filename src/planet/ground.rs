use std::collections::{BTreeMap, BTreeSet};

use algebrix::{DQuat, DVec3, Vec3};

use super::{Planet, TileKey, direction, face_of};
use crate::body::{BodyHandle, RigidBodyDesc};
use crate::error::PhysicsError;
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
        })
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
            let body = world.spawn(RigidBodyDesc::fixed(shape).local_position(center.as_vec3()));
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
