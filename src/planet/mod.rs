use std::f64::consts::FRAC_PI_4;

use algebrix::DVec3;

use crate::error::PhysicsError;

mod climate;
mod edits;
mod ground;

pub use climate::{Climate, ClimateDesc, SECONDS_PER_YEAR};
pub use edits::{PLANET_EDITS_VERSION, PlanetEdit, PlanetEdits, PlanetShape};
pub use ground::{PlanetFrames, PlanetGround, level_at, level_for};

/// Deepest a tile is split, a planet the size of the earth gets tiles of a few meters there.
pub const MAX_LEVEL: u8 = 22;
/// Most tiles one choice may hand back, so a bad detail can not run away.
const MAX_TILES: usize = 1 << 14;

/// A whole planet's ground at its real size, as heights over a sphere. Read from the direction
/// out of the center, so it has no seams between the six faces the tiles sit on, and nothing of
/// it is stored: tiles are worked out where they are needed, finer near whoever looks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetDesc {
    /// Meters to the base sphere.
    pub radius: f64,
    pub seed: u32,
    /// Most the ground rises or falls from the base sphere, meters.
    pub relief: f64,
    /// Longest wave of the relief, meters.
    pub wavelength: f64,
    /// Each one half as long and half as high as the one before.
    pub octaves: u32,
    /// Height of the sea over the base sphere, meters.
    pub sea_level: f64,
    /// Mass times the gravitational constant, m^3/s^2, for its pull.
    pub mu: f64,
    /// Seconds for one turn about its y axis against the stars, zero for none.
    pub day: f64,
}

impl PlanetDesc {
    /// The earth's size, with land and sea.
    pub fn earth(seed: u32) -> Self {
        Self {
            radius: 6_371_000.0,
            seed,
            relief: 4_000.0,
            wavelength: 2_000_000.0,
            octaves: 16,
            sea_level: 0.0,
            mu: 3.986_004_418e14,
            day: 86_164.1,
        }
    }

    /// Ground over the base sphere along `direction`, as [`Planet::height`].
    pub(crate) fn height(&self, direction: DVec3) -> f64 {
        let unit = direction.normalize();
        let mut sum = 0.0;
        let mut amplitude = 1.0;
        let mut total = 0.0;
        let mut scale = self.radius / self.wavelength;
        for octave in 0..self.octaves {
            let at = unit * scale;
            sum += amplitude * value_noise(at, self.seed.wrapping_add(octave));
            total += amplitude;
            amplitude *= 0.5;
            scale *= 2.0;
        }
        self.relief * sum / total
    }

    pub(crate) fn validate(&self) -> Result<(), PhysicsError> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let valid = positive(self.radius)
            && self.relief.is_finite()
            && self.relief >= 0.0
            && self.relief < self.radius / 2.0
            && positive(self.wavelength)
            && (1..=24).contains(&self.octaves)
            && self.sea_level.is_finite()
            && self.sea_level.abs() < self.radius / 2.0
            && positive(self.mu)
            && self.day.is_finite()
            && self.day >= 0.0;
        if valid {
            Ok(())
        } else {
            Err(PhysicsError::InvalidShape)
        }
    }
}

/// One tile of the cube the sphere is made from: a face, how many times split, and where on
/// that face. `x` and `y` count tiles across the face at that level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileKey {
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl TileKey {
    /// Corners of the tile on its face, from -1 to 1.
    pub fn bounds(self) -> ((f64, f64), (f64, f64)) {
        let span = 2.0 / f64::from(1u32 << self.level);
        let low = (
            -1.0 + span * f64::from(self.x),
            -1.0 + span * f64::from(self.y),
        );
        (low, (low.0 + span, low.1 + span))
    }

    fn children(self) -> [Self; 4] {
        let child = |dx: u32, dy: u32| Self {
            face: self.face,
            level: self.level + 1,
            x: self.x * 2 + dx,
            y: self.y * 2 + dy,
        };
        [child(0, 0), child(1, 0), child(0, 1), child(1, 1)]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Planet {
    desc: PlanetDesc,
}

impl Planet {
    /// # Errors
    /// Fails on a radius, relief or wave that is not positive and finite, relief or sea past
    /// half the radius, or octaves out of 1 to 24.
    pub fn new(desc: PlanetDesc) -> Result<Self, PhysicsError> {
        desc.validate()?;
        Ok(Self { desc })
    }

    pub fn desc(&self) -> &PlanetDesc {
        &self.desc
    }

    /// Ground over the base sphere straight out along `direction`, meters. `direction` need
    /// not be unit length, it must not be zero.
    pub fn height(&self, direction: DVec3) -> f64 {
        self.desc.height(direction)
    }

    /// The point on the ground straight out along `direction`, from the center.
    pub fn surface(&self, direction: DVec3) -> DVec3 {
        let unit = direction.normalize();
        unit * (self.desc.radius + self.height(unit))
    }

    /// Tiles covering the whole planet, split until each is no wider than `detail` times its
    /// distance from `viewer`, from the center, and no deeper than `max_level`. In face, level
    /// and place order.
    ///
    /// # Errors
    /// Fails on a non finite viewer, a detail that is not positive and finite, or more tiles
    /// than one choice may hand back.
    pub fn tiles(
        &self,
        viewer: DVec3,
        detail: f64,
        max_level: u8,
    ) -> Result<Vec<TileKey>, PhysicsError> {
        let finite = viewer.x.is_finite() && viewer.y.is_finite() && viewer.z.is_finite();
        if !finite || !(detail.is_finite() && detail > 0.0) {
            return Err(PhysicsError::OutOfRange);
        }
        let max_level = max_level.min(MAX_LEVEL);
        let mut out = Vec::new();
        let mut open: Vec<TileKey> = (0..6)
            .rev()
            .map(|face| TileKey {
                face,
                level: 0,
                x: 0,
                y: 0,
            })
            .collect();
        while let Some(tile) = open.pop() {
            let ((u0, v0), (u1, v1)) = tile.bounds();
            let center = self.surface(direction(tile.face, (u0 + u1) / 2.0, (v0 + v1) / 2.0));
            // the tile's widest span on the ground, from its corners
            let corners = [(u0, v0), (u1, v0), (u0, v1), (u1, v1)]
                .map(|(u, v)| self.desc.radius * direction(tile.face, u, v));
            let width = corners[0]
                .distance(corners[3])
                .max(corners[1].distance(corners[2]));
            let near = (center.distance(viewer) - width / 2.0).max(1.0);
            if tile.level < max_level && width > detail * near {
                // children go on in reverse so they come off in order
                open.extend(tile.children().into_iter().rev());
            } else {
                out.push(tile);
                if out.len() > MAX_TILES {
                    return Err(PhysicsError::OutOfRange);
                }
            }
        }
        Ok(out)
    }

    /// Heights of a tile at `samples` by `samples` points from corner to corner, x fastest.
    pub fn tile_heights(&self, tile: TileKey, samples: usize) -> Vec<f64> {
        let ((u0, v0), (u1, v1)) = tile.bounds();
        let steps = samples.max(2) - 1;
        let mut out = Vec::with_capacity(samples.max(2).pow(2));
        for j in 0..=steps {
            for i in 0..=steps {
                let t = (i as f64 / steps as f64, j as f64 / steps as f64);
                let at = direction(tile.face, u0 + (u1 - u0) * t.0, v0 + (v1 - v0) * t.1);
                out.push(self.height(at));
            }
        }
        out
    }
}

/// The face's outward axis and the two axes across it, so each face is seen from outside.
fn axes(face: u8) -> (DVec3, DVec3, DVec3) {
    match face {
        0 => (DVec3::X, DVec3::new(0.0, 0.0, -1.0), DVec3::Y),
        1 => (DVec3::new(-1.0, 0.0, 0.0), DVec3::Z, DVec3::Y),
        2 => (DVec3::Y, DVec3::X, DVec3::new(0.0, 0.0, -1.0)),
        3 => (DVec3::new(0.0, -1.0, 0.0), DVec3::X, DVec3::Z),
        4 => (DVec3::Z, DVec3::X, DVec3::Y),
        _ => (
            DVec3::new(0.0, 0.0, -1.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::Y,
        ),
    }
}

/// Unit direction out of the center for a place on a face, `u` and `v` from -1 to 1. Bent by a
/// tangent so cells come out near the same size over the whole face, not four times smaller at
/// its corners.
pub fn direction(face: u8, u: f64, v: f64) -> DVec3 {
    let (out, across, up) = axes(face);
    let bend = |t: f64| (t * FRAC_PI_4).tan();
    (out + across * bend(u) + up * bend(v)).normalize()
}

/// The face and place on it a direction goes through. Inverse of [`direction`].
pub fn face_of(direction: DVec3) -> (u8, f64, f64) {
    let (ax, ay, az) = (direction.x.abs(), direction.y.abs(), direction.z.abs());
    let face = if ax >= ay && ax >= az {
        if direction.x > 0.0 { 0 } else { 1 }
    } else if ay >= az {
        if direction.y > 0.0 { 2 } else { 3 }
    } else if direction.z > 0.0 {
        4
    } else {
        5
    };
    let (out, across, up) = axes(face);
    let depth = direction.dot(out);
    let unbend = |t: f64| (t / depth).atan() / FRAC_PI_4;
    (
        face,
        unbend(direction.dot(across)),
        unbend(direction.dot(up)),
    )
}

fn hash(x: i64, y: i64, z: i64, seed: u32) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ (z as u64).wrapping_mul(0x1656_67b1_9e37_79f9)
        ^ u64::from(seed).wrapping_mul(0x27d4_eb2f_1656_67c5);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// In -1 to 1, smooth, the same on every machine.
fn value_noise(at: DVec3, seed: u32) -> f64 {
    let floor = DVec3::new(at.x.floor(), at.y.floor(), at.z.floor());
    let (ix, iy, iz) = (floor.x as i64, floor.y as i64, floor.z as i64);
    let (u, v, w) = (
        fade(at.x - floor.x),
        fade(at.y - floor.y),
        fade(at.z - floor.z),
    );
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let corner = |dx: i64, dy: i64, dz: i64| hash(ix + dx, iy + dy, iz + dz, seed);
    let plane = |dz: i64| {
        lerp(
            lerp(corner(0, 0, dz), corner(1, 0, dz), u),
            lerp(corner(0, 1, dz), corner(1, 1, dz), u),
            v,
        )
    };
    lerp(plane(0), plane(1), w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn earth() -> Planet {
        Planet::new(PlanetDesc::earth(7)).unwrap()
    }

    /// A spread of directions, some near face edges and corners.
    fn directions() -> Vec<DVec3> {
        let mut out = Vec::new();
        for i in 0..40 {
            let t = i as f64 * 0.37;
            out.push(DVec3::new(t.sin() * 3.0, (t * 1.7).cos(), (t * 0.3).sin() - 0.5).normalize());
        }
        out.push(DVec3::new(1.0, 1.0, 1.0).normalize());
        out.push(DVec3::new(1.0, 1.0, 1e-9).normalize());
        out
    }

    #[test]
    fn faces_map_there_and_back() {
        for dir in directions() {
            let (face, u, v) = face_of(dir);
            assert!(u.abs() <= 1.0 + 1e-12 && v.abs() <= 1.0 + 1e-12);
            assert!(direction(face, u, v).distance(dir) < 1e-12, "{dir:?}");
        }
        // every face is seen from outside and faces meet edge to edge
        for face in 0..6 {
            let (out, across, up) = axes(face);
            assert!((across.cross(up).dot(out) - 1.0).abs() < 1e-12, "{face}");
            for (u, v) in [(1.0, 0.3), (-1.0, -0.2), (0.4, 1.0), (0.1, -1.0)] {
                let edge = direction(face, u, v);
                let (other, ou, ov) = face_of(edge);
                assert!(direction(other, ou, ov).distance(edge) < 1e-12);
            }
        }
    }

    #[test]
    fn the_ground_is_smooth_bounded_and_seamless() {
        let planet = earth();
        let relief = planet.desc().relief;
        for dir in directions() {
            let height = planet.height(dir);
            assert!(height.abs() <= relief, "{height}");
            // a meter over, on whichever face that is, the ground has not jumped
            let side = dir.cross(DVec3::Y).normalize();
            let next = (dir * planet.desc().radius + side).normalize();
            assert!((planet.height(next) - height).abs() < 5.0, "{dir:?}");
            assert_eq!(planet.height(dir), height);
        }
    }

    #[test]
    fn a_meter_stays_a_meter_at_earth_size() {
        let planet = earth();
        let dir = DVec3::new(0.3, 0.8, -0.52).normalize();
        let side = dir.cross(DVec3::Z).normalize();
        let a = planet.surface(dir);
        let b = planet.surface(a + side * 0.25);
        let apart = a.distance(b);
        assert!(apart > 0.24 && apart < 0.3, "{apart}");
    }

    #[test]
    fn tiles_are_fine_near_the_viewer_and_few_from_orbit() {
        let planet = earth();
        let dir = DVec3::new(0.2, 0.9, 0.1).normalize();
        let standing = planet.surface(dir) + dir * 2.0;
        let near = planet.tiles(standing, 1.0, MAX_LEVEL).unwrap();
        assert!(near.len() < 2000, "{}", near.len());
        let deepest = near.iter().map(|tile| tile.level).max().unwrap();
        assert!(deepest >= 18, "{deepest}");
        // the tile under the viewer is the deepest
        let (face, u, v) = face_of(dir);
        let under = near
            .iter()
            .find(|tile| {
                let ((u0, v0), (u1, v1)) = tile.bounds();
                tile.face == face && (u0..=u1).contains(&u) && (v0..=v1).contains(&v)
            })
            .unwrap();
        assert_eq!(under.level, deepest);
        let orbit = planet.tiles(dir * 3.0e7, 1.0, MAX_LEVEL).unwrap();
        assert!(orbit.len() <= 48, "{}", orbit.len());
        // the same answer twice
        assert_eq!(planet.tiles(standing, 1.0, MAX_LEVEL).unwrap(), near);
    }

    #[test]
    fn tiles_cover_every_face_once() {
        let planet = earth();
        let tiles = planet
            .tiles(DVec3::new(7.0e6, 1.0e5, 0.0), 1.0, 12)
            .unwrap();
        // area on the cube adds up to six whole faces
        let area: f64 = tiles
            .iter()
            .map(|tile| 4.0 / f64::from(1u32 << tile.level).powi(2))
            .sum();
        assert!((area - 24.0).abs() < 1e-9, "{area}");
    }

    #[test]
    fn bad_input_is_refused() {
        let mut desc = PlanetDesc::earth(1);
        desc.radius = f64::NAN;
        assert!(Planet::new(desc).is_err());
        let mut desc = PlanetDesc::earth(1);
        desc.octaves = 0;
        assert!(Planet::new(desc).is_err());
        let planet = earth();
        assert!(planet.tiles(DVec3::ZERO, 0.0, 10).is_err());
        assert!(planet.tiles(DVec3::splat(f64::NAN), 1.0, 10).is_err());
        // a detail that splits everywhere is stopped
        assert!(
            planet
                .tiles(DVec3::new(6.4e6, 0.0, 0.0), 1e-9, MAX_LEVEL)
                .is_err()
        );
    }
}
