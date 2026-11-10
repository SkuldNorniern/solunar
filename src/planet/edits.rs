use algebrix::{DQuat, DVec3, Vec3};

use super::PlanetGround;
use crate::error::PhysicsError;
use crate::sdf::{Edit, EditOp, EditShape, SdfDesc};
use crate::snapshot::{Reader, SnapshotError, Writer};

const MAGIC: [u8; 4] = *b"SLPE";
pub const PLANET_EDITS_VERSION: u32 = 1;
/// Most edits one store holds or loads.
const MAX_EDITS: usize = 1 << 22;

/// An edit's shape from the planet's center, in its axes, so it stays put whatever region or
/// anchor looks at it later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanetShape {
    Sphere {
        center: DVec3,
        radius: f64,
    },
    Box {
        center: DVec3,
        half_extents: DVec3,
        rotation: DQuat,
    },
    Capsule {
        a: DVec3,
        b: DVec3,
        radius: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetEdit {
    pub shape: PlanetShape,
    pub op: EditOp,
}

/// Every change dug into or built onto a planet's ground, in the order made. Only these are
/// kept: the ground itself is worked out again from the planet wherever a region is made.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanetEdits {
    edits: Vec<PlanetEdit>,
}

fn finite(v: DVec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

impl PlanetShape {
    fn is_valid(&self) -> bool {
        match *self {
            Self::Sphere { center, radius } => finite(center) && positive(radius),
            Self::Box {
                center,
                half_extents,
                rotation,
            } => {
                let q = [rotation.x, rotation.y, rotation.z, rotation.w];
                let length = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                finite(center)
                    && finite(half_extents)
                    && half_extents.x > 0.0
                    && half_extents.y > 0.0
                    && half_extents.z > 0.0
                    && (length - 1.0).abs() < 1e-3
            }
            Self::Capsule { a, b, radius } => finite(a) && finite(b) && positive(radius),
        }
    }
}

impl PlanetEdits {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.edits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn edits(&self) -> &[PlanetEdit] {
        &self.edits
    }

    /// Adds `edit` after the others.
    ///
    /// # Errors
    /// Fails on a shape that is not finite, a size that is not positive, a box turned by a
    /// rotation that is not one, or a full store.
    pub fn push(&mut self, edit: PlanetEdit) -> Result<(), PhysicsError> {
        if !edit.shape.is_valid() {
            return Err(PhysicsError::InvalidShape);
        }
        if self.edits.len() >= MAX_EDITS {
            return Err(PhysicsError::OutOfRange);
        }
        self.edits.push(edit);
        Ok(())
    }

    /// The edits that reach the region `desc` of `ground`, in its own space and in order, to
    /// apply to a region made there.
    pub fn local(&self, ground: &PlanetGround, desc: &SdfDesc) -> Vec<Edit> {
        let corner = desc.origin.to_dvec3();
        let cells = desc.brick_cells as f32 * desc.voxel;
        let size = Vec3::new(
            desc.bricks[0] as f32 * cells,
            desc.bricks[1] as f32 * cells,
            desc.bricks[2] as f32 * cells,
        );
        self.edits
            .iter()
            .filter_map(|edit| {
                let local = ground.edit_to_region(corner, edit);
                let (low, high) = local.bounds();
                let reaches = low.x <= size.x
                    && low.y <= size.y
                    && low.z <= size.z
                    && high.x >= 0.0
                    && high.y >= 0.0
                    && high.z >= 0.0;
                reaches.then_some(local)
            })
            .collect()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Writer::with_capacity(16 + self.edits.len() * 80);
        out.bytes(&MAGIC);
        out.u32(PLANET_EDITS_VERSION);
        out.len(self.edits.len());
        for edit in &self.edits {
            out.u8(match edit.op {
                EditOp::Add => 0,
                EditOp::Excavate => 1,
                EditOp::Tunnel => 2,
            });
            match edit.shape {
                PlanetShape::Sphere { center, radius } => {
                    out.u8(0);
                    out.dvec3(center);
                    out.f64(radius);
                }
                PlanetShape::Box {
                    center,
                    half_extents,
                    rotation,
                } => {
                    out.u8(1);
                    out.dvec3(center);
                    out.dvec3(half_extents);
                    for value in [rotation.x, rotation.y, rotation.z, rotation.w] {
                        out.f64(value);
                    }
                }
                PlanetShape::Capsule { a, b, radius } => {
                    out.u8(2);
                    out.dvec3(a);
                    out.dvec3(b);
                    out.f64(radius);
                }
            }
        }
        let sum = crate::snapshot::fnv1a(&out.data);
        out.u64(sum);
        out.data
    }

    /// # Errors
    /// Fails on bytes that are not a store of this version, cut short, changed, or holding a
    /// shape [`Self::push`] would refuse.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        let bad = SnapshotError::Invalid("planet edits");
        if bytes.len() < 20 || bytes[..4] != MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        let (body, sum) = bytes.split_at(bytes.len() - 8);
        if crate::snapshot::fnv1a(body) != u64::from_le_bytes(sum.try_into().map_err(|_| bad)?) {
            return Err(SnapshotError::BadChecksum);
        }
        let mut reader = Reader {
            data: body,
            at: 4,
            version: 0,
        };
        let version = reader.u32()?;
        if version != PLANET_EDITS_VERSION {
            return Err(SnapshotError::UnsupportedVersion(version));
        }
        // the smallest record is an op, a kind and a ball
        let count = reader.count(34, MAX_EDITS)?;
        let mut edits = Vec::with_capacity(count);
        for _ in 0..count {
            let op = match reader.u8()? {
                0 => EditOp::Add,
                1 => EditOp::Excavate,
                2 => EditOp::Tunnel,
                _ => return Err(bad),
            };
            let shape = match reader.u8()? {
                0 => PlanetShape::Sphere {
                    center: reader.dvec3()?,
                    radius: reader.f64()?,
                },
                1 => {
                    let (center, half_extents) = (reader.dvec3()?, reader.dvec3()?);
                    let (x, y, z, w) = (reader.f64()?, reader.f64()?, reader.f64()?, reader.f64()?);
                    PlanetShape::Box {
                        center,
                        half_extents,
                        rotation: DQuat::new(x, y, z, w),
                    }
                }
                2 => PlanetShape::Capsule {
                    a: reader.dvec3()?,
                    b: reader.dvec3()?,
                    radius: reader.f64()?,
                },
                _ => return Err(bad),
            };
            if !shape.is_valid() {
                return Err(bad);
            }
            edits.push(PlanetEdit { shape, op });
        }
        if reader.at != body.len() {
            return Err(bad);
        }
        Ok(Self { edits })
    }
}

impl PlanetGround {
    /// `edit`, made in the region whose origin is `corner` in this ground's frame, as an edit of
    /// the planet to keep in [`PlanetEdits`].
    pub fn edit_to_planet(&self, corner: DVec3, edit: &Edit) -> PlanetEdit {
        let back = self.turn().conjugate();
        let at = |local: Vec3| self.anchor() + back.mul_vec3(DVec3::from_vec3(local) + corner);
        let shape = match edit.shape {
            EditShape::Sphere { center, radius } => PlanetShape::Sphere {
                center: at(center),
                radius: f64::from(radius),
            },
            EditShape::Box {
                center,
                half_extents,
                rotation,
            } => PlanetShape::Box {
                center: at(center),
                half_extents: DVec3::from_vec3(half_extents),
                rotation: (back * DQuat::from_quat(rotation.normalize())).normalize(),
            },
            EditShape::Capsule { a, b, radius } => PlanetShape::Capsule {
                a: at(a),
                b: at(b),
                radius: f64::from(radius),
            },
        };
        PlanetEdit { shape, op: edit.op }
    }

    /// A kept planet edit in the region whose origin is `corner` in this ground's frame.
    pub fn edit_to_region(&self, corner: DVec3, edit: &PlanetEdit) -> Edit {
        let local =
            |planet: DVec3| (self.turn().mul_vec3(planet - self.anchor()) - corner).as_vec3();
        let shape = match edit.shape {
            PlanetShape::Sphere { center, radius } => EditShape::Sphere {
                center: local(center),
                radius: radius as f32,
            },
            PlanetShape::Box {
                center,
                half_extents,
                rotation,
            } => EditShape::Box {
                center: local(center),
                half_extents: half_extents.as_vec3(),
                rotation: (self.turn() * rotation).normalize().as_quat(),
            },
            PlanetShape::Capsule { a, b, radius } => EditShape::Capsule {
                a: local(a),
                b: local(b),
                radius: radius as f32,
            },
        };
        Edit { shape, op: edit.op }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EditOp, Material, PhysicsSettings, PhysicsWorld, Planet, PlanetDesc, SdfTerrain,
        Sequential, level_for,
    };

    fn ground(direction: DVec3) -> PlanetGround {
        let planet = Planet::new(PlanetDesc::earth(3)).unwrap();
        let anchor = planet.surface(direction);
        let level = level_for(&planet, 2.5);
        PlanetGround::new(planet, anchor, level).unwrap()
    }

    fn region(ground: &PlanetGround, at: DVec3) -> SdfDesc {
        ground
            .sdf_desc(ground.ground_under(at), 12.0, (0.5, 8), Material::default())
            .unwrap()
    }

    #[test]
    fn a_region_holds_the_planets_own_ground() {
        let ground = ground(DVec3::new(0.3, 0.8, 0.5));
        // away from the anchor, where the frame's up has turned off the planet's
        for at in [DVec3::ZERO, DVec3::new(900.0, 0.0, -400.0)] {
            let desc = region(&ground, at);
            let volume = crate::SdfVolume::new(desc).unwrap();
            let corner = desc.origin.to_dvec3();
            for (dx, dz) in [(0.0, 0.0), (3.3, -5.1), (-7.0, 6.2)] {
                let on = ground.ground_under(at + DVec3::new(dx, 0.0, dz));
                let d = volume.distance((on - corner).as_vec3());
                assert!(d.abs() < 0.02, "{at:?} {dx} {dz} {d}");
                let above = volume.distance((on - corner).as_vec3() + Vec3::Y);
                assert!(above > 0.9 && above < 1.01, "{above}");
            }
        }
    }

    #[test]
    fn a_hole_kept_as_bytes_comes_back_in_the_same_place_from_another_anchor() {
        let first = ground(DVec3::new(0.3, 0.8, 0.5));
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let desc = region(&first, DVec3::ZERO);
        let corner = desc.origin.to_dvec3();
        let mut terrain = SdfTerrain::new_in(&mut world, desc, first.frame(), &Sequential).unwrap();
        let center = (first.ground_under(DVec3::ZERO) - corner).as_vec3();
        let dig = Edit {
            shape: EditShape::Sphere {
                center,
                radius: 2.0,
            },
            op: EditOp::Excavate,
        };
        terrain.edit(&mut world, dig).unwrap();
        let mut kept = PlanetEdits::new();
        kept.push(first.edit_to_planet(corner, &dig)).unwrap();
        let hole = terrain.volume().distance(center);
        assert!(hole > 1.9, "{hole}");
        // a ray down the hole in the ground's frame finds its floor
        let down = crate::Ray {
            origin: crate::GridPosition::from_dvec3(
                first.ground_under(DVec3::ZERO) + DVec3::Y * 5.0,
            )
            .unwrap(),
            direction: Vec3::NEG_Y,
            max_distance: 20.0,
        };
        let floor = terrain.raycast(&world, &down).unwrap().unwrap();
        assert!((floor.distance - 7.0).abs() < 0.3, "{}", floor.distance);
        let kept = PlanetEdits::from_bytes(&kept.to_bytes()).unwrap();

        // a ground anchored 300 m off, so its frame is turned against the first
        let planet_hole = first.to_planet(first.ground_under(DVec3::ZERO));
        let second = ground(first.anchor() + first.turn().conjugate().mul_vec3(DVec3::X * 300.0));
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let there = second.to_world(planet_hole);
        let desc = region(&second, there);
        let corner = desc.origin.to_dvec3();
        let mut terrain =
            SdfTerrain::new_in(&mut world, desc, second.frame(), &Sequential).unwrap();
        let local = kept.local(&second, &desc);
        assert_eq!(local.len(), 1);
        for edit in local {
            terrain.edit(&mut world, edit).unwrap();
        }
        let again = terrain.volume().distance((there - corner).as_vec3());
        assert!((again - hole).abs() < 0.01, "{again} {hole}");
        // and a region far from it does not take it
        let away = region(&second, there + DVec3::new(200.0, 0.0, 0.0));
        assert!(kept.local(&second, &away).is_empty());
    }

    #[test]
    fn a_region_on_a_turning_planet_stands_in_its_frame_and_rays_find_it() {
        let planet = Planet::new(PlanetDesc::earth(3)).unwrap();
        let anchor = planet.surface(DVec3::new(0.3, 0.8, 0.5));
        let level = level_for(&planet, 2.5);
        let mut world = PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap();
        let (ground, frames) = PlanetGround::on_planet(&mut world, planet, anchor, level).unwrap();
        let desc = region(&ground, DVec3::ZERO);
        let terrain = SdfTerrain::new_in(&mut world, desc, frames.ground, &Sequential).unwrap();
        let top = ground.ground_under(DVec3::ZERO);
        let down = crate::Ray {
            origin: crate::GridPosition::from_dvec3(top + DVec3::Y * 5.0).unwrap(),
            direction: Vec3::NEG_Y,
            max_distance: 20.0,
        };
        let hit = terrain.raycast(&world, &down).unwrap().unwrap();
        assert!((hit.distance - 5.0).abs() < 0.1, "{}", hit.distance);
    }

    #[test]
    fn ground_prepared_away_from_the_world_is_the_same_as_made_in_it() {
        let ground = ground(DVec3::new(0.3, 0.8, 0.5));
        let desc = region(&ground, DVec3::ZERO);
        let center = (ground.ground_under(DVec3::ZERO) - desc.origin.to_dvec3()).as_vec3();
        let edits = [Edit {
            shape: EditShape::Sphere {
                center,
                radius: 2.5,
            },
            op: EditOp::Excavate,
        }];
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let mut made = SdfTerrain::new_in(&mut world, desc, ground.frame(), &Sequential).unwrap();
        for edit in edits {
            made.edit(&mut world, edit).unwrap();
        }
        let prepared = SdfTerrain::prepare(desc, &edits, ground.frame(), &Sequential).unwrap();
        let mut other = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        let attached = SdfTerrain::attach(&mut other, prepared).unwrap();
        assert_eq!(made.bodies().count(), attached.bodies().count());
        assert_eq!(
            made.volume().material_volume(),
            attached.volume().material_volume()
        );
        for (key, ..) in made.volume().surface_bricks() {
            assert_eq!(
                made.volume().brick_mesh(key),
                attached.volume().brick_mesh(key)
            );
        }
        let hole = attached.volume().distance(center);
        assert!(hole > 2.0, "{hole}");
    }

    #[test]
    fn a_box_keeps_its_turn_through_the_planet() {
        let first = ground(DVec3::new(0.3, 0.8, 0.5));
        let second = ground(DVec3::new(0.31, 0.8, 0.49));
        let corner = DVec3::new(-5.0, -2.0, 3.0);
        let edit = Edit {
            shape: EditShape::Box {
                center: Vec3::new(1.0, 2.0, 3.0),
                half_extents: Vec3::new(1.0, 0.5, 2.0),
                rotation: algebrix::Quat::from_axis_angle(Vec3::Y, 0.7),
            },
            op: EditOp::Add,
        };
        let kept = first.edit_to_planet(corner, &edit);
        let back = first.edit_to_region(corner, &kept);
        let (
            EditShape::Box {
                center, rotation, ..
            },
            EditShape::Box {
                center: c,
                rotation: r,
                ..
            },
        ) = (edit.shape, back.shape)
        else {
            panic!()
        };
        assert!((center - c).length() < 1e-4);
        assert!(rotation.dot(r).abs() > 1.0 - 1e-6);
        // seen from another ground the box's corners are the same planet points
        let other = second.edit_to_region(DVec3::ZERO, &kept);
        let EditShape::Box {
            center: o,
            rotation: q,
            ..
        } = other.shape
        else {
            panic!()
        };
        let tip = |at: Vec3, turn: algebrix::Quat, ground: &PlanetGround, corner: DVec3| {
            ground.to_planet(DVec3::from_vec3(at + turn.rotate_vec3(Vec3::X)) + corner)
        };
        let a = tip(center, rotation, &first, corner);
        let b = tip(o, q, &second, DVec3::ZERO);
        assert!((a - b).length() < 1e-2, "{:?}", a - b);
    }

    #[test]
    fn bad_stores_are_refused_and_never_panic() {
        let mut kept = PlanetEdits::new();
        for i in 0..20 {
            let at = DVec3::new(6.4e6, f64::from(i), 0.0);
            kept.push(PlanetEdit {
                shape: PlanetShape::Capsule {
                    a: at,
                    b: at + DVec3::X,
                    radius: 0.5,
                },
                op: EditOp::Tunnel,
            })
            .unwrap();
        }
        assert!(
            kept.push(PlanetEdit {
                shape: PlanetShape::Sphere {
                    center: DVec3::ZERO,
                    radius: f64::NAN,
                },
                op: EditOp::Add,
            })
            .is_err()
        );
        let bytes = kept.to_bytes();
        assert_eq!(PlanetEdits::from_bytes(&bytes).unwrap(), kept);
        assert!(PlanetEdits::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        assert!(PlanetEdits::from_bytes(b"nope").is_err());
        let mut seed = 7_u64;
        for _ in 0..2000 {
            let mut broken = bytes.clone();
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let at = (seed >> 33) as usize % broken.len();
            broken[at] ^= 1 << ((seed >> 20) % 8);
            if at < broken.len() / 2 {
                broken.truncate(broken.len() - (seed as usize % 5));
            }
            let _ = PlanetEdits::from_bytes(&broken);
        }
    }

    #[test]
    fn carved_tiles_stay_out_until_the_region_goes() {
        let mut ground = ground(DVec3::new(0.3, 0.8, 0.5));
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        ground.update(&mut world, &[DVec3::ZERO], 30.0).unwrap();
        let all = ground.tile_count();
        // tiles here are about 40 m wide
        let half = 70.0;
        let low = DVec3::new(-half, -50.0, -half);
        let high = DVec3::new(half, 50.0, half);
        ground.carve(&mut world, Some((low, high))).unwrap();
        let carved = ground.tile_count();
        assert!(carved < all, "{carved} {all}");
        ground.update(&mut world, &[DVec3::ZERO], 30.0).unwrap();
        assert_eq!(ground.tile_count(), carved);
        ground.carve(&mut world, None).unwrap();
        ground.update(&mut world, &[DVec3::ZERO], 30.0).unwrap();
        assert_eq!(ground.tile_count(), all);
        // two neighbouring halves take the same tiles as the whole, those across them too
        let halves = [
            (low, DVec3::new(0.0, high.y, high.z)),
            (DVec3::new(0.0, low.y, low.z), high),
        ];
        ground.carve_all(&mut world, &halves).unwrap();
        assert_eq!(ground.tile_count(), carved);
    }
}
