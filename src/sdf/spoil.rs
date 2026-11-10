use algebrix::{DVec3, Quat, Vec3};

use super::{Edit, EditOp, EditReport, EditShape, SdfTerrain};
use crate::dispatch::{JobDispatch, Sequential};
use crate::error::PhysicsError;
use crate::fracture::{
    BreakableDesc, BreakableHandle, Debris, FractureMaterial, FracturePattern, MAX_CELLS,
    PatternBond, PatternPiece, voronoi_box_within,
};
use crate::material::Material;
use crate::world::PhysicsWorld;
use crate::{QueryFilter, Shape};

/// Samples a side when guessing how much solid a dig takes, for its piece count.
const GUESS_SIDE: usize = 3;
/// Halvings when finding how far a piece may grow and stay in the hole.
const FIT_STEPS: usize = 10;
/// Gap kept to the hole's wall, in voxels: the wall's mesh lies within this of the edit.
const WALL_GAP: f32 = 0.1;
/// A piece shrunk below this share of its size is left as loose ground.
const SMALLEST: f32 = 0.3;

/// What digging leaves of the ground it takes out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Spoil {
    /// Soil, sand and snow: counted as mass only, no bodies.
    Loose,
    /// Rock and ice: what comes out is one bonded mass cut by the material's own pattern,
    /// which cracks apart by its strength as it falls and is hit, as any breakable. Its density
    /// is the ground's, the rest comes from `material`.
    Fragments {
        material: FractureMaterial,
        debris: Option<Debris>,
    },
}

/// What a dig did.
#[derive(Debug, Clone, PartialEq)]
pub struct Dig {
    pub report: EditReport,
    /// The pieces dug out, none for loose ground or when nothing solid was taken.
    pub spoil: Option<BreakableHandle>,
    /// Kilograms in the pieces.
    pub fragment_mass: f64,
    /// Kilograms of what was taken that is not in a piece: loose ground, the rim the pieces
    /// had to leave so they never start inside the walls. Never below zero.
    pub loose_mass: f64,
}

impl SdfTerrain {
    /// Excavates `edit` and turns what it took into `spoil`. Pieces stand where the rock was,
    /// each wholly inside the hole and inside the old ground, so none starts overlapping what
    /// stays. The same ground, edit and `seed` give the same pieces.
    ///
    /// # Errors
    /// Fails on an edit that is not an excavation, a bad shape, with other edits waiting, or
    /// when the world will not take a brick or the pieces. The hole stays when only the pieces
    /// fail.
    pub fn dig(
        &mut self,
        world: &mut PhysicsWorld,
        edit: Edit,
        spoil: &Spoil,
        seed: u64,
    ) -> Result<Dig, PhysicsError> {
        self.dig_with(world, edit, spoil, seed, &Sequential)
    }

    /// [`Self::dig`] with its bricks rebuilt on the host's workers. The result does not depend
    /// on them.
    ///
    /// # Errors
    /// See [`Self::dig`], and a dispatcher that skips jobs.
    pub fn dig_with(
        &mut self,
        world: &mut PhysicsWorld,
        edit: Edit,
        spoil: &Spoil,
        seed: u64,
        dispatch: &dyn JobDispatch,
    ) -> Result<Dig, PhysicsError> {
        if edit.op != EditOp::Excavate {
            return Err(PhysicsError::InvalidShape);
        }
        edit.shape.validate()?;
        let pieces = match *spoil {
            Spoil::Loose => None,
            Spoil::Fragments { material, debris } => {
                self.spoil_of(world, &edit, &material, debris, seed)?
            }
        };
        let report = self.edit_with(world, edit, dispatch)?;
        let (mut spoil, mut fragment_mass) = (None, 0.0);
        if let Some(desc) = pieces {
            let volume: f64 = desc
                .pattern
                .pieces
                .iter()
                .map(|piece| f64::from(piece.volume))
                .sum();
            fragment_mass = volume * f64::from(desc.material.density);
            spoil = Some(world.add_breakable(desc)?);
        }
        Ok(Dig {
            loose_mass: (report.removed_mass - fragment_mass).max(0.0),
            report,
            spoil,
            fragment_mass,
        })
    }

    /// The pieces `edit` would dig out of the ground as it is now, before it is done.
    fn spoil_of(
        &self,
        world: &PhysicsWorld,
        edit: &Edit,
        kind: &FractureMaterial,
        debris: Option<Debris>,
        seed: u64,
    ) -> Result<Option<BreakableDesc>, PhysicsError> {
        let volume = self.volume();
        let (low, high) = edit.bounds();
        let (middle, half) = ((low + high) * 0.5, (high - low) * 0.5);
        let shape = edit.shape;
        let margin = volume.desc().voxel * WALL_GAP;
        let in_hole = |p: Vec3| shape.distance(p) <= -margin;
        let taken = |p: Vec3| in_hole(p) && volume.distance(p) <= 0.0;
        let mut inside = 0;
        for i in 0..GUESS_SIDE.pow(3) {
            let cell = |k: usize| (k as f32 + 0.5) / GUESS_SIDE as f32 * 2.0 - 1.0;
            let at = (
                i % GUESS_SIDE,
                i / GUESS_SIDE % GUESS_SIDE,
                i / GUESS_SIDE / GUESS_SIDE,
            );
            if taken(middle + half * Vec3::new(cell(at.0), cell(at.1), cell(at.2))) {
                inside += 1;
            }
        }
        if inside == 0 {
            return Ok(None);
        }
        let solid = 8.0 * half.x * half.y * half.z * inside as f32 / GUESS_SIDE.pow(3) as f32;
        let cells = ((solid * kind.pieces_per_cubic_meter).round() as usize)
            .clamp(kind.fewest_pieces.max(1), MAX_CELLS);
        let pattern = voronoi_box_within(half, cells, seed, &hole_planes(&shape, middle, margin))?;
        // each piece shrunk about its middle until it lies wholly in the hole
        let mut kept: Vec<(usize, f32)> = Vec::new();
        for (index, piece) in pattern.pieces.iter().enumerate() {
            let at = |p: Vec3, scale: f32| middle + piece.center + (p - piece.center) * scale;
            if !taken(at(piece.center, 0.0)) {
                continue;
            }
            // each corner as far out as it stays in the hole, the piece by the nearest one;
            // the hole's shape first, it is cheap, the ground only where the shape allows
            let fit = |inside: &dyn Fn(Vec3) -> bool, mut scale: f32| {
                for p in &piece.points {
                    if inside(at(*p, scale)) {
                        continue;
                    }
                    let (mut fits, mut out) = (0.0, scale);
                    for _ in 0..FIT_STEPS {
                        let mid = (fits + out) * 0.5;
                        if inside(at(*p, mid)) {
                            fits = mid;
                        } else {
                            out = mid;
                        }
                    }
                    scale = fits;
                }
                scale
            };
            let mut scale = fit(&in_hole, 1.0);
            // the ground's distance never says more than the truth, so a middle deeper than
            // the piece reaches holds all of it; only pieces near the top check each corner
            let reach = piece
                .points
                .iter()
                .map(|p| (*p - piece.center).length())
                .fold(0.0, f32::max)
                * scale;
            let deep = volume.distance(at(piece.center, 0.0)) < -reach;
            if !deep && !piece.points.iter().all(|p| taken(at(*p, scale))) {
                scale = fit(&taken, scale);
            }
            // shrinking for one corner may pull another back in, never out
            if scale >= SMALLEST
                && (deep || piece.points.iter().all(|p| taken(at(*p, scale))))
                && !self.crowded(world, middle + piece.center, &piece.points, scale)?
            {
                kept.push((index, scale));
            }
        }
        if kept.is_empty() {
            return Ok(None);
        }
        let mut renumber = vec![None; pattern.pieces.len()];
        let pieces: Vec<PatternPiece> = kept
            .iter()
            .enumerate()
            .map(|(new, &(old, scale))| {
                renumber[old] = Some((new as u32, scale));
                let piece = &pattern.pieces[old];
                PatternPiece {
                    points: piece
                        .points
                        .iter()
                        .map(|p| piece.center + (*p - piece.center) * scale)
                        .collect(),
                    volume: piece.volume * scale * scale * scale,
                    center: piece.center,
                }
            })
            .collect();
        let bonds = pattern
            .bonds
            .iter()
            .filter_map(|bond| {
                let (a, sa) = renumber[bond.a as usize]?;
                let (b, sb) = renumber[bond.b as usize]?;
                let scale = sa.min(sb);
                Some(PatternBond {
                    a: a.min(b),
                    b: a.max(b),
                    area: bond.area * scale * scale,
                    normal: if a < b { bond.normal } else { -bond.normal },
                })
            })
            .collect();
        let position = volume
            .desc()
            .origin
            .shifted(DVec3::from_vec3(middle))
            .ok_or(PhysicsError::OutOfRange)?;
        let pattern = FracturePattern {
            pieces,
            bonds,
            half,
        };
        Ok(Some(BreakableDesc {
            frame: self.frame(),
            material: Material {
                density: volume.desc().material.density,
                friction: kind.friction,
                restitution: kind.restitution,
                ..Material::default()
            },
            strength: kind.strength,
            crack_spread: kind.crack_spread,
            debris,
            // dug rock breaks on down to an eighth of the material's own piece
            smallest_piece: 1.0 / (8.0 * kind.pieces_per_cubic_meter),
            ..BreakableDesc::new(pattern, position)
        }))
    }
}

impl SdfTerrain {
    /// Whether a piece of `points` about `center`, shrunk to `scale`, would start touching a
    /// body that is not ground, as pieces of an earlier dig lying in the hole. Its bounding
    /// ball is asked, so near counts as touching.
    fn crowded(
        &self,
        world: &PhysicsWorld,
        center: Vec3,
        points: &[Vec3],
        scale: f32,
    ) -> Result<bool, PhysicsError> {
        let middle = points.iter().fold(Vec3::ZERO, |sum, p| sum + *p) / points.len() as f32;
        let reach = points
            .iter()
            .map(|p| (*p - middle).length())
            .fold(0.0, f32::max)
            * scale;
        let position = self
            .volume()
            .desc()
            .origin
            .shifted(DVec3::from_vec3(center))
            .ok_or(PhysicsError::OutOfRange)?;
        let filter = QueryFilter {
            frame: self.frame(),
            ..QueryFilter::default()
        };
        let mut near = Vec::new();
        world.overlap(
            &Shape::sphere(reach.max(1e-3)),
            position,
            Quat::IDENTITY,
            &filter,
            &mut near,
        )?;
        Ok(near.iter().any(|body| !self.is_brick(*body)))
    }
}

/// Planes around `shape`, `margin` inside it, from `middle`: a box's own faces, a ball or a
/// capsule by planes touching it in 14 directions. The corners between those reach a little
/// past a round shape, the fit after trims them.
fn hole_planes(shape: &EditShape, middle: Vec3, margin: f32) -> Vec<(DVec3, f64)> {
    let plane = |normal: Vec3, reach: f32| (DVec3::from_vec3(normal), f64::from(reach - margin));
    let round = |a: Vec3, b: Vec3, radius: f32| {
        let mut planes = Vec::with_capacity(14);
        for i in 0..27 {
            let step = |k: usize| (k % 3) as f32 - 1.0;
            let direction = Vec3::new(step(i), step(i / 3), step(i / 9));
            // the axes and the corners, edges between them add corners to pieces for little
            let zeros = [direction.x, direction.y, direction.z]
                .iter()
                .filter(|v| **v == 0.0)
                .count();
            if zeros == 3 || zeros == 1 {
                continue;
            }
            let normal = direction.normalize();
            let reach = normal.dot(a - middle).max(normal.dot(b - middle)) + radius;
            planes.push(plane(normal, reach));
        }
        planes
    };
    match *shape {
        EditShape::Sphere { center, radius } => round(center, center, radius),
        EditShape::Capsule { a, b, radius } => round(a, b, radius),
        EditShape::Box {
            center,
            half_extents,
            rotation,
        } => {
            let turn = rotation.normalize();
            let mut planes = Vec::with_capacity(6);
            for (axis, half) in [
                (Vec3::X, half_extents.x),
                (Vec3::Y, half_extents.y),
                (Vec3::Z, half_extents.z),
            ] {
                for sign in [1.0, -1.0] {
                    let normal = turn.rotate_vec3(axis * sign);
                    planes.push(plane(normal, normal.dot(center - middle) + half));
                }
            }
            planes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::testing::world;
    use crate::{Base, EditShape, GridPosition, SdfDesc};

    fn ground(world: &mut PhysicsWorld) -> SdfTerrain {
        let desc = SdfDesc {
            origin: GridPosition::ORIGIN,
            voxel: 0.5,
            brick_cells: 8,
            bricks: [4, 3, 4],
            base: Base::Hills {
                seed: 11,
                height: 8.0,
                amplitude: 1.0,
                wavelength: 14.0,
                octaves: 2,
            },
            material: Material {
                density: 2600.0,
                ..Material::default()
            },
        };
        SdfTerrain::new(world, desc).unwrap()
    }

    fn bore() -> Edit {
        Edit {
            shape: EditShape::Sphere {
                center: Vec3::new(8.0, 6.5, 8.0),
                radius: 1.2,
            },
            op: EditOp::Excavate,
        }
    }

    fn rock(strength: f32) -> Spoil {
        Spoil::Fragments {
            material: FractureMaterial {
                strength,
                ..FractureMaterial::STONE
            },
            debris: None,
        }
    }

    #[test]
    fn rock_comes_out_in_pieces_inside_the_hole_and_the_mass_adds_up() {
        let mut world = world();
        let mut terrain = ground(&mut world);
        let dig = terrain.dig(&mut world, bore(), &rock(2.0e6), 7).unwrap();
        let spoil = dig.spoil.unwrap();
        assert!(dig.report.removed_mass > 0.0);
        assert!(dig.fragment_mass > 0.3 * dig.report.removed_mass, "{dig:?}");
        assert!(
            dig.fragment_mass <= dig.report.removed_mass * 1.02,
            "{dig:?}"
        );
        let sum = dig.fragment_mass + dig.loose_mass;
        assert!(sum >= dig.report.removed_mass - 1e-6, "{dig:?}");
        // nothing starts inside the walls, so the first step throws nothing out
        world.step_fixed().unwrap();
        for body in world.breakable_bodies(spoil) {
            let state = world.body_state(body).unwrap();
            assert!(state.linear_velocity.length() < 0.5, "{state:?}");
        }
    }

    #[test]
    fn loose_ground_is_mass_only_and_the_same_dig_is_the_same_pieces() {
        let mut world = world();
        let mut terrain = ground(&mut world);
        let dig = terrain.dig(&mut world, bore(), &Spoil::Loose, 7).unwrap();
        assert!(dig.spoil.is_none() && dig.fragment_mass == 0.0);
        assert_eq!(dig.loose_mass, dig.report.removed_mass);
        let pieces = |seed: u64| {
            let mut fresh = crate::world::testing::world();
            let terrain = ground(&mut fresh);
            let Spoil::Fragments { material, debris } = rock(2.0e6) else {
                unreachable!()
            };
            terrain
                .spoil_of(&fresh, &bore(), &material, debris, seed)
                .unwrap()
                .unwrap()
                .pattern
        };
        assert_eq!(pieces(3), pieces(3));
        assert_ne!(pieces(3), pieces(4));
        let tunnel = Edit {
            op: EditOp::Tunnel,
            ..bore()
        };
        assert_eq!(
            terrain.dig(&mut world, tunnel, &Spoil::Loose, 7),
            Err(PhysicsError::InvalidShape)
        );
    }

    #[test]
    fn a_dig_on_workers_is_the_same_as_on_one_thread() {
        let run = |dispatch: &dyn JobDispatch| {
            let mut world = world();
            let mut terrain = ground(&mut world);
            let dig = terrain
                .dig_with(&mut world, bore(), &rock(2.0e6), 7, dispatch)
                .unwrap();
            (dig, world.save_snapshot())
        };
        let (one, one_world) = run(&Sequential);
        let (many, many_world) = run(&crate::dispatch::testing::Threads(4));
        assert_eq!(one, many);
        assert_eq!(one_world, many_world);
    }

    #[test]
    fn weak_rock_crumbles_where_strong_rock_stays_in_lumps() {
        let parts = |strength: f32| {
            let mut world = world();
            let mut terrain = ground(&mut world);
            let dig = terrain.dig(&mut world, bore(), &rock(strength), 7).unwrap();
            for _ in 0..240 {
                world.step_fixed().unwrap();
            }
            world.breakable_bodies(dig.spoil.unwrap()).len()
        };
        let (strong, weak) = (parts(2.0e7), parts(5.0e2));
        assert!(weak > strong, "{weak} {strong}");
    }
}
