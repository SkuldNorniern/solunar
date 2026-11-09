use algebrix::{DVec3, Vec3};

use crate::error::PhysicsError;

/// Most cells one pattern may have.
pub const MAX_CELLS: usize = 256;
/// Most hull points a piece may keep, a convex shape's limit.
const MAX_POINTS: usize = 64;
/// Points closer than this, times the box size, are one point.
const MERGE: f64 = 1e-7;

/// A box cut into convex pieces that tile it with no gap or overlap, and which pieces share a
/// face and how big it is.
#[derive(Debug, Clone, PartialEq)]
pub struct FracturePattern {
    pub pieces: Vec<PatternPiece>,
    pub bonds: Vec<PatternBond>,
    pub half: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatternPiece {
    /// Hull points, from the box's middle.
    pub points: Vec<Vec3>,
    pub volume: f32,
    pub center: Vec3,
}

/// Two pieces that share a face of `area` square meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatternBond {
    pub a: u32,
    pub b: u32,
    pub area: f32,
}

/// A face and what is on its other side: `None` the box's outside, else that cell.
type Face = (Vec<DVec3>, Option<u32>);

/// Keeps the part of each face where `normal · x <= offset`, and closes the cut with a new face.
fn clip(faces: Vec<Face>, normal: DVec3, offset: f64, tag: Option<u32>, merge: f64) -> Vec<Face> {
    let side = |point: DVec3| normal.dot(point) - offset;
    let mut kept = Vec::with_capacity(faces.len() + 1);
    let mut cut: Vec<DVec3> = Vec::new();
    for (face, other) in faces {
        let mut out = Vec::with_capacity(face.len() + 1);
        for (i, &here) in face.iter().enumerate() {
            let next = face[(i + 1) % face.len()];
            let (d0, d1) = (side(here), side(next));
            if d0 <= merge {
                out.push(here);
                if d0.abs() <= merge {
                    cut.push(here);
                }
            }
            if (d0 > merge && d1 < -merge) || (d0 < -merge && d1 > merge) {
                let at = here + (next - here) * (d0 / (d0 - d1));
                out.push(at);
                cut.push(at);
            }
        }
        if out.len() >= 3 {
            kept.push((out, other));
        }
    }
    let mut unique: Vec<DVec3> = Vec::new();
    for point in cut {
        if unique.iter().all(|seen| (*seen - point).length() > merge) {
            unique.push(point);
        }
    }
    if unique.len() >= 3 {
        let middle = unique.iter().fold(DVec3::ZERO, |sum, p| sum + *p) / unique.len() as f64;
        let u = (unique[0] - middle).normalize();
        let v = normal.normalize().cross(u);
        unique.sort_by(|p, q| {
            let angle = |x: DVec3| (x - middle).dot(v).atan2((x - middle).dot(u));
            angle(*p).total_cmp(&angle(*q))
        });
        kept.push((unique, tag));
    }
    kept
}

fn area(face: &[DVec3]) -> f64 {
    let mut sum = DVec3::ZERO;
    for i in 1..face.len().saturating_sub(1) {
        sum += (face[i] - face[0]).cross(face[i + 1] - face[0]);
    }
    sum.length() * 0.5
}

/// Volume and center of a convex polytope from its faces, whichever way they wind.
fn measure(faces: &[Face]) -> (f64, DVec3) {
    let points: Vec<DVec3> = faces
        .iter()
        .flat_map(|(face, _)| face.iter().copied())
        .collect();
    let inside = points.iter().fold(DVec3::ZERO, |sum, p| sum + *p) / points.len() as f64;
    let (mut volume, mut moment) = (0.0, DVec3::ZERO);
    for (face, _) in faces {
        for i in 1..face.len() - 1 {
            let (a, b, c) = (face[0] - inside, face[i] - inside, face[i + 1] - inside);
            let tetra = a.dot(b.cross(c)).abs() / 6.0;
            volume += tetra;
            moment += (a + b + c) * (tetra / 4.0);
        }
    }
    let center = if volume > 0.0 {
        inside + moment / volume
    } else {
        inside
    };
    (volume, center)
}

/// A small fast generator, the same numbers for the same seed everywhere.
fn next(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1_u64 << 53) as f64
}

/// The box `half` from its middle cut into `cells` voronoi pieces around seeded points. The
/// same seed gives the same pieces.
///
/// # Errors
/// Fails on a box that is not finite and positive, or no cells, or more than [`MAX_CELLS`].
pub fn voronoi_box(half: Vec3, cells: usize, seed: u64) -> Result<FracturePattern, PhysicsError> {
    if !half.is_finite() || half.min_element() <= 0.0 || cells == 0 || cells > MAX_CELLS {
        return Err(PhysicsError::InvalidShape);
    }
    let h = DVec3::from_vec3(half);
    let merge = MERGE * h.length();
    let mut state = seed;
    let seeds: Vec<DVec3> = (0..cells)
        .map(|_| {
            // kept off the walls so no cell is a sliver against them
            let pick = |half: f64, state: &mut u64| (next(state) * 2.0 - 1.0) * half * 0.95;
            DVec3::new(
                pick(h.x, &mut state),
                pick(h.y, &mut state),
                pick(h.z, &mut state),
            )
        })
        .collect();
    let corner = |sx: f64, sy: f64, sz: f64| DVec3::new(sx * h.x, sy * h.y, sz * h.z);
    let box_faces = || -> Vec<Face> {
        let quad = |a, b, c, d| (vec![a, b, c, d], None);
        vec![
            quad(
                corner(1., -1., -1.),
                corner(1., 1., -1.),
                corner(1., 1., 1.),
                corner(1., -1., 1.),
            ),
            quad(
                corner(-1., -1., -1.),
                corner(-1., -1., 1.),
                corner(-1., 1., 1.),
                corner(-1., 1., -1.),
            ),
            quad(
                corner(-1., 1., -1.),
                corner(-1., 1., 1.),
                corner(1., 1., 1.),
                corner(1., 1., -1.),
            ),
            quad(
                corner(-1., -1., -1.),
                corner(1., -1., -1.),
                corner(1., -1., 1.),
                corner(-1., -1., 1.),
            ),
            quad(
                corner(-1., -1., 1.),
                corner(1., -1., 1.),
                corner(1., 1., 1.),
                corner(-1., 1., 1.),
            ),
            quad(
                corner(-1., -1., -1.),
                corner(-1., 1., -1.),
                corner(1., 1., -1.),
                corner(1., -1., -1.),
            ),
        ]
    };
    let mut pieces = Vec::with_capacity(cells);
    let mut faces_of = Vec::with_capacity(cells);
    for (i, own) in seeds.iter().enumerate() {
        let mut faces = box_faces();
        // nearer neighbors first, they cut the most, later ones often miss
        let mut others: Vec<usize> = (0..cells).filter(|j| *j != i).collect();
        others.sort_by(|a, b| {
            (seeds[*a] - *own)
                .length_squared()
                .total_cmp(&(seeds[*b] - *own).length_squared())
        });
        for j in others {
            let normal = seeds[j] - *own;
            if normal.length() <= merge {
                continue;
            }
            let offset = normal.dot((seeds[j] + *own) * 0.5);
            faces = clip(faces, normal, offset, Some(j as u32), merge);
            if faces.len() < 4 {
                break;
            }
        }
        if faces.len() < 4 {
            return Err(PhysicsError::InvalidShape);
        }
        let (volume, center) = measure(&faces);
        let mut points: Vec<DVec3> = Vec::new();
        for (face, _) in &faces {
            for point in face {
                if points.iter().all(|seen| (*seen - *point).length() > merge) {
                    points.push(*point);
                }
            }
        }
        if points.len() > MAX_POINTS {
            return Err(PhysicsError::InvalidShape);
        }
        pieces.push(PatternPiece {
            points: points.iter().map(|p| p.as_vec3()).collect(),
            volume: volume as f32,
            center: center.as_vec3(),
        });
        faces_of.push(faces);
    }
    let mut bonds = Vec::new();
    for (i, faces) in faces_of.iter().enumerate() {
        for (face, other) in faces {
            let Some(j) = *other else {
                continue;
            };
            let face_area = area(face);
            if (j as usize) > i && face_area > merge * merge {
                bonds.push(PatternBond {
                    a: i as u32,
                    b: j,
                    area: face_area as f32,
                });
            }
        }
    }
    Ok(FracturePattern {
        pieces,
        bonds,
        half,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_tile_the_box_and_bond_where_they_touch() {
        for (half, cells, seed) in [
            (Vec3::new(1.0, 1.0, 1.0), 1, 1),
            (Vec3::new(1.0, 1.0, 1.0), 12, 7),
            (Vec3::new(2.0, 1.5, 0.15), 40, 3),
            (Vec3::new(0.5, 0.5, 0.5), 96, 11),
        ] {
            let pattern = voronoi_box(half, cells, seed).unwrap();
            assert_eq!(pattern.pieces.len(), cells);
            let whole = 8.0 * f64::from(half.x * half.y * half.z);
            let sum: f64 = pattern.pieces.iter().map(|p| f64::from(p.volume)).sum();
            assert!((sum - whole).abs() < whole * 1e-4, "{sum} of {whole}");
            for piece in &pattern.pieces {
                assert!(piece.volume > 0.0);
                for point in &piece.points {
                    let room = half * 1.0001;
                    let inside = point.x.abs() <= room.x
                        && point.y.abs() <= room.y
                        && point.z.abs() <= room.z;
                    assert!(inside, "{point:?} outside {half:?}");
                }
            }
            if cells > 1 {
                // every piece holds on to something
                for i in 0..cells as u32 {
                    assert!(
                        pattern.bonds.iter().any(|b| b.a == i || b.b == i),
                        "{i} alone"
                    );
                }
            }
            for bond in &pattern.bonds {
                assert!(bond.a < bond.b && bond.area > 0.0);
            }
        }
        assert_eq!(voronoi_box(Vec3::ONE, 12, 7), voronoi_box(Vec3::ONE, 12, 7));
        assert_ne!(voronoi_box(Vec3::ONE, 12, 7), voronoi_box(Vec3::ONE, 12, 8));
        assert!(voronoi_box(Vec3::new(1.0, 0.0, 1.0), 4, 1).is_err());
        assert!(voronoi_box(Vec3::ONE, 0, 1).is_err());
        assert!(voronoi_box(Vec3::ONE, MAX_CELLS + 1, 1).is_err());
    }
}
