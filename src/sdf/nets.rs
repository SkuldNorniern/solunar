use algebrix::Vec3;

use crate::collision::trimesh::MIN_TRIANGLE_CROSS;

/// Triangles for one brick, in its own space: the brick's low corner is the origin. Front
/// faces point out of the material.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BrickMesh {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

impl BrickMesh {
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }
}

const NONE: u32 = u32::MAX;

/// Corner pairs of a cube's 12 edges, corners numbered by x + 2y + 4z.
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// `values` holds `side`³ samples, x fastest, sample `i` at `(i - 1) * voxel` from the brick
/// corner. The brick owns the lattice edges that start at samples 1 to `side - 2`, so
/// neighbors never make the same face twice and meet with no gap.
pub(crate) fn surface_nets(values: &[f32], side: usize, voxel: f32) -> BrickMesh {
    let cells = side - 1;
    let at = |x: usize, y: usize, z: usize| values[x + side * (y + side * z)];
    let corner = |x: usize, y: usize, z: usize| {
        Vec3::new(x as f32 - 1.0, y as f32 - 1.0, z as f32 - 1.0) * voxel
    };

    let mut cell_vertex = vec![NONE; cells * cells * cells];
    let mut mesh = BrickMesh::default();
    let mut vertex_of = |x: usize, y: usize, z: usize, mesh: &mut BrickMesh| -> u32 {
        let slot = &mut cell_vertex[x + cells * (y + cells * z)];
        if *slot != NONE {
            return *slot;
        }
        let mut d = [0.0_f32; 8];
        for (index, value) in d.iter_mut().enumerate() {
            *value = at(x + (index & 1), y + ((index >> 1) & 1), z + (index >> 2));
        }
        let mut sum = Vec3::ZERO;
        let mut count = 0.0;
        for (a, b) in EDGES {
            if (d[a] < 0.0) != (d[b] < 0.0) {
                let t = d[a] / (d[a] - d[b]);
                let pa = Vec3::new((a & 1) as f32, ((a >> 1) & 1) as f32, (a >> 2) as f32);
                let pb = Vec3::new((b & 1) as f32, ((b >> 1) & 1) as f32, (b >> 2) as f32);
                sum += pa + (pb - pa) * t;
                count += 1.0;
            }
        }
        let local = sum / count;
        // trilinear gradient of the cell at the vertex
        let (u, v, w) = (local.x, local.y, local.z);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let gx = lerp(
            lerp(d[1] - d[0], d[3] - d[2], v),
            lerp(d[5] - d[4], d[7] - d[6], v),
            w,
        );
        let gy = lerp(
            lerp(d[2] - d[0], d[3] - d[1], u),
            lerp(d[6] - d[4], d[7] - d[5], u),
            w,
        );
        let gz = lerp(
            lerp(d[4] - d[0], d[5] - d[1], u),
            lerp(d[6] - d[2], d[7] - d[3], u),
            v,
        );
        let normal = Vec3::new(gx, gy, gz).normalize_or(Vec3::Y);
        let id = mesh.vertices.len() as u32;
        mesh.vertices.push(corner(x, y, z) + local * voxel);
        mesh.normals.push(normal);
        *slot = id;
        id
    };

    let min_area = voxel * voxel * 1e-6;
    for z in 1..side - 1 {
        for y in 1..side - 1 {
            for x in 1..side - 1 {
                let here = at(x, y, z) < 0.0;
                // (axis step, u step, v step) with u x v along the axis
                let axes = [
                    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
                    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
                ];
                for (step, u, v) in axes {
                    let there = at(x + step.0, y + step.1, z + step.2) < 0.0;
                    if here == there {
                        continue;
                    }
                    let cell = |du: usize, dv: usize| {
                        (
                            x - u.0 * (1 - du) - v.0 * (1 - dv),
                            y - u.1 * (1 - du) - v.1 * (1 - dv),
                            z - u.2 * (1 - du) - v.2 * (1 - dv),
                        )
                    };
                    let mut quad = [0u32; 4];
                    for (slot, (du, dv)) in quad.iter_mut().zip([(0, 0), (1, 0), (1, 1), (0, 1)]) {
                        let (cx, cy, cz) = cell(du, dv);
                        *slot = vertex_of(cx, cy, cz, &mut mesh);
                    }
                    if !here {
                        quad.reverse();
                    }
                    let p = |i: u32| mesh.vertices[i as usize];
                    let split_02 = (p(quad[0]) - p(quad[2])).length_squared()
                        <= (p(quad[1]) - p(quad[3])).length_squared();
                    let pair = if split_02 {
                        [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
                    } else {
                        [[quad[0], quad[1], quad[3]], [quad[1], quad[2], quad[3]]]
                    };
                    for triangle in pair {
                        let [a, b, c] = triangle.map(p);
                        // under the mesh's own floor at a fine voxel, a sliver failed the edit
                        let cross = (b - a).cross(c - a).length();
                        if cross * 0.5 > min_area && cross > MIN_TRIANGLE_CROSS {
                            mesh.triangles.push(triangle);
                        }
                    }
                }
            }
        }
    }
    compact(mesh)
}

/// Drops vertices no kept triangle uses.
fn compact(mesh: BrickMesh) -> BrickMesh {
    let mut map = vec![NONE; mesh.vertices.len()];
    let mut out = BrickMesh::default();
    for triangle in &mesh.triangles {
        let mut kept = [0; 3];
        for (slot, index) in kept.iter_mut().zip(triangle) {
            let mapped = &mut map[*index as usize];
            if *mapped == NONE {
                *mapped = out.vertices.len() as u32;
                out.vertices.push(mesh.vertices[*index as usize]);
                out.normals.push(mesh.normals[*index as usize]);
            }
            *slot = *mapped;
        }
        out.triangles.push(kept);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn sphere(side: usize, voxel: f32, center: Vec3, radius: f32) -> Vec<f32> {
        let mut values = Vec::with_capacity(side * side * side);
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let p = Vec3::new(x as f32 - 1.0, y as f32 - 1.0, z as f32 - 1.0) * voxel;
                    values.push((p - center).length() - radius);
                }
            }
        }
        values
    }

    #[test]
    fn a_sphere_inside_one_brick_is_closed_and_faces_out() {
        let (side, voxel) = (18, 0.5);
        let center = Vec3::splat(4.0);
        let mesh = surface_nets(&sphere(side, voxel, center, 2.6), side, voxel);
        assert!(!mesh.is_empty());
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        for [a, b, c] in &mesh.triangles {
            for (from, to) in [(*a, *b), (*b, *c), (*c, *a)] {
                *edges.entry((from.min(to), from.max(to))).or_default() +=
                    if from < to { 1 } else { -1 };
            }
        }
        // closed and consistently wound: every edge used once each way
        assert!(edges.values().all(|count| *count == 0));
        for (index, vertex) in mesh.vertices.iter().enumerate() {
            let out = (*vertex - center).normalize();
            assert!(((*vertex - center).length() - 2.6).abs() < voxel * 0.5);
            assert!(mesh.normals[index].dot(out) > 0.8);
        }
        for [a, b, c] in &mesh.triangles {
            let (pa, pb, pc) = (
                mesh.vertices[*a as usize],
                mesh.vertices[*b as usize],
                mesh.vertices[*c as usize],
            );
            let facing = (pb - pa).cross(pc - pa);
            assert!(facing.dot((pa + pb + pc) / 3.0 - center) > 0.0);
        }
    }

    #[test]
    fn no_crossing_gives_no_triangles() {
        let side = 10;
        assert!(surface_nets(&vec![1.0; side * side * side], side, 1.0).is_empty());
        assert!(surface_nets(&vec![-1.0; side * side * side], side, 1.0).is_empty());
    }
}
