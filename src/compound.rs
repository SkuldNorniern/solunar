use algebrix::{DQuat, DVec3, Mat3, Quat, Vec3};

use crate::collision::trimesh::{Aabb, TriMesh, triangle_contact};
use crate::collision::{Bounds, RawManifold, contacts};
use crate::{GridPosition, MassProperties, PhysicsError, PhysicsWorld, Shape};

pub(crate) const MAX_COMPOUND_CHILDREN: usize = 1 << 16;
const JACOBI_SWEEPS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CompoundId(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshId(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompoundChild {
    pub shape: Shape,
    pub position: Vec3,
    pub rotation: Quat,
    /// Times the body's density, for parts of a heavier material.
    pub density_scale: f32,
}

impl CompoundChild {
    pub fn new(shape: Shape, position: Vec3) -> Self {
        Self {
            shape,
            position,
            rotation: Quat::IDENTITY,
            density_scale: 1.0,
        }
    }
}

/// The solver needs a body centered on its mass with diagonal inertia, so a compound is
/// rebased. Something placed at `at` turned by `turn` becomes a body at `at + turn * center`,
/// turned by `turn * axes`. [`Self::placed`] does this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompoundInfo {
    pub shape: Shape,
    pub center: Vec3,
    pub axes: Quat,
}

impl CompoundInfo {
    pub fn placed(&self, at: Vec3, turn: Quat) -> (Vec3, Quat) {
        (
            at + turn.rotate_vec3(self.center),
            (turn * self.axes).normalize(),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Compound {
    pub(crate) children: Vec<CompoundChild>,
    /// At density 1.
    pub(crate) unit: MassProperties,
    box_center: Vec3,
    box_half: Vec3,
    radius: f32,
}

impl Compound {
    /// Children are already rebased here. Snapshot loading comes through this too.
    pub(crate) fn from_rebased(
        children: Vec<CompoundChild>,
        unit: MassProperties,
    ) -> Result<Self, PhysicsError> {
        if children.is_empty() || children.len() > MAX_COMPOUND_CHILDREN {
            return Err(PhysicsError::InvalidShape);
        }
        let mut low = Vec3::splat(f32::INFINITY);
        let mut high = Vec3::splat(f32::NEG_INFINITY);
        let mut radius = 0.0_f32;
        for child in &children {
            validate_child(child)?;
            let reach = child.shape.bounding_radius();
            low = low.min(child.position - Vec3::splat(reach));
            high = high.max(child.position + Vec3::splat(reach));
            radius = radius.max(child.position.length() + reach);
        }
        if !radius.is_finite()
            || !unit.mass.is_finite()
            || unit.mass <= 0.0
            || !unit.inertia.is_finite()
            || unit.inertia.min_element() < 0.0
        {
            return Err(PhysicsError::InvalidShape);
        }
        Ok(Self {
            children,
            unit,
            box_center: (low + high) * 0.5,
            box_half: (high - low) * 0.5,
            radius,
        })
    }
}

fn validate_child(child: &CompoundChild) -> Result<(), PhysicsError> {
    let squared = child.rotation.length_squared();
    let good = !matches!(child.shape, Shape::Compound(_) | Shape::TriMesh(_))
        && child.position.is_finite()
        && squared.is_finite()
        && squared > 1e-12
        && child.density_scale.is_finite()
        && child.density_scale > 0.0;
    if !good {
        return Err(PhysicsError::InvalidShape);
    }
    child.shape.validate()
}

/// Shared shape data. Bodies name it by id, so `Shape` stays small and copyable.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ShapeAssets {
    pub(crate) compounds: Vec<Compound>,
    pub(crate) meshes: Vec<TriMesh>,
}

impl ShapeAssets {
    fn compound(&self, id: CompoundId) -> Option<&Compound> {
        self.compounds.get(id.0 as usize)
    }

    pub(crate) fn mesh(&self, shape: &Shape) -> Option<&TriMesh> {
        match *shape {
            Shape::TriMesh(id) => self.meshes.get(id.0 as usize),
            _ => None,
        }
    }

    pub(crate) fn validate(&self, shape: &Shape) -> Result<(), PhysicsError> {
        match *shape {
            Shape::TriMesh(_) => self
                .mesh(shape)
                .map(|_| ())
                .ok_or(PhysicsError::InvalidShape),
            Shape::Compound(id) => self
                .compound(id)
                .map(|_| ())
                .ok_or(PhysicsError::InvalidShape),
            _ => shape.validate(),
        }
    }

    pub(crate) fn mass_properties(&self, shape: &Shape, density: f32) -> MassProperties {
        match *shape {
            Shape::Compound(id) => self.compound(id).map_or(
                MassProperties {
                    mass: 0.0,
                    inertia: Vec3::ZERO,
                },
                |compound| MassProperties {
                    mass: compound.unit.mass * density,
                    inertia: compound.unit.inertia * density,
                },
            ),
            _ => shape.mass_properties(density),
        }
    }

    pub(crate) fn bounding_radius(&self, shape: &Shape) -> f32 {
        match *shape {
            Shape::Compound(id) => self
                .compound(id)
                .map_or(f32::INFINITY, |compound| compound.radius),
            Shape::TriMesh(_) => self.mesh(shape).map_or(f32::INFINITY, |mesh| mesh.radius),
            _ => shape.bounding_radius(),
        }
    }

    pub(crate) fn bounds(
        &self,
        shape: &Shape,
        position: GridPosition,
        rotation: Quat,
        margin: f32,
    ) -> Bounds {
        let local = match *shape {
            Shape::Compound(id) => self
                .compound(id)
                .map(|compound| (compound.box_center, compound.box_half)),
            Shape::TriMesh(_) => self.mesh(shape).map(|mesh| {
                (
                    (mesh.bounds.min + mesh.bounds.max) * 0.5,
                    (mesh.bounds.max - mesh.bounds.min) * 0.5,
                )
            }),
            _ => None,
        };
        if let Some((box_center, half)) = local {
            let extent = rotation.rotate_vec3(Vec3::X * half.x).abs()
                + rotation.rotate_vec3(Vec3::Y * half.y).abs()
                + rotation.rotate_vec3(Vec3::Z * half.z).abs();
            let center = position.to_dvec3() + DVec3::from_vec3(rotation.rotate_vec3(box_center));
            let extent = DVec3::from_vec3(extent + Vec3::splat(margin * 0.5));
            return Bounds {
                min: center - extent,
                max: center + extent,
            };
        }
        Bounds::of(shape, position, rotation, margin)
    }

    /// Calls `visit` with the part index, shape, offset from the body and rotation of each part.
    pub(crate) fn for_each_part(
        &self,
        shape: &Shape,
        rotation: Quat,
        mut visit: impl FnMut(u32, &Shape, Vec3, Quat),
    ) {
        match *shape {
            Shape::Compound(id) => {
                if let Some(compound) = self.compound(id) {
                    for (index, child) in compound.children.iter().enumerate() {
                        visit(
                            index as u32,
                            &child.shape,
                            rotation.rotate_vec3(child.position),
                            (rotation * child.rotation).normalize(),
                        );
                    }
                }
            }
            Shape::TriMesh(_) => {}
            _ => visit(0, shape, Vec3::ZERO, rotation),
        }
    }

    /// Every touching part pair of two bodies. `center_b` goes from A to B.
    /// `visit` gets the part indexes, the manifold and the offset of A's part from A.
    pub(crate) fn for_each_contact(
        &self,
        (shape_a, rotation_a): (&Shape, Quat),
        (shape_b, rotation_b): (&Shape, Quat),
        center_b: Vec3,
        margin: f32,
        mut visit: impl FnMut(u32, u32, &RawManifold, Vec3),
    ) {
        if let Some(mesh) = self.mesh(shape_a) {
            let inverse = rotation_a.conjugate();
            self.for_each_part(shape_b, rotation_b, |part_b, part, at_b, turn_b| {
                let center = inverse.rotate_vec3(center_b + at_b);
                let turn = inverse * turn_b;
                let area = Aabb::around(center, part.bounding_radius() + margin);
                mesh.for_each_near(area, |triangle| {
                    if let Some(raw) = triangle_contact(mesh, triangle, part, turn, center, margin)
                    {
                        let mut world = RawManifold::new(rotation_a.rotate_vec3(raw.normal));
                        for point in &raw.points[..raw.count] {
                            world.push(rotation_a.rotate_vec3(point.position), point.depth);
                        }
                        visit(triangle, part_b, &world, Vec3::ZERO);
                    }
                });
            });
            return;
        }
        if let Some(mesh) = self.mesh(shape_b) {
            let inverse = rotation_b.conjugate();
            self.for_each_part(shape_a, rotation_a, |part_a, part, at_a, turn_a| {
                let center = inverse.rotate_vec3(at_a - center_b);
                let turn = inverse * turn_a;
                let area = Aabb::around(center, part.bounding_radius() + margin);
                mesh.for_each_near(area, |triangle| {
                    if let Some(raw) = triangle_contact(mesh, triangle, part, turn, center, margin)
                    {
                        let mut world = RawManifold::new(-rotation_b.rotate_vec3(raw.normal));
                        for point in &raw.points[..raw.count] {
                            let position = rotation_b.rotate_vec3(point.position) + center_b - at_a;
                            world.push(position, point.depth);
                        }
                        visit(part_a, triangle, &world, at_a);
                    }
                });
            });
            return;
        }
        self.for_each_part(shape_a, rotation_a, |part_a, shape_a, at_a, turn_a| {
            let reach_a = shape_a.bounding_radius();
            self.for_each_part(shape_b, rotation_b, |part_b, shape_b, at_b, turn_b| {
                let center = center_b + at_b - at_a;
                let reach = reach_a + shape_b.bounding_radius() + margin;
                if center.length_squared() > reach * reach {
                    return;
                }
                if let Some(raw) = contacts(shape_a, turn_a, shape_b, turn_b, center, margin) {
                    visit(part_a, part_b, &raw, at_a);
                }
            });
        });
    }

    pub(crate) fn any_contact(
        &self,
        a: (&Shape, Quat),
        b: (&Shape, Quat),
        center_b: Vec3,
        margin: f32,
    ) -> bool {
        let mut found = false;
        self.for_each_contact(a, b, center_b, margin, |_, _, _, _| found = true);
        found
    }
}

/// Eigenvalues and right handed eigenvector columns of a symmetric matrix (cyclic Jacobi).
fn principal_axes(mut matrix: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..JACOBI_SWEEPS {
        let off = matrix[0][1].abs() + matrix[0][2].abs() + matrix[1][2].abs();
        let scale = matrix[0][0].abs() + matrix[1][1].abs() + matrix[2][2].abs();
        if off <= scale * 1e-15 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if matrix[p][q] == 0.0 {
                continue;
            }
            let theta = (matrix[q][q] - matrix[p][p]) / (2.0 * matrix[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = (t * t + 1.0).sqrt().recip();
            let s = t * c;
            for row in &mut matrix {
                let (one, two) = (row[p], row[q]);
                row[p] = c * one - s * two;
                row[q] = s * one + c * two;
            }
            let (row_p, row_q) = (matrix[p], matrix[q]);
            for k in 0..3 {
                matrix[p][k] = c * row_p[k] - s * row_q[k];
                matrix[q][k] = s * row_p[k] + c * row_q[k];
            }
            for row in &mut vectors {
                let (one, two) = (row[p], row[q]);
                row[p] = c * one - s * two;
                row[q] = s * one + c * two;
            }
        }
    }
    let column = |k: usize| DVec3::new(vectors[0][k], vectors[1][k], vectors[2][k]);
    if column(0).cross(column(1)).dot(column(2)) < 0.0 {
        for row in &mut vectors {
            row[2] = -row[2];
        }
    }
    ([matrix[0][0], matrix[1][1], matrix[2][2]], vectors)
}

fn add_outer(matrix: &mut [[f64; 3]; 3], scale: f64, axis: DVec3) {
    let parts = [axis.x, axis.y, axis.z];
    for (row, left) in matrix.iter_mut().zip(parts) {
        for (cell, right) in row.iter_mut().zip(parts) {
            *cell += scale * left * right;
        }
    }
}

fn rebased(children: &[CompoundChild]) -> Result<(Compound, Vec3, Quat), PhysicsError> {
    let mut mass = 0.0_f64;
    let mut weighted = DVec3::ZERO;
    for child in children {
        validate_child(child)?;
        let part = f64::from(child.shape.mass_properties(child.density_scale).mass);
        mass += part;
        weighted += DVec3::from_vec3(child.position) * part;
    }
    if !mass.is_finite() || mass <= 0.0 {
        return Err(PhysicsError::InvalidShape);
    }
    let center = weighted / mass;
    let mut inertia = [[0.0; 3]; 3];
    for child in children {
        let part = child.shape.mass_properties(child.density_scale);
        let turn = DQuat::from_quat(child.rotation.normalize());
        for (axis, value) in [
            (DVec3::X, part.inertia.x),
            (DVec3::Y, part.inertia.y),
            (DVec3::Z, part.inertia.z),
        ] {
            add_outer(&mut inertia, f64::from(value), turn.mul_vec3(axis));
        }
        // parallel axis
        let arm = DVec3::from_vec3(child.position) - center;
        let part_mass = f64::from(part.mass);
        for (index, row) in inertia.iter_mut().enumerate() {
            row[index] += part_mass * arm.length_squared();
        }
        add_outer(&mut inertia, -part_mass, arm);
    }
    let (values, vectors) = principal_axes(inertia);
    let column = |k: usize| {
        Vec3::new(
            vectors[0][k] as f32,
            vectors[1][k] as f32,
            vectors[2][k] as f32,
        )
    };
    let axes = Quat::from_mat3(&Mat3::from_cols(column(0), column(1), column(2))).normalize();
    let back = axes.conjugate();
    let center = center.as_vec3();
    let moved = children
        .iter()
        .map(|child| CompoundChild {
            position: back.rotate_vec3(child.position - center),
            rotation: (back * child.rotation.normalize()).normalize(),
            ..*child
        })
        .collect();
    let unit = MassProperties {
        mass: mass as f32,
        inertia: Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32),
    };
    Ok((Compound::from_rebased(moved, unit)?, center, axes))
}

impl PhysicsWorld {
    fn push_mesh(&mut self, mesh: TriMesh) -> Result<Shape, PhysicsError> {
        let id =
            u32::try_from(self.bodies.assets.meshes.len()).map_err(|_| PhysicsError::OutOfRange)?;
        self.bodies.assets.meshes.push(mesh);
        Ok(Shape::TriMesh(MeshId(id)))
    }

    /// Heights go row by row, x across, z down, `columns` by `rows`. The first post sits at the
    /// body origin.
    ///
    /// # Errors
    /// Fails on a bad size, spacing or height.
    pub fn add_heightfield(
        &mut self,
        columns: u32,
        rows: u32,
        spacing_x: f32,
        spacing_z: f32,
        heights: &[f32],
    ) -> Result<Shape, PhysicsError> {
        self.push_mesh(TriMesh::from_heights(
            columns, rows, spacing_x, spacing_z, heights,
        )?)
    }

    /// Static triangles, front faces counter clockwise. Bodies collide from the front only.
    ///
    /// # Errors
    /// Fails on no triangles, non finite vertices, bad indexes or triangles with no area.
    pub fn add_mesh(
        &mut self,
        vertices: &[Vec3],
        triangles: &[[u32; 3]],
    ) -> Result<Shape, PhysicsError> {
        self.push_mesh(TriMesh::from_triangles(
            vertices.to_vec(),
            triangles.to_vec(),
        )?)
    }

    /// Registers a shape made of parts. Parts can not be compounds themselves.
    /// Mass adds up part by part, so overlapping parts count twice.
    ///
    /// # Errors
    /// Fails on no parts, too many parts or a bad part.
    pub fn add_compound(
        &mut self,
        children: &[CompoundChild],
    ) -> Result<CompoundInfo, PhysicsError> {
        if children.is_empty() || children.len() > MAX_COMPOUND_CHILDREN {
            return Err(PhysicsError::InvalidShape);
        }
        let id = u32::try_from(self.bodies.assets.compounds.len())
            .map_err(|_| PhysicsError::OutOfRange)?;
        let (compound, center, axes) = rebased(children)?;
        self.bodies.assets.compounds.push(compound);
        Ok(CompoundInfo {
            shape: Shape::Compound(CompoundId(id)),
            center,
            axes,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    use super::*;
    use crate::{CollisionFilter, PhysicsSettings, QueryFilter, Ray, RigidBodyDesc, ShapeCast};

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings::default()).unwrap()
    }

    fn no_gravity() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        })
        .unwrap()
    }

    fn ground(world: &mut PhysicsWorld) {
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(50.0, 1.0, 50.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
    }

    fn cube_at(x: f32, y: f32, z: f32) -> CompoundChild {
        CompoundChild::new(Shape::cuboid(Vec3::splat(0.5)), Vec3::new(x, y, z))
    }

    fn near(a: f32, b: f32, tolerance: f32) {
        assert!((a - b).abs() <= tolerance, "{a} != {b}");
    }

    fn at(x: f32, y: f32, z: f32) -> GridPosition {
        GridPosition::from_local(Vec3::new(x, y, z)).unwrap()
    }

    #[test]
    fn jacobi_finds_a_known_matrix() {
        let (values, vectors) = principal_axes([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]]);
        let mut sorted = values;
        sorted.sort_by(f64::total_cmp);
        for (value, expected) in sorted.iter().zip([1.0, 3.0, 5.0]) {
            assert!((value - expected).abs() < 1e-12, "{value}");
        }
        let column = |k: usize| DVec3::new(vectors[0][k], vectors[1][k], vectors[2][k]);
        assert!((column(0).cross(column(1)).dot(column(2)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn two_cubes_have_textbook_mass_and_inertia() {
        let mut world = world();
        let info = world
            .add_compound(&[cube_at(-1.0, 3.0, 0.0), cube_at(1.0, 3.0, 0.0)])
            .unwrap();
        near(info.center.y, 3.0, 1e-5);
        near(info.center.x, 0.0, 1e-5);
        let unit = world.bodies.assets.compounds[0].unit;
        near(unit.mass, 2.0, 1e-5);
        // each cube: 1/6 about its center, plus 1 * 1^2 off the axis it does not sit on
        let mut values = [unit.inertia.x, unit.inertia.y, unit.inertia.z];
        values.sort_by(f32::total_cmp);
        near(values[0], 2.0 / 6.0, 1e-4);
        near(values[1], 2.0 / 6.0 + 2.0, 1e-4);
        near(values[2], 2.0 / 6.0 + 2.0, 1e-4);
    }

    #[test]
    fn a_single_part_matches_that_shape_as_a_plain_body() {
        let mut world = world();
        let half = Vec3::new(0.3, 0.6, 0.9);
        let info = world
            .add_compound(&[CompoundChild::new(Shape::cuboid(half), Vec3::ZERO)])
            .unwrap();
        let unit = world.bodies.assets.compounds[0].unit;
        let plain = Shape::cuboid(half).mass_properties(1.0);
        near(unit.mass, plain.mass, 1e-5);
        let mut left = [unit.inertia.x, unit.inertia.y, unit.inertia.z];
        let mut right = [plain.inertia.x, plain.inertia.y, plain.inertia.z];
        left.sort_by(f32::total_cmp);
        right.sort_by(f32::total_cmp);
        for (a, b) in left.iter().zip(right) {
            near(*a, b, 1e-4);
        }
        assert!(info.center.length() < 1e-5);
    }

    #[test]
    fn heavier_part_pulls_the_center_over() {
        let mut world = world();
        let heavy = CompoundChild {
            density_scale: 3.0,
            ..cube_at(1.0, 0.0, 0.0)
        };
        let info = world
            .add_compound(&[cube_at(-1.0, 0.0, 0.0), heavy])
            .unwrap();
        near(info.center.x, 0.5, 1e-5);
    }

    #[test]
    fn turned_inertia_gets_real_axes() {
        // an l shape has no inertia axes along its own sides
        let mut world = world();
        let info = world
            .add_compound(&[
                cube_at(0.0, 0.0, 0.0),
                cube_at(1.0, 0.0, 0.0),
                cube_at(2.0, 0.0, 0.0),
                cube_at(0.0, 1.0, 0.0),
            ])
            .unwrap();
        assert!(info.axes.to_array()[3].abs() < 0.9999, "{:?}", info.axes);
        // spinning about a principal axis stays exactly so, a wrong axis would wobble
        let mut world = no_gravity();
        let info = world
            .add_compound(&[
                cube_at(0.0, 0.0, 0.0),
                cube_at(1.0, 0.0, 0.0),
                cube_at(2.0, 0.0, 0.0),
                cube_at(0.0, 1.0, 0.0),
            ])
            .unwrap();
        let unit = world.bodies.assets.compounds[0].unit;
        let values = [unit.inertia.x, unit.inertia.y, unit.inertia.z];
        // the largest and smallest axes are the stable ones
        let largest = (0..3)
            .max_by(|a, b| values[*a].total_cmp(&values[*b]))
            .unwrap();
        let axis = [Vec3::X, Vec3::Y, Vec3::Z][largest];
        let body = world
            .spawn(RigidBodyDesc::dynamic(info.shape).angular_velocity(axis * 5.0))
            .unwrap();
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        let spin = world.body_state(body).unwrap().angular_velocity;
        assert!((spin - axis * 5.0).length() < 1e-2, "{spin:?}");
    }

    #[test]
    fn bad_compounds_fail() {
        let mut world = world();
        let good = world.add_compound(&[cube_at(0.0, 0.0, 0.0)]).unwrap();
        let bad_position = CompoundChild::new(Shape::sphere(1.0), Vec3::new(f32::NAN, 0.0, 0.0));
        let bad_scale = CompoundChild {
            density_scale: 0.0,
            ..cube_at(0.0, 0.0, 0.0)
        };
        let bad_turn = CompoundChild {
            rotation: Quat::new(0.0, 0.0, 0.0, 0.0),
            ..cube_at(0.0, 0.0, 0.0)
        };
        for children in [
            vec![],
            vec![CompoundChild::new(good.shape, Vec3::ZERO)],
            vec![CompoundChild::new(Shape::sphere(-1.0), Vec3::ZERO)],
            vec![bad_position],
            vec![bad_scale],
            vec![bad_turn],
        ] {
            assert_eq!(
                world.add_compound(&children),
                Err(PhysicsError::InvalidShape),
                "{children:?}"
            );
        }
        let unknown = Shape::Compound(CompoundId(99));
        assert!(world.spawn(RigidBodyDesc::dynamic(unknown)).is_err());
    }

    fn table(world: &mut PhysicsWorld) -> CompoundInfo {
        // a top with four legs at the corners, like the textbook case
        world
            .add_compound(&[
                CompoundChild::new(
                    Shape::cuboid(Vec3::new(1.0, 0.1, 1.0)),
                    Vec3::new(0.0, 1.0, 0.0),
                ),
                CompoundChild::new(
                    Shape::cuboid(Vec3::new(0.1, 0.45, 0.1)),
                    Vec3::new(0.9, 0.45, 0.9),
                ),
                CompoundChild::new(
                    Shape::cuboid(Vec3::new(0.1, 0.45, 0.1)),
                    Vec3::new(-0.9, 0.45, 0.9),
                ),
                CompoundChild::new(
                    Shape::cuboid(Vec3::new(0.1, 0.45, 0.1)),
                    Vec3::new(0.9, 0.45, -0.9),
                ),
                CompoundChild::new(
                    Shape::cuboid(Vec3::new(0.1, 0.45, 0.1)),
                    Vec3::new(-0.9, 0.45, -0.9),
                ),
            ])
            .unwrap()
    }

    #[test]
    fn table_stands_on_its_legs_and_sleeps() {
        let mut world = world();
        ground(&mut world);
        let info = table(&mut world);
        let (place, turn) = info.placed(Vec3::new(0.0, 0.02, 0.0), Quat::IDENTITY);
        let body = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        let state = world.body_state(body).unwrap();
        assert!(state.asleep, "{state:?}");
        // the bottom of the legs ends up on the floor, not the body center
        near(state.position.local.y, info.center.y, 0.02);
        let up = state
            .rotation
            .rotate_vec3(info.axes.conjugate().rotate_vec3(Vec3::Y));
        assert!(up.y > 0.999, "{up:?}");
    }

    #[test]
    fn something_small_falls_between_the_legs_and_lands_on_the_floor() {
        let mut world = world();
        ground(&mut world);
        let info = table(&mut world);
        let (place, turn) = info.placed(Vec3::ZERO, Quat::IDENTITY);
        world
            .spawn(
                RigidBodyDesc::fixed(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        let marble = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.1)).local_position(Vec3::new(0.0, 0.5, 0.0)),
            )
            .unwrap();
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        near(
            world.body_state(marble).unwrap().position.local.y,
            0.1,
            0.02,
        );
    }

    #[test]
    fn a_ball_dropped_on_top_lands_on_the_top_part() {
        let mut world = world();
        ground(&mut world);
        let info = table(&mut world);
        let (place, turn) = info.placed(Vec3::ZERO, Quat::IDENTITY);
        world
            .spawn(
                RigidBodyDesc::fixed(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        let ball = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.2)).local_position(Vec3::new(0.3, 2.0, 0.2)),
            )
            .unwrap();
        for _ in 0..180 {
            world.step_fixed().unwrap();
        }
        near(world.body_state(ball).unwrap().position.local.y, 1.3, 0.02);
    }

    #[test]
    fn dumbbell_tips_onto_its_ends_when_off_balance() {
        let mut world = world();
        ground(&mut world);
        let info = world
            .add_compound(&[
                CompoundChild::new(Shape::sphere(0.5), Vec3::new(-1.0, 0.0, 0.0)),
                CompoundChild::new(Shape::sphere(0.5), Vec3::new(1.0, 0.0, 0.0)),
            ])
            .unwrap();
        let tilt = Quat::from_rotation_z(FRAC_PI_4);
        let (place, turn) = info.placed(Vec3::new(0.0, 2.0, 0.0), tilt);
        let body = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        for _ in 0..600 {
            world.step_fixed().unwrap();
        }
        // lying flat with both balls down
        near(world.body_state(body).unwrap().position.local.y, 0.5, 0.03);
    }

    #[test]
    fn momentum_stays_equal_when_a_compound_hits_a_body() {
        let mut world = no_gravity();
        let info = world
            .add_compound(&[cube_at(-0.5, 0.0, 0.0), cube_at(0.5, 0.0, 0.0)])
            .unwrap();
        let mover = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .density(1.0)
                    .rotation(info.axes)
                    .linear_velocity(Vec3::new(3.0, 0.0, 0.0)),
            )
            .unwrap();
        let target = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.5))
                    .density(1.0)
                    .local_position(Vec3::new(3.0, 0.0, 0.0)),
            )
            .unwrap();
        for _ in 0..120 {
            world.step_fixed().unwrap();
        }
        let mass_b = 4.0 / 3.0 * std::f32::consts::PI * 0.125;
        let total = world.body_state(mover).unwrap().linear_velocity * 2.0
            + world.body_state(target).unwrap().linear_velocity * mass_b;
        assert!(
            (total - Vec3::new(6.0, 0.0, 0.0)).length() < 1e-2,
            "{total:?}"
        );
        assert!(world.body_state(target).unwrap().linear_velocity.x > 0.1);
    }

    #[test]
    fn queries_see_the_parts_and_not_the_gap_between_them() {
        let mut world = world();
        let info = world
            .add_compound(&[cube_at(-2.0, 0.0, 0.0), cube_at(2.0, 0.0, 0.0)])
            .unwrap();
        let (place, turn) = info.placed(Vec3::ZERO, Quat::IDENTITY);
        let body = world
            .spawn(
                RigidBodyDesc::fixed(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        let filter = QueryFilter::default();
        let down = |x: f32| Ray {
            origin: at(x, 5.0, 0.0),
            direction: -Vec3::Y,
            max_distance: 100.0,
        };
        let hit = world.raycast(&down(2.0), &filter).unwrap().unwrap();
        assert_eq!(hit.body, body);
        near(hit.distance, 4.5, 1e-4);
        near(hit.normal.y, 1.0, 1e-3);
        assert_eq!(world.raycast(&down(0.0), &filter).unwrap(), None);

        let cast = |x: f32| ShapeCast {
            shape: Shape::sphere(0.25),
            rotation: Quat::IDENTITY,
            origin: at(x, 5.0, 0.0),
            direction: -Vec3::Y,
            max_distance: 100.0,
        };
        let swept = world.shape_cast(&cast(-2.0), &filter).unwrap().unwrap();
        near(swept.distance, 4.25, 1e-3);
        assert_eq!(world.shape_cast(&cast(0.0), &filter).unwrap(), None);

        let mut out = Vec::new();
        world
            .overlap(
                &Shape::sphere(0.25),
                at(0.0, 0.0, 0.0),
                Quat::IDENTITY,
                &filter,
                &mut out,
            )
            .unwrap();
        assert!(out.is_empty());
        world
            .overlap(
                &Shape::sphere(0.25),
                at(1.6, 0.0, 0.0),
                Quat::IDENTITY,
                &filter,
                &mut out,
            )
            .unwrap();
        assert_eq!(out, [body]);

        let nearest = world
            .nearest_point(at(0.0, 0.0, 0.0), 100.0, &filter)
            .unwrap()
            .unwrap();
        near(nearest.distance, 1.5, 1e-4);
        let inside = world
            .nearest_point(at(2.0, 0.25, 0.0), 100.0, &filter)
            .unwrap()
            .unwrap();
        near(inside.distance, -0.25, 1e-4);
    }

    #[test]
    fn a_compound_can_not_be_the_query_shape_yet() {
        let mut world = world();
        let info = world.add_compound(&[cube_at(0.0, 0.0, 0.0)]).unwrap();
        let filter = QueryFilter::default();
        let cast = ShapeCast {
            shape: info.shape,
            rotation: Quat::IDENTITY,
            origin: at(0.0, 0.0, 0.0),
            direction: Vec3::X,
            max_distance: 1.0,
        };
        assert_eq!(
            world.shape_cast(&cast, &filter),
            Err(PhysicsError::InvalidShape)
        );
        let mut out = Vec::new();
        assert_eq!(
            world.overlap(
                &info.shape,
                at(0.0, 0.0, 0.0),
                Quat::IDENTITY,
                &filter,
                &mut out
            ),
            Err(PhysicsError::InvalidShape)
        );
    }

    #[test]
    fn filters_apply_to_the_whole_body() {
        let mut world = world();
        ground(&mut world);
        let info = world
            .add_compound(&[cube_at(-0.5, 0.0, 0.0), cube_at(0.5, 0.0, 0.0)])
            .unwrap();
        let body = world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .filter(CollisionFilter { layers: 2, mask: 2 })
                    .local_position(Vec3::new(0.0, 1.0, 0.0))
                    .rotation(info.axes),
            )
            .unwrap();
        for _ in 0..60 {
            world.step_fixed().unwrap();
        }
        assert!(world.body_state(body).unwrap().position.local.y < -1.0);
    }

    #[test]
    fn placed_keeps_a_part_where_it_was_asked_to_be() {
        let mut world = world();
        let info = world
            .add_compound(&[
                cube_at(1.0, 2.0, 3.0),
                cube_at(4.0, 2.0, 3.0),
                cube_at(1.0, 5.0, 3.0),
            ])
            .unwrap();
        let turn = Quat::from_rotation_y(FRAC_PI_2);
        let (place, rotation) = info.placed(Vec3::new(10.0, 0.0, 0.0), turn);
        let first = world.bodies.assets.compounds[0].children[0];
        let found = place + rotation.rotate_vec3(first.position);
        let wanted = Vec3::new(10.0, 0.0, 0.0) + turn.rotate_vec3(Vec3::new(1.0, 2.0, 3.0));
        assert!((found - wanted).length() < 1e-4, "{found:?} {wanted:?}");
    }

    #[test]
    fn snapshot_with_a_compound_continues_exactly() {
        let mut world = world();
        ground(&mut world);
        let info = table(&mut world);
        let (place, turn) = info.placed(Vec3::new(0.0, 0.5, 0.0), Quat::from_rotation_x(0.1));
        world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(place)
                    .rotation(turn),
            )
            .unwrap();
        for _ in 0..30 {
            world.step_fixed().unwrap();
        }
        let bytes = world.save_snapshot();
        let mut loaded = PhysicsWorld::load_snapshot(&bytes).unwrap();
        assert_eq!(loaded.save_snapshot(), bytes);
        assert_eq!(loaded.bodies, world.bodies);
        for _ in 0..60 {
            world.step_fixed().unwrap();
            loaded.step_fixed().unwrap();
        }
        assert_eq!(loaded.state_hash(), world.state_hash());
    }
}
