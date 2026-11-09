use algebrix::Vec3;

use crate::{
    BodyHandle, CompoundChild, GridPosition, JointHandle, PhysicsError, PhysicsWorld, Shape,
};

/// A pair of bodies starting or stopping to push on each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactEvent {
    Began(BodyHandle, BodyHandle),
    Ended(BodyHandle, BodyHandle),
}

/// A contact from the last step. `load` is the push along `normal` per second, in newtons.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactView {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub point: GridPosition,
    pub load: f32,
}

/// Islands solved in the last step. Sleeping islands are not counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IslandStats {
    pub count: usize,
    pub largest_bodies: usize,
    pub largest_constraints: usize,
}

/// Read only views for tools and debug drawing. Nothing here changes the simulation.
impl PhysicsWorld {
    pub fn island_stats(&self) -> IslandStats {
        let islands = self.contact_scratch.islands();
        let mut stats = IslandStats {
            count: islands.count(),
            ..IslandStats::default()
        };
        for (bodies, constraints) in islands.sizes() {
            stats.largest_bodies = stats.largest_bodies.max(bodies);
            stats.largest_constraints = stats.largest_constraints.max(constraints);
        }
        stats
    }

    pub fn body_handles(&self) -> impl Iterator<Item = BodyHandle> + '_ {
        (0..self.bodies.len()).map(|dense| self.bodies.handle_at(dense))
    }

    /// # Errors
    /// Fails on a stale handle.
    pub fn body_shape(&self, body: BodyHandle) -> Result<Shape, PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        Ok(self.bodies.shape[dense])
    }

    /// Parts in body axes. `None` when the shape is not a known compound.
    pub fn compound_parts(&self, shape: &Shape) -> Option<&[CompoundChild]> {
        match *shape {
            Shape::Compound(_) => self.bodies.assets.parts(shape),
            _ => None,
        }
    }

    /// Vertices and triangles in body axes. `None` when the shape is not a known mesh.
    pub fn mesh_triangles(&self, shape: &Shape) -> Option<(&[Vec3], &[[u32; 3]])> {
        let mesh = self.bodies.assets.mesh(shape)?;
        Some((&mesh.vertices, &mesh.triangles))
    }

    /// Triangles of a hull in body axes, for drawing. False when the shape is not a known hull.
    pub fn convex_triangles(&self, shape: &Shape, out: &mut Vec<[Vec3; 3]>) -> bool {
        match self.bodies.assets.convex(shape) {
            Some(hull) => {
                hull.triangles(out);
                true
            }
            None => {
                out.clear();
                false
            }
        }
    }

    /// Fills out with the hull's edges in body axes. False when the shape is not a known hull.
    pub fn convex_edges(&self, shape: &Shape, out: &mut Vec<(Vec3, Vec3)>) -> bool {
        match self.bodies.assets.convex(shape) {
            Some(hull) => {
                hull.outline(out);
                true
            }
            None => {
                out.clear();
                false
            }
        }
    }

    pub fn joint_handles(&self) -> impl Iterator<Item = JointHandle> + '_ {
        self.joints.handles()
    }

    /// Where the joint sits on each body. These stay together while it holds.
    ///
    /// # Errors
    /// Fails on a stale handle.
    pub fn joint_anchors(
        &self,
        joint: JointHandle,
    ) -> Result<(GridPosition, GridPosition), PhysicsError> {
        let desc = self.joints.get(joint)?.desc;
        let at = |body: BodyHandle, anchor: Vec3| {
            let dense = self.bodies.dense_index(body)?;
            let arm = self.bodies.rotation[dense].rotate_vec3(anchor);
            self.bodies
                .position(dense)
                .shifted(algebrix::DVec3::from_vec3(arm))
                .ok_or(PhysicsError::OutOfRange)
        };
        Ok((at(desc.a, desc.anchor_a)?, at(desc.b, desc.anchor_b)?))
    }

    /// Fills `out` from the contact cache. Sleeping pairs keep their last load.
    pub fn contacts(&self, out: &mut Vec<ContactView>) {
        out.clear();
        let part = self.settings.fixed_timestep / self.settings.substeps as f32;
        for entry in &self.contact_cache {
            let Ok(dense) = self.bodies.dense_index(entry.a) else {
                continue;
            };
            let rotation = self.bodies.rotation[dense];
            let origin = self.bodies.position(dense);
            for point in &entry.points[..entry.count] {
                let arm = rotation.rotate_vec3(point.local_a);
                if let Some(at) = origin.shifted(algebrix::DVec3::from_vec3(arm)) {
                    out.push(ContactView {
                        body_a: entry.a,
                        body_b: entry.b,
                        point: at,
                        load: point.impulse.x / part,
                    });
                }
            }
        }
    }

    pub fn sleeping_body_count(&self) -> usize {
        self.bodies.asleep.iter().filter(|asleep| **asleep).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::testing::world;
    use crate::{CompoundChild, JointDesc, RigidBodyDesc};

    #[test]
    fn every_body_is_listed_and_stale_ones_are_not() {
        let mut world = world();
        let a = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(1.0)))
            .unwrap();
        let b = world
            .spawn(RigidBodyDesc::fixed(Shape::sphere(2.0)))
            .unwrap();
        assert_eq!(world.body_handles().collect::<Vec<_>>(), [a, b]);
        assert_eq!(world.body_shape(b).unwrap(), Shape::sphere(2.0));
        world.despawn(a).unwrap();
        assert_eq!(world.body_handles().collect::<Vec<_>>(), [b]);
        assert_eq!(world.body_shape(a), Err(PhysicsError::InvalidHandle));
    }

    #[test]
    fn compounds_and_meshes_show_their_data() {
        let mut world = world();
        let info = world
            .add_compound(&[
                CompoundChild::new(Shape::sphere(0.5), Vec3::X),
                CompoundChild::new(Shape::sphere(0.5), -Vec3::X),
            ])
            .unwrap();
        assert_eq!(world.compound_parts(&info.shape).map(<[_]>::len), Some(2));
        assert_eq!(world.compound_parts(&Shape::sphere(1.0)), None);
        let mesh = world
            .add_mesh(&[Vec3::ZERO, Vec3::X, Vec3::Z], &[[0, 2, 1]])
            .unwrap();
        let (vertices, triangles) = world.mesh_triangles(&mesh).unwrap();
        assert_eq!((vertices.len(), triangles.len()), (3, 1));
        assert_eq!(world.mesh_triangles(&info.shape), None);
        let mut points = Vec::new();
        for x in [-0.5, 0.5] {
            for y in [-0.5, 0.5] {
                for z in [-0.5, 0.5] {
                    points.push(Vec3::new(x, y, z));
                }
            }
        }
        let cube = world.add_convex(&points).unwrap();
        let mut edges = Vec::new();
        assert!(world.convex_edges(&cube.shape, &mut edges));
        assert_eq!(edges.len(), 12);
        for (a, b) in &edges {
            assert!(((*a - *b).length() - 1.0).abs() < 1e-4);
        }
        assert!(!world.convex_edges(&mesh, &mut edges));
        let mut triangles = Vec::new();
        assert!(world.convex_triangles(&cube.shape, &mut triangles));
        // six square faces of two triangles, total area is the surface of the 1 m cube
        assert_eq!(triangles.len(), 12);
        let area: f32 = triangles
            .iter()
            .map(|[a, b, c]| (*b - *a).cross(*c - *a).length() * 0.5)
            .sum();
        assert!((area - 6.0).abs() < 1e-3, "{area}");
        assert!(!world.convex_triangles(&mesh, &mut triangles));
        assert!(triangles.is_empty());
        assert!(edges.is_empty());
    }

    #[test]
    fn joint_anchors_meet_where_the_joint_holds() {
        let mut world = world();
        let hook = world
            .spawn(RigidBodyDesc::fixed(Shape::sphere(0.1)))
            .unwrap();
        let bob = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::sphere(0.25))
                    .local_position(Vec3::new(0.0, -2.0, 0.0)),
            )
            .unwrap();
        let joint = world
            .add_joint(JointDesc::ball(hook, bob).anchors(Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0)))
            .unwrap();
        assert_eq!(world.joint_handles().collect::<Vec<_>>(), [joint]);
        let (a, b) = world.joint_anchors(joint).unwrap();
        assert!(b.offset_from(a).length() < 1e-5);
        world.remove_joint(joint).unwrap();
        assert_eq!(world.joint_handles().count(), 0);
        assert_eq!(world.joint_anchors(joint), Err(PhysicsError::InvalidHandle));
    }

    #[test]
    fn contacts_report_the_weight_they_carry() {
        let mut world = world();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(5.0, 1.0, 5.0)))
                    .local_position(Vec3::new(0.0, -1.0, 0.0)),
            )
            .unwrap();
        world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .density(10.0)
                    .local_position(Vec3::new(0.0, 0.5, 0.0)),
            )
            .unwrap();
        for _ in 0..20 {
            world.step_fixed().unwrap();
        }
        let mut seen = Vec::new();
        world.contacts(&mut seen);
        assert_eq!(seen.len(), 4);
        let total: f32 = seen.iter().map(|contact| contact.load).sum();
        assert!((total - 98.1).abs() < 1.0, "{total}");
        for contact in &seen {
            assert!(contact.point.local.y.abs() < 0.03, "{contact:?}");
        }
    }

    #[test]
    fn looking_does_not_change_the_state() {
        let mut world = world();
        world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(1.0)))
            .unwrap();
        for _ in 0..5 {
            world.step_fixed().unwrap();
        }
        let before = world.state_hash();
        let mut seen = Vec::new();
        world.contacts(&mut seen);
        let _ = world.sleeping_body_count();
        let _ = world.body_handles().count();
        assert_eq!(world.state_hash(), before);
    }
}
