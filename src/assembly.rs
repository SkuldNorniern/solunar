use algebrix::{DVec3, Quat, Vec3};

use crate::body::{BodyHandle, BodyKind, RigidBodyDesc, apply_diagonal, momentum_from_velocity};
use crate::compound::CompoundChild;
use crate::error::PhysicsError;
use crate::shape::Shape;
use crate::world::PhysicsWorld;

/// Changing what a body is made of while it keeps its handle: a ship that loses a wing, a
/// building that takes a new floor, two parts welded into one. Between steps only. A body with
/// joints or that a breakable owns is refused, they hold its parts in their own axes.
impl PhysicsWorld {
    /// The parts of `body`'s compound, in its axes about its center of mass.
    ///
    /// # Errors
    /// Fails on a stale handle or a body that is not a compound.
    pub fn compound_children(&self, body: BodyHandle) -> Result<Vec<CompoundChild>, PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        match self.bodies.shape[dense] {
            Shape::Compound(id) => Ok(self
                .bodies
                .assets
                .compound(id)
                .ok_or(PhysicsError::InvalidShape)?
                .children
                .clone()),
            _ => Err(PhysicsError::InvalidShape),
        }
    }

    /// Makes `body` of `children`, given in its axes about its center of mass as
    /// [`Self::compound_children`] gives them. The parts stay where they are, the body moves to
    /// their new center of mass and turns to their principal axes, and every point of it keeps
    /// the speed it had. Its contacts are found again.
    ///
    /// # Errors
    /// Fails on a stale handle, bad children, a body with joints or owned by a breakable.
    pub fn reshape_compound(
        &mut self,
        body: BodyHandle,
        children: &[CompoundChild],
    ) -> Result<(), PhysicsError> {
        let dense = self.editable(body)?;
        let info = self.add_owned_compound(children)?;
        let (rotation, position) = (self.bodies.rotation[dense], self.bodies.position(dense));
        let shift = rotation.rotate_vec3(info.center);
        let moved = position
            .shifted(DVec3::from_vec3(shift))
            .ok_or(PhysicsError::OutOfRange)?;
        let turned = (rotation * info.axes).normalize();
        let linear = self.bodies.linear[dense] + self.bodies.angular[dense].cross(shift);
        let angular = self.bodies.angular[dense];
        self.set_shape(dense, info.shape, (moved, turned), (linear, angular));
        Ok(())
    }

    /// Takes `parts`, indexes into [`Self::compound_children`], off `body` into a new body of the
    /// same kind and material, each going on with the motion its parts had. At least one part
    /// must stay.
    ///
    /// # Errors
    /// As [`Self::reshape_compound`], and on no parts, all of them, or an index past the end.
    pub fn split_compound(
        &mut self,
        body: BodyHandle,
        parts: &[usize],
    ) -> Result<BodyHandle, PhysicsError> {
        let dense = self.editable(body)?;
        let children = self.compound_children(body)?;
        let mut taken = vec![false; children.len()];
        for part in parts {
            *taken.get_mut(*part).ok_or(PhysicsError::OutOfRange)? = true;
        }
        let count = taken.iter().filter(|taken| **taken).count();
        if count == 0 || count == children.len() {
            return Err(PhysicsError::OutOfRange);
        }
        let (kept, gone): (Vec<_>, Vec<_>) =
            children.iter().zip(&taken).partition(|(_, taken)| !**taken);
        let kept: Vec<CompoundChild> = kept.into_iter().map(|(child, _)| *child).collect();
        let gone: Vec<CompoundChild> = gone.into_iter().map(|(child, _)| *child).collect();
        let (rotation, position) = (self.bodies.rotation[dense], self.bodies.position(dense));
        let (linear, angular) = (self.bodies.linear[dense], self.bodies.angular[dense]);
        let info = self.add_owned_compound(&gone)?;
        let shift = rotation.rotate_vec3(info.center);
        let mut desc = RigidBodyDesc::new(self.bodies.kind[dense], info.shape)
            .material(self.bodies.material[dense])
            .position(
                position
                    .shifted(DVec3::from_vec3(shift))
                    .ok_or(PhysicsError::OutOfRange)?,
            )
            .rotation((rotation * info.axes).normalize());
        desc.frame = self.bodies.frame[dense];
        desc.filter = self.bodies.filter[dense];
        if self.bodies.kind[dense] == BodyKind::Dynamic {
            desc = desc
                .linear_velocity(linear + angular.cross(shift))
                .angular_velocity(angular);
        }
        let made = self.spawn(desc)?;
        self.reshape_compound(body, &kept)?;
        Ok(made)
    }

    /// Welds `other` onto `body`, both dynamic and in the same frame, and despawns `other`. The
    /// two keep their momentum and angular momentum together. A part that is not a compound
    /// goes in as one part.
    ///
    /// # Errors
    /// As [`Self::reshape_compound`] for either, and on bodies that are not dynamic, in different
    /// frames, the same body, or more parts than a compound holds.
    pub fn merge_bodies(
        &mut self,
        body: BodyHandle,
        other: BodyHandle,
    ) -> Result<(), PhysicsError> {
        if body == other {
            return Err(PhysicsError::InvalidHandle);
        }
        let (a, b) = (self.editable(body)?, self.editable(other)?);
        let dynamic = |dense: usize| self.bodies.kind[dense] == BodyKind::Dynamic;
        if !dynamic(a) || !dynamic(b) {
            return Err(PhysicsError::NotDynamic);
        }
        if self.bodies.frame[a] != self.bodies.frame[b] {
            return Err(PhysicsError::InvalidFrame);
        }
        let parts = |world: &Self, dense: usize| -> Result<Vec<CompoundChild>, PhysicsError> {
            match world.bodies.shape[dense] {
                Shape::Compound(id) => Ok(world
                    .bodies
                    .assets
                    .compound(id)
                    .ok_or(PhysicsError::InvalidShape)?
                    .children
                    .clone()),
                shape => Ok(vec![CompoundChild::new(shape, Vec3::ZERO)]),
            }
        };
        let (turn_a, turn_b) = (self.bodies.rotation[a], self.bodies.rotation[b]);
        let offset = self
            .bodies
            .position(b)
            .offset_from(self.bodies.position(a))
            .as_vec3();
        // b's parts in a's axes, each heavier or lighter as b's density is to a's
        let into_a = turn_a.conjugate() * turn_b;
        let scale = self.bodies.material[b].density / self.bodies.material[a].density;
        let mut children = parts(self, a)?;
        for child in parts(self, b)? {
            children.push(CompoundChild {
                position: turn_a.conjugate().rotate_vec3(offset)
                    + into_a.rotate_vec3(child.position),
                rotation: (into_a * child.rotation).normalize(),
                density_scale: child.density_scale * scale,
                ..child
            });
        }
        let (mass_a, mass_b) = (
            self.bodies.inverse_mass[a].recip(),
            self.bodies.inverse_mass[b].recip(),
        );
        let total = mass_a + mass_b;
        let middle = offset * (mass_b / total);
        let linear = (self.bodies.linear[a] * mass_a + self.bodies.linear[b] * mass_b) / total;
        // angular momentum about the joint center: each spin, and each moving round it
        let spin_of = |dense: usize, arm: Vec3, mass: f32| {
            let velocity = self.bodies.linear[dense] - linear;
            self.bodies.momentum[dense] + arm.cross(velocity) * mass
        };
        let momentum = spin_of(a, -middle, mass_a) + spin_of(b, offset - middle, mass_b);
        self.despawn(other)?;
        let a = self.bodies.dense_index(body)?;
        let info = self.add_owned_compound(&children)?;
        let shift = turn_a.rotate_vec3(info.center);
        let moved = self
            .bodies
            .position(a)
            .shifted(DVec3::from_vec3(shift))
            .ok_or(PhysicsError::OutOfRange)?;
        let turned = (turn_a * info.axes).normalize();
        self.set_shape(a, info.shape, (moved, turned), (linear, Vec3::ZERO));
        let inverse_inertia = self.bodies.inverse_inertia[a];
        self.bodies.momentum[a] = momentum;
        self.bodies.angular[a] = apply_diagonal(turned, inverse_inertia, momentum);
        Ok(())
    }

    /// The body's dense index if its parts may change.
    fn editable(&self, body: BodyHandle) -> Result<usize, PhysicsError> {
        let dense = self.bodies.dense_index(body)?;
        if self.joints.touching(body).next().is_some() {
            return Err(PhysicsError::BodyHasJoints);
        }
        if self.breakables.holds(body) {
            return Err(PhysicsError::ShapeInUse);
        }
        Ok(dense)
    }

    /// Swaps in a shape, moves the body to its new center and axes with these velocities, works
    /// out mass and inertia again and drops the body's contacts, found again next step.
    fn set_shape(
        &mut self,
        dense: usize,
        shape: Shape,
        (position, rotation): (crate::GridPosition, Quat),
        (linear, angular): (Vec3, Vec3),
    ) {
        let old = std::mem::replace(&mut self.bodies.shape[dense], shape);
        self.free_owned(old);
        let bodies = &mut self.bodies;
        let kind = bodies.kind[dense];
        if kind == BodyKind::Dynamic {
            let mass = bodies
                .assets
                .mass_properties(&shape, bodies.material[dense].density);
            bodies.inverse_mass[dense] = mass.mass.recip();
            bodies.inverse_inertia[dense] = mass.inertia.recip();
        }
        bodies.cell[dense] = position.cell;
        bodies.local[dense] = position.local;
        bodies.rotation[dense] = rotation;
        if kind == BodyKind::Dynamic {
            bodies.linear[dense] = linear;
            bodies.angular[dense] = angular;
            bodies.momentum[dense] =
                momentum_from_velocity(kind, rotation, bodies.inverse_inertia[dense], angular);
        }
        bodies.tree.clear();
        bodies.wake(dense);
        let body = bodies.handle_at(dense);
        self.wake_contacts_of(body);
        self.contact_cache
            .retain(|entry| entry.a != body && entry.b != body);
        self.recheck_sleepers |= kind != BodyKind::Dynamic;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::testing::at;
    use crate::world::testing::{ground, no_gravity, world};

    /// Four 1 m cubes in a row along x, a body of them spinning about y and drifting.
    fn rod(world: &mut PhysicsWorld) -> BodyHandle {
        let parts: Vec<CompoundChild> = (0..4)
            .map(|index| {
                CompoundChild::new(
                    Shape::cuboid(Vec3::splat(0.5)),
                    Vec3::new(index as f32 - 1.5, 0.0, 0.0),
                )
            })
            .collect();
        let info = world.add_compound(&parts).unwrap();
        let (center, turn) = info.placed(Vec3::new(0.0, 5.0, 0.0), Quat::IDENTITY);
        world
            .spawn(
                RigidBodyDesc::dynamic(info.shape)
                    .local_position(center)
                    .rotation(turn)
                    .linear_velocity(Vec3::new(1.0, 0.0, 0.0))
                    .angular_velocity(Vec3::new(0.0, 2.0, 0.0)),
            )
            .unwrap()
    }

    fn momentum(world: &PhysicsWorld, bodies: &[BodyHandle]) -> (Vec3, f32) {
        let mut sum = Vec3::ZERO;
        let mut mass = 0.0;
        for body in bodies {
            let dense = world.bodies.dense_index(*body).unwrap();
            let part = world.bodies.inverse_mass[dense].recip();
            sum += world.bodies.linear[dense] * part;
            mass += part;
        }
        (sum, mass)
    }

    #[test]
    fn a_split_off_part_goes_on_with_its_share_and_mass_is_kept() {
        let mut world = no_gravity();
        let body = rod(&mut world);
        let (before, mass) = momentum(&world, &[body]);
        let children = world.compound_children(body).unwrap();
        assert_eq!(children.len(), 4);
        // the end cube furthest along x in the body's axes
        let end = (0..4)
            .max_by(|a, b| {
                let along = |i: usize| {
                    world
                        .body_state(body)
                        .unwrap()
                        .rotation
                        .rotate_vec3(children[i].position)
                        .x
                };
                along(*a).total_cmp(&along(*b))
            })
            .unwrap();
        let piece = world.split_compound(body, &[end]).unwrap();
        let (after, split_mass) = momentum(&world, &[body, piece]);
        assert!((split_mass - mass).abs() < 1e-3 * mass);
        assert!(
            (after - before).length() < 1e-3 * before.length(),
            "{before:?} {after:?}"
        );
        // the end was swinging round, it flies off faster than the rest
        let speed = |handle| world.body_state(handle).unwrap().linear_velocity.length();
        assert!(speed(piece) > speed(body));
        assert_eq!(world.compound_children(body).unwrap().len(), 3);
        // where the parts were, they still are
        let position = |handle| world.body_state(handle).unwrap().position.to_dvec3();
        let gap = (position(piece) - position(body)).length();
        assert!((gap - 2.0).abs() < 1e-3, "{gap}");
    }

    #[test]
    fn merging_keeps_momentum_and_angular_momentum() {
        let mut world = no_gravity();
        let a = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .local_position(Vec3::ZERO)
                    .linear_velocity(Vec3::new(0.0, 0.0, 2.0)),
            )
            .unwrap();
        let b = world
            .spawn(
                RigidBodyDesc::dynamic(Shape::cuboid(Vec3::splat(0.5)))
                    .density(3000.0)
                    .local_position(Vec3::new(1.0, 0.0, 0.0))
                    .linear_velocity(Vec3::new(0.0, 0.0, -1.0)),
            )
            .unwrap();
        let (before, mass) = momentum(&world, &[a, b]);
        // about the pair's center, of mass at x 0.75: a goes +z on the left, b -z on the right
        let center = Vec3::new(0.75, 0.0, 0.0);
        let turn_about = |world: &PhysicsWorld, body: BodyHandle| {
            let dense = world.bodies.dense_index(body).unwrap();
            let m = world.bodies.inverse_mass[dense].recip();
            let arm = world.body_state(body).unwrap().position.local - center;
            world.bodies.momentum[dense] + arm.cross(world.bodies.linear[dense]) * m
        };
        let spin = turn_about(&world, a) + turn_about(&world, b);
        world.merge_bodies(a, b).unwrap();
        assert!(world.body_state(b).is_err());
        let (after, merged) = momentum(&world, &[a]);
        assert!((merged - mass).abs() < 1e-2, "{merged} {mass}");
        assert!((after - before).length() < 1e-2, "{before:?} {after:?}");
        let state = world.body_state(a).unwrap();
        assert!(
            (state.position.local - center).length() < 1e-4,
            "{:?}",
            state.position
        );
        let dense = world.bodies.dense_index(a).unwrap();
        assert!(
            (world.bodies.momentum[dense] - spin).length() < 1e-2,
            "{spin:?}"
        );
        assert!(state.angular_velocity.y.abs() > 0.1);
    }

    #[test]
    fn a_reshaped_body_falls_and_rests_on_its_new_parts() {
        let mut world = world();
        ground(&mut world);
        let body = rod(&mut world);
        let children = world.compound_children(body).unwrap();
        world.reshape_compound(body, &children[..1]).unwrap();
        for _ in 0..240 {
            world.step_fixed().unwrap();
        }
        let state = world.body_state(body).unwrap();
        assert!(
            (state.position.local.y - 0.5).abs() < 0.05,
            "{:?}",
            state.position
        );
        assert!(state.linear_velocity.length() < 0.05);
    }

    fn live(world: &PhysicsWorld) -> usize {
        world
            .bodies
            .assets
            .compounds
            .iter()
            .filter(|compound| compound.life != crate::compound::Life::Freed)
            .count()
    }

    #[test]
    fn edits_free_the_compounds_they_replace_but_never_a_users() {
        let mut world = no_gravity();
        let body = rod(&mut world);
        let first = world.bodies.shape[world.bodies.dense_index(body).unwrap()];
        assert_eq!(live(&world), 1);
        let children = world.compound_children(body).unwrap();
        world.reshape_compound(body, &children[..3]).unwrap();
        world.reshape_compound(body, &children[..2]).unwrap();
        // the user's first and the last edit's
        assert_eq!(live(&world), 2);
        assert!(
            world
                .bodies
                .assets
                .compound(match first {
                    Shape::Compound(id) => id,
                    _ => unreachable!(),
                })
                .is_some()
        );
        let piece = world.split_compound(body, &[0]).unwrap();
        world.merge_bodies(body, piece).unwrap();
        assert_eq!(live(&world), 2);
        world.despawn(body).unwrap();
        assert_eq!(live(&world), 1);
        // the user's own goes when asked, and its id finds nothing after
        world.remove_compound(first).unwrap();
        assert_eq!(live(&world), 0);
        assert!(world.remove_compound(first).is_err());
        assert!(world.spawn(RigidBodyDesc::dynamic(first)).is_err());
    }

    #[test]
    fn a_compound_in_use_is_not_removed_and_freed_slots_survive_snapshots() {
        let mut world = no_gravity();
        let body = rod(&mut world);
        let shape = world.bodies.shape[world.bodies.dense_index(body).unwrap()];
        assert_eq!(world.remove_compound(shape), Err(PhysicsError::ShapeInUse));
        let children = world.compound_children(body).unwrap();
        world.reshape_compound(body, &children[1..]).unwrap();
        world.reshape_compound(body, &children[2..]).unwrap();
        world.remove_compound(shape).unwrap();
        for _ in 0..10 {
            world.step_fixed().unwrap();
        }
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        assert_eq!(
            loaded.bodies.assets.compounds,
            world.bodies.assets.compounds
        );
        for _ in 0..30 {
            world.step_fixed().unwrap();
            loaded.step_fixed().unwrap();
        }
        assert_eq!(loaded.save_snapshot(), world.save_snapshot());
    }

    #[test]
    fn bad_edits_are_refused() {
        let mut world = no_gravity();
        let body = rod(&mut world);
        assert!(world.split_compound(body, &[]).is_err());
        assert!(world.split_compound(body, &[0, 1, 2, 3]).is_err());
        assert!(world.split_compound(body, &[9]).is_err());
        assert!(world.merge_bodies(body, body).is_err());
        let fixed = world
            .spawn(RigidBodyDesc::fixed(Shape::sphere(1.0)).local_position(at(9.0, 0.0, 0.0).local))
            .unwrap();
        assert!(world.merge_bodies(body, fixed).is_err());
        assert!(world.compound_children(fixed).is_err());
        assert_eq!(world.compound_children(body).unwrap().len(), 4);
    }
}
