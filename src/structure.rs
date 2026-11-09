use algebrix::{Quat, Vec3};

use crate::body::{BodyHandle, BodyKind};
use crate::compound::part_mass;
use crate::error::PhysicsError;
use crate::shape::Shape;
use crate::snapshot::{Reader, SnapshotError, Writer};
use crate::world::PhysicsWorld;

/// Most links a structure holds.
pub const MAX_LINKS: usize = 1 << 16;
/// Load over strength past which a link tires: each step there adds this share of the ratio to
/// its damage, so a link held at 60% of its strength gives out after about 40 seconds.
const FATIGUE_FROM: f32 = 0.5;
const FATIGUE_RATE: f32 = 0.0007;
/// Conjugate gradient passes for a structure's loads, and how close it must come.
const LOAD_PASSES: usize = 400;
/// Seconds a link's load is averaged over before it breaks it. A rigid hit goes through in one
/// step, its force a step's impulse over the step, so read alone a 1 m/s bump broke a ship and
/// the step length decided what held. Averaged, a one step spike counts by its impulse, a load
/// that stays counts in full after a few steps.
const STRAIN_TIME: f32 = 0.05;
const LOAD_TOLERANCE: f32 = 1e-5;

/// Two parts of a compound body held together, a weld, a bolted frame, a docking collar. The
/// parts are indexes into the body's [`PhysicsWorld::compound_children`] when the structure is
/// made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkDesc {
    pub a: u32,
    pub b: u32,
    /// Newtons it carries before it breaks.
    pub strength: f32,
}

/// A link as it stands: how loaded it was last step and how tired.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Link {
    pub a: u32,
    pub b: u32,
    pub strength: f32,
    /// Newtons through it last step.
    pub load: f32,
    /// Load over strength averaged over the last 50 ms or so, it breaks at 1.
    pub strain: f32,
    /// 0 to 1, from loads held past half the strength. It breaks at 1.
    pub damage: f32,
    pub broken: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StructureHandle {
    index: u32,
    generation: u32,
}

/// A body of a structure split off from another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructureEvent {
    pub structure: StructureHandle,
    pub from: BodyHandle,
    pub body: BodyHandle,
    pub parts: u32,
    pub mass: f32,
}

/// The parts of one body, structure part ids in the order of its compound's children.
#[derive(Debug, Clone, PartialEq)]
struct Group {
    body: BodyHandle,
    parts: Vec<u32>,
    /// Its motion as the step began, for the push its parts need.
    before: (Vec3, Vec3),
}

#[derive(Debug, Clone, PartialEq)]
struct Structure {
    links: Vec<Link>,
    groups: Vec<Group>,
    /// Pushes the host put on parts for the next step, newtons at each part's middle.
    pushes: Vec<(u32, Vec3)>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Structures {
    slots: Vec<Option<Structure>>,
    generations: Vec<u32>,
    events: Vec<StructureEvent>,
}

impl Structures {
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    pub(crate) fn holds(&self, body: BodyHandle) -> bool {
        self.slots
            .iter()
            .flatten()
            .any(|structure| structure.groups.iter().any(|group| group.body == body))
    }

    pub(crate) fn save(&self, out: &mut Writer) {
        out.len(self.slots.len());
        for (slot, generation) in self.slots.iter().zip(&self.generations) {
            out.u32(*generation);
            let Some(structure) = slot else {
                out.u8(0);
                continue;
            };
            out.u8(1);
            out.len(structure.links.len());
            for link in &structure.links {
                out.u32(link.a);
                out.u32(link.b);
                out.f32(link.strength);
                out.f32(link.load);
                out.f32(link.strain);
                out.f32(link.damage);
                out.u8(u8::from(link.broken));
            }
            out.len(structure.groups.len());
            for group in &structure.groups {
                out.body(group.body);
                out.len(group.parts.len());
                for part in &group.parts {
                    out.u32(*part);
                }
            }
        }
    }

    pub(crate) fn load(reader: &mut Reader<'_>, max: usize) -> Result<Self, SnapshotError> {
        let bad = SnapshotError::Invalid("structure");
        let count = reader.count(5, max)?;
        let mut structures = Self::default();
        for _ in 0..count {
            structures.generations.push(reader.u32()?);
            let slot = match reader.u8()? {
                0 => None,
                1 => {
                    let links = reader.count(25, MAX_LINKS)?;
                    let mut structure = Structure {
                        links: Vec::with_capacity(links),
                        groups: Vec::new(),
                        pushes: Vec::new(),
                    };
                    for _ in 0..links {
                        let link = Link {
                            a: reader.u32()?,
                            b: reader.u32()?,
                            strength: reader.f32()?,
                            load: reader.f32()?,
                            strain: reader.f32()?,
                            damage: reader.f32()?,
                            broken: match reader.u8()? {
                                0 => false,
                                1 => true,
                                _ => return Err(bad),
                            },
                        };
                        let positive = link.strength.is_finite() && link.strength > 0.0;
                        if !positive
                            || !link.load.is_finite()
                            || !link.strain.is_finite()
                            || !(0.0..=1.0).contains(&link.damage)
                        {
                            return Err(bad);
                        }
                        structure.links.push(link);
                    }
                    let groups = reader.count(12, MAX_LINKS)?;
                    let mut seen = Vec::new();
                    for _ in 0..groups {
                        let body = reader.handle()?;
                        let parts = reader.count(4, MAX_LINKS)?;
                        let mut ids = Vec::with_capacity(parts);
                        for _ in 0..parts {
                            ids.push(reader.u32()?);
                        }
                        seen.extend_from_slice(&ids);
                        structure.groups.push(Group {
                            body,
                            parts: ids,
                            before: (Vec3::ZERO, Vec3::ZERO),
                        });
                    }
                    seen.sort_unstable();
                    let parts = seen.len() as u32;
                    let unique = seen.windows(2).all(|pair| pair[0] != pair[1]);
                    let in_range = structure.links.iter().all(|l| l.a < parts && l.b < parts);
                    if !unique || seen.last().is_some_and(|last| *last >= parts) || !in_range {
                        return Err(bad);
                    }
                    Some(structure)
                }
                _ => return Err(bad),
            };
            structures.slots.push(slot);
        }
        Ok(structures)
    }
}

impl PhysicsWorld {
    /// Holds `body`'s compound together by `links` between its parts: each step the loads its
    /// motion, contacts and pushes put on it are spread over them, and a link past its strength,
    /// or tired out by load after load, breaks. When the parts are no longer all joined, each
    /// lot of them goes on as its own body. The body's handle stays with the heaviest lot.
    ///
    /// # Errors
    /// Fails on a stale handle, a body that is not a compound, a part index past its parts, a
    /// link of a part to itself or a non finite or non positive strength, too many links, or a
    /// body with joints or owned by a breakable or another structure.
    pub fn add_structure(
        &mut self,
        body: BodyHandle,
        links: &[LinkDesc],
    ) -> Result<StructureHandle, PhysicsError> {
        let parts = self.compound_children(body)?.len() as u32;
        if self.joints.touching(body).next().is_some() {
            return Err(PhysicsError::BodyHasJoints);
        }
        if self.breakables.holds(body) || self.structures.holds(body) {
            return Err(PhysicsError::ShapeInUse);
        }
        let valid = links.len() <= MAX_LINKS
            && links.iter().all(|link| {
                link.a < parts
                    && link.b < parts
                    && link.a != link.b
                    && link.strength.is_finite()
                    && link.strength > 0.0
            });
        if !valid {
            return Err(PhysicsError::InvalidShape);
        }
        let structure = Structure {
            links: links
                .iter()
                .map(|link| Link {
                    a: link.a,
                    b: link.b,
                    strength: link.strength,
                    load: 0.0,
                    strain: 0.0,
                    damage: 0.0,
                    broken: false,
                })
                .collect(),
            groups: vec![Group {
                body,
                parts: (0..parts).collect(),
                before: (Vec3::ZERO, Vec3::ZERO),
            }],
            pushes: Vec::new(),
        };
        let structures = &mut self.structures;
        let index = match structures.slots.iter().position(Option::is_none) {
            Some(index) => index,
            None => {
                structures.slots.push(None);
                structures.generations.push(0);
                structures.slots.len() - 1
            }
        };
        structures.slots[index] = Some(structure);
        Ok(StructureHandle {
            index: index as u32,
            generation: structures.generations[index],
        })
    }

    /// Ends `structure`, its bodies keep going as they are.
    ///
    /// # Errors
    /// Fails on a stale handle.
    pub fn remove_structure(&mut self, structure: StructureHandle) -> Result<(), PhysicsError> {
        self.structure(structure)?;
        let index = structure.index as usize;
        self.structures.slots[index] = None;
        self.structures.generations[index] = self.structures.generations[index].wrapping_add(1);
        Ok(())
    }

    /// The links as they stood after the last step.
    ///
    /// # Errors
    /// Fails on a stale handle.
    pub fn structure_links(&self, structure: StructureHandle) -> Result<&[Link], PhysicsError> {
        Ok(&self.structure(structure)?.links)
    }

    /// The bodies it is in now, each with its parts, ids as the structure was made.
    ///
    /// # Errors
    /// Fails on a stale handle.
    pub fn structure_bodies(
        &self,
        structure: StructureHandle,
    ) -> Result<Vec<(BodyHandle, Vec<u32>)>, PhysicsError> {
        Ok(self
            .structure(structure)?
            .groups
            .iter()
            .map(|group| (group.body, group.parts.clone()))
            .collect())
    }

    /// Pushes part `part` with `force` newtons through its middle over the next step: a thruster
    /// or a crane on that part. The body is pushed and the links carry it.
    ///
    /// # Errors
    /// Fails on a stale handle, an unknown part or a non finite force.
    pub fn push_part(
        &mut self,
        structure: StructureHandle,
        part: u32,
        force: Vec3,
    ) -> Result<(), PhysicsError> {
        if !force.is_finite() {
            return Err(PhysicsError::NonFinite("part push"));
        }
        let (body, at) = {
            let held = self.structure(structure)?;
            let group = held
                .groups
                .iter()
                .find(|group| group.parts.contains(&part))
                .ok_or(PhysicsError::OutOfRange)?;
            let index = group.parts.iter().position(|id| *id == part).unwrap_or(0);
            (group.body, index)
        };
        let children = self.compound_children(body)?;
        let rotation = self.body_state(body)?.rotation;
        let arm = rotation.rotate_vec3(children[at].position);
        self.apply_force(body, force)?;
        self.apply_torque(body, arm.cross(force))?;
        if let Some(held) = self.structures.slots[structure.index as usize].as_mut() {
            held.pushes.push((part, force));
        }
        Ok(())
    }

    /// Breaks a link now, as a charge or a cutter would. The parts it held come apart after the
    /// next step if nothing else holds them.
    ///
    /// # Errors
    /// Fails on a stale handle or a link past the end.
    pub fn break_link(
        &mut self,
        structure: StructureHandle,
        link: usize,
    ) -> Result<(), PhysicsError> {
        self.structure(structure)?;
        let held = self.structures.slots[structure.index as usize]
            .as_mut()
            .ok_or(PhysicsError::InvalidHandle)?;
        held.links
            .get_mut(link)
            .ok_or(PhysicsError::OutOfRange)?
            .broken = true;
        Ok(())
    }

    /// Bodies split off by the last step.
    pub fn structure_events(&self) -> &[StructureEvent] {
        &self.structures.events
    }

    fn structure(&self, handle: StructureHandle) -> Result<&Structure, PhysicsError> {
        let index = handle.index as usize;
        match (
            self.structures.slots.get(index),
            self.structures.generations.get(index),
        ) {
            (Some(Some(structure)), Some(generation)) if *generation == handle.generation => {
                Ok(structure)
            }
            _ => Err(PhysicsError::InvalidHandle),
        }
    }

    pub(crate) fn remember_structure_motion(&mut self) {
        for structure in self.structures.slots.iter_mut().flatten() {
            for group in &mut structure.groups {
                if let Ok(dense) = self.bodies.dense_index(group.body) {
                    group.before = (self.bodies.linear[dense], self.bodies.angular[dense]);
                }
            }
        }
    }

    /// After a step: each structure's loads, then what broke and what came apart.
    pub(crate) fn update_structures(&mut self) -> Result<(), PhysicsError> {
        self.structures.events.clear();
        if self.structures.is_empty() {
            return Ok(());
        }
        let dt = self.settings.fixed_timestep;
        for index in 0..self.structures.slots.len() {
            let Some(mut structure) = self.structures.slots[index].take() else {
                continue;
            };
            let handle = StructureHandle {
                index: index as u32,
                generation: self.structures.generations[index],
            };
            let result = self.load_structure(&mut structure, dt);
            let result = result.and_then(|()| self.split_structure(handle, &mut structure));
            structure.pushes.clear();
            self.structures.slots[index] = Some(structure);
            result?;
        }
        Ok(())
    }

    /// What each part needs from its links, f = pushed on it less its mass times how it sped
    /// up, gravity left out as it pulls every part alike. Spread over the links by their
    /// strength as the least load that carries it, a weighted graph laplacian solved by
    /// conjugate gradients from zero, so no state is kept between steps.
    fn load_structure(&self, structure: &mut Structure, dt: f32) -> Result<(), PhysicsError> {
        let parts = structure
            .groups
            .iter()
            .map(|group| group.parts.len())
            .sum::<usize>();
        let mut need = vec![Vec3::ZERO; parts];
        for group in &structure.groups {
            let Ok(dense) = self.bodies.dense_index(group.body) else {
                continue;
            };
            if self.bodies.kind[dense] != BodyKind::Dynamic {
                continue;
            }
            let Shape::Compound(id) = self.bodies.shape[dense] else {
                continue;
            };
            let Some(compound) = self.bodies.assets.compound(id) else {
                continue;
            };
            let rotation: Quat = self.bodies.rotation[dense];
            let (linear, angular) = (self.bodies.linear[dense], self.bodies.angular[dense]);
            let gravity = self.gravity.at(
                self.bodies.frame[dense],
                self.bodies.position(dense),
                linear,
            );
            let speeding = (linear - group.before.0) / dt - gravity;
            let turning = (angular - group.before.1) / dt;
            let density = self.bodies.material[dense].density;
            for (child, part) in compound.children.iter().zip(&group.parts) {
                let mass = part_mass(
                    &child.shape,
                    child.density_scale * density,
                    &self.bodies.assets.convexes,
                )
                .mass;
                let arm = rotation.rotate_vec3(child.position);
                let along = speeding + turning.cross(arm) + angular.cross(angular.cross(arm));
                need[*part as usize] -= along * mass;
            }
        }
        // contacts on each part, both sides, the part numbered as the compound's children
        let mut owners: Vec<(BodyHandle, usize)> = structure
            .groups
            .iter()
            .enumerate()
            .map(|(index, group)| (group.body, index))
            .collect();
        owners.sort_unstable();
        for entry in &self.contact_cache {
            for (body, part, sign) in [(entry.a, entry.part_a, -1.0), (entry.b, entry.part_b, 1.0)]
            {
                let Ok(at) = owners.binary_search_by_key(&body, |owner| owner.0) else {
                    continue;
                };
                let group = &structure.groups[owners[at].1];
                let Some(&id) = group.parts.get(part as usize) else {
                    continue;
                };
                let push: f32 = entry.points[..entry.count]
                    .iter()
                    .map(|point| point.pushed)
                    .sum();
                let Ok(dense) = self.bodies.dense_index(entry.a) else {
                    continue;
                };
                // the cache keeps its normal in a's axes, b is pushed along it
                let normal = self.bodies.rotation[dense].rotate_vec3(entry.normal);
                need[id as usize] += normal * (sign * push / dt);
            }
        }
        for (part, force) in &structure.pushes {
            if let Some(slot) = need.get_mut(*part as usize) {
                *slot += *force;
            }
        }
        let potentials = solve_loads(&structure.links, &structure.groups, &mut need, parts);
        for link in &mut structure.links {
            if link.broken {
                link.load = 0.0;
                link.strain = 0.0;
                continue;
            }
            let carried = potentials[link.a as usize] - potentials[link.b as usize];
            let ratio = carried.length();
            link.load = ratio * link.strength;
            link.strain += (ratio - link.strain) * (dt / STRAIN_TIME).min(1.0);
            if ratio > FATIGUE_FROM {
                link.damage = (link.damage + ratio * FATIGUE_RATE).min(1.0);
            }
            if link.strain >= 1.0 || link.damage >= 1.0 {
                link.broken = true;
            }
        }
        if potentials.iter().any(|value| !value.is_finite()) {
            return Err(PhysicsError::NonFinite("structure loads"));
        }
        Ok(())
    }

    /// Each body's parts still joined: the lot with the most parts keeps the body, the rest go
    /// as bodies of their own.
    fn split_structure(
        &mut self,
        handle: StructureHandle,
        structure: &mut Structure,
    ) -> Result<(), PhysicsError> {
        let groups = std::mem::take(&mut structure.groups);
        for group in groups {
            if self.bodies.dense_index(group.body).is_err() {
                continue;
            }
            let lots = joined(&group.parts, &structure.links);
            if lots.len() < 2 {
                structure.groups.push(group);
                continue;
            }
            let keep = (0..lots.len())
                .max_by_key(|lot| (lots[*lot].len(), std::cmp::Reverse(*lot)))
                .unwrap_or(0);
            let mut remaining = group.parts.clone();
            for (index, lot) in lots.iter().enumerate() {
                if index == keep {
                    continue;
                }
                let at: Vec<usize> = lot
                    .iter()
                    .filter_map(|id| remaining.iter().position(|part| part == id))
                    .collect();
                let mass = {
                    let dense = self.bodies.dense_index(group.body)?;
                    let children = self.compound_children(group.body)?;
                    let density = self.bodies.material[dense].density;
                    at.iter()
                        .map(|index| {
                            let child = &children[*index];
                            part_mass(
                                &child.shape,
                                child.density_scale * density,
                                &self.bodies.assets.convexes,
                            )
                            .mass
                        })
                        .sum::<f32>()
                };
                let made = self.split_compound(group.body, &at)?;
                remaining.retain(|part| !lot.contains(part));
                self.structures.events.push(StructureEvent {
                    structure: handle,
                    from: group.body,
                    body: made,
                    parts: lot.len() as u32,
                    mass,
                });
                structure.groups.push(Group {
                    body: made,
                    parts: lot.clone(),
                    before: (Vec3::ZERO, Vec3::ZERO),
                });
            }
            structure.groups.push(Group {
                body: group.body,
                parts: remaining,
                before: group.before,
            });
        }
        Ok(())
    }
}

/// The least loads through unbroken `links` that carry `need`, as each part's potential, a
/// link's load over its strength being the potentials' difference. Within each joined lot the
/// needs are made to sum to zero first, what is left over is the lot's own speeding up.
fn solve_loads(links: &[Link], groups: &[Group], need: &mut [Vec3], parts: usize) -> Vec<Vec3> {
    for group in groups {
        for lot in joined(&group.parts, links) {
            let mean = lot.iter().map(|id| need[*id as usize]).sum::<Vec3>() / lot.len() as f32;
            for id in &lot {
                need[*id as usize] -= mean;
            }
        }
    }
    let apply = |x: &[Vec3], out: &mut [Vec3]| {
        out.iter_mut().for_each(|value| *value = Vec3::ZERO);
        for link in links.iter().filter(|link| !link.broken) {
            let flow = (x[link.a as usize] - x[link.b as usize]) * link.strength;
            out[link.a as usize] += flow;
            out[link.b as usize] -= flow;
        }
    };
    let dot = |a: &[Vec3], b: &[Vec3]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(x.dot(*y)))
            .sum::<f64>()
    };
    let mut x = vec![Vec3::ZERO; parts];
    let mut r = need.to_vec();
    let mut p = r.clone();
    let mut lp = vec![Vec3::ZERO; parts];
    let start = dot(&r, &r);
    let mut rr = start;
    for _ in 0..LOAD_PASSES {
        if rr <= start * f64::from(LOAD_TOLERANCE * LOAD_TOLERANCE) || rr == 0.0 {
            break;
        }
        apply(&p, &mut lp);
        let curve = dot(&p, &lp);
        if curve <= 0.0 {
            break;
        }
        let step = (rr / curve) as f32;
        for i in 0..parts {
            x[i] += p[i] * step;
            r[i] -= lp[i] * step;
        }
        let next = dot(&r, &r);
        let keep = (next / rr) as f32;
        for i in 0..parts {
            p[i] = r[i] + p[i] * keep;
        }
        rr = next;
    }
    x
}

/// The parts of `parts` still joined by unbroken links, each lot in the order of `parts`, lots
/// by their first part.
fn joined(parts: &[u32], links: &[Link]) -> Vec<Vec<u32>> {
    fn root(parent: &mut [usize], mut at: usize) -> usize {
        while parent[at] != at {
            parent[at] = parent[parent[at]];
            at = parent[at];
        }
        at
    }
    let top = parts.iter().max().map_or(0, |most| *most as usize + 1);
    let mut place = vec![usize::MAX; top];
    for (index, part) in parts.iter().enumerate() {
        place[*part as usize] = index;
    }
    let index_of = |id: u32| {
        place
            .get(id as usize)
            .copied()
            .filter(|at| *at != usize::MAX)
    };
    let mut parent: Vec<usize> = (0..parts.len()).collect();
    for link in links.iter().filter(|link| !link.broken) {
        if let (Some(a), Some(b)) = (index_of(link.a), index_of(link.b)) {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    let mut lots: Vec<(usize, Vec<u32>)> = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let top = root(&mut parent, index);
        match lots.iter_mut().find(|lot| lot.0 == top) {
            Some(lot) => lot.1.push(*part),
            None => lots.push((top, vec![*part])),
        }
    }
    lots.into_iter().map(|lot| lot.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compound::CompoundChild;
    use crate::world::testing::no_gravity;
    use crate::{RigidBodyDesc, Shape};

    /// `count` 1 m cubes of 1000 kg/m^3 in a row along x, each welded to the next.
    fn rod(world: &mut PhysicsWorld, count: u32, strength: f32) -> (BodyHandle, StructureHandle) {
        let parts: Vec<CompoundChild> = (0..count)
            .map(|index| {
                CompoundChild::new(
                    Shape::cuboid(Vec3::splat(0.5)),
                    Vec3::new(index as f32, 0.0, 0.0),
                )
            })
            .collect();
        let info = world.add_compound(&parts).unwrap();
        let (center, turn) = info.placed(Vec3::ZERO, Quat::IDENTITY);
        let mut desc = RigidBodyDesc::dynamic(info.shape)
            .local_position(center)
            .rotation(turn);
        desc.can_sleep = false;
        let body = world.spawn(desc).unwrap();
        let links: Vec<LinkDesc> = (0..count - 1)
            .map(|index| LinkDesc {
                a: index,
                b: index + 1,
                strength,
            })
            .collect();
        let structure = world.add_structure(body, &links).unwrap();
        (body, structure)
    }

    /// Each part's x in world axes, by part index.
    fn along_x(world: &PhysicsWorld, body: BodyHandle) -> Vec<f32> {
        let rotation = world.body_state(body).unwrap().rotation;
        world
            .compound_children(body)
            .unwrap()
            .iter()
            .map(|child| rotation.rotate_vec3(child.position).x)
            .collect()
    }

    #[test]
    fn a_pulled_rod_carries_what_the_parts_beyond_need() {
        let mut world = no_gravity();
        let (body, structure) = rod(&mut world, 5, 1.0e9);
        let x = along_x(&world, body);
        let end = (0..5).max_by(|a, b| x[*a].total_cmp(&x[*b])).unwrap() as u32;
        let force = 10_000.0;
        world
            .push_part(structure, end, Vec3::new(force, 0.0, 0.0))
            .unwrap();
        world.step_fixed().unwrap();
        // the link with k parts on the far side from the pulled end carries F k / 5
        let mut loads: Vec<(f32, f32)> = world
            .structure_links(structure)
            .unwrap()
            .iter()
            .map(|link| (x[link.a as usize].min(x[link.b as usize]), link.load))
            .collect();
        loads.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (k, (_, load)) in loads.iter().enumerate() {
            let expected = force * (k + 1) as f32 / 5.0;
            assert!((load - expected).abs() < 0.01 * force, "{loads:?}");
        }
    }

    #[test]
    fn a_weak_link_breaks_and_both_sides_keep_their_motion() {
        let mut world = no_gravity();
        let (body, structure) = rod(&mut world, 4, 1.0e9);
        let x = along_x(&world, body);
        let end = (0..4).min_by(|a, b| x[*a].total_cmp(&x[*b])).unwrap() as u32;
        // the link from the end part to the rest made weak
        let mut descs: Vec<LinkDesc> = world
            .structure_links(structure)
            .unwrap()
            .iter()
            .map(|link| LinkDesc {
                a: link.a,
                b: link.b,
                strength: link.strength,
            })
            .collect();
        for desc in &mut descs {
            if desc.a == end || desc.b == end {
                desc.strength = 3_000.0;
            }
        }
        world.remove_structure(structure).unwrap();
        let structure = world.add_structure(body, &descs).unwrap();
        // 20 kN back on the end part: its link to the rest carries 15 kN, past the weak 3
        world
            .push_part(structure, end, Vec3::new(-20_000.0, 0.0, 0.0))
            .unwrap();
        world.step_fixed().unwrap();
        let events = world.structure_events().to_vec();
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0].from, body);
        assert!((events[0].mass - 1000.0).abs() < 1.0);
        let bodies = world.structure_bodies(structure).unwrap();
        assert_eq!(bodies.len(), 2);
        // all of it still weighs 4000 kg and goes with the push it got
        let (mut mass, mut momentum) = (0.0, Vec3::ZERO);
        for (part, _) in &bodies {
            let dense = world.bodies.dense_index(*part).unwrap();
            let m = world.bodies.inverse_mass[dense].recip();
            mass += m;
            momentum += world.bodies.linear[dense] * m;
        }
        assert!((mass - 4000.0).abs() < 1.0, "{mass}");
        let pushed = -20_000.0 / 60.0;
        assert!(
            (momentum.x - pushed).abs() < 0.01 * pushed.abs(),
            "{momentum:?}"
        );
        // the break comes after the step, the whole took that push. pushed again only the end
        // part goes with it
        world
            .push_part(structure, end, Vec3::new(-20_000.0, 0.0, 0.0))
            .unwrap();
        world.step_fixed().unwrap();
        let speed = |part: BodyHandle| world.body_state(part).unwrap().linear_velocity.x;
        assert!(speed(events[0].body) < speed(body) - 0.3);
    }

    fn spin(world: &mut PhysicsWorld, body: BodyHandle, rate: f32) {
        let dense = world.bodies.dense_index(body).unwrap();
        let (rotation, inverse) = (
            world.bodies.rotation[dense],
            world.bodies.inverse_inertia[dense],
        );
        let turn = Vec3::new(0.0, rate, 0.0);
        world.bodies.angular[dense] = turn;
        world.bodies.momentum[dense] =
            crate::body::momentum_from_velocity(BodyKind::Dynamic, rotation, inverse, turn);
    }

    #[test]
    fn flown_into_a_wall_its_front_breaks_off() {
        let mut world = no_gravity();
        world
            .spawn(
                RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(0.5, 5.0, 5.0)))
                    .local_position(Vec3::new(6.0, 0.0, 0.0)),
            )
            .unwrap();
        let (body, structure) = rod(&mut world, 5, 2.0e5);
        let dense = world.bodies.dense_index(body).unwrap();
        world.bodies.linear[dense] = Vec3::new(30.0, 0.0, 0.0);
        let mut broke = false;
        for _ in 0..60 {
            world.step_fixed().unwrap();
            broke |= !world.structure_events().is_empty();
        }
        assert!(broke);
        // the same flown slowly holds together
        let mut calm = no_gravity();
        calm.spawn(
            RigidBodyDesc::fixed(Shape::cuboid(Vec3::new(0.5, 5.0, 5.0)))
                .local_position(Vec3::new(6.0, 0.0, 0.0)),
        )
        .unwrap();
        let (slow, held) = rod(&mut calm, 5, 2.0e5);
        let dense = calm.bodies.dense_index(slow).unwrap();
        calm.bodies.linear[dense] = Vec3::new(1.0, 0.0, 0.0);
        for _ in 0..300 {
            calm.step_fixed().unwrap();
        }
        assert_eq!(calm.structure_bodies(held).unwrap().len(), 1);
        let _ = structure;
    }

    #[test]
    fn spun_too_fast_it_flies_apart_and_slower_it_holds() {
        // the outer links carry the ends' m w^2 r: 1000 x 400 x 2 = 800 kN at 20 rad/s
        let mut world = no_gravity();
        let (body, structure) = rod(&mut world, 5, 2.0e5);
        spin(&mut world, body, 20.0);
        world.step_fixed().unwrap();
        assert!(!world.structure_events().is_empty());
        assert!(world.structure_bodies(structure).unwrap().len() > 1);
        // 8 kN at 2 rad/s
        let mut calm = no_gravity();
        let (slow, held) = rod(&mut calm, 5, 2.0e5);
        spin(&mut calm, slow, 2.0);
        for _ in 0..60 {
            calm.step_fixed().unwrap();
        }
        assert_eq!(calm.structure_bodies(held).unwrap().len(), 1);
        let most = calm
            .structure_links(held)
            .unwrap()
            .iter()
            .fold(0.0_f32, |most, link| most.max(link.load));
        assert!((most - 12_000.0).abs() < 1_500.0, "{most}");
    }

    #[test]
    fn a_link_held_near_its_strength_tires_and_gives() {
        let mut world = no_gravity();
        let (_, structure) = rod(&mut world, 2, 10_000.0);
        // pulled apart with 6 kN each way: 60% of its strength
        let mut broke_at = None;
        for step in 0..60 * 120 {
            world
                .push_part(structure, 0, Vec3::new(-6_000.0, 0.0, 0.0))
                .unwrap();
            world
                .push_part(structure, 1, Vec3::new(6_000.0, 0.0, 0.0))
                .unwrap();
            world.step_fixed().unwrap();
            if !world.structure_events().is_empty() {
                broke_at = Some(step);
                break;
            }
        }
        // 1 / (0.6 x 0.0007 x 60) is about 40 s
        let seconds = broke_at.unwrap() as f32 / 60.0;
        assert!((30.0..50.0).contains(&seconds), "{seconds}");
    }

    #[test]
    fn a_structure_continues_exactly_after_a_snapshot() {
        let mut world = no_gravity();
        let (body, structure) = rod(&mut world, 4, 5.0e4);
        world.break_link(structure, 1).unwrap();
        world.step_fixed().unwrap();
        assert_eq!(world.structure_bodies(structure).unwrap().len(), 2);
        let mut loaded = PhysicsWorld::load_snapshot(&world.save_snapshot()).unwrap();
        for _ in 0..30 {
            for side in [&mut world, &mut loaded] {
                side.push_part(structure, 3, Vec3::new(0.0, 100.0, 0.0))
                    .unwrap();
                side.step_fixed().unwrap();
            }
        }
        assert_eq!(loaded.save_snapshot(), world.save_snapshot());
        // its bodies are not reshaped or parked from outside
        let children = world.compound_children(body).unwrap();
        assert!(world.reshape_compound(body, &children).is_err());
        assert!(world.add_structure(body, &[]).is_err());
    }

    #[test]
    fn bad_structures_are_refused() {
        let mut world = no_gravity();
        let lone = world
            .spawn(RigidBodyDesc::dynamic(Shape::sphere(1.0)))
            .unwrap();
        assert!(world.add_structure(lone, &[]).is_err());
        let parts: Vec<CompoundChild> = (0..2)
            .map(|i| CompoundChild::new(Shape::sphere(0.5), Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let info = world.add_compound(&parts).unwrap();
        let pair = world.spawn(RigidBodyDesc::dynamic(info.shape)).unwrap();
        let link = |a, b, strength| LinkDesc { a, b, strength };
        for bad in [
            link(0, 0, 1.0),
            link(0, 2, 1.0),
            link(0, 1, 0.0),
            link(0, 1, f32::NAN),
        ] {
            assert!(world.add_structure(pair, &[bad]).is_err(), "{bad:?}");
        }
        let held = world.add_structure(pair, &[link(0, 1, 1.0)]).unwrap();
        assert!(world.break_link(held, 5).is_err());
        assert!(world.push_part(held, 9, Vec3::X).is_err());
        world.remove_structure(held).unwrap();
        assert!(world.structure_links(held).is_err());
    }
}
