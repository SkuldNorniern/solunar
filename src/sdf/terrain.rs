use std::collections::{BTreeMap, HashMap, VecDeque};

use algebrix::{DVec3, Vec3};

use crate::frame::FrameId;
use crate::{
    BodyHandle, BodyKind, CastHit, CollisionFilter, JobDispatch, PhysicsError, PhysicsWorld,
    QueryFilter, Ray, RigidBodyDesc, Sequential, Shape,
};

use super::jobs::{BrickJob, BuiltBricks, JobBrick, MAX_BATCH, Queued};
use super::volume::{EditEntry, SdfVolume};
use super::{
    BrickKey, BrickMesh, Edit, EditId, EditOp, EditReport, EditShape, GROUND_LAYER, SdfDesc,
};

/// A body counts as swallowed when its center is nearer the surface than this share of its radius.
const EJECT_DEPTH: f32 = 0.25;
/// How far inside the material a body's center must be before it is rescued, meters. Its sample
/// is the nearest one, so a little under a surface is not told apart from on it.
const RESCUE_DEPTH: f32 = 0.3;
/// Room left between an ejected body and the surface it lands on, meters.
const EJECT_GAP: f32 = 0.02;

/// An [`SdfVolume`] in a world: each surface brick is a fixed body with its triangles. Changes
/// land between steps, every brick of one edit at once, so a step never sees half a hole.
#[derive(Debug)]
pub struct SdfTerrain {
    volume: SdfVolume,
    bodies: HashMap<BrickKey, (BodyHandle, Shape)>,
    queue: VecDeque<Queued>,
    /// What the job handed out holds, until it commits or is abandoned.
    flying: Vec<Queued>,
    token: u64,
    /// The frame its bricks stand in.
    frame: FrameId,
}

impl SdfTerrain {
    /// # Errors
    /// Fails on a bad description or a brick the world will not take. Nothing stays behind.
    pub fn new(world: &mut PhysicsWorld, desc: SdfDesc) -> Result<Self, PhysicsError> {
        Self::new_with(world, desc, &Sequential)
    }

    /// [`Self::new`] with bricks built on the host's workers.
    ///
    /// # Errors
    /// See [`Self::new`], and a dispatcher that skips jobs.
    pub fn new_with(
        world: &mut PhysicsWorld,
        desc: SdfDesc,
        dispatch: &dyn JobDispatch,
    ) -> Result<Self, PhysicsError> {
        Self::new_in(world, desc, FrameId::ROOT, dispatch)
    }

    /// [`Self::new_with`] with its bricks in `frame`, as a planet's ground in its
    /// [`PlanetGround::frame`](crate::PlanetGround::frame).
    ///
    /// # Errors
    /// See [`Self::new_with`], and a frame the world does not have.
    pub fn new_in(
        world: &mut PhysicsWorld,
        desc: SdfDesc,
        frame: FrameId,
        dispatch: &dyn JobDispatch,
    ) -> Result<Self, PhysicsError> {
        let mut terrain = Self {
            volume: SdfVolume::new_with(desc, dispatch)?,
            bodies: HashMap::new(),
            queue: VecDeque::new(),
            flying: Vec::new(),
            token: 0,
            frame,
        };
        let mut keys: Vec<BrickKey> = terrain
            .volume
            .surface_bricks()
            .map(|(key, ..)| key)
            .collect();
        keys.sort_unstable();
        for key in keys {
            let mesh = terrain.volume.brick_mesh(key).cloned().unwrap_or_default();
            match terrain.collider(world, key, &mesh) {
                Ok(Some(body)) => {
                    terrain.bodies.insert(key, body);
                }
                Ok(None) => {}
                Err(error) => {
                    terrain.remove(world);
                    return Err(error);
                }
            }
        }
        Ok(terrain)
    }

    pub fn volume(&self) -> &SdfVolume {
        &self.volume
    }

    /// See [`SdfVolume::set_cache_limit`].
    pub fn set_cache_limit(&mut self, bytes: usize) {
        self.volume.set_cache_limit(bytes);
    }

    /// The first ground along `ray`, whatever other bodies are in the way.
    ///
    /// # Errors
    /// Fails on a bad ray.
    pub fn raycast(
        &self,
        world: &PhysicsWorld,
        ray: &Ray,
    ) -> Result<Option<CastHit>, PhysicsError> {
        world.raycast(
            ray,
            &QueryFilter {
                mask: GROUND_LAYER,
                ..QueryFilter::default()
            },
        )
    }

    pub fn body_of(&self, key: BrickKey) -> Option<BodyHandle> {
        self.bodies.get(&key).map(|(body, _)| *body)
    }

    pub fn bodies(&self) -> impl Iterator<Item = BodyHandle> + '_ {
        self.bodies.values().map(|(body, _)| *body)
    }

    fn collider(
        &self,
        world: &mut PhysicsWorld,
        key: BrickKey,
        mesh: &BrickMesh,
    ) -> Result<Option<(BodyHandle, Shape)>, PhysicsError> {
        if mesh.is_empty() {
            return Ok(None);
        }
        let shape = world.add_mesh(&mesh.vertices, &mesh.triangles)?;
        self.place(world, key, shape).map(Some)
    }

    /// A fixed body for a brick's mesh. On failure the mesh is gone too.
    fn place(
        &self,
        world: &mut PhysicsWorld,
        key: BrickKey,
        shape: Shape,
    ) -> Result<(BodyHandle, Shape), PhysicsError> {
        let corner = DVec3::from_vec3(self.volume.brick_corner(key));
        let placed = self
            .volume
            .desc
            .origin
            .shifted(corner)
            .ok_or(PhysicsError::OutOfRange)
            .and_then(|position| {
                let mut desc = RigidBodyDesc::fixed(shape)
                    .position(position)
                    .material(self.volume.desc.material)
                    .filter(CollisionFilter {
                        layers: CollisionFilter::default().layers | GROUND_LAYER,
                        mask: u32::MAX,
                    });
                desc.frame = self.frame;
                world.spawn(desc)
            });
        match placed {
            Ok(body) => Ok((body, shape)),
            Err(error) => {
                world.remove_mesh(shape)?;
                Err(error)
            }
        }
    }

    /// Asks for an edit. It lands when a job holding it commits, so it can be built away from
    /// the step. Edits land in the order they were asked for.
    ///
    /// # Errors
    /// Fails on a bad shape.
    pub fn queue_edit(&mut self, edit: Edit) -> Result<EditId, PhysicsError> {
        edit.shape.validate()?;
        let id = EditId(self.volume.next_edit);
        self.volume.next_edit += 1;
        let entry = self.volume.entry(edit);
        self.queue.push_back(Queued::Edit(id, entry));
        Ok(id)
    }

    /// Asks for a tunnel to be taken out, the ground it cut comes back.
    ///
    /// # Errors
    /// Fails when `id` is not a tunnel here, or it is already on its way out.
    pub fn queue_close_tunnel(&mut self, id: EditId) -> Result<(), PhysicsError> {
        let entry = self.open_tunnel(id)?;
        self.queue.push_back(Queued::Close(id, entry));
        Ok(())
    }

    /// Asks for a tunnel to be somewhere else: the hole moves and the ground closes behind it.
    /// It keeps its place among the edits, and a cost of only the bricks it leaves and reaches. A
    /// move still waiting is replaced, so asking faster than jobs finish does not pile up.
    ///
    /// # Errors
    /// Fails on a bad shape, when `id` is not a tunnel here, or it is on its way out.
    pub fn queue_move_tunnel(&mut self, id: EditId, shape: EditShape) -> Result<(), PhysicsError> {
        shape.validate()?;
        let old = self.open_tunnel(id)?;
        let new = self.volume.entry(Edit {
            shape,
            op: EditOp::Tunnel,
        });
        if let Some(Queued::Move {
            id: waiting,
            new: place,
            ..
        }) = self.queue.back_mut()
            && *waiting == id
        {
            *place = new;
            return Ok(());
        }
        self.queue.push_back(Queued::Move { id, old, new });
        Ok(())
    }

    /// Where a tunnel will be once everything asked for has landed.
    fn open_tunnel(&self, id: EditId) -> Result<EditEntry, PhysicsError> {
        let mut latest = self.volume.edits.get(&id).copied();
        for item in self.flying.iter().chain(&self.queue) {
            match *item {
                Queued::Edit(other, entry)
                | Queued::Move {
                    id: other,
                    new: entry,
                    ..
                } if other == id => {
                    latest = Some(entry);
                }
                Queued::Close(other, _) if other == id => latest = None,
                _ => {}
            }
        }
        latest
            .filter(|entry| entry.edit.op == EditOp::Tunnel)
            .ok_or(PhysicsError::InvalidHandle)
    }

    /// Edits asked for and not landed yet.
    pub fn pending(&self) -> usize {
        self.queue.len() + self.flying.len()
    }

    /// Whether a job is out and its result not committed.
    pub fn job_out(&self) -> bool {
        !self.flying.is_empty()
    }

    /// The next batch to build, `None` with one still out or nothing asked for. A batch is
    /// edits of one kind in a row, so what it did to the ground can be told apart.
    pub fn next_job(&mut self) -> Option<BrickJob> {
        if !self.flying.is_empty() {
            return None;
        }
        let first = self.queue.pop_front()?;
        self.flying.push(first);
        if let Queued::Edit(_, entry) = first {
            while self.flying.len() < MAX_BATCH {
                match self.queue.front() {
                    Some(Queued::Edit(_, next)) if next.edit.op == entry.edit.op => {}
                    _ => break,
                }
                if let Some(next) = self.queue.pop_front() {
                    self.flying.push(next);
                }
            }
        }
        let mut bricks: BTreeMap<BrickKey, JobBrick> = BTreeMap::new();
        for item in &self.flying {
            match *item {
                Queued::Edit(id, entry) => {
                    for key in self.volume.reached(&entry) {
                        let brick = bricks.entry(key).or_insert_with(|| {
                            let kept = self
                                .volume
                                .bricks
                                .get(&key)
                                .and_then(|brick| brick.values.clone());
                            let edits = if kept.is_some() {
                                Vec::new()
                            } else {
                                self.volume.entries_of(key, None, None)
                            };
                            JobBrick {
                                key,
                                kept,
                                edits,
                                added: Vec::new(),
                                closed: None,
                            }
                        });
                        brick.edits.push(entry);
                        brick.added.push(id);
                    }
                }
                Queued::Close(id, entry) => {
                    for key in self.volume.reached(&entry) {
                        bricks.insert(
                            key,
                            JobBrick {
                                key,
                                kept: None,
                                edits: self.volume.entries_of(key, None, Some(id)),
                                added: Vec::new(),
                                closed: Some(id),
                            },
                        );
                    }
                }
                Queued::Move { id, old, new } => {
                    let reaches = self.volume.reached(&new);
                    let mut keys = self.volume.reached(&old);
                    keys.extend(reaches.iter().copied());
                    for key in keys {
                        let moved = reaches.contains(&key);
                        bricks.entry(key).or_insert_with(|| JobBrick {
                            key,
                            kept: None,
                            edits: self.volume.entries_moved(key, id, moved.then_some(&new)),
                            added: if moved { vec![id] } else { Vec::new() },
                            closed: Some(id),
                        });
                    }
                }
            }
        }
        self.token += 1;
        Some(BrickJob {
            lattice: self.volume.lattice(),
            bricks: bricks.into_values().collect(),
            token: self.token,
        })
    }

    /// Drops the edits of the job that is out, its result is refused from now on.
    pub fn abandon(&mut self) {
        self.flying.clear();
        self.token += 1;
    }

    /// Puts a finished job in the world, every brick at once. Call it between steps.
    ///
    /// # Errors
    /// Fails on a result of a job abandoned or already committed, or a brick the world will not
    /// take. Then nothing changed.
    pub fn commit(
        &mut self,
        world: &mut PhysicsWorld,
        built: BuiltBricks,
    ) -> Result<EditReport, PhysicsError> {
        if built.token != self.token || self.flying.is_empty() {
            return Err(PhysicsError::InvalidHandle);
        }
        let mut rebuilt = built.rebuilt;
        let mut fresh = Vec::new();
        for item in &mut rebuilt {
            let Some(collider) = item.collider.take() else {
                continue;
            };
            let placed = world
                .push_mesh(collider)
                .and_then(|shape| self.place(world, item.key, shape));
            match placed {
                Ok(body) => fresh.push((item.key, body)),
                Err(error) => {
                    let bodies: Vec<_> = fresh.iter().map(|(_, body)| *body).collect();
                    release(world, &bodies);
                    return Err(error);
                }
            }
        }
        let keys: Vec<BrickKey> = rebuilt.iter().map(|item| item.key).collect();
        let old: Vec<_> = keys
            .iter()
            .filter_map(|key| self.bodies.remove(key))
            .collect();
        release(world, &old);
        self.bodies.extend(fresh);
        let (revision, change) = self.volume.commit(rebuilt);
        let volume_change = change as f64 * f64::from(self.volume.desc.voxel).powi(3);
        let density = f64::from(self.volume.desc.material.density);
        let mut report = EditReport {
            edit: EditId(0),
            edits: self.flying.len(),
            revision,
            bricks: keys.len(),
            volume_change,
            removed_mass: 0.0,
            ejected: 0,
        };
        for item in std::mem::take(&mut self.flying) {
            match item {
                Queued::Edit(id, entry) => {
                    self.volume.edits.insert(id, entry);
                    report.edit = id;
                    match entry.edit.op {
                        EditOp::Excavate => {
                            report.removed_mass = (-volume_change).max(0.0) * density;
                        }
                        EditOp::Add => report.ejected += self.eject(world, &entry)?,
                        EditOp::Tunnel => {}
                    }
                }
                Queued::Close(id, entry) => {
                    self.volume.edits.remove(&id);
                    report.edit = id;
                    report.ejected += self.eject(world, &entry)?;
                }
                Queued::Move { id, old, new } => {
                    self.volume.edits.insert(id, new);
                    report.edit = id;
                    report.ejected += self.eject(world, &old)?;
                }
            }
        }
        Ok(report)
    }

    /// Queue, build and commit in one call. Only with nothing else waiting.
    fn run_now(
        &mut self,
        world: &mut PhysicsWorld,
        dispatch: &dyn JobDispatch,
    ) -> Result<EditReport, PhysicsError> {
        let Some(job) = self.next_job() else {
            return Err(PhysicsError::TerrainBusy);
        };
        let done = job
            .run(dispatch)
            .and_then(|built| self.commit(world, built));
        if done.is_err() {
            self.abandon();
        }
        done
    }

    /// Applies one edit to the volume and the world.
    ///
    /// # Errors
    /// Fails on a bad shape, with other edits waiting, or when the world will not take a new
    /// brick. Then nothing changed.
    pub fn edit(
        &mut self,
        world: &mut PhysicsWorld,
        edit: Edit,
    ) -> Result<EditReport, PhysicsError> {
        self.edit_with(world, edit, &Sequential)
    }

    /// [`Self::edit`] with its bricks built on the host's workers. The result does not depend
    /// on them.
    ///
    /// # Errors
    /// See [`Self::edit`], and a dispatcher that skips jobs.
    pub fn edit_with(
        &mut self,
        world: &mut PhysicsWorld,
        edit: Edit,
        dispatch: &dyn JobDispatch,
    ) -> Result<EditReport, PhysicsError> {
        if self.pending() > 0 {
            return Err(PhysicsError::TerrainBusy);
        }
        self.queue_edit(edit)?;
        self.run_now(world, dispatch)
    }

    /// Takes a tunnel out again, the ground it cut comes back.
    ///
    /// # Errors
    /// Fails when `id` is not a tunnel here, with other edits waiting, or the world will not
    /// take a brick.
    pub fn remove_tunnel(
        &mut self,
        world: &mut PhysicsWorld,
        id: EditId,
    ) -> Result<EditReport, PhysicsError> {
        self.remove_tunnel_with(world, id, &Sequential)
    }

    /// [`Self::remove_tunnel`] with its bricks built on the host's workers.
    ///
    /// # Errors
    /// See [`Self::remove_tunnel`], and a dispatcher that skips jobs.
    pub fn remove_tunnel_with(
        &mut self,
        world: &mut PhysicsWorld,
        id: EditId,
        dispatch: &dyn JobDispatch,
    ) -> Result<EditReport, PhysicsError> {
        if self.pending() > 0 {
            return Err(PhysicsError::TerrainBusy);
        }
        self.queue_close_tunnel(id)?;
        self.run_now(world, dispatch)
    }

    /// Moves dynamic bodies that new material swallowed straight up to the surface.
    /// Triangles only push from the front, so a body left under them would fall out of the world.
    fn eject(&self, world: &mut PhysicsWorld, entry: &EditEntry) -> Result<usize, PhysicsError> {
        let step = self.volume.desc.voxel;
        let origin = self.volume.desc.origin;
        let (low, high) = (
            Vec3::new(entry.low.x as f32, entry.low.y as f32, entry.low.z as f32) * step,
            Vec3::new(
                entry.high.x as f32,
                entry.high.y as f32,
                entry.high.z as f32,
            ) * step,
        );
        let mut buried = Vec::new();
        for body in world.body_handles() {
            let state = world.body_state(body)?;
            if state.kind != BodyKind::Dynamic {
                continue;
            }
            let at = state.position.offset_from(origin).as_vec3();
            if at.max(low) != at || at.min(high) != at {
                continue;
            }
            let shape = world.body_shape(body)?;
            let radius = world.bodies.assets.bounding_radius(&shape);
            // bodies at rest sit half a radius or more above the surface, so this leaves them be
            if self.volume.distance(at) < radius * EJECT_DEPTH {
                buried.push((body, at, radius));
            }
        }
        self.lift(world, buried)
    }

    /// Lifts awake dynamic bodies whose center got inside the material or under the volume.
    /// Triangles only push from the front, so hard pressure or a fast small body can get past
    /// them, and then nothing would stop it. Asleep bodies are left out, they did not move. Call
    /// it between steps. Returns how many were lifted.
    ///
    /// # Errors
    /// Fails when a body can not be placed in range.
    pub fn rescue(&self, world: &mut PhysicsWorld) -> Result<usize, PhysicsError> {
        let origin = self.volume.desc.origin;
        let top = self.volume.brick_corner(BrickKey(self.volume.desc.bricks));
        let mut lost = Vec::new();
        for dense in 0..world.bodies.len() {
            if world.bodies.kind[dense] != BodyKind::Dynamic || world.bodies.asleep[dense] {
                continue;
            }
            let mut at = world.bodies.position(dense).offset_from(origin).as_vec3();
            if at.x < 0.0 || at.z < 0.0 || at.x > top.x || at.z > top.z || at.y > top.y {
                continue;
            }
            let under = at.y < 0.0;
            if under || self.volume.quick_distance(at) < -RESCUE_DEPTH {
                at.y = at.y.max(0.0);
                let radius = world
                    .bodies
                    .assets
                    .bounding_radius(&world.bodies.shape[dense]);
                lost.push((world.bodies.handle_at(dense), at, radius));
            }
        }
        if lost.is_empty() {
            return Ok(0);
        }
        self.lift(world, lost)
    }

    /// Moves bodies straight up until each is clear of the ground and of the others.
    fn lift(
        &self,
        world: &mut PhysicsWorld,
        mut buried: Vec<(BodyHandle, Vec3, f32)>,
    ) -> Result<usize, PhysicsError> {
        let step = self.volume.desc.voxel;
        let origin = self.volume.desc.origin;
        // lowest first, so a column of bodies lands in the order it stood
        buried.sort_by(|a, b| a.1.y.total_cmp(&b.1.y));
        let mut placed: Vec<(Vec3, f32)> = Vec::with_capacity(buried.len());
        let top = self
            .volume
            .brick_corner(BrickKey(self.volume.desc.bricks))
            .y;
        for (body, mut at, radius) in buried {
            let clear = |at: Vec3, placed: &[(Vec3, f32)]| {
                self.volume.distance(at) >= radius + EJECT_GAP
                    && placed
                        .iter()
                        .all(|(other, reach)| (at - *other).length() >= radius + reach)
            };
            while at.y < top && !clear(at, &placed) {
                at.y += step;
            }
            placed.push((at, radius));
            let position = origin
                .shifted(DVec3::from_vec3(at))
                .ok_or(PhysicsError::OutOfRange)?;
            world.teleport(body, position)?;
            world.set_velocity(body, Vec3::ZERO, Vec3::ZERO)?;
        }
        Ok(placed.len())
    }

    /// Takes every brick body out of the world.
    pub fn remove(self, world: &mut PhysicsWorld) {
        let bodies: Vec<_> = self.bodies.into_values().collect();
        release(world, &bodies);
    }
}

fn release(world: &mut PhysicsWorld, bodies: &[(BodyHandle, Shape)]) {
    for (body, shape) in bodies {
        // both are ours and alive, so these only fail if someone else removed them first
        let _ = world.despawn(*body);
        let _ = world.remove_mesh(*shape);
    }
}
