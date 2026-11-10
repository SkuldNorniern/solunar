use std::sync::{Arc, Mutex, PoisonError};

use crate::JobDispatch;
use crate::PhysicsError;
use crate::collision::trimesh::TriMesh;

use super::compute::{BrickSamples, CpuEdits, EditSpan, SdfCompute};
use super::volume::{EditEntry, SdfVolume};
use super::{BrickKey, BrickMesh, EditId};

/// Most edits one job takes, so a long queue still commits in steps.
pub(super) const MAX_BATCH: usize = 64;

/// A brick built again but not yet in the volume.
#[derive(Debug)]
pub(super) struct Rebuilt {
    pub(super) key: BrickKey,
    pub(super) inside: u64,
    pub(super) mesh: BrickMesh,
    pub(super) values: Option<Arc<Vec<f32>>>,
    pub(super) collider: Option<TriMesh>,
    pub(super) added: Vec<EditId>,
    pub(super) closed: Option<EditId>,
}

/// One brick's work: start from its kept samples and run `edits`, or from scratch with `edits`
/// being every edit that reaches it.
#[derive(Debug)]
pub(super) struct JobBrick {
    pub(super) key: BrickKey,
    pub(super) kept: Option<Arc<Vec<f32>>>,
    pub(super) edits: Vec<EditEntry>,
    pub(super) added: Vec<EditId>,
    pub(super) closed: Option<EditId>,
}

/// Building a batch of bricks, cut off from the terrain and the world, so it can run on any
/// thread while the world steps. Comes from [`SdfTerrain::next_job`](super::SdfTerrain::next_job).
#[derive(Debug)]
pub struct BrickJob {
    pub(super) lattice: SdfVolume,
    pub(super) bricks: Vec<JobBrick>,
    pub(super) token: u64,
}

impl BrickJob {
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// Samples, meshes and readies colliders for every brick. The result does not depend on
    /// `dispatch`.
    ///
    /// # Errors
    /// Fails on triangles a collider will not take, or a dispatcher that skips jobs. Then call
    /// [`SdfTerrain::abandon`](super::SdfTerrain::abandon).
    pub fn run(self, dispatch: &dyn JobDispatch) -> Result<BuiltBricks, PhysicsError> {
        self.run_inner(dispatch, None)
    }

    /// [`Self::run`] with the edits applied by `compute`, as a gpu. Base sampling and meshing
    /// stay on `dispatch`. A compute that turns the work down leaves it to the cpu, the result
    /// is the same either way.
    ///
    /// # Errors
    /// As [`Self::run`].
    pub fn run_with(
        self,
        dispatch: &dyn JobDispatch,
        compute: &dyn SdfCompute,
    ) -> Result<BuiltBricks, PhysicsError> {
        self.run_inner(dispatch, Some(compute))
    }

    fn run_inner(
        self,
        dispatch: &dyn JobDispatch,
        compute: Option<&dyn SdfCompute>,
    ) -> Result<BuiltBricks, PhysicsError> {
        let Some(compute) = compute else {
            return self.run_cpu(dispatch);
        };
        let lattice = &self.lattice;
        // a brick with no edits that is known solid or empty is built as it is
        let starts = in_parallel(dispatch, &self.bricks, |brick| match &brick.kept {
            Some(kept) => Some(Vec::clone(kept)),
            None if brick.edits.is_empty() && lattice.known(brick.key) => None,
            None => Some(lattice.base_samples(brick.key)),
        })?;
        let spans: Vec<Vec<EditSpan>> = self
            .bricks
            .iter()
            .map(|brick| lattice.spans(brick.key, &brick.edits))
            .collect();
        let side = lattice.sample_side();
        let step = lattice.desc().voxel;
        let mut trial = starts.clone();
        let mut asked: Vec<BrickSamples<'_>> = trial
            .iter_mut()
            .zip(&self.bricks)
            .zip(&spans)
            .filter_map(|((values, brick), edits)| {
                let values = values.as_mut()?;
                Some(BrickSamples {
                    low: lattice.brick_samples(brick.key).0,
                    side,
                    voxel: step,
                    edits,
                    values,
                })
            })
            .collect();
        let done = compute.apply(&mut asked)
            && asked
                .iter()
                .all(|brick| brick.values.len() == side * side * side);
        drop(asked);
        let values = if done {
            trial
        } else {
            let mut values = starts;
            let mut again: Vec<BrickSamples<'_>> = values
                .iter_mut()
                .zip(&self.bricks)
                .zip(&spans)
                .filter_map(|((values, brick), edits)| {
                    Some(BrickSamples {
                        low: lattice.brick_samples(brick.key).0,
                        side,
                        voxel: step,
                        edits,
                        values: values.as_mut()?,
                    })
                })
                .collect();
            CpuEdits.apply(&mut again);
            drop(again);
            values
        };
        let work: Vec<(&JobBrick, Option<Vec<f32>>)> = self.bricks.iter().zip(values).collect();
        let built = in_parallel(dispatch, &work, |(brick, values)| {
            let item = match values {
                Some(values) => lattice.finish(brick.key, values.clone()),
                None => lattice.build(brick.key, &brick.edits),
            };
            ready(item, brick)
        })?;
        let rebuilt = built
            .into_iter()
            .collect::<Result<Vec<_>, PhysicsError>>()?;
        Ok(BuiltBricks {
            rebuilt,
            token: self.token,
        })
    }

    fn run_cpu(self, dispatch: &dyn JobDispatch) -> Result<BuiltBricks, PhysicsError> {
        let lattice = &self.lattice;
        let built = in_parallel(dispatch, &self.bricks, |brick| {
            let mut item = match &brick.kept {
                Some(kept) => {
                    let mut values = Vec::clone(kept);
                    for entry in &brick.edits {
                        lattice.apply(brick.key, &mut values, entry);
                    }
                    lattice.finish(brick.key, values)
                }
                None => lattice.build(brick.key, &brick.edits),
            };
            item.added.clone_from(&brick.added);
            item.closed = brick.closed;
            if !item.mesh.is_empty() {
                item.collider = Some(TriMesh::from_triangles(
                    item.mesh.vertices.clone(),
                    item.mesh.triangles.clone(),
                )?);
            }
            Ok(item)
        })?;
        let rebuilt = built
            .into_iter()
            .collect::<Result<Vec<_>, PhysicsError>>()?;
        Ok(BuiltBricks {
            rebuilt,
            token: self.token,
        })
    }
}

/// A brick's result with what the job kept of it and its collider made.
fn ready(mut item: Rebuilt, brick: &JobBrick) -> Result<Rebuilt, PhysicsError> {
    item.added.clone_from(&brick.added);
    item.closed = brick.closed;
    if !item.mesh.is_empty() {
        item.collider = Some(TriMesh::from_triangles(
            item.mesh.vertices.clone(),
            item.mesh.triangles.clone(),
        )?);
    }
    Ok(item)
}

/// A finished [`BrickJob`], for [`SdfTerrain::commit`](super::SdfTerrain::commit).
#[derive(Debug)]
pub struct BuiltBricks {
    pub(super) rebuilt: Vec<Rebuilt>,
    pub(super) token: u64,
}

/// Work waiting its turn, in the order it was asked for.
#[derive(Debug, Clone, Copy)]
pub(super) enum Queued {
    Edit(EditId, EditEntry),
    Close(EditId, EditEntry),
    Move {
        id: EditId,
        old: EditEntry,
        new: EditEntry,
    },
}

/// `work` for each item on the dispatcher's workers, results in item order.
pub(super) fn in_parallel<T: Sync, R: Send>(
    dispatch: &dyn JobDispatch,
    items: &[T],
    work: impl Fn(&T) -> R + Sync,
) -> Result<Vec<R>, PhysicsError> {
    let slots: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    dispatch.run(items.len(), &|index| {
        let result = work(&items[index]);
        *slots[index].lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
    });
    slots
        .into_iter()
        .map(|slot| slot.into_inner().unwrap_or_else(PoisonError::into_inner))
        .collect::<Option<Vec<R>>>()
        .ok_or(PhysicsError::DispatchIncomplete)
}
