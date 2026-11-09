use std::sync::Mutex;

use crate::{CastHit, JobDispatch, PhysicsError, PhysicsWorld, QueryFilter, Ray, ShapeCast};

/// Queries a job answers together, so a job's setup is not paid per query.
const PER_JOB: usize = 64;

impl PhysicsWorld {
    /// Every ray, spread over `dispatch`. `out` gets one answer a ray, in order, each the same as
    /// [`Self::raycast`] gives alone, on any number of workers.
    pub fn raycast_batch(
        &self,
        rays: &[Ray],
        filter: &QueryFilter,
        dispatch: &dyn JobDispatch,
        out: &mut Vec<Result<Option<CastHit>, PhysicsError>>,
    ) {
        self.batch(rays, dispatch, out, &|ray| self.raycast(ray, filter));
    }

    /// Whether each ray is blocked, spread over `dispatch`, as [`Self::ray_blocked`] answers
    /// one. For many light or sound paths at once.
    pub fn ray_blocked_batch(
        &self,
        rays: &[Ray],
        filter: &QueryFilter,
        dispatch: &dyn JobDispatch,
        out: &mut Vec<Result<bool, PhysicsError>>,
    ) {
        self.batch(rays, dispatch, out, &|ray| self.ray_blocked(ray, filter));
    }

    /// Every cast, spread over `dispatch`, as [`Self::raycast_batch`] does rays.
    pub fn shape_cast_batch(
        &self,
        casts: &[ShapeCast],
        filter: &QueryFilter,
        dispatch: &dyn JobDispatch,
        out: &mut Vec<Result<Option<CastHit>, PhysicsError>>,
    ) {
        self.batch(casts, dispatch, out, &|cast| self.shape_cast(cast, filter));
    }

    fn batch<Q: Sync, A: Send>(
        &self,
        queries: &[Q],
        dispatch: &dyn JobDispatch,
        out: &mut Vec<A>,
        one: &(dyn Fn(&Q) -> A + Sync),
    ) {
        out.clear();
        if queries.is_empty() {
            return;
        }
        // built here once, not raced for by the first job of every worker
        self.bodies.tree.get(&self.bodies);
        let chunks: Vec<&[Q]> = queries.chunks(PER_JOB).collect();
        let answers: Vec<Mutex<Vec<A>>> = chunks.iter().map(|_| Mutex::new(Vec::new())).collect();
        dispatch.run(chunks.len(), &|job| {
            let found: Vec<A> = chunks[job].iter().map(one).collect();
            if let Ok(mut slot) = answers[job].lock() {
                *slot = found;
            }
        });
        out.reserve(queries.len());
        for slot in answers {
            out.extend(slot.into_inner().unwrap_or_default());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dispatch::testing::Threads;
    use algebrix::{Quat, Vec3};

    use super::*;
    use crate::{GridPosition, PhysicsSettings, RigidBodyDesc, Sequential, Shape};

    #[test]
    fn batches_answer_as_single_queries_do_on_any_worker_count() {
        let mut state = 0x5eed_1234_9876_abcd_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 1000.0 - 1.0
        };
        let mut world = PhysicsWorld::new(PhysicsSettings::default()).unwrap();
        for _ in 0..400 {
            let at = Vec3::new(next(), next(), next()) * 30.0;
            world
                .spawn(RigidBodyDesc::fixed(Shape::sphere(0.8)).local_position(at))
                .unwrap();
        }
        let at = |v: Vec3| GridPosition::from_local(v).unwrap();
        let rays: Vec<Ray> = (0..300)
            .map(|_| {
                let from = Vec3::new(next(), next(), next()) * 40.0;
                // toward the field, a little off
                let toward = -from + Vec3::new(next(), next(), next()) * 15.0;
                Ray {
                    origin: at(from),
                    direction: toward,
                    max_distance: 80.0,
                }
            })
            .collect();
        let casts: Vec<ShapeCast> = rays
            .iter()
            .map(|ray| ShapeCast {
                shape: Shape::sphere(0.3),
                rotation: Quat::IDENTITY,
                origin: ray.origin,
                direction: ray.direction,
                max_distance: ray.max_distance,
            })
            .collect();
        let filter = QueryFilter::default();
        let alone: Vec<_> = rays.iter().map(|ray| world.raycast(ray, &filter)).collect();
        let cast_alone: Vec<_> = casts
            .iter()
            .map(|cast| world.shape_cast(cast, &filter))
            .collect();
        assert!(
            alone
                .iter()
                .filter(|hit| matches!(hit, Ok(Some(_))))
                .count()
                > 30
        );
        let mut out = Vec::new();
        world.raycast_batch(&rays, &filter, &Sequential, &mut out);
        assert_eq!(out, alone);
        for workers in [2, 7] {
            world.raycast_batch(&rays, &filter, &Threads(workers), &mut out);
            assert_eq!(out, alone);
            world.shape_cast_batch(&casts, &filter, &Threads(workers), &mut out);
            assert_eq!(out, cast_alone);
        }
        // blocked exactly where the nearest hit is in reach, on any worker count
        let wanted: Vec<_> = alone
            .iter()
            .map(|hit| hit.as_ref().map(Option::is_some).map_err(Clone::clone))
            .collect();
        let mut blocked = Vec::new();
        for workers in [1, 3] {
            world.ray_blocked_batch(&rays, &filter, &Threads(workers), &mut blocked);
            assert_eq!(blocked, wanted);
        }
        // a bad ray answers with its own error, the rest still answer
        let mut mixed = rays[..3].to_vec();
        mixed[1].direction = Vec3::ZERO;
        world.raycast_batch(&mixed, &filter, &Threads(2), &mut out);
        assert!(out[1].is_err() && out[0] == alone[0] && out[2] == alone[2]);
        world.raycast_batch(&[], &filter, &Threads(2), &mut out);
        assert!(out.is_empty());
    }
}
