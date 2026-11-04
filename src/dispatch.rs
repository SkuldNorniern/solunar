/// Host side job runner. Lets a host share its workers with the solver.
pub trait JobDispatch {
    fn parallelism(&self) -> usize;

    /// Runs `job` once for every index in `0..jobs`, returns when all are finished.
    fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync));
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sequential;

impl JobDispatch for Sequential {
    fn parallelism(&self) -> usize {
        1
    }

    fn run(&self, jobs: usize, job: &(dyn Fn(usize) + Sync)) {
        for index in 0..jobs {
            job(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn sequential_runs_every_job_once() {
        let total = AtomicUsize::new(0);
        Sequential.run(5, &|index| {
            total.fetch_add(index + 1, Ordering::Relaxed);
        });
        assert_eq!(total.into_inner(), 15);
    }
}
