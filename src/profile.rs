use std::time::Duration;

const CAPACITY: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepReport {
    pub samples: usize,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub max: Duration,
}

/// Keeps the last 4096 step times.
#[derive(Debug, Clone, Default)]
pub(crate) struct StepTimes {
    samples: Vec<Duration>,
    next: usize,
}

impl StepTimes {
    pub(crate) fn record(&mut self, time: Duration) {
        if self.samples.len() < CAPACITY {
            self.samples.push(time);
        } else {
            self.samples[self.next] = time;
        }
        self.next = (self.next + 1) % CAPACITY;
    }

    pub(crate) fn report(&self) -> Option<StepReport> {
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let max = *sorted.last()?;
        Some(StepReport {
            samples: sorted.len(),
            p50: percentile(&sorted, 50),
            p95: percentile(&sorted, 95),
            p99: percentile(&sorted, 99),
            max,
        })
    }
}

fn percentile(sorted: &[Duration], percent: usize) -> Duration {
    let rank = (sorted.len() * percent).div_ceil(100).max(1);
    sorted[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_has_no_report() {
        assert_eq!(StepTimes::default().report(), None);
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let mut times = StepTimes::default();
        for ms in (1..=100).rev() {
            times.record(Duration::from_millis(ms));
        }
        let report = times.report().unwrap();
        assert_eq!(report.samples, 100);
        assert_eq!(report.p50, Duration::from_millis(50));
        assert_eq!(report.p95, Duration::from_millis(95));
        assert_eq!(report.p99, Duration::from_millis(99));
        assert_eq!(report.max, Duration::from_millis(100));
    }

    #[test]
    fn old_samples_get_replaced() {
        let mut times = StepTimes::default();
        times.record(Duration::from_secs(9));
        for _ in 0..CAPACITY {
            times.record(Duration::from_millis(1));
        }
        let report = times.report().unwrap();
        assert_eq!(report.samples, CAPACITY);
        assert_eq!(report.max, Duration::from_millis(1));
    }
}
