// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use crate::RamaRollingStats;

pub(crate) const RLLM_ROLLING_ENV: &str = "RLLM_ROLLING";

#[derive(Debug, Clone, Copy)]
pub(crate) struct RollingExecutorConfig {
    pub enabled: bool,
    pub worker_count: usize,
    pub min_rows_per_worker: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct RollingExecutor {
    config: RollingExecutorConfig,
    stats: RamaRollingStats,
}

impl RollingExecutor {
    pub(crate) fn new(config: RollingExecutorConfig) -> Self {
        Self {
            config: RollingExecutorConfig {
                enabled: config.enabled,
                worker_count: config.worker_count.max(1),
                min_rows_per_worker: config.min_rows_per_worker.max(1),
            },
            stats: RamaRollingStats::default(),
        }
    }

    pub(crate) fn from_env(available_workers: usize) -> Option<Self> {
        if !parse_rolling_enabled(std::env::var(RLLM_ROLLING_ENV).ok().as_deref()) {
            return None;
        }
        Some(Self::new(RollingExecutorConfig {
            enabled: true,
            worker_count: available_workers.max(1),
            min_rows_per_worker: 4,
        }))
    }

    pub(crate) fn effective_workers_for_rows(&self, rows: usize) -> usize {
        if !self.config.enabled {
            return 1;
        }
        let workers = self.config.worker_count.min(rows).max(1);
        if rows < workers.saturating_mul(self.config.min_rows_per_worker) {
            1
        } else {
            workers
        }
    }

    pub(crate) fn record_parallel_batch(&mut self, workers: usize, scratch_bytes: usize) {
        self.stats.submitted_tasks = self.stats.submitted_tasks.saturating_add(workers);
        self.stats.worker_wakeups = self.stats.worker_wakeups.saturating_add(workers);
        self.stats.peak_scratch_bytes = self.stats.peak_scratch_bytes.max(scratch_bytes);
    }

    pub(crate) fn record_sequential_fallback(&mut self) {
        self.stats.sequential_fallbacks = self.stats.sequential_fallbacks.saturating_add(1);
    }

    pub(crate) fn take_stats(&mut self) -> RamaRollingStats {
        let stats = self.stats;
        self.stats = RamaRollingStats::default();
        stats
    }
}

pub(crate) fn parse_rolling_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_env_parser_accepts_explicit_truthy_values() {
        assert!(parse_rolling_enabled(Some("1")));
        assert!(parse_rolling_enabled(Some("true")));
        assert!(parse_rolling_enabled(Some("yes")));
        assert!(parse_rolling_enabled(Some("on")));
        assert!(!parse_rolling_enabled(Some("0")));
        assert!(!parse_rolling_enabled(Some("false")));
        assert!(!parse_rolling_enabled(Some("")));
        assert!(!parse_rolling_enabled(None));
    }

    #[test]
    fn executor_policy_skips_small_row_counts() {
        let executor = RollingExecutor::new(RollingExecutorConfig {
            enabled: true,
            worker_count: 4,
            min_rows_per_worker: 4,
        });

        assert_eq!(executor.effective_workers_for_rows(0), 1);
        assert_eq!(executor.effective_workers_for_rows(3), 1);
        assert_eq!(executor.effective_workers_for_rows(16), 4);
        assert_eq!(executor.effective_workers_for_rows(6), 1);
    }

    #[test]
    fn executor_records_batches_and_fallbacks() {
        let mut executor = RollingExecutor::new(RollingExecutorConfig {
            enabled: true,
            worker_count: 3,
            min_rows_per_worker: 2,
        });

        executor.record_parallel_batch(3, 128);
        executor.record_sequential_fallback();
        let stats = executor.take_stats();

        assert_eq!(stats.submitted_tasks, 3);
        assert_eq!(stats.worker_wakeups, 3);
        assert_eq!(stats.sequential_fallbacks, 1);
        assert_eq!(stats.peak_scratch_bytes, 128);
        assert!(executor.take_stats().is_empty());
    }
}
