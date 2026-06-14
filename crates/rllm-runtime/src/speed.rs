pub const RLLM_EXPERIMENTAL_SPEED_ENV: &str = "RLLM_EXPERIMENTAL_SPEED";
pub const RLLM_TURBO_TOPK_ENV: &str = "RLLM_TURBO_TOPK";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaExperimentalSpeedConfig {
    pub enabled: bool,
    pub turbo_topk: Option<usize>,
}

impl RamaExperimentalSpeedConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: parse_experimental_speed_enabled(
                std::env::var(RLLM_EXPERIMENTAL_SPEED_ENV).ok().as_deref(),
            ),
            turbo_topk: parse_turbo_topk(std::env::var(RLLM_TURBO_TOPK_ENV).ok().as_deref()),
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            turbo_topk: None,
        }
    }

    pub fn topk_for_input(self, input_len: usize, default_topk: usize) -> usize {
        if input_len == 0 {
            return 0;
        }
        self.turbo_topk
            .unwrap_or(default_topk.max(1))
            .min(input_len)
            .max(1)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaExperimentalSpeedStats {
    pub sparse_projection_calls: usize,
    pub exact_fallbacks: usize,
    pub selected_topk_sum: usize,
    pub max_selected_topk: usize,
    pub estimated_skipped_madds: usize,
    pub peak_scratch_bytes: usize,
}

impl RamaExperimentalSpeedStats {
    pub fn add_assign(&mut self, other: Self) {
        self.sparse_projection_calls = self
            .sparse_projection_calls
            .saturating_add(other.sparse_projection_calls);
        self.exact_fallbacks = self.exact_fallbacks.saturating_add(other.exact_fallbacks);
        self.selected_topk_sum = self
            .selected_topk_sum
            .saturating_add(other.selected_topk_sum);
        self.max_selected_topk = self.max_selected_topk.max(other.max_selected_topk);
        self.estimated_skipped_madds = self
            .estimated_skipped_madds
            .saturating_add(other.estimated_skipped_madds);
        self.peak_scratch_bytes = self.peak_scratch_bytes.max(other.peak_scratch_bytes);
    }

    pub fn record_sparse_projection(
        &mut self,
        selected_topk: usize,
        input_len: usize,
        out_features: usize,
        projection_count: usize,
    ) {
        self.sparse_projection_calls = self.sparse_projection_calls.saturating_add(1);
        self.selected_topk_sum = self.selected_topk_sum.saturating_add(selected_topk);
        self.max_selected_topk = self.max_selected_topk.max(selected_topk);
        let skipped_per_row = input_len.saturating_sub(selected_topk);
        let skipped = skipped_per_row
            .saturating_mul(out_features)
            .saturating_mul(projection_count.max(1));
        self.estimated_skipped_madds = self.estimated_skipped_madds.saturating_add(skipped);
        let scratch = selected_topk.saturating_mul(std::mem::size_of::<usize>());
        self.peak_scratch_bytes = self.peak_scratch_bytes.max(scratch);
    }

    pub fn record_exact_fallback(&mut self) {
        self.exact_fallbacks = self.exact_fallbacks.saturating_add(1);
    }

    pub fn is_empty(self) -> bool {
        self.sparse_projection_calls == 0
            && self.exact_fallbacks == 0
            && self.selected_topk_sum == 0
            && self.max_selected_topk == 0
            && self.estimated_skipped_madds == 0
            && self.peak_scratch_bytes == 0
    }
}

pub fn parse_experimental_speed_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn parse_turbo_topk(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

pub fn select_top_abs_indices(input: &[f32], topk: usize) -> Vec<usize> {
    let limit = topk.min(input.len());
    if limit == 0 {
        return Vec::new();
    }

    let mut scored: Vec<(usize, f32)> = input
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, value.abs()))
        .collect();
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut indices: Vec<usize> = scored.into_iter().take(limit).map(|(idx, _)| idx).collect();
    indices.sort_unstable();
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experimental_speed_env_parser_accepts_truthy_values() {
        assert!(parse_experimental_speed_enabled(Some("1")));
        assert!(parse_experimental_speed_enabled(Some("true")));
        assert!(parse_experimental_speed_enabled(Some("yes")));
        assert!(parse_experimental_speed_enabled(Some("on")));
        assert!(!parse_experimental_speed_enabled(Some("0")));
        assert!(!parse_experimental_speed_enabled(Some("false")));
        assert!(!parse_experimental_speed_enabled(Some("")));
        assert!(!parse_experimental_speed_enabled(None));
    }

    #[test]
    fn parse_turbo_topk_keeps_only_positive_values() {
        assert_eq!(parse_turbo_topk(Some("256")), Some(256));
        assert_eq!(parse_turbo_topk(Some("1")), Some(1));
        assert_eq!(parse_turbo_topk(Some("0")), None);
        assert_eq!(parse_turbo_topk(Some("-7")), None);
        assert_eq!(parse_turbo_topk(Some("bad")), None);
        assert_eq!(parse_turbo_topk(None), None);
    }

    #[test]
    fn top_abs_indices_are_deterministic_and_sorted_for_access() {
        let input = [0.5, -4.0, 3.0, 4.0, -0.25];
        assert_eq!(select_top_abs_indices(&input, 3), vec![1, 2, 3]);
        assert_eq!(select_top_abs_indices(&input, 99), vec![0, 1, 2, 3, 4]);
        assert!(select_top_abs_indices(&input, 0).is_empty());
    }

    #[test]
    fn config_chooses_bounded_topk() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            turbo_topk: Some(512),
        };
        assert_eq!(config.topk_for_input(2048, 256), 512);
        assert_eq!(config.topk_for_input(128, 256), 128);

        let defaulted = RamaExperimentalSpeedConfig {
            enabled: true,
            turbo_topk: None,
        };
        assert_eq!(defaulted.topk_for_input(2048, 256), 256);
        assert_eq!(defaulted.topk_for_input(32, 256), 32);
    }

    #[test]
    fn stats_record_sparse_calls_and_merge() {
        let mut stats = RamaExperimentalSpeedStats::default();
        assert!(stats.is_empty());
        stats.record_sparse_projection(4, 16, 3, 64);
        stats.record_exact_fallback();

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_sparse_projection(2, 8, 1, 32);
        stats.add_assign(other);

        assert_eq!(stats.sparse_projection_calls, 2);
        assert_eq!(stats.exact_fallbacks, 1);
        assert_eq!(stats.selected_topk_sum, 6);
        assert_eq!(stats.max_selected_topk, 4);
        assert_eq!(stats.estimated_skipped_madds, 2496);
        assert_eq!(stats.peak_scratch_bytes, 32);
        assert!(!stats.is_empty());
    }
}
