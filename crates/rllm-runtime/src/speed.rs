pub const RLLM_EXPERIMENTAL_SPEED_ENV: &str = "RLLM_EXPERIMENTAL_SPEED";
pub const RLLM_AIP_POLICY_ENV: &str = "RLLM_AIP_POLICY";
pub const RLLM_AIP_TOPK_ENV: &str = "RLLM_AIP_TOPK";
pub const RLLM_AIP_LM_HEAD_ROWS_ENV: &str = "RLLM_AIP_LM_HEAD_ROWS";
pub const RLLM_AIP_COLUMN_CACHE_ENV: &str = "RLLM_AIP_COLUMN_CACHE";
pub const RLLM_AIP_INPUT_TILES_ENV: &str = "RLLM_AIP_INPUT_TILES";
pub const RLLM_TURBO_TOPK_ENV: &str = "RLLM_TURBO_TOPK";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RamaAipPolicyKind {
    #[default]
    Quality,
    Speed,
}

impl RamaAipPolicyKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Speed => "speed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamaAipProjectionKind {
    MlpGateUp,
    MlpDown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaAipProjectionDecision {
    pub enabled: bool,
    pub topk: usize,
}

impl RamaAipProjectionDecision {
    pub fn exact() -> Self {
        Self {
            enabled: false,
            topk: 0,
        }
    }

    pub fn aip(topk: usize) -> Self {
        Self {
            enabled: true,
            topk,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaExperimentalSpeedConfig {
    pub enabled: bool,
    pub aip_policy: RamaAipPolicyKind,
    pub aip_topk: Option<usize>,
    pub aip_lm_head_rows: Option<usize>,
    pub aip_column_cache: bool,
    pub aip_input_tiles: bool,
}

impl RamaExperimentalSpeedConfig {
    pub fn from_env() -> Self {
        let aip_topk = parse_aip_topk(std::env::var(RLLM_AIP_TOPK_ENV).ok().as_deref())
            .or_else(|| parse_turbo_topk(std::env::var(RLLM_TURBO_TOPK_ENV).ok().as_deref()));

        Self {
            enabled: parse_experimental_speed_enabled(
                std::env::var(RLLM_EXPERIMENTAL_SPEED_ENV).ok().as_deref(),
            ),
            aip_policy: parse_aip_policy(std::env::var(RLLM_AIP_POLICY_ENV).ok().as_deref())
                .unwrap_or_default(),
            aip_topk,
            aip_lm_head_rows: parse_aip_lm_head_rows(
                std::env::var(RLLM_AIP_LM_HEAD_ROWS_ENV).ok().as_deref(),
            ),
            aip_column_cache: parse_aip_column_cache_enabled(
                std::env::var(RLLM_AIP_COLUMN_CACHE_ENV).ok().as_deref(),
            ),
            aip_input_tiles: parse_aip_input_tiles_enabled(
                std::env::var(RLLM_AIP_INPUT_TILES_ENV).ok().as_deref(),
            ),
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            aip_policy: RamaAipPolicyKind::Quality,
            aip_topk: None,
            aip_lm_head_rows: None,
            aip_column_cache: false,
            aip_input_tiles: false,
        }
    }

    pub fn topk_for_input(self, input_len: usize, default_topk: usize) -> usize {
        if input_len == 0 {
            return 0;
        }
        self.aip_topk
            .unwrap_or(default_topk.max(1))
            .min(input_len)
            .max(1)
    }

    pub fn aip_decision_for_projection(
        self,
        layer_index: usize,
        total_layers: usize,
        projection: RamaAipProjectionKind,
        input_len: usize,
        default_topk: usize,
    ) -> RamaAipProjectionDecision {
        if !self.enabled || input_len == 0 || layer_index >= total_layers {
            return RamaAipProjectionDecision::exact();
        }

        match self.aip_policy {
            RamaAipPolicyKind::Speed => {
                RamaAipProjectionDecision::aip(self.topk_for_input(input_len, default_topk))
            }
            RamaAipPolicyKind::Quality => {
                if projection != RamaAipProjectionKind::MlpGateUp
                    || !quality_policy_allows_layer(layer_index, total_layers)
                {
                    return RamaAipProjectionDecision::exact();
                }
                RamaAipProjectionDecision::aip(self.topk_for_input(input_len, default_topk))
            }
        }
    }

    pub fn lm_head_prefix_rows(self, vocab_size: usize) -> Option<usize> {
        if !self.enabled || vocab_size == 0 {
            return None;
        }
        self.aip_lm_head_rows
            .map(|rows| rows.min(vocab_size).max(1))
            .filter(|rows| *rows < vocab_size)
    }
}

fn quality_policy_allows_layer(layer_index: usize, total_layers: usize) -> bool {
    if total_layers < 4 || layer_index >= total_layers {
        return false;
    }
    let exact_edge_layers = total_layers / 4;
    layer_index >= exact_edge_layers && layer_index < total_layers.saturating_sub(exact_edge_layers)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaExperimentalSpeedStats {
    pub aip_policy: Option<RamaAipPolicyKind>,
    pub sparse_projection_calls: usize,
    pub exact_fallbacks: usize,
    pub selected_topk_sum: usize,
    pub max_selected_topk: usize,
    pub estimated_skipped_madds: usize,
    pub peak_scratch_bytes: usize,
    pub lm_head_prefix_rows: usize,
    pub lm_head_vocab_rows: usize,
    pub column_cache_hits: usize,
    pub column_cache_misses: usize,
    pub column_cache_resident_columns: usize,
    pub column_cache_resident_bytes: usize,
    pub input_tile_range_reads: usize,
    pub input_tile_range_bytes: usize,
}

impl RamaExperimentalSpeedStats {
    pub fn add_assign(&mut self, other: Self) {
        if self.aip_policy.is_none() {
            self.aip_policy = other.aip_policy;
        }
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
        if self.lm_head_prefix_rows == 0 {
            self.lm_head_prefix_rows = other.lm_head_prefix_rows;
            self.lm_head_vocab_rows = other.lm_head_vocab_rows;
        }
        self.column_cache_hits = self
            .column_cache_hits
            .saturating_add(other.column_cache_hits);
        self.column_cache_misses = self
            .column_cache_misses
            .saturating_add(other.column_cache_misses);
        self.column_cache_resident_columns = self
            .column_cache_resident_columns
            .max(other.column_cache_resident_columns);
        self.column_cache_resident_bytes = self
            .column_cache_resident_bytes
            .max(other.column_cache_resident_bytes);
        self.input_tile_range_reads = self
            .input_tile_range_reads
            .saturating_add(other.input_tile_range_reads);
        self.input_tile_range_bytes = self
            .input_tile_range_bytes
            .saturating_add(other.input_tile_range_bytes);
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

    pub fn record_aip_policy(&mut self, policy: RamaAipPolicyKind) {
        if self.aip_policy.is_none() {
            self.aip_policy = Some(policy);
        }
    }

    pub fn record_lm_head_prefix(&mut self, prefix_rows: usize, vocab_rows: usize) {
        if prefix_rows > 0 {
            self.lm_head_prefix_rows = prefix_rows;
            self.lm_head_vocab_rows = vocab_rows;
        }
    }

    pub fn record_column_cache(
        &mut self,
        hits: usize,
        misses: usize,
        resident_columns: usize,
        resident_bytes: usize,
    ) {
        self.column_cache_hits = self.column_cache_hits.saturating_add(hits);
        self.column_cache_misses = self.column_cache_misses.saturating_add(misses);
        self.column_cache_resident_columns =
            self.column_cache_resident_columns.max(resident_columns);
        self.column_cache_resident_bytes = self.column_cache_resident_bytes.max(resident_bytes);
    }

    pub fn record_input_tile_ranges(&mut self, reads: usize, bytes: usize) {
        self.input_tile_range_reads = self.input_tile_range_reads.saturating_add(reads);
        self.input_tile_range_bytes = self.input_tile_range_bytes.saturating_add(bytes);
    }

    pub fn is_empty(self) -> bool {
        self.aip_policy.is_none()
            && self.sparse_projection_calls == 0
            && self.exact_fallbacks == 0
            && self.selected_topk_sum == 0
            && self.max_selected_topk == 0
            && self.estimated_skipped_madds == 0
            && self.peak_scratch_bytes == 0
            && self.lm_head_prefix_rows == 0
            && self.lm_head_vocab_rows == 0
            && self.column_cache_hits == 0
            && self.column_cache_misses == 0
            && self.column_cache_resident_columns == 0
            && self.column_cache_resident_bytes == 0
            && self.input_tile_range_reads == 0
            && self.input_tile_range_bytes == 0
    }
}

pub fn parse_experimental_speed_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn parse_aip_policy(value: Option<&str>) -> Option<RamaAipPolicyKind> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("quality") => Some(RamaAipPolicyKind::Quality),
        Some("speed") => Some(RamaAipPolicyKind::Speed),
        _ => None,
    }
}

pub fn parse_aip_topk(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

pub fn parse_aip_lm_head_rows(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_column_cache_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn parse_aip_input_tiles_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn parse_turbo_topk(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
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
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: Some(512),
            aip_lm_head_rows: None,
            aip_column_cache: false,
            aip_input_tiles: false,
        };
        assert_eq!(config.topk_for_input(2048, 256), 512);
        assert_eq!(config.topk_for_input(128, 256), 128);

        let defaulted = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: None,
            aip_lm_head_rows: None,
            aip_column_cache: false,
            aip_input_tiles: false,
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

    #[test]
    fn parse_aip_policy_accepts_quality_and_speed() {
        assert_eq!(
            parse_aip_policy(Some("quality")),
            Some(RamaAipPolicyKind::Quality)
        );
        assert_eq!(
            parse_aip_policy(Some("speed")),
            Some(RamaAipPolicyKind::Speed)
        );
        assert_eq!(
            parse_aip_policy(Some(" QUALITY ")),
            Some(RamaAipPolicyKind::Quality)
        );
        assert_eq!(parse_aip_policy(Some("bad")), None);
        assert_eq!(parse_aip_policy(None), None);
    }

    #[test]
    fn parse_aip_topk_keeps_only_positive_values() {
        assert_eq!(parse_aip_topk(Some("128")), Some(128));
        assert_eq!(parse_aip_topk(Some("1")), Some(1));
        assert_eq!(parse_aip_topk(Some("0")), None);
        assert_eq!(parse_aip_topk(Some("-2")), None);
        assert_eq!(parse_aip_topk(Some("bad")), None);
        assert_eq!(parse_aip_topk(None), None);
    }

    #[test]
    fn parse_aip_lm_head_rows_keeps_only_positive_values() {
        assert_eq!(parse_aip_lm_head_rows(Some("4096")), Some(4096));
        assert_eq!(parse_aip_lm_head_rows(Some("1")), Some(1));
        assert_eq!(parse_aip_lm_head_rows(Some("0")), None);
        assert_eq!(parse_aip_lm_head_rows(Some("-2")), None);
        assert_eq!(parse_aip_lm_head_rows(Some("bad")), None);
        assert_eq!(parse_aip_lm_head_rows(None), None);
    }

    #[test]
    fn parse_aip_column_cache_enabled_accepts_explicit_truthy_values() {
        assert!(parse_aip_column_cache_enabled(Some("1")));
        assert!(parse_aip_column_cache_enabled(Some("true")));
        assert!(parse_aip_column_cache_enabled(Some("yes")));
        assert!(parse_aip_column_cache_enabled(Some("on")));
        assert!(!parse_aip_column_cache_enabled(Some("0")));
        assert!(!parse_aip_column_cache_enabled(Some("false")));
        assert!(!parse_aip_column_cache_enabled(Some("")));
        assert!(!parse_aip_column_cache_enabled(None));
    }

    #[test]
    fn parse_aip_input_tiles_enabled_accepts_explicit_truthy_values() {
        assert!(parse_aip_input_tiles_enabled(Some("1")));
        assert!(parse_aip_input_tiles_enabled(Some("true")));
        assert!(parse_aip_input_tiles_enabled(Some("yes")));
        assert!(parse_aip_input_tiles_enabled(Some("on")));
        assert!(!parse_aip_input_tiles_enabled(Some("0")));
        assert!(!parse_aip_input_tiles_enabled(Some("false")));
        assert!(!parse_aip_input_tiles_enabled(Some("")));
        assert!(!parse_aip_input_tiles_enabled(None));
    }

    #[test]
    fn lm_head_prefix_rows_are_bounded_and_only_when_enabled() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: None,
            aip_lm_head_rows: Some(512),
            aip_column_cache: false,
            aip_input_tiles: false,
        };
        assert_eq!(config.lm_head_prefix_rows(128_256), Some(512));
        assert_eq!(config.lm_head_prefix_rows(512), None);
        assert_eq!(config.lm_head_prefix_rows(256), None);

        let disabled = RamaExperimentalSpeedConfig {
            enabled: false,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: None,
            aip_lm_head_rows: Some(512),
            aip_column_cache: false,
            aip_input_tiles: false,
        };
        assert_eq!(disabled.lm_head_prefix_rows(128_256), None);
    }

    #[test]
    fn quality_policy_uses_only_middle_layer_gate_up() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Quality,
            aip_topk: Some(96),
            aip_lm_head_rows: None,
            aip_column_cache: false,
            aip_input_tiles: false,
        };

        assert_eq!(
            config.aip_decision_for_projection(0, 8, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        assert_eq!(
            config.aip_decision_for_projection(7, 8, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        assert_eq!(
            config.aip_decision_for_projection(3, 8, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(96)
        );
        assert_eq!(
            config.aip_decision_for_projection(3, 8, RamaAipProjectionKind::MlpDown, 8192, 512),
            RamaAipProjectionDecision::exact()
        );
    }

    #[test]
    fn quality_policy_stays_exact_for_tiny_layer_counts() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Quality,
            aip_topk: Some(64),
            aip_lm_head_rows: None,
            aip_column_cache: false,
            aip_input_tiles: false,
        };

        assert_eq!(
            config.aip_decision_for_projection(0, 1, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        assert_eq!(
            config.aip_decision_for_projection(1, 3, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
    }

    #[test]
    fn speed_policy_uses_aip_for_gate_up_and_down() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: Some(128),
            aip_lm_head_rows: None,
            aip_column_cache: false,
            aip_input_tiles: false,
        };

        assert_eq!(
            config.aip_decision_for_projection(0, 1, RamaAipProjectionKind::MlpGateUp, 2048, 256),
            RamaAipProjectionDecision::aip(128)
        );
        assert_eq!(
            config.aip_decision_for_projection(0, 1, RamaAipProjectionKind::MlpDown, 8192, 512),
            RamaAipProjectionDecision::aip(128)
        );
    }

    #[test]
    fn disabled_config_always_selects_exact() {
        let config = RamaExperimentalSpeedConfig::disabled();

        assert_eq!(
            config.aip_decision_for_projection(3, 8, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
    }

    #[test]
    fn stats_record_policy_without_losing_sparse_counts() {
        let mut stats = RamaExperimentalSpeedStats::default();
        stats.record_aip_policy(RamaAipPolicyKind::Quality);
        stats.record_sparse_projection(4, 16, 3, 2);

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_aip_policy(RamaAipPolicyKind::Speed);
        other.record_exact_fallback();
        stats.add_assign(other);

        assert_eq!(stats.aip_policy, Some(RamaAipPolicyKind::Quality));
        assert_eq!(stats.sparse_projection_calls, 1);
        assert_eq!(stats.exact_fallbacks, 1);
        assert!(!stats.is_empty());
    }
}
