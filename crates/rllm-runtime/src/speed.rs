use std::cmp::Ordering;

pub const RLLM_EXPERIMENTAL_SPEED_ENV: &str = "RLLM_EXPERIMENTAL_SPEED";
pub const RLLM_AIP_POLICY_ENV: &str = "RLLM_AIP_POLICY";
pub const RLLM_AIP_TOPK_ENV: &str = "RLLM_AIP_TOPK";
pub const RLLM_AIP_ATTENTION_TOPK_ENV: &str = "RLLM_AIP_ATTENTION_TOPK";
pub const RLLM_AIP_ATTENTION_LOCALITY_WINDOW_ENV: &str = "RLLM_AIP_ATTENTION_LOCALITY_WINDOW";
pub const RLLM_AIP_ATTENTION_LOCALITY_EXTRA_ENV: &str = "RLLM_AIP_ATTENTION_LOCALITY_EXTRA";
pub const RLLM_AIP_MLP_TOPK_ENV: &str = "RLLM_AIP_MLP_TOPK";
pub const RLLM_AIP_DOWN_TOPK_ENV: &str = "RLLM_AIP_DOWN_TOPK";
pub const RLLM_AIP_EDGE_LAYERS_ENV: &str = "RLLM_AIP_EDGE_LAYERS";
pub const RLLM_AIP_EDGE_TOPK_ENV: &str = "RLLM_AIP_EDGE_TOPK";
pub const RLLM_AIP_EXACT_EDGE_LAYERS_ENV: &str = "RLLM_AIP_EXACT_EDGE_LAYERS";
pub const RLLM_AIP_EXACT_EDGE_PROJECTION_ENV: &str = "RLLM_AIP_EXACT_EDGE_PROJECTION";
pub const RLLM_AIP_EXACT_LAYER_ENV: &str = "RLLM_AIP_EXACT_LAYER";
pub const RLLM_AIP_EXACT_LAYER_PROJECTION_ENV: &str = "RLLM_AIP_EXACT_LAYER_PROJECTION";
pub const RLLM_AIP_LM_HEAD_TOPK_ENV: &str = "RLLM_AIP_LM_HEAD_TOPK";
pub const RLLM_AIP_LM_HEAD_RESCORE_ENV: &str = "RLLM_AIP_LM_HEAD_RESCORE";
pub const RLLM_AIP_LM_HEAD_RESCORE_GAP_MILLI_ENV: &str = "RLLM_AIP_LM_HEAD_RESCORE_GAP_MILLI";
pub const RLLM_AIP_LM_HEAD_AGREEMENT_ENV: &str = "RLLM_AIP_LM_HEAD_AGREEMENT";
pub const RLLM_AIP_LM_HEAD_EXACT_EVERY_ENV: &str = "RLLM_AIP_LM_HEAD_EXACT_EVERY";
pub const RLLM_AIP_LAYER_DRIFT_PROBE_ENV: &str = "RLLM_AIP_LAYER_DRIFT_PROBE";
pub const RLLM_AIP_LM_HEAD_ROWS_ENV: &str = "RLLM_AIP_LM_HEAD_ROWS";
pub const RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI_ENV: &str = "RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI";
pub const RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE_ENV: &str =
    "RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE";
pub const RLLM_AIP_LM_HEAD_NOVELTY_WINDOW_ENV: &str = "RLLM_AIP_LM_HEAD_NOVELTY_WINDOW";
pub const RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI_ENV: &str = "RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI";
pub const RLLM_AIP_LM_HEAD_NOVELTY_REPEAT_PENALTY_MILLI_ENV: &str =
    "RLLM_AIP_LM_HEAD_NOVELTY_REPEAT_PENALTY_MILLI";
pub const RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI_ENV: &str =
    "RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI";
pub const RLLM_AIP_COLUMN_CACHE_ENV: &str = "RLLM_AIP_COLUMN_CACHE";
pub const RLLM_AIP_INPUT_TILES_ENV: &str = "RLLM_AIP_INPUT_TILES";
pub const RLLM_AIP_EXACT_PREFILL_ENV: &str = "RLLM_AIP_EXACT_PREFILL";
pub const RLLM_AIP_NO_REPEAT_LAST_ENV: &str = "RLLM_AIP_NO_REPEAT_LAST";
pub const RLLM_AIP_REPEAT_RUN_LIMIT_ENV: &str = "RLLM_AIP_REPEAT_RUN_LIMIT";
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
    Attention,
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
    pub aip_attention_topk: Option<usize>,
    pub aip_attention_locality_window: Option<usize>,
    pub aip_attention_locality_extra: Option<usize>,
    pub aip_mlp_topk: Option<usize>,
    pub aip_down_topk: Option<usize>,
    pub aip_edge_layers: Option<usize>,
    pub aip_edge_topk: Option<usize>,
    pub aip_exact_edge_layers: Option<usize>,
    pub aip_exact_edge_projection: Option<RamaAipProjectionKind>,
    pub aip_exact_layer: Option<usize>,
    pub aip_exact_layer_projection: Option<RamaAipProjectionKind>,
    pub aip_lm_head_topk: Option<usize>,
    pub aip_lm_head_rescore: Option<usize>,
    pub aip_lm_head_rescore_gap_milli: Option<usize>,
    pub aip_lm_head_agreement: bool,
    pub aip_lm_head_rows: Option<usize>,
    pub aip_lm_head_repeat_margin_milli: Option<usize>,
    pub aip_lm_head_repeat_margin_adaptive: bool,
    pub aip_lm_head_novelty_window: Option<usize>,
    pub aip_lm_head_novelty_gap_milli: Option<usize>,
    pub aip_lm_head_novelty_repeat_penalty_milli: Option<usize>,
    pub aip_lm_head_novelty_retention_milli: Option<usize>,
    pub aip_column_cache: bool,
    pub aip_input_tiles: bool,
    pub aip_no_repeat_last: bool,
    pub aip_repeat_run_limit: Option<usize>,
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
            aip_attention_topk: parse_aip_topk(
                std::env::var(RLLM_AIP_ATTENTION_TOPK_ENV).ok().as_deref(),
            ),
            aip_attention_locality_window: parse_aip_attention_locality_window(
                std::env::var(RLLM_AIP_ATTENTION_LOCALITY_WINDOW_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_attention_locality_extra: parse_aip_attention_locality_extra(
                std::env::var(RLLM_AIP_ATTENTION_LOCALITY_EXTRA_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_mlp_topk: parse_aip_topk(std::env::var(RLLM_AIP_MLP_TOPK_ENV).ok().as_deref()),
            aip_down_topk: parse_aip_topk(std::env::var(RLLM_AIP_DOWN_TOPK_ENV).ok().as_deref()),
            aip_edge_layers: parse_aip_edge_layers(
                std::env::var(RLLM_AIP_EDGE_LAYERS_ENV).ok().as_deref(),
            ),
            aip_edge_topk: parse_aip_edge_topk(
                std::env::var(RLLM_AIP_EDGE_TOPK_ENV).ok().as_deref(),
            ),
            aip_exact_edge_layers: parse_aip_exact_edge_layers(
                std::env::var(RLLM_AIP_EXACT_EDGE_LAYERS_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_exact_edge_projection: parse_aip_exact_edge_projection(
                std::env::var(RLLM_AIP_EXACT_EDGE_PROJECTION_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_exact_layer: parse_aip_exact_layer(
                std::env::var(RLLM_AIP_EXACT_LAYER_ENV).ok().as_deref(),
            ),
            aip_exact_layer_projection: parse_aip_exact_layer_projection(
                std::env::var(RLLM_AIP_EXACT_LAYER_PROJECTION_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_lm_head_topk: parse_aip_topk(
                std::env::var(RLLM_AIP_LM_HEAD_TOPK_ENV).ok().as_deref(),
            ),
            aip_lm_head_rescore: parse_aip_lm_head_rescore(
                std::env::var(RLLM_AIP_LM_HEAD_RESCORE_ENV).ok().as_deref(),
            ),
            aip_lm_head_rescore_gap_milli: parse_aip_lm_head_rescore_gap_milli(
                std::env::var(RLLM_AIP_LM_HEAD_RESCORE_GAP_MILLI_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_lm_head_agreement: parse_aip_lm_head_agreement_enabled(
                std::env::var(RLLM_AIP_LM_HEAD_AGREEMENT_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_lm_head_rows: parse_aip_lm_head_rows(
                std::env::var(RLLM_AIP_LM_HEAD_ROWS_ENV).ok().as_deref(),
            ),
            aip_lm_head_repeat_margin_milli: parse_aip_lm_head_repeat_margin_milli(
                std::env::var(RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_lm_head_repeat_margin_adaptive: parse_aip_lm_head_repeat_margin_adaptive_enabled(
                std::env::var(RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_lm_head_novelty_window: parse_aip_lm_head_novelty_window(
                std::env::var(RLLM_AIP_LM_HEAD_NOVELTY_WINDOW_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_lm_head_novelty_gap_milli: parse_aip_lm_head_novelty_gap_milli(
                std::env::var(RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_lm_head_novelty_repeat_penalty_milli:
                parse_aip_lm_head_novelty_repeat_penalty_milli(
                    std::env::var(RLLM_AIP_LM_HEAD_NOVELTY_REPEAT_PENALTY_MILLI_ENV)
                        .ok()
                        .as_deref(),
                ),
            aip_lm_head_novelty_retention_milli: parse_aip_lm_head_novelty_retention_milli(
                std::env::var(RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_column_cache: parse_aip_column_cache_enabled(
                std::env::var(RLLM_AIP_COLUMN_CACHE_ENV).ok().as_deref(),
            ),
            aip_input_tiles: parse_aip_input_tiles_enabled(
                std::env::var(RLLM_AIP_INPUT_TILES_ENV).ok().as_deref(),
            ),
            aip_no_repeat_last: parse_aip_no_repeat_last_enabled(
                std::env::var(RLLM_AIP_NO_REPEAT_LAST_ENV).ok().as_deref(),
            ),
            aip_repeat_run_limit: parse_aip_repeat_run_limit(
                std::env::var(RLLM_AIP_REPEAT_RUN_LIMIT_ENV).ok().as_deref(),
            ),
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            aip_policy: RamaAipPolicyKind::Quality,
            aip_topk: None,
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        }
    }

    pub fn topk_for_input(self, input_len: usize, default_topk: usize) -> usize {
        self.topk_for_input_with_override(input_len, default_topk, None)
    }

    fn topk_for_input_with_override(
        self,
        input_len: usize,
        default_topk: usize,
        topk_override: Option<usize>,
    ) -> usize {
        if input_len == 0 {
            return 0;
        }
        topk_override
            .or(self.aip_topk)
            .unwrap_or(default_topk.max(1))
            .min(input_len)
            .max(1)
    }

    pub fn topk_for_projection(
        self,
        projection: RamaAipProjectionKind,
        input_len: usize,
        default_topk: usize,
    ) -> usize {
        let topk_override = match projection {
            RamaAipProjectionKind::Attention => self.aip_attention_topk,
            RamaAipProjectionKind::MlpGateUp => self.aip_mlp_topk,
            RamaAipProjectionKind::MlpDown => self.aip_down_topk,
        };
        self.topk_for_input_with_override(input_len, default_topk, topk_override)
    }

    fn edge_topk_for_layer(
        self,
        layer_index: usize,
        total_layers: usize,
        input_len: usize,
    ) -> Option<usize> {
        if input_len == 0 || layer_index >= total_layers {
            return None;
        }
        let edge_topk = self.aip_edge_topk?;
        let edge_layers = self.aip_edge_layers.unwrap_or(1).min(total_layers);
        if layer_index < edge_layers || layer_index >= total_layers.saturating_sub(edge_layers) {
            Some(edge_topk.min(input_len).max(1))
        } else {
            None
        }
    }

    pub fn attention_locality_enabled_for_layer(
        self,
        layer_index: usize,
        total_layers: usize,
    ) -> bool {
        if !self.enabled
            || self.aip_attention_locality_window.is_none()
            || self.aip_attention_locality_extra.is_none()
            || layer_index >= total_layers
        {
            return false;
        }
        let edge_layers = self.aip_edge_layers.unwrap_or(1).min(total_layers);
        layer_index < edge_layers || layer_index >= total_layers.saturating_sub(edge_layers)
    }

    fn exact_edge_projection(
        self,
        layer_index: usize,
        total_layers: usize,
        projection: RamaAipProjectionKind,
    ) -> bool {
        if layer_index >= total_layers {
            return false;
        }
        let Some(edge_layers) = self.aip_exact_edge_layers else {
            return false;
        };
        let edge_layers = edge_layers.min(total_layers);
        if layer_index >= edge_layers && layer_index < total_layers.saturating_sub(edge_layers) {
            return false;
        }
        self.aip_exact_edge_projection
            .map(|exact_projection| exact_projection == projection)
            .unwrap_or(true)
    }

    fn exact_layer_projection(
        self,
        layer_index: usize,
        total_layers: usize,
        projection: RamaAipProjectionKind,
    ) -> bool {
        if layer_index >= total_layers {
            return false;
        }
        let Some(exact_layer) = self.aip_exact_layer else {
            return false;
        };
        if exact_layer == 0 || layer_index + 1 != exact_layer {
            return false;
        }
        self.aip_exact_layer_projection
            .map(|exact_projection| exact_projection == projection)
            .unwrap_or(true)
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
        if self.exact_layer_projection(layer_index, total_layers, projection) {
            return RamaAipProjectionDecision::exact();
        }
        if self.exact_edge_projection(layer_index, total_layers, projection) {
            return RamaAipProjectionDecision::exact();
        }

        match self.aip_policy {
            RamaAipPolicyKind::Speed => {
                let topk = self
                    .edge_topk_for_layer(layer_index, total_layers, input_len)
                    .unwrap_or_else(|| {
                        self.topk_for_projection(projection, input_len, default_topk)
                    });
                RamaAipProjectionDecision::aip(topk)
            }
            RamaAipPolicyKind::Quality => {
                if projection != RamaAipProjectionKind::MlpGateUp
                    || !quality_policy_allows_layer(layer_index, total_layers)
                {
                    return RamaAipProjectionDecision::exact();
                }
                RamaAipProjectionDecision::aip(self.topk_for_projection(
                    projection,
                    input_len,
                    default_topk,
                ))
            }
        }
    }

    pub fn lm_head_topk_for_input(self, input_len: usize, default_topk: usize) -> usize {
        self.topk_for_input_with_override(input_len, default_topk, self.aip_lm_head_topk)
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
pub struct RamaLayerDriftProbeStats {
    pub samples: usize,
    pub layers: usize,
    pub mismatch_layers: usize,
    pub first_mismatch_layer: usize,
    pub max_l2_milli: usize,
    pub max_cosine_gap_milli: usize,
    pub max_exact_margin_milli: usize,
}

impl RamaLayerDriftProbeStats {
    pub fn add_assign(&mut self, other: Self) {
        self.samples = self.samples.saturating_add(other.samples);
        self.layers = self.layers.saturating_add(other.layers);
        self.mismatch_layers = self.mismatch_layers.saturating_add(other.mismatch_layers);
        self.first_mismatch_layer =
            min_non_zero(self.first_mismatch_layer, other.first_mismatch_layer);
        self.max_l2_milli = self.max_l2_milli.max(other.max_l2_milli);
        self.max_cosine_gap_milli = self.max_cosine_gap_milli.max(other.max_cosine_gap_milli);
        self.max_exact_margin_milli = self
            .max_exact_margin_milli
            .max(other.max_exact_margin_milli);
    }

    pub fn record(
        &mut self,
        layers: usize,
        mismatch_layers: usize,
        first_mismatch_layer: Option<usize>,
        max_l2_milli: usize,
        max_cosine_gap_milli: usize,
        max_exact_margin_milli: usize,
    ) {
        if layers == 0 {
            return;
        }
        self.samples = self.samples.saturating_add(1);
        self.layers = self.layers.saturating_add(layers);
        self.mismatch_layers = self.mismatch_layers.saturating_add(mismatch_layers);
        if let Some(layer) = first_mismatch_layer.filter(|layer| *layer > 0) {
            self.first_mismatch_layer = min_non_zero(self.first_mismatch_layer, layer);
        }
        self.max_l2_milli = self.max_l2_milli.max(max_l2_milli);
        self.max_cosine_gap_milli = self.max_cosine_gap_milli.max(max_cosine_gap_milli);
        self.max_exact_margin_milli = self.max_exact_margin_milli.max(max_exact_margin_milli);
    }

    pub fn is_empty(self) -> bool {
        self.samples == 0
            && self.layers == 0
            && self.mismatch_layers == 0
            && self.first_mismatch_layer == 0
            && self.max_l2_milli == 0
            && self.max_cosine_gap_milli == 0
            && self.max_exact_margin_milli == 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaLayerAttributionProbeStats {
    pub samples: usize,
    pub layer: usize,
    pub attention_l2_milli: usize,
    pub attention_cosine_gap_milli: usize,
    pub gate_up_l2_milli: usize,
    pub gate_up_cosine_gap_milli: usize,
    pub down_l2_milli: usize,
    pub down_cosine_gap_milli: usize,
}

impl RamaLayerAttributionProbeStats {
    pub fn add_assign(&mut self, other: Self) {
        self.samples = self.samples.saturating_add(other.samples);
        self.layer = min_non_zero(self.layer, other.layer);
        self.attention_l2_milli = self.attention_l2_milli.max(other.attention_l2_milli);
        self.attention_cosine_gap_milli = self
            .attention_cosine_gap_milli
            .max(other.attention_cosine_gap_milli);
        self.gate_up_l2_milli = self.gate_up_l2_milli.max(other.gate_up_l2_milli);
        self.gate_up_cosine_gap_milli = self
            .gate_up_cosine_gap_milli
            .max(other.gate_up_cosine_gap_milli);
        self.down_l2_milli = self.down_l2_milli.max(other.down_l2_milli);
        self.down_cosine_gap_milli = self.down_cosine_gap_milli.max(other.down_cosine_gap_milli);
    }

    pub fn record(
        &mut self,
        layer: usize,
        attention_l2_milli: usize,
        attention_cosine_gap_milli: usize,
        gate_up_l2_milli: usize,
        gate_up_cosine_gap_milli: usize,
        down_l2_milli: usize,
        down_cosine_gap_milli: usize,
    ) {
        if layer == 0 {
            return;
        }
        self.samples = self.samples.saturating_add(1);
        self.layer = min_non_zero(self.layer, layer);
        self.attention_l2_milli = self.attention_l2_milli.max(attention_l2_milli);
        self.attention_cosine_gap_milli = self
            .attention_cosine_gap_milli
            .max(attention_cosine_gap_milli);
        self.gate_up_l2_milli = self.gate_up_l2_milli.max(gate_up_l2_milli);
        self.gate_up_cosine_gap_milli = self.gate_up_cosine_gap_milli.max(gate_up_cosine_gap_milli);
        self.down_l2_milli = self.down_l2_milli.max(down_l2_milli);
        self.down_cosine_gap_milli = self.down_cosine_gap_milli.max(down_cosine_gap_milli);
    }

    pub fn is_empty(self) -> bool {
        self.samples == 0
            && self.layer == 0
            && self.attention_l2_milli == 0
            && self.attention_cosine_gap_milli == 0
            && self.gate_up_l2_milli == 0
            && self.gate_up_cosine_gap_milli == 0
            && self.down_l2_milli == 0
            && self.down_cosine_gap_milli == 0
    }
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
    pub attention_locality_uses: usize,
    pub attention_locality_added_indices: usize,
    pub attention_locality_max_selected: usize,
    pub lm_head_prefix_rows: usize,
    pub lm_head_vocab_rows: usize,
    pub lm_head_rescore_checks: usize,
    pub lm_head_rescore_uses: usize,
    pub lm_head_rescore_gap_skips: usize,
    pub lm_head_rescore_max_gap_milli: usize,
    pub column_cache_hits: usize,
    pub column_cache_misses: usize,
    pub column_cache_resident_columns: usize,
    pub column_cache_resident_bytes: usize,
    pub input_tile_range_reads: usize,
    pub input_tile_range_bytes: usize,
    pub lm_head_agreement_samples: usize,
    pub lm_head_agreement_sparse_argmax_matches: usize,
    pub lm_head_agreement_selected_matches: usize,
    pub lm_head_agreement_exact_in_sparse_topk: usize,
    pub lm_head_agreement_max_topk: usize,
    pub lm_head_exact_checks: usize,
    pub lm_head_exact_switches: usize,
    pub lm_head_repeat_margin_checks: usize,
    pub lm_head_repeat_margin_switches: usize,
    pub lm_head_repeat_margin_max_gap_milli: usize,
    pub lm_head_repeat_margin_adaptive_throttles: usize,
    pub lm_head_repeat_margin_min_effective_milli: usize,
    pub lm_head_phrase_novelty_checks: usize,
    pub lm_head_phrase_novelty_switches: usize,
    pub lm_head_phrase_novelty_max_ngram: usize,
    pub lm_head_phrase_novelty_gap_skips: usize,
    pub lm_head_phrase_novelty_max_gap_milli: usize,
    pub lm_head_phrase_novelty_soft_choices: usize,
    pub lm_head_phrase_novelty_retentions: usize,
    pub layer_drift_probe: RamaLayerDriftProbeStats,
    pub layer_attribution_probe: RamaLayerAttributionProbeStats,
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
        self.attention_locality_uses = self
            .attention_locality_uses
            .saturating_add(other.attention_locality_uses);
        self.attention_locality_added_indices = self
            .attention_locality_added_indices
            .saturating_add(other.attention_locality_added_indices);
        self.attention_locality_max_selected = self
            .attention_locality_max_selected
            .max(other.attention_locality_max_selected);
        if self.lm_head_prefix_rows == 0 {
            self.lm_head_prefix_rows = other.lm_head_prefix_rows;
            self.lm_head_vocab_rows = other.lm_head_vocab_rows;
        }
        self.lm_head_rescore_checks = self
            .lm_head_rescore_checks
            .saturating_add(other.lm_head_rescore_checks);
        self.lm_head_rescore_uses = self
            .lm_head_rescore_uses
            .saturating_add(other.lm_head_rescore_uses);
        self.lm_head_rescore_gap_skips = self
            .lm_head_rescore_gap_skips
            .saturating_add(other.lm_head_rescore_gap_skips);
        self.lm_head_rescore_max_gap_milli = self
            .lm_head_rescore_max_gap_milli
            .max(other.lm_head_rescore_max_gap_milli);
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
        self.lm_head_agreement_samples = self
            .lm_head_agreement_samples
            .saturating_add(other.lm_head_agreement_samples);
        self.lm_head_agreement_sparse_argmax_matches = self
            .lm_head_agreement_sparse_argmax_matches
            .saturating_add(other.lm_head_agreement_sparse_argmax_matches);
        self.lm_head_agreement_selected_matches = self
            .lm_head_agreement_selected_matches
            .saturating_add(other.lm_head_agreement_selected_matches);
        self.lm_head_agreement_exact_in_sparse_topk = self
            .lm_head_agreement_exact_in_sparse_topk
            .saturating_add(other.lm_head_agreement_exact_in_sparse_topk);
        self.lm_head_agreement_max_topk = self
            .lm_head_agreement_max_topk
            .max(other.lm_head_agreement_max_topk);
        self.lm_head_exact_checks = self
            .lm_head_exact_checks
            .saturating_add(other.lm_head_exact_checks);
        self.lm_head_exact_switches = self
            .lm_head_exact_switches
            .saturating_add(other.lm_head_exact_switches);
        self.lm_head_repeat_margin_checks = self
            .lm_head_repeat_margin_checks
            .saturating_add(other.lm_head_repeat_margin_checks);
        self.lm_head_repeat_margin_switches = self
            .lm_head_repeat_margin_switches
            .saturating_add(other.lm_head_repeat_margin_switches);
        self.lm_head_repeat_margin_max_gap_milli = self
            .lm_head_repeat_margin_max_gap_milli
            .max(other.lm_head_repeat_margin_max_gap_milli);
        self.lm_head_repeat_margin_adaptive_throttles = self
            .lm_head_repeat_margin_adaptive_throttles
            .saturating_add(other.lm_head_repeat_margin_adaptive_throttles);
        self.lm_head_repeat_margin_min_effective_milli = min_non_zero(
            self.lm_head_repeat_margin_min_effective_milli,
            other.lm_head_repeat_margin_min_effective_milli,
        );
        self.lm_head_phrase_novelty_checks = self
            .lm_head_phrase_novelty_checks
            .saturating_add(other.lm_head_phrase_novelty_checks);
        self.lm_head_phrase_novelty_switches = self
            .lm_head_phrase_novelty_switches
            .saturating_add(other.lm_head_phrase_novelty_switches);
        self.lm_head_phrase_novelty_max_ngram = self
            .lm_head_phrase_novelty_max_ngram
            .max(other.lm_head_phrase_novelty_max_ngram);
        self.lm_head_phrase_novelty_gap_skips = self
            .lm_head_phrase_novelty_gap_skips
            .saturating_add(other.lm_head_phrase_novelty_gap_skips);
        self.lm_head_phrase_novelty_max_gap_milli = self
            .lm_head_phrase_novelty_max_gap_milli
            .max(other.lm_head_phrase_novelty_max_gap_milli);
        self.lm_head_phrase_novelty_soft_choices = self
            .lm_head_phrase_novelty_soft_choices
            .saturating_add(other.lm_head_phrase_novelty_soft_choices);
        self.lm_head_phrase_novelty_retentions = self
            .lm_head_phrase_novelty_retentions
            .saturating_add(other.lm_head_phrase_novelty_retentions);
        self.layer_drift_probe.add_assign(other.layer_drift_probe);
        self.layer_attribution_probe
            .add_assign(other.layer_attribution_probe);
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

    pub fn record_lm_head_rescore(&mut self, used: bool, gap_milli: usize) {
        self.lm_head_rescore_checks = self.lm_head_rescore_checks.saturating_add(1);
        if used {
            self.lm_head_rescore_uses = self.lm_head_rescore_uses.saturating_add(1);
        } else {
            self.lm_head_rescore_gap_skips = self.lm_head_rescore_gap_skips.saturating_add(1);
        }
        self.lm_head_rescore_max_gap_milli = self.lm_head_rescore_max_gap_milli.max(gap_milli);
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

    pub fn record_attention_locality(&mut self, selected_topk: usize, base_topk: usize) {
        self.attention_locality_uses = self.attention_locality_uses.saturating_add(1);
        self.attention_locality_added_indices = self
            .attention_locality_added_indices
            .saturating_add(selected_topk.saturating_sub(base_topk));
        self.attention_locality_max_selected =
            self.attention_locality_max_selected.max(selected_topk);
    }

    pub fn record_lm_head_agreement(
        &mut self,
        sparse_argmax_matches_exact: bool,
        selected_matches_exact: bool,
        exact_in_sparse_topk: bool,
        sparse_topk: usize,
    ) {
        self.lm_head_agreement_samples = self.lm_head_agreement_samples.saturating_add(1);
        if sparse_argmax_matches_exact {
            self.lm_head_agreement_sparse_argmax_matches = self
                .lm_head_agreement_sparse_argmax_matches
                .saturating_add(1);
        }
        if selected_matches_exact {
            self.lm_head_agreement_selected_matches =
                self.lm_head_agreement_selected_matches.saturating_add(1);
        }
        if exact_in_sparse_topk {
            self.lm_head_agreement_exact_in_sparse_topk = self
                .lm_head_agreement_exact_in_sparse_topk
                .saturating_add(1);
        }
        self.lm_head_agreement_max_topk = self.lm_head_agreement_max_topk.max(sparse_topk);
    }

    pub fn record_lm_head_exact_check(&mut self, switched: bool) {
        self.lm_head_exact_checks = self.lm_head_exact_checks.saturating_add(1);
        if switched {
            self.lm_head_exact_switches = self.lm_head_exact_switches.saturating_add(1);
        }
    }

    pub fn record_lm_head_repeat_margin(&mut self, switched: bool, gap_milli: usize) {
        self.lm_head_repeat_margin_checks = self.lm_head_repeat_margin_checks.saturating_add(1);
        if switched {
            self.lm_head_repeat_margin_switches =
                self.lm_head_repeat_margin_switches.saturating_add(1);
        }
        self.lm_head_repeat_margin_max_gap_milli =
            self.lm_head_repeat_margin_max_gap_milli.max(gap_milli);
    }

    pub fn record_lm_head_repeat_margin_adaptive_throttle(
        &mut self,
        effective_margin_milli: usize,
    ) {
        self.lm_head_repeat_margin_adaptive_throttles = self
            .lm_head_repeat_margin_adaptive_throttles
            .saturating_add(1);
        self.lm_head_repeat_margin_min_effective_milli = min_non_zero(
            self.lm_head_repeat_margin_min_effective_milli,
            effective_margin_milli,
        );
    }

    pub fn record_lm_head_phrase_novelty(&mut self, switched: bool, ngram_len: usize) {
        self.lm_head_phrase_novelty_checks = self.lm_head_phrase_novelty_checks.saturating_add(1);
        if switched {
            self.lm_head_phrase_novelty_switches =
                self.lm_head_phrase_novelty_switches.saturating_add(1);
        }
        self.lm_head_phrase_novelty_max_ngram =
            self.lm_head_phrase_novelty_max_ngram.max(ngram_len);
    }

    pub fn record_lm_head_phrase_novelty_gap_skip(&mut self, gap_milli: usize) {
        self.lm_head_phrase_novelty_gap_skips =
            self.lm_head_phrase_novelty_gap_skips.saturating_add(1);
        self.lm_head_phrase_novelty_max_gap_milli =
            self.lm_head_phrase_novelty_max_gap_milli.max(gap_milli);
    }

    pub fn record_lm_head_phrase_novelty_soft_choice(&mut self) {
        self.lm_head_phrase_novelty_soft_choices =
            self.lm_head_phrase_novelty_soft_choices.saturating_add(1);
    }

    pub fn record_lm_head_phrase_novelty_retention(&mut self) {
        self.lm_head_phrase_novelty_retentions =
            self.lm_head_phrase_novelty_retentions.saturating_add(1);
    }

    pub fn record_layer_drift_probe(
        &mut self,
        layers: usize,
        mismatch_layers: usize,
        first_mismatch_layer: Option<usize>,
        max_l2_milli: usize,
        max_cosine_gap_milli: usize,
        max_exact_margin_milli: usize,
    ) {
        self.layer_drift_probe.record(
            layers,
            mismatch_layers,
            first_mismatch_layer,
            max_l2_milli,
            max_cosine_gap_milli,
            max_exact_margin_milli,
        );
    }

    pub fn record_layer_attribution_probe(
        &mut self,
        layer: usize,
        attention_l2_milli: usize,
        attention_cosine_gap_milli: usize,
        gate_up_l2_milli: usize,
        gate_up_cosine_gap_milli: usize,
        down_l2_milli: usize,
        down_cosine_gap_milli: usize,
    ) {
        self.layer_attribution_probe.record(
            layer,
            attention_l2_milli,
            attention_cosine_gap_milli,
            gate_up_l2_milli,
            gate_up_cosine_gap_milli,
            down_l2_milli,
            down_cosine_gap_milli,
        );
    }

    pub fn is_empty(self) -> bool {
        self.aip_policy.is_none()
            && self.sparse_projection_calls == 0
            && self.exact_fallbacks == 0
            && self.selected_topk_sum == 0
            && self.max_selected_topk == 0
            && self.estimated_skipped_madds == 0
            && self.peak_scratch_bytes == 0
            && self.attention_locality_uses == 0
            && self.attention_locality_added_indices == 0
            && self.attention_locality_max_selected == 0
            && self.lm_head_prefix_rows == 0
            && self.lm_head_vocab_rows == 0
            && self.lm_head_rescore_checks == 0
            && self.lm_head_rescore_uses == 0
            && self.lm_head_rescore_gap_skips == 0
            && self.lm_head_rescore_max_gap_milli == 0
            && self.column_cache_hits == 0
            && self.column_cache_misses == 0
            && self.column_cache_resident_columns == 0
            && self.column_cache_resident_bytes == 0
            && self.input_tile_range_reads == 0
            && self.input_tile_range_bytes == 0
            && self.lm_head_agreement_samples == 0
            && self.lm_head_agreement_sparse_argmax_matches == 0
            && self.lm_head_agreement_selected_matches == 0
            && self.lm_head_agreement_exact_in_sparse_topk == 0
            && self.lm_head_agreement_max_topk == 0
            && self.lm_head_exact_checks == 0
            && self.lm_head_exact_switches == 0
            && self.lm_head_repeat_margin_checks == 0
            && self.lm_head_repeat_margin_switches == 0
            && self.lm_head_repeat_margin_max_gap_milli == 0
            && self.lm_head_repeat_margin_adaptive_throttles == 0
            && self.lm_head_repeat_margin_min_effective_milli == 0
            && self.lm_head_phrase_novelty_checks == 0
            && self.lm_head_phrase_novelty_switches == 0
            && self.lm_head_phrase_novelty_max_ngram == 0
            && self.lm_head_phrase_novelty_gap_skips == 0
            && self.lm_head_phrase_novelty_max_gap_milli == 0
            && self.lm_head_phrase_novelty_soft_choices == 0
            && self.lm_head_phrase_novelty_retentions == 0
            && self.layer_drift_probe.is_empty()
            && self.layer_attribution_probe.is_empty()
    }
}

fn min_non_zero(lhs: usize, rhs: usize) -> usize {
    match (lhs, rhs) {
        (0, value) => value,
        (value, 0) => value,
        (left, right) => left.min(right),
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

pub fn parse_aip_attention_locality_window(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_attention_locality_extra(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_lm_head_repeat_margin_milli(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_lm_head_novelty_window(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_lm_head_novelty_gap_milli(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_lm_head_novelty_repeat_penalty_milli(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_lm_head_novelty_retention_milli(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_lm_head_repeat_margin_adaptive_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn parse_aip_lm_head_rescore(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_lm_head_rescore_gap_milli(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_lm_head_exact_every(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_lm_head_agreement_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn parse_aip_layer_drift_probe_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
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

pub fn parse_aip_exact_prefill_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn parse_aip_no_repeat_last_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn parse_aip_repeat_run_limit(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_edge_layers(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_edge_topk(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_exact_edge_layers(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_exact_edge_projection(value: Option<&str>) -> Option<RamaAipProjectionKind> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("attention" | "attn") => Some(RamaAipProjectionKind::Attention),
        Some("mlp-gate-up" | "mlp_gate_up" | "gate-up" | "gateup") => {
            Some(RamaAipProjectionKind::MlpGateUp)
        }
        Some("mlp-down" | "mlp_down" | "down") => Some(RamaAipProjectionKind::MlpDown),
        _ => None,
    }
}

pub fn parse_aip_exact_layer(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_exact_layer_projection(value: Option<&str>) -> Option<RamaAipProjectionKind> {
    parse_aip_exact_edge_projection(value)
}

pub fn parse_turbo_topk(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn select_top_abs_indices(input: &[f32], topk: usize) -> Vec<usize> {
    let limit = topk.min(input.len());
    if limit == 0 {
        return Vec::new();
    }
    if limit <= 16 {
        return select_top_abs_indices_small(input, limit);
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

pub fn select_top_abs_indices_with_recent(
    input: &[f32],
    topk: usize,
    recent: &[usize],
    extra: usize,
) -> Vec<usize> {
    let mut selected = select_top_abs_indices(input, topk);
    let mut added = 0usize;
    for &index in recent {
        if index >= input.len() || selected.contains(&index) {
            continue;
        }
        selected.push(index);
        added = added.saturating_add(1);
        if added >= extra {
            break;
        }
    }
    selected.sort_unstable();
    selected
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RamaAttentionLocalityCache {
    recent: Vec<usize>,
}

impl RamaAttentionLocalityCache {
    pub fn recent(&self) -> &[usize] {
        &self.recent
    }

    pub fn clear(&mut self) {
        self.recent.clear();
    }

    pub fn record(&mut self, selected: &[usize], window: usize) {
        if window == 0 {
            self.recent.clear();
            return;
        }
        let mut next = Vec::with_capacity(window.min(selected.len() + self.recent.len()));
        for &index in selected {
            push_unique_bounded(&mut next, index, window);
        }
        for &index in &self.recent {
            push_unique_bounded(&mut next, index, window);
        }
        self.recent = next;
    }
}

fn push_unique_bounded(values: &mut Vec<usize>, value: usize, limit: usize) {
    if values.len() < limit && !values.contains(&value) {
        values.push(value);
    }
}

fn select_top_abs_indices_small(input: &[f32], limit: usize) -> Vec<usize> {
    let mut winners: Vec<(usize, f32)> = Vec::with_capacity(limit);
    for (idx, value) in input.iter().enumerate() {
        let candidate = (idx, value.abs());
        if winners.len() < limit {
            winners.push(candidate);
            continue;
        }

        let Some((worst_idx, worst)) = winners
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| compare_top_abs_candidates(left, right))
        else {
            continue;
        };
        if compare_top_abs_candidates(&candidate, worst) == Ordering::Less {
            winners[worst_idx] = candidate;
        }
    }
    winners.sort_unstable_by_key(|(idx, _)| *idx);
    winners.into_iter().map(|(idx, _)| idx).collect()
}

fn compare_top_abs_candidates(left: &(usize, f32), right: &(usize, f32)) -> Ordering {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
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
    fn small_top_abs_selector_matches_full_sort_ranking() {
        let input: [f32; 18] = [
            0.1, -9.0, 4.0, -2.0, 7.0, -7.0, 0.0, 3.5, -4.0, 1.0, 6.0, -8.0, 8.0, -0.5, 2.5, -6.0,
            5.0, -3.0,
        ];

        let mut scored: Vec<(usize, f32)> = input
            .iter()
            .enumerate()
            .map(|(idx, value)| (idx, (*value).abs()))
            .collect();
        scored.sort_by(compare_top_abs_candidates);
        let mut expected: Vec<usize> = scored.into_iter().take(8).map(|(idx, _)| idx).collect();
        expected.sort_unstable();

        assert_eq!(select_top_abs_indices(&input, 8), expected);
    }

    #[test]
    fn attention_locality_selection_adds_recent_indices_with_bounds() {
        let input = [0.1, -9.0, 8.0, 0.2, 0.3, 0.4];
        let selected = select_top_abs_indices_with_recent(&input, 2, &[4, 1, 99, 5], 2);

        assert_eq!(selected, vec![1, 2, 4, 5]);
    }

    #[test]
    fn attention_locality_cache_keeps_recent_unique_indices() {
        let mut cache = RamaAttentionLocalityCache::default();

        cache.record(&[1, 2, 3], 4);
        cache.record(&[2, 5], 4);

        assert_eq!(cache.recent(), &[2, 5, 1, 3]);
        cache.clear();
        assert!(cache.recent().is_empty());
    }

    #[test]
    fn config_chooses_bounded_topk() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: Some(512),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };
        assert_eq!(config.topk_for_input(2048, 256), 512);
        assert_eq!(config.topk_for_input(128, 256), 128);

        let defaulted = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: None,
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
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
    fn parse_aip_exact_prefill_enabled_accepts_explicit_truthy_values() {
        assert!(parse_aip_exact_prefill_enabled(Some("1")));
        assert!(parse_aip_exact_prefill_enabled(Some("true")));
        assert!(parse_aip_exact_prefill_enabled(Some("yes")));
        assert!(parse_aip_exact_prefill_enabled(Some("on")));
        assert!(!parse_aip_exact_prefill_enabled(Some("0")));
        assert!(!parse_aip_exact_prefill_enabled(Some("false")));
        assert!(!parse_aip_exact_prefill_enabled(Some("")));
        assert!(!parse_aip_exact_prefill_enabled(None));
    }

    #[test]
    fn parse_aip_no_repeat_last_enabled_accepts_explicit_truthy_values() {
        assert!(parse_aip_no_repeat_last_enabled(Some("1")));
        assert!(parse_aip_no_repeat_last_enabled(Some("true")));
        assert!(parse_aip_no_repeat_last_enabled(Some("yes")));
        assert!(parse_aip_no_repeat_last_enabled(Some("on")));
        assert!(!parse_aip_no_repeat_last_enabled(Some("0")));
        assert!(!parse_aip_no_repeat_last_enabled(Some("false")));
        assert!(!parse_aip_no_repeat_last_enabled(Some("")));
        assert!(!parse_aip_no_repeat_last_enabled(None));
    }

    #[test]
    fn parse_aip_repeat_run_limit_keeps_only_positive_values() {
        assert_eq!(parse_aip_repeat_run_limit(Some("2")), Some(2));
        assert_eq!(parse_aip_repeat_run_limit(Some("1")), Some(1));
        assert_eq!(parse_aip_repeat_run_limit(Some("0")), None);
        assert_eq!(parse_aip_repeat_run_limit(Some("-2")), None);
        assert_eq!(parse_aip_repeat_run_limit(Some("bad")), None);
        assert_eq!(parse_aip_repeat_run_limit(None), None);
    }

    #[test]
    fn parse_aip_edge_controls_keep_only_positive_values() {
        assert_eq!(parse_aip_edge_layers(Some("2")), Some(2));
        assert_eq!(parse_aip_edge_layers(Some("1")), Some(1));
        assert_eq!(parse_aip_edge_layers(Some("0")), None);
        assert_eq!(parse_aip_edge_layers(Some("-2")), None);
        assert_eq!(parse_aip_edge_layers(Some("bad")), None);
        assert_eq!(parse_aip_edge_layers(None), None);

        assert_eq!(parse_aip_edge_topk(Some("8")), Some(8));
        assert_eq!(parse_aip_edge_topk(Some("1")), Some(1));
        assert_eq!(parse_aip_edge_topk(Some("0")), None);
        assert_eq!(parse_aip_edge_topk(Some("-2")), None);
        assert_eq!(parse_aip_edge_topk(Some("bad")), None);
        assert_eq!(parse_aip_edge_topk(None), None);
    }

    #[test]
    fn parse_aip_exact_edge_layers_keeps_only_positive_values() {
        assert_eq!(parse_aip_exact_edge_layers(Some("2")), Some(2));
        assert_eq!(parse_aip_exact_edge_layers(Some("1")), Some(1));
        assert_eq!(parse_aip_exact_edge_layers(Some("0")), None);
        assert_eq!(parse_aip_exact_edge_layers(Some("-2")), None);
        assert_eq!(parse_aip_exact_edge_layers(Some("bad")), None);
        assert_eq!(parse_aip_exact_edge_layers(None), None);
    }

    #[test]
    fn parse_aip_attention_locality_controls_keep_only_positive_values() {
        assert_eq!(parse_aip_attention_locality_window(Some("16")), Some(16));
        assert_eq!(parse_aip_attention_locality_window(Some("0")), None);
        assert_eq!(parse_aip_attention_locality_window(Some("bad")), None);
        assert_eq!(parse_aip_attention_locality_extra(Some("4")), Some(4));
        assert_eq!(parse_aip_attention_locality_extra(Some("-1")), None);
        assert_eq!(parse_aip_attention_locality_extra(None), None);
    }

    #[test]
    fn parse_aip_exact_edge_projection_accepts_known_projection_names() {
        assert_eq!(parse_aip_exact_edge_projection(Some("all")), None);
        assert_eq!(
            parse_aip_exact_edge_projection(Some("attention")),
            Some(RamaAipProjectionKind::Attention)
        );
        assert_eq!(
            parse_aip_exact_edge_projection(Some("attn")),
            Some(RamaAipProjectionKind::Attention)
        );
        assert_eq!(
            parse_aip_exact_edge_projection(Some("mlp-gate-up")),
            Some(RamaAipProjectionKind::MlpGateUp)
        );
        assert_eq!(
            parse_aip_exact_edge_projection(Some("gateup")),
            Some(RamaAipProjectionKind::MlpGateUp)
        );
        assert_eq!(
            parse_aip_exact_edge_projection(Some("mlp-down")),
            Some(RamaAipProjectionKind::MlpDown)
        );
        assert_eq!(
            parse_aip_exact_edge_projection(Some("down")),
            Some(RamaAipProjectionKind::MlpDown)
        );
        assert_eq!(parse_aip_exact_edge_projection(Some("bad")), None);
        assert_eq!(parse_aip_exact_edge_projection(None), None);
    }

    #[test]
    fn parse_aip_exact_layer_controls_keep_only_positive_values() {
        assert_eq!(parse_aip_exact_layer(Some("2")), Some(2));
        assert_eq!(parse_aip_exact_layer(Some("1")), Some(1));
        assert_eq!(parse_aip_exact_layer(Some("0")), None);
        assert_eq!(parse_aip_exact_layer(Some("-2")), None);
        assert_eq!(parse_aip_exact_layer(Some("bad")), None);
        assert_eq!(parse_aip_exact_layer(None), None);
    }

    #[test]
    fn parse_aip_exact_layer_projection_accepts_known_projection_names() {
        assert_eq!(parse_aip_exact_layer_projection(Some("all")), None);
        assert_eq!(
            parse_aip_exact_layer_projection(Some("attention")),
            Some(RamaAipProjectionKind::Attention)
        );
        assert_eq!(
            parse_aip_exact_layer_projection(Some("gateup")),
            Some(RamaAipProjectionKind::MlpGateUp)
        );
        assert_eq!(
            parse_aip_exact_layer_projection(Some("down")),
            Some(RamaAipProjectionKind::MlpDown)
        );
        assert_eq!(parse_aip_exact_layer_projection(Some("bad")), None);
        assert_eq!(parse_aip_exact_layer_projection(None), None);
    }

    #[test]
    fn speed_policy_exact_layer_projection_overrides_only_target_layer_projection() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: Some(4),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: Some(2),
            aip_exact_layer_projection: Some(RamaAipProjectionKind::Attention),
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };

        assert_eq!(
            config.aip_decision_for_projection(1, 4, RamaAipProjectionKind::Attention, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        assert_eq!(
            config.aip_decision_for_projection(1, 4, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(4)
        );
        assert_eq!(
            config.aip_decision_for_projection(0, 4, RamaAipProjectionKind::Attention, 2048, 128),
            RamaAipProjectionDecision::aip(4)
        );
    }

    #[test]
    fn speed_policy_exact_edge_layers_override_sparse_projection() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: Some(128),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: Some(1),
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };

        assert_eq!(
            config.aip_decision_for_projection(0, 4, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        assert_eq!(
            config.aip_decision_for_projection(3, 4, RamaAipProjectionKind::MlpDown, 8192, 128),
            RamaAipProjectionDecision::exact()
        );
        assert_eq!(
            config.aip_decision_for_projection(1, 4, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(128)
        );
    }

    #[test]
    fn speed_policy_exact_edge_projection_filter_only_overrides_matching_projection() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: Some(4),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: Some(1),
            aip_exact_edge_projection: Some(RamaAipProjectionKind::MlpDown),
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };

        assert_eq!(
            config.aip_decision_for_projection(0, 4, RamaAipProjectionKind::MlpDown, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        assert_eq!(
            config.aip_decision_for_projection(0, 4, RamaAipProjectionKind::Attention, 2048, 128),
            RamaAipProjectionDecision::aip(4)
        );
        assert_eq!(
            config.aip_decision_for_projection(1, 4, RamaAipProjectionKind::MlpDown, 2048, 128),
            RamaAipProjectionDecision::aip(4)
        );
    }

    #[test]
    fn parse_aip_lm_head_repeat_margin_milli_keeps_only_positive_values() {
        assert_eq!(
            parse_aip_lm_head_repeat_margin_milli(Some("250")),
            Some(250)
        );
        assert_eq!(parse_aip_lm_head_repeat_margin_milli(Some("1")), Some(1));
        assert_eq!(parse_aip_lm_head_repeat_margin_milli(Some("0")), None);
        assert_eq!(parse_aip_lm_head_repeat_margin_milli(Some("-2")), None);
        assert_eq!(parse_aip_lm_head_repeat_margin_milli(Some("bad")), None);
        assert_eq!(parse_aip_lm_head_repeat_margin_milli(None), None);
    }

    #[test]
    fn parse_aip_lm_head_novelty_window_keeps_only_positive_values() {
        assert_eq!(parse_aip_lm_head_novelty_window(Some("16")), Some(16));
        assert_eq!(parse_aip_lm_head_novelty_window(Some("1")), Some(1));
        assert_eq!(parse_aip_lm_head_novelty_window(Some("0")), None);
        assert_eq!(parse_aip_lm_head_novelty_window(Some("-2")), None);
        assert_eq!(parse_aip_lm_head_novelty_window(Some("bad")), None);
        assert_eq!(parse_aip_lm_head_novelty_window(None), None);
    }

    #[test]
    fn parse_aip_lm_head_novelty_gap_milli_keeps_only_positive_values() {
        assert_eq!(parse_aip_lm_head_novelty_gap_milli(Some("250")), Some(250));
        assert_eq!(parse_aip_lm_head_novelty_gap_milli(Some("1")), Some(1));
        assert_eq!(parse_aip_lm_head_novelty_gap_milli(Some("0")), None);
        assert_eq!(parse_aip_lm_head_novelty_gap_milli(Some("-2")), None);
        assert_eq!(parse_aip_lm_head_novelty_gap_milli(Some("bad")), None);
        assert_eq!(parse_aip_lm_head_novelty_gap_milli(None), None);
    }

    #[test]
    fn parse_aip_lm_head_novelty_repeat_penalty_milli_keeps_only_positive_values() {
        assert_eq!(
            parse_aip_lm_head_novelty_repeat_penalty_milli(Some("150")),
            Some(150)
        );
        assert_eq!(
            parse_aip_lm_head_novelty_repeat_penalty_milli(Some("1")),
            Some(1)
        );
        assert_eq!(
            parse_aip_lm_head_novelty_repeat_penalty_milli(Some("0")),
            None
        );
        assert_eq!(
            parse_aip_lm_head_novelty_repeat_penalty_milli(Some("-2")),
            None
        );
        assert_eq!(
            parse_aip_lm_head_novelty_repeat_penalty_milli(Some("bad")),
            None
        );
        assert_eq!(parse_aip_lm_head_novelty_repeat_penalty_milli(None), None);
    }

    #[test]
    fn parse_aip_lm_head_novelty_retention_milli_keeps_only_positive_values() {
        assert_eq!(
            parse_aip_lm_head_novelty_retention_milli(Some("100")),
            Some(100)
        );
        assert_eq!(
            parse_aip_lm_head_novelty_retention_milli(Some("1")),
            Some(1)
        );
        assert_eq!(parse_aip_lm_head_novelty_retention_milli(Some("0")), None);
        assert_eq!(parse_aip_lm_head_novelty_retention_milli(Some("-2")), None);
        assert_eq!(parse_aip_lm_head_novelty_retention_milli(Some("bad")), None);
        assert_eq!(parse_aip_lm_head_novelty_retention_milli(None), None);
    }

    #[test]
    fn parse_aip_lm_head_repeat_margin_adaptive_enabled_accepts_explicit_truthy_values() {
        assert!(parse_aip_lm_head_repeat_margin_adaptive_enabled(Some("1")));
        assert!(parse_aip_lm_head_repeat_margin_adaptive_enabled(Some(
            "true"
        )));
        assert!(parse_aip_lm_head_repeat_margin_adaptive_enabled(Some(
            "yes"
        )));
        assert!(parse_aip_lm_head_repeat_margin_adaptive_enabled(Some("on")));
        assert!(!parse_aip_lm_head_repeat_margin_adaptive_enabled(Some("0")));
        assert!(!parse_aip_lm_head_repeat_margin_adaptive_enabled(Some(
            "false"
        )));
        assert!(!parse_aip_lm_head_repeat_margin_adaptive_enabled(Some("")));
        assert!(!parse_aip_lm_head_repeat_margin_adaptive_enabled(None));
    }

    #[test]
    fn parse_aip_lm_head_rescore_keeps_only_positive_values() {
        assert_eq!(parse_aip_lm_head_rescore(Some("8")), Some(8));
        assert_eq!(parse_aip_lm_head_rescore(Some("1")), Some(1));
        assert_eq!(parse_aip_lm_head_rescore(Some("0")), None);
        assert_eq!(parse_aip_lm_head_rescore(Some("-2")), None);
        assert_eq!(parse_aip_lm_head_rescore(Some("bad")), None);
        assert_eq!(parse_aip_lm_head_rescore(None), None);
    }

    #[test]
    fn parse_aip_lm_head_rescore_gap_milli_keeps_only_positive_values() {
        assert_eq!(parse_aip_lm_head_rescore_gap_milli(Some("250")), Some(250));
        assert_eq!(parse_aip_lm_head_rescore_gap_milli(Some("1")), Some(1));
        assert_eq!(parse_aip_lm_head_rescore_gap_milli(Some("0")), None);
        assert_eq!(parse_aip_lm_head_rescore_gap_milli(Some("-2")), None);
        assert_eq!(parse_aip_lm_head_rescore_gap_milli(Some("bad")), None);
        assert_eq!(parse_aip_lm_head_rescore_gap_milli(None), None);
    }

    #[test]
    fn parse_aip_lm_head_agreement_enabled_accepts_explicit_truthy_values() {
        assert!(parse_aip_lm_head_agreement_enabled(Some("1")));
        assert!(parse_aip_lm_head_agreement_enabled(Some("true")));
        assert!(parse_aip_lm_head_agreement_enabled(Some("yes")));
        assert!(parse_aip_lm_head_agreement_enabled(Some("on")));
        assert!(!parse_aip_lm_head_agreement_enabled(Some("0")));
        assert!(!parse_aip_lm_head_agreement_enabled(Some("false")));
        assert!(!parse_aip_lm_head_agreement_enabled(Some("")));
        assert!(!parse_aip_lm_head_agreement_enabled(None));
    }

    #[test]
    fn parse_aip_layer_drift_probe_enabled_accepts_explicit_truthy_values() {
        assert!(parse_aip_layer_drift_probe_enabled(Some("1")));
        assert!(parse_aip_layer_drift_probe_enabled(Some("true")));
        assert!(parse_aip_layer_drift_probe_enabled(Some("yes")));
        assert!(parse_aip_layer_drift_probe_enabled(Some("on")));
        assert!(!parse_aip_layer_drift_probe_enabled(Some("0")));
        assert!(!parse_aip_layer_drift_probe_enabled(Some("false")));
        assert!(!parse_aip_layer_drift_probe_enabled(Some("")));
        assert!(!parse_aip_layer_drift_probe_enabled(None));
    }

    #[test]
    fn parse_aip_lm_head_exact_every_keeps_only_positive_values() {
        assert_eq!(parse_aip_lm_head_exact_every(Some("4")), Some(4));
        assert_eq!(parse_aip_lm_head_exact_every(Some("1")), Some(1));
        assert_eq!(parse_aip_lm_head_exact_every(Some("0")), None);
        assert_eq!(parse_aip_lm_head_exact_every(Some("-2")), None);
        assert_eq!(parse_aip_lm_head_exact_every(Some("bad")), None);
        assert_eq!(parse_aip_lm_head_exact_every(None), None);
    }

    #[test]
    fn stats_record_lm_head_agreement_and_merge() {
        let mut stats = RamaExperimentalSpeedStats::default();
        stats.record_lm_head_agreement(true, false, true, 4);
        stats.record_lm_head_agreement(false, true, false, 2);

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_lm_head_agreement(true, true, true, 8);
        stats.add_assign(other);

        assert_eq!(stats.lm_head_agreement_samples, 3);
        assert_eq!(stats.lm_head_agreement_sparse_argmax_matches, 2);
        assert_eq!(stats.lm_head_agreement_selected_matches, 2);
        assert_eq!(stats.lm_head_agreement_exact_in_sparse_topk, 2);
        assert_eq!(stats.lm_head_agreement_max_topk, 8);
        assert!(!stats.is_empty());
    }

    #[test]
    fn stats_record_layer_drift_probe_and_merge() {
        let mut stats = RamaExperimentalSpeedStats::default();
        stats.record_layer_drift_probe(4, 2, Some(2), 1_250, 15, 900);

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_layer_drift_probe(4, 1, Some(1), 1_000, 50, 1_200);
        stats.add_assign(other);

        assert_eq!(stats.layer_drift_probe.samples, 2);
        assert_eq!(stats.layer_drift_probe.layers, 8);
        assert_eq!(stats.layer_drift_probe.mismatch_layers, 3);
        assert_eq!(stats.layer_drift_probe.first_mismatch_layer, 1);
        assert_eq!(stats.layer_drift_probe.max_l2_milli, 1_250);
        assert_eq!(stats.layer_drift_probe.max_cosine_gap_milli, 50);
        assert_eq!(stats.layer_drift_probe.max_exact_margin_milli, 1_200);
        assert!(!stats.is_empty());
    }

    #[test]
    fn stats_record_layer_attribution_probe_and_merge() {
        let mut stats = RamaExperimentalSpeedStats::default();
        stats.record_layer_attribution_probe(2, 100, 10, 200, 20, 300, 30);

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_layer_attribution_probe(3, 150, 15, 180, 25, 400, 5);
        stats.add_assign(other);

        assert_eq!(stats.layer_attribution_probe.samples, 2);
        assert_eq!(stats.layer_attribution_probe.layer, 2);
        assert_eq!(stats.layer_attribution_probe.attention_l2_milli, 150);
        assert_eq!(stats.layer_attribution_probe.attention_cosine_gap_milli, 15);
        assert_eq!(stats.layer_attribution_probe.gate_up_l2_milli, 200);
        assert_eq!(stats.layer_attribution_probe.gate_up_cosine_gap_milli, 25);
        assert_eq!(stats.layer_attribution_probe.down_l2_milli, 400);
        assert_eq!(stats.layer_attribution_probe.down_cosine_gap_milli, 30);
        assert!(!stats.is_empty());
    }

    #[test]
    fn stats_record_attention_locality_and_merge() {
        let mut stats = RamaExperimentalSpeedStats::default();
        stats.record_attention_locality(8, 4);

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_attention_locality(6, 4);
        stats.add_assign(other);

        assert_eq!(stats.attention_locality_uses, 2);
        assert_eq!(stats.attention_locality_added_indices, 6);
        assert_eq!(stats.attention_locality_max_selected, 8);
        assert!(!stats.is_empty());
    }

    #[test]
    fn stats_record_lm_head_exact_checks_and_switches() {
        let mut stats = RamaExperimentalSpeedStats::default();
        stats.record_lm_head_exact_check(false);
        stats.record_lm_head_exact_check(true);

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_lm_head_exact_check(true);
        stats.add_assign(other);

        assert_eq!(stats.lm_head_exact_checks, 3);
        assert_eq!(stats.lm_head_exact_switches, 2);
        assert!(!stats.is_empty());
    }

    #[test]
    fn stats_record_lm_head_rescore_checks_uses_and_skips() {
        let mut stats = RamaExperimentalSpeedStats::default();
        stats.record_lm_head_rescore(false, 900);
        stats.record_lm_head_rescore(true, 125);

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_lm_head_rescore(true, 250);
        stats.add_assign(other);

        assert_eq!(stats.lm_head_rescore_checks, 3);
        assert_eq!(stats.lm_head_rescore_uses, 2);
        assert_eq!(stats.lm_head_rescore_gap_skips, 1);
        assert_eq!(stats.lm_head_rescore_max_gap_milli, 900);
        assert!(!stats.is_empty());
    }

    #[test]
    fn stats_record_lm_head_repeat_margin_and_merge() {
        let mut stats = RamaExperimentalSpeedStats::default();
        stats.record_lm_head_repeat_margin(false, 25);
        stats.record_lm_head_repeat_margin(true, 125);
        stats.record_lm_head_repeat_margin_adaptive_throttle(125);
        stats.record_lm_head_phrase_novelty(false, 2);
        stats.record_lm_head_phrase_novelty_gap_skip(500);
        stats.record_lm_head_phrase_novelty_soft_choice();
        stats.record_lm_head_phrase_novelty_retention();

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_lm_head_repeat_margin(true, 75);
        other.record_lm_head_repeat_margin_adaptive_throttle(250);
        other.record_lm_head_phrase_novelty(true, 3);
        other.record_lm_head_phrase_novelty_gap_skip(250);
        other.record_lm_head_phrase_novelty_soft_choice();
        other.record_lm_head_phrase_novelty_retention();
        stats.add_assign(other);

        assert_eq!(stats.lm_head_repeat_margin_checks, 3);
        assert_eq!(stats.lm_head_repeat_margin_switches, 2);
        assert_eq!(stats.lm_head_repeat_margin_max_gap_milli, 125);
        assert_eq!(stats.lm_head_repeat_margin_adaptive_throttles, 2);
        assert_eq!(stats.lm_head_repeat_margin_min_effective_milli, 125);
        assert_eq!(stats.lm_head_phrase_novelty_checks, 2);
        assert_eq!(stats.lm_head_phrase_novelty_switches, 1);
        assert_eq!(stats.lm_head_phrase_novelty_max_ngram, 3);
        assert_eq!(stats.lm_head_phrase_novelty_gap_skips, 2);
        assert_eq!(stats.lm_head_phrase_novelty_max_gap_milli, 500);
        assert_eq!(stats.lm_head_phrase_novelty_soft_choices, 2);
        assert_eq!(stats.lm_head_phrase_novelty_retentions, 2);
        assert!(!stats.is_empty());
    }

    #[test]
    fn lm_head_prefix_rows_are_bounded_and_only_when_enabled() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: None,
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: Some(512),
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };
        assert_eq!(config.lm_head_prefix_rows(128_256), Some(512));
        assert_eq!(config.lm_head_prefix_rows(512), None);
        assert_eq!(config.lm_head_prefix_rows(256), None);

        let disabled = RamaExperimentalSpeedConfig {
            enabled: false,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: None,
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: Some(512),
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };
        assert_eq!(disabled.lm_head_prefix_rows(128_256), None);
    }

    #[test]
    fn quality_policy_uses_only_middle_layer_gate_up() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Quality,
            aip_topk: Some(96),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
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
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
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
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
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
    fn projection_specific_topk_overrides_global_topk() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: Some(4),
            aip_attention_topk: Some(8),
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: Some(2),
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: Some(16),
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };

        assert_eq!(
            config.aip_decision_for_projection(0, 1, RamaAipProjectionKind::Attention, 2048, 128),
            RamaAipProjectionDecision::aip(8)
        );
        assert_eq!(
            config.aip_decision_for_projection(0, 1, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(4)
        );
        assert_eq!(
            config.aip_decision_for_projection(0, 1, RamaAipProjectionKind::MlpDown, 8192, 512),
            RamaAipProjectionDecision::aip(2)
        );
        assert_eq!(config.lm_head_topk_for_input(2048, 128), 16);
    }

    #[test]
    fn speed_policy_can_raise_topk_on_edge_layers() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: Some(4),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: Some(6),
            aip_edge_layers: Some(1),
            aip_edge_topk: Some(8),
            aip_exact_edge_layers: None,
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };

        assert_eq!(
            config.aip_decision_for_projection(0, 4, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(8)
        );
        assert_eq!(
            config.aip_decision_for_projection(1, 4, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(4)
        );
        assert_eq!(
            config.aip_decision_for_projection(2, 4, RamaAipProjectionKind::MlpDown, 8192, 512),
            RamaAipProjectionDecision::aip(6)
        );
        assert_eq!(
            config.aip_decision_for_projection(3, 4, RamaAipProjectionKind::MlpDown, 8192, 512),
            RamaAipProjectionDecision::aip(8)
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
