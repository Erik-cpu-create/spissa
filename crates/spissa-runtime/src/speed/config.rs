// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

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
pub const RLLM_AIP_EXACT_PREFIX_LAYERS_ENV: &str = "RLLM_AIP_EXACT_PREFIX_LAYERS";
pub const RLLM_AIP_EXACT_PERIODIC_LAYERS_ENV: &str = "RLLM_AIP_EXACT_PERIODIC_LAYERS";
pub const RLLM_AIP_LAYER_TOPK_OVERRIDES_ENV: &str = "RLLM_AIP_LAYER_TOPK_OVERRIDES";
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
    Mlp,             // Gate-Up and Down
    AttentionGateUp, // Attention and Gate-Up
    AttentionDown,   // Attention and Down
    All,             // All projections exact
}

impl RamaAipProjectionKind {
    pub fn matches(self, other: Self) -> bool {
        match (self, other) {
            (Self::Attention, Self::Attention) => true,
            (Self::MlpGateUp, Self::MlpGateUp) => true,
            (Self::MlpDown, Self::MlpDown) => true,
            (Self::Mlp, Self::MlpGateUp | Self::MlpDown) => true,
            (Self::AttentionGateUp, Self::Attention | Self::MlpGateUp) => true,
            (Self::AttentionDown, Self::Attention | Self::MlpDown) => true,
            (Self::All, _) => true,
            _ => self == other,
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub aip_exact_prefix_layers: Option<usize>,
    pub aip_exact_periodic_layers: Option<usize>,
    pub aip_layer_topk_overrides: [u16; 128],
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

impl Default for RamaExperimentalSpeedConfig {
    fn default() -> Self {
        Self::disabled()
    }
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
            aip_exact_prefix_layers: parse_aip_exact_prefix_layers(
                std::env::var(RLLM_AIP_EXACT_PREFIX_LAYERS_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_exact_periodic_layers: parse_aip_exact_periodic_layers(
                std::env::var(RLLM_AIP_EXACT_PERIODIC_LAYERS_ENV)
                    .ok()
                    .as_deref(),
            ),
            aip_layer_topk_overrides: parse_aip_layer_topk_overrides(
                std::env::var(RLLM_AIP_LAYER_TOPK_OVERRIDES_ENV)
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
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
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

pub fn parse_aip_exact_prefix_layers(value: Option<&str>) -> Option<usize> {
    value.and_then(|v| v.parse().ok()).filter(|v| *v > 0)
}

pub fn parse_aip_lm_head_rows(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_topk(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
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

pub fn parse_aip_layer_topk_overrides(value: Option<&str>) -> [u16; 128] {
    let mut overrides = [0u16; 128];
    let Some(val) = value else {
        return overrides;
    };
    for pair in val.split(',') {
        let mut parts = pair.split(':');
        if let (Some(layer_str), Some(topk_str)) = (parts.next(), parts.next()) {
            if let (Ok(layer), Ok(topk)) = (
                layer_str.trim().parse::<usize>(),
                topk_str.trim().parse::<u16>(),
            ) {
                if layer < 128 {
                    overrides[layer] = topk;
                }
            }
        }
    }
    overrides
}

pub fn parse_aip_exact_edge_layers(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_exact_periodic_layers(value: Option<&str>) -> Option<usize> {
    parse_aip_topk(value)
}

pub fn parse_aip_exact_edge_projection(value: Option<&str>) -> Option<RamaAipProjectionKind> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("attention" | "attn") => Some(RamaAipProjectionKind::Attention),
        Some("mlp-gate-up" | "mlp_gate_up" | "gate-up" | "gateup") => {
            Some(RamaAipProjectionKind::MlpGateUp)
        }
        Some("mlp-down" | "mlp_down" | "down") => Some(RamaAipProjectionKind::MlpDown),
        Some("mlp" | "mlps") => Some(RamaAipProjectionKind::Mlp),
        Some(
            "attention-gate-up" | "attention_gate_up" | "attention-gateup" | "attention_gateup"
            | "attn-gateup",
        ) => Some(RamaAipProjectionKind::AttentionGateUp),
        Some("attention-down" | "attention_down" | "attn-down") => {
            Some(RamaAipProjectionKind::AttentionDown)
        }
        Some("all") => Some(RamaAipProjectionKind::All),
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
