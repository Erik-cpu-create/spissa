// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speed::{
        parse_aip_attention_locality_extra, parse_aip_attention_locality_window,
        parse_aip_column_cache_enabled, parse_aip_edge_layers, parse_aip_edge_topk,
        parse_aip_exact_edge_layers, parse_aip_exact_edge_projection, parse_aip_exact_layer,
        parse_aip_exact_layer_projection, parse_aip_input_tiles_enabled,
        parse_aip_layer_drift_probe_enabled, parse_aip_lm_head_agreement_enabled,
        parse_aip_lm_head_exact_every, parse_aip_lm_head_novelty_gap_milli,
        parse_aip_lm_head_novelty_repeat_penalty_milli, parse_aip_lm_head_novelty_retention_milli,
        parse_aip_lm_head_novelty_window, parse_aip_lm_head_repeat_margin_adaptive_enabled,
        parse_aip_lm_head_repeat_margin_milli, parse_aip_lm_head_rescore,
        parse_aip_lm_head_rescore_gap_milli, parse_aip_lm_head_rows, parse_aip_no_repeat_last_enabled,
        parse_aip_policy, parse_aip_repeat_run_limit, parse_aip_topk, parse_experimental_speed_enabled,
        parse_turbo_topk, select_top_abs_indices, select_top_abs_indices_with_recent,
        RamaAipPolicyKind, RamaAipProjectionDecision, RamaAipProjectionKind,
        RamaAttentionLocalityCache, RamaExperimentalSpeedConfig, RamaExperimentalSpeedStats,
    };

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
        assert_eq!(
            parse_aip_exact_edge_projection(Some("all")),
            Some(RamaAipProjectionKind::All)
        );
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
        assert_eq!(
            parse_aip_exact_layer_projection(Some("all")),
            Some(RamaAipProjectionKind::All)
        );
        assert_eq!(
            parse_aip_exact_layer_projection(Some("mlp")),
            Some(RamaAipProjectionKind::Mlp)
        );
        assert_eq!(
            parse_aip_exact_layer_projection(Some("attention-gateup")),
            Some(RamaAipProjectionKind::AttentionGateUp)
        );
        assert_eq!(
            parse_aip_exact_layer_projection(Some("attention-down")),
            Some(RamaAipProjectionKind::AttentionDown)
        );
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
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
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
    fn speed_policy_exact_layer_projection_applies_globally_when_no_layer_specified() {
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
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: Some(RamaAipProjectionKind::MlpDown),
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
            config.aip_decision_for_projection(3, 4, RamaAipProjectionKind::MlpDown, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        assert_eq!(
            config.aip_decision_for_projection(1, 4, RamaAipProjectionKind::Attention, 2048, 128),
            RamaAipProjectionDecision::aip(4)
        );
    }

    #[test]
    fn speed_policy_combination_exact_layer_projection_overrides_projections() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: RamaAipPolicyKind::Speed,
            aip_topk: Some(4),
            aip_layer_topk_overrides: [0; 128],
            aip_exact_layer_projection: Some(RamaAipProjectionKind::Mlp),
            ..RamaExperimentalSpeedConfig::disabled()
        };

        // For Mlp: Gate-Up and Down should be exact
        assert_eq!(
            config.aip_decision_for_projection(0, 4, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        assert_eq!(
            config.aip_decision_for_projection(0, 4, RamaAipProjectionKind::MlpDown, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        // Attention should be sparse (AIP)
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
            aip_down_topk: Some(6),
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: Some(1),
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
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
            RamaAipProjectionDecision::aip(6)
        );
    }

    #[test]
    fn speed_policy_exact_periodic_layers_override_sparse_projection() {
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
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: Some(4),
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
        };

        // Layer 0 is exact (0 % 4 == 0)
        assert_eq!(
            config.aip_decision_for_projection(0, 16, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        // Layer 1 is sparse
        assert_eq!(
            config.aip_decision_for_projection(1, 16, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(128)
        );
        // Layer 4 is exact (4 % 4 == 0)
        assert_eq!(
            config.aip_decision_for_projection(4, 16, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::exact()
        );
        // Layer 7 is sparse
        assert_eq!(
            config.aip_decision_for_projection(7, 16, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(128)
        );
    }

    #[test]
    fn parse_aip_layer_topk_overrides_parses_comma_colon_pairs() {
        let overrides = parse_aip_layer_topk_overrides(Some("0:16, 2 :8, 127:32"));
        assert_eq!(overrides[0], 16);
        assert_eq!(overrides[1], 0);
        assert_eq!(overrides[2], 8);
        assert_eq!(overrides[127], 32);

        let empty = parse_aip_layer_topk_overrides(None);
        assert_eq!(empty[0], 0);
    }

    #[test]
    fn speed_policy_layer_topk_overrides_takes_precedence() {
        let mut overrides = [0u16; 128];
        overrides[2] = 16; // layer 2 override to 16

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
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: overrides,
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

        // Layer 0 is not overridden, uses global topk=4
        assert_eq!(
            config.aip_decision_for_projection(0, 16, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(4)
        );
        // Layer 2 is overridden to 16
        assert_eq!(
            config.aip_decision_for_projection(2, 16, RamaAipProjectionKind::MlpGateUp, 2048, 128),
            RamaAipProjectionDecision::aip(16)
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
        stats.record_layer_drift_probe(4, 2, Some(2), 100, 5, 1_250, 15, 900);

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_layer_drift_probe(4, 1, Some(1), 0, 0, 1_000, 50, 1_200);
        stats.add_assign(other);

        assert_eq!(stats.layer_drift_probe.samples, 2);
        assert_eq!(stats.layer_drift_probe.layers, 8);
        assert_eq!(stats.layer_drift_probe.mismatch_layers, 3);
        assert_eq!(stats.layer_drift_probe.first_mismatch_layer, 1);
        assert_eq!(stats.layer_drift_probe.pre_mismatch_max_l2_milli, 100);
        assert_eq!(stats.layer_drift_probe.pre_mismatch_max_cosine_gap_milli, 5);
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
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
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
