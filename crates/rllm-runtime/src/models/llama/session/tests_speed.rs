// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

    #[test]
    fn sparse_lm_head_rescore_candidates_respects_confidence_gap() {
        let mut config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: Some(2),
            aip_lm_head_rescore_gap_milli: Some(250),
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };
        let mut stats = RamaExperimentalSpeedStats::default();
        assert_eq!(
            sparse_lm_head_rescore_candidates(&[0.0, 5.0, 4.8, 1.0], &[], config, &mut stats)
                .unwrap(),
            Some(vec![1, 2])
        );
        assert_eq!(stats.lm_head_rescore_checks, 1);
        assert_eq!(stats.lm_head_rescore_uses, 1);
        assert_eq!(stats.lm_head_rescore_gap_skips, 0);
        assert_eq!(stats.lm_head_rescore_max_gap_milli, 200);

        config.aip_lm_head_rescore_gap_milli = Some(100);
        let mut stats = RamaExperimentalSpeedStats::default();
        assert_eq!(
            sparse_lm_head_rescore_candidates(&[0.0, 5.0, 4.8, 1.0], &[], config, &mut stats)
                .unwrap(),
            None
        );
        assert_eq!(stats.lm_head_rescore_checks, 1);
        assert_eq!(stats.lm_head_rescore_uses, 0);
        assert_eq!(stats.lm_head_rescore_gap_skips, 1);
        assert_eq!(stats.lm_head_rescore_max_gap_milli, 200);
    }

    #[test]
    fn rescored_lm_head_token_still_respects_repeat_run_limit() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: Some(2),
            aip_lm_head_rescore_gap_milli: Some(250),
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };
        let mut stats = RamaExperimentalSpeedStats::default();
        let mut repeat_state = LmHeadRepeatMarginState::default();
        let mut novelty_state = LmHeadPhraseNoveltyState::default();

        assert_eq!(
            apply_rescored_lm_head_controllers(
                &[0.1, 3.0, 2.0],
                1,
                &[1],
                2,
                config,
                &mut stats,
                &mut repeat_state,
                &mut novelty_state,
            )
            .unwrap(),
            2
        );
    }

    #[test]
    fn sparse_lm_head_argmax_no_repeat_last_only_skips_decode_token() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_input_tiles: true,
            aip_no_repeat_last: true,
            aip_repeat_run_limit: None,
        };

        assert_eq!(
            sample_sparse_lm_head_argmax(&[0.1, 3.0, 2.0], &[1], 1, config).unwrap(),
            2
        );
        assert_eq!(
            sample_sparse_lm_head_argmax(&[0.1, 3.0, 2.0], &[7, 1], 1, config).unwrap(),
            1
        );
    }

    #[test]
    fn sparse_lm_head_argmax_repeat_run_limit_skips_after_allowed_run() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };

        assert_eq!(
            sample_sparse_lm_head_argmax(&[0.1, 3.0, 2.0], &[1], 1, config).unwrap(),
            1
        );
        assert_eq!(
            sample_sparse_lm_head_argmax(&[0.1, 3.0, 2.0], &[1], 2, config).unwrap(),
            2
        );
    }

    #[test]
    fn sparse_lm_head_argmax_repeat_margin_uses_next_candidate_only_on_small_gap() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: Some(250),
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };

        assert_eq!(
            sample_sparse_lm_head_argmax(&[0.1, 3.0, 2.9], &[1], 1, config).unwrap(),
            2
        );
        assert_eq!(
            sample_sparse_lm_head_argmax(&[0.1, 3.0, 2.0], &[1], 1, config).unwrap(),
            1
        );
        assert_eq!(
            sample_sparse_lm_head_argmax(&[0.1, 3.0, 3.1], &[1], 1, config).unwrap(),
            2
        );
        assert_eq!(
            sample_sparse_lm_head_argmax(&[0.1, 3.0, 2.9], &[1], 0, config).unwrap(),
            1
        );
    }

    #[test]
    fn sparse_lm_head_argmax_repeat_margin_records_controller_stats() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: Some(250),
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };
        let mut stats = RamaExperimentalSpeedStats::default();

        assert_eq!(
            sample_sparse_lm_head_argmax_with_stats(&[0.1, 3.0, 2.9], &[1], 1, config, &mut stats)
                .unwrap(),
            2
        );
        assert_eq!(stats.lm_head_repeat_margin_checks, 1);
        assert_eq!(stats.lm_head_repeat_margin_switches, 1);
        assert_eq!(stats.lm_head_repeat_margin_max_gap_milli, 100);
    }

    #[test]
    fn sparse_lm_head_argmax_adaptive_repeat_margin_throttles_switch_streak() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: Some(500),
            aip_lm_head_repeat_margin_adaptive: true,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };
        let mut stats = RamaExperimentalSpeedStats::default();
        let mut state = LmHeadRepeatMarginState::default();

        for _ in 0..3 {
            assert_eq!(
                sample_sparse_lm_head_argmax_with_adaptive_state(
                    &[0.1, 3.0, 2.7],
                    &[1],
                    1,
                    config,
                    &mut stats,
                    &mut state
                )
                .unwrap(),
                2
            );
        }

        assert_eq!(
            sample_sparse_lm_head_argmax_with_adaptive_state(
                &[0.1, 3.0, 2.7],
                &[1],
                1,
                config,
                &mut stats,
                &mut state
            )
            .unwrap(),
            1
        );
        assert_eq!(stats.lm_head_repeat_margin_checks, 4);
        assert_eq!(stats.lm_head_repeat_margin_switches, 3);
        assert_eq!(stats.lm_head_repeat_margin_adaptive_throttles, 1);
        assert_eq!(stats.lm_head_repeat_margin_min_effective_milli, 125);
    }

    #[test]
    fn sparse_lm_head_argmax_phrase_novelty_skips_repeated_ngram() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: Some(16),
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };
        let mut stats = RamaExperimentalSpeedStats::default();
        let mut repeat_state = LmHeadRepeatMarginState::default();
        let mut novelty_state = LmHeadPhraseNoveltyState::default();
        for token in [1, 2, 3, 1, 2] {
            novelty_state.push(token);
        }

        assert_eq!(
            sample_sparse_lm_head_argmax_with_controller_state(
                &[0.0, 0.1, 0.2, 3.0, 2.9],
                &[2],
                1,
                config,
                &mut stats,
                &mut repeat_state,
                &mut novelty_state
            )
            .unwrap(),
            4
        );
        assert_eq!(stats.lm_head_phrase_novelty_checks, 1);
        assert_eq!(stats.lm_head_phrase_novelty_switches, 1);
        assert_eq!(stats.lm_head_phrase_novelty_max_ngram, 3);
    }

    #[test]
    fn sparse_lm_head_argmax_phrase_novelty_keeps_confident_top_candidate() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: Some(16),
            aip_lm_head_novelty_gap_milli: Some(250),
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };
        let mut stats = RamaExperimentalSpeedStats::default();
        let mut repeat_state = LmHeadRepeatMarginState::default();
        let mut novelty_state = LmHeadPhraseNoveltyState::default();
        for token in [1, 2, 3, 1, 2] {
            novelty_state.push(token);
        }

        assert_eq!(
            sample_sparse_lm_head_argmax_with_controller_state(
                &[0.0, 0.1, 0.2, 3.0, 2.0],
                &[2],
                1,
                config,
                &mut stats,
                &mut repeat_state,
                &mut novelty_state
            )
            .unwrap(),
            3
        );
        assert_eq!(stats.lm_head_phrase_novelty_checks, 1);
        assert_eq!(stats.lm_head_phrase_novelty_switches, 0);
        assert_eq!(stats.lm_head_phrase_novelty_gap_skips, 1);
        assert_eq!(stats.lm_head_phrase_novelty_max_gap_milli, 1000);
    }

    #[test]
    fn sparse_lm_head_argmax_phrase_novelty_soft_ranking_keeps_close_repeat_candidate() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: Some(16),
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: Some(150),
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };
        let mut stats = RamaExperimentalSpeedStats::default();
        let mut repeat_state = LmHeadRepeatMarginState::default();
        let mut novelty_state = LmHeadPhraseNoveltyState::default();
        for token in [1, 2, 3, 1, 2, 4, 1, 2] {
            novelty_state.push(token);
        }

        assert_eq!(
            sample_sparse_lm_head_argmax_with_controller_state(
                &[0.0, 0.1, 0.2, 3.0, 2.95, 2.7],
                &[2],
                1,
                config,
                &mut stats,
                &mut repeat_state,
                &mut novelty_state
            )
            .unwrap(),
            4
        );
        assert_eq!(stats.lm_head_phrase_novelty_checks, 1);
        assert_eq!(stats.lm_head_phrase_novelty_switches, 1);
        assert_eq!(stats.lm_head_phrase_novelty_soft_choices, 1);
    }

    #[test]
    fn sparse_lm_head_argmax_phrase_novelty_retention_keeps_top_candidate_when_fallback_is_weak() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: Some(16),
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: Some(150),
            aip_lm_head_novelty_retention_milli: Some(100),
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };
        let mut stats = RamaExperimentalSpeedStats::default();
        let mut repeat_state = LmHeadRepeatMarginState::default();
        let mut novelty_state = LmHeadPhraseNoveltyState::default();
        for token in [1, 2, 3, 1, 2, 4, 1, 2] {
            novelty_state.push(token);
        }

        assert_eq!(
            sample_sparse_lm_head_argmax_with_controller_state(
                &[0.0, 0.1, 0.2, 3.0, 2.95, 2.7],
                &[2],
                1,
                config,
                &mut stats,
                &mut repeat_state,
                &mut novelty_state
            )
            .unwrap(),
            3
        );
        assert_eq!(stats.lm_head_phrase_novelty_checks, 1);
        assert_eq!(stats.lm_head_phrase_novelty_switches, 0);
        assert_eq!(stats.lm_head_phrase_novelty_retentions, 1);
    }

    #[test]
    fn sparse_lm_head_argmax_phrase_novelty_retention_switches_to_close_candidate() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: Some(16),
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: Some(100),
            aip_column_cache: false,
            aip_input_tiles: true,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: Some(2),
        };
        let mut stats = RamaExperimentalSpeedStats::default();
        let mut repeat_state = LmHeadRepeatMarginState::default();
        let mut novelty_state = LmHeadPhraseNoveltyState::default();
        for token in [1, 2, 3, 1, 2] {
            novelty_state.push(token);
        }

        assert_eq!(
            sample_sparse_lm_head_argmax_with_controller_state(
                &[0.0, 0.1, 0.2, 3.0, 2.95],
                &[2],
                1,
                config,
                &mut stats,
                &mut repeat_state,
                &mut novelty_state
            )
            .unwrap(),
            4
        );
        assert_eq!(stats.lm_head_phrase_novelty_checks, 1);
        assert_eq!(stats.lm_head_phrase_novelty_switches, 1);
        assert_eq!(stats.lm_head_phrase_novelty_retentions, 0);
    }

    #[test]
    fn sparse_lm_head_rescore_candidates_only_when_top_token_repeats() {
        let config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
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
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: Some(3),
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
            aip_input_tiles: true,
            aip_no_repeat_last: true,
            aip_repeat_run_limit: None,
        };

        let mut stats = RamaExperimentalSpeedStats::default();
        assert_eq!(
            sparse_lm_head_rescore_candidates(&[0.1, 3.0, 2.0], &[1], config, &mut stats).unwrap(),
            Some(vec![2, 0])
        );
        assert_eq!(
            sparse_lm_head_rescore_candidates(&[0.1, 3.0, 2.0], &[0], config, &mut stats).unwrap(),
            None
        );
        assert_eq!(
            sparse_lm_head_rescore_candidates(&[0.1, 3.0, 2.0], &[7, 1], config, &mut stats)
                .unwrap(),
            None
        );
    }

    #[test]
    fn sparse_lm_head_agreement_records_raw_selected_and_topk_hits() {
        let mut stats = RamaExperimentalSpeedStats::default();

        record_sparse_lm_head_agreement_sample(&mut stats, &[0.1, 4.0, 3.0, 2.0], 2, 1, 2).unwrap();
        record_sparse_lm_head_agreement_sample(&mut stats, &[0.1, 4.0, 3.0, 2.0], 2, 2, 2).unwrap();

        assert_eq!(stats.lm_head_agreement_samples, 2);
        assert_eq!(stats.lm_head_agreement_sparse_argmax_matches, 1);
        assert_eq!(stats.lm_head_agreement_selected_matches, 1);
        assert_eq!(stats.lm_head_agreement_exact_in_sparse_topk, 2);
        assert_eq!(stats.lm_head_agreement_max_topk, 2);
    }

    #[test]
    fn llama_session_quality_policy_uses_fewer_aip_calls_than_speed_policy() {
        let path = temp_path("aip-quality-vs-speed");
        write_bf16_mlp_speed_model_with_layers(&path, 8, 4);

        let mut quality_model = LazyRllmModel::open(&path).unwrap();
        let quality_prepared = prepared_with_layers(4);
        let mut quality_budget = MemoryBudget::unbounded();
        let mut quality_adapter = LlamaRamaSessionAdapter::new(
            &mut quality_model,
            &quality_prepared,
            &mut quality_budget,
        )
        .unwrap();
        quality_adapter.enable_experimental_speed_for_test(crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Quality,
            aip_topk: Some(1),
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
        });
        quality_adapter
            .append_tokens(&[0], &mut quality_budget, true)
            .unwrap();
        let quality_stats = quality_adapter
            .take_last_experimental_speed_stats()
            .unwrap();

        let mut speed_model = LazyRllmModel::open(&path).unwrap();
        let speed_prepared = prepared_with_layers(4);
        let mut speed_budget = MemoryBudget::unbounded();
        let mut speed_adapter =
            LlamaRamaSessionAdapter::new(&mut speed_model, &speed_prepared, &mut speed_budget)
                .unwrap();
        speed_adapter.enable_experimental_speed_for_test(crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
            aip_topk: Some(1),
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
        });
        speed_adapter
            .append_tokens(&[0], &mut speed_budget, true)
            .unwrap();
        let speed_stats = speed_adapter.take_last_experimental_speed_stats().unwrap();

        assert_eq!(
            quality_stats.aip_policy,
            Some(crate::RamaAipPolicyKind::Quality)
        );
        assert_eq!(
            speed_stats.aip_policy,
            Some(crate::RamaAipPolicyKind::Speed)
        );
        assert!(quality_stats.sparse_projection_calls > 0);
        assert!(quality_stats.sparse_projection_calls < speed_stats.sparse_projection_calls);
        assert_eq!(quality_stats.max_selected_topk, 1);
        assert_eq!(speed_stats.max_selected_topk, 1);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_clears_phase_timings_after_failed_append() {
        let path = temp_path("phase-timing-failure");
        write_post_cache_failure_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(2);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();

        let result = adapter.append_tokens(&[0], &mut budget, false);

        assert!(result.is_err());
        assert!(adapter.take_last_phase_timings().is_none());
        assert_eq!(adapter.context_len(), 0);
        std::fs::remove_file(path).ok();
    }

