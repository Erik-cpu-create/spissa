// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

use crate::speed::RamaAipPolicyKind;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaLayerDriftProbeStats {
    pub samples: usize,
    pub layers: usize,
    pub mismatch_layers: usize,
    pub first_mismatch_layer: usize,
    pub pre_mismatch_max_l2_milli: usize,
    pub pre_mismatch_max_cosine_gap_milli: usize,
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
        self.pre_mismatch_max_l2_milli = self
            .pre_mismatch_max_l2_milli
            .max(other.pre_mismatch_max_l2_milli);
        self.pre_mismatch_max_cosine_gap_milli = self
            .pre_mismatch_max_cosine_gap_milli
            .max(other.pre_mismatch_max_cosine_gap_milli);
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
        pre_mismatch_max_l2_milli: usize,
        pre_mismatch_max_cosine_gap_milli: usize,
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
        if let Some(first) = first_mismatch_layer {
            self.first_mismatch_layer = min_non_zero(self.first_mismatch_layer, first);
        }
        self.pre_mismatch_max_l2_milli = self
            .pre_mismatch_max_l2_milli
            .max(pre_mismatch_max_l2_milli);
        self.pre_mismatch_max_cosine_gap_milli = self
            .pre_mismatch_max_cosine_gap_milli
            .max(pre_mismatch_max_cosine_gap_milli);
        self.max_l2_milli = self.max_l2_milli.max(max_l2_milli);
        self.max_cosine_gap_milli = self.max_cosine_gap_milli.max(max_cosine_gap_milli);
        self.max_exact_margin_milli = self.max_exact_margin_milli.max(max_exact_margin_milli);
    }

    pub fn is_empty(self) -> bool {
        self.samples == 0
            && self.layers == 0
            && self.mismatch_layers == 0
            && self.first_mismatch_layer == 0
            && self.pre_mismatch_max_l2_milli == 0
            && self.pre_mismatch_max_cosine_gap_milli == 0
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
        pre_mismatch_max_l2_milli: usize,
        pre_mismatch_max_cosine_gap_milli: usize,
        max_l2_milli: usize,
        max_cosine_gap_milli: usize,
        max_exact_margin_milli: usize,
    ) {
        self.layer_drift_probe.record(
            layers,
            mismatch_layers,
            first_mismatch_layer,
            pre_mismatch_max_l2_milli,
            pre_mismatch_max_cosine_gap_milli,
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
