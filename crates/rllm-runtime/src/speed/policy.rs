// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use crate::speed::{
    RamaAipPolicyKind, RamaAipProjectionDecision, RamaAipProjectionKind,
    RamaExperimentalSpeedConfig,
};
use std::cmp::Ordering;

impl RamaExperimentalSpeedConfig {
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
            _ => None,
        };
        self.topk_for_input_with_override(input_len, default_topk, topk_override)
    }

    fn edge_topk_for_layer(
        self,
        layer_index: usize,
        total_layers: usize,
        input_len: usize,
    ) -> Option<usize> {
        if !self.exact_edge_layer(layer_index, total_layers) {
            return None;
        }
        self.aip_edge_topk.map(|topk| topk.min(input_len).max(1))
    }

    pub fn exact_edge_layer(self, layer_index: usize, total_layers: usize) -> bool {
        if !self.enabled || total_layers == 0 {
            return false;
        }
        let Some(edge_layers) = self.aip_edge_layers else {
            return false;
        };
        let edge_layers = edge_layers.min(total_layers);
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
            .map(|exact_projection| exact_projection.matches(projection))
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
            return self
                .aip_exact_layer_projection
                .map(|exact| exact.matches(projection))
                .unwrap_or(false);
        };
        if exact_layer == 0 || layer_index + 1 != exact_layer {
            return false;
        }
        self.aip_exact_layer_projection
            .map(|exact_projection| exact_projection.matches(projection))
            .unwrap_or(true)
    }

    fn exact_prefix_projection(self, layer_index: usize, total_layers: usize) -> bool {
        if layer_index >= total_layers {
            return false;
        }
        let Some(prefix_layers) = self.aip_exact_prefix_layers else {
            return false;
        };
        layer_index < prefix_layers.min(total_layers)
    }

    fn exact_periodic_projection(self, layer_index: usize, total_layers: usize) -> bool {
        if layer_index >= total_layers {
            return false;
        }
        let Some(periodic_layers) = self.aip_exact_periodic_layers else {
            return false;
        };
        if periodic_layers == 0 {
            return false;
        }
        layer_index % periodic_layers == 0
    }

    pub fn attention_locality_enabled_for_layer(
        self,
        layer_index: usize,
        total_layers: usize,
    ) -> bool {
        if !self.enabled
            || self.aip_attention_locality_window.is_none()
            || layer_index >= total_layers
        {
            return false;
        }
        let edge_layers = self.aip_exact_edge_layers.unwrap_or(0).min(total_layers);
        if layer_index < edge_layers || layer_index >= total_layers.saturating_sub(edge_layers) {
            return true;
        }
        false
    }

    pub fn aip_decision_for_projection(
        self,
        layer_index: usize,
        total_layers: usize,
        projection: RamaAipProjectionKind,
        input_len: usize,
        default_topk: usize,
    ) -> RamaAipProjectionDecision {
        if !self.enabled {
            return RamaAipProjectionDecision::exact();
        }

        if self.exact_edge_projection(layer_index, total_layers, projection) {
            return RamaAipProjectionDecision::exact();
        }

        if self.exact_layer_projection(layer_index, total_layers, projection) {
            return RamaAipProjectionDecision::exact();
        }

        if self.exact_prefix_projection(layer_index, total_layers) {
            return RamaAipProjectionDecision::exact();
        }

        if self.exact_periodic_projection(layer_index, total_layers) {
            return RamaAipProjectionDecision::exact();
        }

        match self.aip_policy {
            RamaAipPolicyKind::Speed => {
                let layer_topk = if layer_index < self.aip_layer_topk_overrides.len() {
                    let val = self.aip_layer_topk_overrides[layer_index];
                    if val > 0 {
                        Some(val as usize)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let topk = if let Some(val) = layer_topk {
                    val.min(input_len).max(1)
                } else {
                    self.edge_topk_for_layer(layer_index, total_layers, input_len)
                        .unwrap_or_else(|| {
                            self.topk_for_projection(projection, input_len, default_topk)
                        })
                };
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

pub(crate) fn compare_top_abs_candidates(left: &(usize, f32), right: &(usize, f32)) -> Ordering {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
}
