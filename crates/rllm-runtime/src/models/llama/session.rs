use crate::models::llama::api::{
    decode_vector_tensor, require_config_usize, require_model_config, validate_llama_shape,
};
use crate::models::llama::generate::{
    streaming_llama_transformer_block_with_timing, LlamaStreamingBlockConfig,
};
use crate::models::llama::model::{
    LayerDecodedLlamaRamaTransformer, OwnedLlamaStreamingBlockParameters,
    OwnedLlamaStreamingBlockTensorNames,
};
use crate::rolling::RollingExecutor;
#[cfg(test)]
use crate::rolling::RollingExecutorConfig;
use crate::rotary::KvCache;
use crate::speed::{
    parse_aip_exact_prefill_enabled, parse_aip_lm_head_exact_every, RLLM_AIP_EXACT_PREFILL_ENV,
    RLLM_AIP_LM_HEAD_EXACT_EVERY_ENV,
};
use crate::streaming::{
    streaming_tile_linear_argmax_candidate_rows_range_from_model,
    streaming_tile_linear_argmax_prefix_from_model,
    streaming_tile_linear_argmax_with_rolling_from_model,
};
use crate::{
    embedding_lookup, rms_norm, sample_argmax_excluding, sample_top_p, select_top_indices_by_value,
    streaming_input_tiled_sparse_tile_linear_from_model, streaming_tile_linear_argmax_from_model,
    streaming_tile_linear_from_model, LazyRllmModel, MemoryBudget, Result, RuntimeError,
};
use crate::{RamaAttentionLocalityCache, RamaExperimentalSpeedConfig, RamaExperimentalSpeedStats};
use crate::{
    RamaRollingStats, RamaSessionAdapter, RamaSessionPhaseTimings, RamaSessionStep,
    RamaTransformerPhaseTimings, SparseColumnCache, StreamingLinearConfig,
    StreamingTileLinearConfig, DEFAULT_STREAMING_TILE_ELEMENTS,
};
use std::time::Instant;

pub struct LlamaRamaSessionAdapter<'a> {
    model: &'a mut LazyRllmModel,
    prepared: LayerDecodedLlamaRamaTransformer,
    hidden_size: usize,
    intermediate_size: usize,
    head_dim: usize,
    vocab_size: usize,
    embedding_data: Vec<f32>,
    layer_norms: Vec<OwnedLlamaStreamingBlockParameters>,
    caches: Vec<KvCache>,
    last_phase_timings: Option<RamaSessionPhaseTimings>,
    rolling_executor: Option<RollingExecutor>,
    last_rolling_stats: Option<RamaRollingStats>,
    experimental_speed_config: RamaExperimentalSpeedConfig,
    exact_prefill: bool,
    lm_head_exact_every: Option<usize>,
    last_experimental_speed_stats: Option<RamaExperimentalSpeedStats>,
    sparse_column_cache: SparseColumnCache,
    attention_locality_caches: Vec<RamaAttentionLocalityCache>,
    collect_transformer_detail_timing: bool,
    last_generated_token: Option<usize>,
    last_generated_token_run: usize,
    generated_token_count_in_turn: usize,
    lm_head_repeat_margin_state: LmHeadRepeatMarginState,
    lm_head_phrase_novelty_state: LmHeadPhraseNoveltyState,
}

#[derive(Debug, Clone, Copy, Default)]
struct LmHeadRepeatMarginState {
    consecutive_switches: usize,
}

impl LmHeadRepeatMarginState {
    fn effective_margin_milli(self, base_margin_milli: usize, adaptive: bool) -> usize {
        const SWITCH_STREAK_THRESHOLD: usize = 3;
        const THROTTLE_DIVISOR: usize = 4;

        if adaptive && self.consecutive_switches >= SWITCH_STREAK_THRESHOLD {
            (base_margin_milli / THROTTLE_DIVISOR).max(1)
        } else {
            base_margin_milli
        }
    }

    fn record_margin_check(&mut self, switched: bool) {
        if switched {
            self.consecutive_switches = self.consecutive_switches.saturating_add(1);
        } else {
            self.consecutive_switches = 0;
        }
    }

    fn reset(&mut self) {
        self.consecutive_switches = 0;
    }
}

#[derive(Debug, Clone, Copy)]
struct LmHeadPhraseNoveltyState {
    recent: [usize; Self::MAX_WINDOW],
    len: usize,
}

impl Default for LmHeadPhraseNoveltyState {
    fn default() -> Self {
        Self {
            recent: [0; Self::MAX_WINDOW],
            len: 0,
        }
    }
}

impl LmHeadPhraseNoveltyState {
    const MAX_WINDOW: usize = 32;
    const MAX_NGRAM: usize = 4;
    const MIN_NGRAM: usize = 2;

    fn push(&mut self, token_id: usize) {
        if self.len < Self::MAX_WINDOW {
            self.recent[self.len] = token_id;
            self.len += 1;
        } else {
            self.recent.copy_within(1..Self::MAX_WINDOW, 0);
            self.recent[Self::MAX_WINDOW - 1] = token_id;
        }
    }

    fn reset(&mut self) {
        self.len = 0;
    }

    fn repeated_ngram_len(self, candidate: usize, window: usize) -> Option<usize> {
        let history_len = self
            .len
            .min(window.clamp(Self::MIN_NGRAM, Self::MAX_WINDOW));
        if history_len < Self::MIN_NGRAM {
            return None;
        }
        for ngram_len in (Self::MIN_NGRAM..=Self::MAX_NGRAM).rev() {
            if history_len < ngram_len.saturating_mul(2).saturating_sub(1)
                || self.len < ngram_len.saturating_sub(1)
            {
                continue;
            }
            if self.candidate_repeats_ngram(candidate, history_len, ngram_len) {
                return Some(ngram_len);
            }
        }
        None
    }

    fn candidate_repeats_ngram(
        self,
        candidate: usize,
        history_len: usize,
        ngram_len: usize,
    ) -> bool {
        let suffix_start = self.len - (ngram_len - 1);
        let earliest_start = self.len - history_len;
        let Some(latest_start) = suffix_start.checked_sub(ngram_len) else {
            return false;
        };
        if latest_start < earliest_start {
            return false;
        }

        for start in earliest_start..=latest_start {
            let mut matches = true;
            for offset in 0..ngram_len {
                let suffix_token = if offset + 1 == ngram_len {
                    candidate
                } else {
                    self.recent[suffix_start + offset]
                };
                if self.recent[start + offset] != suffix_token {
                    matches = false;
                    break;
                }
            }
            if matches {
                return true;
            }
        }
        false
    }
}

fn tensor_shape_usize(model: &LazyRllmModel, name: &str) -> Result<Vec<usize>> {
    model
        .tensor(name)?
        .shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                RuntimeError::Shape(format!("tensor {name} dimension {dim} overflows usize"))
            })
        })
        .collect()
}

fn validate_matrix_with_columns(
    model: &LazyRllmModel,
    name: &str,
    expected_cols: usize,
) -> Result<usize> {
    let shape = tensor_shape_usize(model, name)?;
    if shape.len() != 2 {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} must be rank-2 [rows, {expected_cols}], got {:?}",
            shape
        )));
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows == 0 {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} must have non-zero row count, got {:?}",
            shape
        )));
    }
    if cols != expected_cols {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected [rows, {expected_cols}]",
            shape
        )));
    }
    Ok(rows)
}

fn validate_matrix_shape(
    model: &LazyRllmModel,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<()> {
    let shape = tensor_shape_usize(model, name)?;
    if shape != [expected_rows, expected_cols] {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected [{expected_rows}, {expected_cols}]",
            shape
        )));
    }
    Ok(())
}

fn checked_projection_rows(label: &str, heads: usize, head_dim: usize) -> Result<usize> {
    heads.checked_mul(head_dim).ok_or_else(|| {
        RuntimeError::Shape(format!(
            "llama session {label} projection row count overflow"
        ))
    })
}

fn lm_head_exact_check_due(
    exact_every: Option<usize>,
    is_decode_step: bool,
    generated_token_count_in_turn: usize,
) -> bool {
    is_decode_step
        && exact_every.is_some_and(|every| {
            every > 0 && generated_token_count_in_turn.saturating_add(1) % every == 0
        })
}

#[cfg(test)]
fn sample_sparse_lm_head_argmax(
    logits: &[f32],
    appended_tokens: &[usize],
    previous_token_run: usize,
    config: RamaExperimentalSpeedConfig,
) -> Result<usize> {
    sample_sparse_lm_head_argmax_inner(
        logits,
        appended_tokens,
        previous_token_run,
        config,
        None,
        None,
        None,
    )
}

#[cfg(test)]
fn sample_sparse_lm_head_argmax_with_stats(
    logits: &[f32],
    appended_tokens: &[usize],
    previous_token_run: usize,
    config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
) -> Result<usize> {
    sample_sparse_lm_head_argmax_inner(
        logits,
        appended_tokens,
        previous_token_run,
        config,
        Some(stats),
        None,
        None,
    )
}

#[cfg(test)]
fn sample_sparse_lm_head_argmax_with_adaptive_state(
    logits: &[f32],
    appended_tokens: &[usize],
    previous_token_run: usize,
    config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
    state: &mut LmHeadRepeatMarginState,
) -> Result<usize> {
    sample_sparse_lm_head_argmax_inner(
        logits,
        appended_tokens,
        previous_token_run,
        config,
        Some(stats),
        Some(state),
        None,
    )
}

fn sample_sparse_lm_head_argmax_with_controller_state(
    logits: &[f32],
    appended_tokens: &[usize],
    previous_token_run: usize,
    config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
    repeat_margin_state: &mut LmHeadRepeatMarginState,
    phrase_novelty_state: &mut LmHeadPhraseNoveltyState,
) -> Result<usize> {
    sample_sparse_lm_head_argmax_inner(
        logits,
        appended_tokens,
        previous_token_run,
        config,
        Some(stats),
        Some(repeat_margin_state),
        Some(phrase_novelty_state),
    )
}

fn sample_sparse_lm_head_argmax_inner(
    logits: &[f32],
    appended_tokens: &[usize],
    previous_token_run: usize,
    config: RamaExperimentalSpeedConfig,
    mut stats: Option<&mut RamaExperimentalSpeedStats>,
    mut repeat_margin_state: Option<&mut LmHeadRepeatMarginState>,
    phrase_novelty_state: Option<&LmHeadPhraseNoveltyState>,
) -> Result<usize> {
    let excluded_token = if appended_tokens.len() == 1 {
        let previous = appended_tokens.first().copied();
        let repeat_limit_reached = config
            .aip_repeat_run_limit
            .is_some_and(|limit| previous_token_run >= limit);
        if config.aip_no_repeat_last || repeat_limit_reached {
            previous
        } else {
            None
        }
    } else {
        None
    };
    let mut novelty_excluded_token = excluded_token;
    if excluded_token.is_some() {
        if config.aip_lm_head_repeat_margin_adaptive {
            if let Some(state) = repeat_margin_state.as_deref_mut() {
                state.reset();
            }
        }
    }

    let mut selected_token = None;
    if excluded_token.is_none() {
        if let Some(margin_milli) = config.aip_lm_head_repeat_margin_milli {
            if appended_tokens.len() == 1 && previous_token_run > 0 {
                let previous = appended_tokens[0];
                let (best_idx, best_value, second) = top_two_sparse_logits(logits)?;
                if best_idx == previous {
                    if let Some((second_idx, second_value)) = second {
                        let effective_margin_milli = repeat_margin_state
                            .as_deref()
                            .map(|state| {
                                state.effective_margin_milli(
                                    margin_milli,
                                    config.aip_lm_head_repeat_margin_adaptive,
                                )
                            })
                            .unwrap_or(margin_milli);
                        if effective_margin_milli < margin_milli {
                            if let Some(stats) = stats.as_deref_mut() {
                                stats.record_lm_head_repeat_margin_adaptive_throttle(
                                    effective_margin_milli,
                                );
                            }
                        }
                        let margin = effective_margin_milli as f32 / 1000.0;
                        let gap = best_value - second_value;
                        if gap.is_finite() && gap <= margin {
                            if let Some(stats) = stats.as_deref_mut() {
                                stats.record_lm_head_repeat_margin(true, gap_to_milli(gap));
                            }
                            if config.aip_lm_head_repeat_margin_adaptive {
                                if let Some(state) = repeat_margin_state.as_deref_mut() {
                                    state.record_margin_check(true);
                                }
                            }
                            selected_token = Some(second_idx);
                            novelty_excluded_token = Some(previous);
                        }
                        if selected_token.is_none() {
                            if let Some(stats) = stats.as_deref_mut() {
                                stats.record_lm_head_repeat_margin(false, gap_to_milli(gap));
                            }
                            if config.aip_lm_head_repeat_margin_adaptive {
                                if let Some(state) = repeat_margin_state.as_deref_mut() {
                                    state.record_margin_check(false);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let selected_token = selected_token.unwrap_or(sample_argmax_excluding(logits, excluded_token)?);
    Ok(apply_phrase_novelty_controller(
        logits,
        selected_token,
        novelty_excluded_token,
        config,
        stats,
        phrase_novelty_state,
    ))
}

fn apply_phrase_novelty_controller(
    logits: &[f32],
    selected_token: usize,
    excluded_token: Option<usize>,
    config: RamaExperimentalSpeedConfig,
    mut stats: Option<&mut RamaExperimentalSpeedStats>,
    phrase_novelty_state: Option<&LmHeadPhraseNoveltyState>,
) -> usize {
    let Some(window) = config.aip_lm_head_novelty_window else {
        return selected_token;
    };
    let Some(state) = phrase_novelty_state else {
        return selected_token;
    };
    let Some(repeated_ngram) = state.repeated_ngram_len(selected_token, window) else {
        if let Some(stats) = stats.as_deref_mut() {
            stats.record_lm_head_phrase_novelty(false, 0);
        }
        return selected_token;
    };

    let candidates = top_sparse_candidates_by_value(logits, 4, excluded_token);
    if let Some(gap_limit_milli) = config.aip_lm_head_novelty_gap_milli {
        if let Some(gap_milli) = selected_candidate_gap_milli(logits, &candidates, selected_token) {
            if gap_milli > gap_limit_milli {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.record_lm_head_phrase_novelty(false, repeated_ngram);
                    stats.record_lm_head_phrase_novelty_gap_skip(gap_milli);
                }
                return selected_token;
            }
        }
    }

    if let Some(retention_milli) = config.aip_lm_head_novelty_retention_milli {
        match phrase_novelty_retention_decision(
            logits,
            selected_token,
            &candidates,
            state,
            window,
            config.aip_lm_head_novelty_repeat_penalty_milli,
            retention_milli,
        ) {
            PhraseNoveltyRetentionDecision::Switch(candidate) => {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.record_lm_head_phrase_novelty(true, repeated_ngram);
                }
                return candidate;
            }
            PhraseNoveltyRetentionDecision::Retain => {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.record_lm_head_phrase_novelty(false, repeated_ngram);
                    stats.record_lm_head_phrase_novelty_retention();
                }
                return selected_token;
            }
        }
    }

    if let Some(repeat_penalty_milli) = config.aip_lm_head_novelty_repeat_penalty_milli {
        if let Some(candidate) = select_soft_phrase_novelty_candidate(
            logits,
            selected_token,
            &candidates,
            state,
            window,
            repeat_penalty_milli,
        ) {
            if let Some(stats) = stats.as_deref_mut() {
                stats.record_lm_head_phrase_novelty(true, repeated_ngram);
                stats.record_lm_head_phrase_novelty_soft_choice();
            }
            return candidate;
        }
    }

    for candidate in candidates {
        if candidate == selected_token || Some(candidate) == excluded_token {
            continue;
        }
        if state.repeated_ngram_len(candidate, window).is_none() {
            if let Some(stats) = stats.as_deref_mut() {
                stats.record_lm_head_phrase_novelty(true, repeated_ngram);
            }
            return candidate;
        }
    }

    if let Some(stats) = stats.as_deref_mut() {
        stats.record_lm_head_phrase_novelty(false, repeated_ngram);
    }
    selected_token
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhraseNoveltyRetentionDecision {
    Switch(usize),
    Retain,
}

fn phrase_novelty_retention_decision(
    logits: &[f32],
    selected_token: usize,
    candidates: &[usize],
    state: &LmHeadPhraseNoveltyState,
    window: usize,
    repeat_penalty_milli: Option<usize>,
    retention_milli: usize,
) -> PhraseNoveltyRetentionDecision {
    let Some(selected_value) = logits.get(selected_token).copied() else {
        return PhraseNoveltyRetentionDecision::Retain;
    };
    let mut best: Option<(usize, usize)> = None;
    for &candidate in candidates {
        if candidate == selected_token {
            continue;
        }
        let candidate_repeats = state.repeated_ngram_len(candidate, window).is_some();
        if candidate_repeats && repeat_penalty_milli.is_none() {
            continue;
        }
        let candidate_value = logits.get(candidate).copied().unwrap_or(f32::NEG_INFINITY);
        let gap_milli = gap_to_milli(selected_value - candidate_value);
        let repeat_penalty = if candidate_repeats {
            repeat_penalty_milli.unwrap_or(0)
        } else {
            0
        };
        let score = gap_milli.saturating_add(repeat_penalty);
        if best.is_none_or(|(best_candidate, best_score)| {
            score < best_score || (score == best_score && candidate < best_candidate)
        }) {
            best = Some((candidate, score));
        }
    }

    match best {
        Some((candidate, score)) if score <= retention_milli => {
            PhraseNoveltyRetentionDecision::Switch(candidate)
        }
        _ => PhraseNoveltyRetentionDecision::Retain,
    }
}

fn select_soft_phrase_novelty_candidate(
    logits: &[f32],
    selected_token: usize,
    candidates: &[usize],
    state: &LmHeadPhraseNoveltyState,
    window: usize,
    repeat_penalty_milli: usize,
) -> Option<usize> {
    let selected_value = logits.get(selected_token).copied()?;
    let mut best: Option<(usize, usize)> = None;
    for &candidate in candidates {
        if candidate == selected_token {
            continue;
        }
        let candidate_value = logits.get(candidate).copied().unwrap_or(f32::NEG_INFINITY);
        let gap_milli = gap_to_milli(selected_value - candidate_value);
        let repeat_penalty = if state.repeated_ngram_len(candidate, window).is_some() {
            repeat_penalty_milli
        } else {
            0
        };
        let score = gap_milli.saturating_add(repeat_penalty);
        if best.is_none_or(|(best_candidate, best_score)| {
            score < best_score || (score == best_score && candidate < best_candidate)
        }) {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

fn top_sparse_candidates_by_value(
    logits: &[f32],
    limit: usize,
    excluded_token: Option<usize>,
) -> Vec<usize> {
    let limit = limit.min(logits.len());
    if limit == 0 {
        return Vec::new();
    }
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(limit);
    for (idx, &value) in logits.iter().enumerate() {
        if Some(idx) == excluded_token {
            continue;
        }
        let insert_at = top
            .iter()
            .position(|&(existing_idx, existing_value)| {
                value > existing_value || (value == existing_value && idx < existing_idx)
            })
            .unwrap_or(top.len());
        if insert_at < limit {
            top.insert(insert_at, (idx, value));
            top.truncate(limit);
        }
    }
    top.into_iter().map(|(idx, _)| idx).collect()
}

fn selected_candidate_gap_milli(
    logits: &[f32],
    candidates: &[usize],
    selected_token: usize,
) -> Option<usize> {
    let selected_rank = candidates
        .iter()
        .position(|candidate| *candidate == selected_token)?;
    let next_candidate = candidates.get(selected_rank + 1).copied()?;
    let gap = logits
        .get(selected_token)
        .copied()
        .unwrap_or(f32::NEG_INFINITY)
        - logits
            .get(next_candidate)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
    Some(gap_to_milli(gap))
}

fn gap_to_milli(gap: f32) -> usize {
    if !gap.is_finite() || gap <= 0.0 {
        0
    } else {
        (gap * 1000.0).round() as usize
    }
}

fn top_two_sparse_logits(logits: &[f32]) -> Result<(usize, f32, Option<(usize, f32)>)> {
    if logits.is_empty() {
        return Err(RuntimeError::InvalidTensorData(
            "cannot sample from empty logits".to_string(),
        ));
    }
    let mut best_idx = 0usize;
    let mut best_value = logits[0];
    let mut second: Option<(usize, f32)> = None;

    for (idx, &value) in logits.iter().enumerate().skip(1) {
        if value > best_value {
            second = Some((best_idx, best_value));
            best_idx = idx;
            best_value = value;
        } else if second.is_none_or(|(_, second_value)| value > second_value) {
            second = Some((idx, value));
        }
    }

    Ok((best_idx, best_value, second))
}

fn sparse_lm_head_rescore_candidates(
    logits: &[f32],
    appended_tokens: &[usize],
    config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
) -> Result<Option<Vec<usize>>> {
    let Some(candidate_count) = config.aip_lm_head_rescore else {
        return Ok(None);
    };
    if config.aip_no_repeat_last {
        if appended_tokens.len() != 1 {
            return Ok(None);
        }
        let Some(previous) = appended_tokens.first().copied() else {
            return Ok(None);
        };
        if sample_argmax_excluding(logits, None)? != previous {
            return Ok(None);
        }
        let mut candidates = select_top_indices_by_value(logits, candidate_count);
        candidates.retain(|token_id| *token_id != previous);
        if candidates.is_empty() {
            return Ok(None);
        }
        return Ok(confidence_gated_lm_head_rescore_candidates(
            logits, candidates, config, stats,
        ));
    }

    let candidates = select_top_indices_by_value(logits, candidate_count);
    if candidates.is_empty() {
        Ok(None)
    } else {
        Ok(confidence_gated_lm_head_rescore_candidates(
            logits, candidates, config, stats,
        ))
    }
}

fn confidence_gated_lm_head_rescore_candidates(
    logits: &[f32],
    candidates: Vec<usize>,
    config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
) -> Option<Vec<usize>> {
    let gap_milli = rescore_candidate_gap_milli(logits, &candidates);
    let use_rescore = config
        .aip_lm_head_rescore_gap_milli
        .is_none_or(|limit| gap_milli <= limit);
    stats.record_lm_head_rescore(use_rescore, gap_milli);
    use_rescore.then_some(candidates)
}

fn rescore_candidate_gap_milli(logits: &[f32], candidates: &[usize]) -> usize {
    let Some((&first, rest)) = candidates.split_first() else {
        return 0;
    };
    let Some(&second) = rest.first() else {
        return 0;
    };
    let first_value = logits.get(first).copied().unwrap_or(f32::NEG_INFINITY);
    let second_value = logits.get(second).copied().unwrap_or(f32::NEG_INFINITY);
    gap_to_milli(first_value - second_value)
}

fn apply_rescored_lm_head_controllers(
    logits: &[f32],
    rescored_token_id: usize,
    appended_tokens: &[usize],
    previous_token_run: usize,
    config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
    repeat_margin_state: &mut LmHeadRepeatMarginState,
    phrase_novelty_state: &mut LmHeadPhraseNoveltyState,
) -> Result<usize> {
    let repeats_previous = appended_tokens.len() == 1
        && appended_tokens
            .first()
            .copied()
            .is_some_and(|previous| previous == rescored_token_id);
    let repeat_limit_reached = config
        .aip_repeat_run_limit
        .is_some_and(|limit| previous_token_run >= limit);
    let repeat_controller_needed = repeats_previous
        && (config.aip_no_repeat_last
            || repeat_limit_reached
            || config.aip_lm_head_repeat_margin_milli.is_some());
    if repeat_controller_needed {
        return sample_sparse_lm_head_argmax_with_controller_state(
            logits,
            appended_tokens,
            previous_token_run,
            config,
            stats,
            repeat_margin_state,
            phrase_novelty_state,
        );
    }

    Ok(apply_phrase_novelty_controller(
        logits,
        rescored_token_id,
        None,
        config,
        Some(stats),
        Some(phrase_novelty_state),
    ))
}

fn record_sparse_lm_head_agreement_sample(
    stats: &mut RamaExperimentalSpeedStats,
    sparse_logits: &[f32],
    selected_token_id: usize,
    exact_token_id: usize,
    sparse_topk: usize,
) -> Result<()> {
    let sparse_argmax = sample_argmax_excluding(sparse_logits, None)?;
    let candidates = select_top_indices_by_value(sparse_logits, sparse_topk);
    stats.record_lm_head_agreement(
        sparse_argmax == exact_token_id,
        selected_token_id == exact_token_id,
        candidates.contains(&exact_token_id),
        candidates.len(),
    );
    Ok(())
}

fn validate_layer_tensor_shapes(
    model: &LazyRllmModel,
    layer_names: &OwnedLlamaStreamingBlockTensorNames,
    hidden_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
) -> Result<()> {
    let q_rows = checked_projection_rows("q", q_heads, head_dim)?;
    let kv_rows = checked_projection_rows("kv", kv_heads, head_dim)?;

    validate_matrix_shape(model, &layer_names.q_weight, q_rows, hidden_size)?;
    validate_matrix_shape(model, &layer_names.k_weight, kv_rows, hidden_size)?;
    validate_matrix_shape(model, &layer_names.v_weight, kv_rows, hidden_size)?;
    validate_matrix_shape(model, &layer_names.o_weight, hidden_size, q_rows)?;
    validate_matrix_shape(
        model,
        &layer_names.gate_weight,
        intermediate_size,
        hidden_size,
    )?;
    validate_matrix_shape(
        model,
        &layer_names.up_weight,
        intermediate_size,
        hidden_size,
    )?;
    validate_matrix_shape(
        model,
        &layer_names.down_weight,
        hidden_size,
        intermediate_size,
    )?;
    Ok(())
}

impl<'a> LlamaRamaSessionAdapter<'a> {
    pub fn new(
        model: &'a mut LazyRllmModel,
        prepared: &LayerDecodedLlamaRamaTransformer,
        budget: &mut MemoryBudget,
    ) -> Result<Self> {
        if prepared.layers.is_empty() {
            return Err(RuntimeError::Shape(
                "llama session requires at least one layer".to_string(),
            ));
        }

        let model_config = require_model_config(model, "llama")?;
        let hidden_size = require_config_usize("hidden_size", model_config.hidden_size)?;
        let intermediate_size =
            require_config_usize("intermediate_size", model_config.intermediate_size)?;
        if intermediate_size == 0 {
            return Err(RuntimeError::Shape(
                "llama session intermediate_size must be non-zero".to_string(),
            ));
        }
        if prepared.final_layernorm_weight.len() != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "llama session final_layernorm_weight len {} does not match hidden_size {hidden_size}",
                prepared.final_layernorm_weight.len()
            )));
        }
        let head_dim = validate_llama_shape(
            hidden_size,
            prepared.config.num_heads,
            prepared.config.num_key_value_heads,
        )?;
        let max_seq_len = prepared.config.max_seq_len.ok_or_else(|| {
            RuntimeError::InvalidTensorData("llama session config requires max_seq_len".to_string())
        })?;

        let vocab_size =
            validate_matrix_with_columns(model, &prepared.embedding_weight, hidden_size)?;
        validate_matrix_shape(model, &prepared.lm_head_weight, vocab_size, hidden_size)?;
        for layer_names in &prepared.layers {
            validate_layer_tensor_shapes(
                model,
                layer_names,
                hidden_size,
                prepared.config.num_heads,
                prepared.config.num_key_value_heads,
                head_dim,
                intermediate_size,
            )?;
        }

        let embedding_data = model
            .decode_tensor(&prepared.embedding_weight, budget)?
            .data;

        let mut layer_norms = Vec::with_capacity(prepared.layers.len());
        for i in 0..prepared.layers.len() {
            layer_norms.push(OwnedLlamaStreamingBlockParameters {
                input_layernorm_weight: decode_vector_tensor(
                    model,
                    &format!("model.layers.{i}.input_layernorm.weight"),
                    hidden_size,
                )?,
                post_attention_layernorm_weight: decode_vector_tensor(
                    model,
                    &format!("model.layers.{i}.post_attention_layernorm.weight"),
                    hidden_size,
                )?,
            });
        }

        let mut caches = Vec::with_capacity(prepared.layers.len());
        for _ in 0..prepared.layers.len() {
            caches.push(KvCache::new(
                prepared.config.num_key_value_heads,
                head_dim,
                max_seq_len,
            )?);
        }
        let attention_locality_caches =
            vec![RamaAttentionLocalityCache::default(); prepared.layers.len()];

        Ok(Self {
            model,
            prepared: prepared.clone(),
            hidden_size,
            intermediate_size,
            head_dim,
            vocab_size,
            embedding_data,
            layer_norms,
            caches,
            last_phase_timings: None,
            rolling_executor: RollingExecutor::from_env(
                crate::streaming::streaming_available_threads(),
            ),
            last_rolling_stats: None,
            experimental_speed_config: RamaExperimentalSpeedConfig::from_env(),
            exact_prefill: parse_aip_exact_prefill_enabled(
                std::env::var(RLLM_AIP_EXACT_PREFILL_ENV).ok().as_deref(),
            ),
            lm_head_exact_every: parse_aip_lm_head_exact_every(
                std::env::var(RLLM_AIP_LM_HEAD_EXACT_EVERY_ENV)
                    .ok()
                    .as_deref(),
            ),
            last_experimental_speed_stats: None,
            sparse_column_cache: SparseColumnCache::from_env(),
            attention_locality_caches,
            collect_transformer_detail_timing: false,
            last_generated_token: None,
            last_generated_token_run: 0,
            generated_token_count_in_turn: 0,
            lm_head_repeat_margin_state: LmHeadRepeatMarginState::default(),
            lm_head_phrase_novelty_state: LmHeadPhraseNoveltyState::default(),
        })
    }

    pub fn set_transformer_detail_timing(&mut self, enabled: bool) {
        self.collect_transformer_detail_timing = enabled;
    }

    fn record_generated_token(&mut self, token_id: usize, reset_run: bool) {
        if reset_run {
            self.lm_head_phrase_novelty_state.reset();
            self.generated_token_count_in_turn = 0;
        }
        if !reset_run && self.last_generated_token == Some(token_id) {
            self.last_generated_token_run = self.last_generated_token_run.saturating_add(1);
        } else {
            self.last_generated_token = Some(token_id);
            self.last_generated_token_run = 1;
        }
        self.generated_token_count_in_turn = self.generated_token_count_in_turn.saturating_add(1);
        self.lm_head_phrase_novelty_state.push(token_id);
    }

    #[cfg(test)]
    pub(crate) fn enable_rolling_executor_for_test(
        &mut self,
        worker_count: usize,
        min_rows_per_worker: usize,
    ) {
        self.rolling_executor = Some(RollingExecutor::new(RollingExecutorConfig {
            enabled: true,
            worker_count,
            min_rows_per_worker,
        }));
    }

    #[cfg(test)]
    pub(crate) fn enable_experimental_speed_for_test(
        &mut self,
        config: RamaExperimentalSpeedConfig,
    ) {
        self.experimental_speed_config = config;
    }

    #[cfg(test)]
    pub(crate) fn enable_exact_prefill_for_test(&mut self, enabled: bool) {
        self.exact_prefill = enabled;
    }

    fn append_tokens_inner(
        &mut self,
        tokens: &[usize],
        budget: &mut MemoryBudget,
        emit_logits: bool,
    ) -> Result<Option<RamaSessionStep>> {
        if tokens.is_empty() {
            return Err(RuntimeError::InvalidTensorData(
                "llama session append requires at least one token".to_string(),
            ));
        }
        let seq_len = tokens.len();
        let position_offset = self.context_len();
        let projected_len = position_offset.checked_add(seq_len).ok_or_else(|| {
            RuntimeError::Shape("llama session context length overflow".to_string())
        })?;
        if projected_len > self.max_seq_len() {
            return Err(RuntimeError::Shape(format!(
                "llama session context would reach {projected_len}, max_seq_len {}",
                self.max_seq_len()
            )));
        }
        let is_decode_step =
            tokens.len() == 1 && self.last_generated_token == tokens.first().copied();
        let experimental_speed_config = if self.exact_prefill && emit_logits && !is_decode_step {
            RamaExperimentalSpeedConfig::disabled()
        } else {
            self.experimental_speed_config
        };
        if !is_decode_step {
            for cache in &mut self.attention_locality_caches {
                cache.clear();
            }
        }

        let mut phase_timings = RamaSessionPhaseTimings::default();
        let mut experimental_speed_stats = RamaExperimentalSpeedStats::default();
        let phase_start = Instant::now();
        let mut hidden = embedding_lookup(
            &self.embedding_data,
            self.vocab_size,
            self.hidden_size,
            tokens,
        )?;
        phase_timings.embedding_ms += phase_start.elapsed().as_secs_f64() * 1000.0;

        let phase_start = Instant::now();
        for (i, layer_names) in self.prepared.layers.iter().enumerate() {
            let config = LlamaStreamingBlockConfig {
                seq_len,
                hidden_size: self.hidden_size,
                q_heads: self.prepared.config.num_heads,
                kv_heads: self.prepared.config.num_key_value_heads,
                head_dim: self.head_dim,
                intermediate_size: self.intermediate_size,
                rms_norm_eps: self.prepared.config.rms_norm_eps,
                rope_theta: self.prepared.config.rope_theta,
                causal: self.prepared.config.causal,
                position_offset,
                layer_index: i,
                total_layers: self.prepared.layers.len(),
                experimental_speed: experimental_speed_config,
            };
            let mut transformer_detail = RamaTransformerPhaseTimings::default();
            let transformer_detail_timing = if self.collect_transformer_detail_timing {
                Some(&mut transformer_detail)
            } else {
                None
            };
            let experimental_stats_ref = if experimental_speed_config.enabled {
                Some(&mut experimental_speed_stats)
            } else {
                None
            };
            let sparse_column_cache = if experimental_speed_config.aip_column_cache {
                Some(&mut self.sparse_column_cache)
            } else {
                None
            };
            let attention_locality_cache = if experimental_speed_config
                .attention_locality_enabled_for_layer(i, self.prepared.layers.len())
            {
                Some(&mut self.attention_locality_caches[i])
            } else {
                None
            };
            hidden = streaming_llama_transformer_block_with_timing(
                self.model,
                &hidden,
                layer_names,
                &self.layer_norms[i],
                config,
                budget,
                Some(&mut self.caches[i]),
                transformer_detail_timing,
                experimental_stats_ref,
                sparse_column_cache,
                attention_locality_cache,
            )?;
            if self.collect_transformer_detail_timing {
                phase_timings
                    .transformer_detail
                    .add_assign(transformer_detail);
            }
        }
        phase_timings.transformer_ms += phase_start.elapsed().as_secs_f64() * 1000.0;

        if !emit_logits {
            self.last_phase_timings = Some(phase_timings);
            self.last_experimental_speed_stats = Some(experimental_speed_stats);
            return Ok(None);
        }

        let phase_start = Instant::now();
        let hidden = rms_norm(
            &hidden,
            &self.prepared.final_layernorm_weight,
            seq_len,
            self.hidden_size,
            self.prepared.config.rms_norm_eps,
        )?;
        phase_timings.final_norm_ms += phase_start.elapsed().as_secs_f64() * 1000.0;

        let phase_start = Instant::now();
        let last_hidden = &hidden[(seq_len - 1) * self.hidden_size..];
        let previous_token_run = if is_decode_step {
            self.last_generated_token_run
        } else {
            self.lm_head_repeat_margin_state.reset();
            self.lm_head_phrase_novelty_state.reset();
            0
        };
        let lm_head_config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: self.hidden_size,
                out_features: self.vocab_size,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        };
        let (token_id, logits) = match self.prepared.config.sampling {
            crate::StreamingSamplingConfig::Argmax => {
                let token_id = if let Some(prefix_rows) =
                    experimental_speed_config.lm_head_prefix_rows(self.vocab_size)
                {
                    experimental_speed_stats
                        .record_aip_policy(experimental_speed_config.aip_policy);
                    experimental_speed_stats.record_lm_head_prefix(prefix_rows, self.vocab_size);
                    streaming_tile_linear_argmax_prefix_from_model(
                        self.model,
                        &self.prepared.lm_head_weight,
                        last_hidden,
                        None,
                        lm_head_config,
                        prefix_rows,
                        budget,
                    )?
                } else if experimental_speed_config.enabled
                    && experimental_speed_config.aip_input_tiles
                {
                    experimental_speed_stats
                        .record_aip_policy(experimental_speed_config.aip_policy);
                    let sparse_config = RamaExperimentalSpeedConfig {
                        enabled: true,
                        aip_policy: experimental_speed_config.aip_policy,
                        aip_topk: Some(
                            experimental_speed_config.lm_head_topk_for_input(self.hidden_size, 128),
                        ),
                        aip_attention_topk: None,
                        aip_attention_locality_window: None,
                        aip_attention_locality_extra: None,
                        aip_mlp_topk: None,
                        aip_down_topk: None,
                        aip_edge_layers: None,
                        aip_edge_topk: None,
                        aip_exact_edge_layers: None,
                        aip_exact_edge_projection: None,
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
                        aip_repeat_run_limit: None,
                    };
                    match streaming_input_tiled_sparse_tile_linear_from_model(
                        self.model,
                        &self.prepared.lm_head_weight,
                        last_hidden,
                        None,
                        lm_head_config,
                        sparse_config,
                        &mut experimental_speed_stats,
                        budget,
                    )? {
                        Some(logits) => {
                            let sparse_lm_head_topk = sparse_config.aip_topk.unwrap_or(0);
                            let selected_token_id = if let Some(candidates) =
                                sparse_lm_head_rescore_candidates(
                                    &logits,
                                    tokens,
                                    experimental_speed_config,
                                    &mut experimental_speed_stats,
                                )? {
                                match streaming_tile_linear_argmax_candidate_rows_range_from_model(
                                    self.model,
                                    &self.prepared.lm_head_weight,
                                    last_hidden,
                                    None,
                                    lm_head_config,
                                    &candidates,
                                    budget,
                                )? {
                                    Some(token_id) => apply_rescored_lm_head_controllers(
                                        &logits,
                                        token_id,
                                        tokens,
                                        previous_token_run,
                                        experimental_speed_config,
                                        &mut experimental_speed_stats,
                                        &mut self.lm_head_repeat_margin_state,
                                        &mut self.lm_head_phrase_novelty_state,
                                    )?,
                                    None => sample_sparse_lm_head_argmax_with_controller_state(
                                        &logits,
                                        tokens,
                                        previous_token_run,
                                        experimental_speed_config,
                                        &mut experimental_speed_stats,
                                        &mut self.lm_head_repeat_margin_state,
                                        &mut self.lm_head_phrase_novelty_state,
                                    )?,
                                }
                            } else {
                                sample_sparse_lm_head_argmax_with_controller_state(
                                    &logits,
                                    tokens,
                                    previous_token_run,
                                    experimental_speed_config,
                                    &mut experimental_speed_stats,
                                    &mut self.lm_head_repeat_margin_state,
                                    &mut self.lm_head_phrase_novelty_state,
                                )?
                            };
                            let exact_check_due = lm_head_exact_check_due(
                                self.lm_head_exact_every,
                                is_decode_step,
                                self.generated_token_count_in_turn,
                            );
                            if exact_check_due || experimental_speed_config.aip_lm_head_agreement {
                                let exact_token_id = streaming_tile_linear_argmax_from_model(
                                    self.model,
                                    &self.prepared.lm_head_weight,
                                    last_hidden,
                                    None,
                                    lm_head_config,
                                    budget,
                                )?;
                                if exact_check_due {
                                    experimental_speed_stats.record_lm_head_exact_check(
                                        selected_token_id != exact_token_id,
                                    );
                                    if experimental_speed_config.aip_lm_head_agreement {
                                        record_sparse_lm_head_agreement_sample(
                                            &mut experimental_speed_stats,
                                            &logits,
                                            selected_token_id,
                                            exact_token_id,
                                            sparse_lm_head_topk,
                                        )?;
                                    }
                                    exact_token_id
                                } else {
                                    record_sparse_lm_head_agreement_sample(
                                        &mut experimental_speed_stats,
                                        &logits,
                                        selected_token_id,
                                        exact_token_id,
                                        sparse_lm_head_topk,
                                    )?;
                                    selected_token_id
                                }
                            } else {
                                selected_token_id
                            }
                        }
                        None => {
                            if let Some(executor) = self.rolling_executor.as_mut() {
                                let token = streaming_tile_linear_argmax_with_rolling_from_model(
                                    self.model,
                                    &self.prepared.lm_head_weight,
                                    last_hidden,
                                    None,
                                    lm_head_config,
                                    budget,
                                    Some(executor),
                                )?;
                                self.last_rolling_stats = Some(executor.take_stats());
                                token
                            } else {
                                streaming_tile_linear_argmax_from_model(
                                    self.model,
                                    &self.prepared.lm_head_weight,
                                    last_hidden,
                                    None,
                                    lm_head_config,
                                    budget,
                                )?
                            }
                        }
                    }
                } else if let Some(executor) = self.rolling_executor.as_mut() {
                    let token = streaming_tile_linear_argmax_with_rolling_from_model(
                        self.model,
                        &self.prepared.lm_head_weight,
                        last_hidden,
                        None,
                        lm_head_config,
                        budget,
                        Some(executor),
                    )?;
                    self.last_rolling_stats = Some(executor.take_stats());
                    token
                } else {
                    streaming_tile_linear_argmax_from_model(
                        self.model,
                        &self.prepared.lm_head_weight,
                        last_hidden,
                        None,
                        lm_head_config,
                        budget,
                    )?
                };
                (token_id, None)
            }
            crate::StreamingSamplingConfig::TopP {
                temperature,
                top_p,
                seed,
            } => {
                let logits = streaming_tile_linear_from_model(
                    self.model,
                    &self.prepared.lm_head_weight,
                    last_hidden,
                    None,
                    lm_head_config,
                    budget,
                )?;
                let token_id = sample_top_p(&logits, temperature, top_p, seed)?;
                (token_id, Some(logits))
            }
        };
        phase_timings.lm_head_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
        self.last_phase_timings = Some(phase_timings);
        self.last_experimental_speed_stats = Some(experimental_speed_stats);
        self.record_generated_token(token_id, !is_decode_step);
        Ok(Some(RamaSessionStep {
            token_id,
            logits,
            cached_context_len_after: self.context_len(),
        }))
    }
}

impl RamaSessionAdapter for LlamaRamaSessionAdapter<'_> {
    fn context_len(&self) -> usize {
        self.caches.first().map(KvCache::len).unwrap_or(0)
    }

    fn max_seq_len(&self) -> usize {
        self.prepared.config.max_seq_len.unwrap_or(0)
    }

    fn context_memory_bytes(&self) -> usize {
        self.caches.iter().map(KvCache::resident_bytes).sum()
    }

    fn append_tokens(
        &mut self,
        tokens: &[usize],
        budget: &mut MemoryBudget,
        emit_logits: bool,
    ) -> Result<Option<RamaSessionStep>> {
        self.last_phase_timings = None;
        self.last_experimental_speed_stats = None;
        let old_lens: Vec<usize> = self.caches.iter().map(KvCache::len).collect();
        let old_last_generated_token = self.last_generated_token;
        let old_last_generated_token_run = self.last_generated_token_run;
        let old_generated_token_count_in_turn = self.generated_token_count_in_turn;
        match self.append_tokens_inner(tokens, budget, emit_logits) {
            Ok(step) => Ok(step),
            Err(error) => {
                for (cache, len) in self.caches.iter_mut().zip(old_lens) {
                    let _ = cache.truncate(len);
                }
                self.last_phase_timings = None;
                self.last_experimental_speed_stats = None;
                self.last_generated_token = old_last_generated_token;
                self.last_generated_token_run = old_last_generated_token_run;
                self.generated_token_count_in_turn = old_generated_token_count_in_turn;
                Err(error)
            }
        }
    }

    fn take_last_phase_timings(&mut self) -> Option<RamaSessionPhaseTimings> {
        self.last_phase_timings.take()
    }

    fn take_last_rolling_stats(&mut self) -> Option<RamaRollingStats> {
        self.last_rolling_stats.take()
    }

    fn take_last_experimental_speed_stats(&mut self) -> Option<RamaExperimentalSpeedStats> {
        self.last_experimental_speed_stats.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::llama::model::{LlamaRamaBuildConfig, OwnedLlamaStreamingBlockTensorNames};
    use crate::{RamaSessionAdapter, StreamingSamplingConfig};
    use rllm_container::{DType, GlobalMetadata, ModelConfigMetadata, RllmWriter, TensorMeta};
    use sha2::{Digest, Sha256};

    const VOCAB_SIZE: usize = 3;
    const HIDDEN_SIZE: usize = 2;
    const INTERMEDIATE_SIZE: usize = 3;

    fn sha256_array(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn bf16_bytes(values: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rllm-llama-session-{name}-{}.rllm",
            std::process::id()
        ))
    }

    fn add_f32_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        shape: Vec<u64>,
        values: &[f32],
    ) {
        let bytes = f32_bytes(values);
        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape,
            dtype: DType::Fp32,
            original_size_bytes: bytes.len() as u64,
            compressed_size_bytes: bytes.len() as u64,
            original_sha256: sha256_array(&bytes),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(tensor_id, "rtc-raw-v1", &bytes, &bytes, 0)
            .unwrap();
    }

    fn add_bf16_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        shape: Vec<u64>,
        values: &[u16],
    ) {
        let bytes = bf16_bytes(values);
        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape,
            dtype: DType::Bf16,
            original_size_bytes: bytes.len() as u64,
            compressed_size_bytes: bytes.len() as u64,
            original_sha256: sha256_array(&bytes),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(tensor_id, "rtc-raw-v1", &bytes, &bytes, 0)
            .unwrap();
    }

    fn llama_metadata() -> GlobalMetadata {
        let mut metadata = GlobalMetadata::new_test();
        metadata.model_config = Some(ModelConfigMetadata {
            architecture_type: Some("llama".to_string()),
            hidden_size: Some(HIDDEN_SIZE as u64),
            intermediate_size: Some(INTERMEDIATE_SIZE as u64),
            num_attention_heads: Some(1),
            num_key_value_heads: Some(1),
            max_position_embeddings: Some(8),
            rms_norm_eps: Some(1e-5),
            rope_theta: Some(10_000.0),
            vocab_size: Some(VOCAB_SIZE as u64),
            ..Default::default()
        });
        metadata
    }

    fn llama_metadata_with_vocab(vocab_size: usize) -> GlobalMetadata {
        let mut metadata = llama_metadata();
        metadata.model_config.as_mut().unwrap().vocab_size = Some(vocab_size as u64);
        metadata
    }

    fn layer_names(layer_idx: usize) -> OwnedLlamaStreamingBlockTensorNames {
        OwnedLlamaStreamingBlockTensorNames {
            q_weight: format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
            k_weight: format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
            v_weight: format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
            o_weight: format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
            gate_weight: format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
            up_weight: format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
            down_weight: format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
        }
    }

    fn prepared_with_layers(layer_count: usize) -> LayerDecodedLlamaRamaTransformer {
        LayerDecodedLlamaRamaTransformer {
            config: LlamaRamaBuildConfig {
                max_new_tokens: 1,
                max_seq_len: Some(8),
                num_heads: 1,
                num_key_value_heads: 1,
                causal: true,
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
                sampling: StreamingSamplingConfig::Argmax,
            },
            embedding_weight: "model.embed_tokens.weight".to_string(),
            layers: (0..layer_count).map(layer_names).collect(),
            lm_head_weight: "lm_head.weight".to_string(),
            final_layernorm_weight: vec![1.0, 1.0],
            pinned_lm_head_weight: None,
            resident_parameter_bytes: 0,
            max_layer_parameter_bytes: 0,
        }
    }

    fn add_base_tensors(writer: &mut RllmWriter, tensor_id: &mut u64, lm_head_shape: Vec<u64>) {
        add_f32_tensor(
            writer,
            *tensor_id,
            "model.embed_tokens.weight",
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
            &[0.5, -1.0, 1.25, 0.75, -0.5, 0.25],
        );
        *tensor_id += 1;

        let lm_head_values = vec![0.25; lm_head_shape.iter().product::<u64>() as usize];
        add_f32_tensor(
            writer,
            *tensor_id,
            "lm_head.weight",
            lm_head_shape,
            &lm_head_values,
        );
        *tensor_id += 1;
    }

    fn add_layer_norms(writer: &mut RllmWriter, tensor_id: &mut u64, layer_idx: usize) {
        add_f32_tensor(
            writer,
            *tensor_id,
            &format!("model.layers.{layer_idx}.input_layernorm.weight"),
            vec![HIDDEN_SIZE as u64],
            &[1.0, 1.0],
        );
        *tensor_id += 1;
        add_f32_tensor(
            writer,
            *tensor_id,
            &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
            vec![HIDDEN_SIZE as u64],
            &[1.0, 1.0],
        );
        *tensor_id += 1;
    }

    fn zero_values(shape: &[u64]) -> Vec<f32> {
        vec![0.0; shape.iter().product::<u64>() as usize]
    }

    fn add_zero_f32_tensor(
        writer: &mut RllmWriter,
        tensor_id: &mut u64,
        name: &str,
        shape: Vec<u64>,
    ) {
        let values = zero_values(&shape);
        add_f32_tensor(writer, *tensor_id, name, shape, &values);
        *tensor_id += 1;
    }

    fn add_layer_projection_tensors(
        writer: &mut RllmWriter,
        tensor_id: &mut u64,
        layer_idx: usize,
        o_shape: Vec<u64>,
        down_shape: Vec<u64>,
        short_q_data: bool,
    ) {
        let prefix = format!("model.layers.{layer_idx}");
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        let hidden_square = vec![hidden, hidden];

        if short_q_data {
            add_f32_tensor(
                writer,
                *tensor_id,
                &format!("{prefix}.self_attn.q_proj.weight"),
                hidden_square.clone(),
                &[0.0],
            );
            *tensor_id += 1;
        } else {
            add_zero_f32_tensor(
                writer,
                tensor_id,
                &format!("{prefix}.self_attn.q_proj.weight"),
                hidden_square.clone(),
            );
        }
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.self_attn.k_proj.weight"),
            hidden_square.clone(),
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.self_attn.v_proj.weight"),
            hidden_square.clone(),
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.self_attn.o_proj.weight"),
            o_shape,
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.mlp.down_proj.weight"),
            down_shape,
        );
    }

    fn add_complete_layer(writer: &mut RllmWriter, tensor_id: &mut u64, layer_idx: usize) {
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        add_layer_norms(writer, tensor_id, layer_idx);
        add_layer_projection_tensors(
            writer,
            tensor_id,
            layer_idx,
            vec![hidden, hidden],
            vec![hidden, intermediate],
            false,
        );
    }

    fn add_layer_with_bad_o_projection(
        writer: &mut RllmWriter,
        tensor_id: &mut u64,
        layer_idx: usize,
    ) {
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        add_layer_norms(writer, tensor_id, layer_idx);
        add_layer_projection_tensors(
            writer,
            tensor_id,
            layer_idx,
            vec![hidden - 1, hidden],
            vec![hidden, intermediate],
            false,
        );
    }

    fn add_layer_with_bad_down_projection(
        writer: &mut RllmWriter,
        tensor_id: &mut u64,
        layer_idx: usize,
    ) {
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        add_layer_norms(writer, tensor_id, layer_idx);
        add_layer_projection_tensors(
            writer,
            tensor_id,
            layer_idx,
            vec![hidden, hidden],
            vec![hidden, intermediate - 1],
            false,
        );
    }

    fn add_layer_with_runtime_q_failure(
        writer: &mut RllmWriter,
        tensor_id: &mut u64,
        layer_idx: usize,
    ) {
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        add_layer_norms(writer, tensor_id, layer_idx);
        add_layer_projection_tensors(
            writer,
            tensor_id,
            layer_idx,
            vec![hidden, hidden],
            vec![hidden, intermediate],
            true,
        );
    }

    fn write_bad_attention_projection_model(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(
            &mut writer,
            &mut tensor_id,
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_layer_with_bad_o_projection(&mut writer, &mut tensor_id, 0);
        writer.finalize().unwrap();
    }

    fn write_bad_mlp_projection_model(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(
            &mut writer,
            &mut tensor_id,
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_layer_with_bad_down_projection(&mut writer, &mut tensor_id, 0);
        writer.finalize().unwrap();
    }

    fn add_complete_layer_zero(writer: &mut RllmWriter, tensor_id: &mut u64) {
        add_complete_layer(writer, tensor_id, 0);
    }

    fn write_constructor_model(path: &std::path::Path, lm_head_shape: Vec<u64>) {
        let mut writer = RllmWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(&mut writer, &mut tensor_id, lm_head_shape);
        add_layer_norms(&mut writer, &mut tensor_id, 0);
        writer.finalize().unwrap();
    }

    fn write_bf16_lm_head_model(path: &std::path::Path, vocab_size: usize) {
        let mut writer = RllmWriter::new(path, llama_metadata_with_vocab(vocab_size)).unwrap();
        let mut tensor_id = 0u64;
        add_f32_tensor(
            &mut writer,
            tensor_id,
            "model.embed_tokens.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0.0; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            "lm_head.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0x0000; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_complete_layer_zero(&mut writer, &mut tensor_id);
        writer.finalize().unwrap();
    }

    fn write_bf16_mlp_speed_model(path: &std::path::Path, vocab_size: usize) {
        let mut writer = RllmWriter::new(path, llama_metadata_with_vocab(vocab_size)).unwrap();
        let mut tensor_id = 0u64;
        add_f32_tensor(
            &mut writer,
            tensor_id,
            "model.embed_tokens.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0.0; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            "lm_head.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0x0000; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_layer_norms(&mut writer, &mut tensor_id, 0);
        let prefix = "model.layers.0";
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.q_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.k_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.v_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
            &[0x0000; HIDDEN_SIZE * INTERMEDIATE_SIZE],
        );
        writer.finalize().unwrap();
    }

    fn write_bf16_mlp_speed_model_with_layers(
        path: &std::path::Path,
        vocab_size: usize,
        layer_count: usize,
    ) {
        let mut writer = RllmWriter::new(path, llama_metadata_with_vocab(vocab_size)).unwrap();
        let mut tensor_id = 0u64;
        add_f32_tensor(
            &mut writer,
            tensor_id,
            "model.embed_tokens.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0.0; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            "lm_head.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0x0000; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        for layer_idx in 0..layer_count {
            add_layer_norms(&mut writer, &mut tensor_id, layer_idx);
            let prefix = format!("model.layers.{layer_idx}");
            add_zero_f32_tensor(
                &mut writer,
                &mut tensor_id,
                &format!("{prefix}.self_attn.q_proj.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            add_zero_f32_tensor(
                &mut writer,
                &mut tensor_id,
                &format!("{prefix}.self_attn.k_proj.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            add_zero_f32_tensor(
                &mut writer,
                &mut tensor_id,
                &format!("{prefix}.self_attn.v_proj.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            add_zero_f32_tensor(
                &mut writer,
                &mut tensor_id,
                &format!("{prefix}.self_attn.o_proj.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            add_bf16_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.gate_proj.weight"),
                vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
                &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
            );
            tensor_id += 1;
            add_bf16_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.up_proj.weight"),
                vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
                &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
            );
            tensor_id += 1;
            add_bf16_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.down_proj.weight"),
                vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
                &[0x0000; HIDDEN_SIZE * INTERMEDIATE_SIZE],
            );
            tensor_id += 1;
        }
        writer.finalize().unwrap();
    }

    fn write_post_cache_failure_model(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(
            &mut writer,
            &mut tensor_id,
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_complete_layer_zero(&mut writer, &mut tensor_id);
        add_layer_with_runtime_q_failure(&mut writer, &mut tensor_id, 1);
        writer.finalize().unwrap();
    }

    #[test]
    fn llama_session_new_rejects_empty_prepared_layers() {
        let path = temp_path("empty-prepared-layers");
        write_constructor_model(&path, vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64]);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(0);
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("at least one layer")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_new_rejects_malformed_lm_head_shape() {
        let path = temp_path("malformed-lm-head");
        write_constructor_model(&path, vec![(VOCAB_SIZE - 1) as u64, HIDDEN_SIZE as u64]);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("lm_head.weight")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_new_rejects_malformed_final_layernorm_shape() {
        let path = temp_path("malformed-final-layernorm");
        write_constructor_model(&path, vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64]);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut prepared = prepared_with_layers(1);
        prepared.final_layernorm_weight = vec![1.0];
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("final_layernorm_weight")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_new_rejects_malformed_attention_projection_shape() {
        let path = temp_path("malformed-attention-projection");
        write_bad_attention_projection_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("self_attn.o_proj.weight")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_new_rejects_malformed_mlp_projection_shape() {
        let path = temp_path("malformed-mlp-projection");
        write_bad_mlp_projection_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("mlp.down_proj.weight")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_append_rolls_back_all_caches_after_post_cache_layer_failure() {
        let path = temp_path("rollback-post-cache-failure");
        write_post_cache_failure_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(2);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();

        let result = adapter.append_tokens(&[0], &mut budget, false);

        assert!(result.is_err());
        assert_eq!(adapter.context_len(), 0);
        assert_eq!(adapter.context_memory_bytes(), 0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_records_phase_timings_for_logits_append() {
        let path = temp_path("phase-timing-logits");
        write_post_cache_failure_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
        adapter.set_transformer_detail_timing(true);

        let step = adapter.append_tokens(&[0], &mut budget, true).unwrap();
        let timings = adapter.take_last_phase_timings().unwrap();

        assert!(step.is_some());
        assert!(timings.embedding_ms >= 0.0);
        assert!(timings.transformer_ms >= 0.0);
        assert_eq!(timings.transformer_detail.profiled_layers, 1);
        assert!(timings.transformer_detail.attention_total_ms() >= 0.0);
        assert!(timings.transformer_detail.mlp_total_ms() >= 0.0);
        assert!(timings.final_norm_ms >= 0.0);
        assert!(timings.lm_head_ms >= 0.0);
        assert!(timings.total_ms() >= 0.0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_reports_rolling_stats_when_executor_is_enabled() {
        let path = temp_path("llama-session-rolling");
        write_bf16_lm_head_model(&path, 8);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
        adapter.enable_rolling_executor_for_test(4, 1);

        let _ = adapter.append_tokens(&[1], &mut budget, true).unwrap();
        let stats = adapter.take_last_rolling_stats().unwrap();

        assert!(stats.submitted_tasks > 0);
        assert!(stats.worker_wakeups > 0);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_reports_experimental_speed_stats_when_enabled_for_test() {
        let path = temp_path("experimental-speed-stats");
        write_bf16_mlp_speed_model(&path, 8);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
        adapter.enable_experimental_speed_for_test(crate::RamaExperimentalSpeedConfig {
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
            aip_exact_edge_projection: None,
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

        adapter.append_tokens(&[0], &mut budget, true).unwrap();
        let stats = adapter.take_last_experimental_speed_stats().unwrap();

        assert!(stats.sparse_projection_calls > 0);
        assert!(stats.max_selected_topk <= 1);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_exact_prefill_keeps_decode_aip_enabled() {
        let path = temp_path("exact-prefill-speed-decode");
        write_bf16_mlp_speed_model(&path, 8);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
        adapter.enable_experimental_speed_for_test(crate::RamaExperimentalSpeedConfig {
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
            aip_exact_edge_projection: None,
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
        adapter.enable_exact_prefill_for_test(true);

        let first_step = adapter
            .append_tokens(&[0, 1], &mut budget, true)
            .unwrap()
            .unwrap();
        let prompt_stats = adapter.take_last_experimental_speed_stats().unwrap();
        assert_eq!(prompt_stats.sparse_projection_calls, 0);

        adapter
            .append_tokens(&[first_step.token_id], &mut budget, true)
            .unwrap();
        let decode_stats = adapter.take_last_experimental_speed_stats().unwrap();
        assert!(decode_stats.sparse_projection_calls > 0);
        assert_eq!(
            decode_stats.aip_policy,
            Some(crate::RamaAipPolicyKind::Speed)
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn lm_head_exact_periodic_check_targets_decode_token_indices() {
        assert!(!lm_head_exact_check_due(None, true, 3));
        assert!(!lm_head_exact_check_due(Some(4), false, 3));
        assert!(!lm_head_exact_check_due(Some(4), true, 2));
        assert!(lm_head_exact_check_due(Some(4), true, 3));
        assert!(lm_head_exact_check_due(Some(1), true, 0));
    }

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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
            aip_exact_edge_projection: None,
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
}
