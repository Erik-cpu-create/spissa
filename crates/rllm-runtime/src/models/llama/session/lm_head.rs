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

#[derive(Debug, Clone)]
struct LlamaLayerDriftProbeOutput {
    hidden: Vec<f32>,
    token_id: usize,
    exact_margin_milli: usize,
}

fn positive_f64_to_milli(value: f64) -> usize {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let milli = (value * 1000.0).round();
    if milli >= usize::MAX as f64 {
        usize::MAX
    } else {
        milli as usize
    }
}

fn hidden_l2_milli(lhs: &[f32], rhs: &[f32]) -> usize {
    let sum_sq = lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| {
            let delta = *left as f64 - *right as f64;
            delta * delta
        })
        .sum::<f64>();
    positive_f64_to_milli(sum_sq.sqrt())
}

fn hidden_cosine_gap_milli(lhs: &[f32], rhs: &[f32]) -> usize {
    let (dot, lhs_sq, rhs_sq) =
        lhs.iter()
            .zip(rhs)
            .fold((0.0f64, 0.0f64, 0.0f64), |acc, (left, right)| {
                let left = *left as f64;
                let right = *right as f64;
                (
                    acc.0 + left * right,
                    acc.1 + left * left,
                    acc.2 + right * right,
                )
            });
    if lhs_sq <= f64::EPSILON || rhs_sq <= f64::EPSILON {
        return 0;
    }
    let cosine = (dot / (lhs_sq.sqrt() * rhs_sq.sqrt())).clamp(-1.0, 1.0);
    positive_f64_to_milli(1.0 - cosine)
}

fn optional_vector_metrics(lhs: Option<&[f32]>, rhs: Option<&[f32]>) -> (usize, usize) {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => (hidden_l2_milli(lhs, rhs), hidden_cosine_gap_milli(lhs, rhs)),
        _ => (0, 0),
    }
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

