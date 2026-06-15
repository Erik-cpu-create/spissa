use crate::{MemoryBudget, RamaExperimentalSpeedStats, Result, RuntimeError};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct RamaSessionStep {
    pub token_id: usize,
    pub logits: Option<Vec<f32>>,
    pub cached_context_len_after: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RamaTransformerPhaseTimings {
    pub attention_norm_ms: f64,
    pub q_projection_ms: f64,
    pub k_projection_ms: f64,
    pub v_projection_ms: f64,
    pub rotary_ms: f64,
    pub attention_ms: f64,
    pub kv_append_ms: f64,
    pub o_projection_ms: f64,
    pub attention_residual_ms: f64,
    pub mlp_norm_ms: f64,
    pub gate_projection_ms: f64,
    pub up_projection_ms: f64,
    pub activation_multiply_ms: f64,
    pub down_projection_ms: f64,
    pub mlp_residual_ms: f64,
    pub profiled_layers: usize,
}

impl RamaTransformerPhaseTimings {
    pub fn add_assign(&mut self, other: RamaTransformerPhaseTimings) {
        self.attention_norm_ms += other.attention_norm_ms;
        self.q_projection_ms += other.q_projection_ms;
        self.k_projection_ms += other.k_projection_ms;
        self.v_projection_ms += other.v_projection_ms;
        self.rotary_ms += other.rotary_ms;
        self.attention_ms += other.attention_ms;
        self.kv_append_ms += other.kv_append_ms;
        self.o_projection_ms += other.o_projection_ms;
        self.attention_residual_ms += other.attention_residual_ms;
        self.mlp_norm_ms += other.mlp_norm_ms;
        self.gate_projection_ms += other.gate_projection_ms;
        self.up_projection_ms += other.up_projection_ms;
        self.activation_multiply_ms += other.activation_multiply_ms;
        self.down_projection_ms += other.down_projection_ms;
        self.mlp_residual_ms += other.mlp_residual_ms;
        self.profiled_layers = self.profiled_layers.saturating_add(other.profiled_layers);
    }

    pub fn attention_total_ms(&self) -> f64 {
        self.attention_norm_ms
            + self.q_projection_ms
            + self.k_projection_ms
            + self.v_projection_ms
            + self.rotary_ms
            + self.attention_ms
            + self.kv_append_ms
            + self.o_projection_ms
            + self.attention_residual_ms
    }

    pub fn mlp_total_ms(&self) -> f64 {
        self.mlp_norm_ms
            + self.gate_projection_ms
            + self.up_projection_ms
            + self.activation_multiply_ms
            + self.down_projection_ms
            + self.mlp_residual_ms
    }

    pub fn total_ms(&self) -> f64 {
        self.attention_total_ms() + self.mlp_total_ms()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RamaSessionPhaseTimings {
    pub embedding_ms: f64,
    pub transformer_ms: f64,
    pub transformer_detail: RamaTransformerPhaseTimings,
    pub final_norm_ms: f64,
    pub lm_head_ms: f64,
}

impl RamaSessionPhaseTimings {
    pub fn add_assign(&mut self, other: RamaSessionPhaseTimings) {
        self.embedding_ms += other.embedding_ms;
        self.transformer_ms += other.transformer_ms;
        self.transformer_detail.add_assign(other.transformer_detail);
        self.final_norm_ms += other.final_norm_ms;
        self.lm_head_ms += other.lm_head_ms;
    }

    pub fn total_ms(&self) -> f64 {
        self.embedding_ms + self.transformer_ms + self.final_norm_ms + self.lm_head_ms
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaRollingStats {
    pub submitted_tasks: usize,
    pub worker_wakeups: usize,
    pub sequential_fallbacks: usize,
    pub peak_scratch_bytes: usize,
}

impl RamaRollingStats {
    pub fn add_assign(&mut self, other: RamaRollingStats) {
        self.submitted_tasks = self.submitted_tasks.saturating_add(other.submitted_tasks);
        self.worker_wakeups = self.worker_wakeups.saturating_add(other.worker_wakeups);
        self.sequential_fallbacks = self
            .sequential_fallbacks
            .saturating_add(other.sequential_fallbacks);
        self.peak_scratch_bytes = self.peak_scratch_bytes.max(other.peak_scratch_bytes);
    }

    pub fn is_empty(self) -> bool {
        self.submitted_tasks == 0
            && self.worker_wakeups == 0
            && self.sequential_fallbacks == 0
            && self.peak_scratch_bytes == 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RamaRepetitionStats {
    pub generated_tokens: usize,
    pub unique_generated_tokens: usize,
    pub max_repeated_token_run: usize,
    pub repeated_token_ratio: f64,
}

impl RamaRepetitionStats {
    pub fn from_tokens(tokens: &[usize]) -> Self {
        if tokens.is_empty() {
            return Self::default();
        }

        let mut unique = std::collections::HashSet::new();
        let mut max_run = 1usize;
        let mut current_run = 1usize;
        let mut adjacent_repeats = 0usize;
        unique.insert(tokens[0]);

        for window in tokens.windows(2) {
            unique.insert(window[1]);
            if window[0] == window[1] {
                adjacent_repeats = adjacent_repeats.saturating_add(1);
                current_run = current_run.saturating_add(1);
                max_run = max_run.max(current_run);
            } else {
                current_run = 1;
            }
        }

        let denominator = tokens.len().saturating_sub(1);
        let repeated_token_ratio = if denominator == 0 {
            0.0
        } else {
            adjacent_repeats as f64 / denominator as f64
        };

        Self {
            generated_tokens: tokens.len(),
            unique_generated_tokens: unique.len(),
            max_repeated_token_run: max_run,
            repeated_token_ratio,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RamaSessionTurnMetrics {
    pub input_tokens: usize,
    pub generated_tokens: usize,
    pub new_prefill_tokens: usize,
    pub flushed_pending_tokens: usize,
    pub replayed_tokens: usize,
    pub ttft_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub end_to_end_ms: f64,
    pub decode_tok_s: f64,
    pub end_to_end_tok_s: f64,
    pub context_memory_bytes: usize,
    pub peak_transient_bytes: usize,
    pub rolling_stats: RamaRollingStats,
    pub experimental_speed_stats: RamaExperimentalSpeedStats,
    pub phase_timings: RamaSessionPhaseTimings,
    pub prefill_phase_timings: RamaSessionPhaseTimings,
    pub decode_phase_timings: RamaSessionPhaseTimings,
    pub repetition_stats: RamaRepetitionStats,
}

#[derive(Debug, Clone)]
pub struct RamaSessionTurnResult {
    pub input_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub token_history: Vec<usize>,
    pub metrics: RamaSessionTurnMetrics,
}

/// Model-specific adapter used by [`RamaChatSession`] to append tokens into a
/// cached runtime context.
///
/// Implementations must make each [`Self::append_tokens`] call transactional:
/// session retry safety depends on adapter state either advancing exactly as
/// specified for a successful append or being fully rolled back before return.
pub trait RamaSessionAdapter {
    fn context_len(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn context_memory_bytes(&self) -> usize;

    /// Append `tokens` to the adapter's cached context.
    ///
    /// On a protocol-valid success, the adapter context length must increase by
    /// exactly `tokens.len()` tokens. When `emit_logits` is `true`, this method
    /// must return `Ok(Some(step))`, where `step.token_id` is the sampled next
    /// token from the final appended token. When `emit_logits` is `false`, this
    /// method must return `Ok(None)`.
    ///
    /// Any `Err`, `emit_logits == true` returning `Ok(None)`, or
    /// `emit_logits == false` returning `Ok(Some(_))` must leave adapter state
    /// unchanged from the call boundary. Real adapters with multi-layer KV cache
    /// must checkpoint and roll back all internal cache writes before returning
    /// one of those failure or protocol-invalid outcomes.
    fn append_tokens(
        &mut self,
        tokens: &[usize],
        budget: &mut MemoryBudget,
        emit_logits: bool,
    ) -> Result<Option<RamaSessionStep>>;

    fn take_last_phase_timings(&mut self) -> Option<RamaSessionPhaseTimings> {
        None
    }

    fn take_last_rolling_stats(&mut self) -> Option<RamaRollingStats> {
        None
    }

    fn take_last_experimental_speed_stats(&mut self) -> Option<RamaExperimentalSpeedStats> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct RamaChatSession<A> {
    adapter: A,
    token_history: Vec<usize>,
    pending_uncached_token: Option<usize>,
}

impl<A: RamaSessionAdapter> RamaChatSession<A> {
    pub fn new(adapter: A) -> Self {
        Self {
            adapter,
            token_history: Vec::new(),
            pending_uncached_token: None,
        }
    }

    pub fn token_history(&self) -> &[usize] {
        &self.token_history
    }

    pub fn cached_context_len(&self) -> usize {
        self.adapter.context_len()
    }

    pub fn pending_uncached_token(&self) -> Option<usize> {
        self.pending_uncached_token
    }

    pub fn context_memory_bytes(&self) -> usize {
        self.adapter.context_memory_bytes()
    }

    pub fn adapter(&self) -> &A {
        &self.adapter
    }

    pub fn generate_turn(
        &mut self,
        user_token_ids: &[usize],
        max_new_tokens: usize,
        budget: &mut MemoryBudget,
        mut on_token: impl FnMut(usize) -> bool,
    ) -> Result<RamaSessionTurnResult> {
        if user_token_ids.is_empty() {
            return Err(RuntimeError::InvalidTensorData(
                "chat session turn requires at least one user token".to_string(),
            ));
        }
        if max_new_tokens == 0 {
            return Err(RuntimeError::InvalidTensorData(
                "chat session max_new_tokens must be greater than zero".to_string(),
            ));
        }
        let projected_visible_len = self
            .token_history
            .len()
            .checked_add(user_token_ids.len())
            .and_then(|value| value.checked_add(max_new_tokens))
            .ok_or_else(|| {
                RuntimeError::Shape("chat session context length overflow".to_string())
            })?;
        if projected_visible_len > self.adapter.max_seq_len() {
            return Err(RuntimeError::Shape(format!(
                "chat session context would reach {projected_visible_len} tokens, max_seq_len {}",
                self.adapter.max_seq_len()
            )));
        }

        budget.reset_peak();
        let turn_start = Instant::now();
        let mut flushed_pending_tokens = 0usize;
        let mut phase_timings = RamaSessionPhaseTimings::default();
        let mut prefill_phase_timings = RamaSessionPhaseTimings::default();
        let mut decode_phase_timings = RamaSessionPhaseTimings::default();
        let mut rolling_stats = RamaRollingStats::default();
        let mut experimental_speed_stats = RamaExperimentalSpeedStats::default();
        if let Some(token) = self.pending_uncached_token {
            let emitted = self.adapter.append_tokens(&[token], budget, false)?;
            if emitted.is_some() {
                return Err(RuntimeError::InvalidTensorData(
                    "chat session pending-token flush unexpectedly emitted logits".to_string(),
                ));
            }
            if let Some(timings) = self.adapter.take_last_phase_timings() {
                phase_timings.add_assign(timings);
            }
            if let Some(stats) = self.adapter.take_last_rolling_stats() {
                rolling_stats.add_assign(stats);
            }
            if let Some(stats) = self.adapter.take_last_experimental_speed_stats() {
                experimental_speed_stats.add_assign(stats);
            }
            self.pending_uncached_token = None;
            flushed_pending_tokens = 1;
        }

        let prefill_start = Instant::now();
        let first_step = self
            .adapter
            .append_tokens(user_token_ids, budget, true)?
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData(
                    "chat session prefill did not emit first token".to_string(),
                )
            })?;
        if let Some(timings) = self.adapter.take_last_phase_timings() {
            prefill_phase_timings.add_assign(timings);
            phase_timings.add_assign(timings);
        }
        if let Some(stats) = self.adapter.take_last_rolling_stats() {
            rolling_stats.add_assign(stats);
        }
        if let Some(stats) = self.adapter.take_last_experimental_speed_stats() {
            experimental_speed_stats.add_assign(stats);
        }
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = turn_start.elapsed().as_secs_f64() * 1000.0;

        self.token_history.extend_from_slice(user_token_ids);
        let mut generated_token_ids = vec![first_step.token_id];
        self.token_history.push(first_step.token_id);
        self.pending_uncached_token = Some(first_step.token_id);
        if !on_token(first_step.token_id) {
            let end_to_end_ms = turn_start.elapsed().as_secs_f64() * 1000.0;
            return Ok(self.turn_result(
                user_token_ids,
                generated_token_ids.clone(),
                RamaSessionTurnMetrics {
                    input_tokens: user_token_ids.len(),
                    generated_tokens: 1,
                    new_prefill_tokens: user_token_ids.len(),
                    flushed_pending_tokens,
                    replayed_tokens: 0,
                    ttft_ms,
                    prefill_ms,
                    decode_ms: 0.0,
                    end_to_end_ms,
                    decode_tok_s: 0.0,
                    end_to_end_tok_s: 1.0 / (end_to_end_ms / 1000.0).max(f64::EPSILON),
                    context_memory_bytes: self.adapter.context_memory_bytes(),
                    peak_transient_bytes: budget.peak_bytes(),
                    rolling_stats,
                    experimental_speed_stats,
                    phase_timings,
                    prefill_phase_timings,
                    decode_phase_timings,
                    repetition_stats: RamaRepetitionStats::from_tokens(&generated_token_ids),
                },
            ));
        }

        let decode_start = Instant::now();
        while generated_token_ids.len() < max_new_tokens {
            let previous = self.pending_uncached_token.ok_or_else(|| {
                RuntimeError::InvalidTensorData("chat session missing pending token".to_string())
            })?;
            let step = self
                .adapter
                .append_tokens(&[previous], budget, true)?
                .ok_or_else(|| {
                    RuntimeError::InvalidTensorData(
                        "chat session decode did not emit next token".to_string(),
                    )
                })?;
            if let Some(timings) = self.adapter.take_last_phase_timings() {
                decode_phase_timings.add_assign(timings);
                phase_timings.add_assign(timings);
            }
            if let Some(stats) = self.adapter.take_last_rolling_stats() {
                rolling_stats.add_assign(stats);
            }
            if let Some(stats) = self.adapter.take_last_experimental_speed_stats() {
                experimental_speed_stats.add_assign(stats);
            }
            generated_token_ids.push(step.token_id);
            self.token_history.push(step.token_id);
            self.pending_uncached_token = Some(step.token_id);
            if !on_token(step.token_id) {
                break;
            }
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let end_to_end_ms = turn_start.elapsed().as_secs_f64() * 1000.0;
        let decode_steps = generated_token_ids.len().saturating_sub(1);

        Ok(self.turn_result(
            user_token_ids,
            generated_token_ids.clone(),
            RamaSessionTurnMetrics {
                input_tokens: user_token_ids.len(),
                generated_tokens: generated_token_ids.len(),
                new_prefill_tokens: user_token_ids.len(),
                flushed_pending_tokens,
                replayed_tokens: 0,
                ttft_ms,
                prefill_ms,
                decode_ms,
                end_to_end_ms,
                decode_tok_s: if decode_steps == 0 {
                    0.0
                } else {
                    decode_steps as f64 / (decode_ms / 1000.0).max(f64::EPSILON)
                },
                end_to_end_tok_s: generated_token_ids.len() as f64
                    / (end_to_end_ms / 1000.0).max(f64::EPSILON),
                context_memory_bytes: self.adapter.context_memory_bytes(),
                peak_transient_bytes: budget.peak_bytes(),
                rolling_stats,
                experimental_speed_stats,
                phase_timings,
                prefill_phase_timings,
                decode_phase_timings,
                repetition_stats: RamaRepetitionStats::from_tokens(&generated_token_ids),
            },
        ))
    }

    fn turn_result(
        &self,
        input_token_ids: &[usize],
        generated_token_ids: Vec<usize>,
        metrics: RamaSessionTurnMetrics,
    ) -> RamaSessionTurnResult {
        RamaSessionTurnResult {
            input_token_ids: input_token_ids.to_vec(),
            generated_token_ids,
            token_history: self.token_history.clone(),
            metrics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct RecordingAdapter {
        max_seq_len: usize,
        context_len: usize,
        sample_base: usize,
        steps: Vec<Result<Option<RamaSessionStep>>>,
        appends: Vec<(Vec<usize>, bool)>,
        faults: Vec<AppendFault>,
        transient_bytes: usize,
        phase_timings: Vec<RamaSessionPhaseTimings>,
        rolling_stats: Vec<RamaRollingStats>,
        experimental_speed_stats: Vec<RamaExperimentalSpeedStats>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum AppendFaultKind {
        Error,
        None,
        UnexpectedLogits,
    }

    #[derive(Debug, Clone)]
    struct AppendFault {
        tokens: Vec<usize>,
        emit_logits: bool,
        kind: AppendFaultKind,
    }

    impl RecordingAdapter {
        fn new(max_seq_len: usize) -> Self {
            Self {
                max_seq_len,
                context_len: 0,
                sample_base: 100,
                steps: Vec::new(),
                appends: Vec::new(),
                faults: Vec::new(),
                transient_bytes: 0,
                phase_timings: Vec::new(),
                rolling_stats: Vec::new(),
                experimental_speed_stats: Vec::new(),
            }
        }

        fn with_fault(
            mut self,
            tokens: &[usize],
            emit_logits: bool,
            kind: AppendFaultKind,
        ) -> Self {
            self.faults.push(AppendFault {
                tokens: tokens.to_vec(),
                emit_logits,
                kind,
            });
            self
        }

        fn with_transient_bytes(mut self, transient_bytes: usize) -> Self {
            self.transient_bytes = transient_bytes;
            self
        }

        fn take_fault(&mut self, tokens: &[usize], emit_logits: bool) -> Option<AppendFaultKind> {
            let index = self
                .faults
                .iter()
                .position(|fault| fault.tokens == tokens && fault.emit_logits == emit_logits)?;
            Some(self.faults.remove(index).kind)
        }
    }

    impl RamaSessionAdapter for RecordingAdapter {
        fn context_len(&self) -> usize {
            self.context_len
        }

        fn max_seq_len(&self) -> usize {
            self.max_seq_len
        }

        fn context_memory_bytes(&self) -> usize {
            self.context_len * 16
        }

        fn append_tokens(
            &mut self,
            tokens: &[usize],
            budget: &mut MemoryBudget,
            emit_logits: bool,
        ) -> Result<Option<RamaSessionStep>> {
            if tokens.is_empty() {
                return Err(RuntimeError::InvalidTensorData(
                    "recording adapter rejects empty append".to_string(),
                ));
            }
            if self.context_len + tokens.len() > self.max_seq_len {
                return Err(RuntimeError::Shape(
                    "recording adapter overflow".to_string(),
                ));
            }
            if let Some(fault) = self.take_fault(tokens, emit_logits) {
                return match fault {
                    AppendFaultKind::Error => Err(RuntimeError::InvalidTensorData(
                        "recording adapter injected failure".to_string(),
                    )),
                    AppendFaultKind::None => Ok(None),
                    AppendFaultKind::UnexpectedLogits => Ok(Some(RamaSessionStep {
                        token_id: self.sample_base + self.appends.len() + 1,
                        logits: None,
                        cached_context_len_after: self.context_len,
                    })),
                };
            }
            let scripted_step = if emit_logits && !self.steps.is_empty() {
                match self.steps.remove(0) {
                    Ok(Some(step)) => Some(step),
                    Ok(None) => return Ok(None),
                    Err(err) => return Err(err),
                }
            } else {
                None
            };
            if self.transient_bytes > 0 {
                budget.reserve(self.transient_bytes, "recording adapter transient")?;
                budget.release(self.transient_bytes, "recording adapter transient")?;
            }
            self.appends.push((tokens.to_vec(), emit_logits));
            self.context_len += tokens.len();
            if emit_logits {
                if let Some(step) = scripted_step {
                    Ok(Some(step))
                } else {
                    let token_id = self.sample_base + self.appends.len();
                    Ok(Some(RamaSessionStep {
                        token_id,
                        logits: None,
                        cached_context_len_after: self.context_len,
                    }))
                }
            } else {
                Ok(None)
            }
        }

        fn take_last_phase_timings(&mut self) -> Option<RamaSessionPhaseTimings> {
            if self.phase_timings.is_empty() {
                None
            } else {
                Some(self.phase_timings.remove(0))
            }
        }

        fn take_last_rolling_stats(&mut self) -> Option<RamaRollingStats> {
            if self.rolling_stats.is_empty() {
                None
            } else {
                Some(self.rolling_stats.remove(0))
            }
        }

        fn take_last_experimental_speed_stats(&mut self) -> Option<RamaExperimentalSpeedStats> {
            if self.experimental_speed_stats.is_empty() {
                None
            } else {
                Some(self.experimental_speed_stats.remove(0))
            }
        }
    }

    #[test]
    fn second_turn_flushes_pending_tail_and_appends_only_new_user_tokens() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16));
        let mut budget = MemoryBudget::unbounded();

        let turn1 = session
            .generate_turn(&[1, 2], 2, &mut budget, |_| true)
            .unwrap();
        assert_eq!(turn1.generated_token_ids, [101, 102]);
        assert_eq!(turn1.metrics.replayed_tokens, 0);
        assert_eq!(session.pending_uncached_token(), Some(102));
        assert_eq!(session.cached_context_len(), 3);

        let turn2 = session
            .generate_turn(&[3], 1, &mut budget, |_| true)
            .unwrap();
        assert_eq!(turn2.metrics.flushed_pending_tokens, 1);
        assert_eq!(turn2.metrics.new_prefill_tokens, 1);
        assert_eq!(turn2.metrics.replayed_tokens, 0);
        assert_eq!(turn2.generated_token_ids, [104]);
        assert_eq!(
            session.adapter().appends,
            vec![
                (vec![1, 2], true),
                (vec![101], true),
                (vec![102], false),
                (vec![3], true),
            ]
        );
    }

    #[test]
    fn overflow_is_rejected_before_mutating_adapter_or_history() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(3));
        let mut budget = MemoryBudget::unbounded();

        let result = session.generate_turn(&[1, 2], 2, &mut budget, |_| true);

        assert!(result.is_err());
        assert!(session.token_history().is_empty());
        assert!(session.adapter().appends.is_empty());
    }

    #[test]
    fn empty_input_is_rejected_before_mutating_adapter_or_history() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16));
        let mut budget = MemoryBudget::unbounded();

        let result = session.generate_turn(&[], 1, &mut budget, |_| true);

        assert!(matches!(result, Err(RuntimeError::InvalidTensorData(_))));
        assert!(session.token_history().is_empty());
        assert_eq!(session.pending_uncached_token(), None);
        assert!(session.adapter().appends.is_empty());
    }

    #[test]
    fn zero_max_new_tokens_is_rejected_before_mutating_adapter_or_history() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16));
        let mut budget = MemoryBudget::unbounded();

        let result = session.generate_turn(&[1], 0, &mut budget, |_| true);

        assert!(matches!(result, Err(RuntimeError::InvalidTensorData(_))));
        assert!(session.token_history().is_empty());
        assert_eq!(session.pending_uncached_token(), None);
        assert!(session.adapter().appends.is_empty());
    }

    #[test]
    fn first_turn_prefill_error_preserves_empty_visible_state() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16).with_fault(
            &[1, 2],
            true,
            AppendFaultKind::Error,
        ));
        let mut budget = MemoryBudget::unbounded();

        let result = session.generate_turn(&[1, 2], 1, &mut budget, |_| true);

        assert!(result.is_err());
        assert!(session.token_history().is_empty());
        assert_eq!(session.pending_uncached_token(), None);
        assert_eq!(session.cached_context_len(), 0);
        assert!(session.adapter().appends.is_empty());
    }

    #[test]
    fn first_turn_prefill_none_preserves_empty_visible_state() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16).with_fault(
            &[1, 2],
            true,
            AppendFaultKind::None,
        ));
        let mut budget = MemoryBudget::unbounded();

        let result = session.generate_turn(&[1, 2], 1, &mut budget, |_| true);

        assert!(result.is_err());
        assert!(session.token_history().is_empty());
        assert_eq!(session.pending_uncached_token(), None);
        assert_eq!(session.cached_context_len(), 0);
        assert!(session.adapter().appends.is_empty());
    }

    #[test]
    fn second_turn_prefill_error_after_flush_keeps_coherent_retry_state() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16).with_fault(
            &[2],
            true,
            AppendFaultKind::Error,
        ));
        let mut budget = MemoryBudget::unbounded();

        session
            .generate_turn(&[1], 1, &mut budget, |_| true)
            .unwrap();
        assert_eq!(session.pending_uncached_token(), Some(101));
        assert_eq!(session.cached_context_len(), 1);

        let result = session.generate_turn(&[2], 1, &mut budget, |_| true);

        assert!(result.is_err());
        assert_eq!(session.pending_uncached_token(), None);
        assert_eq!(session.token_history(), &[1, 101]);
        assert_eq!(session.cached_context_len(), 2);
        assert_eq!(
            session.adapter().appends,
            vec![(vec![1], true), (vec![101], false)]
        );

        let retry = session
            .generate_turn(&[2], 1, &mut budget, |_| true)
            .unwrap();
        assert_eq!(retry.metrics.flushed_pending_tokens, 0);
        assert_eq!(retry.generated_token_ids, [103]);
        assert_eq!(session.pending_uncached_token(), Some(103));
        assert_eq!(session.token_history(), &[1, 101, 2, 103]);
        assert_eq!(
            session.adapter().appends,
            vec![(vec![1], true), (vec![101], false), (vec![2], true)]
        );
    }

    #[test]
    fn flush_failure_preserves_pending_tail_and_can_retry_turn() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16).with_fault(
            &[101],
            false,
            AppendFaultKind::Error,
        ));
        let mut budget = MemoryBudget::unbounded();

        let turn1 = session
            .generate_turn(&[1], 1, &mut budget, |_| true)
            .unwrap();
        assert_eq!(turn1.generated_token_ids, [101]);

        let result = session.generate_turn(&[2], 1, &mut budget, |_| true);

        assert!(result.is_err());
        assert_eq!(session.pending_uncached_token(), Some(101));
        assert_eq!(session.token_history(), &[1, 101]);
        assert_eq!(session.cached_context_len(), 1);

        let retry = session
            .generate_turn(&[2], 1, &mut budget, |_| true)
            .unwrap();
        assert_eq!(retry.metrics.flushed_pending_tokens, 1);
        assert_eq!(retry.generated_token_ids, [103]);
        assert_eq!(session.pending_uncached_token(), Some(103));
        assert_eq!(session.token_history(), &[1, 101, 2, 103]);
    }

    #[test]
    fn decode_failure_preserves_pending_tail_and_visible_history_can_continue() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16).with_fault(
            &[101],
            true,
            AppendFaultKind::Error,
        ));
        let mut budget = MemoryBudget::unbounded();

        let result = session.generate_turn(&[1], 2, &mut budget, |_| true);

        assert!(result.is_err());
        assert_eq!(session.pending_uncached_token(), Some(101));
        assert_eq!(session.token_history(), &[1, 101]);
        assert_eq!(session.cached_context_len(), 1);

        let retry = session
            .generate_turn(&[2], 1, &mut budget, |_| true)
            .unwrap();
        assert_eq!(retry.metrics.flushed_pending_tokens, 1);
        assert_eq!(retry.generated_token_ids, [103]);
        assert_eq!(session.pending_uncached_token(), Some(103));
        assert_eq!(session.token_history(), &[1, 101, 2, 103]);
    }

    #[test]
    fn flush_unexpected_logits_preserves_pending_tail() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16).with_fault(
            &[101],
            false,
            AppendFaultKind::UnexpectedLogits,
        ));
        let mut budget = MemoryBudget::unbounded();

        session
            .generate_turn(&[1], 1, &mut budget, |_| true)
            .unwrap();
        let result = session.generate_turn(&[2], 1, &mut budget, |_| true);

        assert!(result.is_err());
        assert_eq!(session.pending_uncached_token(), Some(101));
        assert_eq!(session.token_history(), &[1, 101]);
    }

    #[test]
    fn decode_none_preserves_pending_tail() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16).with_fault(
            &[101],
            true,
            AppendFaultKind::None,
        ));
        let mut budget = MemoryBudget::unbounded();

        let result = session.generate_turn(&[1], 2, &mut budget, |_| true);

        assert!(result.is_err());
        assert_eq!(session.pending_uncached_token(), Some(101));
        assert_eq!(session.token_history(), &[1, 101]);
        assert_eq!(session.cached_context_len(), 1);
    }

    #[test]
    fn peak_transient_bytes_reports_per_turn_peak_after_reset() {
        let mut session = RamaChatSession::new(RecordingAdapter::new(16).with_transient_bytes(64));
        let mut budget = MemoryBudget::new(1024);
        budget.reserve(512, "old peak").unwrap();
        budget.release(512, "old peak").unwrap();
        assert_eq!(budget.peak_bytes(), 512);

        let turn = session
            .generate_turn(&[1], 1, &mut budget, |_| true)
            .unwrap();

        assert_eq!(turn.metrics.peak_transient_bytes, 64);
        assert_eq!(budget.current_bytes(), 0);
    }

    #[test]
    fn turn_metrics_collect_adapter_phase_timings() {
        let mut adapter = RecordingAdapter::new(16);
        adapter.phase_timings.push(RamaSessionPhaseTimings {
            embedding_ms: 1.0,
            transformer_ms: 2.0,
            final_norm_ms: 3.0,
            lm_head_ms: 4.0,
            ..Default::default()
        });
        adapter.phase_timings.push(RamaSessionPhaseTimings {
            embedding_ms: 10.0,
            transformer_ms: 20.0,
            final_norm_ms: 30.0,
            lm_head_ms: 40.0,
            ..Default::default()
        });
        let mut session = RamaChatSession::new(adapter);
        let mut budget = MemoryBudget::unbounded();

        let result = session
            .generate_turn(&[1, 2], 2, &mut budget, |_| true)
            .unwrap();

        assert_eq!(result.metrics.phase_timings.embedding_ms, 11.0);
        assert_eq!(result.metrics.phase_timings.transformer_ms, 22.0);
        assert_eq!(result.metrics.phase_timings.final_norm_ms, 33.0);
        assert_eq!(result.metrics.phase_timings.lm_head_ms, 44.0);
    }

    #[test]
    fn turn_metrics_split_prefill_and_decode_phase_timings() {
        let mut adapter = RecordingAdapter::new(16);
        adapter.phase_timings.push(RamaSessionPhaseTimings {
            embedding_ms: 1.0,
            transformer_ms: 2.0,
            final_norm_ms: 3.0,
            lm_head_ms: 4.0,
            ..Default::default()
        });
        adapter.phase_timings.push(RamaSessionPhaseTimings {
            embedding_ms: 10.0,
            transformer_ms: 20.0,
            final_norm_ms: 30.0,
            lm_head_ms: 40.0,
            ..Default::default()
        });
        let mut session = RamaChatSession::new(adapter);
        let mut budget = MemoryBudget::unbounded();

        let result = session
            .generate_turn(&[1, 2], 2, &mut budget, |_| true)
            .unwrap();

        assert_eq!(result.metrics.prefill_phase_timings.embedding_ms, 1.0);
        assert_eq!(result.metrics.prefill_phase_timings.transformer_ms, 2.0);
        assert_eq!(result.metrics.prefill_phase_timings.final_norm_ms, 3.0);
        assert_eq!(result.metrics.prefill_phase_timings.lm_head_ms, 4.0);
        assert_eq!(result.metrics.decode_phase_timings.embedding_ms, 10.0);
        assert_eq!(result.metrics.decode_phase_timings.transformer_ms, 20.0);
        assert_eq!(result.metrics.decode_phase_timings.final_norm_ms, 30.0);
        assert_eq!(result.metrics.decode_phase_timings.lm_head_ms, 40.0);
    }

    #[test]
    fn turn_metrics_collect_adapter_rolling_stats() {
        let mut adapter = RecordingAdapter::new(16);
        adapter.steps = vec![
            Ok(Some(RamaSessionStep {
                token_id: 7,
                logits: None,
                cached_context_len_after: 1,
            })),
            Ok(Some(RamaSessionStep {
                token_id: 8,
                logits: None,
                cached_context_len_after: 2,
            })),
        ];
        adapter.rolling_stats.push(RamaRollingStats {
            submitted_tasks: 3,
            worker_wakeups: 2,
            sequential_fallbacks: 1,
            peak_scratch_bytes: 64,
        });
        adapter.rolling_stats.push(RamaRollingStats {
            submitted_tasks: 5,
            worker_wakeups: 4,
            sequential_fallbacks: 2,
            peak_scratch_bytes: 32,
        });
        let mut session = RamaChatSession::new(adapter);
        let mut budget = MemoryBudget::unbounded();

        let result = session
            .generate_turn(&[1], 2, &mut budget, |_| true)
            .unwrap();

        assert_eq!(result.generated_token_ids, [7, 8]);
        assert_eq!(result.metrics.rolling_stats.submitted_tasks, 8);
        assert_eq!(result.metrics.rolling_stats.worker_wakeups, 6);
        assert_eq!(result.metrics.rolling_stats.sequential_fallbacks, 3);
        assert_eq!(result.metrics.rolling_stats.peak_scratch_bytes, 64);
    }

    #[test]
    fn turn_metrics_collect_adapter_experimental_speed_stats() {
        let mut adapter = RecordingAdapter::new(16);
        adapter.steps = vec![
            Ok(Some(RamaSessionStep {
                token_id: 7,
                logits: None,
                cached_context_len_after: 1,
            })),
            Ok(Some(RamaSessionStep {
                token_id: 8,
                logits: None,
                cached_context_len_after: 2,
            })),
        ];
        adapter
            .experimental_speed_stats
            .push(RamaExperimentalSpeedStats {
                aip_policy: None,
                sparse_projection_calls: 2,
                exact_fallbacks: 1,
                selected_topk_sum: 128,
                max_selected_topk: 64,
                estimated_skipped_madds: 1024,
                peak_scratch_bytes: 512,
                attention_locality_uses: 0,
                attention_locality_added_indices: 0,
                attention_locality_max_selected: 0,
                lm_head_prefix_rows: 0,
                lm_head_vocab_rows: 0,
                lm_head_rescore_checks: 0,
                lm_head_rescore_uses: 0,
                lm_head_rescore_gap_skips: 0,
                lm_head_rescore_max_gap_milli: 0,
                column_cache_hits: 0,
                column_cache_misses: 0,
                column_cache_resident_columns: 0,
                column_cache_resident_bytes: 0,
                input_tile_range_reads: 0,
                input_tile_range_bytes: 0,
                lm_head_agreement_samples: 0,
                lm_head_agreement_sparse_argmax_matches: 0,
                lm_head_agreement_selected_matches: 0,
                lm_head_agreement_exact_in_sparse_topk: 0,
                lm_head_agreement_max_topk: 0,
                lm_head_exact_checks: 0,
                lm_head_exact_switches: 0,
                lm_head_repeat_margin_checks: 0,
                lm_head_repeat_margin_switches: 0,
                lm_head_repeat_margin_max_gap_milli: 0,
                lm_head_repeat_margin_adaptive_throttles: 0,
                lm_head_repeat_margin_min_effective_milli: 0,
                lm_head_phrase_novelty_checks: 0,
                lm_head_phrase_novelty_switches: 0,
                lm_head_phrase_novelty_max_ngram: 0,
                lm_head_phrase_novelty_gap_skips: 0,
                lm_head_phrase_novelty_max_gap_milli: 0,
                lm_head_phrase_novelty_soft_choices: 0,
                lm_head_phrase_novelty_retentions: 0,
                layer_drift_probe: Default::default(),
            });
        adapter
            .experimental_speed_stats
            .push(RamaExperimentalSpeedStats {
                aip_policy: None,
                sparse_projection_calls: 3,
                exact_fallbacks: 0,
                selected_topk_sum: 256,
                max_selected_topk: 128,
                estimated_skipped_madds: 2048,
                peak_scratch_bytes: 256,
                attention_locality_uses: 0,
                attention_locality_added_indices: 0,
                attention_locality_max_selected: 0,
                lm_head_prefix_rows: 0,
                lm_head_vocab_rows: 0,
                lm_head_rescore_checks: 0,
                lm_head_rescore_uses: 0,
                lm_head_rescore_gap_skips: 0,
                lm_head_rescore_max_gap_milli: 0,
                column_cache_hits: 0,
                column_cache_misses: 0,
                column_cache_resident_columns: 0,
                column_cache_resident_bytes: 0,
                input_tile_range_reads: 0,
                input_tile_range_bytes: 0,
                lm_head_agreement_samples: 0,
                lm_head_agreement_sparse_argmax_matches: 0,
                lm_head_agreement_selected_matches: 0,
                lm_head_agreement_exact_in_sparse_topk: 0,
                lm_head_agreement_max_topk: 0,
                lm_head_exact_checks: 0,
                lm_head_exact_switches: 0,
                lm_head_repeat_margin_checks: 0,
                lm_head_repeat_margin_switches: 0,
                lm_head_repeat_margin_max_gap_milli: 0,
                lm_head_repeat_margin_adaptive_throttles: 0,
                lm_head_repeat_margin_min_effective_milli: 0,
                lm_head_phrase_novelty_checks: 0,
                lm_head_phrase_novelty_switches: 0,
                lm_head_phrase_novelty_max_ngram: 0,
                lm_head_phrase_novelty_gap_skips: 0,
                lm_head_phrase_novelty_max_gap_milli: 0,
                lm_head_phrase_novelty_soft_choices: 0,
                lm_head_phrase_novelty_retentions: 0,
                layer_drift_probe: Default::default(),
            });
        let mut session = RamaChatSession::new(adapter);
        let mut budget = MemoryBudget::unbounded();

        let result = session
            .generate_turn(&[1], 2, &mut budget, |_| true)
            .unwrap();

        assert_eq!(result.generated_token_ids, [7, 8]);
        assert_eq!(
            result
                .metrics
                .experimental_speed_stats
                .sparse_projection_calls,
            5
        );
        assert_eq!(result.metrics.experimental_speed_stats.exact_fallbacks, 1);
        assert_eq!(
            result.metrics.experimental_speed_stats.selected_topk_sum,
            384
        );
        assert_eq!(
            result.metrics.experimental_speed_stats.max_selected_topk,
            128
        );
        assert_eq!(
            result
                .metrics
                .experimental_speed_stats
                .estimated_skipped_madds,
            3072
        );
        assert_eq!(
            result.metrics.experimental_speed_stats.peak_scratch_bytes,
            512
        );
    }

    #[test]
    fn repetition_stats_from_tokens_detects_runs_and_unique_count() {
        let stats = RamaRepetitionStats::from_tokens(&[7, 7, 7, 8, 9, 9]);

        assert_eq!(stats.generated_tokens, 6);
        assert_eq!(stats.unique_generated_tokens, 3);
        assert_eq!(stats.max_repeated_token_run, 3);
        assert!((stats.repeated_token_ratio - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn repetition_stats_from_empty_tokens_are_zero() {
        let stats = RamaRepetitionStats::from_tokens(&[]);

        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(stats.unique_generated_tokens, 0);
        assert_eq!(stats.max_repeated_token_run, 0);
        assert_eq!(stats.repeated_token_ratio, 0.0);
    }

    #[test]
    fn turn_metrics_include_repetition_stats() {
        let mut adapter = RecordingAdapter::new(16);
        adapter.steps = vec![
            Ok(Some(RamaSessionStep {
                token_id: 7,
                logits: None,
                cached_context_len_after: 1,
            })),
            Ok(Some(RamaSessionStep {
                token_id: 7,
                logits: None,
                cached_context_len_after: 2,
            })),
            Ok(Some(RamaSessionStep {
                token_id: 8,
                logits: None,
                cached_context_len_after: 3,
            })),
            Ok(Some(RamaSessionStep {
                token_id: 8,
                logits: None,
                cached_context_len_after: 4,
            })),
        ];
        let mut session = RamaChatSession::new(adapter);
        let mut budget = MemoryBudget::unbounded();

        let result = session
            .generate_turn(&[1], 4, &mut budget, |_| true)
            .unwrap();

        assert_eq!(result.generated_token_ids, [7, 7, 8, 8]);
        assert_eq!(result.metrics.repetition_stats.generated_tokens, 4);
        assert_eq!(result.metrics.repetition_stats.unique_generated_tokens, 2);
        assert_eq!(result.metrics.repetition_stats.max_repeated_token_run, 2);
        assert!(
            (result.metrics.repetition_stats.repeated_token_ratio - (2.0 / 3.0)).abs()
                < f64::EPSILON
        );
    }
}
