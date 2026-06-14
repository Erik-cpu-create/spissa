use crate::{MemoryBudget, Result, RuntimeError};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct RamaSessionStep {
    pub token_id: usize,
    pub logits: Option<Vec<f32>>,
    pub cached_context_len_after: usize,
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
}

#[derive(Debug, Clone)]
pub struct RamaSessionTurnResult {
    pub input_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub token_history: Vec<usize>,
    pub metrics: RamaSessionTurnMetrics,
}

pub trait RamaSessionAdapter {
    fn context_len(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn context_memory_bytes(&self) -> usize;

    fn append_tokens(
        &mut self,
        tokens: &[usize],
        budget: &mut MemoryBudget,
        emit_logits: bool,
    ) -> Result<Option<RamaSessionStep>>;
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

    pub fn adapter_mut(&mut self) -> &mut A {
        &mut self.adapter
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

        let turn_start = Instant::now();
        let mut flushed_pending_tokens = 0usize;
        if let Some(token) = self.pending_uncached_token.take() {
            let emitted = self.adapter.append_tokens(&[token], budget, false)?;
            if emitted.is_some() {
                return Err(RuntimeError::InvalidTensorData(
                    "chat session pending-token flush unexpectedly emitted logits".to_string(),
                ));
            }
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
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = turn_start.elapsed().as_secs_f64() * 1000.0;

        self.token_history.extend_from_slice(user_token_ids);
        let mut generated_token_ids = vec![first_step.token_id];
        self.token_history.push(first_step.token_id);
        self.pending_uncached_token = Some(first_step.token_id);
        if !on_token(first_step.token_id) {
            return Ok(self.turn_result(
                user_token_ids,
                generated_token_ids,
                RamaSessionTurnMetrics {
                    input_tokens: user_token_ids.len(),
                    generated_tokens: 1,
                    new_prefill_tokens: user_token_ids.len(),
                    flushed_pending_tokens,
                    replayed_tokens: 0,
                    ttft_ms,
                    prefill_ms,
                    decode_ms: 0.0,
                    end_to_end_ms: turn_start.elapsed().as_secs_f64() * 1000.0,
                    decode_tok_s: 0.0,
                    end_to_end_tok_s: 1000.0 / ttft_ms.max(f64::EPSILON),
                    context_memory_bytes: self.adapter.context_memory_bytes(),
                    peak_transient_bytes: budget.peak_bytes(),
                },
            ));
        }

        let decode_start = Instant::now();
        while generated_token_ids.len() < max_new_tokens {
            let previous = self.pending_uncached_token.take().ok_or_else(|| {
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

    #[derive(Debug, Clone)]
    struct RecordingAdapter {
        max_seq_len: usize,
        context_len: usize,
        sample_base: usize,
        appends: Vec<(Vec<usize>, bool)>,
    }

    impl RecordingAdapter {
        fn new(max_seq_len: usize) -> Self {
            Self {
                max_seq_len,
                context_len: 0,
                sample_base: 100,
                appends: Vec::new(),
            }
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
            _budget: &mut MemoryBudget,
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
            self.appends.push((tokens.to_vec(), emit_logits));
            self.context_len += tokens.len();
            if emit_logits {
                let token_id = self.sample_base + self.appends.len();
                Ok(Some(RamaSessionStep {
                    token_id,
                    logits: None,
                    cached_context_len_after: self.context_len,
                }))
            } else {
                Ok(None)
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
}
