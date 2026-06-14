# RAMA Generic Chat Session R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build R1 persistent chat-session inference so later chat turns append only new tokens and keep model KV-cache alive across turns.

**Architecture:** Add a small generic `RamaChatSession` orchestration layer in `rllm-runtime`, backed first by a LLaMA-family adapter for `LayerDecodedLlamaRamaTransformer`. The session keeps token history and a pending uncached tail token, while the adapter owns model-family details, pinned tensors, and per-layer KV-cache.

**Tech Stack:** Rust 2021, existing `LazyRllmModel`, `MemoryBudget`, `KvCache`, LLaMA streaming block path, Clap CLI, Markdown benchmark reports.

**Execution Status:** Completed on 2026-06-14. Implementation, review, and first
SmolLM2 benchmark evidence are recorded in the linked benchmark trial report.
The first text-transcript benchmark is classified as inconclusive because
strict tokenizer equivalence failed before turn 2.

---

## File Structure

- Modify `crates/rllm-runtime/src/rotary.rs`
  - Add `KvCache::truncate` and `KvCache::resident_bytes` for rollback and reporting.
- Create `crates/rllm-runtime/src/session.rs`
  - Define generic session structs, metrics, adapter trait, and deterministic test adapter.
- Modify `crates/rllm-runtime/src/lib.rs`
  - Export the session API.
- Modify `crates/rllm-runtime/src/models/llama/api.rs`
  - Make existing LLaMA metadata helpers `pub(crate)` for reuse by the session adapter.
- Modify `crates/rllm-runtime/src/models/llama/mod.rs`
  - Export the LLaMA session adapter module.
- Create `crates/rllm-runtime/src/models/llama/session.rs`
  - Implement `LlamaRamaSessionAdapter` over the current layer-decoded LLaMA path.
- Modify `crates/rllm-cli/src/commands/mod.rs`
  - Register the new command module.
- Create `crates/rllm-cli/src/commands/chat_session.rs`
  - Implement scripted two-turn benchmark harness and report writer.
- Modify `crates/rllm-cli/src/main.rs`
  - Add the `chat-session` subcommand.
- Update `docs/benchmarks/trials/index.md`
  - Add R1 trial row after the first run writes evidence.
- Create `docs/benchmarks/trials/active/2026-06-14-r1-session-smollm2.md`
  - Store first R1 benchmark report.

## Key Design Decision: Pending Tail Token

Autoregressive generation caches the input token used to produce the next token,
not the sampled next token itself. Therefore, after a turn ends, the final
assistant token is visible in `token_history` but not yet present in KV-cache.
R1 must keep `pending_uncached_token: Option<usize>` and flush it before the
next user turn. This prevents history replay while preserving correct context.

---

### Task 1: Add KV-cache Rollback Primitives

**Files:**
- Modify: `crates/rllm-runtime/src/rotary.rs`

- [ ] **Step 1: Add failing tests for cache truncation**

Add these tests inside the existing `#[cfg(test)] mod tests` in `crates/rllm-runtime/src/rotary.rs`:

```rust
#[test]
fn kv_cache_truncate_rolls_back_len_and_buffers() {
    let mut cache = KvCache::new(2, 3, 4).unwrap();
    cache
        .append(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            2,
        )
        .unwrap();

    cache.truncate(1).unwrap();

    assert_eq!(cache.len(), 1);
    assert_eq!(cache.keys(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(cache.values(), &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);
    assert_eq!(cache.resident_bytes(), 48);
}

#[test]
fn kv_cache_truncate_rejects_growth() {
    let mut cache = KvCache::new(1, 2, 4).unwrap();
    cache.append(&[1.0, 2.0], &[3.0, 4.0], 1).unwrap();

    let result = cache.truncate(2);

    assert!(result.is_err());
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.keys(), &[1.0, 2.0]);
    assert_eq!(cache.values(), &[3.0, 4.0]);
}
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
cargo test -p rllm-runtime kv_cache_truncate -- --nocapture
```

Expected result before implementation: compile failure because `truncate` and
`resident_bytes` do not exist on `KvCache`.

- [ ] **Step 3: Implement rollback helpers**

Add these methods to `impl KvCache` in `crates/rllm-runtime/src/rotary.rs`:

```rust
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(RuntimeError::Shape(format!(
                "KV cache truncate cannot grow from {} to {len}",
                self.len
            )));
        }
        let value_len = len
            .checked_mul(self.token_width())
            .ok_or_else(|| RuntimeError::Shape("KV truncate length overflow".to_string()))?;
        self.keys.truncate(value_len);
        self.values.truncate(value_len);
        self.len = len;
        Ok(())
    }

    pub fn resident_bytes(&self) -> usize {
        self.len
            .saturating_mul(self.token_width())
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<f32>())
    }
```

- [ ] **Step 4: Verify the focused tests pass**

Run:

```bash
cargo test -p rllm-runtime kv_cache_truncate -- --nocapture
```

Expected result: both new `kv_cache_truncate_*` tests pass.

- [ ] **Step 5: Commit only this task's files when committing is requested**

```bash
git add crates/rllm-runtime/src/rotary.rs
git commit -m "feat(runtime): add kv cache rollback helpers"
```

---

### Task 2: Add Generic Session Core

**Files:**
- Create: `crates/rllm-runtime/src/session.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [ ] **Step 1: Create the session core API**

Create `crates/rllm-runtime/src/session.rs` with this implementation:

```rust
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

    fn append_tokens(&mut self, tokens: &[usize], budget: &mut MemoryBudget, emit_logits: bool)
        -> Result<Option<RamaSessionStep>>;
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
            .ok_or_else(|| RuntimeError::Shape("chat session context length overflow".to_string()))?;
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
```

- [ ] **Step 2: Export the module**

Modify `crates/rllm-runtime/src/lib.rs`:

```rust
mod session;
```

Add the export near other `pub use` lines:

```rust
pub use session::{
    RamaChatSession, RamaSessionAdapter, RamaSessionStep, RamaSessionTurnMetrics,
    RamaSessionTurnResult,
};
```

- [ ] **Step 3: Add deterministic adapter tests**

Append these tests to `crates/rllm-runtime/src/session.rs`:

```rust
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
                return Err(RuntimeError::Shape("recording adapter overflow".to_string()));
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
```

- [ ] **Step 4: Run focused session tests**

Run:

```bash
cargo test -p rllm-runtime session::tests -- --nocapture
```

Expected result: both session tests pass.

- [ ] **Step 5: Commit only this task's files when committing is requested**

```bash
git add crates/rllm-runtime/src/session.rs crates/rllm-runtime/src/lib.rs
git commit -m "feat(runtime): add generic chat session core"
```

---

### Task 3: Add LLaMA Session Adapter

**Files:**
- Modify: `crates/rllm-runtime/src/models/llama/api.rs`
- Modify: `crates/rllm-runtime/src/models/llama/mod.rs`
- Create: `crates/rllm-runtime/src/models/llama/session.rs`

- [ ] **Step 1: Expose existing LLaMA helpers within the crate**

Change these function signatures in `crates/rllm-runtime/src/models/llama/api.rs`:

```rust
pub(crate) fn require_model_config<'a>(
    model: &'a LazyRllmModel,
    architecture: &str,
) -> Result<&'a ModelConfigMetadata> {
```

```rust
pub(crate) fn require_config_usize(field_name: &str, value: Option<u64>) -> Result<usize> {
```

```rust
pub(crate) fn validate_llama_shape(
    hidden_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
) -> Result<usize> {
```

```rust
pub(crate) fn decode_vector_tensor(
    model: &mut LazyRllmModel,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>> {
```

- [ ] **Step 2: Add the adapter module export**

Modify `crates/rllm-runtime/src/models/llama/mod.rs`:

```rust
pub mod api;
pub mod generate;
pub mod model;
pub mod session;

pub use api::*;
pub use generate::*;
pub use model::*;
pub use session::*;
```

- [ ] **Step 3: Create LLaMA adapter**

Create `crates/rllm-runtime/src/models/llama/session.rs`:

```rust
use crate::models::llama::api::{
    decode_vector_tensor, require_config_usize, require_model_config, validate_llama_shape,
};
use crate::models::llama::generate::{
    streaming_llama_transformer_block, LlamaStreamingBlockConfig,
};
use crate::models::llama::model::{
    LayerDecodedLlamaRamaTransformer, OwnedLlamaStreamingBlockParameters,
};
use crate::rotary::KvCache;
use crate::{
    embedding_lookup, rms_norm, sample_argmax, sample_top_p, LazyRllmModel, MemoryBudget, Result,
    RuntimeError,
};
use crate::{RamaSessionAdapter, RamaSessionStep};

pub struct LlamaRamaSessionAdapter<'a> {
    model: &'a mut LazyRllmModel,
    prepared: LayerDecodedLlamaRamaTransformer,
    hidden_size: usize,
    intermediate_size: usize,
    head_dim: usize,
    vocab_size: usize,
    embedding_data: Vec<f32>,
    layer_norms: Vec<OwnedLlamaStreamingBlockParameters>,
    lm_head_weight_data: Vec<f32>,
    caches: Vec<KvCache>,
}

impl<'a> LlamaRamaSessionAdapter<'a> {
    pub fn new(
        model: &'a mut LazyRllmModel,
        prepared: &LayerDecodedLlamaRamaTransformer,
        budget: &mut MemoryBudget,
    ) -> Result<Self> {
        let model_config = require_model_config(model, "llama")?;
        let hidden_size = require_config_usize("hidden_size", model_config.hidden_size)?;
        let intermediate_size =
            require_config_usize("intermediate_size", model_config.intermediate_size)?;
        let head_dim = validate_llama_shape(
            hidden_size,
            prepared.config.num_heads,
            prepared.config.num_key_value_heads,
        )?;
        let max_seq_len = prepared.config.max_seq_len.ok_or_else(|| {
            RuntimeError::InvalidTensorData(
                "llama session config requires max_seq_len".to_string(),
            )
        })?;

        let embedding_data = model.decode_tensor(&prepared.embedding_weight, budget)?.data;
        let vocab_size = embedding_data.len() / hidden_size;
        let lm_head_weight_data = model.decode_tensor(&prepared.lm_head_weight, budget)?.data;

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

        Ok(Self {
            model,
            prepared: prepared.clone(),
            hidden_size,
            intermediate_size,
            head_dim,
            vocab_size,
            embedding_data,
            layer_norms,
            lm_head_weight_data,
            caches,
        })
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
        let projected_len = position_offset + seq_len;
        if projected_len > self.max_seq_len() {
            return Err(RuntimeError::Shape(format!(
                "llama session context would reach {projected_len}, max_seq_len {}",
                self.max_seq_len()
            )));
        }

        let mut hidden =
            embedding_lookup(&self.embedding_data, self.vocab_size, self.hidden_size, tokens)?;
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
            };
            hidden = streaming_llama_transformer_block(
                self.model,
                &hidden,
                layer_names,
                &self.layer_norms[i],
                config,
                budget,
                Some(&mut self.caches[i]),
            )?;
        }

        if !emit_logits {
            return Ok(None);
        }

        let hidden = rms_norm(
            &hidden,
            &self.prepared.final_layernorm_weight,
            seq_len,
            self.hidden_size,
            self.prepared.config.rms_norm_eps,
        )?;
        let last_hidden = &hidden[(seq_len - 1) * self.hidden_size..];
        let mut logits = vec![0.0f32; self.vocab_size];
        for v in 0..self.vocab_size {
            let mut sum = 0.0f32;
            for h in 0..self.hidden_size {
                sum += last_hidden[h] * self.lm_head_weight_data[v * self.hidden_size + h];
            }
            logits[v] = sum;
        }
        let token_id = match self.prepared.config.sampling {
            crate::StreamingSamplingConfig::Argmax => sample_argmax(&logits)?,
            crate::StreamingSamplingConfig::TopP {
                temperature,
                top_p,
                seed,
            } => sample_top_p(&logits, temperature, top_p, seed)?,
        };
        Ok(Some(RamaSessionStep {
            token_id,
            logits: Some(logits),
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
        let old_lens: Vec<usize> = self.caches.iter().map(KvCache::len).collect();
        match self.append_tokens_inner(tokens, budget, emit_logits) {
            Ok(step) => Ok(step),
            Err(error) => {
                for (cache, len) in self.caches.iter_mut().zip(old_lens) {
                    let _ = cache.truncate(len);
                }
                Err(error)
            }
        }
    }
}
```

- [ ] **Step 4: Run focused compile**

Run:

```bash
cargo check -p rllm-runtime
```

Expected result: runtime compiles with the new LLaMA adapter.

- [ ] **Step 5: Run existing LLaMA tests**

Run:

```bash
cargo test -p rllm-runtime models::llama -- --nocapture
```

Expected result: existing LLaMA tests pass after helper visibility changes and
the new adapter module compile.

- [ ] **Step 6: Commit only this task's files when committing is requested**

```bash
git add crates/rllm-runtime/src/models/llama/api.rs crates/rllm-runtime/src/models/llama/mod.rs crates/rllm-runtime/src/models/llama/session.rs
git commit -m "feat(runtime): add llama chat session adapter"
```

---

### Task 4: Add Scripted Chat-Session Benchmark Command

**Files:**
- Modify: `crates/rllm-cli/src/commands/mod.rs`
- Create: `crates/rllm-cli/src/commands/chat_session.rs`
- Modify: `crates/rllm-cli/src/main.rs`

- [ ] **Step 1: Register command module**

Modify `crates/rllm-cli/src/commands/mod.rs`:

```rust
pub mod benchmark;
pub mod chat_session;
pub mod doctor;
pub mod import;
pub mod inspect;
pub mod pack;
pub mod run;
pub mod unpack;
pub mod verify;
```

- [ ] **Step 2: Add CLI variant**

Add this variant to `Commands` in `crates/rllm-cli/src/main.rs`:

```rust
    /// Run a scripted persistent chat-session benchmark
    ChatSession {
        /// Path to .rllm file
        file: String,

        /// Conversation turn text; pass this flag more than once
        #[arg(long = "turn", required = true)]
        turns: Vec<String>,

        /// Maximum assistant tokens per turn
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,

        /// Maximum context length
        #[arg(long, default_value_t = 2048)]
        ctx: usize,

        /// Markdown report output path
        #[arg(long)]
        out: String,
    },
```

Add this match arm:

```rust
        Commands::ChatSession {
            file,
            turns,
            max_new_tokens,
            ctx,
            out,
        } => commands::chat_session::run(&file, &turns, max_new_tokens, ctx, &out),
```

- [ ] **Step 3: Implement command**

Create `crates/rllm-cli/src/commands/chat_session.rs`:

```rust
use anyhow::{Context, Result};
use rllm_runtime::{
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata, LlamaRamaGenerationConfig,
        LlamaRamaSessionAdapter,
    },
    LazyRllmModel, MemoryBudget, RamaChatSession, RamaIntegrityMode, RllmTokenizer,
    StreamingSamplingConfig,
};
use std::fs;
use std::path::Path;

pub fn run(
    file: &str,
    turns: &[String],
    max_new_tokens: usize,
    ctx: usize,
    out: &str,
) -> Result<()> {
    if turns.is_empty() {
        anyhow::bail!("chat-session requires at least one --turn");
    }
    if max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    if ctx == 0 {
        anyhow::bail!("--ctx must be greater than zero");
    }

    let mut model = LazyRllmModel::open(file).with_context(|| format!("failed to open {file}"))?;
    model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
    let tokenizer_meta = model
        .metadata()
        .tokenizer
        .as_ref()
        .context("model metadata does not include tokenizer metadata")?;
    let tokenizer = RllmTokenizer::from_metadata(tokenizer_meta)?;
    let prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(
        &mut model,
        LlamaRamaGenerationConfig {
            max_new_tokens,
            max_seq_len: Some(ctx),
            causal: true,
            sampling: StreamingSamplingConfig::Argmax,
        },
    )?;
    let mut budget = MemoryBudget::unbounded();
    let adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget)?;
    let mut session = RamaChatSession::new(adapter);

    let mut report_turns = Vec::new();
    for (idx, turn) in turns.iter().enumerate() {
        let input_token_ids = tokenizer.encode(turn)?;
        let result = session.generate_turn(&input_token_ids, max_new_tokens, &mut budget, |_| true)?;
        println!(
            "turn {}: input={} generated={} replayed={} ttft_ms={:.2} decode_tok_s={:.2}",
            idx + 1,
            result.metrics.input_tokens,
            result.metrics.generated_tokens,
            result.metrics.replayed_tokens,
            result.metrics.ttft_ms,
            result.metrics.decode_tok_s
        );
        report_turns.push((idx + 1, turn.clone(), result));
    }

    write_report(out, file, max_new_tokens, ctx, &report_turns)?;
    println!("Benchmark report: {out}");
    Ok(())
}

fn write_report(
    out: &str,
    file: &str,
    max_new_tokens: usize,
    ctx: usize,
    turns: &[(usize, String, rllm_runtime::RamaSessionTurnResult)],
) -> Result<()> {
    if let Some(parent) = Path::new(out).parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = String::new();
    body.push_str("# Trial: R1 Persistent Chat Session SmolLM2\n\n");
    body.push_str("Date: 2026-06-14\n");
    body.push_str("Owner: RLLM\n");
    body.push_str("Status: running\n");
    body.push_str("Folder: active\n\n");
    body.push_str("## Hypothesis\n\n");
    body.push_str("Keeping KV-cache alive across turns reduces turn 2 prefill latency because only new user tokens are appended.\n\n");
    body.push_str("## Scope\n\n");
    body.push_str("- Mode: exact-lowram\n");
    body.push_str(&format!("- Model/artifact: `{file}`\n"));
    body.push_str("- Architecture: llama\n");
    body.push_str("- Target device/profile: single CPU, low RAM\n");
    body.push_str("- Expected bottleneck: full-history replay and memory bandwidth\n");
    body.push_str("- Bottleneck tag: cache locality\n\n");
    body.push_str("## Setup\n\n");
    body.push_str("Commands:\n\n```bash\n");
    body.push_str(&format!(
        "cargo run -p rllm-cli -- chat-session {file} --turn \"{}\" --turn \"{}\" --max-new-tokens {max_new_tokens} --ctx {ctx} --out {out}\n",
        turns.first().map(|(_, text, _)| text.as_str()).unwrap_or(""),
        turns.get(1).map(|(_, text, _)| text.as_str()).unwrap_or("")
    ));
    body.push_str("```\n\n");
    body.push_str("## Results\n\n");
    body.push_str("| run | prompt/input tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |\n");
    body.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for (idx, _, result) in turns {
        body.push_str(&format!(
            "| turn {idx} | {} | {} | {:.2} ms | {:.2} | {:.2} | not captured | {} bytes | replayed_tokens={} context_bytes={} |\n",
            result.metrics.input_tokens,
            result.metrics.generated_tokens,
            result.metrics.ttft_ms,
            result.metrics.decode_tok_s,
            result.metrics.end_to_end_tok_s,
            result.metrics.peak_transient_bytes,
            result.metrics.replayed_tokens,
            result.metrics.context_memory_bytes
        ));
    }
    body.push_str("\n## Analysis\n\n");
    body.push_str("Turn 2 is valid only if `replayed_tokens` remains zero and `flushed_pending_tokens` is one when turn 1 generated at least one assistant token.\n\n");
    body.push_str("## Decision\n\n");
    body.push_str("needs follow-up\n\n");
    body.push_str("Reason: compare this report against the existing `llama-test` full-replay baseline before moving it to success or failed.\n\n");
    body.push_str("Paper value:\n\n- use as limitation\n\n");
    body.push_str("## Next Experiment\n\n");
    body.push_str("Run the same turns through the old full-replay chat path and compare turn 2 TTFT, decode tok/s, and memory.\n");
    fs::write(out, body)?;
    Ok(())
}
```

- [ ] **Step 4: Run CLI compile**

Run:

```bash
cargo check -p rllm-cli
```

Expected result: CLI compiles and exposes `chat-session`.

- [ ] **Step 5: Commit only this task's files when committing is requested**

```bash
git add crates/rllm-cli/src/commands/mod.rs crates/rllm-cli/src/commands/chat_session.rs crates/rllm-cli/src/main.rs
git commit -m "feat(cli): add persistent chat session benchmark"
```

---

### Task 5: Run R1 Benchmark Evidence

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-14-r1-session-smollm2.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Build release binary**

Run:

```bash
cargo build --release -p rllm-cli
```

Expected result: release CLI builds successfully.

- [ ] **Step 2: Run old full-replay baseline**

Run:

```bash
printf 'Hello\nContinue\nexit\n' | /usr/bin/time -l target/release/llama-test --model models/SmolLM2-135M-raw.rllm
```

Expected evidence to record in the report:

- turn 1 prefill seconds
- turn 1 decode tok/s
- turn 2 prefill seconds
- turn 2 decode tok/s
- maximum resident set size
- peak memory footprint

- [ ] **Step 3: Run R1 persistent session**

Run:

```bash
cargo run --release -p rllm-cli -- chat-session models/SmolLM2-135M-raw.rllm --turn "Hello" --turn "Continue" --max-new-tokens 64 --ctx 2048 --out docs/benchmarks/trials/active/2026-06-14-r1-session-smollm2.md
```

Expected result:

- report file is written under `docs/benchmarks/trials/active/`
- turn 2 has `replayed_tokens=0`
- turn 2 has `flushed_pending_tokens=1`
- command prints per-turn TTFT and decode tok/s

- [ ] **Step 4: Update trial index**

Add a row to `docs/benchmarks/trials/index.md`:

```markdown
| 2026-06-14 | 2026-06-14-r1-session-smollm2.md | active | SmolLM2-135M raw | exact-lowram | cache locality | llama-test full replay | persistent session, turn 2 replayed_tokens=0 | needs follow-up | use as limitation |
```

If measured turn 2 TTFT improves clearly and decode tok/s does not regress
materially, move the report to `docs/benchmarks/trials/success/` and change the
folder/decision cells to `success` and `accepted`. If it regresses, move it to
`docs/benchmarks/trials/failed/` and keep the bottleneck notes.

- [ ] **Step 5: Commit only benchmark evidence when committing is requested**

```bash
git add docs/benchmarks/trials/active/2026-06-14-r1-session-smollm2.md docs/benchmarks/trials/index.md
git commit -m "docs(benchmarks): record r1 chat session trial"
```

---

### Task 6: Full Verification

**Files:**
- All files touched by Tasks 1 through 5.

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt --check
```

Expected result: formatting is clean. If it fails, run `cargo fmt`, inspect the
diff, then rerun `cargo fmt --check`.

- [ ] **Step 2: Workspace check**

Run:

```bash
cargo check --workspace
```

Expected result: all workspace crates compile.

- [ ] **Step 3: Workspace tests**

Run:

```bash
cargo test --workspace
```

Expected result: all tests pass.

- [ ] **Step 4: Clippy**

Run:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected result: no warnings.

- [ ] **Step 5: Doctor**

Run:

```bash
cargo run --quiet -- doctor
```

Expected result: doctor command completes successfully.

---

## Self-Review

Spec coverage:

- Generic session boundary is covered by Task 2.
- LLaMA/SmolLM2 first adapter is covered by Task 3.
- No full-history replay after turn 1 is covered by Task 2 tests and Task 5 benchmark evidence.
- Benchmark documentation with success/failed routing is covered by Task 5.
- Error handling for empty prompts, max token zero, overflow, and adapter rollback is covered by Tasks 1 through 4.

Known implementation risks:

- `LlamaRamaSessionAdapter` owns `&mut LazyRllmModel`, so only one live adapter can use the model at a time. That is acceptable for R1 and keeps the generic session trait testable without a fake model.
- Final assistant token remains pending until the next turn. The benchmark report must show `flushed_pending_tokens=1` on turn 2 to prove this path is active.
- RSS capture is not inside the Rust command yet. Use `/usr/bin/time -l` for paper-grade RSS until an in-process platform-specific sampler is added.

Execution handoff:

Plan complete and saved to `docs/superpowers/plans/2026-06-14-rama-generic-chat-session-r1.md`. Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task and review between tasks.
2. Inline Execution - execute tasks in this session using checkpoints.
