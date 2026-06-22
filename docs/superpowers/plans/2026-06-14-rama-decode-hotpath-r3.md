# RAMA Decode Hot-Path R3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add R3B decode subphase evidence to the token-native chat-session benchmark, then apply one low-risk LLaMA session hot-path optimization if profiling supports it.

**Architecture:** Keep generic chat-session timing types in `crates/rllm-runtime/src/session.rs`, collect LLaMA-specific subphase timings inside `crates/rllm-runtime/src/models/llama/session.rs`, and only surface summarized evidence in `crates/rllm-cli/src/commands/chat_session_token.rs`. The benchmark continues to use R2 token equality as the correctness gate.

**Tech Stack:** Rust workspace, `rllm-runtime`, `rllm-cli`, Markdown benchmark reports, Cargo test/check/clippy.

---

### Task 1: Add Generic Session Phase Timing Plumbing

**Files:**
- Modify: `crates/rllm-runtime/src/session.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [ ] **Step 1: Add a failing unit test for adapter phase timing aggregation**

In `crates/rllm-runtime/src/session.rs`, extend the existing `RecordingAdapter` test double with a `next_phase_timing` queue and add this test in the same `#[cfg(test)]` module:

```rust
#[test]
fn turn_metrics_collect_adapter_phase_timings() {
    let mut adapter = RecordingAdapter::new(16);
    adapter.phase_timings.push(RamaSessionPhaseTimings {
        embedding_ms: 1.0,
        transformer_ms: 2.0,
        final_norm_ms: 3.0,
        lm_head_ms: 4.0,
    });
    adapter.phase_timings.push(RamaSessionPhaseTimings {
        embedding_ms: 10.0,
        transformer_ms: 20.0,
        final_norm_ms: 30.0,
        lm_head_ms: 40.0,
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
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
cargo test -p rllm-runtime turn_metrics_collect_adapter_phase_timings -- --nocapture
```

Expected: FAIL because `RamaSessionPhaseTimings`, `metrics.phase_timings`, and the adapter timing hook do not exist yet.

- [ ] **Step 3: Add the public phase timing type and adapter hook**

In `crates/rllm-runtime/src/session.rs`, add this type after `RamaSessionStep`:

```rust
#[derive(Debug, Clone, Copy, Default)]
pub struct RamaSessionPhaseTimings {
    pub embedding_ms: f64,
    pub transformer_ms: f64,
    pub final_norm_ms: f64,
    pub lm_head_ms: f64,
}

impl RamaSessionPhaseTimings {
    pub fn add_assign(&mut self, other: RamaSessionPhaseTimings) {
        self.embedding_ms += other.embedding_ms;
        self.transformer_ms += other.transformer_ms;
        self.final_norm_ms += other.final_norm_ms;
        self.lm_head_ms += other.lm_head_ms;
    }

    pub fn total_ms(&self) -> f64 {
        self.embedding_ms + self.transformer_ms + self.final_norm_ms + self.lm_head_ms
    }
}
```

Add a field to `RamaSessionTurnMetrics`:

```rust
pub phase_timings: RamaSessionPhaseTimings,
```

Add this default hook to `RamaSessionAdapter`:

```rust
fn take_last_phase_timings(&mut self) -> Option<RamaSessionPhaseTimings> {
    None
}
```

In `RamaChatSession::generate_turn`, create:

```rust
let mut phase_timings = RamaSessionPhaseTimings::default();
```

After every successful `adapter.append_tokens(...)` call, aggregate:

```rust
if let Some(timings) = self.adapter.take_last_phase_timings() {
    phase_timings.add_assign(timings);
}
```

Set `phase_timings` in every `RamaSessionTurnMetrics` literal in this method.

- [ ] **Step 4: Update `RecordingAdapter` for the test**

In the test double struct, add:

```rust
phase_timings: Vec<RamaSessionPhaseTimings>,
```

Initialize it in `RecordingAdapter::new`:

```rust
phase_timings: Vec::new(),
```

Implement the hook:

```rust
fn take_last_phase_timings(&mut self) -> Option<RamaSessionPhaseTimings> {
    if self.phase_timings.is_empty() {
        None
    } else {
        Some(self.phase_timings.remove(0))
    }
}
```

- [ ] **Step 5: Export the timing type**

In `crates/rllm-runtime/src/lib.rs`, add `RamaSessionPhaseTimings` to the existing `pub use chat_session::{...}` list.

- [ ] **Step 6: Run focused runtime tests**

Run:

```bash
cargo test -p rllm-runtime turn_metrics_collect_adapter_phase_timings -- --nocapture
cargo test -p rllm-runtime chat_session -- --nocapture
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add crates/rllm-runtime/src/session.rs crates/rllm-runtime/src/lib.rs
git commit -m "feat(runtime): collect session phase timings"
```

---

### Task 2: Instrument the LLaMA Session Decode Path

**Files:**
- Modify: `crates/rllm-runtime/src/models/llama/session.rs`

- [ ] **Step 1: Add failing LLaMA timing tests**

In `crates/rllm-runtime/src/models/llama/session.rs`, add tests to the existing module:

```rust
#[test]
fn llama_session_records_phase_timings_for_logits_append() {
    let path = temp_path("phase-timing-logits");
    write_constructor_model(&path, vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64]);
    let mut model = LazyRllmModel::open(&path).unwrap();
    let prepared = prepared_with_layers(0);
    let mut budget = MemoryBudget::unbounded();
    let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();

    let step = adapter.append_tokens(&[0], &mut budget, true).unwrap();
    let timings = adapter.take_last_phase_timings().unwrap();

    assert!(step.is_some());
    assert!(timings.embedding_ms >= 0.0);
    assert!(timings.transformer_ms >= 0.0);
    assert!(timings.final_norm_ms >= 0.0);
    assert!(timings.lm_head_ms >= 0.0);
    assert!(timings.total_ms() >= 0.0);
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
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
cargo test -p rllm-runtime llama_session_records_phase_timings_for_logits_append -- --nocapture
cargo test -p rllm-runtime llama_session_clears_phase_timings_after_failed_append -- --nocapture
```

Expected: FAIL because `LlamaRamaSessionAdapter` does not expose phase timings yet.

- [ ] **Step 3: Add timing storage to the adapter**

In `crates/rllm-runtime/src/models/llama/session.rs`, update imports:

```rust
use crate::{RamaSessionAdapter, RamaSessionPhaseTimings, RamaSessionStep};
use std::time::Instant;
```

Add this field to `LlamaRamaSessionAdapter`:

```rust
last_phase_timings: Option<RamaSessionPhaseTimings>,
```

Initialize it in `new`:

```rust
last_phase_timings: None,
```

- [ ] **Step 4: Measure subphases in `append_tokens_inner`**

Inside `append_tokens_inner`, create a local timing accumulator:

```rust
let mut phase_timings = RamaSessionPhaseTimings::default();
let phase_start = Instant::now();
let mut hidden = embedding_lookup(
    &self.embedding_data,
    self.vocab_size,
    self.hidden_size,
    tokens,
)?;
phase_timings.embedding_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
```

Wrap the transformer layer loop:

```rust
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
phase_timings.transformer_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
```

Before returning for `emit_logits == false`, set:

```rust
self.last_phase_timings = Some(phase_timings);
return Ok(None);
```

Wrap final norm:

```rust
let phase_start = Instant::now();
let hidden = rms_norm(
    &hidden,
    &self.prepared.final_layernorm_weight,
    seq_len,
    self.hidden_size,
    self.prepared.config.rms_norm_eps,
)?;
phase_timings.final_norm_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
```

Wrap LM head and sampling:

```rust
let phase_start = Instant::now();
let last_hidden = &hidden[(seq_len - 1) * self.hidden_size..];
let mut logits = vec![0.0f32; self.vocab_size];
for (v, logit) in logits.iter_mut().enumerate() {
    let row_start = v * self.hidden_size;
    let row = &self.lm_head_weight_data[row_start..row_start + self.hidden_size];
    let mut sum = 0.0f32;
    for (&hidden, &weight) in last_hidden.iter().zip(row.iter()) {
        sum += hidden * weight;
    }
    *logit = sum;
}
let token_id = match self.prepared.config.sampling {
    crate::StreamingSamplingConfig::Argmax => sample_argmax(&logits)?,
    crate::StreamingSamplingConfig::TopP {
        temperature,
        top_p,
        seed,
    } => sample_top_p(&logits, temperature, top_p, seed)?,
};
phase_timings.lm_head_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
self.last_phase_timings = Some(phase_timings);
```

- [ ] **Step 5: Clear stale timings on failure and implement the hook**

At the start of `append_tokens`, before saving cache lengths:

```rust
self.last_phase_timings = None;
```

In the error branch:

```rust
self.last_phase_timings = None;
```

Add to the `impl RamaSessionAdapter for LlamaRamaSessionAdapter<'_>` block:

```rust
fn take_last_phase_timings(&mut self) -> Option<RamaSessionPhaseTimings> {
    self.last_phase_timings.take()
}
```

- [ ] **Step 6: Run focused LLaMA session tests**

Run:

```bash
cargo test -p rllm-runtime llama_session -- --nocapture
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add crates/rllm-runtime/src/models/llama/session.rs
git commit -m "feat(runtime): profile llama session decode phases"
```

---

### Task 3: Surface R3 Phase Timings in the Token-Native Report

**Files:**
- Modify: `crates/rllm-cli/src/commands/chat_session_token.rs`

- [ ] **Step 1: Add failing CLI report test**

In `crates/rllm-cli/src/commands/chat_session_token.rs`, update the test imports:

```rust
use rllm_runtime::RamaSessionPhaseTimings;
use super::{
    format_phase_timing_note, parse_token_turns, token_match_summary,
    validate_report_output_path,
};
```

Add this test:

```rust
#[test]
fn format_phase_timing_note_summarizes_decode_subphases() {
    let note = format_phase_timing_note(RamaSessionPhaseTimings {
        embedding_ms: 1.25,
        transformer_ms: 8.5,
        final_norm_ms: 0.75,
        lm_head_ms: 2.0,
    });

    assert!(note.contains("embedding=1.25ms"));
    assert!(note.contains("transformer=8.50ms"));
    assert!(note.contains("final_norm=0.75ms"));
    assert!(note.contains("lm_head=2.00ms"));
    assert!(note.contains("profiled_total=12.50ms"));
}
```

- [ ] **Step 2: Run the focused CLI test and verify it fails**

Run:

```bash
cargo test -p rllm-cli format_phase_timing_note_summarizes_decode_subphases -- --nocapture
```

Expected: FAIL because `format_phase_timing_note` does not exist yet.

- [ ] **Step 3: Add the report formatter**

In `crates/rllm-cli/src/commands/chat_session_token.rs`, import the timing type:

```rust
use rllm_runtime::RamaSessionPhaseTimings;
```

Add this helper above `write_report`:

```rust
fn format_phase_timing_note(timings: RamaSessionPhaseTimings) -> String {
    format!(
        "embedding={:.2}ms transformer={:.2}ms final_norm={:.2}ms lm_head={:.2}ms profiled_total={:.2}ms",
        timings.embedding_ms,
        timings.transformer_ms,
        timings.final_norm_ms,
        timings.lm_head_ms,
        timings.total_ms()
    )
}
```

- [ ] **Step 4: Include R3 timing evidence in the Markdown report**

In the results table header, add one column before `notes`:

```text
| session phase timing |
```

In each row, add:

```rust
format_phase_timing_note(row.session_result.metrics.phase_timings)
```

Update the `format!` string so the row includes the timing note as its own Markdown table cell.

In `## Analysis`, after the existing valid/inconclusive statement, add:

```rust
body.push_str("R3 phase timing is aggregated from LLaMA session adapter append calls for the measured turn. Treat it as coarse wall-clock evidence for choosing the next hot-path target, not cycle-level profiling.\n\n");
```

In `## Next Experiment`, replace the R2 sentence with:

```rust
body.push_str("Use the R3 phase timing columns to decide whether the next pass should target LM head/sampling, transformer matmul, KV-cache layout, or memory bandwidth.\n");
```

- [ ] **Step 5: Run focused CLI tests**

Run:

```bash
cargo test -p rllm-cli chat_session -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add crates/rllm-cli/src/commands/chat_session_token.rs
git commit -m "feat(cli): report session decode phase timings"
```

---

### Task 4: Run the Pre-Optimization R3 Benchmark Evidence

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-preopt.md`

- [ ] **Step 1: Run full verification before benchmark**

Run:

```bash
cargo fmt --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run --quiet -- doctor
```

Expected: all commands PASS.

- [ ] **Step 2: Run the same token-native benchmark shape as R2**

Run:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-preopt.md'
```

Expected: command exits successfully, token match stays true, and the report includes `session phase timing`.

- [ ] **Step 3: Classify the optimization gate**

Open the pre-optimization report and use this exact gate:

```text
If lm_head_ms is at least 15% of profiled_total_ms on any measured turn, continue to Task 5.
If lm_head_ms is below 15% on every measured turn, skip Task 5 and update the report Analysis/Decision to say transformer or another bucket dominates.
```

This gate prevents implementing an LM-head optimization when the profile does not support it.

- [ ] **Step 4: Commit the pre-optimization report**

Run:

```bash
git add docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-preopt.md
git commit -m "docs: record r3 preopt decode profile"
```

Expected: report is committed before any optimization is attempted.

---

### Task 5: Add Argmax No-Full-Logits Fast Path When the Gate Passes

**Files:**
- Modify: `crates/rllm-runtime/src/models/llama/session.rs`
- Create: `docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-postopt.md`

- [ ] **Step 1: Add failing tests for the argmax fast path**

In `crates/rllm-runtime/src/models/llama/session.rs`, add:

```rust
#[test]
fn llama_session_argmax_fast_path_returns_token_without_logits_vector() {
    let path = temp_path("argmax-fast-path");
    write_constructor_model(&path, vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64]);
    let mut model = LazyRllmModel::open(&path).unwrap();
    let prepared = prepared_with_layers(0);
    let mut budget = MemoryBudget::unbounded();
    let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();

    let step = adapter.append_tokens(&[0], &mut budget, true).unwrap().unwrap();

    assert_eq!(step.token_id, 0);
    assert!(step.logits.is_none());
    assert_eq!(step.cached_context_len_after, 0);
    std::fs::remove_file(path).ok();
}
```

Expected behavior: for argmax sampling, the LLaMA session can return the sampled token without keeping a full logits vector because `RamaChatSession` does not consume logits.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
cargo test -p rllm-runtime llama_session_argmax_fast_path_returns_token_without_logits_vector -- --nocapture
```

Expected: FAIL because current LLaMA session returns `Some(logits)`.

- [ ] **Step 3: Add a streaming argmax helper**

In the `impl LlamaRamaSessionAdapter<'_>` block, add:

```rust
fn sample_argmax_from_lm_head(&self, last_hidden: &[f32]) -> Result<usize> {
    if self.vocab_size == 0 {
        return Err(RuntimeError::InvalidTensorData(
            "cannot sample from empty logits".to_string(),
        ));
    }
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for v in 0..self.vocab_size {
        let row_start = v * self.hidden_size;
        let row = &self.lm_head_weight_data[row_start..row_start + self.hidden_size];
        let mut sum = 0.0f32;
        for (&hidden, &weight) in last_hidden.iter().zip(row.iter()) {
            sum += hidden * weight;
        }
        if v == 0 || sum > best_value {
            best_index = v;
            best_value = sum;
        }
    }
    Ok(best_index)
}
```

- [ ] **Step 4: Use the fast path only for argmax**

Replace the LM-head/sampling block in `append_tokens_inner` with this structure:

```rust
let (token_id, logits) = match self.prepared.config.sampling {
    crate::StreamingSamplingConfig::Argmax => {
        (self.sample_argmax_from_lm_head(last_hidden)?, None)
    }
    crate::StreamingSamplingConfig::TopP {
        temperature,
        top_p,
        seed,
    } => {
        let mut logits = vec![0.0f32; self.vocab_size];
        for (v, logit) in logits.iter_mut().enumerate() {
            let row_start = v * self.hidden_size;
            let row = &self.lm_head_weight_data[row_start..row_start + self.hidden_size];
            let mut sum = 0.0f32;
            for (&hidden, &weight) in last_hidden.iter().zip(row.iter()) {
                sum += hidden * weight;
            }
            *logit = sum;
        }
        let token_id = sample_top_p(&logits, temperature, top_p, seed)?;
        (token_id, Some(logits))
    }
};
```

Keep timing around this whole block so `lm_head_ms` remains comparable.

- [ ] **Step 5: Run focused runtime tests**

Run:

```bash
cargo test -p rllm-runtime llama_session -- --nocapture
cargo test -p rllm-runtime chat_session -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Run full verification**

Run:

```bash
cargo fmt --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run --quiet -- doctor
```

Expected: all commands PASS.

- [ ] **Step 7: Run post-optimization benchmark**

Run:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-postopt.md'
```

Expected: token match stays true and the report includes updated decode tok/s and phase timing.

- [ ] **Step 8: Update the post-optimization report decision**

Edit the post-optimization report:

```text
Decision: success if token histories match and decode tok/s improves or transient allocation falls without decode slowdown.
Decision: failed if token histories match but decode tok/s falls or memory increases counterproductively.
Decision: inconclusive if token histories diverge or timing evidence is incomplete.
```

Record the observed preopt vs postopt delta in `## Analysis`.

- [ ] **Step 9: Commit Task 5**

Run:

```bash
git add crates/rllm-runtime/src/models/llama/session.rs docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-postopt.md
git commit -m "feat(runtime): add llama argmax lm-head fast path"
```

---

### Task 6: Final Review and R3B Summary

**Files:**
- Modify only if needed: `docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-preopt.md`
- Modify only if needed: `docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-postopt.md`

- [ ] **Step 1: Inspect git history and working tree**

Run:

```bash
git log --oneline -6
git status --short --branch
```

Expected: branch is `codex/rama-decode-hotpath-r3`; working tree is clean after the final commit.

- [ ] **Step 2: Compare R2, R3 preopt, and R3 postopt if it exists**

Run:

```bash
rg -n "session decode tok/s|session phase timing|Decision|Reason|Paper value|lm_head|transformer" docs/benchmarks/trials/active/2026-06-14-r2-token-native-smollm2.md docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-preopt.md
test ! -f docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-postopt.md || rg -n "session decode tok/s|session phase timing|Decision|Reason|Paper value|lm_head|transformer" docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-postopt.md
```

Expected: enough evidence exists to summarize the result without opening the whole reports.

- [ ] **Step 3: Final verification**

Run:

```bash
cargo fmt --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run --quiet -- doctor
```

Expected: all commands PASS.

- [ ] **Step 4: Prepare final response**

Summarize with the measured values from the committed reports:

```text
R3B implemented on codex/rama-decode-hotpath-r3.
Verification passed: fmt, check, test, clippy, doctor.
Preopt report path: docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-preopt.md.
Postopt report path: docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-postopt.md if Task 5 ran.
Token equality result: say matched or mismatch exactly as the report shows.
Decode speed result: state the observed tok/s delta from the reports.
Next bottleneck: state the dominant measured timing bucket.
```

Do not claim 30-40 tok/s unless the measured report actually reaches that range.
