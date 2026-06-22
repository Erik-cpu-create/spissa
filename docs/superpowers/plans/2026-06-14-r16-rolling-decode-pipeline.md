# R16 Rolling Decode Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and benchmark an opt-in rolling decode experiment that reduces hot-path friction without changing model weights or increasing RLLM transient memory.

**Architecture:** Add a small runtime-owned rolling executor and telemetry path, then wire it into the LLaMA LM-head argmax path only. The first R16 implementation is opt-in through `RLLM_ROLLING=1` so the accepted default runtime stays stable until benchmark evidence proves the change.

**Tech Stack:** Rust workspace, `rllm-runtime`, `rllm-cli`, existing LLaMA streaming argmax path, existing Markdown benchmark folders.

---

## File Structure

- Create `crates/rllm-runtime/src/rolling.rs`
  - Owns rolling telemetry types, opt-in parsing, executor policy, and reusable per-call stats.
  - Does not know about LLaMA, tensors, tokenization, or CLI formatting.
- Modify `crates/rllm-runtime/src/lib.rs`
  - Registers the private `rolling` module and exports public telemetry types needed by CLI/tests.
- Modify `crates/rllm-runtime/src/session.rs`
  - Adds `RamaRollingStats` to turn metrics and adapter telemetry.
- Modify `crates/rllm-runtime/src/streaming/argmax.rs`
  - Adds an opt-in rolling argmax wrapper around the existing raw 16-bit row-block argmax kernel.
  - The default path remains unchanged when no rolling executor is provided.
- Modify `crates/rllm-runtime/src/streaming/linear.rs`
  - Adds `streaming_tile_linear_argmax_with_rolling_from_model` and keeps the existing public function as a no-rolling wrapper.
- Modify `crates/rllm-runtime/src/streaming/tests.rs`
  - Adds deterministic equality tests for rolling raw BF16 argmax.
- Modify `crates/rllm-runtime/src/models/llama/session.rs`
  - Owns an optional session-level rolling executor for LM-head argmax and reports per-turn rolling stats.
- Modify `crates/rllm-cli/src/bin/llama-test.rs`
  - Prints rolling counters when they are non-zero.
- Modify `crates/rllm-cli/src/commands/chat_session_token.rs`
  - Writes rolling counters into benchmark phase notes when they are non-zero.
- Create `docs/benchmarks/trials/active/2026-06-14-r16-rolling-decode-pipeline.md`
  - Records baseline and opt-in rolling benchmark evidence before classification.

## Task 1: Add Rolling Telemetry to Session Metrics

**Files:**
- Modify: `crates/rllm-runtime/src/session.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [ ] **Step 1: Write the failing session telemetry test**

Add this test inside `#[cfg(test)] mod tests` in `crates/rllm-runtime/src/session.rs`:

```rust
#[test]
fn turn_metrics_collect_adapter_rolling_stats() {
    let mut adapter = RecordingAdapter::new(16);
    adapter.steps = vec![Ok(Some(RamaSessionStep {
        token_id: 7,
        logits: None,
        cached_context_len_after: 1,
    }))];
    adapter.rolling_stats.push(RamaRollingStats {
        submitted_tasks: 3,
        worker_wakeups: 2,
        sequential_fallbacks: 1,
        peak_scratch_bytes: 64,
    });
    let mut session = RamaChatSession::new(adapter);
    let mut budget = MemoryBudget::unbounded();

    let result = session
        .generate_turn(&[1], 1, &mut budget, |_| true)
        .unwrap();

    assert_eq!(result.metrics.rolling_stats.submitted_tasks, 3);
    assert_eq!(result.metrics.rolling_stats.worker_wakeups, 2);
    assert_eq!(result.metrics.rolling_stats.sequential_fallbacks, 1);
    assert_eq!(result.metrics.rolling_stats.peak_scratch_bytes, 64);
}
```

Also extend the existing `RecordingAdapter` test helper with:

```rust
rolling_stats: Vec<RamaRollingStats>,
```

Initialize it in `RecordingAdapter::new`:

```rust
rolling_stats: Vec::new(),
```

Implement the adapter method in the test helper:

```rust
fn take_last_rolling_stats(&mut self) -> Option<RamaRollingStats> {
    if self.rolling_stats.is_empty() {
        None
    } else {
        Some(self.rolling_stats.remove(0))
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cargo test -p rllm-runtime turn_metrics_collect_adapter_rolling_stats -- --nocapture
```

Expected: fail to compile because `RamaRollingStats`, `rolling_stats`, and `take_last_rolling_stats` do not exist.

- [ ] **Step 3: Implement the telemetry type and aggregation**

Add near `RamaSessionPhaseTimings` in `crates/rllm-runtime/src/session.rs`:

```rust
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
```

Add the field to `RamaSessionTurnMetrics`:

```rust
pub rolling_stats: RamaRollingStats,
```

Add the default adapter method:

```rust
fn take_last_rolling_stats(&mut self) -> Option<RamaRollingStats> {
    None
}
```

Inside `generate_turn`, create:

```rust
let mut rolling_stats = RamaRollingStats::default();
```

After every existing `take_last_phase_timings` block, add:

```rust
if let Some(stats) = self.adapter.take_last_rolling_stats() {
    rolling_stats.add_assign(stats);
}
```

Set `rolling_stats` in each `RamaSessionTurnMetrics` construction:

```rust
rolling_stats,
```

Update the `pub use chat_session::{ ... }` block in `crates/rllm-runtime/src/lib.rs`:

```rust
RamaChatSession, RamaRollingStats, RamaSessionAdapter, RamaSessionPhaseTimings, RamaSessionStep,
RamaSessionTurnMetrics, RamaSessionTurnResult, RamaTransformerPhaseTimings,
```

- [ ] **Step 4: Run the telemetry test and session tests**

Run:

```bash
cargo test -p rllm-runtime turn_metrics_collect_adapter_rolling_stats -- --nocapture
cargo test -p rllm-runtime chat_session::tests -- --nocapture
```

Expected: both commands pass.

- [ ] **Step 5: Commit**

```bash
git add crates/rllm-runtime/src/session.rs crates/rllm-runtime/src/lib.rs
git commit -m "feat(runtime): add rolling telemetry to session metrics"
```

## Task 2: Add the Rolling Executor Module

**Files:**
- Create: `crates/rllm-runtime/src/rolling.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [ ] **Step 1: Write the failing rolling policy tests**

Create `crates/rllm-runtime/src/rolling.rs` with only the tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_env_parser_accepts_explicit_truthy_values() {
        assert!(parse_rolling_enabled(Some("1")));
        assert!(parse_rolling_enabled(Some("true")));
        assert!(parse_rolling_enabled(Some("yes")));
        assert!(parse_rolling_enabled(Some("on")));
        assert!(!parse_rolling_enabled(Some("0")));
        assert!(!parse_rolling_enabled(Some("false")));
        assert!(!parse_rolling_enabled(Some("")));
        assert!(!parse_rolling_enabled(None));
    }

    #[test]
    fn executor_policy_skips_small_row_counts() {
        let executor = RollingExecutor::new(RollingExecutorConfig {
            enabled: true,
            worker_count: 4,
            min_rows_per_worker: 4,
        });

        assert_eq!(executor.effective_workers_for_rows(0), 1);
        assert_eq!(executor.effective_workers_for_rows(3), 1);
        assert_eq!(executor.effective_workers_for_rows(16), 4);
        assert_eq!(executor.effective_workers_for_rows(6), 1);
    }

    #[test]
    fn executor_records_batches_and_fallbacks() {
        let mut executor = RollingExecutor::new(RollingExecutorConfig {
            enabled: true,
            worker_count: 3,
            min_rows_per_worker: 2,
        });

        executor.record_parallel_batch(3, 128);
        executor.record_sequential_fallback();
        let stats = executor.take_stats();

        assert_eq!(stats.submitted_tasks, 3);
        assert_eq!(stats.worker_wakeups, 3);
        assert_eq!(stats.sequential_fallbacks, 1);
        assert_eq!(stats.peak_scratch_bytes, 128);
        assert!(executor.take_stats().is_empty());
    }
}
```

Register the module in `crates/rllm-runtime/src/lib.rs` before running the test:

```rust
mod rolling;
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
cargo test -p rllm-runtime rolling::tests -- --nocapture
```

Expected: fail to compile because `parse_rolling_enabled`, `RollingExecutor`, and `RollingExecutorConfig` do not exist.

- [ ] **Step 3: Implement the executor module**

Replace the temporary file content with:

```rust
use crate::RamaRollingStats;

pub(crate) const RLLM_ROLLING_ENV: &str = "RLLM_ROLLING";

#[derive(Debug, Clone, Copy)]
pub(crate) struct RollingExecutorConfig {
    pub enabled: bool,
    pub worker_count: usize,
    pub min_rows_per_worker: usize,
}

impl RollingExecutorConfig {
    pub(crate) fn disabled() -> Self {
        Self {
            enabled: false,
            worker_count: 1,
            min_rows_per_worker: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RollingExecutor {
    config: RollingExecutorConfig,
    stats: RamaRollingStats,
}

impl RollingExecutor {
    pub(crate) fn new(config: RollingExecutorConfig) -> Self {
        Self {
            config: RollingExecutorConfig {
                enabled: config.enabled,
                worker_count: config.worker_count.max(1),
                min_rows_per_worker: config.min_rows_per_worker.max(1),
            },
            stats: RamaRollingStats::default(),
        }
    }

    pub(crate) fn from_env(available_workers: usize) -> Option<Self> {
        if !parse_rolling_enabled(std::env::var(RLLM_ROLLING_ENV).ok().as_deref()) {
            return None;
        }
        Some(Self::new(RollingExecutorConfig {
            enabled: true,
            worker_count: available_workers.max(1),
            min_rows_per_worker: 4,
        }))
    }

    pub(crate) fn effective_workers_for_rows(&self, rows: usize) -> usize {
        if !self.config.enabled {
            return 1;
        }
        let workers = self.config.worker_count.min(rows).max(1);
        if rows < workers.saturating_mul(self.config.min_rows_per_worker) {
            1
        } else {
            workers
        }
    }

    pub(crate) fn record_parallel_batch(&mut self, workers: usize, scratch_bytes: usize) {
        self.stats.submitted_tasks = self.stats.submitted_tasks.saturating_add(workers);
        self.stats.worker_wakeups = self.stats.worker_wakeups.saturating_add(workers);
        self.stats.peak_scratch_bytes = self.stats.peak_scratch_bytes.max(scratch_bytes);
    }

    pub(crate) fn record_sequential_fallback(&mut self) {
        self.stats.sequential_fallbacks = self.stats.sequential_fallbacks.saturating_add(1);
    }

    pub(crate) fn take_stats(&mut self) -> RamaRollingStats {
        let stats = self.stats;
        self.stats = RamaRollingStats::default();
        stats
    }
}

pub(crate) fn parse_rolling_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_env_parser_accepts_explicit_truthy_values() {
        assert!(parse_rolling_enabled(Some("1")));
        assert!(parse_rolling_enabled(Some("true")));
        assert!(parse_rolling_enabled(Some("yes")));
        assert!(parse_rolling_enabled(Some("on")));
        assert!(!parse_rolling_enabled(Some("0")));
        assert!(!parse_rolling_enabled(Some("false")));
        assert!(!parse_rolling_enabled(Some("")));
        assert!(!parse_rolling_enabled(None));
    }

    #[test]
    fn executor_policy_skips_small_row_counts() {
        let executor = RollingExecutor::new(RollingExecutorConfig {
            enabled: true,
            worker_count: 4,
            min_rows_per_worker: 4,
        });

        assert_eq!(executor.effective_workers_for_rows(0), 1);
        assert_eq!(executor.effective_workers_for_rows(3), 1);
        assert_eq!(executor.effective_workers_for_rows(16), 4);
        assert_eq!(executor.effective_workers_for_rows(6), 1);
    }

    #[test]
    fn executor_records_batches_and_fallbacks() {
        let mut executor = RollingExecutor::new(RollingExecutorConfig {
            enabled: true,
            worker_count: 3,
            min_rows_per_worker: 2,
        });

        executor.record_parallel_batch(3, 128);
        executor.record_sequential_fallback();
        let stats = executor.take_stats();

        assert_eq!(stats.submitted_tasks, 3);
        assert_eq!(stats.worker_wakeups, 3);
        assert_eq!(stats.sequential_fallbacks, 1);
        assert_eq!(stats.peak_scratch_bytes, 128);
        assert!(executor.take_stats().is_empty());
    }
}
```

- [ ] **Step 4: Run rolling module tests**

Run:

```bash
cargo test -p rllm-runtime rolling::tests -- --nocapture
```

Expected: all rolling module tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/rllm-runtime/src/lib.rs crates/rllm-runtime/src/rolling.rs
git commit -m "feat(runtime): add rolling executor policy"
```

## Task 3: Add Rolling Raw BF16 Argmax Equality Coverage

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/argmax.rs`
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] **Step 1: Write the failing rolling argmax test**

Add this test after `parallel_raw_bf16_argmax_rows_match_materialized_logits` in `crates/rllm-runtime/src/streaming/tests.rs`:

```rust
#[test]
fn rolling_raw_bf16_argmax_rows_match_materialized_logits_and_record_stats() {
    let weight_bf16 = vec![
        0x3f80, 0x0000, 0x0000, 0x0000, 0x4000, 0x0000, 0x0000, 0x0000, 0x4040, 0xbf80, 0x3f80,
        0x0000, 0x3f00, 0x3f00, 0x3f00, 0xc000, 0x0000, 0x3f80, 0x0000, 0xc040, 0x3f80, 0x3f80,
        0x3f80, 0x3f80,
    ];
    let raw = bf16_bytes(&weight_bf16);
    let weight_f32: Vec<f32> = weight_bf16
        .iter()
        .map(|bits| crate::tensor::bf16_to_f32(*bits))
        .collect();
    let input = vec![1.0, -2.0, 0.5];
    let bias = vec![0.0, 0.25, -0.25, 0.5, 0.0, 1.0, -1.0, 0.75];
    let config = StreamingLinearConfig {
        batch: 1,
        in_features: 3,
        out_features: 8,
    };
    let expected_logits = linear(&input, &weight_f32, Some(&bias), 1, 3, 8).unwrap();
    let expected = sample_argmax(&expected_logits).unwrap();
    let mut executor = crate::rolling::RollingExecutor::new(crate::rolling::RollingExecutorConfig {
        enabled: true,
        worker_count: 4,
        min_rows_per_worker: 1,
    });

    let actual = rolling_raw_16bit_argmax_rows(
        &input,
        &raw,
        0,
        0,
        8,
        config,
        DType::Bf16,
        Some(&bias),
        &mut executor,
    );
    let stats = executor.take_stats();

    assert_eq!(actual.best_index, expected);
    assert_eq!(stats.submitted_tasks, 4);
    assert_eq!(stats.worker_wakeups, 4);
    assert_eq!(stats.sequential_fallbacks, 0);
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cargo test -p rllm-runtime rolling_raw_bf16_argmax_rows_match_materialized_logits_and_record_stats -- --nocapture
```

Expected: fail to compile because `rolling_raw_16bit_argmax_rows` does not exist.

- [ ] **Step 3: Implement the rolling argmax wrapper**

In `crates/rllm-runtime/src/streaming/argmax.rs`, add this import at the top of the included module content:

```rust
use crate::rolling::RollingExecutor;
```

Add this function near `parallel_raw_16bit_argmax_rows`:

```rust
fn rolling_raw_16bit_argmax_rows(
    input: &[f32],
    raw_bytes: &[u8],
    local_row_start: usize,
    out_feature_start: usize,
    rows: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    bias: Option<&[f32]>,
    executor: &mut RollingExecutor,
) -> ArgmaxCandidate {
    let workers = executor.effective_workers_for_rows(rows);
    if workers == 1 {
        executor.record_sequential_fallback();
        return raw_16bit_argmax_rows_range(
            input,
            raw_bytes,
            local_row_start,
            out_feature_start,
            rows,
            config,
            dtype,
            bias,
        );
    }

    executor.record_parallel_batch(workers, std::mem::size_of::<ArgmaxCandidate>() * workers);
    parallel_raw_16bit_argmax_rows(
        input,
        raw_bytes,
        local_row_start,
        out_feature_start,
        rows,
        config,
        dtype,
        bias,
        workers,
    )
}
```

- [ ] **Step 4: Run targeted streaming tests**

Run:

```bash
cargo test -p rllm-runtime rolling_raw_bf16_argmax_rows_match_materialized_logits_and_record_stats -- --nocapture
cargo test -p rllm-runtime parallel_raw_bf16_argmax_rows_match_materialized_logits -- --nocapture
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/rllm-runtime/src/streaming/argmax.rs crates/rllm-runtime/src/streaming/tests.rs
git commit -m "feat(runtime): add rolling raw argmax wrapper"
```

## Task 4: Thread Rolling Through Streaming Argmax

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/argmax.rs`
- Modify: `crates/rllm-runtime/src/streaming/linear.rs`
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [ ] **Step 1: Write the failing end-to-end streaming test**

Add this test after `streaming_tile_linear_argmax_uses_raw_bf16_batch1_path`:

```rust
#[test]
fn streaming_tile_linear_argmax_with_rolling_records_stats() {
    let path = temp_path("tile-linear-argmax-bf16-rolling");
    let weight_bf16 = vec![
        0x3f00, 0xbf80, 0x4000, 0xc000, 0x3e80, 0x3f00, 0x3f80, 0x3f80, 0xbf80, 0x0000, 0xbf00,
        0x3f40, 0x3f80, 0x4000, 0x4040, 0xc040, 0x3f00, 0x3e80, 0x3f00, 0x3f80, 0x4000, 0x4040,
        0x4080, 0x40a0,
    ];
    let weight_f32: Vec<f32> = weight_bf16
        .iter()
        .map(|bits| crate::tensor::bf16_to_f32(*bits))
        .collect();
    let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
    add_bf16_tensor(
        &mut writer,
        0,
        "linear.argmax.bf16.rolling.weight",
        vec![8, 3],
        &weight_bf16,
        weight_bf16.len() * 2,
    );
    writer.finalize().unwrap();

    let input = vec![1.5, -2.0, 0.25];
    let bias = vec![0.0, 0.5, -1.0, 4.0, 0.25, -0.25, 0.75, 1.25];
    let logits = linear(&input, &weight_f32, Some(&bias), 1, 3, 8).unwrap();
    let expected = sample_argmax(&logits).unwrap();
    let mut model = LazyRllmModel::open(&path).unwrap();
    let mut budget = MemoryBudget::new(64);
    let mut executor = crate::rolling::RollingExecutor::new(crate::rolling::RollingExecutorConfig {
        enabled: true,
        worker_count: 4,
        min_rows_per_worker: 1,
    });

    let actual = streaming_tile_linear_argmax_with_rolling_from_model(
        &mut model,
        "linear.argmax.bf16.rolling.weight",
        &input,
        Some(&bias),
        StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: 3,
                out_features: 8,
            },
            tile_elements: 2,
        },
        &mut budget,
        Some(&mut executor),
    )
    .unwrap();
    let stats = executor.take_stats();

    assert_eq!(actual, expected);
    assert!(stats.submitted_tasks > 0);
    assert_eq!(budget.current_bytes(), 0);

    std::fs::remove_file(&path).ok();
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cargo test -p rllm-runtime streaming_tile_linear_argmax_with_rolling_records_stats -- --nocapture
```

Expected: fail to compile because `streaming_tile_linear_argmax_with_rolling_from_model` does not exist.

- [ ] **Step 3: Add optional rolling parameters through the raw argmax path**

Change `accumulate_raw_16bit_chunk_argmax` and `accumulate_raw_16bit_chunk_argmax_row_blocked` signatures in `argmax.rs`:

```rust
fn accumulate_raw_16bit_chunk_argmax(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    state: &mut StreamingLinearArgmaxState<'_>,
    weight_name: &str,
    rolling: Option<&mut RollingExecutor>,
) -> Result<()>
```

Inside the full-row block, replace the candidate selection with:

```rust
let candidate = if let Some(executor) = rolling {
    rolling_raw_16bit_argmax_rows(
        input,
        raw_bytes,
        local_idx,
        out_feature,
        full_rows,
        config,
        dtype,
        state.bias,
        executor,
    )
} else {
    parallel_raw_16bit_argmax_rows(
        input,
        raw_bytes,
        local_idx,
        out_feature,
        full_rows,
        config,
        dtype,
        state.bias,
        worker_count,
    )
};
```

Pass `rolling` through from the wrapper function to the row-blocked function.

- [ ] **Step 4: Rename the existing argmax body and add the no-rolling wrapper**

In `crates/rllm-runtime/src/streaming/linear.rs`, rename the current
`streaming_tile_linear_argmax_from_model` function to
`streaming_tile_linear_argmax_with_rolling_from_model` and add the
`rolling` parameter to that renamed function. The renamed function keeps the
same body that currently starts with `validate_tile_linear_config(config)?;`.
Then add this no-rolling wrapper above it:

```rust
pub fn streaming_tile_linear_argmax_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<usize> {
    streaming_tile_linear_argmax_with_rolling_from_model(
        model,
        weight_name,
        input,
        bias,
        config,
        budget,
        None,
    )
}
```

The renamed function signature must be:

```rust
pub fn streaming_tile_linear_argmax_with_rolling_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    budget: &mut MemoryBudget,
    mut rolling: Option<&mut RollingExecutor>,
) -> Result<usize>
```

After the rename, the first two executable lines inside the function remain:

```rust
validate_tile_linear_config(config)?;
validate_linear_shapes(input, bias, config.linear)?;
```

Inside the raw chunk branch, call:

```rust
accumulate_raw_16bit_chunk_argmax(
    input,
    compressed_bytes,
    element_start,
    config.linear,
    tensor.dtype,
    &mut state,
    weight_name,
    rolling.as_deref_mut(),
)
```

Export the new function in `crates/rllm-runtime/src/lib.rs`:

```rust
streaming_tile_linear_argmax_from_model, streaming_tile_linear_argmax_with_rolling_from_model,
```

- [ ] **Step 5: Run streaming tests**

Run:

```bash
cargo test -p rllm-runtime streaming_tile_linear_argmax_with_rolling_records_stats -- --nocapture
cargo test -p rllm-runtime streaming_tile_linear_argmax_uses_raw_bf16_batch1_path -- --nocapture
cargo test -p rllm-runtime streaming_tile_linear_argmax_matches_full_logits_across_split_rows -- --nocapture
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/rllm-runtime/src/streaming/argmax.rs crates/rllm-runtime/src/streaming/linear.rs crates/rllm-runtime/src/streaming/tests.rs crates/rllm-runtime/src/lib.rs
git commit -m "feat(runtime): thread rolling executor through argmax"
```

## Task 5: Enable Opt-In LLaMA Session Rolling

**Files:**
- Modify: `crates/rllm-runtime/src/models/llama/session.rs`

- [ ] **Step 1: Write the failing LLaMA session test**

Add this test near the other LLaMA session tests:

```rust
#[test]
fn llama_session_reports_rolling_stats_when_executor_is_enabled() {
    let path = temp_path("llama-session-rolling");
    write_constructor_model(&path, vec![8, 4]);
    let mut model = LazyRllmModel::open(&path).unwrap();
    let prepared = tiny_prepared(16, 1);
    let mut budget = MemoryBudget::unbounded();
    let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
    adapter.enable_rolling_executor_for_test(4, 1);

    let _ = adapter.append_tokens(&[1], &mut budget, true).unwrap();
    let stats = adapter.take_last_rolling_stats().unwrap();

    assert!(stats.submitted_tasks > 0);
    assert!(stats.worker_wakeups > 0);

    std::fs::remove_file(&path).ok();
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cargo test -p rllm-runtime llama_session_reports_rolling_stats_when_executor_is_enabled -- --nocapture
```

Expected: fail to compile because the adapter has no rolling executor field, no test enable method, and no rolling telemetry method.

- [ ] **Step 3: Add rolling state to `LlamaRamaSessionAdapter`**

Update imports in `session.rs`:

```rust
use crate::rolling::{RollingExecutor, RollingExecutorConfig};
use crate::{
    embedding_lookup, rms_norm, sample_top_p, streaming_tile_linear_argmax_from_model,
    streaming_tile_linear_argmax_with_rolling_from_model, streaming_tile_linear_from_model,
    LazyRllmModel, MemoryBudget, Result, RuntimeError,
};
use crate::{
    RamaRollingStats, RamaSessionAdapter, RamaSessionPhaseTimings, RamaSessionStep,
    RamaTransformerPhaseTimings, StreamingLinearConfig, StreamingTileLinearConfig,
    DEFAULT_STREAMING_TILE_ELEMENTS,
};
```

Add fields:

```rust
rolling_executor: Option<RollingExecutor>,
last_rolling_stats: Option<RamaRollingStats>,
```

Add this public-crate helper in `streaming/mod.rs`:

```rust
pub(crate) fn streaming_available_threads() -> usize {
    available_runtime_threads()
}
```

Use it from session as:

```rust
rolling_executor: RollingExecutor::from_env(crate::streaming::streaming_available_threads()),
last_rolling_stats: None,
```

Add the test-only enable method:

```rust
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
```

- [ ] **Step 4: Use rolling only for argmax LM-head**

In the `StreamingSamplingConfig::Argmax` branch, replace the argmax call with:

```rust
let token_id = if let Some(executor) = self.rolling_executor.as_mut() {
    let token = streaming_tile_linear_argmax_with_rolling_from_model(
        self.model,
        &self.prepared.lm_head_weight,
        &last_hidden,
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
        &last_hidden,
        None,
        lm_head_config,
        budget,
    )?
};
```

Keep the existing non-argmax/logits path unchanged.

Implement the adapter method:

```rust
fn take_last_rolling_stats(&mut self) -> Option<RamaRollingStats> {
    self.last_rolling_stats.take()
}
```

- [ ] **Step 5: Run LLaMA session and streaming tests**

Run:

```bash
cargo test -p rllm-runtime llama_session_reports_rolling_stats_when_executor_is_enabled -- --nocapture
cargo test -p rllm-runtime models::llama::session::tests -- --nocapture
cargo test -p rllm-runtime streaming::tests::streaming_tile_linear_argmax_with_rolling_records_stats -- --nocapture
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/rllm-runtime/src/models/llama/session.rs crates/rllm-runtime/src/streaming/mod.rs
git commit -m "feat(runtime): enable opt-in llama rolling argmax"
```

## Task 6: Expose Rolling Counters in CLI Output

**Files:**
- Modify: `crates/rllm-cli/src/bin/llama-test.rs`
- Modify: `crates/rllm-cli/src/commands/chat_session_token.rs`

- [ ] **Step 1: Write failing CLI formatting tests**

In `crates/rllm-cli/src/bin/llama-test.rs`, add:

```rust
fn format_rolling_suffix(stats: rllm_runtime::RamaRollingStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        format!(
            " | Rolling: tasks={} wakeups={} fallbacks={} scratch={} bytes",
            stats.submitted_tasks,
            stats.worker_wakeups,
            stats.sequential_fallbacks,
            stats.peak_scratch_bytes
        )
    }
}
```

Add tests:

```rust
#[test]
fn rolling_suffix_is_empty_without_activity() {
    assert_eq!(format_rolling_suffix(rllm_runtime::RamaRollingStats::default()), "");
}

#[test]
fn rolling_suffix_reports_nonzero_activity() {
    let suffix = format_rolling_suffix(rllm_runtime::RamaRollingStats {
        submitted_tasks: 8,
        worker_wakeups: 8,
        sequential_fallbacks: 1,
        peak_scratch_bytes: 64,
    });
    assert!(suffix.contains("tasks=8"));
    assert!(suffix.contains("fallbacks=1"));
}
```

In `crates/rllm-cli/src/commands/chat_session_token.rs`, extend `format_phase_timing_note` or add a sibling helper:

```rust
fn format_rolling_note(stats: rllm_runtime::RamaRollingStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        format!(
            " rolling_tasks={} rolling_wakeups={} rolling_fallbacks={} rolling_scratch_bytes={}",
            stats.submitted_tasks,
            stats.worker_wakeups,
            stats.sequential_fallbacks,
            stats.peak_scratch_bytes
        )
    }
}
```

Add a unit test:

```rust
#[test]
fn format_rolling_note_reports_nonzero_stats() {
    let note = format_rolling_note(rllm_runtime::RamaRollingStats {
        submitted_tasks: 4,
        worker_wakeups: 4,
        sequential_fallbacks: 2,
        peak_scratch_bytes: 32,
    });

    assert!(note.contains("rolling_tasks=4"));
    assert!(note.contains("rolling_fallbacks=2"));
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cargo test -p rllm-cli rolling_suffix -- --nocapture
cargo test -p rllm-cli format_rolling_note -- --nocapture
```

Expected: fail until the helper functions are placed correctly and wired into output formatting.

- [ ] **Step 3: Wire helpers into output**

In `llama-test.rs`, append the suffix to the metrics line:

```rust
let rolling_suffix = format_rolling_suffix(result.metrics.rolling_stats);
println!(
    "\n[TTFT/Prefill: {:.2}s | Decode: {:.2} tok/s | E2E: {:.2} tok/s | Total: {} tokens | Context: {} tokens | Peak: {} bytes{}]",
    result.metrics.ttft_ms / 1000.0,
    result.metrics.decode_tok_s,
    result.metrics.end_to_end_tok_s,
    result.metrics.generated_tokens,
    session.token_history().len(),
    result.metrics.peak_transient_bytes,
    rolling_suffix
);
```

In `chat_session_token.rs`, append `format_rolling_note(result.session_result.metrics.rolling_stats)` to the existing phase timing note string before writing the Markdown row.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
cargo test -p rllm-cli rolling_suffix -- --nocapture
cargo test -p rllm-cli format_rolling_note -- --nocapture
cargo test -p rllm-cli chat_session_token::tests -- --nocapture
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/rllm-cli/src/bin/llama-test.rs crates/rllm-cli/src/commands/chat_session_token.rs
git commit -m "feat(cli): report rolling decode counters"
```

## Task 7: Benchmark and Classify R16

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-14-r16-rolling-decode-pipeline.md`
- Modify: `docs/benchmarks/trials/index.md`
- Move after analysis: from `active/` to `success/`, `failed/`, or `inconclusive/`

- [ ] **Step 1: Build release binary**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: release build succeeds.

- [ ] **Step 2: Run SmolLM2 baseline and rolling trial**

Run baseline:

```bash
printf 'good morning\nexit\n' | \
  RLLM_THREADS=1 /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

Run rolling:

```bash
printf 'good morning\nexit\n' | \
  RLLM_ROLLING=1 /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

Expected: both runs complete and print identical generated token counts. The rolling run prints non-zero rolling counters.

- [ ] **Step 3: Run Llama 3.2 1B baseline and rolling trial**

Run baseline:

```bash
printf 'good morning\nexit\n' | \
  RLLM_THREADS=1 /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

Run rolling:

```bash
printf 'good morning\nexit\n' | \
  RLLM_ROLLING=1 /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

Expected: both runs complete. If Llama 1B takes too long, reduce `--max-new-tokens` to `4` and record the reason in the report.

- [ ] **Step 4: Write the active benchmark report**

Create the report using this structure:

```markdown
# Trial: R16 Rolling Decode Pipeline

Date: 2026-06-14
Owner: RLLM
Status: active
Folder: active

## Hypothesis

An opt-in rolling executor can reduce decode friction by reusing scheduling
state and measuring worker activity without duplicating model tensors, KV cache,
or full logits buffers.

## Scope

- Mode: exact-lowram
- Models/artifacts: `models/SmolLM2-135M-raw.spsa`, `models/Llama-3.2-1B-Instruct-raw.spsa`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Bottleneck tag: rolling decode pipeline

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | rolling tasks | rolling fallbacks | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Analysis

Add one measured row for each command from Steps 2-3. Compare decode tok/s
first, then RAM/RSS, then rolling counters.

## Decision

active

## Next Experiment

Classify after analysis.
```

Do not commit the report until the results table contains measured rows for the
commands that actually ran.

- [ ] **Step 5: Classify the report**

Use these rules:

- Move to `success/` only if token counts are stable, rolling counters are non-zero, RLLM peak transient memory stays flat, SmolLM2 decode improves, and Llama 1B does not regress.
- Move to `failed/` if decode regresses or rolling counters show the mechanism is active but not useful.
- Move to `inconclusive/` if counters are zero, token generation fails, or measurements are too noisy for a clear decision.

Update `docs/benchmarks/trials/index.md` with one R16 row after moving the report.

- [ ] **Step 6: Commit benchmark evidence**

```bash
git add docs/benchmarks/trials/index.md docs/benchmarks/trials/active/2026-06-14-r16-rolling-decode-pipeline.md
git commit -m "docs(benchmarks): record r16 rolling decode trial"
```

If the report was moved out of `active/`, stage the moved path instead:

```bash
git add docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-14-r16-rolling-decode-pipeline.md
git commit -m "docs(benchmarks): record r16 rolling decode trial"
```

Use `failed/` or `inconclusive/` in the path when that is the actual classification.

## Task 8: Full Verification and Merge Decision

**Files:**
- Inspect: full workspace

- [ ] **Step 1: Run full verification**

Run:

```bash
cargo fmt --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run --quiet -- doctor
```

Expected: all commands exit `0`.

- [ ] **Step 2: Decide whether code stays**

Use the R16 benchmark decision:

- If `success`, keep the opt-in rolling implementation and consider a separate R17/R16B default-enable trial.
- If `failed`, revert runtime and CLI code from Tasks 1-6 while keeping the failed benchmark report and index update.
- If `inconclusive`, keep only telemetry code if it helps future benchmark clarity; otherwise revert runtime and CLI code and keep the inconclusive report.

- [ ] **Step 3: Commit cleanup when needed**

If code is reverted after a failed or inconclusive trial, commit the cleanup:

```bash
git add crates/rllm-runtime crates/rllm-cli docs/benchmarks/trials
git commit -m "chore(runtime): remove rejected r16 rolling code"
```

- [ ] **Step 4: Final status check**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: working tree is clean, and the latest commits show the R16 code/evidence path that matches the benchmark decision.

## Self-Review Notes

- Spec coverage: the plan covers cyclic reuse through a rolling executor, no weight compression, no external runtime copying, benchmark classification, and low-RAM telemetry.
- Scope control: the first implementation touches only LM-head argmax. Transformer projection, packed layout, and SIMD are left for later R stages.
- Risk control: `RLLM_ROLLING=1` keeps default behavior unchanged until evidence supports enabling it.
