# R17 Experimental Speed Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in Turbo Sparse Decode for LLaMA batch-1 MLP projection so RLLM can test an original experimental-speed path without changing default exact-lowram inference.

**Architecture:** Add a small runtime speed module for env parsing, deterministic top-k activation selection, and telemetry. Wire it through session metrics, then add sparse raw-16-bit projection kernels for LLaMA MLP `gate/up` and `down` paths behind `RLLM_EXPERIMENTAL_SPEED=1`. Exact mode stays unchanged and all benchmark evidence is documented as `experimental-speed`.

**Tech Stack:** Rust workspace, `rllm-runtime`, `rllm-cli`, existing `.spsa` raw BF16/FP16 streaming kernels, `cargo test`, local benchmark reports under `docs/benchmarks/trials`.

---

## File Structure

- Create `crates/rllm-runtime/src/speed.rs`
  - Owns `RLLM_EXPERIMENTAL_SPEED`, `RLLM_TURBO_TOPK`, `RamaExperimentalSpeedConfig`, `RamaExperimentalSpeedStats`, and deterministic activation top-k selection.
- Modify `crates/rllm-runtime/src/lib.rs`
  - Adds `mod speed;` and exports the public config/stats/selectors needed by CLI tests and session metrics.
- Modify `crates/rllm-runtime/src/session.rs`
  - Adds experimental-speed stats to turn metrics and adapter collection.
- Modify `crates/rllm-runtime/src/streaming/linear.rs`
  - Adds sparse raw-16-bit batch-1 streaming projection entry points for `silu(gate) * up` and plain linear output.
- Modify `crates/rllm-runtime/src/streaming/kernels.rs`
  - Adds private sparse raw-16-bit chunk accumulators used by `linear.rs`.
- Modify `crates/rllm-runtime/src/streaming/tests.rs`
  - Adds focused unit tests for sparse gate/up and sparse down projection.
- Modify `crates/rllm-runtime/src/models/llama/generate.rs`
  - Allows the LLaMA transformer block to request sparse MLP kernels when experimental mode is enabled.
- Modify `crates/rllm-runtime/src/models/llama/session.rs`
  - Reads experimental config, passes stats through generation, and exposes stats to `RamaChatSession`.
- Modify `crates/rllm-cli/src/bin/llama-test.rs`
  - Prints experimental-speed telemetry after each turn.
- Modify `crates/rllm-cli/src/commands/chat_session_token.rs`
  - Includes experimental-speed telemetry in benchmark reports.
- Create or move `docs/benchmarks/trials/active/2026-06-14-r17-experimental-speed-mode.md`
  - Records the R17 command plan, baseline, experimental result, and decision.
- Modify `docs/benchmarks/trials/index.md`
  - Adds one R17 row after measurement and folder classification.

## Task 1: Experimental Speed Config and Stats

**Files:**
- Create: `crates/rllm-runtime/src/speed.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [ ] **Step 1: Write failing tests for env parsing, top-k selection, and stats**

Create `crates/rllm-runtime/src/speed.rs` with this test scaffold and empty production types above it:

```rust
pub const RLLM_EXPERIMENTAL_SPEED_ENV: &str = "RLLM_EXPERIMENTAL_SPEED";
pub const RLLM_TURBO_TOPK_ENV: &str = "RLLM_TURBO_TOPK";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaExperimentalSpeedConfig {
    pub enabled: bool,
    pub turbo_topk: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaExperimentalSpeedStats {
    pub sparse_projection_calls: usize,
    pub exact_fallbacks: usize,
    pub selected_topk_sum: usize,
    pub max_selected_topk: usize,
    pub estimated_skipped_madds: usize,
    pub peak_scratch_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experimental_speed_env_parser_accepts_truthy_values() {
        assert!(parse_experimental_speed_enabled(Some("1")));
        assert!(parse_experimental_speed_enabled(Some("true")));
        assert!(parse_experimental_speed_enabled(Some("yes")));
        assert!(parse_experimental_speed_enabled(Some("on")));
        assert!(!parse_experimental_speed_enabled(Some("0")));
        assert!(!parse_experimental_speed_enabled(Some("false")));
        assert!(!parse_experimental_speed_enabled(Some("")));
        assert!(!parse_experimental_speed_enabled(None));
    }

    #[test]
    fn parse_turbo_topk_keeps_only_positive_values() {
        assert_eq!(parse_turbo_topk(Some("256")), Some(256));
        assert_eq!(parse_turbo_topk(Some("1")), Some(1));
        assert_eq!(parse_turbo_topk(Some("0")), None);
        assert_eq!(parse_turbo_topk(Some("-7")), None);
        assert_eq!(parse_turbo_topk(Some("bad")), None);
        assert_eq!(parse_turbo_topk(None), None);
    }

    #[test]
    fn top_abs_indices_are_deterministic_and_sorted_for_access() {
        let input = [0.5, -4.0, 3.0, 4.0, -0.25];
        assert_eq!(select_top_abs_indices(&input, 3), vec![1, 2, 3]);
        assert_eq!(select_top_abs_indices(&input, 99), vec![0, 1, 2, 3, 4]);
        assert!(select_top_abs_indices(&input, 0).is_empty());
    }

    #[test]
    fn config_chooses_bounded_topk() {
        let config = RamaExperimentalSpeedConfig {
            enabled: true,
            turbo_topk: Some(512),
        };
        assert_eq!(config.topk_for_input(2048, 256), 512);
        assert_eq!(config.topk_for_input(128, 256), 128);

        let defaulted = RamaExperimentalSpeedConfig {
            enabled: true,
            turbo_topk: None,
        };
        assert_eq!(defaulted.topk_for_input(2048, 256), 256);
        assert_eq!(defaulted.topk_for_input(32, 256), 32);
    }

    #[test]
    fn stats_record_sparse_calls_and_merge() {
        let mut stats = RamaExperimentalSpeedStats::default();
        assert!(stats.is_empty());
        stats.record_sparse_projection(4, 16, 3, 64);
        stats.record_exact_fallback();

        let mut other = RamaExperimentalSpeedStats::default();
        other.record_sparse_projection(2, 8, 1, 32);
        stats.add_assign(other);

        assert_eq!(stats.sparse_projection_calls, 2);
        assert_eq!(stats.exact_fallbacks, 1);
        assert_eq!(stats.selected_topk_sum, 6);
        assert_eq!(stats.max_selected_topk, 4);
        assert_eq!(stats.estimated_skipped_madds, 24);
        assert_eq!(stats.peak_scratch_bytes, 64);
        assert!(!stats.is_empty());
    }
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cargo test -p rllm-runtime speed -- --nocapture
```

Expected: compile failure naming missing functions such as `parse_experimental_speed_enabled`, `parse_turbo_topk`, and `select_top_abs_indices`.

- [ ] **Step 3: Implement speed config and stats**

Replace the production section of `crates/rllm-runtime/src/speed.rs` with:

```rust
pub const RLLM_EXPERIMENTAL_SPEED_ENV: &str = "RLLM_EXPERIMENTAL_SPEED";
pub const RLLM_TURBO_TOPK_ENV: &str = "RLLM_TURBO_TOPK";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaExperimentalSpeedConfig {
    pub enabled: bool,
    pub turbo_topk: Option<usize>,
}

impl RamaExperimentalSpeedConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: parse_experimental_speed_enabled(
                std::env::var(RLLM_EXPERIMENTAL_SPEED_ENV).ok().as_deref(),
            ),
            turbo_topk: parse_turbo_topk(std::env::var(RLLM_TURBO_TOPK_ENV).ok().as_deref()),
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            turbo_topk: None,
        }
    }

    pub fn topk_for_input(self, input_len: usize, default_topk: usize) -> usize {
        if input_len == 0 {
            return 0;
        }
        self.turbo_topk
            .unwrap_or(default_topk.max(1))
            .min(input_len)
            .max(1)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamaExperimentalSpeedStats {
    pub sparse_projection_calls: usize,
    pub exact_fallbacks: usize,
    pub selected_topk_sum: usize,
    pub max_selected_topk: usize,
    pub estimated_skipped_madds: usize,
    pub peak_scratch_bytes: usize,
}

impl RamaExperimentalSpeedStats {
    pub fn add_assign(&mut self, other: Self) {
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

    pub fn is_empty(self) -> bool {
        self.sparse_projection_calls == 0
            && self.exact_fallbacks == 0
            && self.selected_topk_sum == 0
            && self.max_selected_topk == 0
            && self.estimated_skipped_madds == 0
            && self.peak_scratch_bytes == 0
    }
}

pub fn parse_experimental_speed_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn parse_turbo_topk(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

pub fn select_top_abs_indices(input: &[f32], topk: usize) -> Vec<usize> {
    let limit = topk.min(input.len());
    if limit == 0 {
        return Vec::new();
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
    let mut indices: Vec<usize> = scored
        .into_iter()
        .take(limit)
        .map(|(idx, _)| idx)
        .collect();
    indices.sort_unstable();
    indices
}
```

Modify `crates/rllm-runtime/src/lib.rs`:

```rust
mod speed;
```

Add exports next to the session exports:

```rust
pub use speed::{
    parse_experimental_speed_enabled, parse_turbo_topk, select_top_abs_indices,
    RamaExperimentalSpeedConfig, RamaExperimentalSpeedStats,
};
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
cargo test -p rllm-runtime speed -- --nocapture
```

Expected: all speed tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/rllm-runtime/src/speed.rs crates/rllm-runtime/src/lib.rs
git commit -m "feat(runtime): add experimental speed config"
```

## Task 2: Session Metrics Plumbing

**Files:**
- Modify: `crates/rllm-runtime/src/session.rs`

- [ ] **Step 1: Write failing metric aggregation test**

In `crates/rllm-runtime/src/session.rs`, add `RamaExperimentalSpeedStats` to the top import:

```rust
use crate::{MemoryBudget, RamaExperimentalSpeedStats, Result, RuntimeError};
```

In `RecordingAdapter`, add:

```rust
experimental_speed_stats: Vec<RamaExperimentalSpeedStats>,
```

Initialize it in `RecordingAdapter::new`:

```rust
experimental_speed_stats: Vec::new(),
```

Add this trait method inside `impl RamaSessionAdapter for RecordingAdapter`:

```rust
fn take_last_experimental_speed_stats(&mut self) -> Option<RamaExperimentalSpeedStats> {
    if self.experimental_speed_stats.is_empty() {
        None
    } else {
        Some(self.experimental_speed_stats.remove(0))
    }
}
```

Add this test after `turn_metrics_collect_adapter_rolling_stats`:

```rust
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
    adapter.experimental_speed_stats.push(RamaExperimentalSpeedStats {
        sparse_projection_calls: 2,
        exact_fallbacks: 1,
        selected_topk_sum: 128,
        max_selected_topk: 64,
        estimated_skipped_madds: 1024,
        peak_scratch_bytes: 512,
    });
    adapter.experimental_speed_stats.push(RamaExperimentalSpeedStats {
        sparse_projection_calls: 3,
        exact_fallbacks: 0,
        selected_topk_sum: 256,
        max_selected_topk: 128,
        estimated_skipped_madds: 2048,
        peak_scratch_bytes: 256,
    });
    let mut session = RamaChatSession::new(adapter);
    let mut budget = MemoryBudget::unbounded();

    let result = session
        .generate_turn(&[1], 2, &mut budget, |_| true)
        .unwrap();

    assert_eq!(result.generated_token_ids, [7, 8]);
    assert_eq!(result.metrics.experimental_speed_stats.sparse_projection_calls, 5);
    assert_eq!(result.metrics.experimental_speed_stats.exact_fallbacks, 1);
    assert_eq!(result.metrics.experimental_speed_stats.selected_topk_sum, 384);
    assert_eq!(result.metrics.experimental_speed_stats.max_selected_topk, 128);
    assert_eq!(
        result
            .metrics
            .experimental_speed_stats
            .estimated_skipped_madds,
        3072
    );
    assert_eq!(result.metrics.experimental_speed_stats.peak_scratch_bytes, 512);
}
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
cargo test -p rllm-runtime turn_metrics_collect_adapter_experimental_speed_stats -- --nocapture
```

Expected: compile failure because `RamaSessionTurnMetrics` and `RamaSessionAdapter` do not expose experimental-speed stats yet.

- [ ] **Step 3: Implement metrics plumbing**

In `RamaSessionTurnMetrics`, add:

```rust
pub experimental_speed_stats: RamaExperimentalSpeedStats,
```

In `RamaSessionAdapter`, add default method:

```rust
fn take_last_experimental_speed_stats(&mut self) -> Option<RamaExperimentalSpeedStats> {
    None
}
```

In `RamaChatSession::generate_turn`, add:

```rust
let mut experimental_speed_stats = RamaExperimentalSpeedStats::default();
```

After each existing rolling stats collection block, add:

```rust
if let Some(stats) = self.adapter.take_last_experimental_speed_stats() {
    experimental_speed_stats.add_assign(stats);
}
```

In every `RamaSessionTurnMetrics { ... }` literal, add:

```rust
experimental_speed_stats,
```

- [ ] **Step 4: Run focused and session tests**

Run:

```bash
cargo test -p rllm-runtime turn_metrics_collect_adapter_experimental_speed_stats -- --nocapture
cargo test -p rllm-runtime session -- --nocapture
```

Expected: both commands pass.

- [ ] **Step 5: Commit**

```bash
git add crates/rllm-runtime/src/session.rs
git commit -m "feat(runtime): collect experimental speed metrics"
```

## Task 3: Sparse Raw 16-bit MLP Kernels

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/linear.rs`
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [ ] **Step 1: Write failing sparse kernel tests**

In `crates/rllm-runtime/src/streaming/tests.rs`, add tests near the existing raw BF16 streaming tests:

```rust
#[test]
fn sparse_raw_bf16_linear_matches_manual_topk_projection() {
    let path = temp_path("sparse-linear-bf16");
    let weight_bf16 = vec![
        0x3f80, 0x4000, 0x4040, 0x4080, // row 0: 1,2,3,4
        0xbf80, 0xc000, 0xc040, 0xc080, // row 1: -1,-2,-3,-4
        0x3f00, 0x3f80, 0x4000, 0x4040, // row 2: 0.5,1,2,3
    ];
    let input = vec![1.0, -8.0, 2.0, 7.0];
    let selected = select_top_abs_indices(&input, 2);
    assert_eq!(selected, vec![1, 3]);

    let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
    add_bf16_tensor(
        &mut writer,
        0,
        "linear.sparse.bf16.weight",
        vec![3, 4],
        &weight_bf16,
        weight_bf16.len() * 2,
    );
    writer.finalize().unwrap();

    let mut model = LazyRllmModel::open(&path).unwrap();
    let mut budget = MemoryBudget::unbounded();
    let mut stats = RamaExperimentalSpeedStats::default();
    let output = streaming_sparse_tile_linear_from_model(
        &mut model,
        "linear.sparse.bf16.weight",
        &input,
        None,
        StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: 4,
                out_features: 3,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        },
        RamaExperimentalSpeedConfig {
            enabled: true,
            turbo_topk: Some(2),
        },
        &mut stats,
        &mut budget,
    )
    .unwrap()
    .unwrap();

    assert_eq!(output.len(), 3);
    assert!((output[0] - 12.0).abs() < 1e-4);
    assert!((output[1] + 12.0).abs() < 1e-4);
    assert!((output[2] - 13.0).abs() < 1e-4);
    assert_eq!(stats.sparse_projection_calls, 1);
    assert_eq!(stats.max_selected_topk, 2);
}

#[test]
fn sparse_silu_gate_up_matches_manual_topk_projection() {
    let path = temp_path("sparse-gate-up-bf16");
    let gate_bf16 = vec![
        0x3f80, 0x4000, 0x4040, 0x4080,
        0x4000, 0x4040, 0x4080, 0x40a0,
    ];
    let up_bf16 = vec![
        0x3f80, 0x3f80, 0x3f80, 0x3f80,
        0x4000, 0x4000, 0x4000, 0x4000,
    ];
    let input = vec![1.0, -8.0, 2.0, 7.0];

    let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
    add_bf16_tensor(
        &mut writer,
        0,
        "mlp.gate.sparse.bf16.weight",
        vec![2, 4],
        &gate_bf16,
        gate_bf16.len() * 2,
    );
    add_bf16_tensor(
        &mut writer,
        1,
        "mlp.up.sparse.bf16.weight",
        vec![2, 4],
        &up_bf16,
        up_bf16.len() * 2,
    );
    writer.finalize().unwrap();

    let mut model = LazyRllmModel::open(&path).unwrap();
    let mut budget = MemoryBudget::unbounded();
    let mut stats = RamaExperimentalSpeedStats::default();
    let output = streaming_sparse_silu_gate_up_from_model(
        &mut model,
        "mlp.gate.sparse.bf16.weight",
        "mlp.up.sparse.bf16.weight",
        &input,
        StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: 4,
                out_features: 2,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        },
        RamaExperimentalSpeedConfig {
            enabled: true,
            turbo_topk: Some(2),
        },
        &mut stats,
        &mut budget,
    )
    .unwrap()
    .unwrap();

    let selected = select_top_abs_indices(&input, 2);
    let gate_f32: Vec<f32> = gate_bf16
        .iter()
        .map(|bits| crate::tensor::bf16_to_f32(*bits))
        .collect();
    let up_f32: Vec<f32> = up_bf16
        .iter()
        .map(|bits| crate::tensor::bf16_to_f32(*bits))
        .collect();
    let mut expected = Vec::new();
    for row in 0..2 {
        let mut gate_acc = 0.0;
        let mut up_acc = 0.0;
        for &idx in &selected {
            gate_acc += input[idx] * gate_f32[row * 4 + idx];
            up_acc += input[idx] * up_f32[row * 4 + idx];
        }
        expected.push(crate::ops::silu(gate_acc) * up_acc);
    }

    assert_eq!(output.len(), expected.len());
    for (actual, expected) in output.iter().zip(expected) {
        assert!((*actual - expected).abs() < 1e-4);
    }
    assert_eq!(stats.sparse_projection_calls, 1);
    assert_eq!(stats.estimated_skipped_madds, 8);
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cargo test -p rllm-runtime sparse_raw_bf16_linear_matches_manual_topk_projection -- --nocapture
cargo test -p rllm-runtime sparse_silu_gate_up_matches_manual_topk_projection -- --nocapture
```

Expected: compile failure because `streaming_sparse_tile_linear_from_model` and `streaming_sparse_silu_gate_up_from_model` do not exist.

- [ ] **Step 3: Implement sparse entry points in `linear.rs`**

Add imports at the top of included code where needed:

```rust
use crate::{RamaExperimentalSpeedConfig, RamaExperimentalSpeedStats};
```

Add public functions near the existing fused MLP functions:

```rust
pub fn streaming_sparse_tile_linear_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    speed_config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, bias, config.linear)?;
    if !speed_config.enabled || config.linear.batch != 1 || config.linear.in_features == 0 {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let tensor = model.tensor(weight_name)?.clone();
    validate_weight_tensor(&tensor, config.linear)?;
    if !matches!(
        tensor.dtype,
        rllm_container::DType::Fp16 | rllm_container::DType::Bf16
    ) {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let selected = crate::select_top_abs_indices(
        input,
        speed_config.topk_for_input(config.linear.in_features, 256),
    );
    if selected.is_empty() {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    if chunks.is_empty() || chunks.iter().any(|chunk| chunk.codec_id != "rtc-raw-v1") {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let mut output = vec![0.0f32; config.linear.out_features];
    if let Some(bias) = bias {
        output.copy_from_slice(bias);
    }

    let dtype_size = tensor.dtype.size_bytes();
    let mut byte_offset = 0usize;
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} sparse stream reached unaligned byte offset {byte_offset}"
            )));
        }
        let element_start = byte_offset / dtype_size;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;

        model.with_raw_chunk(chunk.chunk_id, budget, |raw_bytes, _budget| {
            if raw_bytes.len() != expected_chunk_bytes {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "chunk {} raw byte len {} does not match metadata {}",
                    chunk.chunk_id,
                    raw_bytes.len(),
                    expected_chunk_bytes
                )));
            }
            accumulate_sparse_raw_16bit_linear_chunk_batch1(
                input,
                &selected,
                &mut output,
                raw_bytes,
                element_start,
                config.linear,
                tensor.dtype,
                weight_name,
            )
        })?;

        byte_offset = byte_offset.checked_add(expected_chunk_bytes).ok_or_else(|| {
            RuntimeError::InvalidTensorData("sparse chunk byte offset overflow".to_string())
        })?;
    }

    stats.record_sparse_projection(
        selected.len(),
        config.linear.in_features,
        config.linear.out_features,
        1,
    );
    Ok(Some(output))
}
```

Add the gate/up variant:

```rust
pub fn streaming_sparse_silu_gate_up_from_model(
    model: &mut LazyRllmModel,
    gate_weight_name: &str,
    up_weight_name: &str,
    input: &[f32],
    config: StreamingTileLinearConfig,
    speed_config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, None, config.linear)?;
    if !speed_config.enabled || config.linear.batch != 1 || config.linear.in_features == 0 {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let gate_tensor = model.tensor(gate_weight_name)?.clone();
    let up_tensor = model.tensor(up_weight_name)?.clone();
    validate_weight_tensor(&gate_tensor, config.linear)?;
    validate_weight_tensor(&up_tensor, config.linear)?;
    if gate_tensor.dtype != up_tensor.dtype
        || !matches!(
            gate_tensor.dtype,
            rllm_container::DType::Fp16 | rllm_container::DType::Bf16
        )
    {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let selected = crate::select_top_abs_indices(
        input,
        speed_config.topk_for_input(config.linear.in_features, 256),
    );
    if selected.is_empty() {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let gate_chunks: Vec<ChunkMeta> = model.chunks_for_tensor(gate_tensor.tensor_id).to_vec();
    let up_chunks: Vec<ChunkMeta> = model.chunks_for_tensor(up_tensor.tensor_id).to_vec();
    if gate_chunks.is_empty() || gate_chunks.len() != up_chunks.len() {
        stats.record_exact_fallback();
        return Ok(None);
    }
    for (gate_chunk, up_chunk) in gate_chunks.iter().zip(up_chunks.iter()) {
        if gate_chunk.codec_id != "rtc-raw-v1"
            || up_chunk.codec_id != "rtc-raw-v1"
            || gate_chunk.chunk_offset_in_tensor != up_chunk.chunk_offset_in_tensor
            || gate_chunk.uncompressed_size != up_chunk.uncompressed_size
        {
            stats.record_exact_fallback();
            return Ok(None);
        }
    }

    let mut output = vec![0.0f32; config.linear.out_features];
    let mut state = SiluGateUpState::new(&mut output);
    let dtype_size = gate_tensor.dtype.size_bytes();
    let mut byte_offset = 0usize;
    for (gate_chunk, up_chunk) in gate_chunks.iter().zip(up_chunks.iter()) {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensors {gate_weight_name}/{up_weight_name} sparse stream reached unaligned byte offset {byte_offset}"
            )));
        }
        let element_start = byte_offset / dtype_size;
        let expected_chunk_bytes = usize::try_from(gate_chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                gate_chunk.chunk_id
            ))
        })?;

        model.with_two_raw_chunks(
            gate_chunk.chunk_id,
            up_chunk.chunk_id,
            budget,
            |gate_bytes, up_bytes, _budget| {
                if gate_bytes.len() != expected_chunk_bytes || up_bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "sparse gate/up raw chunk len mismatch for chunks {}/{}",
                        gate_chunk.chunk_id, up_chunk.chunk_id
                    )));
                }
                accumulate_sparse_silu_gate_up_raw_16bit_chunk_batch1(
                    input,
                    &selected,
                    gate_bytes,
                    up_bytes,
                    element_start,
                    config.linear,
                    gate_tensor.dtype,
                    &mut state,
                    gate_weight_name,
                )
            },
        )?;

        byte_offset = byte_offset.checked_add(expected_chunk_bytes).ok_or_else(|| {
            RuntimeError::InvalidTensorData("sparse gate/up byte offset overflow".to_string())
        })?;
    }
    state.finish(config.linear, gate_weight_name)?;

    stats.record_sparse_projection(
        selected.len(),
        config.linear.in_features,
        config.linear.out_features,
        2,
    );
    Ok(Some(output))
}
```

- [ ] **Step 4: Implement private sparse accumulators in `kernels.rs`**

Add these helpers near the existing raw 16-bit helpers:

```rust
fn accumulate_sparse_raw_16bit_linear_chunk_batch1(
    input: &[f32],
    selected: &[usize],
    output: &mut [f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    weight_name: &str,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Sparse raw 16-bit stream for {weight_name} has odd length"
        )));
    }
    let weight_elements = raw_bytes.len() / 2;
    let element_end = element_start.checked_add(weight_elements).ok_or_else(|| {
        RuntimeError::Shape("sparse raw chunk element range overflow".to_string())
    })?;
    let expected = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("sparse weight element count overflow".to_string()))?;
    if element_end > expected {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} sparse chunk elements {element_start}..{element_end} exceed expected {expected}"
        )));
    }

    let first_row = element_start / config.in_features;
    let last_row = element_end.saturating_sub(1) / config.in_features;
    for out_feature in first_row..=last_row {
        let row_base = out_feature * config.in_features;
        let mut acc = output[out_feature];
        for &in_feature in selected {
            let global = row_base + in_feature;
            if global >= element_start && global < element_end {
                let local = global - element_start;
                acc += input[in_feature] * raw_16bit_weight_at(raw_bytes, local, dtype);
            }
        }
        output[out_feature] = acc;
    }
    Ok(())
}

fn accumulate_sparse_silu_gate_up_raw_16bit_chunk_batch1(
    input: &[f32],
    selected: &[usize],
    gate_bytes: &[u8],
    up_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    state: &mut SiluGateUpState<'_>,
    weight_name: &str,
) -> Result<()> {
    if !gate_bytes.len().is_multiple_of(2) || !up_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Sparse raw gate/up stream for {weight_name} has odd length"
        )));
    }
    let weight_elements = gate_bytes.len() / 2;
    let element_end = element_start.checked_add(weight_elements).ok_or_else(|| {
        RuntimeError::Shape("sparse gate/up chunk element range overflow".to_string())
    })?;
    let expected = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("sparse gate/up element count overflow".to_string()))?;
    if element_end > expected {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} sparse gate/up chunk elements {element_start}..{element_end} exceed expected {expected}"
        )));
    }

    let first_row = element_start / config.in_features;
    let last_row = element_end.saturating_sub(1) / config.in_features;
    for out_feature in first_row..=last_row {
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic sparse row {out_feature}, current {}",
                state.current_out_feature
            )));
        }

        let row_base = out_feature * config.in_features;
        for &in_feature in selected {
            let global = row_base + in_feature;
            if global >= element_start && global < element_end {
                let local = global - element_start;
                let x = input[in_feature];
                state.gate_acc += x * raw_16bit_weight_at(gate_bytes, local, dtype);
                state.up_acc += x * raw_16bit_weight_at(up_bytes, local, dtype);
            }
        }

        if element_end >= row_base + config.in_features {
            state.finish_current(config, weight_name)?;
        }
    }
    Ok(())
}
```

- [ ] **Step 5: Export sparse streaming functions**

In `crates/rllm-runtime/src/lib.rs`, add the two exports in the `pub use streaming::{...}` block:

```rust
streaming_sparse_silu_gate_up_from_model, streaming_sparse_tile_linear_from_model,
```

- [ ] **Step 6: Run sparse kernel tests**

Run:

```bash
cargo test -p rllm-runtime sparse_raw_bf16_linear_matches_manual_topk_projection -- --nocapture
cargo test -p rllm-runtime sparse_silu_gate_up_matches_manual_topk_projection -- --nocapture
cargo test -p rllm-runtime streaming -- --nocapture
```

Expected: all commands pass.

- [ ] **Step 7: Commit**

```bash
git add crates/rllm-runtime/src/streaming/linear.rs crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs crates/rllm-runtime/src/lib.rs
git commit -m "feat(runtime): add sparse raw mlp kernels"
```

## Task 4: Wire Sparse MLP Into LLaMA Session

**Files:**
- Modify: `crates/rllm-runtime/src/models/llama/generate.rs`
- Modify: `crates/rllm-runtime/src/models/llama/session.rs`

- [ ] **Step 1: Write failing LLaMA session stats test**

In `crates/rllm-runtime/src/models/llama/session.rs`, add this test after the rolling stats test:

```rust
#[test]
fn llama_session_reports_experimental_speed_stats_when_enabled_for_test() {
    let path = temp_path("experimental-speed-stats");
    write_bf16_lm_head_model(&path, 8);
    let mut model = LazyRllmModel::open(&path).unwrap();
    let prepared = prepared_with_layers(1);
    let mut budget = MemoryBudget::unbounded();
    let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
    adapter.enable_experimental_speed_for_test(RamaExperimentalSpeedConfig {
        enabled: true,
        turbo_topk: Some(1),
    });

    adapter.append_tokens(&[0], &mut budget, true).unwrap();
    let stats = adapter.take_last_experimental_speed_stats().unwrap();

    assert!(stats.sparse_projection_calls > 0);
    assert!(stats.max_selected_topk <= 1);
}
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
cargo test -p rllm-runtime llama_session_reports_experimental_speed_stats_when_enabled_for_test -- --nocapture
```

Expected: compile failure because `enable_experimental_speed_for_test` and adapter stats method are not implemented.

- [ ] **Step 3: Extend LLaMA block config and function signature**

In `crates/rllm-runtime/src/models/llama/generate.rs`, import:

```rust
use crate::{
    RamaExperimentalSpeedConfig, RamaExperimentalSpeedStats,
    streaming_sparse_silu_gate_up_from_model, streaming_sparse_tile_linear_from_model,
};
```

Add this field to `LlamaStreamingBlockConfig`:

```rust
pub experimental_speed: RamaExperimentalSpeedConfig,
```

Change `streaming_llama_transformer_block_with_timing` signature by adding:

```rust
mut experimental_speed_stats: Option<&mut RamaExperimentalSpeedStats>,
```

Update `streaming_llama_transformer_block` to pass `None` for stats:

```rust
streaming_llama_transformer_block_with_timing(
    model, input, names, params, config, budget, cache, None, None,
)
```

- [ ] **Step 4: Prefer sparse MLP kernels when enabled**

In the MLP gate/up section, replace the first fused call with:

```rust
let sparse_gate_up = if config.experimental_speed.enabled {
    if let Some(stats) = experimental_speed_stats.as_deref_mut() {
        streaming_sparse_silu_gate_up_from_model(
            model,
            &names.gate_weight,
            &names.up_weight,
            &mlp_input,
            mlp_config,
            config.experimental_speed,
            stats,
            budget,
        )?
    } else {
        None
    }
} else {
    None
};

let fused_gate_up = if sparse_gate_up.is_some() {
    sparse_gate_up
} else {
    streaming_silu_gate_up_from_model(
        model,
        &names.gate_weight,
        &names.up_weight,
        &mlp_input,
        mlp_config,
        budget,
    )?
};
```

In the down projection section, replace the exact call with:

```rust
let sparse_down = if config.experimental_speed.enabled {
    if let Some(stats) = experimental_speed_stats.as_deref_mut() {
        streaming_sparse_tile_linear_from_model(
            model,
            &names.down_weight,
            &gate,
            None,
            down_config,
            config.experimental_speed,
            stats,
            budget,
        )?
    } else {
        None
    }
} else {
    None
};
let down = if let Some(sparse_down) = sparse_down {
    sparse_down
} else {
    streaming_tile_linear_from_model(
        model,
        &names.down_weight,
        &gate,
        None,
        down_config,
        budget,
    )?
};
```

- [ ] **Step 5: Wire config and stats through session adapter**

In `crates/rllm-runtime/src/models/llama/session.rs`, add fields:

```rust
experimental_speed_config: RamaExperimentalSpeedConfig,
last_experimental_speed_stats: Option<RamaExperimentalSpeedStats>,
```

Initialize in `new`:

```rust
experimental_speed_config: RamaExperimentalSpeedConfig::from_env(),
last_experimental_speed_stats: None,
```

Add test-only helper:

```rust
#[cfg(test)]
pub(crate) fn enable_experimental_speed_for_test(
    &mut self,
    config: RamaExperimentalSpeedConfig,
) {
    self.experimental_speed_config = config;
}
```

At the start of `append_tokens_inner`, add:

```rust
let mut experimental_speed_stats = RamaExperimentalSpeedStats::default();
```

When constructing `LlamaStreamingBlockConfig`, add:

```rust
experimental_speed: self.experimental_speed_config,
```

Pass stats to the transformer block:

```rust
let experimental_stats_ref = if self.experimental_speed_config.enabled {
    Some(&mut experimental_speed_stats)
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
)?;
```

Before every successful return from `append_tokens_inner`, set:

```rust
self.last_experimental_speed_stats = Some(experimental_speed_stats);
```

In `impl RamaSessionAdapter`, add:

```rust
fn take_last_experimental_speed_stats(&mut self) -> Option<RamaExperimentalSpeedStats> {
    self.last_experimental_speed_stats.take()
}
```

- [ ] **Step 6: Run LLaMA and runtime tests**

Run:

```bash
cargo test -p rllm-runtime llama_session_reports_experimental_speed_stats_when_enabled_for_test -- --nocapture
cargo test -p rllm-runtime llama::session -- --nocapture
cargo test -p rllm-runtime -- --nocapture
```

Expected: all commands pass.

- [ ] **Step 7: Commit**

```bash
git add crates/rllm-runtime/src/models/llama/generate.rs crates/rllm-runtime/src/models/llama/session.rs
git commit -m "feat(runtime): enable llama experimental sparse mlp"
```

## Task 5: CLI and Benchmark Report Telemetry

**Files:**
- Modify: `crates/rllm-cli/src/bin/llama-test.rs`
- Modify: `crates/rllm-cli/src/commands/chat_session_token.rs`
- Create: `docs/benchmarks/trials/active/2026-06-14-r17-experimental-speed-mode.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Write failing CLI formatting tests**

In `crates/rllm-cli/src/bin/llama-test.rs`, add:

```rust
fn format_experimental_speed_suffix(stats: rllm_runtime::RamaExperimentalSpeedStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        format!(
            " | ExperimentalSpeed: sparse_calls={} fallbacks={} max_topk={} skipped_madds={} scratch={} bytes",
            stats.sparse_projection_calls,
            stats.exact_fallbacks,
            stats.max_selected_topk,
            stats.estimated_skipped_madds,
            stats.peak_scratch_bytes
        )
    }
}
```

Add tests:

```rust
#[test]
fn experimental_speed_suffix_is_empty_without_activity() {
    assert_eq!(
        format_experimental_speed_suffix(rllm_runtime::RamaExperimentalSpeedStats::default()),
        ""
    );
}

#[test]
fn experimental_speed_suffix_reports_nonzero_activity() {
    let suffix = format_experimental_speed_suffix(rllm_runtime::RamaExperimentalSpeedStats {
        sparse_projection_calls: 4,
        exact_fallbacks: 1,
        selected_topk_sum: 256,
        max_selected_topk: 128,
        estimated_skipped_madds: 2048,
        peak_scratch_bytes: 512,
    });

    assert!(suffix.contains("sparse_calls=4"));
    assert!(suffix.contains("fallbacks=1"));
    assert!(suffix.contains("max_topk=128"));
    assert!(suffix.contains("skipped_madds=2048"));
}
```

In `crates/rllm-cli/src/commands/chat_session_token.rs`, add:

```rust
fn format_experimental_speed_note(stats: rllm_runtime::RamaExperimentalSpeedStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        format!(
            " experimental_sparse_calls={} experimental_fallbacks={} experimental_max_topk={} experimental_skipped_madds={} experimental_scratch_bytes={}",
            stats.sparse_projection_calls,
            stats.exact_fallbacks,
            stats.max_selected_topk,
            stats.estimated_skipped_madds,
            stats.peak_scratch_bytes
        )
    }
}
```

Add a unit test next to `format_rolling_note_reports_nonzero_stats`:

```rust
#[test]
fn format_experimental_speed_note_reports_nonzero_stats() {
    let note = format_experimental_speed_note(rllm_runtime::RamaExperimentalSpeedStats {
        sparse_projection_calls: 4,
        exact_fallbacks: 1,
        selected_topk_sum: 256,
        max_selected_topk: 128,
        estimated_skipped_madds: 2048,
        peak_scratch_bytes: 512,
    });

    assert!(note.contains("experimental_sparse_calls=4"));
    assert!(note.contains("experimental_fallbacks=1"));
    assert!(note.contains("experimental_max_topk=128"));
}
```

- [ ] **Step 2: Run CLI tests and verify failure or unused function warning**

Run:

```bash
cargo test -p rllm-cli experimental_speed -- --nocapture
```

Expected: tests compile after the function is added, but the main output path does not yet use the suffix.

- [ ] **Step 3: Wire CLI output**

In `llama-test.rs`, change:

```rust
let rolling_suffix = format_rolling_suffix(result.metrics.rolling_stats);
```

to:

```rust
let rolling_suffix = format_rolling_suffix(result.metrics.rolling_stats);
let experimental_speed_suffix =
    format_experimental_speed_suffix(result.metrics.experimental_speed_stats);
```

Change the metrics print format to include both suffixes:

```rust
"\n[TTFT/Prefill: {:.2}s | Decode: {:.2} tok/s | E2E: {:.2} tok/s | Total: {} tokens | Context: {} tokens | Peak: {} bytes{}{}]",
```

and pass:

```rust
rolling_suffix,
experimental_speed_suffix
```

In `chat_session_token.rs`, update:

```rust
let phase_note = format_phase_timing_note(row.session_result.metrics.phase_timings)
    + &format_rolling_note(row.session_result.metrics.rolling_stats);
```

to:

```rust
let phase_note = format_phase_timing_note(row.session_result.metrics.phase_timings)
    + &format_rolling_note(row.session_result.metrics.rolling_stats)
    + &format_experimental_speed_note(row.session_result.metrics.experimental_speed_stats);
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
cargo test -p rllm-cli experimental_speed -- --nocapture
cargo test -p rllm-cli -- --nocapture
```

Expected: all CLI tests pass.

- [ ] **Step 5: Create active R17 benchmark report**

Create `docs/benchmarks/trials/active/2026-06-14-r17-experimental-speed-mode.md`:

```markdown
# Trial: R17 Experimental Speed Mode

Date: 2026-06-14
Owner: RLLM
Status: active
Folder: active

## Hypothesis

Turbo Sparse Decode can improve Llama 3.2 1B Instruct CPU-only decode speed by
reducing raw BF16 MLP projection work without changing model weights or default
exact-lowram behavior.

## Scope

- Mode: experimental-speed
- Models/artifacts: `models/SmolLM2-135M-raw.spsa`, `models/Llama-3.2-1B-Instruct-raw.spsa`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Bottleneck tag: sparse MLP projection
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Top-k: `RLLM_TURBO_TOPK=256`

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=256 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | sparse calls | fallbacks | max top-k | skipped madds | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Analysis

Pending measurement.

## Decision

active

Reason: measurement pending.

Paper value:

- pending
```

- [ ] **Step 6: Run full verification before benchmark**

Run:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all commands pass.

- [ ] **Step 7: Commit CLI and active report**

```bash
git add crates/rllm-cli/src/bin/llama-test.rs crates/rllm-cli/src/commands/chat_session_token.rs docs/benchmarks/trials/active/2026-06-14-r17-experimental-speed-mode.md
git commit -m "feat(cli): report experimental speed telemetry"
```

## Task 6: Measure and Classify R17

**Files:**
- Modify or move: `docs/benchmarks/trials/active/2026-06-14-r17-experimental-speed-mode.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Build release binary**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: release build succeeds.

- [ ] **Step 2: Measure exact Llama 1B baseline**

Run:

```bash
printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

Record:

- generated token count
- TTFT/prefill
- decode tok/s
- end-to-end tok/s
- max RSS from `/usr/bin/time -l`
- RLLM peak transient bytes from CLI output

- [ ] **Step 3: Measure experimental Llama 1B top-k 256**

Run:

```bash
printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=256 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

Record the same metrics plus experimental-speed suffix:

- sparse calls
- fallbacks
- max top-k
- skipped multiply-add estimate
- sparse scratch bytes

- [ ] **Step 4: If top-k 256 works, sweep one smaller and one larger top-k**

Run:

```bash
printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=512 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

Expected: at least one top-k setting has nonzero sparse calls and improves decode tok/s versus exact baseline.

- [ ] **Step 5: Measure SmolLM2 control**

Run:

```bash
printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

Expected: SmolLM2 remains runnable and reports experimental-speed telemetry.

- [ ] **Step 6: Update report decision**

Use this decision rule:

- `success`: Llama 1B decode speed improves at least 2x and RLLM peak transient memory does not rise beyond sparse scratch.
- `success with limitation`: speed improves less than 2x but sparse telemetry proves the path is active and memory remains controlled.
- `failed`: speed regresses, output immediately degenerates, or sparse fallbacks dominate.
- `inconclusive`: measurements are too noisy or a command cannot complete reliably.

Move the report to the matching folder with `git mv`, for example:

```bash
git mv docs/benchmarks/trials/active/2026-06-14-r17-experimental-speed-mode.md docs/benchmarks/trials/success/2026-06-14-r17-experimental-speed-mode.md
```

Update `docs/benchmarks/trials/index.md` with the final row.

- [ ] **Step 7: Final verification and commit**

Run:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all commands pass.

Commit:

```bash
git add docs/benchmarks/trials/index.md docs/benchmarks/trials
git commit -m "docs(benchmarks): record r17 experimental speed trial"
```

## Self-Review Checklist

- Spec coverage: Task 1 covers env config, top-k selection, and telemetry. Task 2 covers session metrics. Task 3 covers sparse raw projection kernels. Task 4 wires LLaMA MLP gate/up and down. Task 5 exposes CLI/report telemetry. Task 6 records benchmark evidence and classifies the trial.
- Exact mode safety: all sparse behavior is gated by `RLLM_EXPERIMENTAL_SPEED=1`; fallback paths keep exact kernels.
- Originality: all code is RLLM-native and uses existing local kernel patterns; no third-party runtime code is copied.
- Test coverage: unit tests cover parser, selector, stats merge, session aggregation, sparse linear, sparse gate/up, LLaMA adapter telemetry, and CLI formatting.
- Benchmark coverage: Llama 3.2 1B is primary; SmolLM2 is control; reports use `experimental-speed`.
