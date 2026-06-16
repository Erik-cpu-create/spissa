# R80 Q8 Prefill Scaled Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce exact Q8 Llama prefill CPU arithmetic by converting each Q8_0 block to scaled `f32` weights once per block during `batch > 1` prefill.

**Architecture:** Keep the existing streaming chunk traversal and exact math. Add a stack-local scaled-block helper in `crates/rllm-runtime/src/streaming/kernels.rs`, route only the `batch > 1` full-block Q8 path through it, and keep batch-1 decode behavior unchanged.

**Tech Stack:** Rust, existing RLLM streaming kernels, existing `cargo test` and documented benchmark harness.

---

## Files

- Modify `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add `q8_0_scaled_block`.
  - Add `f32_dot_32`.
  - In `accumulate_q8_0_chunk`, when `config.batch > 1` and a block stays within one input row with `block_len == 32`, compute scaled weights once and dot those weights against each batch input row.
  - Leave partial-row and batch-1 paths unchanged.

- Modify `crates/rllm-runtime/src/streaming/tests.rs`
  - Add a helper-level test proving `q8_0_scaled_block` applies scale and signed i8 conversion exactly.
  - Existing Q8 streaming tests must continue passing.

- Add `docs/benchmarks/trials/active/2026-06-16-r80-q8-prefill-scaled-block.md`
  - Record command, output, timings, memory, and decision.

- Modify `docs/benchmarks/trials/index.md`
  - Add R80 row after benchmark.

## Task 1: Red Test

**Files:**
- Modify `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] Add this test near the existing Q8 tests:

```rust
#[test]
fn q8_0_scaled_block_applies_scale_once() {
    let mut q = [0i8; 32];
    for (idx, value) in q.iter_mut().enumerate() {
        *value = idx as i8 - 16;
    }
    let q8 = q8_0_block_bytes(0.5, &q);

    let scaled = q8_0_scaled_block(&q8[2..34], 0.5);

    assert_eq!(scaled[0], -8.0);
    assert_eq!(scaled[16], 0.0);
    assert_eq!(scaled[31], 7.5);
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime q8_0_scaled_block_applies_scale_once
```

Expected: fail with `cannot find function q8_0_scaled_block`.

## Task 2: Green Implementation

**Files:**
- Modify `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] Add helper functions after `q8_0_dot_i8_f32`:

```rust
fn q8_0_scaled_block(qs: &[u8], scale: f32) -> [f32; 32] {
    let mut scaled = [0.0f32; 32];
    for idx in 0..32 {
        scaled[idx] = scale * (qs[idx] as i8) as f32;
    }
    scaled
}

fn f32_dot_32(weights: &[f32; 32], input: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut idx = 0usize;
    while idx < 32 {
        acc0 += weights[idx] * input[idx];
        acc1 += weights[idx + 1] * input[idx + 1];
        acc2 += weights[idx + 2] * input[idx + 2];
        acc3 += weights[idx + 3] * input[idx + 3];
        idx += 4;
    }
    (acc0 + acc1) + (acc2 + acc3)
}
```

- [ ] In `accumulate_q8_0_chunk`, replace only the full-block `config.batch > 1` case:

```rust
if config.batch > 1 && block_len == 32 && in_feature + block_len <= config.in_features {
    let scaled = q8_0_scaled_block(qs, scale);
    for batch_idx in 0..config.batch {
        let input_start = batch_idx * config.in_features + in_feature;
        let output_idx = batch_idx * config.out_features + out_feature;
        output[output_idx] += f32_dot_32(&scaled, &input[input_start..]);
    }
} else if in_feature + block_len <= config.in_features {
    ...
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime q8_0_scaled_block_applies_scale_once
```

Expected: pass.

## Task 3: Regression Tests

**Files:**
- No additional files.

- [ ] Run:

```sh
cargo test -p rllm-runtime q8_0
```

Expected: all Q8 tests pass.

- [ ] Run:

```sh
cargo test -p rllm-cli --bin llama-test
```

Expected: all `llama-test` tests pass.

## Task 4: Benchmark

**Files:**
- Modify `docs/benchmarks/trials/active/2026-06-16-r80-q8-prefill-scaled-block.md`
- Modify `docs/benchmarks/trials/index.md`

- [ ] Build release:

```sh
cargo build --release --bin llama-test
```

- [ ] Run the non-traced benchmark:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases"
```

Expected quality: output starts with `No`.

Compare against R78 baseline:

- prefill: 26.75 s
- MLP total: 20,324.73 ms
- output: `No`

## Task 5: Decision and Commit

**Files:**
- Modify benchmark report and index.

- [ ] If prefill improves without changing output, move report to `docs/benchmarks/trials/success/`.
- [ ] If prefill regresses or output changes, move report to `docs/benchmarks/trials/failed/`.
- [ ] Commit:

```sh
git add crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/*/2026-06-16-r80-q8-prefill-scaled-block.md docs/superpowers/plans/2026-06-16-r80-q8-prefill-scaled-block.md
git commit -m "perf(runtime): add q8 prefill scaled block path"
```

## Self-Review

- Spec coverage: The plan targets R79's measured CPU arithmetic bottleneck and avoids IO/decode work.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: Function names and paths match the current streaming kernel/test module layout.
