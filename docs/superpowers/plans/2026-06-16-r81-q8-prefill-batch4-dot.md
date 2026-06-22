# R81 Q8 Prefill Batch4 Dot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the accepted R80 exact Q8 prefill path by reusing each scaled Q8 block across four prompt tokens at a time.

**Architecture:** Keep R80's stack-local `[f32; 32]` scaled block. Add a batch4 dot helper that loads each scaled weight once and accumulates four output rows, then route only `config.batch > 1` full-block prefill through batch4 with scalar remainder handling.

**Tech Stack:** Rust, RLLM streaming kernels, existing benchmark harness documented under `docs/benchmarks`.

---

## Files

- Modify `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add `accumulate_f32_dot_32_batch4`.
  - Use it inside the R80 scaled-block branch of `accumulate_q8_0_chunk`.
  - Keep batch-1, partial block, multiply-into, and argmax paths unchanged.

- Modify `crates/rllm-runtime/src/streaming/tests.rs`
  - Add a helper-level test for the new batch4 function.

- Add `docs/benchmarks/trials/active/2026-06-16-r81-q8-prefill-batch4-dot.md`
  - Record tests, command, benchmark output, and decision.

- Modify `docs/benchmarks/trials/index.md`
  - Add R81 row after measurement.

## Task 1: Red Test

**Files:**
- Modify `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] Add this test after `q8_0_scaled_block_applies_scale_once`:

```rust
#[test]
fn f32_dot_32_batch4_accumulates_four_outputs() {
    let mut weights = [0.0f32; 32];
    weights[0] = 1.0;
    weights[1] = 2.0;

    let mut input = vec![0.0f32; 4 * 32];
    input[0] = 1.0;
    input[1] = 10.0;
    input[32] = 2.0;
    input[33] = 20.0;
    input[64] = 3.0;
    input[65] = 30.0;
    input[96] = 4.0;
    input[97] = 40.0;

    let mut output = vec![0.5f32, 1.5, 2.5, 3.5];

    accumulate_f32_dot_32_batch4(&weights, &input, 32, &mut output, 1, 0);

    assert_eq!(output, vec![21.5, 43.5, 65.5, 87.5]);
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime f32_dot_32_batch4_accumulates_four_outputs
```

Expected: fail with `cannot find function accumulate_f32_dot_32_batch4`.

## Task 2: Green Implementation

**Files:**
- Modify `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] Add helper after `f32_dot_32`:

```rust
fn accumulate_f32_dot_32_batch4(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    let mut acc0 = output[out_feature];
    let mut acc1 = output[output_stride + out_feature];
    let mut acc2 = output[output_stride * 2 + out_feature];
    let mut acc3 = output[output_stride * 3 + out_feature];
    let mut idx = 0usize;
    while idx < 32 {
        let weight = weights[idx];
        acc0 += weight * input[idx];
        acc1 += weight * input[input_stride + idx];
        acc2 += weight * input[input_stride * 2 + idx];
        acc3 += weight * input[input_stride * 3 + idx];
        idx += 1;
    }
    output[out_feature] = acc0;
    output[output_stride + out_feature] = acc1;
    output[output_stride * 2 + out_feature] = acc2;
    output[output_stride * 3 + out_feature] = acc3;
}
```

- [ ] Replace the R80 branch body in `accumulate_q8_0_chunk`:

```rust
let scaled = q8_0_scaled_block(qs, scale);
let mut batch_idx = 0usize;
while batch_idx + 4 <= config.batch {
    let input_start = batch_idx * config.in_features + in_feature;
    let output_start = batch_idx * config.out_features;
    accumulate_f32_dot_32_batch4(
        &scaled,
        &input[input_start..],
        config.in_features,
        &mut output[output_start..],
        config.out_features,
        out_feature,
    );
    batch_idx += 4;
}
while batch_idx < config.batch {
    let input_start = batch_idx * config.in_features + in_feature;
    let output_idx = batch_idx * config.out_features + out_feature;
    output[output_idx] += f32_dot_32(&scaled, &input[input_start..]);
    batch_idx += 1;
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime f32_dot_32_batch4_accumulates_four_outputs
```

Expected: pass.

## Task 3: Regression Tests

Run:

```sh
cargo test -p rllm-runtime q8_0
cargo test -p rllm-cli --bin llama-test
```

Expected:

- Q8 runtime tests pass.
- `llama-test` tests pass.

## Task 4: Benchmark

Run:

```sh
cargo build --release --bin llama-test
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases"
```

Expected quality: output remains `No`.

Compare against:

- R78 baseline prefill: 26.75 s
- R80 best prefill: 22.06 s
- R80 best MLP total: 17,174.73 ms

## Task 5: Decision and Commit

- If R81 improves or ties R80 without memory/output regression, move report to `success`.
- If R81 regresses R80 materially, revert runtime changes and record as `failed`.
- Commit accepted implementation and report:

```sh
git add crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/*/2026-06-16-r81-q8-prefill-batch4-dot.md docs/superpowers/plans/2026-06-16-r81-q8-prefill-batch4-dot.md
git commit -m "perf(runtime): add q8 prefill batch4 dot path"
```

## Self-Review

- Spec coverage: R81 targets R80's next inner-loop opportunity without changing model quality or memory policy.
- Placeholder scan: No placeholder steps remain.
- Type consistency: Function names and file paths match the current RLLM streaming module.
