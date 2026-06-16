# R83 Q8 Multiply-Into Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce exact Q8 Llama prefill time by applying the accepted R80/R81 scaled-block batch4 optimization to the `up_proj` multiply-into path.

**Architecture:** Keep the existing gate/up/down MLP structure. Add a helper that accumulates one scaled Q8 block into four `StreamingLinearMultiplyIntoState` batch accumulators, then route only `config.batch > 1` full-block Q8 `multiply_into` chunks through it.

**Tech Stack:** Rust, RLLM streaming Q8 kernels, existing `llama-test` benchmark harness.

---

## Files

- Modify `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add `accumulate_f32_dot_32_batch4_into`.
  - Use it in `accumulate_q8_0_chunk_multiply_into` for `config.batch > 1`, full 32-element blocks that stay within one row.
  - Keep batch-1 decode behavior unchanged.

- Modify `crates/rllm-runtime/src/streaming/tests.rs`
  - Add a helper-level test proving batch4 multiply-into accumulates into an existing accumulator vector.

- Add `docs/benchmarks/trials/active/2026-06-16-r83-q8-multiply-into-prefill.md`
  - Record tests, benchmark command, output, timing, memory, and decision.

- Modify `docs/benchmarks/trials/index.md`
  - Add R83 row after measurement.

## Task 1: Red Test

**Files:**
- Modify `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] Add this test after `f32_dot_32_batch4_accumulates_four_outputs`:

```rust
#[test]
fn f32_dot_32_batch4_into_accumulates_existing_values() {
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

    let mut accumulators = vec![0.5f32, 1.5, 2.5, 3.5];

    accumulate_f32_dot_32_batch4_into(&weights, &input, 32, &mut accumulators, 0);

    assert_eq!(accumulators, vec![21.5, 43.5, 65.5, 87.5]);
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime f32_dot_32_batch4_into_accumulates_existing_values
```

Expected: fail with `cannot find function accumulate_f32_dot_32_batch4_into`.

## Task 2: Green Implementation

**Files:**
- Modify `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] Add helper after `accumulate_f32_dot_32_batch4`:

```rust
fn accumulate_f32_dot_32_batch4_into(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    accumulators: &mut [f32],
    accumulator_start: usize,
) {
    let mut acc0 = accumulators[accumulator_start];
    let mut acc1 = accumulators[accumulator_start + 1];
    let mut acc2 = accumulators[accumulator_start + 2];
    let mut acc3 = accumulators[accumulator_start + 3];
    let mut idx = 0usize;
    while idx < 32 {
        let weight = weights[idx];
        acc0 += weight * input[idx];
        acc1 += weight * input[input_stride + idx];
        acc2 += weight * input[input_stride * 2 + idx];
        acc3 += weight * input[input_stride * 3 + idx];
        idx += 1;
    }
    accumulators[accumulator_start] = acc0;
    accumulators[accumulator_start + 1] = acc1;
    accumulators[accumulator_start + 2] = acc2;
    accumulators[accumulator_start + 3] = acc3;
}
```

- [ ] In `accumulate_q8_0_chunk_multiply_into`, add a scaled-block branch before the existing in-row branch:

```rust
if config.batch > 1 && block_len == 32 && in_feature + block_len <= config.in_features {
    advance_multiply_state_to_row(state, out_feature, config, weight_name)?;
    let scaled = q8_0_scaled_block(qs, scale);
    let mut batch_idx = 0usize;
    while batch_idx + 4 <= config.batch {
        let input_start = batch_idx * config.in_features + in_feature;
        accumulate_f32_dot_32_batch4_into(
            &scaled,
            &input[input_start..],
            config.in_features,
            &mut state.current_acc,
            batch_idx,
        );
        batch_idx += 4;
    }
    while batch_idx < config.batch {
        let input_start = batch_idx * config.in_features + in_feature;
        state.current_acc[batch_idx] += f32_dot_32(&scaled, &input[input_start..]);
        batch_idx += 1;
    }
    if in_feature + block_len == config.in_features {
        state.finish_current(config, weight_name)?;
    }
} else if in_feature + block_len <= config.in_features {
    ...
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime f32_dot_32_batch4_into_accumulates_existing_values
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
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
```

Expected quality: output remains `No`.

Compare against:

- R78 baseline prefill: 26.75 s
- R81 best verify-once prefill: 21.41 s
- R82 best unchecked prefill: 16.38 s
- R82 best unchecked MLP total: 14,096.94 ms

## Task 5: Decision and Commit

- If R83 improves R82 unchecked prefill without output or memory regression, move report to `success`.
- If R83 regresses, revert runtime changes and record the report under `failed`.
- Commit accepted implementation and report:

```sh
git add crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/*/2026-06-16-r83-q8-multiply-into-prefill.md docs/superpowers/plans/2026-06-16-r83-q8-multiply-into-prefill.md
git commit -m "perf(runtime): add q8 multiply-into prefill batch4 path"
```

## Self-Review

- Spec coverage: R83 targets the measured remaining expensive `up_proj` multiply-into path.
- Placeholder scan: No placeholder instructions remain.
- Type consistency: Helper names match the current Q8 kernel naming style.
