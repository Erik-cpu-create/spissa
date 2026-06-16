# R85 Q8 MLP Batch8 Direct Dot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Llama 3.2 1B Q8 exact-lowram prefill time by optimizing the shared Q8 MLP dot path used by gate/up/down without increasing resident RAM.

**Architecture:** R85 keeps the current `.rllm` container, Q8_0 block format, and streaming low-RAM execution. It adds scalar batch8 direct-dot micro-kernels for Q8_0 blocks so prefill can process eight prompt rows per 32-weight block without first materializing a temporary `[f32; 32]` scaled block. The first implementation is portable Rust; target-specific SIMD is left for a later stage only if this slice proves the bottleneck and correctness remain stable.

**Tech Stack:** Rust, `rllm-runtime` streaming kernels, Q8_0 block weights, `llama-test`, benchmark docs.

---

## Evidence From R84

R84 measured:

- Ollama CPU-only prompt eval: `0.285813s / 34 prompt tokens`
- RLLM unchecked prefill: `13.94s / 55 context tokens`
- RLLM trace `chunk_compute_closure`: `11836.07ms`
- RLLM trace `chunk_read`: `3.12ms`
- MLP buckets: down `3354.26ms`, gate `3337.44ms`, up `3102.48ms`

Therefore R85 must target the shared Q8 MLP compute path, not IO, checksum, or a single projection.

## Success Gate

R85 is accepted only if all conditions hold:

- Output sanity prompt remains `No`.
- Internal peak transient remains `1050673152 bytes` or lower.
- Best of three unchecked prefill runs beats the R83 best `11.45s`.
- Prefill MLP total decreases versus the R84 measured `10703.88ms`.
- The benchmark report records success or failure honestly in `docs/benchmarks/trials`.

If the patch only beats the noisy R84 run `13.94s` but does not beat `11.45s`, mark the trial inconclusive or failed, not success.

## Files

- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add batch8 direct Q8_0 dot helpers.
  - Use them in normal linear and multiply-into Q8_0 block loops.

- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
  - Add unit tests for batch8 direct helpers.

- Create: `docs/benchmarks/trials/active/2026-06-16-r85-q8-mlp-batch8-direct-dot.md`
  - Record implementation, commands, raw output, and decision.

- Modify: `docs/benchmarks/trials/index.md`
  - Add the R85 trial row after measurement.

## Task 1: Add Active Benchmark Report

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-16-r85-q8-mlp-batch8-direct-dot.md`

- [ ] Create the report skeleton:

```markdown
# R85: Q8 MLP Batch8 Direct Dot

## Status

Active.

## Hypothesis

RLLM prefill is dominated by shared Q8 MLP compute. Processing eight prompt
rows per Q8_0 block directly, without materializing `[f32; 32]` for every
block, should reduce gate/up/down time while keeping exact output and low RAM.

## Baseline

- R83 best unchecked prefill: `11.45s`
- R84 measured unchecked prefill: `13.94s`
- R84 MLP total: `10703.88ms`
- R84 peak transient: `1050673152 bytes`
- R84 output: `No`

## Commands

Pending.

## Results

Pending.

## Decision

Pending.
```

- [ ] Verify the report exists:

```sh
test -f docs/benchmarks/trials/active/2026-06-16-r85-q8-mlp-batch8-direct-dot.md
```

Expected: exit code `0`.

## Task 2: Write Batch8 Unit Tests First

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] Add tests near the existing Q8 helper tests:

```rust
#[test]
fn q8_0_dot_32_batch8_accumulates_eight_outputs() {
    let mut q = [0i8; 32];
    q[0] = 1;
    q[1] = 2;
    let q8 = q8_0_block_bytes(0.5, &q);

    let mut input = vec![0.0f32; 8 * 32];
    for batch_idx in 0..8 {
        input[batch_idx * 32] = (batch_idx + 1) as f32;
        input[batch_idx * 32 + 1] = ((batch_idx + 1) * 10) as f32;
    }

    let mut output = vec![1.0f32; 8];

    accumulate_q8_0_dot_32_batch8(
        &q8[2..34],
        0.5,
        &input,
        32,
        &mut output,
        1,
        0,
    );

    assert_eq!(
        output,
        vec![11.5, 22.0, 32.5, 43.0, 53.5, 64.0, 74.5, 85.0]
    );
}

#[test]
fn q8_0_dot_32_batch8_into_accumulates_existing_values() {
    let mut q = [0i8; 32];
    q[0] = 1;
    q[1] = 2;
    let q8 = q8_0_block_bytes(0.5, &q);

    let mut input = vec![0.0f32; 8 * 32];
    for batch_idx in 0..8 {
        input[batch_idx * 32] = (batch_idx + 1) as f32;
        input[batch_idx * 32 + 1] = ((batch_idx + 1) * 10) as f32;
    }

    let mut accumulators = vec![1.0f32; 8];

    accumulate_q8_0_dot_32_batch8_into(
        &q8[2..34],
        0.5,
        &input,
        32,
        &mut accumulators,
        0,
    );

    assert_eq!(
        accumulators,
        vec![11.5, 22.0, 32.5, 43.0, 53.5, 64.0, 74.5, 85.0]
    );
}
```

- [ ] Run the new tests and confirm they fail because helpers do not exist:

```sh
cargo test -p rllm-runtime q8_0_dot_32_batch8 -- --nocapture
```

Expected: compile failure mentioning missing `accumulate_q8_0_dot_32_batch8` and `accumulate_q8_0_dot_32_batch8_into`.

## Task 3: Implement Batch8 Direct Q8 Helpers

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] Add helpers after `accumulate_f32_dot_32_batch4_into`:

```rust
fn accumulate_q8_0_dot_32_batch8(
    qs: &[u8],
    scale: f32,
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
    let mut acc4 = output[output_stride * 4 + out_feature];
    let mut acc5 = output[output_stride * 5 + out_feature];
    let mut acc6 = output[output_stride * 6 + out_feature];
    let mut acc7 = output[output_stride * 7 + out_feature];

    let mut idx = 0usize;
    while idx < 32 {
        let weight = scale * (qs[idx] as i8) as f32;
        acc0 += weight * input[idx];
        acc1 += weight * input[input_stride + idx];
        acc2 += weight * input[input_stride * 2 + idx];
        acc3 += weight * input[input_stride * 3 + idx];
        acc4 += weight * input[input_stride * 4 + idx];
        acc5 += weight * input[input_stride * 5 + idx];
        acc6 += weight * input[input_stride * 6 + idx];
        acc7 += weight * input[input_stride * 7 + idx];
        idx += 1;
    }

    output[out_feature] = acc0;
    output[output_stride + out_feature] = acc1;
    output[output_stride * 2 + out_feature] = acc2;
    output[output_stride * 3 + out_feature] = acc3;
    output[output_stride * 4 + out_feature] = acc4;
    output[output_stride * 5 + out_feature] = acc5;
    output[output_stride * 6 + out_feature] = acc6;
    output[output_stride * 7 + out_feature] = acc7;
}

fn accumulate_q8_0_dot_32_batch8_into(
    qs: &[u8],
    scale: f32,
    input: &[f32],
    input_stride: usize,
    accumulators: &mut [f32],
    accumulator_start: usize,
) {
    let mut acc0 = accumulators[accumulator_start];
    let mut acc1 = accumulators[accumulator_start + 1];
    let mut acc2 = accumulators[accumulator_start + 2];
    let mut acc3 = accumulators[accumulator_start + 3];
    let mut acc4 = accumulators[accumulator_start + 4];
    let mut acc5 = accumulators[accumulator_start + 5];
    let mut acc6 = accumulators[accumulator_start + 6];
    let mut acc7 = accumulators[accumulator_start + 7];

    let mut idx = 0usize;
    while idx < 32 {
        let weight = scale * (qs[idx] as i8) as f32;
        acc0 += weight * input[idx];
        acc1 += weight * input[input_stride + idx];
        acc2 += weight * input[input_stride * 2 + idx];
        acc3 += weight * input[input_stride * 3 + idx];
        acc4 += weight * input[input_stride * 4 + idx];
        acc5 += weight * input[input_stride * 5 + idx];
        acc6 += weight * input[input_stride * 6 + idx];
        acc7 += weight * input[input_stride * 7 + idx];
        idx += 1;
    }

    accumulators[accumulator_start] = acc0;
    accumulators[accumulator_start + 1] = acc1;
    accumulators[accumulator_start + 2] = acc2;
    accumulators[accumulator_start + 3] = acc3;
    accumulators[accumulator_start + 4] = acc4;
    accumulators[accumulator_start + 5] = acc5;
    accumulators[accumulator_start + 6] = acc6;
    accumulators[accumulator_start + 7] = acc7;
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime q8_0_dot_32_batch8 -- --nocapture
```

Expected: both new tests pass.

## Task 4: Wire Batch8 Into Q8 Normal Linear

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] In `accumulate_q8_0_chunk`, replace the `config.batch > 1` full-block branch with batch8 first, then existing batch4, then scalar tail:

```rust
if config.batch > 1 && block_len == 32 && in_feature + block_len <= config.in_features {
    let mut batch_idx = 0usize;
    while batch_idx + 8 <= config.batch {
        let input_start = batch_idx * config.in_features + in_feature;
        let output_start = batch_idx * config.out_features;
        accumulate_q8_0_dot_32_batch8(
            qs,
            scale,
            &input[input_start..],
            config.in_features,
            &mut output[output_start..],
            config.out_features,
            out_feature,
        );
        batch_idx += 8;
    }
    if batch_idx + 4 <= config.batch {
        let scaled = q8_0_scaled_block(qs, scale);
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
    }
    while batch_idx < config.batch {
        let input_start = batch_idx * config.in_features + in_feature;
        let output_idx = batch_idx * config.out_features + out_feature;
        output[output_idx] += scale * q8_0_dot_i8_f32(qs, &input[input_start..], 32);
        batch_idx += 1;
    }
}
```

- [ ] Run existing Q8 tests:

```sh
cargo test -p rllm-runtime q8_0 -- --nocapture
```

Expected: all Q8 tests pass.

## Task 5: Wire Batch8 Into Q8 Multiply-Into

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] In `accumulate_q8_0_chunk_multiply_into`, replace the `config.batch > 1` full-block branch with batch8 first, then existing batch4, then scalar tail:

```rust
if config.batch > 1 && block_len == 32 && in_feature + block_len <= config.in_features {
    advance_multiply_state_to_row(state, out_feature, config, weight_name)?;
    let mut batch_idx = 0usize;
    while batch_idx + 8 <= config.batch {
        let input_start = batch_idx * config.in_features + in_feature;
        accumulate_q8_0_dot_32_batch8_into(
            qs,
            scale,
            &input[input_start..],
            config.in_features,
            &mut state.current_acc,
            batch_idx,
        );
        batch_idx += 8;
    }
    if batch_idx + 4 <= config.batch {
        let scaled = q8_0_scaled_block(qs, scale);
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
    }
    while batch_idx < config.batch {
        let input_start = batch_idx * config.in_features + in_feature;
        state.current_acc[batch_idx] +=
            scale * q8_0_dot_i8_f32(qs, &input[input_start..], 32);
        batch_idx += 1;
    }
    if in_feature + block_len == config.in_features {
        state.finish_current(config, weight_name)?;
    }
}
```

- [ ] Run existing multiply-into tests:

```sh
cargo test -p rllm-runtime multiply_into -- --nocapture
```

Expected: all multiply-into tests pass.

## Task 6: Build and Benchmark

**Files:**
- Modify: `docs/benchmarks/trials/active/2026-06-16-r85-q8-mlp-batch8-direct-dot.md`

- [ ] Build:

```sh
cargo build --release --bin llama-test
```

Expected: release build succeeds.

- [ ] Run three unchecked benchmark passes:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
```

Expected each run:

- output includes `No`
- `TTFT/Prefill` is recorded
- `PrefillProfile` is recorded
- internal peak transient remains close to `1050673152 bytes`
- `/usr/bin/time -l` max RSS is recorded

- [ ] Run one trace if a benchmark pass beats `11.45s`:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace target/r85-rllm-trace.json"
jq '[.summary.duration_by_phase[] | {phase,event_count,total_ms}]' target/r85-rllm-trace.json
jq '[.summary.duration_by_tensor_bucket[] | {bucket,event_count,total_ms}] | sort_by(.total_ms) | reverse' target/r85-rllm-trace.json
```

Expected: trace shows reduced MLP buckets versus R84.

## Task 7: Decide, Document, and Commit

**Files:**
- Modify: `docs/benchmarks/trials/active/2026-06-16-r85-q8-mlp-batch8-direct-dot.md`
- Move to one of:
  - `docs/benchmarks/trials/success/2026-06-16-r85-q8-mlp-batch8-direct-dot.md`
  - `docs/benchmarks/trials/failed/2026-06-16-r85-q8-mlp-batch8-direct-dot.md`
  - `docs/benchmarks/trials/inconclusive/2026-06-16-r85-q8-mlp-batch8-direct-dot.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] If accepted, keep code and move report to `success`.
- [ ] If slower or not better than `11.45s`, revert runtime changes and move report to `failed` or `inconclusive` with the raw numbers.
- [ ] Ensure generated traces are not staged:

```sh
git status --short target/r85-rllm-trace.json
```

Expected: no staged trace file.

- [ ] Run final verification:

```sh
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo build --release --bin llama-test
git diff --check
```

Expected: tests pass, build succeeds, `git diff --check` has no output.

- [ ] Commit accepted R85:

```sh
git add crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/*/2026-06-16-r85-q8-mlp-batch8-direct-dot.md
git commit -m "perf(runtime): add q8 batch8 prefill dot path"
```

- [ ] Commit rejected R85 docs-only evidence:

```sh
git add docs/benchmarks/trials/index.md docs/benchmarks/trials/*/2026-06-16-r85-q8-mlp-batch8-direct-dot.md
git commit -m "docs(bench): record r85 q8 batch8 direct dot result"
```

## Self-Review

- Spec coverage: The plan targets the R84 bottleneck: shared Q8 MLP compute.
- Low-RAM constraint: No resident full-model repack buffer is introduced.
- Correctness gate: Output, tests, profile, and memory are checked before acceptance.
- Benchmark gate: A speedup must beat the prior best `11.45s`, not only a noisy current run.
- Placeholder scan: No `TBD` or open-ended "add tests later" steps remain.
