# R95 REELANE-Q8 Batch4 Accumulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test and, only if proven, promote a runtime-shaped unrolled f32 batch4 accumulator for the R93 `batch_gt1_scaled` Q8 prefill hotspot.

**Architecture:** R95 fixes the lab mismatch exposed by R94: the existing lab `scaled_f32_dot32_batch4` calls four independent f32 dot kernels, while runtime uses one four-lane accumulator loop. R95 adds runtime-shaped lab variants and a `REELANE-Q8-BATCH4-LAB` candidate that unrolls four indices at a time while preserving per-lane accumulation order. If the lab gate passes, the runtime helper `accumulate_f32_dot_32_batch4` and multiply-into sibling get the same unroll.

**Tech Stack:** Rust, `rllm-runtime`, `q8-microbench`, `llama-test`, benchmark trial docs.

---

## Why This Stage Exists

R94 proved that scalar Q8 fusion is slower than the current scaled-block path:

- current lab `scaled_f32_dot32_batch4`: `38,622,250ns`
- R94 `reeflow_i8_scaled_batch4`: `46,713,375ns`
- diff: `0`
- runtime was not touched

The next low-risk target is the current winner itself: keep the scaled f32 block, but reduce loop overhead in the batch4 accumulator that R93 identified as the dominant runtime branch.

## Scope

Allowed:

- add runtime-shaped scaled batch4 lab variants
- add `reelane_f32_dot32_batch4` lab candidate
- unroll runtime `accumulate_f32_dot_32_batch4`
- unroll runtime `accumulate_f32_dot_32_batch4_into`
- benchmark against a fresh R95 pre-control
- write success or failed report

Not allowed:

- changing Q8 quantization, container format, model artifact, prompt formatting, or output sampling
- introducing architecture-specific intrinsics in R95
- using permanent repack buffers
- changing accumulator order within a single output lane
- promoting runtime code if lab does not beat the current runtime-shaped baseline

## Success Gate

R95 is accepted only if:

- `reelane_f32_dot32_batch4` has `max_abs_diff <= 0.0001`
- `reelane_f32_dot32_batch4` beats `scaled_f32_dot32_batch4_runtime` in lab
- output remains `No`
- peak transient remains `1,050,673,152 bytes`
- best post-change prefill beats R95 pre-control best prefill
- `cargo test -p rllm-runtime` passes

R95 is rejected if:

- lab candidate is not faster than runtime-shaped baseline
- runtime prefill is not faster
- output changes
- memory grows

If rejected after runtime promotion, revert the runtime helper changes and keep only useful lab/report evidence.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add lab variants:
    - `scaled_f32_dot32_batch4_runtime`
    - `reelane_f32_dot32_batch4`
  - Add helpers:
    - `accumulate_scaled_batch4_runtime`
    - `accumulate_reelane_scaled_batch4`
  - Update tests.
- Modify if lab passes: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Unroll `accumulate_f32_dot_32_batch4`
  - Unroll `accumulate_f32_dot_32_batch4_into`
- Create success or failed R95 report under `docs/benchmarks/trials/`
- Modify: `docs/benchmarks/trials/index.md`
- Modify: this plan checklist.

## Task 1: Add Runtime-Shaped Lab Variants

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [x] **Step 1: Add variants to `run_suite`**

Insert after `scaled_f32_dot32_batch4`:

```rust
{
    let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
        scaled_f32_dot32_batch4_runtime(
            &q8,
            scale,
            &input,
            config.batch,
            config.in_features,
        )
    });
    ("scaled_f32_dot32_batch4_runtime", elapsed_ns, output)
},
{
    let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
        reelane_f32_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
    });
    ("reelane_f32_dot32_batch4", elapsed_ns, output)
},
```

- [x] **Step 2: Add runtime-shaped baseline**

Add:

```rust
pub fn scaled_f32_dot32_batch4_runtime(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = scaled_block(&q8[offset + 2..offset + 34], scale);
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            accumulate_scaled_batch4_runtime(
                &scaled,
                &input[batch_idx * in_features + in_feature..],
                in_features,
                &mut output,
                batch_idx,
            );
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}
```

- [x] **Step 3: Add REELANE lab candidate**

Add:

```rust
pub fn reelane_f32_dot32_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = scaled_block(&q8[offset + 2..offset + 34], scale);
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            accumulate_reelane_scaled_batch4(
                &scaled,
                &input[batch_idx * in_features + in_feature..],
                in_features,
                &mut output,
                batch_idx,
            );
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}
```

- [x] **Step 4: Add helpers**

Add near `accumulate_scaled_batch4`:

```rust
fn accumulate_scaled_batch4_runtime(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = output[batch_idx];
    let mut acc1 = output[batch_idx + 1];
    let mut acc2 = output[batch_idx + 2];
    let mut acc3 = output[batch_idx + 3];
    let mut idx = 0usize;
    while idx < 32 {
        let weight = scaled[idx];
        acc0 += weight * input[idx];
        acc1 += weight * input[stride + idx];
        acc2 += weight * input[stride * 2 + idx];
        acc3 += weight * input[stride * 3 + idx];
        idx += 1;
    }
    output[batch_idx] = acc0;
    output[batch_idx + 1] = acc1;
    output[batch_idx + 2] = acc2;
    output[batch_idx + 3] = acc3;
}

fn accumulate_reelane_scaled_batch4(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = output[batch_idx];
    let mut acc1 = output[batch_idx + 1];
    let mut acc2 = output[batch_idx + 2];
    let mut acc3 = output[batch_idx + 3];
    let mut idx = 0usize;
    while idx < 32 {
        let weight0 = scaled[idx];
        let weight1 = scaled[idx + 1];
        let weight2 = scaled[idx + 2];
        let weight3 = scaled[idx + 3];

        acc0 += weight0 * input[idx];
        acc1 += weight0 * input[stride + idx];
        acc2 += weight0 * input[stride * 2 + idx];
        acc3 += weight0 * input[stride * 3 + idx];

        acc0 += weight1 * input[idx + 1];
        acc1 += weight1 * input[stride + idx + 1];
        acc2 += weight1 * input[stride * 2 + idx + 1];
        acc3 += weight1 * input[stride * 3 + idx + 1];

        acc0 += weight2 * input[idx + 2];
        acc1 += weight2 * input[stride + idx + 2];
        acc2 += weight2 * input[stride * 2 + idx + 2];
        acc3 += weight2 * input[stride * 3 + idx + 2];

        acc0 += weight3 * input[idx + 3];
        acc1 += weight3 * input[stride + idx + 3];
        acc2 += weight3 * input[stride * 2 + idx + 3];
        acc3 += weight3 * input[stride * 3 + idx + 3];

        idx += 4;
    }
    output[batch_idx] = acc0;
    output[batch_idx + 1] = acc1;
    output[batch_idx + 2] = acc2;
    output[batch_idx + 3] = acc3;
}
```

- [x] **Step 5: Update lab test variants**

Expected variant list:

```rust
[
    "baseline_i8_dot32_batch4",
    "scaled_f32_dot32_batch4",
    "scaled_f32_dot32_batch4_runtime",
    "reelane_f32_dot32_batch4",
    "reeflow_i8_scaled_batch4",
    "unrolled_i8_dot32_batch4",
]
```

- [x] **Step 6: Run lab tests**

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
```

Expected: pass.

## Task 2: Lab Gate

**Files:**
- Generate: `target/r95-reelane-lab.json`
- Generate: `target/r95-reelane-lab.md`

- [x] **Step 1: Build and run microbench**

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r95-reelane-lab.json \
  --markdown target/r95-reelane-lab.md \
  --iters 2000 \
  --batch 55
```

Expected:

- `reelane_f32_dot32_batch4 max_abs_diff <= 0.00010000`
- `reelane_f32_dot32_batch4` faster than `scaled_f32_dot32_batch4_runtime`

- [x] **Step 2: Stop or continue**

If lab fails, do not touch runtime. Write failed report.

If lab passes, continue to Task 3.

## Task 3: Runtime Promotion

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Capture pre-control**

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r95-pre-control.txt 2> target/r95-pre-control.time
```

- [ ] **Step 2: Unroll `accumulate_f32_dot_32_batch4`**

Replace the loop with the `accumulate_reelane_scaled_batch4` body adapted to `output_stride` and `out_feature`.

- [ ] **Step 3: Unroll `accumulate_f32_dot_32_batch4_into`**

Replace the loop with the same four-index unroll adapted to `accumulators` and `accumulator_start`.

- [ ] **Step 4: Run focused Q8 tests**

```bash
cargo test -p rllm-runtime q8_0 -- --nocapture
```

Expected: pass.

## Task 4: Runtime Gate and Report

**Files:**
- Generate: `target/r95-run1.txt`, `target/r95-run2.txt`, `target/r95-run3.txt`
- Generate: matching `.time` files
- Generate: `target/r95-profile.txt`, `target/r95-profile.time`
- Create success or failed R95 report
- Modify benchmark index

- [ ] **Step 1: Run three post-change runtime trials**

```bash
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r95-run${i}.txt" 2> "target/r95-run${i}.time"
done
```

- [ ] **Step 2: Run profiled trial**

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r95-profile.txt 2> target/r95-profile.time
```

- [x] **Step 3: Decide and write report**

Accept only if post-change best prefill beats pre-control best prefill, output is `No`, and peak transient stays unchanged.

If runtime fails, revert `streaming/kernels.rs` and write failed report.

- [x] **Step 4: Final verification and commit**

```bash
cargo fmt --check
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-cli --bin llama-test q8_kernel_profile -- --nocapture
cargo test -p rllm-runtime
git diff --check
git add crates/rllm-runtime/src/q8_kernel_lab.rs crates/rllm-runtime/src/streaming/kernels.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-16-r95-reelane-q8-batch4-accumulator.md docs/benchmarks/trials/failed/2026-06-16-r95-reelane-q8-batch4-accumulator.md docs/superpowers/plans/2026-06-16-r95-reelane-q8-batch4-accumulator.md
git commit -m "bench(runtime): gate reelane q8 batch4 accumulator"
```

Stage only the report path that exists.
