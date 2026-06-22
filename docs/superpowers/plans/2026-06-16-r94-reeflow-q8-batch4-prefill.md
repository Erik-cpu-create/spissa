# R94 REEFLOW-Q8 Batch4 Prefill Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the measured `batch_gt1_scaled` Q8 prefill hotspot with a gated portable fused batch4 kernel if it beats the current runtime.

**Architecture:** R94 adds a lab variant named `REEFLOW-Q8-BATCH4-LAB` that fuses Q8 scaling and batch4 accumulation without materializing a `[f32; 32]` scaled block. If the lab gate passes, the same helper is promoted into the existing Q8 runtime `batch_gt1_scaled` branch for linear and multiply-into paths. The change must preserve exact output, peak transient memory, single-thread CPU-only benchmark discipline, and the R93 `Q8KernelProfile` attribution.

**Tech Stack:** Rust, `rllm-runtime`, `q8-microbench`, `llama-test`, existing benchmark trial docs.

---

## Why This Stage Exists

R93 measured the real runtime branches and found the dominant Q8 path:

- `batch_gt1_scaled`: `30,408,704` calls
- elapsed: `9931.51-10717.01ms`
- output stayed correct: `No`
- peak transient stayed unchanged: `1,050,673,152` bytes

R94 therefore targets only this hot path. It must not repeat the R92 mistake of optimizing a batch1 decode-shaped path while prefill is dominated by batch>1 work.

## Scope

Allowed:

- add `REEFLOW-Q8-BATCH4-LAB` to `q8_kernel_lab`
- add portable fused Q8 i8-scale batch4 helpers to `streaming/kernels.rs`
- replace the current `q8_0_scaled_block` + `accumulate_f32_dot_32_batch4` sequence only inside the existing `batch_gt1_scaled` runtime branch
- keep tail batches using the existing `f32_dot_32`/scaled block path unless benchmark evidence says otherwise
- write success or failed benchmark report

Not allowed:

- changing `.spsa` format, Q8 format, tokenizer, chat template, model loader, or RAM budget logic
- adding permanent repack buffers or resident f32 caches
- changing output sampling or answer correctness checks
- changing default threading discipline for the benchmark
- claiming success if runtime does not beat the R94 pre-control

## Success Gate

R94 is accepted only if all are true:

- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture` passes
- `cargo test -p rllm-runtime q8_0 -- --nocapture` passes
- `cargo test -p rllm-runtime` passes
- `cargo test -p rllm-cli --bin llama-test q8_kernel_profile -- --nocapture` passes
- lab variant `reeflow_i8_scaled_batch4` has `max_abs_diff <= 0.0001`
- runtime output remains `No`
- runtime peak transient remains `1,050,673,152 bytes`
- post-change best prefill beats R94 pre-control best prefill
- R93 profiler still reports `batch_gt1_scaled` as the measured path when enabled

R94 is rejected if:

- lab variant is slower than `scaled_f32_dot32_batch4`
- runtime prefill does not improve over R94 pre-control
- output changes
- peak transient grows
- the kernel makes decode materially worse

If rejected, runtime code must be reverted, but lab/report evidence may remain.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add `reeflow_i8_scaled_batch4` lab variant.
  - Add helper `accumulate_reeflow_i8_scaled_batch4`.
  - Update lab tests to require the new variant.
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add runtime helpers:
    - `accumulate_q8_0_i8_scaled_batch4`
    - `accumulate_q8_0_i8_scaled_batch4_into`
  - Use them only inside the current `config.batch > 1 && block_len == 32` branch.
- Create: `docs/benchmarks/trials/success/2026-06-16-r94-reeflow-q8-batch4-prefill.md` or `docs/benchmarks/trials/failed/2026-06-16-r94-reeflow-q8-batch4-prefill.md`
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `docs/superpowers/plans/2026-06-16-r94-reeflow-q8-batch4-prefill.md`
  - Check off completed steps during execution.

## Task 1: Add REEFLOW Lab Variant

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [x] **Step 1: Add the lab variant to `run_suite`**

In the array of batch4 variants after `scaled_f32_dot32_batch4`, add:

```rust
{
    let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
        reflow_i8_scaled_batch4(&q8, scale, &input, config.batch, config.in_features)
    });
    ("reeflow_i8_scaled_batch4", elapsed_ns, output)
},
```

- [x] **Step 2: Add the lab implementation**

Add this function near the existing batch4 lab variants:

```rust
pub fn reflow_i8_scaled_batch4(
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
        let qs = &q8[offset + 2..offset + 34];
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            accumulate_reeflow_i8_scaled_batch4(
                qs,
                scale,
                &input[batch_idx * in_features + in_feature..],
                in_features,
                &mut output,
                batch_idx,
            );
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled_block(qs, scale), &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}
```

Add this helper near `accumulate_scaled_batch4`:

```rust
fn accumulate_reeflow_i8_scaled_batch4(
    qs: &[u8],
    scale: f32,
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
        let weight = scale * (qs[idx] as i8) as f32;
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
```

- [x] **Step 3: Update lab tests**

Update the expected variants in `q8_kernel_lab_reports_required_ree_variants`:

```rust
assert_eq!(
    variants,
    [
        "baseline_i8_dot32_batch4",
        "scaled_f32_dot32_batch4",
        "reeflow_i8_scaled_batch4",
        "unrolled_i8_dot32_batch4"
    ]
);
```

- [x] **Step 4: Run lab tests**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
```

Expected: pass.

## Task 2: Run Lab Gate

**Files:**
- Generate: `target/r94-reeflow-lab.json`
- Generate: `target/r94-reeflow-lab.md`

- [x] **Step 1: Build microbench**

Run:

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
```

Expected: pass.

- [x] **Step 2: Run batch55 lab**

Run:

```bash
target/release/q8-microbench \
  --json target/r94-reeflow-lab.json \
  --markdown target/r94-reeflow-lab.md \
  --iters 2000 \
  --batch 55
```

Expected:

- `reeflow_i8_scaled_batch4 max_abs_diff <= 0.00010000`
- `reeflow_i8_scaled_batch4` elapsed lower than `scaled_f32_dot32_batch4`

- [x] **Step 3: Stop or continue based on lab**

If lab fails, do not touch runtime code. Write a failed report.

If lab passes, continue to Task 3.

## Task 3: Capture R94 Pre-Control

**Files:**
- Generate: `target/r94-pre-control.txt`
- Generate: `target/r94-pre-control.time`

- [ ] **Step 1: Build release CLI before runtime change**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: pass.

- [ ] **Step 2: Run pre-control**

Run:

```bash
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r94-pre-control.txt 2> target/r94-pre-control.time
```

Expected:

- answer contains `> No`
- metrics line has no `Q8KernelProfile`
- record prefill, decode, MLP total, peak transient, max RSS, elapsed

## Task 4: Promote REEFLOW Runtime Helper

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Add runtime helper for normal output**

Add near `accumulate_f32_dot_32_batch4`:

```rust
fn accumulate_q8_0_i8_scaled_batch4(
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
    let mut idx = 0usize;
    while idx < 32 {
        let weight = scale * (qs[idx] as i8) as f32;
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

- [ ] **Step 2: Add runtime helper for multiply-into output**

Add near `accumulate_f32_dot_32_batch4_into`:

```rust
fn accumulate_q8_0_i8_scaled_batch4_into(
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
    let mut idx = 0usize;
    while idx < 32 {
        let weight = scale * (qs[idx] as i8) as f32;
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

- [ ] **Step 3: Replace batch4 normal-output call site**

Inside `accumulate_q8_0_chunk`, replace:

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

with:

```rust
let mut batch_idx = 0usize;
while batch_idx + 4 <= config.batch {
    let input_start = batch_idx * config.in_features + in_feature;
    let output_start = batch_idx * config.out_features;
    accumulate_q8_0_i8_scaled_batch4(
        qs,
        scale,
        &input[input_start..],
        config.in_features,
        &mut output[output_start..],
        config.out_features,
        out_feature,
    );
    batch_idx += 4;
}
if batch_idx < config.batch {
    let scaled = q8_0_scaled_block(qs, scale);
    while batch_idx < config.batch {
        let input_start = batch_idx * config.in_features + in_feature;
        let output_idx = batch_idx * config.out_features + out_feature;
        output[output_idx] += f32_dot_32(&scaled, &input[input_start..]);
        batch_idx += 1;
    }
}
```

- [ ] **Step 4: Replace batch4 multiply-into call site**

Inside `accumulate_q8_0_chunk_multiply_into`, replace the analogous `scaled` + `accumulate_f32_dot_32_batch4_into` block with:

```rust
let mut batch_idx = 0usize;
while batch_idx + 4 <= config.batch {
    let input_start = batch_idx * config.in_features + in_feature;
    accumulate_q8_0_i8_scaled_batch4_into(
        qs,
        scale,
        &input[input_start..],
        config.in_features,
        &mut state.current_acc,
        batch_idx,
    );
    batch_idx += 4;
}
if batch_idx < config.batch {
    let scaled = q8_0_scaled_block(qs, scale);
    while batch_idx < config.batch {
        let input_start = batch_idx * config.in_features + in_feature;
        state.current_acc[batch_idx] += f32_dot_32(&scaled, &input[input_start..]);
        batch_idx += 1;
    }
}
```

- [ ] **Step 5: Run focused correctness tests**

Run:

```bash
cargo test -p rllm-runtime q8_0 -- --nocapture
```

Expected: pass.

## Task 5: Runtime Gate

**Files:**
- Generate: `target/r94-run1.txt`
- Generate: `target/r94-run1.time`
- Generate: `target/r94-run2.txt`
- Generate: `target/r94-run2.time`
- Generate: `target/r94-run3.txt`
- Generate: `target/r94-run3.time`
- Generate: `target/r94-profile.txt`
- Generate: `target/r94-profile.time`

- [ ] **Step 1: Build release CLI**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: pass.

- [ ] **Step 2: Run three post-change runtime trials**

Run:

```bash
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r94-run${i}.txt" 2> "target/r94-run${i}.time"
done
```

Expected for every run:

- answer contains `> No`
- peak transient is `1,050,673,152 bytes`
- metrics line contains no `Q8KernelProfile`

- [ ] **Step 3: Run one profiled attribution trial**

Run:

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r94-profile.txt 2> target/r94-profile.time
```

Expected:

- answer contains `> No`
- profile suffix contains `Q8KernelProfile`
- top branch still reports `batch_gt1_scaled`

- [ ] **Step 4: Decide promotion**

Accept runtime code only if best post-change prefill beats R94 pre-control best
prefill and all correctness/memory checks pass.

If it fails, revert only the runtime helper promotion in `streaming/kernels.rs`.
Keep lab code if it produced useful evidence.

## Task 6: Report and Final Verification

**Files:**
- Create success/failed R94 report under `docs/benchmarks/trials/`
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `docs/superpowers/plans/2026-06-16-r94-reeflow-q8-batch4-prefill.md`

- [x] **Step 1: Write report**

Report must include:

- lab table from `target/r94-reeflow-lab.md`
- R94 pre-control metrics
- post-change runtime metrics
- profiled attribution row
- correctness result (`No`)
- peak transient result
- decision: accepted or rejected
- next experiment recommendation

- [x] **Step 2: Update benchmark index**

Add one row to `docs/benchmarks/trials/index.md` with:

- trial: `2026-06-16-r94-reeflow-q8-batch4-prefill.md`
- folder: `success` or `failed`
- model: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- mode: `exact-lowram runtime, REEFLOW-Q8-BATCH4`
- bottleneck tag: `CPU arithmetic / Q8 batch4 prefill`
- baseline: R94 pre-control prefill
- result: best post-change prefill
- decision
- paper value

- [x] **Step 3: Run final verification**

Run:

```bash
cargo fmt --check
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-cli --bin llama-test q8_kernel_profile -- --nocapture
cargo test -p rllm-runtime
git diff --check
```

Expected: all pass.

- [ ] **Step 4: Commit R94**

Run:

```bash
git add crates/rllm-runtime/src/q8_kernel_lab.rs crates/rllm-runtime/src/streaming/kernels.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-16-r94-reeflow-q8-batch4-prefill.md docs/benchmarks/trials/failed/2026-06-16-r94-reeflow-q8-batch4-prefill.md docs/superpowers/plans/2026-06-16-r94-reeflow-q8-batch4-prefill.md
git commit -m "bench(runtime): gate reeflow q8 batch4 prefill kernel"
```

Stage only the report path that exists.
