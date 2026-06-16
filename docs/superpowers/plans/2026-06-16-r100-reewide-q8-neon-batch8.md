# R100 REEWIDE Q8 NEON Batch8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test and, only if gated by evidence, promote a wider aarch64 NEON Q8 prefill kernel for the normal linear path that covers R99's hottest MLP gate/down buckets.

**Architecture:** R100 adds a lab variant named `REEWIDE-Q8-NEON-BATCH8-LAB` that reuses R98's NEON Q8 scale/dequant and accumulates eight prompt rows per 32-weight block. Runtime promotion is limited to `accumulate_q8_0_chunk`, the normal linear path used by `gate_proj` and `down_proj`; `multiply_into` for `up_proj` remains on the R98 batch4 path unless a later stage proves batch8 helps it too.

**Tech Stack:** Rust, `rllm-runtime`, aarch64 NEON intrinsics, existing `q8-microbench`, `llama-test --profile-phases`, `RLLM_THREADS=1`.

---

## Evidence Inputs

R99 normal trace:

- output: `No`
- prefill: `9.29s`
- `chunk_compute_closure`: `6538.46ms`
- `mlp.gate_proj`: `2384.54ms`
- `mlp.down_proj`: `1785.53ms`
- `mlp.up_proj`: `1357.44ms`
- profiled `batch_gt1_scaled`: `5853.47ms`

This makes the normal Q8 linear path a better R100 target than batch1 decode or
`up_proj` multiply-into.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add `reewide_neon_f32_dot32_batch8`.
  - Add aarch64 helper `accumulate_neon_scaled_batch8`.
  - Add the variant to the report and unit variant list.
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add wrapper `accumulate_f32_dot_32_batch8_reewide`.
  - Add aarch64 helper `accumulate_f32_dot_32_batch8_neon`.
  - Use batch8 first only inside `accumulate_q8_0_chunk` full-block `batch_gt1_scaled`.
  - Leave `accumulate_q8_0_chunk_multiply_into` unchanged.
- Create: `docs/benchmarks/trials/success/2026-06-16-r100-reewide-q8-neon-batch8.md` or `docs/benchmarks/trials/failed/2026-06-16-r100-reewide-q8-neon-batch8.md`
- Modify: `docs/benchmarks/trials/index.md`

## Gates

Lab gate:

- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture` passes.
- `target/release/q8-microbench --json target/r100-reewide-lab.json --markdown target/r100-reewide-lab.md --iters 2000 --batch 55` reports `reewide_neon_f32_dot32_batch8`.
- `reewide_neon_f32_dot32_batch8 max_abs_diff <= 0.0001`.
- `reewide_neon_f32_dot32_batch8` beats `reecast_neon_scale_batch4`.

Runtime gate:

- Build `llama-test` release.
- Run one pre-control and three candidate runs with `RLLM_THREADS=1`.
- Output remains `No`.
- Peak transient remains `1,050,673,152 bytes`.
- Best candidate prefill beats the immediate pre-control.
- If candidate best is slower than R98/R99 best range (`9.28-9.29s`), accept only as marginal/diagnostic and state that clearly.

Revert rule:

- If lab gate fails, do not promote runtime changes.
- If runtime gate clearly regresses best prefill and offers no branch-profile win, revert runtime changes and record a failed trial.

## Task 1: Add REEWIDE Lab Variant

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [ ] **Step 1: Add the aarch64 lab result**

Inside the existing `#[cfg(target_arch = "aarch64")]` block after `reecast_neon_scale_batch4`, add:

```rust
let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
    reewide_neon_f32_dot32_batch8(&q8, scale, &input, config.batch, config.in_features)
});
results.push(Q8KernelBenchResult {
    variant: "reewide_neon_f32_dot32_batch8".to_string(),
    elapsed_ns,
    checksum: checksum(&output),
    max_abs_diff: max_abs_diff(&baseline_output, &output),
    speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
});
```

- [ ] **Step 2: Add the lab function**

Add near `reecast_neon_scale_batch4`:

```rust
#[cfg(target_arch = "aarch64")]
pub fn reewide_neon_f32_dot32_batch8(
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
        let scaled = unsafe { scaled_block_neon(&q8[offset + 2..offset + 34], scale) };
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 8 <= batch {
            unsafe {
                accumulate_neon_scaled_batch8(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 8;
        }
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled_batch4(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
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

- [ ] **Step 3: Add the NEON batch8 helper**

Add after `accumulate_neon_scaled_batch4`:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon_scaled_batch8(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut acc4 = vdupq_n_f32(0.0);
    let mut acc5 = vdupq_n_f32(0.0);
    let mut acc6 = vdupq_n_f32(0.0);
    let mut acc7 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let weights = vld1q_f32(scaled.as_ptr().add(idx));
        acc0 = vfmaq_f32(acc0, weights, vld1q_f32(input.as_ptr().add(idx)));
        acc1 = vfmaq_f32(acc1, weights, vld1q_f32(input.as_ptr().add(stride + idx)));
        acc2 = vfmaq_f32(acc2, weights, vld1q_f32(input.as_ptr().add(stride * 2 + idx)));
        acc3 = vfmaq_f32(acc3, weights, vld1q_f32(input.as_ptr().add(stride * 3 + idx)));
        acc4 = vfmaq_f32(acc4, weights, vld1q_f32(input.as_ptr().add(stride * 4 + idx)));
        acc5 = vfmaq_f32(acc5, weights, vld1q_f32(input.as_ptr().add(stride * 5 + idx)));
        acc6 = vfmaq_f32(acc6, weights, vld1q_f32(input.as_ptr().add(stride * 6 + idx)));
        acc7 = vfmaq_f32(acc7, weights, vld1q_f32(input.as_ptr().add(stride * 7 + idx)));
        idx += 4;
    }
    output[batch_idx] += vaddvq_f32(acc0);
    output[batch_idx + 1] += vaddvq_f32(acc1);
    output[batch_idx + 2] += vaddvq_f32(acc2);
    output[batch_idx + 3] += vaddvq_f32(acc3);
    output[batch_idx + 4] += vaddvq_f32(acc4);
    output[batch_idx + 5] += vaddvq_f32(acc5);
    output[batch_idx + 6] += vaddvq_f32(acc6);
    output[batch_idx + 7] += vaddvq_f32(acc7);
}
```

- [ ] **Step 4: Update the variant test**

Inside `q8_kernel_lab_reports_required_ree_variants`, add:

```rust
#[cfg(target_arch = "aarch64")]
assert!(variants.contains(&"reewide_neon_f32_dot32_batch8"));
```

- [ ] **Step 5: Run lab tests**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
```

Expected: PASS.

## Task 2: Run Lab Gate

**Files:**
- No source changes.

- [ ] **Step 1: Build q8-microbench**

Run:

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
```

Expected: PASS.

- [ ] **Step 2: Run R100 lab**

Run:

```bash
target/release/q8-microbench \
  --json target/r100-reewide-lab.json \
  --markdown target/r100-reewide-lab.md \
  --iters 2000 \
  --batch 55
```

Expected:

- output includes `reewide_neon_f32_dot32_batch8`
- `max_abs_diff <= 0.0001`
- elapsed lower than `reecast_neon_scale_batch4`

## Task 3: Promote Runtime Normal Path If Lab Passes

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Add the wrapper**

Add near `accumulate_f32_dot_32_batch4_reevec`:

```rust
fn accumulate_f32_dot_32_batch8_reewide(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return accumulate_f32_dot_32_batch8_neon(
            weights,
            input,
            input_stride,
            output,
            output_stride,
            out_feature,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        accumulate_f32_dot_32_batch4(
            weights,
            input,
            input_stride,
            output,
            output_stride,
            out_feature,
        );
        accumulate_f32_dot_32_batch4(
            weights,
            &input[input_stride * 4..],
            input_stride,
            &mut output[output_stride * 4..],
            output_stride,
            out_feature,
        );
    }
}
```

- [ ] **Step 2: Add the runtime NEON helper**

Add after `accumulate_f32_dot_32_batch4_neon`:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_f32_dot_32_batch8_neon(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut acc4 = vdupq_n_f32(0.0);
    let mut acc5 = vdupq_n_f32(0.0);
    let mut acc6 = vdupq_n_f32(0.0);
    let mut acc7 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let w = vld1q_f32(weights.as_ptr().add(idx));
        acc0 = vfmaq_f32(acc0, w, vld1q_f32(input.as_ptr().add(idx)));
        acc1 = vfmaq_f32(acc1, w, vld1q_f32(input.as_ptr().add(input_stride + idx)));
        acc2 = vfmaq_f32(acc2, w, vld1q_f32(input.as_ptr().add(input_stride * 2 + idx)));
        acc3 = vfmaq_f32(acc3, w, vld1q_f32(input.as_ptr().add(input_stride * 3 + idx)));
        acc4 = vfmaq_f32(acc4, w, vld1q_f32(input.as_ptr().add(input_stride * 4 + idx)));
        acc5 = vfmaq_f32(acc5, w, vld1q_f32(input.as_ptr().add(input_stride * 5 + idx)));
        acc6 = vfmaq_f32(acc6, w, vld1q_f32(input.as_ptr().add(input_stride * 6 + idx)));
        acc7 = vfmaq_f32(acc7, w, vld1q_f32(input.as_ptr().add(input_stride * 7 + idx)));
        idx += 4;
    }
    output[out_feature] += vaddvq_f32(acc0);
    output[output_stride + out_feature] += vaddvq_f32(acc1);
    output[output_stride * 2 + out_feature] += vaddvq_f32(acc2);
    output[output_stride * 3 + out_feature] += vaddvq_f32(acc3);
    output[output_stride * 4 + out_feature] += vaddvq_f32(acc4);
    output[output_stride * 5 + out_feature] += vaddvq_f32(acc5);
    output[output_stride * 6 + out_feature] += vaddvq_f32(acc6);
    output[output_stride * 7 + out_feature] += vaddvq_f32(acc7);
}
```

- [ ] **Step 3: Use batch8 first in normal Q8 chunk accumulation**

Inside `accumulate_q8_0_chunk`, replace the current batch4 loop in the
`config.batch > 1 && block_len == 32` branch with:

```rust
let scaled = q8_0_scaled_block_reecast(qs, scale);
let mut batch_idx = 0usize;
while batch_idx + 8 <= config.batch {
    let input_start = batch_idx * config.in_features + in_feature;
    let output_start = batch_idx * config.out_features;
    accumulate_f32_dot_32_batch8_reewide(
        &scaled,
        &input[input_start..],
        config.in_features,
        &mut output[output_start..],
        config.out_features,
        out_feature,
    );
    batch_idx += 8;
}
while batch_idx + 4 <= config.batch {
    let input_start = batch_idx * config.in_features + in_feature;
    let output_start = batch_idx * config.out_features;
    accumulate_f32_dot_32_batch4_reevec(
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

- [ ] **Step 4: Run focused streaming tests**

Run:

```bash
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
```

Expected: PASS.

## Task 4: Runtime Benchmark

**Files:**
- No source changes.

- [ ] **Step 1: Build llama-test**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: PASS.

- [ ] **Step 2: Run pre-control**

Run:

```bash
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r100-pre-control.txt 2> target/r100-pre-control.time
```

Expected: output `No`.

- [ ] **Step 3: Run three candidate trials**

Run:

```bash
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r100-run${i}.txt" 2> "target/r100-run${i}.time"
done
```

Expected: all outputs `No`.

- [ ] **Step 4: Run profiled candidate**

Run:

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r100-profile.txt 2> target/r100-profile.time
```

Expected: output `No`; profile still attributes work to `batch_gt1_scaled`.

## Task 5: Report, Verify, Commit

**Files:**
- Create: R100 report under `docs/benchmarks/trials/success/` or `failed/`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Write the report**

Report must include:

- lab table
- runtime table
- output correctness
- prefill/decode/MLP/gate/up/down
- peak transient and max RSS
- Q8 branch profile
- final decision

- [ ] **Step 2: Verify formatting and diff**

Run:

```bash
cargo fmt --check
git diff --check
git status --short
```

Expected: fmt/check pass; status shows only R100 intended files.

- [ ] **Step 3: Commit**

Run:

```bash
git add crates/rllm-runtime/src/q8_kernel_lab.rs crates/rllm-runtime/src/streaming/kernels.rs docs/superpowers/plans/2026-06-16-r100-reewide-q8-neon-batch8.md docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-16-r100-reewide-q8-neon-batch8.md docs/benchmarks/trials/failed/2026-06-16-r100-reewide-q8-neon-batch8.md
git commit -m "bench(runtime): gate reewide q8 neon batch8"
```

Expected: commit succeeds.
