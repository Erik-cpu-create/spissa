# R96 REEVEC-Q8 NEON Batch4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test and, only if proven, promote an aarch64 NEON f32x4 fast path for the R93 `batch_gt1_scaled` Q8 prefill hotspot while keeping the portable scalar fallback for every other CPU.

**Architecture:** R96 keeps the current winning scaled-block design and changes only the batch4 f32 accumulator implementation behind a small architecture boundary. The lab adds `REEVEC-Q8-NEON-BATCH4-LAB` only on `target_arch = "aarch64"`; non-aarch64 builds keep existing portable variants. If lab and runtime gates pass, runtime uses a wrapper that calls NEON on aarch64 and existing scalar code elsewhere.

**Tech Stack:** Rust stable, `std::arch::aarch64` NEON intrinsics, `rllm-runtime`, `q8-microbench`, `llama-test`, benchmark trial docs.

---

## Why This Stage Exists

R94 and R95 proved that scalar rewrites are not enough:

- R94 `reeflow_i8_scaled_batch4`: exact but slower than `scaled_f32_dot32_batch4`
- R95 `reelane_f32_dot32_batch4`: exact but slower than `scaled_f32_dot32_batch4_runtime`
- R95 also showed the runtime-shaped scalar baseline is strong: `34,953,292ns`

R96 therefore stops reshaping scalar loops and tests an architecture-aware vector fast path. This is still universal because the fast path is behind `#[cfg(target_arch = "aarch64")]`; all other CPUs keep the current portable scalar code.

## Scope

Allowed:

- add aarch64-only NEON lab variant
- add aarch64-only NEON runtime helper
- keep portable scalar helper unchanged as fallback
- promote runtime only if lab and runtime gates pass
- document success or failure

Not allowed:

- removing portable CPU support
- changing `.rllm` format, Q8 format, tokenizer, prompt formatting, sampling, or memory budget logic
- adding permanent repack buffers or resident f32 caches
- changing the R93 profiler API
- claiming a universal speedup if only aarch64 was tested

## Success Gate

R96 is accepted only if all are true:

- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture` passes
- `cargo test -p rllm-runtime q8_0 -- --nocapture` passes
- `cargo test -p rllm-runtime` passes
- `cargo test -p rllm-cli --bin llama-test q8_kernel_profile -- --nocapture` passes
- NEON lab variant has `max_abs_diff <= 0.0001`
- NEON lab beats `scaled_f32_dot32_batch4_runtime`
- runtime output remains `No`
- runtime peak transient remains `1,050,673,152 bytes`
- best post-change prefill beats R96 pre-control best prefill

R96 is rejected if:

- NEON lab is not faster than runtime-shaped scalar baseline
- runtime prefill does not improve
- output changes
- memory grows
- non-aarch64 fallback compile path is broken

If rejected after runtime promotion, revert only runtime helper changes and keep useful lab/report evidence.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add aarch64-only lab function `reevec_neon_f32_dot32_batch4`.
  - Add aarch64-only helper `accumulate_neon_scaled_batch4`.
  - Add conditional test expectations.
- Modify if lab passes: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add scalar fallback helper wrapper.
  - Add aarch64-only NEON helper.
  - Replace calls to `accumulate_f32_dot_32_batch4` and `accumulate_f32_dot_32_batch4_into` with wrappers only after lab passes.
- Create success or failed R96 report under `docs/benchmarks/trials/`.
- Modify: `docs/benchmarks/trials/index.md`.
- Modify: this plan checklist.

## Task 1: Add AArch64 NEON Lab Variant

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [x] **Step 1: Import NEON intrinsics behind cfg**

Add near the top of the file:

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
```

- [x] **Step 2: Push NEON result conditionally**

After the existing batch4 result loop, add:

```rust
#[cfg(target_arch = "aarch64")]
{
    let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
        reevec_neon_f32_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
    });
    results.push(Q8KernelBenchResult {
        variant: "reevec_neon_f32_dot32_batch4".to_string(),
        elapsed_ns,
        checksum: checksum(&output),
        max_abs_diff: max_abs_diff(&baseline_output, &output),
        speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
    });
}
```

- [x] **Step 3: Add NEON lab implementation**

Add near the other batch4 lab variants:

```rust
#[cfg(target_arch = "aarch64")]
pub fn reevec_neon_f32_dot32_batch4(
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

- [x] **Step 4: Add NEON lab helper**

Add near the other lab helpers:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon_scaled_batch4(
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
    let mut idx = 0usize;
    while idx < 32 {
        let weights = vld1q_f32(scaled.as_ptr().add(idx));
        let x0 = vld1q_f32(input.as_ptr().add(idx));
        let x1 = vld1q_f32(input.as_ptr().add(stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(stride * 3 + idx));
        acc0 = vfmaq_f32(acc0, weights, x0);
        acc1 = vfmaq_f32(acc1, weights, x1);
        acc2 = vfmaq_f32(acc2, weights, x2);
        acc3 = vfmaq_f32(acc3, weights, x3);
        idx += 4;
    }
    output[batch_idx] += vaddvq_f32(acc0);
    output[batch_idx + 1] += vaddvq_f32(acc1);
    output[batch_idx + 2] += vaddvq_f32(acc2);
    output[batch_idx + 3] += vaddvq_f32(acc3);
}
```

- [x] **Step 5: Update lab test expectations conditionally**

In `q8_kernel_lab_reports_required_ree_variants`, keep the existing exact vector list for portable variants, then add:

```rust
#[cfg(target_arch = "aarch64")]
assert!(variants.contains(&"reevec_neon_f32_dot32_batch4"));
```

Do not require the NEON variant on non-aarch64.

- [x] **Step 6: Run lab tests**

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
```

Expected: pass.

## Task 2: Lab Gate

**Files:**
- Generate: `target/r96-reevec-lab.json`
- Generate: `target/r96-reevec-lab.md`

- [x] **Step 1: Build and run microbench**

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r96-reevec-lab.json \
  --markdown target/r96-reevec-lab.md \
  --iters 2000 \
  --batch 55
```

Expected on aarch64:

- output includes `reevec_neon_f32_dot32_batch4`
- `reevec_neon_f32_dot32_batch4 max_abs_diff <= 0.00010000`
- `reevec_neon_f32_dot32_batch4` faster than `scaled_f32_dot32_batch4_runtime`

- [x] **Step 2: Stop or continue**

If lab fails, do not touch runtime. Write failed report.

If lab passes, continue to Task 3.

## Task 3: Runtime Promotion

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [x] **Step 1: Capture R96 pre-control**

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r96-pre-control.txt 2> target/r96-pre-control.time
```

Expected: answer `No`, peak transient `1,050,673,152 bytes`.

- [x] **Step 2: Add runtime NEON import**

At the top of the module scope that owns `kernels.rs`, add:

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
```

- [x] **Step 3: Add runtime wrapper helpers**

Add:

```rust
fn accumulate_f32_dot_32_batch4_reevec(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return accumulate_f32_dot_32_batch4_neon(
            weights,
            input,
            input_stride,
            output,
            output_stride,
            out_feature,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_f32_dot_32_batch4(
        weights,
        input,
        input_stride,
        output,
        output_stride,
        out_feature,
    );
}
```

Add a matching `accumulate_f32_dot_32_batch4_into_reevec` wrapper with portable fallback to `accumulate_f32_dot_32_batch4_into`.

- [x] **Step 4: Add runtime NEON helpers**

Add aarch64-only helpers that mirror the lab helper, but write to `output_stride/out_feature` and `accumulators/accumulator_start`.

- [x] **Step 5: Replace runtime call sites**

Replace only the two hot branch calls:

- `accumulate_f32_dot_32_batch4(...)` -> `accumulate_f32_dot_32_batch4_reevec(...)`
- `accumulate_f32_dot_32_batch4_into(...)` -> `accumulate_f32_dot_32_batch4_into_reevec(...)`

Keep tail batch code unchanged.

- [x] **Step 6: Run focused Q8 tests**

```bash
cargo test -p rllm-runtime q8_0 -- --nocapture
```

Expected: pass.

## Task 4: Runtime Gate

**Files:**
- Generate: `target/r96-run1.txt`, `target/r96-run2.txt`, `target/r96-run3.txt`
- Generate matching `.time` files
- Generate: `target/r96-profile.txt`, `target/r96-profile.time`

- [x] **Step 1: Build release CLI**

```bash
cargo build --release -p rllm-cli --bin llama-test
```

- [x] **Step 2: Run three post-change runtime trials**

```bash
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r96-run${i}.txt" 2> "target/r96-run${i}.time"
done
```

Expected:

- every run answers `No`
- peak transient stays `1,050,673,152 bytes`
- best prefill beats R96 pre-control best prefill

- [x] **Step 3: Run profiled attribution trial**

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r96-profile.txt 2> target/r96-profile.time
```

Expected: `Q8KernelProfile` exists and top branch remains `batch_gt1_scaled`.

- [x] **Step 4: Decide**

Accept runtime only if all runtime gate checks pass. Otherwise revert runtime helper changes and write failed report.

## Task 5: Report and Commit

**Files:**
- Create success or failed R96 report.
- Modify: `docs/benchmarks/trials/index.md`.
- Modify: this plan checklist.

- [x] **Step 1: Write R96 report**

Include:

- lab table
- whether NEON was available
- pre-control runtime metrics if runtime promotion was attempted
- post-change runtime metrics if attempted
- correctness result
- peak transient result
- decision
- next experiment

- [x] **Step 2: Update benchmark index**

Add one row for `2026-06-16-r96-reevec-q8-neon-batch4.md`.

- [x] **Step 3: Final verification**

```bash
cargo fmt --check
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-cli --bin llama-test q8_kernel_profile -- --nocapture
cargo test -p rllm-runtime
git diff --check
```

- [x] **Step 4: Commit**

```bash
git add crates/rllm-runtime/src/q8_kernel_lab.rs crates/rllm-runtime/src/streaming/mod.rs crates/rllm-runtime/src/streaming/kernels.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-16-r96-reevec-q8-neon-batch4.md docs/benchmarks/trials/failed/2026-06-16-r96-reevec-q8-neon-batch4.md docs/superpowers/plans/2026-06-16-r96-reevec-q8-neon-batch4.md
git commit -m "bench(runtime): gate reevec q8 neon batch4 kernel"
```

Stage only files that exist.
