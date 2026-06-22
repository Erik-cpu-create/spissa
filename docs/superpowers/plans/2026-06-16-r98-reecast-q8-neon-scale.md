# R98 REECAST-Q8 NEON Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test and, only if proven, promote an aarch64 NEON Q8 scale/dequant helper for the remaining R97 `batch_gt1_scaled` prefill bottleneck.

**Architecture:** R98 keeps the R96 NEON batch4 accumulator and targets the scalar `q8_0_scaled_block` step that converts 32 signed Q8 bytes into a scaled f32 block. The lab adds `REECAST-Q8-NEON-SCALE-LAB`, comparing scalar `scaled_block` plus NEON accumulator against NEON scale/dequant plus the same NEON accumulator. Runtime promotion is allowed only if lab and runtime gates pass, with portable scalar fallback preserved for non-aarch64 CPUs.

**Tech Stack:** Rust stable, `std::arch::aarch64` NEON intrinsics, `rllm-runtime`, `q8-microbench`, `llama-test`, benchmark trial docs.

---

## Why This Stage Exists

R97 confirmed the next target:

- post-R96 prefill controls: `9.31-9.85s`
- normal MLP total: `6823.51-7455.54ms`
- profiled `batch_gt1_scaled`: `5845.56-6138.07ms`
- batch1 complete-row paths remain small

R96 already vectorized the f32 batch4 accumulator. The remaining plausible per-block cost is scalar Q8 scale/dequant in `q8_0_scaled_block`.

## Scope

Allowed:

- add aarch64-only NEON scale/dequant lab helper
- add aarch64-only runtime scale/dequant helper
- keep portable scalar fallback unchanged
- promote runtime only after lab passes
- benchmark normal and profiled runtime after promotion

Not allowed:

- changing Q8 format, `.spsa` format, tokenizer, prompt, sampling, or RAM budget logic
- adding resident f32 caches
- changing the R96 NEON accumulator semantics
- touching batch1 decode paths
- claiming universal speedup without non-aarch64 measurements

## Success Gate

R98 is accepted only if:

- `reecast_neon_scale_batch4` has `max_abs_diff <= 0.0001`
- lab `reecast_neon_scale_batch4` beats `reevec_neon_f32_dot32_batch4`
- runtime output remains `No`
- internal peak transient remains `1,050,673,152 bytes`
- best post-change prefill beats R98 pre-control best prefill
- `cargo test -p rllm-runtime` passes

R98 is rejected if:

- lab candidate is slower than R96 lab winner
- runtime prefill does not improve
- output changes
- internal peak transient grows

If rejected after runtime promotion, revert only runtime scale/dequant changes and keep useful lab/report evidence.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add `reecast_neon_scale_batch4` aarch64-only lab variant.
  - Add `scaled_block_neon` aarch64-only helper.
  - Add conditional test expectation.
- Modify if lab passes: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add `q8_0_scaled_block_reecast` wrapper.
  - Add aarch64-only `q8_0_scaled_block_neon`.
  - Replace only hot branch `q8_0_scaled_block` calls used before batch4 accumulators.
- Create success or failed R98 report.
- Modify benchmark index and this plan checklist.

## Task 1: Add NEON Scale/Dequant Lab

- [x] **Step 1: Add lab result conditionally**

After the existing `reevec_neon_f32_dot32_batch4` result push, add:

```rust
let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
    reecast_neon_scale_batch4(&q8, scale, &input, config.batch, config.in_features)
});
results.push(Q8KernelBenchResult {
    variant: "reecast_neon_scale_batch4".to_string(),
    elapsed_ns,
    checksum: checksum(&output),
    max_abs_diff: max_abs_diff(&baseline_output, &output),
    speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
});
```

Keep this inside `#[cfg(target_arch = "aarch64")]`.

- [x] **Step 2: Add lab candidate function**

Add near `reevec_neon_f32_dot32_batch4`:

```rust
#[cfg(target_arch = "aarch64")]
pub fn reecast_neon_scale_batch4(
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

- [x] **Step 3: Add `scaled_block_neon` helper**

Add near `scaled_block`:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn scaled_block_neon(qs: &[u8], scale: f32) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    let scale_vec = vdupq_n_f32(scale);
    let mut offset = 0usize;
    while offset < 32 {
        let q_i8 = vld1q_s8(qs.as_ptr().add(offset) as *const i8);
        let low_i16 = vmovl_s8(vget_low_s8(q_i8));
        let high_i16 = vmovl_s8(vget_high_s8(q_i8));

        let low_low_i32 = vmovl_s16(vget_low_s16(low_i16));
        let low_high_i32 = vmovl_s16(vget_high_s16(low_i16));
        let high_low_i32 = vmovl_s16(vget_low_s16(high_i16));
        let high_high_i32 = vmovl_s16(vget_high_s16(high_i16));

        vst1q_f32(
            out.as_mut_ptr().add(offset),
            vmulq_f32(vcvtq_f32_s32(low_low_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 4),
            vmulq_f32(vcvtq_f32_s32(low_high_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 8),
            vmulq_f32(vcvtq_f32_s32(high_low_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 12),
            vmulq_f32(vcvtq_f32_s32(high_high_i32), scale_vec),
        );
        offset += 16;
    }
    out
}
```

- [x] **Step 4: Update test expectation**

In `q8_kernel_lab_reports_required_ree_variants`, add:

```rust
#[cfg(target_arch = "aarch64")]
assert!(variants.contains(&"reecast_neon_scale_batch4"));
```

- [x] **Step 5: Run lab tests**

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
```

Expected: pass.

## Task 2: Run Lab Gate

- [x] **Step 1: Build and run R98 microbench**

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r98-reecast-lab.json \
  --markdown target/r98-reecast-lab.md \
  --iters 2000 \
  --batch 55
```

Expected:

- output includes `reecast_neon_scale_batch4`
- diff is `0` or within `0.0001`
- `reecast_neon_scale_batch4` is faster than `reevec_neon_f32_dot32_batch4`

- [x] **Step 2: Stop or continue**

If lab fails, do not touch runtime. Write failed report.

If lab passes, continue to Task 3.

## Task 3: Runtime Promotion

- [x] **Step 1: Capture R98 pre-control**

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r98-pre-control.txt 2> target/r98-pre-control.time
```

- [x] **Step 2: Add runtime scale/dequant wrapper**

In `streaming/kernels.rs`, add:

```rust
fn q8_0_scaled_block_reecast(qs: &[u8], scale: f32) -> [f32; 32] {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return q8_0_scaled_block_neon(qs, scale);
    }

    #[cfg(not(target_arch = "aarch64"))]
    q8_0_scaled_block(qs, scale)
}
```

- [x] **Step 3: Add runtime NEON scale/dequant helper**

Add `q8_0_scaled_block_neon` mirroring the lab `scaled_block_neon`.

- [x] **Step 4: Replace hot branch scale calls**

In `accumulate_q8_0_chunk` and `accumulate_q8_0_chunk_multiply_into`, replace only the first hot-branch `let scaled = q8_0_scaled_block(qs, scale);` with:

```rust
let scaled = q8_0_scaled_block_reecast(qs, scale);
```

Do not change batch1 row helpers.

- [x] **Step 5: Run focused Q8 tests**

```bash
cargo test -p rllm-runtime q8_0 -- --nocapture
```

Expected: pass.

## Task 4: Runtime Gate

- [x] **Step 1: Run post-change runtime trials**

```bash
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r98-run${i}.txt" 2> "target/r98-run${i}.time"
done
```

- [x] **Step 2: Run one profiled trial**

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r98-profile.txt 2> target/r98-profile.time
```

- [x] **Step 3: Decide**

Accept runtime only if output is `No`, peak transient is unchanged, and best prefill beats R98 pre-control best prefill.

If runtime fails, revert runtime scale/dequant changes and keep lab/report evidence.

## Task 5: Report and Commit

- [x] **Step 1: Write report**

Create success or failed report under `docs/benchmarks/trials/`.

- [x] **Step 2: Update benchmark index**

Add one R98 row.

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
git add crates/rllm-runtime/src/q8_kernel_lab.rs crates/rllm-runtime/src/streaming/kernels.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-16-r98-reecast-q8-neon-scale.md docs/benchmarks/trials/failed/2026-06-16-r98-reecast-q8-neon-scale.md docs/superpowers/plans/2026-06-16-r98-reecast-q8-neon-scale.md
git commit -m "bench(runtime): gate reecast q8 neon scale kernel"
```

Stage only files that exist.
