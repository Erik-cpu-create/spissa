# R104 REETAIL Q8 NEON Tail3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the R103-identified normal-path Q8 scalar tail cost for batch 55 by replacing the 3-token remainder loop with a small batch3 accumulator.

**Architecture:** R104 adds a lab variant named `REETAIL-Q8-NEON-TAIL3-LAB`. It keeps the current R98/R103 batch4 path and changes only the remainder path when exactly 3 batch rows remain. Runtime promotion is allowed only for `accumulate_q8_0_chunk` normal linear path, not multiply-into, and only if the lab beats `reecast_neon_scale_batch4`.

**Tech Stack:** Rust, `rllm-runtime`, aarch64 NEON intrinsics, existing `q8-microbench`, `llama-test --profile-phases`, `RLLM_Q8_KERNEL_PROFILE=1`.

**Final status:** Lab passed, runtime gate failed, runtime code reverted. Final report: `docs/benchmarks/trials/failed/2026-06-17-r104-reetail-q8-neon-tail3.md`.

---

## Evidence Inputs

R103 detail profile:

- `batch_gt1_normal_batch4`: `3551.82ms`
- `batch_gt1_normal_tail`: `1030.26ms`
- `batch_gt1_normal_scale`: `507.11ms`

With batch 55, the normal path has 13 groups of four and a 3-token remainder for
every full Q8 block. The tail is large enough to justify a narrow remainder
specialization.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add `reetail_neon_tail3_batch4`.
  - Add aarch64 helper `accumulate_neon_scaled_tail3`.
  - Add variant assertion in `q8_kernel_lab_reports_required_ree_variants`.
- Modify only if lab passes: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add `accumulate_f32_dot_32_tail3_reetail`.
  - Add aarch64 helper `accumulate_f32_dot_32_tail3_neon`.
  - Use it only in `accumulate_q8_0_chunk` when `config.batch - batch_idx == 3`.
- Create: `docs/benchmarks/trials/success/2026-06-17-r104-reetail-q8-neon-tail3.md` or `docs/benchmarks/trials/failed/2026-06-17-r104-reetail-q8-neon-tail3.md`
- Modify: `docs/benchmarks/trials/index.md`

## Gates

Lab gate:

- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture` passes.
- `q8-microbench` includes `reetail_neon_tail3_batch4`.
- `reetail_neon_tail3_batch4 max_abs_diff <= 0.0001`.
- Long lab run shows `reetail_neon_tail3_batch4` beats `reecast_neon_scale_batch4`.

Runtime gate, only if lab passes:

- `llama-test` release builds.
- control and candidate runs output `No`.
- peak transient remains `1,050,673,152 bytes`.
- `RLLM_Q8_KERNEL_PROFILE=1` shows `batch_gt1_normal_tail` lower than R103's
  `1030.26ms`, or at least candidate prefill improves over immediate control.

Revert rule:

- If lab gate fails, do not touch runtime.
- If runtime gate fails, revert runtime code and keep lab/report evidence.

## Task 1: Add Failing Lab Test

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [ ] **Step 1: Add variant expectation**

Inside `q8_kernel_lab_reports_required_ree_variants`, add:

```rust
#[cfg(target_arch = "aarch64")]
assert!(variants.contains(&"reetail_neon_tail3_batch4"));
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab_reports_required_ree_variants -- --nocapture
```

Expected: FAIL because `reetail_neon_tail3_batch4` is missing.

## Task 2: Add REETAIL Lab Variant

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [ ] **Step 1: Register the variant**

Inside the aarch64 lab block after `reeside_prescaled_f32_batch4`, add:

```rust
let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
    reetail_neon_tail3_batch4(&q8, scale, &input, config.batch, config.in_features)
});
results.push(Q8KernelBenchResult {
    variant: "reetail_neon_tail3_batch4".to_string(),
    elapsed_ns,
    checksum: checksum(&output),
    max_abs_diff: max_abs_diff(&baseline_output, &output),
    speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
});
```

- [ ] **Step 2: Add lab function**

Add near `reecast_neon_scale_batch4`:

```rust
#[cfg(target_arch = "aarch64")]
pub fn reetail_neon_tail3_batch4(
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
        if batch - batch_idx == 3 {
            unsafe {
                accumulate_neon_scaled_tail3(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
        } else {
            while batch_idx < batch {
                output[batch_idx] +=
                    dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
                batch_idx += 1;
            }
        }
    }
    output
}
```

- [ ] **Step 3: Add tail3 helper**

Add after `accumulate_neon_scaled_batch4`:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon_scaled_tail3(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let weights = vld1q_f32(scaled.as_ptr().add(idx));
        acc0 = vfmaq_f32(acc0, weights, vld1q_f32(input.as_ptr().add(idx)));
        acc1 = vfmaq_f32(acc1, weights, vld1q_f32(input.as_ptr().add(stride + idx)));
        acc2 = vfmaq_f32(acc2, weights, vld1q_f32(input.as_ptr().add(stride * 2 + idx)));
        idx += 4;
    }
    output[batch_idx] += vaddvq_f32(acc0);
    output[batch_idx + 1] += vaddvq_f32(acc1);
    output[batch_idx + 2] += vaddvq_f32(acc2);
}
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
```

Expected: PASS.

## Task 3: Run Lab Gate

**Files:**
- No source changes.

- [ ] **Step 1: Build microbench**

Run:

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
```

- [ ] **Step 2: Run standard lab**

Run:

```bash
target/release/q8-microbench \
  --json target/r104-reetail-lab.json \
  --markdown target/r104-reetail-lab.md \
  --iters 2000 \
  --batch 55
```

- [ ] **Step 3: Run long lab**

Run:

```bash
target/release/q8-microbench \
  --json target/r104-reetail-lab-long.json \
  --markdown target/r104-reetail-lab-long.md \
  --iters 10000 \
  --batch 55
```

Use the long run for the final lab decision.

## Task 4: Runtime Promotion If Lab Passes

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Add wrapper and helper**

Port the lab helper to runtime as:

- `accumulate_f32_dot_32_tail3_reetail`
- `accumulate_f32_dot_32_tail3_neon`

- [ ] **Step 2: Replace normal tail3 only**

Inside `accumulate_q8_0_chunk`, after the batch4 loop, use tail3 only when
`config.batch - batch_idx == 3`. Keep existing scalar loop for all other tails.

- [ ] **Step 3: Test**

Run:

```bash
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
```

Expected: PASS.

## Task 5: Runtime Benchmark If Promoted

**Files:**
- No source changes.

- [ ] **Step 1: Build llama-test**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

- [ ] **Step 2: Run control and candidate**

Run one control before promotion if needed, then three candidate runs:

```bash
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r104-run${i}.txt" 2> "target/r104-run${i}.time"
done
```

- [ ] **Step 3: Run profiler**

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r104-profile.txt 2> target/r104-profile.time
```

Expected: output `No`.

## Task 6: Report, Verify, Commit

**Files:**
- Create: R104 report under `docs/benchmarks/trials/success/` or `failed/`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Write report**

Include lab results, runtime promotion decision, runtime results if promoted,
and R105 recommendation.

- [ ] **Step 2: Verify**

Run:

```bash
cargo fmt --check
git diff --check
git status --short
```

- [ ] **Step 3: Commit**

Run:

```bash
git add crates/rllm-runtime/src/q8_kernel_lab.rs crates/rllm-runtime/src/streaming/kernels.rs docs/superpowers/plans/2026-06-17-r104-reetail-q8-neon-tail3.md docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r104-reetail-q8-neon-tail3.md docs/benchmarks/trials/failed/2026-06-17-r104-reetail-q8-neon-tail3.md
git commit -m "bench(runtime): gate reetail q8 neon tail3"
```
