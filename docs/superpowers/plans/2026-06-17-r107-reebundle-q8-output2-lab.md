# R107 REEBUNDLE Q8 Output2 Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether bundling two output features per Q8 input block can reduce loop/setup overhead before touching the real runtime.

**Architecture:** R107 adds a lab-only kernel named `REEBUNDLE-Q8-OUTPUT2-LAB` to `q8_kernel_lab.rs`. The lab models two adjacent output rows that share the same input batch slice and `in_feature` block index, then compares a bundled NEON implementation against a separate exact output2 baseline. Runtime promotion is allowed only if the lab wins a long benchmark with exact output; otherwise runtime code must remain unchanged.

**Tech Stack:** Rust, `rllm-runtime`, aarch64 NEON intrinsics, existing `q8-microbench`, existing benchmark report structure under `docs/benchmarks/trials/`.

---

## Evidence Inputs

R106 diagnostic:

- `batch_gt1_normal_batch4`: `25008.82ms` under heavy split instrumentation
- `batch_gt1_normal_batch4_kernel`: `6015.19ms`
- `batch_gt1_normal_batch4_setup`: `5241.44ms`
- residual loop/instrumentation overhead: `13752.19ms`

R106 conclusion:

- The remaining batch4 cost cannot be explained by NEON dot math alone.
- R107 should reduce the number of loop/setup events or use coarser work units.
- Do not add persistent sidecars.
- Do not increase peak transient memory.

Rejected paths to avoid:

- Tail-only specialization: R104 failed runtime gate.
- Inline hints: R105 failed runtime gate.
- Pre-scaled sidecar: R102 failed lab and costs extra storage.
- Batch8 / block64: R100/R101 failed lab.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add output2 deterministic Q8 row-pair helpers.
  - Add output2 exact baseline.
  - Add `reebundle_neon_output2_batch4` and NEON helper.
  - Register and test the lab variant.
- Create on success: `docs/benchmarks/trials/success/2026-06-17-r107-reebundle-q8-output2-lab.md`
- Create on failure: `docs/benchmarks/trials/failed/2026-06-17-r107-reebundle-q8-output2-lab.md`
- Modify: `docs/benchmarks/trials/index.md`

Runtime files such as `crates/rllm-runtime/src/streaming/kernels.rs` must not be modified until the lab gate passes.

## Lab Shape

Current `q8_kernel_lab.rs` mostly models one output feature:

```text
q8 blocks for one output row -> output[batch]
```

R107 output2 lab must model two output features:

```text
q8 blocks for row0 + q8 blocks for row1 -> output[batch * 2]
```

Use row-major output layout:

```text
output[batch_idx * 2 + 0] = row0 result for that batch item
output[batch_idx * 2 + 1] = row1 result for that batch item
```

The bundled candidate may reuse the same input batch slices while applying two
different scaled Q8 blocks. This models the runtime idea of processing adjacent
output features for the same `in_feature` block group.

## Gates

Lab correctness gates:

- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture` passes.
- New variant `reebundle_neon_output2_batch4` appears on aarch64.
- `max_abs_diff <= 0.0001` against the output2 baseline.

Lab performance gates:

- Standard lab:

```bash
target/release/q8-microbench --json target/r107-reebundle-lab.json --markdown target/r107-reebundle-lab.md --iters 2000 --batch 55
```

- Long lab:

```bash
target/release/q8-microbench --json target/r107-reebundle-lab-long.json --markdown target/r107-reebundle-lab-long.md --iters 10000 --batch 55
```

- The long lab must show `reebundle_neon_output2_batch4` faster than the output2 baseline.
- The report must state whether the result is strong enough for an R108 runtime-gated prototype.

Runtime gate:

- No runtime promotion in R107 unless explicitly approved after the lab result.
- If the lab fails, runtime code remains untouched and the report goes to `failed`.

## Task 1: Add RED Test for Output2 Variant

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [x] **Step 1: Add variant expectation**

Inside `q8_kernel_lab_reports_required_ree_variants`, add:

```rust
#[cfg(target_arch = "aarch64")]
assert!(variants.contains(&"reebundle_neon_output2_batch4"));
```

- [x] **Step 2: Verify RED**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab_reports_required_ree_variants -- --nocapture
```

Expected: FAIL because `reebundle_neon_output2_batch4` is not registered yet.

## Task 2: Add Output2 Baseline and Variant Registration

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [x] **Step 1: Add row-pair deterministic Q8 helper**

Add after `deterministic_q8_blocks`:

```rust
fn deterministic_q8_row_pair_blocks(blocks_per_row: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks_per_row * 2 * 34);
    for row in 0..2 {
        for block in 0..blocks_per_row {
            bytes.extend_from_slice(&crate::tensor::f32_to_fp16(0.125).to_le_bytes());
            for idx in 0..32 {
                let q = ((((row + 1) * 11 + block * 7 + idx * 3) as i16 % 17) - 8) as i8;
                bytes.push(q as u8);
            }
        }
    }
    bytes
}
```

- [x] **Step 2: Add output2 baseline function**

Add near `baseline_i8_dot32_batch4`:

```rust
pub fn baseline_i8_dot32_output2_batch4(
    q8_pair: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
    blocks_per_row: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * 2];
    for row in 0..2 {
        let row_offset = row * blocks_per_row * 34;
        for block in 0..blocks_per_row {
            let offset = row_offset + block * 34;
            let qs = &q8_pair[offset + 2..offset + 34];
            let in_feature = block * 32;
            for batch_idx in 0..batch {
                output[batch_idx * 2 + row] +=
                    scale * dot_i8_f32(qs, &input[batch_idx * in_features + in_feature..]);
            }
        }
    }
    output
}
```

- [x] **Step 3: Register output2 baseline and candidate**

In `run_suite`, after the existing aarch64 `reetail_neon_tail3_batch4` result, add:

```rust
let q8_pair = deterministic_q8_row_pair_blocks(config.blocks_per_row);
let (output2_baseline_ns, output2_baseline) = time_variant(config.iters, config.batch * 2, || {
    baseline_i8_dot32_output2_batch4(
        &q8_pair,
        scale,
        &input,
        config.batch,
        config.in_features,
        config.blocks_per_row,
    )
});
results.push(Q8KernelBenchResult {
    variant: "baseline_i8_dot32_output2_batch4".to_string(),
    elapsed_ns: output2_baseline_ns,
    checksum: checksum(&output2_baseline),
    max_abs_diff: 0.0,
    speedup_vs_baseline: 1.0,
});

let (elapsed_ns, output) = time_variant(config.iters, config.batch * 2, || {
    reebundle_neon_output2_batch4(
        &q8_pair,
        scale,
        &input,
        config.batch,
        config.in_features,
        config.blocks_per_row,
    )
});
results.push(Q8KernelBenchResult {
    variant: "reebundle_neon_output2_batch4".to_string(),
    elapsed_ns,
    checksum: checksum(&output),
    max_abs_diff: max_abs_diff(&output2_baseline, &output),
    speedup_vs_baseline: output2_baseline_ns as f64 / elapsed_ns.max(1) as f64,
});
```

This intentionally compares output2 candidate against output2 baseline, not the
single-output baseline.

## Task 3: Add REEBUNDLE Output2 Candidate

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [x] **Step 1: Add candidate function**

Add near the other aarch64 lab functions:

```rust
#[cfg(target_arch = "aarch64")]
pub fn reebundle_neon_output2_batch4(
    q8_pair: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
    blocks_per_row: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * 2];
    let row_stride = blocks_per_row * 34;
    for block in 0..blocks_per_row {
        let first_offset = block * 34;
        let second_offset = row_stride + block * 34;
        let first_scaled = unsafe { scaled_block_neon(&q8_pair[first_offset + 2..first_offset + 34], scale) };
        let second_scaled = unsafe { scaled_block_neon(&q8_pair[second_offset + 2..second_offset + 34], scale) };
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_output2_batch4(
                    &first_scaled,
                    &second_scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx * 2] +=
                dot_f32_32(&first_scaled, &input[batch_idx * in_features + in_feature..]);
            output[batch_idx * 2 + 1] +=
                dot_f32_32(&second_scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}
```

Rustfmt may wrap the two `scaled_block_neon` lines; keep the body equivalent.

- [x] **Step 2: Add output2 NEON helper**

Add after `accumulate_neon_scaled_batch4`:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon_output2_batch4(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut first0 = vdupq_n_f32(0.0);
    let mut first1 = vdupq_n_f32(0.0);
    let mut first2 = vdupq_n_f32(0.0);
    let mut first3 = vdupq_n_f32(0.0);
    let mut second0 = vdupq_n_f32(0.0);
    let mut second1 = vdupq_n_f32(0.0);
    let mut second2 = vdupq_n_f32(0.0);
    let mut second3 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let x0 = vld1q_f32(input.as_ptr().add(idx));
        let x1 = vld1q_f32(input.as_ptr().add(stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(stride * 3 + idx));
        let first_weights = vld1q_f32(first.as_ptr().add(idx));
        let second_weights = vld1q_f32(second.as_ptr().add(idx));
        first0 = vfmaq_f32(first0, first_weights, x0);
        first1 = vfmaq_f32(first1, first_weights, x1);
        first2 = vfmaq_f32(first2, first_weights, x2);
        first3 = vfmaq_f32(first3, first_weights, x3);
        second0 = vfmaq_f32(second0, second_weights, x0);
        second1 = vfmaq_f32(second1, second_weights, x1);
        second2 = vfmaq_f32(second2, second_weights, x2);
        second3 = vfmaq_f32(second3, second_weights, x3);
        idx += 4;
    }
    output[batch_idx * 2] += vaddvq_f32(first0);
    output[batch_idx * 2 + 1] += vaddvq_f32(second0);
    output[(batch_idx + 1) * 2] += vaddvq_f32(first1);
    output[(batch_idx + 1) * 2 + 1] += vaddvq_f32(second1);
    output[(batch_idx + 2) * 2] += vaddvq_f32(first2);
    output[(batch_idx + 2) * 2 + 1] += vaddvq_f32(second2);
    output[(batch_idx + 3) * 2] += vaddvq_f32(first3);
    output[(batch_idx + 3) * 2 + 1] += vaddvq_f32(second3);
}
```

- [x] **Step 3: Run lab tests**

Run:

```bash
cargo fmt
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
```

Expected: both existing q8 lab tests pass and `reebundle_neon_output2_batch4`
is present on aarch64.

## Task 4: Run R107 Lab Benchmarks

**Files:**
- No source changes.

- [x] **Step 1: Build microbench**

Run:

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
```

Expected: build succeeds.

- [x] **Step 2: Run standard lab**

Run:

```bash
target/release/q8-microbench --json target/r107-reebundle-lab.json --markdown target/r107-reebundle-lab.md --iters 2000 --batch 55
```

Expected:

- output includes `baseline_i8_dot32_output2_batch4`
- output includes `reebundle_neon_output2_batch4`
- `reebundle_neon_output2_batch4 max_abs_diff=0.00000000` or at most `0.0001`

- [x] **Step 3: Run long lab**

Run:

```bash
target/release/q8-microbench --json target/r107-reebundle-lab-long.json --markdown target/r107-reebundle-lab-long.md --iters 10000 --batch 55
```

Expected:

- same correctness requirements as standard lab
- use this run for the accept/reject decision

- [x] **Step 4: Decide lab gate**

Accept only if:

```text
reebundle_neon_output2_batch4 elapsed_ns < baseline_i8_dot32_output2_batch4 elapsed_ns
reebundle_neon_output2_batch4 max_abs_diff <= 0.0001
```

If the lab passes, the report may recommend an R108 runtime prototype. If the
lab fails, do not touch runtime and report the failed result.

## Task 5: Benchmark Report and Index

**Files:**
- Create one:
  - `docs/benchmarks/trials/success/2026-06-17-r107-reebundle-q8-output2-lab.md`
  - `docs/benchmarks/trials/failed/2026-06-17-r107-reebundle-q8-output2-lab.md`
- Modify: `docs/benchmarks/trials/index.md`

- [x] **Step 1: Write report**

The report must include:

- hypothesis
- REE kernel lineage: `REEBUNDLE-Q8-OUTPUT2-LAB`
- lab shape and why output2 has its own baseline
- exact commands
- standard and long lab tables
- correctness result
- accept/reject decision
- next experiment

Use measured values from:

```text
target/r107-reebundle-lab.md
target/r107-reebundle-lab-long.md
```

- [x] **Step 2: Update index**

Add one row after R106 in `docs/benchmarks/trials/index.md`. The row must state:

- status folder
- output2 baseline elapsed
- bundled candidate elapsed
- max abs diff
- whether runtime promotion is allowed for R108

- [x] **Step 3: Scan for unfinished markers**

Run:

```bash
rg -n "T[B]D|T[O]DO" docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r107-reebundle-q8-output2-lab.md docs/benchmarks/trials/failed/2026-06-17-r107-reebundle-q8-output2-lab.md
```

Expected: no matches in the R107 report or index row.

## Task 6: Final Verification and Commit

**Files:**
- All files changed by prior tasks.

- [x] **Step 1: Run final checks**

Run:

```bash
cargo fmt --check
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
git diff --check
```

Expected:

- all commands pass
- no whitespace errors

- [x] **Step 2: Review final diff**

Run:

```bash
git diff --stat
git diff -- crates/rllm-runtime/src/q8_kernel_lab.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r107-reebundle-q8-output2-lab.md docs/benchmarks/trials/failed/2026-06-17-r107-reebundle-q8-output2-lab.md docs/superpowers/plans/2026-06-17-r107-reebundle-q8-output2-lab.md
```

Expected:

- source changes are limited to lab code
- runtime streaming kernel is untouched
- report and index match measured values

- [x] **Step 3: Commit**

Run:

```bash
git add crates/rllm-runtime/src/q8_kernel_lab.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r107-reebundle-q8-output2-lab.md docs/benchmarks/trials/failed/2026-06-17-r107-reebundle-q8-output2-lab.md docs/superpowers/plans/2026-06-17-r107-reebundle-q8-output2-lab.md
git commit -m "bench(runtime): gate reebundle q8 output2 lab"
```

If one report path does not exist, remove that path from `git add` and commit only the existing report.

## Self-Review

- Spec coverage: R107 follows R106's loop/setup attribution and avoids another tiny dot hint.
- Placeholder scan: all commands, paths, labels, thresholds, and report requirements are explicit.
- Type consistency: output2 candidate compares against output2 baseline, not the single-output baseline.
- Scope: no runtime streaming kernel, model format, container, packer, tokenizer, or persistent sidecar changes are included.
