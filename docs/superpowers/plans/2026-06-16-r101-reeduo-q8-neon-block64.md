# R101 REEDUO Q8 NEON Block64 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether pairing adjacent Q8 blocks into a 64-weight NEON accumulator reduces exact prefill cost without increasing resident RAM.

**Architecture:** R101 adds a lab variant named `REEDUO-Q8-NEON-BLOCK64-LAB`. It dequantizes two adjacent Q8_0 blocks into a bounded stack `[f32; 64]`, then accumulates four prompt rows across 64 inputs before one horizontal reduction/write. Runtime promotion is allowed only if the lab beats R98/R100 `reecast_neon_scale_batch4`; if accepted, promotion is limited to the normal Q8 linear path used by gate/down and only for adjacent full blocks inside the same row.

**Tech Stack:** Rust, `rllm-runtime`, aarch64 NEON intrinsics, existing `q8-microbench`, `llama-test --profile-phases`, `RLLM_THREADS=1`.

---

## Evidence Inputs

R99 showed:

- `mlp.gate_proj`: `2384.54ms`
- `mlp.down_proj`: `1785.53ms`
- `mlp.up_proj`: `1357.44ms`
- `batch_gt1_scaled`: `5853.47ms` in profiled run

R100 showed:

- batch8 NEON was exact but slower than batch4
- `reecast_neon_scale_batch4`: `17565500ns`
- `reewide_neon_f32_dot32_batch8`: `24298208ns`

So R101 must not widen batch accumulators. It should reduce per-block overhead
while keeping the proven batch4 shape.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add lab variant `reeduo_neon_block64_batch4`.
  - Add helper `scaled_pair_block_neon`.
  - Add helper `accumulate_neon_scaled64_batch4`.
  - Add variant assertion to the lab test.
- Modify only if lab passes: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add runtime block64 wrapper/helper.
  - Pair adjacent full blocks inside `accumulate_q8_0_chunk`.
  - Leave multiply-into unchanged for this stage.
- Create: `docs/benchmarks/trials/success/2026-06-16-r101-reeduo-q8-neon-block64.md` or `docs/benchmarks/trials/failed/2026-06-16-r101-reeduo-q8-neon-block64.md`
- Modify: `docs/benchmarks/trials/index.md`

## Gates

Lab gate:

- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture` passes.
- `q8-microbench` includes `reeduo_neon_block64_batch4`.
- `reeduo_neon_block64_batch4 max_abs_diff <= 0.0001`.
- `reeduo_neon_block64_batch4` beats `reecast_neon_scale_batch4`.

Runtime gate, only if lab passes:

- release `llama-test` builds.
- pre-control plus three candidate runs complete with output `No`.
- peak transient remains `1,050,673,152 bytes`.
- best candidate prefill beats immediate pre-control.
- profile still attributes work under `batch_gt1_scaled`.

Revert rule:

- If lab gate fails, do not touch runtime.
- If runtime gate fails, revert runtime code and keep only lab/report evidence.

## Task 1: Add Failing Lab Test

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [ ] **Step 1: Add variant expectation**

Inside `q8_kernel_lab_reports_required_ree_variants`, add:

```rust
#[cfg(target_arch = "aarch64")]
assert!(variants.contains(&"reeduo_neon_block64_batch4"));
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab_reports_required_ree_variants -- --nocapture
```

Expected: FAIL because `reeduo_neon_block64_batch4` is missing.

## Task 2: Add REEDUO Lab Variant

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [ ] **Step 1: Register the variant**

Inside the existing aarch64 lab block after `reewide_neon_f32_dot32_batch8`, add:

```rust
let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
    reeduo_neon_block64_batch4(&q8, scale, &input, config.batch, config.in_features)
});
results.push(Q8KernelBenchResult {
    variant: "reeduo_neon_block64_batch4".to_string(),
    elapsed_ns,
    checksum: checksum(&output),
    max_abs_diff: max_abs_diff(&baseline_output, &output),
    speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
});
```

- [ ] **Step 2: Add `reeduo_neon_block64_batch4`**

Add near the other aarch64 lab variants:

```rust
#[cfg(target_arch = "aarch64")]
pub fn reeduo_neon_block64_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    let mut block = 0usize;
    while block + 1 < blocks {
        let first_offset = block * 34;
        let second_offset = first_offset + 34;
        let scaled = unsafe {
            scaled_pair_block_neon(
                &q8[first_offset + 2..first_offset + 34],
                &q8[second_offset + 2..second_offset + 34],
                scale,
            )
        };
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled64_batch4(
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
                dot_f32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
        block += 2;
    }
    while block < blocks {
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
        block += 1;
    }
    output
}
```

- [ ] **Step 3: Add pair scale helper**

Add after `scaled_block_neon`:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn scaled_pair_block_neon(first: &[u8], second: &[u8], scale: f32) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    let first_scaled = scaled_block_neon(first, scale);
    let second_scaled = scaled_block_neon(second, scale);
    out[..32].copy_from_slice(&first_scaled);
    out[32..].copy_from_slice(&second_scaled);
    out
}
```

- [ ] **Step 4: Add 64-wide batch4 helper**

Add after `accumulate_neon_scaled_batch4`:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon_scaled64_batch4(
    scaled: &[f32; 64],
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
    while idx < 64 {
        let weights = vld1q_f32(scaled.as_ptr().add(idx));
        acc0 = vfmaq_f32(acc0, weights, vld1q_f32(input.as_ptr().add(idx)));
        acc1 = vfmaq_f32(acc1, weights, vld1q_f32(input.as_ptr().add(stride + idx)));
        acc2 = vfmaq_f32(acc2, weights, vld1q_f32(input.as_ptr().add(stride * 2 + idx)));
        acc3 = vfmaq_f32(acc3, weights, vld1q_f32(input.as_ptr().add(stride * 3 + idx)));
        idx += 4;
    }
    output[batch_idx] += vaddvq_f32(acc0);
    output[batch_idx + 1] += vaddvq_f32(acc1);
    output[batch_idx + 2] += vaddvq_f32(acc2);
    output[batch_idx + 3] += vaddvq_f32(acc3);
}
```

- [ ] **Step 5: Add generic f32 dot helper**

Add near `dot_f32_32`:

```rust
fn dot_f32(weights: &[f32], input: &[f32]) -> f32 {
    weights
        .iter()
        .zip(input.iter())
        .map(|(weight, value)| weight * value)
        .sum()
}
```

- [ ] **Step 6: Verify GREEN**

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

Expected: PASS.

- [ ] **Step 2: Run microbench**

Run:

```bash
target/release/q8-microbench \
  --json target/r101-reeduo-lab.json \
  --markdown target/r101-reeduo-lab.md \
  --iters 2000 \
  --batch 55
```

Expected:

- `reeduo_neon_block64_batch4` appears.
- `max_abs_diff <= 0.0001`.
- Runtime promotion allowed only if elapsed is lower than `reecast_neon_scale_batch4`.

## Task 4: Runtime Promotion If Lab Passes

**Files:**
- Modify only if lab passes: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Add runtime block64 helper**

If lab passes, port the lab helper into `streaming/kernels.rs` as
`accumulate_f32_dot_64_batch4_reeduo` and `accumulate_f32_dot_64_batch4_neon`.

- [ ] **Step 2: Pair adjacent full blocks**

Inside `accumulate_q8_0_chunk`, consume two blocks at a time only when:

- both blocks are full length
- both blocks are inside the same row
- `config.batch > 1`
- `in_feature + 64 <= config.in_features`

Fallback to the current R98/R96 block32 path for all other cases.

- [ ] **Step 3: Run focused test**

Run:

```bash
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
```

Expected: PASS.

## Task 5: Report, Verify, Commit

**Files:**
- Create: R101 benchmark report under `docs/benchmarks/trials/success/` or `failed/`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Write report**

Report must include:

- lab command output
- lab table
- whether runtime promotion happened
- exactness
- final decision
- next experiment

- [ ] **Step 2: Verify**

Run:

```bash
cargo fmt --check
git diff --check
git status --short
```

Expected: pass; only intended files changed.

- [ ] **Step 3: Commit**

Run:

```bash
git add crates/rllm-runtime/src/q8_kernel_lab.rs crates/rllm-runtime/src/streaming/kernels.rs docs/superpowers/plans/2026-06-16-r101-reeduo-q8-neon-block64.md docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-16-r101-reeduo-q8-neon-block64.md docs/benchmarks/trials/failed/2026-06-16-r101-reeduo-q8-neon-block64.md
git commit -m "bench(runtime): gate reeduo q8 neon block64"
```

Expected: commit succeeds.
