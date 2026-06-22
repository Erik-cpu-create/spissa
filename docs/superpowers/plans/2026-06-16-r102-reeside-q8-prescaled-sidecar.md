# R102 REESIDE Q8 Prescaled Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether a pack-time pre-scaled sidecar can beat the current R98 Q8 runtime micro-kernel before changing the `.spsa` container or packer.

**Architecture:** R102 adds a lab-only variant named `REESIDE-Q8-PRESCALED-SIDECAR-LAB`. The lab precomputes Q8 blocks into `[f32; 32]` sidecar blocks outside the timed loop, then runs the same proven batch4 NEON accumulator. If the lab wins, R103 can design a real `.spsa` sidecar format with explicit storage/RAM accounting; if it loses, we avoid a large packer/runtime change.

**Tech Stack:** Rust, `rllm-runtime`, aarch64 NEON intrinsics, existing `q8-microbench`, benchmark trial docs.

---

## Evidence Inputs

R99:

- remaining hotspot: `batch_gt1_scaled`
- MLP normal path gate/down dominates prefill
- chunk read is tiny compared with compute

R100/R101:

- batch8 widening failed
- adjacent block64 pairing failed
- current best measured Q8 micro-kernel shape remains R98 `reecast_neon_scale_batch4`

R102 tests whether the remaining cost is worth moving out of runtime and into
pack-time storage.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add sidecar precompute helper `prescaled_sidecar_blocks`.
  - Add lab variant `reeside_prescaled_f32_batch4`.
  - Add variant assertion in `q8_kernel_lab_reports_required_ree_variants`.
- Create: `docs/benchmarks/trials/success/2026-06-16-r102-reeside-q8-prescaled-sidecar.md` or `docs/benchmarks/trials/failed/2026-06-16-r102-reeside-q8-prescaled-sidecar.md`
- Modify: `docs/benchmarks/trials/index.md`

No runtime/container/packer changes are allowed in R102.

## Gates

Lab gate:

- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture` passes.
- `q8-microbench` includes `reeside_prescaled_f32_batch4`.
- `max_abs_diff <= 0.0001`.
- `reeside_prescaled_f32_batch4` beats `reecast_neon_scale_batch4` in a long run.

Decision:

- If sidecar wins: mark R102 as successful diagnostic and make R103 a packer/runtime sidecar design.
- If sidecar loses: mark R102 failed and stop sidecar work.

## Task 1: Add Failing Variant Test

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [ ] **Step 1: Add variant expectation**

Inside `q8_kernel_lab_reports_required_ree_variants`, add:

```rust
#[cfg(target_arch = "aarch64")]
assert!(variants.contains(&"reeside_prescaled_f32_batch4"));
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab_reports_required_ree_variants -- --nocapture
```

Expected: FAIL because `reeside_prescaled_f32_batch4` is missing.

## Task 2: Add REESIDE Lab Variant

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [ ] **Step 1: Precompute sidecar before the aarch64 result block**

Inside `run_suite`, after `let scale = 0.125f32;`, add:

```rust
#[cfg(target_arch = "aarch64")]
let prescaled_sidecar = prescaled_sidecar_blocks(&q8, scale);
```

- [ ] **Step 2: Register the variant**

Inside the aarch64 lab block after `reeduo_neon_block64_batch4`, add:

```rust
let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
    reeside_prescaled_f32_batch4(
        &prescaled_sidecar,
        &input,
        config.batch,
        config.in_features,
    )
});
results.push(Q8KernelBenchResult {
    variant: "reeside_prescaled_f32_batch4".to_string(),
    elapsed_ns,
    checksum: checksum(&output),
    max_abs_diff: max_abs_diff(&baseline_output, &output),
    speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
});
```

- [ ] **Step 3: Add the sidecar precompute helper**

Add after `scaled_block_neon`:

```rust
#[cfg(target_arch = "aarch64")]
fn prescaled_sidecar_blocks(q8: &[u8], scale: f32) -> Vec<[f32; 32]> {
    let blocks = q8.len() / 34;
    let mut sidecar = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let offset = block * 34;
        sidecar.push(unsafe { scaled_block_neon(&q8[offset + 2..offset + 34], scale) });
    }
    sidecar
}
```

- [ ] **Step 4: Add the sidecar lab function**

Add near the other aarch64 lab functions:

```rust
#[cfg(target_arch = "aarch64")]
pub fn reeside_prescaled_f32_batch4(
    sidecar: &[[f32; 32]],
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    for (block, scaled) in sidecar.iter().enumerate() {
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled_batch4(
                    scaled,
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
                dot_f32_32(scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}
```

- [ ] **Step 5: Verify GREEN**

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

- [ ] **Step 2: Run standard lab**

Run:

```bash
target/release/q8-microbench \
  --json target/r102-reeside-lab.json \
  --markdown target/r102-reeside-lab.md \
  --iters 2000 \
  --batch 55
```

Expected: output includes `reeside_prescaled_f32_batch4`.

- [ ] **Step 3: Run long lab**

Run:

```bash
target/release/q8-microbench \
  --json target/r102-reeside-lab-long.json \
  --markdown target/r102-reeside-lab-long.md \
  --iters 10000 \
  --batch 55
```

Expected: use this long run for the final decision.

## Task 4: Report, Verify, Commit

**Files:**
- Create: R102 report under `docs/benchmarks/trials/success/` or `failed/`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Write report**

Report must include:

- standard and long lab tables
- exactness
- sidecar storage implication: 128 bytes per 32-weight block vs 34 bytes Q8
- explicit statement that no runtime/container change happened
- R103 recommendation

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
git add crates/rllm-runtime/src/q8_kernel_lab.rs docs/superpowers/plans/2026-06-16-r102-reeside-q8-prescaled-sidecar.md docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-16-r102-reeside-q8-prescaled-sidecar.md docs/benchmarks/trials/failed/2026-06-16-r102-reeside-q8-prescaled-sidecar.md
git commit -m "bench(runtime): gate reeside q8 prescaled sidecar"
```

Expected: commit succeeds.
