# R86 Portable Q8 Prefill Kernel Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve RLLM exact-lowram Q8 prefill through a portable Q8 kernel interface, keeping scalar correctness as the universal baseline and allowing CPU-specific optimized backends without locking RLLM to Apple Silicon.

**Architecture:** R86 creates a small `streaming/q8_kernel.rs` layer for the hot Q8_0 32-element dot operations used by MLP prefill. `kernels.rs` routes full-block Q8 paths through this layer, while the layer defaults to scalar reference math on every CPU. The first optimized backend may be aarch64 NEON because the current benchmark machine can verify it, but the public call shape remains universal so x86 AVX2/FMA, lower-end ARM NEON, or bounded tile/repack backends can be added later without touching the streaming call-sites again.

**Tech Stack:** Rust, `rllm-runtime` streaming kernels, Q8_0 block weights, optional `std::arch::aarch64` intrinsics, `llama-test`, benchmark docs.

---

## Why This Order

We still fix prefill first because R84 proved the bottleneck:

- `chunk_compute_closure`: `11836.07ms`
- `chunk_read`: `3.12ms`
- MLP buckets: down `3354.26ms`, gate `3337.44ms`, up `3102.48ms`

But we do not hardcode an Apple-only solution. R86 builds the portable kernel boundary first, then uses the current `arm64` machine only as the first measurable backend. This keeps the product direction aligned with low-end CPU and IoT targets.

## Evidence From R84/R85

R84:

- Ollama CPU-only prompt eval: `0.285813s / 34 prompt tokens`
- RLLM unchecked prefill: `13.94s / 55 context tokens`
- RLLM trace showed IO is not the blocker.

R85:

- Portable scalar batch8 direct-dot kept output `No`.
- Best prefill was `12.68s`, failing the strict R83 gate of `11.45s`.
- Runtime changes were reverted.
- Conclusion: widening scalar loops alone is not enough.

## Success Gate

R86 is accepted only if all conditions hold:

- Output sanity prompt remains `No`.
- Internal peak transient remains `1050673152 bytes` or lower.
- Best of three unchecked prefill runs beats the R83 best `11.45s`.
- Prefill MLP total decreases versus the R84 measured `10703.88ms`.
- The scalar backend tests pass on every platform.
- Any optimized backend is covered by scalar-reference equivalence tests.
- Benchmark report records success or failure honestly.

If the abstraction lands but the optimized path does not beat `11.45s`, revert runtime optimization and either keep only the abstraction if it has no measurable regression, or record R86 as failed docs-only evidence.

## Files

- Modify: `crates/rllm-runtime/src/streaming/mod.rs`
  - Add `mod q8_kernel;` before the `include!("kernels.rs")` line.

- Create: `crates/rllm-runtime/src/streaming/q8_kernel.rs`
  - Owns portable Q8_0 32-element dot wrappers.
  - Provides scalar reference backend on every target.
  - Optionally provides aarch64 NEON backend behind `#[cfg(target_arch = "aarch64")]`.

- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Route full-block Q8 normal linear and multiply-into branches through `q8_kernel`.
  - Keep partial-block and cross-row fallback behavior unchanged.

- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
  - Add scalar equivalence tests for the q8 kernel layer.

- Create: `docs/benchmarks/trials/active/2026-06-16-r86-portable-q8-prefill-kernel-layer.md`
  - Record commands, raw measurements, and final decision.

- Modify: `docs/benchmarks/trials/index.md`
  - Add R86 trial row after measurement.

## Task 1: Add Active Benchmark Report

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-16-r86-portable-q8-prefill-kernel-layer.md`

- [ ] Add this report skeleton:

```markdown
# R86: Portable Q8 Prefill Kernel Layer

## Status

Active.

## Hypothesis

RLLM prefill is dominated by shared Q8 MLP compute. A portable Q8 kernel layer
lets RLLM keep scalar correctness on all CPUs while enabling CPU-specific
optimized dot paths for the same call-sites. The first optimized backend should
reduce gate/up/down time without changing the model format or RAM invariant.

## Baseline

- R83 best unchecked prefill: `11.45s`
- R84 measured unchecked prefill: `13.94s`
- R84 MLP total: `10703.88ms`
- R85 best unchecked prefill: `12.68s` but rejected
- R84/R85 peak transient: `1050673152 bytes`
- Baseline output: `No`

## Commands

Pending.

## Results

Pending.

## Decision

Pending.
```

- [ ] Verify the report exists:

```sh
test -f docs/benchmarks/trials/active/2026-06-16-r86-portable-q8-prefill-kernel-layer.md
```

Expected: exit code `0`.

## Task 2: Add Q8 Kernel Module and Failing Tests

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/mod.rs`
- Create: `crates/rllm-runtime/src/streaming/q8_kernel.rs`
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] Add the module declaration in `crates/rllm-runtime/src/streaming/mod.rs` immediately before the include list:

```rust
mod q8_kernel;
```

- [ ] Create `crates/rllm-runtime/src/streaming/q8_kernel.rs` with only the scalar reference helper first:

```rust
#[inline]
pub(super) fn dot_i8_f32_scalar(qs: &[u8], input: &[f32], len: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut idx = 0usize;
    while idx + 4 <= len {
        acc += (qs[idx] as i8) as f32 * input[idx]
            + (qs[idx + 1] as i8) as f32 * input[idx + 1]
            + (qs[idx + 2] as i8) as f32 * input[idx + 2]
            + (qs[idx + 3] as i8) as f32 * input[idx + 3];
        idx += 4;
    }
    while idx < len {
        acc += (qs[idx] as i8) as f32 * input[idx];
        idx += 1;
    }
    acc
}
```

- [ ] Add tests near existing Q8 helper tests in `crates/rllm-runtime/src/streaming/tests.rs`:

```rust
#[test]
fn q8_kernel_dot32_matches_scalar_reference() {
    let mut q = [0i8; 32];
    for (idx, value) in q.iter_mut().enumerate() {
        *value = (idx as i8 % 17) - 8;
    }
    let q8 = q8_0_block_bytes(0.25, &q);
    let input: Vec<f32> = (0..32).map(|idx| idx as f32 * 0.125 - 1.5).collect();

    let expected = 0.25 * q8_0_dot_i8_f32(&q8[2..34], &input, 32);
    let actual = q8_kernel::dot32(&q8[2..34], 0.25, &input);

    assert!((actual - expected).abs() <= 1.0e-4, "actual={actual} expected={expected}");
}

#[test]
fn q8_kernel_batch4_matches_scaled_reference() {
    let mut q = [0i8; 32];
    for (idx, value) in q.iter_mut().enumerate() {
        *value = (idx as i8 % 13) - 6;
    }
    let q8 = q8_0_block_bytes(0.125, &q);
    let scaled = q8_0_scaled_block(&q8[2..34], 0.125);

    let mut input = vec![0.0f32; 4 * 32];
    for batch_idx in 0..4 {
        for feature_idx in 0..32 {
            input[batch_idx * 32 + feature_idx] =
                batch_idx as f32 * 0.5 + feature_idx as f32 * 0.03125 - 0.75;
        }
    }

    let mut expected = vec![0.5f32, 1.5, 2.5, 3.5];
    let mut actual = expected.clone();

    accumulate_f32_dot_32_batch4(&scaled, &input, 32, &mut expected, 1, 0);
    q8_kernel::accumulate_dot32_batch4(&q8[2..34], 0.125, &input, 32, &mut actual, 1, 0);

    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() <= 1.0e-4, "idx={idx} actual={a} expected={e}");
    }
}

#[test]
fn q8_kernel_batch4_into_matches_scaled_reference() {
    let mut q = [0i8; 32];
    for (idx, value) in q.iter_mut().enumerate() {
        *value = (idx as i8 % 11) - 5;
    }
    let q8 = q8_0_block_bytes(0.5, &q);
    let scaled = q8_0_scaled_block(&q8[2..34], 0.5);

    let mut input = vec![0.0f32; 4 * 32];
    for batch_idx in 0..4 {
        for feature_idx in 0..32 {
            input[batch_idx * 32 + feature_idx] =
                batch_idx as f32 * -0.25 + feature_idx as f32 * 0.0625 + 0.25;
        }
    }

    let mut expected = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut actual = expected.clone();

    accumulate_f32_dot_32_batch4_into(&scaled, &input, 32, &mut expected, 0);
    q8_kernel::accumulate_dot32_batch4_into(&q8[2..34], 0.5, &input, 32, &mut actual, 0);

    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() <= 1.0e-4, "idx={idx} actual={a} expected={e}");
    }
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime q8_kernel -- --nocapture
```

Expected red result:

- compile error for missing `q8_kernel::dot32`
- compile error for missing `q8_kernel::accumulate_dot32_batch4`
- compile error for missing `q8_kernel::accumulate_dot32_batch4_into`

## Task 3: Implement Portable Scalar Kernel Interface

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/q8_kernel.rs`

- [ ] Extend `q8_kernel.rs` with the portable public wrappers:

```rust
#[inline]
pub(super) fn dot32(qs: &[u8], scale: f32, input: &[f32]) -> f32 {
    debug_assert!(qs.len() >= 32);
    debug_assert!(input.len() >= 32);
    scale * dot_i8_f32_scalar(qs, input, 32)
}

#[inline]
pub(super) fn accumulate_dot32_batch4(
    qs: &[u8],
    scale: f32,
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    output[out_feature] += dot32(qs, scale, input);
    output[output_stride + out_feature] += dot32(qs, scale, &input[input_stride..]);
    output[output_stride * 2 + out_feature] += dot32(qs, scale, &input[input_stride * 2..]);
    output[output_stride * 3 + out_feature] += dot32(qs, scale, &input[input_stride * 3..]);
}

#[inline]
pub(super) fn accumulate_dot32_batch4_into(
    qs: &[u8],
    scale: f32,
    input: &[f32],
    input_stride: usize,
    accumulators: &mut [f32],
    accumulator_start: usize,
) {
    accumulators[accumulator_start] += dot32(qs, scale, input);
    accumulators[accumulator_start + 1] += dot32(qs, scale, &input[input_stride..]);
    accumulators[accumulator_start + 2] += dot32(qs, scale, &input[input_stride * 2..]);
    accumulators[accumulator_start + 3] += dot32(qs, scale, &input[input_stride * 3..]);
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime q8_kernel -- --nocapture
```

Expected: the three new tests pass.

## Task 4: Wire Portable Interface Into Q8 Full-Block Paths

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] In `accumulate_q8_0_chunk`, replace the full-block batch branch body:

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
    q8_kernel::accumulate_dot32_batch4(
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
while batch_idx < config.batch {
    let input_start = batch_idx * config.in_features + in_feature;
    let output_idx = batch_idx * config.out_features + out_feature;
    output[output_idx] += q8_kernel::dot32(qs, scale, &input[input_start..]);
    batch_idx += 1;
}
```

- [ ] In `accumulate_q8_0_chunk_multiply_into`, replace the analogous full-block batch branch body with:

```rust
let mut batch_idx = 0usize;
while batch_idx + 4 <= config.batch {
    let input_start = batch_idx * config.in_features + in_feature;
    q8_kernel::accumulate_dot32_batch4_into(
        qs,
        scale,
        &input[input_start..],
        config.in_features,
        &mut state.current_acc,
        batch_idx,
    );
    batch_idx += 4;
}
while batch_idx < config.batch {
    let input_start = batch_idx * config.in_features + in_feature;
    state.current_acc[batch_idx] += q8_kernel::dot32(qs, scale, &input[input_start..]);
    batch_idx += 1;
}
```

- [ ] Run targeted tests:

```sh
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-runtime multiply_into -- --nocapture
```

Expected: all targeted tests pass.

## Task 5: Add Optional AArch64 NEON Backend Behind Same Interface

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/q8_kernel.rs`

- [ ] Add aarch64 imports at the top of `q8_kernel.rs`:

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, int16x8_t, int8x8_t, vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vget_high_s16,
    vget_low_s16, vld1_s8, vld1q_f32, vmovl_s16, vmovl_s8, vmulq_n_f32, vaddvq_f32,
};
```

- [ ] Replace `dot32` with a dispatch wrapper:

```rust
#[inline]
pub(super) fn dot32(qs: &[u8], scale: f32, input: &[f32]) -> f32 {
    debug_assert!(qs.len() >= 32);
    debug_assert!(input.len() >= 32);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot32_neon(qs, scale, input);
    }

    scale * dot_i8_f32_scalar(qs, input, 32)
}
```

- [ ] Add NEON implementation below the scalar helper:

```rust
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn load_weight_pair_f32x4(qs: *const u8, offset: usize, scale: f32) -> (float32x4_t, float32x4_t) {
    let q8: int8x8_t = unsafe { vld1_s8(qs.add(offset) as *const i8) };
    let q16: int16x8_t = unsafe { vmovl_s8(q8) };
    let lo = unsafe { vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16))) };
    let hi = unsafe { vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16))) };
    (vmulq_n_f32(lo, scale), vmulq_n_f32(hi, scale))
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot32_neon(qs: &[u8], scale: f32, input: &[f32]) -> f32 {
    let qs_ptr = qs.as_ptr();
    let input_ptr = input.as_ptr();
    let mut acc = vdupq_n_f32(0.0);

    let mut offset = 0usize;
    while offset < 32 {
        let (w0, w1) = unsafe { load_weight_pair_f32x4(qs_ptr, offset, scale) };
        let x0 = unsafe { vld1q_f32(input_ptr.add(offset)) };
        let x1 = unsafe { vld1q_f32(input_ptr.add(offset + 4)) };
        acc = unsafe { vfmaq_f32(acc, w0, x0) };
        acc = unsafe { vfmaq_f32(acc, w1, x1) };
        offset += 8;
    }

    unsafe { vaddvq_f32(acc) }
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime q8_kernel -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
```

Expected: tests pass. If a NEON import is unused, remove that import only.

## Task 6: Build and Benchmark

**Files:**
- Modify: `docs/benchmarks/trials/active/2026-06-16-r86-portable-q8-prefill-kernel-layer.md`

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

- [ ] Run one trace only if a benchmark pass beats `11.45s`:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace target/r86-rllm-trace.json"
jq '[.summary.duration_by_phase[] | {phase,event_count,total_ms}]' target/r86-rllm-trace.json
jq '[.summary.duration_by_tensor_bucket[] | {bucket,event_count,total_ms}] | sort_by(.total_ms) | reverse' target/r86-rllm-trace.json
```

Expected if accepted: trace shows reduced MLP buckets versus R84/R85.

## Task 7: Decide, Document, and Commit

**Files:**
- Modify: `docs/benchmarks/trials/active/2026-06-16-r86-portable-q8-prefill-kernel-layer.md`
- Move to one of:
  - `docs/benchmarks/trials/success/2026-06-16-r86-portable-q8-prefill-kernel-layer.md`
  - `docs/benchmarks/trials/failed/2026-06-16-r86-portable-q8-prefill-kernel-layer.md`
  - `docs/benchmarks/trials/inconclusive/2026-06-16-r86-portable-q8-prefill-kernel-layer.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] If accepted, keep runtime code and move report to `success`.
- [ ] If optimized backend is slower but abstraction alone is neutral, keep abstraction only and record optimized backend as rejected.
- [ ] If the whole change regresses or does not beat `11.45s`, revert runtime changes and move report to `failed` or `inconclusive` with raw numbers.
- [ ] Ensure generated trace is not staged:

```sh
git status --short target/r86-rllm-trace.json
```

Expected: no staged trace file.

- [ ] Run final verification:

```sh
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo build --release --bin llama-test
git diff --check
```

Expected: tests pass, build succeeds, `git diff --check` has no output.

- [ ] Commit accepted R86:

```sh
git add crates/rllm-runtime/src/streaming/mod.rs crates/rllm-runtime/src/streaming/q8_kernel.rs crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/*/2026-06-16-r86-portable-q8-prefill-kernel-layer.md
git commit -m "perf(runtime): add portable q8 prefill kernel layer"
```

- [ ] Commit rejected R86 docs-only evidence:

```sh
git add docs/benchmarks/trials/index.md docs/benchmarks/trials/*/2026-06-16-r86-portable-q8-prefill-kernel-layer.md
git commit -m "docs(bench): record r86 portable q8 kernel result"
```

## Self-Review

- Spec coverage: This plan fixes prefill first while making the kernel call boundary portable.
- Universal target: Scalar fallback exists on every CPU, optimized backends are additive.
- Low-RAM constraint: No full-model resident repack buffer is introduced.
- Correctness gate: Wrapper tests compare every optimized path to scalar/reference math.
- Benchmark gate: A speedup must beat `11.45s`, not only a noisy current run.
- Placeholder scan: No open-ended implementation steps remain.
