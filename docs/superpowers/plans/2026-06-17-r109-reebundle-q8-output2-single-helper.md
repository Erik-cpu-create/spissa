# R109 REEBUNDLE Q8 Output2 Single Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace R108's conservative two-call output2 wrapper with a single output2 batch4 helper and accept it only if same-turn runtime evidence improves or stays safely neutral.

**Architecture:** Keep R108's output2 routing and safety gate intact. Change only the helper used after two adjacent output rows are proven safe: one helper should load the four batch input lanes once and accumulate two output features in the same 32-element loop, with a portable scalar fallback and an aarch64 NEON implementation.

**Tech Stack:** Rust, `rllm-runtime`, aarch64 NEON intrinsics, `rllm-cli` `llama-test`, benchmark reports under `docs/benchmarks/trials/`.

---

## Evidence Inputs

R108 accepted `REEBUNDLE-Q8-OUTPUT2`:

- same-turn control prefill: `9.28s`
- best candidate prefill: `7.55s`
- output: `No`
- RLLM peak transient: unchanged at `1,050,673,152 bytes`
- limitation: max RSS increased in candidate runs

R108's helper is intentionally conservative:

```rust
fn accumulate_f32_dot_32_output2_batch4_reebundle(...) {
    accumulate_f32_dot_32_batch4_reevec(first, ... first_out_feature);
    accumulate_f32_dot_32_batch4_reevec(second, ... first_out_feature + 1);
}
```

R109 tests whether a single helper that keeps both output features in one loop reduces helper overhead without changing math.

## Boundary

- Runtime owner: `crates/rllm-runtime/src/streaming/kernels.rs`
- Test owner: `crates/rllm-runtime/src/streaming/tests.rs`
- Benchmark docs owner: `docs/benchmarks/trials/`

Do not change R108 pair detection, Q8 block format, model/container format, tokenizer, prompt template, memory budget logic, Q8 multiply-into, Q8 argmax, or batch1 fast path.

## Files

- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Replace the output2 wrapper body with a real single-loop scalar helper.
  - Add aarch64 NEON single-loop helper behind `#[cfg(target_arch = "aarch64")]`.
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
  - Add a helper-level output2 test that fails if the helper stops accumulating both adjacent output features correctly.
- Create on success: `docs/benchmarks/trials/success/2026-06-17-r109-reebundle-q8-output2-single-helper.md`
- Create on failure: `docs/benchmarks/trials/failed/2026-06-17-r109-reebundle-q8-output2-single-helper.md`
- Modify: `docs/benchmarks/trials/index.md`

## Gates

Correctness:

```bash
cargo test -p rllm-runtime output2_batch4 -- --nocapture
cargo test -p rllm-runtime q8_0_output2 -- --nocapture
cargo test -p rllm-runtime q8_profile_records_sorts_and_resets_rows -- --nocapture
```

Build:

```bash
cargo fmt --check
cargo build --release -p rllm-cli --bin llama-test
git diff --check
```

Runtime:

Use same prompt and flags as R108:

```bash
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r109-pre-control.txt 2> target/r109-pre-control.time
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r109-run${i}.txt" 2> "target/r109-run${i}.time"
done
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r109-profile.txt 2> target/r109-profile.time
```

Acceptance:

- output remains exactly `No`
- RLLM peak transient stays at or below same-turn control
- best candidate prefill is at least `3%` faster than same-turn control, or profile row `batch_gt1_normal_output2_batch4` improves without no-profile regression
- max RSS does not exceed the R108 worst observed `2,496,315,392 bytes`

Rejection:

- output changes
- RLLM peak transient increases
- prefill regresses versus same-turn control
- max RSS exceeds R108 worst observed
- profile row disappears

If rejected, revert runtime code but keep the failed report and plan.

## Task 1: RED Helper Test

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`

- [x] **Step 1: Add direct helper test**

Add this test near the existing batch4 helper tests:

```rust
#[test]
fn output2_batch4_helper_accumulates_two_adjacent_features() {
    let mut first = [0.0f32; 32];
    let mut second = [0.0f32; 32];
    first[0] = 1.0;
    first[1] = 2.0;
    second[0] = 3.0;
    second[1] = 4.0;

    let mut input = vec![0.0f32; 4 * 32];
    input[0] = 1.0;
    input[1] = 10.0;
    input[32] = 2.0;
    input[33] = 20.0;
    input[64] = 3.0;
    input[65] = 30.0;
    input[96] = 4.0;
    input[97] = 40.0;

    let mut output = vec![0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

    accumulate_f32_dot_32_output2_batch4_reebundle(
        &first,
        &second,
        &input,
        32,
        &mut output,
        2,
        0,
    );

    assert_eq!(output, vec![21.5, 44.5, 44.5, 89.5, 67.5, 134.5, 90.5, 179.5]);
}
```

- [x] **Step 2: Verify RED or behavior lock**

Run:

```bash
cargo test -p rllm-runtime output2_batch4 -- --nocapture
```

Expected: PASS on R108 because the wrapper already computes correct output. This is acceptable for R109 because it locks helper behavior before refactoring internals.

## Task 2: Implement Single Output2 Helper

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [x] **Step 1: Replace wrapper body with cfg dispatch**

Change `accumulate_f32_dot_32_output2_batch4_reebundle` to:

```rust
fn accumulate_f32_dot_32_output2_batch4_reebundle(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return accumulate_f32_dot_32_output2_batch4_neon(
            first,
            second,
            input,
            input_stride,
            output,
            output_stride,
            first_out_feature,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_f32_dot_32_output2_batch4_scalar(
        first,
        second,
        input,
        input_stride,
        output,
        output_stride,
        first_out_feature,
    );
}
```

- [x] **Step 2: Add scalar fallback**

Add:

```rust
fn accumulate_f32_dot_32_output2_batch4_scalar(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
) {
    let mut first0 = output[first_out_feature];
    let mut first1 = output[output_stride + first_out_feature];
    let mut first2 = output[output_stride * 2 + first_out_feature];
    let mut first3 = output[output_stride * 3 + first_out_feature];
    let second_out_feature = first_out_feature + 1;
    let mut second0 = output[second_out_feature];
    let mut second1 = output[output_stride + second_out_feature];
    let mut second2 = output[output_stride * 2 + second_out_feature];
    let mut second3 = output[output_stride * 3 + second_out_feature];
    let mut idx = 0usize;
    while idx < 32 {
        let x0 = input[idx];
        let x1 = input[input_stride + idx];
        let x2 = input[input_stride * 2 + idx];
        let x3 = input[input_stride * 3 + idx];
        let fw = first[idx];
        let sw = second[idx];
        first0 += fw * x0;
        first1 += fw * x1;
        first2 += fw * x2;
        first3 += fw * x3;
        second0 += sw * x0;
        second1 += sw * x1;
        second2 += sw * x2;
        second3 += sw * x3;
        idx += 1;
    }
    output[first_out_feature] = first0;
    output[output_stride + first_out_feature] = first1;
    output[output_stride * 2 + first_out_feature] = first2;
    output[output_stride * 3 + first_out_feature] = first3;
    output[second_out_feature] = second0;
    output[output_stride + second_out_feature] = second1;
    output[output_stride * 2 + second_out_feature] = second2;
    output[output_stride * 3 + second_out_feature] = second3;
}
```

- [x] **Step 3: Add aarch64 NEON helper**

Add:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_f32_dot_32_output2_batch4_neon(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
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
        let x1 = vld1q_f32(input.as_ptr().add(input_stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(input_stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(input_stride * 3 + idx));
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
    let second_out_feature = first_out_feature + 1;
    output[first_out_feature] += vaddvq_f32(first0);
    output[output_stride + first_out_feature] += vaddvq_f32(first1);
    output[output_stride * 2 + first_out_feature] += vaddvq_f32(first2);
    output[output_stride * 3 + first_out_feature] += vaddvq_f32(first3);
    output[second_out_feature] += vaddvq_f32(second0);
    output[output_stride + second_out_feature] += vaddvq_f32(second1);
    output[output_stride * 2 + second_out_feature] += vaddvq_f32(second2);
    output[output_stride * 3 + second_out_feature] += vaddvq_f32(second3);
}
```

- [x] **Step 4: Verify helper and runtime tests**

Run:

```bash
cargo fmt
cargo test -p rllm-runtime output2_batch4 -- --nocapture
cargo test -p rllm-runtime q8_0_output2 -- --nocapture
```

Expected: PASS.

## Task 3: Runtime Benchmark Gate

**Files:**
- Runtime outputs under `target/r109-*`

- [x] **Step 1: Run pre-control before helper change if possible**

If helper change is already applied, use current `HEAD~1` or a temporary worktree for R108 code. Preferred command:

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r109-pre-control.txt 2> target/r109-pre-control.time
```

- [x] **Step 2: Run candidate trials**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r109-run${i}.txt" 2> "target/r109-run${i}.time"
done
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r109-profile.txt 2> target/r109-profile.time
```

- [x] **Step 3: Decide**

Accept only if:

- output `No`
- peak transient unchanged
- best prefill beats same-turn control by at least `3%`, or no-profile is neutral and profile row improves
- max RSS does not exceed `2,496,315,392`

Otherwise revert runtime helper change and create failed report.

## Task 4: Report, Index, Verification, Commit

**Files:**
- Create success or failed R109 report
- Modify `docs/benchmarks/trials/index.md`
- Modify this plan checklist

- [x] **Step 1: Write report**

Use exact measured numbers from `target/r109-*`. Include output, prefill, decode tok/s, MLP, gate/up/down, peak transient, max RSS, elapsed, and `batch_gt1_normal_output2_batch4`.

- [x] **Step 2: Update index**

Add one row for R109 with success or failed folder.

- [x] **Step 3: Final verification**

Run:

```bash
cargo fmt --check
cargo test -p rllm-runtime output2_batch4 -- --nocapture
cargo test -p rllm-runtime q8_0_output2 -- --nocapture
cargo test -p rllm-runtime q8_profile_records_sorts_and_resets_rows -- --nocapture
cargo build --release -p rllm-cli --bin llama-test
git diff --check
```

- [x] **Step 4: Commit**

If success:

```bash
git add crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r109-reebundle-q8-output2-single-helper.md docs/superpowers/plans/2026-06-17-r109-reebundle-q8-output2-single-helper.md
git commit -m "bench(runtime): gate reebundle q8 output2 helper"
```

If failed and runtime change was reverted:

```bash
git add docs/benchmarks/trials/index.md docs/benchmarks/trials/failed/2026-06-17-r109-reebundle-q8-output2-single-helper.md docs/superpowers/plans/2026-06-17-r109-reebundle-q8-output2-single-helper.md
git commit -m "bench(runtime): reject reebundle q8 output2 helper"
```
