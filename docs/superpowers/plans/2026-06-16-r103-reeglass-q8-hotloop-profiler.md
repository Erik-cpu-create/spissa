# R103 REEGLASS Q8 Hotloop Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the current `batch_gt1_scaled` Q8 profiler bucket into actionable hot-loop subpaths so R104 can target the real remaining overhead.

**Architecture:** R103 extends the existing opt-in `RLLM_Q8_KERNEL_PROFILE=1` profiler. It keeps default runtime behavior unchanged and adds detailed rows for normal linear and multiply-into Q8 batch-greater-than-one work: scale/dequant, batch4 dot, scalar tail, multiply state advance, and multiply finish. The existing aggregate `batch_gt1_scaled` row remains for continuity.

**Tech Stack:** Rust, `rllm-runtime`, existing `q8_profile`, `llama-test --profile-phases`, single-thread CPU benchmark flow.

---

## Evidence Inputs

R99 showed:

- prefill `9.29s`
- `batch_gt1_scaled` profiled elapsed `5853.47ms`
- gate/down/up MLP dominates

R100-R102 showed:

- batch8 NEON failed
- adjacent block64 pairing failed
- pre-scaled f32 sidecar failed

The remaining useful move is profiler attribution inside the hot loop.

## Files

- Modify: `crates/rllm-runtime/src/q8_profile.rs`
  - Add detailed `Q8KernelPath` variants.
  - Keep `ree_kernel` name stable or update to `REEGLASS-Q8-HOTLOOP-PROFILER`.
  - Extend tests to cover sorting and new path names.
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Record aggregate detail rows only when `RLLM_Q8_KERNEL_PROFILE=1`.
  - Use per-chunk local counters and record once per chunk, not per inner operation.
- Create: `docs/benchmarks/trials/success/2026-06-16-r103-reeglass-q8-hotloop-profiler.md`
- Modify: `docs/benchmarks/trials/index.md`

## Detailed Rows

Add these paths:

```rust
BatchGt1NormalScale,
BatchGt1NormalBatch4,
BatchGt1NormalTail,
BatchGt1MultiplyAdvance,
BatchGt1MultiplyScale,
BatchGt1MultiplyBatch4,
BatchGt1MultiplyTail,
BatchGt1MultiplyFinish,
```

Names:

```text
batch_gt1_normal_scale
batch_gt1_normal_batch4
batch_gt1_normal_tail
batch_gt1_multiply_advance
batch_gt1_multiply_scale
batch_gt1_multiply_batch4
batch_gt1_multiply_tail
batch_gt1_multiply_finish
```

## Gates

- `cargo test -p rllm-runtime q8_profile -- --nocapture` passes.
- `cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture` passes.
- `cargo build --release -p rllm-cli --bin llama-test` passes.
- With no env, `llama-test` output remains unchanged.
- With `RLLM_Q8_KERNEL_PROFILE=1`, suffix includes detailed rows.
- Profiled run output remains `No`.
- Peak transient remains `1,050,673,152 bytes`.

## Task 1: Add Failing Profiler Test

**Files:**
- Modify: `crates/rllm-runtime/src/q8_profile.rs`

- [ ] **Step 1: Extend the unit test**

Add assertions to `q8_profile_records_sorts_and_resets_rows`:

```rust
record_q8_kernel_path(
    Q8KernelPath::BatchGt1NormalBatch4,
    4,
    8,
    0,
    16,
    Duration::from_nanos(40),
);
record_q8_kernel_path(
    Q8KernelPath::BatchGt1MultiplyFinish,
    2,
    2,
    2,
    0,
    Duration::from_nanos(5),
);
```

Then assert:

```rust
assert_eq!(snapshot.rows[0].path, "batch_gt1_normal_batch4");
assert!(snapshot
    .rows
    .iter()
    .any(|row| row.path == "batch_gt1_multiply_finish"));
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p rllm-runtime q8_profile -- --nocapture
```

Expected: compile failure because the new variants do not exist.

## Task 2: Add Q8 Profile Variants

**Files:**
- Modify: `crates/rllm-runtime/src/q8_profile.rs`

- [ ] **Step 1: Add enum variants**

Add the detailed variants listed above to `Q8KernelPath`.

- [ ] **Step 2: Add string names**

Map each new variant in `Q8KernelPath::as_str()`.

- [ ] **Step 3: Update profiler kernel label**

Change snapshot label to:

```rust
ree_kernel: "REEGLASS-Q8-HOTLOOP-PROFILER",
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cargo test -p rllm-runtime q8_profile -- --nocapture
```

Expected: PASS.

## Task 3: Instrument Normal Q8 Hot Loop

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Add local counters**

Inside `accumulate_q8_0_chunk`, after validation, create local profile counters:

```rust
let mut normal_scale_elapsed = Duration::ZERO;
let mut normal_scale_calls = 0u64;
let mut normal_batch4_elapsed = Duration::ZERO;
let mut normal_batch4_calls = 0u64;
let mut normal_batch4_items = 0u64;
let mut normal_tail_elapsed = Duration::ZERO;
let mut normal_tail_calls = 0u64;
let mut normal_tail_items = 0u64;
```

- [ ] **Step 2: Time scale/dequant**

Around `q8_0_scaled_block_reecast(qs, scale)`:

```rust
let scale_start = profile_enabled.then(Instant::now);
let scaled = q8_0_scaled_block_reecast(qs, scale);
if let Some(scale_start) = scale_start {
    normal_scale_elapsed += scale_start.elapsed();
    normal_scale_calls += 1;
}
```

- [ ] **Step 3: Time batch4 and tail separately**

Time the `while batch_idx + 4 <= config.batch` loop as `BatchGt1NormalBatch4`.
Time the scalar remainder loop as `BatchGt1NormalTail`.

- [ ] **Step 4: Record detail rows once after the block loop**

Before `Ok(())`, add:

```rust
if profile_enabled {
    record_q8_kernel_path(
        Q8KernelPath::BatchGt1NormalScale,
        normal_scale_calls,
        normal_scale_calls,
        0,
        0,
        normal_scale_elapsed,
    );
    record_q8_kernel_path(
        Q8KernelPath::BatchGt1NormalBatch4,
        normal_batch4_calls,
        normal_batch4_calls,
        0,
        normal_batch4_items,
        normal_batch4_elapsed,
    );
    record_q8_kernel_path(
        Q8KernelPath::BatchGt1NormalTail,
        normal_tail_calls,
        normal_tail_calls,
        0,
        normal_tail_items,
        normal_tail_elapsed,
    );
}
```

## Task 4: Instrument Multiply-Into Q8 Hot Loop

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Add local counters**

Inside `accumulate_q8_0_chunk_multiply_into`, add counters for advance, scale,
batch4, tail, finish.

- [ ] **Step 2: Time operations**

Time:

- `advance_multiply_state_to_row`
- `q8_0_scaled_block_reecast`
- batch4 loop
- scalar tail loop
- `state.finish_current`

- [ ] **Step 3: Record detail rows once**

Record `BatchGt1MultiplyAdvance`, `BatchGt1MultiplyScale`,
`BatchGt1MultiplyBatch4`, `BatchGt1MultiplyTail`, and
`BatchGt1MultiplyFinish`.

## Task 5: Runtime Evidence

**Files:**
- No source changes.

- [ ] **Step 1: Build llama-test**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

- [ ] **Step 2: Run profile**

Run:

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r103-profile.txt 2> target/r103-profile.time
```

Expected:

- output `No`
- suffix includes `REEGLASS-Q8-HOTLOOP-PROFILER`
- suffix includes at least one detailed path

## Task 6: Report, Verify, Commit

**Files:**
- Create: `docs/benchmarks/trials/success/2026-06-16-r103-reeglass-q8-hotloop-profiler.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Write report**

Include:

- commands
- output correctness
- prefill/decode and peak
- detailed profile rows
- R104 recommendation

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
git add crates/rllm-runtime/src/q8_profile.rs crates/rllm-runtime/src/streaming/kernels.rs docs/superpowers/plans/2026-06-16-r103-reeglass-q8-hotloop-profiler.md docs/benchmarks/trials/success/2026-06-16-r103-reeglass-q8-hotloop-profiler.md docs/benchmarks/trials/index.md
git commit -m "bench(runtime): profile q8 hotloop detail"
```
