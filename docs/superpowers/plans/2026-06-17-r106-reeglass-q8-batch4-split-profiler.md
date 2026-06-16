# R106 REEGLASS Q8 Batch4 Split Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in Q8 batch4 split profiler so the next kernel stage targets measured loop/setup/kernel cost instead of guessing.

**Architecture:** R106 extends the existing `RLLM_Q8_KERNEL_PROFILE=1` profiler with more granular rows for the normal `batch_gt1_normal_batch4` path. Default runtime behavior stays unchanged when profiling is disabled. The profiler records setup time around per-call slice/index preparation and kernel time around the existing batch4 helper call; the existing aggregate rows remain so R103-R105 comparisons stay readable.

**Tech Stack:** Rust, `rllm-runtime`, existing Q8 profile state in `q8_profile.rs`, streaming Q8 runtime in `streaming/kernels.rs`, `llama-test --profile-phases`, `/usr/bin/time -l`.

---

## Evidence Inputs

R103 accepted diagnostic:

- `batch_gt1_scaled`: `10589.93ms`
- `batch_gt1_normal_batch4`: `3551.82ms`
- `batch_gt1_normal_tail`: `1030.26ms`
- `batch_gt1_normal_scale`: `507.11ms`

R104 rejected runtime gate:

- `REETAIL-Q8-NEON-TAIL3-LAB` passed lab but failed runtime.
- Runtime `batch_gt1_normal_tail` rose instead of falling.
- Conclusion: tiny isolated kernel wins are not reliable without better runtime attribution.

R105 rejected runtime gate:

- `REEINLINE-Q8-BATCH4-CALLSITE` kept output and memory correct.
- `batch_gt1_normal_batch4` regressed from same-turn `3727.99ms` to `3867.66ms`.
- Conclusion: compiler hints are not enough; R106 must split the batch4 bucket.

## Files

- Modify: `crates/rllm-runtime/src/q8_profile.rs`
  - Add new `Q8KernelPath` variants and string labels.
  - Update profile unit test to assert the new rows are recordable and sorted.
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add local accumulators for normal batch4 setup and kernel time.
  - Record the new detail rows only when `RLLM_Q8_KERNEL_PROFILE=1`.
  - Do not change math, helper signatures, memory layout, or non-profile runtime path.
- Modify: `crates/rllm-cli/src/bin/llama-test.rs`
  - Update the Q8 profile suffix test so the new labels are visible in output formatting.
  - Do not change CLI flags or default output.
- Create on success: `docs/benchmarks/trials/success/2026-06-17-r106-reeglass-q8-batch4-split-profiler.md`
- Modify: `docs/benchmarks/trials/index.md`

No model artifact, container, packer, tokenizer, or memory-budget changes are allowed in R106.

## New Profile Rows

Add these `Q8KernelPath` variants:

```rust
BatchGt1NormalBatch4Setup,
BatchGt1NormalBatch4Kernel,
```

Use these output labels:

```text
batch_gt1_normal_batch4_setup
batch_gt1_normal_batch4_kernel
```

Semantics:

- `batch_gt1_normal_batch4_setup`: time spent inside the batch4 loop preparing `input_start`, `output_start`, and slices before calling the helper.
- `batch_gt1_normal_batch4_kernel`: time spent inside the existing `accumulate_f32_dot_32_batch4_reevec(...)` helper call.
- Existing `batch_gt1_normal_batch4`: total loop time from before the first batch4 iteration until after the last batch4 iteration.
- Estimated loop overhead for analysis is:

```text
batch_gt1_normal_batch4 - batch_gt1_normal_batch4_setup - batch_gt1_normal_batch4_kernel
```

Do not record an explicit overhead row in R106. Compute it in the report so the runtime profiler stays small.

## Gates

Correctness gates:

- Default non-profile behavior remains unchanged.
- Profile behavior remains opt-in through `RLLM_Q8_KERNEL_PROFILE=1`.
- Visible output must remain exactly:

```text
No
```

- `Peak` in `llama-test` output must remain `1050673152 bytes`.
- `cargo test -p rllm-runtime q8_profile -- --nocapture` passes.
- `cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture` passes.
- `cargo test -p rllm-cli --bin llama-test q8_kernel_profile_suffix -- --nocapture` passes.

Diagnostic gate:

- Profile output must include all three rows:

```text
batch_gt1_normal_batch4
batch_gt1_normal_batch4_setup
batch_gt1_normal_batch4_kernel
```

- The report must identify whether R107 should target setup/slice overhead, kernel math, loop overhead, or a broader output-feature tiling design.

Acceptance:

- R106 is accepted if it gives actionable attribution without changing default runtime behavior, even if profiled runs are slower due to instrumentation.

## Task 1: Extend Q8 Profile Paths

**Files:**
- Modify: `crates/rllm-runtime/src/q8_profile.rs`

- [ ] **Step 1: Add enum variants**

In `Q8KernelPath`, add the new variants immediately after `BatchGt1NormalBatch4`:

```rust
BatchGt1NormalBatch4Setup,
BatchGt1NormalBatch4Kernel,
```

- [ ] **Step 2: Add string labels**

In `impl Q8KernelPath { pub fn as_str(...) }`, add:

```rust
Self::BatchGt1NormalBatch4Setup => "batch_gt1_normal_batch4_setup",
Self::BatchGt1NormalBatch4Kernel => "batch_gt1_normal_batch4_kernel",
```

Place them next to the existing `BatchGt1NormalBatch4` match arm.

- [ ] **Step 3: Update q8 profile unit test**

In `q8_profile_records_sorts_and_resets_rows`, add records for the new paths:

```rust
record_q8_kernel_path(
    Q8KernelPath::BatchGt1NormalBatch4Setup,
    4,
    4,
    0,
    16,
    Duration::from_nanos(15),
);
record_q8_kernel_path(
    Q8KernelPath::BatchGt1NormalBatch4Kernel,
    4,
    4,
    0,
    16,
    Duration::from_nanos(45),
);
```

Then assert both labels are present:

```rust
assert!(snapshot
    .rows
    .iter()
    .any(|row| row.path == "batch_gt1_normal_batch4_setup"));
assert!(snapshot
    .rows
    .iter()
    .any(|row| row.path == "batch_gt1_normal_batch4_kernel"));
```

- [ ] **Step 4: Run runtime profile test**

Run:

```bash
cargo test -p rllm-runtime q8_profile -- --nocapture
```

Expected:

```text
test q8_profile::tests::q8_profile_records_sorts_and_resets_rows ... ok
```

## Task 2: Instrument Normal Batch4 Split

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Add local accumulators**

Inside `fn accumulate_q8_0_chunk(...)`, near the existing `normal_batch4_*`
locals, add:

```rust
let mut normal_batch4_setup_elapsed = std::time::Duration::ZERO;
let mut normal_batch4_setup_calls = 0u64;
let mut normal_batch4_kernel_elapsed = std::time::Duration::ZERO;
let mut normal_batch4_kernel_calls = 0u64;
let mut normal_batch4_kernel_items = 0u64;
```

- [ ] **Step 2: Split timing inside the batch4 loop**

Replace the current batch4 loop body:

```rust
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
```

with:

```rust
while batch_idx + 4 <= config.batch {
    let setup_start = profile_enabled.then(Instant::now);
    let input_start = batch_idx * config.in_features + in_feature;
    let output_start = batch_idx * config.out_features;
    if let Some(setup_start) = setup_start {
        normal_batch4_setup_elapsed += setup_start.elapsed();
        normal_batch4_setup_calls += 1;
    }

    let kernel_start = profile_enabled.then(Instant::now);
    accumulate_f32_dot_32_batch4_reevec(
        &scaled,
        &input[input_start..],
        config.in_features,
        &mut output[output_start..],
        config.out_features,
        out_feature,
    );
    if let Some(kernel_start) = kernel_start {
        normal_batch4_kernel_elapsed += kernel_start.elapsed();
        normal_batch4_kernel_calls += 1;
        normal_batch4_kernel_items += 4;
    }

    batch_idx += 4;
}
```

This keeps instrumentation opt-in because `profile_enabled.then(Instant::now)`
does not call `Instant::now` when profiling is disabled.

- [ ] **Step 3: Record new rows**

Inside the existing `if profile_enabled { ... }` recording block for normal Q8,
after `BatchGt1NormalBatch4`, add:

```rust
record_q8_kernel_path(
    Q8KernelPath::BatchGt1NormalBatch4Setup,
    normal_batch4_setup_calls,
    normal_batch4_setup_calls,
    0,
    0,
    normal_batch4_setup_elapsed,
);
record_q8_kernel_path(
    Q8KernelPath::BatchGt1NormalBatch4Kernel,
    normal_batch4_kernel_calls,
    normal_batch4_kernel_calls,
    0,
    normal_batch4_kernel_items,
    normal_batch4_kernel_elapsed,
);
```

- [ ] **Step 4: Format and run targeted runtime test**

Run:

```bash
cargo fmt
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
```

Expected:

```text
test streaming::tests::streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch ... ok
```

## Task 3: Update CLI Profile Suffix Test

**Files:**
- Modify: `crates/rllm-cli/src/bin/llama-test.rs`

- [ ] **Step 1: Add recorded rows in test**

In `q8_kernel_profile_suffix_reports_recorded_rows`, after the existing
`Batch1CompleteMultiply` record, add:

```rust
rllm_runtime::record_q8_kernel_path(
    rllm_runtime::Q8KernelPath::BatchGt1NormalBatch4Setup,
    3,
    3,
    0,
    0,
    std::time::Duration::from_micros(250),
);
rllm_runtime::record_q8_kernel_path(
    rllm_runtime::Q8KernelPath::BatchGt1NormalBatch4Kernel,
    3,
    3,
    0,
    12,
    std::time::Duration::from_micros(750),
);
```

- [ ] **Step 2: Add suffix assertions**

Add:

```rust
assert!(suffix.contains("batch_gt1_normal_batch4_setup calls=3 blocks=3"));
assert!(suffix.contains("batch_gt1_normal_batch4_kernel calls=3 blocks=3"));
```

- [ ] **Step 3: Run CLI suffix test**

Run:

```bash
cargo test -p rllm-cli --bin llama-test q8_kernel_profile_suffix -- --nocapture
```

Expected:

```text
test tests::q8_kernel_profile_suffix_reports_recorded_rows ... ok
test tests::q8_kernel_profile_suffix_is_empty_without_profile_rows ... ok
```

## Task 4: Benchmark R106 Profiler

**Files:**
- No source changes in this task.

- [ ] **Step 1: Build release runner**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: build succeeds.

- [ ] **Step 2: Run non-profile control**

Run:

```bash
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r106-control.txt 2> target/r106-control.time
```

Expected:

- output contains `> No`
- `Peak: 1050673152 bytes`
- no `Q8KernelProfile` suffix appears because the env flag is not set

- [ ] **Step 3: Run split-profile benchmark**

Run:

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r106-profile.txt 2> target/r106-profile.time
```

Expected:

- output contains `> No`
- `Peak: 1050673152 bytes`
- `Q8KernelProfile` includes:

```text
batch_gt1_normal_batch4
batch_gt1_normal_batch4_setup
batch_gt1_normal_batch4_kernel
```

- [ ] **Step 4: Compute loop overhead for report**

From `target/r106-profile.txt`, compute:

```text
batch4_loop_overhead_ms =
    batch_gt1_normal_batch4_elapsed_ms
    - batch_gt1_normal_batch4_setup_elapsed_ms
    - batch_gt1_normal_batch4_kernel_elapsed_ms
```

Record the result in the benchmark report. If the value is negative because
per-iteration timing overhead exceeds the aggregate timer precision, report it
as instrumentation overhead and mark R107 as needing a coarser profiler before
kernel work.

## Task 5: Write Benchmark Report

**Files:**
- Create: `docs/benchmarks/trials/success/2026-06-17-r106-reeglass-q8-batch4-split-profiler.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Create report**

The report must include:

- hypothesis
- REE kernel lineage: `REEGLASS-Q8-BATCH4-SPLIT-PROFILER`
- model path
- exact commands
- non-profile control result
- split-profile result
- Q8 profile table with setup/kernel/aggregate rows
- computed batch4 loop overhead
- decision: accepted diagnostic
- next experiment: named R107 target based on measured split

Use measured values from:

```text
target/r106-control.txt
target/r106-control.time
target/r106-profile.txt
target/r106-profile.time
```

- [ ] **Step 2: Update index**

Add one row to `docs/benchmarks/trials/index.md` after R105. Include:

- date: `2026-06-17`
- report filename: `2026-06-17-r106-reeglass-q8-batch4-split-profiler.md`
- status: `success`
- kernel lineage: `REEGLASS-Q8-BATCH4-SPLIT-PROFILER`
- baseline context: R105 failed inline gate
- result context: split rows and computed overhead
- next target: the R107 recommendation from the report

- [ ] **Step 3: Scan report for unfinished markers**

Run:

```bash
rg -n "T[B]D|T[O]DO" docs/benchmarks/trials/success/2026-06-17-r106-reeglass-q8-batch4-split-profiler.md docs/benchmarks/trials/index.md
```

Expected: no matches in the R106 report or index row.

## Task 6: Final Verification and Commit

**Files:**
- All files changed by prior tasks.

- [ ] **Step 1: Run final checks**

Run:

```bash
cargo fmt --check
cargo test -p rllm-runtime q8_profile -- --nocapture
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
cargo test -p rllm-cli --bin llama-test q8_kernel_profile_suffix -- --nocapture
cargo build --release -p rllm-cli --bin llama-test
git diff --check
```

Expected:

- all commands pass
- no whitespace errors

- [ ] **Step 2: Review final diff**

Run:

```bash
git diff --stat
git diff -- crates/rllm-runtime/src/q8_profile.rs crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-cli/src/bin/llama-test.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r106-reeglass-q8-batch4-split-profiler.md docs/superpowers/plans/2026-06-17-r106-reeglass-q8-batch4-split-profiler.md
```

Expected:

- runtime code only adds opt-in profiling rows and local timing accumulators
- no non-profile math changes
- report and index match measured values

- [ ] **Step 3: Commit**

Run:

```bash
git add crates/rllm-runtime/src/q8_profile.rs crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-cli/src/bin/llama-test.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r106-reeglass-q8-batch4-split-profiler.md docs/superpowers/plans/2026-06-17-r106-reeglass-q8-batch4-split-profiler.md
git commit -m "bench(runtime): split q8 batch4 profile"
```

## Self-Review

- Spec coverage: R106 directly follows R105 and avoids another speculative kernel.
- Placeholder scan: all commands, paths, labels, thresholds, and report requirements are explicit.
- Type consistency: all new labels map from `Q8KernelPath` through existing profile rows and suffix formatting.
- Scope: no model format, container, packer, tokenizer, or non-profile runtime math changes are included.
