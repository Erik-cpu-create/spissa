# R105 REEINLINE Q8 Batch4 Callsite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether forcing the Q8 batch4 hot callsite to inline reduces the real Llama 3.2 1B Q8 prefill bottleneck without changing output, memory, model format, or math.

**Architecture:** R105 is a narrow runtime-gated optimization named `REEINLINE-Q8-BATCH4-CALLSITE`. It touches only the existing Q8 normal batch4 helper call path in `streaming/kernels.rs` and keeps the same `REECAST` scaled-block data layout plus the same NEON dot algorithm. The change is accepted only if the real model profile shows `batch_gt1_normal_batch4` drops versus R103/R104 while output remains `No` and peak transient memory stays unchanged.

**Tech Stack:** Rust, `rllm-runtime`, aarch64 NEON intrinsics, existing `llama-test --profile-phases`, `RLLM_Q8_KERNEL_PROFILE=1`, `/usr/bin/time -l`.

---

## Evidence Inputs

R103 detail profile:

- `batch_gt1_scaled`: `10589.93ms`
- `batch_gt1_normal_batch4`: `3551.82ms`
- `batch_gt1_normal_tail`: `1030.26ms`
- `batch_gt1_normal_scale`: `507.11ms`

R104 rejected lesson:

- `REETAIL-Q8-NEON-TAIL3-LAB` passed synthetic lab.
- Runtime candidate did not reduce the target tail bucket.
- Final runtime code was reverted.
- R105 must not target the 3-row tail again.

## Files

- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add `#[inline(always)]` to the existing Q8 batch4 call wrapper and aarch64 implementation.
  - Optionally add `#[inline(always)]` to the existing Q8 scaled-block wrapper if the first runtime profile does not improve and source inspection confirms it is still a hot wrapper.
- Create on success: `docs/benchmarks/trials/success/2026-06-17-r105-reeinline-q8-batch4-callsite.md`
- Create on failure: `docs/benchmarks/trials/failed/2026-06-17-r105-reeinline-q8-batch4-callsite.md`
- Modify: `docs/benchmarks/trials/index.md`

No container, packer, metadata, or model artifact changes are allowed in R105.

## Gates

Correctness and memory gates:

- Visible output must remain exactly:

```text
No
```

- `Peak` in `llama-test` output must remain `1050673152 bytes`.
- `streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch` must pass.
- `cargo fmt --check` and `git diff --check` must pass.

Performance gate:

- Profiled `batch_gt1_normal_batch4` must be lower than R103 `3551.82ms`, or a same-turn pre-control must establish a higher current baseline and the candidate must improve against that same-turn baseline.
- No-profile prefill should be at least neutral versus the current R103/R104 range around `9.0s` to `9.9s`.
- If `batch_gt1_normal_batch4` does not improve, revert runtime code and report R105 as failed.

## Task 1: Capture Pre-Control

**Files:**
- No source changes.

- [ ] **Step 1: Confirm clean worktree before runtime edits**

Run:

```bash
git status --short
```

Expected: only this plan file may be uncommitted. If unrelated files are present, inspect them before proceeding and do not overwrite user changes.

- [ ] **Step 2: Build release runner**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: build succeeds.

- [ ] **Step 3: Run same-turn pre-control**

Run:

```bash
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r105-pre-control.txt 2> target/r105-pre-control.time
```

Expected:

- `target/r105-pre-control.txt` contains `> No`.
- `Peak: 1050673152 bytes`.
- The prefill line is captured for comparison.

- [ ] **Step 4: Run pre-control profile**

Run:

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r105-pre-profile.txt 2> target/r105-pre-profile.time
```

Expected:

- `target/r105-pre-profile.txt` contains `> No`.
- `Q8KernelProfile` includes `batch_gt1_normal_batch4`.
- Record `batch_gt1_normal_batch4` and `batch_gt1_scaled`.

## Task 2: Inline Batch4 Hot Helpers

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Add inline attributes to the batch4 wrapper and NEON implementation**

Change this function header:

```rust
fn accumulate_f32_dot_32_batch4_reevec(
```

to:

```rust
#[inline(always)]
fn accumulate_f32_dot_32_batch4_reevec(
```

Change this function header:

```rust
unsafe fn accumulate_f32_dot_32_batch4_neon(
```

to:

```rust
#[inline(always)]
unsafe fn accumulate_f32_dot_32_batch4_neon(
```

Do not change the function bodies in this step.

- [ ] **Step 2: Format**

Run:

```bash
cargo fmt
```

Expected: completes without output or with normal rustfmt output only.

- [ ] **Step 3: Run targeted correctness test**

Run:

```bash
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
```

Expected:

```text
test streaming::tests::streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch ... ok
```

- [ ] **Step 4: Build release runner**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: build succeeds.

## Task 3: Candidate Runtime Benchmark

**Files:**
- No additional source changes.

- [ ] **Step 1: Run three no-profile candidate trials**

Run:

```bash
for i in 1 2 3; do RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r105-run${i}.txt" 2> "target/r105-run${i}.time"; done
```

Expected for each `target/r105-run*.txt`:

- output contains `> No`
- `Peak: 1050673152 bytes`
- prefill, decode tok/s, MLP total, gate/up/down are available in `PrefillProfile`

- [ ] **Step 2: Run candidate profile**

Run:

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r105-profile.txt 2> target/r105-profile.time
```

Expected:

- output contains `> No`
- `Q8KernelProfile` includes `batch_gt1_normal_batch4`
- `batch_gt1_normal_batch4` is compared against R103 `3551.82ms` and the same-turn pre-control profile

- [ ] **Step 3: Decision check**

Accept the runtime change only if all of these are true:

```text
output == No
peak_transient == 1050673152
candidate batch_gt1_normal_batch4 < same-turn pre-control batch_gt1_normal_batch4
candidate best no-profile prefill <= same-turn pre-control prefill
```

If any condition fails, run:

```bash
git diff -- crates/rllm-runtime/src/streaming/kernels.rs
```

Then revert only the R105 runtime edit by removing the two `#[inline(always)]` attributes added in Task 2.

## Task 4: Optional Scaled-Block Wrapper Inline Retry

**Files:**
- Modify only if Task 3 has a mixed result: `crates/rllm-runtime/src/streaming/kernels.rs`

Use this task only when Task 3 shows output and memory are correct but `batch_gt1_normal_batch4` is neutral while `batch_gt1_scaled` remains high. Skip this task if Task 3 clearly passes or clearly fails.

- [ ] **Step 1: Add inline attributes to scaled-block wrappers**

Change:

```rust
fn q8_0_scaled_block_reecast(qs: &[u8], scale: f32) -> [f32; 32] {
```

to:

```rust
#[inline(always)]
fn q8_0_scaled_block_reecast(qs: &[u8], scale: f32) -> [f32; 32] {
```

Change:

```rust
unsafe fn q8_0_scaled_block_neon(qs: &[u8], scale: f32) -> [f32; 32] {
```

to:

```rust
#[inline(always)]
unsafe fn q8_0_scaled_block_neon(qs: &[u8], scale: f32) -> [f32; 32] {
```

- [ ] **Step 2: Re-run verification and candidate profile**

Run:

```bash
cargo fmt
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r105-inline-scale-profile.txt 2> target/r105-inline-scale-profile.time
```

Expected:

- output contains `> No`
- `Peak: 1050673152 bytes`
- `batch_gt1_normal_batch4` or `batch_gt1_scaled` improves versus same-turn pre-control

- [ ] **Step 3: Revert optional retry if it fails**

If the optional retry does not improve the target profile rows, remove the scaled-block inline attributes from `q8_0_scaled_block_reecast` and `q8_0_scaled_block_neon`. Keep the batch4 inline attributes only if Task 3 passed.

## Task 5: Benchmark Report

**Files:**
- Create one:
  - `docs/benchmarks/trials/success/2026-06-17-r105-reeinline-q8-batch4-callsite.md`
  - `docs/benchmarks/trials/failed/2026-06-17-r105-reeinline-q8-batch4-callsite.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Write the report**

Use the exact measured values from:

```text
target/r105-pre-control.txt
target/r105-pre-control.time
target/r105-pre-profile.txt
target/r105-pre-profile.time
target/r105-run1.txt
target/r105-run1.time
target/r105-run2.txt
target/r105-run2.time
target/r105-run3.txt
target/r105-run3.time
target/r105-profile.txt
target/r105-profile.time
target/r105-inline-scale-profile.txt
target/r105-inline-scale-profile.time
```

If `target/r105-inline-scale-profile.*` was not created, state that the optional retry was skipped because the batch4-only decision was already clear.

The report must include:

- hypothesis
- REE kernel lineage: `REEINLINE-Q8-BATCH4-CALLSITE`
- model path
- exact commands
- output correctness
- prefill/decode table
- `Q8KernelProfile` comparison table
- decision: accepted or rejected
- next experiment

- [ ] **Step 2: Update index**

Add one row to `docs/benchmarks/trials/index.md` after R104. The row must use
the exact report status folder, the exact measured `batch_gt1_normal_batch4`
profile value, the best no-profile prefill value, output correctness, and peak
transient memory. Before saving, run the scan below and make sure it returns no
matches:

```bash
rg -n "T[B]D|T[O]DO" docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r105-reeinline-q8-batch4-callsite.md docs/benchmarks/trials/failed/2026-06-17-r105-reeinline-q8-batch4-callsite.md
```

Expected: no matches in the R105 report or index row.

## Task 6: Final Verification and Commit

**Files:**
- All files changed by prior tasks.

- [ ] **Step 1: Run final checks**

Run:

```bash
cargo fmt --check
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
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
git diff -- crates/rllm-runtime/src/streaming/kernels.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r105-reeinline-q8-batch4-callsite.md docs/benchmarks/trials/failed/2026-06-17-r105-reeinline-q8-batch4-callsite.md docs/superpowers/plans/2026-06-17-r105-reeinline-q8-batch4-callsite.md
```

Expected:

- source diff contains only accepted runtime inline attributes, or no runtime source diff if the runtime gate failed and was reverted
- report and index match the measured decision

- [ ] **Step 3: Commit**

Run:

```bash
git add crates/rllm-runtime/src/streaming/kernels.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r105-reeinline-q8-batch4-callsite.md docs/benchmarks/trials/failed/2026-06-17-r105-reeinline-q8-batch4-callsite.md docs/superpowers/plans/2026-06-17-r105-reeinline-q8-batch4-callsite.md
git commit -m "bench(runtime): gate reeinline q8 batch4 callsite"
```

If one report path does not exist, remove that path from `git add` and commit the existing report path only.

## Self-Review

- Spec coverage: R105 directly follows R104's next experiment and targets the larger `batch_gt1_normal_batch4` bucket.
- Placeholder scan: the plan has concrete commands, paths, metrics, thresholds, and revert criteria.
- Type/signature consistency: all function names match existing source in `crates/rllm-runtime/src/streaming/kernels.rs`.
- Scope: no model format, container, packer, tokenizer, or memory-layout changes are included.
