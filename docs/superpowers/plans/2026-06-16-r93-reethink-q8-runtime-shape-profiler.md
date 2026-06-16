# R93 REETHINK-Q8 Runtime Shape Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add an opt-in `REETHINK-Q8-SHAPE-PROFILER` that measures the real Q8 runtime branches before designing another kernel.

**Architecture:** R93 adds a disabled-by-default runtime profiler for Q8_0 streaming paths. It records branch-level counts and elapsed time for real `llama-test` runs, then prints a compact `Q8KernelProfile` suffix only when `RLLM_Q8_KERNEL_PROFILE=1` is set. R93 must not change Q8 math, prompt formatting, model loading, or default runtime behavior.

**Tech Stack:** Rust `std::sync::{Mutex, OnceLock}`, `std::time::Instant`, `rllm-runtime`, `llama-test`, existing benchmark report workflow.

---

## Why This Stage Exists

R91 and R92 proved that microbenchmarks can be useful but insufficient:

- R91 `REEDOT-LAB` batch55 scaled block won in lab.
- R92 `REEBORN-Q8-BATCH1-LAB` batch1 scaled block won in lab.
- R92 runtime candidate failed: decode fell to `0.41/0.58/1.17 tok/s` versus R88 single-thread baseline `1.48-1.53 tok/s`.

The next failure mode to avoid is optimizing the wrong runtime branch again. R93 therefore instruments the exact runtime Q8 paths and produces measured attribution before any new kernel is attempted.

## Scope

Allowed:

- add opt-in Q8 runtime branch profiler
- expose a runtime snapshot/reset API
- print profile output in `llama-test` only when profiling is enabled
- benchmark RLLM with the same Llama 3.2 1B Q8 rowchunks prompt
- write a success or failed diagnostic trial report

Not allowed:

- changing Q8 output math
- changing `streaming/kernels.rs` branch decisions
- adding new kernel behavior
- changing pack/import/container formats
- changing prompt formatting, tokenizer, session semantics, or answer quality logic
- claiming speed improvement from R93

## Success Gate

R93 is accepted if:

- `cargo test -p rllm-runtime q8_profile -- --nocapture` passes
- `cargo test -p rllm-cli --bin llama-test q8_kernel_profile -- --nocapture` passes
- `cargo test -p rllm-runtime q8_0 -- --nocapture` passes
- default `llama-test` output is unchanged when `RLLM_Q8_KERNEL_PROFILE` is not set
- a profiled benchmark run prints `Q8KernelProfile`
- report identifies the top Q8 branch by elapsed milliseconds and call count

R93 is rejected if:

- profiler output is missing or obviously incomplete
- default runtime behavior changes
- output answer changes from `No`
- profiling overhead makes the measurement unusable and no attribution can be trusted

## Files

- Create: `crates/rllm-runtime/src/q8_profile.rs`
  - Owns profiler state, record API, snapshot/reset API, and tests.
- Modify: `crates/rllm-runtime/src/lib.rs`
  - Add `mod q8_profile;`
  - Export `q8_kernel_profile_enabled`, `q8_kernel_profile_snapshot_and_reset`, and profile data types.
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add opt-in profiling around Q8_0 branch paths only.
  - No math or branch behavior changes.
- Modify: `crates/rllm-cli/src/bin/llama-test.rs`
  - Add a `Q8KernelProfile` suffix to the metrics line only when a profile snapshot exists.
- Create: `docs/benchmarks/trials/success/2026-06-16-r93-reethink-q8-runtime-shape-profiler.md` or `docs/benchmarks/trials/failed/2026-06-16-r93-reethink-q8-runtime-shape-profiler.md`
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `docs/superpowers/plans/2026-06-16-r93-reethink-q8-runtime-shape-profiler.md`
  - Check off completed steps during execution.

## Task 1: Add Profiler State

**Files:**
- Create: `crates/rllm-runtime/src/q8_profile.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [x] **Step 1: Create profiler module with tests**

Create `crates/rllm-runtime/src/q8_profile.rs`:

```rust
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

const PROFILE_ENV: &str = "RLLM_Q8_KERNEL_PROFILE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q8KernelPath {
    BatchGt1Scaled,
    Batch1CompleteLinear,
    Batch1CompleteMultiply,
    Batch1CompleteArgmax,
    ContiguousI8Dot,
    SplitRowScalar,
}

impl Q8KernelPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BatchGt1Scaled => "batch_gt1_scaled",
            Self::Batch1CompleteLinear => "batch1_complete_linear",
            Self::Batch1CompleteMultiply => "batch1_complete_multiply",
            Self::Batch1CompleteArgmax => "batch1_complete_argmax",
            Self::ContiguousI8Dot => "contiguous_i8_dot",
            Self::SplitRowScalar => "split_row_scalar",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8KernelProfileRow {
    pub path: &'static str,
    pub calls: u64,
    pub blocks: u64,
    pub rows: u64,
    pub batch_items: u64,
    pub elapsed_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8KernelProfileSnapshot {
    pub ree_kernel: &'static str,
    pub rows: Vec<Q8KernelProfileRow>,
}

#[derive(Debug, Default)]
struct Q8KernelProfileState {
    rows: Vec<Q8KernelProfileRow>,
}

static PROFILE: OnceLock<Mutex<Q8KernelProfileState>> = OnceLock::new();

pub fn q8_kernel_profile_enabled() -> bool {
    matches!(
        std::env::var(PROFILE_ENV)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
    )
}

pub fn record_q8_kernel_path(
    path: Q8KernelPath,
    calls: u64,
    blocks: u64,
    rows: u64,
    batch_items: u64,
    elapsed: Duration,
) {
    if calls == 0 && blocks == 0 && rows == 0 && batch_items == 0 {
        return;
    }

    let mutex = PROFILE.get_or_init(|| Mutex::new(Q8KernelProfileState::default()));
    let mut state = mutex.lock().expect("Q8 profile mutex poisoned");
    let key = path.as_str();
    if let Some(row) = state.rows.iter_mut().find(|row| row.path == key) {
        row.calls += calls;
        row.blocks += blocks;
        row.rows += rows;
        row.batch_items += batch_items;
        row.elapsed_ns += elapsed.as_nanos();
        return;
    }

    state.rows.push(Q8KernelProfileRow {
        path: key,
        calls,
        blocks,
        rows,
        batch_items,
        elapsed_ns: elapsed.as_nanos(),
    });
}

pub fn q8_kernel_profile_snapshot_and_reset() -> Option<Q8KernelProfileSnapshot> {
    let mutex = PROFILE.get()?;
    let mut state = mutex.lock().expect("Q8 profile mutex poisoned");
    if state.rows.is_empty() {
        return None;
    }
    let mut rows = std::mem::take(&mut state.rows);
    rows.sort_by(|left, right| right.elapsed_ns.cmp(&left.elapsed_ns));
    Some(Q8KernelProfileSnapshot {
        ree_kernel: "REETHINK-Q8-SHAPE-PROFILER",
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_profile_records_sorts_and_resets_rows() {
        let _ = q8_kernel_profile_snapshot_and_reset();

        record_q8_kernel_path(
            Q8KernelPath::ContiguousI8Dot,
            1,
            2,
            3,
            4,
            Duration::from_nanos(10),
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1Scaled,
            1,
            2,
            3,
            4,
            Duration::from_nanos(30),
        );
        record_q8_kernel_path(
            Q8KernelPath::ContiguousI8Dot,
            2,
            3,
            4,
            5,
            Duration::from_nanos(20),
        );

        let snapshot = q8_kernel_profile_snapshot_and_reset().unwrap();
        assert_eq!(snapshot.ree_kernel, "REETHINK-Q8-SHAPE-PROFILER");
        assert_eq!(snapshot.rows[0].path, "contiguous_i8_dot");
        assert_eq!(snapshot.rows[0].calls, 3);
        assert_eq!(snapshot.rows[0].elapsed_ns, 30);
        assert_eq!(snapshot.rows[1].path, "batch_gt1_scaled");
        assert!(q8_kernel_profile_snapshot_and_reset().is_none());
    }
}
```

- [x] **Step 2: Export profiler API from runtime**

In `crates/rllm-runtime/src/lib.rs`, add near the module list:

```rust
mod q8_profile;
```

Add near the public exports:

```rust
pub use q8_profile::{
    q8_kernel_profile_enabled, q8_kernel_profile_snapshot_and_reset,
    record_q8_kernel_path, Q8KernelPath, Q8KernelProfileRow,
    Q8KernelProfileSnapshot,
};
```

- [x] **Step 3: Run profiler tests**

Run:

```bash
cargo test -p rllm-runtime q8_profile -- --nocapture
```

Expected:

- PASS.

## Task 2: Instrument Q8 Runtime Branches

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [x] **Step 1: Import profiler helpers**

At the top of `crates/rllm-runtime/src/streaming/kernels.rs`, add:

```rust
use crate::{q8_kernel_profile_enabled, record_q8_kernel_path, Q8KernelPath};
use std::time::Instant;
```

If `std::time::Instant` is already imported in the file, reuse the existing import.

- [x] **Step 2: Add branch counters to `accumulate_q8_0_chunk`**

Inside `accumulate_q8_0_chunk`, after the batch1 complete-row early return check, add:

```rust
let profile_enabled = q8_kernel_profile_enabled();
let profile_started = profile_enabled.then(Instant::now);
let mut scaled_blocks = 0u64;
let mut contiguous_blocks = 0u64;
let mut split_blocks = 0u64;
let mut scaled_batch_items = 0u64;
let mut contiguous_batch_items = 0u64;
let mut split_batch_items = 0u64;
```

Inside the `config.batch > 1 && block_len == 32` branch, increment:

```rust
scaled_blocks += 1;
scaled_batch_items += config.batch as u64;
```

Inside the contiguous `else if in_feature + block_len <= config.in_features` branch, increment:

```rust
contiguous_blocks += 1;
contiguous_batch_items += config.batch as u64;
```

Inside the split-row `else` branch, increment:

```rust
split_blocks += 1;
split_batch_items += config.batch as u64;
```

Before `Ok(())`, add:

```rust
if let Some(started) = profile_started {
    let elapsed = started.elapsed();
    record_q8_kernel_path(
        Q8KernelPath::BatchGt1Scaled,
        u64::from(scaled_blocks > 0),
        scaled_blocks,
        0,
        scaled_batch_items,
        elapsed,
    );
    record_q8_kernel_path(
        Q8KernelPath::ContiguousI8Dot,
        u64::from(contiguous_blocks > 0),
        contiguous_blocks,
        0,
        contiguous_batch_items,
        elapsed,
    );
    record_q8_kernel_path(
        Q8KernelPath::SplitRowScalar,
        u64::from(split_blocks > 0),
        split_blocks,
        0,
        split_batch_items,
        elapsed,
    );
}
```

This deliberately records shared elapsed time for branch groups in the same chunk pass. R93 needs shape attribution first, not cycle-perfect per-branch profiling.

- [x] **Step 3: Instrument batch1 complete-row early return paths**

In each complete-row helper, wrap the row loop with timing only when profiling is enabled:

```rust
let profile_started = q8_kernel_profile_enabled().then(Instant::now);
```

Before `Ok(true)` in `accumulate_q8_0_chunk_batch1_complete_rows`, add:

```rust
if let Some(started) = profile_started {
    record_q8_kernel_path(
        Q8KernelPath::Batch1CompleteLinear,
        1,
        (row_count * blocks_per_row) as u64,
        row_count as u64,
        config.batch as u64,
        started.elapsed(),
    );
}
```

Before `Ok(true)` in `accumulate_q8_0_chunk_multiply_into_batch1_complete_rows`, add the same call with `Q8KernelPath::Batch1CompleteMultiply`.

Before `Ok(true)` in `accumulate_q8_0_chunk_argmax_batch1_complete_rows`, add the same call with `Q8KernelPath::Batch1CompleteArgmax`.

- [x] **Step 4: Run Q8 tests**

Run:

```bash
cargo test -p rllm-runtime q8_0 -- --nocapture
```

Expected:

- PASS.
- No output changes when `RLLM_Q8_KERNEL_PROFILE` is not set.

## Task 3: Print Profile in `llama-test`

**Files:**
- Modify: `crates/rllm-cli/src/bin/llama-test.rs`

- [x] **Step 1: Add formatter function**

Add near `format_phase_profile_suffix`:

```rust
fn format_q8_kernel_profile_suffix() -> String {
    let Some(snapshot) = rllm_runtime::q8_kernel_profile_snapshot_and_reset() else {
        return String::new();
    };
    let mut suffix = format!(" | Q8KernelProfile: ree={}", snapshot.ree_kernel);
    for row in snapshot.rows.iter().take(6) {
        suffix.push_str(&format!(
            " {} calls={} blocks={} rows={} batch_items={} elapsed_ms={:.2}",
            row.path,
            row.calls,
            row.blocks,
            row.rows,
            row.batch_items,
            row.elapsed_ns as f64 / 1_000_000.0
        ));
    }
    suffix
}
```

- [x] **Step 2: Add formatter tests**

Add tests inside `#[cfg(test)] mod tests`:

```rust
#[test]
fn q8_kernel_profile_suffix_is_empty_without_profile_rows() {
    let _ = rllm_runtime::q8_kernel_profile_snapshot_and_reset();
    assert_eq!(format_q8_kernel_profile_suffix(), "");
}

#[test]
fn q8_kernel_profile_suffix_reports_recorded_rows() {
    let _ = rllm_runtime::q8_kernel_profile_snapshot_and_reset();
    rllm_runtime::record_q8_kernel_path(
        rllm_runtime::Q8KernelPath::BatchGt1Scaled,
        1,
        2,
        3,
        4,
        std::time::Duration::from_micros(1500),
    );
    let suffix = format_q8_kernel_profile_suffix();
    assert!(suffix.contains("Q8KernelProfile: ree=REETHINK-Q8-SHAPE-PROFILER"));
    assert!(suffix.contains("batch_gt1_scaled"));
    assert!(suffix.contains("elapsed_ms=1.50"));
    assert_eq!(format_q8_kernel_profile_suffix(), "");
}
```

- [x] **Step 3: Append suffix to metrics line**

In the turn result block, after `phase_profile_suffix`, add:

```rust
let q8_kernel_profile_suffix = format_q8_kernel_profile_suffix();
```

Update the metrics `println!` format to append one more `{}` before the closing bracket and pass `q8_kernel_profile_suffix` as the final argument.

- [x] **Step 4: Run CLI tests**

Run:

```bash
cargo test -p rllm-cli --bin llama-test q8_kernel_profile -- --nocapture
```

Expected:

- PASS.

## Task 4: Benchmark R93

**Files:**
- Generated: `target/r93-q8-profile-run1.txt`
- Generated: `target/r93-q8-profile-run1.time`
- Generated: `target/r93-q8-profile-run2.txt`
- Generated: `target/r93-q8-profile-run2.time`

- [x] **Step 1: Build release CLI**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected:

- PASS.

- [x] **Step 2: Run default control without profiler**

Run:

```bash
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r93-control.txt 2> target/r93-control.time
```

Expected:

- output includes `No`
- output does not include `Q8KernelProfile`

- [x] **Step 3: Run profiled benchmark**

Run two profiled runs:

```bash
for i in 1 2; do
  RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r93-q8-profile-run${i}.txt" 2> "target/r93-q8-profile-run${i}.time"
done
```

Expected:

- each run includes `No`
- each run includes `Q8KernelProfile`
- profile rows include elapsed milliseconds and block counts

- [x] **Step 4: Extract profile rows**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
for path in sorted(Path("target").glob("r93-q8-profile-run*.txt")):
    text = path.read_text()
    line = next((line for line in text.splitlines() if "Q8KernelProfile:" in line), "")
    print(f"{path.name}: {line}")
PY
```

Expected:

- prints one profile line per run
- top rows are readable enough to identify the dominant Q8 path

## Task 5: Report and Final Verification

**Files:**
- Create: `docs/benchmarks/trials/success/2026-06-16-r93-reethink-q8-runtime-shape-profiler.md` or `docs/benchmarks/trials/failed/2026-06-16-r93-reethink-q8-runtime-shape-profiler.md`
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `docs/superpowers/plans/2026-06-16-r93-reethink-q8-runtime-shape-profiler.md`

- [x] **Step 1: Write report**

If the profiler printed usable rows, create the success report. Include:

```text
REE kernel: REETHINK-Q8-SHAPE-PROFILER
Control output: No
Profiled output: No
Top Q8 path by elapsed_ms:
Top Q8 path by blocks:
Peak transient:
Max RSS:
Decision:
Next experiment:
```

If the profiler did not print usable rows, create the failed report with the command output and failure reason.

- [x] **Step 2: Update benchmark index**

Append one R93 row to `docs/benchmarks/trials/index.md` with:

```text
2026-06-16 | 2026-06-16-r93-reethink-q8-runtime-shape-profiler.md | success or failed | Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm | exact-lowram diagnostic, REETHINK-Q8-SHAPE-PROFILER | CPU arithmetic / Q8 runtime branch attribution | R92 showed lab/runtime mismatch | profile identifies dominant branch or fails to produce attribution | decision | paper value
```

- [x] **Step 3: Run final verification**

Run:

```bash
cargo fmt --check
cargo test -p rllm-runtime q8_profile -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-cli --bin llama-test q8_kernel_profile -- --nocapture
cargo test -p rllm-runtime
git diff --check
```

Expected:

- all commands pass

- [x] **Step 4: Commit R93**

Run:

```bash
git add \
  crates/rllm-runtime/src/q8_profile.rs \
  crates/rllm-runtime/src/lib.rs \
  crates/rllm-runtime/src/streaming/kernels.rs \
  crates/rllm-cli/src/bin/llama-test.rs \
  docs/benchmarks/trials/index.md \
  docs/benchmarks/trials/success/2026-06-16-r93-reethink-q8-runtime-shape-profiler.md \
  docs/benchmarks/trials/failed/2026-06-16-r93-reethink-q8-runtime-shape-profiler.md \
  docs/superpowers/plans/2026-06-16-r93-reethink-q8-runtime-shape-profiler.md
git commit -m "bench(runtime): add q8 runtime shape profiler"
```

Only stage the report path that exists.

## Self-Review

- Spec coverage: R93 explains the R91/R92 lab/runtime mismatch and produces runtime branch attribution before any new kernel work.
- Placeholder scan: The report fields are concrete labels to fill from measured output; no `TBD`, `TODO`, or unspecified code remains.
- Type consistency: `Q8KernelPath`, `Q8KernelProfileRow`, and `Q8KernelProfileSnapshot` are consistently named across runtime and CLI.
- Risk: Profiling adds overhead when enabled, so R93 must not compare profiled timing directly against non-profiled timing for speed claims. It is an attribution stage, not an optimization stage.
