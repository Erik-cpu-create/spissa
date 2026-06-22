# Bounded Q8 Prefill Repack Tile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce prefill CPU time on Llama 3.2 1B (`q8_transformer_keepio-rowchunks.spsa`) by adding a bounded, row-tiled Q8 repack path in the streaming matmul kernels without widening transient RAM.

**Architecture:** Keep the Q8 bottleneck fix inside `crates/rllm-runtime/src/streaming/kernels.rs` and use explicit bounded scratch for dequantizing short Q8 row tiles (for both `accumulate_q8_0_chunk*` paths). No full-chunk or global Q8 dequantization. Reuse existing `--profile-phases` pipeline from `llama-test`.

**Tech Stack:** Rust, `rllm-runtime`, `rllm-cli` (`llama-test`), `streaming` kernels, benchmark traces/docs.

---

## Why This Stage Next

R85 and R86 proved scalar widening (batch8, NEON backend) is not enough. The profile evidence says prefill still spends most time in `streaming_mlp`:

- R84 prefill `13.94s`, MLP `10.7s`.
- R85 best prefill `12.68s` (failed).
- R86 best prefill `12.17s` (failed).

The only concrete next target is to cut Q8 complete-row arithmetic cost with cache-aware repacking while respecting low-RAM constraints.

## Success Gate

R87 is accepted only if all of these pass:

- Prompt result on `Answer yes or no: is fire cold?` remains exactly `No`.
- Best of three unchecked prefill runs with `--profile-phases` is `<= 11.45s` and is no worse than baseline by more than `-5%` (`0.9%` margin of noise is acceptable only if `--profile-phases` and trace files are stable across runs).
- `rllm` peak transient does not exceed `1,100,000,000 bytes` and does not increase by more than `+3%` over current `1050673152`.
- MLP prefill total is reduced vs R84 (`10703.88ms`) and R86 (`~9446.64ms`) baselines.
- Gate/Up/Down bucket traces each show at least `-10%` movement in the best run, or a minimum of one bucket reduces by `>= 500ms`.
- Runtime changes are reverted if gate fails, and the trial report is marked failed with honest numbers.

## Files

- Create: `docs/superpowers/plans/2026-06-16-r87-bounded-q8-prefill-repack.md` (this file)
- Create: `docs/benchmarks/trials/active/2026-06-16-r87-q8-prefill-repack-tile.md`
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
- If benchmark harness needs a stable run path, modify:
  - `scripts/phase79e_prefill_timing_benchmark.py` (optional, only if automated repeatability currently inconsistent)

## Task 1: Create Active Trial Skeleton

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-16-r87-q8-prefill-repack-tile.md`

- [ ] Add this report scaffold:

```markdown
# R87: Bounded Q8 Prefill Repack Tile

## Status

Active.

## Hypothesis

The remaining prefill bottleneck is exact Q8 row math. A bounded row-tile repack (small in-memory `[f32]` window) should cut branch/quant conversion overhead in complete-row Q8 paths while preserving low RAM.

## Baseline

- R83 best prefill: `11.45s` (exact, trusted benchmark mode)
- R84 baseline: `13.94s` prefill
- R85 best: `12.68s`
- R86 best: `12.17s`
- Reference peak transient: `1050673152 bytes`
- Reference buckets: down/gate/up around `3337/3354/3102 ms`

## Commands

Pending.

## Results

Pending.

## Decision

Pending.
```

- [ ] Add placeholders for all runs, command lines, trace filenames, and three-run summary tables.

## Task 2: Add Bounded Row-Tile Repack Helpers (Scalar-Correct)

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] Add constants and helpers near Q8 utilities:

```rust
const Q8_PREFILL_TILE_ROWS: usize = 4;
const Q8_PREFILL_TILE_FEATURES: usize = 32 * 8;

#[inline]
fn q8_0_rows_in_tile(row_index: usize, rows: usize, total_rows: usize) -> usize {
    (row_index + rows).min(total_rows)
        .saturating_sub(row_index)
}

#[inline]
fn q8_0_scale_and_dequantize_row(
    tile: &mut [f32],
    q8_row: &[u8],
    tile_base_row: usize,
    row_idx: usize,
    scale: f32,
) -> usize {
    let out_row = row_idx - tile_base_row;
    let row_base = out_row * Q8_PREFILL_TILE_FEATURES;
    let mut input_idx = 0usize;
    while input_idx < 32 {
        tile[row_base + input_idx] = scale * (q8_row[input_idx] as i8) as f32;
        input_idx += 1;
    }
    row_base
}
```

These names are intentional for bounded scratch only and should only be enabled in batch-1 complete-row Q8 paths.

- [ ] Add a row-tiled complete-row accumulator:

```rust
fn q8_0_complete_row_tile_len(config: StreamingLinearConfig, config_in_features: usize) -> usize {
    Q8_PREFILL_TILE_ROWS.min(config_in_features.div_ceil(32)).min(32)
}
```

- [ ] Implement:

```rust
fn accumulate_q8_0_chunk_batch1_complete_rows_tiled(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<bool> { /* batch-1 only complete-row path with bounded repack */ }
```

Pseudo-behavior:
- Keep existing guard checks (`q8_0_complete_row_span`).
- Build a fixed `[f32]` tile scratch of size `Q8_PREFILL_TILE_ROWS * Q8_PREFILL_TILE_FEATURES`.
- For each row-tile:
  - Dequantize up to `Q8_PREFILL_TILE_ROWS *` blocks × 32 bytes into scratch.
  - Run 32-way accumulation for each row in the tile using input slices.
- Return `Ok(false)` if shape is incompatible; otherwise `Ok(true)`.

## Task 3: Wire Row-Tile Path Into Q8 Kernels

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] In `accumulate_q8_0_chunk`, replace the old `batch1_complete_rows` fast path with:

```rust
if accumulate_q8_0_chunk_batch1_complete_rows_tiled(
    input,
    output,
    q8_bytes,
    element_start,
    config,
    weight_name,
)? {
    return Ok(());
}
```

- [ ] In `accumulate_q8_0_chunk_multiply_into`, add the same tile fast path for `batch == 1` with stateful output handling.
- [ ] Keep scalar fallback (`q8_0_dot_i8_f32`) for non-complete-row, batch>1, and tail-block cases unchanged.

## Task 4: Add Deterministic Tests for Row-Tile Repack Correctness

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] Add a new test for output-tile accumulation equivalence:

```rust
#[test]
fn q8_0_batch1_row_tiled_complete_rows_match_reference() {
    // Build deterministic 2-8 row weight bytes from q8_0_block_bytes().
    // Compare output from accumulate_q8_0_chunk with prebuilt scalar path output.
}
```

- [ ] Add a fallback test for partial rows and partial tail blocks:

```rust
#[test]
fn q8_0_row_tiled_repack_falls_back_for_non_aligned_inputs() { ... }
```

- [ ] Add multiply-into tiled/fallback equivalence:

```rust
#[test]
fn q8_0_batch1_row_tiled_multiply_into_matches_reference() { ... }
```

- [ ] Run targeted tests:

```sh
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-runtime streaming_tile_linear -- --nocapture
```

Expected: PASS, no shape regression on existing Q8 block tests.

## Task 5: Benchmark R87 and Record Evidence

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-16-r87-q8-prefill-repack-tile.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] Build release binary:

```sh
cargo build --release -p rllm-cli --bin llama-test
```

- [ ] Execute three runs (unchecked trust mode for speed signal, keep strict-mode baseline separately if needed):

```sh
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace /tmp/r87-rllm-trace-run1.json"
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace /tmp/r87-rllm-trace-run2.json"
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace /tmp/r87-rllm-trace-run3.json"
```

- [ ] Capture:
  - `No` response text.
  - Three prefill durations.
  - `peak` and `max RSS`.
  - `mlp`, `gate`, `down`, `up` bucket durations.

- [ ] If run passes gate:
  - Move report from `active/` to `success/` and update `docs/benchmarks/trials/index.md`.
- [ ] If run fails:
  - Revert only runtime changes, move report to `failed/` with exact measurements and reason.

## Task 6: Validation Before Closure

**Files:**
- Modify: `docs/benchmarks/trials/active/2026-06-16-r87-q8-prefill-repack-tile.md`

- [ ] Add a final section:
  - exact timing table.
  - bucket deltas vs R86/R84.
  - memory delta and output validity.

- [ ] Update `docs/benchmarks/trials/index.md` status row from `active` to `success` or `failed`.

## Post-Plan Execution Check

1. Verify every success criterion maps to a concrete test/command in the tasks.
2. Verify there are no placeholders such as `TODO`, `TBD`, or `implement this`.
3. Verify all plan tasks keep scope limited to Q8 row complete-row prefill kernels.
4. Resolve by either:
   - Commit + run benchmark report, or
   - Revert runtime and commit docs-only evidence if gate fails.

After saving the plan, choose execution mode:

**1) Subagent-Driven (recommended)** - I execute each task with independent reviews between steps, and run bench after each kernel milestone.

**2) Inline Execution** - I execute all tasks in this session and use checkpoint reviews after Task 3 and Task 5.

Which approach?
