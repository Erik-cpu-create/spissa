# R78 CPU Q8 Prefill Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify and reduce the CPU-only exact-Q8 Llama 3.2 1B prefill bottleneck while preserving RLLM's low-RAM, no-GPU, non-sparse quality target.

**Architecture:** Keep the change inside existing runtime and benchmark boundaries. `llama-test --profile-phases` exposes prefill and decode timing from existing `RamaSessionTurnMetrics`; `streaming/kernels.rs` owns Q8 CPU accumulation kernels; benchmark evidence is recorded under `docs/benchmarks/trials/` and summarized in `docs/benchmarks/trials/index.md`.

**Tech Stack:** Rust, Cargo tests, RLLM `.rllm` artifacts, `llama-test`, Q8_0 packed block kernels, Markdown benchmark trial reports.

---

## File Structure

- Modify: `crates/rllm-cli/src/bin/llama-test.rs`
  - Responsibility: expose existing prefill/decode phase timings in CLI output when `--profile-phases` is enabled.
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Responsibility: add a CPU Q8_0 complete-row fast path for `batch > 1` prefill.
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
  - Responsibility: test Q8_0 batch-prefill fast-path correctness.
- Create: `docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md`
  - Responsibility: benchmark report following `docs/benchmarks/templates/trial-report.md`.
- Modify: `docs/benchmarks/trials/index.md`
  - Responsibility: add or update the R78 row with measured result and decision.

## Benchmark Rules To Follow

Every benchmark task must follow `docs/benchmarks/README.md`:

- Include hypothesis, model/artifact path, exact command, runtime context, TTFT/prefill, decode tok/s, end-to-end tok/s, RSS, peak transient memory, bottleneck tag, result table, analysis, decision, and next experiment.
- Store the report under `docs/benchmarks/trials/active/` while running.
- Move the report to `success/`, `failed/`, or `inconclusive/` only after measurement.
- Update `docs/benchmarks/trials/index.md` whenever the trial is added or moved.
- Do not claim a speedup without before/after numbers.

---

### Task 1: Expose Prefill And Decode Phase Profiles

**Files:**
- Modify: `crates/rllm-cli/src/bin/llama-test.rs`

- [ ] **Step 1: Write the failing formatter test**

Replace the existing `phase_profile_suffix_reports_decode_subphases_and_overhead` test with:

```rust
#[test]
fn phase_profile_suffix_reports_prefill_and_decode_subphases() {
    let prefill = rllm_runtime::RamaSessionPhaseTimings {
        embedding_ms: 2.0,
        transformer_ms: 50.0,
        transformer_detail: rllm_runtime::RamaTransformerPhaseTimings {
            q_projection_ms: 10.0,
            k_projection_ms: 11.0,
            v_projection_ms: 12.0,
            attention_ms: 13.0,
            gate_projection_ms: 14.0,
            up_projection_ms: 15.0,
            down_projection_ms: 16.0,
            profiled_layers: 16,
            ..Default::default()
        },
        final_norm_ms: 17.0,
        lm_head_ms: 18.0,
    };
    let decode = rllm_runtime::RamaSessionPhaseTimings {
        embedding_ms: 1.0,
        transformer_ms: 20.0,
        transformer_detail: rllm_runtime::RamaTransformerPhaseTimings {
            q_projection_ms: 2.0,
            k_projection_ms: 3.0,
            v_projection_ms: 4.0,
            attention_ms: 5.0,
            gate_projection_ms: 6.0,
            up_projection_ms: 7.0,
            down_projection_ms: 8.0,
            profiled_layers: 16,
            ..Default::default()
        },
        final_norm_ms: 9.0,
        lm_head_ms: 10.0,
    };

    let suffix = format_phase_profile_suffix(prefill, 60.0, decode, 44.0);

    assert!(suffix.contains("PrefillProfile: prefill_total=60.00ms"));
    assert!(suffix.contains("profiled=87.00ms"));
    assert!(suffix.contains("attention_total=46.00ms"));
    assert!(suffix.contains("mlp_total=45.00ms"));
    assert!(suffix.contains("lm_head=18.00ms"));
    assert!(suffix.contains("DecodeProfile: decode_total=44.00ms"));
    assert!(suffix.contains("profiled=40.00ms"));
    assert!(suffix.contains("overhead=4.00ms"));
    assert!(suffix.contains("attention_total=14.00ms"));
    assert!(suffix.contains("mlp_total=21.00ms"));
    assert!(suffix.contains("lm_head=10.00ms"));
    assert!(suffix.contains("layers=16"));
}
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
cargo test -p rllm-cli --bin llama-test phase_profile_suffix_reports_prefill_and_decode_subphases -- --nocapture
```

Expected: fail with `this function takes 2 arguments but 4 arguments were supplied`.

- [ ] **Step 3: Implement minimal formatter split**

Change `format_phase_profile_suffix` to accept prefill and decode timings:

```rust
fn format_phase_profile_suffix(
    prefill_timings: RamaSessionPhaseTimings,
    prefill_wall_ms: f64,
    decode_timings: RamaSessionPhaseTimings,
    decode_wall_ms: f64,
) -> String {
    if prefill_timings.total_ms() == 0.0 && decode_timings.total_ms() == 0.0 {
        return String::new();
    }

    format!(
        "{}{}",
        format_phase_profile_segment(
            "PrefillProfile",
            "prefill",
            prefill_timings,
            prefill_wall_ms,
        ),
        format_phase_profile_segment("DecodeProfile", "decode", decode_timings, decode_wall_ms)
    )
}
```

Add helper next to it:

```rust
fn format_phase_profile_segment(
    label: &str,
    total_label: &str,
    timings: RamaSessionPhaseTimings,
    wall_ms: f64,
) -> String {
    let detail = timings.transformer_detail;
    let profiled_total_ms = timings.total_ms();
    let overhead_ms = (wall_ms - profiled_total_ms).max(0.0);
    format!(
        " | {label}: {total_label}_total={:.2}ms profiled={:.2}ms overhead={:.2}ms embedding={:.2}ms transformer={:.2}ms attention_total={:.2}ms mlp_total={:.2}ms final_norm={:.2}ms lm_head={:.2}ms layers={} q={:.2}ms k={:.2}ms v={:.2}ms attn={:.2}ms gate={:.2}ms up={:.2}ms down={:.2}ms",
        wall_ms,
        profiled_total_ms,
        overhead_ms,
        timings.embedding_ms,
        timings.transformer_ms,
        detail.attention_total_ms(),
        detail.mlp_total_ms(),
        timings.final_norm_ms,
        timings.lm_head_ms,
        detail.profiled_layers,
        detail.q_projection_ms,
        detail.k_projection_ms,
        detail.v_projection_ms,
        detail.attention_ms,
        detail.gate_projection_ms,
        detail.up_projection_ms,
        detail.down_projection_ms
    )
}
```

Update the call site:

```rust
let phase_profile_suffix = if args.profile_phases {
    format_phase_profile_suffix(
        result.metrics.prefill_phase_timings,
        result.metrics.prefill_ms,
        result.metrics.decode_phase_timings,
        result.metrics.decode_ms,
    )
} else {
    String::new()
};
```

- [ ] **Step 4: Run tests and formatter**

Run:

```bash
rustfmt --check crates/rllm-cli/src/bin/llama-test.rs
cargo test -p rllm-cli --bin llama-test
cargo test -p rllm-cli
```

Expected: all pass.

- [ ] **Step 5: Commit profiler output**

Run:

```bash
git add crates/rllm-cli/src/bin/llama-test.rs
git commit -m "feat(cli): report prefill phase profile"
```

---

### Task 2: Create R78 Active Benchmark Report

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Create active trial report**

Create `docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md`:

```markdown
# Trial: R78 CPU Q8 Prefill Kernel

Date: 2026-06-16
Owner: RLLM
Status: running
Folder: active

## Hypothesis

Llama 3.2 1B Q8 exact-lowram prefill is dominated by CPU Q8 MLP projections.
Adding a Q8_0 complete-row fast path for `batch > 1` should reduce prefill time
without changing generated text, peak transient memory, or CPU-only semantics.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Architecture: Llama 3.2 1B Instruct
- Target device/profile: local CPU-only RLLM release build
- Expected bottleneck: Q8 MLP projection prefill
- Bottleneck tag: CPU arithmetic

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf '%s\nquit\n' 'Answer yes or no: is fire cold?' \
  | target/release/llama-test \
      --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm \
      --chat-template llama3 \
      --max-new-tokens 4 \
      --profile-phases
```

Runtime context:

- build profile: release
- OS: macOS
- GPU: not used by RLLM
- relevant config: `--chat-template llama3`, `--profile-phases`

## Results

| run | prompt/input tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | | | | | | | | |
| trial | | | | | | | | |

## Analysis

Fill this after baseline and trial runs. Include the phase-profile breakdown for
prefill and decode, especially `attention_total`, `mlp_total`, `gate`, `up`,
`down`, and `lm_head`.

## Decision

needs follow-up

Reason: waiting for before/after measurements.

Paper value:

- not paper-worthy yet

## Next Experiment

Decide after the measured R78 result.
```

- [ ] **Step 2: Add active index row**

Append one row to `docs/benchmarks/trials/index.md`:

```markdown
| 2026-06-16 | 2026-06-16-r78-cpu-q8-prefill-kernel.md | active | Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm | exact-lowram | CPU arithmetic | R77 rowchunks answered 3/3 sanity prompts correctly but prefill stayed ~28-30s | running | planned | CPU-only low-RAM prefill bottleneck evidence |
```

- [ ] **Step 3: Commit active report scaffold**

Run:

```bash
git add docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md docs/benchmarks/trials/index.md
git commit -m "docs(bench): add r78 cpu q8 prefill trial"
```

---

### Task 3: Capture Baseline Phase Profile

**Files:**
- Modify: `docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md`

- [ ] **Step 1: Build release binary**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: `Finished release profile`.

- [ ] **Step 2: Run baseline profile**

Run:

```bash
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases" \
  2>&1 | tee /tmp/rllm-r78-baseline-profile-20260616.txt
```

Expected output contains:

```text
PrefillProfile: prefill_total=
DecodeProfile: decode_total=
```

- [ ] **Step 3: Update report baseline row**

Add the measured values to the `baseline` row. For the current known baseline run, use these values if rerun variance is close:

```markdown
| baseline | 55 | 2 | 28.67s | 1.29 | 0.07 | measured by `/usr/bin/time -l` | 1050673152 | output `No`; prefill transformer 27583.10ms, prefill MLP 22730.50ms, gate/up/down 7606.92/7525.10/7586.69ms |
```

- [ ] **Step 4: Commit baseline evidence**

Run:

```bash
git add docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md
git commit -m "docs(bench): record r78 baseline profile"
```

---

### Task 4: Add Q8 Complete-Row Fast Path For Batch Prefill

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] **Step 1: Write failing kernel test**

Add this test near existing Q8 row fast path tests in `crates/rllm-runtime/src/streaming/tests.rs`:

```rust
#[test]
fn q8_0_batch_prefill_row_fast_path_accumulates_complete_rows() {
    let mut row0 = [0i8; 32];
    let mut row1 = [0i8; 32];
    row0.fill(1);
    row1.fill(2);
    let mut q8 = q8_0_block_bytes(1.0, &row0);
    q8.extend_from_slice(&q8_0_block_bytes(1.0, &row1));

    let input = vec![
        1.0f32; 32
    ]
    .into_iter()
    .chain(vec![2.0f32; 32])
    .collect::<Vec<_>>();
    let mut output = vec![0.5f32, 1.5, 2.5, 3.5];
    let config = StreamingLinearConfig {
        batch: 2,
        in_features: 32,
        out_features: 2,
    };

    let used_fast_path = accumulate_q8_0_chunk_batch_complete_rows(
        &input,
        &mut output,
        &q8,
        0,
        config,
        "linear.q8.batch.rows.weight",
    )
    .unwrap();

    assert!(used_fast_path);
    assert_eq!(output, vec![32.5, 65.5, 66.5, 131.5]);
}
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
cargo test -p rllm-runtime q8_0_batch_prefill_row_fast_path_accumulates_complete_rows -- --nocapture
```

Expected: fail with `cannot find function accumulate_q8_0_chunk_batch_complete_rows`.

- [ ] **Step 3: Implement fast path helper**

In `crates/rllm-runtime/src/streaming/kernels.rs`, add this helper before `accumulate_q8_0_chunk_batch1_complete_rows`:

```rust
fn accumulate_q8_0_chunk_batch_complete_rows(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<bool> {
    let Some((first_row, row_count, blocks_per_row)) =
        q8_0_complete_row_span(q8_bytes, element_start, config)?
    else {
        return Ok(false);
    };
    let row_end = first_row.checked_add(row_count).ok_or_else(|| {
        RuntimeError::Shape("Q8_0 batch row fast path row range overflow".to_string())
    })?;
    if row_end > config.out_features {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} Q8_0 batch row fast path rows {first_row}..{row_end} exceed expected {}",
            config.out_features
        )));
    }

    for local_row in 0..row_count {
        let out_feature = first_row + local_row;
        let first_block = local_row * blocks_per_row;
        for batch_idx in 0..config.batch {
            let mut acc = output[batch_idx * config.out_features + out_feature];
            let input_base = batch_idx * config.in_features;
            for block_in_row in 0..blocks_per_row {
                let block_offset = (first_block + block_in_row) * 34;
                let scale = q8_0_block_scale(q8_bytes, block_offset);
                let input_start = input_base + block_in_row * 32;
                acc += scale
                    * q8_0_dot_i8_f32(
                        &q8_bytes[block_offset + 2..block_offset + 34],
                        &input[input_start..],
                        32,
                    );
            }
            output[batch_idx * config.out_features + out_feature] = acc;
        }
    }

    Ok(true)
}
```

Update `accumulate_q8_0_chunk` to call it before the generic loop:

```rust
if accumulate_q8_0_chunk_batch_complete_rows(
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

Remove the old direct call to `accumulate_q8_0_chunk_batch1_complete_rows` from `accumulate_q8_0_chunk`; keep the batch1 helper for multiply/argmax tests and specialized paths.

- [ ] **Step 4: Run kernel tests**

Run:

```bash
cargo test -p rllm-runtime q8_0_batch_prefill_row_fast_path_accumulates_complete_rows -- --nocapture
cargo test -p rllm-runtime q8_0_batch1_row_fast_path_accumulates_complete_rows -- --nocapture
cargo test -p rllm-runtime streaming::tests:: -- --nocapture
```

Expected: all pass.

- [ ] **Step 5: Run broader checks**

Run:

```bash
rustfmt --check crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs
cargo test -p rllm-runtime
cargo test -p rllm-cli
```

Expected: all pass.

- [ ] **Step 6: Commit kernel change**

Run:

```bash
git add crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs
git commit -m "perf(runtime): add q8 batch row fast path"
```

---

### Task 5: Benchmark R78 Trial And Decide

**Files:**
- Modify: `docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md`
- Modify: `docs/benchmarks/trials/index.md`
- Move if accepted/rejected: `docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md`

- [ ] **Step 1: Build release binary**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: `Finished release profile`.

- [ ] **Step 2: Run trial profile**

Run:

```bash
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases" \
  2>&1 | tee /tmp/rllm-r78-trial-profile-20260616.txt
```

Expected output contains:

```text
PrefillProfile: prefill_total=
DecodeProfile: decode_total=
```

- [ ] **Step 3: Update report with before/after table**

Fill the `trial` row with measured TTFT/prefill, decode tok/s, E2E tok/s, RSS, peak transient memory, output text, and phase profile. Compare against baseline:

```markdown
| trial | 55 | 2 | <measured>s | <measured> | <measured> | <measured> | <measured> | output `<text>`; prefill transformer <ms>, prefill MLP <ms>, gate/up/down <ms>/<ms>/<ms> |
```

- [ ] **Step 4: Route trial status**

If prefill improves by at least 10% and output remains `No` or equivalent correct yes/no answer, move to success:

```bash
git mv docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md docs/benchmarks/trials/success/2026-06-16-r78-cpu-q8-prefill-kernel.md
```

If improvement is under 10%, move to failed:

```bash
git mv docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md docs/benchmarks/trials/failed/2026-06-16-r78-cpu-q8-prefill-kernel.md
```

If variance makes the result unclear, move to inconclusive:

```bash
git mv docs/benchmarks/trials/active/2026-06-16-r78-cpu-q8-prefill-kernel.md docs/benchmarks/trials/inconclusive/2026-06-16-r78-cpu-q8-prefill-kernel.md
```

- [ ] **Step 5: Update index row**

Update the R78 row in `docs/benchmarks/trials/index.md` with final folder, baseline metric, trial metric, decision, and paper value.

- [ ] **Step 6: Commit benchmark decision**

Run:

```bash
git add docs/benchmarks/trials/index.md docs/benchmarks/trials
git commit -m "docs(bench): record r78 q8 prefill kernel result"
```

---

## Self-Review

- Spec coverage: The plan preserves CPU-only, low-RAM, exact/non-sparse quality. It adds profiler evidence first, then targets the measured Q8 MLP prefill bottleneck, then records benchmark evidence in the canonical docs.
- Placeholder scan: No `TBD`, unbounded TODO, or unspecified tests remain. Measurement values are explicitly filled during benchmark tasks because they must come from live runs.
- Type consistency: The plan uses existing types `RamaSessionPhaseTimings`, `RamaTransformerPhaseTimings`, `StreamingLinearConfig`, and existing Q8 helper style in `streaming/kernels.rs`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-16-r78-cpu-q8-prefill-kernel.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
