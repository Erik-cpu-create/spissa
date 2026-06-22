# R92 REEBORN-Q8 Batch1 Decode Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide with measurement whether a `REEBORN-Q8` batch1 scaled-block runtime kernel can improve decode token speed without hurting exact Q8 output or low-RAM behavior.

**Architecture:** R92 extends the R91 `REEDOT-LAB` benchmark with batch1 decode-shaped variants before touching inference. Runtime integration is allowed only if the batch1 lab candidate clears the gate; otherwise R92 stops as negative evidence and writes a failed trial report. If integration is allowed, the smallest runtime change is limited to Q8_0 complete-row batch1 paths in `streaming/kernels.rs`.

**Tech Stack:** Rust, `rllm-runtime`, `q8-microbench`, existing `llama-test` benchmark flow, benchmark docs under `docs/benchmarks`.

---

## Why This Stage Exists

R91 proved `scaled_f32_dot32_batch4` is faster in a prefill-shaped lab:

- baseline: `112889458ns`
- scaled block batch4: `47816750ns`
- speedup: `2.361x`
- max diff: `0`

That does not prove batch1 decode will improve. Current runtime already uses the scaled-block path for `config.batch > 1`, while batch1 complete-row paths still use `scale * q8_0_dot_i8_f32(...)`. R92 tests that missing decode shape first.

## Scope

Allowed:

- extend `q8_kernel_lab` with batch1 complete-row benchmark variants
- name the candidate `REEBORN-Q8-BATCH1-LAB`
- run batch1 and batch55 microbench gates
- integrate only the batch1 complete-row Q8 paths if the lab gate passes
- benchmark with the existing Llama 3.2 1B Q8 rowchunks prompt
- write one R92 trial report and update the benchmark index

Not allowed:

- changing pack/import/container formats
- adding architecture-specific SIMD
- adding allocation-heavy repack buffers
- changing prompt formatting, tokenizer behavior, or model quality logic
- touching non-Q8 runtime paths
- claiming runtime improvement from lab numbers alone

## Success Gate

R92 runtime integration is allowed only if:

- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture` passes
- `target/release/q8-microbench --json target/r92-q8-batch1.json --markdown target/r92-q8-batch1.md --iters 2000 --batch 1` completes
- every batch1 candidate reports `max_abs_diff <= 0.0001`
- `scaled_f32_dot32_batch1_row` reaches at least `1.15x` over `baseline_i8_dot32_batch1_row`

If the lab gate fails, stop and write a failed R92 report. Do not edit runtime.

If runtime integration happens, accept only if:

- `cargo test -p rllm-runtime q8_0 -- --nocapture` passes
- `cargo test -p rllm-runtime` passes
- `llama-test` output on `Answer yes or no: is fire cold?` remains `No`
- peak transient memory does not increase
- decode tok/s improves by at least `5%` on the measured prompt, or prefill does not regress while decode timing improves in phase output

If runtime benchmark fails this gate, revert runtime code and keep only lab/report negative evidence.

## Files

- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Add batch1 row-shaped baseline and scaled-block variants.
  - Keep R91 variants unchanged.
- Modify: `crates/rllm-runtime/src/bin/q8-microbench.rs`
  - No behavior change expected unless output needs clearer kernel naming.
- Conditional modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Only if lab gate passes.
  - Replace batch1 complete-row Q8 dot loops with measured scaled-block helper usage.
- Create after measurement: `docs/benchmarks/trials/success/2026-06-16-r92-reeborn-q8-batch1-decode-gate.md` or `docs/benchmarks/trials/failed/2026-06-16-r92-reeborn-q8-batch1-decode-gate.md`
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `docs/superpowers/plans/2026-06-16-r92-reeborn-q8-batch1-decode-gate.md`
  - Check off completed steps during execution.

## Task 1: Add Batch1 Lab Gate

**Files:**
- Modify: `crates/rllm-runtime/src/q8_kernel_lab.rs`

- [x] **Step 1: Add a failing test for batch1 variants**

Add this test inside the existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn q8_kernel_lab_reports_batch1_decode_gate_variants() {
    let report = run_suite(Q8KernelBenchConfig {
        batch: 1,
        in_features: 2048,
        blocks_per_row: 64,
        out_features: 8192,
        iters: 2,
    });

    let variants = report
        .results
        .iter()
        .map(|result| result.variant.as_str())
        .collect::<Vec<_>>();

    assert!(variants.contains(&"baseline_i8_dot32_batch1_row"));
    assert!(variants.contains(&"scaled_f32_dot32_batch1_row"));

    for result in report
        .results
        .iter()
        .filter(|result| result.variant.ends_with("_batch1_row"))
    {
        assert!(
            result.max_abs_diff <= 0.0001,
            "{} diff {} exceeded tolerance",
            result.variant,
            result.max_abs_diff
        );
    }
}
```

- [x] **Step 2: Run red test**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab_reports_batch1_decode_gate_variants -- --nocapture
```

Expected:

- FAIL because `baseline_i8_dot32_batch1_row` and `scaled_f32_dot32_batch1_row` do not exist yet.

- [x] **Step 3: Add batch1 benchmark helpers**

Add helpers beside the existing lab helper functions:

```rust
pub fn baseline_i8_dot32_batch1_row(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    in_features: usize,
) -> Vec<f32> {
    let blocks = q8.len() / 34;
    let mut output = vec![0.0f32; 1];
    for block in 0..blocks {
        let offset = block * 34;
        let in_feature = block * 32;
        output[0] += scale * dot_i8_f32(&q8[offset + 2..offset + 34], &input[in_feature..]);
    }
    assert_eq!(blocks * 32, in_features);
    output
}

pub fn scaled_f32_dot32_batch1_row(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    in_features: usize,
) -> Vec<f32> {
    let blocks = q8.len() / 34;
    let mut output = vec![0.0f32; 1];
    for block in 0..blocks {
        let offset = block * 34;
        let in_feature = block * 32;
        let scaled = scaled_block(&q8[offset + 2..offset + 34], scale);
        output[0] += dot_f32_32(&scaled, &input[in_feature..]);
    }
    assert_eq!(blocks * 32, in_features);
    output
}
```

- [x] **Step 4: Register batch1 variants in `run_suite`**

After the existing three R91 variants, add a conditional block for `config.batch == 1`:

```rust
if config.batch == 1 {
    let (baseline_batch1_ns, baseline_batch1_output) =
        time_variant(config.iters, 1, || {
            baseline_i8_dot32_batch1_row(&q8, scale, &input, config.in_features)
        });

    results.push(Q8KernelBenchResult {
        variant: "baseline_i8_dot32_batch1_row".to_string(),
        elapsed_ns: baseline_batch1_ns,
        checksum: checksum(&baseline_batch1_output),
        max_abs_diff: 0.0,
        speedup_vs_baseline: 1.0,
    });

    let (scaled_batch1_ns, scaled_batch1_output) =
        time_variant(config.iters, 1, || {
            scaled_f32_dot32_batch1_row(&q8, scale, &input, config.in_features)
        });

    results.push(Q8KernelBenchResult {
        variant: "scaled_f32_dot32_batch1_row".to_string(),
        elapsed_ns: scaled_batch1_ns,
        checksum: checksum(&scaled_batch1_output),
        max_abs_diff: max_abs_diff(&baseline_batch1_output, &scaled_batch1_output),
        speedup_vs_baseline: baseline_batch1_ns as f64 / scaled_batch1_ns.max(1) as f64,
    });
}
```

- [x] **Step 5: Run green tests**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
```

Expected:

- PASS.
- Both R91 and R92 lab variants are present.

## Task 2: Run Lab Gate

**Files:**
- Generated: `target/r92-q8-batch1.json`
- Generated: `target/r92-q8-batch1.md`

- [x] **Step 1: Build the microbench binary**

Run:

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
```

Expected:

- PASS.

- [x] **Step 2: Run batch1 gate**

Run:

```bash
target/release/q8-microbench \
  --json target/r92-q8-batch1.json \
  --markdown target/r92-q8-batch1.md \
  --iters 2000 \
  --batch 1
```

Expected:

- command exits 0
- `scaled_f32_dot32_batch1_row` appears in stdout and JSON
- `max_abs_diff` is `0` or `<= 0.0001`

- [x] **Step 3: Decide whether runtime integration is allowed**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("target/r92-q8-batch1.json").read_text())
rows = {row["variant"]: row for row in data["results"]}
candidate = rows["scaled_f32_dot32_batch1_row"]
print(candidate["speedup_vs_baseline"])
raise SystemExit(0 if candidate["max_abs_diff"] <= 0.0001 and candidate["speedup_vs_baseline"] >= 1.15 else 1)
PY
```

Expected:

- exit 0 means proceed to Task 3
- exit 1 means skip Task 3, write failed report in Task 4

## Task 3: Conditional Runtime Integration

Only execute this task if Task 2 Step 3 exits 0.

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [x] **Step 1: Add a named runtime marker**

Near the Q8 helpers in `streaming/kernels.rs`, add:

```rust
const REEBORN_Q8_BATCH1: &str = "REEBORN-Q8-BATCH1";
```

Do not log it yet. This constant anchors the lineage in the runtime source without changing output.

- [x] **Step 2: Update complete-row linear batch1 path**

In `accumulate_q8_0_chunk_batch1_complete_rows`, replace:

```rust
acc += scale
    * q8_0_dot_i8_f32(
        &q8_bytes[block_offset + 2..block_offset + 34],
        &input[input_start..],
        32,
    );
```

with:

```rust
let scaled = q8_0_scaled_block(&q8_bytes[block_offset + 2..block_offset + 34], scale);
acc += f32_dot_32(&scaled, &input[input_start..]);
```

- [x] **Step 3: Update complete-row multiply-into batch1 path**

In `accumulate_q8_0_chunk_multiply_into_batch1_complete_rows`, replace the same `scale * q8_0_dot_i8_f32(...)` block with:

```rust
let scaled = q8_0_scaled_block(&q8_bytes[block_offset + 2..block_offset + 34], scale);
acc += f32_dot_32(&scaled, &input[input_start..]);
```

- [x] **Step 4: Update complete-row argmax batch1 path**

In `accumulate_q8_0_chunk_argmax_batch1_complete_rows`, replace the same `scale * q8_0_dot_i8_f32(...)` block with:

```rust
let scaled = q8_0_scaled_block(&q8_bytes[block_offset + 2..block_offset + 34], scale);
acc += f32_dot_32(&scaled, &input[input_start..]);
```

- [x] **Step 5: Run Q8 tests**

Run:

```bash
cargo test -p rllm-runtime q8_0 -- --nocapture
```

Expected:

- PASS, 9 Q8 tests.

## Task 4: Benchmark and Report

**Files:**
- Create: `docs/benchmarks/trials/success/2026-06-16-r92-reeborn-q8-batch1-decode-gate.md` or `docs/benchmarks/trials/failed/2026-06-16-r92-reeborn-q8-batch1-decode-gate.md`
- Modify: `docs/benchmarks/trials/index.md`

- [x] **Step 1: Build release CLI**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected:

- PASS.

- [x] **Step 2: Run benchmark**

Run three measured runs:

```bash
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r92-run${i}.txt" 2> "target/r92-run${i}.time"
done
```

Expected:

- each run exits 0
- answer remains `No`
- phase output includes prefill/decode timing
- `/usr/bin/time -l` captures maximum RSS

- [x] **Step 3: Write report**

Use the trial-report template and include:

```text
REE kernel: REEBORN-Q8-BATCH1 if runtime integration happened
REE kernel: REEBORN-Q8-BATCH1-LAB if lab failed and runtime was skipped
```

Decision rules:

```text
accepted = runtime integration happened and benchmark gate passed
failed = lab gate failed, runtime gate failed, output changed, memory increased, or timing regressed
```

- [x] **Step 4: Update index**

Append one R92 row to `docs/benchmarks/trials/index.md` with:

```text
date | report filename | status | artifact/model | mode | bottleneck | baseline | result | decision | paper value
```

## Task 5: Final Verification and Commit

**Files:**
- all touched files

- [x] **Step 1: Run final verification**

Run:

```bash
cargo fmt --check
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-runtime
git diff --check
```

Expected:

- all commands pass

- [x] **Step 2: Confirm conditional runtime state**

If lab gate failed:

```bash
git diff -- crates/rllm-runtime/src/streaming/kernels.rs
```

Expected:

- no diff.

If runtime integration happened:

```bash
rg -n "REEBORN-Q8-BATCH1|q8_0_scaled_block|f32_dot_32" crates/rllm-runtime/src/streaming/kernels.rs
```

Expected:

- `REEBORN-Q8-BATCH1` exists
- only Q8 complete-row batch1 paths changed

- [x] **Step 3: Commit**

Run:

```bash
git add \
  crates/rllm-runtime/src/q8_kernel_lab.rs \
  crates/rllm-runtime/src/streaming/kernels.rs \
  docs/benchmarks/trials/index.md \
  docs/benchmarks/trials/success/2026-06-16-r92-reeborn-q8-batch1-decode-gate.md \
  docs/benchmarks/trials/failed/2026-06-16-r92-reeborn-q8-batch1-decode-gate.md \
  docs/superpowers/plans/2026-06-16-r92-reeborn-q8-batch1-decode-gate.md
git commit -m "bench(runtime): gate reeborn q8 batch1 decode kernel"
```

Only stage the report path that exists.

## Self-Review

- Spec coverage: The plan keeps the REE naming requirement, starts with lab evidence, and only allows runtime integration after a measurable batch1 win.
- Placeholder scan: No `TBD`, `TODO`, or empty result table is required before measurement.
- Type consistency: Variant names are exact: `baseline_i8_dot32_batch1_row`, `scaled_f32_dot32_batch1_row`, `REEBORN-Q8-BATCH1-LAB`, and `REEBORN-Q8-BATCH1`.
- Risk: The likely outcome is that batch1 scaled-block may fail because constructing a scaled block for one row can cost more than direct `i8` conversion. If so, this stage still produces useful negative evidence and avoids another R89/R90-style runtime regression.
