# R91 Q8 Kernel Microbench First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cheap, deterministic Q8 kernel benchmark lab before touching the full runtime again.

**Architecture:** R91 adds the first REE kernel lab name, `REEDOT-LAB`, plus a lab-only Q8 benchmark module and a `q8-microbench` binary in `rllm-runtime`. The binary measures current Q8 dot helper shapes and candidate variants on deterministic Llama 3.2 1B-like dimensions, writes machine-readable JSON plus a Markdown summary, and does not route any candidate into inference. Runtime optimization only happens in a later stage if a candidate wins by measurement.

**Tech Stack:** Rust `std::time::Instant`, deterministic test vectors, `serde_json`, `rllm-runtime`, benchmark docs under `docs/benchmarks`.

---

## Why This Stage Exists

R89 and R90 both failed after full-runtime implementation:

- R89 best prefill: `12.23s`, slower than R88 `10.24s`
- R90 best prefill: `18.84s`, much slower than R88 `10.24s`
- Both preserved output and memory, but their kernel ideas regressed hot-path time

R91 changes the process: no more full-runtime kernel attempts until a microbenchmark shows a strong isolated win.

## Scope

R91 is allowed to add lab code and docs only:

- allowed: REE kernel lineage naming for `REEDOT-LAB`
- allowed: `q8-microbench` binary
- allowed: deterministic Q8 kernel lab helpers
- allowed: tests proving candidate variants numerically match baseline
- allowed: benchmark trial docs
- not allowed: changing `streaming/kernels.rs` inference behavior
- not allowed: changing Llama session behavior
- not allowed: changing pack/import/container formats

## Success Gate

R91 is accepted as a useful stage if:

- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture` passes
- `cargo build --release -p rllm-runtime --bin q8-microbench` passes
- `target/release/q8-microbench --json target/r91-q8-microbench.json --markdown target/r91-q8-microbench.md --iters 2000` completes
- every candidate reports `max_abs_diff <= 0.0001`
- benchmark output includes at least these variants:
  - `baseline_i8_dot32_batch4`
  - `scaled_f32_dot32_batch4`
  - `unrolled_i8_dot32_batch4`
- a candidate may be proposed for R92 only if its median-ish elapsed time is at least `1.50x` faster than `baseline_i8_dot32_batch4`

If no candidate reaches `1.50x`, R91 still succeeds as negative evidence, but no runtime optimization plan should be executed from it.

The benchmark report must name this trial's kernel lineage as `REEDOT-LAB`.

## Files

- Create: `crates/rllm-runtime/src/q8_kernel_lab.rs`
  - Deterministic Q8 data generation.
  - Baseline and candidate micro-kernel functions.
  - Testable `run_suite(config)` API returning structured metrics.
- Modify: `crates/rllm-runtime/src/lib.rs`
  - Add `#[doc(hidden)] pub mod q8_kernel_lab;`
- Create: `crates/rllm-runtime/src/bin/q8-microbench.rs`
  - CLI wrapper around `q8_kernel_lab::run_suite`.
  - Writes JSON and Markdown result files.
- Modify: `crates/rllm-runtime/Cargo.toml`
  - Add an explicit binary target for `q8-microbench`.
- Create: `docs/benchmarks/trials/success/2026-06-16-r91-q8-kernel-microbench-first.md`
  - Created after first benchmark run with actual numbers.
- Modify: `docs/benchmarks/trials/index.md`
  - Add R91 row after measurement.

## Task 1: Add Lab API and Red Tests

**Files:**
- Create: `crates/rllm-runtime/src/q8_kernel_lab.rs`
- Modify: `crates/rllm-runtime/src/lib.rs`

- [x] **Step 1: Create initial lab module with types and baseline**

Create `crates/rllm-runtime/src/q8_kernel_lab.rs`:

```rust
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct Q8KernelBenchConfig {
    pub batch: usize,
    pub in_features: usize,
    pub blocks_per_row: usize,
    pub out_features: usize,
    pub iters: usize,
}

impl Default for Q8KernelBenchConfig {
    fn default() -> Self {
        Self {
            batch: 55,
            in_features: 2048,
            blocks_per_row: 64,
            out_features: 8192,
            iters: 2000,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Q8KernelBenchResult {
    pub variant: String,
    pub elapsed_ns: u128,
    pub checksum: f32,
    pub max_abs_diff: f32,
    pub speedup_vs_baseline: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Q8KernelBenchReport {
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub iters: usize,
    pub results: Vec<Q8KernelBenchResult>,
}

pub fn run_suite(config: Q8KernelBenchConfig) -> Q8KernelBenchReport {
    assert_eq!(config.in_features % 32, 0);
    assert_eq!(config.blocks_per_row, config.in_features / 32);
    let input = deterministic_input(config.batch, config.in_features);
    let q8 = deterministic_q8_blocks(config.blocks_per_row);
    let scale = 0.125f32;

    let (baseline_ns, baseline_output) = time_variant(config.iters, config.batch, || {
        baseline_i8_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
    });
    let baseline_checksum = checksum(&baseline_output);

    let mut results = Vec::new();
    results.push(Q8KernelBenchResult {
        variant: "baseline_i8_dot32_batch4".to_string(),
        elapsed_ns: baseline_ns,
        checksum: baseline_checksum,
        max_abs_diff: 0.0,
        speedup_vs_baseline: 1.0,
    });

    for (variant, elapsed_ns, output) in [
        {
            let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
                scaled_f32_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
            });
            ("scaled_f32_dot32_batch4", elapsed_ns, output)
        },
        {
            let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
                unrolled_i8_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
            });
            ("unrolled_i8_dot32_batch4", elapsed_ns, output)
        },
    ] {
        let max_abs_diff = max_abs_diff(&baseline_output, &output);
        results.push(Q8KernelBenchResult {
            variant: variant.to_string(),
            elapsed_ns,
            checksum: checksum(&output),
            max_abs_diff,
            speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
        });
    }

    Q8KernelBenchReport {
        batch: config.batch,
        in_features: config.in_features,
        out_features: config.out_features,
        iters: config.iters,
        results,
    }
}

fn deterministic_input(batch: usize, in_features: usize) -> Vec<f32> {
    (0..batch * in_features)
        .map(|idx| (idx as f32 % 97.0) * 0.00390625 - 0.1875)
        .collect()
}

fn deterministic_q8_blocks(blocks_per_row: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks_per_row * 34);
    for block in 0..blocks_per_row {
        bytes.extend_from_slice(&crate::tensor::f32_to_fp16(0.125).to_le_bytes());
        for idx in 0..32 {
            bytes.push((((block * 7 + idx * 3) as i16 % 17) - 8) as i8 as u8);
        }
    }
    bytes
}

fn time_variant(
    iters: usize,
    output_len: usize,
    mut f: impl FnMut() -> Vec<f32>,
) -> (u128, Vec<f32>) {
    let warmup = f();
    assert_eq!(warmup.len(), output_len);
    let started = Instant::now();
    let mut output = Vec::new();
    for _ in 0..iters {
        output = f();
        std::hint::black_box(&output);
    }
    (started.elapsed().as_nanos(), output)
}

pub fn baseline_i8_dot32_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let qs = &q8[offset + 2..offset + 34];
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            for lane in 0..4 {
                output[batch_idx + lane] += scale
                    * dot_i8_f32(qs, &input[(batch_idx + lane) * in_features + in_feature..]);
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                scale * dot_i8_f32(qs, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

pub fn scaled_f32_dot32_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = scaled_block(&q8[offset + 2..offset + 34], scale);
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            accumulate_scaled_batch4(&scaled, &input[batch_idx * in_features + in_feature..], in_features, &mut output, batch_idx);
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] += dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

pub fn unrolled_i8_dot32_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let qs = &q8[offset + 2..offset + 34];
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            let mut acc = [0.0f32; 4];
            for idx in 0..32 {
                let weight = (qs[idx] as i8) as f32;
                acc[0] += weight * input[batch_idx * in_features + in_feature + idx];
                acc[1] += weight * input[(batch_idx + 1) * in_features + in_feature + idx];
                acc[2] += weight * input[(batch_idx + 2) * in_features + in_feature + idx];
                acc[3] += weight * input[(batch_idx + 3) * in_features + in_feature + idx];
            }
            output[batch_idx] += scale * acc[0];
            output[batch_idx + 1] += scale * acc[1];
            output[batch_idx + 2] += scale * acc[2];
            output[batch_idx + 3] += scale * acc[3];
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                scale * dot_i8_f32(qs, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

fn dot_i8_f32(qs: &[u8], input: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut idx = 0usize;
    while idx + 4 <= 32 {
        acc += (qs[idx] as i8) as f32 * input[idx]
            + (qs[idx + 1] as i8) as f32 * input[idx + 1]
            + (qs[idx + 2] as i8) as f32 * input[idx + 2]
            + (qs[idx + 3] as i8) as f32 * input[idx + 3];
        idx += 4;
    }
    acc
}

fn scaled_block(qs: &[u8], scale: f32) -> [f32; 32] {
    let mut scaled = [0.0f32; 32];
    for idx in 0..32 {
        scaled[idx] = scale * (qs[idx] as i8) as f32;
    }
    scaled
}

fn dot_f32_32(weights: &[f32; 32], input: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..32 {
        acc += weights[idx] * input[idx];
    }
    acc
}

fn accumulate_scaled_batch4(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    batch_start: usize,
) {
    for idx in 0..32 {
        let weight = weights[idx];
        output[batch_start] += weight * input[idx];
        output[batch_start + 1] += weight * input[input_stride + idx];
        output[batch_start + 2] += weight * input[input_stride * 2 + idx];
        output[batch_start + 3] += weight * input[input_stride * 3 + idx];
    }
}

fn checksum(values: &[f32]) -> f32 {
    values.iter().copied().sum()
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_kernel_lab_variants_match_baseline() {
        let report = run_suite(Q8KernelBenchConfig {
            batch: 7,
            in_features: 64,
            blocks_per_row: 2,
            out_features: 16,
            iters: 2,
        });

        assert_eq!(report.results.len(), 3);
        for result in report.results.iter().skip(1) {
            assert!(
                result.max_abs_diff <= 0.0001,
                "{} diff {}",
                result.variant,
                result.max_abs_diff
            );
        }
    }
}
```

- [x] **Step 2: Expose the lab module**

Modify `crates/rllm-runtime/src/lib.rs` and add:

```rust
#[doc(hidden)]
pub mod q8_kernel_lab;
```

- [x] **Step 3: Run red/green lab tests**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
```

Expected: PASS with `q8_kernel_lab_variants_match_baseline`.

## Task 2: Add q8-microbench Binary

**Files:**
- Modify: `crates/rllm-runtime/Cargo.toml`
- Create: `crates/rllm-runtime/src/bin/q8-microbench.rs`

- [x] **Step 1: Register binary target**

Append to `crates/rllm-runtime/Cargo.toml`:

```toml
[[bin]]
name = "q8-microbench"
path = "src/bin/q8-microbench.rs"
```

- [x] **Step 2: Add CLI binary**

Create `crates/rllm-runtime/src/bin/q8-microbench.rs`:

```rust
use rllm_runtime::q8_kernel_lab::{run_suite, Q8KernelBenchConfig, Q8KernelBenchReport};
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = value_after(&args, "--json").unwrap_or_else(|| "target/r91-q8-microbench.json".to_string());
    let markdown = value_after(&args, "--markdown").unwrap_or_else(|| "target/r91-q8-microbench.md".to_string());
    let iters = value_after(&args, "--iters")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2000);

    let report = run_suite(Q8KernelBenchConfig {
        iters,
        ..Q8KernelBenchConfig::default()
    });

    write_json(&PathBuf::from(json), &report);
    write_markdown(&PathBuf::from(markdown), &report);

    for result in &report.results {
        println!(
            "{} elapsed_ns={} speedup={:.3} max_abs_diff={:.8} checksum={:.6}",
            result.variant,
            result.elapsed_ns,
            result.speedup_vs_baseline,
            result.max_abs_diff,
            result.checksum
        );
    }
}

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn write_json(path: &PathBuf, report: &Q8KernelBenchReport) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create JSON parent directory");
    }
    let body = serde_json::to_string_pretty(report).expect("serialize report");
    fs::write(path, body).expect("write JSON report");
}

fn write_markdown(path: &PathBuf, report: &Q8KernelBenchReport) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create Markdown parent directory");
    }
    let mut body = String::new();
    body.push_str("# R91 Q8 Kernel Microbench Output\n\n");
    body.push_str(&format!(
        "- batch: `{}`\n- in_features: `{}`\n- out_features: `{}`\n- iters: `{}`\n\n",
        report.batch, report.in_features, report.out_features, report.iters
    ));
    body.push_str("| variant | elapsed ns | speedup vs baseline | max abs diff | checksum |\n");
    body.push_str("|---|---:|---:|---:|---:|\n");
    for result in &report.results {
        body.push_str(&format!(
            "| {} | {} | {:.3} | {:.8} | {:.6} |\n",
            result.variant,
            result.elapsed_ns,
            result.speedup_vs_baseline,
            result.max_abs_diff,
            result.checksum
        ));
    }
    fs::write(path, body).expect("write Markdown report");
}
```

- [x] **Step 3: Build binary**

Run:

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
```

Expected: release binary builds successfully.

## Task 3: Run Microbench and Record Trial

**Files:**
- Create after benchmark: `docs/benchmarks/trials/success/2026-06-16-r91-q8-kernel-microbench-first.md`
- Modify after benchmark: `docs/benchmarks/trials/index.md`

- [x] **Step 1: Run benchmark**

Run:

```bash
target/release/q8-microbench \
  --json target/r91-q8-microbench.json \
  --markdown target/r91-q8-microbench.md \
  --iters 2000
```

Expected stdout contains all three variants:

```text
baseline_i8_dot32_batch4
scaled_f32_dot32_batch4
unrolled_i8_dot32_batch4
```

- [x] **Step 2: Create R91 trial report from measured files**

Create `docs/benchmarks/trials/success/2026-06-16-r91-q8-kernel-microbench-first.md` after the benchmark run. Use the measured table from `target/r91-q8-microbench.md` and this exact decision rule:

```markdown
# R91: Q8 Kernel Microbench First

## Status

Active.

## Hypothesis

RLLM should not run another full-runtime Q8 kernel experiment until an isolated Q8 microbenchmark shows a candidate is at least 1.50x faster than the current baseline and numerically equivalent.

## Scope

- Mode: exact-lowram kernel lab
- Model/artifact: no model load; deterministic Llama 3.2 1B-like dimensions
- Architecture: Q8_0 dot32 batch4 kernels
- Target device/profile: Mac CPU
- Expected bottleneck: CPU arithmetic / cache locality
- Bottleneck tag: CPU arithmetic

## Setup

Commands:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench --json target/r91-q8-microbench.json --markdown target/r91-q8-microbench.md --iters 2000
```

## Results

Paste the measured table from `target/r91-q8-microbench.md`.

## Analysis

State whether any candidate reached `1.50x` speedup with `max_abs_diff <= 0.0001`.

## Decision

Use one of these exact outcomes:

- `success with candidate`: only when a candidate reaches `1.50x`
- `failed as optimization source`: when no candidate reaches `1.50x`

## Next Experiment

If a candidate reaches `1.50x`, write R92 to integrate only that candidate into `streaming/kernels.rs`.
If no candidate reaches `1.50x`, write R92 as a diagnostic trace stage instead of a kernel implementation stage.
```

- [x] **Step 3: Update index row**

Add one row to `docs/benchmarks/trials/index.md` after R90. Use actual measured result text, not generic wording:

```markdown
| 2026-06-16 | 2026-06-16-r91-q8-kernel-microbench-first.md | active | deterministic Q8 Llama-like lab | exact-lowram kernel lab | CPU arithmetic | R88 full-runtime best prefill 10.24s; R89/R90 full-runtime kernel attempts regressed | measured microbench result from target/r91-q8-microbench.md | active | prevents another full-runtime kernel attempt without isolated evidence |
```

## Task 4: Verification and Commit

**Files:**
- `crates/rllm-runtime/src/q8_kernel_lab.rs`
- `crates/rllm-runtime/src/bin/q8-microbench.rs`
- `crates/rllm-runtime/Cargo.toml`
- `crates/rllm-runtime/src/lib.rs`
- `docs/benchmarks/trials/success/2026-06-16-r91-q8-kernel-microbench-first.md`
- `docs/benchmarks/trials/index.md`

- [x] **Step 1: Run verification commands**

Run:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
```

Expected:

- lab tests pass
- existing Q8 tests pass
- release microbench binary builds

- [x] **Step 2: Check no inference runtime file changed**

Run:

```bash
git diff -- crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/linear.rs crates/rllm-runtime/src/models/llama/session/mod.rs
```

Expected: no diff.

- [ ] **Step 3: Commit R91**

Run:

```bash
git add \
  crates/rllm-runtime/Cargo.toml \
  crates/rllm-runtime/src/lib.rs \
  crates/rllm-runtime/src/q8_kernel_lab.rs \
  crates/rllm-runtime/src/bin/q8-microbench.rs \
  docs/benchmarks/trials/success/2026-06-16-r91-q8-kernel-microbench-first.md \
  docs/benchmarks/trials/index.md \
  docs/superpowers/plans/2026-06-16-r91-q8-kernel-microbench-first.md
git commit -m "bench(runtime): add q8 kernel microbench lab"
```

## Self-Review

- Spec coverage: R91 creates a cheap benchmark gate before any future full-runtime Q8 kernel changes.
- Placeholder scan: The benchmark trial file is created only after measurement; no empty result rows are committed.
- Type consistency: `Q8KernelBenchConfig`, `Q8KernelBenchResult`, and `Q8KernelBenchReport` are used consistently by the lab module and binary.
- Risk: The lab duplicates current kernel shapes instead of extracting private runtime helpers. This is intentional for R91 because the goal is evidence without changing inference behavior.
