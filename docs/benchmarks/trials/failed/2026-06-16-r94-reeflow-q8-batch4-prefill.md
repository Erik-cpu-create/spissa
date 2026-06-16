# R94: REEFLOW-Q8 Batch4 Prefill Kernel

Date: 2026-06-16
Owner: RLLM
Status: rejected at lab gate
Folder: failed

## Hypothesis

R93 showed that `batch_gt1_scaled` dominates Q8 runtime elapsed time for the
Llama 3.2 1B Q8 rowchunks prefill path. R94 tested whether a portable fused
batch4 kernel could beat the current scaled-block path by avoiding materializing
a `[f32; 32]` temporary for each Q8 block.

## Scope

- Mode: exact-lowram lab gate
- REE kernel lineage: `REEFLOW-Q8-BATCH4-LAB`
- Model shape: Llama 3.2 1B-like Q8 row, batch 55, in_features 2048
- Target runtime branch: R93 `batch_gt1_scaled`
- Bottleneck tag: CPU arithmetic / Q8 batch4 prefill

R94 did not touch runtime production code because the lab gate failed.

## Setup

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r94-reeflow-lab.json \
  --markdown target/r94-reeflow-lab.md \
  --iters 2000 \
  --batch 55
```

## Lab Results

| variant | elapsed ns | speedup vs baseline | max abs diff | checksum |
|---|---:|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 101978667 | 1.000x | 0.00000000 | -15.000977 |
| `scaled_f32_dot32_batch4` | 38622250 | 2.640x | 0.00000000 | -15.000977 |
| `reeflow_i8_scaled_batch4` | 46713375 | 2.183x | 0.00000000 | -15.000977 |
| `unrolled_i8_dot32_batch4` | 90643750 | 1.125x | 0.00000000 | -15.000977 |

## Analysis

The fused `REEFLOW-Q8-BATCH4-LAB` variant was numerically exact against the
baseline, but it did not beat the existing `scaled_f32_dot32_batch4` path. It
ran at `46,713,375ns` versus `38,622,250ns` for the current scaled-block lab
path.

The likely reason is that the current path pays for one scale materialization
per Q8 block, then reuses the f32 block across all batch lanes. The fused variant
removed the temporary, but still performed scalar conversion and scaling inside
the accumulator loop and did not create enough locality or instruction-level
parallelism to beat the existing f32 dot path.

Because the lab gate failed, R94 made no runtime code change and ran no runtime
promotion benchmark.

## Decision

rejected at lab gate

Reason: `REEFLOW-Q8-BATCH4-LAB` was exact but slower than the current
`scaled_f32_dot32_batch4` lab path.

Paper value:

- useful negative evidence that simply fusing scale into the batch4 accumulator
  is not the next viable prefill kernel
- supports keeping the current scaled-block path until a stronger batch4 kernel
  passes lab

## Next Experiment

R95 should not try another "remove the scaled block" scalar fusion. The next
candidate should target the current winner more directly, such as unrolling the
runtime-shaped f32 batch4 accumulator or adding architecture-specific dot
support behind a portable fallback.
