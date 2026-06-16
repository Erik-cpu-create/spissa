# R95: REELANE-Q8 Batch4 Accumulator

Date: 2026-06-16
Owner: RLLM
Status: rejected at lab gate
Folder: failed

## Hypothesis

R94 showed that fusing Q8 scale directly into a scalar batch4 accumulator was
slower than the existing scaled-block path. R95 tested a narrower target: keep
the scaled f32 block, but unroll the runtime-shaped four-lane accumulator while
preserving per-lane accumulation order.

## Scope

- Mode: exact-lowram lab gate
- REE kernel lineage: `REELANE-Q8-BATCH4-LAB`
- Model shape: Llama 3.2 1B-like Q8 row, batch 55, in_features 2048
- Target runtime branch: R93 `batch_gt1_scaled`
- Bottleneck tag: CPU arithmetic / Q8 batch4 prefill

R95 did not touch runtime production code because the lab gate failed.

## Setup

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r95-reelane-lab.json \
  --markdown target/r95-reelane-lab.md \
  --iters 2000 \
  --batch 55
```

## Lab Results

| variant | elapsed ns | speedup vs baseline | max abs diff | checksum |
|---|---:|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 100824084 | 1.000x | 0.00000000 | -15.000977 |
| `scaled_f32_dot32_batch4` | 38454750 | 2.622x | 0.00000000 | -15.000977 |
| `scaled_f32_dot32_batch4_runtime` | 34953292 | 2.885x | 0.00000000 | -15.000977 |
| `reelane_f32_dot32_batch4` | 47990833 | 2.101x | 0.00000000 | -15.000977 |
| `reeflow_i8_scaled_batch4` | 46779042 | 2.155x | 0.00000000 | -15.000977 |
| `unrolled_i8_dot32_batch4` | 88505708 | 1.139x | 0.00000000 | -15.000977 |

## Analysis

`REELANE-Q8-BATCH4-LAB` was exact, but slower than the runtime-shaped baseline.
The candidate measured `47,990,833ns` while `scaled_f32_dot32_batch4_runtime`
measured `34,953,292ns`.

The useful finding is that the runtime-shaped scaled f32 batch4 accumulator is
already faster in the lab than the old lab baseline that called four independent
dot kernels. That means the current runtime helper shape is not the obvious next
scalar bottleneck. Manual four-index unrolling made the compiler/runtime
scheduling worse, not better.

Because the lab gate failed, R95 made no runtime code change and ran no runtime
promotion benchmark.

## Decision

rejected at lab gate

Reason: `REELANE-Q8-BATCH4-LAB` was exact but slower than
`scaled_f32_dot32_batch4_runtime`.

Paper value:

- useful negative evidence that manual scalar unrolling of the current f32
  batch4 accumulator is not the next viable kernel
- useful diagnostic evidence that future lab gates should compare against the
  runtime-shaped baseline, not only the older four-dot lab baseline

## Next Experiment

R96 should stop trying scalar loop reshapes for this branch. The next credible
path is an architecture-aware kernel behind a portable fallback, or a data-layout
experiment that changes the amount of work the hot path performs without
materializing large resident buffers.
