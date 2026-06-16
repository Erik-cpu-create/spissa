# R100: REEWIDE-Q8 NEON Batch8

Date: 2026-06-16
Owner: RLLM
Status: failed lab gate
Folder: failed

## Hypothesis

R99 showed the remaining prefill hotspot is normal Q8 MLP work, especially
`mlp.gate_proj` and `mlp.down_proj`. R100 tested whether widening the existing
R98 aarch64 NEON batch4 path to batch8 could reduce loop overhead for those
normal linear projections.

## Scope

- Mode: exact-lowram lab gate
- REE kernel lineage: `REEWIDE-Q8-NEON-BATCH8-LAB`
- Model-shaped synthetic row: batch 55, in_features 2048, blocks_per_row 64
- Target runtime path if accepted: `accumulate_q8_0_chunk` full-block
  `batch_gt1_scaled`
- Bottleneck tag: CPU arithmetic / Q8 NEON batch widening

Runtime promotion was skipped because the lab gate failed.

## Setup

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r100-reewide-lab.json \
  --markdown target/r100-reewide-lab.md \
  --iters 2000 \
  --batch 55
```

## Lab Results

| variant | elapsed ns | speedup vs baseline | max abs diff |
|---|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 93016333 | 1.000x | 0.00000000 |
| `scaled_f32_dot32_batch4` | 40492083 | 2.297x | 0.00000000 |
| `scaled_f32_dot32_batch4_runtime` | 34502458 | 2.696x | 0.00000000 |
| `reelane_f32_dot32_batch4` | 48885541 | 1.903x | 0.00000000 |
| `reeflow_i8_scaled_batch4` | 44241750 | 2.102x | 0.00000000 |
| `unrolled_i8_dot32_batch4` | 89286458 | 1.042x | 0.00000000 |
| `reevec_neon_f32_dot32_batch4` | 18286917 | 5.086x | 0.00000000 |
| `reecast_neon_scale_batch4` | 17565500 | 5.295x | 0.00000000 |
| `reewide_neon_f32_dot32_batch8` | 24298208 | 3.828x | 0.00000000 |

`REEWIDE` was exact but slower than the current R98 lab winner:

- R98/R100 current winner `reecast_neon_scale_batch4`: `17.57ms`
- R100 candidate `reewide_neon_f32_dot32_batch8`: `24.30ms`
- regression vs `reecast`: about `38.3%`

## Analysis

Batch8 reduces the number of outer batch groups, but the aarch64 helper needs
eight vector accumulators and eight input vector loads per inner step. In this
shape, that added register pressure and load scheduling cost beats any loop
overhead reduction.

The result matches the earlier R85 lesson at a different layer: widening the
batch shape is not automatically a win. The existing R96/R98 batch4 NEON shape
is still the best measured lab shape for this artifact.

Because the lab gate failed, R100 did not touch runtime `streaming/kernels.rs`
and did not run runtime promotion trials.

## Decision

failed lab gate

Reason: `reewide_neon_f32_dot32_batch8` preserved exact output but was slower
than `reecast_neon_scale_batch4`, the current lab/runtime winner.

Paper value:

- useful negative evidence
- closes the "just widen to batch8" path for aarch64 NEON prefill
- supports keeping R100+ work focused on layout/data movement or a different
  projection-level strategy, not more accumulator widening

## Next Experiment

R101 should not widen the batch accumulator again. The next useful target is a
bounded gate/down layout experiment or a trace-guided data-movement reduction
that reduces repeated scale/dequant/dot work without increasing resident RAM.
