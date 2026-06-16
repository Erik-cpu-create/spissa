# R102: REESIDE-Q8 Prescaled Sidecar

Date: 2026-06-16
Owner: RLLM
Status: failed lab gate
Folder: failed

## Hypothesis

R99 showed the remaining prefill bottleneck is Q8 `batch_gt1_scaled`, while R100
and R101 showed that batch8 widening and adjacent block64 pairing do not beat
the current R98 kernel. R102 tested whether moving Q8 scale/dequant out of the
runtime hot loop into a pack-time pre-scaled sidecar would be enough to beat
`REECAST`.

## Scope

- Mode: exact-lowram lab gate
- REE kernel lineage: `REESIDE-Q8-PRESCALED-SIDECAR-LAB`
- Model-shaped synthetic row: batch 55, in_features 2048, blocks_per_row 64
- Sidecar shape: one `[f32; 32]` pre-scaled block per Q8 block
- Bottleneck tag: Q8 pack-time sidecar / runtime dequant avoidance

Runtime, packer, and container changes were skipped because the lab gate failed.

## Setup

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r102-reeside-lab.json \
  --markdown target/r102-reeside-lab.md \
  --iters 2000 \
  --batch 55
target/release/q8-microbench \
  --json target/r102-reeside-lab-long.json \
  --markdown target/r102-reeside-lab-long.md \
  --iters 10000 \
  --batch 55
```

## Lab Results

Primary 2000-iteration run:

| variant | elapsed ns | speedup vs baseline | max abs diff |
|---|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 98828459 | 1.000x | 0.00000000 |
| `scaled_f32_dot32_batch4` | 35257500 | 2.803x | 0.00000000 |
| `scaled_f32_dot32_batch4_runtime` | 31736333 | 3.114x | 0.00000000 |
| `reelane_f32_dot32_batch4` | 45815875 | 2.157x | 0.00000000 |
| `reeflow_i8_scaled_batch4` | 42256125 | 2.339x | 0.00000000 |
| `unrolled_i8_dot32_batch4` | 84350583 | 1.172x | 0.00000000 |
| `reevec_neon_f32_dot32_batch4` | 17315084 | 5.708x | 0.00000000 |
| `reecast_neon_scale_batch4` | 16861167 | 5.861x | 0.00000000 |
| `reewide_neon_f32_dot32_batch8` | 18403875 | 5.370x | 0.00000000 |
| `reeduo_neon_block64_batch4` | 18053708 | 5.474x | 0.00000000 |
| `reeside_prescaled_f32_batch4` | 18310208 | 5.397x | 0.00000000 |

Long 10000-iteration run:

| variant | elapsed ns | speedup vs baseline | max abs diff |
|---|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 413013834 | 1.000x | 0.00000000 |
| `scaled_f32_dot32_batch4` | 191895334 | 2.152x | 0.00000000 |
| `scaled_f32_dot32_batch4_runtime` | 172243875 | 2.398x | 0.00000000 |
| `reelane_f32_dot32_batch4` | 239257166 | 1.726x | 0.00000000 |
| `reeflow_i8_scaled_batch4` | 232151959 | 1.779x | 0.00000000 |
| `unrolled_i8_dot32_batch4` | 445044083 | 0.928x | 0.00000000 |
| `reevec_neon_f32_dot32_batch4` | 89579625 | 4.611x | 0.00000000 |
| `reecast_neon_scale_batch4` | 89522042 | 4.614x | 0.00000000 |
| `reewide_neon_f32_dot32_batch8` | 95126542 | 4.342x | 0.00000000 |
| `reeduo_neon_block64_batch4` | 89586167 | 4.610x | 0.00000000 |
| `reeside_prescaled_f32_batch4` | 96465625 | 4.281x | 0.00000000 |

## Storage Implication

The tested sidecar stores 32 pre-scaled `f32` values per Q8 block:

- Q8_0 block: `34 bytes`
- REESIDE sidecar block: `128 bytes`
- storage multiplier: about `3.76x` over Q8_0 blocks

This would be acceptable only if it produced a material runtime win. It did not
in this lab.

## Analysis

`REESIDE` was exact, but it was slower than `REECAST` in both the standard and
long runs. The long run is the decision source:

- `reecast_neon_scale_batch4`: `89.52ms`
- `reeside_prescaled_f32_batch4`: `96.47ms`
- `REESIDE` regression: about `7.8%`

This means R98's NEON scale/dequant is not the dominant remaining cost inside
the lab shape. Pre-scaling weights into a larger f32 sidecar increases memory
traffic enough to lose despite removing Q8 scale/dequant from the timed loop.

Because the lab gate failed, R102 did not change `.rllm`, packer, importer, or
runtime streaming paths.

## Decision

failed lab gate

Reason: `reeside_prescaled_f32_batch4` preserved exact output but did not beat
`reecast_neon_scale_batch4`; the storage tradeoff is not justified.

Paper value:

- useful negative evidence
- closes the simple pre-scaled f32 sidecar path
- confirms the remaining bottleneck is not just per-block scale/dequant

## Next Experiment

R103 should stop targeting Q8 scale/dequant and simple f32 sidecars. The next
candidate should profile or reduce higher-level overhead around the streaming
linear loop: chunk/event frequency, output indexing/write pattern, or row-group
scheduling for gate/down without expanding weights to f32.
