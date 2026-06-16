# R101: REEDUO-Q8 NEON Block64

Date: 2026-06-16
Owner: RLLM
Status: failed lab gate
Folder: failed

## Hypothesis

R100 showed that widening the batch accumulator to eight rows is slower than the
current R98 batch4 NEON shape. R101 tested a different path: pair two adjacent
Q8 blocks into a 64-weight batch4 kernel so each pair performs one horizontal
reduction/write instead of two.

## Scope

- Mode: exact-lowram lab gate
- REE kernel lineage: `REEDUO-Q8-NEON-BLOCK64-LAB`
- Model-shaped synthetic row: batch 55, in_features 2048, blocks_per_row 64
- Target runtime path if accepted: `accumulate_q8_0_chunk` full-block
  `batch_gt1_scaled`
- Bottleneck tag: CPU arithmetic / Q8 NEON block pairing

Runtime promotion was skipped because the stable lab gate failed.

## Setup

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r101-reeduo-lab.json \
  --markdown target/r101-reeduo-lab.md \
  --iters 2000 \
  --batch 55
target/release/q8-microbench \
  --json target/r101-reeduo-lab-long.json \
  --markdown target/r101-reeduo-lab-long.md \
  --iters 10000 \
  --batch 55
```

## Lab Results

Primary 2000-iteration run:

| variant | elapsed ns | speedup vs baseline | max abs diff |
|---|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 92963458 | 1.000x | 0.00000000 |
| `scaled_f32_dot32_batch4` | 36575833 | 2.542x | 0.00000000 |
| `scaled_f32_dot32_batch4_runtime` | 33588042 | 2.768x | 0.00000000 |
| `reelane_f32_dot32_batch4` | 45843333 | 2.028x | 0.00000000 |
| `reeflow_i8_scaled_batch4` | 43204041 | 2.152x | 0.00000000 |
| `unrolled_i8_dot32_batch4` | 85936791 | 1.082x | 0.00000000 |
| `reevec_neon_f32_dot32_batch4` | 16118167 | 5.768x | 0.00000000 |
| `reecast_neon_scale_batch4` | 16670209 | 5.577x | 0.00000000 |
| `reewide_neon_f32_dot32_batch8` | 17483834 | 5.317x | 0.00000000 |
| `reeduo_neon_block64_batch4` | 16723000 | 5.559x | 0.00000000 |

Long 10000-iteration run:

| variant | elapsed ns | speedup vs baseline | max abs diff |
|---|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 396953583 | 1.000x | 0.00000000 |
| `scaled_f32_dot32_batch4` | 187386916 | 2.118x | 0.00000000 |
| `scaled_f32_dot32_batch4_runtime` | 163020708 | 2.435x | 0.00000000 |
| `reelane_f32_dot32_batch4` | 229361750 | 1.731x | 0.00000000 |
| `reeflow_i8_scaled_batch4` | 220220792 | 1.803x | 0.00000000 |
| `unrolled_i8_dot32_batch4` | 431140708 | 0.921x | 0.00000000 |
| `reevec_neon_f32_dot32_batch4` | 86372583 | 4.596x | 0.00000000 |
| `reecast_neon_scale_batch4` | 85465750 | 4.645x | 0.00000000 |
| `reewide_neon_f32_dot32_batch8` | 91322042 | 4.347x | 0.00000000 |
| `reeduo_neon_block64_batch4` | 87420375 | 4.541x | 0.00000000 |

Additional 2000-iteration repeats were noisy:

| run | `reecast_neon_scale_batch4` | `reeduo_neon_block64_batch4` | decision |
|---|---:|---:|---|
| run1 | 16670209ns | 16723000ns | `reeduo` slower |
| run2 | 21598458ns | 27121667ns | `reeduo` slower |
| run3 | 21337625ns | 20434958ns | `reeduo` faster |
| long | 85465750ns | 87420375ns | `reeduo` slower |

## Analysis

`REEDUO` preserves exact output, but it does not reliably beat the current
`REECAST` winner. The long run is the best stability check in this R101 slice,
and it shows `reeduo_neon_block64_batch4` slower by about `2.3%`.

The likely reason is that block64 reduces horizontal reductions and output
writes, but pays extra stack movement by assembling `[f32; 64]` from two scaled
blocks. The current R98 `reecast` path already keeps the hot operation small
enough for NEON scheduling, so pairing two blocks does not produce a clear win.

Because the lab gate failed, R101 did not touch runtime `streaming/kernels.rs`
and did not run runtime promotion trials.

## Decision

failed lab gate

Reason: `reeduo_neon_block64_batch4` was exact but did not beat
`reecast_neon_scale_batch4` in the long lab run.

Paper value:

- useful negative evidence
- closes the simple adjacent-block pairing path
- reinforces that R98 batch4 plus NEON scale remains the best measured Q8
  micro-kernel shape so far

## Next Experiment

R102 should stop reshaping the 32/64 weight micro-kernel. The remaining likely
path is a higher-level layout experiment: store a gate/down optimized sidecar or
row-group metadata at pack time so runtime does less per-block work without
raising resident RAM.
