# R107: REEBUNDLE Q8 Output2 Lab

Date: 2026-06-17
Owner: RLLM
Status: accepted lab
Folder: success

## Hypothesis

R106 showed that Q8 batch4 runtime cost is not only the NEON dot helper. A large
share is setup, loop, and instrumentation around the helper. R107 tests whether
bundling two output features for the same input block can reduce loop/setup work
before attempting a real runtime change.

## Scope

- Mode: exact-lowram lab
- REE kernel lineage: `REEBUNDLE-Q8-OUTPUT2-LAB`
- Artifact shape: synthetic Llama 3.2 1B-like Q8 row pair
- Batch: 55
- Input features: 2048
- Blocks per row: 64
- Output2 layout: `output[batch_idx * 2 + row]`
- Bottleneck tag: CPU arithmetic / Q8 loop-level bundling

Runtime streaming code was not changed in R107.

## Setup

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench --json target/r107-reebundle-lab.json --markdown target/r107-reebundle-lab.md --iters 2000 --batch 55
target/release/q8-microbench --json target/r107-reebundle-lab-long.json --markdown target/r107-reebundle-lab-long.md --iters 10000 --batch 55
```

## Lab Shape

R107 uses a separate output2 baseline because the candidate computes two output
features, not one. Comparing it against the single-output baseline would inflate
or distort the speedup claim.

```text
baseline_i8_dot32_output2_batch4:
  row0 blocks + row1 blocks -> output[batch * 2]

reebundle_neon_output2_batch4:
  row0 scaled block + row1 scaled block + shared input batch slice -> output[batch * 2]
```

Both paths use exact deterministic Q8 data and compare the full interleaved
output vector.

## Results

Standard lab:

| variant | elapsed ns | speedup vs own baseline | max abs diff | checksum |
|---|---:|---:|---:|---:|
| `baseline_i8_dot32_output2_batch4` | 278282291 | 1.000x | 0.00000000 | -14.851562 |
| `reebundle_neon_output2_batch4` | 44999750 | 6.184x | 0.00000000 | -14.851562 |

Long lab:

| variant | elapsed ns | speedup vs own baseline | max abs diff | checksum |
|---|---:|---:|---:|---:|
| `baseline_i8_dot32_output2_batch4` | 1035911084 | 1.000x | 0.00000000 | -14.851562 |
| `reebundle_neon_output2_batch4` | 163494375 | 6.336x | 0.00000000 | -14.851562 |

For context, the same long run also measured current single-output lab variants:

| variant | elapsed ns | max abs diff |
|---|---:|---:|
| `reecast_neon_scale_batch4` | 106365959 | 0.00000000 |
| `reetail_neon_tail3_batch4` | 102422583 | 0.00000000 |

Those rows are not the acceptance baseline for R107, but they show the output2
candidate is doing roughly twice the work in `163.49ms`, which is promising
enough for a runtime prototype gate.

## Analysis

R107 passes the lab gate. The bundled output2 kernel is exact against its own
output2 baseline and is much faster than the scalar output2 baseline in both
standard and long runs. This does not prove runtime speedup yet: the lab does
not model chunk boundaries, row ordering, output strides, lazy loading, or
profile overhead. It does validate that an output-feature bundling direction is
more credible than another tiny callsite hint.

The runtime implementation still needs careful gating because the real
`accumulate_q8_0_chunk` currently streams one output feature per block. R108
must prove it can detect adjacent output rows safely and keep peak transient
memory unchanged.

## Decision

accepted lab

Reason: `reebundle_neon_output2_batch4` beat `baseline_i8_dot32_output2_batch4`
in the long lab by `6.336x` with `max_abs_diff=0.00000000`.

Paper value:

- positive lab evidence for loop-level Q8 output-feature bundling
- supports an R108 runtime-gated prototype
- does not claim end-to-end runtime speedup yet

## Next Experiment

R108 should create a runtime-gated `REEBUNDLE-Q8-OUTPUT2` prototype in
`accumulate_q8_0_chunk` only when two adjacent output rows are available in the
same chunk and share the same input block index. It must compare against a
same-turn runtime control and revert if output, peak transient memory, or
profile/pre-fill gates fail.
