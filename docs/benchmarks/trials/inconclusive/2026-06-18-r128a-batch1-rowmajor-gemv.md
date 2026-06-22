# Trial: R128a batch1 row-major GEMV (write-once foundation)

Date: 2026-06-18
Owner: RLLM
Status: inconclusive (correct foundation; standalone speed within noise)
Folder: inconclusive

## Hypothesis

Research (llama.cpp source + arxiv) said the biggest no-repack decode lever is
holding the accumulator in a register across all in-blocks and writing `output`
ONCE per row, instead of the block-major path's per-block read-modify-write
(~2.5x per the llama.cpp kernel analysis). Test that standalone on the current q8
layout.

## Scope

- Mode: exact-lowram runtime
- REE kernel: REEBORN-Q8-SDOT (batch1 row-major)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Architecture: LLaMA 3.2 1B, Q8_0
- Target device/profile: Apple A18 Pro, single-thread
- Bottleneck tag: CPU arithmetic (per-block dot overhead)

## Setup

Added `accumulate_q8_0_chunk_int8_batch1_rowmajor`: for batch=1 row-aligned
chunks, iterate output rows; per row accumulate `w_scale[b]*a_scale[b]*sdot` into
a register across all blocks (same b-order as block-major → bit-identical f32),
write `output[out_feature]` once. Dispatched from
`accumulate_q8_0_chunk_int8_activation` when batch==1 + in_features%32==0 +
row-aligned; else the R127 block-major path.

## Results

| run | gen tok | decode tok/s (best of 5) | output |
|---|---:|---:|---|
| R127 | 24 | ~2.66 | fire `No`, sea baseline |
| R128a | 24 | ~2.94 (noisy 1.93–2.94) | byte-identical |

~1.1x, within thermal noise (machine heavily loaded after a long session). 268
runtime tests pass; output bit-identical.

## Analysis

Standalone write-once is a **much smaller win than predicted** because the
per-block cost is dominated by the `i8_dot32` call itself (asm setup + the
horizontal `addv` reduce + f32 scale), not by the output memory write (which was
L1-resident). The llama.cpp "register-hold ~2.5x" is inseparable from its
**repacked, 4-column-vectorized** inner kernel — the wins compound there, they do
not decompose into a standalone write-once.

R128a is still correct, bit-exact, and the right structural foundation: it groups
work by output row, which is the prerequisite for the actual lever (process
multiple rows per asm block with independent sdot accumulators + a shared
activation load).

## Decision

inconclusive

Reason: bit-exact and structurally sound, but the standalone speed gain is within
measurement noise on a thermally-loaded machine. Kept as the row-major foundation
for R128b. Not claimed as a speed win.

Paper value:

- use as negative evidence: standalone "write output once" is not the decode
  lever on the current q8 layout; the per-block dot (call + addv) dominates.

## Next Experiment

R128b: single-asm 4-row ILP kernel — process 4 output rows per pass with 4
independent sdot accumulator chains and ONE shared activation-block load, deferring
per-block reduces where possible. This targets the per-block dot overhead the
write-once change could not. Lab-first (q8-microbench) to de-risk the asm, then
promote. Measure on a thermally-quiet machine. R129 (later): 4-column repacked
layout for the remaining gap (costs RAM/repack — deferred per "RAM later").
