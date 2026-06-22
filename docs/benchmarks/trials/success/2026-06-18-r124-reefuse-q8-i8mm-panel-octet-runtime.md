# R124: REEFUSE-Q8-I8MM-PANEL output-octet runtime (ILP)

Date: 2026-06-18
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

R123 proved in the lab that 4 independent `smmla` accumulator chains (output8)
beat the single-chain `output2` panel by ~1.46x (latency-hiding, exact). R124
promotes that into the runtime panel: an output-octet kernel that processes 8
output rows per pass with 4 independent tiles, handling the runtime's per-block
per-row weight scales (the lab used a single scale).

## Scope

- Mode: exact-lowram runtime gate
- REE kernel lineage: REEFUSE-Q8-I8MM-PANEL (output-octet)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Device: Apple A18 Pro, single-thread (`RLLM_THREADS=1`)
- Bottleneck tag: CPU arithmetic / Q8 i8mm GEMM (smmla latency)

## What was done

- `smmla_accumulate_output_octet` — 8 output rows, four packed weight panels, four
  independent `smmla` accumulator tiles per K-block, activation `v0` loaded once
  per K-segment and reused across all four panels. Per-block per-row weight scales
  folded in scalar post-`smmla` (same pattern as `output_pair`; the loop-carried
  vector accumulator was already known fragile). Odd-batch scalar tail covers all
  8 rows. Same R119 inline-asm discipline: typed `out(vreg)` tiles, never a memory
  write through an `in(reg)` pointer.
- `accumulate_q8_0_chunk_panel_smmla` dispatch restructured: output-octets first
  (`r += 8`), then a pair remainder (`r += 2`, 0-3 pairs), then the odd output row
  (scalar). Covers any output-row count exactly.

## Results

Best-of-10 single-thread prefill (`Answer yes or no: is fire cold?`, output `No`):

| config | prefill |
|---|---:|
| R121 (output_pair panel) | 1.43s |
| **R124 (output_octet panel)** | **1.24s** |

- **~1.15x end-to-end** (1.43s → 1.24s). The kernel win is ~1.46x (R123) but the
  paneled matmul is not all of prefill — chunk decode, lm_head argmax, attention
  softmax/RoPE, and the per-matmul activation pack are unchanged — so the matmul
  speedup dilutes to ~1.15x overall.
- **Parity exact:** first-token full-vocab logits top-1 match (`2822` = `No`),
  top-10 10/10, max abs diff **0.3720 — identical to R121**. The octet is the same
  int8 arithmetic as `output_pair`, only the `smmla` are reordered into
  independent chains, so the result is bit-for-bit unchanged (no added error).
- 7 new octet boundary unit tests (even/odd batch × out_features hitting
  octet+pair+odd splits, multi-octet realistic) + 82 streaming + 268 runtime tests
  pass.

## Analysis

The matmul is now near its single-thread compute floor (4-chain ILP saturates the
`smmla` latency-hiding). The remaining prefill cost is increasingly **non-matmul
overhead**: streaming chunk decode (~1.5 GB raw copy across the pass), the lm_head
argmax over 128256 vocab, and the per-matmul activation quant+pack. Those, not the
GEMM, are the next levers to reach sub-1s.

Total single-thread prefill across the R-series on this model: f32 control 6.05s →
R119 3.3s → R121 1.43s → **R124 1.24s** (~4.9x over f32), output `No` throughout,
parity preserved.

## Decision

accepted

Reason: promoting the R123 output8 ILP kernel to the runtime octet cut
single-thread prefill 1.43s → 1.24s (~1.15x) with bit-for-bit identical output
(logit diff 0.3720 = R121, output `No`), validated by 7 new octet-boundary tests
plus full parity.

Paper value:

- the R118/R119 packed panel was latency-bound on one accumulator; a 4-chain
  output-octet promotes the R123 lab ILP win into the runtime, exact
- with the GEMM near its single-thread floor, the prefill bottleneck shifts to
  non-matmul overhead (chunk decode, lm_head, activation pack)

## Next Experiment

R125: attack the non-matmul prefill overhead now that the GEMM is saturated —
profile the new breakdown (decode vs lm_head vs activation-pack vs attention) and
target the largest. Candidates: panel the lm_head argmax, reduce per-matmul
activation re-quantization, and cut the per-chunk weight-panel Vec allocations
(4 panels + 4 scale Vecs are allocated per chunk in the octet path).
