# Trial: R129 interleaved 4-output GEMV (lab) — REGRESSION

Date: 2026-06-18
Owner: RLLM
Status: rejected
Folder: failed

## Hypothesis

R128b found a no-repack ceiling and pointed to the 4-column interleaved repack
(llama.cpp's q8_0 4x8 GEMV) as the real lever: lay weights so one `sdot`'s 4 lanes
hold 4 DIFFERENT outputs → no per-block `addv`, one vectorized scale-FMA per block.
This would justify a `.rllm` weight-format change. Test the interleaved kernel in
the lab BEFORE committing to the format change.

## Scope

- Mode: exact-lowram lab (q8-microbench)
- REE kernel: REEBORN-Q8-SDOT (interleaved 4-output)
- Shape: batch=1, in_features 2048, blocks_per_row 64
- Target device/profile: Apple A18 Pro, single-thread

## Setup

`q8_kernel_lab.rs`: `pack_w_interleaved_x4` (4 rows → 8 segments × `[r0|r1|r2|r3]`
16-byte groups per block), `r129_x4_interleaved` (1 int32 tile, 8 `sdot`,
`ld1r`-broadcast activation, no addv, `vfmaq` scale once/block) and
`r129_x4_interleaved_2t` (8 segments split across 2 tiles for ILP). Compared to the
R128b no-repack kernels (`r128_x4_ilp` = 4 independent sdot chains).

## Results

batch1, in=2048, best of 3 (speedup vs the per-block baseline):

| kernel | layout | chains | speedup | diff |
|---|---|---:|---:|---:|
| `r128_x4_ilp` (R128b) | none (current q8) | 4 | **~1.6x** | 0.0 |
| `r129_x4_interleaved` | interleaved | 1 | ~1.1–1.27x | 2.4e-7 |
| `r129_x4_interleaved_2t` | interleaved | 2 | ~1.1–1.19x | 2.4e-7 |

The interleaved kernels are CORRECT (diff = fma rounding only) but both **slower
than R128b's non-interleaved 4-chain ILP**.

## Analysis

The interleaved repack does NOT reduce the `sdot` count for a fixed output count
(still 8 sdots per 4 outputs per block) — it only moves the reduction from
per-output `addv` into lane-packing. Its cost: the 4 outputs share 1–2 accumulator
tiles, so only 1–2 independent dependency chains, vs R128b's 4 independent
per-row chains. On these cores, **ILP (more independent chains) matters more than
removing the `addv`** — so the interleave is a net regression here.

This overturns the R128b doc's projection that the interleaved `.rllm` format was
the lever. It is not — at least not for batch=1 GEMV in this form. (llama.cpp's ~3x
from the repack appears to come from contiguous bandwidth at scale + its overall
kernel structure, not from the per-block instruction mix in isolation.)

## Decision

rejected

Reason: the interleaved 4-output kernel (1-tile and 2-tile) is measurably slower
than the no-repack 4-row ILP kernel (R128b), so a `.rllm` interleaved weight-format
change would REGRESS decode, not improve it. Lab-first measurement avoided a large
wasted format migration.

Paper value:

- use as negative evidence: the 4-column interleaved repack does not beat a
  4-independent-chain non-interleaved sdot GEMV for batch=1 on Apple A18; ILP wins
  over addv-elimination.

## Next Experiment

Promote R128b (the no-repack 4-row ILP kernel, ~1.6x in lab) to the runtime batch1
path — the real, format-free decode win. Then re-profile on a cool machine to see
the new floor and the next bottleneck.
