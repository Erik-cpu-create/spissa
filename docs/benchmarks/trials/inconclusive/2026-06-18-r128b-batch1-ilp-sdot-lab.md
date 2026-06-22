# Trial: R128b batch1 4-row ILP sdot (lab)

Date: 2026-06-18
Owner: RLLM
Status: inconclusive (correct + asm de-risked; gain modest, no-repack ceiling reached)
Folder: inconclusive

## Hypothesis

After R128a (write-once, ~1.1x), the remaining no-repack decode lever is
instruction-level parallelism: process 4 output rows in ONE asm block with 4
independent sdot accumulator chains and a single shared activation-block load, so
the chains pipeline and hide sdot latency. Test the relative speedup in the lab
(thermal-robust) before promoting.

## Scope

- Mode: exact-lowram lab (q8-microbench)
- REE kernel: REEBORN-Q8-SDOT (batch1 4-row ILP)
- Shape: batch=1, in_features 2048, blocks_per_row 64
- Target device/profile: Apple A18 Pro, single-thread

## Setup

Added to `q8_kernel_lab.rs`: `r128_x4_baseline` (4 rows, per-row-per-block sdot via
`r128_sdot_block` = the R128a structure) and `r128_x4_ilp` (4 rows in one asm
block: 4 independent sdot chains, activation loaded once into v0/v1 and shared,
each accumulator reduced in Rust via `vaddvq_s32`, scaled per block). Same R119
asm discipline (typed `out(vreg)` tiles, no memory write via `in(reg)` pointer).

```bash
q8-microbench --batch 1 --in-features 2048 --iters 20000
```

## Results

| variant | speedup vs per-block baseline | max abs diff |
|---|---:|---:|
| `r128_x4_baseline_sdot` | 1.00x | — |
| `r128_x4_ilp_sdot` | 1.06–1.67x (median ~1.2x) | **0.00000000** |

Bit-exact (diff 0.0). Noisy (machine thermally loaded after a long session).

## Analysis

The ILP win is real but **modest (~1.2x)**, far below the ~1.7x ideal. The 8 sdots
per block are identical in count both ways; the ILP only saves 3 of 4 activation
loads per block and lets the 4 chains pipeline — but the per-block `vaddvq` reduce
+ f32 scale (which do NOT vectorize across the 4 outputs without a repack) cap the
gain.

**Key strategic finding:** no-repack kernel tweaks are tapped. R128a (~1.1x) +
R128b (~1.2x) ≈ ~1.3-1.5x total over R127 → decode ~2.66 → ~3.7 tok/s, still ~10x
short of the ~36 tok/s floor. The dominant per-block cost is the `addv` reduce,
which only the **4-column interleaved repack** removes: with weights interleaved
so one sdot's 4 lanes hold 4 DIFFERENT outputs (one K-segment each), the 4 lanes
ARE the 4 outputs — accumulate across blocks with NO per-block reduce, one
vectorized scale-FMA per block. That is llama.cpp's q8_0 4x8 GEMV and the source
of its ~13x.

Crucially, **pre-packing the interleaved layout in the `.spsa` is NOT a RAM cost**
— it is a one-time format change, a single mmap'd copy (the pack IS the storage).
llama.cpp pays RAM for it (repack at load = a 2nd resident copy, 3.30 GB); RLLM can
bake it into its own format → the decode speed at the low-RAM point. This makes the
repack the rare lever that improves decode WITHOUT a RAM tradeoff.

## Decision

inconclusive

Reason: the 4-row ILP kernel is correct (bit-exact) and the asm is de-risked, but
the standalone gain (~1.2x) confirms the no-repack ceiling. Not promoted to the
runtime; kept as validated scaffolding for the R129 kernel (which reuses the 4-row
sdot + shared-load pattern on the interleaved layout).

Paper value:

- use as limitation: no-repack batch1 kernel optimization tops out ~1.3-1.5x over
  R127; the per-block `addv` reduce is the wall.
- use as positive evidence: the 4-row ILP sdot pattern is correct and de-risked.

## Next Experiment

R129 (the real decode lever): add a 4-row-interleaved q8 weight layout to the
`.spsa` pack format, and a decode GEMV kernel that holds 4 outputs in one sdot's 4
lanes (no per-block `addv`), register-accumulates across blocks, scales once per
block. Low-RAM-compatible (format change, 1 copy). Substantial multi-stage work
(packer + loader + kernel + parity) — do it fresh on a thermally-quiet machine for
trustworthy decode measurement.
