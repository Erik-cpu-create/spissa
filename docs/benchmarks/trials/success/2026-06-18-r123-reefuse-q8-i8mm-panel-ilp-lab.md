# R123: REEFUSE-Q8-I8MM-PANEL ILP (output4 / output8 lab)

Date: 2026-06-18
Owner: RLLM
Status: accepted lab
Folder: success

## Hypothesis

After R121 (all three MLP projections paneled), single-thread prefill is ~1.43s
and threading is a measured dead-end (the streaming/low-RAM design forces
per-chunk `thread::scope` spawns whose overhead + cold thread-local activation
cache cancel the parallel gain; measured neutral at every thread count). The
prefill is **compute-bound on `smmla` throughput**: ~53 G MACs / 32 MACs-per-smmla
≈ 1.66 G smmla, and the R118/R119 panel kernel (`output2`) serializes the 4
`smmla` of each K-block into ONE accumulator register — a 4-deep dependency chain
that stalls on `smmla`'s ~3-cycle latency. Hypothesis: more **independent
accumulator chains** (instruction-level parallelism) hide the latency.

## Scope

- Mode: exact-lowram lab (REE microbench, `q8-microbench` bin)
- REE kernel lineage: REEFUSE-Q8-I8MM-PANEL (ILP variants output4 / output8)
- Shape: synthetic Llama-prefill panel, batch 55, in_features 2048, blocks_per_row 64
- Device: Apple A18 Pro

## What was added (`q8_kernel_lab.rs`)

- `reefuse_smmla_panel_output4` — 4 output rows (two weight row-pairs), **2
  independent accumulator tiles**, activation `v0` loaded once per K-segment and
  reused across both weight panels (halves activation loads).
- `reefuse_smmla_panel_output8` — 8 output rows (four weight row-pairs), **4
  independent accumulator tiles**, activation loaded once and reused across all
  four weight panels (quarters activation loads). Odd-batch scalar tail included.
- Honest baselines: `output2_x2(4rows)` / `output2_x4(8rows)` run the current
  `output2` panel two/four times over the same rows. Both pack outside the timed
  loop; same R119 inline-asm discipline (typed `out(vreg)` tiles, never a memory
  write through an `in(reg)` pointer).

## Results

Speedup over the current panel kernel for the same output rows (best-of-3,
realistic shape, iters 4000-5000):

| kernel | indep. chains | speedup vs current panel | max abs diff |
|---|---:|---:|---:|
| `output2` (R118/R119, current) | 1 | 1.00x | — |
| `output4` | 2 | 1.19–1.36x | 0.00988 |
| **`output8`** | **4** | **1.46–1.47x** | 0.00988 |

Correctness exact: `max_abs_diff = 0.00988` is the R111-validated int8
activation-quant error, identical to `output2` — the ILP restructure adds no
error. Lab correctness test (`q8_kernel_lab_reports_required_ree_variants`) +
batch1 gate test pass.

## Analysis

ILP scales with independent chains: 1 chain (output2) → 2 (output4, ~1.2–1.36x)
→ 4 (output8, ~1.46x), confirming the kernel was latency-bound on the single
accumulator's dependency chain, not on `smmla` issue width or memory. output8's
1.46x is stable, suggesting we are near the practical chain-count ceiling (4
chains saturate the latency-hiding; more chains add register pressure with little
gain). The activation-load amortization (4x for output8) compounds the win.

Projected runtime impact: the paneled matmuls are ~90% of the 1.43s prefill, so a
1.46x kernel should take prefill to roughly **~1.0s** — the sub-1s target — while
staying exact.

## Decision

accepted lab

Reason: `output8` (4 independent `smmla` accumulator chains + shared activation
load) is a stable 1.46x over the current `output2` panel at realistic
prefill shape, bit-for-bit within the int8 activation-quant tolerance. This is the
single-thread lever that threading could not provide.

Paper value:

- the R118/R119 packed panel was latency-bound on one accumulator chain; ILP via
  multiple independent `smmla` tiles is a clean ~1.46x with zero added error
- on heterogeneous mobile cores with a streaming/low-RAM design, single-thread
  ILP beats batch-row threading (which the per-chunk spawn model defeats)

## Next Experiment

R124: promote `output8` into the runtime panel
(`smmla_accumulate_output_pair` → an output-octet kernel), handling the runtime's
**per-block per-row weight scales** (the lab uses a single scale), packing 4
weight panels per octet, and an output-row remainder path (octets → pairs → odd
row). Re-measure end-to-end prefill (target ~1.0s) and re-validate parity (top-1,
top-10, output `No`). Apply the R119 inline-asm rules throughout.
