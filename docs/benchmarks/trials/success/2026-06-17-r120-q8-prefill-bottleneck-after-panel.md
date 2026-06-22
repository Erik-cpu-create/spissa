# R120: Q8 Prefill Bottleneck After Panel (threading neutral)

Date: 2026-06-17
Owner: RLLM
Status: accepted diagnostic
Folder: success

## Hypothesis

R119 made the gate/attention/down Q8 matmuls ~6x faster via the i8mm packed
panel, taking single-thread prefill from ~6.8s to ~3.3s. R120 tries to stack
threading on top — thread the panel's output-pair loop across cores — and, when
that fails to help, attributes the new bottleneck.

## Scope

- Mode: exact-lowram diagnostic + runtime trial
- REE kernel lineage: REEFUSE-Q8-I8MM-PANEL (threading attempt, reverted)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Device: Apple A18 Pro
- Bottleneck tag: scheduler / kernel coverage

## What was tried

Threaded the panel over output pairs: each worker owns a disjoint range of output
columns (raw `*mut f32` via a `Send` wrapper, no `split_at_mut` since columns are
strided), shares the packed activation read-only, and the R115 batch-split is
bypassed when the panel is active so the full batch stays packed once. Correct
(12 panel unit tests + parity bit-identical) but **neutral**.

## Results

Prefill best-of-3 (output `No`, parity held throughout):

| config | prefill |
|---|---:|
| `RLLM_THREADS=1` (f32) | 6.5s |
| `RLLM_THREADS=1` + panel | 3.30s |
| `RLLM_THREADS=6` + panel (threaded) | 3.31s |

Threading the panel does **nothing** (3.31s vs 3.30s). Phase profile at panel
single-thread shows why — and what the real bottleneck is:

| MLP projection | f32 (panel off) | panel on | sped up? |
|---|---:|---:|---|
| gate (2048→8192) | 2051ms | **338ms** | yes, ~6x |
| down (8192→2048) | 1539ms | **336ms** | yes, ~4.6x |
| **up (2048→8192)** | 1306ms | **1357ms** | **NO** |

## Analysis

R120 (threading the panel) is neutral because the matmuls the panel covers
(gate/down) are already so fast (~336ms) that threading them is lost in noise and
per-chunk spawn overhead — and the actual bottleneck, **up_proj (1357ms), does not
go through the panel at all**.

up_proj is computed by `streaming_tile_linear_multiply_into_from_model` →
`accumulate_q8_0_chunk_multiply_into`, the fused `silu(gate) * up_proj(x)` kernel
(it multiplies each up output feature into the gate buffer via a row-state
machine). R119 only converted `accumulate_q8_0_chunk` (gate, attention q/k/v/o,
down), not the multiply-into kernel. So up is still on the slow f32 path while
gate/down are paneled.

The threaded-panel code was reverted (neutral complexity, plus per-chunk
`thread::scope` spawn overhead that does not pay off while up dominates).

## Decision

accepted diagnostic (threading change reverted; R119 panel kept)

Reason: measured that threading the panel is neutral (3.31s vs 3.30s) and
attributed the remaining prefill cost to the unpaneled up_proj
(`accumulate_q8_0_chunk_multiply_into`, 1357ms vs gate/down 336ms).

Paper value:

- high-value attribution: after R119, the single dominant prefill cost is the
  fused up_proj multiply-into kernel, not threading
- threading the panel does not help while one matmul is unpaneled and the rest are
  already ~6x faster

## Next Experiment

R121: extend the i8mm packed panel to up_proj — compute up via the panel into a
scratch buffer per matmul, then `target[b][f] *= up[b][f] + bias[f]` (the
multiply-into), done in `streaming_tile_linear_multiply_into_from_model` so the
row-state machine is bypassed for eligible chunks. Projected: up 1357ms → ~340ms,
MLP ~2040ms → ~1010ms, prefill ~3.3s → ~2.3s, all exact (same activation-quant
tolerance as R119). Add a multiply-into panel unit test against the existing
state-machine path. Then revisit threading once all three MLP projections are
paneled.
