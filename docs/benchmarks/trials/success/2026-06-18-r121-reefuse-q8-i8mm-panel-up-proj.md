# R121: REEFUSE-Q8-I8MM-PANEL up_proj (multiply-into)

Date: 2026-06-18
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

R119 made gate / attention / down ~6x faster via the i8mm packed panel but left
the LLaMA `up_proj` on the slow f32 dequant path: it is computed by the fused
`silu(gate) * up_proj(x)` multiply-into kernel
(`accumulate_q8_0_chunk_multiply_into`), which R119 never converted. R120
attributed the entire post-R119 prefill cost to it (up 1357ms vs gate/down
336ms). R121 routes `up_proj` through the same packed panel: compute the full Q8
linear into a scratch buffer with the R119 kernel, then apply
`target[b][f] *= up[b][f] + bias[f]` at the caller, bypassing the fused row-state
machine for panel-eligible chunks.

## Scope

- Mode: exact-lowram runtime gate
- REE kernel lineage: REEFUSE-Q8-I8MM-PANEL (multiply-into extension)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Device: Apple A18 Pro (single-thread, `RLLM_THREADS=1`)
- Bottleneck tag: CPU arithmetic / Q8 i8mm GEMM (kernel coverage)

## What was done

`streaming_tile_linear_multiply_into_from_model` gains a fast path
(`try_panel_multiply_into_up`): when the weight is Q8_0, `RLLM_Q8_ACTIVATION` is
on, and `batch >= 2`, it accumulates every chunk through
`accumulate_q8_0_chunk_panel_smmla` into a `[batch * out_features]` scratch
(threaded `element_start`, budget-accounted), then the caller applies
`target *= up + bias`. Any non-panel-eligible chunk makes the helper return
`None`, discards the scratch, restores the budget, and falls through to the
unchanged fused state machine — so the fused path is never partially applied.

The multiply-into state machine itself is untouched; the panel is a parallel
bypass that reuses the already-validated R119 kernel.

## Results

Best-of-3 prefill, single-thread, same session (`Answer yes or no: is fire
cold?`, output stayed `No` throughout):

| config | prefill (best of 3) | vs f32 |
|---|---:|---:|
| `RLLM_THREADS=1` (f32 control) | 6.05s | 1.0x |
| R119 (gate/down int8, up=f32) | 2.55s | 2.4x |
| **R121 (gate/down/up int8)** | **1.48s** | **4.1x** |

- **R121 vs R119: 2.55s → 1.48s (~1.7x).** Larger than the up GEMM alone because
  `up_proj` shares the MLP input with `gate_proj`, so the panel path reuses the
  thread-local packed-activation cache instead of re-dequantizing — the f32 path
  could not.
- **Parity holds** (first-token full-vocab logits, `RLLM_FULL_LOGITS=1`): top-1
  match (`2822` = `No`), top-10 overlap 10/10, max abs diff **0.3720**. The small
  rise from R119's 0.2997 is exactly the added `up_proj` int8 activation quant;
  output token unchanged.
- 4 new R121 unit tests (`r121_multiply_into_tests`: even, odd-out+chunks,
  realistic up 53x2048x16, single-chunk-full) + 79 streaming / 265 runtime tests
  pass.

## Analysis

The post-R119 bottleneck identified in R120 is removed: all three MLP projections
(gate, up, down) now run on the i8mm packed panel. The extra speedup beyond the
matmul is the activation-cache reuse — gate and up are two projections of the
same `x`, so packing once and reusing it for up is free. Output is preserved and
the quant error stays at the validated int8-activation level.

## Decision

accepted

Reason: `RLLM_Q8_ACTIVATION=1` cut single-thread prefill from R119's 2.55s to
1.48s (~1.7x; 4.1x over the 6.05s f32 control) by paneling `up_proj` via the
multiply-into bypass, output stayed `No`, first-token logits matched the f32
control (top-1, top-10 10/10, diff 0.3720 = activation-quant only), and the fused
state machine remains the exact fallback for non-panel chunks.

Paper value:

- completing panel coverage of all three MLP projections (not just gate/down)
  gives a super-linear prefill win because the fused projections share packed
  activations
- the multiply-into projection can be paneled as a scratch-compute + element-wise
  apply without touching the fragile fused row-state machine

## Next Experiment

R122: revisit threading now that all three MLP projections are paneled (R120
found threading neutral only because up dominated single-thread). Also profile
the new prefill breakdown — with the MLP near ~1.0s, the next levers are likely
attention (q/k/v/o already paneled), lm_head, and the per-matmul activation-pack
overhead that the cache only partly amortizes across layers.
