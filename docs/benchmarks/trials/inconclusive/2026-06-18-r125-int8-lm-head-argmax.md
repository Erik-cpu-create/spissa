# Trial: R125 int8 lm_head argmax (gated)

Date: 2026-06-18
Owner: RLLM
Status: inconclusive
Folder: inconclusive

## Hypothesis

lm_head profiled as a large slice of prefill. Routing the batch1 lm_head argmax
through an int8 path (quantize the input vector once, `i8_dot32`/sdot directly
against the q8 weight, no per-weight f32 dequant) should cut it, like gate/up/down.

## Scope

- Mode: exact-lowram runtime gate
- REE kernel: REEBORN-Q8-SDOT (argmax variant)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Architecture: LLaMA 3.2 1B, Q8_0
- Target device/profile: Apple A18 Pro, single-thread
- Bottleneck tag: CPU arithmetic / memory bandwidth

## Setup

Added a gated int8 branch in `accumulate_q8_0_chunk_argmax_batch1_complete_rows`
(quantize input once via `quantize_seg32_i8`, sdot per weight row), behind
`RLLM_Q8_ACTIVATION`. Measured prefill best-of-10 + first-token parity.

## Results

| run | input tok | gen tok | prefill | decode tok/s | notes |
|---|---:|---:|---:|---:|---|
| R124 baseline | 54 | 1 | 1.24s | — | f32 lm_head argmax |
| R125 int8 lm_head | 54 | 1 | 1.24s | untested | output `No`, top-1 match |

Prefill unchanged (1.24s). Output token `No` preserved; first-token top-1 match.

## Analysis

Neutral **on prefill** because prefill runs lm_head exactly once (last position
only), so it is a tiny slice of a 54-token prefill — there is nothing to win
there. The change was reverted to keep a clean R124 baseline.

BUT the decode effect was never measured: in **decode**, lm_head runs every token
(~20% of decode time in profiling), and R125 already uses the efficient pattern
(quantize input once + sdot) that R127 will generalize to gate/up/down. So R125
is essentially "R127 applied to lm_head". Its decode benefit and per-token parity
(the output token is decided here, so quant error is most sensitive) are open.

## Decision

inconclusive

Reason: prefill-neutral (lm_head is one call in prefill), decode effect and
decode-wide parity untested. Reverted from the tree to keep the R124 baseline,
but the pattern is correct and should be re-evaluated as part of R127.

Paper value:

- use as limitation: prefill lm_head is not a lever (single call); decode lm_head
  may be, pending measurement + token-parity validation.

## Next Experiment

Fold lm_head into R127: once the batch1 activation-quant cache + tight sdot
accumulation are in place for gate/up/down, re-introduce int8 lm_head and measure
decode tok/s + multi-token output parity (top-1 over a generation, not just the
first token).
