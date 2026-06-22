# Trial: R126 zero-copy raw Q8 chunk access

Date: 2026-06-18
Owner: RLLM
Status: inconclusive
Folder: inconclusive

## Hypothesis

The decode bottleneck is the per-token weight copy: every Q8 chunk goes through
`with_decoded_chunk` → `codec.decode` → `encoded.to_vec()` (the rtc-raw identity
codec allocates + copies the whole chunk), so ~the entire model is re-copied from
the mmap every token. Reading raw chunks zero-copy (`with_raw_chunk`, already used
by the fp16 path) should eliminate that and speed up decode (and prefill).

## Scope

- Mode: exact-lowram runtime
- REE kernel: none (IO/decode path change)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa` (codec rtc-raw-v1)
- Architecture: LLaMA 3.2 1B, Q8_0
- Target device/profile: Apple A18 Pro, single-thread
- Bottleneck tag: allocation / IO-decode

## Setup

Routed the four hot Q8 linear paths (normal linear, multiply-into/up, the R121
panel up prefetch, and lm_head argmax) to `with_raw_chunk` when
`chunk.codec_id == "rtc-raw-v1"`, falling back to `with_decoded_chunk` for
compressed codecs. Identical bytes (raw = identity). Measured decode + prefill +
output parity.

## Results

| run | input tok | gen tok | prefill | decode tok/s | notes |
|---|---:|---:|---:|---:|---|
| R124 baseline | 54 | 16 | ~1.24s | 1.43 | with_decoded_chunk (.to_vec) |
| R126 zero-copy | 54 | 16 | ~1.24s | 1.49 | output coherent + `No`, parity held |

Decode 1.43 → 1.49 tok/s (within noise). Prefill unchanged. Output preserved.

## Analysis

**Hypothesis rejected: the `.to_vec()` copy is NOT the decode bottleneck.** Decode
phase profile is unchanged — MLP (gate/up/down) still ~68%. The per-token copy is
cheap relative to the dominant cost.

The valuable outcome is the corrected root-cause it forced: the batch=1 int8
kernel (`accumulate_q8_0_chunk_int8_activation`) **re-quantizes each input segment
once per output row** (e.g. 8192× redundant for gate). The panel path caches the
activation quant for prefill (batch≥2); the batch1 decode path does not. That
redundant quantization (not memcpy, not the already-SIMD `sdot`) is the decode
cost → R127.

The zero-copy change itself is correct and lossless, and avoids a "decoded chunk"
buffer allocation + budget reservation in bounded-memory mode, so it is plausibly
RAM-positive in the low-ram-fast profile — but that was not measured here
(test used `--rama-integrity unchecked` = unbounded budget).

## Decision

inconclusive

Reason: speed hypothesis rejected (decode/prefill neutral, within noise), but the
change is correct + lossless and may reduce peak RSS in bounded mode (untested).
Kept in the tree pending a bounded-mode RAM measurement; its real contribution
was exposing the batch1 redundant-quantization root-cause.

Paper value:

- use as negative evidence: per-token raw-chunk copy is not the decode bottleneck.
- use as positive evidence (pending): zero-copy avoids a per-chunk decode buffer
  alloc — measure peak RSS under a bounded memory budget.

## Next Experiment

R127: eliminate the batch1 redundant activation quantization (quantize input once
per matmul, reuse across output rows, accumulate per row into one sdot register).
Separately, measure R126 peak RSS under `--memory-budget` (bounded) to confirm or
drop the RAM-positive claim.
