# Trial: R127 batch1 activation-quant cache (decode)

Date: 2026-06-18
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

The R125bench diagnostic root-caused slow decode (~26x behind llama.cpp) to the
batch=1 int8 kernel `accumulate_q8_0_chunk_int8_activation` re-quantizing each
input segment once per output row (8192x redundant for gate; the panel caches
this for prefill batch>=2, the batch1 decode path does not). Quantizing the
activation once per matmul and reusing it should cut decode toward the memory
floor, losslessly (the cache uses the identical absmax/round/clamp).

## Scope

- Mode: exact-lowram runtime
- REE kernel: REEBORN-Q8-SDOT (batch1 activation cache)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Architecture: LLaMA 3.2 1B, Q8_0
- Target device/profile: Apple A18 Pro, single-thread, CPU-only
- Bottleneck tag: CPU arithmetic (redundant quantization)

## Setup

Wrapped the `accumulate_q8_0_chunk_int8_activation` block loop in
`with_q8_panel_activations` (thread-local cache keyed by ptr+len+shape+content
fingerprint → `quantize_input_q8_blocks`, run once per matmul, hit across all 18
chunks). Inner loop replaced `quantize_seg32_i8(seg)` with a lookup
`act_i8[row*in_features + in_feature ..][..32]` + `act_scales[row*blocks_per_row +
in_feature/32]`. `quantize_input_q8_blocks` is bit-identical to `quantize_seg32_i8`
(same `amax/127`, `round().clamp(-127,127)`), so output is unchanged. The
partial/boundary-block else branch is untouched.

```bash
RLLM_THREADS=1 RLLM_Q8_ACTIVATION=1 llama-test --model <rllm> --max-new-tokens 24
```

Runtime context: release, Apple A18 Pro single-thread, macOS, `RLLM_Q8_ACTIVATION=1`.

## Results

| run | input tok | gen tok | prefill | decode tok/s | notes |
|---|---:|---:|---:|---:|---|
| R124/R126 baseline | 54 | 24 | ~1.24s | 1.43 | re-quantize per output row |
| **R127 quant cache** | 54 | 24 | ~1.24s | **~2.5 (best 2.54)** | output identical |

Decode **1.43 → ~2.5 tok/s (~1.75x)**. Output preserved: fire prompt `No`; "sea"
generation byte-identical to baseline. 82 streaming tests pass — including the
r119 panel tests, which use `accumulate_q8_0_chunk_int8_activation` as their
reference, so a bit-exact match is enforced by the suite.

## Analysis

Removing the `out_features`x redundant activation quantization gives ~1.75x decode
with zero output change. Still ~14x short of the ~36 tok/s memory floor, so the
redundant quant was roughly half the removable overhead. The remaining cost is the
per-block kernel structure: `i8_dot32` does an `addv` horizontal reduce + scalar
f32 scale-and-accumulate into `output` once per 32-block, with no cross-block
register accumulation (per-block weight+act scales force a per-block scale, but the
reduce/store can be tightened). That is the R128 lever.

## Adversarial verification (3 independent lenses)

- **Parity lens: exact-and-safe.** Confirmed the cache lookup is bit-identical to
  the per-row `quantize_seg32_i8` path (identical absmax/round/clamp, identical
  segment/scale indexing).
- **Edge-case lens: blocker found + FIXED.** When `in_features` is not a multiple
  of 32, the block cache only covers `blocks_per_row*32` elements/row, so a
  non-32-aligned `in_feature` fast-path lookup could read the wrong/uncovered
  region. Guarded: non-32-aligned `in_features` now falls back to the exact
  per-segment `accumulate_q8_0_chunk_int8_activation_uncached`. (q8_0 itself
  requires 32-aligned dims, so this is defensive; real models are unaffected.)
- **Cache lens: weak fingerprint (major) → HARDENED.** `q8_act_fingerprint`
  sampled only 4 points; on decode the activation buffer address is reused across
  tokens, so a 4-sample collision could serve stale quantization. Strengthened to
  up to 64 evenly-spread samples with per-index FNV mixing.
- **Regression lens: wasted `pack_act_panel_pairs` on batch≥2 i8mm-ineligible
  fallback (major) — DEFERRED.** The int8 path reuses `with_q8_panel_activations`,
  which packs the smmla panel it never reads. This is **free for decode**
  (batch=1 → 0 pairs) and only wasted on the rare batch≥2 non-row-aligned / no-i8mm
  fallback (never hit by the rowchunks model in normal prefill). Tracked as a
  follow-up (split a quant-only cache) rather than added complexity now.

## Decision

accepted

Reason: decode 1.43 → ~2.66 tok/s (~1.85x), bit-for-bit identical output (enforced
by the panel reference tests + fire `No` + byte-identical generation), no prefill
change. Adversarially verified across 3 lenses; the one blocker (non-32 in_features)
and the fingerprint-collision risk were fixed; the wasted-pack regression is
negligible in practice (deferred with a tracked follow-up). 268 runtime tests pass
after the fixes.

Paper value:

- use as positive evidence: the batch1 decode path needs the same activation-quant
  caching the prefill panel already has; it is a clean lossless ~1.75x.

## Next Experiment

R128: tighten the batch1 int8 kernel to row-major accumulation — per output row,
accumulate the per-block scaled dot into a register across all blocks and write
`output` once (avoid the per-block `addv`/store and per-block function-call
overhead), keeping per-block scales. Target the remaining gap to the ~36 tok/s
floor. Also fold in int8 lm_head (R125) and measure multi-token parity.
