# R157a — rANS as a real .rllm container codec: `pack --codec rans` lossless (GO)

- Date: 2026-06-20
- Codec: rtc-rans-v1 (`RansCodec: TensorCodec`)
- Models: pythia-160m (375 MB), Gemma 3 1B IT bf16 (2.0 GB) — both packed + verified
- Verdict: **GO** — rANS is now a first-class `.rllm` codec. `rllm pack --codec rans`
  produces a **lossless** container the runtime decodes natively; Gemma 1B packs to
  **2.0 GB → 1.325 GB (−33.7%)** and `rllm verify` reports **LOSSLESS VERIFIED**. The
  rANS invention moves from research/sidecar to a real, usable feature.

## What changed

- **`RansCodec` implements `TensorCodec`** (`rtc-codec/src/rans.rs`): encode splits each
  bf16 pair into (exponent, residual), 4-lane interleaved-rANS the exponent, stores the
  residual raw; decode reverses it. The (exp,residual) split is a *bijective bit
  rearrangement* of any even-length bytes, so it is **lossless regardless of dtype** and
  only *compresses* when the high byte is low-entropy (real bf16). Odd-length chunks →
  raw fallback (FLAG_RAW). Roundtrip unit tests (bf16 + raw fallback) green.
- **Registered `"rtc-rans-v1"`** in all three codec dispatch sites (they were duplicated):
  `loader.rs::codec_for_id` (runtime decode), `verify.rs`, `unpack.rs`.
- **`rllm pack --codec rans`**: new `PackCodecPolicy::Rans` (+ `CODEC_RANS_V1` const). Tries
  `[RansCodec, RawCodec]` and keeps the smallest lossless per chunk (raw where rANS doesn't
  help, e.g. non-bf16).

## Results

```
$ rllm pack /tmp/gemma1b --out gemma1b-rans.rllm --codec rans ...
  Encoded 2189 chunks   Codec policy: rtc-rans-v1
  Original 1,999,771,904 B -> Compressed 1,324,908,500 B   ratio 66.3%  (1.3 GB file)

$ rllm verify gemma1b/model.safetensors gemma1b-rans.rllm
  [OK] Verified 340 tensors, 1,999,771,904 bytes total
  [OK] LOSSLESS VERIFIED

$ rllm pack models/pythia-160m --codec rans  -> 375 MB -> 296 MB (78.9%)
$ rllm verify ...pythia... -> [OK] LOSSLESS VERIFIED (184 tensors)
```

The Gemma 1B packed size (1.325 GB) matches the R156b in-memory measurement (1.324 GB)
exactly. Suites: rtc-codec 54 / rllm-runtime lib 296, 0 warnings.

## Analysis

- **The milestone "packed to .rllm" is real:** before R157a, rANS lived only as a library
  + experimental sidecars; the container's `codec_for_id` knew only raw/rle/huff (not even
  bitplane/dfloat). Now `rllm pack --codec rans` yields a genuine lossless `.rllm` and the
  runtime/verify/unpack all decode it.
- **Lossless verified end-to-end** on two real models (byte-exact vs original safetensors),
  not just synthetic roundtrips.
- **Three duplicated codec dispatchers** (loader/verify/unpack) is a latent smell — a future
  cleanup could unify them, but for R157a all three were updated.

## Decision

**GO.** rANS is a first-class lossless `.rllm` codec. A user can now `rllm pack --codec rans`
and ship a 1.3 GB lossless Gemma 1B (vs 2.0 GB raw). This is the resident/container path;
the >RAM streaming forward-pass integration is R157b/c.

## Next (R157b/c)

- R157b: wire one layer's forward pass to stream+decode projections via rANS (lossless vs
  resident). Needs flexible `block_rows` for tensors not %256 (e.g. down_proj).
- R157c: full forward pass + prefill (GEMM) + e2e tok/s on a model > device RAM.

## Verification status

- [x] `RansCodec` roundtrip (bf16 compresses + raw fallback), unit.
- [x] `rllm pack --codec rans` → lossless `.rllm`; `rllm verify` LOSSLESS VERIFIED on
      pythia-160m (184 tensors) and Gemma 3 1B bf16 (340 tensors, 2.0 GB → 1.325 GB).
- [x] Registered in loader/verify/unpack; rtc-codec 54 / rllm-runtime 296, 0 warnings.
