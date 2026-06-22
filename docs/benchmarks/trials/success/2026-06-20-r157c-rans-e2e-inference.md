# R157c — rANS-packed model runs end-to-end inference, LOSSLESS (GO; speed decode-bound)

- Date: 2026-06-20
- Model: Gemma 3 1B IT packed `--codec rans` (1.325 GB); run via `gemma-test`
- Verdict: **GO (correctness)** — a rANS-packed `.spsa` runs full generation through the
  existing engine and produces **token-identical output to the bf16 resident model**
  (lossless e2e). **Speed is decode-bound** (0.28 tok/s vs bf16's 2.19) because the
  container decodes the whole tensor per access; the streaming forward-pass optimization
  is deferred (R158).

## Finding

No forward-pass rewrite was needed for correctness: the container's `codec_for_id`
(R157a) decodes rANS tensors on access, so the existing Gemma generation path runs a
rANS-packed model as-is.

```
$ gemma-test --model gemma1b-rans.spsa --prompt "The capital of France is" -n 12
  Paris. The largest city in France is Paris.
  token ids: [9079,236761,108,818,7488,3207,528,7001,563,9079,236761,108]
  Tokens: 12 in 43.45s (0.28 tok/s) | peak transient 1.81 GB

$ gemma-test --model gemma-3-1b-it-rawcodec.spsa (bf16)  ...
  token ids: [9079,236761,108,818,7488,3207,528,7001,563,9079,236761,108]  (IDENTICAL)
  Tokens: 12 in 5.47s (2.19 tok/s) | peak transient 16 KB
```

**Identical token ids → rANS whole-model inference is bit-identical (lossless) to bf16.**

## Analysis (honest)

- **Correctness milestone reached:** you can `rllm pack --codec rans` and *run* the model;
  the output is bit-identical to bf16. rANS is a fully-functional lossless inference codec,
  not just storage.
- **Speed is decode-bound, and slower than bf16 here:** 0.28 tok/s vs 2.19. Two reasons:
  (1) bf16-raw uses a **zero-copy mmap** (no decode, 16 KB transient); rANS **decodes the
  whole tensor per access** (1.81 GB transient). (2) No q8/streaming fast path on this
  route. On a model that **fits RAM**, zero-copy bf16 wins; rANS's value is the **smaller
  file (1.3 GB) and the > RAM regime** where bf16 cannot fit at all.
- **The streaming infrastructure (R153–R157b) is NOT yet on this route.** gemma-test on a
  rANS model uses the generic container *resident* decode (whole-tensor), not
  `streaming_rans_gemv_parallel`. Wiring the streaming/parallel decode into the forward
  pass — plus decoding only the rows needed, caching, and the >RAM cold path — is the
  R158 optimization.

## Decision

**GO on correctness** — rANS inference runs end-to-end and lossless on a real model. This
is the "rANS runs in the runtime" milestone. Speed in the resident regime is decode-bound
(slower than zero-copy bf16); making it fast is R158 (streaming forward-pass + the >RAM
capacity-bound regime where it wins).

## Next (R158 — speed)

- Route the forward-pass projection matmuls through `streaming_rans_gemv_parallel`
  (parallel decode, R154 oversubscription) instead of whole-tensor container decode.
- Avoid the 1.8 GB transient (stream/decode in blocks, reuse buffers).
- Measure on a model genuinely > device RAM, where rANS (1.3 GB) fits and bf16 (2.0 GB)
  thrashes/can't — the regime where the capacity-bound win (R150a/R154) applies.

## Verification status

- [x] rANS-packed Gemma 1B runs full generation via the existing engine.
- [x] Output token-identical to bf16 resident (lossless e2e).
- [x] Speed measured honestly: 0.28 tok/s (decode-bound) vs bf16 2.19 (zero-copy); RSS
      1.81 GB vs 16 KB. Streaming/perf = R158.
