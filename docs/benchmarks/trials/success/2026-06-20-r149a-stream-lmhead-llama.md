# R149a — Llama streaming lm-head lossless vs resident (real weights, w=5)

- Date: 2026-06-20
- Kernel lineage: REESTREAM (reused from R148 — `streaming_bitplane_gemv` + sidecar writer/reader)
- Model: Llama-3.2-1B-Instruct-raw.rllm (raw bf16; tied lm-head `model.embed_tokens.weight`, shape [vocab=128256, hidden=2048]; confirmed w=5, vocab%256==0)
- Verdict: **GO** (lossless integration proven on real w=5 weights)

## Scope

Prove correctness of the streaming lm-head path on **real model weights**, not a synthetic
fixture. R148 proved REESTREAM lossless on a synthetic w=5 fixture; R149a-Gemma (failed/, honest
record) showed the real Gemma 3 1B embedding is w=6 and the decode kernel is w=5-only → BLOCKED
pending R149b. Per Erik's decision, correctness is proven NOW on a confirmed-w=5 real model
(Llama 3.2 1B) via a **targeted lossless test**, not full generation.

Path under test: `write_lmhead_sidecar` → `stream_lmhead_from_sidecar` (sidecar bit-plane
streaming GEMV) vs the model's resident bf16 lm-head GEMV (`lm_head_logits_parallel_bf16`),
both fed the identical activation vector and the identical real tensor.

## Hypothesis

The streaming lm-head logits are **bit-identical** to the resident bf16 lm-head GEMV when run
on real Llama weights.

## Method

Single `#[ignore]` integration test `r149a_llama_streaming_lmhead_lossless`
(`crates/rllm-runtime/src/streaming/bitplane_stream.rs`, mod `bitplane_stream_tests`):

1. `write_lmhead_sidecar(model, "model.embed_tokens.weight", 256, /tmp/llama1b-lmhead.sidecar)`.
2. Load the resident bf16 tensor via `with_raw_tensor`.
3. Deterministic activation `act[i] = sin(i*0.011)*0.4`, length = hidden (2048).
4. `resident  = lm_head_logits_parallel_bf16(act, bf16, vocab, hidden)`.
5. `streamed  = stream_lmhead_from_sidecar(sidecar, act)`.
6. `assert_eq!(streamed, resident)` — exact, bit-for-bit, all 128256 logits.

Run:
`cargo test -p rllm-runtime --release r149a_llama_streaming_lmhead_lossless -- --ignored --nocapture`

## Results

```
running 1 test
R149a OK: Llama streaming lm-head == resident, 128256 logits bit-identical
test streaming::bitplane_stream_tests::r149a_llama_streaming_lmhead_lossless ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 302 filtered out; finished in 7.10s
```

All 128256 logits **bit-identical** (exact `assert_eq!`, not approximate). Full lib suite green:
`cargo test -p rllm-runtime --lib` → 293 passed, 0 failed, 10 ignored.

## Analysis

Correctness-first. This proves the **kernel + sidecar + lm-head integration is lossless on REAL
model weights** (w=5 Llama embedding), closing the gap left by R148 (synthetic fixture only).
The write path (palette extraction, bit-plane framing) and the read path (NEON tbl decode →
streaming GEMV) reproduce the resident bf16 GEMV exactly.

Deferred, on purpose:
- **Full-generation wiring**: Gemma's generation lm-head branch is already wired (R149a T3,
  opt-in `RLLM_STREAM_LMHEAD`) but BLOCKED because Gemma 3 1B is w=6 — that is R149b's
  decode-kernel generalization, not in scope here. Llama uses a fused-argmax path for its own
  generation, so the Llama generation gate is not the correctness target here.
- **Speed**: REESTREAM only wins in the capacity-bound regime (model > RAM, cold device, per
  R147/R148). No speed claim is made here; this trial is correctness-only.

## Decision

**GO** — lossless streaming lm-head integration proven bit-identical to resident on real w=5
Llama weights. The kernel/sidecar/lm-head stack is correct; what remains is generalization and
speed, both scheduled.

## Next

- **R149b**: generalize the decode kernel to w=6 (palette > 32) so Gemma's already-wired
  generation path runs, then re-run the identical-token generation gate on Gemma 3 1B.
- Speed demo only in the >RAM / cold regime where REESTREAM is designed to win.
