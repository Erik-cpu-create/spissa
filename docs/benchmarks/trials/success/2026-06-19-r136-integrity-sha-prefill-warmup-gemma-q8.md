# Trial: R136 integrity SHA was the prefill/warmup cost — dedup + parallel prewarm

Date: 2026-06-19
Owner: RLLM
Status: accepted (prefill 7.3s→1.2s; decode-step-1 warmup ~6.2s→~0; integrity preserved)
Folder: success

## Hypothesis

After R134 (residency) and R135 (int8 sdot layers), the `--fast` end-to-end
tok/s was still low and dominated by two NON-decode costs: a ~7.3s prefill and a
one-time ~6.2s stall on the first decode token. Hypothesis: these are SHA-256
integrity verification (RLLM verifies every byte lossless; Ollama does not),
not kernel compute — and they are partly redundant and fully parallelizable.

## Scope

- Mode: fast-lowram runtime (q8, codec rtc-raw-v1)
- REE kernel: none — integrity/IO change (SHA dedup + parallel verification), not an execution kernel
- Model/artifact: `models/gemma-3-4b-it-q8.rllm` (~4.5 GB q8 + 1.34 GB bf16 tied embedding)
- Architecture: Gemma 3 4B, Q8_0
- Target device/profile: Apple Silicon, 8 GB RAM, CPU only
- Expected bottleneck: SHA-256 integrity (verification), not kernels
- Bottleneck tag: IO/decode (integrity verification)

## Setup

```bash
# isolate the SHA cost from compute (diagnostic knob added this trial)
RLLM_INTEGRITY=unchecked RLLM_Q8_KERNEL_PROFILE=1 ./target/release/gemma-test \
  --model models/gemma-3-4b-it-q8.rllm --prompt "The capital of France is" --fast ...
# normal (VerifyOnce) with the two fixes:
./target/release/gemma-test --model models/gemma-3-4b-it-q8.rllm \
  --prompt "The capital of France is" --fast --max-new-tokens 32 --ctx 256
```

Runtime context: release; Apple Silicon ARM64 CPU-only; 8 GB; macOS; `--fast`
(mlock + int8 activation), VerifyOnce integrity. SHA-256 via the `sha2` crate.

## Results

Diagnostic — `--fast` with integrity on vs off (per-step profile):

| phase | VerifyOnce (before fixes) | RLLM_INTEGRITY=unchecked | inference |
|---|---:|---:|---|
| step 0 prefill (6 tok) | 7029–7284 ms | 1139 ms | ~1.1s is compute; ~5.9s was SHA |
| step 1 decode (warmup) | 6251 ms | 95 ms | the entire warmup was redundant SHA |
| steps ≥2 decode (layers) | ~88 ms | ~88 ms | decode layers never paid SHA |

Root cause: in VerifyOnce the q8 weights were SHA-verified per-chunk during
prefill (recorded in `verified_compressed_chunks`) and then RE-hashed whole-tensor
by the decode fast-path (`with_raw_tensor` → `verified_tensors`) — the same
4.5 GB hashed twice. SHA-256 ≈ 0.7 GB/s/core on ARM ⇒ ~6 s per pass.

Fix 1 (dedup, chunk→tensor bridge): `with_raw_tensor` skips the whole-tensor
hash when all of the tensor's rtc-raw-v1 chunks are already verified. Kills the
step-1 warmup (6251 ms → 88 ms). Zero integrity loss (bytes already verified).

Fix 2 (parallel prewarm): `prewarm_chunk_integrity()` SHA-verifies all chunks
across cores at startup (one shared read-only mmap, disjoint shards). Moves the
remaining inline SHA out of prefill.

End-to-end after both fixes (`--fast`, VerifyOnce kept on):

| metric | before (this session start) | after R134–R136 | Ollama CPU |
|---|---:|---:|---|
| integrity prewarm (one-time) | n/a (smeared ~12s inline) | ~2.7 s parallel | none (no verify) |
| prefill (6 tok) | ~7.3 s | ~1.2 s | ~0.5 s |
| decode-step-1 warmup | ~6.2 s | ~0 | none |
| decode steady-state | layers 90ms + lm_head 165ms = ~255 ms/tok (~3.9 tok/s) | same | ~80 ms/tok (12.4 tok/s) |
| RSS | 3.97 GB (f32 embed, thrash) | ~5.06 GB (resident, pinned) | ~3.95 GB |

Output coherent and correct: "Paris is a global center for art, fashion,
gastronomy and culture. It is home to many famous landmarks, including the
Eiffel Tower, the ...". 283 runtime + 17 container tests green.

## Analysis

The dominant "slowness" of RLLM vs Ollama for prefill+startup was SHA-256
integrity verification — the lossless guarantee Ollama does not provide — not
kernel quality. Isolating it (RLLM_INTEGRITY=unchecked) showed prefill compute
is only ~1.1s and the warmup is ~0. Fix 1 removes a redundant second hash of the
same bytes; Fix 2 parallelizes the single remaining pass and moves it to a brief
startup step. Integrity is fully preserved: every byte is still SHA-verified
exactly once.

Honest remaining gaps vs Ollama: (a) decode ~3.9 tok/s vs 12.4 (~3.2x) — now
dominated by lm_head (bf16 tied embedding, 1.34 GB/token @ ~8 GB/s) rather than
the q8 layers (~90 ms, efficient); (b) prefill 1.2s vs 0.5s (~2.4x); (c) a ~2.7s
one-time integrity prewarm that Ollama simply does not do.

## Decision

accepted

Reason: prefill 7.3s→1.2s, warmup ~6.2s→0, integrity preserved, output
unchanged, tests green. Adds `RLLM_INTEGRITY` diagnostic knob.

Paper value:

- use as positive evidence (a lossless runtime can keep bit-exact integrity yet
  push verification off the hot path via dedup + parallel prewarm) with
  limitation (the integrity pass is a one-time cost Ollama avoids entirely).

## Next Experiment

lm_head: ~165 ms/token is now the largest single decode cost — it reads the
1.34 GB bf16 tied embedding per token at ~8 GB/s (vs the q8 layers' ~39 GB/s).
Optimize the bf16 lm_head GEMV kernel (vectorized bf16→f32 + better threading)
toward the layer kernel's bandwidth, staying lossless (no quantizing the output
embedding). Secondary: prefill compute 1.2s→ closer to Ollama 0.5s.
