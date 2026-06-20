# R160 — decode-once-at-load: lossless rANS steady decode 6.3× (≈ bf16 speed) (GO; RAM follow-up)

- Date: 2026-06-20
- Model: Gemma 3 1B IT, `--codec rans`, `RLLM_DECODE_RESIDENT=1`; gemma-test + /usr/bin/time -l
- Verdict: **GO (speed)** — caching the decoded weights so they're decoded ONCE (not per
  token) takes lossless rANS steady decode from **3130 → 497 ms/token (6.3×, ≈ bf16's 450)**,
  token-identical. The per-token-decode penalty (R157c) was the wrong architecture; this
  fixes it. Open follow-up: peak RAM 3.5 GB (cache holds compressed + decoded).

## Insight

The slow rANS inference (0.27 tok/s) re-decoded every weight EVERY token. Weights don't
change — decode once, reuse. Opt-in `RLLM_DECODE_RESIDENT=1` adds a `decoded_cache`
(chunk_id → bytes) in `with_decoded_chunk`: first token decodes + caches all chunks,
subsequent tokens skip read/verify/decode entirely → bf16-resident speed.

## Results — measured

```
                            steady decode   RAM        tokens        lossless
rANS per-token (R157c)       3130 ms        2.26 GB    identical     yes
rANS decode-once (R160)       497 ms        3.50 GB    identical     yes
bf16 (zero-copy)              450 ms        2.34 GB    identical     yes
```
Steady decode ≈ bf16 (497 vs 450 ms). Default (cache off) unchanged — opt-in, no
regression. rllm-runtime lib 296, 0 warnings.

## Analysis (honest)

- **Speed complaint cracked:** decode-once → steady ~2 tok/s ≈ bf16, lossless, from a
  1.3 GB file. The per-token decode wall was an architecture mistake, not physics.
- **RAM is the open issue:** 3.5 GB because the cache holds the decoded bf16 (~2 GB) ON
  TOP of the still-mmap'd compressed model (1.3 GB). The compressed pages are reclaimable
  (file-backed) after warmup, but the peak counts them. Fix: drop/`madvise(DONTNEED)` the
  compressed after decode → ~2 GB (≈ bf16).
- **The clean product today (no new code):** ship rANS (1.3 GB), `rllm unpack` → bf16
  locally (one-time), run at full bf16 speed + RAM. R160 is the seamless in-engine version
  (one command, currently +RAM).

## Decision

**GO on speed** — lossless decode-once runs at ≈ bf16 speed from a 35%-smaller file. The
"lossless is too slow" problem is solved for the fits-in-RAM/decompress-resident case.
RAM reduction (drop compressed post-decode) is the immediate follow-up.

## Next

- Drop/`madvise` the compressed pages after decode-all → peak RAM ≈ bf16 → clean win
  (smaller file + ≈ bf16 speed + ≈ bf16 RAM + lossless).
- Eager decode-all at load (vs lazy first-token) to remove the one-time first-token stall.

## Verification status

- [x] Steady decode 3130 → 497 ms (6.3×, ≈ bf16), token-identical (lossless).
- [x] Opt-in (RLLM_DECODE_RESIDENT); default path unchanged; lib 296, 0 warnings.
- [~] Peak RAM 3.5 GB (holds compressed + decoded) — follow-up to drop compressed.
