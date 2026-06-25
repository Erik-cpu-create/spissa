# Research Trial: E8 — W = lossy baseline + lossless residual (speculative lane)

Date: 2026-06-25
Owner: REEBORN
Status: rejected
Folder: failed

## Hypothesis

The web-recovered survey (agent B) flagged "W = U·V + lossless residual, or a lossless residual
against an already-lossy q4/q3 baseline" as apparently unclaimed. Gamble: decomposing W into a
lossy baseline + a losslessly-coded residual beats coding W directly. (Low-rank U·V is already
disproven by REEFORM, net 11.73; this tests the q4-baseline twist.)

## Scope

- Experiment type: codec-design / decomposition
- REE codec: REEBORN
- Lossless reference: vs bf16 (bit-exact)
- Model/artifact + weight source: Llama-3.2-1B-Instruct (same source)
- Finding tag: information-theoretic limit (null) — chain rule

## Method

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp8_residual.py
```

q4 per-block-32 absmax baseline. Measured H(u16), H(q4code), the joint (q4code, u16) → residual
H(u16|q4code) and the scale-redundancy H(q4code|u16). Decisive test: does q4 + scale + residual
beat the direct floor H(u16)?

## Results

| quantity | b/w |
|---|---:|
| direct floor H(u16) | **10.5483** |
| H(q4 code) | 3.5955 |
| H(u16 \| q4 code) — the residual | 8.3608 |
| H(q4 code \| u16) — scale redundancy | 1.4080 |
| fp16 scale stream | 0.5000 |
| **TOTAL q4 + scale + residual** | **12.4563** |

→ **LOSES by +1.9080 b/w (~18% worse)** than coding W directly.

## Analysis

The chain rule settles it: H(q4code) + H(u16|q4code) = H(u16) + H(q4code|u16) ≥ H(u16). The q4 code
is not a pure function of a single weight — it carries **1.41 b/w of block-scale information**
(H(q4code|u16) = 1.41) that becomes pure overhead once the residual is also stored, **plus** the
0.5 b/w scale stream. So baseline+residual re-partitions the same 10.55 bits and adds ~1.9 b/w of
overhead. A non-free baseline can NEVER beat direct coding for standalone lossless — confirmed by
measurement, not just theory.

The one real (but known) niche: if a q4 model is **already stored** for inference, upgrading it to
exact bf16 costs only the residual H(u16|q4) = 8.36 b/w vs 10.55 for a separate bf16 copy → saves
2.19 b/w in the store-BOTH case. This is **scalable / progressive (embedded) coding** (JPEG2000,
SVC) — a known concept, and 8.36 is still dominated by the incompressible mantissa.

## Decision

rejected

Reason: baseline+residual loses by 1.9 b/w for standalone lossless (chain rule). The only value is
a known scalable-coding niche (lossless upgrade sidecar for an existing q4 model).

Paper value: negative evidence + a clean chain-rule demonstration that no non-free decomposition
beats direct coding for standalone lossless weights.

## Next Experiment

Standalone-lossless RATIO is now exhaustively closed (E2–E8 + the prior-art survey + REEFORM +
the repo's own `rtc-dfloat-v1`). The only genuinely-open original path is the EDGE niche:
fastest-decoding lossless weight codec on ARM/CPU in the model>RAM streaming regime (verified
unoccupied by external work) → build in Rust + benchmark decode GB/s (runtime trial rNN).
