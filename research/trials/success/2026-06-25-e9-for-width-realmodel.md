# Research Trial: E9 — REEBORN-FOR exponent width on real Llama-1B

Date: 2026-06-25
Owner: REEBORN
Status: accepted
Folder: success

## Hypothesis

The FOR decode speed is distribution-independent (proved in the edge bench: 6.18× rANS), so the
open question is the real RATIO: what fixed exponent `width` do actual per-tensor ranges need, and
what b/w does REEBORN-FOR (raw 8-bit significand + per-tensor fixed-width exponent) reach on a real
model?

## Scope

- Experiment type: codec-validation (edge codec ratio)
- REE codec: REEBORN-FOR (`rtc-reeborn-for`)
- Lossless reference: vs bf16 (bit-exact)
- Model/artifact: Llama-3.2-1B-Instruct, 1236M weights, 113 2-D bf16 tensors
- Finding tag: codec validation

## Method

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp9_for_width_realmodel.py
```

Per tensor: width = ceil(log2(exp_max − exp_min + 1)) (pure, no escape). Also a global
mode-centered width+escape variant (rare out-of-window exps cost width+8).

## Results

| | b/w | vs bf16 |
|---|---:|---:|
| per-tensor exp width | 72.6% → 5-bit, 27.4% → 6-bit (avg 5.274) | — |
| **REEBORN-FOR pure** (significand 8 + per-tensor width + hdr) | **13.274** | 1.21× |
| **REEBORN-FOR width-3 + escape** (9.0% escape) | **11.720** | 1.37× |
| DFloat / rANS (reference) | ~10.6 | 1.51× |

## Analysis

REEBORN-FOR has a clean ratio↔decode-simplicity knob:
- **Pure per-tensor fixed-width (13.27 b/w):** fully branch-free, fastest + most NEON-friendly decode.
- **Width-3 + escape (11.72 b/w):** only ~1.1 b/w above the rANS floor, still coderless (a small 9%
  escape branch), much better ratio.

**CORRECTION (measured, not assumed):** the escape variant's per-symbol branch is expensive — its
decode benched at **0.856 Gw/s** (vs pure fixed-width 1.80 Gw/s), see
`docs/benchmarks/trials/success/2026-06-25-reeborn-for-edge-decode.md`. In the >RAM cold regime
(read ~1.74 GB/s) that flips the picture:

| codec | b/w | decode Gw/s | >RAM net | vs raw |
|---|---:|---:|---|---:|
| raw bf16 | 16 | — | 0.870 (read-bound) | 1.00× |
| **FOR pure (branch-free)** | 13.27 | 1.80 | 1.05 (read-bound) | **1.21×** |
| FOR width+escape | 11.72 | 0.856 | 0.856 (DECODE-bound) | 0.98× (break-even) |

So the **pure branch-free fixed-width FOR is the edge winner** (read-bound, 1.21× over raw); the
escape variant's better ratio is wasted because its branch drops decode below the read rate, making
it decode-bound and ~break-even with raw. **For edge, branch-free decode matters more than ratio** —
the opposite of my initial assumption (an earlier draft wrongly called escape the best operating point).

## Decision

accepted

Reason: REEBORN-FOR is the validated edge codec — raw significand + **pure per-tensor branch-free
fixed-width exponent** (13.27 b/w on Llama-1B, decode 1.80 Gw/s, read-bound 1.21× over raw in >RAM).
The width+escape variant (11.72 b/w) is REJECTED for edge: its branch drops decode to 0.856 Gw/s →
decode-bound → break-even with raw. Branch-free wins on the edge axis.

Paper value: positive — real-model confirmation of the FOR edge codec ratio, with a documented
ratio/decode knob.

## Caveats / next

- The width+escape decode SPEED is not yet benched (9% escape branch may shave throughput — expected
  still ≫ rANS). Bench it next.
- Then: NEON-vectorize the pure fixed-width unpack; wire `rtc-reeborn-for` into the container +
  streaming GEMV; e2e >RAM benchmark vs raw and rANS on Mac, then the ARM phone.
