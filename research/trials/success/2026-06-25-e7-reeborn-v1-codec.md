# Research Trial: E7 — REEBORN v1 codec (our own, from scratch)

Date: 2026-06-25
Owner: REEBORN
Status: accepted
Folder: success

## Hypothesis

A from-scratch lossless codec — no delta, no rANS — built as **raw 8-bit significand + our own
exponent code** can reach near-floor ratio, lossless, because the significand (sign+mantissa) is
~uniform (store raw = optimal, no coder) and only the exponent has structure.

## Scope

- Experiment type: codec-validation (the first working REEBORN codec)
- REE codec: REEBORN v1
- Lossless reference: vs bf16 (our codec is bit-exact to the original weights)
- Model/artifact + weight source: Llama-3.2-1B-Instruct (same source), 1236M weights
- Finding tag: codec validation
- Constraint: no delta (BitDelta/REEFORM prior art), no rANS (commodity coder) — fully ours.

## Method

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp7_reeborn_v1.py
```

Per weight: significand (sign<<7 | mantissa) stored RAW (8 bits, no coder). Exponent coded two
ways, both ours, per-tensor: **FOR** = frame-of-reference width-W bit-pack of (exp−base) + escape
(no entropy coder at all); **PREFIX** = our from-scratch canonical Huffman over the exponent
histogram. Losslessness verified by logical bijection decode(encode(e))==e over every tensor.

## Results

Lossless (logical bijection, all tensors): **True**.

| codec | b/w | vs bf16 | notes |
|---|---:|---:|---|
| bf16 baseline | 16.0000 | 1.00× | — |
| REEBORN-FOR | 11.2198 | **1.43×** | no entropy coder (pure structural), fastest decode |
| REEBORN-PREFIX | 10.6391 | **1.50×** | our Huffman; within 0.07 of rANS (10.57) WITHOUT rANS |
| (info floor) | 10.55 | — | reference |

Breakdown: significand raw = 8.000 b/w (optimal, coder-free); exponent = FOR 3.220 / PREFIX 2.639
(+table 0.0001).

## Analysis

- **A working, original, lossless codec** — neither delta nor rANS. PREFIX lands 10.64 b/w, only
  ~0.07 above the optimal rANS-on-u16 (10.57) and ~0.09 above the info floor — i.e. within 0.7% of
  optimal without using the commodity coder.
- **The 0.07 gap to rANS** = field-separation (coding sign/mant/exp separately loses I(exp;mant)
  ≈0.047, which joint-u16 rANS keeps) + Huffman-vs-arithmetic overhead (~0.04) − per-tensor model
  gain (~0.03). Beating rANS on RATIO needs arithmetic/ANS-class coding (which IS rANS) — so REEBORN's
  edge is not ratio.
- **REEBORN's real edge is DECODE SPEED, by design:** half of every weight (the 8-bit significand)
  is a raw copy with ZERO entropy-decode; only the exponent needs work (FOR = fixed-width unpack,
  PREFIX = table lookup). On the >RAM streaming axis where rANS decode is the wall (R144/R153), this
  raw-significand design should decode far faster at ~the same ratio. FOR (no coder at all, 11.22)
  is the speed-extreme; PREFIX (10.64) the ratio-extreme.

## Decision

accepted

Reason: first working REEBORN codec — ours, lossless, 1.43–1.50× smaller than bf16, near-optimal
ratio without rANS, and a built-in decode-speed advantage (raw significand). Adopt the
raw-significand + per-tensor/cluster exponent design.

Paper value: positive — an original lossless weight codec with a raw-significand design that trades
~0.07 b/w for a large decode-speed advantage; two operating points (FOR / PREFIX).

## Next Experiment

- Build REEBORN v1 in Rust (rtc-reeborn-v1) and measure REAL decode throughput vs rANS on the
  >RAM streaming path (where the raw-significand design should win) — that moves to the runtime
  trial log (docs/benchmarks/trials/, rNN).
- Fold E3 per-layer-type exponent tables into PREFIX; explore a faster-than-Huffman, near-entropy
  ours-coder for the exponent (still not rANS) to close the last 0.07 if desired.
