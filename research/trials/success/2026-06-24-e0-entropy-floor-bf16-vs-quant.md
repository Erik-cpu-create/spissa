# Research Trial: E0 — entropy floor, bf16 vs quantized

Date: 2026-06-24
Owner: REEBORN
Status: accepted
Folder: success

## Hypothesis

"<4 bits/weight, 100% lossless" is only achievable for some definitions of "lossless".
Shannon's source-coding theorem is a hard floor; the real research question is *which
representation* we measure entropy on. Measuring the true entropy of Llama-1B weights in
different representations should show: lossless-vs-fp is bounded by the bf16 symbol entropy
(expected ~10 b/w, mantissa near-random → <4 impossible); lossless-vs-quantized is bounded
by the q-code entropy (expected ~3–3.6 b/w → <4 reachable).

## Scope

- Experiment type: entropy-floor
- REE codec: REEBORN (design input)
- Lossless reference: both — vs fp/bf16 AND vs q4/q3 checkpoint
- Model/artifact + weight source: Llama-3.2-1B-Instruct, `models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors`
- Dtype / param count: BF16, 1.236 B (norms excluded; lm_head tied to embed)
- Finding tag: information-theoretic limit + null result

## Method

Script:

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp0_entropy.py
```

Per-32 symmetric absmax quant (q4 = ±7 levels, q3 = ±3). Measures: order-0 entropy of the
16-bit bf16 symbols (+ sign / exponent / mantissa decomposition); q4/q3 code entropy;
conditional entropy H(code | left) and H(code | up) over 2-D adjacency.

Runtime context: numpy 2.5.0 (venv `/tmp/reeborn-venv`); macOS arm64; ~38 s.

## Results

| metric (ALL, bits/weight) | value |
|---|---:|
| bf16 order-0 entropy | 10.55 |
| — sign / exp / mantissa | 1.00 / 2.63 / 6.96 |
| q4 code entropy | 3.60 |
| q3 code entropy | 2.43 |
| H(q4 code \| left) / H(q4 code \| up) | 3.59 / 3.59 |

(linear vs embed near-identical, ±0.03 b/w; see `research/reeborn/RESULTS.md`.)

## Analysis

1. **Lossless-vs-fp <4 b/w is impossible.** bf16 floor = 10.55 (independently matches the
   project's REEFORM ~10.6). Mantissa entropy 6.96/7 → 99.4% random/incompressible; only the
   exponent (2.63/8) is compressible. Reaching <4 needs deleting ~6.5 bits of near-uniform
   mantissa entropy = a 2.6× Shannon violation. Not a tuning gap — a law.
2. **Lossless-vs-quantized <4 b/w is real.** q4 codes 3.60 (< nominal 4.0 → a flat-4-bit GGUF
   leaves 0.40 b/w on the table); q3 2.43. REEBORN commits to this baseline.
3. **Spatial conditional modelling is a NULL result.** H(code|left) = H(code|up) = 3.59 vs
   marginal 3.60 → 0.01 b/w. Adjacent weights are statistically independent (random-matrix view
   of trained weights, confirmed). A stateless rANS over the marginal is within 0.01 bit of the
   order-1 optimum → do NOT build a context-modelling arithmetic coder.

## Decision

accepted

Reason: information-theoretic floors are decisive and reproducible. REEBORN targets
lossless-vs-quantized (q4/q3); spatial context modelling is dropped.

Paper value: positive evidence (quantized-lossless floor) + null evidence (no spatial redundancy).

## Next Experiment

E1 — attack the scale stream: the 0.50 b/w fp16 per-block scale overhead now exceeds the
0.40 b/w code-entropy savings. See `2026-06-24-e1-scale-stream-compression.md`.
