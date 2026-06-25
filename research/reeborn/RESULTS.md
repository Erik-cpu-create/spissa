# REEBORN — Research Synthesis

> **Canonical per-experiment records live in `research/trials/`** (the offline-analysis trial
> log, separate from the Rust runtime-benchmark log at `docs/benchmarks/trials/`). This file is
> the project-level synthesis — the locked budget and design decisions across experiments.
>
> - E0 → `research/trials/success/2026-06-24-e0-entropy-floor-bf16-vs-quant.md`
> - E1 → `research/trials/success/2026-06-24-e1-scale-stream-compression.md`

## Experiment 0 — entropy floor (bf16 vs quantized)

**Model:** Llama-3.2-1B-Instruct (bf16, original safetensors), 1.236B params (norms excluded).
**Tool:** `research/reeborn/exp0_entropy.py` (numpy, per-32 symmetric absmax quant).
**Date:** 2026-06-24.

## Raw numbers (bits/weight)

| category | Mweights | bf16_H | sign | exp | mant | q4_H | q4+scale | q3_H | q3+scale | q4\|left | q4\|up |
|----------|---------:|-------:|-----:|----:|-----:|-----:|---------:|-----:|---------:|--------:|-------:|
| linear   |    973.1 | 10.560 | 1.00 | 2.64 | 6.97 | 3.60 |     4.10 | 2.43 |     2.93 |    3.60 |   3.60 |
| embed    |    262.7 | 10.489 | 1.00 | 2.58 | 6.96 | 3.57 |     4.07 | 2.41 |     2.91 |    3.57 |   3.56 |
| **ALL**  |   1235.7 | 10.548 | 1.00 | 2.63 | 6.96 | 3.60 |     4.10 | 2.43 |     2.93 |    3.59 |   3.59 |

`+scale` = fp16 scale per 32-block = 0.50 b/w overhead.

## Findings

**1. True-lossless vs bf16 cannot reach <4 bits — Shannon, not dogma.**
Order-0 bf16 entropy = **10.55 b/w** (independently matches the project's REEFORM ~10.6).
Breakdown: sign **1.00** (random), exponent **2.63/8** (compressible), mantissa **6.96/7**
(99.4% random, ~incompressible). Reaching <4 bits true-lossless needs deleting ~6.5 bits
of near-uniform mantissa entropy → a 2.6× Shannon violation → impossible. REEFORM-style
plane decorrelation can only *approach* 10.55, never go below ~exp+mantissa floor.

**2. Lossless vs a quantized checkpoint (REEBORN's home) reaches <4 bits easily.**
- q4 codes: marginal entropy **3.60** vs nominal 4.0 → flat-4-bit GGUF wastes **0.40 b/w**.
- q3 codes: **2.43**, +scale **2.93** → clean **sub-3-bit** lossless-vs-q3.
- Key surprise: the fp16 **scale overhead (0.50 b/w) now exceeds the code savings (0.40)**.
  q4+scale = 4.10. So the dominant remaining slack is in the *scale stream*, not the codes.

**3. Spatial conditional modelling is a NULL RESULT — drop it.**
H(code|left) = H(code|up) = 3.59 vs marginal 3.60 → **0.01 b/w** gain. Adjacent weights are
statistically independent (trained weights ≈ random matrix, confirmed). A stateless rANS over
the marginal code distribution is within 0.01 bit of the best order-1 model. **Do NOT build a
context-modelling arithmetic coder** — it would add complexity for ~zero gain.

## Redirected REEBORN design

| Lever | Status | Evidence |
|-------|--------|----------|
| 1. rANS entropy-code codes to marginal floor | **ADOPT** | q4 4.0→3.60, q3 3.0→2.43 |
| 2. ~~Cross-block spatial conditional modelling~~ | **DROP** | 0.01 b/w (null) |
| 3. Scale-stream compression (delta/entropy/larger blocks/fp8 scale) | **#1 PRIORITY** | overhead 0.50 > code savings 0.40 |
| 4. Outlier sidechannel (shrink per-block absmax → tighter codes+scales) | **MEASURE** | — |

**Honest target (bit-exact to the quantized checkpoint):** q4 ≈ 3.8 b/w, q3 ≈ 2.6 b/w
(vs GGUF q4_0 4.5 → ~16–40% smaller, identical quality).

**"Lossless" scope:** relative to the q4/q3 checkpoint. The fp→quant step is lossy;
REEBORN adds zero further loss. Win vs GGUF = same codes, smaller file.

---

# Experiment 1 Results — the scale stream (`research/reeborn/exp1_scale.py`)

Block-size sweep + scale-stream compressibility + SQNR (Llama-3.2-1B, 2026-06-24).

### Q4 (levels ±7)
| B | code_H | scale_raw | scale_ent | scale_dlt | total_raw | total_best | SQNR_dB |
|---|-------:|----------:|----------:|----------:|----------:|-----------:|--------:|
| 16  | 3.725 | 1.000 | 0.498 | 0.444 | 4.725 | 4.169 | 21.19 |
| 32  | 3.595 | 0.500 | 0.244 | 0.214 | 4.095 | **3.809** | 19.96 |
| 64  | 3.460 | 0.250 | 0.121 | 0.104 | 3.710 | 3.565 | 18.95 |
| 128 | 3.329 | 0.125 | 0.060 | 0.051 | 3.454 | 3.380 | 18.07 |
| 256 | 3.207 | 0.062 | 0.030 | 0.025 | 3.270 | 3.233 | 17.28 |

### Q3 (levels ±3)
| B | code_H | scale_dlt | total_best | SQNR_dB |
|---|-------:|----------:|-----------:|--------:|
| 32  | 2.427 | 0.214 | **2.641** | 12.62 |
| 64  | 2.287 | 0.104 | 2.391 | 11.62 |
| 256 | 2.040 | 0.025 | 2.066 |  9.97 |

### Findings
1. **Scale entropy-coding is a free ~2× win.** The fp16 scale carries only ~7.8 bits of real
   entropy. Overhead 0.500 → 0.244 (entropy-coded fp16) → 0.214 (log2-scale delta). Lossless
   re-code of the same scales → zero quality cost. (E0 hypothesis confirmed.)
2. **REEBORN q4 @ B32 = 3.81 b/w**, bit-exact to the q4 checkpoint, identical quality, 15%
   smaller than GGUF q4_0 (4.5). **q3 @ B32 = 2.64 b/w.**
3. **Block size = a clean bits↔quality dial.** code_H and scale both fall as B grows (bigger
   per-block absmax pulls codes toward 0) but SQNR drops ~1 dB per doubling. No free lunch:
   B 32→256 saves 0.58 b/w (q4) at a 2.7 dB SQNR cost.
4. Every lever so far trades bits vs quality. The **outlier sidechannel (E2)** is the only one
   that can improve BOTH: pulling 1–2 extreme weights/block out shrinks the block absmax →
   tighter codes for the other 31 (size↓) AND finer resolution (SQNR↑).

### Locked theoretical budget (E0+E1)
`q4 @ B32 = 3.60 (rANS codes) + 0.21 (log-delta scale) = 3.81 b/w` · `q3 @ B32 = 2.64 b/w`.
SQNR ~20 dB looks low but LLMs tolerate weight noise — real quality = perplexity/output,
needs pack+run (E3+).

## Next experiments
- **E2 (outliers):** fraction of fat-tail weights; effect of an outlier sidechannel on block
  absmax → code entropy, scale, AND SQNR. The one lever that improves size and quality together.
- **E3 (rANS prototype, Rust):** real round-trip q4/q3 → rANS + scale codec → bit-exact decode;
  measure achieved b/w vs the 3.81 / 2.64 predictions; then pack+run for perplexity/output.
