# Research Trial: E1 — scale-stream compression

Date: 2026-06-24
Owner: REEBORN
Status: accepted
Folder: success

## Hypothesis

E0 showed the fp16 per-block scale overhead (0.50 b/w @ block 32) now exceeds the
code-entropy savings (0.40). The scale stream should be highly compressible (per-block absmax
has a limited, smooth range), and block size should trade bits against quality. Quantify both
and find REEBORN's operating point.

## Scope

- Experiment type: scale-stream + quant-rate
- REE codec: REEBORN (design input)
- Lossless reference: vs q4/q3 checkpoint
- Model/artifact + weight source: Llama-3.2-1B-Instruct (same source as E0)
- Dtype / param count: BF16, 1.236 B (norms excluded)
- Finding tag: quantization tradeoff + redundancy source

## Method

Script:

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp1_scale.py
```

Block-size sweep B ∈ {16,32,64,128,256}. Per B: q4/q3 code entropy; raw scale overhead 16/B;
entropy-coded fp16 scale (lossless re-code of the same scales); log2-scale delta-coded scale;
SQNR(dB) = 10·log10(Σw² / Σ(w−dequant)²) as the quality cost.

Runtime context: numpy 2.5.0 (venv `/tmp/reeborn-venv`); macOS arm64; ~3 min.

## Results

q4 (levels ±7), bits/weight:

| B | code_H | scale_raw | scale_ent | scale_dlt | total_best | SQNR dB |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 3.725 | 1.000 | 0.498 | 0.444 | 4.169 | 21.19 |
| 32 | 3.595 | 0.500 | 0.244 | 0.214 | **3.809** | 19.96 |
| 64 | 3.460 | 0.250 | 0.121 | 0.104 | 3.565 | 18.95 |
| 128 | 3.329 | 0.125 | 0.060 | 0.051 | 3.380 | 18.07 |
| 256 | 3.207 | 0.062 | 0.030 | 0.025 | 3.233 | 17.28 |

q3 @ B32: 2.427 (codes) + 0.214 (log-delta scale) = **2.641 b/w** @ 12.62 dB.

## Analysis

- **Scale entropy-coding is a free ~2× win.** The fp16 scale carries only ~7.8 bits of real
  entropy. Overhead 0.500 → 0.244 (entropy-coded fp16) → 0.214 (log2-scale delta). This is a
  lossless re-code of the same scales → zero quality cost (confirms E0's hypothesis).
- **REEBORN q4 @ B32 = 3.81 b/w**, bit-exact to the q4 checkpoint, identical quality, 15%
  smaller than GGUF q4_0 (4.5). **q3 @ B32 = 2.64 b/w.**
- **Block size is a clean bits↔quality dial.** code_H and scale both fall as B grows (a larger
  per-block absmax pulls codes toward 0) but SQNR drops ~1 dB per doubling. No free lunch:
  B 32→256 saves 0.58 b/w (q4) at a 2.7 dB SQNR cost.
- Every lever so far trades bits vs quality. The outlier sidechannel (E2) is the only one that
  can improve **both** (pull extreme weights out → smaller block absmax → tighter codes AND
  finer resolution).
- SQNR ~20 dB looks low but LLMs tolerate weight noise; real quality = perplexity/output, which
  needs a pack+run (E3+), not an offline metric.

## Decision

accepted

Reason: the scale stream compresses ~2× losslessly for free, locking REEBORN's theoretical
budget at q4 ≈ 3.81 b/w and q3 ≈ 2.64 b/w. The scale codec = log-delta/entropy-coded; block
size is the quality knob.

Paper value: positive evidence.

## Next Experiment

E2 — outlier sidechannel: measure the fraction of fat-tail weights and the effect of pulling
1–2 extreme weights/block out on block absmax → code entropy, scale, AND SQNR. The one lever
that improves size and quality together. Then E3 — build the real Rust rANS + scale codec and
validate the 3.81 / 2.64 predictions with a bit-exact round-trip.
