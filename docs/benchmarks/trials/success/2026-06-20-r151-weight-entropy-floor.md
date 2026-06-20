# R151 — Shannon-entropy floor of real bf16 weights (measurement that scopes the codec)

- Date: 2026-06-20
- Type: measurement/analysis (informs codec design — not a kernel GO/NO-GO)
- Model: Gemma 3 1B IT (`gemma-3-1b-it-rawcodec.rllm`, raw bf16 everything)
- Tensors: `embed_tokens.weight`, `layers.0.self_attn.q_proj.weight`,
  `layers.0.mlp.gate_proj.weight`, `layers.0.mlp.down_proj.weight`
- Verdict: **floor ≈ 10.5 bits/weight**; lossless compression of LLM weights is
  **entirely an exponent-coding problem** (mantissa is irreducible noise).

## Why

Instead of guessing whether a lower-bit lossless codec is possible, measure the
**theoretical floor** of the real weights first (`H(exponent) + H(residual)`), then
design to it. Run via the public-API integration test
`tests/weight_entropy_analysis.rs` (`--ignored --nocapture`).

## Results (consistent across all four tensors)

| quantity | value | reading |
|---|---|---|
| **H(exponent)** | **~2.6 bits** (26–34 distinct) | bit-plane pays a fixed 5–6-bit index → ~3 bits/weight wasted |
| H(delta-exponent along row) | 3.2 bits (**higher**) | delta/2D-structure coding is a **dead end** |
| **H(residual byte = sign+mantissa)** | **~7.9 / 8 bits** | nearly incompressible |
| per-bit mantissa entropy | **0.98–1.00** (≈ uniform) | the 7 mantissa bits + sign are **white noise** |
| **Shannon floor (order-0)** | **~10.5 bits/weight** | the true lossless minimum |

Per-tensor floor: embed 10.52, q_proj 10.35, gate_proj 10.53, down_proj 10.58.
Current bit-plane: 13–14 bits. q8 (lossy ref): ~8.5 bits.

## Analysis — three definitive conclusions

1. **Lossless = exponent-coding, full stop.** The mantissa (7 bits) + sign are
   measured to be ~uniform random → information-theoretically incompressible (~8 bits
   irreducible). The *only* compressible part is the exponent (2.6 bits of info, paid
   as 5–6 in bit-plane). All lossless headroom lives there.
2. **The +2-bit gap to q8 is the mantissa q8 discards** — information, not a codec
   failure. So "lossless as small/fast as q8" is provably impossible; the honest
   target is "best lossless ≈ floor, at streaming-decode speed."
3. **Delta/2D exponent structure is dead** (H(delta) > H(order-0)). Cross it off.

## Decision → next invention (R152)

Bit-plane (13–14) sits ~2.4–3.5 bits above the 10.5 floor — all recoverable by
entropy-coding the exponent. **dfloat Huffman already hit ~10.6 (the floor) but died
on serial decode (R142).** So the invention is a **SIMD-parallel / interleaved ANS
(rANS/tANS) exponent codec** that reaches the floor at decode throughput that stays
hidden under the cold-read bandwidth (~1.3–1.7 GB/s, R150a) — a far lower bar than
R142's in-RAM 12 GB/s. Expected: lossless **~10.5 bits (1.31 B/weight, ~20% smaller
than bit-plane)**, parallel-decodable. Still ~24% bigger than q8, but that is the
irreducible mantissa — now known, not assumed.

## Verification status

- [x] Floor measured on 4 real tensors (embed + attention + MLP), consistent ~10.5.
- [x] Mantissa shown ~uniform (per-bit H ≈ 1.0) → incompressible.
- [x] Delta-exponent shown worse than order-0 → 2D direction killed.
- [x] Reproducible: `cargo test -p rllm-runtime --release --test weight_entropy_analysis -- --ignored --nocapture`.
