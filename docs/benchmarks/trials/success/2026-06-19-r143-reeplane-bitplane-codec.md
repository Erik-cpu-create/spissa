# Trial: R143 — REEPLANE `rtc-bitplane-v1` SIMD-decodable lossless bf16 codec

Date: 2026-06-19
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

A fixed-width palette-index bit-plane codec, decoded with a branchless NEON
`tbl`-gather kernel, can decode the real Llama 1B bf16 embedding **fast enough to
clear the bandwidth budget** — unlike Huffman (R142, NO-GO: 0.18 Gweight/s, killed
by its per-symbol serial dependency chain). It accepts a worse ratio (~13
bits/weight, 19% saving, vs Huffman's 10.6 / 34%) to buy SIMD-parallel decode.
If decode clears the budget, lossless compressed-resident inference is viable on
CPU — the open niche the R140 series identified.

## Scope

- Mode: experimental (compressed-resident codec; decode throughput gate, Phase B)
- REE kernel: REEPLANE (working name; Erik's final call before any paper use)
- Model/artifact: `Llama-3.2-1B-Instruct-raw.spsa` tied bf16 embedding (262.7M weights, 525 MB; palette 32, w=5)
- Architecture: LLaMA 3.2 1B bf16 embedding/LM head
- Target device/profile: Apple A18 Pro (2 P + 4 E), macOS; release build
- Expected bottleneck: index-plane unpack + palette gather (NEON `tbl`)
- Bottleneck tag: IO/decode

## Setup

Commands:

```bash
# Codec + scalar decode + NEON-vs-scalar parity (all green):
cargo test -p rtc-codec --lib bitplane decode_neon_w5 -- --nocapture

# Real 525MB sample (reused from R142):
cargo test -p rllm-runtime --release dump_bf16_embedding_sample -- --ignored --nocapture

# Feasibility gate (single-core NEON decode throughput):
cargo test -p rtc-codec --release bitplane_neon_decode_feasibility -- --ignored --nocapture
```

Runtime context:

- build profile: release
- CPU: Apple A18 Pro (2 P + 4 E)
- RAM: encode + decode of the 525 MB sample in-process
- OS: macOS (Darwin 25.5.0)
- relevant env/config: none (pure codec micro-benchmark)

## Results

Compression (real embedding): palette = 32 distinct exponents → index width
`w = 5`, **13.000 bits/weight = 81% of bf16 (19% saving)**.

Single-core decode of the 262.7M-weight embedding (8 warm iters, materializing):

| metric | scalar bitplane | NEON REEPLANE | ratio |
|---|---:|---:|---:|
| throughput (Gweight/s) | 0.888 | **5.07** | **5.7×** |
| time / decode | ~296 ms | 51.8 ms | — |
| bit-identical to scalar? | — | yes (parity test green) | — |

Verdict (threshold identical to R142, so the arc composes):

| quantity | value |
|---|---:|
| NEON single-core | 5.07 Gweight/s |
| aggregate (×3.5, A18 2P+4E) | **17.7 Gweight/s** |
| GO threshold (≥12) | cleared |
| RAM-bound ceiling for 13-bit read (~14.7 Gw/s agg) | cleared |
| **VERDICT** | **🟢 GO** |

For reference: R142 Huffman buffered decode was 0.18 Gweight/s (0.6 aggregate,
NO-GO). REEPLANE is **~28× faster** end of arc. All `rtc-codec` tests pass (44 +
3 ignored benches); NEON decode is bit-identical to scalar across sizes 32..4099
including tail boundaries, plus roundtrip-lossless over palettes 1..64 and the
>64-exponent raw fallback.

## Analysis

The gate passes cleanly, and it resolves the central question of the R140-R143
arc. The two failure modes are now both characterized and one is beaten:

- **Ratio path (Huffman, R142):** 10.6 bits/weight (34% saving) but decode is a
  per-symbol serial chain → 0.18 Gweight/s, ~20× below budget. NO-GO.
- **Speed path (bit-plane, R143):** 13 bits/weight (19% saving) but decode is
  branchless — NEON `tbl4` gathers 8 exponents per call, per-lane shifts unpack
  the 5-bit indices, and the bf16 reconstruct is pure SIMD. The `tbl` gather more
  than recovers the 5-bit-unpack cost the scout had skipped (byte-aligned scalar
  was 1.44; real packed NEON is 5.07). 17.7 Gweight/s aggregate.

Where 17.7 Gweight/s lands against the physics (from R141): plain bf16 decode is
~12 Gweight/s (bandwidth-bound, 22 ms/token for the embedding). A 13-bit
compressed read is ~17.9 ms (the new RAM floor); decoding at ≥14.7 Gweight/s
aggregate means decode is *not* the bottleneck → the GEMV becomes bound by the
reduced read → **~1.23× faster decode + 19% less resident RAM, lossless**. The
measured 17.7 clears that, and it is the *materializing* number (the fused kernel
skips the 525 MB DRAM store, so the real fused throughput is higher still).

The honest resolution of the frontier tradeoff:

> For lossless bf16 on CPU you cannot have both the best ratio and fast decode.
> The entropy-optimal codec (Huffman, 34%) decodes too slowly (serial); the
> SIMD-fast codec (bit-plane, 19%) gives up ~15% of the ratio but decodes ~28×
> faster — fast enough to win. **The fast-decode side clears the bar; lossless
> compressed-resident is viable on CPU/ARM at ~1.2×/19%.**

Caveat (honest): this is the **decode-throughput gate (Phase B)**, a proxy. It
does not yet prove the e2e win — that is Phase C (fuse REEPLANE decode→bfdot into
a resident GEMV, no DRAM store, measure decode tok/s + RSS). The gate clearing by
margin is strong evidence Phase C will show the projected ~1.2×/19%, but the e2e
number must be measured, not assumed.

## Decision

accepted (GO)

Reason: NEON bit-plane decode reaches 5.07 Gweight/s single-core (17.7 aggregate),
clearing the ≥12 GO threshold and the 13-bit RAM-bound ceiling, at 13 bits/weight
(19% RAM), bit-identical to scalar. Decode is no longer the bottleneck — the first
GO in the lossless-compressed-resident arc. Proceed to Phase C (fused
decode→bfdot resident GEMV) under its own spec.

Paper value:

- use as positive evidence: lossless compressed-resident inference is decode-viable
  on CPU/ARM (the open niche; SOTA is all GPU). Combined with R142 (Huffman too
  slow) it cleanly demonstrates the ratio-vs-decode-speed tradeoff and that the
  fast-decode side wins — a complete, honest frontier result.

## Next Experiment

Phase C (own spec): fuse REEPLANE decode→bfdot into a resident GEMV for the tied
bf16 LM head — keep the bit-plane planes resident (~19% less RAM), decode a tile
into registers, feed R141's `bf16_row_dot_bf16`, never materialize bf16 to DRAM.
Wire behind `--fast`, measure decode tok/s + RSS vs plain bf16. Target: demonstrate
the projected ~1.2× decode / 19% RAM, lossless, end-to-end.
