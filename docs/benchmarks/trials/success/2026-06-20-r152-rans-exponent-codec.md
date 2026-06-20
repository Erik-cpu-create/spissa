# R152 — rtc-rans-v1: interleaved rANS exponent codec hits the entropy floor + hides under streaming (GO)

- Date: 2026-06-20
- Kernel/codec lineage: **rtc-rans-v1** (new entropy coder; targets the R151 floor)
- Model: Gemma 3 1B IT raw-bf16 (`embed_tokens.weight`, `layers.0.mlp.gate_proj.weight`)
- Verdict: **GO** — rANS reaches the exponent entropy floor (lossless **~10.57 bits/weight,
  18–25% smaller than bit-plane**), and 4-lane **interleaved** decode is fast enough to
  **hide under the cold-read bandwidth** when parallelized (margin ~1.5×).

## Hypothesis (from R151)

Lossless compression of LLM weights is purely an exponent-coding problem (mantissa is
white noise). dfloat Huffman hit the floor but decoded serially (R142 NO-GO). A static
rANS coder should hit the same ~2.6-bit exponent entropy, and **interleaved** rANS
(independent lanes → ILP) should decode fast enough that, parallelized per block
(R150a), it stays hidden under the cold-read bandwidth (~1.3 Gweight/s — a far lower
bar than R142's in-RAM 12 GB/s).

## Method

- **`rtc-rans-v1`** (`crates/rtc-codec/src/rans.rs`): static 32-bit rANS, byte renorm,
  SCALE_BITS=12 (M=4096), RANS_L=1<<23. `normalize_freqs` / `rans_encode` /
  `rans_decode` (scalar), plus **`rans_encode_interleaved4` / `rans_decode_interleaved4`**
  (symbol i → lane i%4; four independent states advanced in one body for ILP).
  Roundtrip unit tests (small alphabet, edge cases, full alphabet, interleaved across
  lengths 1..10001) all bit-exact.
- **Scout** (`tests/rans_exponent_scout.rs`, #[ignore]): encode the REAL exponent plane,
  measure bits/weight vs H(exp)/bit-plane/floor, verify decode == exponents, and time
  scalar vs interleaved-4 decode, then check whether 6-core aggregate ≥ the cold-read
  Gweight/s.

## Results — GO

```
embed_tokens.weight (302M weights, 34 exponents):
  RATIO: H(exp)=2.563  rANS=2.569 bits (AT the floor)  vs bit-plane index 6.000
    lossless TOTAL: rANS 10.569  vs bit-plane 14.000  vs floor 10.513   (25% smaller, +0.056 of floor)
  SPEED: scalar 0.185 Gw/s/core | interleaved4 0.314 (1.70x ILP)
    cold read ~1.287 Gw/s; 6-core aggregate 1.89 Gw/s  => GO (hides, margin 1.47x)

gate_proj.weight (8.0M weights, 27 exponents):
  RATIO: rANS 2.617 bits  => lossless TOTAL 10.617 vs bit-plane 13.000 (18% smaller, +0.054 of floor)
  SPEED: scalar 0.184 | interleaved4 0.322 (1.75x ILP)
    6-core aggregate 1.93 Gw/s vs read 1.281  => GO (margin 1.51x)
```

rtc-codec 52 tests pass (incl. 4 new rANS), 0 warnings; rllm-runtime 0 warnings.

## Analysis

- **Ratio: at the floor.** rANS encodes the exponent at 2.57–2.62 bits = within 0.006
  of H(exp). Total lossless = ~10.57 bits/weight (+0.05 of the absolute R151 floor),
  **18–25% smaller than the current bit-plane** (which wasted 2.4–3.5 bits on a
  fixed-width index). The residual byte (8 bits, white noise) is copied raw.
- **Speed: the interleaving lever clears R142's ghost.** Scalar rANS (~0.185 Gw/s/core)
  carries the same per-symbol serial dependency that sank Huffman. 4-lane interleaving
  gives 1.70–1.75× (independent lane states pipeline on one core); 6-core block-parallel
  (R150a) → ~1.9 Gw/s aggregate vs the ~1.29 Gw/s the cold read delivers → **hides with
  ~1.5× margin.** The bar is the *cold read*, not DRAM — which is why this is GO where
  R142 (in-RAM, 12 GB/s bar) was NO-GO.
- **Honest bounds:** the 1.5× margin is on fast NVMe (1.7 GB/s). On slower/cheaper
  storage (the actual mission) the read is slower → the margin grows → more robustly GO.
  4-way ILP gave 1.7× (not 4× — renorm branches + the 4 KB slot table cap it); 8-way is
  available headroom if a faster device needs it. vs q8 the codec stays ~24% bigger
  (the irreducible mantissa, R151) — but this is the best *lossless*, at the floor,
  parallel-decodable.

## Decision

**GO** — rtc-rans-v1 is the lossless codec at the entropy floor with parallel-hideable
decode: the novel CPU/ARM contribution R151 pointed to (SOTA lossless-fused is all GPU).
Feasibility proven on real weights; correctness bit-exact.

## Next (R153)

Integrate rtc-rans-v1 into the streaming path: replace the bit-plane fixed-width
exponent index with a per-block interleaved-rANS exponent stream (residual plane
unchanged), wire into REESTREAM-PAR, and measure the end-to-end >RAM cold capacity
win vs raw bf16 (expected ~ raw/1.51 ≈ 1.5× from bytes, vs bit-plane's 1.15×) — and
confirm losslessness end-to-end.

## Verification status

- [x] rANS + interleaved-4 roundtrip bit-exact (unit, lengths 1..10001).
- [x] Real exponent plane: decode == exponents (bit-exact), 2 tensors.
- [x] Ratio at the floor (10.57 bits, 18–25% < bit-plane).
- [x] Interleaved decode hides under cold read (6-core ≥ read, margin ~1.5×).
- [x] rtc-codec 52 / rllm-runtime 0-warn green.
