# R159 — bit-plane as a fast lossless container codec: 4× rANS inference (GO, with honest ceiling)

- Date: 2026-06-20
- Codec: rtc-bitplane-v1 (`BitplaneCodec` wired into the container + NEON decode)
- Model: Gemma 3 1B IT, `pack --codec bitplane`; speed/RAM/tokens via `gemma-test` + `/usr/bin/time -l`
- Verdict: **GO** — bit-plane is a real, registered, lossless `.rllm` codec whose
  **NEON fixed-width decode makes lossless-compressed inference 4× faster than rANS**
  (1.08 vs 0.27 tok/s), lossless-verified. Honest ceiling: it still loses to bf16
  in-RAM (1.08 vs 1.7) because bf16 is zero-copy and any codec pays a decode cost.

## Why bit-plane for speed (the R158c finding)

Parallel rANS decode only bought +13% — because rANS entropy decode is **sequential**
per lane (lane-parallel only, no SIMD within a lane). Bit-plane is **fixed-width**, so
its decode is a branchless **NEON tbl-gather** (R143, ~6 Gw/s/core vs rANS ~0.3) — ~20×
faster decode. The trade is ratio: 14 bits/weight (bit-plane) vs 10.5 (rANS).

## Change

- `BitplaneCodec::decode`: use `decode_bitplane_row_into` (NEON w=5/6, scalar tail) on
  aarch64 instead of the scalar `BufferedBitReader` loop.
- `BitplaneCodec::encode`: dtype-agnostic (the bf16 (exp,residual) split is bijective on
  any even-length bytes; pack passes dtype "u8"); odd-length / palette>64 → byte-based
  raw fallback (`raw_chunk`).
- Registered "rtc-bitplane-v1" in all 3 dispatchers (loader/verify/unpack);
  `PackCodecPolicy::Bitplane` + `--codec bitplane`.

## Results — measured

```
                file     peak RAM   speed        lossless
rANS            1.30 GB  2.26 GB    0.27 tok/s   yes
bit-plane(R159) 1.62 GB  2.56 GB    1.08 tok/s   yes (verify: LOSSLESS VERIFIED)
bf16            1.90 GB  2.34 GB    1.71 tok/s   yes
q8              1.10 GB  2.10 GB    fast         NO (lossy)
```
Tokens identical to bf16 (`[9079,236761,108,818,7488,3207,...]`). rtc-codec 55 /
rllm-runtime lib 296, 0 warnings.

## Analysis — honest

- **bit-plane IS the fast lossless codec:** 4× rANS inference, lossless, via NEON decode.
  Within the compressed-lossless options, it's the speed choice (rANS is the size choice).
- **In-RAM, bf16 still wins** (1.7 vs 1.08, and bf16 RAM 2.34 < bit-plane 2.56) — bf16 is
  zero-copy (no decode); any codec's decode is additive (R144/R145). Compression's home is
  the **>RAM regime**, where bf16 can't fit and a fast-decode codec (bit-plane) runs.
- **bit-plane uses more RAM than rANS/bf16** here because its compressed body is bigger
  (14 vs 10.5 bit; and it decodes per chunk). The speed/size/RAM trade is now explicit.

## The lossless codec-choice matrix (the real deliverable)

- **Fits RAM, want fastest:** bf16 (zero-copy).
- **> RAM, want smallest:** rANS (1.3 GB, slow decode).
- **> RAM, want fastest lossless:** bit-plane (1.62 GB, NEON decode, 4× rANS).
- Lossy OK: q8 (smallest + fast, not bit-exact).

## Decision

**GO** — bit-plane is a first-class fast lossless codec; it makes lossless-compressed
inference 4× faster than rANS. The honest boundary stands: in-RAM, bf16 wins (no decode);
compression is a >RAM play.

## Next

- Parallel NEON bit-plane decode (container path is single-threaded) → push past 1.08.
- The >RAM demo: a model whose bf16 doesn't fit but bit-plane does — where fast-decode
  bit-plane is the only lossless option that runs.

## Verification status

- [x] `pack --codec bitplane` → LOSSLESS VERIFIED (340 tensors byte-exact).
- [x] Inference token-identical to bf16; 1.08 tok/s = 4× rANS (0.27).
- [x] Registered in 3 dispatchers; rtc-codec 55 / runtime 296, 0 warnings.
