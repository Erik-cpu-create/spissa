# Spec: R143 — `rtc-bitplane-v1` SIMD-decodable lossless bf16 codec

Date: 2026-06-19
Status: design (approved to draft)
REE kernel (working name): **REEPLANE** (bit-plane palette decode; final name: Erik's call before any report/paper use)

## Honest positioning (read first)

This builds the **proof-of-concept** that lossless compressed-resident inference
is achievable on CPU/ARM — the open niche the R140-series identified (every SOTA
weight-compression paper is GPU/Tensor-Core). It is deliberately a *modest* win,
and we state the ceiling up front:

> **Ceiling: ~1.2× decode speed + ~19% RAM/bandwidth saving, lossless** (bf16
> weights bit-exact). Measured on the real Llama 1B bf16 embedding.

The R140→R142 arc established the binding science. R142 proved the entropy-optimal
codec (Huffman, 10.6 bits/weight, 34% saving) cannot be decoded fast enough on CPU
— its per-symbol serial dependency chain caps decode at ~0.18 Gweight/s
(~20× below the bandwidth budget). The fundamental tradeoff this spec accepts:

> **For lossless bf16 on CPU you can have a good ratio XOR fast decode, not both.**
> The bf16 exponent carries only ~2.6 bits of entropy; the residual
> (sign + 7-bit mantissa) is ~8 incompressible bits. Huffman reaches the 2.6-bit
> floor but decodes serially; a **fixed-width palette index** decodes branchlessly
> (SIMD `tbl` gather) but spends 5 bits on the exponent instead of 2.6.

R143 takes the fast-decode side: a fixed-width, SIMD-decodable layout. A scalar
proof (auto-vectorized) already hit ~1.44 Gweight/s/core (5.0 aggregate, MARGINAL
edge) vs Huffman's 0.18 — a ~7× decode speedup from removing the serial chain. The
open question — the one this spec gates everything on — is whether a hand-tuned
**NEON `tbl`** decode clears the GO bar.

## Scope & decomposition

Three phases, each gating the next. This spec fully specifies **Phase A** (codec)
and **Phase B** (NEON decode + the decisive throughput gate); **Phase C** (fused
decode→bfdot resident GEMV) is designed at high level and gets its own spec only
if Phase B lands GO/MARGINAL.

- **Phase A — `rtc-bitplane-v1` codec.** Encode + scalar decode, bit-identical
  lossless, measured compression ratio (expected ~13 bits/weight).
- **Phase B — NEON decode kernel + feasibility gate.** The crux: does
  `tbl`-gather decode clear the throughput bar? Honest GO/MARGINAL/NO-GO. If
  NO-GO, stop — that completes the frontier (even SIMD can't), an honest negative.
- **Phase C — fused decode→bfdot (design-level only, gated).** Decode a tile into
  registers and feed R141's bfdot in the same pass; never materialize bf16 to
  DRAM. Measure e2e decode tok/s + resident RAM. Demonstrates the 1.2×/19% result.

## Design

### Format `rtc-bitplane-v1` (new codec, new ID)

Per bf16 tensor, the encoder splits each weight into `(exponent, residual)` exactly
as `rtc-dfloat-v1` does (`split_bf16`: exponent = bits 14..7, residual = sign|mantissa,
reused verbatim). Then, instead of Huffman-coding the exponent stream:

1. **Palette:** collect the distinct exponents, sorted ascending → `palette: Vec<u8>`
   (length `P`). If `P > 64`, the tensor is **not** bit-plane-compressible at a
   useful width → emit a **raw-fallback** chunk (store the original bf16 bytes,
   marked in the header). Lossless either way. (Real LLM bf16 weights cluster
   tightly: the Llama 1B embedding has exactly 32 distinct exponents.)
2. **Index width:** `w = max(1, ceil(log2(P)))` bits (≤6). Each weight's exponent
   becomes its palette index, packed at `w` bits.
3. **Two planes:**
   - **index plane:** `n` indices × `w` bits, bit-packed in **byte-aligned groups**:
     8 indices pack into exactly `w` bytes (e.g. `w=5` → 8 indices / 5 bytes, no
     straddle across the group boundary), so each group unpacks with a fixed
     shuffle + shift pattern (the standard columnar bit-unpacking technique). No
     variable-length codes → no serial dependency.
   - **residual plane:** `n` bytes, one residual each (incompressible, byte-aligned
     for trivial access and SIMD interleave).

Encoded layout:

```
[ magic: "RTCB" (4) ][ version: u8 = 1 ][ flags: u8 ][ num_weights: u64 ]
[ palette_len P: u8 ][ index_width w: u8 ][ palette: P × u8 ]
[ index_plane: ceil(n*w/8) bytes ][ residual_plane: n bytes ]
```

`flags` bit0 = raw-fallback (then the body is the original bf16 bytes, no palette).
Size at `w=5`: `n·(5/8 + 1) = 1.625 n` bytes = **13 bits/weight** (19% < bf16).

### Decode — two paths

- **Scalar reference (`decode`, the lossless oracle):** for each weight, unpack
  its `w`-bit index, `exp = palette[idx]`, `join_bf16(exp, residual)` (reused
  verbatim), write the bf16. Bit-identical to the original tensor — the contract
  every other path is checked against.
- **NEON kernel `bitplane_decode_neon` (REEPLANE, Phase B):** process a block of
  weights at a time — unpack `w`-bit indices into bytes (SIMD shift/mask), gather
  exponents from the palette with `tbl`/`tbl2` (the palette ≤64 bytes lives in 1-4
  NEON registers), zip the exponent bytes with the residual bytes into bf16 pairs,
  store. Scalar tail for the remainder. `std::arch` / inline asm, consistent with
  the existing `sdot`/`bfdot` kernels — no new dependency, `cargo build` only.

### Phase B feasibility gate (the decisive measurement)

Mirror R142's gate exactly. Encode the real 525 MB Llama 1B bf16 embedding
(`/tmp/rllm-bf16-sample.bin`), then time the NEON decode in the **fused pattern**
(decode into registers + a multiply-accumulate against a small L1-resident
activation, **no DRAM output write** — the same shape the Phase C kernel runs).
Report single-core Gweight/s, the speedup vs the scalar bit-plane decode, and the
verdict against the R141-derived threshold (identical to R142 so results compose):

| single-core decode | aggregate (×3.5) | verdict |
|---|---|---|
| ≥ ~3.4 Gweight/s | ≥ ~12 Gweight/s | 🟢 **GO** — SIMD decode clears the bandwidth budget; build Phase C |
| ~1.4–3.4 Gweight/s | ~5–12 Gweight/s | 🟡 **MARGINAL** — RAM win (19%) holds; speed roughly neutral. Decide per goal |
| < ~1.4 Gweight/s | < ~5 Gweight/s | 🔴 **NO-GO** — even branchless SIMD can't; codec stays storage-only, frontier complete |

Honest framing: the scalar proof was 1.44 Gweight/s/core (5.0 aggregate, MARGINAL
edge) — but it used a **byte-aligned** index (16 bits/weight, no real compression),
so it was *optimistic on the unpack axis*. The real codec packs indices at `w=5`
bits, adding a genuine 5-bit-unpack cost the scout did not pay. So Phase B is a
true unknown: NEON `tbl` + zip must *more than recover* that unpack cost (vs the
scalar byte-aligned baseline) to clear GO. A MARGINAL or NO-GO here is a real,
expected possibility — that is exactly what the gate exists to find. The ×3.5 is
the A18's 2 P + 4 E cores; the bench measures one P-core.

### Phase C (design-level only — separate spec if Phase B is GO/MARGINAL)

Fused compressed-resident GEMV: keep the bit-plane planes resident (≈19% less RAM
than bf16), decode a weight tile into a register/L1 scratch with REEPLANE, feed
R141's `bf16_row_dot_bf16` (bfdot) in the same pass so bf16 never hits DRAM. Wire
behind the existing `--fast` path for the tied bf16 LM head, measure decode tok/s +
resident RSS vs plain bf16. Target: demonstrate the lossless 1.2×/19% e2e result.

## Non-goals

- Beating the 1.2×/19% ceiling — it is the bf16 lossless limit on CPU, stated and
  accepted. (Bigger wins require lossy quantization, a different thesis.)
- Phase C implementation in this plan (designed, gated behind Phase B).
- Registering the codec in the runtime `codec_for_id` for general decode — Phase C
  concern. Phase A/B are codec-crate + bench only.
- Compressing the residual (mantissa is ~incompressible; not worth the complexity).
- GPU. Sub-bf16 precision. KV-cache compression. q8/q4 paths.

## Testing

- **Lossless (hard rule):** `decode(encode(x)) == x` byte-for-byte on skewed,
  single-exponent, palette-overflow (>64 exponents → raw fallback), random bf16,
  and tail-boundary sizes. Empty and single-weight tensors.
- **NEON-vs-scalar parity:** `bitplane_decode_neon` is bit-identical to the scalar
  `decode` on the same inputs (the SIMD path must not drift a bit).
- **Ratio sanity:** the real embedding encodes to ~13 bits/weight (palette 32, w=5).
- **Honest metrics:** Phase B single-core Gweight/s + speedup vs scalar bit-plane +
  the GO/MARGINAL/NO-GO verdict — reported plainly, NO-GO included.
- Existing `rtc-codec` tests stay green (new codec is additive).

## Originality & dependencies (doctrine)

- **Original code.** Palette + fixed-width index plane + `tbl`-gather decode is a
  standard SIMD-decompression pattern; this implementation is written from scratch
  in this repo, reusing the crate's own `split_bf16`/`join_bf16`. No external
  decompression library, no port of any runtime.
- **No new dependencies.** NEON via `std::arch`/inline asm (built in). `cargo build`
  stays the only requirement.
- **Lossless by default** preserved end to end; proven by the bit-identical tests.

## Components / isolation

- `crates/rtc-codec/src/bitplane.rs` — `BitplaneCodec` (encode + scalar decode) +
  its own bit-packing helpers + tests. One responsibility, isolated from dfloat.
- `bitplane_decode_neon` (Phase B) — additive NEON kernel, parity-tested vs scalar.
- Phase B feasibility bench — `#[ignore]`, reads the real sample, prints the verdict.
- Phase C fused kernel — separate spec, gated.

## Prior art (cited honestly)

Palette/dictionary coding + SIMD table-lookup decode are standard (e.g. byte-level
dictionary codecs, `PSHUFB`/`tbl` gathers in fast decoders). RLLM's distinct
position is the combination this spec proves: a lossless bf16 weight codec whose
decode is fast enough on CPU/ARM to feed an exact-weight bf16 GEMV from a resident
compressed buffer — the open CPU/edge gap from the R140 work, where the SOTA
(DFloat11, NeuZip, Huff-LLM, EntroLLM) is GPU or slow-serial on CPU.
