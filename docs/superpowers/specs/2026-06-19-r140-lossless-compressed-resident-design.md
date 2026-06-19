# Spec: R140 — Lossless bf16 compressed-resident + fused dequant (CPU/NEON)

Date: 2026-06-19
Status: design (approved to draft)

## Honest positioning (read first)

This is **not novel research and not a path to beating llama.cpp.** The technique
— losslessly compress bf16 weights via exponent entropy coding and decode them
on-the-fly during the matmul — is published and implemented: **DFloat11**
(NeurIPS '25, open code), **ZipServ** (ASPLOS '26), **Huff-LLM**, **EntroLLM**,
Cloudflare's Unweight. R140 is a **CPU/NEON open-source implementation of that
known technique**, which today exists only on GPU / NPU / proprietary engines.

Why do it anyway:
- It fills a real, narrow gap: there is no open, usable, CPU/ARM,
  lossless-bf16-compressed-resident runtime (llama.cpp/Ollama don't do it).
- The one genuine use-case: run an **exact** model that barely fits (or doesn't
  fit) RAM, on-device — "slow but runs / runs exactly" beats "doesn't run."
- Deep learning value (porting DFloat11-class decode to NEON is hard).

Why it is NOT a speed win: lossless weights are bf16 → float matmul, which cannot
use the int8 `sdot` fast path (sdot needs int8 = lossy). So the full-bf16 path is
SLOWER than q8 `--fast`. The compression buys RAM, not speed. The q8 `--fast`
path stays the recommended fast mode.

## Goal

Add a lossless bf16 compression codec and a fused decode-matmul kernel so RLLM can
keep weights losslessly compressed (~11 bits/weight) resident in RAM and decode
them per-tile inside the matmul — proving the compressed-resident capability
end-to-end on a real hot path, with an honest RAM/speed report.

### Primary target (contained, useful): the tied bf16 embedding / LM head

In the q8 `keep_io` models the tied embedding is already bf16 (1.34 GB for Gemma).
Compressing IT losslessly is a **clean RAM win with zero quality and zero layer-
speed change** (the transformer layers stay q8/sdot; only the embedding/LM-head
read is affected). The LM head is bandwidth-bound (R137), so fused-decode there
may be neutral-to-slightly-faster (read ~11 bits vs 16). This is the first
deliverable: codec + compressed-resident embedding + fused LM-head decode.

### Secondary target (capability demo): full-bf16 lossless model

Apply the same codec to all weights of a fully-bf16 model → an EXACT model at
~11 bits/weight. Slower than q8 (bf16 matmul). Measured and reported honestly as
the "exact, reduced-RAM, slower" capability. Not the primary deliverable.

## Non-goals

- Beating q8 `--fast` on speed (impossible — lossless = float matmul).
- GPU. Novel compression research. Sub-11-bit lossless (entropy floor).
- KV-cache compression.

## Design

### 1. Codec `rtc-dfloat-v1` (field-split lossless bf16)

bf16 = `[sign:1][exp:8][mantissa:7]`. Per tensor, split into two streams:

- **Exponent stream** — the 8-bit exponents, entropy-coded. LLM weights cluster
  on a few exponents (low entropy), so canonical **Huffman** → ~2.6 bits/exp. One
  Huffman table per tensor, stored in the tensor/chunk metadata.
- **Residual stream** — `sign|mantissa` (8 bits), stored RAW (≈incompressible).

Total ≈ 2.6 + 8 ≈ **~11 bits/weight** (validated in Phase 1; matches DFloat11/
ZipNN's ~30% reduction on bf16).

**Tile-granular** layout — the unit of independent decode is a TILE (one output
row, or a fixed block, e.g. 256/512 weights). Each tile = its own exponent
bitstream offset + raw residual span, so `decode_range`/the kernel can decode one
tile without touching others (required for fusion). The `TensorCodec` trait
already has `decode_range`; `rtc-dfloat-v1` implements it at tile granularity.

**Fast decode (the risk)** — NOT bit-by-bit (that is why the old `auto` codec was
6× slow). Use a **canonical-Huffman LUT decode**: a single 2^L-entry table (L =
max code length, ~8–10 for ~32 exponent symbols → ≤2 KB LUT, L1-resident) maps the
next L bits → `(symbol, code_len)`; one lookup + bit-advance per exponent. Then
reassemble `bf16 = (exp << 7) | residual` — vectorizable. Only the variable-length
exponent unpack is serial; the residual read and bf16 reassembly are SIMD.

### 2. Compressed-resident layout

`rtc-dfloat-v1` chunks store the compressed (exponent-bitstream + residual + per-
tensor Huffman table) bytes in the `.rllm`. They are mmap'd and kept **compressed
in RAM** (no full decompress at load). Per-byte SHA-256 still verifies the stored
compressed bytes (lossless integrity preserved). `pack --quantize none --codec
dfloat` (or a dedicated flag) produces such a tensor.

### 3. Fused decode kernel `REEFUSE-DFLOAT-BF16` (name pending Erik)

The LM-head/bf16 GEMV decodes a weight tile from the compressed stream into a
small reused cache-resident scratch, then feeds the existing `bf16_row_dot_f32`
(R137) NEON dot — decode and consume in the same pass (no whole-tensor f32
materialization). Integrates where the bf16 dot reads weights
(`lm_head_logits_*` / the streaming bf16 path). Gated so the exact bf16 path is
unaffected when the tensor isn't dfloat-coded.

## Implementation phases (de-risk inside one plan)

1. **Codec + measure (go/no-go).** Implement `rtc-dfloat-v1` encode/decode +
   lossless round-trip test. Measure on real weights: actual bits/weight (target
   ~11) and **standalone decode throughput (GB/s)**. If decode is far slower than
   the bandwidth it saves, stop here and record the negative result.
2. **Pack path.** `rllm pack` can emit dfloat-coded bf16 tensors; verify in
   `inspect` + `verify`.
3. **Compressed-resident + fused LM-head kernel** for the embedding target.
4. **Measure end-to-end:** RAM + decode tok/s + output parity vs the bf16 and q8
   models, on the embedding target. Then (stretch) the full-bf16 model.

## Testing

- **Lossless round-trip, bit-exact**: `decode(encode(bf16)) == bf16` for random +
  real tensors (the lossless contract).
- **Ratio + throughput**: measured bits/weight and decode GB/s (reported honestly).
- **decode_range correctness**: a tile decodes identically standalone vs whole.
- **End-to-end**: embedding-target RAM (expect Gemma −~0.4 GB), LM-head tok/s
  (expect neutral/slight change), output parity (bit-identical embedding ⇒
  identical logits within the existing fast-path tolerance).
- Existing q8/bf16 tests stay green (codec is additive, gated).

## Originality & dependencies (doctrine)

- **Original code, borrowed idea.** The exponent-split + entropy-decode TECHNIQUE
  is DFloat11's (cited). The IMPLEMENTATION is written from scratch in Rust —
  implemented from the technique/algorithm (canonical Huffman + LUT decode + bf16
  field split are standard, well-understood building blocks), NOT by reading or
  porting DFloat11's CUDA/Python. This matches the RLLM doctrine: studying prior
  work to learn ≠ wrapping/copying.
- **No external dependencies.** Implements its own self-contained canonical Huffman
  (lengths + length-limiting to 15 bits + canonical code assignment + LUT decode)
  directly in `rtc-codec/src/dfloat.rs` — no import from `huff.rs`. Plus a new
  LUT decoder and the bf16 field split — all pure Rust. The NEON kernel uses
  `std::arch` (built in). No zstd/flate2/entropy crates. Consistent with RTC's
  "custom codecs, no external generic compression libraries" rule; `cargo build`
  stays the only requirement.

## Prior art (cited honestly)

DFloat11 (arXiv 2504.11651, NeurIPS'25, this is the same exponent-split scheme),
ZipServ (2603.17435, fused decompress-GEMM), Huff-LLM (2502.00922), EntroLLM
(2505.02380, edge-ARM), Cloudflare Unweight. R140 is the CPU/NEON open
implementation of this line.

## Components / isolation

- `rtc-dfloat-v1` (rtc-codec crate): `encode`/`decode`/`decode_range`; implements
  its own self-contained canonical Huffman in `dfloat.rs` — no import from `huff.rs`.
  Testable standalone.
- Fused decode kernel (rllm-runtime streaming): decode-tile + bf16 dot; depends on
  the codec's tile decode + `bf16_row_dot_f32`.
- Pack/runtime wiring: additive, gated on the new codec id.

## Feasibility result (measured)

**Measured on:** `model.embed_tokens.weight` from `Llama-3.2-1B-Instruct-raw.rllm`
(262,668,288 bf16 weights = 525,336,576 bytes).

| Metric | Value |
|--------|-------|
| bits/weight | **10.625** |
| compressed ratio | 66.4% of raw bf16 |
| decode throughput | **0.02 GB/s** (bf16-out) |
| decode latency | ~29,990 ms/decode (full tensor) |

**GO/NO-GO for R140b (fused compressed-resident kernel): NO-GO.**

The codec ratio is excellent — 10.625 bits/weight (target was ~11), confirming the
compression thesis. However, the scalar bit-reader decode runs at only 0.02 GB/s,
roughly **800× slower** than the ~16 GB/s memory bandwidth it would save. Even
tiled/per-row use does not rescue this: the Huffman bit-reader is inherently serial
per symbol and will bottleneck any fused matmul kernel far harder than raw bf16
bandwidth.

**What this means:**

- R140a (codec + lossless on-disk storage) is a **GO** — the codec is correct,
  lossless, and achieves the target ratio. It is useful for producing smaller
  `.rllm` files on disk.
- R140b (compressed-resident + fused kernel) is a **NO-GO** at the current decode
  speed. The gap (0.02 GB/s vs ~5 GB/s needed) requires a SIMD/NEON LUT decode
  path (vectorized 8-weight-at-a-time LUT lookup + bf16 reassembly) before
  compressed-resident makes sense on CPU. That is a substantially larger kernel
  effort and should be a separate, explicitly-scoped plan if pursued.
- This is recorded as a **useful negative result**, not a failure: the ratio
  numbers validate the compression approach; the decode bottleneck is the expected
  risk that the feasibility gate was designed to catch.
