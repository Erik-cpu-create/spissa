# Spec: R144 — Fused REEPLANE decode→bfdot GEMV (Phase C, e2e proof)

Date: 2026-06-19
Status: design (approved to draft)
REE kernel (working name): **REEFUSE-PLANE-DOT** (fuses REEPLANE decode + REEFLOW-BF16-DOT; final name: Erik's call before any report/paper use)

## Honest positioning (read first)

This is **Phase C** of the lossless-compressed-resident arc: fuse the two
validated halves into one kernel and measure whether the end-to-end GEMV is
actually faster and smaller, lossless.

- **R141 (REEFLOW-BF16-DOT):** exact-bf16 `bfdot` row-dot. The compute half.
- **R143 (REEPLANE):** fixed-width bit-plane codec, NEON decode 17.7 Gweight/s
  aggregate (GO), 13 bits/weight. The bytes-read half.

Phase B (R143) measured *decode throughput in isolation* — a proxy. R144 measures
the **actual lm-head GEMV**: keep the bit-plane planes resident (≈19% less RAM
than bf16), decode each weight row into a small L1 scratch, and `bfdot` it against
the activation in the same pass — bf16 never materializes to DRAM. The question:

> Is `decode→bfdot` from a 13-bit resident buffer faster than reading 16-bit bf16
> from DRAM and `bfdot`-ing it, and is it bit-identical (lossless)?

Honest scope: this is the GEMV-level proof, not full-model generation. The
bit-plane codec only applies to the tied bf16 embedding / LM head (the transformer
layers are q8, a different codec), so the LM-head GEMV **is** the lever — measuring
it in isolation is the clean, decisive test. If positive, wiring it behind
`--fast` for real generation is a trivial follow-up (its own small spec); if the
fused GEMV is not faster, we learn that honestly before any plumbing.

## Goal

Build a fused kernel `lm_head_logits_rows_bitplane` that computes the LM-head
logits GEMV directly from the resident `rtc-bitplane-v1` planes (palette + index
plane + residual plane) via per-row REEPLANE decode into a reused scratch + R141
`bf16_row_dot_bf16`. Benchmark it against the existing plain-bf16 GEMV
(`lm_head_logits_rows_bf16`) on the real 525 MB Llama 1B embedding: per-GEMV time,
resident bytes, and bit-identical logit parity. Report a GO/MARGINAL/NO-GO verdict
on the e2e-of-GEMV win.

## Design

### Row byte-alignment (why per-row decode is clean)

For the tied embedding, `hidden = 2048`, palette = 32 → `w = 5`. Each row's index
plane is `hidden * w = 2048 * 5 = 10240` bits = **1280 bytes** (byte-aligned), and
its residual plane is `hidden = 2048` bytes. So row `r`'s planes start at byte
offsets `r * 1280` (index) and `r * 2048` (residual) — no per-row framing or
bit-straddle. The kernel requires `hidden * w % 8 == 0` (true here); other shapes
fall back to the plain path. This is a benchmark-scope constraint, stated, not a
general codec limitation.

### 1. `decode_neon_w5_into` (rtc-codec, no-alloc)

Expose `pub fn decode_neon_w5_into(palette: &[u8], idx_plane: &[u8], residuals:
&[u8], n: usize, out: &mut [u8])` — the existing `decode_w5_neon_inner` already
writes into a caller `out: &mut [u8]`; this is a thin `pub` wrapper (DRY, no new
decode logic). It lets the fused kernel decode one row into a reused 2-byte×hidden
scratch with **zero allocation per row**.

### 2. Fused GEMV `lm_head_logits_rows_bitplane` (rllm-runtime)

New file `crates/rllm-runtime/src/streaming/bitplane_gemv.rs`, wired via the
existing `include!` pattern so it shares module scope with `bf16_row_dot_bf16` and
`Bf16DotActivation` (no cross-module plumbing). Signature mirrors
`lm_head_logits_rows_bf16`:

```
fn lm_head_logits_rows_bitplane(
    last_hidden: &[f32], palette: &[u8], idx_plane: &[u8], residuals: &[u8],
    hidden: usize, row_offset: usize, out: &mut [f32],
)
```

Per call: build the bf16 activation once (`Bf16DotActivation::new`, R141), allocate
one row scratch `Vec<u8>` of `hidden*2` bytes, then for each output row: pass the
row's planes as **open-ended slices** `&idx_plane[r*1280..]` and
`&residuals[r*2048..]` (NOT row-exact slices) into `decode_neon_w5_into(...,
&mut scratch)`, and `act.row_dot(&scratch, hidden)` (the bfdot path). The
open-ended slice matters: the NEON 8-byte group load can read a few bytes past a
row's 1280-byte index span — with the open slice those bytes belong to the next
row (in-bounds, harmless), and the final row is covered by the kernel's existing
`simd_groups` guard + scalar tail. bf16 stays in the L1-resident scratch; never
written to DRAM.

### 3. Benchmark (`#[ignore]`, rllm-runtime)

Reads `/tmp/rllm-bf16-sample.bin` (the 262.7M-weight embedding = vocab 128256 ×
hidden 2048). Encode with `BitplaneCodec`, keep the planes resident. With a random
f32 activation (`hidden=2048`), under `RLLM_BF16_DOT=1` (so both paths use bfdot):

- **plain bf16 path:** time `lm_head_logits_rows_bf16` over all 128256 rows.
- **fused bit-plane path:** time `lm_head_logits_rows_bitplane` over all rows.
- **resident bytes:** plane total (≈427 MB = 164 MB index + 263 MB residual, 13
  bits/weight) vs bf16 (525 MB) — the ~19% saving.
- **lossless parity:** assert the fused logits **equal** the plain-bf16-bfdot
  logits bit-for-bit (same exact bf16 weights, same bfdot kernel) — the lossless
  e2e proof.

Verdict (honest): fused faster than plain? by how much? RAM saved? Report
GO (faster + smaller), MARGINAL (smaller, ≈same speed), NO-GO (slower).

## Non-goals

- Full `--fast` model wiring, packing models with `rtc-bitplane-v1`, registering
  in `codec_for_id` — a follow-up spec if R144 is GO/MARGINAL.
- General `(hidden, w)` shapes in the fused kernel (only the embedding's `w=5`,
  `hidden=2048`); other shapes use the plain path.
- Multi-threading the GEMV (the existing parallel lm-head wrapper can call the
  fused per-row path later; the bench measures single-core to compare cleanly).
- Beating the ~1.2×/19% ceiling. GPU. q8/q4 layers. KV-cache.

## Testing

- **Lossless parity (hard rule):** fused logits bit-identical to plain-bf16-bfdot
  logits on a small synthetic embedding (e.g. vocab 64 × hidden 2048, 32
  exponents) — same weights + same kernel → exact equality. A unit test, not just
  the bench.
- **Decode-into correctness:** `decode_neon_w5_into` produces the same bytes as
  `decode_neon_w5` (the allocating variant) — small parity test in rtc-codec.
- **Honest metrics:** the bench reports plain vs fused ms/GEMV, resident bytes, the
  exact-parity confirmation, and the verdict — MARGINAL/NO-GO stated plainly.
- Existing rtc-codec + rllm-runtime tests stay green (additive).

## Originality & dependencies (doctrine)

- **Original code.** The fused kernel composes two in-house kernels (REEPLANE
  decode, R141 bfdot) written from scratch in this repo. No external runtime.
- **No new dependencies.** rllm-runtime already depends on rtc-codec; NEON via
  `std::arch`. `cargo build` stays the only requirement.
- **Lossless by default** preserved: the fused path decodes bit-exact bf16 weights;
  proven bit-identical to the read-bf16 path by the parity test.

## Components / isolation

- `decode_neon_w5_into` (rtc-codec) — thin no-alloc wrapper over existing inner;
  parity-tested vs the allocating variant.
- `lm_head_logits_rows_bitplane` (rllm-runtime `bitplane_gemv.rs`) — fused per-row
  decode→bfdot; reuses `Bf16DotActivation` + `bf16_row_dot_bf16`.
- Benchmark — `#[ignore]`, reads the real sample, prints plain-vs-fused + verdict.
- Full `--fast` wiring — separate follow-up spec, gated on this verdict.

## Prior art (cited honestly)

Fused decompress-in-kernel (weights never materialize uncompressed) is the GPU
SOTA idea (DFloat11, NeuZip, Cloudflare Unweight). R144's distinct position is the
CPU/ARM realization: a resident bit-plane buffer decoded per-tile into registers
and fed to a native bf16 `bfdot`, lossless — the open CPU/edge gap from the R140
work, now assembled from the R141 + R143 building blocks.
