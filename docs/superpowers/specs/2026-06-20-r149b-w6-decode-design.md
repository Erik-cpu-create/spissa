# Spec: R149b — w=6 bit-plane decode (REEPLANE-W6), unblock Gemma streaming lm-head

Date: 2026-06-20
Status: **DONE (GO)** — see `docs/benchmarks/trials/success/2026-06-20-r149b-w6-decode-gemma.md`
REE kernel: **REEPLANE-W6** (new decode variant; extends R143 REEPLANE / R148 REESTREAM)

## Honest positioning (read first)

R149a proved the streaming lm-head stack lossless on **real** Llama weights (w=5,
128256 logits bit-identical) but was **BLOCKED on Gemma 3 1B**: Gemma's tied
embedding has 34 distinct bf16 exponents → `index_width = 6`, and the streaming
decode kernel (`decode16_w5_into` / `decode_neon_w5`) is hard-wired to w=5
(palette ≤ 32). The sidecar writer errors `width 6 != 5` by design.

R149b's job: add a **w=6 SIMD decode** (palette 33–64) and thread the width `w`
through the sidecar so the **already-wired** Gemma generation path
(`RLLM_STREAM_LMHEAD`, `gemma/api.rs:507`) runs and produces an **identical token
sequence** vs the resident lm-head. Correctness-first; speed (capacity-bound,
> RAM) stays deferred (Gemma 1B fits RAM).

## Root cause (verified)

- Palette = the set of distinct bf16 **exponent bytes** (`split_bf16`: exponent =
  bits 14..7; residual = sign<<7 | mantissa). `w = ⌈log₂(palette_len)⌉`.
- The bitstream is **MSB-first** (`BitWriter::write`, `BufferedBitReader`).
- The w=5 NEON kernels exploit `8·5 = 40 bits = 5 bytes` periodicity and gather
  exponents with `vqtbl2q_u8` (32-entry table). Two hard limits for w=6:
  1. the index-extraction tables are 5-bit specific;
  2. `vqtbl2q` addresses only 32 palette entries, but w=6 needs up to 64.

## Design

### 1. `decode16_w6_into` (REEPLANE-W6 SIMD kernel) — `rtc-codec/src/bitplane.rs`

Mirror `decode16_w5_into`, 16 weights/iter, with the w=6 constants:

- Period: `4·6 = 24 bits = 3 bytes`; 16 indices = 96 bits = **12-byte** group
  stride (`g*12`); residual load `g*16`.
- Index extraction (derived, MSB-first 2-byte big-endian window per lane):
  - `bidx_hi: [0,0,1,2,3,3,4,5,6,6,7,8,9,9,10,11]`, `bidx_lo = bidx_hi + 1`
  - right-shift = `10 − (6j mod 8)`, period-4 → `neg_shift = [-10,-4,-6,-8]×2`
  - mask = `0x3f`
- Exponent gather: palette zero-padded to **64 bytes** as `uint8x16x4_t`, gathered
  with **`vqtbl4q_u8`** (single NEON TBL over 4 regs = 64-entry table; indices 0–63).
- bf16 reconstruction identical to w=5 (sign = `(res&0x80)<<8`, exp = `e<<7`,
  mant = `res&0x7f`) — factor into an `#[inline(always)] assemble_bf16(res16, exp16)`
  helper, used by the new kernel (existing frozen w=5 kernels left untouched).
- 16-group bound: `groups16 = if len≥16 {(len−16)/12 + 1} else {0}`,
  `simd16 = min(n/16, groups16)`; **scalar tail** for the remainder via
  `decode_scalar_w(.., w=6, ..)` (byte_off = `done·6/8`, `done` a multiple of 16 ⇒
  byte-aligned).

### 2. `decode_bitplane_row_into` (width dispatcher) — codec crate owns kernel choice

`pub fn decode_bitplane_row_into(palette, idx, res, n, w, out)`:
`w=5 → decode16_w5_into`, `w=6 → decode16_w6_into`, `else → decode_scalar_w`
(general `BufferedBitReader` loop, lossless for any w≤6 — robustness, not the hot
path). This is the **single** width-dispatch entry point the runtime calls; the
runtime stays width-agnostic (correct layering: codec selects the kernel).

### 3. Width propagation through the sidecar — `bitplane_stream.rs`

- `write_lmhead_sidecar`: reject raw-fallback chunks (`flags & FLAG_RAW`, i.e.
  palette > 64 → not bit-plane streamable); accept `w ∈ 1..=6`; require
  `hidden·w % 8 == 0` (row byte-alignment) and `vocab % block_rows == 0`;
  `row_idx = hidden·w/8`. Header **v2**: `RLMH`, ver=2, hidden, vocab, block_rows,
  palette_len, **w**, palette, then framed `[B×index ++ B×residual]` blocks.
- `stream_lmhead_from_sidecar`: read header; ver 1 ⇒ w=5 (palette at 18),
  ver 2 ⇒ w = head[18] (palette at 19); pass `w` through.
- `streaming_bitplane_gemv`: add `w` param; `row_idx = hidden·w/8`; decode each row
  via `decode_bitplane_row_into(.., w, ..)`.
- **No change to `gemma/api.rs`** — the `RLLM_STREAM_LMHEAD` branch already calls
  `stream_lmhead_from_sidecar`.

## Testing (TDD gates)

1. **Unit (fast, no model):** `decode16_w6_matches_scalar_bit_for_bit` — encode a
   34-exponent tensor (w=6), decode via `decode16_w6_into`, assert bit-identical to
   scalar `BitplaneCodec::decode` across n ∈ {64,80,96,1000,4096,4099,65536} (covers
   SIMD/scalar boundary + tails). Plus dispatcher parity (`decode_bitplane_row_into`
   == scalar for w=5 and w=6).
2. **Unit sidecar round-trip (w=6 synthetic):** build a 34-exponent embedding,
   write a v2 sidecar, `stream_lmhead_from_sidecar` == single-thread decode+dot
   reference, bit-for-bit.
3. **Lossless e2e (hard rule), real Gemma 3 1B:** `#[ignore]` test —
   `write_lmhead_sidecar(gemma-3-1b-it-rawcodec.spsa, embed, 256, sidecar)` then
   generate a fixed prompt twice (resident vs `RLLM_STREAM_LMHEAD=<sidecar>`);
   assert **identical token-id sequence**. This is the gate R149a-Gemma could not
   reach.
4. Existing suites stay green (rtc-codec + rllm-runtime lib); streaming path is
   opt-in/env-gated, default unchanged.

## Non-goals

- Speed: Gemma 1B fits RAM; the capacity-bound win (> RAM / cold) is a later phase.
- Streaming the transformer projections (only lm-head).
- w > 6 SIMD (palette > 64 is the codec's raw fallback — not bit-plane streamable).

## Originality & doctrine

- **Original code.** New `vqtbl4q` w=6 kernel + width dispatch written from scratch;
  reuses `BitplaneCodec`, `BufferedBitReader`, `bf16_row_dot_f32`, R148 streaming.
- **No new dependencies.** **Lossless by default** preserved (opt-in, proven
  bit-identical). **Honest metrics:** a failed identical-token gate is reported as
  NO-GO, never fudged.
