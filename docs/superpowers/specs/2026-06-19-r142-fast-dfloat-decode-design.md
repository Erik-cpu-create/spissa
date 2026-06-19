# Spec: R142 — Fast `rtc-dfloat-v1` decode (feasibility gate)

Date: 2026-06-19
Status: design (approved to draft)
REE kernel (working name): **REEDRIP** (decode-drip → feeds the R141 bfdot kernel; final name: Erik's call before any report/paper use)

## Honest positioning (read first)

This is the **feasibility gate for Phase 2** (compressed-resident). It does NOT
build compressed-resident inference. It answers ONE question with a measured
number, cheaply, before any larger build:

> Can the `rtc-dfloat-v1` lossless bf16 codec decode fast enough that reading
> ~10.6 bits/weight from RAM and decoding on the fly beats reading 16-bit bf16
> straight from RAM?

R140a measured the existing decoder at **0.02 GB/s** — the `BitReader.peek(n)`
loop reads bit-by-bit (up to 15 div/mod iterations per exponent symbol), and the
stream is one monolithic blob per tensor. That number killed R140b. R142 rewrites
ONLY the decode hot path (buffered bit-reader + LUT, pre-allocated output),
re-measures, and gates everything downstream on the result.

**This is high-risk and we say so up front.** R140a decoded at ~0.01 Gweight/s
single-core — and we now know exactly why: `peek` does a div + a mod **per bit**,
up to 15 bits × 262M symbols ≈ 34 s of div/mod, which is essentially the entire
26 s decode. Removing it (a buffered shift/mask window) should give a large
multiple. But a clean GO needs single-core decode around **~3.4 Gweight/s** (so
all cores aggregate past the GEMV's ~12 Gweight/s bandwidth rate — see Threshold).
Well-tuned table-driven Huffman decoders land ~0.5–3 Gweight/s of bf16 output per
core, so this realistically lands in the **MARGINAL-to-GO band, not a slam dunk**.
R142 measures exactly where. If it falls short, we record the honest negative and
stop — exactly as R140b did.

## The combined vision (where this sits)

- **R140a — codec (done, on main).** Lossless bf16 codec `rtc-dfloat-v1`, 10.6
  bits/weight. Decodes at 0.02 GB/s (naive reader). Used for disk storage.
- **R141 — bfdot compute (done, on main).** Exact-bf16 GEMV kernel. Validated,
  ~1.2× isolated, opt-in. Proved the decode bottleneck is **bytes read**, not
  compute (the bf16 LM head is memory-bandwidth-bound at ~24 GB/s).
- **R142 — fast decode feasibility (THIS spec).** Make the decoder fast in
  isolation, measure Gweight/s, GO/NO-GO. The missing link: if decode is fast
  enough, fewer bytes are read per row → the GEMV stops being bandwidth-bound →
  R141's compute win finally surfaces.
- **Phase 2 proper (separate spec, gated on R142 GO).** Per-row framing in the
  codec stream + fused decode→bfdot across cores → compressed + lossless + fast,
  all at once.

## Goal

Add a fast decode path to `DfloatCodec` (`crates/rtc-codec/src/dfloat.rs`) that
produces **bit-identical** output to the existing `decode()` but uses a buffered
64-bit-window bit-reader and a pre-allocated output buffer instead of the
per-bit `peek`/`advance` loop. Measure its decode throughput on the real Llama
1B bf16 embedding (525 MB) and report GO / MARGINAL / NO-GO against a
physics-derived threshold. **No codec format change. No runtime wiring. No fused
kernel.** Those are gated behind this measurement.

## Design

### Bottleneck (measured, R140a)

`BitReader::peek(n)` (`dfloat.rs:89`) computes each bit with a `byte = abs/8`,
`bit = 7-(abs%8)` — a div and a mod **per bit**, up to `MAX_CODE_LEN = 15` bits
per exponent symbol, for 262M symbols. The decode loop also does
`out.extend_from_slice(&bits.to_le_bytes())` per weight (a 2-byte push with a
capacity check each time). Both are the hot path.

### 1. `BufferedBitReader` (`crates/rtc-codec/src/bitreader_fast.rs`, new file)

A bit-reader that holds a 64-bit window so a peek is one shift + one mask:

- `bitbuf: u64` — up to 64 buffered bits, MSB-aligned (next bit at bit 63).
- `bitcnt: u32` — valid bits currently in `bitbuf`.
- `bytes: &[u8]`, `pos: usize` — refill cursor.
- `refill(&mut self)` — while `bitcnt <= 56`, pull the next byte and OR it into
  `bitbuf` at the right offset, advancing `bitcnt` by 8. Past end-of-input,
  refill with zero bits (matches the current reader's zero-padding contract so
  the final symbols decode identically).
- `peek(n) -> u32` — `(self.bitbuf >> (64 - n)) as u32` (n ≤ 15 ≤ 32, safe).
- `consume(n)` — `self.bitbuf <<= n; self.bitcnt -= n;`.

The decode loop calls `refill()` once per symbol (cheap branch when the window is
already full), `peek(max_len)`, indexes the existing `DecodeLut`, then
`consume(len)`. SWAR only (shift/mask on `u64`) — no intrinsics, portable, pure
Rust.

### 2. `DfloatCodec::decode_fast(encoded, meta) -> Result<Vec<u8>>`

Same header parse and `validate_lengths` + `build_decode_lut` as `decode()`
(reuse those verbatim — no duplication of the Huffman logic). Differences:

- Bit-read exponents via `BufferedBitReader` instead of `BitReader`.
- Pre-allocate `let mut out = vec![0u8; num_weights * 2];` and write each weight
  with `out[2*i] = lo; out[2*i+1] = hi;` (indexed writes, no per-element grow).
- Identical error handling (invalid Huffman code, truncation) so behavior matches
  on corrupt input too.

`decode()` stays unchanged as the reference. `decode_fast()` is additive.

### 3. Bit-identical parity test

A unit test that encodes several real-shaped inputs (skewed exponents, single
exponent, random bf16 via the existing xorshift pattern in
`dfloat_roundtrip_is_bit_exact`) and asserts `decode_fast(enc) == decode(enc)`
**byte-for-byte**. Lossless is non-negotiable; the fast path must not drift a
single bit. Also re-run against the corrupt-length-table input to confirm both
paths reject it.

### 4. `#[ignore]` feasibility bench

Mirror the existing `dfloat_feasibility_ratio_and_throughput` (`dfloat.rs:579`):
read `/tmp/rllm-bf16-sample.bin` (the real 525 MB Llama 1B bf16 embedding, dumped
by the existing `dump_bf16_embedding_sample` test in `lazy.rs:1227`), encode once,
then time `decode_fast()` over several warm iterations. Print:

- bits/weight + ratio (sanity — should match R140a's 10.6),
- **single-core decode throughput** in GB/s (bf16-out) AND **Gweight/s**,
- the speedup vs the R140a naive `decode()` (run both, same input),
- the GO/MARGINAL/NO-GO verdict computed against the threshold below.

### Threshold (physics-derived, from R141 measurements)

Two reference points from R141's bandwidth measurement of the bf16 LM head GEMV
(262.7M weights = 128256×2048, 525 MB at ~24 GB/s, bandwidth-bound):

- **Speed to beat:** the plain bf16 GEMV runs at **262.7M / ~22 ms ≈ 12
  Gweight/s** aggregate. Compressed-resident wins on speed when decode keeps up
  with this.
- **RAM-read floor:** compressed-resident reads only ~349 MB/token (10.6
  bits/weight) ≈ **14.5 ms** of RAM — the wall-clock floor. Hitting it needs
  decode ≥ **18.1 Gweight/s** aggregate (the 1.5× ceiling).

Why **aggregate decode vs the RAM read** is the right comparison: in a fused loop
the RAM read is bus-bound (~24 GB/s, shared across cores) while decode is compute
that parallelizes across cores — they genuinely overlap. So compressed-resident
wall-clock ≈ **max(14.5 ms RAM read, parallel-decode time)**, plus a small margin
for tile-boundary serialization. Decode ≥ 12 Gw/s aggregate finishes inside the
22 ms plain-bf16 budget (a win); ≥ 18 finishes inside the 14.5 ms RAM floor (the
full 1.5× win).

The bench measures **single-core** decode (the test runs on one P-core); the
verdict scales by the A18's effective parallelism — **~3.5 P-equivalent** (2 P + 4
slower E cores ≈ 2 + 4×0.4). This is a **proxy**: it isolates decode throughput,
not the real fused kernel. A GO means "build Phase 2 and measure the fused number
for real."

| Single-core decode | Aggregate (×3.5) | Verdict |
|---|---|---|
| ≥ ~3.4 Gweight/s | ≥ ~12 Gweight/s | 🟢 **GO** — compressed-resident is faster than plain bf16 AND smaller; build Phase 2 (≥18 agg = full 1.5× ceiling) |
| ~1.4–3.4 Gweight/s | ~5–12 Gweight/s | 🟡 **MARGINAL** — GEMV slightly slower than plain bf16, but RAM drops 525→349 MB; a real win when RAM is the binding constraint (the mission). Decide per goal |
| < ~1.4 Gweight/s | < ~5 Gweight/s | 🔴 **NO-GO** — decode dominates; keep the codec for storage, record the honest negative |

The verdict is reported honestly whichever way it lands; MARGINAL or NO-GO is a
valid, publishable result (it bounds the lossless-fast frontier on CPU).

## Implementation phases (de-risk inside one plan)

1. `BufferedBitReader` + unit test that it round-trips the same `(code, len)`
   streams the existing `BitReader` test uses (`bit_writer_reader_roundtrips_variable_codes`).
2. `decode_fast()` + the bit-identical parity test vs `decode()`.
3. The `#[ignore]` feasibility bench + run it on the real 525 MB sample; record
   single-core Gweight/s, the speedup vs naive, and the verdict.
4. Trial report (REEDRIP) with the measured number and GO/MARGINAL/NO-GO, placed
   in the folder matching the decision; index row added.

## Non-goals (explicitly gated behind a GO)

- Per-row / per-tile framing of the codec stream (a format change). Phase 2 proper.
- Fused decode→bfdot kernel. Phase 2 proper.
- Registering `DfloatCodec` in `codec_for_id` (`loader.rs:121`) for runtime use.
  Not needed to measure decode throughput; deferred to Phase 2 proper.
- Multi-threaded decode. R142 measures one core and extrapolates; real
  parallelism is Phase 2 proper.
- SIMD/NEON decode. The buffered SWAR reader is the cheap first lever; only if it
  lands MARGINAL-but-close do we consider SIMD in a follow-up.
- KV-cache compression. GPU. Sub-bf16 precision.

## Testing

- **Lossless parity (hard rule):** `decode_fast(enc) == decode(enc)` byte-for-byte
  on skewed, single-exponent, and random bf16 inputs. A single differing byte
  fails the gate.
- **Reader unit test:** `BufferedBitReader` reproduces the existing `BitReader`
  variable-code round-trip exactly.
- **Corrupt input:** both paths reject the corrupt-length-table payload.
- **Honest metrics:** single-core Gweight/s + GB/s on the real 525 MB sample, the
  speedup vs the R140a naive decoder, and the verdict vs the physics threshold —
  all in the trial report, including a MARGINAL/NO-GO outcome stated plainly.
- Existing `rtc-codec` tests stay green (the fast path is additive).

## Originality & dependencies (doctrine)

- **Original code.** A buffered 64-bit-window bit-reader is a standard entropy-
  decoding technique; this implementation is written from scratch in this repo.
  No external decompression library, no port of any runtime.
- **No new dependencies.** SWAR on `u64` via the standard library only.
  `cargo build` stays the only requirement.
- **Lossless by default** is preserved end to end: only the decode mechanism
  changes, output is bit-identical, proven by the parity test.

## Components / isolation

- `BufferedBitReader` (`bitreader_fast.rs`) — standalone, unit-tested against the
  existing reader's contract.
- `DfloatCodec::decode_fast` — additive method, reuses the existing header parse,
  `validate_lengths`, and `build_decode_lut`; isolated from `decode()`.
- Feasibility bench — `#[ignore]`, reads the real sample, prints the verdict.
- Phase 2 codec/kernel work — separate spec, gated on this result.

## Prior art (cited honestly)

Buffered/table-driven Huffman decoding is well established (zlib, Huff0/FSE in
Zstd). RLLM's distinct position is the combination this gate is testing: a
lossless bf16 codec decoded fast enough to feed an exact-weight bf16 GEMV on CPU,
toward compressed-resident inference — the open CPU gap from the R140/R141 work.
