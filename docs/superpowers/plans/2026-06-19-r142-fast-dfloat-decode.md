# R142 — Fast `rtc-dfloat-v1` decode (feasibility gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a buffered-bit-reader fast decode path to `DfloatCodec` that is bit-identical to the existing `decode()`, then measure its single-core throughput on the real 525 MB Llama 1B bf16 embedding and report a GO / MARGINAL / NO-GO verdict against the R141-derived threshold.

**Architecture:** A new `BufferedBitReader` (64-bit MSB-aligned window, SWAR shift/mask) replaces the per-bit `peek`/`advance` of the original reader. A new additive method `DfloatCodec::decode_fast` reuses the existing header parse (extracted to a shared helper), `validate_lengths`, and `build_decode_lut`, but reads exponents through the buffered reader and writes the output into a pre-allocated buffer by index. `decode()` stays as the bit-identical reference; an `#[ignore]` bench measures `decode_fast` on the real embedding.

**Tech Stack:** Rust (stable), crate `rtc-codec`, no new dependencies. Standard-library SWAR on `u64`. Tests via `cargo test -p rtc-codec`.

## Global Constraints

- Pure Rust, **no new dependencies**; `cargo build` stays the only requirement. SWAR only — no SIMD/intrinsics in this gate.
- **Lossless / bit-identical (hard rule):** `decode_fast(enc)` must equal `decode(enc)` byte-for-byte. A single differing byte fails the gate.
- **No codec format change.** The on-disk/encoded layout `[num_weights:u64][lengths:256][exp_len:u64][exp_stream][residuals]` is unchanged. Encode is untouched.
- **No runtime wiring, no fused kernel, no per-row framing.** Do NOT register `DfloatCodec` in `codec_for_id` (`crates/rllm-runtime/src/loader.rs:121`). Those are Phase-2-proper, gated behind this measurement.
- **REE kernel working name: REEDRIP** (Erik's final call before any paper/report use) — use it in the trial report Scope line.
- **Honest metrics:** report single-core Gweight/s, the speedup vs the naive decoder, and the verdict — including a MARGINAL or NO-GO outcome stated plainly.
- Existing `rtc-codec` tests must stay green (the fast path is additive; the header-parse extraction is behavior-preserving).

## File Structure

- **Create** `crates/rtc-codec/src/bitreader_fast.rs` — `BufferedBitReader` (64-bit window) + its unit test. One responsibility: fast MSB-first bit reading.
- **Modify** `crates/rtc-codec/src/lib.rs` — add `mod bitreader_fast;` (private module, one line).
- **Modify** `crates/rtc-codec/src/dfloat.rs` — extract `parse_chunk` helper (used by `decode` and `decode_fast`), add `DfloatCodec::decode_fast`, add parity/tail/corrupt tests + the `#[ignore]` feasibility bench.
- **Create** `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r142-reedrip-fast-dfloat-decode.md` — trial report.
- **Modify** `docs/benchmarks/trials/index.md` — add the R142 row.
- **Modify** memory `rllm-speed-thesis-streaming-vs-resident.md` — record the measured R142 number.

---

### Task 1: `BufferedBitReader` (64-bit window)

**Files:**
- Create: `crates/rtc-codec/src/bitreader_fast.rs`
- Modify: `crates/rtc-codec/src/lib.rs` (add `mod bitreader_fast;`)
- Test: in `crates/rtc-codec/src/bitreader_fast.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `crate::dfloat::BitWriter` (pub, in `dfloat.rs`) for the round-trip test only.
- Produces: `pub struct BufferedBitReader<'a>` with `pub fn new(bytes: &'a [u8]) -> Self`, `pub fn refill(&mut self)`, `pub fn peek(&self, n: u8) -> u32`, `pub fn consume(&mut self, n: u8)`. Contract: bits consumed MSB-first; reads past end yield zero bits (matches the original `BitReader`); after `refill()` at least 57 valid bits are buffered, so any `peek(n)` with `n <= 32` is valid.

- [ ] **Step 1: Add the module declaration**

In `crates/rtc-codec/src/lib.rs`, add to the module block (the existing block is `mod codec; mod dfloat; mod error; mod huff; mod raw; mod rle;`):

```rust
mod bitreader_fast;
```

- [ ] **Step 2: Write the failing test**

Create `crates/rtc-codec/src/bitreader_fast.rs` with ONLY the test module first (so it fails to compile → the strongest "fails first"):

```rust
//! Buffered bit-reader for fast canonical-Huffman decode.
//!
//! The original `BitReader` (in `dfloat.rs`) computes each bit with a byte
//! index + bit index — a div and a mod per bit, up to `MAX_CODE_LEN` bits per
//! symbol. For 262M symbols that is tens of seconds (it dominated R140a's
//! 0.02 GB/s decode). This reader keeps a 64-bit MSB-aligned window, so a peek
//! is one shift and a symbol lookup costs one shift + one mask.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfloat::BitWriter;

    #[test]
    fn buffered_reader_matches_bitwriter_stream() {
        // Same (code, len) pairs the original BitReader test uses, MSB-first.
        let symbols = [(0b1u32, 1u8), (0b01, 2), (0b001, 3), (0b0, 1), (0b101, 3)];
        let mut w = BitWriter::new();
        for &(c, l) in &symbols {
            w.write(c, l);
        }
        let bytes = w.finish();
        let mut r = BufferedBitReader::new(&bytes);
        for &(c, l) in &symbols {
            r.refill();
            assert_eq!(r.peek(l), c, "peek mismatch for {c:#b}/{l}");
            r.consume(l);
        }
    }

    #[test]
    fn buffered_reader_zero_pads_past_end() {
        // One byte 0b1010_0000; after consuming the 3 real-ish bits the reader
        // must keep returning zeros (no panic, no garbage).
        let bytes = [0b1010_0000u8];
        let mut r = BufferedBitReader::new(&bytes);
        r.refill();
        assert_eq!(r.peek(4), 0b1010);
        r.consume(8);
        r.refill();
        assert_eq!(r.peek(8), 0, "past end-of-input must read as zeros");
    }
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p rtc-codec --lib buffered_reader -- --nocapture`
Expected: FAIL — compile error, `cannot find type BufferedBitReader in this scope`.

- [ ] **Step 4: Implement `BufferedBitReader`**

Insert this ABOVE the `#[cfg(test)]` module in `crates/rtc-codec/src/bitreader_fast.rs`:

```rust
/// A bit-reader over a byte slice that buffers up to 64 bits in a register-wide
/// window. Bits are consumed most-significant-first, matching `BitWriter`'s
/// MSB-first output and the canonical-Huffman code layout. Reads past the end of
/// the input yield zero bits (the same zero-padding contract as the original
/// `BitReader`), so the final symbols decode identically.
pub struct BufferedBitReader<'a> {
    bytes: &'a [u8],
    pos: usize,   // index of the next byte to pull into the window
    bitbuf: u64,  // buffered bits, next bit to read at the MSB (bit 63)
    bitcnt: u32,  // number of valid bits currently in `bitbuf` (0..=64)
}

impl<'a> BufferedBitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        let mut r = Self { bytes, pos: 0, bitbuf: 0, bitcnt: 0 };
        r.refill();
        r
    }

    /// Pull bytes into the window until it holds more than 56 bits, i.e. at least
    /// 57 valid bits — enough for any code up to 32 bits. Past end-of-input no
    /// real bytes remain, so zero bytes are folded in: the low bits stay zero,
    /// reproducing the original reader's zero-padding. `bitcnt` never exceeds 64.
    #[inline]
    pub fn refill(&mut self) {
        while self.bitcnt <= 56 {
            let byte = if self.pos < self.bytes.len() {
                let b = self.bytes[self.pos];
                self.pos += 1;
                b
            } else {
                0
            };
            // Place the new byte just below the bits already buffered.
            self.bitbuf |= (byte as u64) << (56 - self.bitcnt);
            self.bitcnt += 8;
        }
    }

    /// Peek the next `n` bits (1..=32) as an integer, MSB-first. Caller must have
    /// called `refill()` so at least `n` bits are buffered.
    #[inline]
    pub fn peek(&self, n: u8) -> u32 {
        debug_assert!((1..=32).contains(&n));
        (self.bitbuf >> (64 - n as u32)) as u32
    }

    /// Advance past `n` consumed bits (`n <= bitcnt`, `n <= 32`).
    #[inline]
    pub fn consume(&mut self, n: u8) {
        debug_assert!(n as u32 <= self.bitcnt);
        self.bitbuf <<= n as u32;
        self.bitcnt -= n as u32;
    }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p rtc-codec --lib buffered_reader -- --nocapture`
Expected: PASS — both `buffered_reader_matches_bitwriter_stream` and `buffered_reader_zero_pads_past_end`.

- [ ] **Step 6: Verify the whole crate still builds and tests green**

Run: `cargo test -p rtc-codec`
Expected: PASS — all existing tests plus the 2 new ones; 0 failures.

- [ ] **Step 7: Commit**

```bash
git add crates/rtc-codec/src/bitreader_fast.rs crates/rtc-codec/src/lib.rs
git commit -m "feat(rtc-codec): BufferedBitReader 64-bit-window reader (R142 REEDRIP)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `decode_fast` + bit-identical parity

**Files:**
- Modify: `crates/rtc-codec/src/dfloat.rs` (extract `parse_chunk`, refactor `decode` to use it, add `decode_fast`, add tests)
- Test: in `crates/rtc-codec/src/dfloat.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `BufferedBitReader` (Task 1); existing `validate_lengths`, `build_decode_lut`, `join_bf16`, `DecodeLut` in `dfloat.rs`.
- Produces: `fn parse_chunk(encoded: &[u8]) -> Result<(usize, [u8; 256], &[u8], &[u8])>` (private; returns `(num_weights, lengths, exp_stream, residuals)`); `impl DfloatCodec { pub fn decode_fast(&self, encoded: &[u8]) -> Result<Vec<u8>> }`. `decode_fast` ignores `DecodeMeta` exactly as `decode` does (size comes from the chunk header).

- [ ] **Step 1: Write the failing parity test**

Add to the `#[cfg(test)] mod tests` in `crates/rtc-codec/src/dfloat.rs`:

```rust
#[test]
fn decode_fast_matches_decode_bit_for_bit() {
    use crate::{DecodeMeta, EncodeMeta, TensorCodec};
    let codec = DfloatCodec;
    let dmeta = DecodeMeta { codec_id: "rtc-dfloat-v1".into(), uncompressed_size: 0 };

    let mut inputs: Vec<Vec<u8>> = Vec::new();
    // (a) skewed exponents
    {
        let mut b = Vec::new();
        for i in 0..4096u16 {
            let exp: u16 = if i % 8 == 0 { 0x40 } else { 0x3F };
            let bits = (((i >> 6) & 1) << 15) | (exp << 7) | (i & 0x7F);
            b.extend_from_slice(&bits.to_le_bytes());
        }
        inputs.push(b);
    }
    // (b) single exponent
    {
        let mut b = Vec::new();
        for i in 0..512u16 {
            let bits = (((i >> 7) & 1) << 15) | (0x40u16 << 7) | (i & 0x7F);
            b.extend_from_slice(&bits.to_le_bytes());
        }
        inputs.push(b);
    }
    // (c) full-entropy random bf16 (xorshift over the full 16-bit space)
    {
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut b = Vec::new();
        for _ in 0..8192 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bits = (state >> 32) as u16;
            b.extend_from_slice(&bits.to_le_bytes());
        }
        inputs.push(b);
    }

    for (k, bytes) in inputs.iter().enumerate() {
        let meta = EncodeMeta {
            name: "w".into(),
            shape: vec![(bytes.len() / 2) as u64],
            dtype: "bf16".into(),
        };
        let enc = codec.encode(bytes, &meta).unwrap();
        let slow = codec.decode(&enc.data, &dmeta).unwrap();
        let fast = codec.decode_fast(&enc.data).unwrap();
        assert_eq!(&slow, bytes, "case {k}: slow decode must roundtrip");
        assert_eq!(fast, slow, "case {k}: decode_fast must equal decode byte-for-byte");
    }
}

#[test]
fn decode_fast_matches_decode_on_tail_boundaries() {
    use crate::{DecodeMeta, EncodeMeta, TensorCodec};
    let codec = DfloatCodec;
    let dmeta = DecodeMeta { codec_id: "rtc-dfloat-v1".into(), uncompressed_size: 0 };
    for n in [0usize, 1, 2, 3, 5, 7, 9, 15, 17, 31, 33] {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut bytes = Vec::new();
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.extend_from_slice(&((state >> 48) as u16).to_le_bytes());
        }
        let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &meta).unwrap();
        let slow = codec.decode(&enc.data, &dmeta).unwrap();
        let fast = codec.decode_fast(&enc.data).unwrap();
        assert_eq!(fast, slow, "n={n}: decode_fast must equal decode at tail boundary");
        assert_eq!(fast, bytes, "n={n}: decode_fast must be lossless");
    }
}

#[test]
fn decode_fast_rejects_invalid_length_table() {
    use crate::{EncodeMeta, TensorCodec};
    let bits: u16 = 0x3F80;
    let bytes: Vec<u8> = (0..8).flat_map(|_| bits.to_le_bytes()).collect();
    let codec = DfloatCodec;
    let emeta = EncodeMeta { name: "w".into(), shape: vec![8], dtype: "bf16".into() };
    let enc = codec.encode(&bytes, &emeta).unwrap();
    let mut corrupt = enc.data.clone();
    corrupt[8] = 30; // length > MAX_CODE_LEN (15) → must be rejected
    assert!(codec.decode_fast(&corrupt).is_err(), "decode_fast must reject corrupt length table");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rtc-codec --lib decode_fast -- --nocapture`
Expected: FAIL — compile error, `no method named decode_fast found for struct DfloatCodec`.

- [ ] **Step 3: Add the import and the shared `parse_chunk` helper**

At the top of `crates/rtc-codec/src/dfloat.rs`, below the existing `use` lines, add:

```rust
use crate::bitreader_fast::BufferedBitReader;
```

Then add this private free function (place it just above `pub struct DfloatCodec;`):

```rust
/// Parse a rtc-dfloat-v1 chunk into its sections, shared by `decode` and
/// `decode_fast` so the framing logic lives in exactly one place. Returns
/// `(num_weights, lengths, exp_stream, residuals)`.
fn parse_chunk(encoded: &[u8]) -> Result<(usize, [u8; 256], &[u8], &[u8])> {
    let err = || CodecError::InvalidData("truncated rtc-dfloat-v1 chunk".to_string());
    if encoded.len() < 8 + 256 + 8 {
        return Err(err());
    }
    let num_weights = u64::from_le_bytes(encoded[0..8].try_into().map_err(|_| err())?) as usize;
    let mut lengths = [0u8; 256];
    lengths.copy_from_slice(&encoded[8..8 + 256]);
    let exp_len = u64::from_le_bytes(encoded[264..272].try_into().map_err(|_| err())?) as usize;
    let exp_start: usize = 272;
    let exp_end = exp_start.checked_add(exp_len).ok_or_else(err)?;
    let res_end = exp_end.checked_add(num_weights).ok_or_else(err)?;
    if encoded.len() < res_end {
        return Err(err());
    }
    Ok((num_weights, lengths, &encoded[exp_start..exp_end], &encoded[exp_end..res_end]))
}
```

- [ ] **Step 4: Refactor `decode` to use `parse_chunk` (behavior-preserving)**

Replace the body of `fn decode(...)` (currently the header-parsing block at `dfloat.rs:332-369`) so its head uses the helper. The new `decode` body:

```rust
fn decode(&self, encoded: &[u8], _meta: &crate::codec::DecodeMeta) -> Result<Vec<u8>> {
    let (num_weights, lengths, exp_stream, residuals) = parse_chunk(encoded)?;
    validate_lengths(&lengths)?;
    let lut = build_decode_lut(&lengths);
    let mut reader = BitReader::new(exp_stream);
    let mut out = Vec::with_capacity(num_weights * 2);
    for &res in residuals.iter() {
        let window = reader.peek(lut.max_len);
        let (exp, len) = lut.entries[window as usize];
        if len == 0 {
            return Err(CodecError::InvalidData(
                "rtc-dfloat-v1: invalid Huffman code in exponent stream".into(),
            ));
        }
        reader.advance(len);
        let bits = join_bf16(exp, res);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(out)
}
```

- [ ] **Step 5: Add the `decode_fast` inherent method**

Add a new `impl` block just below the `impl TensorCodec for DfloatCodec` block:

```rust
impl DfloatCodec {
    /// Fast decode: identical output to [`TensorCodec::decode`], but reads
    /// exponents through the buffered 64-bit-window reader and writes the output
    /// by index into a pre-allocated buffer (no per-bit div/mod, no per-element
    /// push). Bit-identical and lossless; proven by
    /// `decode_fast_matches_decode_bit_for_bit`. Additive and not yet wired into
    /// the runtime — this is the R142 feasibility building block.
    pub fn decode_fast(&self, encoded: &[u8]) -> Result<Vec<u8>> {
        let (num_weights, lengths, exp_stream, residuals) = parse_chunk(encoded)?;
        validate_lengths(&lengths)?;
        let lut = build_decode_lut(&lengths);
        let max_len = lut.max_len;
        let mut reader = BufferedBitReader::new(exp_stream);
        let mut out = vec![0u8; num_weights * 2];
        for (i, &res) in residuals.iter().enumerate() {
            reader.refill();
            let window = reader.peek(max_len);
            let (exp, len) = lut.entries[window as usize];
            if len == 0 {
                return Err(CodecError::InvalidData(
                    "rtc-dfloat-v1: invalid Huffman code in exponent stream".into(),
                ));
            }
            reader.consume(len);
            let bits = join_bf16(exp, res);
            let le = bits.to_le_bytes();
            out[2 * i] = le[0];
            out[2 * i + 1] = le[1];
        }
        Ok(out)
    }
}
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `cargo test -p rtc-codec --lib decode_fast -- --nocapture`
Expected: PASS — `decode_fast_matches_decode_bit_for_bit`, `decode_fast_matches_decode_on_tail_boundaries`, `decode_fast_rejects_invalid_length_table`.

- [ ] **Step 7: Run the full crate suite (guards the `decode` refactor)**

Run: `cargo test -p rtc-codec`
Expected: PASS — every existing dfloat test (incl. `dfloat_roundtrip_is_bit_exact`, `dfloat_decode_rejects_invalid_length_table`) stays green, proving the `parse_chunk` extraction did not change `decode`'s behavior.

- [ ] **Step 8: Commit**

```bash
git add crates/rtc-codec/src/dfloat.rs
git commit -m "feat(rtc-codec): decode_fast bit-identical buffered decode (R142 REEDRIP)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Feasibility bench + measurement

**Files:**
- Modify: `crates/rtc-codec/src/dfloat.rs` (add `#[ignore]` bench)

**Interfaces:**
- Consumes: `DfloatCodec::decode_fast` (Task 2), the existing `dump_bf16_embedding_sample` test (`crates/rllm-runtime/src/lazy.rs:1227`) which writes `/tmp/rllm-bf16-sample.bin` from `models/Llama-3.2-1B-Instruct-raw.spsa`.
- Produces: the measured single-core Gweight/s, the speedup vs naive, and the GO/MARGINAL/NO-GO verdict (printed; transcribed into the Task 4 trial report).

- [ ] **Step 1: Add the `#[ignore]` feasibility bench**

Add to the `#[cfg(test)] mod tests` in `crates/rtc-codec/src/dfloat.rs`:

```rust
#[test]
#[ignore]
fn dfloat_fast_decode_feasibility() {
    use crate::{DecodeMeta, EncodeMeta, TensorCodec};
    let bytes = std::fs::read("/tmp/rllm-bf16-sample.bin")
        .expect("run dump_bf16_embedding_sample first (see plan Task 3, Step 2)");
    let num_weights = bytes.len() / 2;
    let codec = DfloatCodec;
    let emeta =
        EncodeMeta { name: "embed".into(), shape: vec![num_weights as u64], dtype: "bf16".into() };
    let enc = codec.encode(&bytes, &emeta).unwrap();
    let bits_per_weight = (enc.data.len() as f64 * 8.0) / num_weights as f64;
    let dmeta = DecodeMeta { codec_id: "rtc-dfloat-v1".into(), uncompressed_size: bytes.len() as u64 };

    // Correctness on the real sample before timing.
    let fast = codec.decode_fast(&enc.data).unwrap();
    assert_eq!(fast, bytes, "decode_fast must be lossless on the real embedding");

    // Warm, timed fast decode.
    let iters = 5;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let d = codec.decode_fast(&enc.data).unwrap();
        std::hint::black_box(&d);
    }
    let fast_s = t.elapsed().as_secs_f64() / iters as f64;
    let fast_gw = (num_weights as f64 / 1e9) / fast_s; // Gweight/s
    let fast_gbps = (bytes.len() as f64 / 1e9) / fast_s; // GB/s bf16-out

    // One pass of the naive decoder for the speedup ratio (it is ~26 s).
    let t = std::time::Instant::now();
    let slow = codec.decode(&enc.data, &dmeta).unwrap();
    std::hint::black_box(&slow);
    let slow_s = t.elapsed().as_secs_f64();
    let slow_gw = (num_weights as f64 / 1e9) / slow_s;

    let agg = fast_gw * 3.5; // A18: 2 P + 4 E ≈ 3.5 P-equivalent
    let verdict = if agg >= 12.0 {
        "GO"
    } else if agg >= 5.0 {
        "MARGINAL"
    } else {
        "NO-GO"
    };

    eprintln!(
        "\n=== R142 REEDRIP fast-decode FEASIBILITY ===\n\
         weights={num_weights}  bits/weight={bits_per_weight:.3}\n\
         fast single-core: {fast_gw:.2} Gweight/s  ({fast_gbps:.2} GB/s bf16-out, {:.1} ms/decode)\n\
         naive single-core: {slow_gw:.4} Gweight/s  (speedup {:.0}x)\n\
         aggregate (x3.5): {agg:.1} Gweight/s\n\
         threshold: GO>=12, MARGINAL 5-12, NO-GO<5 (Gweight/s aggregate)\n\
         VERDICT: {verdict}\n",
        fast_s * 1000.0,
        fast_gw / slow_gw,
    );
}
```

- [ ] **Step 2: Generate the real bf16 sample**

Run (needs the local `models/Llama-3.2-1B-Instruct-raw.spsa`, 2.3 GB, already on disk):

```bash
cargo test -p rllm-runtime --release dump_bf16_embedding_sample -- --ignored --nocapture
```
Expected: prints `wrote 525336576 bf16 bytes to /tmp/rllm-bf16-sample.bin`.

- [ ] **Step 3: Run the feasibility bench (release) and capture the verdict**

Run:

```bash
cargo test -p rtc-codec --release dfloat_fast_decode_feasibility -- --ignored --nocapture
```
Expected: the `=== R142 REEDRIP fast-decode FEASIBILITY ===` block prints. **Record verbatim**: `bits/weight` (~10.6 sanity-check), fast `Gweight/s` + `GB/s` + `ms/decode`, the `speedup` vs naive, the `aggregate`, and the `VERDICT`. These numbers are the deliverable; they feed Task 4.

- [ ] **Step 4: Commit**

```bash
git add crates/rtc-codec/src/dfloat.rs
git commit -m "test(rtc-codec): R142 REEDRIP fast-decode feasibility bench + measurement

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Trial report + index + memory

**Files:**
- Create: `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r142-reedrip-fast-dfloat-decode.md`
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md`

**Interfaces:**
- Consumes: the measured numbers + verdict from Task 3 Step 3, and the template `docs/benchmarks/templates/trial-report.md`.
- Produces: a trial report in the folder matching the verdict (`success/` for GO, `inconclusive/` for MARGINAL, `failed/` for NO-GO), an index row, and a one-line memory update.

- [ ] **Step 1: Read the template and a recent example**

Run: `cat docs/benchmarks/templates/trial-report.md docs/benchmarks/trials/inconclusive/2026-06-19-r141-reeflow-bf16-dot.md`
Expected: see the required section structure (`# Trial: …`, `Date`/`Owner`/`Status`/`Folder` block, `## Hypothesis`, `## Scope`, `## Setup`, `## Results`, `## Analysis`, `## Decision`, `## Next Experiment`).

- [ ] **Step 2: Write the trial report**

Create `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r142-reedrip-fast-dfloat-decode.md` following that structure. Required content, filled from Task 3's measured numbers:
- **Scope → REE kernel:** `REEDRIP (working name; Erik's final call before any paper use)`. Mode: `compressed-resident feasibility (codec decode only)`. Artifact: `Llama-3.2-1B-Instruct-raw.spsa` embedding (262.7M bf16). Device: Apple A18 Pro (2 P + 4 E). Bottleneck tag: `codec decode throughput vs memory bandwidth`.
- **Hypothesis:** a buffered 64-bit-window reader makes `rtc-dfloat-v1` decode fast enough that compressed-resident (read ~10.6 bits/weight, decode on the fly) beats reading 16-bit bf16 from RAM.
- **Results:** the table — naive vs fast single-core Gweight/s + the speedup; fast GB/s + ms/decode; aggregate (×3.5); the GO/MARGINAL/NO-GO verdict; bit-identical parity confirmed (all `decode_fast` tests green).
- **Analysis:** compare aggregate decode to the 12 Gweight/s plain-bf16 rate and the 18.1 ceiling; state honestly whether decode clears the bandwidth budget. If MARGINAL, note the RAM win (525→349 MB resident) stands regardless.
- **Decision:** the verdict, with the gate for Phase-2-proper (per-row framing + fused decode→bfdot) being GO or close-MARGINAL.
- **Next Experiment:** if GO/close — Phase 2 proper (per-row framing + fused decode→bfdot, registering `DfloatCodec` in `codec_for_id`); if NO-GO — codec stays storage-only, consider SIMD decode as a separate lever.

- [ ] **Step 3: Add the index row**

In `docs/benchmarks/trials/index.md`, add a row for R142 mirroring the R141 row's columns (date | trial | folder | model | mode | bottleneck tag | baseline | result | decision | paper value). Baseline = R140a naive 0.02 GB/s; result = the measured fast Gweight/s + verdict.

- [ ] **Step 4: Update memory**

Append the measured R142 number to `rllm-speed-thesis-streaming-vs-resident.md` (honest one-liner: buffered reader lifted dfloat decode from 0.01 Gweight/s to `<measured>` single-core / `<agg>` aggregate → `<verdict>`; the bytes-read lever from R141 is now quantified on the decode side).

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/trials/ "/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md"
git commit -m "docs(bench): R142 REEDRIP fast-decode trial (<verdict>) + index + memory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Verification (end-to-end)

1. `cargo test -p rtc-codec` → all green, including the 2 reader tests and the 3 `decode_fast` tests (bit-identical parity, tail boundaries, corrupt rejection).
2. `cargo build` (workspace) → compiles with no new dependencies (doctrine check).
3. The `#[ignore]` bench printed the `=== R142 REEDRIP fast-decode FEASIBILITY ===` block with a real verdict on the 525 MB sample.
4. Trial report exists in the folder matching the verdict, with all required sections and the measured numbers; `docs/benchmarks/trials/index.md` has the R142 row; memory updated.
5. `git grep -n "decode_fast" crates/rllm-runtime crates/rllm-cli` → **no hits** (confirms no runtime wiring happened; the gate stayed isolated).

## Out of scope (gated behind a GO — do NOT build here)

- Per-row / per-tile framing of the codec stream (format change). Phase 2 proper.
- Fused decode→bfdot kernel; registering `DfloatCodec` in `codec_for_id`. Phase 2 proper.
- Multi-threaded decode; SIMD/NEON decode. (×3.5 is an extrapolation, not a build.)
- KV-cache compression, GPU, sub-bf16 precision.
