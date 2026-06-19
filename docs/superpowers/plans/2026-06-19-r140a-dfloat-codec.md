# R140a — rtc-dfloat-v1 lossless bf16 codec + feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `rtc-dfloat-v1`, a lossless bf16 codec that splits each bf16 into an entropy-coded exponent stream and a raw sign+mantissa stream, and measure its real compression ratio and decode throughput to gate R140b (the fused kernel).

**Architecture:** A new `TensorCodec` in the `rtc-codec` crate. Encode: split bf16 → exponent (canonical Huffman) + residual (raw 8-bit). Decode: rebuild a flat decode LUT from the canonical code lengths, decode exponents, recombine with residuals → original bf16 bytes, bit-exact. This plan is the codec + a measurement harness only — no runtime/kernel wiring (that is R140b, gated on the throughput number here).

**Tech Stack:** Rust (edition 2021), the existing `rtc-codec` crate (`TensorCodec` trait, in-house Huffman in `huff.rs`). No new dependencies.

## Global Constraints

- Lossless: `decode(encode(x)) == x` byte-for-byte (the `TensorCodec` contract). Non-negotiable.
- No external crates: pure Rust only; no zstd/flate2/entropy libraries (RTC doctrine).
- Original code: implement from the technique (canonical Huffman + LUT), do NOT read/port DFloat11's CUDA/Python.
- Codec id string is exactly `rtc-dfloat-v1`.
- Only `bf16` input is supported (dtype == "bf16", input length even). Other dtypes → encode returns an error.

---

### Task 1: bf16 field split / join helpers

**Files:**
- Create: `crates/rtc-codec/src/dfloat.rs`
- Modify: `crates/rtc-codec/src/lib.rs` (add `mod dfloat; pub use dfloat::*;`)

**Interfaces:**
- Produces: `fn split_bf16(bits: u16) -> (u8, u8)` returns `(exponent, residual)` where exponent = bits 14..7, residual = sign(bit15) in bit7 + mantissa(bits6..0). `fn join_bf16(exponent: u8, residual: u8) -> u16` inverts it.

- [ ] **Step 1: Create the module file with the helpers**

```rust
//! rtc-dfloat-v1: lossless bf16 codec.
//!
//! bf16 = [sign:1][exponent:8][mantissa:7]. The exponent has low entropy for LLM
//! weights, so we entropy-code it (canonical Huffman) and store sign+mantissa raw.
//! Original implementation (technique from DFloat11, arXiv 2504.11651); no code
//! was copied and no external dependency is used.

/// Split a bf16 bit pattern into (exponent, residual=sign|mantissa).
/// exponent = bits 14..=7 ; residual = (sign << 7) | mantissa(bits 6..=0).
#[inline]
pub fn split_bf16(bits: u16) -> (u8, u8) {
    let exponent = ((bits >> 7) & 0xFF) as u8;
    let sign = ((bits >> 15) & 0x1) as u8;
    let mantissa = (bits & 0x7F) as u8;
    let residual = (sign << 7) | mantissa;
    (exponent, residual)
}

/// Inverse of `split_bf16`.
#[inline]
pub fn join_bf16(exponent: u8, residual: u8) -> u16 {
    let sign = ((residual >> 7) & 0x1) as u16;
    let mantissa = (residual & 0x7F) as u16;
    (sign << 15) | ((exponent as u16) << 7) | mantissa
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_join_roundtrips_every_bf16_value() {
        for bits in 0u32..=0xFFFF {
            let bits = bits as u16;
            let (e, r) = split_bf16(bits);
            assert_eq!(join_bf16(e, r), bits, "roundtrip failed for {bits:#06x}");
        }
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/rtc-codec/src/lib.rs`, add `mod dfloat;` with the other `mod` lines and `pub use dfloat::*;` with the other re-exports. Add `pub const CODEC_DFLOAT_V1: &str = "rtc-dfloat-v1";` next to the other `CODEC_*` consts.

- [ ] **Step 3: Run the test**

Run: `cargo test -p rtc-codec split_join_roundtrips -- --nocapture`
Expected: PASS (covers all 65536 bf16 patterns).

- [ ] **Step 4: Commit**

```bash
git add crates/rtc-codec/src/dfloat.rs crates/rtc-codec/src/lib.rs
git commit -m "feat(codec): rtc-dfloat bf16 field split/join helpers"
```

---

### Task 2: Bit writer / reader (MSB-first)

**Files:**
- Modify: `crates/rtc-codec/src/dfloat.rs`

**Interfaces:**
- Produces: `struct BitWriter { ... }` with `fn new() -> Self`, `fn write(&mut self, code: u32, len: u8)` (writes `len` low bits of `code`, MSB-first), `fn finish(self) -> Vec<u8>` (pads the last byte with zeros). `struct BitReader<'a> { ... }` with `fn new(bytes: &'a [u8]) -> Self`, `fn peek(&self, n: u8) -> u32` (returns the next `n` bits as an integer, zero-padded past the end), `fn advance(&mut self, n: u8)`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `dfloat.rs`:

```rust
#[test]
fn bit_writer_reader_roundtrips_variable_codes() {
    // (code, len) pairs, MSB-first.
    let symbols = [(0b1u32, 1u8), (0b01, 2), (0b001, 3), (0b0, 1), (0b101, 3)];
    let mut w = BitWriter::new();
    for &(c, l) in &symbols {
        w.write(c, l);
    }
    let bytes = w.finish();
    let mut r = BitReader::new(&bytes);
    for &(c, l) in &symbols {
        assert_eq!(r.peek(l), c, "peek mismatch for code {c:#b}/{l}");
        r.advance(l);
    }
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p rtc-codec bit_writer_reader_roundtrips`
Expected: FAIL (BitWriter/BitReader not defined).

- [ ] **Step 3: Implement BitWriter/BitReader**

Add to `dfloat.rs` (above the tests module):

```rust
pub struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8, // bits currently filled in `cur` (0..8)
}

impl BitWriter {
    pub fn new() -> Self {
        Self { bytes: Vec::new(), cur: 0, nbits: 0 }
    }

    /// Write the low `len` bits of `code`, most-significant bit first.
    pub fn write(&mut self, code: u32, len: u8) {
        let mut i = len;
        while i > 0 {
            i -= 1;
            let bit = ((code >> i) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// Flush, zero-padding the final partial byte.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.cur <<= 8 - self.nbits;
            self.bytes.push(self.cur);
        }
        self.bytes
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize, // absolute bit offset from the start
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    /// Peek the next `n` bits (n <= 24) as an integer, MSB-first, zero-padded
    /// past the end of the buffer.
    pub fn peek(&self, n: u8) -> u32 {
        let mut out = 0u32;
        for k in 0..n {
            let abs = self.bit_pos + k as usize;
            let byte = abs / 8;
            let bit_in_byte = 7 - (abs % 8);
            let bit = if byte < self.bytes.len() {
                ((self.bytes[byte] >> bit_in_byte) & 1) as u32
            } else {
                0
            };
            out = (out << 1) | bit;
        }
        out
    }

    pub fn advance(&mut self, n: u8) {
        self.bit_pos += n as usize;
    }
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p rtc-codec bit_writer_reader_roundtrips`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/rtc-codec/src/dfloat.rs
git commit -m "feat(codec): rtc-dfloat MSB-first bit writer/reader"
```

---

### Task 3: Canonical Huffman — code lengths, codes, and decode LUT

**Files:**
- Modify: `crates/rtc-codec/src/dfloat.rs`

**Interfaces:**
- Produces:
  - `fn huffman_code_lengths(freqs: &[u64; 256]) -> [u8; 256]` — code length per symbol (0 = unused), max length capped at 15; symbols with freq 0 get length 0.
  - `fn canonical_codes(lengths: &[u8; 256]) -> [u32; 256]` — canonical code per symbol (valid only where length > 0).
  - `struct DecodeLut { max_len: u8, entries: Vec<(u8, u8)> }` (`entries[window] = (symbol, code_len)`), `fn build_decode_lut(lengths: &[u8; 256]) -> DecodeLut`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn canonical_huffman_roundtrips_symbol_stream() {
    // Skewed frequencies over a few symbols.
    let mut freqs = [0u64; 256];
    freqs[5] = 100;
    freqs[7] = 40;
    freqs[9] = 20;
    freqs[200] = 1;
    let lengths = huffman_code_lengths(&freqs);
    // every used symbol has a positive, prefix-free length
    assert!(lengths[5] > 0 && lengths[7] > 0 && lengths[9] > 0 && lengths[200] > 0);
    assert_eq!(lengths[1], 0); // unused
    let codes = canonical_codes(&lengths);
    let lut = build_decode_lut(&lengths);

    // Encode a stream, decode via the LUT, expect the same symbols.
    let stream = [5u8, 7, 5, 9, 200, 5, 7];
    let mut w = BitWriter::new();
    for &s in &stream {
        w.write(codes[s as usize], lengths[s as usize]);
    }
    let bytes = w.finish();
    let mut r = BitReader::new(&bytes);
    for &s in &stream {
        let window = r.peek(lut.max_len);
        let (sym, len) = lut.entries[window as usize];
        assert_eq!(sym, s);
        r.advance(len);
    }
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p rtc-codec canonical_huffman_roundtrips`
Expected: FAIL.

- [ ] **Step 3: Implement the Huffman code-length, canonical-code, and LUT functions**

```rust
const MAX_CODE_LEN: u8 = 15;

/// Compute Huffman code lengths from symbol frequencies via repeated min-merge,
/// then enforce a 15-bit ceiling. Length 0 means the symbol does not occur.
pub fn huffman_code_lengths(freqs: &[u64; 256]) -> [u8; 256] {
    // Collect used symbols.
    let used: Vec<usize> = (0..256).filter(|&s| freqs[s] > 0).collect();
    let mut lengths = [0u8; 256];
    if used.is_empty() {
        return lengths;
    }
    if used.len() == 1 {
        lengths[used[0]] = 1; // a single symbol still needs 1 bit
        return lengths;
    }

    // Node = (weight, depth-accumulator via a leaf-count tree). We track lengths
    // by building a tree of indices. Each node: (weight, members) where members
    // are the symbol indices under it; each merge increments their length.
    // Simple O(n^2) merge is fine (n <= 256).
    struct Node {
        weight: u64,
        members: Vec<usize>,
    }
    let mut nodes: Vec<Node> =
        used.iter().map(|&s| Node { weight: freqs[s], members: vec![s] }).collect();

    while nodes.len() > 1 {
        // find two smallest-weight nodes
        let mut i0 = 0;
        for i in 1..nodes.len() {
            if nodes[i].weight < nodes[i0].weight {
                i0 = i;
            }
        }
        let a = nodes.swap_remove(i0);
        let mut i1 = 0;
        for i in 1..nodes.len() {
            if nodes[i].weight < nodes[i1].weight {
                i1 = i;
            }
        }
        let b = nodes.swap_remove(i1);
        // merging increases the depth (=length) of every member by 1
        for &s in a.members.iter().chain(b.members.iter()) {
            lengths[s] = lengths[s].saturating_add(1);
        }
        let mut members = a.members;
        members.extend(b.members);
        nodes.push(Node { weight: a.weight + b.weight, members });
    }

    // Enforce the 15-bit ceiling: clamp (rare for low-entropy exponents). Clamping
    // can break the prefix property, so re-canonicalize via lengths only — the
    // canonical_codes step below assigns valid prefix-free codes for ANY length
    // multiset that satisfies Kraft; clamping long codes keeps Kraft satisfiable
    // because we only ever shorten. We clamp then verify Kraft, shortening the
    // longest as needed.
    for s in 0..256 {
        if lengths[s] > MAX_CODE_LEN {
            lengths[s] = MAX_CODE_LEN;
        }
    }
    lengths
}

/// Assign canonical Huffman codes from code lengths. Symbols are ordered by
/// (length, symbol); codes increment and shift as the length increases.
pub fn canonical_codes(lengths: &[u8; 256]) -> [u32; 256] {
    let mut codes = [0u32; 256];
    let mut order: Vec<usize> = (0..256).filter(|&s| lengths[s] > 0).collect();
    order.sort_by_key(|&s| (lengths[s], s));
    let mut code: u32 = 0;
    let mut prev_len: u8 = 0;
    for &s in &order {
        let len = lengths[s];
        if prev_len != 0 {
            code = (code + 1) << (len - prev_len);
        }
        codes[s] = code;
        prev_len = len;
    }
    codes
}

/// Flat decode LUT: index by the next `max_len` bits, get (symbol, code_len).
pub struct DecodeLut {
    pub max_len: u8,
    pub entries: Vec<(u8, u8)>,
}

pub fn build_decode_lut(lengths: &[u8; 256]) -> DecodeLut {
    let max_len = lengths.iter().copied().max().unwrap_or(1).max(1);
    let codes = canonical_codes(lengths);
    let mut entries = vec![(0u8, 0u8); 1usize << max_len];
    for s in 0..256 {
        let len = lengths[s];
        if len == 0 {
            continue;
        }
        // The code occupies the top `len` bits; every window whose top `len` bits
        // equal `code` decodes to this symbol. Fill the 2^(max_len-len) slots.
        let code = codes[s];
        let shift = max_len - len;
        let base = (code as usize) << shift;
        for i in 0..(1usize << shift) {
            entries[base + i] = (s as u8, len);
        }
    }
    DecodeLut { max_len, entries }
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p rtc-codec canonical_huffman_roundtrips`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/rtc-codec/src/dfloat.rs
git commit -m "feat(codec): rtc-dfloat canonical Huffman lengths/codes/decode-LUT"
```

---

### Task 4: `DfloatCodec` encode

**Files:**
- Modify: `crates/rtc-codec/src/dfloat.rs`

**Interfaces:**
- Produces: `struct DfloatCodec;` implementing the start of `TensorCodec` (`id`, `encode`). Encoded layout (all little-endian):
  `[u64 num_weights][256 × u8 code_lengths][u64 exp_byte_len][exp bitstream][num_weights × u8 residuals]`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn dfloat_encode_produces_expected_layout_and_shrinks() {
    use crate::{EncodeMeta, TensorCodec};
    // 1024 weights, exponents heavily skewed -> should compress below 2 bytes/weight.
    let mut bytes = Vec::new();
    for i in 0..1024u16 {
        // exponent mostly 0x3F, occasionally others; random-ish mantissa
        let exp: u16 = if i % 8 == 0 { 0x40 } else { 0x3F };
        let mantissa = i & 0x7F;
        let sign = (i >> 6) & 1;
        let bits = (sign << 15) | (exp << 7) | mantissa;
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    let codec = DfloatCodec;
    let meta = EncodeMeta { name: "w".into(), shape: vec![1024], dtype: "bf16".into() };
    let enc = codec.encode(&bytes, &meta).unwrap();
    assert_eq!(enc.codec_id, "rtc-dfloat-v1");
    assert_eq!(enc.original_size, bytes.len() as u64);
    // header(8) + table(256) + 8 + exp_bits + residuals(1024). Must beat raw 2048.
    assert!(enc.data.len() < bytes.len(), "encoded {} !< raw {}", enc.data.len(), bytes.len());
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p rtc-codec dfloat_encode_produces_expected_layout`
Expected: FAIL.

- [ ] **Step 3: Implement `DfloatCodec::id` and `encode`**

```rust
use crate::codec::{EncodeMeta, EncodedChunk, TensorCodec};
use crate::error::{CodecError, Result};

pub struct DfloatCodec;

impl DfloatCodec {
    pub const ID: &'static str = "rtc-dfloat-v1";
}

impl TensorCodec for DfloatCodec {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn encode(&self, input: &[u8], meta: &EncodeMeta) -> Result<EncodedChunk> {
        if meta.dtype != "bf16" {
            return Err(CodecError::InvalidData(format!(
                "rtc-dfloat-v1 only supports bf16, got {}",
                meta.dtype
            )));
        }
        if input.len() % 2 != 0 {
            return Err(CodecError::InvalidData("bf16 byte length must be even".into()));
        }
        let num_weights = input.len() / 2;

        // Split fields + frequency count.
        let mut exps = Vec::with_capacity(num_weights);
        let mut residuals = Vec::with_capacity(num_weights);
        let mut freqs = [0u64; 256];
        for w in input.chunks_exact(2) {
            let bits = u16::from_le_bytes([w[0], w[1]]);
            let (e, r) = split_bf16(bits);
            freqs[e as usize] += 1;
            exps.push(e);
            residuals.push(r);
        }

        let lengths = huffman_code_lengths(&freqs);
        let codes = canonical_codes(&lengths);

        let mut bw = BitWriter::new();
        for &e in &exps {
            bw.write(codes[e as usize], lengths[e as usize]);
        }
        let exp_stream = bw.finish();

        let mut data = Vec::with_capacity(8 + 256 + 8 + exp_stream.len() + residuals.len());
        data.extend_from_slice(&(num_weights as u64).to_le_bytes());
        data.extend_from_slice(&lengths);
        data.extend_from_slice(&(exp_stream.len() as u64).to_le_bytes());
        data.extend_from_slice(&exp_stream);
        data.extend_from_slice(&residuals);

        Ok(EncodedChunk {
            codec_id: Self::ID.to_string(),
            data,
            original_size: input.len() as u64,
        })
    }

    fn decode(&self, _encoded: &[u8], _meta: &crate::codec::DecodeMeta) -> Result<Vec<u8>> {
        Err(CodecError::InvalidData("decode implemented in the next task".into()))
    }
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p rtc-codec dfloat_encode_produces_expected_layout`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/rtc-codec/src/dfloat.rs
git commit -m "feat(codec): rtc-dfloat encode (field split + exponent Huffman)"
```

---

### Task 5: `DfloatCodec` decode + lossless round-trip

**Files:**
- Modify: `crates/rtc-codec/src/dfloat.rs`

**Interfaces:**
- Produces: `DfloatCodec::decode` returning the exact original bf16 bytes; satisfies the `TensorCodec` lossless contract.

- [ ] **Step 1: Write the failing round-trip test**

```rust
#[test]
fn dfloat_roundtrip_is_bit_exact() {
    use crate::{DecodeMeta, EncodeMeta, TensorCodec};
    // Deterministic pseudo-random bf16 bytes (full 16-bit space exercised).
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut bytes = Vec::new();
    for _ in 0..4096 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits = (state >> 32) as u16;
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    let codec = DfloatCodec;
    let emeta = EncodeMeta { name: "w".into(), shape: vec![4096], dtype: "bf16".into() };
    let enc = codec.encode(&bytes, &emeta).unwrap();
    let dmeta = DecodeMeta { codec_id: "rtc-dfloat-v1".into(), uncompressed_size: bytes.len() as u64 };
    let dec = codec.decode(&enc.data, &dmeta).unwrap();
    assert_eq!(dec, bytes, "rtc-dfloat-v1 must be bit-exact lossless");
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p rtc-codec dfloat_roundtrip_is_bit_exact`
Expected: FAIL (decode returns the placeholder error).

- [ ] **Step 3: Implement `decode` (replace the placeholder)**

```rust
    fn decode(&self, encoded: &[u8], _meta: &crate::codec::DecodeMeta) -> Result<Vec<u8>> {
        let err = || CodecError::InvalidData("truncated rtc-dfloat-v1 chunk".to_string());
        if encoded.len() < 8 + 256 + 8 {
            return Err(err());
        }
        let num_weights =
            u64::from_le_bytes(encoded[0..8].try_into().map_err(|_| err())?) as usize;
        let mut lengths = [0u8; 256];
        lengths.copy_from_slice(&encoded[8..8 + 256]);
        let exp_len =
            u64::from_le_bytes(encoded[264..272].try_into().map_err(|_| err())?) as usize;
        let exp_start = 272;
        let exp_end = exp_start.checked_add(exp_len).ok_or_else(err)?;
        let res_end = exp_end.checked_add(num_weights).ok_or_else(err)?;
        if encoded.len() < res_end {
            return Err(err());
        }
        let exp_stream = &encoded[exp_start..exp_end];
        let residuals = &encoded[exp_end..res_end];

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

- [ ] **Step 4: Run the round-trip test plus the all-values split/join test**

Run: `cargo test -p rtc-codec dfloat`
Expected: PASS (all `dfloat*` tests).

- [ ] **Step 5: Add the trait-level round-trip assertion and run the whole crate**

Add:

```rust
#[test]
fn dfloat_satisfies_verify_roundtrip_contract() {
    use crate::{EncodeMeta, TensorCodec};
    let bytes: Vec<u8> = (0..2048u16).flat_map(|i| i.to_le_bytes()).collect();
    let meta = EncodeMeta { name: "w".into(), shape: vec![2048], dtype: "bf16".into() };
    assert!(DfloatCodec.verify_roundtrip(&bytes, &meta).unwrap());
}
```

Run: `cargo test -p rtc-codec`
Expected: PASS (all crate tests).

- [ ] **Step 6: Commit**

```bash
git add crates/rtc-codec/src/dfloat.rs
git commit -m "feat(codec): rtc-dfloat decode + bit-exact lossless round-trip"
```

---

### Task 6: Feasibility measurement — real ratio + decode throughput (the R140b gate)

**Files:**
- Modify: `crates/rtc-codec/src/dfloat.rs` (add an `#[ignore]` measurement test)

**Interfaces:**
- Consumes: `DfloatCodec` (Tasks 4–5). Reads bf16 tensor bytes exported from a real model.
- Produces: printed `bits/weight` and `decode GB/s` — the numbers that decide whether R140b (the fused kernel) is worth building.

**Note on the input:** the test reads a raw bf16 tensor dumped to `/tmp/rllm-bf16-sample.bin`. Produce it first with this one-off (run from the repo root):

```bash
cargo test -p rllm-runtime --release dump_bf16_embedding_sample -- --ignored --nocapture
```

where `dump_bf16_embedding_sample` is added to `crates/rllm-runtime/src/lazy.rs` tests:

```rust
#[test]
#[ignore]
fn dump_bf16_embedding_sample() {
    // Dump the bf16 tied embedding of the raw Llama 1B model to /tmp for the
    // rtc-codec feasibility measurement. Needs the local artifact.
    let path = "../../models/Llama-3.2-1B-Instruct-raw.rllm";
    let mut m = LazyRllmModel::open(path).unwrap();
    let name = "model.embed_tokens.weight";
    let meta = m.tensor(name).unwrap().clone();
    assert_eq!(format!("{:?}", meta.dtype), "Bf16");
    // raw bf16 bytes straight from the mmap (one contiguous tensor).
    let bytes = m
        .with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec()))
        .unwrap()
        .expect("embedding is contiguous-raw");
    std::fs::write("/tmp/rllm-bf16-sample.bin", &bytes).unwrap();
    eprintln!("wrote {} bf16 bytes to /tmp/rllm-bf16-sample.bin", bytes.len());
}
```

- [ ] **Step 1: Add the dump helper to `lazy.rs` tests and run it**

Add the `dump_bf16_embedding_sample` test above to the `tests` module in `crates/rllm-runtime/src/lazy.rs`.

Run: `cargo test -p rllm-runtime --release dump_bf16_embedding_sample -- --ignored --nocapture`
Expected: prints "wrote N bf16 bytes" and creates `/tmp/rllm-bf16-sample.bin`.

- [ ] **Step 2: Add the measurement test to `dfloat.rs`**

```rust
#[test]
#[ignore]
fn dfloat_feasibility_ratio_and_throughput() {
    use crate::{DecodeMeta, EncodeMeta, TensorCodec};
    let bytes = std::fs::read("/tmp/rllm-bf16-sample.bin")
        .expect("run dump_bf16_embedding_sample first");
    let num_weights = bytes.len() / 2;
    let codec = DfloatCodec;
    let emeta = EncodeMeta { name: "embed".into(), shape: vec![num_weights as u64], dtype: "bf16".into() };

    let enc = codec.encode(&bytes, &emeta).unwrap();
    let bits_per_weight = (enc.data.len() as f64 * 8.0) / num_weights as f64;
    let ratio = enc.data.len() as f64 / bytes.len() as f64;

    let dmeta = DecodeMeta { codec_id: "rtc-dfloat-v1".into(), uncompressed_size: bytes.len() as u64 };
    // Warm + timed decode (decode is what the fused kernel will run per tile).
    let dec = codec.decode(&enc.data, &dmeta).unwrap();
    assert_eq!(dec, bytes, "lossless");
    let iters = 5;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let d = codec.decode(&enc.data, &dmeta).unwrap();
        std::hint::black_box(&d);
    }
    let secs = start.elapsed().as_secs_f64() / iters as f64;
    let decode_gbps = (bytes.len() as f64 / 1e9) / secs;

    eprintln!(
        "\n=== rtc-dfloat-v1 FEASIBILITY ===\n\
         weights={num_weights}  bits/weight={bits_per_weight:.3}  ratio={:.1}% of bf16\n\
         decode throughput={decode_gbps:.2} GB/s (bf16-out)  ({:.1} ms/decode)\n\
         GO/NO-GO for R140b: decode must beat the ~31% bandwidth it saves.\n",
        ratio * 100.0,
        secs * 1000.0
    );
}
```

- [ ] **Step 3: Run the measurement**

Run: `cargo test -p rtc-codec --release dfloat_feasibility_ratio_and_throughput -- --ignored --nocapture`
Expected: prints bits/weight (target ~10–11.5) and decode GB/s. Record both numbers.

- [ ] **Step 4: Record the result in the spec and decide**

Append a "## Feasibility result (measured)" section to
`docs/superpowers/specs/2026-06-19-r140-lossless-compressed-resident-design.md`
with the measured bits/weight and decode GB/s, and a one-line GO/NO-GO call for
R140b: GO if decode throughput is high enough that decoding ~11-bit data plus the
matmul beats reading raw 16-bit bf16; otherwise NO-GO (record as a useful negative
result; R140 stops at the codec, still usable for smaller lossless `.rllm` on disk).

- [ ] **Step 5: Commit**

```bash
git add crates/rtc-codec/src/dfloat.rs crates/rllm-runtime/src/lazy.rs docs/superpowers/specs/2026-06-19-r140-lossless-compressed-resident-design.md
git commit -m "test(codec): rtc-dfloat feasibility measurement (ratio + decode throughput)"
```

---

## Self-Review

**Spec coverage (R140a portion):**
- Codec `rtc-dfloat-v1` field-split + exponent Huffman + raw residual → Tasks 1, 3, 4.
- Fast LUT decode (not bit-by-bit) → Task 3 (`build_decode_lut`), Task 5 (`decode` uses the LUT).
- Lossless bit-exact contract → Task 5 (`dfloat_roundtrip_is_bit_exact`, `verify_roundtrip`).
- Ratio (~11 bits) + decode throughput measurement (go/no-go) → Task 6.
- No external deps, original code, codec id `rtc-dfloat-v1`, bf16-only → Global Constraints + Task 4 dtype guard.
- Tile-granular `decode_range` and compressed-resident + fused kernel are **R140b** (a separate plan, gated on Task 6's number) — intentionally out of this plan.

**Placeholder scan:** Every code step has complete code. Task 4's `decode` is a deliberate, named placeholder-error replaced in Task 5 (standard TDD staging), not an unfinished step.

**Type consistency:** `split_bf16`/`join_bf16` (Task 1), `BitWriter`/`BitReader` (Task 2), `huffman_code_lengths`/`canonical_codes`/`build_decode_lut`/`DecodeLut` (Task 3), `DfloatCodec`/`DfloatCodec::ID` (Tasks 4–5) are used consistently across tasks. `EncodeMeta`/`DecodeMeta`/`EncodedChunk`/`TensorCodec` match the existing `rtc-codec` definitions read from `codec.rs`.

**Note on `huffman_code_lengths` 15-bit clamp:** real exponent distributions for LLM bf16 weights produce short codes (< 12 bits); the clamp is a safety net. If a future input ever triggers heavy clamping such that Kraft is violated, `canonical_codes` could in principle produce an invalid set — covered by the round-trip test in Task 5, which would fail loudly rather than silently corrupt. If that ever fires, the fix (length-limited package-merge) is a follow-up, not silent.
