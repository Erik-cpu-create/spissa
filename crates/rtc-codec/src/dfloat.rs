// Copyright (c) 2026 Erik. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! rtc-dfloat-v1: lossless bf16 codec.
//!
//! bf16 = [sign:1][exponent:8][mantissa:7]. The exponent has low entropy for LLM
//! weights, so we entropy-code it (canonical Huffman) and store sign+mantissa raw.
//! Original implementation (technique from DFloat11, arXiv 2504.11651); no code
//! was copied and no external dependency is used.

use crate::bitreader_fast::BufferedBitReader;
use crate::codec::{EncodeMeta, EncodedChunk, TensorCodec};

/// Maximum Huffman code length in bits. Caps the decode LUT at 2^15 = 32 768 entries,
/// which is safe even for singleton exponents in real LLM embedding tables.
const MAX_CODE_LEN: u8 = 15;
use crate::error::{CodecError, Result};

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

/// Compute Huffman code lengths from symbol frequencies via repeated min-merge.
/// Length 0 means the symbol does not occur.
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

    // Length-limit to MAX_CODE_LEN bits, then repair the Kraft inequality. A
    // singleton exponent (e.g. one that occurs once among millions of weights, as
    // in real LLM embeddings) gets a very long Huffman code; we clamp it and
    // lengthen the rarest still-shrinkable codes until the set is a valid prefix
    // code. The decode LUT is then bounded to 2^MAX_CODE_LEN entries. Lossless is
    // preserved: only code LENGTHS change, and both encode and decode use the same
    // stored lengths table.
    for len in lengths.iter_mut() {
        if *len > MAX_CODE_LEN {
            *len = MAX_CODE_LEN;
        }
    }
    let budget: u64 = 1u64 << MAX_CODE_LEN;
    loop {
        let mut k: u64 = 0;
        for &len in lengths.iter() {
            if len > 0 {
                k += 1u64 << (MAX_CODE_LEN - len);
            }
        }
        if k <= budget {
            break;
        }
        // Overfull: lengthen the rarest still-shrinkable code (largest len < MAX).
        let mut best: Option<usize> = None;
        for s in 0..256 {
            if lengths[s] > 0 && lengths[s] < MAX_CODE_LEN {
                match best {
                    Some(b) if lengths[b] >= lengths[s] => {}
                    _ => best = Some(s),
                }
            }
        }
        match best {
            Some(s) => lengths[s] += 1,
            None => break, // all used symbols at MAX; k <= budget already holds
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

/// Validate a decoded length table before building the LUT: bounded max length
/// and the Kraft inequality, so canonical_codes/build_decode_lut cannot overflow
/// or index out of bounds on corrupt input.
fn validate_lengths(lengths: &[u8; 256]) -> Result<()> {
    let max_len = lengths.iter().copied().max().unwrap_or(0);
    if max_len > MAX_CODE_LEN {
        return Err(CodecError::InvalidData(format!(
            "rtc-dfloat-v1: corrupt length table, max length {max_len} > {MAX_CODE_LEN}"
        )));
    }
    if max_len == 0 {
        return Ok(());
    }
    let mut sum: u64 = 0;
    for &len in lengths.iter() {
        if len > 0 {
            sum += 1u64 << (max_len - len);
        }
    }
    if sum > (1u64 << max_len) {
        return Err(CodecError::InvalidData(
            "rtc-dfloat-v1: corrupt length table violates the Kraft inequality".into(),
        ));
    }
    Ok(())
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

pub struct DfloatCodec;

impl DfloatCodec {
    pub const ID: &'static str = "rtc-dfloat-v1";

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

    #[test]
    fn dfloat_satisfies_verify_roundtrip_contract() {
        use crate::{EncodeMeta, TensorCodec};
        let bytes: Vec<u8> = (0..2048u16).flat_map(|i| i.to_le_bytes()).collect();
        let meta = EncodeMeta { name: "w".into(), shape: vec![2048], dtype: "bf16".into() };
        assert!(DfloatCodec.verify_roundtrip(&bytes, &meta).unwrap());
    }

    #[test]
    fn dfloat_roundtrip_empty_tensor() {
        use crate::{DecodeMeta, EncodeMeta, TensorCodec};
        let bytes: Vec<u8> = Vec::new();
        let codec = DfloatCodec;
        let emeta = EncodeMeta { name: "w".into(), shape: vec![0], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &emeta).unwrap();
        let dmeta = DecodeMeta {
            codec_id: "rtc-dfloat-v1".into(),
            uncompressed_size: 0,
        };
        let dec = codec.decode(&enc.data, &dmeta).unwrap();
        assert_eq!(dec, bytes);
    }

    #[test]
    fn dfloat_roundtrip_single_weight() {
        use crate::{DecodeMeta, EncodeMeta, TensorCodec};
        // one bf16 = 2 bytes
        let bits: u16 = 0x3F80; // 1.0 in bf16
        let bytes = bits.to_le_bytes().to_vec();
        let codec = DfloatCodec;
        let emeta = EncodeMeta { name: "w".into(), shape: vec![1], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &emeta).unwrap();
        let dmeta = DecodeMeta {
            codec_id: "rtc-dfloat-v1".into(),
            uncompressed_size: bytes.len() as u64,
        };
        let dec = codec.decode(&enc.data, &dmeta).unwrap();
        assert_eq!(dec, bytes);
    }

    #[test]
    fn dfloat_roundtrip_all_identical() {
        use crate::{DecodeMeta, EncodeMeta, TensorCodec};
        let bits: u16 = 0x4000; // 2.0 in bf16
        let bytes: Vec<u8> =
            (0..512).flat_map(|_| bits.to_le_bytes()).collect();
        let codec = DfloatCodec;
        let emeta =
            EncodeMeta { name: "w".into(), shape: vec![512], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &emeta).unwrap();
        let dmeta = DecodeMeta {
            codec_id: "rtc-dfloat-v1".into(),
            uncompressed_size: bytes.len() as u64,
        };
        let dec = codec.decode(&enc.data, &dmeta).unwrap();
        assert_eq!(dec, bytes);
    }

    #[test]
    fn dfloat_roundtrip_single_exponent() {
        use crate::{DecodeMeta, EncodeMeta, TensorCodec};
        // 512 weights all with exponent 0x40, varying sign and low 7 mantissa bits.
        let mut bytes = Vec::with_capacity(512 * 2);
        for i in 0..512u16 {
            let exp: u16 = 0x40;
            let mantissa = i & 0x7F;
            let sign = (i >> 7) & 1;
            let bits = (sign << 15) | (exp << 7) | mantissa;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let codec = DfloatCodec;
        let emeta =
            EncodeMeta { name: "w".into(), shape: vec![512], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &emeta).unwrap();
        let dmeta = DecodeMeta {
            codec_id: "rtc-dfloat-v1".into(),
            uncompressed_size: bytes.len() as u64,
        };
        let dec = codec.decode(&enc.data, &dmeta).unwrap();
        assert_eq!(dec, bytes, "single-exponent round-trip must be bit-exact");
    }

    #[test]
    fn dfloat_decode_rejects_invalid_length_table() {
        use crate::{DecodeMeta, EncodeMeta, TensorCodec};
        // Encode a few weights to get a valid payload.
        let bits: u16 = 0x3F80;
        let bytes: Vec<u8> = (0..8).flat_map(|_| bits.to_le_bytes()).collect();
        let codec = DfloatCodec;
        let emeta = EncodeMeta { name: "w".into(), shape: vec![8], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &emeta).unwrap();

        // Corrupt the length-table region (bytes 8..264) by setting one entry to 30.
        let mut corrupt = enc.data.clone();
        corrupt[8] = 30; // length > MAX_CODE_LEN (15) → must be rejected
        let dmeta = DecodeMeta {
            codec_id: "rtc-dfloat-v1".into(),
            uncompressed_size: bytes.len() as u64,
        };
        let result = codec.decode(&corrupt, &dmeta);
        assert!(result.is_err(), "decode must return Err on corrupt length table");
    }

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
        assert!(
            codec.decode_fast(&corrupt).is_err(),
            "decode_fast must reject corrupt length table"
        );
    }

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
        let dmeta =
            DecodeMeta { codec_id: "rtc-dfloat-v1".into(), uncompressed_size: bytes.len() as u64 };

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
}
