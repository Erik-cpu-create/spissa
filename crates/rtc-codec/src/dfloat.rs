//! rtc-dfloat-v1: lossless bf16 codec.
//!
//! bf16 = [sign:1][exponent:8][mantissa:7]. The exponent has low entropy for LLM
//! weights, so we entropy-code it (canonical Huffman) and store sign+mantissa raw.
//! Original implementation (technique from DFloat11, arXiv 2504.11651); no code
//! was copied and no external dependency is used.

use crate::codec::{EncodeMeta, EncodedChunk, TensorCodec};
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

    // No length cap: the min-merge above produces valid (Kraft-satisfying) Huffman
    // lengths. Inputs are bf16 EXPONENT bytes (256 symbols, low-entropy), whose code
    // lengths stay well under 16, so the decode LUT (2^max_len entries) stays small.
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
        let exp_start: usize = 272;
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
