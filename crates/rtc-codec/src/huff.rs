// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! rtc-huff-v1: byte-level Huffman codec
//!
//! This is a small in-house static Huffman codec for tensor chunks.
//! It builds a frequency table per chunk, serializes that table, then
//! bit-packs the Huffman-coded byte stream.
//!
//! Format:
//! - 256 × u32 LE: byte frequency table
//! - u64 LE: encoded bit length
//! - N bytes: MSB-first bitstream

use crate::codec::{DecodeMeta, EncodeMeta, EncodedChunk, TensorCodec};
use crate::error::{CodecError, Result};
use crate::CODEC_HUFF_V1;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const SYMBOL_COUNT: usize = 256;
const FREQ_TABLE_BYTES: usize = SYMBOL_COUNT * 4;
const BIT_LEN_BYTES: usize = 8;
const HEADER_BYTES: usize = FREQ_TABLE_BYTES + BIT_LEN_BYTES;

#[derive(Debug)]
enum Node {
    Leaf(u8),
    Branch(Box<Node>, Box<Node>),
}

struct HeapItem {
    freq: u64,
    order: usize,
    node: Box<Node>,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.freq == other.freq && self.order == other.order
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-first, so reverse ordering to pop lowest frequency first.
        other
            .freq
            .cmp(&self.freq)
            .then_with(|| other.order.cmp(&self.order))
    }
}

#[derive(Clone, Copy, Debug)]
struct FlatNode {
    left: u16,
    right: u16,
    symbol: u8,
    is_leaf: bool,
}

#[derive(Clone, Copy, Debug)]
struct LutEntry {
    symbol: u8,
    consumed: u8,
    next_node: u16,
    is_leaf: bool,
}

/// Byte-level static Huffman codec.
pub struct HuffmanCodec;

impl HuffmanCodec {
    pub fn new() -> Self {
        Self
    }

    fn frequency_table(input: &[u8]) -> [u32; SYMBOL_COUNT] {
        let mut freqs = [0u32; SYMBOL_COUNT];
        for &byte in input {
            freqs[byte as usize] += 1;
        }
        freqs
    }

    fn build_tree(freqs: &[u32; SYMBOL_COUNT]) -> Result<Option<Box<Node>>> {
        let mut heap = BinaryHeap::new();
        let mut order = 0usize;

        for (symbol, &freq) in freqs.iter().enumerate() {
            if freq > 0 {
                heap.push(HeapItem {
                    freq: freq as u64,
                    order,
                    node: Box::new(Node::Leaf(symbol as u8)),
                });
                order += 1;
            }
        }

        if heap.is_empty() {
            return Ok(None);
        }

        while heap.len() > 1 {
            let left = heap.pop().expect("heap has left node");
            let right = heap.pop().expect("heap has right node");

            heap.push(HeapItem {
                freq: left.freq + right.freq,
                order,
                node: Box::new(Node::Branch(left.node, right.node)),
            });
            order += 1;
        }

        Ok(Some(heap.pop().expect("heap has root").node))
    }

    fn assign_codes(
        node: &Node,
        code: u64,
        len: u8,
        codes: &mut [Option<(u64, u8)>; SYMBOL_COUNT],
    ) -> Result<()> {
        match node {
            Node::Leaf(symbol) => {
                // Single-symbol chunks get a synthetic 1-bit code, though encode
                // special-cases them to zero payload bits.
                let final_len = if len == 0 { 1 } else { len };
                codes[*symbol as usize] = Some((code, final_len));
                Ok(())
            }
            Node::Branch(left, right) => {
                if len >= 63 {
                    // Keep the codec simple and safe. Extremely pathological
                    // distributions can fall back to raw codec selection.
                    return Err(CodecError::General(
                        "Huffman code length exceeded 63 bits".to_string(),
                    ));
                }
                Self::assign_codes(left, code << 1, len + 1, codes)?;
                Self::assign_codes(right, (code << 1) | 1, len + 1, codes)?;
                Ok(())
            }
        }
    }

    fn count_nonzero(freqs: &[u32; SYMBOL_COUNT]) -> usize {
        freqs.iter().filter(|&&f| f > 0).count()
    }

    fn single_symbol(freqs: &[u32; SYMBOL_COUNT]) -> Option<u8> {
        let mut found = None;
        for (symbol, &freq) in freqs.iter().enumerate() {
            if freq > 0 {
                if found.is_some() {
                    return None;
                }
                found = Some(symbol as u8);
            }
        }
        found
    }

    fn write_header(freqs: &[u32; SYMBOL_COUNT], bit_len: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + payload.len());
        for &freq in freqs {
            out.extend_from_slice(&freq.to_le_bytes());
        }
        out.extend_from_slice(&bit_len.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn read_header(encoded: &[u8]) -> Result<([u32; SYMBOL_COUNT], u64, &[u8])> {
        if encoded.len() < HEADER_BYTES {
            return Err(CodecError::InvalidData(format!(
                "Huffman payload too short: expected at least {HEADER_BYTES} bytes, got {}",
                encoded.len()
            )));
        }

        let mut freqs = [0u32; SYMBOL_COUNT];
        for (i, freq) in freqs.iter_mut().enumerate() {
            let start = i * 4;
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&encoded[start..start + 4]);
            *freq = u32::from_le_bytes(bytes);
        }

        let mut bit_len_bytes = [0u8; 8];
        bit_len_bytes.copy_from_slice(&encoded[FREQ_TABLE_BYTES..HEADER_BYTES]);
        let bit_len = u64::from_le_bytes(bit_len_bytes);

        Ok((freqs, bit_len, &encoded[HEADER_BYTES..]))
    }

    fn encode_bits(
        input: &[u8],
        codes: &[Option<(u64, u8)>; SYMBOL_COUNT],
    ) -> Result<(u64, Vec<u8>)> {
        let mut payload = Vec::new();
        let mut current = 0u8;
        let mut filled = 0u8;
        let mut bit_len = 0u64;

        for &byte in input {
            let (code, len) = codes[byte as usize].ok_or_else(|| {
                CodecError::General(format!("Missing Huffman code for byte {byte}"))
            })?;

            for shift in (0..len).rev() {
                let bit = ((code >> shift) & 1) as u8;
                current = (current << 1) | bit;
                filled += 1;
                bit_len += 1;

                if filled == 8 {
                    payload.push(current);
                    current = 0;
                    filled = 0;
                }
            }
        }

        if filled > 0 {
            current <<= 8 - filled;
            payload.push(current);
        }

        Ok((bit_len, payload))
    }

    fn flatten_tree(node: &Node, flat_nodes: &mut Vec<FlatNode>) -> u16 {
        let idx = flat_nodes.len() as u16;
        flat_nodes.push(FlatNode {
            left: 0,
            right: 0,
            symbol: 0,
            is_leaf: false,
        });
        match node {
            Node::Leaf(symbol) => {
                flat_nodes[idx as usize] = FlatNode {
                    left: 0,
                    right: 0,
                    symbol: *symbol,
                    is_leaf: true,
                };
            }
            Node::Branch(left, right) => {
                let left_idx = Self::flatten_tree(left, flat_nodes);
                let right_idx = Self::flatten_tree(right, flat_nodes);
                flat_nodes[idx as usize] = FlatNode {
                    left: left_idx,
                    right: right_idx,
                    symbol: 0,
                    is_leaf: false,
                };
            }
        }
        idx
    }

    fn build_lut(flat_nodes: &[FlatNode]) -> [LutEntry; 256] {
        let mut lut = [LutEntry {
            symbol: 0,
            consumed: 0,
            next_node: 0,
            is_leaf: false,
        }; 256];

        for prefix in 0..256 {
            let mut cursor_idx = 0; // Root is index 0
            let mut consumed = 0;
            let mut is_leaf = false;
            let mut symbol = 0;

            for step in 0..8 {
                let bit = (prefix >> (7 - step)) & 1;
                let node = &flat_nodes[cursor_idx as usize];
                cursor_idx = if bit == 0 { node.left } else { node.right };
                consumed += 1;

                let next_node = &flat_nodes[cursor_idx as usize];
                if next_node.is_leaf {
                    is_leaf = true;
                    symbol = next_node.symbol;
                    break;
                }
            }

            lut[prefix as usize] = LutEntry {
                symbol,
                consumed,
                next_node: cursor_idx,
                is_leaf,
            };
        }

        lut
    }

    fn decode_bits(
        root: &Node,
        bit_len: u64,
        payload: &[u8],
        expected_size: usize,
    ) -> Result<Vec<u8>> {
        let available_bits = (payload.len() as u64) * 8;
        if bit_len > available_bits {
            return Err(CodecError::InvalidData(format!(
                "Huffman bit length {bit_len} exceeds payload capacity {available_bits}"
            )));
        }

        // 1. Flatten the tree
        let mut flat_nodes = Vec::with_capacity(512);
        let root_idx = Self::flatten_tree(root, &mut flat_nodes);

        // 2. Build the LUT
        let lut = Self::build_lut(&flat_nodes);

        // 3. Decode
        let mut out = Vec::with_capacity(expected_size);
        let mut cursor_idx = root_idx;

        let mut bit_buf = 0u64;
        let mut bit_count = 0u32;
        let mut byte_idx = 0;
        let mut bits_consumed_total = 0u64;

        while bits_consumed_total < bit_len {
            // Refill bit buffer
            while bit_count <= 56 && byte_idx < payload.len() {
                bit_buf = (bit_buf << 8) | (payload[byte_idx] as u64);
                bit_count += 8;
                byte_idx += 1;
            }

            let bits_left = bit_len - bits_consumed_total;

            if cursor_idx == 0 && bits_left >= 8 && bit_count >= 8 {
                let shift = bit_count - 8;
                let val = ((bit_buf >> shift) & 0xFF) as usize;
                let entry = unsafe { lut.get_unchecked(val) };

                if entry.is_leaf {
                    out.push(entry.symbol);
                    if out.len() > expected_size {
                        return Err(CodecError::InvalidData(
                            "Huffman decoded more bytes than expected".to_string(),
                        ));
                    }
                    bit_count -= entry.consumed as u32;
                    bits_consumed_total += entry.consumed as u64;
                    cursor_idx = 0;
                } else {
                    cursor_idx = entry.next_node;
                    bit_count -= 8;
                    bits_consumed_total += 8;
                }
            } else {
                let bits_to_process = std::cmp::min(bits_left, 1);
                if bits_to_process == 0 {
                    break;
                }
                if bit_count == 0 {
                    break;
                }
                let bit = ((bit_buf >> (bit_count - 1)) & 1) as u8;
                bit_count -= 1;
                bits_consumed_total += 1;

                let node = unsafe { flat_nodes.get_unchecked(cursor_idx as usize) };
                cursor_idx = if bit == 0 { node.left } else { node.right };

                let next_node = unsafe { flat_nodes.get_unchecked(cursor_idx as usize) };
                if next_node.is_leaf {
                    out.push(next_node.symbol);
                    if out.len() > expected_size {
                        return Err(CodecError::InvalidData(
                            "Huffman decoded more bytes than expected".to_string(),
                        ));
                    }
                    cursor_idx = 0;
                }
            }
        }

        if out.len() != expected_size {
            return Err(CodecError::InvalidData(format!(
                "Huffman decoded size mismatch: expected {}, got {}",
                expected_size,
                out.len()
            )));
        }

        Ok(out)
    }
}

impl Default for HuffmanCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorCodec for HuffmanCodec {
    fn id(&self) -> &'static str {
        CODEC_HUFF_V1
    }

    fn encode(&self, input: &[u8], _meta: &EncodeMeta) -> Result<EncodedChunk> {
        let freqs = Self::frequency_table(input);

        if input.is_empty() {
            return Ok(EncodedChunk {
                codec_id: CODEC_HUFF_V1.to_string(),
                data: Self::write_header(&freqs, 0, &[]),
                original_size: 0,
            });
        }

        if Self::count_nonzero(&freqs) == 1 {
            // Frequency table already identifies the repeated byte; no bitstream needed.
            return Ok(EncodedChunk {
                codec_id: CODEC_HUFF_V1.to_string(),
                data: Self::write_header(&freqs, 0, &[]),
                original_size: input.len() as u64,
            });
        }

        let root = Self::build_tree(&freqs)?.ok_or_else(|| {
            CodecError::General("Huffman tree missing for non-empty input".to_string())
        })?;

        let mut codes: [Option<(u64, u8)>; SYMBOL_COUNT] = std::array::from_fn(|_| None);
        Self::assign_codes(&root, 0, 0, &mut codes)?;

        let (bit_len, payload) = Self::encode_bits(input, &codes)?;

        Ok(EncodedChunk {
            codec_id: CODEC_HUFF_V1.to_string(),
            data: Self::write_header(&freqs, bit_len, &payload),
            original_size: input.len() as u64,
        })
    }

    fn decode(&self, encoded: &[u8], meta: &DecodeMeta) -> Result<Vec<u8>> {
        let (freqs, bit_len, payload) = Self::read_header(encoded)?;
        let expected_size = meta.uncompressed_size as usize;
        let freq_sum: u64 = freqs.iter().map(|&f| f as u64).sum();

        if freq_sum != meta.uncompressed_size {
            return Err(CodecError::InvalidData(format!(
                "Huffman frequency sum mismatch: expected {}, got {}",
                meta.uncompressed_size, freq_sum
            )));
        }

        if expected_size == 0 {
            return Ok(Vec::new());
        }

        if let Some(symbol) = Self::single_symbol(&freqs) {
            if bit_len != 0 || !payload.is_empty() {
                return Err(CodecError::InvalidData(
                    "Single-symbol Huffman chunk must have empty bitstream".to_string(),
                ));
            }
            return Ok(vec![symbol; expected_size]);
        }

        let root = Self::build_tree(&freqs)?.ok_or_else(|| {
            CodecError::InvalidData("Missing Huffman tree for non-empty chunk".to_string())
        })?;

        Self::decode_bits(&root, bit_len, payload, expected_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::tests::assert_roundtrip;

    #[test]
    fn test_huff_empty() {
        let codec = HuffmanCodec::new();
        assert_roundtrip(&codec, b"", "empty");
    }

    #[test]
    fn test_huff_single_symbol() {
        let codec = HuffmanCodec::new();
        let data = vec![42u8; 10_000];
        assert_roundtrip(&codec, &data, "single-symbol");

        let meta = EncodeMeta {
            name: "single".to_string(),
            shape: vec![data.len() as u64],
            dtype: "u8".to_string(),
        };
        let encoded = codec.encode(&data, &meta).unwrap();
        assert_eq!(encoded.data.len(), HEADER_BYTES);
    }

    #[test]
    fn test_huff_biased_distribution_compresses() {
        let codec = HuffmanCodec::new();
        let mut data = Vec::new();
        data.extend(std::iter::repeat_n(0, 80_000));
        data.extend(std::iter::repeat_n(1, 15_000));
        for i in 0..5_000 {
            data.push((i % 64) as u8);
        }

        assert_roundtrip(&codec, &data, "biased");

        let meta = EncodeMeta {
            name: "biased".to_string(),
            shape: vec![data.len() as u64],
            dtype: "u8".to_string(),
        };
        let encoded = codec.encode(&data, &meta).unwrap();
        assert!(encoded.data.len() < data.len() / 2);
    }

    #[test]
    fn test_huff_all_byte_values() {
        let codec = HuffmanCodec::new();
        let data: Vec<u8> = (0..=255).cycle().take(100_000).collect();
        assert_roundtrip(&codec, &data, "all-byte-values");
    }

    #[test]
    fn test_huff_pseudorandom_roundtrip() {
        let codec = HuffmanCodec::new();
        let data: Vec<u8> = (0..100_000).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        assert_roundtrip(&codec, &data, "pseudorandom");
    }

    #[test]
    fn test_huff_rejects_truncated_header() {
        let codec = HuffmanCodec::new();
        let meta = DecodeMeta {
            codec_id: CODEC_HUFF_V1.to_string(),
            uncompressed_size: 1,
        };
        assert!(codec.decode(&[0u8; 10], &meta).is_err());
    }
}
