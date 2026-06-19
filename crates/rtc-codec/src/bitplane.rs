//! rtc-bitplane-v1: SIMD-decodable lossless bf16 codec.
//!
//! Like rtc-dfloat-v1 it splits bf16 into (exponent, residual), but instead of
//! Huffman-coding the exponent it stores a fixed-width palette index per weight.
//! Fixed width => branchless, SIMD `tbl`-gather decode (no per-symbol serial
//! dependency). Trades ratio (~13 bits/weight vs Huffman's 10.6) for fast decode.

use crate::bitreader_fast::BufferedBitReader;
use crate::codec::{DecodeMeta, EncodeMeta, EncodedChunk, TensorCodec};
use crate::dfloat::{join_bf16, split_bf16, BitWriter};
use crate::error::{CodecError, Result};

/// Minimum bits to index a palette of `p` entries. `p<=1` needs 0 bits
/// (every weight uses palette[0]); otherwise ceil(log2(p)).
pub fn index_width(p: usize) -> u8 {
    if p <= 1 {
        0
    } else {
        (usize::BITS - (p - 1).leading_zeros()) as u8
    }
}

pub struct BitplaneCodec;

impl BitplaneCodec {
    pub const ID: &'static str = "rtc-bitplane-v1";
}

const HEADER_LEN: usize = 4 + 1 + 1 + 8 + 1 + 1; // magic+version+flags+n+palette_len+width = 16
const FLAG_RAW: u8 = 0x01;

impl TensorCodec for BitplaneCodec {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn encode(&self, input: &[u8], meta: &EncodeMeta) -> Result<EncodedChunk> {
        if meta.dtype != "bf16" {
            return Err(CodecError::InvalidData(format!(
                "rtc-bitplane-v1 only supports bf16, got {}",
                meta.dtype
            )));
        }
        if input.len() % 2 != 0 {
            return Err(CodecError::InvalidData("bf16 byte length must be even".into()));
        }
        let n = input.len() / 2;

        let mut present = [false; 256];
        let mut exps = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        for w in input.chunks_exact(2) {
            let bits = u16::from_le_bytes([w[0], w[1]]);
            let (e, r) = split_bf16(bits);
            present[e as usize] = true;
            exps.push(e);
            residuals.push(r);
        }
        let palette: Vec<u8> =
            (0..256usize).filter(|&i| present[i]).map(|i| i as u8).collect();

        let mut data = Vec::with_capacity(HEADER_LEN + palette.len() + n * 2);
        data.extend_from_slice(b"RTCB");
        data.push(1); // version

        if palette.len() > 64 {
            // Raw fallback: not usefully palette-compressible.
            data.push(FLAG_RAW);
            data.extend_from_slice(&(n as u64).to_le_bytes());
            data.push(0); // palette_len
            data.push(0); // index_width
            data.extend_from_slice(input);
            return Ok(EncodedChunk {
                codec_id: Self::ID.to_string(),
                data,
                original_size: input.len() as u64,
            });
        }

        let w = index_width(palette.len());
        data.push(0); // flags
        data.extend_from_slice(&(n as u64).to_le_bytes());
        data.push(palette.len() as u8);
        data.push(w);
        data.extend_from_slice(&palette);

        if w > 0 {
            let mut map = [0u8; 256];
            for (i, &e) in palette.iter().enumerate() {
                map[e as usize] = i as u8;
            }
            let mut bw = BitWriter::new();
            for &e in &exps {
                bw.write(map[e as usize] as u32, w);
            }
            data.extend_from_slice(&bw.finish());
        }
        data.extend_from_slice(&residuals);

        Ok(EncodedChunk {
            codec_id: Self::ID.to_string(),
            data,
            original_size: input.len() as u64,
        })
    }

    fn decode(&self, encoded: &[u8], _meta: &DecodeMeta) -> Result<Vec<u8>> {
        let err = || CodecError::InvalidData("truncated rtc-bitplane-v1 chunk".to_string());
        if encoded.len() < HEADER_LEN || &encoded[0..4] != b"RTCB" || encoded[4] != 1 {
            return Err(err());
        }
        let flags = encoded[5];
        let n = u64::from_le_bytes(encoded[6..14].try_into().map_err(|_| err())?) as usize;
        let p = encoded[14] as usize;
        let w = encoded[15];
        let mut off = HEADER_LEN;

        if flags & FLAG_RAW != 0 {
            let body = encoded.get(off..).ok_or_else(err)?;
            if body.len() < n * 2 {
                return Err(err());
            }
            return Ok(body[..n * 2].to_vec());
        }

        let palette = encoded.get(off..off + p).ok_or_else(err)?;
        off += p;
        let idx_bytes = (n * w as usize + 7) / 8;
        let idx_plane = encoded.get(off..off + idx_bytes).ok_or_else(err)?;
        off += idx_bytes;
        let residuals = encoded.get(off..off + n).ok_or_else(err)?;

        let mut out = vec![0u8; n * 2];
        if n == 0 {
            return Ok(out);
        }
        if w == 0 {
            if p == 0 {
                return Err(err());
            }
            let exp = palette[0];
            for i in 0..n {
                let bits = join_bf16(exp, residuals[i]);
                out[2 * i] = bits as u8;
                out[2 * i + 1] = (bits >> 8) as u8;
            }
            return Ok(out);
        }

        let mut reader = BufferedBitReader::new(idx_plane);
        for i in 0..n {
            reader.refill();
            let idx = reader.peek(w) as usize;
            reader.consume(w);
            if idx >= p {
                return Err(CodecError::InvalidData(
                    "rtc-bitplane-v1: palette index out of range".into(),
                ));
            }
            let bits = join_bf16(palette[idx], residuals[i]);
            out[2 * i] = bits as u8;
            out[2 * i + 1] = (bits >> 8) as u8;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecodeMeta;

    fn dmeta() -> DecodeMeta {
        DecodeMeta { codec_id: "rtc-bitplane-v1".into(), uncompressed_size: 0 }
    }

    // Build bf16 bytes whose exponents cycle through `distinct` values, with
    // pseudo-random sign+mantissa, so the palette has exactly `distinct` entries.
    fn make_bf16(distinct: usize, n: usize) -> Vec<u8> {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let exp = (96 + (i % distinct)) as u16 & 0xFF; // exponents in a tight band
            let sign = ((state >> 31) & 1) as u16;
            let mant = (state & 0x7F) as u16;
            let bits = (sign << 15) | (exp << 7) | mant;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    #[test]
    fn bitplane_roundtrip_bit_exact_various_palettes() {
        let codec = BitplaneCodec;
        for &distinct in &[1usize, 2, 3, 17, 32, 64] {
            let bytes = make_bf16(distinct, 1000);
            let meta = EncodeMeta { name: "w".into(), shape: vec![1000], dtype: "bf16".into() };
            let enc = codec.encode(&bytes, &meta).unwrap();
            // not raw-fallback for palette <= 64 => must be smaller than bf16
            assert!(enc.data.len() < bytes.len(), "distinct={distinct}: must compress");
            let dec = codec.decode(&enc.data, &dmeta()).unwrap();
            assert_eq!(dec, bytes, "distinct={distinct}: must be bit-exact lossless");
        }
    }

    #[test]
    fn bitplane_roundtrip_raw_fallback_over_64_exponents() {
        let codec = BitplaneCodec;
        let bytes = make_bf16(120, 2000); // >64 distinct => raw fallback
        let meta = EncodeMeta { name: "w".into(), shape: vec![2000], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &meta).unwrap();
        let dec = codec.decode(&enc.data, &dmeta()).unwrap();
        assert_eq!(dec, bytes, "raw-fallback must be bit-exact lossless");
    }

    #[test]
    fn bitplane_roundtrip_tail_and_edge_sizes() {
        let codec = BitplaneCodec;
        for &n in &[0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 100] {
            let bytes = make_bf16(32, n); // w=5
            let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
            let enc = codec.encode(&bytes, &meta).unwrap();
            let dec = codec.decode(&enc.data, &dmeta()).unwrap();
            assert_eq!(dec, bytes, "n={n}: must be bit-exact");
        }
    }

    #[test]
    fn bitplane_index_width_is_ceil_log2() {
        assert_eq!(index_width(1), 0);
        assert_eq!(index_width(2), 1);
        assert_eq!(index_width(3), 2);
        assert_eq!(index_width(4), 2);
        assert_eq!(index_width(5), 3);
        assert_eq!(index_width(32), 5);
        assert_eq!(index_width(33), 6);
        assert_eq!(index_width(64), 6);
    }
}
