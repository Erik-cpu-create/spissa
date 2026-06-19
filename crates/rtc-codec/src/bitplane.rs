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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_w5_neon_inner(
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    n: usize,
    out: &mut [u8],
) {
    use std::arch::aarch64::*;

    // Palette padded to 32 bytes for vtbl4 (indices < palette.len() <= 32).
    let mut pal = [0u8; 32];
    pal[..palette.len()].copy_from_slice(palette);
    let pal_tbl = uint8x8x4_t(
        vld1_u8(pal.as_ptr()),
        vld1_u8(pal.as_ptr().add(8)),
        vld1_u8(pal.as_ptr().add(16)),
        vld1_u8(pal.as_ptr().add(24)),
    );

    // For 8 indices (5 bits each) packed MSB-first in 5 bytes: index j occupies
    // bits [5j, 5j+5); read a 2-byte big-endian window at byte 5j/8, then
    // right-shift by (11 - 5j%8) and mask 0x1f.
    let bidx_hi: [u8; 8] = [0, 0, 1, 1, 2, 3, 3, 4];
    let bidx_lo: [u8; 8] = [1, 1, 2, 2, 3, 4, 4, 5];
    let neg_shift: [i16; 8] = [-11, -6, -9, -4, -7, -10, -5, -8];
    let vhi = vld1_u8(bidx_hi.as_ptr());
    let vlo = vld1_u8(bidx_lo.as_ptr());
    let vshift = vld1q_s16(neg_shift.as_ptr());
    let mask5 = vdupq_n_u16(0x1f);
    let mask80 = vdupq_n_u16(0x80);
    let mask7f = vdupq_n_u16(0x7f);

    // SIMD-safe groups: each iteration loads 8 bytes at offset g*5, so require
    // g*5 + 8 <= idx_plane.len(); also g*8 + 8 <= n.
    let groups_by_bytes = if idx_plane.len() >= 8 { (idx_plane.len() - 8) / 5 + 1 } else { 0 };
    let simd_groups = core::cmp::min(n / 8, groups_by_bytes);

    let out_u16 = out.as_mut_ptr() as *mut u16;
    for g in 0..simd_groups {
        let grp = vld1_u8(idx_plane.as_ptr().add(g * 5)); // 8 bytes (need up to byte 5)
        let hi = vtbl1_u8(grp, vhi);
        let lo = vtbl1_u8(grp, vlo);
        let window = vorrq_u16(vshlq_n_u16(vmovl_u8(hi), 8), vmovl_u8(lo));
        let idx16 = vandq_u16(vshlq_u16(window, vshift), mask5);
        let idx8 = vmovn_u16(idx16);
        let exp8 = vtbl4_u8(pal_tbl, idx8);
        let res8 = vld1_u8(residuals.as_ptr().add(g * 8));
        let res16 = vmovl_u8(res8);
        let exp16 = vmovl_u8(exp8);
        let sign = vshlq_n_u16(vandq_u16(res16, mask80), 8);
        let ep = vshlq_n_u16(exp16, 7);
        let mant = vandq_u16(res16, mask7f);
        let bf16 = vorrq_u16(vorrq_u16(sign, ep), mant);
        vst1q_u16(out_u16.add(g * 8), bf16);
    }

    // Scalar tail: weights [simd_groups*8 .. n]. simd_groups*8 is a multiple of 8,
    // so its bit offset (×5) is a multiple of 40 bits = 5 bytes => byte-aligned.
    let tail_start = simd_groups * 8;
    if tail_start < n {
        let byte_off = (tail_start / 8) * 5;
        let mut reader = BufferedBitReader::new(&idx_plane[byte_off..]);
        for i in tail_start..n {
            reader.refill();
            let idx = reader.peek(5) as usize;
            reader.consume(5);
            let bits = join_bf16(palette[idx], residuals[i]);
            out[2 * i] = bits as u8;
            out[2 * i + 1] = (bits >> 8) as u8;
        }
    }
}

/// NEON `w=5` bit-plane decode to bf16 bytes. Bit-identical to scalar `decode`.
#[cfg(target_arch = "aarch64")]
pub fn decode_neon_w5(palette: &[u8], idx_plane: &[u8], residuals: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * 2];
    unsafe { decode_w5_neon_inner(palette, idx_plane, residuals, n, &mut out) };
    out
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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn decode_neon_w5_matches_scalar_bit_for_bit() {
        let codec = BitplaneCodec;
        // Sizes >= 32 so make_bf16(32, n) yields all 32 distinct exponents (w=5),
        // covering tail cases (n%8 in {0,1,7,3}) and the SIMD/scalar boundary.
        for &n in &[32usize, 33, 39, 40, 47, 64, 1000, 4096, 4099] {
            let bytes = make_bf16(32, n);
            let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
            let enc = codec.encode(&bytes, &meta).unwrap();
            assert_eq!(&enc.data[0..4], b"RTCB");
            let p = enc.data[14] as usize;
            let w = enc.data[15];
            assert_eq!(w, 5, "n={n}: expected w=5 for 32 exponents");
            let mut off = 16;
            let palette = &enc.data[off..off + p];
            off += p;
            let idx_bytes = (n * 5 + 7) / 8;
            let idx_plane = &enc.data[off..off + idx_bytes];
            off += idx_bytes;
            let residuals = &enc.data[off..off + n];

            let scalar = codec
                .decode(
                    &enc.data,
                    &DecodeMeta { codec_id: "rtc-bitplane-v1".into(), uncompressed_size: 0 },
                )
                .unwrap();
            let neon = decode_neon_w5(palette, idx_plane, residuals, n);
            assert_eq!(neon, scalar, "n={n}: NEON decode must equal scalar bit-for-bit");
            assert_eq!(neon, bytes, "n={n}: NEON decode must be lossless");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn bitplane_neon_decode_feasibility() {
        let bytes = std::fs::read("/tmp/rllm-bf16-sample.bin")
            .expect("run dump_bf16_embedding_sample first (see plan Task 3, Step 2)");
        let n = bytes.len() / 2;
        let codec = BitplaneCodec;
        let meta = EncodeMeta { name: "embed".into(), shape: vec![n as u64], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &meta).unwrap();
        let bits_per_weight = (enc.data.len() as f64 * 8.0) / n as f64;
        let p = enc.data[14] as usize;
        let w = enc.data[15];
        assert_eq!(w, 5, "expected w=5 (32 exponents) for the real embedding");
        let mut off = 16;
        let palette = enc.data[off..off + p].to_vec();
        off += p;
        let idx_bytes = (n * 5 + 7) / 8;
        let idx_plane = enc.data[off..off + idx_bytes].to_vec();
        off += idx_bytes;
        let residuals = enc.data[off..off + n].to_vec();

        // Correctness on the real sample.
        let neon = decode_neon_w5(&palette, &idx_plane, &residuals, n);
        assert_eq!(neon, bytes, "NEON decode must be lossless on the real embedding");

        // Timed NEON decode (materializing; the fused kernel would skip the store,
        // so this is a conservative floor).
        let iters = 8;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let d = decode_neon_w5(&palette, &idx_plane, &residuals, n);
            std::hint::black_box(&d);
        }
        let neon_s = t.elapsed().as_secs_f64() / iters as f64;
        let neon_gw = (n as f64 / 1e9) / neon_s;

        // Scalar bitplane decode for the speedup ratio (one pass).
        let t = std::time::Instant::now();
        let sc = codec
            .decode(
                &enc.data,
                &DecodeMeta { codec_id: "rtc-bitplane-v1".into(), uncompressed_size: 0 },
            )
            .unwrap();
        std::hint::black_box(&sc);
        let scalar_s = t.elapsed().as_secs_f64();
        let scalar_gw = (n as f64 / 1e9) / scalar_s;

        let agg = neon_gw * 3.5;
        let verdict = if agg >= 12.0 { "GO" } else if agg >= 5.0 { "MARGINAL" } else { "NO-GO" };
        eprintln!(
            "\n=== R143 REEPLANE bit-plane NEON decode FEASIBILITY ===\n\
             weights={n}  bits/weight={bits_per_weight:.3}  palette={p} w={w}\n\
             NEON single-core: {neon_gw:.2} Gweight/s  ({:.1} ms/decode, materializing)\n\
             scalar bitplane: {scalar_gw:.3} Gweight/s  (NEON speedup {:.1}x)\n\
             aggregate (x3.5): {agg:.1} Gweight/s\n\
             threshold: GO>=12, MARGINAL 5-12, NO-GO<5 (Gweight/s aggregate)\n\
             VERDICT: {verdict}\n",
            neon_s * 1000.0,
            neon_gw / scalar_gw,
        );
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
