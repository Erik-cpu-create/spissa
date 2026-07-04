// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

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

    /// Verbatim raw-fallback chunk (odd length or palette > 64). The header's u64 field
    /// holds the BYTE length so decode returns exactly `input` (works for odd input too).
    fn raw_chunk(input: &[u8]) -> EncodedChunk {
        let mut data = Vec::with_capacity(HEADER_LEN + input.len());
        data.extend_from_slice(b"RTCB");
        data.push(1);
        data.push(FLAG_RAW);
        data.extend_from_slice(&(input.len() as u64).to_le_bytes());
        data.push(0); // palette_len
        data.push(0); // index_width
        data.extend_from_slice(input);
        EncodedChunk {
            codec_id: Self::ID.to_string(),
            data,
            original_size: input.len() as u64,
        }
    }
}

const HEADER_LEN: usize = 4 + 1 + 1 + 8 + 1 + 1; // magic+version+flags+n+palette_len+width = 16
const FLAG_RAW: u8 = 0x01;

impl TensorCodec for BitplaneCodec {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn encode(&self, input: &[u8], meta: &EncodeMeta) -> Result<EncodedChunk> {
        let _ = meta; // dtype-agnostic: the (exp,residual) bf16 split is a bijection on any
                      // even-length bytes (pack passes dtype "u8"); it only compresses for
                      // real bf16. Odd length -> raw fallback.
        if !input.len().is_multiple_of(2) {
            return Ok(Self::raw_chunk(input));
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
        let palette: Vec<u8> = (0..256usize)
            .filter(|&i| present[i])
            .map(|i| i as u8)
            .collect();

        if palette.len() > 64 {
            // Not usefully palette-compressible → raw fallback.
            return Ok(Self::raw_chunk(input));
        }

        let mut data = Vec::with_capacity(HEADER_LEN + palette.len() + n * 2);
        data.extend_from_slice(b"RTCB");
        data.push(1); // version
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
            // n holds the BYTE length for raw chunks (see `raw_chunk`).
            let body = encoded.get(off..).ok_or_else(err)?;
            if body.len() < n {
                return Err(err());
            }
            return Ok(body[..n].to_vec());
        }

        let palette = encoded.get(off..off + p).ok_or_else(err)?;
        off += p;
        let idx_bytes = (n * w as usize).div_ceil(8);
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

        // R159: NEON fixed-width decode (~20× the scalar BufferedBitReader) — this is
        // why bit-plane is the FAST lossless codec for inference vs rANS's sequential
        // entropy decode. decode_bitplane_row_into dispatches w=5/6 to SIMD, else scalar.
        #[cfg(target_arch = "aarch64")]
        {
            decode_bitplane_row_into(palette, idx_plane, residuals, n, w, &mut out);
            Ok(out)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
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
    let groups_by_bytes = if idx_plane.len() >= 8 {
        (idx_plane.len() - 8) / 5 + 1
    } else {
        0
    };
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
    decode_neon_w5_into(palette, idx_plane, residuals, n, &mut out);
    out
}

/// NEON `w=5` bit-plane decode into a caller-provided buffer (`out.len() >= n*2`),
/// no allocation. Used by the fused per-row GEMV so decode lands in an L1 scratch.
#[cfg(target_arch = "aarch64")]
pub fn decode_neon_w5_into(
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    n: usize,
    out: &mut [u8],
) {
    assert!(out.len() >= n * 2, "decode_neon_w5_into: out too small");
    unsafe { decode_w5_neon_inner(palette, idx_plane, residuals, n, out) };
}

/// R146 SCOUT: 16-wide w=5 decode (vqtbl2q gathers 16 exponents/lookup vs the
/// 8-wide vtbl4's 8). Processes 16 weights/iter; scalar tail via the 8-wide path.
///
/// # Safety
/// The caller must ensure the CPU supports NEON (guaranteed on `aarch64`, but the
/// `#[target_feature(enable = "neon")]` contract still requires it), and that
/// `palette`, `idx_plane`, `residuals`, and `out` are sized consistently with `n`
/// for the w=5 layout — the function reads/writes those buffers without bounds checks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn decode16_w5_into(
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    n: usize,
    out: &mut [u8],
) {
    use std::arch::aarch64::*;
    // Palette in 2x uint8x16 for vqtbl2q (indices < 32).
    let mut pal = [0u8; 32];
    pal[..palette.len()].copy_from_slice(palette);
    let pal2 = uint8x16x2_t(vld1q_u8(pal.as_ptr()), vld1q_u8(pal.as_ptr().add(16)));

    // 16 indices (5 bits) packed in 10 bytes; lane j reads a 2-byte window at
    // byte 5j/8, right-shift (11 - 5j%8), mask 0x1f. The shift pattern repeats
    // every 8 lanes (5*8=40 is byte-aligned).
    let bidx_hi: [u8; 16] = [0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9];
    let bidx_lo: [u8; 16] = [1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 9, 9, 10];
    let neg_shift: [i16; 8] = [-11, -6, -9, -4, -7, -10, -5, -8];
    let vhi = vld1q_u8(bidx_hi.as_ptr());
    let vlo = vld1q_u8(bidx_lo.as_ptr());
    let vshift = vld1q_s16(neg_shift.as_ptr());
    let mask5 = vdupq_n_u16(0x1f);
    let mask80 = vdupq_n_u16(0x80);
    let mask7f = vdupq_n_u16(0x7f);

    // 16-group is safe while its 16-byte load stays in bounds (needs up to byte 10).
    let groups16 = if idx_plane.len() >= 16 {
        (idx_plane.len() - 16) / 10 + 1
    } else {
        0
    };
    let simd16 = core::cmp::min(n / 16, groups16);
    let out_u16 = out.as_mut_ptr() as *mut u16;

    for g in 0..simd16 {
        let grp = vld1q_u8(idx_plane.as_ptr().add(g * 10));
        let hi = vqtbl1q_u8(grp, vhi);
        let lo = vqtbl1q_u8(grp, vlo);
        // 16-bit windows, low and high halves.
        let win_lo = vorrq_u16(
            vshlq_n_u16(vmovl_u8(vget_low_u8(hi)), 8),
            vmovl_u8(vget_low_u8(lo)),
        );
        let win_hi = vorrq_u16(
            vshlq_n_u16(vmovl_u8(vget_high_u8(hi)), 8),
            vmovl_u8(vget_high_u8(lo)),
        );
        let idx_lo = vandq_u16(vshlq_u16(win_lo, vshift), mask5);
        let idx_hi = vandq_u16(vshlq_u16(win_hi, vshift), mask5);
        let idx16 = vcombine_u8(vmovn_u16(idx_lo), vmovn_u16(idx_hi));
        let exp16 = vqtbl2q_u8(pal2, idx16);
        let res16 = vld1q_u8(residuals.as_ptr().add(g * 16));
        // reconstruct bf16 in two halves
        let res_lo = vmovl_u8(vget_low_u8(res16));
        let exp_lo = vmovl_u8(vget_low_u8(exp16));
        let bf_lo = vorrq_u16(
            vorrq_u16(
                vshlq_n_u16(vandq_u16(res_lo, mask80), 8),
                vshlq_n_u16(exp_lo, 7),
            ),
            vandq_u16(res_lo, mask7f),
        );
        let res_hi = vmovl_u8(vget_high_u8(res16));
        let exp_hi = vmovl_u8(vget_high_u8(exp16));
        let bf_hi = vorrq_u16(
            vorrq_u16(
                vshlq_n_u16(vandq_u16(res_hi, mask80), 8),
                vshlq_n_u16(exp_hi, 7),
            ),
            vandq_u16(res_hi, mask7f),
        );
        vst1q_u16(out_u16.add(g * 16), bf_lo);
        vst1q_u16(out_u16.add(g * 16 + 8), bf_hi);
    }

    // Scalar/8-wide tail for the remaining weights, reusing the proven path.
    let done = simd16 * 16;
    if done < n {
        let byte_off = (done / 8) * 5; // done is a multiple of 16 => byte-aligned
        decode_w5_neon_inner(
            palette,
            &idx_plane[byte_off..],
            &residuals[done..],
            n - done,
            &mut out[done * 2..],
        );
    }
}

/// Scalar `w`-bit bit-plane row decode (any `w` in 1..=8). The lossless reference
/// the SIMD kernels reproduce; used as their byte-aligned tail and as the
/// dispatcher fallback for widths without a SIMD path. Mirrors the inner loop of
/// `BitplaneCodec::decode` exactly (MSB-first `BufferedBitReader`).
///
/// Only the aarch64 SIMD dispatch (`decode_bitplane_row_into` / `decode16_w6_into`)
/// calls this; on other targets `BitplaneCodec::decode` inlines the scalar loop, so
/// gate it to aarch64 to avoid a dead-code warning there.
#[cfg(target_arch = "aarch64")]
fn decode_scalar_w(
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    n: usize,
    w: u8,
    out: &mut [u8],
) {
    let mut reader = BufferedBitReader::new(idx_plane);
    for i in 0..n {
        reader.refill();
        let idx = reader.peek(w) as usize;
        reader.consume(w);
        let bits = join_bf16(palette[idx], residuals[i]);
        out[2 * i] = bits as u8;
        out[2 * i + 1] = (bits >> 8) as u8;
    }
}

/// REEPLANE-W6 (R149b): 16-wide w=6 bit-plane decode for palettes of 33–64
/// exponents. Same shape as `decode16_w5_into`, but the index layout uses the w=6
/// period (4 indices = 24 bits = 3 bytes ⇒ 16 indices = 12-byte group stride) and
/// exponents are gathered from a **64-entry** palette via `vqtbl4q_u8` (a single
/// TBL over 4 registers) — `vqtbl2q`'s 32-entry table cannot reach indices ≥ 32.
/// Bit-identical to scalar `BitplaneCodec::decode` for w=6.
///
/// # Safety
/// The caller must ensure the CPU supports NEON (guaranteed on `aarch64`, but the
/// `#[target_feature(enable = "neon")]` contract still requires it), and that
/// `palette`, `idx_plane`, `residuals`, and `out` are sized consistently with `n`
/// for the w=6 layout — the function reads/writes those buffers without bounds checks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn decode16_w6_into(
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    n: usize,
    out: &mut [u8],
) {
    use std::arch::aarch64::*;
    // Palette in 4x uint8x16 for vqtbl4q (indices < 64).
    let mut pal = [0u8; 64];
    pal[..palette.len()].copy_from_slice(palette);
    let pal4 = uint8x16x4_t(
        vld1q_u8(pal.as_ptr()),
        vld1q_u8(pal.as_ptr().add(16)),
        vld1q_u8(pal.as_ptr().add(32)),
        vld1q_u8(pal.as_ptr().add(48)),
    );

    // 16 indices (6 bits) packed in 12 bytes; lane j reads a 2-byte big-endian
    // window at byte 6j/8, right-shift (10 - 6j%8), mask 0x3f. The shift pattern
    // has period 4 (6*4 = 24 = 3 bytes byte-aligned), so an 8-entry table repeats.
    let bidx_hi: [u8; 16] = [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11];
    let bidx_lo: [u8; 16] = [1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12];
    let neg_shift: [i16; 8] = [-10, -4, -6, -8, -10, -4, -6, -8];
    let vhi = vld1q_u8(bidx_hi.as_ptr());
    let vlo = vld1q_u8(bidx_lo.as_ptr());
    let vshift = vld1q_s16(neg_shift.as_ptr());
    let mask6 = vdupq_n_u16(0x3f);
    let mask80 = vdupq_n_u16(0x80);
    let mask7f = vdupq_n_u16(0x7f);

    // 16-group is safe while its 16-byte load stays in bounds (needs up to byte 12).
    let groups16 = if idx_plane.len() >= 16 {
        (idx_plane.len() - 16) / 12 + 1
    } else {
        0
    };
    let simd16 = core::cmp::min(n / 16, groups16);
    let out_u16 = out.as_mut_ptr() as *mut u16;

    for g in 0..simd16 {
        let grp = vld1q_u8(idx_plane.as_ptr().add(g * 12));
        let hi = vqtbl1q_u8(grp, vhi);
        let lo = vqtbl1q_u8(grp, vlo);
        let win_lo = vorrq_u16(
            vshlq_n_u16(vmovl_u8(vget_low_u8(hi)), 8),
            vmovl_u8(vget_low_u8(lo)),
        );
        let win_hi = vorrq_u16(
            vshlq_n_u16(vmovl_u8(vget_high_u8(hi)), 8),
            vmovl_u8(vget_high_u8(lo)),
        );
        let idx_lo = vandq_u16(vshlq_u16(win_lo, vshift), mask6);
        let idx_hi = vandq_u16(vshlq_u16(win_hi, vshift), mask6);
        let idx16 = vcombine_u8(vmovn_u16(idx_lo), vmovn_u16(idx_hi));
        let exp16 = vqtbl4q_u8(pal4, idx16);
        let res16 = vld1q_u8(residuals.as_ptr().add(g * 16));
        // reconstruct bf16 in two halves (identical to the w=5 path)
        let res_lo = vmovl_u8(vget_low_u8(res16));
        let exp_lo = vmovl_u8(vget_low_u8(exp16));
        let bf_lo = vorrq_u16(
            vorrq_u16(
                vshlq_n_u16(vandq_u16(res_lo, mask80), 8),
                vshlq_n_u16(exp_lo, 7),
            ),
            vandq_u16(res_lo, mask7f),
        );
        let res_hi = vmovl_u8(vget_high_u8(res16));
        let exp_hi = vmovl_u8(vget_high_u8(exp16));
        let bf_hi = vorrq_u16(
            vorrq_u16(
                vshlq_n_u16(vandq_u16(res_hi, mask80), 8),
                vshlq_n_u16(exp_hi, 7),
            ),
            vandq_u16(res_hi, mask7f),
        );
        vst1q_u16(out_u16.add(g * 16), bf_lo);
        vst1q_u16(out_u16.add(g * 16 + 8), bf_hi);
    }

    // Scalar tail for the remaining weights.
    let done = simd16 * 16;
    if done < n {
        let byte_off = done * 6 / 8; // done is a multiple of 16 => byte-aligned
        decode_scalar_w(
            palette,
            &idx_plane[byte_off..],
            &residuals[done..],
            n - done,
            6,
            &mut out[done * 2..],
        );
    }
}

/// Decode one bit-plane row of `n` weights at index width `w` into `out`
/// (`out.len() >= n*2`). The single width-dispatch entry point: SIMD for w∈{5,6}
/// (REEPLANE / REEPLANE-W6), scalar otherwise. Callers (e.g. the streaming GEMV)
/// stay width-agnostic — kernel selection lives here in the codec crate.
#[cfg(target_arch = "aarch64")]
pub fn decode_bitplane_row_into(
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    n: usize,
    w: u8,
    out: &mut [u8],
) {
    assert!(
        out.len() >= n * 2,
        "decode_bitplane_row_into: out too small"
    );
    match w {
        5 => unsafe { decode16_w5_into(palette, idx_plane, residuals, n, out) },
        6 => unsafe { decode16_w6_into(palette, idx_plane, residuals, n, out) },
        _ => decode_scalar_w(palette, idx_plane, residuals, n, w, out),
    }
}

#[cfg(test)]
#[path = "tests_bitplane.rs"]
mod tests;
