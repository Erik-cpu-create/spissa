// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

//! rtc-reeborn-for: frame-of-reference fixed-width exponent codec (coderless).
//!
//! REEBORN edge-decode path. Instead of an entropy coder (rANS/Huffman) with a per-symbol
//! state update + table gather + renormalization, pack each exponent as a fixed `width`-bit
//! offset from a per-tensor `base`. Decode is a branch-light fixed-width unpack — no divide,
//! no table lookup, no renorm — so it is far cheaper per symbol and cleanly NEON-vectorizable.
//! The trade is ratio: `width` bits/exp (e.g. 5) vs rANS ~2.6, for much higher decode
//! throughput in the model>RAM streaming regime where decode bandwidth is the wall.

use crate::codec::{DecodeMeta, EncodeMeta, EncodedChunk, TensorCodec};
use crate::dfloat::{join_bf16, split_bf16};
use crate::error::{CodecError, Result};

/// Encode `symbols` as fixed-`width`-bit offsets from `base` (MSB-first bit packing).
/// Requires every symbol in `[base, base + 2^width - 1]`; the caller picks base/width to
/// cover the tensor's exponent range (widen by a bit to absorb any outlier).
pub fn for_encode(symbols: &[u8], base: u8, width: u32) -> Vec<u8> {
    debug_assert!((1..=8).contains(&width));
    let mut out = Vec::with_capacity((symbols.len() * width as usize).div_ceil(8));
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for &s in symbols {
        debug_assert!(
            s >= base && ((s - base) as u16) < (1u16 << width),
            "symbol out of FOR range"
        );
        acc = (acc << width) | (s - base) as u64;
        nbits += width;
        while nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    if nbits > 0 {
        out.push((acc << (8 - nbits)) as u8);
    }
    out
}

/// Decode `n` fixed-`width`-bit offsets from `stream` into `out`, adding `base`.
/// Inverse of [`for_encode`]; branch-light, no table/state — the edge-decode lever.
pub fn for_decode_into(stream: &[u8], n: usize, base: u8, width: u32, out: &mut [u8]) {
    let mask: u64 = (1u64 << width) - 1;
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    let mut pos: usize = 0;
    for o in out.iter_mut().take(n) {
        while nbits < width {
            acc = (acc << 8) | stream[pos] as u64;
            pos += 1;
            nbits += 8;
        }
        nbits -= width;
        *o = base.wrapping_add(((acc >> nbits) & mask) as u8);
    }
}

/// Faster FOR decode: unpacks 8 codes per iteration. 8 codes × W bits = 8·W bits = exactly W
/// bytes, so every group of 8 is byte-aligned — the body reads W bytes into a u64 and extracts
/// 8 codes with fixed shifts, branch-free (the autovectorizable / NEON-friendly form). The tail
/// (n mod 8) uses the bit-reader. Bit-identical to [`for_decode_into`].
pub fn for_decode8_into(stream: &[u8], n: usize, base: u8, width: u32, out: &mut [u8]) {
    let mask = (1u64 << width) - 1;
    let w = width as usize;
    let groups = n / 8;
    let mut pos = 0usize;
    for g in 0..groups {
        let mut acc: u64 = 0;
        for k in 0..w {
            acc = (acc << 8) | stream[pos + k] as u64;
        }
        pos += w;
        let o = g * 8;
        for i in 0..8usize {
            out[o + i] = base.wrapping_add(((acc >> (width * (7 - i as u32))) & mask) as u8);
        }
    }
    // tail: the remaining n mod 8 codes are packed continuously from `pos` (group boundaries
    // are byte-aligned, so pos = groups*w here).
    let done = groups * 8;
    if done < n {
        let mut acc: u64 = 0;
        let mut nbits: u32 = 0;
        for o in out.iter_mut().take(n).skip(done) {
            while nbits < width {
                acc = (acc << 8) | stream[pos] as u64;
                pos += 1;
                nbits += 8;
            }
            nbits -= width;
            *o = base.wrapping_add(((acc >> nbits) & mask) as u8);
        }
    }
}

/// FOR with escape: width-W code of `sym - base` in `[0, 2^W-2]`, or the all-ones escape
/// (`2^W-1`) followed by the raw 8-bit symbol for out-of-window values. Avoids inflating
/// `width` for rare outlier exponents, so the ratio is much closer to the entropy floor,
/// while staying coderless (no table/state) — just one predictable branch per symbol.
pub fn for_escape_encode(symbols: &[u8], base: u8, width: u32) -> Vec<u8> {
    debug_assert!((1..=8).contains(&width));
    let span = (1u64 << width) - 1; // reserved escape sentinel
    let mut out = Vec::new();
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    macro_rules! put {
        ($val:expr, $w:expr) => {{
            acc = (acc << $w) | ($val as u64);
            nbits += $w;
            while nbits >= 8 {
                nbits -= 8;
                out.push((acc >> nbits) as u8);
            }
        }};
    }
    for &s in symbols {
        let d = s.wrapping_sub(base) as u64;
        if d < span {
            put!(d, width);
        } else {
            put!(span, width);
            put!(s, 8u32);
        }
    }
    if nbits > 0 {
        out.push((acc << (8 - nbits)) as u8);
    }
    out
}

/// Decode `n` symbols from a [`for_escape_encode`] stream into `out`. Inverse; coderless.
pub fn for_escape_decode_into(stream: &[u8], n: usize, base: u8, width: u32, out: &mut [u8]) {
    let span = (1u64 << width) - 1;
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    let mut pos: usize = 0;
    macro_rules! get {
        ($w:expr) => {{
            while nbits < $w {
                acc = (acc << 8) | stream[pos] as u64;
                pos += 1;
                nbits += 8;
            }
            nbits -= $w;
            (acc >> nbits) & ((1u64 << $w) - 1)
        }};
    }
    for o in out.iter_mut().take(n) {
        let code = get!(width);
        *o = if code == span {
            get!(8u32) as u8
        } else {
            base.wrapping_add(code as u8)
        };
    }
}

/// rtc-reeborn-for-v1: lossless bf16 codec for the EDGE / model>RAM streaming regime.
///
/// Raw 8-bit significand + per-tensor pure fixed-width FOR exponent (no entropy coder) → a
/// branch-free, NEON-friendly decode that stays read-bound where rANS/Huffman are decode-bound.
/// Larger b/w than rANS (~13 vs ~10.6 on Llama-1B) but ~6× faster decode — the trade that wins
/// in the model>RAM regime (validated in research/trials + docs/benchmarks/trials).
pub struct ReebornForCodec;

impl ReebornForCodec {
    pub const ID: &'static str = "rtc-reeborn-for-v1";
    const FLAG_FOR: u8 = 0; // bf16 path: raw significand + FOR exponent
    const FLAG_RAW: u8 = 1; // non-bf16 / odd-length fallback: bytes stored verbatim

    /// Per-tensor base = min exponent, width = ceil(log2(range)) covering the full range
    /// losslessly (no escape). width ∈ [1, 8].
    fn pick_base_width(exps: &[u8]) -> (u8, u32) {
        let lo = exps.iter().copied().min().unwrap_or(0);
        let hi = exps.iter().copied().max().unwrap_or(0);
        let range = (hi - lo) as u32 + 1;
        (lo, (32 - (range - 1).leading_zeros()).max(1))
    }
}

impl TensorCodec for ReebornForCodec {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn encode(&self, input: &[u8], _meta: &EncodeMeta) -> Result<EncodedChunk> {
        // Odd-length chunks fall back to raw (FLAG_RAW). Even-length bytes are treated as bf16
        // bit patterns — split_bf16 is bijective on any u16, so this stays lossless for ANY tensor
        // (dtype-string independent, like rtc-rans-v1); the smallest-lossless picker keeps RawCodec
        // when FOR does not shrink the chunk.
        if !input.len().is_multiple_of(2) {
            let mut data = Vec::with_capacity(1 + input.len());
            data.push(Self::FLAG_RAW);
            data.extend_from_slice(input);
            return Ok(EncodedChunk {
                codec_id: Self::ID.to_string(),
                data,
                original_size: input.len() as u64,
            });
        }
        let n = input.len() / 2;
        let mut exps = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        for w in input.chunks_exact(2) {
            let (e, r) = split_bf16(u16::from_le_bytes([w[0], w[1]]));
            exps.push(e);
            residuals.push(r);
        }
        let (base, width) = Self::pick_base_width(&exps);
        let stream = for_encode(&exps, base, width);
        // layout: FLAG_FOR | n(u64) | base(u8) | width(u8) | stream_len(u64) | for_stream | residuals
        let mut data = Vec::with_capacity(19 + stream.len() + residuals.len());
        data.push(Self::FLAG_FOR);
        data.extend_from_slice(&(n as u64).to_le_bytes());
        data.push(base);
        data.push(width as u8);
        data.extend_from_slice(&(stream.len() as u64).to_le_bytes());
        data.extend_from_slice(&stream);
        data.extend_from_slice(&residuals);
        Ok(EncodedChunk {
            codec_id: Self::ID.to_string(),
            data,
            original_size: input.len() as u64,
        })
    }

    fn decode(&self, encoded: &[u8], _meta: &DecodeMeta) -> Result<Vec<u8>> {
        let err = || CodecError::InvalidData("truncated rtc-reeborn-for-v1 chunk".into());
        let flag = *encoded.first().ok_or_else(err)?;
        let body = &encoded[1..];
        if flag == Self::FLAG_RAW {
            return Ok(body.to_vec());
        }
        if body.len() < 18 {
            return Err(err());
        }
        let n = u64::from_le_bytes(body[0..8].try_into().unwrap()) as usize;
        let base = body[8];
        let width = body[9] as u32;
        if !(1..=8).contains(&width) {
            return Err(CodecError::InvalidData(
                "rtc-reeborn-for-v1: bad width".into(),
            ));
        }
        let slen = u64::from_le_bytes(body[10..18].try_into().unwrap()) as usize;
        if slen < (n * width as usize).div_ceil(8) {
            return Err(err()); // stream too short for n width-bit codes
        }
        let stream = body.get(18..18 + slen).ok_or_else(err)?;
        let residuals = body.get(18 + slen..18 + slen + n).ok_or_else(err)?;
        let mut exps = vec![0u8; n];
        for_decode8_into(stream, n, base, width, &mut exps); // branch-free group-of-8, 1.44x faster
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            out.extend_from_slice(&join_bf16(exps[i], residuals[i]).to_le_bytes());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_roundtrip_fixed_width() {
        for &(base, width) in &[(104u8, 5u32), (96, 6), (0, 8), (120, 3)] {
            let span = 1u32 << width;
            let mut state = 0x1111_2222u32;
            let syms: Vec<u8> = (0..5000)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    base.wrapping_add((state % span) as u8)
                })
                .collect();
            let enc = for_encode(&syms, base, width);
            let mut out = vec![0u8; syms.len()];
            for_decode_into(&enc, syms.len(), base, width, &mut out);
            assert_eq!(
                out, syms,
                "FOR roundtrip must be bit-exact (base={base}, width={width})"
            );
        }
    }

    #[test]
    fn for_decode8_matches_scalar() {
        for &(base, width) in &[(104u8, 5u32), (96, 6), (110, 4), (0, 8), (120, 3)] {
            for &len in &[0usize, 1, 7, 8, 9, 16, 17, 23, 1000, 10003] {
                let span = 1u32 << width;
                let mut state = 0x5151_2626u32;
                let syms: Vec<u8> = (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        base.wrapping_add((state % span) as u8)
                    })
                    .collect();
                let enc = for_encode(&syms, base, width);
                let mut a = vec![0u8; len];
                let mut b = vec![0u8; len];
                for_decode_into(&enc, len, base, width, &mut a);
                for_decode8_into(&enc, len, base, width, &mut b);
                assert_eq!(
                    a, syms,
                    "scalar decode wrong (base={base},width={width},len={len})"
                );
                assert_eq!(
                    b, syms,
                    "decode8 wrong (base={base},width={width},len={len})"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn reeborn_for_decode8_bench() {
        use std::time::Instant;
        let n = 40_000_000usize;
        let (base, width) = (104u8, 5u32);
        let span = 1u32 << width;
        let mut state = 0x2468_ACE0u32;
        let syms: Vec<u8> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                base.wrapping_add((state % span) as u8)
            })
            .collect();
        let enc = for_encode(&syms, base, width);
        let reps = 3usize;
        let gw = |s: f64| (n as f64 * reps as f64 / 1e9) / s;
        let mut out = vec![0u8; n];
        let t0 = Instant::now();
        for _ in 0..reps {
            for_decode_into(&enc, n, base, width, &mut out);
            std::hint::black_box(&out);
        }
        let scalar = gw(t0.elapsed().as_secs_f64());
        let t1 = Instant::now();
        for _ in 0..reps {
            for_decode8_into(&enc, n, base, width, &mut out);
            std::hint::black_box(&out);
        }
        let d8 = gw(t1.elapsed().as_secs_f64());
        assert_eq!(&out[..n], &syms[..]);
        eprintln!("\n########## REEBORN-FOR decode8 ({n} exp, w={width}) ##########");
        eprintln!("  for_decode_into  (per-code) = {scalar:.3} Gweight/s/core");
        eprintln!(
            "  for_decode8_into (group-8)  = {d8:.3} Gweight/s/core  ({:.2}x)",
            d8 / scalar
        );
        eprintln!("##########");
    }

    #[test]
    fn for_escape_roundtrip() {
        for &(base, width) in &[(112u8, 4u32), (118, 3), (100, 5)] {
            let mut state = 0x9999_7777u32;
            let syms: Vec<u8> = (0..8000)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (state & 0xFF) as u8 // full 0..255 range => exercises the escape path
                })
                .collect();
            let enc = for_escape_encode(&syms, base, width);
            let mut out = vec![0u8; syms.len()];
            for_escape_decode_into(&enc, syms.len(), base, width, &mut out);
            assert_eq!(
                out, syms,
                "FOR-escape roundtrip must be bit-exact (base={base}, width={width})"
            );
        }
    }

    // Does the escape variant (best ratio, ~11.7 b/w on Llama) keep the fast coderless decode?
    // Run: cargo test -p rtc-codec reeborn_for_escape_decode_bench -- --ignored --nocapture --release
    #[test]
    #[ignore]
    fn reeborn_for_escape_decode_bench() {
        use std::time::Instant;
        let n = 40_000_000usize;
        let mut state = 0x2468_ACE0u32;
        let syms: Vec<u8> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if state % 100 < 70 {
                    (118 + (state % 3)) as u8
                } else {
                    (104 + (state % 24)) as u8
                }
            })
            .collect();
        let (base, width) = (112u8, 4u32); // window [112,126]
        let reps = 3usize;
        let gw = |secs: f64| (n as f64 * reps as f64 / 1e9) / secs;
        let mut out = vec![0u8; n];
        let enc = for_escape_encode(&syms, base, width);
        let bits = (enc.len() as f64 * 8.0) / n as f64;
        let span = (1u32 << width) - 1;
        let esc = syms
            .iter()
            .filter(|&&s| (s.wrapping_sub(base) as u32) >= span)
            .count();
        let t0 = Instant::now();
        for _ in 0..reps {
            for_escape_decode_into(&enc, n, base, width, &mut out);
            std::hint::black_box(&out);
        }
        let g = gw(t0.elapsed().as_secs_f64());
        assert_eq!(&out[..n], &syms[..], "escape decode must be bit-exact");
        eprintln!("\n########## REEBORN-FOR escape decode ({n} exp symbols) ##########");
        eprintln!("  width {width} + escape ({:.1}% escape): {bits:.2} bits/exp, decode {g:.3} Gweight/s/core",
            100.0 * esc as f64 / n as f64);
        eprintln!("  context: pure-5bit 5.00 bits @ ~1.80 Gw/s | rANS-4lane 3.15 bits @ 0.29 Gw/s");
        eprintln!("##########");
    }

    // REEBORN edge thesis: coderless FOR decode vs 4-lane rANS — ratio cost vs decode speed.
    // Run: cargo test -p rtc-codec reeborn_for_vs_rans_decode_bench -- --ignored --nocapture --release
    #[test]
    #[ignore]
    fn reeborn_for_vs_rans_decode_bench() {
        use crate::{
            count_symbols, normalize_freqs, rans_build_tables, rans_decode_interleaved4_into,
            rans_encode_interleaved4,
        };
        use std::time::Instant;
        let n = 40_000_000usize;
        let mut state = 0x2468_ACE0u32;
        let syms: Vec<u8> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if state % 100 < 70 {
                    (118 + (state % 3)) as u8
                } else {
                    (104 + (state % 24)) as u8
                }
            })
            .collect();
        let (base, width) = (104u8, 5u32); // covers 104..135
        let reps = 3usize;
        let gw = |secs: f64| (n as f64 * reps as f64 / 1e9) / secs;
        let mut out = vec![0u8; n];

        let fstream = for_encode(&syms, base, width);
        let for_bits = (fstream.len() as f64 * 8.0) / n as f64;
        let t0 = Instant::now();
        for _ in 0..reps {
            for_decode_into(&fstream, n, base, width, &mut out);
            std::hint::black_box(&out);
        }
        let for_gw = gw(t0.elapsed().as_secs_f64());
        assert_eq!(&out[..n], &syms[..], "FOR decode must be bit-exact");

        let freq = normalize_freqs(&count_symbols(&syms));
        let s4 = rans_encode_interleaved4(&syms, &freq);
        let rans_bits = (s4.iter().map(|l| l.len()).sum::<usize>() as f64 * 8.0) / n as f64;
        let t = rans_build_tables(&freq);
        let sl4 = [&s4[0][..], &s4[1][..], &s4[2][..], &s4[3][..]];
        let t1 = Instant::now();
        for _ in 0..reps {
            rans_decode_interleaved4_into(sl4, n, &t, &mut out);
            std::hint::black_box(&out);
        }
        let rans_gw = gw(t1.elapsed().as_secs_f64());

        eprintln!("\n########## REEBORN-FOR vs rANS decode ({n} exp symbols) ##########");
        eprintln!("  FOR  (coderless, {width}-bit): {for_bits:.2} bits/exp, decode {for_gw:.3} Gweight/s/core");
        eprintln!("  rANS (4-lane)               : {rans_bits:.2} bits/exp, decode {rans_gw:.3} Gweight/s/core");
        eprintln!(
            "  -> FOR decodes {:.2}x faster, at {:.2}x the exponent bits",
            for_gw / rans_gw,
            for_bits / rans_bits
        );
        eprintln!("##########");
    }

    #[test]
    fn reeborn_for_codec_bf16_roundtrip_and_compresses() {
        let mut state = 0xC0DE_1234u32;
        let n = 50_000usize;
        let mut bytes = Vec::with_capacity(n * 2);
        for k in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let exp = ((110 + (k % 18)) as u16) & 0xFF; // 18 exponents -> width 5
            let bits = (((state >> 31) & 1) as u16) << 15 | (exp << 7) | (state & 0x7F) as u16;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let meta = EncodeMeta {
            name: "w".into(),
            shape: vec![n as u64],
            dtype: "bf16".into(),
        };
        let enc = ReebornForCodec.encode(&bytes, &meta).unwrap();
        assert_eq!(enc.codec_id, "rtc-reeborn-for-v1");
        assert!(
            enc.data.len() < bytes.len(),
            "must compress: {} >= {}",
            enc.data.len(),
            bytes.len()
        );
        let dmeta = DecodeMeta {
            codec_id: enc.codec_id.clone(),
            uncompressed_size: bytes.len() as u64,
        };
        assert_eq!(
            ReebornForCodec.decode(&enc.data, &dmeta).unwrap(),
            bytes,
            "bf16 roundtrip must be bit-exact"
        );
    }

    #[test]
    fn reeborn_for_codec_constant_and_empty() {
        let mk = |n: u64| EncodeMeta {
            name: "w".into(),
            shape: vec![n],
            dtype: "bf16".into(),
        };
        let mut constant = Vec::new();
        for _ in 0..1000 {
            constant.extend_from_slice(&0x3F80u16.to_le_bytes()); // 1.0, single exponent (width 1)
        }
        for bytes in [Vec::<u8>::new(), constant] {
            let enc = ReebornForCodec
                .encode(&bytes, &mk((bytes.len() / 2) as u64))
                .unwrap();
            let dmeta = DecodeMeta {
                codec_id: enc.codec_id.clone(),
                uncompressed_size: bytes.len() as u64,
            };
            assert_eq!(
                ReebornForCodec.decode(&enc.data, &dmeta).unwrap(),
                bytes,
                "constant/empty roundtrip"
            );
        }
    }

    #[test]
    fn reeborn_for_codec_raw_fallback_roundtrip() {
        // non-bf16 (u8) and odd-length chunks must round-trip via the raw fallback.
        let bytes: Vec<u8> = (0..1235u32).map(|x| (x * 7) as u8).collect();
        let meta = EncodeMeta {
            name: "idx".into(),
            shape: vec![bytes.len() as u64],
            dtype: "u8".into(),
        };
        let enc = ReebornForCodec.encode(&bytes, &meta).unwrap();
        let dmeta = DecodeMeta {
            codec_id: enc.codec_id.clone(),
            uncompressed_size: bytes.len() as u64,
        };
        assert_eq!(
            ReebornForCodec.decode(&enc.data, &dmeta).unwrap(),
            bytes,
            "raw-fallback roundtrip"
        );
    }
}
