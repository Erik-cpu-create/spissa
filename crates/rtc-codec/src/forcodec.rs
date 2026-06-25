// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! rtc-reeborn-for: frame-of-reference fixed-width exponent codec (coderless).
//!
//! REEBORN edge-decode path. Instead of an entropy coder (rANS/Huffman) with a per-symbol
//! state update + table gather + renormalization, pack each exponent as a fixed `width`-bit
//! offset from a per-tensor `base`. Decode is a branch-light fixed-width unpack — no divide,
//! no table lookup, no renorm — so it is far cheaper per symbol and cleanly NEON-vectorizable.
//! The trade is ratio: `width` bits/exp (e.g. 5) vs rANS ~2.6, for much higher decode
//! throughput in the model>RAM streaming regime where decode bandwidth is the wall.

/// Encode `symbols` as fixed-`width`-bit offsets from `base` (MSB-first bit packing).
/// Requires every symbol in `[base, base + 2^width - 1]`; the caller picks base/width to
/// cover the tensor's exponent range (widen by a bit to absorb any outlier).
pub fn for_encode(symbols: &[u8], base: u8, width: u32) -> Vec<u8> {
    debug_assert!((1..=8).contains(&width));
    let mut out = Vec::with_capacity((symbols.len() * width as usize).div_ceil(8));
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for &s in symbols {
        debug_assert!(s >= base && ((s - base) as u16) < (1u16 << width), "symbol out of FOR range");
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
        *o = if code == span { get!(8u32) as u8 } else { base.wrapping_add(code as u8) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_roundtrip_fixed_width() {
        for &(base, width) in &[(104u8, 5u32), (96, 6), (0, 8), (120, 3)] {
            let span = (1u32 << width) as u32;
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
            assert_eq!(out, syms, "FOR roundtrip must be bit-exact (base={base}, width={width})");
        }
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
            assert_eq!(out, syms, "FOR-escape roundtrip must be bit-exact (base={base}, width={width})");
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
                if state % 100 < 70 { (118 + (state % 3)) as u8 } else { (104 + (state % 24)) as u8 }
            })
            .collect();
        let (base, width) = (112u8, 4u32); // window [112,126]
        let reps = 3usize;
        let gw = |secs: f64| (n as f64 * reps as f64 / 1e9) / secs;
        let mut out = vec![0u8; n];
        let enc = for_escape_encode(&syms, base, width);
        let bits = (enc.len() as f64 * 8.0) / n as f64;
        let span = (1u32 << width) - 1;
        let esc = syms.iter().filter(|&&s| (s.wrapping_sub(base) as u32) >= span).count();
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
                if state % 100 < 70 { (118 + (state % 3)) as u8 } else { (104 + (state % 24)) as u8 }
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
        eprintln!("  -> FOR decodes {:.2}x faster, at {:.2}x the exponent bits", for_gw / rans_gw, for_bits / rans_bits);
        eprintln!("##########");
    }
}
