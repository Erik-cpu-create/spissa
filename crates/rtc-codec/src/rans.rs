// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! rtc-rans-v1: static range-ANS (rANS) entropy coder for the bit-plane exponent
//! symbols (R152).
//!
//! R151 measured the lossless floor of LLM bf16 weights at ~10.5 bits/weight, of
//! which only the exponent (~2.6 bits) is compressible — the residual (sign+mantissa,
//! ~8 bits) is white noise. Bit-plane pays a fixed 5–6-bit index for the exponent, so
//! ~3 bits/weight are wasted. dfloat Huffman hits the entropy floor but decodes
//! serially (R142 NO-GO). rANS reaches the same ratio and, crucially, can be split
//! into independent per-block streams decoded in parallel (the R150a REESTREAM-PAR
//! structure) so decode hides under the cold-read bandwidth.
//!
//! Static 32-bit rANS, byte renormalization (Giesen's rans_byte scheme): SCALE_BITS=12
//! (M=4096 total frequency), RANS_L=1<<23. The frequency table is tiny (≤64 used
//! symbols) and stored alongside the stream.

const SCALE_BITS: u32 = 12;
const M: u32 = 1 << SCALE_BITS;
const RANS_L: u32 = 1 << 23;

/// Normalize symbol counts to frequencies summing to exactly `M` (each used symbol
/// gets freq ≥ 1). Deterministic; encoder and decoder build the same table.
pub fn normalize_freqs(counts: &[u32; 256]) -> [u32; 256] {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut freq = [0u32; 256];
    if total == 0 {
        return freq;
    }
    let mut sum: u32 = 0;
    for s in 0..256 {
        if counts[s] > 0 {
            let f = ((counts[s] as u64 * M as u64) / total).max(1) as u32;
            freq[s] = f;
            sum += f;
        }
    }
    // Fix the rounding drift so the table sums to exactly M.
    if sum < M {
        let s = (0..256).max_by_key(|&s| freq[s]).unwrap();
        freq[s] += M - sum;
    } else {
        let mut excess = sum - M;
        while excess > 0 {
            let s = (0..256).filter(|&s| freq[s] > 1).max_by_key(|&s| freq[s]).unwrap();
            let take = excess.min(freq[s] - 1);
            freq[s] -= take;
            excess -= take;
        }
    }
    freq
}

fn cum_table(freq: &[u32; 256]) -> [u32; 257] {
    let mut cum = [0u32; 257];
    for s in 0..256 {
        cum[s + 1] = cum[s] + freq[s];
    }
    cum
}

fn slot_table(cum: &[u32; 257]) -> Vec<u8> {
    let mut t = vec![0u8; M as usize];
    for s in 0..256 {
        for slot in cum[s]..cum[s + 1] {
            t[slot as usize] = s as u8;
        }
    }
    t
}

/// rANS-encode `symbols` with a frequency table normalized to `M`. Returns the byte
/// stream (4-byte final state followed by the renormalization bytes in decode order).
pub fn rans_encode(symbols: &[u8], freq: &[u32; 256]) -> Vec<u8> {
    let cum = cum_table(freq);
    let mut renorm = Vec::with_capacity(symbols.len() / 2 + 8);
    let mut x: u32 = RANS_L;
    // Encode in reverse so decode emits symbols in forward order (rANS is LIFO).
    for &s in symbols.iter().rev() {
        let f = freq[s as usize];
        debug_assert!(f > 0, "encoding a symbol with zero frequency");
        let c = cum[s as usize];
        let x_max = ((RANS_L >> SCALE_BITS) << 8).wrapping_mul(f);
        while x >= x_max {
            renorm.push((x & 0xff) as u8);
            x >>= 8;
        }
        x = ((x / f) << SCALE_BITS) + (x % f) + c;
    }
    let mut stream = Vec::with_capacity(renorm.len() + 4);
    stream.push(x as u8);
    stream.push((x >> 8) as u8);
    stream.push((x >> 16) as u8);
    stream.push((x >> 24) as u8);
    renorm.reverse();
    stream.extend_from_slice(&renorm);
    stream
}

/// Decode `n` symbols from a stream produced by [`rans_encode`] with the same `freq`.
/// Bit-exact inverse of `rans_encode`.
pub fn rans_decode(stream: &[u8], n: usize, freq: &[u32; 256]) -> Vec<u8> {
    let cum = cum_table(freq);
    let slot2sym = slot_table(&cum);
    let mut x: u32 = (stream[0] as u32)
        | (stream[1] as u32) << 8
        | (stream[2] as u32) << 16
        | (stream[3] as u32) << 24;
    let mut pos = 4usize;
    let mask = M - 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let slot = x & mask;
        let s = slot2sym[slot as usize];
        out.push(s);
        let f = freq[s as usize];
        let c = cum[s as usize];
        x = f * (x >> SCALE_BITS) + slot - c;
        while x < RANS_L {
            x = (x << 8) | stream[pos] as u32;
            pos += 1;
        }
    }
    out
}

/// Interleaved 4-lane rANS encode: symbol `i` goes to lane `i % 4`, each lane is an
/// independent rANS stream. Decoding the 4 lanes round-robin breaks the per-symbol
/// serial state dependency (4 independent state updates pipeline on one core) — the
/// ILP lever that lets rANS keep up with the streaming read where serial rANS/Huffman
/// (R142) could not.
pub fn rans_encode_interleaved4(symbols: &[u8], freq: &[u32; 256]) -> [Vec<u8>; 4] {
    core::array::from_fn(|j| {
        let sub: Vec<u8> = symbols.iter().skip(j).step_by(4).copied().collect();
        rans_encode(&sub, freq)
    })
}

/// Decode `n` symbols from a 4-lane interleaved stream (inverse of
/// [`rans_encode_interleaved4`]). The four lanes advance independently in the inner
/// body so the CPU overlaps them.
pub fn rans_decode_interleaved4(streams: &[Vec<u8>; 4], n: usize, freq: &[u32; 256]) -> Vec<u8> {
    let cum = cum_table(freq);
    let slot2sym = slot_table(&cum);
    let mask = M - 1;
    let init = |s: &[u8]| -> u32 {
        (s[0] as u32) | (s[1] as u32) << 8 | (s[2] as u32) << 16 | (s[3] as u32) << 24
    };
    let (mut x0, mut x1, mut x2, mut x3) =
        (init(&streams[0]), init(&streams[1]), init(&streams[2]), init(&streams[3]));
    let (mut p0, mut p1, mut p2, mut p3) = (4usize, 4usize, 4usize, 4usize);
    let (s0, s1, s2, s3) = (&streams[0], &streams[1], &streams[2], &streams[3]);
    let mut out = vec![0u8; n];

    let mut base = 0usize;
    while base < n {
        // Lane 0 (always present: base < n).
        let sl = x0 & mask;
        let s = slot2sym[sl as usize];
        out[base] = s;
        x0 = freq[s as usize] * (x0 >> SCALE_BITS) + sl - cum[s as usize];
        while x0 < RANS_L {
            x0 = (x0 << 8) | s0[p0] as u32;
            p0 += 1;
        }
        if base + 1 < n {
            let sl = x1 & mask;
            let s = slot2sym[sl as usize];
            out[base + 1] = s;
            x1 = freq[s as usize] * (x1 >> SCALE_BITS) + sl - cum[s as usize];
            while x1 < RANS_L {
                x1 = (x1 << 8) | s1[p1] as u32;
                p1 += 1;
            }
        }
        if base + 2 < n {
            let sl = x2 & mask;
            let s = slot2sym[sl as usize];
            out[base + 2] = s;
            x2 = freq[s as usize] * (x2 >> SCALE_BITS) + sl - cum[s as usize];
            while x2 < RANS_L {
                x2 = (x2 << 8) | s2[p2] as u32;
                p2 += 1;
            }
        }
        if base + 3 < n {
            let sl = x3 & mask;
            let s = slot2sym[sl as usize];
            out[base + 3] = s;
            x3 = freq[s as usize] * (x3 >> SCALE_BITS) + sl - cum[s as usize];
            while x3 < RANS_L {
                x3 = (x3 << 8) | s3[p3] as u32;
                p3 += 1;
            }
        }
        base += 4;
    }
    out
}

/// Count symbol occurrences (for building a frequency table).
pub fn count_symbols(symbols: &[u8]) -> [u32; 256] {
    let mut counts = [0u32; 256];
    for &s in symbols {
        counts[s as usize] += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(symbols: &[u8]) {
        let counts = count_symbols(symbols);
        let freq = normalize_freqs(&counts);
        assert_eq!(freq.iter().sum::<u32>(), M, "freqs must sum to M");
        let stream = rans_encode(symbols, &freq);
        let decoded = rans_decode(&stream, symbols.len(), &freq);
        assert_eq!(decoded, symbols, "rANS roundtrip must be bit-exact");
    }

    #[test]
    fn rans_roundtrip_small_alphabet() {
        // Skewed distribution over a small alphabet (like real exponents).
        let mut state = 0x1234_5678u32;
        let syms: Vec<u8> = (0..10000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                // bias toward a few exponent values around 96..130
                (96 + (state % 34)) as u8
            })
            .collect();
        roundtrip(&syms);
    }

    #[test]
    fn rans_roundtrip_edge_cases() {
        roundtrip(&[7u8]); // single symbol
        roundtrip(&[3u8; 1000]); // constant
        roundtrip(&(0..=255u8).cycle().take(5000).collect::<Vec<_>>()); // full alphabet
    }

    #[test]
    fn rans_interleaved4_roundtrip_matches() {
        for &len in &[1usize, 2, 3, 4, 5, 7, 8, 1000, 10001] {
            let mut state = 0xABCD_1234u32;
            let syms: Vec<u8> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (96 + (state % 34)) as u8
                })
                .collect();
            let counts = count_symbols(&syms);
            let freq = normalize_freqs(&counts);
            let streams = rans_encode_interleaved4(&syms, &freq);
            let decoded = rans_decode_interleaved4(&streams, syms.len(), &freq);
            assert_eq!(decoded, syms, "interleaved-4 roundtrip must be bit-exact (len={len})");
        }
    }

    #[test]
    fn rans_beats_fixed_width_on_skewed_exponents() {
        // 34 distinct exponents but concentrated => H ~2.6 bits; fixed-width index
        // would be ceil(log2(34)) = 6 bits. rANS should land well under 6.
        let mut state = 0xDEAD_BEEFu32;
        let syms: Vec<u8> = (0..200_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                // strong bias: ~70% on 3 values, rest spread over 34
                let r = state % 100;
                if r < 70 {
                    (100 + (state % 3)) as u8
                } else {
                    (96 + (state % 34)) as u8
                }
            })
            .collect();
        let counts = count_symbols(&syms);
        let freq = normalize_freqs(&counts);
        let stream = rans_encode(&syms, &freq);
        let bits_per_sym = (stream.len() as f64 * 8.0) / syms.len() as f64;
        assert!(bits_per_sym < 6.0, "rANS {bits_per_sym:.2} bits/sym should beat 6-bit fixed width");
        assert_eq!(rans_decode(&stream, syms.len(), &freq), syms);
    }
}
