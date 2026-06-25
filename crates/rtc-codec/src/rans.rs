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

use crate::codec::{DecodeMeta, EncodeMeta, EncodedChunk, TensorCodec};
use crate::dfloat::{join_bf16, split_bf16};
use crate::error::{CodecError, Result as CodecResult};

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

/// Precomputed decode tables (cumulative frequencies + slot→symbol map) for one
/// frequency table. Build once and reuse across many blocks/streams — building the
/// 4096-entry slot table per call dominates streaming decode otherwise.
pub struct RansDecodeTables {
    freq: [u32; 256],
    cum: [u32; 257],
    slot2sym: Vec<u8>,
}

/// Build the reusable decode tables for `freq`.
pub fn rans_build_tables(freq: &[u32; 256]) -> RansDecodeTables {
    let cum = cum_table(freq);
    let slot2sym = slot_table(&cum);
    RansDecodeTables { freq: *freq, cum, slot2sym }
}

/// Decode `n` symbols from a 4-lane interleaved stream (inverse of
/// [`rans_encode_interleaved4`]). Allocates + builds tables; for hot loops use
/// [`rans_decode_interleaved4_into`] with precomputed tables and a reused buffer.
pub fn rans_decode_interleaved4(streams: [&[u8]; 4], n: usize, freq: &[u32; 256]) -> Vec<u8> {
    let t = rans_build_tables(freq);
    let mut out = vec![0u8; n];
    rans_decode_interleaved4_into(streams, n, &t, &mut out);
    out
}

/// Decode `n` symbols into `out` (`out.len() >= n`) using precomputed `tables` and no
/// allocation. The four lanes advance independently in the body so the CPU overlaps
/// them (the ILP lever).
pub fn rans_decode_interleaved4_into(
    streams: [&[u8]; 4],
    n: usize,
    tables: &RansDecodeTables,
    out: &mut [u8],
) {
    let freq = &tables.freq;
    let cum = &tables.cum;
    let slot2sym = &tables.slot2sym;
    let mask = M - 1;
    let init = |s: &[u8]| -> u32 {
        (s[0] as u32) | (s[1] as u32) << 8 | (s[2] as u32) << 16 | (s[3] as u32) << 24
    };
    let (mut x0, mut x1, mut x2, mut x3) =
        (init(streams[0]), init(streams[1]), init(streams[2]), init(streams[3]));
    let (mut p0, mut p1, mut p2, mut p3) = (4usize, 4usize, 4usize, 4usize);
    let (s0, s1, s2, s3) = (streams[0], streams[1], streams[2], streams[3]);

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
}

/// Interleaved 8-lane rANS encode — same construction as [`rans_encode_interleaved4`]
/// but 8 independent streams. More lanes = more independent state chains for a wide
/// out-of-order core to overlap on decode (the REEBORN edge-decode lever the project
/// flagged at R152/R153 but had not pushed past 4 lanes).
pub fn rans_encode_interleaved8(symbols: &[u8], freq: &[u32; 256]) -> [Vec<u8>; 8] {
    core::array::from_fn(|j| {
        let sub: Vec<u8> = symbols.iter().skip(j).step_by(8).copied().collect();
        rans_encode(&sub, freq)
    })
}

/// Decode `n` symbols from an 8-lane interleaved stream into `out` using precomputed
/// `tables`. The eight lane states live in separate locals (own register chains) and are
/// stepped per iteration, so a wide core overlaps more independent work than the 4-lane
/// body — the throughput lever for lossless decode in the model>RAM streaming regime.
pub fn rans_decode_interleaved8_into(
    streams: [&[u8]; 8],
    n: usize,
    tables: &RansDecodeTables,
    out: &mut [u8],
) {
    let freq = &tables.freq;
    let cum = &tables.cum;
    let slot2sym = &tables.slot2sym;
    let mask = M - 1;
    let init = |s: &[u8]| -> u32 {
        (s[0] as u32) | (s[1] as u32) << 8 | (s[2] as u32) << 16 | (s[3] as u32) << 24
    };
    let (mut x0, mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7) = (
        init(streams[0]), init(streams[1]), init(streams[2]), init(streams[3]),
        init(streams[4]), init(streams[5]), init(streams[6]), init(streams[7]),
    );
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (4usize, 4usize, 4usize, 4usize, 4usize, 4usize, 4usize, 4usize);
    let (s0, s1, s2, s3, s4, s5, s6, s7) = (
        streams[0], streams[1], streams[2], streams[3],
        streams[4], streams[5], streams[6], streams[7],
    );
    macro_rules! step {
        ($x:ident, $p:ident, $s:ident, $idx:expr) => {
            if $idx < n {
                let sl = $x & mask;
                let sym = slot2sym[sl as usize];
                out[$idx] = sym;
                $x = freq[sym as usize] * ($x >> SCALE_BITS) + sl - cum[sym as usize];
                while $x < RANS_L {
                    $x = ($x << 8) | $s[$p] as u32;
                    $p += 1;
                }
            }
        };
    }
    let mut base = 0usize;
    while base < n {
        step!(x0, p0, s0, base);
        step!(x1, p1, s1, base + 1);
        step!(x2, p2, s2, base + 2);
        step!(x3, p3, s3, base + 3);
        step!(x4, p4, s4, base + 4);
        step!(x5, p5, s5, base + 5);
        step!(x6, p6, s6, base + 6);
        step!(x7, p7, s7, base + 7);
        base += 8;
    }
}

/// Convenience: build tables + decode an 8-lane interleaved stream.
pub fn rans_decode_interleaved8(streams: [&[u8]; 8], n: usize, freq: &[u32; 256]) -> Vec<u8> {
    let t = rans_build_tables(freq);
    let mut out = vec![0u8; n];
    rans_decode_interleaved8_into(streams, n, &t, &mut out);
    out
}

/// Parallel 4-lane interleaved decode: each lane (an independent rANS stream) is
/// decoded on its OWN thread, then the four subsequences are interleaved. ~4× the
/// single-thread `rans_decode_interleaved4` on large chunks — the container decode
/// path is otherwise single-threaded and dominates rANS inference (R158c).
pub fn rans_decode_interleaved4_parallel(streams: [&[u8]; 4], n: usize, freq: &[u32; 256]) -> Vec<u8> {
    let counts = [0usize, 1, 2, 3].map(|j| if j < n { (n - j).div_ceil(4) } else { 0 });
    let subs: Vec<Vec<u8>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|j| {
                let stream = streams[j];
                let cnt = counts[j];
                s.spawn(move || rans_decode(stream, cnt, freq))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = vec![0u8; n];
    for (j, sub) in subs.iter().enumerate() {
        for (k, &sym) in sub.iter().enumerate() {
            out[j + 4 * k] = sym;
        }
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

/// rtc-rans-v1 container codec: lossless bf16 weights at the entropy floor (~10.5
/// bits/weight). Splits bf16 into (exponent, residual), entropy-codes the exponent
/// with 4-lane interleaved rANS, and stores the residual raw. Non-bf16 / odd-length
/// chunks fall back to raw (FLAG_RAW) so the codec is safe for any tensor.
pub struct RansCodec;

impl RansCodec {
    pub const ID: &'static str = "rtc-rans-v1";
}

// Header: magic "RTCR"(4) + version(1) + flags(1) + n(u64). bf16 path then appends
// freq[256] u16 (512) + lane_len[4] u32 (16) + lanes + residual; raw path appends bytes.
const RANS_HEADER: usize = 4 + 1 + 1 + 8;
const RANS_FLAG_RAW: u8 = 0x01;

impl TensorCodec for RansCodec {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn encode(&self, input: &[u8], meta: &EncodeMeta) -> CodecResult<EncodedChunk> {
        let _ = meta; // the (exp,residual) split is a bijective bit rearrangement of any
                      // even-length bytes, so this is lossless regardless of dtype; it only
                      // *compresses* when the high byte is low-entropy (real bf16). The pack
                      // smallest-lossless picker keeps raw when rANS doesn't help.
        let mut data = Vec::with_capacity(RANS_HEADER + input.len());
        data.extend_from_slice(b"RTCR");
        data.push(1);
        // Raw fallback only for odd-length chunks (no bf16 pairing).
        if input.len() % 2 != 0 {
            data.push(RANS_FLAG_RAW);
            data.extend_from_slice(&(input.len() as u64).to_le_bytes());
            data.extend_from_slice(input);
            return Ok(EncodedChunk { codec_id: Self::ID.into(), data, original_size: input.len() as u64 });
        }
        let n = input.len() / 2;
        let mut exp = vec![0u8; n];
        let mut res = vec![0u8; n];
        for i in 0..n {
            let (e, r) = split_bf16(u16::from_le_bytes([input[2 * i], input[2 * i + 1]]));
            exp[i] = e;
            res[i] = r;
        }
        let freq = normalize_freqs(&count_symbols(&exp));
        let lanes = rans_encode_interleaved4(&exp, &freq);
        data.push(0); // flags
        data.extend_from_slice(&(n as u64).to_le_bytes());
        for f in freq.iter() {
            data.extend_from_slice(&(*f as u16).to_le_bytes());
        }
        for l in &lanes {
            data.extend_from_slice(&(l.len() as u32).to_le_bytes());
        }
        for l in &lanes {
            data.extend_from_slice(l);
        }
        data.extend_from_slice(&res);
        Ok(EncodedChunk { codec_id: Self::ID.into(), data, original_size: input.len() as u64 })
    }

    fn decode(&self, encoded: &[u8], _meta: &DecodeMeta) -> CodecResult<Vec<u8>> {
        let err = || CodecError::InvalidData("truncated rtc-rans-v1 chunk".into());
        if encoded.len() < RANS_HEADER || &encoded[0..4] != b"RTCR" || encoded[4] != 1 {
            return Err(err());
        }
        let flags = encoded[5];
        let n = u64::from_le_bytes(encoded[6..14].try_into().map_err(|_| err())?) as usize;
        let mut off = RANS_HEADER;
        if flags & RANS_FLAG_RAW != 0 {
            return encoded.get(off..off + n).map(|s| s.to_vec()).ok_or_else(err);
        }
        let freq_bytes = encoded.get(off..off + 512).ok_or_else(err)?;
        let mut freq = [0u32; 256];
        for (s, f) in freq.iter_mut().enumerate() {
            *f = u16::from_le_bytes([freq_bytes[s * 2], freq_bytes[s * 2 + 1]]) as u32;
        }
        off += 512;
        let lens = encoded.get(off..off + 16).ok_or_else(err)?;
        let ll = |k: usize| u32::from_le_bytes(lens[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
        let (l0, l1, l2, l3) = (ll(0), ll(1), ll(2), ll(3));
        off += 16;
        let lane0 = encoded.get(off..off + l0).ok_or_else(err)?;
        off += l0;
        let lane1 = encoded.get(off..off + l1).ok_or_else(err)?;
        off += l1;
        let lane2 = encoded.get(off..off + l2).ok_or_else(err)?;
        off += l2;
        let lane3 = encoded.get(off..off + l3).ok_or_else(err)?;
        off += l3;
        let residual = encoded.get(off..off + n).ok_or_else(err)?;
        // Parallel decode for large chunks (4 lanes → 4 threads); single-thread below
        // the threshold where the thread-spawn overhead would dominate.
        let exp = if n >= 65_536 {
            rans_decode_interleaved4_parallel([lane0, lane1, lane2, lane3], n, &freq)
        } else {
            rans_decode_interleaved4([lane0, lane1, lane2, lane3], n, &freq)
        };
        let mut out = vec![0u8; n * 2];
        for i in 0..n {
            let bits = join_bf16(exp[i], residual[i]);
            out[2 * i] = bits as u8;
            out[2 * i + 1] = (bits >> 8) as u8;
        }
        Ok(out)
    }
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
            let sl = [&streams[0][..], &streams[1][..], &streams[2][..], &streams[3][..]];
            let decoded = rans_decode_interleaved4(sl, syms.len(), &freq);
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

    #[test]
    fn rans_interleaved4_parallel_matches_serial() {
        for &len in &[1usize, 4, 7, 1000, 70_000, 100_001] {
            let mut state = 0x1357_9BDFu32;
            let syms: Vec<u8> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (96 + (state % 34)) as u8
                })
                .collect();
            let freq = normalize_freqs(&count_symbols(&syms));
            let streams = rans_encode_interleaved4(&syms, &freq);
            let sl = [&streams[0][..], &streams[1][..], &streams[2][..], &streams[3][..]];
            let serial = rans_decode_interleaved4(sl, len, &freq);
            let parallel = rans_decode_interleaved4_parallel(sl, len, &freq);
            assert_eq!(parallel, serial, "parallel decode must equal serial (len={len})");
            assert_eq!(parallel, syms, "parallel decode must be lossless (len={len})");
        }
    }

    // RansCodec (container TensorCodec) round-trips bf16 bit-exact and compresses.
    #[test]
    fn rans_codec_bf16_roundtrip_and_compresses() {
        // 34-exponent bf16 (Gemma-class). build via the bit pattern.
        let mut state = 0xF00D_BEEFu32;
        let n = 50_000usize;
        let mut bytes = Vec::with_capacity(n * 2);
        for k in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let exp = (96 + (k % 34)) as u16 & 0xFF;
            let bits = (((state >> 31) & 1) as u16) << 15 | (exp << 7) | (state & 0x7F) as u16;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
        let enc = RansCodec.encode(&bytes, &meta).unwrap();
        assert_eq!(enc.codec_id, "rtc-rans-v1");
        assert!(enc.data.len() < bytes.len(), "must compress: {} >= {}", enc.data.len(), bytes.len());
        let dmeta = DecodeMeta { codec_id: "rtc-rans-v1".into(), uncompressed_size: bytes.len() as u64 };
        assert_eq!(RansCodec.decode(&enc.data, &dmeta).unwrap(), bytes, "bf16 roundtrip must be bit-exact");
    }

    // Odd-length chunks fall back to raw and still round-trip.
    #[test]
    fn rans_codec_raw_fallback_roundtrip() {
        let bytes: Vec<u8> = (0..1235u32).map(|x| (x * 7) as u8).collect(); // ODD length => raw fallback
        let meta = EncodeMeta { name: "idx".into(), shape: vec![bytes.len() as u64], dtype: "u8".into() };
        let enc = RansCodec.encode(&bytes, &meta).unwrap();
        let dmeta = DecodeMeta { codec_id: "rtc-rans-v1".into(), uncompressed_size: bytes.len() as u64 };
        assert_eq!(RansCodec.decode(&enc.data, &dmeta).unwrap(), bytes, "raw-fallback roundtrip must be bit-exact");
    }

    #[test]
    fn rans_interleaved8_roundtrip_matches() {
        for &len in &[1usize, 2, 7, 8, 9, 15, 16, 17, 1000, 10003] {
            let mut state = 0x0BAD_F00Du32;
            let syms: Vec<u8> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (96 + (state % 34)) as u8
                })
                .collect();
            let freq = normalize_freqs(&count_symbols(&syms));
            let streams = rans_encode_interleaved8(&syms, &freq);
            let sl = [
                &streams[0][..], &streams[1][..], &streams[2][..], &streams[3][..],
                &streams[4][..], &streams[5][..], &streams[6][..], &streams[7][..],
            ];
            let decoded = rans_decode_interleaved8(sl, syms.len(), &freq);
            assert_eq!(decoded, syms, "interleaved-8 roundtrip must be bit-exact (len={len})");
        }
    }

    // REEBORN edge-decode lever: does going 4 -> 8 lanes lift single-core rANS decode
    // throughput on this machine? Run: cargo test -p rtc-codec reeborn_edge_lane_bench
    // -- --ignored --nocapture
    #[test]
    #[ignore]
    fn reeborn_edge_lane_bench() {
        use std::time::Instant;
        let n = 40_000_000usize; // 40M exponents (~one big tensor's worth)
        let mut state = 0x2468_ACE0u32;
        let syms: Vec<u8> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                // realistic bf16 exponent peak: ~70% on 118..120, rest spread 104..127
                if state % 100 < 70 { (118 + (state % 3)) as u8 } else { (104 + (state % 24)) as u8 }
            })
            .collect();
        let freq = normalize_freqs(&count_symbols(&syms));
        let s1 = rans_encode(&syms, &freq);
        let s4 = rans_encode_interleaved4(&syms, &freq);
        let s8 = rans_encode_interleaved8(&syms, &freq);
        let t = rans_build_tables(&freq);
        let mut out = vec![0u8; n];
        let reps = 3usize;
        let gw = |secs: f64| (n as f64 * reps as f64 / 1e9) / secs;

        let t0 = Instant::now();
        for _ in 0..reps {
            let d = rans_decode(&s1, n, &freq);
            std::hint::black_box(&d);
        }
        let scalar = gw(t0.elapsed().as_secs_f64());

        let sl4 = [&s4[0][..], &s4[1][..], &s4[2][..], &s4[3][..]];
        let t1 = Instant::now();
        for _ in 0..reps {
            rans_decode_interleaved4_into(sl4, n, &t, &mut out);
            std::hint::black_box(&out);
        }
        let l4 = gw(t1.elapsed().as_secs_f64());

        let sl8 = [
            &s8[0][..], &s8[1][..], &s8[2][..], &s8[3][..],
            &s8[4][..], &s8[5][..], &s8[6][..], &s8[7][..],
        ];
        let t2 = Instant::now();
        for _ in 0..reps {
            rans_decode_interleaved8_into(sl8, n, &t, &mut out);
            std::hint::black_box(&out);
        }
        let l8 = gw(t2.elapsed().as_secs_f64());
        assert_eq!(&out[..n], &syms[..], "8-lane decode must stay bit-exact in the bench");

        let bits = (s1.len() as f64 * 8.0) / n as f64;
        eprintln!("\n########## REEBORN edge lane bench ({n} exp symbols, {bits:.3} bits/sym) ##########");
        eprintln!("  scalar   (1 lane)  = {scalar:.3} Gweight/s/core");
        eprintln!("  interleaved4       = {l4:.3} Gweight/s/core  ({:.2}x vs scalar)", l4 / scalar);
        eprintln!("  interleaved8       = {l8:.3} Gweight/s/core  ({:.2}x vs scalar, {:.2}x vs 4-lane)", l8 / scalar, l8 / l4);
        eprintln!("##########");
    }
}
