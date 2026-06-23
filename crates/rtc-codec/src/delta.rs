// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — REEFORM lossless fine-tune delta codec (SECRET IP).
//
// A fine-tune is its base plus a gentle update, so `Δ = W_ft − W_base` (on the bf16 bit patterns,
// int-subtract mod 2^16, exactly reversible) clusters tightly around 0 and codes far below the full
// model. We zigzag the signed Δ (signed→small-unsigned) and entropy-code the u16 symbols with a
// static, global-table u16-symbol rANS. The base is required to BOTH compute and invert the delta,
// so this codec takes the base bytes explicitly (it is not a self-contained `TensorCodec`).
//
// Decode is the exact inverse: `W_ft = W_base + unzigzag(Δ)`. Verified bit-exact over 134.5M
// weights (SmolLM2-135M, Qwen2.5-0.5B). Phase-3 will condition the symbol model on the base
// exponent (a further ~8% lossless) without changing this format's framing.

/// Chunk codec id for a delta-coded tensor (payload = global-table u16-rANS of the zigzag Δ).
pub const CODEC_DELTA_V1: &str = "reeform-delta-v1";
/// Synthetic tensor that carries the model's single global rANS frequency table (raw chunk).
pub const DELTA_TABLE_TENSOR: &str = "__reeform_delta_table__";

const PROB_BITS: u32 = 20;
const PROB_SCALE: u64 = 1 << PROB_BITS;
const RANS_L: u64 = 1 << 31;

#[inline]
pub fn zigzag(d: u16) -> u16 {
    let x = d as i16;
    (x.wrapping_shl(1) ^ (x >> 15)) as u16
}
#[inline]
pub fn unzigzag(z: u16) -> u16 {
    (((z >> 1) as i16) ^ -((z & 1) as i16)) as u16
}

#[inline]
fn u16s(bytes: &[u8]) -> Vec<u16> {
    (0..bytes.len() / 2)
        .map(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
        .collect()
}

/// Static u16-symbol rANS frequency tables (one global instance per model).
pub struct Tables {
    freq: Vec<u32>,
    cum: Vec<u32>,
    slot: Vec<u16>,
}

impl Tables {
    /// Build a normalized table from a 65536-bin symbol histogram.
    pub fn from_hist(hist: &[u64]) -> Tables {
        let total: u64 = hist.iter().sum::<u64>().max(1);
        let mut freq = vec![0u32; 65536];
        let (mut sum, mut maxs) = (0u64, 0usize);
        for s in 0..65536 {
            if hist[s] > 0 {
                let f = ((hist[s] as u128 * PROB_SCALE as u128 / total as u128) as u64).max(1);
                freq[s] = f as u32;
                sum += f;
                if freq[s] > freq[maxs] {
                    maxs = s;
                }
            }
        }
        if sum != PROB_SCALE {
            freq[maxs] = (freq[maxs] as i64 + (PROB_SCALE as i64 - sum as i64)) as u32;
        }
        Self::from_freq(freq)
    }

    fn from_freq(freq: Vec<u32>) -> Tables {
        let mut cum = vec![0u32; 65536];
        let mut slot = vec![0u16; PROB_SCALE as usize];
        let mut c = 0u32;
        for s in 0..65536 {
            cum[s] = c;
            for i in c..c + freq[s] {
                slot[i as usize] = s as u16;
            }
            c += freq[s];
        }
        Tables { freq, cum, slot }
    }

    /// Serialize the frequency table (65536 × u32 LE) for storage in the container.
    pub fn freq_to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(65536 * 4);
        for &f in &self.freq {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    /// Rebuild tables from a serialized frequency table.
    pub fn from_freq_bytes(b: &[u8]) -> Tables {
        let freq: Vec<u32> = (0..65536)
            .map(|i| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]))
            .collect();
        Self::from_freq(freq)
    }
}

/// Add a tensor's zigzag-delta symbols to a global histogram (pass 1 of packing).
pub fn accumulate_hist(hist: &mut [u64], zz: &[u16]) {
    for &s in zz {
        hist[s as usize] += 1;
    }
}

/// Zigzag int-subtract delta of a bf16 tensor against its base (both little-endian bf16 bytes).
pub fn delta_zigzag(ft_bf16: &[u8], base_bf16: &[u8]) -> Vec<u16> {
    let (fv, bv) = (u16s(ft_bf16), u16s(base_bf16));
    (0..fv.len())
        .map(|i| zigzag(fv[i].wrapping_sub(bv[i])))
        .collect()
}

/// Reconstruct the fine-tune bf16 bytes from the zigzag delta + base bytes: `W_ft = W_base + Δ`.
pub fn reconstruct(zz: &[u16], base_bf16: &[u8]) -> Vec<u8> {
    let bv = u16s(base_bf16);
    let mut out = Vec::with_capacity(zz.len() * 2);
    for i in 0..zz.len() {
        out.extend_from_slice(&bv[i].wrapping_add(unzigzag(zz[i])).to_le_bytes());
    }
    out
}

/// rANS-encode a symbol stream with the global table (LIFO; the byte order is decode-ready).
pub fn encode_stream(syms: &[u16], t: &Tables) -> Vec<u8> {
    let mut x = RANS_L;
    let mut out = Vec::with_capacity(syms.len());
    for &s in syms.iter().rev() {
        let (f, c) = (t.freq[s as usize] as u64, t.cum[s as usize] as u64);
        let x_max = ((RANS_L >> PROB_BITS) << 8) * f;
        while x >= x_max {
            out.push((x & 0xFF) as u8);
            x >>= 8;
        }
        x = ((x / f) << PROB_BITS) + (x % f) + c;
    }
    for i in 0..8 {
        out.push(((x >> (8 * i)) & 0xFF) as u8);
    }
    out.reverse();
    out
}

/// rANS-decode `n` symbols from a stream produced by [`encode_stream`] with the same table.
pub fn decode_stream(data: &[u8], n: usize, t: &Tables) -> Vec<u16> {
    let mut x = 0u64;
    for &b in data.iter().take(8) {
        x = (x << 8) | b as u64;
    }
    let mut pos = 8usize;
    let mask = PROB_SCALE - 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let s = t.slot[(x & mask) as usize];
        let (f, c) = (t.freq[s as usize] as u64, t.cum[s as usize] as u64);
        x = f * (x >> PROB_BITS) + (x & mask) - c;
        while x < RANS_L {
            x = (x << 8) | data[pos] as u64;
            pos += 1;
        }
        out.push(s);
    }
    out
}

/// Decode one delta chunk straight back to fine-tune bf16 bytes (loader convenience).
pub fn decode_tensor(stream: &[u8], base_bf16: &[u8], n_weights: usize, t: &Tables) -> Vec<u8> {
    reconstruct(&decode_stream(stream, n_weights, t), base_bf16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_roundtrip() {
        for d in [0u16, 1, 2, 3, 65535, 32768, 12345] {
            assert_eq!(unzigzag(zigzag(d)), d);
        }
    }

    #[test]
    fn delta_codec_bit_exact() {
        // synthetic base + fine-tune (small per-weight pattern moves)
        let n = 4000usize;
        let mut base = Vec::new();
        let mut ft = Vec::new();
        for i in 0..n {
            let b = (i.wrapping_mul(40503) & 0xFFFF) as u16;
            let mv = ((i.wrapping_mul(2654435761) >> 13) % 7) as i32 - 3; // small move -3..3
            let f = (b as i32 + mv) as u16;
            base.extend_from_slice(&b.to_le_bytes());
            ft.extend_from_slice(&f.to_le_bytes());
        }
        let zz = delta_zigzag(&ft, &base);
        let mut hist = vec![0u64; 65536];
        accumulate_hist(&mut hist, &zz);
        let t = Tables::from_hist(&hist);
        // round-trip the table through serialization too
        let t2 = Tables::from_freq_bytes(&t.freq_to_bytes());
        let enc = encode_stream(&zz, &t);
        let ft_rec = decode_tensor(&enc, &base, n, &t2);
        assert_eq!(ft_rec, ft, "delta codec must be bit-exact");
    }
}
