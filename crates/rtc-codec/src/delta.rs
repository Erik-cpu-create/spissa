// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada
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
/// Generic on-disk id (the secret project codename is kept out of the binary + `.spsa` files).
pub const CODEC_DELTA_V1: &str = "rtc-delta-v1";
/// Synthetic tensor that carries the model's codebook (raw chunk). Generic on-disk name.
pub const DELTA_TABLE_TENSOR: &str = "__rtc_delta_table__";

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

// ===== Phase-3: BASE-EXPONENT-CONDITIONED coding (the REEFORM invention) =====
//
// The int-pattern delta magnitude is Δvalue / ULP(W_base), and the bf16 ULP is set by the base
// EXPONENT — so the base exponent predicts the delta symbol, and the decoder has the base for free.
// We code each symbol with the table for ITS base exponent. A `Codebook` carries one global table
// plus per-exponent tables only where one pays for its own (compact) storage (a hybrid that keeps
// the small dense exponents specialised and the sparse large ones on the global table). Measured
// ~7% further lossless, cross-family, bit-exact. rANS switches tables per symbol (all share the
// same PROB_BITS, so the state math is identical); encoder and decoder agree on the table because
// both derive the exponent from the base at the same position.

/// Per-weight base bf16 exponent — the coding context (decoder recomputes it from the base).
pub fn base_exps(base_bf16: &[u8]) -> Vec<u8> {
    (0..base_bf16.len() / 2)
        .map(|i| {
            let p = u16::from_le_bytes([base_bf16[2 * i], base_bf16[2 * i + 1]]);
            ((p >> 7) & 0xFF) as u8
        })
        .collect()
}

/// Global table + optional per-exponent tables (indexed by exponent 0..=255; `None` → use global).
pub struct Codebook {
    global: Tables,
    per_exp: Vec<Option<Tables>>,
}

impl Codebook {
    #[inline]
    fn table(&self, exp: u8) -> &Tables {
        self.per_exp[exp as usize].as_ref().unwrap_or(&self.global)
    }

    /// Serialize: global table, then a count + (exponent, table) for each specialised exponent.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        write_table(&mut out, &self.global);
        let exps: Vec<usize> = (0..256).filter(|&e| self.per_exp[e].is_some()).collect();
        out.extend_from_slice(&(exps.len() as u16).to_le_bytes());
        for e in exps {
            out.push(e as u8);
            write_table(&mut out, self.per_exp[e].as_ref().unwrap());
        }
        out
    }

    /// Inverse of [`Codebook::serialize`].
    pub fn deserialize(b: &[u8]) -> Codebook {
        let mut p = 0usize;
        let global = read_table(b, &mut p);
        let n = u16::from_le_bytes([b[p], b[p + 1]]) as usize;
        p += 2;
        let mut per_exp: Vec<Option<Tables>> = (0..256).map(|_| None).collect();
        for _ in 0..n {
            let e = b[p] as usize;
            p += 1;
            per_exp[e] = Some(read_table(b, &mut p));
        }
        Codebook { global, per_exp }
    }
}

/// Compact table storage: max occupied symbol + freqs up to it (u16, escaping ≥0xFFFF to u32).
fn write_table(out: &mut Vec<u8>, t: &Tables) {
    let max_sym = (0..65536).rev().find(|&s| t.freq[s] > 0).unwrap_or(0);
    out.extend_from_slice(&(max_sym as u32).to_le_bytes());
    for s in 0..=max_sym {
        let f = t.freq[s];
        if f < 0xFFFF {
            out.extend_from_slice(&(f as u16).to_le_bytes());
        } else {
            out.extend_from_slice(&[0xFF, 0xFF]);
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
}
fn read_table(b: &[u8], p: &mut usize) -> Tables {
    let max_sym = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap()) as usize;
    *p += 4;
    let mut freq = vec![0u32; 65536];
    for fs in freq.iter_mut().take(max_sym + 1) {
        let lo = u16::from_le_bytes([b[*p], b[*p + 1]]);
        *p += 2;
        *fs = if lo == 0xFFFF {
            let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
            *p += 4;
            v
        } else {
            lo as u32
        };
    }
    Tables::from_freq(freq)
}
fn table_len(t: &Tables) -> usize {
    let max_sym = (0..65536).rev().find(|&s| t.freq[s] > 0).unwrap_or(0);
    4 + (0..=max_sym)
        .map(|s| if t.freq[s] < 0xFFFF { 2 } else { 6 })
        .sum::<usize>()
}
fn est_bits(hist: &[u64], t: &Tables) -> f64 {
    let mut bits = 0.0;
    #[allow(clippy::needless_range_loop)] // s indexes the 2^16 symbol histogram directly
    for s in 0..65536 {
        if hist[s] > 0 {
            let p = t.freq[s] as f64 / PROB_SCALE as f64;
            bits += hist[s] as f64 * -p.log2();
        }
    }
    bits
}

/// Accumulate a tensor's symbols into the global + per-exponent histograms (pass 1 of packing).
/// `per_exp_hist` is a flat 256×65536 table indexed `[exp*65536 + symbol]`.
pub fn accumulate_bx(per_exp_hist: &mut [u64], global_hist: &mut [u64], zz: &[u16], exps: &[u8]) {
    for i in 0..zz.len() {
        let s = zz[i] as usize;
        global_hist[s] += 1;
        per_exp_hist[exps[i] as usize * 65536 + s] += 1;
    }
}

/// Decide the codebook: keep a per-exponent table only where it codes its exponent's symbols
/// smaller than the global table does, including the table's own compact storage cost.
pub fn build_codebook(per_exp_hist: &[u64], global_hist: &[u64]) -> Codebook {
    let global = Tables::from_hist(global_hist);
    let mut per_exp: Vec<Option<Tables>> = (0..256).map(|_| None).collect();
    for e in 0..256 {
        let h = &per_exp_hist[e * 65536..(e + 1) * 65536];
        if h.iter().all(|&c| c == 0) {
            continue;
        }
        let t = Tables::from_hist(h);
        let pay = est_bits(h, &t) + (table_len(&t) * 8) as f64;
        if pay < est_bits(h, &global) {
            per_exp[e] = Some(t);
        }
    }
    Codebook { global, per_exp }
}

/// rANS-encode the zigzag Δ, switching to each symbol's base-exponent table.
pub fn encode_bx(zz: &[u16], exps: &[u8], cb: &Codebook) -> Vec<u8> {
    let mut x = RANS_L;
    let mut out = Vec::with_capacity(zz.len());
    for i in (0..zz.len()).rev() {
        let t = cb.table(exps[i]);
        let s = zz[i] as usize;
        let (f, c) = (t.freq[s] as u64, t.cum[s] as u64);
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

/// Inverse of [`encode_bx`] — `exps` (length = symbol count) selects the table at each position.
pub fn decode_bx(data: &[u8], exps: &[u8], cb: &Codebook) -> Vec<u16> {
    let mut x = 0u64;
    for &b in data.iter().take(8) {
        x = (x << 8) | b as u64;
    }
    let mut pos = 8usize;
    let mask = PROB_SCALE - 1;
    let mut out = Vec::with_capacity(exps.len());
    for &e in exps {
        let t = cb.table(e);
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

/// Encode a fine-tune tensor as a base-exponent-conditioned delta chunk (pack convenience).
pub fn encode_tensor_bx(ft_bf16: &[u8], base_bf16: &[u8], cb: &Codebook) -> Vec<u8> {
    encode_bx(&delta_zigzag(ft_bf16, base_bf16), &base_exps(base_bf16), cb)
}

/// Decode a base-exponent-conditioned delta chunk back to fine-tune bf16 bytes (loader).
pub fn decode_tensor_bx(stream: &[u8], base_bf16: &[u8], cb: &Codebook) -> Vec<u8> {
    let exps = base_exps(base_bf16);
    reconstruct(&decode_bx(stream, &exps, cb), base_bf16)
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

    #[test]
    fn delta_bx_bit_exact() {
        // base spans several exponents; the fine-tune nudges patterns by small per-weight moves.
        let n = 6000usize;
        let (mut base, mut ft) = (Vec::new(), Vec::new());
        for i in 0..n {
            let exp = (i % 40) as u16; // vary the base exponent across the range
            let mant = (i.wrapping_mul(2654435761) >> 11) as u16 & 0x7F;
            let b = (exp << 7) | mant;
            let mv = ((i.wrapping_mul(40503) >> 9) % 9) as i32 - 4;
            let f = (b as i32 + mv) as u16;
            base.extend_from_slice(&b.to_le_bytes());
            ft.extend_from_slice(&f.to_le_bytes());
        }
        let zz = delta_zigzag(&ft, &base);
        let exps = base_exps(&base);
        let mut gh = vec![0u64; 65536];
        let mut ph = vec![0u64; 256 * 65536];
        accumulate_bx(&mut ph, &mut gh, &zz, &exps);
        let cb = build_codebook(&ph, &gh);
        // round-trip the codebook through (de)serialization too
        let cb2 = Codebook::deserialize(&cb.serialize());
        let enc = encode_tensor_bx(&ft, &base, &cb);
        let ft_rec = decode_tensor_bx(&enc, &base, &cb2);
        assert_eq!(ft_rec, ft, "base-exp delta codec must be bit-exact");
    }
}
