// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — research instrument (REEFORM, SECRET IP).
//
// THE BASE-CONDITIONED LEVER. The int-pattern delta's magnitude is Δvalue / ULP(W_base), and the
// bf16 ULP is set by the base EXPONENT. So the base exponent mechanically predicts the delta
// symbol — and the decoder already has the base (FREE side-info). This measures the true ceiling:
// H0(Δ) vs H(Δ | base-exponent), the achievable rate of a base-conditioned entropy coder. We also
// try (base-exp, base-mantissa-hi) to see if finer base context helps. Static joint histogram =
// clean information rate; with 256 contexts over 134M weights the per-context table/learning cost
// is negligible, so H_cond is essentially achievable.
//
//   reeform-basecond <ft.safetensors> <base.safetensors>

use anyhow::Result;
use spissa_container::DType;
use spissa_import::SafetensorsReader;

#[inline]
fn zigzag(d: u16) -> u16 {
    let x = d as i16;
    (x.wrapping_shl(1) ^ (x >> 15)) as u16
}
fn u16s(bytes: &[u8]) -> Vec<u16> {
    (0..bytes.len() / 2)
        .map(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
        .collect()
}
#[inline]
fn entropy(counts: &[u32], total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut h = 0.0;
    for &c in counts {
        if c != 0 {
            let p = c as f64 / t;
            h -= p * p.log2();
        }
    }
    h
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (ft, base) = (&a[1], &a[2]);
    let mut rb = SafetensorsReader::open(base)?;
    let mut rf = SafetensorsReader::open(ft)?;
    let bnames: std::collections::HashSet<String> =
        rb.list_tensors().iter().map(|s| s.to_string()).collect();
    let fnames: Vec<String> = rf.list_tensors().iter().map(|s| s.to_string()).collect();

    const S: usize = 65536;
    let mut marg = vec![0u32; S];
    // ctx = base exponent (256)
    let mut joint_e = vec![0u32; 256 * S];
    let mut tot_e = vec![0u64; 256];
    // ctx = base exponent (8) + top mantissa bit (1) = 512, to test finer base context
    let mut joint_em = vec![0u32; 512 * S];
    let mut tot_em = vec![0u64; 512];
    let mut n = 0u64;
    // CORE transformer only (exclude embed/lm_head) — robustness: is the lever in real weights too?
    let mut cmarg = vec![0u32; S];
    let mut cjoint = vec![0u32; 256 * S];
    let mut ctot = vec![0u64; 256];
    let mut cn = 0u64;

    for name in &fnames {
        let m = rf.to_rllm_meta(name)?;
        if m.dtype != DType::Bf16 || !bnames.contains(name) {
            continue;
        }
        let bm = rb.to_rllm_meta(name)?;
        if bm.shape != m.shape {
            continue;
        }
        let fb = rf.read_tensor(name)?;
        let bb = rb.read_tensor(name)?;
        if fb.len() != bb.len() {
            continue;
        }
        let ln = name.to_lowercase();
        let is_core = !(ln.contains("embed")
            || ln.contains("lm_head")
            || ln.contains("wte")
            || ln.contains("wpe"));
        let (bv, fv) = (u16s(&bb), u16s(&fb));
        for i in 0..fv.len() {
            let s = zigzag(fv[i].wrapping_sub(bv[i])) as usize;
            let e = ((bv[i] >> 7) & 0xFF) as usize;
            let em = e * 2 + ((bv[i] >> 6) & 1) as usize;
            marg[s] += 1;
            joint_e[e * S + s] += 1;
            tot_e[e] += 1;
            joint_em[em * S + s] += 1;
            tot_em[em] += 1;
            n += 1;
            if is_core {
                cmarg[s] += 1;
                cjoint[e * S + s] += 1;
                ctot[e] += 1;
                cn += 1;
            }
        }
    }

    let h0 = entropy(&marg, n);
    let mut h_e = 0.0;
    for e in 0..256 {
        if tot_e[e] != 0 {
            h_e += (tot_e[e] as f64 / n as f64) * entropy(&joint_e[e * S..(e + 1) * S], tot_e[e]);
        }
    }
    let mut h_em = 0.0;
    for c in 0..512 {
        if tot_em[c] != 0 {
            h_em += (tot_em[c] as f64 / n as f64) * entropy(&joint_em[c * S..(c + 1) * S], tot_em[c]);
        }
    }
    let mib = |bpw: f64| bpw * n as f64 / 8.0 / 1048576.0;
    println!("=== REEFORM base-conditioned ceiling (static conditional entropy) ===");
    println!("weights {n}");
    println!("H0(Δ)                      : {:.4} bit/w   ({:.1} MiB)  [our shipped ≈ this]", h0, mib(h0));
    println!(
        "H(Δ | base-exp)            : {:.4} bit/w   ({:.1} MiB)   gain {:+.4}",
        h_e,
        mib(h_e),
        h_e - h0
    );
    println!(
        "H(Δ | base-exp + mant-hi)  : {:.4} bit/w   ({:.1} MiB)   gain {:+.4}",
        h_em,
        mib(h_em),
        h_em - h0
    );
    println!(
        "→ base-exp lever = {:.1}% further lossless on top of the delta",
        100.0 * (h0 - h_e) / h0
    );
    // CORE-only robustness (exclude embed/lm_head)
    let ch0 = entropy(&cmarg, cn);
    let mut ch_e = 0.0;
    for e in 0..256 {
        if ctot[e] != 0 {
            ch_e += (ctot[e] as f64 / cn as f64) * entropy(&cjoint[e * S..(e + 1) * S], ctot[e]);
        }
    }
    println!("--- CORE transformer only (no embed/lm_head): {cn} weights ---");
    println!(
        "H0 {:.4} → H(Δ|base-exp) {:.4}   gain {:+.4}  ({:.1}% further)  ← lever in REAL weights?",
        ch0,
        ch_e,
        ch_e - ch0,
        100.0 * (ch0 - ch_e) / ch0
    );
    Ok(())
}
