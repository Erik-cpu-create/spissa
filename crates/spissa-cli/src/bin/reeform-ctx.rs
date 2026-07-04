// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada
//
// HONEST STRUCTURE AUDIT of the fine-tune delta. Conditional entropy H(X|ctx) ≤ H(X) ALWAYS, so a
// raw H1 < H0 proves nothing — the honest test is the ACTUAL coded length of an ADAPTIVE coder
// (decoder learns the same tables online; model cost paid via Krichevsky–Trofimov add-½, no free
// side-info). More contexts ⇒ more learning cost ⇒ finite-sample bias is automatically penalised.
//
// We probe two axes that could carry real exploitable structure in a smooth update field:
//   • CLASS (magnitude = bit-length of zigzag Δ): order-0 / left / up / left+up / per-col / per-row
//   • SIGN  (direction of the pattern delta, for nonzero Δ): order-0 / left / up
// Mantissa stays raw (shared cost). Any context that beats order-0 AFTER the KT penalty is real.
//
//   reeform-ctx <ft.safetensors> <base.safetensors>

use anyhow::Result;
use spissa_container::DType;
use spissa_import::SafetensorsReader;

const A: usize = 17; // classes 0..=16

#[inline]
fn zigzag(d: u16) -> u16 {
    let x = d as i16;
    (x.wrapping_shl(1) ^ (x >> 15)) as u16
}
#[inline]
fn class_of(z: u16) -> usize {
    if z == 0 {
        0
    } else {
        16 - z.leading_zeros() as usize
    }
}
#[inline]
fn mant_bits(c: usize) -> u64 {
    if c == 0 {
        0
    } else {
        (c - 1) as u64
    }
}
fn u16s(bytes: &[u8]) -> Vec<u16> {
    (0..bytes.len() / 2)
        .map(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
        .collect()
}

/// Adaptive KT model: a bank of count tables over an `A`-symbol alphabet. -log2 p = coded bits.
struct Adaptive {
    cnt: Vec<u32>,
    tot: Vec<u32>,
    a: usize,
}
impl Adaptive {
    fn new(n_ctx: usize, a: usize) -> Self {
        Adaptive {
            cnt: vec![0u32; n_ctx * a],
            tot: vec![0u32; n_ctx],
            a,
        }
    }
    #[inline]
    fn code(&mut self, ctx: usize, sym: usize) -> f64 {
        let i = ctx * self.a + sym;
        let p = (self.cnt[i] as f64 + 0.5) / (self.tot[ctx] as f64 + 0.5 * self.a as f64);
        self.cnt[i] += 1;
        self.tot[ctx] += 1;
        -p.log2()
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (ft, base) = (&a[1], &a[2]);
    let mut rb = SafetensorsReader::open(base)?;
    let mut rf = SafetensorsReader::open(ft)?;
    let bnames: std::collections::HashSet<String> =
        rb.list_tensors().iter().map(|s| s.to_string()).collect();
    let fnames: Vec<String> = rf.list_tensors().iter().map(|s| s.to_string()).collect();

    // Global class models (shared across the whole model).
    let mut c0 = Adaptive::new(1, A);
    let mut cl = Adaptive::new(A, A);
    let mut cu = Adaptive::new(A, A);
    let mut clu = Adaptive::new(A * A, A);
    let (mut b0, mut bl, mut bu, mut blu) = (0f64, 0f64, 0f64, 0f64);
    // Per-col / per-row class models are reset PER MATRIX (col j / row i differ across matrices).
    let (mut bcol, mut brow) = (0f64, 0f64);
    // Global sign models (sign of nonzero pattern-delta; 0=edge/zero ctx).
    let mut s0 = Adaptive::new(1, 2);
    let mut sl = Adaptive::new(3, 2); // ctx: 0 none, 1 neg, 2 pos (left)
    let mut su = Adaptive::new(3, 2); // (up)
    let (mut bs0, mut bsl, mut bsu) = (0f64, 0f64, 0f64);
    // BASE as FREE side-info (decoder has it): class|base-exponent, sign|base-sign.
    let mut cbe = Adaptive::new(256, A); // ctx = base bf16 exponent
    let mut sbs = Adaptive::new(2, 2); // ctx = base sign (weight-decay hypothesis)
    let (mut bbe, mut bsbs) = (0f64, 0f64);
    let mut bmant = 0u64;
    let (mut nweights, mut nnz) = (0u64, 0u64);

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
        let (bv, fv) = (u16s(&bb), u16s(&fb));
        let cols = (*m.shape.last().unwrap_or(&1) as usize).max(1);
        let rows = fv.len() / cols;
        // Precompute class + sign-ctx per element.
        let cls: Vec<u8> = (0..fv.len())
            .map(|i| class_of(zigzag(fv[i].wrapping_sub(bv[i]))) as u8)
            .collect();
        let sgn: Vec<u8> = (0..fv.len())
            .map(|i| {
                let d = fv[i].wrapping_sub(bv[i]) as i16;
                if d == 0 {
                    0
                } else if d < 0 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let mut col_m = Adaptive::new(cols, A);
        let mut row_m = Adaptive::new(rows, A);
        let mut prev = vec![0u8; cols];
        let mut prev_s = vec![0u8; cols];
        for r in 0..rows {
            let (mut left, mut left_s) = (0u8, 0u8);
            let bi = r * cols;
            for c in 0..cols {
                let k = cls[bi + c] as usize;
                let base_pat = bv[bi + c];
                b0 += c0.code(0, k);
                bl += cl.code(left as usize, k);
                bu += cu.code(prev[c] as usize, k);
                blu += clu.code(left as usize * A + prev[c] as usize, k);
                bcol += col_m.code(c, k);
                brow += row_m.code(r, k);
                bbe += cbe.code(((base_pat >> 7) & 0xFF) as usize, k);
                bmant += mant_bits(k);
                // sign (only nonzero carries a sign bit; zigzag already charges it as a mant bit)
                let sg = sgn[bi + c];
                if sg != 0 {
                    let s = (sg - 1) as usize; // 0 neg, 1 pos
                    bs0 += s0.code(0, s);
                    bsl += sl.code(left_s as usize, s);
                    bsu += su.code(prev_s[c] as usize, s);
                    bsbs += sbs.code((base_pat >> 15) as usize, s);
                    nnz += 1;
                }
                left = k as u8;
                left_s = sg;
                prev[c] = k as u8;
                prev_s[c] = sg;
            }
        }
        nweights += fv.len() as u64;
    }

    let n = nweights as f64;
    let mant = bmant as f64;
    println!("=== REEFORM structure audit (adaptive, model-cost PAID) ===");
    println!(
        "weights {nweights} · nonzero Δ {nnz} ({:.2}%)",
        100.0 * nnz as f64 / n
    );
    println!("mantissa (raw, shared)   : {:.4} bit/w", mant / n);
    println!("--- CLASS coding: class-bits/w  (+mant = total) ---");
    let pr = |label: &str, bits: f64, ref0: f64| {
        println!(
            "{label:22}: {:.4}  (total {:.4}, Δvs0 {:+.4})",
            bits / n,
            (bits + mant) / n,
            (bits - ref0) / n
        );
    };
    pr("order-0", b0, b0);
    pr("ctx=left", bl, b0);
    pr("ctx=up", bu, b0);
    pr("ctx=left+up", blu, b0);
    pr("ctx=per-col", bcol, b0);
    pr("ctx=per-row", brow, b0);
    pr("ctx=BASE-exponent", bbe, b0);
    println!("--- SIGN coding (of nonzero Δ): bit per nonzero ---");
    let m2 = nnz as f64;
    println!(
        "order-0 {:.4} · ctx=left {:.4} (Δ {:+.4}) · ctx=up {:.4} (Δ {:+.4})",
        bs0 / m2,
        bsl / m2,
        (bsl - bs0) / m2,
        bsu / m2,
        (bsu - bs0) / m2
    );
    println!(
        "ctx=BASE-sign {:.4} (Δ {:+.4})  ← weight-decay hypothesis",
        bsbs / m2,
        (bsbs - bs0) / m2
    );
    let best_cls = [b0, bl, bu, blu, bcol, brow]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    println!(
        "BEST class total {:.4} bit/w  (best Δ vs order-0 {:+.4})   [our shipped u16-rANS ref 7.74]",
        (best_cls + mant) / n,
        (best_cls - b0) / n
    );
    Ok(())
}
