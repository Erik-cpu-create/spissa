// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada
#![allow(dead_code)]
//
//! REEFORM Phase-0 probe: measure whether LLM bf16 weights carry CONDITIONAL (higher-order)
//! structure that the order-0 entropy floor (~10.6 bit/weight, what rANS/bit-plane/DFloat11
//! hit) is blind to. If a lossless order-1 context model codes the fields below the order-0
//! floor, the floor is breakable losslessly — go.
//!
//! Run: reeform-probe [model.safetensors]   (default: models/smollm2-135m/model.safetensors)

use spissa_container::DType;
use spissa_import::SafetensorsReader;
use std::collections::HashMap;

/// Order-0 entropy (bits/symbol): -Σ p log2 p over the empirical symbol distribution.
fn h0(syms: &[u32]) -> f64 {
    if syms.is_empty() {
        return 0.0;
    }
    let mut hist: HashMap<u32, u64> = HashMap::new();
    for &s in syms {
        *hist.entry(s).or_insert(0) += 1;
    }
    let n = syms.len() as f64;
    -hist
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            p * p.log2()
        })
        .sum::<f64>()
}

/// Order-1 conditional entropy H(cur | prev), context reset at each row boundary so the
/// predictor never crosses rows. This IS the achievable rate of a lossless order-1 context
/// coder on the field. `field_bits` ≤ 12 (sign/exp/mantissa) so the (prev<<12|cur) key fits.
fn h1_rowaware(syms: &[u32], ncols: usize, field_bits: u32) -> f64 {
    let shift = field_bits.max(1);
    let mut joint: HashMap<u32, u64> = HashMap::new();
    let mut prev_marg: HashMap<u32, u64> = HashMap::new();
    let mut npairs = 0u64;
    for i in 1..syms.len() {
        if ncols > 0 && i % ncols == 0 {
            continue; // row start: no left neighbour
        }
        let p = syms[i - 1];
        let c = syms[i];
        *joint.entry((p << shift) | c).or_insert(0) += 1;
        *prev_marg.entry(p).or_insert(0) += 1;
        npairs += 1;
    }
    if npairs == 0 {
        return h0(syms);
    }
    // H(cur|prev) = -1/N Σ joint(p,c) log2( joint(p,c) / marg(p) )
    let mut acc = 0.0;
    for (&key, &cnt) in &joint {
        let p = key >> shift;
        let pc = prev_marg[&p] as f64;
        acc += cnt as f64 * (cnt as f64 / pc).log2();
    }
    -acc / npairs as f64
}

/// Order-2 conditional entropy H(cur | prev, prev2), row-aware. Field must be small.
fn h2_rowaware(syms: &[u32], ncols: usize, field_bits: u32) -> f64 {
    let s = field_bits.max(1);
    let mut joint: HashMap<u32, u64> = HashMap::new();
    let mut ctx: HashMap<u32, u64> = HashMap::new();
    let mut n = 0u64;
    for i in 2..syms.len() {
        if ncols > 0 && (i % ncols == 0 || i % ncols == 1) {
            continue;
        }
        let key2 = (syms[i - 2] << s) | syms[i - 1];
        let key = (key2 << s) | syms[i];
        *joint.entry(key).or_insert(0) += 1;
        *ctx.entry(key2).or_insert(0) += 1;
        n += 1;
    }
    if n == 0 {
        return h0(syms);
    }
    let mut acc = 0.0;
    for (&key, &cnt) in &joint {
        let c2 = key >> s;
        let cc = ctx[&c2] as f64;
        acc += cnt as f64 * (cnt as f64 / cc).log2();
    }
    -acc / n as f64
}

/// H(cur | up-neighbour) = vertical/column context (position i predicted from i-ncols).
fn h1_col(syms: &[u32], ncols: usize, field_bits: u32) -> f64 {
    let s = field_bits.max(1);
    let mut joint: HashMap<u32, u64> = HashMap::new();
    let mut marg: HashMap<u32, u64> = HashMap::new();
    let mut n = 0u64;
    for i in ncols..syms.len() {
        let p = syms[i - ncols];
        let c = syms[i];
        *joint.entry((p << s) | c).or_insert(0) += 1;
        *marg.entry(p).or_insert(0) += 1;
        n += 1;
    }
    if n == 0 {
        return h0(syms);
    }
    let mut acc = 0.0;
    for (&k, &cnt) in &joint {
        acc += cnt as f64 * (cnt as f64 / marg[&(k >> s)] as f64).log2();
    }
    -acc / n as f64
}

/// Per-column conditional entropy H(value | column): the achievable rate of a context-
/// adaptive coder whose context is the output-channel (column). Captures per-channel scale
/// — the property that makes per-channel quantization work.
fn h0_percol(syms: &[u32], ncols: usize) -> f64 {
    if ncols == 0 || syms.len() < ncols {
        return h0(syms);
    }
    let nrows = syms.len() / ncols;
    let mut total = 0.0f64;
    let mut col = Vec::with_capacity(nrows);
    for j in 0..ncols {
        col.clear();
        let mut i = j;
        while i < syms.len() {
            col.push(syms[i]);
            i += ncols;
        }
        total += h0(&col) * col.len() as f64;
    }
    total / syms.len() as f64
}

/// H(cur | other) where `other` is a parallel stream (e.g. same position in the previous
/// LAYER). The achievable rate of predicting one tensor's field from another's.
fn h1_cross(cur: &[u32], other: &[u32], field_bits: u32) -> f64 {
    let s = field_bits.max(1);
    let n = cur.len().min(other.len());
    let mut joint: HashMap<u32, u64> = HashMap::new();
    let mut marg: HashMap<u32, u64> = HashMap::new();
    for i in 0..n {
        *joint.entry((other[i] << s) | cur[i]).or_insert(0) += 1;
        *marg.entry(other[i]).or_insert(0) += 1;
    }
    if n == 0 {
        return h0(cur);
    }
    let mut acc = 0.0;
    for (&k, &cnt) in &joint {
        acc += cnt as f64 * (cnt as f64 / marg[&(k >> s)] as f64).log2();
    }
    -acc / n as f64
}

/// Strip `.layers.<N>.` (or `.h.<N>.`) → ("model.layers.{}.role", N) so tensors that play
/// the same role in consecutive layers can be matched for cross-layer prediction.
fn layer_role(name: &str) -> Option<(String, usize)> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, p) in parts.iter().enumerate() {
        if (*p == "layers" || *p == "h") && i + 1 < parts.len() {
            if let Ok(n) = parts[i + 1].parse::<usize>() {
                let mut key = parts.clone();
                key[i + 1] = "{}";
                return Some((key.join("."), n));
            }
        }
    }
    None
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/smollm2-135m/model.safetensors".to_string());
    eprintln!("[reeform-probe] {path}");
    let mut r = SafetensorsReader::open(&path)?;
    let names: Vec<String> = r.list_tensors().iter().map(|s| s.to_string()).collect();

    // Load all bf16 2-D matrices once (135M ≈ 270 MB of u16 — fine on this machine).
    struct Tn {
        full: Vec<u32>,
        exp: Vec<u32>,
        ncols: usize,
    }
    let mut tn: HashMap<String, Tn> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for name in &names {
        let meta = r.to_rllm_meta(name)?;
        if meta.dtype != DType::Bf16 || meta.shape.len() != 2 {
            continue;
        }
        let bytes = r.read_tensor(name)?;
        let ncols = *meta.shape.last().unwrap() as usize;
        let n = bytes.len() / 2;
        let mut full = Vec::with_capacity(n);
        let mut exp = Vec::with_capacity(n);
        for k in 0..n {
            let v = u16::from_le_bytes([bytes[2 * k], bytes[2 * k + 1]]) as u32;
            full.push(v);
            exp.push((v >> 7) & 0xFF);
        }
        tn.insert(name.clone(), Tn { full, exp, ncols });
        order.push(name.clone());
    }

    // --- Per-tensor: order-0 floor + spatial (row vs column) exp context ---
    let mut w = 0u64;
    let (mut h0f, mut h0e, mut h0s, mut h0m) = (0.0f64, 0.0, 0.0, 0.0);
    let (mut e_row, mut e_col) = (0.0f64, 0.0);
    let (mut e_pcol, mut f_pcol, mut m_pcol) = (0.0f64, 0.0, 0.0);
    for name in &order {
        let t = &tn[name];
        let c = t.full.len() as f64;
        let sign: Vec<u32> = t.full.iter().map(|v| (v >> 15) & 1).collect();
        let man: Vec<u32> = t.full.iter().map(|v| v & 0x7F).collect();
        h0f += h0(&t.full) * c;
        h0e += h0(&t.exp) * c;
        h0s += h0(&sign) * c;
        h0m += h0(&man) * c;
        e_row += h1_rowaware(&t.exp, t.ncols, 8) * c;
        e_col += h1_col(&t.exp, t.ncols, 8) * c;
        e_pcol += h0_percol(&t.exp, t.ncols) * c;
        f_pcol += h0_percol(&t.full, t.ncols) * c;
        m_pcol += h0_percol(&man, t.ncols) * c;
        w += t.full.len() as u64;
    }

    // --- Cross-layer: predict layer L's exp / full from layer L-1 (residual-stream bet) ---
    let mut roles: HashMap<String, Vec<(usize, String)>> = HashMap::new();
    for name in &order {
        if let Some((role, n)) = layer_role(name) {
            roles.entry(role).or_default().push((n, name.clone()));
        }
    }
    let (mut clw, mut cl_e0, mut cl_e, mut cl_f0, mut cl_fd) = (0u64, 0.0f64, 0.0, 0.0, 0.0);
    for (_role, mut list) in roles {
        list.sort();
        for pair in list.windows(2) {
            let (a, b) = (&tn[&pair[0].1], &tn[&pair[1].1]);
            if a.full.len() != b.full.len() {
                continue;
            }
            let c = b.full.len() as f64;
            cl_e0 += h0(&b.exp) * c; // exp baseline (this layer alone)
            cl_e += h1_cross(&b.exp, &a.exp, 8) * c; // exp | prev-layer exp
            cl_f0 += h0(&b.full) * c; // full baseline
            let delta: Vec<u32> = b
                .full
                .iter()
                .zip(&a.full)
                .map(|(x, y)| x.wrapping_sub(*y) & 0xFFFF)
                .collect();
            cl_fd += h0(&delta) * c; // H0 of cross-layer integer delta
            clw += b.full.len() as u64;
        }
    }

    let wf = w as f64;
    let floor = h0f / wf;
    println!(
        "\n=== REEFORM Phase-0 — SmolLM2-135M bf16, {} matrices, {w} weights ===\n",
        order.len()
    );
    println!("ORDER-0 FLOOR (rANS / bit-plane / DFloat11 all land here):");
    println!(
        "  H0(full)  = {floor:.4} bit/weight    sign {:.4} | exp {:.4} | man {:.4}",
        h0s / wf,
        h0e / wf,
        h0m / wf
    );
    println!("\nEXPONENT (the only field with < random entropy, so the only lever):");
    println!("  H0(exp)            = {:.4}", h0e / wf);
    println!("  H(exp | left)      = {:.4}   (row neighbour)", e_row / wf);
    println!(
        "  H(exp | up)        = {:.4}   (column neighbour)",
        e_col / wf
    );
    println!(
        "  H(exp | column)    = {:.4}   (per-channel adaptive) *",
        e_pcol / wf
    );
    println!(
        "  H(man | column)    = {:.4}   (vs H0(man) {:.4}) *",
        m_pcol / wf,
        h0m / wf
    );
    println!(
        "  H(full | column)   = {:.4}   (vs floor {:.4}) *",
        f_pcol / wf,
        floor
    );
    println!("  * per-column empirical entropy ignores model-transmit cost — treat as an");
    println!("    OPTIMISTIC bound; only a LARGE gap survives the per-channel model overhead.");
    if clw > 0 {
        let cw = clw as f64;
        println!(
            "\nCROSS-LAYER (residual-stream bet, {} layer-pairs-worth of weights):",
            clw
        );
        println!("  H0(exp)            = {:.4}   (this layer)", cl_e0 / cw);
        println!(
            "  H(exp | prevlayer) = {:.4}   (predict exp from previous layer)",
            cl_e / cw
        );
        println!("  H0(full)           = {:.4}", cl_f0 / cw);
        println!(
            "  H0(full Δ prevlayer)= {:.4}  (cross-layer integer delta residual)",
            cl_fd / cw
        );
    }

    // Best lossless rate we can construct from the measured structure: keep sign(1)+man(7),
    // code exp with its best available context (min of row/col).
    let best_exp = (e_row / wf).min(e_col / wf).min(h0e / wf);
    let ours = h0s / wf + best_exp + h0m / wf;
    let win = floor - ours;
    println!("\n--- VERDICT (spatial) ---");
    println!("  floor {floor:.4} | ours(best-exp-context) {ours:.4} | Δ {win:+.4} bit/weight");
    if win > 0.001 {
        println!(
            "  ✅ structure found — {:.2}% lossless gain available",
            win / floor * 100.0
        );
    } else {
        println!("  ❌ spatial exp context ≈ null. Real signal (if any) is cross-layer/low-rank — see above.");
    }
    Ok(())
}
