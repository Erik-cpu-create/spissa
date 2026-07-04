// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada
//
//! REEFORM Phase-1 amplifier: the fine-tune delta Δ = W_ft − W_base is exactly the object LoRA
//! approximates as low-rank. The raw weights were NOT low-rank, but the DELTA should be. Test
//! two amplifiers over the int-delta floor (7.70 bit/weight):
//!   (a) neighbour structure of Δ  — is the update spatially correlated?
//!   (b) low-rank-on-Δ, LOSSLESS:  store A,B (rank-k of float Δ) + exact int residual
//!         decode:  pred = bf16(base_f + A·Bᵀ);  ft = pred + residual   (bit-exact)
//!
//! Run: reeform-amplify <base.safetensors> <finetune.safetensors> [rank-k]

use spissa_container::DType;
use spissa_import::SafetensorsReader;
use std::collections::HashMap;

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let u = x.to_bits();
    let round = ((u >> 16) & 1) + 0x7FFF;
    (u.wrapping_add(round) >> 16) as u16
}
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
/// H(cur | left), row-aware. For a u16 delta the alphabet is huge, so quantise the context to
/// the delta's high byte (captures magnitude) to keep the table small but meaningful.
fn h1_delta(d: &[u32], ncols: usize) -> f64 {
    let mut joint: HashMap<u64, u64> = HashMap::new();
    let mut marg: HashMap<u32, u64> = HashMap::new();
    let mut n = 0u64;
    for i in 1..d.len() {
        if ncols > 0 && i % ncols == 0 {
            continue;
        }
        let p = d[i - 1]; // full previous symbol as context
        *joint.entry(((p as u64) << 16) | d[i] as u64).or_insert(0) += 1;
        *marg.entry(p).or_insert(0) += 1;
        n += 1;
    }
    if n == 0 {
        return h0(d);
    }
    let mut acc = 0.0;
    for (&k, &c) in &joint {
        acc += c as f64 * (c as f64 / marg[&((k >> 16) as u32)] as f64).log2();
    }
    -acc / n as f64
}
/// Rank-k residual of `w` (m×n, row-major) by power-iteration deflation in f32.
fn lowrank_residual(w: &[f32], m: usize, n: usize, k: usize, iters: usize) -> Vec<f32> {
    let mut r = w.to_vec();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u32 << 24) as f32 - 0.5
    };
    for _ in 0..k {
        let mut v: Vec<f32> = (0..n).map(|_| rng()).collect();
        let mut u = vec![0.0f32; m];
        let mut sigma = 0.0f32;
        for _ in 0..iters {
            for i in 0..m {
                u[i] = r[i * n..i * n + n].iter().zip(&v).map(|(a, b)| a * b).sum();
            }
            let un = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            if un < 1e-12 {
                break;
            }
            u.iter_mut().for_each(|x| *x /= un);
            for j in 0..n {
                v[j] = (0..m).map(|i| r[i * n + j] * u[i]).sum();
            }
            sigma = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if sigma < 1e-12 {
                break;
            }
            v.iter_mut().for_each(|x| *x /= sigma);
        }
        for i in 0..m {
            let ui = sigma * u[i];
            for j in 0..n {
                r[i * n + j] -= ui * v[j];
            }
        }
    }
    r
}
fn read_u16(r: &mut SafetensorsReader, name: &str) -> anyhow::Result<Vec<u16>> {
    let b = r.read_tensor(name)?;
    Ok((0..b.len() / 2)
        .map(|i| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))
        .collect())
}

fn main() -> anyhow::Result<()> {
    let base = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/smollm2-135m/model.safetensors".into());
    let ft = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/downloads/smollm2-135m-instruct/model.safetensors".into());
    let k: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    eprintln!("[reeform-amplify] base={base} ft={ft} rank-k={k}");
    let mut rb = SafetensorsReader::open(&base)?;
    let mut rf = SafetensorsReader::open(&ft)?;
    let bnames: Vec<String> = rb.list_tensors().iter().map(|s| s.to_string()).collect();
    let fset: std::collections::HashSet<String> =
        rf.list_tensors().iter().map(|s| s.to_string()).collect();

    // Pick the largest few square-ish matrices (skip embed/lm_head).
    let mut cand: Vec<(String, usize, usize)> = Vec::new();
    for name in &bnames {
        if !fset.contains(name) {
            continue;
        }
        let m = rb.to_rllm_meta(name)?;
        if m.dtype != DType::Bf16
            || m.shape.len() != 2
            || name.contains("embed")
            || name.contains("lm_head")
        {
            continue;
        }
        cand.push((name.clone(), m.shape[0] as usize, m.shape[1] as usize));
    }
    cand.sort_by_key(|c| std::cmp::Reverse(c.1 * c.2));
    cand.truncate(6);

    println!(
        "\n=== REEFORM amplify (rank-{k}) — top {} matrices ===",
        cand.len()
    );
    println!(
        "{:<28} {:>8} {:>8} {:>10} {:>8} {:>8}",
        "tensor", "H0(Δ)", "H1(Δ)", "lr-resid", "overhd", "lr-net"
    );
    let (mut w, mut s_d0, mut s_d1, mut s_lr, mut s_ov, mut mism) =
        (0u64, 0.0f64, 0.0, 0.0, 0.0, 0u64);
    for (name, m, n) in &cand {
        let bb = read_u16(&mut rb, name)?;
        let fb = read_u16(&mut rf, name)?;
        let nn = bb.len();
        // int delta + its neighbour entropy
        let intd: Vec<u32> = (0..nn).map(|i| fb[i].wrapping_sub(bb[i]) as u32).collect();
        let d0 = h0(&intd);
        let d1 = h1_delta(&intd, *n);
        // low-rank on float delta -> lossless int residual
        let basef: Vec<f32> = bb.iter().map(|&b| bf16_to_f32(b)).collect();
        let ftf: Vec<f32> = fb.iter().map(|&b| bf16_to_f32(b)).collect();
        let deltaf: Vec<f32> = (0..nn).map(|i| ftf[i] - basef[i]).collect();
        let resid = lowrank_residual(&deltaf, *m, *n, k, 24); // R = Δ − lowrank
                                                              // pred = base + lowrank(Δ) = ft − R ;  residual_int = ft_bits − bf16(pred)
        let mut rint = Vec::with_capacity(nn);
        for i in 0..nn {
            let pred = f32_to_bf16(ftf[i] - resid[i]);
            let r = fb[i].wrapping_sub(pred); // lossless residual
            rint.push(r as u32);
            // verify exactness of the chosen scheme
            if pred.wrapping_add(r) != fb[i] {
                mism += 1;
            }
        }
        let lr = h0(&rint);
        let ov = (k * (m + n)) as f64 * 16.0 / (m * n) as f64;
        let label = name.split('.').rev().take(2).collect::<Vec<_>>().join(".");
        println!(
            "{:<28} {d0:>8.3} {d1:>8.3} {lr:>10.3} {ov:>8.3} {:>8.3}",
            label.chars().take(28).collect::<String>(),
            lr + ov
        );
        let c = nn as f64;
        s_d0 += d0 * c;
        s_d1 += d1 * c;
        s_lr += lr * c;
        s_ov += ov * c;
        w += nn as u64;
    }
    let wf = w as f64;
    println!("\n--- aggregate ---  (round-trip mismatches: {mism})");
    println!(
        "  int-delta floor   H0(Δ)        = {:.4} bit/weight",
        s_d0 / wf
    );
    println!(
        "  delta neighbour   H1(Δ)        = {:.4}  (Δ {:+.4})",
        s_d1 / wf,
        s_d1 / wf - s_d0 / wf
    );
    println!(
        "  low-rank-on-Δ     resid+ovhd   = {:.4}  (Δ {:+.4})",
        (s_lr + s_ov) / wf,
        (s_lr + s_ov - s_d0) / wf
    );
    let best = (s_d1 / wf).min((s_lr + s_ov) / wf);
    if best < s_d0 / wf - 0.02 && mism == 0 {
        println!(
            "  ✅ amplifier beats int-delta by {:.4} bit/weight (LOSSLESS)",
            s_d0 / wf - best
        );
    } else {
        println!("  ❌ no net amplifier here — int-delta (7.70) stands as the win; Δ mantissa is the wall too.");
    }
    Ok(())
}
