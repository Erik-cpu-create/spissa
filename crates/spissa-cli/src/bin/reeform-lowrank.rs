// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. CONFIDENTIAL RESEARCH (REEFORM).
//
//! REEFORM low-rank probe: does removing a rank-k component from a weight matrix yield a
//! residual whose lossless entropy (esp. the exponent) drops enough — net of the U,V cost —
//! to beat the order-0 floor? This is the last untested structural lever (global low-rank,
//! invisible to neighbour / cross-layer / per-channel probes). bf16 → f32, power-iteration
//! deflation for top-k singular triplets, residual re-quantised to bf16, measure entropy.

use spissa_container::DType;
use spissa_import::SafetensorsReader;
use std::collections::HashMap;

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
fn f32_to_bf16(x: f32) -> u16 {
    // round-to-nearest-even bf16
    let u = x.to_bits();
    let round = ((u >> 16) & 1) + 0x7FFF;
    ((u.wrapping_add(round)) >> 16) as u16
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

fn h0_bits(w: &[u16]) -> (f64, f64) {
    let full: Vec<u32> = w.iter().map(|&v| v as u32).collect();
    let exp: Vec<u32> = w.iter().map(|&v| ((v >> 7) & 0xFF) as u32).collect();
    (h0(&full), h0(&exp))
}

/// Top-k singular triplets by power iteration with deflation, all in f32. Returns residual
/// `W - Σ σ u vᵀ` flattened row-major.
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
            // u = R v
            for i in 0..m {
                let row = &r[i * n..i * n + n];
                u[i] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
            }
            let un = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            if un < 1e-12 {
                break;
            }
            for x in &mut u {
                *x /= un;
            }
            // v = Rᵀ u
            for j in 0..n {
                v[j] = (0..m).map(|i| r[i * n + j] * u[i]).sum();
            }
            sigma = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if sigma < 1e-12 {
                break;
            }
            for x in &mut v {
                *x /= sigma;
            }
        }
        // deflate: R -= sigma * u vᵀ
        for i in 0..m {
            let ui = sigma * u[i];
            let row = &mut r[i * n..i * n + n];
            for j in 0..n {
                row[j] -= ui * v[j];
            }
        }
    }
    r
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/smollm2-135m/model.safetensors".to_string());
    let k: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    eprintln!("[reeform-lowrank] {path}  rank-k={k}");
    let mut r = SafetensorsReader::open(&path)?;
    let names: Vec<String> = r.list_tensors().iter().map(|s| s.to_string()).collect();

    // Pick the largest few 2-D bf16 matrices (skip embeddings — tied/huge/special).
    let mut cand: Vec<(String, usize, usize)> = Vec::new();
    for name in &names {
        let meta = r.to_rllm_meta(name)?;
        if meta.dtype != DType::Bf16 || meta.shape.len() != 2 {
            continue;
        }
        if name.contains("embed") || name.contains("lm_head") {
            continue;
        }
        let (m, n) = (meta.shape[0] as usize, meta.shape[1] as usize);
        cand.push((name.clone(), m, n));
    }
    cand.sort_by_key(|c| std::cmp::Reverse(c.1 * c.2));
    cand.truncate(6);

    println!("\n=== REEFORM low-rank probe — {path} (rank-{k}) ===");
    println!("{:<46} {:>8} {:>8} {:>8} {:>8}", "tensor (m×n)", "H0(W)", "H0(R)", "expW→R", "overhd");
    let (mut sw, mut sr, mut sov) = (0.0f64, 0.0, 0.0);
    let mut wt = 0u64;
    for (name, m, n) in &cand {
        let bytes = r.read_tensor(name)?;
        let w16: Vec<u16> = (0..bytes.len() / 2)
            .map(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
            .collect();
        let wf: Vec<f32> = w16.iter().map(|&b| bf16_to_f32(b)).collect();
        let (h0w, expw) = h0_bits(&w16);
        let res = lowrank_residual(&wf, *m, *n, k, 24);
        let r16: Vec<u16> = res.iter().map(|&x| f32_to_bf16(x)).collect();
        let (h0r, expr) = h0_bits(&r16);
        // overhead of storing U,V as bf16: k*(m+n)*16 bits over m*n weights
        let overhead = (k * (m + n)) as f64 * 16.0 / (m * n) as f64;
        let label = format!("{} ({}×{})", name.split('.').next_back().unwrap_or(name), m, n);
        println!(
            "{:<46} {h0w:>8.3} {h0r:>8.3} {expw:>4.2}→{expr:<3.2} {overhead:>7.3}",
            label.chars().take(46).collect::<String>()
        );
        let c = (m * n) as f64;
        sw += h0w * c;
        sr += h0r * c;
        sov += overhead * c;
        wt += (m * n) as u64;
    }
    let wf = wt as f64;
    let baseline = sw / wf;
    let residual = sr / wf;
    let overhead = sov / wf;
    let net = residual + overhead;
    println!("\n--- aggregate (top {} matrices) ---", cand.len());
    println!("  floor H0(W)            = {baseline:.4} bit/weight");
    println!("  H0(residual)           = {residual:.4}  (+ U,V overhead {overhead:.4})");
    println!("  NET low-rank lossless  = {net:.4} bit/weight");
    let win = baseline - net;
    if win > 0.01 {
        println!("  ✅ low-rank BEATS the floor by {win:.4} bit/weight ({:.2}%, LOSSLESS)", win / baseline * 100.0);
    } else {
        println!("  ❌ low-rank net {:+.4} — residual entropy + U,V cost ≥ floor. Lever is dead.", -win);
    }
    Ok(())
}
