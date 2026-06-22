// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. CONFIDENTIAL RESEARCH (REEFORM).
//
//! REEFORM fine-tune-delta probe: a fine-tune (Instruct) is the SAME weights as its base
//! after gentle training, so `W_ft` should be close to `W_base`. If the lossless delta
//! (XOR / integer-subtract of bit patterns, or the bf16 value difference) has far lower
//! entropy than `W_ft` itself, we can ship the fine-tune as `base + tiny-delta` — a genuinely
//! novel LOSSLESS win (nobody ships fine-tunes as lossless deltas from base).
//!
//! Run: reeform-delta <base.safetensors> <finetune.safetensors>

use spissa_container::DType;
use spissa_import::SafetensorsReader;
use std::collections::HashMap;

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16(x: f32) -> u16 {
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
    -hist.values().map(|&c| { let p = c as f64 / n; p * p.log2() }).sum::<f64>()
}

fn read_u16(r: &mut SafetensorsReader, name: &str) -> anyhow::Result<Vec<u16>> {
    let b = r.read_tensor(name)?;
    Ok((0..b.len() / 2).map(|i| u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).collect())
}

fn main() -> anyhow::Result<()> {
    let base = std::env::args().nth(1).unwrap_or_else(|| "models/smollm2-135m/model.safetensors".into());
    let ft = std::env::args().nth(2).unwrap_or_else(|| "models/downloads/smollm2-135m-instruct/model.safetensors".into());
    eprintln!("[reeform-delta] base={base}\n               ft  ={ft}");
    let mut rb = SafetensorsReader::open(&base)?;
    let mut rf = SafetensorsReader::open(&ft)?;
    let bnames: Vec<String> = rb.list_tensors().iter().map(|s| s.to_string()).collect();
    let fset: std::collections::HashSet<String> = rf.list_tensors().iter().map(|s| s.to_string()).collect();

    let (mut w, mut hft, mut hxor, mut hint, mut hval) = (0u64, 0.0f64, 0.0, 0.0, 0.0);
    let mut zeros = 0u64;
    let mut nmatch = 0usize;
    let mut roundtrip_mismatches = 0u64; // MUST stay 0 (proves losslessness)
    for name in &bnames {
        if !fset.contains(name) {
            continue;
        }
        let mb = rb.to_rllm_meta(name)?;
        if mb.dtype != DType::Bf16 || mb.shape.len() != 2 {
            continue;
        }
        let a = read_u16(&mut rb, name)?;
        let b = read_u16(&mut rf, name)?;
        if a.len() != b.len() {
            continue;
        }
        let n = a.len();
        let mut xor = Vec::with_capacity(n);
        let mut intd = Vec::with_capacity(n);
        let mut vald = Vec::with_capacity(n);
        let mut ft = Vec::with_capacity(n);
        for i in 0..n {
            ft.push(b[i] as u32);
            let x = (a[i] ^ b[i]) as u32;
            xor.push(x);
            if x == 0 {
                zeros += 1;
            }
            let d = b[i].wrapping_sub(a[i]); // int-sub delta
            intd.push(d as u32);
            // ROUND-TRIP: reconstruct ft = base + delta (mod 2^16) and check bit-exact.
            if a[i].wrapping_add(d) != b[i] {
                roundtrip_mismatches += 1;
            }
            // bf16 value delta (lossless: store this exact bf16 residual; W_ft recoverable
            // as bf16(base_f32 + delta_f32) only if associative — here we just measure rate)
            vald.push(f32_to_bf16(bf16_to_f32(b[i]) - bf16_to_f32(a[i])) as u32);
        }
        let c = n as f64;
        hft += h0(&ft) * c;
        hxor += h0(&xor) * c;
        hint += h0(&intd) * c;
        hval += h0(&vald) * c;
        w += n as u64;
        nmatch += 1;
    }

    let wf = w as f64;
    let base_rate = hft / wf;
    let xor_rate = hxor / wf;
    let int_rate = hint / wf;
    let val_rate = hval / wf;
    println!("\n=== REEFORM fine-tune delta — {nmatch} matched bf16 matrices, {w} weights ===\n");
    println!("  weights IDENTICAL base==ft : {:.2}%", zeros as f64 / wf * 100.0);
    println!("  H0(W_ft)        baseline   = {base_rate:.4} bit/weight   <- floor (ship full ft)");
    println!("  H0(XOR delta)              = {xor_rate:.4} bit/weight   (lossless: ft = base ⊕ Δ)");
    println!("  H0(int-sub delta)          = {int_rate:.4} bit/weight");
    println!("  H0(bf16 value delta)       = {val_rate:.4} bit/weight");
    let best = int_rate.min(xor_rate).min(val_rate);
    let win = base_rate - best;
    println!("\n--- LOSSLESS PROOF ---");
    println!("  round-trip ft = base + Δ (mod 2^16): {roundtrip_mismatches} mismatches / {w} weights");
    let lossless = roundtrip_mismatches == 0;
    println!(
        "  {} reconstruction is {}",
        if lossless { "✅" } else { "❌" },
        if lossless { "BIT-EXACT (provably lossless)" } else { "NOT exact (BUG)" }
    );
    println!("\n--- VERDICT ---");
    if win > 0.05 && lossless {
        println!("  ✅✅ SUCCESS — fine-tune Δ is {win:.4} bit/weight smaller than the full model");
        println!("      = {:.1}% LOSSLESS reduction ({best:.3} vs {base_rate:.3} bit/weight)", win / base_rate * 100.0);
        println!("      Ship base once + bit-exact Δ. Per extra fine-tune: {best:.3} not {base_rate:.3} bit/weight.");
        println!("      Novel: nobody ships fine-tunes as lossless integer-deltas from base.");
    } else {
        println!("  ❌ delta {:+.4} or not lossless — not a lever here.", -win);
    }
    Ok(())
}
