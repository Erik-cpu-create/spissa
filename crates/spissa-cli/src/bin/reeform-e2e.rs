// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. CONFIDENTIAL RESEARCH (REEFORM).
//
//! REEFORM end-to-end proof: the fine-tune-delta invention through the SHIPPED rANS codec, with
//! REAL bytes (not theoretical entropy) and full reconstruction. For each matched bf16 matrix:
//!   baseline = rANS(W_ft)                         — ship the full fine-tune
//!   ours     = rANS(Δ),  Δ = W_ft − W_base        — ship base once + compressed delta
//!   decode ours → Δ → W_ft = W_base + Δ → assert bit-exact vs the original.
//!
//! Run: reeform-e2e <base.safetensors> <finetune.safetensors>

use rtc_codec::{DecodeMeta, EncodeMeta, RansCodec, TensorCodec};
use spissa_container::DType;
use spissa_import::SafetensorsReader;

fn read_u16(r: &mut SafetensorsReader, name: &str) -> anyhow::Result<Vec<u16>> {
    let b = r.read_tensor(name)?;
    Ok((0..b.len() / 2).map(|i| u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).collect())
}
fn enc_len(bytes: &[u8], dtype: &str) -> anyhow::Result<(Vec<u8>, usize)> {
    let meta = EncodeMeta { name: "x".into(), shape: vec![(bytes.len() / 2) as u64], dtype: dtype.into() };
    let chunk = RansCodec.encode(bytes, &meta)?;
    Ok((chunk.data, bytes.len()))
}

fn main() -> anyhow::Result<()> {
    let base = std::env::args().nth(1).unwrap_or_else(|| "models/smollm2-135m/model.safetensors".into());
    let ft = std::env::args().nth(2).unwrap_or_else(|| "models/downloads/smollm2-135m-instruct/model.safetensors".into());
    eprintln!("[reeform-e2e] base={base} ft={ft}");
    let mut rb = SafetensorsReader::open(&base)?;
    let mut rf = SafetensorsReader::open(&ft)?;
    let bnames: Vec<String> = rb.list_tensors().iter().map(|s| s.to_string()).collect();
    let fset: std::collections::HashSet<String> = rf.list_tensors().iter().map(|s| s.to_string()).collect();

    let (mut weights, mut bytes_ft_raw) = (0u64, 0u64);
    let (mut comp_ft, mut comp_delta) = (0u64, 0u64);
    let mut tensors = 0usize;
    let mut all_exact = true;
    for name in &bnames {
        if !fset.contains(name) {
            continue;
        }
        let m = rb.to_rllm_meta(name)?;
        if m.dtype != DType::Bf16 || m.shape.len() != 2 {
            continue;
        }
        let b = read_u16(&mut rb, name)?;
        let f = read_u16(&mut rf, name)?;
        if b.len() != f.len() {
            continue;
        }
        let n = b.len();
        // raw ft bytes (LE) and int-delta bytes (LE)
        let mut ft_bytes = Vec::with_capacity(n * 2);
        let mut d_bytes = Vec::with_capacity(n * 2);
        let mut delta = Vec::with_capacity(n);
        for i in 0..n {
            ft_bytes.extend_from_slice(&f[i].to_le_bytes());
            let d = f[i].wrapping_sub(b[i]); // wrapped int delta
            // zigzag the SIGNED interpretation so small |Δ| → small unsigned → high byte ≈ 0,
            // letting the byte-wise rANS recover the u16-symbol structure it otherwise loses.
            let x = d as i16;
            let z = (x.wrapping_shl(1) ^ (x >> 15)) as u16;
            delta.push(z);
            d_bytes.extend_from_slice(&z.to_le_bytes());
        }
        // baseline: rANS the full fine-tune (bf16 path)
        let (cft, _) = enc_len(&ft_bytes, "bf16")?;
        // ours: rANS the zigzag delta. Try both the byte path and the 16-bit-symbol (bf16)
        // path — the latter's exponent-field coder crushes the mostly-zero high bits of the
        // zigzag — and keep whichever is smaller (both are lossless).
        let (cd_u8, _) = enc_len(&d_bytes, "u8")?;
        let (cd_bf, _) = enc_len(&d_bytes, "bf16")?;
        let cd = if cd_bf.len() < cd_u8.len() { cd_bf } else { cd_u8 };
        // decode ours and reconstruct ft, verify bit-exact
        let dmeta = DecodeMeta { codec_id: "rtc-rans-v1".into(), uncompressed_size: d_bytes.len() as u64 };
        let dec = RansCodec.decode(&cd, &dmeta)?;
        let dec16: Vec<u16> = (0..dec.len() / 2).map(|i| u16::from_le_bytes([dec[2 * i], dec[2 * i + 1]])).collect();
        for i in 0..n {
            // un-zigzag → signed Δ → ft = base + Δ
            let z = dec16[i];
            let x = ((z >> 1) as i16) ^ -((z & 1) as i16);
            let recon = b[i].wrapping_add(x as u16);
            if recon != f[i] {
                all_exact = false;
            }
        }
        let _ = delta;
        weights += n as u64;
        bytes_ft_raw += (n * 2) as u64;
        comp_ft += cft.len() as u64;
        comp_delta += cd.len() as u64;
        tensors += 1;
    }

    let wf = weights as f64;
    let bft = comp_ft as f64 * 8.0 / wf;
    let bdelta = comp_delta as f64 * 8.0 / wf;
    println!("\n=== REEFORM end-to-end (real rANS bytes) — {tensors} matrices, {weights} weights ===\n");
    println!("  raw bf16                       = 16.0000 bit/weight");
    println!("  rANS(full fine-tune)  baseline = {bft:.4} bit/weight   ({} bytes)", comp_ft);
    println!("  rANS(delta)  OURS              = {bdelta:.4} bit/weight   ({} bytes)", comp_delta);
    println!("  reconstruction ft = base + Δ   : {}", if all_exact { "✅ BIT-EXACT" } else { "❌ MISMATCH (bug)" });
    let win = bft - bdelta;
    println!("\n--- VERDICT (real codec, real bytes) ---");
    if win > 0.05 && all_exact {
        println!("  ✅ delta is {win:.4} bit/weight smaller = {:.1}% LOSSLESS reduction vs shipping the full fine-tune", win / bft * 100.0);
        println!("     {} MB → {} MB per fine-tune (base stored once)", comp_ft / 1_048_576, comp_delta / 1_048_576);
    } else {
        println!("  result: baseline {bft:.3} vs delta {bdelta:.3}");
    }
    let _ = bytes_ft_raw;
    Ok(())
}
