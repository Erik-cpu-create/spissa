// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — research instrument (REEFORM, SECRET IP).
//
// Stage-B realisation probe: how big is a base-exponent-conditioned delta coder in REAL bytes?
// Partition the zigzag Δ by the base bf16 exponent (the decoder recomputes it for free from the
// base it already has), give each occupied exponent its own static u16-rANS table, and measure the
// real encoded stream + a COMPACT table cost (freqs stored only up to the exponent's max symbol).
// Compares to the single-global-table coder (today's 7.74) and the ideal H(Δ|base-exp) (7.10).
//
//   reeform-bxtable <ft.safetensors> <base.safetensors>

use anyhow::Result;
use rtc_codec::delta::{encode_stream, Tables};
use spissa_container::DType;
use spissa_import::SafetensorsReader;

#[inline]
fn zigzag(d: u16) -> u16 {
    let x = d as i16;
    (x.wrapping_shl(1) ^ (x >> 15)) as u16
}
fn u16s(b: &[u8]) -> Vec<u16> {
    (0..b.len() / 2)
        .map(|i| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))
        .collect()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (ft, base) = (&a[1], &a[2]);
    let mut rb = SafetensorsReader::open(base)?;
    let mut rf = SafetensorsReader::open(ft)?;
    let bn: std::collections::HashSet<String> =
        rb.list_tensors().iter().map(|s| s.to_string()).collect();
    let fnames: Vec<String> = rf.list_tensors().iter().map(|s| s.to_string()).collect();

    // Bucket the zigzag deltas by base exponent (256 possible), preserving order within a bucket.
    let mut buckets: Vec<Vec<u16>> = (0..256).map(|_| Vec::new()).collect();
    let mut global: Vec<u16> = Vec::new();
    for name in &fnames {
        let m = rf.to_rllm_meta(name)?;
        if m.dtype != DType::Bf16 || !bn.contains(name) {
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
        for i in 0..fv.len() {
            let z = zigzag(fv[i].wrapping_sub(bv[i]));
            buckets[((bv[i] >> 7) & 0xFF) as usize].push(z);
            global.push(z);
        }
    }
    let n: u64 = global.len() as u64;

    // (1) baseline: one global table (today's shipped delta coder).
    let mut ghist = vec![0u64; 65536];
    for &s in &global {
        ghist[s as usize] += 1;
    }
    let gt = Tables::from_hist(&ghist);
    let g_enc = encode_stream(&global, &gt).len() as u64;
    let g_tbl = compact_table_bytes(&ghist);

    // (2) base-exponent-conditioned: one table per occupied exponent, compact storage.
    // (3) HYBRID: per group, use a per-exponent table OR the (already-stored) global table —
    //     whichever encodes smaller including the table cost. Drops wasteful large-exp tables.
    let (mut enc_total, mut tbl_total, mut groups) = (0u64, 0u64, 0u64);
    let (mut hy_enc, mut hy_tbl, mut hy_tables) = (0u64, g_tbl, 0u64); // hybrid always keeps global
    for b in &buckets {
        if b.is_empty() {
            continue;
        }
        groups += 1;
        let mut h = vec![0u64; 65536];
        for &s in b {
            h[s as usize] += 1;
        }
        let t = Tables::from_hist(&h);
        let per_exp_enc = encode_stream(b, &t).len() as u64;
        let per_exp_tbl = compact_table_bytes(&h);
        enc_total += per_exp_enc;
        tbl_total += per_exp_tbl;
        // hybrid: per-exp (enc+tbl) vs global table (enc only, table already paid)
        let global_enc = encode_stream(b, &gt).len() as u64;
        if per_exp_enc + per_exp_tbl < global_enc {
            hy_enc += per_exp_enc;
            hy_tbl += per_exp_tbl;
            hy_tables += 1;
        } else {
            hy_enc += global_enc;
        }
    }

    let bpw = |bytes: u64| bytes as f64 * 8.0 / n as f64;
    let mib = |bytes: u64| bytes as f64 / 1048576.0;
    println!("=== REEFORM Stage-B realisation: base-exp-conditioned coder (REAL bytes) ===");
    println!("weights {n}");
    println!(
        "global (1 table)      : enc {:.4} + tbl {:.4} = {:.4} bit/w   (tbl {:.2} MiB)",
        bpw(g_enc),
        bpw(g_tbl),
        bpw(g_enc + g_tbl),
        mib(g_tbl)
    );
    println!(
        "base-exp ({groups} tables) : enc {:.4} + tbl {:.4} = {:.4} bit/w   (tbl {:.2} MiB)",
        bpw(enc_total),
        bpw(tbl_total),
        bpw(enc_total + tbl_total),
        mib(tbl_total)
    );
    println!(
        "→ Δ vs global = {:+.4} bit/w  ({:.1}% smaller)   [ideal H(Δ|base-exp)=7.10]",
        bpw(enc_total + tbl_total) - bpw(g_enc + g_tbl),
        100.0 * (1.0 - (enc_total + tbl_total) as f64 / (g_enc + g_tbl) as f64)
    );
    println!(
        "HYBRID ({hy_tables} per-exp + global): enc {:.4} + tbl {:.4} = {:.4} bit/w   (tbl {:.2} MiB)",
        bpw(hy_enc),
        bpw(hy_tbl),
        bpw(hy_enc + hy_tbl),
        mib(hy_tbl)
    );
    println!(
        "→ HYBRID vs global = {:+.4} bit/w  ({:.1}% smaller, BIT-EXACT)",
        bpw(hy_enc + hy_tbl) - bpw(g_enc + g_tbl),
        100.0 * (1.0 - (hy_enc + hy_tbl) as f64 / (g_enc + g_tbl) as f64)
    );
    Ok(())
}

/// Compact freq-table cost: store freqs only up to the highest occupied symbol, 2 bytes each
/// (freqs are normalized to ≤ PROB_SCALE so a u16-ish varint is plenty; we model 2 bytes/entry).
fn compact_table_bytes(hist: &[u64]) -> u64 {
    let max_sym = (0..65536).rev().find(|&s| hist[s] > 0).unwrap_or(0);
    // 4 bytes header (max_sym) + 2 bytes per freq entry up to max_sym.
    4 + (max_sym as u64 + 1) * 2
}
