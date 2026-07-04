// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada
//
// Dumps the RAW zigzag-int-delta byte stream (Δ = W_ft − W_base, bf16 patterns) to a file —
// the exact input our u16-rANS coder sees. Lets us compare a GENERIC codec (zstd/gzip) against
// our coder on the *identical* delta, to test whether the delta redundancy is something only
// our coder captures (it is not — entropy coders are near-optimal; the win is the delta framing).
//
//   reeform-dumpdelta <ft.safetensors> <base.safetensors> <out.bin>

use anyhow::Result;
use spissa_container::DType;
use spissa_import::SafetensorsReader;
use std::io::Write;

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

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (ft, base, out) = (&a[1], &a[2], &a[3]);
    let mut rb = SafetensorsReader::open(base)?;
    let mut rf = SafetensorsReader::open(ft)?;
    let bnames: std::collections::HashSet<String> =
        rb.list_tensors().iter().map(|s| s.to_string()).collect();
    let fnames: Vec<String> = rf.list_tensors().iter().map(|s| s.to_string()).collect();
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    let (mut wdelta, mut wraw) = (0u64, 0u64);
    for name in &fnames {
        let m = rf.to_rllm_meta(name)?;
        let fbytes = rf.read_tensor(name)?;
        if m.dtype == DType::Bf16 && bnames.contains(name) {
            let bm = rb.to_rllm_meta(name)?;
            if bm.shape == m.shape {
                let bbytes = rb.read_tensor(name)?;
                if bbytes.len() == fbytes.len() {
                    let (bv, fv) = (u16s(&bbytes), u16s(&fbytes));
                    for i in 0..fv.len() {
                        f.write_all(&zigzag(fv[i].wrapping_sub(bv[i])).to_le_bytes())?;
                    }
                    wdelta += fv.len() as u64;
                    continue;
                }
            }
        }
        f.write_all(&fbytes)?; // non-matching: raw fallback (matches .spsd kind=1)
        wraw += fbytes.len() as u64;
    }
    f.flush()?;
    eprintln!("delta-coded weights: {wdelta} · raw-fallback bytes: {wraw}");
    Ok(())
}
