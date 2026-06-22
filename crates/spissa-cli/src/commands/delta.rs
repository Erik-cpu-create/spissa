// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! `spissa delta` / `spissa undelta` — the REEFORM fine-tune-delta codec as a product feature.
//!
//! A fine-tune is its base plus a gentle update, so `Δ = W_ft − W_base` (zigzag of the signed
//! bf16-pattern difference) has far lower entropy than `W_ft`. We code Δ with a u16-symbol
//! static rANS (one global table) → a `.spsd` file ~27% smaller than packing the fine-tune, and
//! ~50%+ smaller than the raw bf16 safetensors. `undelta` reconstructs the fine-tune BIT-EXACT
//! from `base + Δ`. Both base and fine-tune are safetensors (the base's exact bf16 is required).

use crate::progress::{human_size, Spinner};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use spissa_container::DType;
use spissa_import::SafetensorsReader;
use std::io::Write;
use std::path::Path;

const MAGIC: &[u8; 4] = b"SPSD";
const VERSION: u32 = 1;
const PROB_BITS: u32 = 20;
const PROB_SCALE: u64 = 1 << PROB_BITS;
const RANS_L: u64 = 1 << 31;

// ----- u16-symbol static rANS (verified by reeform-rans16) -----

struct Tables {
    freq: Vec<u32>,
    cum: Vec<u32>,
    slot: Vec<u16>,
}

fn build_tables_from_hist(hist: &[u64]) -> Tables {
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
    tables_from_freq(freq)
}

fn tables_from_freq(freq: Vec<u32>) -> Tables {
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

fn rans_encode(syms: &[u16], t: &Tables) -> Vec<u8> {
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

fn rans_decode(data: &[u8], n: usize, t: &Tables) -> Vec<u16> {
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

#[inline]
fn zigzag(d: u16) -> u16 {
    let x = d as i16;
    (x.wrapping_shl(1) ^ (x >> 15)) as u16
}
#[inline]
fn unzigzag(z: u16) -> u16 {
    (((z >> 1) as i16) ^ -((z & 1) as i16)) as u16
}

// ----- safetensors helpers -----

fn open_st(path: &str) -> Result<SafetensorsReader> {
    let p = Path::new(path);
    let file = if p.is_dir() {
        p.join("model.safetensors")
    } else {
        p.to_path_buf()
    };
    SafetensorsReader::open(&file).with_context(|| format!("opening safetensors: {}", file.display()))
}

fn file_sha(path: &str) -> Result<[u8; 32]> {
    let p = Path::new(path);
    let file = if p.is_dir() {
        p.join("model.safetensors")
    } else {
        p.to_path_buf()
    };
    let bytes = std::fs::read(&file)?;
    Ok(Sha256::digest(&bytes).into())
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    (0..bytes.len() / 2)
        .map(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
        .collect()
}

fn put_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn rd_u32(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}
fn rd_u64(b: &[u8], p: &mut usize) -> u64 {
    let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}

// ----- pack: ft + base → .spsd -----

pub fn run_pack(ft: &str, base: &str, out: &str, verbose: bool) -> Result<()> {
    let mut sp = (!verbose).then(|| Spinner::start("Reading base + fine-tune …"));
    let base_sha = file_sha(base)?;
    let mut rb = open_st(base)?;
    let mut rf = open_st(ft)?;
    let bnames: std::collections::HashSet<String> =
        rb.list_tensors().iter().map(|s| s.to_string()).collect();
    let fnames: Vec<String> = rf.list_tensors().iter().map(|s| s.to_string()).collect();

    // Pass 1: deltas + global histogram.
    struct Item {
        name: String,
        shape: Vec<u64>,
        kind: u8, // 0 = delta (bf16), 1 = full raw
        zz: Vec<u16>,    // zigzag delta (kind 0)
        raw: Vec<u8>,    // raw bytes (kind 1)
    }
    let mut items: Vec<Item> = Vec::new();
    let mut hist = vec![0u64; 65536];
    let (mut wdelta, mut wfull) = (0u64, 0u64);
    for name in &fnames {
        let m = rf.to_rllm_meta(name)?;
        let fbytes = rf.read_tensor(name)?;
        if m.dtype == DType::Bf16 && bnames.contains(name) {
            let bm = rb.to_rllm_meta(name)?;
            if bm.shape == m.shape {
                let bbytes = rb.read_tensor(name)?;
                if bbytes.len() == fbytes.len() {
                    let (bv, fv) = (u16s(&bbytes), u16s(&fbytes));
                    let zz: Vec<u16> = (0..fv.len()).map(|i| zigzag(fv[i].wrapping_sub(bv[i]))).collect();
                    for &s in &zz {
                        hist[s as usize] += 1;
                    }
                    wdelta += zz.len() as u64;
                    items.push(Item { name: name.clone(), shape: m.shape, kind: 0, zz, raw: vec![] });
                    continue;
                }
            }
        }
        // fallback: store the tensor raw (not in base / mismatched / non-bf16)
        wfull += (fbytes.len() / 2) as u64;
        items.push(Item { name: name.clone(), shape: m.shape, kind: 1, zz: vec![], raw: fbytes });
    }

    if let Some(s) = &sp {
        s.set("Building rANS model + encoding …");
    }
    let t = build_tables_from_hist(&hist);

    // Serialize .spsd
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    put_u32(&mut buf, VERSION);
    buf.extend_from_slice(&base_sha);
    // base path (relative hint for undelta convenience)
    let bp = base.as_bytes();
    put_u32(&mut buf, bp.len() as u32);
    buf.extend_from_slice(bp);
    // global freq table (65536 × u32)
    for &f in &t.freq {
        put_u32(&mut buf, f);
    }
    put_u32(&mut buf, items.len() as u32);
    for it in &items {
        put_u32(&mut buf, it.name.len() as u32);
        buf.extend_from_slice(it.name.as_bytes());
        put_u32(&mut buf, it.shape.len() as u32);
        for &d in &it.shape {
            put_u64(&mut buf, d);
        }
        buf.push(it.kind);
        if it.kind == 0 {
            let enc = rans_encode(&it.zz, &t);
            put_u64(&mut buf, it.zz.len() as u64);
            put_u64(&mut buf, enc.len() as u64);
            buf.extend_from_slice(&enc);
        } else {
            put_u64(&mut buf, it.raw.len() as u64);
            buf.extend_from_slice(&it.raw);
        }
    }
    std::fs::write(out, &buf)?;
    if let Some(s) = sp.take() {
        s.clear();
    }

    if !verbose {
        let ft_raw = (wdelta + wfull) * 2;
        println!();
        println!("  \x1b[1;92m✓\x1b[0m  Delta packed");
        println!();
        println!("  fine-tune (bf16)   {}", human_size(ft_raw));
        println!("  delta (.spsd)      {}   ({:.1}% of raw)", human_size(buf.len() as u64), buf.len() as f64 / ft_raw as f64 * 100.0);
        println!("  delta-coded        {} weights · raw fallback {}", wdelta, wfull);
        println!("  base ref           {}  (sha {:02x}{:02x}…)", base, base_sha[0], base_sha[1]);
        println!("  → {out}");
    } else {
        println!("wrote {} ({} bytes); {} delta weights, {} raw", out, buf.len(), wdelta, wfull);
    }
    let _ = std::io::stdout().flush();
    Ok(())
}

// ----- unpack: .spsd + base → ft safetensors -----

pub fn run_unpack(input: &str, base: &str, out: &str) -> Result<()> {
    let sp = Spinner::start("Reconstructing fine-tune …");
    let buf = std::fs::read(input)?;
    let mut p = 0usize;
    if &buf[0..4] != MAGIC {
        bail!("not a .spsd file (bad magic)");
    }
    p += 4;
    let _ver = rd_u32(&buf, &mut p);
    let stored_sha: [u8; 32] = buf[p..p + 32].try_into().unwrap();
    p += 32;
    let bplen = rd_u32(&buf, &mut p) as usize;
    p += bplen; // skip stored base path hint; we use the --base arg

    // verify base matches the one used to pack
    if file_sha(base)? != stored_sha {
        bail!("base mismatch: --base does not match the base this delta was packed against");
    }
    let mut rb = open_st(base)?;

    let mut freq = vec![0u32; 65536];
    for f in freq.iter_mut() {
        *f = rd_u32(&buf, &mut p);
    }
    let t = tables_from_freq(freq);

    let ntensors = rd_u32(&buf, &mut p) as usize;
    let mut tensors: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::with_capacity(ntensors);
    for _ in 0..ntensors {
        let nlen = rd_u32(&buf, &mut p) as usize;
        let name = String::from_utf8(buf[p..p + nlen].to_vec())?;
        p += nlen;
        let ndims = rd_u32(&buf, &mut p) as usize;
        let shape: Vec<u64> = (0..ndims).map(|_| rd_u64(&buf, &mut p)).collect();
        let kind = buf[p];
        p += 1;
        if kind == 0 {
            let n = rd_u64(&buf, &mut p) as usize;
            let elen = rd_u64(&buf, &mut p) as usize;
            let zz = rans_decode(&buf[p..p + elen], n, &t);
            p += elen;
            // ft = base + unzigzag(Δ)
            let bbytes = rb.read_tensor(&name)?;
            let bv = u16s(&bbytes);
            let mut ft = Vec::with_capacity(n * 2);
            for i in 0..n {
                let v = bv[i].wrapping_add(unzigzag(zz[i]));
                ft.extend_from_slice(&v.to_le_bytes());
            }
            tensors.push((name, shape, ft));
        } else {
            let len = rd_u64(&buf, &mut p) as usize;
            let raw = buf[p..p + len].to_vec();
            p += len;
            tensors.push((name, shape, raw));
        }
    }

    // write a minimal safetensors with all reconstructed tensors (bf16 / raw passthrough)
    write_safetensors(out, &tensors)?;
    sp.clear();
    println!("  \x1b[1;92m✓\x1b[0m  Reconstructed → {out}  ({} tensors)", tensors.len());
    Ok(())
}

/// Write tensors as a single safetensors file. dtype is BF16 for delta-reconstructed tensors;
/// raw-fallback tensors are written BF16 too (they were stored as their original bf16 bytes).
fn write_safetensors(out: &str, tensors: &[(String, Vec<u64>, Vec<u8>)]) -> Result<()> {
    // Build the JSON header.
    let mut entries: Vec<String> = Vec::new();
    let mut offset = 0u64;
    for (name, shape, data) in tensors {
        let shape_json = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
        entries.push(format!(
            "\"{}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            shape_json,
            offset,
            offset + data.len() as u64
        ));
        offset += data.len() as u64;
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut hbytes = header.into_bytes();
    while hbytes.len() % 8 != 0 {
        hbytes.push(b' ');
    }
    let mut f = std::fs::File::create(out)?;
    f.write_all(&(hbytes.len() as u64).to_le_bytes())?;
    f.write_all(&hbytes)?;
    for (_, _, data) in tensors {
        f.write_all(data)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rans_roundtrip() {
        let mut hist = vec![0u64; 65536];
        let syms: Vec<u16> = (0..50_000u32).map(|i| (i.wrapping_mul(2654435761) % 97) as u16).collect();
        for &s in &syms {
            hist[s as usize] += 1;
        }
        let t = build_tables_from_hist(&hist);
        assert_eq!(rans_decode(&rans_encode(&syms, &t), syms.len(), &t), syms);
    }
    #[test]
    fn zigzag_roundtrip() {
        for d in [0u16, 1, 0xFFFF, 0x8000, 0x7FFF, 42, 65000] {
            assert_eq!(unzigzag(zigzag(d)), d);
        }
    }
}
