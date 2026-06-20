// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE.
//
// Tests for REESTREAM-RANS (R153): synthetic lossless parity, real-Gemma lossless
// gate, and the >RAM cold capacity-bound bench. Split from rans_stream.rs to keep the
// production module within the modular-code-guard budget; `super::*` reaches the
// streaming module (build_rans_sidecar, stream_lmhead_from_rans_sidecar, …).

use super::*;
use rtc_codec::split_bf16;

// Synthetic bf16 embedding cycling `distinct` exponents (+ pseudo-random sign/mantissa).
fn make_embedding_distinct(vocab: usize, hidden: usize, distinct: usize) -> Vec<u8> {
    let mut state = 0x51A7_F00D_1234_5678u64;
    let mut out = Vec::with_capacity(vocab * hidden * 2);
    for k in 0..vocab * hidden {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let exp = (96 + (k % distinct)) as u16 & 0xFF;
        let bits = (((state >> 31) & 1) as u16) << 15 | (exp << 7) | (state & 0x7F) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

// Reference: resident bf16 lm-head GEMV over the exact same weights.
fn reference_logits(bf16: &[u8], act: &[f32], vocab: usize, hidden: usize) -> Vec<f32> {
    lm_head_logits_parallel_bf16(act, bf16, vocab, hidden)
}

#[test]
fn rans_sidecar_streams_equal_to_reference() {
    // Build a synthetic embedding, frame it as an RLMR sidecar via the shared builder,
    // and confirm the streamed logits match the resident bf16 GEMV bit-for-bit.
    let (vocab, hidden, b) = (512usize, 1152usize, 64usize);
    let bf16 = make_embedding_distinct(vocab, hidden, 34); // w=6-class exponent spread
    let n = vocab * hidden;
    let (mut exp, mut res) = (vec![0u8; n], vec![0u8; n]);
    for i in 0..n {
        let (e, r) = split_bf16(u16::from_le_bytes([bf16[2 * i], bf16[2 * i + 1]]));
        exp[i] = e;
        res[i] = r;
    }
    let freq = rtc_codec::normalize_freqs(&rtc_codec::count_symbols(&exp));
    let sidecar = build_rans_sidecar(&exp, &res, &freq, vocab, hidden, b);
    let path = "/tmp/r153_rans_unit.sidecar";
    std::fs::write(path, &sidecar).unwrap();

    let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.017).sin() * 0.3).collect();
    let reference = reference_logits(&bf16, &act, vocab, hidden);
    let streamed = stream_lmhead_from_rans_sidecar(path, &act).unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(streamed, reference, "rANS sidecar stream must equal resident bf16 GEMV bit-for-bit");
}

#[test]
#[ignore]
fn r153_gemma_rans_lmhead_lossless() {
    let model = "../../models/gemma-3-1b-it-rawcodec.rllm";
    let tname = "model.embed_tokens.weight";
    let sidecar = "/tmp/gemma1b-lmhead-rans.sidecar";
    write_lmhead_sidecar_rans(model, tname, 256, sidecar).unwrap();
    let mut m = crate::LazyRllmModel::open(model).unwrap();
    let meta = m.tensor(tname).unwrap().clone();
    let vocab = meta.shape[0] as usize;
    let hidden = meta.shape[1] as usize;
    let bf16 = m.with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec())).unwrap().unwrap();
    let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin() * 0.4).collect();
    let resident = lm_head_logits_parallel_bf16(&act, &bf16, vocab, hidden);
    let streamed = stream_lmhead_from_rans_sidecar(sidecar, &act).unwrap();
    let _ = std::fs::remove_file(sidecar);
    assert_eq!(streamed, resident, "R153: rANS streaming lm-head must equal resident bf16 bit-for-bit on real Gemma");
    eprintln!("R153 OK: Gemma rANS streaming lm-head == resident, {vocab} logits bit-identical");
}

// R156a: the rANS streaming GEMV must generalize beyond the lm-head to a transformer
// BODY projection (different shape) and stay lossless — the foundation for whole-model
// rANS. gate_proj [6912×1152] (6912 % 256 == 0). Streamed W·x == resident bf16 W·x.
#[test]
#[ignore]
fn r156a_gemma_body_projection_lossless() {
    let model = "../../models/gemma-3-1b-it-rawcodec.rllm";
    let tname = "model.layers.0.mlp.gate_proj.weight";
    let sidecar = "/tmp/r156a_gate_proj.sidecar";
    write_lmhead_sidecar_rans(model, tname, 256, sidecar).unwrap();
    let mut m = crate::LazyRllmModel::open(model).unwrap();
    let meta = m.tensor(tname).unwrap().clone();
    let out_features = meta.shape[0] as usize;
    let in_features = meta.shape[1] as usize;
    let w = m.with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec())).unwrap().unwrap();
    let act: Vec<f32> = (0..in_features).map(|i| ((i as f32) * 0.013).cos() * 0.5).collect();
    let resident = lm_head_logits_parallel_bf16(&act, &w, out_features, in_features);
    let streamed = stream_lmhead_from_rans_sidecar(sidecar, &act).unwrap();
    let _ = std::fs::remove_file(sidecar);
    assert_eq!(streamed, resident, "R156a: rANS-streamed body projection W·x must equal resident bf16 W·x");
    eprintln!("R156a OK: gate_proj [{out_features}×{in_features}] rANS stream == resident, {out_features} outputs bit-identical");
}

// Append `k` copies of `src` to `path` (streamed write); for building >RAM files.
fn replicate(src: &[u8], k: usize, path: &str) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    for _ in 0..k {
        f.write_all(src).unwrap();
    }
    f.sync_all().unwrap();
}

// Cold (F_NOCACHE) read of `path` partitioned across `n_threads` workers, zero compute
// — the FAIR raw baseline (same concurrency as the rANS streamer).
fn cold_parallel_read_ms(path: &str, total: usize, chunk: usize, n_threads: usize) -> f64 {
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::io::AsRawFd;
    extern "C" { fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32; }
    const F_NOCACHE: i32 = 48;
    let nblk = total / chunk;
    let per = nblk.div_ceil(n_threads.max(1));
    let t = std::time::Instant::now();
    std::thread::scope(|s| {
        let mut start = 0usize;
        while start < nblk {
            let cnt = per.min(nblk - start);
            s.spawn(move || {
                let mut f = std::fs::File::open(path).unwrap();
                unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
                if f.seek(SeekFrom::Start((start * chunk) as u64)).is_err() { return; }
                let mut buf = vec![0u8; chunk];
                for _ in 0..cnt {
                    if f.read_exact(&mut buf).is_err() { break; }
                    std::hint::black_box(&buf);
                }
            });
            start += cnt;
        }
    });
    t.elapsed().as_secs_f64() * 1000.0
}

// R153: end-to-end capacity-bound win. rANS sidecar (replicated >RAM, cold) streamed
// in parallel vs a FAIR cold parallel-raw read. Expect ~ raw/rANS byte ratio (~1.5x,
// up from bit-plane's 1.15x) since rANS reads ~34% fewer bytes than raw bf16.
#[test]
#[ignore]
fn r153_rans_capacity_bound() {
    use std::io::{Read, Seek, SeekFrom};
    let model = "../../models/gemma-3-1b-it-rawcodec.rllm";
    let tname = "model.embed_tokens.weight";
    let one = "/tmp/r153_rans_one.sidecar";
    let rans_big = "/tmp/r153_rans_big.bin";
    let raw_big = "/tmp/r153_raw_big.bin";
    write_lmhead_sidecar_rans(model, tname, 256, one).unwrap();

    let (hidden, vocab, block_rows, freq, block_lens, header_len) = read_rans_header(one).unwrap();
    let num_blocks = block_lens.len();
    // Read the sidecar body (blocks only).
    let mut bf = std::fs::File::open(one).unwrap();
    bf.seek(SeekFrom::Start(header_len)).unwrap();
    let mut body = Vec::new();
    bf.read_to_end(&mut body).unwrap();
    let _ = std::fs::remove_file(one);

    // Raw bf16 plane for the baseline.
    let mut m = crate::LazyRllmModel::open(model).unwrap();
    let meta = m.tensor(tname).unwrap().clone();
    let raw = m.with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec())).unwrap().unwrap();

    // Replicate both > 8 GB so reads are genuinely cold (k copies = k vocab-sweeps).
    let k = 20usize; // raw ~12 GB, rANS body ~8 GB
    replicate(&raw, k, raw_big);
    replicate(&body, k, rans_big);
    let raw_total = raw.len() * k;
    let rans_total = body.len() * k;
    drop(raw);
    drop(body);

    // Extended block tables over the k-replicated body (file is body-only, offset 0).
    let mut ext_lens = Vec::with_capacity(num_blocks * k);
    let mut offsets = Vec::with_capacity(num_blocks * k);
    let mut acc = 0u64;
    for _ in 0..k {
        for &l in &block_lens {
            offsets.push(acc);
            ext_lens.push(l);
            acc += l as u64;
        }
    }

    let cores = std::thread::available_parallelism().map(usize::from).unwrap_or(6);
    let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin() * 0.4).collect();
    let raw_ms = cold_parallel_read_ms(raw_big, raw_total, block_rows * hidden * 2, cores);
    let rans_ms = {
        let mut out = vec![0f32; vocab * k];
        let t = std::time::Instant::now();
        streaming_rans_gemv_parallel(
            rans_big, &freq, hidden, block_rows, num_blocks * k, &offsets, &ext_lens, &act, &mut out, true, cores,
        );
        std::hint::black_box(&out);
        t.elapsed().as_secs_f64() * 1000.0
    };
    let _ = std::fs::remove_file(raw_big);
    let _ = std::fs::remove_file(rans_big);

    let (raw_gb, rans_gb) = (raw_total as f64 / 1e9, rans_total as f64 / 1e9);
    let speedup = raw_ms / rans_ms;
    let verdict = if rans_ms < raw_ms { "GO -- rANS stream beats raw bf16, cold, >RAM" }
        else if rans_ms < raw_ms * 1.05 { "MARGINAL" } else { "NO-GO" };
    eprintln!(
        "\n=== R153 REESTREAM-RANS capacity-bound (Gemma lm-head, >RAM cold, {cores} cores, k={k}) ===\n\
         raw bf16  parallel  {raw_gb:.2} GB -> {raw_ms:.0} ms  ({:.2} GB/s)\n\
         rANS      parallel  {rans_gb:.2} GB -> {rans_ms:.0} ms  ({:.2} GB/s, decode hidden)\n\
         bytes: {:.0}% fewer   SPEEDUP vs raw: {speedup:.2}x   (bit-plane R150a was 1.15x)\n\
         VERDICT: {verdict}\n",
        raw_gb / (raw_ms / 1e3),
        rans_gb / (rans_ms / 1e3),
        (1.0 - rans_total as f64 / raw_total as f64) * 100.0,
    );
}

// R154 lever #1: oversubscribe threads so a worker blocked on a cold (F_NOCACHE) read
// lets others decode — the cheap form of read/decode overlap. Sweep thread counts for
// both raw and rANS; if rANS's best approaches the 1.51x byte ratio, oversubscription
// alone closes R153's gap (1.05x).
#[test]
#[ignore]
fn r154_rans_thread_sweep() {
    use std::io::{Read, Seek, SeekFrom};
    let model = "../../models/gemma-3-1b-it-rawcodec.rllm";
    let tname = "model.embed_tokens.weight";
    let one = "/tmp/r154_rans_one.sidecar";
    let rans_big = "/tmp/r154_rans_big.bin";
    let raw_big = "/tmp/r154_raw_big.bin";
    write_lmhead_sidecar_rans(model, tname, 256, one).unwrap();
    let (hidden, vocab, block_rows, freq, block_lens, header_len) = read_rans_header(one).unwrap();
    let num_blocks = block_lens.len();
    let mut bf = std::fs::File::open(one).unwrap();
    bf.seek(SeekFrom::Start(header_len)).unwrap();
    let mut body = Vec::new();
    bf.read_to_end(&mut body).unwrap();
    let _ = std::fs::remove_file(one);
    let mut m = crate::LazyRllmModel::open(model).unwrap();
    let meta = m.tensor(tname).unwrap().clone();
    let raw = m.with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec())).unwrap().unwrap();

    let k = 20usize;
    replicate(&raw, k, raw_big);
    replicate(&body, k, rans_big);
    let raw_total = raw.len() * k;
    drop(raw);
    drop(body);
    let mut ext_lens = Vec::with_capacity(num_blocks * k);
    let mut offsets = Vec::with_capacity(num_blocks * k);
    let mut acc = 0u64;
    for _ in 0..k {
        for &l in &block_lens {
            offsets.push(acc);
            ext_lens.push(l);
            acc += l as u64;
        }
    }
    let cores = std::thread::available_parallelism().map(usize::from).unwrap_or(6);
    let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin() * 0.4).collect();
    let mut out = vec![0f32; vocab * k];

    eprintln!("\n=== R154 thread sweep (Gemma lm-head, >RAM cold, {cores} cores, k={k}) ===");
    eprintln!("  nt | raw bf16 (12GB) | rANS (8GB)      | rANS vs raw");
    for &nt in &[cores, cores * 2, cores * 3] {
        let raw_ms = cold_parallel_read_ms(raw_big, raw_total, block_rows * hidden * 2, nt);
        let t = std::time::Instant::now();
        streaming_rans_gemv_parallel(
            rans_big, &freq, hidden, block_rows, num_blocks * k, &offsets, &ext_lens, &act, &mut out, true, nt,
        );
        std::hint::black_box(&out);
        let rans_ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  {nt:2} | {raw_ms:6.0} ms       | {rans_ms:6.0} ms        | {:.2}x", raw_ms / rans_ms);
    }
    let _ = std::fs::remove_file(raw_big);
    let _ = std::fs::remove_file(rans_big);
    eprintln!();
}
