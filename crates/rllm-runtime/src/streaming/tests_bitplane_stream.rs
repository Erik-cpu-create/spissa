// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

// Unit + capacity-bound tests for the streaming bit-plane lm-head, split from
// bitplane_stream.rs to keep the production module under the modular-code-guard
// line budget (test code is exempt). Declared via #[path] as mod
// bitplane_stream_tests; `super::*` resolves to the streaming module.
    use super::*;
    use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};

    // bf16 embedding cycling `distinct` exponents (=> w = ceil(log2(distinct))),
    // pseudo-random sign+mantissa. distinct=32 => w=5 (Llama), 34 => w=6 (Gemma).
    fn make_embedding_distinct(vocab: usize, hidden: usize, distinct: usize) -> Vec<u8> {
        let mut state = 0x0BAD_F00D_1234_5678u64;
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

    fn make_embedding(vocab: usize, hidden: usize) -> Vec<u8> {
        make_embedding_distinct(vocab, hidden, 32)
    }

    // Frame the flat planes into [B×index ++ B×residual] contiguous blocks.
    fn frame_blocks(idx_plane: &[u8], residuals: &[u8], hidden: usize, vocab: usize, b: usize, w: usize) -> Vec<u8> {
        let row_idx = hidden * w / 8;
        let mut framed = Vec::new();
        for blk in 0..(vocab / b) {
            for r in 0..b {
                let row = blk * b + r;
                framed.extend_from_slice(&idx_plane[row * row_idx..(row + 1) * row_idx]);
            }
            for r in 0..b {
                let row = blk * b + r;
                framed.extend_from_slice(&residuals[row * hidden..(row + 1) * hidden]);
            }
        }
        framed
    }

    #[test]
    fn streaming_gemv_matches_reference_bit_for_bit() {
        let (vocab, hidden, b) = (128usize, 2048usize, 64usize);
        let bf16 = make_embedding(vocab, hidden);
        let enc = BitplaneCodec
            .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![(vocab * hidden) as u64], dtype: "bf16".into() })
            .unwrap();
        let p = enc.data[14] as usize;
        assert_eq!(enc.data[15], 5);
        let mut off = 16;
        let palette = enc.data[off..off + p].to_vec();
        off += p;
        let row_idx = hidden * 5 / 8;
        let idx_bytes = vocab * row_idx;
        let idx_plane = &enc.data[off..off + idx_bytes];
        off += idx_bytes;
        let residuals = &enc.data[off..off + vocab * hidden];
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.021).sin() * 0.3).collect();

        // single-thread reference: decode each row + dot
        let mut reference = vec![0f32; vocab];
        for (r, slot) in reference.iter_mut().enumerate() {
            let dec = rtc_codec::decode_neon_w5(&palette, &idx_plane[r * row_idx..], &residuals[r * hidden..], hidden);
            *slot = bf16_row_dot_f32(&act, &dec, hidden);
        }

        // block-framed file + streaming kernel
        let framed = frame_blocks(idx_plane, residuals, hidden, vocab, b, 5);
        let path = "/tmp/r148_unit.bin";
        std::fs::write(path, &framed).unwrap();
        let mut out = vec![0f32; vocab];
        streaming_bitplane_gemv(path, &palette, hidden, 5, b, vocab / b, &act, &mut out, false, 0);
        let _ = std::fs::remove_file(path);

        assert_eq!(out, reference, "streaming GEMV must equal single-thread decode+dot bit-for-bit");
    }

    #[test]
    fn lmhead_sidecar_streams_equal_to_reference() {
        // Build a tiny synthetic "model" embedding directly as a sidecar and confirm
        // stream_lmhead_from_sidecar == single-thread decode+dot reference.
        let (vocab, hidden, b) = (128usize, 1152usize, 64usize);
        let bf16 = make_embedding(vocab, hidden);
        let enc = BitplaneCodec
            .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![(vocab * hidden) as u64], dtype: "bf16".into() })
            .unwrap();
        assert_eq!(enc.data[15], 5);
        let p = enc.data[14] as usize;
        let palette = enc.data[16..16 + p].to_vec();
        let row_idx = hidden * 5 / 8;
        let idx_plane = enc.data[16 + p..16 + p + vocab * row_idx].to_vec();
        let residuals = enc.data[16 + p + vocab * row_idx..16 + p + vocab * row_idx + vocab * hidden].to_vec();

        // write a sidecar by hand (same format write_lmhead_sidecar produces)
        let mut sc = Vec::new();
        sc.extend_from_slice(b"RLMH");
        sc.push(1);
        sc.extend_from_slice(&(hidden as u32).to_le_bytes());
        sc.extend_from_slice(&(vocab as u32).to_le_bytes());
        sc.extend_from_slice(&(b as u32).to_le_bytes());
        sc.push(p as u8);
        sc.extend_from_slice(&palette);
        sc.extend_from_slice(&frame_blocks(&idx_plane, &residuals, hidden, vocab, b, 5));
        let path = "/tmp/r149a_unit.sidecar";
        std::fs::write(path, &sc).unwrap();

        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.019).sin() * 0.3).collect();
        let mut reference = vec![0f32; vocab];
        for (r, slot) in reference.iter_mut().enumerate() {
            let dec = rtc_codec::decode_neon_w5(&palette, &idx_plane[r * row_idx..], &residuals[r * hidden..], hidden);
            *slot = bf16_row_dot_f32(&act, &dec, hidden);
        }
        let streamed = stream_lmhead_from_sidecar(path, &act).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(streamed, reference, "sidecar stream must equal decode+dot reference bit-for-bit");
    }

    // R149b: w=6 (34 exponents, Gemma's case) round-trips through a v2 sidecar and
    // streams bit-identical to a single-thread scalar decode+dot reference. hidden=1152
    // is Gemma's; 1152*6/8 = 864 is byte-aligned. Unit-level gate, no real model needed.
    #[test]
    fn w6_sidecar_streams_equal_to_reference() {
        let (vocab, hidden, b) = (256usize, 1152usize, 64usize);
        let bf16 = make_embedding_distinct(vocab, hidden, 34); // 34 exponents => w=6
        let enc = BitplaneCodec
            .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![(vocab * hidden) as u64], dtype: "bf16".into() })
            .unwrap();
        assert_eq!(enc.data[15], 6, "expected w=6 for 34 exponents");
        let p = enc.data[14] as usize;
        let palette = enc.data[16..16 + p].to_vec();
        let row_idx = hidden * 6 / 8;
        let idx_plane = enc.data[16 + p..16 + p + vocab * row_idx].to_vec();
        let residuals = enc.data[16 + p + vocab * row_idx..16 + p + vocab * row_idx + vocab * hidden].to_vec();

        // hand-write a v2 sidecar (the format write_lmhead_sidecar produces)
        let mut sc = Vec::new();
        sc.extend_from_slice(b"RLMH");
        sc.push(2);
        sc.extend_from_slice(&(hidden as u32).to_le_bytes());
        sc.extend_from_slice(&(vocab as u32).to_le_bytes());
        sc.extend_from_slice(&(b as u32).to_le_bytes());
        sc.push(p as u8);
        sc.push(6u8); // w
        sc.extend_from_slice(&palette);
        sc.extend_from_slice(&frame_blocks(&idx_plane, &residuals, hidden, vocab, b, 6));
        let path = "/tmp/r149b_w6_unit.sidecar";
        std::fs::write(path, &sc).unwrap();

        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.017).cos() * 0.3).collect();
        let mut reference = vec![0f32; vocab];
        let mut dec = vec![0u8; hidden * 2];
        for (r, slot) in reference.iter_mut().enumerate() {
            rtc_codec::decode_bitplane_row_into(
                &palette, &idx_plane[r * row_idx..], &residuals[r * hidden..], hidden, 6, &mut dec,
            );
            *slot = bf16_row_dot_f32(&act, &dec, hidden);
        }
        let streamed = stream_lmhead_from_sidecar(path, &act).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(streamed, reference, "w=6 sidecar stream must equal decode+dot reference bit-for-bit");
    }

    // R150a: REESTREAM-PAR. Parallel decode (n_threads ∈ {1,2,4}) must equal the
    // single-thread decode+dot reference bit-for-bit, for both w=5 and w=6 — proving
    // the default streaming path stays lossless regardless of thread count.
    #[test]
    fn streaming_parallel_matches_single_thread() {
        for &(distinct, w) in &[(32usize, 5usize), (34usize, 6usize)] {
            let (vocab, hidden, b) = (256usize, 1152usize, 64usize);
            let bf16 = make_embedding_distinct(vocab, hidden, distinct);
            let enc = BitplaneCodec
                .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![(vocab * hidden) as u64], dtype: "bf16".into() })
                .unwrap();
            assert_eq!(enc.data[15] as usize, w);
            let p = enc.data[14] as usize;
            let palette = enc.data[16..16 + p].to_vec();
            let row_idx = hidden * w / 8;
            let idx_plane = enc.data[16 + p..16 + p + vocab * row_idx].to_vec();
            let residuals = enc.data[16 + p + vocab * row_idx..16 + p + vocab * row_idx + vocab * hidden].to_vec();
            let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();

            // single-thread reference: decode each row + dot
            let mut reference = vec![0f32; vocab];
            let mut dec = vec![0u8; hidden * 2];
            for (r, slot) in reference.iter_mut().enumerate() {
                rtc_codec::decode_bitplane_row_into(
                    &palette, &idx_plane[r * row_idx..], &residuals[r * hidden..], hidden, w as u8, &mut dec,
                );
                *slot = bf16_row_dot_f32(&act, &dec, hidden);
            }

            let framed = frame_blocks(&idx_plane, &residuals, hidden, vocab, b, w);
            let path = format!("/tmp/r150a_par_w{w}.bin");
            std::fs::write(&path, &framed).unwrap();
            for &nt in &[1usize, 2, 4] {
                let mut out = vec![0f32; vocab];
                streaming_bitplane_gemv_parallel(&path, &palette, hidden, w, b, vocab / b, &act, &mut out, false, 0, nt);
                assert_eq!(out, reference, "w={w} n_threads={nt}: parallel must equal single-thread bit-for-bit");
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    #[ignore]
    fn streaming_gemv_capacity_bound_bench() {
        use std::io::{Read, Write};
        use std::os::unix::io::AsRawFd;
        extern "C" {
            fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32;
        }
        const F_NOCACHE: i32 = 48;

        let bytes = std::fs::read("/tmp/rllm-bf16-sample.bin")
            .expect("run dump_bf16_embedding_sample first");
        let hidden = 2048usize;
        let n = bytes.len() / 2;
        let vocab = n / hidden;
        let b = 256usize; // block rows
        assert_eq!(vocab % b, 0, "vocab must be a multiple of block size");
        let enc = BitplaneCodec
            .encode(&bytes, &EncodeMeta { name: "e".into(), shape: vec![n as u64], dtype: "bf16".into() })
            .unwrap();
        let p = enc.data[14] as usize;
        let mut off = 16;
        let palette = enc.data[off..off + p].to_vec();
        off += p;
        let row_idx = hidden * 5 / 8;
        let idx_bytes = vocab * row_idx;
        let idx_plane = &enc.data[off..off + idx_bytes];
        off += idx_bytes;
        let residuals = &enc.data[off..off + n];
        let framed = frame_blocks(idx_plane, residuals, hidden, vocab, b, 5);

        // Replicate both files > RAM (~3 GB free) so reads are genuinely cold.
        let k = 12usize;
        let raw_path = "/tmp/r148_raw.bin";
        let comp_path = "/tmp/r148_comp.bin";
        {
            let mut fr = std::fs::File::create(raw_path).unwrap();
            let mut fc = std::fs::File::create(comp_path).unwrap();
            for _ in 0..k {
                fr.write_all(&bytes).unwrap();
                fc.write_all(&framed).unwrap();
            }
        }
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();

        // raw bf16 stream cold: pure read of the bytes raw MUST move (the strongest,
        // fairest baseline -- in a real pipelined raw path the dot is hidden under the
        // read too, so we give raw the benefit of zero compute).
        let raw_ms = {
            let mut f = std::fs::File::open(raw_path).unwrap();
            unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
            let mut buf = vec![0u8; bytes.len()];
            let t = std::time::Instant::now();
            for _ in 0..k {
                f.read_exact(&mut buf).unwrap();
                std::hint::black_box(&buf);
            }
            t.elapsed().as_secs_f64() * 1000.0
        };

        // R148 pipelined stream cold: blocks across the whole replicated file.
        let comp_ms = {
            let total_blocks = (vocab / b) * k;
            let mut out = vec![0f32; total_blocks * b];
            let t = std::time::Instant::now();
            streaming_bitplane_gemv(comp_path, &palette, hidden, 5, b, total_blocks, &act, &mut out, true, 0);
            std::hint::black_box(&out);
            t.elapsed().as_secs_f64() * 1000.0
        };

        let _ = std::fs::remove_file(raw_path);
        let _ = std::fs::remove_file(comp_path);
        let raw_gb = (bytes.len() * k) as f64 / 1e9;
        let comp_gb = (framed.len() * k) as f64 / 1e9;
        eprintln!(
            "\n=== R148 REESTREAM pipelined capacity-bound BENCH (cold SSD, > RAM) ===\n\
             raw bf16   stream {raw_gb:.1} GB -> {raw_ms:.0} ms  ({:.2} GB/s)\n\
             pipelined  stream {comp_gb:.1} GB -> {comp_ms:.0} ms  ({:.2} GB/s, decode overlapped)\n\
             SPEEDUP vs raw: {:.2}x   (R147 un-pipelined scout was 1.13x)\n\
             VERDICT: {}\n",
            raw_gb / (raw_ms / 1e3),
            comp_gb / (comp_ms / 1e3),
            raw_ms / comp_ms,
            if comp_ms < raw_ms { "GO -- pipelined streaming bit-plane beats raw bf16 from SSD" } else { "NO-GO" }
        );
    }

    #[test]
    #[ignore]
    fn write_gemma_lmhead_sidecar() {
        write_lmhead_sidecar(
            "../../models/gemma-3-1b-it-rawcodec.rllm",
            "model.embed_tokens.weight",
            256,
            "/tmp/gemma1b-lmhead.sidecar",
        )
        .unwrap();
        eprintln!("wrote /tmp/gemma1b-lmhead.sidecar");
    }

    #[test]
    #[ignore]
    fn r149a_llama_streaming_lmhead_lossless() {
        let model = "../../models/Llama-3.2-1B-Instruct-raw.rllm";
        let tname = "model.embed_tokens.weight";
        let sidecar = "/tmp/llama1b-lmhead.sidecar";
        write_lmhead_sidecar(model, tname, 256, sidecar).unwrap();
        let mut m = crate::LazyRllmModel::open(model).unwrap();
        let meta = m.tensor(tname).unwrap().clone();
        let vocab = meta.shape[0] as usize;
        let hidden = meta.shape[1] as usize;
        let bf16 = m.with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec())).unwrap().unwrap();
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin() * 0.4).collect();
        let resident = lm_head_logits_parallel_bf16(&act, &bf16, vocab, hidden);
        let streamed = stream_lmhead_from_sidecar(sidecar, &act).unwrap();
        let _ = std::fs::remove_file(sidecar);
        assert_eq!(streamed, resident, "streaming lm-head must equal resident bf16 lm-head bit-for-bit on real Llama weights");
        eprintln!("R149a OK: Llama streaming lm-head == resident, {} logits bit-identical", vocab);
    }

    // R149b: the gate R149a-Gemma could not reach. Real Gemma 3 1B embedding is w=6
    // (34 exponents); the REEPLANE-W6 streaming lm-head must equal the resident bf16
    // lm-head GEMV bit-for-bit on all 262144 logits, for an identical activation.
    #[test]
    #[ignore]
    fn r149b_gemma_streaming_lmhead_lossless() {
        let model = "../../models/gemma-3-1b-it-rawcodec.rllm";
        let tname = "model.embed_tokens.weight";
        let sidecar = "/tmp/gemma1b-lmhead.sidecar";
        write_lmhead_sidecar(model, tname, 256, sidecar).unwrap();
        let mut m = crate::LazyRllmModel::open(model).unwrap();
        let meta = m.tensor(tname).unwrap().clone();
        let vocab = meta.shape[0] as usize;
        let hidden = meta.shape[1] as usize;
        let bf16 = m.with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec())).unwrap().unwrap();
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin() * 0.4).collect();
        let resident = lm_head_logits_parallel_bf16(&act, &bf16, vocab, hidden);
        let streamed = stream_lmhead_from_sidecar(sidecar, &act).unwrap();
        let _ = std::fs::remove_file(sidecar);
        assert_eq!(streamed, resident, "w=6 streaming lm-head must equal resident bf16 lm-head bit-for-bit on real Gemma weights");
        eprintln!("R149b OK: Gemma (w=6) streaming lm-head == resident, {vocab} logits bit-identical");
    }

    // Cold (F_NOCACHE) single blocking read of the whole file, zero compute — the
    // naive raw baseline. Returns mean ms/pass over `iters`.
    fn cold_blocking_read_ms(path: &str, total_bytes: usize, iters: usize) -> f64 {
        use std::io::{Read, Seek, SeekFrom};
        use std::os::unix::io::AsRawFd;
        extern "C" { fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32; }
        const F_NOCACHE: i32 = 48;
        let mut f = std::fs::File::open(path).unwrap();
        unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
        let mut buf = vec![0u8; total_bytes];
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f.seek(SeekFrom::Start(0)).unwrap();
            f.read_exact(&mut buf).unwrap();
            std::hint::black_box(&buf);
        }
        t.elapsed().as_secs_f64() * 1000.0 / iters as f64
    }

    // Cold (F_NOCACHE) double-buffered read of `path` in `chunk_bytes` blocks, zero
    // compute — the fair pipelined-raw baseline for the capacity-bound bench. Returns
    // mean ms/pass over `iters`.
    fn cold_pipelined_read_ms(path: &str, total_bytes: usize, chunk_bytes: usize, iters: usize) -> f64 {
        use std::io::Read;
        use std::os::unix::io::AsRawFd;
        use std::sync::mpsc::sync_channel;
        extern "C" { fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32; }
        const F_NOCACHE: i32 = 48;
        let nblk = total_bytes / chunk_bytes;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let (full_tx, full_rx) = sync_channel::<Vec<u8>>(2);
            let (empty_tx, empty_rx) = sync_channel::<Vec<u8>>(2);
            empty_tx.send(vec![0u8; chunk_bytes]).unwrap();
            empty_tx.send(vec![0u8; chunk_bytes]).unwrap();
            std::thread::scope(|s| {
                s.spawn(move || {
                    let mut f = std::fs::File::open(path).unwrap();
                    unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
                    for _ in 0..nblk {
                        let mut buf = match empty_rx.recv() { Ok(b) => b, Err(_) => break };
                        if f.read_exact(&mut buf).is_err() { break; }
                        if full_tx.send(buf).is_err() { break; }
                    }
                });
                while let Ok(buf) = full_rx.recv() {
                    std::hint::black_box(&buf);
                    let _ = empty_tx.send(buf);
                }
            });
        }
        t.elapsed().as_secs_f64() * 1000.0 / iters as f64
    }

    // R149c: capacity-bound speed demo on the REAL Gemma lm-head, read COLD (F_NOCACHE).
    // Proves the R143->R149 thesis on real weights through the real streamer: a cold
    // pipelined bit-plane stream (read + REEPLANE-W6 decode + dot) beats a cold raw-bf16
    // read (given zero compute — the conservative baseline). 8 GB box => F_NOCACHE forces
    // the capacity-bound regime. Lossless gate inside. Reports GB/s, speedup, verdict.
    #[test]
    #[ignore]
    fn r149c_real_lmhead_capacity_bound() {
        use std::io::Read; // for the sidecar header read; cold-read helpers own F_NOCACHE

        let model = "../../models/gemma-3-1b-it-rawcodec.rllm";
        let tname = "model.embed_tokens.weight";
        let sidecar = "/tmp/gemma1b-lmhead.sidecar";
        let raw_path = "/tmp/gemma1b-lmhead-raw.bin";
        write_lmhead_sidecar(model, tname, 256, sidecar).unwrap();

        let mut m = crate::LazyRllmModel::open(model).unwrap();
        let meta = m.tensor(tname).unwrap().clone();
        let vocab = meta.shape[0] as usize;
        let hidden = meta.shape[1] as usize;
        let bf16 = m.with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec())).unwrap().unwrap();
        std::fs::write(raw_path, &bf16).unwrap();
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin() * 0.4).collect();

        // Lossless gate: streamed (warm) == resident, on the real logits.
        let resident = lm_head_logits_parallel_bf16(&act, &bf16, vocab, hidden);
        let streamed = stream_lmhead_from_sidecar(sidecar, &act).unwrap();
        assert_eq!(streamed, resident, "R149c lossless gate: streamed lm-head must equal resident");

        // Parse sidecar header for the direct cold streaming call.
        let mut head = [0u8; 256];
        let n_head = std::fs::File::open(sidecar).unwrap().read(&mut head).unwrap();
        assert!(n_head >= 19 && &head[0..4] == b"RLMH" && head[4] == 2);
        let p = head[17] as usize;
        let w = head[18] as usize;
        let palette = head[19..19 + p].to_vec();
        let header_len = (19 + p) as u64;
        let block_rows = 256usize;
        let num_blocks = vocab / block_rows;

        let raw_bytes = bf16.len();
        let comp_bytes = std::fs::metadata(sidecar).unwrap().len() as usize - header_len as usize;
        let iters = 5usize;

        // Cold raw bf16: naive single blocking F_NOCACHE read (zero compute).
        let raw_ms = cold_blocking_read_ms(raw_path, raw_bytes, iters);

        // Cold PIPELINED raw bf16: fair baseline (same reader/cadence, zero decode) —
        // isolates the compression contribution from the pipelining contribution. A real
        // raw runtime would pipeline its reads too, never block on one 604 MB read.
        let pipe_raw_ms = cold_pipelined_read_ms(raw_path, raw_bytes, block_rows * hidden * 2, iters);

        // Cold pipelined bit-plane: read + REEPLANE-W6 decode + dot, F_NOCACHE.
        let comp_ms = {
            let mut out = vec![0f32; vocab];
            let t = std::time::Instant::now();
            for _ in 0..iters {
                streaming_bitplane_gemv(
                    sidecar, &palette, hidden, w, block_rows, num_blocks, &act, &mut out, true, header_len,
                );
                std::hint::black_box(&out);
            }
            t.elapsed().as_secs_f64() * 1000.0 / iters as f64
        };
        let _ = std::fs::remove_file(sidecar);
        let _ = std::fs::remove_file(raw_path);

        let raw_gb = raw_bytes as f64 / 1e9;
        let comp_gb = comp_bytes as f64 / 1e9;
        let byte_ratio = comp_bytes as f64 / raw_bytes as f64;
        let ram_save = (1.0 - byte_ratio) * 100.0;
        // Honest decomposition: the apples-to-apples win is comp vs PIPELINED raw
        // (both double-buffered) — that is compression's own contribution. The pipeline
        // contribution is the separate raw_ms -> pipe_raw_ms gain. Don't conflate them.
        let compression_speedup = pipe_raw_ms / comp_ms;
        let pipeline_speedup = raw_ms / pipe_raw_ms;
        let total_speedup = raw_ms / comp_ms;
        // Verdict is on the FAIR baseline: does compression itself win, cold?
        let verdict = if comp_ms < pipe_raw_ms {
            "GO -- bit-plane beats the FAIR pipelined-raw baseline (compression wins, cold)"
        } else if comp_ms < pipe_raw_ms * 1.05 {
            "MARGINAL (within 5% of pipelined raw)"
        } else {
            "NO-GO (pipelined raw cold read still faster)"
        };
        eprintln!(
            "\n=== R149c capacity-bound REAL lm-head (Gemma 3 1B, w={w}, cold F_NOCACHE, {iters} iters) ===\n\
             lossless gate: OK ({vocab} logits identical)\n\
             raw bf16   single-read  {raw_gb:.3} GB -> {raw_ms:.0} ms  ({:.2} GB/s, naive blocking)\n\
             raw bf16   PIPELINED    {raw_gb:.3} GB -> {pipe_raw_ms:.0} ms  ({:.2} GB/s, fair baseline, zero decode)\n\
             bit-plane  PIPELINED    {comp_gb:.3} GB -> {comp_ms:.0} ms  ({:.2} GB/s, decode+dot pipelined)\n\
             bytes: {ram_save:.0}% fewer  ({byte_ratio:.3} ratio)\n\
             compression effect (comp vs pipelined-raw): {compression_speedup:.2}x   <- the honest win\n\
             pipeline effect (single-read -> pipelined raw): {pipeline_speedup:.2}x\n\
             total vs naive raw: {total_speedup:.2}x\n\
             VERDICT: {verdict}\n",
            raw_gb / (raw_ms / 1e3),
            raw_gb / (pipe_raw_ms / 1e3),
            comp_gb / (comp_ms / 1e3),
        );
    }

    // Cold (F_NOCACHE) read of `path` partitioned across `n_threads` workers (each
    // seeks to its block range, reads sequentially, zero compute). The FAIR baseline
    // for the parallel streamer — gives raw the SAME concurrent-read parallelism so
    // the comparison isolates compression from concurrent-read bandwidth.
    fn cold_parallel_read_ms(path: &str, total_bytes: usize, chunk_bytes: usize, n_threads: usize, iters: usize) -> f64 {
        use std::io::{Read, Seek, SeekFrom};
        use std::os::unix::io::AsRawFd;
        extern "C" { fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32; }
        const F_NOCACHE: i32 = 48;
        let nblk = total_bytes / chunk_bytes;
        let n_threads = n_threads.clamp(1, nblk.max(1));
        let per = nblk.div_ceil(n_threads);
        let t = std::time::Instant::now();
        for _ in 0..iters {
            std::thread::scope(|s| {
                let mut start = 0usize;
                while start < nblk {
                    let cnt = per.min(nblk - start);
                    s.spawn(move || {
                        let mut f = std::fs::File::open(path).unwrap();
                        unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
                        if f.seek(SeekFrom::Start((start * chunk_bytes) as u64)).is_err() { return; }
                        let mut buf = vec![0u8; chunk_bytes];
                        for _ in 0..cnt {
                            if f.read_exact(&mut buf).is_err() { break; }
                            std::hint::black_box(&buf);
                        }
                    });
                    start += cnt;
                }
            });
        }
        t.elapsed().as_secs_f64() * 1000.0 / iters as f64
    }

    // Append `k` copies of `src` to `path` (streamed write). Used to build genuinely
    // >RAM files so F_NOCACHE reads are truly cold (a 600 MB file fits 8 GB RAM and
    // gives unreliable, cache-polluted timings — see R149c variance).
    fn replicate_to_file(src: &[u8], k: usize, path: &str) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for _ in 0..k {
            f.write_all(src).unwrap();
        }
        f.sync_all().unwrap();
    }

    // R150a: does PARALLEL decode (REESTREAM-PAR) make streaming the COMPRESSED lm-head
    // beat reading raw bf16, in the GENUINE capacity-bound regime (files > RAM, cold)?
    // R149c was decode-bound (single consumer, ~8 GB/s) and lost; with N-thread decode
    // (aggregate > read bandwidth) the path becomes read-bound and should win by the
    // ~12% byte ratio. Both sides cold over >8 GB files (8 GB box) + F_NOCACHE.
    // (A 600 MB fits-in-RAM file gives F_NOCACHE-variance noise — R149c's flaw.)
    #[test]
    #[ignore]
    fn r150a_parallel_lmhead_capacity_bound() {
        use std::io::{Read, Seek, SeekFrom};
        let model = "../../models/gemma-3-1b-it-rawcodec.rllm";
        let tname = "model.embed_tokens.weight";
        let sidecar = "/tmp/gemma1b-lmhead.sidecar";
        let raw_big = "/tmp/r150a_raw_big.bin";
        let comp_big = "/tmp/r150a_comp_big.bin";
        write_lmhead_sidecar(model, tname, 256, sidecar).unwrap();
        let mut m = crate::LazyRllmModel::open(model).unwrap();
        let meta = m.tensor(tname).unwrap().clone();
        let vocab = meta.shape[0] as usize;
        let hidden = meta.shape[1] as usize;
        let bf16 = m.with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec())).unwrap().unwrap();
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin() * 0.4).collect();
        let resident = lm_head_logits_parallel_bf16(&act, &bf16, vocab, hidden);

        // one framed-comp copy = sidecar minus its header.
        let mut head = [0u8; 256];
        let mut sf = std::fs::File::open(sidecar).unwrap();
        let _ = sf.read(&mut head).unwrap();
        let p = head[17] as usize;
        let w = head[18] as usize;
        let palette = head[19..19 + p].to_vec();
        let header_len = (19 + p) as u64;
        let block_rows = 256usize;
        let num_blocks = vocab / block_rows;
        let comp_copy_len = std::fs::metadata(sidecar).unwrap().len() as usize - header_len as usize;
        let mut comp_copy = vec![0u8; comp_copy_len];
        sf.seek(SeekFrom::Start(header_len)).unwrap();
        sf.read_exact(&mut comp_copy).unwrap();
        let _ = std::fs::remove_file(sidecar);

        // Lossless gate (warm): parallel stream of one copy == resident.
        std::fs::write(comp_big, &comp_copy).unwrap();
        let mut out0 = vec![0f32; vocab];
        let cores = std::thread::available_parallelism().map(usize::from).unwrap_or(4);
        streaming_bitplane_gemv_parallel(comp_big, &palette, hidden, w, block_rows, num_blocks, &act, &mut out0, false, 0, cores);
        assert_eq!(out0, resident, "R150a lossless gate: parallel stream must equal resident");

        // Replicate both > 8 GB so reads are genuinely cold.
        let k = 16usize; // raw ~9.7 GB, comp ~8.4 GB on an 8 GB box
        replicate_to_file(&bf16, k, raw_big);
        replicate_to_file(&comp_copy, k, comp_big);
        drop(bf16);
        drop(comp_copy);
        let raw_total = std::fs::metadata(raw_big).unwrap().len() as usize;
        let comp_total = std::fs::metadata(comp_big).unwrap().len() as usize;

        // Baselines, all cold >RAM: raw 1-reader (context), raw N-reader (FAIR — same
        // concurrency as comp), comp parallel. comp-vs-rawN isolates compression.
        let raw_block = block_rows * hidden * 2;
        let raw1_ms = cold_pipelined_read_ms(raw_big, raw_total, raw_block, 1);
        let raw_n_ms = cold_parallel_read_ms(raw_big, raw_total, raw_block, cores, 1);
        let mut scratch_out = vec![0f32; vocab * k];
        let comp1_ms = {
            let t = std::time::Instant::now();
            streaming_bitplane_gemv_parallel(comp_big, &palette, hidden, w, block_rows, num_blocks * k, &act, &mut scratch_out, true, 0, 1);
            std::hint::black_box(&scratch_out);
            t.elapsed().as_secs_f64() * 1000.0
        };
        let comp_n_ms = {
            let t = std::time::Instant::now();
            streaming_bitplane_gemv_parallel(comp_big, &palette, hidden, w, block_rows, num_blocks * k, &act, &mut scratch_out, true, 0, cores);
            std::hint::black_box(&scratch_out);
            t.elapsed().as_secs_f64() * 1000.0
        };
        let _ = std::fs::remove_file(raw_big);
        let _ = std::fs::remove_file(comp_big);

        let raw_gb = raw_total as f64 / 1e9;
        let comp_gb = comp_total as f64 / 1e9;
        // Honest decomposition: comp-vs-rawN (both N readers) = compression's own win.
        let compression_speedup = raw_n_ms / comp_n_ms;
        let concurrent_read_speedup = raw1_ms / raw_n_ms;
        let verdict = if comp_n_ms < raw_n_ms { "GO -- compression wins vs the FAIR parallel-raw baseline (read-bound, byte ratio)" }
            else if comp_n_ms < raw_n_ms * 1.05 { "MARGINAL (within 5% of parallel raw)" }
            else { "NO-GO (parallel raw still faster)" };
        eprintln!(
            "\n=== R150a PARALLEL capacity-bound real lm-head (Gemma 3 1B, w={w}, >RAM cold, {cores} cores) ===\n\
             lossless gate: OK ({vocab} logits identical)\n\
             raw bf16  1-reader   {raw_gb:.2} GB -> {raw1_ms:.0} ms  ({:.2} GB/s)\n\
             raw bf16  {cores}-reader  {raw_gb:.2} GB -> {raw_n_ms:.0} ms  ({:.2} GB/s, FAIR baseline)\n\
             bit-plane nt=1       {comp_gb:.2} GB -> {comp1_ms:.0} ms  ({:.2} GB/s)\n\
             bit-plane nt={cores}       {comp_gb:.2} GB -> {comp_n_ms:.0} ms  ({:.2} GB/s)\n\
             bytes: {:.0}% fewer\n\
             compression effect (comp vs parallel-raw): {compression_speedup:.2}x   <- the honest win\n\
             concurrent-read effect (1->{cores} readers): {concurrent_read_speedup:.2}x\n\
             VERDICT: {verdict}\n",
            raw_gb / (raw1_ms / 1e3),
            raw_gb / (raw_n_ms / 1e3),
            comp_gb / (comp1_ms / 1e3),
            comp_gb / (comp_n_ms / 1e3),
            (1.0 - comp_total as f64 / raw_total as f64) * 100.0,
        );
    }
