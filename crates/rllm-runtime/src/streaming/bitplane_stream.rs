// Pipelined streaming bit-plane GEMV (R148 REESTREAM, capacity-bound runtime kernel).
//
// Reads block-framed compressed planes sequentially from a file while decoding +
// dotting the previous block, so the cold-SSD read of block N+1 overlaps the
// decode of block N. Reuses rtc-codec decode (R143/R146) + bf16_row_dot_f32.

/// Double-buffer pipelined streaming bit-plane GEMV. Reads `num_blocks` blocks of
/// `block_rows` rows each (`[B×index bytes ++ B×residual bytes]`, w=5) sequentially
/// from `path`; a reader thread streams the next block while this thread decodes +
/// dots the current one. Writes `num_blocks*block_rows` logits. Bit-identical to a
/// single-thread decode+dot. Not yet runtime-wired — the R148 capacity-bound kernel.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn streaming_bitplane_gemv(
    path: &str,
    palette: &[u8],
    hidden: usize,
    block_rows: usize,
    num_blocks: usize,
    last_hidden: &[f32],
    out: &mut [f32],
    nocache: bool,
    data_offset: u64,
) {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::io::AsRawFd;
    use std::sync::mpsc::sync_channel;
    extern "C" {
        fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32;
    }
    const F_NOCACHE: i32 = 48;

    let row_idx = hidden * 5 / 8;
    let row_res = hidden;
    let block_bytes = block_rows * (row_idx + row_res);

    let (full_tx, full_rx) = sync_channel::<(usize, Vec<u8>)>(2);
    let (empty_tx, empty_rx) = sync_channel::<Vec<u8>>(2);
    empty_tx.send(vec![0u8; block_bytes]).unwrap();
    empty_tx.send(vec![0u8; block_bytes]).unwrap();

    std::thread::scope(|s| {
        // reader: fill the spare buffer with block N+1 while the consumer drains N.
        s.spawn(move || {
            let mut f = File::open(path).unwrap();
            if nocache {
                unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
            }
            if data_offset > 0 {
                if f.seek(SeekFrom::Start(data_offset)).is_err() {
                    return;
                }
            }
            for blk in 0..num_blocks {
                let mut buf = match empty_rx.recv() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                if f.read_exact(&mut buf).is_err() {
                    break;
                }
                if full_tx.send((blk, buf)).is_err() {
                    break;
                }
            }
            // full_tx drops here -> consumer's recv() ends after draining.
        });

        // consumer: decode+dot each row of each received block.
        let mut scratch = vec![0u8; hidden * 2];
        while let Ok((blk, buf)) = full_rx.recv() {
            for r in 0..block_rows {
                let idx = &buf[r * row_idx..];
                let res = &buf[block_rows * row_idx + r * row_res..];
                unsafe { rtc_codec::decode16_w5_into(palette, idx, res, hidden, &mut scratch) };
                out[blk * block_rows + r] = bf16_row_dot_f32(last_hidden, &scratch, hidden);
            }
            let _ = empty_tx.send(buf);
        }
    });
}

/// Read a model's tied bf16 embedding/LM-head tensor, bit-plane encode it, and
/// write a block-framed sidecar file the streaming lm-head path consumes.
/// SAFETY/constraints: the tensor must be raw-bf16 readable (pack with `--codec raw`),
/// w must be 5, and `vocab % block_rows == 0`.
#[cfg(target_arch = "aarch64")]
pub fn write_lmhead_sidecar(
    model_path: &str,
    tensor_name: &str,
    block_rows: usize,
    out_path: &str,
) -> crate::Result<()> {
    use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};
    let mut m = crate::LazyRllmModel::open(model_path)?;
    let meta = m.tensor(tensor_name)?.clone();
    let vocab = meta.shape[0] as usize;
    let hidden = meta.shape[1] as usize;
    let bf16 = m
        .with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec()))?
        .ok_or_else(|| crate::RuntimeError::InvalidTensorData(
            "lm-head must be raw bf16 (repack with --codec raw)".into(),
        ))?;
    let n = vocab * hidden;
    let enc = BitplaneCodec
        .encode(&bf16, &EncodeMeta { name: tensor_name.into(), shape: vec![n as u64], dtype: "bf16".into() })
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("bitplane encode: {e}")))?;
    if enc.data[15] != 5 {
        return Err(crate::RuntimeError::InvalidTensorData(format!(
            "lm-head bit-plane width {} != 5; decode16 kernel needs w=5",
            enc.data[15]
        )));
    }
    if vocab % block_rows != 0 {
        return Err(crate::RuntimeError::InvalidTensorData(
            "vocab must be a multiple of block_rows".into(),
        ));
    }
    let p = enc.data[14] as usize;
    let row_idx = hidden * 5 / 8;
    let idx_plane = &enc.data[16 + p..16 + p + vocab * row_idx];
    let residuals = &enc.data[16 + p + vocab * row_idx..16 + p + vocab * row_idx + n];

    let mut sidecar = Vec::with_capacity(18 + p + vocab * (row_idx + hidden));
    sidecar.extend_from_slice(b"RLMH");
    sidecar.push(1);
    sidecar.extend_from_slice(&(hidden as u32).to_le_bytes());
    sidecar.extend_from_slice(&(vocab as u32).to_le_bytes());
    sidecar.extend_from_slice(&(block_rows as u32).to_le_bytes());
    sidecar.push(p as u8);
    sidecar.extend_from_slice(&enc.data[16..16 + p]); // palette
    for blk in 0..vocab / block_rows {
        for r in 0..block_rows {
            let row = blk * block_rows + r;
            sidecar.extend_from_slice(&idx_plane[row * row_idx..(row + 1) * row_idx]);
        }
        for r in 0..block_rows {
            let row = blk * block_rows + r;
            sidecar.extend_from_slice(&residuals[row * hidden..(row + 1) * hidden]);
        }
    }
    std::fs::write(out_path, &sidecar)
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("write sidecar: {e}")))?;
    Ok(())
}

/// Compute lm-head logits by streaming the bit-plane sidecar (R148 kernel).
#[cfg(target_arch = "aarch64")]
pub(crate) fn stream_lmhead_from_sidecar(path: &str, last_hidden: &[f32]) -> crate::Result<Vec<f32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("open sidecar: {e}")))?;
    let mut head = [0u8; 256];
    let got = f.read(&mut head)
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("read sidecar header: {e}")))?;
    if got < 18 || &head[0..4] != b"RLMH" || head[4] != 1 {
        return Err(crate::RuntimeError::InvalidTensorData("bad sidecar header".into()));
    }
    let hidden = u32::from_le_bytes(head[5..9].try_into().unwrap()) as usize;
    let vocab = u32::from_le_bytes(head[9..13].try_into().unwrap()) as usize;
    let block_rows = u32::from_le_bytes(head[13..17].try_into().unwrap()) as usize;
    let p = head[17] as usize;
    if got < 18 + p {
        return Err(crate::RuntimeError::InvalidTensorData("sidecar palette truncated".into()));
    }
    let palette = head[18..18 + p].to_vec();
    let header_len = (18 + p) as u64;
    let num_blocks = vocab / block_rows;
    let mut logits = vec![0f32; vocab];
    streaming_bitplane_gemv(
        path, &palette, hidden, block_rows, num_blocks, last_hidden, &mut logits, false, header_len,
    );
    Ok(logits)
}

#[cfg(all(test, target_arch = "aarch64"))]
mod bitplane_stream_tests {
    use super::*;
    use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};

    fn make_embedding(vocab: usize, hidden: usize) -> Vec<u8> {
        let mut state = 0x0BAD_F00D_1234_5678u64;
        let mut out = Vec::with_capacity(vocab * hidden * 2);
        for k in 0..vocab * hidden {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let exp = (96 + (k % 32)) as u16 & 0xFF;
            let bits = (((state >> 31) & 1) as u16) << 15 | (exp << 7) | (state & 0x7F) as u16;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    // Frame the flat planes into [B×index ++ B×residual] contiguous blocks.
    fn frame_blocks(idx_plane: &[u8], residuals: &[u8], hidden: usize, vocab: usize, b: usize) -> Vec<u8> {
        let row_idx = hidden * 5 / 8;
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
        let framed = frame_blocks(idx_plane, residuals, hidden, vocab, b);
        let path = "/tmp/r148_unit.bin";
        std::fs::write(path, &framed).unwrap();
        let mut out = vec![0f32; vocab];
        streaming_bitplane_gemv(path, &palette, hidden, b, vocab / b, &act, &mut out, false, 0);
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
        sc.extend_from_slice(&frame_blocks(&idx_plane, &residuals, hidden, vocab, b));
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
        let framed = frame_blocks(idx_plane, residuals, hidden, vocab, b);

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
            streaming_bitplane_gemv(comp_path, &palette, hidden, b, total_blocks, &act, &mut out, true, 0);
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
}
