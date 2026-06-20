// Pipelined streaming bit-plane GEMV (R148 REESTREAM, capacity-bound runtime kernel).
//
// Reads block-framed compressed planes sequentially from a file while decoding +
// dotting the previous block, so the cold-SSD read of block N+1 overlaps the
// decode of block N. Reuses rtc-codec decode (R143/R146) + bf16_row_dot_f32.

/// Double-buffer pipelined streaming bit-plane GEMV. Reads `num_blocks` blocks of
/// `block_rows` rows each (`[B×index bytes ++ B×residual bytes]`, index width `w`)
/// sequentially from `path`; a reader thread streams the next block while this
/// thread decodes + dots the current one. Writes `num_blocks*block_rows` logits.
/// Bit-identical to a single-thread decode+dot. `w` selects the decode kernel
/// (REEPLANE w=5 / REEPLANE-W6 w=6 / scalar otherwise) via the codec dispatcher.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn streaming_bitplane_gemv(
    path: &str,
    palette: &[u8],
    hidden: usize,
    w: usize,
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

    let row_idx = hidden * w / 8;
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
                rtc_codec::decode_bitplane_row_into(palette, idx, res, hidden, w as u8, &mut scratch);
                out[blk * block_rows + r] = bf16_row_dot_f32(last_hidden, &scratch, hidden);
            }
            let _ = empty_tx.send(buf);
        }
    });
}

/// Read a model's tied bf16 embedding/LM-head tensor, bit-plane encode it, and
/// write a block-framed sidecar file the streaming lm-head path consumes.
/// SAFETY/constraints: the tensor must be raw-bf16 readable (pack with `--codec raw`),
/// the palette must be ≤ 64 distinct exponents (index width `w ∈ 1..=6`, decodable
/// by the streaming dispatcher), `hidden·w % 8 == 0`, and `vocab % block_rows == 0`.
/// The sidecar header (v2) records `w`, so w=5 (Llama) and w=6 (Gemma) both stream.
/// Sidecar producer: consumed by the streaming tests and a future `rllm` subcommand.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
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
    // enc.data[5] = flags; FLAG_RAW (0x01) => palette > 64 distinct exponents, stored
    // raw (no index plane) and not bit-plane streamable.
    if enc.data[5] & 0x01 != 0 {
        return Err(crate::RuntimeError::InvalidTensorData(
            "lm-head has > 64 distinct bf16 exponents (raw fallback); not bit-plane streamable".into(),
        ));
    }
    let w = enc.data[15] as usize;
    if !(1..=6).contains(&w) {
        return Err(crate::RuntimeError::InvalidTensorData(format!(
            "lm-head bit-plane width {w} unsupported by the streaming decoder (need 1..=6)",
        )));
    }
    if hidden * w % 8 != 0 {
        return Err(crate::RuntimeError::InvalidTensorData(format!(
            "hidden*w must be byte-aligned for row framing (hidden={hidden}, w={w})",
        )));
    }
    if vocab % block_rows != 0 {
        return Err(crate::RuntimeError::InvalidTensorData(
            "vocab must be a multiple of block_rows".into(),
        ));
    }
    let p = enc.data[14] as usize;
    let row_idx = hidden * w / 8;
    let idx_plane = &enc.data[16 + p..16 + p + vocab * row_idx];
    let residuals = &enc.data[16 + p + vocab * row_idx..16 + p + vocab * row_idx + n];

    // Header v2: RLMH, ver=2, hidden, vocab, block_rows, palette_len, w, palette.
    let mut sidecar = Vec::with_capacity(19 + p + vocab * (row_idx + hidden));
    sidecar.extend_from_slice(b"RLMH");
    sidecar.push(2);
    sidecar.extend_from_slice(&(hidden as u32).to_le_bytes());
    sidecar.extend_from_slice(&(vocab as u32).to_le_bytes());
    sidecar.extend_from_slice(&(block_rows as u32).to_le_bytes());
    sidecar.push(p as u8);
    sidecar.push(w as u8);
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
    if got < 18 || &head[0..4] != b"RLMH" {
        return Err(crate::RuntimeError::InvalidTensorData("bad sidecar header".into()));
    }
    let hidden = u32::from_le_bytes(head[5..9].try_into().unwrap()) as usize;
    let vocab = u32::from_le_bytes(head[9..13].try_into().unwrap()) as usize;
    let block_rows = u32::from_le_bytes(head[13..17].try_into().unwrap()) as usize;
    let p = head[17] as usize;
    // v1: w=5 implied, palette at byte 18. v2: w at byte 18, palette at byte 19.
    let (w, pal_off) = match head[4] {
        1 => (5usize, 18usize),
        2 => (head[18] as usize, 19usize),
        v => return Err(crate::RuntimeError::InvalidTensorData(format!("unknown sidecar version {v}"))),
    };
    if got < pal_off + p {
        return Err(crate::RuntimeError::InvalidTensorData("sidecar palette truncated".into()));
    }
    let palette = head[pal_off..pal_off + p].to_vec();
    let header_len = (pal_off + p) as u64;
    let num_blocks = vocab / block_rows;
    // Capacity-bound regime (model > RAM): RLLM_STREAM_NOCACHE=1 reads the sidecar
    // with F_NOCACHE so cold lm-head reads don't thrash the page cache. Default off.
    let nocache = matches!(std::env::var("RLLM_STREAM_NOCACHE").as_deref(), Ok("1") | Ok("true"));
    let mut logits = vec![0f32; vocab];
    streaming_bitplane_gemv(
        path, &palette, hidden, w, block_rows, num_blocks, last_hidden, &mut logits, nocache, header_len,
    );
    Ok(logits)
}

#[cfg(all(test, target_arch = "aarch64"))]
#[path = "tests_bitplane_stream.rs"]
mod bitplane_stream_tests;
