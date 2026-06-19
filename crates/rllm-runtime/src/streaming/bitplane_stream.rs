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
) {
    use std::fs::File;
    use std::io::Read;
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
        streaming_bitplane_gemv(path, &palette, hidden, b, vocab / b, &act, &mut out, false);
        let _ = std::fs::remove_file(path);

        assert_eq!(out, reference, "streaming GEMV must equal single-thread decode+dot bit-for-bit");
    }
}
