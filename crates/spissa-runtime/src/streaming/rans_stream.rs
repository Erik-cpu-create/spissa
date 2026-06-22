// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.
//
// R153 REESTREAM-RANS: parallel streaming GEMV over an rANS-exponent sidecar.
//
// Replaces the bit-plane fixed-width exponent index (5–6 bits) with a per-block
// interleaved-rANS exponent stream at the entropy floor (~2.6 bits), keeping the raw
// residual byte. Reaches ~10.5 bits/weight (R151/R152) vs bit-plane's 13–14, so the
// >RAM cold stream reads ~34% fewer bytes than raw bf16 — the capacity-bound win
// (decode hidden by REESTREAM-PAR's parallel blocks, R150a).
//
// Sidecar "RLMR" v1: header [magic, ver, hidden, vocab, block_rows, freq[256] u16,
// num_blocks, block_len[num_blocks] u32] then per block
// [lane_len[4] u32 ++ lane0..3 (rANS) ++ residual(block_rows*hidden)].

/// rANS-exponent sidecar header offset of the (variable) block-length table tail.
/// magic(4)+ver(1)+hidden(4)+vocab(4)+block_rows(4)+freq(512)+num_blocks(4) = 533.
const RLMR_FIXED_HEADER: usize = 17 + 512 + 4;

/// Write a model's tied bf16 lm-head as an rANS-exponent sidecar (RLMR v1).
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub fn write_lmhead_sidecar_rans(
    model_path: &str,
    tensor_name: &str,
    block_rows: usize,
    out_path: &str,
) -> crate::Result<()> {
    use rtc_codec::{count_symbols, normalize_freqs, split_bf16};
    let mut m = crate::LazySpissaModel::open(model_path)?;
    let meta = m.tensor(tensor_name)?.clone();
    let vocab = meta.shape[0] as usize;
    let hidden = meta.shape[1] as usize;
    let bf16 = m
        .with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec()))?
        .ok_or_else(|| crate::RuntimeError::InvalidTensorData("lm-head must be raw bf16".into()))?;
    let n = vocab * hidden;
    // Pad rows up to a multiple of block_rows with zeros so any row count works
    // (e.g. down_proj [1152×…], not %256). The padding decodes to bf16 zero and the
    // reader truncates to the real `vocab`, so it stays lossless.
    let vocab_padded = vocab.div_ceil(block_rows) * block_rows;
    let mut exp = vec![0u8; vocab_padded * hidden];
    let mut res = vec![0u8; vocab_padded * hidden];
    for i in 0..n {
        let (e, r) = split_bf16(u16::from_le_bytes([bf16[2 * i], bf16[2 * i + 1]]));
        exp[i] = e;
        res[i] = r;
    }
    drop(bf16);
    let freq = normalize_freqs(&count_symbols(&exp)); // includes padding zeros => freq[0] >= 1
    let sidecar = build_rans_sidecar(&exp, &res, &freq, vocab, hidden, block_rows);
    std::fs::write(out_path, &sidecar)
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("write rans sidecar: {e}")))?;
    Ok(())
}

/// Serialize an RLMR v1 sidecar from per-weight exponent + residual planes and a
/// global frequency table. Shared by the model writer and the synthetic tests.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn build_rans_sidecar(
    exp: &[u8],
    res: &[u8],
    freq: &[u32; 256],
    vocab: usize,
    hidden: usize,
    block_rows: usize,
) -> Vec<u8> {
    use rtc_codec::rans_encode_interleaved4;
    // `vocab` is the real row count; exp/res are padded to num_blocks*block_rows rows.
    let num_blocks = vocab.div_ceil(block_rows);
    let bw = block_rows * hidden;
    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    for blk in 0..num_blocks {
        let s = blk * bw;
        let lanes = rans_encode_interleaved4(&exp[s..s + bw], freq);
        let mut body = Vec::with_capacity(16 + bw);
        for l in &lanes {
            body.extend_from_slice(&(l.len() as u32).to_le_bytes());
        }
        for l in &lanes {
            body.extend_from_slice(l);
        }
        body.extend_from_slice(&res[s..s + bw]);
        bodies.push(body);
    }
    let mut out = Vec::with_capacity(RLMR_FIXED_HEADER + num_blocks * 4 + vocab * hidden);
    out.extend_from_slice(b"RLMR");
    out.push(1);
    out.extend_from_slice(&(hidden as u32).to_le_bytes());
    out.extend_from_slice(&(vocab as u32).to_le_bytes());
    out.extend_from_slice(&(block_rows as u32).to_le_bytes());
    for f in freq.iter() {
        out.extend_from_slice(&(*f as u16).to_le_bytes());
    }
    out.extend_from_slice(&(num_blocks as u32).to_le_bytes());
    for b in &bodies {
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    }
    for b in &bodies {
        out.extend_from_slice(b);
    }
    out
}

/// Parse the RLMR header: returns (hidden, vocab, block_rows, freq, block_lens, header_len).
#[cfg(target_arch = "aarch64")]
fn read_rans_header(path: &str) -> crate::Result<(usize, usize, usize, [u32; 256], Vec<u32>, u64)> {
    use std::io::Read;
    let bad = || crate::RuntimeError::InvalidTensorData("bad RLMR sidecar header".into());
    let mut f = std::fs::File::open(path)
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("open rans sidecar: {e}")))?;
    let mut fixed = [0u8; RLMR_FIXED_HEADER];
    f.read_exact(&mut fixed).map_err(|_| bad())?;
    if &fixed[0..4] != b"RLMR" || fixed[4] != 1 {
        return Err(bad());
    }
    let rd = |o: usize| u32::from_le_bytes(fixed[o..o + 4].try_into().unwrap()) as usize;
    let hidden = rd(5);
    let vocab = rd(9);
    let block_rows = rd(13);
    let mut freq = [0u32; 256];
    for (s, fr) in freq.iter_mut().enumerate() {
        let o = 17 + s * 2;
        *fr = u16::from_le_bytes([fixed[o], fixed[o + 1]]) as u32;
    }
    let num_blocks = rd(17 + 512);
    let mut lens_bytes = vec![0u8; num_blocks * 4];
    f.read_exact(&mut lens_bytes).map_err(|_| bad())?;
    let block_lens: Vec<u32> = lens_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let header_len = (RLMR_FIXED_HEADER + num_blocks * 4) as u64;
    Ok((hidden, vocab, block_rows, freq, block_lens, header_len))
}

/// NEON reconstruct of `n` bf16 weights from exponent + residual planes into `out`
/// (`out.len() >= n*2`). bf16 = (sign<<15) | (exp<<7) | mantissa, where the residual
/// byte holds sign (bit 7) + mantissa (bits 0..6). 8 weights/iter; scalar tail.
/// SAFETY: caller guarantees `exp.len()>=n`, `residual.len()>=n`, `out.len()>=n*2`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn reconstruct_bf16_neon(exp: &[u8], residual: &[u8], n: usize, out: &mut [u8]) {
    use std::arch::aarch64::*;
    let mask80 = vdupq_n_u16(0x80);
    let mask7f = vdupq_n_u16(0x7f);
    let out16 = out.as_mut_ptr() as *mut u16;
    let groups = n / 8;
    for g in 0..groups {
        let e = vld1_u8(exp.as_ptr().add(g * 8));
        let r = vld1_u8(residual.as_ptr().add(g * 8));
        let e16 = vmovl_u8(e);
        let r16 = vmovl_u8(r);
        let sign = vshlq_n_u16(vandq_u16(r16, mask80), 8);
        let ep = vshlq_n_u16(e16, 7);
        let mant = vandq_u16(r16, mask7f);
        let bf = vorrq_u16(vorrq_u16(sign, ep), mant);
        vst1q_u16(out16.add(g * 8), bf);
    }
    for i in groups * 8..n {
        let bits = rtc_codec::join_bf16(exp[i], residual[i]);
        out[2 * i] = bits as u8;
        out[2 * i + 1] = (bits >> 8) as u8;
    }
}

/// Reconstruct each row's bf16 from (exponent, residual) and dot it against the
/// activation, writing `block_rows` logits.
#[cfg(target_arch = "aarch64")]
fn rans_decode_dot_block(
    exp: &[u8],
    residual: &[u8],
    hidden: usize,
    block_rows: usize,
    last_hidden: &[f32],
    scratch: &mut [u8],
    out_block: &mut [f32],
) {
    for r in 0..block_rows {
        unsafe { reconstruct_bf16_neon(&exp[r * hidden..], &residual[r * hidden..], hidden, scratch) };
        out_block[r] = bf16_row_dot_f32(last_hidden, &scratch[..hidden * 2], hidden);
    }
}

/// Parallel streaming rANS GEMV (REESTREAM-RANS). Partitions blocks across workers;
/// each seeks to its block range, reads + rANS-decodes the exponent lanes + dots.
/// Bit-identical to a single-thread decode+dot.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn streaming_rans_gemv_parallel(
    path: &str,
    freq: &[u32; 256],
    hidden: usize,
    block_rows: usize,
    num_blocks: usize,
    block_offsets: &[u64],
    block_lens: &[u32],
    last_hidden: &[f32],
    out: &mut [f32],
    nocache: bool,
    n_threads: usize,
) {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32;
    }
    const F_NOCACHE: i32 = 48;
    let bw = block_rows * hidden;
    let n_threads = n_threads.clamp(1, num_blocks.max(1));
    let blocks_per = num_blocks.div_ceil(n_threads);

    std::thread::scope(|s| {
        let mut rest = &mut out[..];
        let mut blk_start = 0usize;
        while blk_start < num_blocks {
            let nblk = blocks_per.min(num_blocks - blk_start);
            let (mine, tail) = rest.split_at_mut(nblk * block_rows);
            rest = tail;
            s.spawn(move || {
                let mut f = File::open(path).unwrap();
                if nocache {
                    unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
                }
                if f.seek(SeekFrom::Start(block_offsets[blk_start])).is_err() {
                    return;
                }
                // Build decode tables once per thread (not per block) and reuse buffers.
                let tables = rtc_codec::rans_build_tables(freq);
                let mut scratch = vec![0u8; hidden * 2];
                let mut exps = vec![0u8; bw];
                let max_len = (blk_start..blk_start + nblk)
                    .map(|b| block_lens[b] as usize)
                    .max()
                    .unwrap_or(0);
                let mut buf = vec![0u8; max_len];
                for bi in 0..nblk {
                    let blk = blk_start + bi;
                    let len = block_lens[blk] as usize;
                    if f.read_exact(&mut buf[..len]).is_err() {
                        return;
                    }
                    let ll = |k: usize| u32::from_le_bytes(buf[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
                    let (l0, l1, l2, l3) = (ll(0), ll(1), ll(2), ll(3));
                    let mut o = 16;
                    let lane0 = &buf[o..o + l0];
                    o += l0;
                    let lane1 = &buf[o..o + l1];
                    o += l1;
                    let lane2 = &buf[o..o + l2];
                    o += l2;
                    let lane3 = &buf[o..o + l3];
                    o += l3;
                    let residual = &buf[o..o + bw];
                    rtc_codec::rans_decode_interleaved4_into([lane0, lane1, lane2, lane3], bw, &tables, &mut exps);
                    let out_block = &mut mine[bi * block_rows..(bi + 1) * block_rows];
                    rans_decode_dot_block(&exps, residual, hidden, block_rows, last_hidden, &mut scratch, out_block);
                }
            });
            blk_start += nblk;
        }
    });
}

/// Compute lm-head logits by streaming the rANS-exponent sidecar (R153).
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub(crate) fn stream_lmhead_from_rans_sidecar(path: &str, last_hidden: &[f32]) -> crate::Result<Vec<f32>> {
    let (hidden, vocab, block_rows, freq, block_lens, header_len) = read_rans_header(path)?;
    let num_blocks = block_lens.len();
    let mut offsets = Vec::with_capacity(num_blocks);
    let mut acc = header_len;
    for &l in &block_lens {
        offsets.push(acc);
        acc += l as u64;
    }
    let nocache = matches!(std::env::var("RLLM_STREAM_NOCACHE").as_deref(), Ok("1") | Ok("true"));
    // R154: oversubscribe (2x cores) by default — workers block on cold reads, so extra
    // threads overlap I/O-wait with rANS decode (1.10x -> 1.39x in the >RAM cold bench).
    // RLLM_STREAM_THREADS overrides.
    let n_threads = std::env::var("RLLM_STREAM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| 2 * std::thread::available_parallelism().map(usize::from).unwrap_or(1));
    // Blocks cover num_blocks*block_rows >= vocab rows (last block zero-padded); stream
    // into the padded buffer then truncate to the real vocab.
    let vocab_padded = num_blocks * block_rows;
    let mut logits = vec![0f32; vocab_padded];
    streaming_rans_gemv_parallel(
        path, &freq, hidden, block_rows, num_blocks, &offsets, &block_lens, last_hidden, &mut logits, nocache, n_threads,
    );
    logits.truncate(vocab);
    Ok(logits)
}

#[cfg(all(test, target_arch = "aarch64"))]
#[path = "tests_rans_stream.rs"]
mod tests_rans_stream;
