// Fused REEPLANE decode -> bfdot GEMV (R144, Phase C).
//
// Computes lm-head logits directly from resident rtc-bitplane-v1 planes: per
// row, decode the row's bf16 weights into an L1 scratch (no DRAM
// materialization) and bfdot against the once-converted activation. Reuses
// rtc-codec's `decode_neon_w5_into` (R143) + this module's `Bf16DotActivation`
// / `bf16_row_dot_bf16` (R141). Not yet wired into the runtime — the Phase C
// building block measured by the R144 bench.

/// Fused bit-plane lm-head GEMV. `palette`/`idx_plane`/`residuals` are the
/// `rtc-bitplane-v1` planes (w=5) of a row-major weight matrix with `hidden`
/// weights per row. For each output row, decode the row's bf16 weights into a
/// reused L1 scratch and `bfdot` against the once-converted activation; bf16 is
/// never written to DRAM. Writes `out.len()` logits starting at `row_offset`.
/// Requires `hidden * 5 % 8 == 0` (rows byte-aligned). Not yet runtime-wired.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn lm_head_logits_rows_bitplane(
    last_hidden: &[f32],
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    hidden: usize,
    row_offset: usize,
    out: &mut [f32],
) {
    debug_assert_eq!((hidden * 5) % 8, 0, "bit-plane row index plane must be byte-aligned");
    let row_idx_bytes = hidden * 5 / 8; // 1280 for hidden=2048
    let act = Bf16DotActivation::new(last_hidden);
    let mut scratch = vec![0u8; hidden * 2];
    for (r, logit) in out.iter_mut().enumerate() {
        let row = row_offset + r;
        // Open-ended slices: NEON group loads may read a few bytes past the row's
        // span into the next row (in-bounds); the last row is covered by the
        // decode kernel's simd_groups guard + scalar tail.
        let idx = &idx_plane[row * row_idx_bytes..];
        let res = &residuals[row * hidden..];
        rtc_codec::decode_neon_w5_into(palette, idx, res, hidden, &mut scratch);
        *logit = act.row_dot(&scratch, hidden);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod bitplane_gemv_tests {
    use super::*;
    use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};

    // vocab*hidden bf16 weights cycling 32 exponents (=> w=5), random mantissa.
    fn make_embedding(vocab: usize, hidden: usize) -> Vec<u8> {
        let mut state = 0xDEAD_BEEF_1234_5678u64;
        let mut out = Vec::with_capacity(vocab * hidden * 2);
        for k in 0..vocab * hidden {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let exp = (96 + (k % 32)) as u16 & 0xFF;
            let sign = ((state >> 31) & 1) as u16;
            let mant = (state & 0x7F) as u16;
            let bits = (sign << 15) | (exp << 7) | mant;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    #[test]
    fn fused_bitplane_gemv_matches_plain_bf16_bit_for_bit() {
        let (vocab, hidden) = (64usize, 2048usize);
        let bf16 = make_embedding(vocab, hidden);
        let enc = BitplaneCodec
            .encode(
                &bf16,
                &EncodeMeta {
                    name: "e".into(),
                    shape: vec![(vocab * hidden) as u64],
                    dtype: "bf16".into(),
                },
            )
            .unwrap();
        let p = enc.data[14] as usize;
        assert_eq!(enc.data[15], 5, "expected w=5");
        let mut off = 16;
        let palette = &enc.data[off..off + p];
        off += p;
        let idx_bytes = (vocab * hidden * 5 + 7) / 8;
        let idx_plane = &enc.data[off..off + idx_bytes];
        off += idx_bytes;
        let residuals = &enc.data[off..off + vocab * hidden];

        // deterministic activation
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();

        let mut plain = vec![0f32; vocab];
        lm_head_logits_rows_bf16(&act, &bf16, hidden, 0, &mut plain);

        let mut fused = vec![0f32; vocab];
        lm_head_logits_rows_bitplane(&act, palette, idx_plane, residuals, hidden, 0, &mut fused);

        assert_eq!(fused, plain, "fused bit-plane GEMV must equal plain bf16 GEMV bit-for-bit");
    }

    #[test]
    #[ignore]
    fn fused_bitplane_gemv_feasibility() {
        // Both paths use bfdot for an apples-to-apples compute comparison.
        std::env::set_var("RLLM_Q8_ACTIVATION", "1");
        std::env::set_var("RLLM_BF16_DOT", "1");

        let bf16 = std::fs::read("/tmp/rllm-bf16-sample.bin")
            .expect("run dump_bf16_embedding_sample first");
        let hidden = 2048usize;
        let n_weights = bf16.len() / 2;
        let vocab = n_weights / hidden;
        assert_eq!(vocab * hidden, n_weights, "sample must be vocab*2048");

        let enc = BitplaneCodec
            .encode(
                &bf16,
                &EncodeMeta {
                    name: "e".into(),
                    shape: vec![n_weights as u64],
                    dtype: "bf16".into(),
                },
            )
            .unwrap();
        assert_eq!(enc.data[15], 5, "expected w=5");
        let p = enc.data[14] as usize;
        let mut off = 16;
        let palette = enc.data[off..off + p].to_vec();
        off += p;
        let idx_bytes = (n_weights * 5 + 7) / 8;
        let idx_plane = enc.data[off..off + idx_bytes].to_vec();
        off += idx_bytes;
        let residuals = enc.data[off..off + n_weights].to_vec();

        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();

        // Correctness + parity on the real sample.
        let mut plain = vec![0f32; vocab];
        lm_head_logits_rows_bf16(&act, &bf16, hidden, 0, &mut plain);
        let mut fused = vec![0f32; vocab];
        lm_head_logits_rows_bitplane(&act, &palette, &idx_plane, &residuals, hidden, 0, &mut fused);
        assert_eq!(fused, plain, "fused must equal plain bf16 (lossless) on the real embedding");

        let iters = 5;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            lm_head_logits_rows_bf16(&act, &bf16, hidden, 0, &mut plain);
            std::hint::black_box(&plain);
        }
        let plain_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let t = std::time::Instant::now();
        for _ in 0..iters {
            lm_head_logits_rows_bitplane(&act, &palette, &idx_plane, &residuals, hidden, 0, &mut fused);
            std::hint::black_box(&fused);
        }
        let fused_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let bf16_mb = bf16.len() as f64 / 1e6;
        let plane_mb = (palette.len() + idx_plane.len() + residuals.len()) as f64 / 1e6;
        let speedup = plain_ms / fused_ms;
        let ram_save = (1.0 - plane_mb / bf16_mb) * 100.0;
        let verdict = if speedup >= 1.05 && plane_mb < bf16_mb {
            "GO (faster + smaller)"
        } else if plane_mb < bf16_mb && speedup >= 0.95 {
            "MARGINAL (smaller, ~same speed)"
        } else {
            "NO-GO (slower)"
        };
        eprintln!(
            "\n=== R144 REEFUSE-PLANE-DOT fused GEMV FEASIBILITY (single-core) ===\n\
             vocab={vocab} hidden={hidden}  (lossless parity: OK)\n\
             plain bf16 GEMV:   {plain_ms:.1} ms/token  (resident {bf16_mb:.0} MB)\n\
             fused bit-plane:   {fused_ms:.1} ms/token  (resident {plane_mb:.0} MB, {ram_save:.0}% less)\n\
             speedup: {speedup:.2}x\n\
             VERDICT: {verdict}\n",
        );
    }

    // Time a GEMV split across `nthreads` OS threads; `f(row_offset, out_slice)`
    // computes a contiguous row range. Returns ms/token (3 warm iters).
    #[cfg(test)]
    fn time_par<F: Fn(usize, &mut [f32]) + Sync>(vocab: usize, nthreads: usize, f: F) -> f64 {
        let iters = 3;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let mut out = vec![0f32; vocab];
            let rows_per = vocab.div_ceil(nthreads);
            let fref = &f;
            std::thread::scope(|s| {
                let mut rest = &mut out[..];
                let mut start = 0usize;
                while start < vocab {
                    let rows = rows_per.min(vocab - start);
                    let (slice, r) = rest.split_at_mut(rows);
                    rest = r;
                    s.spawn(move || fref(start, slice));
                    start += rows;
                }
            });
            std::hint::black_box(&out);
        }
        t.elapsed().as_secs_f64() * 1000.0 / iters as f64
    }

    #[test]
    #[ignore]
    fn fused_bitplane_gemv_multicore_scout() {
        std::env::set_var("RLLM_Q8_ACTIVATION", "1");
        std::env::set_var("RLLM_BF16_DOT", "1");
        let bf16 = std::fs::read("/tmp/rllm-bf16-sample.bin")
            .expect("run dump_bf16_embedding_sample first");
        let hidden = 2048usize;
        let n_weights = bf16.len() / 2;
        let vocab = n_weights / hidden;
        let enc = BitplaneCodec
            .encode(
                &bf16,
                &EncodeMeta { name: "e".into(), shape: vec![n_weights as u64], dtype: "bf16".into() },
            )
            .unwrap();
        let p = enc.data[14] as usize;
        let mut off = 16;
        let palette = enc.data[off..off + p].to_vec();
        off += p;
        let idx_bytes = (n_weights * 5 + 7) / 8;
        let idx_plane = enc.data[off..off + idx_bytes].to_vec();
        off += idx_bytes;
        let residuals = enc.data[off..off + n_weights].to_vec();
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();

        let cores = std::thread::available_parallelism().map(usize::from).unwrap_or(1);
        eprintln!(
            "\n=== R145 multi-core fused GEMV SCOUT (available cores: {cores}) ===\n\
             threads |  plain bf16  |  fused bit-plane | speedup"
        );
        for &nt in &[1usize, 2, 4, 6, 8] {
            let plain = time_par(vocab, nt, |base, slice| {
                lm_head_logits_rows_bf16(&act, &bf16, hidden, base, slice)
            });
            let fused = time_par(vocab, nt, |base, slice| {
                lm_head_logits_rows_bitplane(&act, &palette, &idx_plane, &residuals, hidden, base, slice)
            });
            eprintln!(
                "   {nt:2}   |  {plain:6.1} ms  |   {fused:6.1} ms     | {:.2}x{}",
                plain / fused,
                if fused < plain { "  <-- fused faster" } else { "" }
            );
        }
        eprintln!();
    }
}
