// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

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

/// One logit = dot(activation, decoded weight row), fused: each 8-weight group is
/// decoded straight into a `uint16x8` bf16 vector (R143 logic) and `bfdot`-ed into
/// one of 4 accumulator chains in R141's order — no L1 scratch, decode overlaps
/// the dot via out-of-order execution. Bit-identical to decode-then-bfdot.
/// SAFETY: caller guarantees FEAT_BF16; `hidden % 32 == 0`; `act_bf16.len() >= hidden`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "bf16")]
unsafe fn bitplane_row_dot_bfdot(
    act_bf16: &[u16],
    pal: &[u8; 32],
    idx_row: &[u8],
    res_row: &[u8],
    hidden: usize,
) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(hidden % 32, 0);
    let pal_tbl = uint8x8x4_t(
        vld1_u8(pal.as_ptr()),
        vld1_u8(pal.as_ptr().add(8)),
        vld1_u8(pal.as_ptr().add(16)),
        vld1_u8(pal.as_ptr().add(24)),
    );
    let bidx_hi: [u8; 8] = [0, 0, 1, 1, 2, 3, 3, 4];
    let bidx_lo: [u8; 8] = [1, 1, 2, 2, 3, 4, 4, 5];
    let neg_shift: [i16; 8] = [-11, -6, -9, -4, -7, -10, -5, -8];
    let vhi = vld1_u8(bidx_hi.as_ptr());
    let vlo = vld1_u8(bidx_lo.as_ptr());
    let vshift = vld1q_s16(neg_shift.as_ptr());
    let mask5 = vdupq_n_u16(0x1f);
    let mask80 = vdupq_n_u16(0x80);
    let mask7f = vdupq_n_u16(0x7f);
    let idx_len = idx_row.len();

    // Decode 8 weights of group g into a uint16x8 of bf16. Bounds-checked load so
    // the final row's last group (whose 8-byte load runs past the plane) is safe;
    // the branch is predictable (taken only at the very end).
    let decode_group = |g: usize| -> uint16x8_t {
        let off = g * 5;
        let grp = if off + 8 <= idx_len {
            vld1_u8(idx_row.as_ptr().add(off))
        } else {
            let mut buf = [0u8; 8];
            let avail = idx_len - off;
            core::ptr::copy_nonoverlapping(idx_row.as_ptr().add(off), buf.as_mut_ptr(), avail);
            vld1_u8(buf.as_ptr())
        };
        let hi = vtbl1_u8(grp, vhi);
        let lo = vtbl1_u8(grp, vlo);
        let window = vorrq_u16(vshlq_n_u16(vmovl_u8(hi), 8), vmovl_u8(lo));
        let idx16 = vandq_u16(vshlq_u16(window, vshift), mask5);
        let idx8 = vmovn_u16(idx16);
        let exp8 = vtbl4_u8(pal_tbl, idx8);
        let res8 = vld1_u8(res_row.as_ptr().add(g * 8));
        let res16 = vmovl_u8(res8);
        let exp16 = vmovl_u8(exp8);
        let sign = vshlq_n_u16(vandq_u16(res16, mask80), 8);
        let ep = vshlq_n_u16(exp16, 7);
        let mant = vandq_u16(res16, mask7f);
        vorrq_u16(vorrq_u16(sign, ep), mant)
    };

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let groups = hidden / 8; // multiple of 4 since hidden % 32 == 0
    let aptr = act_bf16.as_ptr();
    let mut g = 0usize;
    while g < groups {
        let w0 = decode_group(g);
        let w1 = decode_group(g + 1);
        let w2 = decode_group(g + 2);
        let w3 = decode_group(g + 3);
        let a0 = vld1q_u16(aptr.add(g * 8));
        let a1 = vld1q_u16(aptr.add((g + 1) * 8));
        let a2 = vld1q_u16(aptr.add((g + 2) * 8));
        let a3 = vld1q_u16(aptr.add((g + 3) * 8));
        core::arch::asm!(
            "bfdot {acc0:v}.4s, {w0:v}.8h, {a0:v}.8h",
            "bfdot {acc1:v}.4s, {w1:v}.8h, {a1:v}.8h",
            "bfdot {acc2:v}.4s, {w2:v}.8h, {a2:v}.8h",
            "bfdot {acc3:v}.4s, {w3:v}.8h, {a3:v}.8h",
            acc0 = inout(vreg) acc0,
            acc1 = inout(vreg) acc1,
            acc2 = inout(vreg) acc2,
            acc3 = inout(vreg) acc3,
            w0 = in(vreg) w0, w1 = in(vreg) w1, w2 = in(vreg) w2, w3 = in(vreg) w3,
            a0 = in(vreg) a0, a1 = in(vreg) a1, a2 = in(vreg) a2, a3 = in(vreg) a3,
            options(nomem, nostack),
        );
        g += 4;
    }
    vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
}

/// Fused bit-plane lm-head GEMV: convert the activation to bf16 once, preload the
/// palette, then dot every row directly from the resident planes via
/// `bitplane_row_dot_bfdot`. Bit-identical to decode-then-bfdot; not yet wired.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn lm_head_logits_bitplane_fused(
    last_hidden: &[f32],
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    hidden: usize,
    row_offset: usize,
    out: &mut [f32],
) {
    debug_assert_eq!((hidden * 5) % 8, 0);
    let row_idx_bytes = hidden * 5 / 8;
    let act_bf16 = convert_f32_to_bf16(last_hidden);
    let mut pal = [0u8; 32];
    pal[..palette.len()].copy_from_slice(palette);
    for (r, logit) in out.iter_mut().enumerate() {
        let row = row_offset + r;
        let idx = &idx_plane[row * row_idx_bytes..];
        let res = &residuals[row * hidden..];
        *logit = unsafe { bitplane_row_dot_bfdot(&act_bf16, &pal, idx, res, hidden) };
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
        std::env::set_var("SPISSA_Q8_ACTIVATION", "1");
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
        std::env::set_var("SPISSA_Q8_ACTIVATION", "1");
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

    #[test]
    #[ignore]
    fn fused_kernel_multicore_bench() {
        if !bf16_dot_available() {
            eprintln!("FEAT_BF16 not present; skipping");
            return;
        }
        // Set so the plain bf16 path also uses bfdot (apples-to-apples). This is an
        // #[ignore] bench run alone, so the global env mutation is safe here.
        std::env::set_var("SPISSA_Q8_ACTIVATION", "1");
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
        let bf16_mb = bf16.len() as f64 / 1e6;
        let plane_mb = (palette.len() + idx_plane.len() + residuals.len()) as f64 / 1e6;

        eprintln!(
            "\n=== R145 tile-fused GEMV multi-core BENCH ===\n\
             resident: bf16 {bf16_mb:.0} MB vs bit-plane {plane_mb:.0} MB ({:.0}% less)\n\
             threads | plain bf16 | R145 fused | speedup",
            (1.0 - plane_mb / bf16_mb) * 100.0
        );
        let mut best = f64::INFINITY;
        let mut best_plain = f64::INFINITY;
        for &nt in &[1usize, 2, 4, 6, 8] {
            let plain = time_par(vocab, nt, |base, slice| {
                lm_head_logits_rows_bf16(&act, &bf16, hidden, base, slice)
            });
            let fused = time_par(vocab, nt, |base, slice| {
                lm_head_logits_bitplane_fused(&act, &palette, &idx_plane, &residuals, hidden, base, slice)
            });
            eprintln!(
                "   {nt:2}   |  {plain:6.1} ms |  {fused:6.1} ms | {:.2}x{}",
                plain / fused,
                if fused < plain { "  <-- WIN" } else { "" }
            );
            if fused < best {
                best = fused;
                best_plain = plain;
            }
        }
        let verdict = if best <= best_plain {
            "GO (faster + 19% less RAM, lossless)"
        } else if best <= best_plain * 1.25 {
            "MARGINAL (close; try 16-wide vqtbl2q / strategy B)"
        } else {
            "NO-GO (decode still loses)"
        };
        eprintln!("\nbest fused {best:.1} ms vs best plain {best_plain:.1} ms => VERDICT: {verdict}\n");
    }

    #[test]
    fn fused_kernel_matches_reference_bit_for_bit() {
        if !bf16_dot_available() {
            return; // FEAT_BF16 required; no-op on non-bf16 hardware
        }
        let (vocab, hidden) = (96usize, 2048usize);
        let bf16 = make_embedding(vocab, hidden);
        let enc = BitplaneCodec
            .encode(
                &bf16,
                &EncodeMeta { name: "e".into(), shape: vec![(vocab * hidden) as u64], dtype: "bf16".into() },
            )
            .unwrap();
        let p = enc.data[14] as usize;
        assert_eq!(enc.data[15], 5);
        let mut off = 16;
        let palette = &enc.data[off..off + p];
        off += p;
        let idx_bytes = (vocab * hidden * 5 + 7) / 8;
        let idx_plane = &enc.data[off..off + idx_bytes];
        off += idx_bytes;
        let residuals = &enc.data[off..off + vocab * hidden];
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();

        // Env-free bfdot reference: decode each row, then R141 bf16_row_dot_bf16.
        let act_bf16 = convert_f32_to_bf16(&act);
        let row_idx_bytes = hidden * 5 / 8;
        let mut reference = vec![0f32; vocab];
        for (r, slot) in reference.iter_mut().enumerate() {
            let decoded = rtc_codec::decode_neon_w5(
                palette,
                &idx_plane[r * row_idx_bytes..],
                &residuals[r * hidden..],
                hidden,
            );
            *slot = unsafe { bf16_row_dot_bf16(&act_bf16, &decoded, hidden) };
        }

        let mut fused = vec![0f32; vocab];
        lm_head_logits_bitplane_fused(&act, palette, idx_plane, residuals, hidden, 0, &mut fused);

        assert_eq!(fused, reference, "fused kernel must equal decode+bfdot reference bit-for-bit");
    }
}
