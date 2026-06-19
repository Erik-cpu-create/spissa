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
}
