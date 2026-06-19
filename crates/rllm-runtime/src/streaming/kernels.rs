fn accumulate_weight_chunk_multiply_into(
    input: &[f32],
    weights: &[f32],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<()> {
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    let element_end = element_start
        .checked_add(weights.len())
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;
    while local_idx < weights.len() {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let row_len = (config.in_features - in_feature).min(weights.len() - local_idx);
        let weight_row = &weights[local_idx..local_idx + row_len];
        for batch_idx in 0..config.batch {
            let input_start = batch_idx * config.in_features + in_feature;
            let input_row = &input[input_start..input_start + row_len];
            let mut acc = state.current_acc[batch_idx];
            let mut idx = 0usize;
            while idx + 4 <= row_len {
                acc += weight_row[idx] * input_row[idx]
                    + weight_row[idx + 1] * input_row[idx + 1]
                    + weight_row[idx + 2] * input_row[idx + 2]
                    + weight_row[idx + 3] * input_row[idx + 3];
                idx += 4;
            }
            while idx < row_len {
                acc += weight_row[idx] * input_row[idx];
                idx += 1;
            }
            state.current_acc[batch_idx] = acc;
        }

        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }
    Ok(())
}

fn accumulate_weight_chunk(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;

    let element_end = element_start
        .checked_add(weights.len())
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;
    while local_idx < weights.len() {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weights.len() - local_idx);
        let weight_row = &weights[local_idx..local_idx + row_len];

        let mut batch_idx = 0usize;
        while batch_idx + 8 <= config.batch {
            let output_idx0 = batch_idx * config.out_features + out_feature;
            let output_idx1 = (batch_idx + 1) * config.out_features + out_feature;
            let output_idx2 = (batch_idx + 2) * config.out_features + out_feature;
            let output_idx3 = (batch_idx + 3) * config.out_features + out_feature;
            let output_idx4 = (batch_idx + 4) * config.out_features + out_feature;
            let output_idx5 = (batch_idx + 5) * config.out_features + out_feature;
            let output_idx6 = (batch_idx + 6) * config.out_features + out_feature;
            let output_idx7 = (batch_idx + 7) * config.out_features + out_feature;
            let mut acc0 = output[output_idx0];
            let mut acc1 = output[output_idx1];
            let mut acc2 = output[output_idx2];
            let mut acc3 = output[output_idx3];
            let mut acc4 = output[output_idx4];
            let mut acc5 = output[output_idx5];
            let mut acc6 = output[output_idx6];
            let mut acc7 = output[output_idx7];
            let input_start0 = batch_idx * config.in_features + in_feature;
            let input_start1 = (batch_idx + 1) * config.in_features + in_feature;
            let input_start2 = (batch_idx + 2) * config.in_features + in_feature;
            let input_start3 = (batch_idx + 3) * config.in_features + in_feature;
            let input_start4 = (batch_idx + 4) * config.in_features + in_feature;
            let input_start5 = (batch_idx + 5) * config.in_features + in_feature;
            let input_start6 = (batch_idx + 6) * config.in_features + in_feature;
            let input_start7 = (batch_idx + 7) * config.in_features + in_feature;
            let mut idx = 0;
            while idx + 4 <= row_len {
                let w = &weight_row[idx..idx + 4];
                let i0 = &input[input_start0 + idx..input_start0 + idx + 4];
                let i1 = &input[input_start1 + idx..input_start1 + idx + 4];
                let i2 = &input[input_start2 + idx..input_start2 + idx + 4];
                let i3 = &input[input_start3 + idx..input_start3 + idx + 4];
                let i4 = &input[input_start4 + idx..input_start4 + idx + 4];
                let i5 = &input[input_start5 + idx..input_start5 + idx + 4];
                let i6 = &input[input_start6 + idx..input_start6 + idx + 4];
                let i7 = &input[input_start7 + idx..input_start7 + idx + 4];

                acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                acc4 += w[0] * i4[0] + w[1] * i4[1] + w[2] * i4[2] + w[3] * i4[3];
                acc5 += w[0] * i5[0] + w[1] * i5[1] + w[2] * i5[2] + w[3] * i5[3];
                acc6 += w[0] * i6[0] + w[1] * i6[1] + w[2] * i6[2] + w[3] * i6[3];
                acc7 += w[0] * i7[0] + w[1] * i7[1] + w[2] * i7[2] + w[3] * i7[3];

                idx += 4;
            }
            while idx < row_len {
                let weight = weight_row[idx];
                acc0 += input[input_start0 + idx] * weight;
                acc1 += input[input_start1 + idx] * weight;
                acc2 += input[input_start2 + idx] * weight;
                acc3 += input[input_start3 + idx] * weight;
                acc4 += input[input_start4 + idx] * weight;
                acc5 += input[input_start5 + idx] * weight;
                acc6 += input[input_start6 + idx] * weight;
                acc7 += input[input_start7 + idx] * weight;
                idx += 1;
            }
            output[output_idx0] = acc0;
            output[output_idx1] = acc1;
            output[output_idx2] = acc2;
            output[output_idx3] = acc3;
            output[output_idx4] = acc4;
            output[output_idx5] = acc5;
            output[output_idx6] = acc6;
            output[output_idx7] = acc7;
            batch_idx += 8;
        }
        while batch_idx + 4 <= config.batch {
            let output_idx0 = batch_idx * config.out_features + out_feature;
            let output_idx1 = (batch_idx + 1) * config.out_features + out_feature;
            let output_idx2 = (batch_idx + 2) * config.out_features + out_feature;
            let output_idx3 = (batch_idx + 3) * config.out_features + out_feature;
            let mut acc0 = output[output_idx0];
            let mut acc1 = output[output_idx1];
            let mut acc2 = output[output_idx2];
            let mut acc3 = output[output_idx3];
            let input_start0 = batch_idx * config.in_features + in_feature;
            let input_start1 = (batch_idx + 1) * config.in_features + in_feature;
            let input_start2 = (batch_idx + 2) * config.in_features + in_feature;
            let input_start3 = (batch_idx + 3) * config.in_features + in_feature;
            let mut idx = 0;
            while idx + 4 <= row_len {
                let w = &weight_row[idx..idx + 4];
                let i0 = &input[input_start0 + idx..input_start0 + idx + 4];
                let i1 = &input[input_start1 + idx..input_start1 + idx + 4];
                let i2 = &input[input_start2 + idx..input_start2 + idx + 4];
                let i3 = &input[input_start3 + idx..input_start3 + idx + 4];

                acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                idx += 4;
            }
            while idx < row_len {
                let weight = weight_row[idx];
                acc0 += input[input_start0 + idx] * weight;
                acc1 += input[input_start1 + idx] * weight;
                acc2 += input[input_start2 + idx] * weight;
                acc3 += input[input_start3 + idx] * weight;
                idx += 1;
            }
            output[output_idx0] = acc0;
            output[output_idx1] = acc1;
            output[output_idx2] = acc2;
            output[output_idx3] = acc3;
            batch_idx += 4;
        }
        while batch_idx < config.batch {
            let input_start = batch_idx * config.in_features + in_feature;
            let input_row = &input[input_start..input_start + row_len];
            let output_idx = batch_idx * config.out_features + out_feature;
            let mut acc = output[output_idx];
            let mut idx = 0;
            while idx + 4 <= row_len {
                let w = &weight_row[idx..idx + 4];
                let i0 = &input_row[idx..idx + 4];
                acc += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                idx += 4;
            }
            while idx < row_len {
                acc += input_row[idx] * weight_row[idx];
                idx += 1;
            }
            output[output_idx] = acc;
            batch_idx += 1;
        }

        local_idx += row_len;
        global_idx += row_len;
    }
    Ok(())
}

const Q8_ACTIVATION_ENV: &str = "RLLM_Q8_ACTIVATION";

/// Opt-in (default off) int8-activation path for Q8 matmul parity validation.
/// When enabled, activations are quantized to int8 per 32-element segment and the
/// dot runs as int8×int8 (ARM `sdot`) instead of dequantizing the weight to f32.
/// This is the REEBORN-Q8 direction gated behind `RLLM_Q8_ACTIVATION` so the exact
/// f32 path stays default until token/logit parity is confirmed on a real model.
fn q8_activation_path_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var(Q8_ACTIVATION_ENV)
                .ok()
                .map(|v| v.trim().to_ascii_lowercase()),
            Some(v) if matches!(v.as_str(), "1" | "true" | "yes" | "on")
        )
    })
}

#[cfg(target_arch = "aarch64")]
fn q8_sdot_available() -> bool {
    static AVAIL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAIL.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

/// Quantize 32 f32 activations to int8 with an absmax scale.
fn quantize_seg32_i8(seg: &[f32]) -> ([i8; 32], f32) {
    let mut amax = 0.0f32;
    for &v in &seg[..32] {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let (scale, inv) = if amax > 0.0 {
        (amax / 127.0, 127.0 / amax)
    } else {
        (0.0, 0.0)
    };
    let mut q = [0i8; 32];
    for k in 0..32 {
        q[k] = (seg[k] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    (q, scale)
}

fn i8_dot32_scalar(w: &[u8], x: &[i8; 32]) -> i32 {
    let mut acc = 0i32;
    for k in 0..32 {
        acc += (w[k] as i8 as i32) * (x[k] as i32);
    }
    acc
}

// Native ARM `sdot` over 32 int8 lanes via inline asm. The `vdotq_s32` intrinsic
// is still nightly-gated (`stdarch_neon_dotprod`); `sdot` works on stable through
// `asm!` + `target_feature(dotprod)`. Caller verifies dotprod via q8_sdot_available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn i8_dot32_sdot(w: &[u8], x: &[i8; 32]) -> i32 {
    use std::arch::asm;
    let mut acc: i32;
    asm!(
        "movi v4.4s, #0",
        "ld1 {{v0.16b, v1.16b}}, [{w}]",
        "ld1 {{v2.16b, v3.16b}}, [{x}]",
        "sdot v4.4s, v0.16b, v2.16b",
        "sdot v4.4s, v1.16b, v3.16b",
        "addv s4, v4.4s",
        "fmov {acc:w}, s4",
        w = in(reg) w.as_ptr(),
        x = in(reg) x.as_ptr(),
        acc = out(reg) acc,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
    );
    acc
}

fn i8_dot32(w: &[u8], x: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        if q8_sdot_available() {
            return unsafe { i8_dot32_sdot(w, x) };
        }
    }
    i8_dot32_scalar(w, x)
}

// ---- REEFUSE-Q8-I8MM-PANEL: runtime promotion of the R118 lab kernel ----
//
// Strategy: when the chunk is row-aligned, `batch >= 2`, and the CPU has i8mm,
// process pairs of adjacent output rows via a packed-panel `smmla` kernel.
// Activation is quantized + packed once per matmul (cached thread-local; key by
// pointer + fingerprint, same shape as R112). Weight pairs pack into local
// scratch per chunk. Per-block weight + activation scales match R111's per-block
// convention (parity-validated). Falls back to the existing R111 naive int8-dot
// path for batch=1, non-i8mm CPUs, non-row-aligned chunks, and odd rows.

#[cfg(target_arch = "aarch64")]
fn q8_i8mm_available() -> bool {
    static AVAIL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAIL.get_or_init(|| std::arch::is_aarch64_feature_detected!("i8mm"))
}

/// Quantize activations to int8 per 32-element K-block, with per-row per-block
/// absmax scale. Layout: `q[row * in_features + b * 32 + k]`,
/// `scales[row * blocks_per_row + b]`.
fn quantize_input_q8_blocks(
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> (Vec<i8>, Vec<f32>) {
    let blocks_per_row = in_features / 32;
    let mut q = vec![0i8; batch * in_features];
    let mut scales = vec![0.0f32; batch * blocks_per_row];
    for row in 0..batch {
        for b in 0..blocks_per_row {
            let off = row * in_features + b * 32;
            let mut amax = 0.0f32;
            for k in 0..32 {
                let a = input[off + k].abs();
                if a > amax {
                    amax = a;
                }
            }
            let (scale, inv) = if amax > 0.0 {
                (amax / 127.0, 127.0 / amax)
            } else {
                (0.0, 0.0)
            };
            scales[row * blocks_per_row + b] = scale;
            for k in 0..32 {
                q[off + k] = (input[off + k] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (q, scales)
}

/// Pack pairs of batch (token) rows into pair-major panels for `smmla`. Per pair
/// per K-block: 4 segments of 16 contiguous bytes `[t0_K0..7 | t1_K0..7]`.
/// Odd-batch tail is left unpacked; the kernel falls back to the raw `act_i8`.
fn pack_act_panel_pairs(act_i8: &[i8], batch: usize, in_features: usize) -> Vec<i8> {
    let pairs = batch / 2;
    let blocks = in_features / 32;
    let mut panel = vec![0i8; pairs * 2 * in_features];
    for p in 0..pairs {
        let r0 = p * 2;
        let r1 = r0 + 1;
        for b in 0..blocks {
            for seg in 0..4 {
                let dst = p * 2 * in_features + b * 64 + seg * 16;
                let src0 = r0 * in_features + b * 32 + seg * 8;
                let src1 = r1 * in_features + b * 32 + seg * 8;
                for k in 0..8 {
                    panel[dst + k] = act_i8[src0 + k];
                    panel[dst + 8 + k] = act_i8[src1 + k];
                }
            }
        }
    }
    panel
}

struct Q8PanelActCache {
    ptr: usize,
    len: usize,
    batch: usize,
    in_features: usize,
    fingerprint: u64,
    act_i8: Vec<i8>,
    act_panel: Vec<i8>,
    act_scales: Vec<f32>,
}

thread_local! {
    static Q8_PANEL_ACT_CACHE: std::cell::RefCell<Option<Q8PanelActCache>> =
        const { std::cell::RefCell::new(None) };
}

fn q8_act_fingerprint(input: &[f32]) -> u64 {
    let n = input.len();
    if n == 0 {
        return 0;
    }
    // Sample up to 64 points spread evenly across the buffer (vs the original 4)
    // and fold them with a per-index FNV-style mix. The cache is keyed by this
    // fingerprint, so on the decode path — where the same activation buffer
    // address is reused across tokens — a richer sample makes a stale-cache
    // collision (two distinct activations matching at every sampled index)
    // astronomically unlikely. Cost is O(64), amortized across the matmul.
    let samples = n.min(64);
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ (n as u64);
    for s in 0..samples {
        let i = if samples == 1 {
            0
        } else {
            (s * (n - 1)) / (samples - 1)
        };
        let v = input[i].to_bits() as u64;
        h ^= v
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .rotate_left((s as u32 * 7 + 13) & 63);
        h = h.wrapping_mul(0x0000_0001_0000_01B3);
    }
    h
}

/// Cache the quantized + panel-packed activation by (ptr, len, shape, content
/// fingerprint) so a single matmul amortizes the quant+pack work across all
/// chunks. Same design as R112's `with_quantized_activations`, extended with the
/// pair-major panel and the per-block scale layout R119 needs.
fn with_q8_panel_activations<R>(
    input: &[f32],
    batch: usize,
    in_features: usize,
    f: impl FnOnce(&[i8], &[i8], &[f32]) -> R,
) -> R {
    let ptr = input.as_ptr() as usize;
    let len = input.len();
    let fp = q8_act_fingerprint(input);
    Q8_PANEL_ACT_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let hit = cache.as_ref().is_some_and(|c| {
            c.ptr == ptr
                && c.len == len
                && c.batch == batch
                && c.in_features == in_features
                && c.fingerprint == fp
        });
        if !hit {
            let (act_i8, act_scales) = quantize_input_q8_blocks(input, batch, in_features);
            let act_panel = pack_act_panel_pairs(&act_i8, batch, in_features);
            *cache = Some(Q8PanelActCache {
                ptr,
                len,
                batch,
                in_features,
                fingerprint: fp,
                act_i8,
                act_panel,
                act_scales,
            });
        }
        let entry = cache.as_ref().unwrap();
        f(&entry.act_i8, &entry.act_panel, &entry.act_scales)
    })
}

/// Pack one weight-row pair from the q8 chunk into a contiguous panel and read
/// the two per-block fp16 scales (one per row). Same layout as
/// `pack_act_panel_pairs` (4 segments of 16 bytes per K-block).
fn pack_q8_weight_pair(
    q8_bytes: &[u8],
    base_r0: usize,
    base_r1: usize,
    blocks_per_row: usize,
    panel: &mut [i8],
    w_scales: &mut [f32],
) {
    for b in 0..blocks_per_row {
        let off0 = base_r0 + b * 34;
        let off1 = base_r1 + b * 34;
        w_scales[b * 2] = q8_0_block_scale(q8_bytes, off0);
        w_scales[b * 2 + 1] = q8_0_block_scale(q8_bytes, off1);
        for seg in 0..4 {
            let dst = b * 64 + seg * 16;
            let src0 = off0 + 2 + seg * 8;
            let src1 = off1 + 2 + seg * 8;
            for k in 0..8 {
                panel[dst + k] = q8_bytes[src0 + k] as i8;
                panel[dst + 8 + k] = q8_bytes[src1 + k] as i8;
            }
        }
    }
}

/// REEFUSE-Q8-I8MM-PANEL inner kernel: accumulate `output[t][out_feature..+2]`
/// for all batch rows `t` against one output-pair worth of packed weight, using
/// `smmla` + per-block per-row scale folded into a register-resident f32 output.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn smmla_accumulate_output_pair(
    weight_panel: &[i8],
    w_scales: &[f32], // 2 per block: [s_w0, s_w1, ...]
    act_panel: &[i8],
    act_i8: &[i8],
    act_scales: &[f32], // batch * blocks_per_row
    batch: usize,
    in_features: usize,
    blocks_per_row: usize,
    output: &mut [f32],
    out_features: usize,
    out_feature: usize,
) {
    let pairs = batch / 2;
    let act_pair_stride = 2 * in_features;
    for p in 0..pairs {
        let t0 = p * 2;
        let t1 = t0 + 1;
        let act_pair_base = act_panel.as_ptr().add(p * act_pair_stride);
        // Accumulate the 2x2 tile into scalars across the K-block loop. A
        // loop-carried `float32x4_t` across the inline `asm!` proved fragile (the
        // asm's vector clobbers collided with the carried accumulator register and
        // corrupted it); scalars are safe and the f32 accumulate cost is
        // negligible next to the smmla.
        let mut o00 = 0.0f32;
        let mut o01 = 0.0f32;
        let mut o10 = 0.0f32;
        let mut o11 = 0.0f32;
        for b in 0..blocks_per_row {
            let s_w0 = w_scales[b * 2];
            let s_w1 = w_scales[b * 2 + 1];
            let s_a0 = act_scales[t0 * blocks_per_row + b];
            let s_a1 = act_scales[t1 * blocks_per_row + b];
            let mut a_ptr = act_pair_base.add(b * 64);
            let mut w_ptr = weight_panel.as_ptr().add(b * 64);
            // Read the int32 tile through a proper `out(vreg)` operand (NOT via an
            // `st1` to a pointer passed as `in(reg)` — that hides the memory write
            // from the compiler and is UB, which manifested as an optimization-
            // dependent heisenbug). Convert to lanes with a NEON store in Rust.
            let tile_acc: int32x4_t;
            std::arch::asm!(
                "movi {acc:v}.4s, #0",
                "ld1 {{v0.16b}}, [{a}], #16",
                "ld1 {{v1.16b}}, [{w}], #16",
                "smmla {acc:v}.4s, v0.16b, v1.16b",
                "ld1 {{v0.16b}}, [{a}], #16",
                "ld1 {{v1.16b}}, [{w}], #16",
                "smmla {acc:v}.4s, v0.16b, v1.16b",
                "ld1 {{v0.16b}}, [{a}], #16",
                "ld1 {{v1.16b}}, [{w}], #16",
                "smmla {acc:v}.4s, v0.16b, v1.16b",
                "ld1 {{v0.16b}}, [{a}], #16",
                "ld1 {{v1.16b}}, [{w}], #16",
                "smmla {acc:v}.4s, v0.16b, v1.16b",
                acc = out(vreg) tile_acc,
                a = inout(reg) a_ptr,
                w = inout(reg) w_ptr,
                out("v0") _,
                out("v1") _,
            );
            let _ = (a_ptr, w_ptr);
            let mut tile = [0i32; 4];
            vst1q_s32(tile.as_mut_ptr(), tile_acc);
            // smmla lanes: [t0*w0, t0*w1, t1*w0, t1*w1]
            o00 += s_w0 * s_a0 * tile[0] as f32;
            o01 += s_w1 * s_a0 * tile[1] as f32;
            o10 += s_w0 * s_a1 * tile[2] as f32;
            o11 += s_w1 * s_a1 * tile[3] as f32;
        }
        output[t0 * out_features + out_feature] += o00;
        output[t0 * out_features + out_feature + 1] += o01;
        output[t1 * out_features + out_feature] += o10;
        output[t1 * out_features + out_feature + 1] += o11;
    }
    // Odd-batch tail: token row (batch-1) is not part of any pair. Compute its
    // contribution against this output-pair via a scalar int8 dot using the raw
    // act_i8 and the packed weight panel (which already has both row 0 and row 1
    // of the output-pair interleaved).
    if batch & 1 != 0 {
        let t = batch - 1;
        let mut o0 = 0.0f32;
        let mut o1 = 0.0f32;
        for b in 0..blocks_per_row {
            let s_w0 = w_scales[b * 2];
            let s_w1 = w_scales[b * 2 + 1];
            let s_a = act_scales[t * blocks_per_row + b];
            let mut d0 = 0i32;
            let mut d1 = 0i32;
            let w_base = weight_panel.as_ptr().add(b * 64);
            for seg in 0..4usize {
                for k in 0..8usize {
                    let a = act_i8[t * in_features + b * 32 + seg * 8 + k] as i32;
                    let w0 = *w_base.add(seg * 16 + k) as i32;
                    let w1 = *w_base.add(seg * 16 + 8 + k) as i32;
                    d0 += a * w0;
                    d1 += a * w1;
                }
            }
            o0 += s_w0 * s_a * d0 as f32;
            o1 += s_w1 * s_a * d1 as f32;
        }
        output[t * out_features + out_feature] += o0;
        output[t * out_features + out_feature + 1] += o1;
    }
}

/// REEFUSE-Q8-I8MM-PANEL output-octet kernel (R124): accumulate EIGHT output rows
/// (`out_feature..+8`) for all batch rows against four packed weight row-pairs,
/// using FOUR independent `smmla` accumulator tiles per K-block. The activation
/// `v0` is loaded once per K-segment and reused across all four weight panels.
/// Four independent chains hide the ~3-cycle `smmla` latency that the single-tile
/// `output_pair` stalls on (lab R123: ~1.46x). Per-block per-row weight scales are
/// folded in scalar post-`smmla`, identical to `smmla_accumulate_output_pair`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
#[allow(clippy::too_many_arguments)]
unsafe fn smmla_accumulate_output_octet(
    wp: &[&[i8]; 4],     // four weight panels, rows (0,1)(2,3)(4,5)(6,7)
    ws: &[&[f32]; 4],    // four scale arrays, 2 per block each
    act_panel: &[i8],
    act_i8: &[i8],
    act_scales: &[f32], // batch * blocks_per_row
    batch: usize,
    in_features: usize,
    blocks_per_row: usize,
    output: &mut [f32],
    out_features: usize,
    out_feature: usize,
) {
    let pairs = batch / 2;
    let act_pair_stride = 2 * in_features;
    for p in 0..pairs {
        let t0 = p * 2;
        let t1 = t0 + 1;
        let act_pair_base = act_panel.as_ptr().add(p * act_pair_stride);
        // acc[token_in_pair][out_row_in_octet]
        let mut acc0 = [0.0f32; 8];
        let mut acc1 = [0.0f32; 8];
        for b in 0..blocks_per_row {
            let mut a_ptr = act_pair_base.add(b * 64);
            let mut w0p = wp[0].as_ptr().add(b * 64);
            let mut w1p = wp[1].as_ptr().add(b * 64);
            let mut w2p = wp[2].as_ptr().add(b * 64);
            let mut w3p = wp[3].as_ptr().add(b * 64);
            let tile0: int32x4_t;
            let tile1: int32x4_t;
            let tile2: int32x4_t;
            let tile3: int32x4_t;
            // Four independent accumulator tiles (acc0..acc3); v0 = shared
            // activation, v1..v4 = the four weight panels. Read each tile through a
            // typed `out(vreg)` operand (never an `st1` via an `in(reg)` pointer —
            // that hides the write from the compiler and is UB; see R119).
            std::arch::asm!(
                "movi {a0:v}.4s, #0",
                "movi {a1:v}.4s, #0",
                "movi {a2:v}.4s, #0",
                "movi {a3:v}.4s, #0",
                "ld1 {{v0.16b}}, [{a}], #16",
                "ld1 {{v1.16b}}, [{w0}], #16",
                "ld1 {{v2.16b}}, [{w1}], #16",
                "ld1 {{v3.16b}}, [{w2}], #16",
                "ld1 {{v4.16b}}, [{w3}], #16",
                "smmla {a0:v}.4s, v0.16b, v1.16b",
                "smmla {a1:v}.4s, v0.16b, v2.16b",
                "smmla {a2:v}.4s, v0.16b, v3.16b",
                "smmla {a3:v}.4s, v0.16b, v4.16b",
                "ld1 {{v0.16b}}, [{a}], #16",
                "ld1 {{v1.16b}}, [{w0}], #16",
                "ld1 {{v2.16b}}, [{w1}], #16",
                "ld1 {{v3.16b}}, [{w2}], #16",
                "ld1 {{v4.16b}}, [{w3}], #16",
                "smmla {a0:v}.4s, v0.16b, v1.16b",
                "smmla {a1:v}.4s, v0.16b, v2.16b",
                "smmla {a2:v}.4s, v0.16b, v3.16b",
                "smmla {a3:v}.4s, v0.16b, v4.16b",
                "ld1 {{v0.16b}}, [{a}], #16",
                "ld1 {{v1.16b}}, [{w0}], #16",
                "ld1 {{v2.16b}}, [{w1}], #16",
                "ld1 {{v3.16b}}, [{w2}], #16",
                "ld1 {{v4.16b}}, [{w3}], #16",
                "smmla {a0:v}.4s, v0.16b, v1.16b",
                "smmla {a1:v}.4s, v0.16b, v2.16b",
                "smmla {a2:v}.4s, v0.16b, v3.16b",
                "smmla {a3:v}.4s, v0.16b, v4.16b",
                "ld1 {{v0.16b}}, [{a}], #16",
                "ld1 {{v1.16b}}, [{w0}], #16",
                "ld1 {{v2.16b}}, [{w1}], #16",
                "ld1 {{v3.16b}}, [{w2}], #16",
                "ld1 {{v4.16b}}, [{w3}], #16",
                "smmla {a0:v}.4s, v0.16b, v1.16b",
                "smmla {a1:v}.4s, v0.16b, v2.16b",
                "smmla {a2:v}.4s, v0.16b, v3.16b",
                "smmla {a3:v}.4s, v0.16b, v4.16b",
                a0 = out(vreg) tile0,
                a1 = out(vreg) tile1,
                a2 = out(vreg) tile2,
                a3 = out(vreg) tile3,
                a = inout(reg) a_ptr,
                w0 = inout(reg) w0p,
                w1 = inout(reg) w1p,
                w2 = inout(reg) w2p,
                w3 = inout(reg) w3p,
                out("v0") _,
                out("v1") _,
                out("v2") _,
                out("v3") _,
                out("v4") _,
            );
            let _ = (a_ptr, w0p, w1p, w2p, w3p);
            let s_a0 = act_scales[t0 * blocks_per_row + b];
            let s_a1 = act_scales[t1 * blocks_per_row + b];
            let tiles = [tile0, tile1, tile2, tile3];
            for (n, tile_acc) in tiles.into_iter().enumerate() {
                let mut tile = [0i32; 4];
                vst1q_s32(tile.as_mut_ptr(), tile_acc);
                let s_w0 = ws[n][b * 2];
                let s_w1 = ws[n][b * 2 + 1];
                // tile lanes: [t0*w0, t0*w1, t1*w0, t1*w1] for this panel's 2 rows.
                acc0[n * 2] += s_w0 * s_a0 * tile[0] as f32;
                acc0[n * 2 + 1] += s_w1 * s_a0 * tile[1] as f32;
                acc1[n * 2] += s_w0 * s_a1 * tile[2] as f32;
                acc1[n * 2 + 1] += s_w1 * s_a1 * tile[3] as f32;
            }
        }
        for orow in 0..8 {
            output[t0 * out_features + out_feature + orow] += acc0[orow];
            output[t1 * out_features + out_feature + orow] += acc1[orow];
        }
    }
    // Odd-batch tail: last token row, all 8 output rows, scalar int8 from the
    // packed weight panels (each panel holds two output rows interleaved).
    if batch & 1 != 0 {
        let t = batch - 1;
        for n in 0..4 {
            let mut o0 = 0.0f32;
            let mut o1 = 0.0f32;
            for b in 0..blocks_per_row {
                let s_w0 = ws[n][b * 2];
                let s_w1 = ws[n][b * 2 + 1];
                let s_a = act_scales[t * blocks_per_row + b];
                let w_base = wp[n].as_ptr().add(b * 64);
                let mut d0 = 0i32;
                let mut d1 = 0i32;
                for seg in 0..4usize {
                    for k in 0..8usize {
                        let a = act_i8[t * in_features + b * 32 + seg * 8 + k] as i32;
                        d0 += a * *w_base.add(seg * 16 + k) as i32;
                        d1 += a * *w_base.add(seg * 16 + 8 + k) as i32;
                    }
                }
                o0 += s_w0 * s_a * d0 as f32;
                o1 += s_w1 * s_a * d1 as f32;
            }
            output[t * out_features + out_feature + n * 2] += o0;
            output[t * out_features + out_feature + n * 2 + 1] += o1;
        }
    }
}

/// Scalar int8 dot for one weight row × all batch rows (handles odd-batch and
/// odd-output-row tails, partial chunks, and non-i8mm CPUs).
fn scalar_int8_row(
    q8_bytes: &[u8],
    base_r: usize,
    act_i8: &[i8],
    act_scales: &[f32],
    batch: usize,
    in_features: usize,
    blocks_per_row: usize,
    output: &mut [f32],
    out_features: usize,
    out_feature: usize,
) {
    for b in 0..blocks_per_row {
        let off = base_r + b * 34;
        let s_w = q8_0_block_scale(q8_bytes, off);
        let in_feat = b * 32;
        for row in 0..batch {
            let aoff = row * in_features + in_feat;
            let s_a = act_scales[row * blocks_per_row + b];
            let mut acc = 0i32;
            for k in 0..32 {
                acc += (q8_bytes[off + 2 + k] as i8 as i32) * (act_i8[aoff + k] as i32);
            }
            output[row * out_features + out_feature] += s_w * s_a * acc as f32;
        }
    }
}

/// Try the R118 packed-panel `smmla` fast path. Returns `Ok(true)` if the chunk
/// was fully processed via the panel kernel; `Ok(false)` means caller should fall
/// back to the existing path.
fn accumulate_q8_0_chunk_panel_smmla(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
) -> Result<bool> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (input, output, q8_bytes, element_start, config);
        return Ok(false);
    }
    #[cfg(target_arch = "aarch64")]
    {
        if !q8_i8mm_available() || config.batch < 2 || config.in_features < 32 {
            return Ok(false);
        }
        if !config.in_features.is_multiple_of(32) {
            return Ok(false);
        }
        if !element_start.is_multiple_of(config.in_features) {
            return Ok(false);
        }
        let blocks_per_row = config.in_features / 32;
        let q8_block_count = q8_bytes.len() / 34;
        if q8_block_count == 0 || !q8_block_count.is_multiple_of(blocks_per_row) {
            return Ok(false);
        }
        let n_rows = q8_block_count / blocks_per_row;
        let out_start = element_start / config.in_features;
        if out_start + n_rows > config.out_features {
            return Ok(false);
        }
        with_q8_panel_activations(input, config.batch, config.in_features, |act_i8, act_panel, act_scales| {
            // Octet (R124): 4 weight panels + 4 scale arrays, reused per octet.
            let mut wpv: [Vec<i8>; 4] =
                std::array::from_fn(|_| vec![0i8; 2 * config.in_features]);
            let mut wsv: [Vec<f32>; 4] =
                std::array::from_fn(|_| vec![0.0f32; 2 * blocks_per_row]);
            // Pair-remainder scratch.
            let mut weight_panel = vec![0i8; 2 * config.in_features];
            let mut w_scales = vec![0.0f32; 2 * blocks_per_row];
            let mut r = 0;
            // Output-octet ILP fast path (8 output rows, 4 independent smmla chains).
            while r + 8 <= n_rows {
                for q in 0..4 {
                    let base_r0 = (r + q * 2) * blocks_per_row * 34;
                    let base_r1 = (r + q * 2 + 1) * blocks_per_row * 34;
                    pack_q8_weight_pair(
                        q8_bytes,
                        base_r0,
                        base_r1,
                        blocks_per_row,
                        &mut wpv[q],
                        &mut wsv[q],
                    );
                }
                let wp = [
                    wpv[0].as_slice(),
                    wpv[1].as_slice(),
                    wpv[2].as_slice(),
                    wpv[3].as_slice(),
                ];
                let ws = [
                    wsv[0].as_slice(),
                    wsv[1].as_slice(),
                    wsv[2].as_slice(),
                    wsv[3].as_slice(),
                ];
                // SAFETY: q8_i8mm_available verified above.
                unsafe {
                    smmla_accumulate_output_octet(
                        &wp,
                        &ws,
                        act_panel,
                        act_i8,
                        act_scales,
                        config.batch,
                        config.in_features,
                        blocks_per_row,
                        output,
                        config.out_features,
                        out_start + r,
                    );
                }
                r += 8;
            }
            // Pair remainder (0..3 output pairs).
            while r + 2 <= n_rows {
                let base_r0 = r * blocks_per_row * 34;
                let base_r1 = (r + 1) * blocks_per_row * 34;
                pack_q8_weight_pair(
                    q8_bytes,
                    base_r0,
                    base_r1,
                    blocks_per_row,
                    &mut weight_panel,
                    &mut w_scales,
                );
                // SAFETY: q8_i8mm_available verified above.
                unsafe {
                    smmla_accumulate_output_pair(
                        &weight_panel,
                        &w_scales,
                        act_panel,
                        act_i8,
                        act_scales,
                        config.batch,
                        config.in_features,
                        blocks_per_row,
                        output,
                        config.out_features,
                        out_start + r,
                    );
                }
                r += 2;
            }
            // Odd last output row in chunk: scalar int8.
            if r < n_rows {
                let base_r = r * blocks_per_row * 34;
                scalar_int8_row(
                    q8_bytes,
                    base_r,
                    act_i8,
                    act_scales,
                    config.batch,
                    config.in_features,
                    blocks_per_row,
                    output,
                    config.out_features,
                    out_start + r,
                );
            }
            Ok::<(), RuntimeError>(())
        })?;
        Ok(true)
    }
}

/// Parity-validation int8-activation Q8 matmul. Accumulates exactly like the f32
/// path (`output[row][out_feature] += ...`) but uses int8×int8 dot with per-row
/// per-segment activation quantization. Boundary/partial blocks fall back to the
/// exact f32 reecast dot.
/// Exact per-segment int8-activation path (pre-R127). Used when `in_features` is
/// not a multiple of 32, where the block-quantization cache layout does not apply.
/// Re-quantizes each input segment per output row.
fn accumulate_q8_0_chunk_int8_activation_uncached(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_elements: usize,
) -> Result<()> {
    let q8_block_count = q8_bytes.len() / 34;
    for block_idx in 0..q8_block_count {
        let block_global_start = element_start + block_idx * 32;
        if block_global_start >= weight_elements {
            break;
        }
        let block_offset = block_idx * 34;
        let w_scale = q8_0_block_scale(q8_bytes, block_offset);
        let wq = &q8_bytes[block_offset + 2..block_offset + 34];
        let block_len = (weight_elements - block_global_start).min(32);
        let out_feature = block_global_start / config.in_features;
        let in_feature = block_global_start % config.in_features;

        if block_len == 32 && in_feature + 32 <= config.in_features {
            for row in 0..config.batch {
                let seg = &input[row * config.in_features + in_feature..][..32];
                let (aq, a_scale) = quantize_seg32_i8(seg);
                let dot = i8_dot32(wq, &aq);
                output[row * config.out_features + out_feature] += w_scale * a_scale * dot as f32;
            }
        } else {
            let scaled = q8_0_scaled_block_reecast(wq, w_scale);
            for row in 0..config.batch {
                let input_start = row * config.in_features + in_feature;
                let mut acc = 0.0f32;
                for k in 0..block_len {
                    acc += scaled[k] * input[input_start + k];
                }
                output[row * config.out_features + out_feature] += acc;
            }
        }
    }
    Ok(())
}

/// R130: batch=1 4-row ILP int8 GEMV core (promotes the R128b lab kernel). One
/// asm block per K-block processes 4 output rows with 4 INDEPENDENT sdot
/// accumulator chains and a single shared activation-block load (v0/v1); the four
/// chains pipeline to hide sdot latency. Reduce each tile in Rust (`vaddvq_s32`),
/// apply per-block per-row weight scale. Bit-identical to the per-row path (same
/// int32 dots, same block-order f32 accumulation). Lab: ~1.6x over per-row sdot;
/// the interleaved repack (R129) was tested and lost to this (fewer ILP chains).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn batch1_x4_ilp(
    q8_bytes: &[u8],
    row_base: usize,
    act_i8: &[i8],
    act_scales: &[f32],
    blocks_per_row: usize,
) -> [f32; 4] {
    let row_stride = blocks_per_row * 34;
    let mut acc = [0.0f32; 4];
    for b in 0..blocks_per_row {
        let a_ptr = act_i8.as_ptr().add(b * 32);
        let off0 = row_base + b * 34;
        let off1 = row_base + row_stride + b * 34;
        let off2 = row_base + 2 * row_stride + b * 34;
        let off3 = row_base + 3 * row_stride + b * 34;
        let w0 = q8_bytes.as_ptr().add(off0 + 2);
        let w1 = q8_bytes.as_ptr().add(off1 + 2);
        let w2 = q8_bytes.as_ptr().add(off2 + 2);
        let w3 = q8_bytes.as_ptr().add(off3 + 2);
        let t0: int32x4_t;
        let t1: int32x4_t;
        let t2: int32x4_t;
        let t3: int32x4_t;
        std::arch::asm!(
            "movi {a0:v}.4s, #0",
            "movi {a1:v}.4s, #0",
            "movi {a2:v}.4s, #0",
            "movi {a3:v}.4s, #0",
            "ld1 {{v0.16b, v1.16b}}, [{a}]",
            "ld1 {{v2.16b, v3.16b}}, [{w0}]",
            "sdot {a0:v}.4s, v2.16b, v0.16b",
            "sdot {a0:v}.4s, v3.16b, v1.16b",
            "ld1 {{v4.16b, v5.16b}}, [{w1}]",
            "sdot {a1:v}.4s, v4.16b, v0.16b",
            "sdot {a1:v}.4s, v5.16b, v1.16b",
            "ld1 {{v6.16b, v7.16b}}, [{w2}]",
            "sdot {a2:v}.4s, v6.16b, v0.16b",
            "sdot {a2:v}.4s, v7.16b, v1.16b",
            "ld1 {{v16.16b, v17.16b}}, [{w3}]",
            "sdot {a3:v}.4s, v16.16b, v0.16b",
            "sdot {a3:v}.4s, v17.16b, v1.16b",
            a0 = out(vreg) t0,
            a1 = out(vreg) t1,
            a2 = out(vreg) t2,
            a3 = out(vreg) t3,
            a = in(reg) a_ptr,
            w0 = in(reg) w0,
            w1 = in(reg) w1,
            w2 = in(reg) w2,
            w3 = in(reg) w3,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v16") _, out("v17") _,
        );
        let s_a = act_scales[b];
        acc[0] += q8_0_block_scale(q8_bytes, off0) * s_a * vaddvq_s32(t0) as f32;
        acc[1] += q8_0_block_scale(q8_bytes, off1) * s_a * vaddvq_s32(t1) as f32;
        acc[2] += q8_0_block_scale(q8_bytes, off2) * s_a * vaddvq_s32(t2) as f32;
        acc[3] += q8_0_block_scale(q8_bytes, off3) * s_a * vaddvq_s32(t3) as f32;
    }
    acc
}

/// R128: batch=1 row-major int8 GEMV (decode fast path). For each output row,
/// accumulate the per-block scaled int8 dot into a register across ALL in-blocks
/// and write `output` ONCE — vs the block-major path's per-block read-modify-write
/// (524288 output writes per gate matmul → 8192). The f32 accumulation order is
/// identical to the block-major path (a row's blocks are contiguous in the q8
/// chunk and processed in the same b=0..blocks_per_row order), so the result is
/// bit-for-bit identical. Activation is quantized once (R127 cache).
/// Caller guarantees batch==1, in_features % 32 == 0, and a row-aligned chunk.
fn accumulate_q8_0_chunk_int8_batch1_rowmajor(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_elements: usize,
    blocks_per_row: usize,
    q8_block_count: usize,
) -> Result<()> {
    let n_rows = q8_block_count / blocks_per_row;
    let out_start = element_start / config.in_features;
    with_q8_panel_activations(input, 1, config.in_features, |act_i8, _panel, act_scales| {
        let mut r = 0usize;
        // R130: 4 output rows per pass via the ILP sdot kernel (dotprod CPUs).
        #[cfg(target_arch = "aarch64")]
        {
            if q8_sdot_available() {
                while r + 4 <= n_rows {
                    let out_feature = out_start + r;
                    if (out_feature + 3) * config.in_features >= weight_elements {
                        break;
                    }
                    let row_base = r * blocks_per_row * 34;
                    // SAFETY: q8_sdot_available() verified dotprod above.
                    let acc = unsafe {
                        batch1_x4_ilp(q8_bytes, row_base, act_i8, act_scales, blocks_per_row)
                    };
                    output[out_feature] += acc[0];
                    output[out_feature + 1] += acc[1];
                    output[out_feature + 2] += acc[2];
                    output[out_feature + 3] += acc[3];
                    r += 4;
                }
            }
        }
        // Remainder rows (and non-dotprod fallback): per-row register accumulate.
        // `r` MUST advance here — otherwise any chunk whose row count is not a
        // multiple of 4 (so the x4 loop above leaves `r < n_rows`) spins forever.
        // Latent until a non-÷4 row-aligned chunk hits it (e.g. Gemma q_proj's
        // 385-row chunk: 385 % 4 == 1).
        while r < n_rows {
            let out_feature = out_start + r;
            if out_feature * config.in_features >= weight_elements {
                break;
            }
            let row_block_base = r * blocks_per_row * 34;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let block_offset = row_block_base + b * 34;
                let w_scale = q8_0_block_scale(q8_bytes, block_offset);
                let wq = &q8_bytes[block_offset + 2..block_offset + 34];
                let a_seg: &[i8; 32] = act_i8[b * 32..b * 32 + 32]
                    .try_into()
                    .expect("32-element activation segment");
                let dot = i8_dot32(wq, a_seg);
                acc += w_scale * act_scales[b] * dot as f32;
            }
            output[out_feature] += acc;
            r += 1;
        }
        Ok::<(), RuntimeError>(())
    })
}

/// R133: int8×int8 sdot GEMV over a contiguous row range of a whole Q8_0 tensor.
/// `out_slice` is the output for rows `[base_row, base_row + out_slice.len())`;
/// `q8_bytes` is the WHOLE tensor (row-major, `blocks_per_row` 32-blocks per row);
/// `act_i8`/`act_scales` are the activation quantized ONCE by the caller. 4-row
/// ILP via `batch1_x4_ilp` on dotprod CPUs, per-row `i8_dot32` remainder/fallback.
fn sdot_int8_batch1_rows_range(
    out_slice: &mut [f32],
    q8_bytes: &[u8],
    act_i8: &[i8],
    act_scales: &[f32],
    base_row: usize,
    blocks_per_row: usize,
) {
    let n = out_slice.len();
    let mut r = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        if q8_sdot_available() {
            while r + 4 <= n {
                let row_base = (base_row + r) * blocks_per_row * 34;
                // SAFETY: q8_sdot_available() verified dotprod.
                let acc = unsafe { batch1_x4_ilp(q8_bytes, row_base, act_i8, act_scales, blocks_per_row) };
                out_slice[r] += acc[0];
                out_slice[r + 1] += acc[1];
                out_slice[r + 2] += acc[2];
                out_slice[r + 3] += acc[3];
                r += 4;
            }
        }
    }
    while r < n {
        let row_base = (base_row + r) * blocks_per_row * 34;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_base + b * 34;
            let w_scale = q8_0_block_scale(q8_bytes, off);
            let wq = &q8_bytes[off + 2..off + 34];
            let a_seg: &[i8; 32] = act_i8[b * 32..b * 32 + 32]
                .try_into()
                .expect("32-element activation segment");
            acc += w_scale * act_scales[b] * i8_dot32(wq, a_seg) as f32;
        }
        out_slice[r] += acc;
        r += 1;
    }
}

/// R133 decode fast-path: int8 sdot GEMV over a WHOLE contiguous Q8_0 tensor
/// (batch=1), parallel across output rows. The activation is quantized to int8
/// ONCE (ggml-style: quantize-once → parallel rows), then each worker owns a
/// disjoint output-row range. Replaces the per-chunk dispatch + scalar i8×f32
/// path for raw-codec q8 decode. Near-exact (int8 activation, quant-only diff —
/// same as llama.cpp q8 inference). `q8_bytes` MUST be the full row-major tensor.
pub(crate) fn accumulate_q8_0_full_tensor_int8_batch1(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    config: StreamingLinearConfig,
) -> Result<()> {
    let in_features = config.in_features;
    let out_features = config.out_features;
    if config.batch != 1 || !in_features.is_multiple_of(32) || in_features == 0 {
        return Err(RuntimeError::Shape(
            "q8 full-tensor fast-path requires batch=1 and in_features%32==0".to_string(),
        ));
    }
    let blocks_per_row = in_features / 32;
    let expected_bytes = out_features
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(34))
        .ok_or_else(|| RuntimeError::Shape("q8 full-tensor size overflow".to_string()))?;
    if q8_bytes.len() != expected_bytes {
        return Err(RuntimeError::Shape(format!(
            "q8 full-tensor byte len {} != expected {expected_bytes} (out={out_features}, bpr={blocks_per_row})",
            q8_bytes.len()
        )));
    }
    if output.len() != out_features {
        return Err(RuntimeError::Shape(format!(
            "q8 full-tensor output len {} != out_features {out_features}",
            output.len()
        )));
    }

    // Quantize the activation row to int8 ONCE for the whole matmul.
    let (act_i8, act_scales) = quantize_input_q8_blocks(input, 1, in_features);

    let threads = effective_runtime_threads(
        std::env::var(RLLM_THREADS_ENV).ok().as_deref(),
        available_runtime_threads(),
    );
    if threads <= 1 || out_features < 2 * MIN_ROWS_PER_PARALLEL_Q8_PREFILL {
        sdot_int8_batch1_rows_range(output, q8_bytes, &act_i8, &act_scales, 0, blocks_per_row);
        return Ok(());
    }

    let workers = threads
        .min(out_features / MIN_ROWS_PER_PARALLEL_Q8_PREFILL)
        .max(1);
    let rows_per_worker = out_features.div_ceil(workers);
    std::thread::scope(|scope| {
        let act_i8 = &act_i8;
        let act_scales = &act_scales;
        let mut rest = &mut output[..];
        let mut base_row = 0usize;
        while base_row < out_features {
            let rows = rows_per_worker.min(out_features - base_row);
            let (chunk, tail) = rest.split_at_mut(rows);
            rest = tail;
            let worker_base = base_row;
            scope.spawn(move || {
                sdot_int8_batch1_rows_range(
                    chunk,
                    q8_bytes,
                    act_i8,
                    act_scales,
                    worker_base,
                    blocks_per_row,
                );
            });
            base_row += rows;
        }
    });
    Ok(())
}

/// R138 prefill fast-path: whole-tensor Q8_0 panel matmul for batch>=2,
/// parallelized across OUTPUT ROWS once per projection (not once per chunk).
///
/// Decode got the whole-tensor treatment in R133; prefill needs the same. The
/// per-chunk path spawned worker threads for EVERY chunk (~238/token), so for a
/// short prompt the thread-spawn overhead beat the work. Splitting by BATCH rows
/// is also wrong: each worker would re-read the WHOLE weight tensor for its few
/// batch rows, tripling weight bandwidth and defeating the point of batching.
///
/// So split by OUTPUT ROWS: each worker reads a DISJOINT slice of weight rows
/// ONCE and computes all batch tokens for them, keeping batch whole (panel stays
/// engaged) and weights read once total. The output is `[batch, out_features]`
/// (batch-major), so a worker's output rows are strided across batch — to keep
/// the parallel section sound (no aliased `&mut`), each worker writes a local
/// `[batch, rows]` buffer and we scatter it into the final output single-threaded
/// afterward (a cheap `out_features*batch` copy).
pub(crate) fn accumulate_q8_0_full_tensor_panel_batch(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    let in_features = config.in_features;
    let out_features = config.out_features;
    let batch = config.batch;
    if batch < 2 || !in_features.is_multiple_of(32) || in_features == 0 {
        return Err(RuntimeError::Shape(
            "q8 panel prefill fast-path requires batch>=2 and in_features%32==0".to_string(),
        ));
    }
    let blocks_per_row = in_features / 32;
    let row_bytes = blocks_per_row * 34;
    let expected_bytes = out_features
        .checked_mul(row_bytes)
        .ok_or_else(|| RuntimeError::Shape("q8 panel prefill size overflow".to_string()))?;
    if q8_bytes.len() != expected_bytes {
        return Err(RuntimeError::Shape(format!(
            "q8 panel prefill byte len {} != expected {expected_bytes} (out={out_features}, bpr={blocks_per_row})",
            q8_bytes.len()
        )));
    }
    if input.len() != batch * in_features || output.len() != batch * out_features {
        return Err(RuntimeError::Shape(
            "q8 panel prefill input/output shape mismatch".to_string(),
        ));
    }

    let threads = effective_runtime_threads(
        std::env::var(RLLM_THREADS_ENV).ok().as_deref(),
        available_runtime_threads(),
    );
    // Enough output rows per worker to amortize the spawn + scatter-merge.
    const MIN_OUT_ROWS_PER_WORKER: usize = 16;
    let workers = threads.min(out_features / MIN_OUT_ROWS_PER_WORKER).max(1);
    if workers <= 1 {
        return accumulate_q8_0_chunk(input, output, q8_bytes, 0, config, weight_name);
    }
    let rows_per_worker = out_features.div_ceil(workers);

    // Each worker computes a disjoint output-row slice into a local [batch, rows]
    // buffer, reading only its weight rows. Returns (row_start, rows, buffer).
    type WorkerOut = Result<(usize, usize, Vec<f32>)>;
    let mut results: Vec<WorkerOut> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut row_start = 0usize;
        while row_start < out_features {
            let rows = rows_per_worker.min(out_features - row_start);
            let w_bytes = &q8_bytes[row_start * row_bytes..(row_start + rows) * row_bytes];
            let worker_config = StreamingLinearConfig { batch, in_features, out_features: rows };
            handles.push(scope.spawn(move || -> WorkerOut {
                let mut local = vec![0.0f32; batch * rows];
                accumulate_q8_0_chunk(input, &mut local, w_bytes, 0, worker_config, weight_name)?;
                Ok((row_start, rows, local))
            }));
            row_start += rows;
        }
        for handle in handles {
            results.push(handle.join().unwrap_or_else(|_| {
                Err(RuntimeError::Shape(
                    "R138 panel prefill worker panicked".to_string(),
                ))
            }));
        }
    });

    // Scatter each worker's local [batch, rows] buffer into the [batch, out_features]
    // output at its column range (single-threaded; disjoint columns).
    for result in results {
        let (row_start, rows, local) = result?;
        for b in 0..batch {
            let src = &local[b * rows..b * rows + rows];
            let dst_off = b * out_features + row_start;
            output[dst_off..dst_off + rows].copy_from_slice(src);
        }
    }
    Ok(())
}

fn accumulate_q8_0_chunk_int8_activation(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_elements: usize,
) -> Result<()> {
    // R127: the block-cache (`quantize_input_q8_blocks`) only quantizes the first
    // `blocks_per_row * 32` elements of each row. When `in_features` is NOT a
    // multiple of 32, `in_feature` can be non-32-aligned and the fast-path segment
    // can run past the cached region, so the cache layout would be wrong. Real
    // transformer dims are always multiples of 32; the non-aligned case falls back
    // to the exact per-segment path.
    if !config.in_features.is_multiple_of(32) {
        return accumulate_q8_0_chunk_int8_activation_uncached(
            input,
            output,
            q8_bytes,
            element_start,
            config,
            weight_elements,
        );
    }
    let q8_block_count = q8_bytes.len() / 34;
    let blocks_per_row = config.in_features / 32;
    // R128: batch=1 decode on a row-aligned chunk uses the row-major
    // register-accumulating GEMV (write output once per row, not per block).
    if config.batch == 1
        && blocks_per_row != 0
        && element_start.is_multiple_of(config.in_features)
        && q8_block_count.is_multiple_of(blocks_per_row)
    {
        return accumulate_q8_0_chunk_int8_batch1_rowmajor(
            input,
            output,
            q8_bytes,
            element_start,
            config,
            weight_elements,
            blocks_per_row,
            q8_block_count,
        );
    }
    // R127: quantize the activation ONCE per matmul (cached thread-local by
    // pointer+fingerprint), then look the per-(row,block) int8 segment + scale up
    // in the inner loop instead of re-quantizing `input` for every output row.
    // `quantize_input_q8_blocks` uses the identical absmax/round/clamp as
    // `quantize_seg32_i8`, so the result is bit-for-bit unchanged. The previous
    // code re-quantized each input segment `out_features` times per chunk; for
    // batch=1 decode (gate: 8192 output rows) that was an 8192x redundancy.
    with_q8_panel_activations(input, config.batch, config.in_features, |act_i8, _panel, act_scales| {
        for block_idx in 0..q8_block_count {
            let block_global_start = element_start + block_idx * 32;
            if block_global_start >= weight_elements {
                break;
            }
            let block_offset = block_idx * 34;
            let w_scale = q8_0_block_scale(q8_bytes, block_offset);
            let wq = &q8_bytes[block_offset + 2..block_offset + 34];
            let block_len = (weight_elements - block_global_start).min(32);
            let out_feature = block_global_start / config.in_features;
            let in_feature = block_global_start % config.in_features;

            if block_len == 32 && in_feature + 32 <= config.in_features {
                let block_in_row = in_feature / 32;
                for row in 0..config.batch {
                    let seg_start = row * config.in_features + in_feature;
                    // SAFETY of unwrap: in_feature + 32 <= in_features guarantees
                    // seg_start + 32 <= (row + 1) * in_features <= act_i8.len().
                    let aq: &[i8; 32] = act_i8[seg_start..seg_start + 32]
                        .try_into()
                        .expect("32-element activation segment");
                    let a_scale = act_scales[row * blocks_per_row + block_in_row];
                    let dot = i8_dot32(wq, aq);
                    output[row * config.out_features + out_feature] +=
                        w_scale * a_scale * dot as f32;
                }
            } else {
                // Partial / boundary block: exact f32 path on the raw input.
                let scaled = q8_0_scaled_block_reecast(wq, w_scale);
                for row in 0..config.batch {
                    let input_start = row * config.in_features + in_feature;
                    let mut acc = 0.0f32;
                    for k in 0..block_len {
                        acc += scaled[k] * input[input_start + k];
                    }
                    output[row * config.out_features + out_feature] += acc;
                }
            }
        }
        Ok::<(), RuntimeError>(())
    })
}

/// Minimum batch rows per worker before the Q8 prefill matmul is parallelized.
///
/// Kept at 4 (threshold `batch >= 8`): below this, the by-batch split spawns
/// threads PER CHUNK (~238 chunks/token) with too little work each, so the
/// thread-spawn overhead makes it SLOWER than single-threaded (measured: batch=6
/// went 1249ms -> 2189ms at threads=8 when this was lowered to 2). Parallelizing
/// short-prompt prefill needs a whole-tensor row-parallel path (R133-style, one
/// spawn per projection), not a lower threshold here.
const MIN_ROWS_PER_PARALLEL_Q8_PREFILL: usize = 4;

/// REEWEAVE-Q8-PREFILL: parallelize one already-decoded Q8 chunk across CPU cores
/// by splitting the batch (prompt token) rows. Each worker owns a contiguous
/// output row-slice (`split_at_mut`) and a contiguous input row-slice, shares the
/// decoded weight bytes read-only, then runs the existing per-row-range kernel.
/// Only engages for batch>1 (prefill); batch1 decode falls through to sequential.
fn accumulate_q8_0_chunk_parallel(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    let threads = effective_runtime_threads(
        std::env::var(RLLM_THREADS_ENV).ok().as_deref(),
        available_runtime_threads(),
    );
    if threads <= 1 || config.batch < 2 * MIN_ROWS_PER_PARALLEL_Q8_PREFILL {
        return accumulate_q8_0_chunk(input, output, q8_bytes, element_start, config, weight_name);
    }
    let workers = threads
        .min(config.batch / MIN_ROWS_PER_PARALLEL_Q8_PREFILL)
        .max(1);
    let rows_per_worker = config.batch.div_ceil(workers);

    let mut results: Vec<Result<()>> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut out_rest = &mut output[..];
        let mut row_start = 0usize;
        while row_start < config.batch {
            let rows = rows_per_worker.min(config.batch - row_start);
            let in_slice =
                &input[row_start * config.in_features..(row_start + rows) * config.in_features];
            let (out_slice, rest) = out_rest.split_at_mut(rows * config.out_features);
            out_rest = rest;
            let mut worker_config = config;
            worker_config.batch = rows;
            handles.push(scope.spawn(move || {
                accumulate_q8_0_chunk(
                    in_slice,
                    out_slice,
                    q8_bytes,
                    element_start,
                    worker_config,
                    weight_name,
                )
            }));
            row_start += rows;
        }
        for handle in handles {
            results.push(handle.join().unwrap_or_else(|_| {
                Err(RuntimeError::Shape(
                    "parallel Q8 prefill worker panicked".to_string(),
                ))
            }));
        }
    });
    for result in results {
        result?;
    }
    Ok(())
}

fn accumulate_q8_0_chunk(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    let profile_enabled = q8_kernel_profile_enabled();
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    validate_q8_0_chunk(q8_bytes, element_start, weight_elements, weight_name)?;
    if q8_activation_path_enabled() {
        // Diagnostic: RLLM_Q8_PANEL=0 disables R119 panel path; useful to confirm
        // the existing R111 parity baseline still holds and isolate panel bugs.
        let panel_enabled = std::env::var("RLLM_Q8_PANEL")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            != Some("0".to_string());
        if panel_enabled
            && accumulate_q8_0_chunk_panel_smmla(input, output, q8_bytes, element_start, config)?
        {
            return Ok(());
        }
        return accumulate_q8_0_chunk_int8_activation(
            input,
            output,
            q8_bytes,
            element_start,
            config,
            weight_elements,
        );
    }
    let mut normal_scale_elapsed = std::time::Duration::ZERO;
    let mut normal_scale_calls = 0u64;
    let mut normal_batch4_elapsed = std::time::Duration::ZERO;
    let mut normal_batch4_calls = 0u64;
    let mut normal_batch4_items = 0u64;
    let mut normal_batch4_setup_elapsed = std::time::Duration::ZERO;
    let mut normal_batch4_setup_calls = 0u64;
    let mut normal_batch4_kernel_elapsed = std::time::Duration::ZERO;
    let mut normal_batch4_kernel_calls = 0u64;
    let mut normal_batch4_kernel_items = 0u64;
    let mut normal_output2_batch4_elapsed = std::time::Duration::ZERO;
    let mut normal_output2_batch4_calls = 0u64;
    let mut normal_output2_batch4_items = 0u64;
    let mut normal_tail_elapsed = std::time::Duration::ZERO;
    let mut normal_tail_calls = 0u64;
    let mut normal_tail_items = 0u64;
    if accumulate_q8_0_chunk_batch1_complete_rows(
        input,
        output,
        q8_bytes,
        element_start,
        config,
        weight_name,
    )? {
        return Ok(());
    }

    let q8_block_count = q8_bytes.len() / 34;
    let mut consumed_as_output2_second = vec![false; q8_block_count];
    for block_idx in 0..q8_block_count {
        if consumed_as_output2_second[block_idx] {
            continue;
        }
        let block_global_start = element_start + block_idx * 32;
        if block_global_start >= weight_elements {
            break;
        }
        let block_offset = block_idx * 34;
        let scale = q8_0_block_scale(q8_bytes, block_offset);
        let qs = &q8_bytes[block_offset + 2..block_offset + 34];
        let block_len = (weight_elements - block_global_start).min(32);
        let out_feature = block_global_start / config.in_features;
        let in_feature = block_global_start % config.in_features;

        if config.batch > 1 && block_len == 32 && in_feature + block_len <= config.in_features {
            if let Some((first_out_feature, second_block_idx)) = q8_output2_pair_offset(
                block_idx,
                q8_block_count,
                element_start,
                weight_elements,
                config,
            ) {
                if second_block_idx != block_idx {
                    let profile_start = profile_enabled.then(Instant::now);
                    let scale_start = profile_enabled.then(Instant::now);
                    let second_offset = second_block_idx * 34;
                    let second_scale = q8_0_block_scale(q8_bytes, second_offset);
                    let second_qs = &q8_bytes[second_offset + 2..second_offset + 34];
                    let first_scaled = q8_0_scaled_block_reecast(qs, scale);
                    let second_scaled = q8_0_scaled_block_reecast(second_qs, second_scale);
                    consumed_as_output2_second[second_block_idx] = true;
                    if let Some(scale_start) = scale_start {
                        normal_scale_elapsed += scale_start.elapsed();
                        normal_scale_calls += 2;
                    }

                    let mut batch_idx = 0usize;
                    let batch4_start_idx = batch_idx;
                    let batch4_start = profile_enabled.then(Instant::now);
                    while batch_idx + 4 <= config.batch {
                        let input_start = batch_idx * config.in_features + in_feature;
                        let output_start = batch_idx * config.out_features;
                        accumulate_f32_dot_32_output2_batch4_reebundle(
                            &first_scaled,
                            &second_scaled,
                            &input[input_start..],
                            config.in_features,
                            &mut output[output_start..],
                            config.out_features,
                            first_out_feature,
                        );
                        batch_idx += 4;
                    }
                    if let Some(batch4_start) = batch4_start {
                        let calls = ((batch_idx - batch4_start_idx) / 4) as u64;
                        normal_output2_batch4_elapsed += batch4_start.elapsed();
                        normal_output2_batch4_calls += calls;
                        normal_output2_batch4_items += calls * 4;
                    }

                    let tail_start_idx = batch_idx;
                    let tail_start = profile_enabled.then(Instant::now);
                    while batch_idx < config.batch {
                        let input_start = batch_idx * config.in_features + in_feature;
                        let output_start = batch_idx * config.out_features;
                        output[output_start + first_out_feature] +=
                            f32_dot_32(&first_scaled, &input[input_start..]);
                        output[output_start + first_out_feature + 1] +=
                            f32_dot_32(&second_scaled, &input[input_start..]);
                        batch_idx += 1;
                    }
                    if let Some(tail_start) = tail_start {
                        let calls = (batch_idx - tail_start_idx) as u64;
                        normal_tail_elapsed += tail_start.elapsed();
                        normal_tail_calls += calls * 2;
                        normal_tail_items += calls * 2;
                    }
                    if let Some(profile_start) = profile_start {
                        record_q8_kernel_path(
                            Q8KernelPath::BatchGt1Scaled,
                            2,
                            2,
                            0,
                            (config.batch * 2) as u64,
                            profile_start.elapsed(),
                        );
                    }
                    continue;
                }
            }

            let profile_start = profile_enabled.then(Instant::now);
            let scale_start = profile_enabled.then(Instant::now);
            let scaled = q8_0_scaled_block_reecast(qs, scale);
            if let Some(scale_start) = scale_start {
                normal_scale_elapsed += scale_start.elapsed();
                normal_scale_calls += 1;
            }
            let mut batch_idx = 0usize;
            let batch4_start_idx = batch_idx;
            let batch4_start = profile_enabled.then(Instant::now);
            while batch_idx + 4 <= config.batch {
                let setup_start = profile_enabled.then(Instant::now);
                let input_start = batch_idx * config.in_features + in_feature;
                let output_start = batch_idx * config.out_features;
                if let Some(setup_start) = setup_start {
                    normal_batch4_setup_elapsed += setup_start.elapsed();
                    normal_batch4_setup_calls += 1;
                }

                let kernel_start = profile_enabled.then(Instant::now);
                accumulate_f32_dot_32_batch4_reevec(
                    &scaled,
                    &input[input_start..],
                    config.in_features,
                    &mut output[output_start..],
                    config.out_features,
                    out_feature,
                );
                if let Some(kernel_start) = kernel_start {
                    normal_batch4_kernel_elapsed += kernel_start.elapsed();
                    normal_batch4_kernel_calls += 1;
                    normal_batch4_kernel_items += 4;
                }
                batch_idx += 4;
            }
            if let Some(batch4_start) = batch4_start {
                let calls = ((batch_idx - batch4_start_idx) / 4) as u64;
                normal_batch4_elapsed += batch4_start.elapsed();
                normal_batch4_calls += calls;
                normal_batch4_items += calls * 4;
            }
            let tail_start_idx = batch_idx;
            let tail_start = profile_enabled.then(Instant::now);
            while batch_idx < config.batch {
                let input_start = batch_idx * config.in_features + in_feature;
                let output_idx = batch_idx * config.out_features + out_feature;
                output[output_idx] += f32_dot_32(&scaled, &input[input_start..]);
                batch_idx += 1;
            }
            if let Some(tail_start) = tail_start {
                let calls = (batch_idx - tail_start_idx) as u64;
                normal_tail_elapsed += tail_start.elapsed();
                normal_tail_calls += calls;
                normal_tail_items += calls;
            }
            if let Some(profile_start) = profile_start {
                record_q8_kernel_path(
                    Q8KernelPath::BatchGt1Scaled,
                    1,
                    1,
                    0,
                    config.batch as u64,
                    profile_start.elapsed(),
                );
            }
        } else if in_feature + block_len <= config.in_features {
            let profile_start = profile_enabled.then(Instant::now);
            for batch_idx in 0..config.batch {
                let input_start = batch_idx * config.in_features + in_feature;
                let output_idx = batch_idx * config.out_features + out_feature;
                output[output_idx] += scale * q8_0_dot_i8_f32(qs, &input[input_start..], block_len);
            }
            if let Some(profile_start) = profile_start {
                record_q8_kernel_path(
                    Q8KernelPath::ContiguousI8Dot,
                    1,
                    1,
                    0,
                    config.batch as u64,
                    profile_start.elapsed(),
                );
            }
        } else {
            let profile_start = profile_enabled.then(Instant::now);
            for (idx, &q) in qs.iter().take(block_len).enumerate() {
                let global_idx = block_global_start + idx;
                let out_feature = global_idx / config.in_features;
                let in_feature = global_idx % config.in_features;
                let weight = scale * (q as i8) as f32;
                for batch_idx in 0..config.batch {
                    output[batch_idx * config.out_features + out_feature] +=
                        input[batch_idx * config.in_features + in_feature] * weight;
                }
            }
            if let Some(profile_start) = profile_start {
                record_q8_kernel_path(
                    Q8KernelPath::SplitRowScalar,
                    1,
                    1,
                    0,
                    config.batch as u64,
                    profile_start.elapsed(),
                );
            }
        }
    }

    if profile_enabled {
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1NormalScale,
            normal_scale_calls,
            normal_scale_calls,
            0,
            0,
            normal_scale_elapsed,
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1NormalBatch4,
            normal_batch4_calls,
            normal_batch4_calls,
            0,
            normal_batch4_items,
            normal_batch4_elapsed,
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1NormalBatch4Setup,
            normal_batch4_setup_calls,
            normal_batch4_setup_calls,
            0,
            0,
            normal_batch4_setup_elapsed,
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1NormalBatch4Kernel,
            normal_batch4_kernel_calls,
            normal_batch4_kernel_calls,
            0,
            normal_batch4_kernel_items,
            normal_batch4_kernel_elapsed,
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1NormalOutput2Batch4,
            normal_output2_batch4_calls,
            normal_output2_batch4_calls,
            normal_output2_batch4_calls * 2,
            normal_output2_batch4_items,
            normal_output2_batch4_elapsed,
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1NormalTail,
            normal_tail_calls,
            normal_tail_calls,
            0,
            normal_tail_items,
            normal_tail_elapsed,
        );
    }

    Ok(())
}

fn q8_output2_pair_offset(
    block_idx: usize,
    q8_block_count: usize,
    element_start: usize,
    weight_elements: usize,
    config: StreamingLinearConfig,
) -> Option<(usize, usize)> {
    let block_global_start = element_start.checked_add(block_idx.checked_mul(32)?)?;
    let out_feature = block_global_start / config.in_features;
    let in_feature = block_global_start % config.in_features;
    if in_feature + 32 > config.in_features {
        return None;
    }
    if out_feature + 1 >= config.out_features {
        return None;
    }
    let next_global_start = (out_feature + 1)
        .checked_mul(config.in_features)?
        .checked_add(in_feature)?;
    if next_global_start + 32 > weight_elements || next_global_start < element_start {
        return None;
    }
    let next_delta = next_global_start - element_start;
    if next_delta % 32 != 0 {
        return None;
    }
    let next_block_idx = next_delta / 32;
    if next_block_idx >= q8_block_count {
        return None;
    }
    Some((out_feature, next_block_idx))
}

fn accumulate_q8_0_chunk_batch1_complete_rows(
    input: &[f32],
    output: &mut [f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<bool> {
    let profile_enabled = q8_kernel_profile_enabled();
    let Some((first_row, row_count, blocks_per_row)) =
        q8_0_complete_row_span(q8_bytes, element_start, config)?
    else {
        return Ok(false);
    };
    let row_end = first_row
        .checked_add(row_count)
        .ok_or_else(|| RuntimeError::Shape("Q8_0 row fast path row range overflow".to_string()))?;
    if row_end > config.out_features {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} Q8_0 row fast path rows {first_row}..{row_end} exceed expected {}",
            config.out_features
        )));
    }

    let profile_start = profile_enabled.then(Instant::now);
    // R132: each output row is an independent dot product, so split the row
    // range across worker threads. Bit-identical to the serial loop (each row
    // keeps the same block-order f32 accumulation); only row ownership changes.
    // Arch-neutral (`std::thread`), so it lifts Intel/x86 batch=1 decode — which
    // otherwise ran single-threaded scalar — as well as ARM.
    q8_0_batch1_complete_rows_parallel(
        input,
        &mut output[first_row..row_end],
        q8_bytes,
        blocks_per_row,
    );
    if let Some(profile_start) = profile_start {
        record_q8_kernel_path(
            Q8KernelPath::Batch1CompleteLinear,
            1,
            (row_count * blocks_per_row) as u64,
            row_count as u64,
            config.batch as u64,
            profile_start.elapsed(),
        );
    }

    Ok(true)
}

/// Compute a contiguous block of complete Q8_0 rows for batch=1, accumulating
/// into `out_slice` (one entry per row). `base_row` is the row index of
/// `out_slice[0]` within the chunk, used to locate each row's Q8_0 blocks. This
/// is the single source of the per-row block-order accumulation that the
/// parallel split below must stay bit-identical to.
fn q8_0_batch1_complete_rows_range(
    input: &[f32],
    out_slice: &mut [f32],
    q8_bytes: &[u8],
    base_row: usize,
    blocks_per_row: usize,
) {
    for (local_offset, out) in out_slice.iter_mut().enumerate() {
        let first_block = (base_row + local_offset) * blocks_per_row;
        let mut acc = *out;
        for block_in_row in 0..blocks_per_row {
            let block_offset = (first_block + block_in_row) * 34;
            let scale = q8_0_block_scale(q8_bytes, block_offset);
            let input_start = block_in_row * 32;
            acc += scale
                * q8_0_dot_i8_f32(
                    &q8_bytes[block_offset + 2..block_offset + 34],
                    &input[input_start..],
                    32,
                );
        }
        *out = acc;
    }
}

/// R132: parallelize the batch=1 complete-rows Q8_0 GEMV across output rows.
/// Rows are independent, so each worker owns a disjoint row range and writes a
/// disjoint slice of `out_span`; the result is bit-identical to the serial path.
/// Honors `RLLM_THREADS`; serial fallback for one core or few rows (where the
/// thread-spawn cost would not pay off).
fn q8_0_batch1_complete_rows_parallel(
    input: &[f32],
    out_span: &mut [f32],
    q8_bytes: &[u8],
    blocks_per_row: usize,
) {
    let row_count = out_span.len();
    let threads = effective_runtime_threads(
        std::env::var(RLLM_THREADS_ENV).ok().as_deref(),
        available_runtime_threads(),
    );
    if threads <= 1 || row_count < 2 * MIN_ROWS_PER_PARALLEL_Q8_PREFILL {
        q8_0_batch1_complete_rows_range(input, out_span, q8_bytes, 0, blocks_per_row);
        return;
    }
    let workers = threads.min(row_count / MIN_ROWS_PER_PARALLEL_Q8_PREFILL).max(1);
    let rows_per_worker = row_count.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut rest = &mut out_span[..];
        let mut base_row = 0usize;
        while base_row < row_count {
            let rows = rows_per_worker.min(row_count - base_row);
            let (chunk, tail) = rest.split_at_mut(rows);
            rest = tail;
            let worker_base = base_row;
            scope.spawn(move || {
                q8_0_batch1_complete_rows_range(input, chunk, q8_bytes, worker_base, blocks_per_row);
            });
            base_row += rows;
        }
    });
}

fn accumulate_q8_0_chunk_multiply_into(
    input: &[f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<()> {
    let profile_enabled = q8_kernel_profile_enabled();
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    validate_q8_0_chunk(q8_bytes, element_start, weight_elements, weight_name)?;
    let mut multiply_advance_elapsed = std::time::Duration::ZERO;
    let mut multiply_advance_calls = 0u64;
    let mut multiply_scale_elapsed = std::time::Duration::ZERO;
    let mut multiply_scale_calls = 0u64;
    let mut multiply_batch4_elapsed = std::time::Duration::ZERO;
    let mut multiply_batch4_calls = 0u64;
    let mut multiply_batch4_items = 0u64;
    let mut multiply_tail_elapsed = std::time::Duration::ZERO;
    let mut multiply_tail_calls = 0u64;
    let mut multiply_tail_items = 0u64;
    let mut multiply_finish_elapsed = std::time::Duration::ZERO;
    let mut multiply_finish_calls = 0u64;
    if accumulate_q8_0_chunk_multiply_into_batch1_complete_rows(
        input,
        q8_bytes,
        element_start,
        config,
        state,
        weight_name,
    )? {
        return Ok(());
    }

    for block_idx in 0..q8_bytes.len() / 34 {
        let block_global_start = element_start + block_idx * 32;
        if block_global_start >= weight_elements {
            break;
        }
        let block_offset = block_idx * 34;
        let scale = q8_0_block_scale(q8_bytes, block_offset);
        let qs = &q8_bytes[block_offset + 2..block_offset + 34];
        let block_len = (weight_elements - block_global_start).min(32);
        let out_feature = block_global_start / config.in_features;
        let in_feature = block_global_start % config.in_features;

        if config.batch > 1 && block_len == 32 && in_feature + block_len <= config.in_features {
            let profile_start = profile_enabled.then(Instant::now);
            let advance_start = profile_enabled.then(Instant::now);
            advance_multiply_state_to_row(state, out_feature, config, weight_name)?;
            if let Some(advance_start) = advance_start {
                multiply_advance_elapsed += advance_start.elapsed();
                multiply_advance_calls += 1;
            }
            let scale_start = profile_enabled.then(Instant::now);
            let scaled = q8_0_scaled_block_reecast(qs, scale);
            if let Some(scale_start) = scale_start {
                multiply_scale_elapsed += scale_start.elapsed();
                multiply_scale_calls += 1;
            }
            let mut batch_idx = 0usize;
            let batch4_start_idx = batch_idx;
            let batch4_start = profile_enabled.then(Instant::now);
            while batch_idx + 4 <= config.batch {
                let input_start = batch_idx * config.in_features + in_feature;
                accumulate_f32_dot_32_batch4_into_reevec(
                    &scaled,
                    &input[input_start..],
                    config.in_features,
                    &mut state.current_acc,
                    batch_idx,
                );
                batch_idx += 4;
            }
            if let Some(batch4_start) = batch4_start {
                let calls = ((batch_idx - batch4_start_idx) / 4) as u64;
                multiply_batch4_elapsed += batch4_start.elapsed();
                multiply_batch4_calls += calls;
                multiply_batch4_items += calls * 4;
            }
            let tail_start_idx = batch_idx;
            let tail_start = profile_enabled.then(Instant::now);
            while batch_idx < config.batch {
                let input_start = batch_idx * config.in_features + in_feature;
                state.current_acc[batch_idx] += f32_dot_32(&scaled, &input[input_start..]);
                batch_idx += 1;
            }
            if let Some(tail_start) = tail_start {
                let calls = (batch_idx - tail_start_idx) as u64;
                multiply_tail_elapsed += tail_start.elapsed();
                multiply_tail_calls += calls;
                multiply_tail_items += calls;
            }
            if in_feature + block_len == config.in_features {
                let finish_start = profile_enabled.then(Instant::now);
                state.finish_current(config, weight_name)?;
                if let Some(finish_start) = finish_start {
                    multiply_finish_elapsed += finish_start.elapsed();
                    multiply_finish_calls += 1;
                }
            }
            if let Some(profile_start) = profile_start {
                record_q8_kernel_path(
                    Q8KernelPath::BatchGt1Scaled,
                    1,
                    1,
                    0,
                    config.batch as u64,
                    profile_start.elapsed(),
                );
            }
        } else if in_feature + block_len <= config.in_features {
            let profile_start = profile_enabled.then(Instant::now);
            advance_multiply_state_to_row(state, out_feature, config, weight_name)?;
            for batch_idx in 0..config.batch {
                let input_start = batch_idx * config.in_features + in_feature;
                state.current_acc[batch_idx] +=
                    scale * q8_0_dot_i8_f32(qs, &input[input_start..], block_len);
            }
            if in_feature + block_len == config.in_features {
                state.finish_current(config, weight_name)?;
            }
            if let Some(profile_start) = profile_start {
                record_q8_kernel_path(
                    Q8KernelPath::ContiguousI8Dot,
                    1,
                    1,
                    0,
                    config.batch as u64,
                    profile_start.elapsed(),
                );
            }
        } else {
            let profile_start = profile_enabled.then(Instant::now);
            for (idx, &q) in qs.iter().take(block_len).enumerate() {
                let global_idx = block_global_start + idx;
                let out_feature = global_idx / config.in_features;
                let in_feature = global_idx % config.in_features;
                advance_multiply_state_to_row(state, out_feature, config, weight_name)?;
                let weight = scale * (q as i8) as f32;
                for batch_idx in 0..config.batch {
                    state.current_acc[batch_idx] +=
                        input[batch_idx * config.in_features + in_feature] * weight;
                }
                if in_feature + 1 == config.in_features {
                    state.finish_current(config, weight_name)?;
                }
            }
            if let Some(profile_start) = profile_start {
                record_q8_kernel_path(
                    Q8KernelPath::SplitRowScalar,
                    1,
                    1,
                    0,
                    config.batch as u64,
                    profile_start.elapsed(),
                );
            }
        }
    }

    if profile_enabled {
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1MultiplyAdvance,
            multiply_advance_calls,
            multiply_advance_calls,
            0,
            0,
            multiply_advance_elapsed,
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1MultiplyScale,
            multiply_scale_calls,
            multiply_scale_calls,
            0,
            0,
            multiply_scale_elapsed,
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1MultiplyBatch4,
            multiply_batch4_calls,
            multiply_batch4_calls,
            0,
            multiply_batch4_items,
            multiply_batch4_elapsed,
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1MultiplyTail,
            multiply_tail_calls,
            multiply_tail_calls,
            0,
            multiply_tail_items,
            multiply_tail_elapsed,
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1MultiplyFinish,
            multiply_finish_calls,
            multiply_finish_calls,
            multiply_finish_calls,
            0,
            multiply_finish_elapsed,
        );
    }

    Ok(())
}

fn accumulate_q8_0_chunk_multiply_into_batch1_complete_rows(
    input: &[f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<bool> {
    let profile_enabled = q8_kernel_profile_enabled();
    let Some((first_row, row_count, blocks_per_row)) =
        q8_0_complete_row_span(q8_bytes, element_start, config)?
    else {
        return Ok(false);
    };
    let row_end = first_row
        .checked_add(row_count)
        .ok_or_else(|| RuntimeError::Shape("Q8_0 multiply row fast path overflow".to_string()))?;
    if row_end > config.out_features {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} Q8_0 multiply row fast path rows {first_row}..{row_end} exceed expected {}",
            config.out_features
        )));
    }

    let profile_start = profile_enabled.then(Instant::now);
    for local_row in 0..row_count {
        let out_feature = first_row + local_row;
        advance_multiply_state_to_row(state, out_feature, config, weight_name)?;
        let mut acc = state.current_acc[0];
        let first_block = local_row * blocks_per_row;
        for block_in_row in 0..blocks_per_row {
            let block_offset = (first_block + block_in_row) * 34;
            let scale = q8_0_block_scale(q8_bytes, block_offset);
            let input_start = block_in_row * 32;
            acc += scale
                * q8_0_dot_i8_f32(
                    &q8_bytes[block_offset + 2..block_offset + 34],
                    &input[input_start..],
                    32,
                );
        }
        state.current_acc[0] = acc;
        state.finish_current(config, weight_name)?;
    }
    if let Some(profile_start) = profile_start {
        record_q8_kernel_path(
            Q8KernelPath::Batch1CompleteMultiply,
            1,
            (row_count * blocks_per_row) as u64,
            row_count as u64,
            config.batch as u64,
            profile_start.elapsed(),
        );
    }

    Ok(true)
}

fn accumulate_q8_0_chunk_argmax(
    input: &[f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearArgmaxState<'_>,
    weight_name: &str,
) -> Result<()> {
    let profile_enabled = q8_kernel_profile_enabled();
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    validate_q8_0_chunk(q8_bytes, element_start, weight_elements, weight_name)?;
    if accumulate_q8_0_chunk_argmax_batch1_complete_rows(
        input,
        q8_bytes,
        element_start,
        config,
        state,
        weight_name,
    )? {
        return Ok(());
    }

    for block_idx in 0..q8_bytes.len() / 34 {
        let block_global_start = element_start + block_idx * 32;
        if block_global_start >= weight_elements {
            break;
        }
        let block_offset = block_idx * 34;
        let scale = q8_0_block_scale(q8_bytes, block_offset);
        let qs = &q8_bytes[block_offset + 2..block_offset + 34];
        let block_len = (weight_elements - block_global_start).min(32);
        let out_feature = block_global_start / config.in_features;
        let in_feature = block_global_start % config.in_features;

        if in_feature + block_len <= config.in_features {
            let profile_start = profile_enabled.then(Instant::now);
            advance_argmax_state_to_row(state, out_feature, config, weight_name)?;
            state.current_acc += scale * q8_0_dot_i8_f32(qs, &input[in_feature..], block_len);
            if in_feature + block_len == config.in_features {
                state.finish_current(config, weight_name)?;
            }
            if let Some(profile_start) = profile_start {
                record_q8_kernel_path(
                    Q8KernelPath::ContiguousI8Dot,
                    1,
                    1,
                    0,
                    config.batch as u64,
                    profile_start.elapsed(),
                );
            }
        } else {
            let profile_start = profile_enabled.then(Instant::now);
            for (idx, &q) in qs.iter().take(block_len).enumerate() {
                let global_idx = block_global_start + idx;
                let out_feature = global_idx / config.in_features;
                let in_feature = global_idx % config.in_features;
                advance_argmax_state_to_row(state, out_feature, config, weight_name)?;
                state.current_acc += input[in_feature] * scale * (q as i8) as f32;
                if in_feature + 1 == config.in_features {
                    state.finish_current(config, weight_name)?;
                }
            }
            if let Some(profile_start) = profile_start {
                record_q8_kernel_path(
                    Q8KernelPath::SplitRowScalar,
                    1,
                    1,
                    0,
                    config.batch as u64,
                    profile_start.elapsed(),
                );
            }
        }
    }

    Ok(())
}

fn accumulate_q8_0_chunk_argmax_batch1_complete_rows(
    input: &[f32],
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearArgmaxState<'_>,
    weight_name: &str,
) -> Result<bool> {
    let profile_enabled = q8_kernel_profile_enabled();
    let Some((first_row, row_count, blocks_per_row)) =
        q8_0_complete_row_span(q8_bytes, element_start, config)?
    else {
        return Ok(false);
    };
    let row_end = first_row
        .checked_add(row_count)
        .ok_or_else(|| RuntimeError::Shape("Q8_0 argmax row fast path overflow".to_string()))?;
    if row_end > config.out_features {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} Q8_0 argmax row fast path rows {first_row}..{row_end} exceed expected {}",
            config.out_features
        )));
    }

    let profile_start = profile_enabled.then(Instant::now);
    for local_row in 0..row_count {
        let out_feature = first_row + local_row;
        advance_argmax_state_to_row(state, out_feature, config, weight_name)?;
        let mut acc = state.current_acc;
        let first_block = local_row * blocks_per_row;
        for block_in_row in 0..blocks_per_row {
            let block_offset = (first_block + block_in_row) * 34;
            let scale = q8_0_block_scale(q8_bytes, block_offset);
            let input_start = block_in_row * 32;
            acc += scale
                * q8_0_dot_i8_f32(
                    &q8_bytes[block_offset + 2..block_offset + 34],
                    &input[input_start..],
                    32,
                );
        }
        state.current_acc = acc;
        state.finish_current(config, weight_name)?;
    }
    if let Some(profile_start) = profile_start {
        record_q8_kernel_path(
            Q8KernelPath::Batch1CompleteArgmax,
            1,
            (row_count * blocks_per_row) as u64,
            row_count as u64,
            config.batch as u64,
            profile_start.elapsed(),
        );
    }

    Ok(true)
}

fn validate_q8_0_chunk(
    q8_bytes: &[u8],
    element_start: usize,
    weight_elements: usize,
    weight_name: &str,
) -> Result<()> {
    if !q8_bytes.len().is_multiple_of(34) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Q8_0 stream for {weight_name} has byte len {} not aligned to 34-byte blocks",
            q8_bytes.len()
        )));
    }
    if element_start > weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} Q8_0 chunk starts at element {element_start}, beyond expected {weight_elements}"
        )));
    }
    Ok(())
}

fn q8_0_complete_row_span(
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
) -> Result<Option<(usize, usize, usize)>> {
    if config.batch != 1
        || config.in_features == 0
        || !config.in_features.is_multiple_of(32)
        || !element_start.is_multiple_of(config.in_features)
    {
        return Ok(None);
    }
    let chunk_elements = quantized_elements_for_bytes(rllm_container::DType::Q8_0, q8_bytes.len())?;
    if chunk_elements == 0 || !chunk_elements.is_multiple_of(config.in_features) {
        return Ok(None);
    }
    Ok(Some((
        element_start / config.in_features,
        chunk_elements / config.in_features,
        config.in_features / 32,
    )))
}

fn q8_0_block_scale(q8_bytes: &[u8], block_offset: usize) -> f32 {
    let scale_bits = u16::from_le_bytes([q8_bytes[block_offset], q8_bytes[block_offset + 1]]);
    crate::tensor::fp16_to_f32(scale_bits)
}

fn q8_0_dot_i8_f32(qs: &[u8], input: &[f32], len: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut idx = 0usize;
    while idx + 4 <= len {
        acc += (qs[idx] as i8) as f32 * input[idx]
            + (qs[idx + 1] as i8) as f32 * input[idx + 1]
            + (qs[idx + 2] as i8) as f32 * input[idx + 2]
            + (qs[idx + 3] as i8) as f32 * input[idx + 3];
        idx += 4;
    }
    while idx < len {
        acc += (qs[idx] as i8) as f32 * input[idx];
        idx += 1;
    }
    acc
}

#[allow(dead_code)]
fn q8_0_scaled_block(qs: &[u8], scale: f32) -> [f32; 32] {
    let mut scaled = [0.0f32; 32];
    for idx in 0..32 {
        scaled[idx] = scale * (qs[idx] as i8) as f32;
    }
    scaled
}

fn q8_0_scaled_block_reecast(qs: &[u8], scale: f32) -> [f32; 32] {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return q8_0_scaled_block_neon(qs, scale);
    }

    #[cfg(not(target_arch = "aarch64"))]
    q8_0_scaled_block(qs, scale)
}

#[cfg(target_arch = "aarch64")]
unsafe fn q8_0_scaled_block_neon(qs: &[u8], scale: f32) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    let scale_vec = vdupq_n_f32(scale);
    let mut offset = 0usize;
    while offset < 32 {
        let q_i8 = vld1q_s8(qs.as_ptr().add(offset) as *const i8);
        let low_i16 = vmovl_s8(vget_low_s8(q_i8));
        let high_i16 = vmovl_s8(vget_high_s8(q_i8));

        let low_low_i32 = vmovl_s16(vget_low_s16(low_i16));
        let low_high_i32 = vmovl_s16(vget_high_s16(low_i16));
        let high_low_i32 = vmovl_s16(vget_low_s16(high_i16));
        let high_high_i32 = vmovl_s16(vget_high_s16(high_i16));

        vst1q_f32(
            out.as_mut_ptr().add(offset),
            vmulq_f32(vcvtq_f32_s32(low_low_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 4),
            vmulq_f32(vcvtq_f32_s32(low_high_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 8),
            vmulq_f32(vcvtq_f32_s32(high_low_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 12),
            vmulq_f32(vcvtq_f32_s32(high_high_i32), scale_vec),
        );
        offset += 16;
    }
    out
}

fn f32_dot_32(weights: &[f32; 32], input: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut idx = 0usize;
    while idx < 32 {
        acc0 += weights[idx] * input[idx];
        acc1 += weights[idx + 1] * input[idx + 1];
        acc2 += weights[idx + 2] * input[idx + 2];
        acc3 += weights[idx + 3] * input[idx + 3];
        idx += 4;
    }
    (acc0 + acc1) + (acc2 + acc3)
}

#[allow(dead_code)]
fn accumulate_f32_dot_32_batch4(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    let mut acc0 = output[out_feature];
    let mut acc1 = output[output_stride + out_feature];
    let mut acc2 = output[output_stride * 2 + out_feature];
    let mut acc3 = output[output_stride * 3 + out_feature];
    let mut idx = 0usize;
    while idx < 32 {
        let weight = weights[idx];
        acc0 += weight * input[idx];
        acc1 += weight * input[input_stride + idx];
        acc2 += weight * input[input_stride * 2 + idx];
        acc3 += weight * input[input_stride * 3 + idx];
        idx += 1;
    }
    output[out_feature] = acc0;
    output[output_stride + out_feature] = acc1;
    output[output_stride * 2 + out_feature] = acc2;
    output[output_stride * 3 + out_feature] = acc3;
}

fn accumulate_f32_dot_32_batch4_reevec(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return accumulate_f32_dot_32_batch4_neon(
            weights,
            input,
            input_stride,
            output,
            output_stride,
            out_feature,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_f32_dot_32_batch4(
        weights,
        input,
        input_stride,
        output,
        output_stride,
        out_feature,
    );
}

fn accumulate_f32_dot_32_output2_batch4_reebundle(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return accumulate_f32_dot_32_output2_batch4_neon(
            first,
            second,
            input,
            input_stride,
            output,
            output_stride,
            first_out_feature,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_f32_dot_32_output2_batch4_scalar(
        first,
        second,
        input,
        input_stride,
        output,
        output_stride,
        first_out_feature,
    );
}

#[cfg(not(target_arch = "aarch64"))]
fn accumulate_f32_dot_32_output2_batch4_scalar(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
) {
    let mut first0 = output[first_out_feature];
    let mut first1 = output[output_stride + first_out_feature];
    let mut first2 = output[output_stride * 2 + first_out_feature];
    let mut first3 = output[output_stride * 3 + first_out_feature];
    let second_out_feature = first_out_feature + 1;
    let mut second0 = output[second_out_feature];
    let mut second1 = output[output_stride + second_out_feature];
    let mut second2 = output[output_stride * 2 + second_out_feature];
    let mut second3 = output[output_stride * 3 + second_out_feature];
    let mut idx = 0usize;
    while idx < 32 {
        let x0 = input[idx];
        let x1 = input[input_stride + idx];
        let x2 = input[input_stride * 2 + idx];
        let x3 = input[input_stride * 3 + idx];
        let fw = first[idx];
        let sw = second[idx];
        first0 += fw * x0;
        first1 += fw * x1;
        first2 += fw * x2;
        first3 += fw * x3;
        second0 += sw * x0;
        second1 += sw * x1;
        second2 += sw * x2;
        second3 += sw * x3;
        idx += 1;
    }
    output[first_out_feature] = first0;
    output[output_stride + first_out_feature] = first1;
    output[output_stride * 2 + first_out_feature] = first2;
    output[output_stride * 3 + first_out_feature] = first3;
    output[second_out_feature] = second0;
    output[output_stride + second_out_feature] = second1;
    output[output_stride * 2 + second_out_feature] = second2;
    output[output_stride * 3 + second_out_feature] = second3;
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_f32_dot_32_output2_batch4_neon(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
) {
    let mut first0 = vdupq_n_f32(0.0);
    let mut first1 = vdupq_n_f32(0.0);
    let mut first2 = vdupq_n_f32(0.0);
    let mut first3 = vdupq_n_f32(0.0);
    let mut second0 = vdupq_n_f32(0.0);
    let mut second1 = vdupq_n_f32(0.0);
    let mut second2 = vdupq_n_f32(0.0);
    let mut second3 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let x0 = vld1q_f32(input.as_ptr().add(idx));
        let x1 = vld1q_f32(input.as_ptr().add(input_stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(input_stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(input_stride * 3 + idx));
        let first_weights = vld1q_f32(first.as_ptr().add(idx));
        let second_weights = vld1q_f32(second.as_ptr().add(idx));
        first0 = vfmaq_f32(first0, first_weights, x0);
        first1 = vfmaq_f32(first1, first_weights, x1);
        first2 = vfmaq_f32(first2, first_weights, x2);
        first3 = vfmaq_f32(first3, first_weights, x3);
        second0 = vfmaq_f32(second0, second_weights, x0);
        second1 = vfmaq_f32(second1, second_weights, x1);
        second2 = vfmaq_f32(second2, second_weights, x2);
        second3 = vfmaq_f32(second3, second_weights, x3);
        idx += 4;
    }
    let second_out_feature = first_out_feature + 1;
    output[first_out_feature] += vaddvq_f32(first0);
    output[output_stride + first_out_feature] += vaddvq_f32(first1);
    output[output_stride * 2 + first_out_feature] += vaddvq_f32(first2);
    output[output_stride * 3 + first_out_feature] += vaddvq_f32(first3);
    output[second_out_feature] += vaddvq_f32(second0);
    output[output_stride + second_out_feature] += vaddvq_f32(second1);
    output[output_stride * 2 + second_out_feature] += vaddvq_f32(second2);
    output[output_stride * 3 + second_out_feature] += vaddvq_f32(second3);
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_f32_dot_32_batch4_neon(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let w = vld1q_f32(weights.as_ptr().add(idx));
        let x0 = vld1q_f32(input.as_ptr().add(idx));
        let x1 = vld1q_f32(input.as_ptr().add(input_stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(input_stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(input_stride * 3 + idx));
        acc0 = vfmaq_f32(acc0, w, x0);
        acc1 = vfmaq_f32(acc1, w, x1);
        acc2 = vfmaq_f32(acc2, w, x2);
        acc3 = vfmaq_f32(acc3, w, x3);
        idx += 4;
    }
    output[out_feature] += vaddvq_f32(acc0);
    output[output_stride + out_feature] += vaddvq_f32(acc1);
    output[output_stride * 2 + out_feature] += vaddvq_f32(acc2);
    output[output_stride * 3 + out_feature] += vaddvq_f32(acc3);
}

#[allow(dead_code)]
fn accumulate_f32_dot_32_batch4_into(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    accumulators: &mut [f32],
    accumulator_start: usize,
) {
    let mut acc0 = accumulators[accumulator_start];
    let mut acc1 = accumulators[accumulator_start + 1];
    let mut acc2 = accumulators[accumulator_start + 2];
    let mut acc3 = accumulators[accumulator_start + 3];
    let mut idx = 0usize;
    while idx < 32 {
        let weight = weights[idx];
        acc0 += weight * input[idx];
        acc1 += weight * input[input_stride + idx];
        acc2 += weight * input[input_stride * 2 + idx];
        acc3 += weight * input[input_stride * 3 + idx];
        idx += 1;
    }
    accumulators[accumulator_start] = acc0;
    accumulators[accumulator_start + 1] = acc1;
    accumulators[accumulator_start + 2] = acc2;
    accumulators[accumulator_start + 3] = acc3;
}

fn accumulate_f32_dot_32_batch4_into_reevec(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    accumulators: &mut [f32],
    accumulator_start: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return accumulate_f32_dot_32_batch4_into_neon(
            weights,
            input,
            input_stride,
            accumulators,
            accumulator_start,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_f32_dot_32_batch4_into(
        weights,
        input,
        input_stride,
        accumulators,
        accumulator_start,
    );
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_f32_dot_32_batch4_into_neon(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    accumulators: &mut [f32],
    accumulator_start: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let w = vld1q_f32(weights.as_ptr().add(idx));
        let x0 = vld1q_f32(input.as_ptr().add(idx));
        let x1 = vld1q_f32(input.as_ptr().add(input_stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(input_stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(input_stride * 3 + idx));
        acc0 = vfmaq_f32(acc0, w, x0);
        acc1 = vfmaq_f32(acc1, w, x1);
        acc2 = vfmaq_f32(acc2, w, x2);
        acc3 = vfmaq_f32(acc3, w, x3);
        idx += 4;
    }
    accumulators[accumulator_start] += vaddvq_f32(acc0);
    accumulators[accumulator_start + 1] += vaddvq_f32(acc1);
    accumulators[accumulator_start + 2] += vaddvq_f32(acc2);
    accumulators[accumulator_start + 3] += vaddvq_f32(acc3);
}

fn advance_multiply_state_to_row(
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    out_feature: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    while state.current_out_feature < out_feature {
        state.finish_current(config, weight_name)?;
    }
    if state.current_out_feature != out_feature {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
            out_feature, state.current_out_feature
        )));
    }
    Ok(())
}

fn advance_argmax_state_to_row(
    state: &mut StreamingLinearArgmaxState<'_>,
    out_feature: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    while state.current_out_feature < out_feature {
        state.finish_current(config, weight_name)?;
    }
    if state.current_out_feature != out_feature {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
            out_feature, state.current_out_feature
        )));
    }
    Ok(())
}

fn accumulate_fused_rle_chunk_u8(
    input: &[f32],
    output: &mut [f32],
    rle_stream: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    if !rle_stream.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "RLE stream for {weight_name} has odd length"
        )));
    }

    let mut current_element = element_start;
    for chunk in rle_stream.chunks_exact(2) {
        let count = chunk[0] as usize;
        let value = chunk[1] as f32;

        let mut i = 0;
        while i < count {
            let out_feature = current_element / config.in_features;
            let in_feature = current_element % config.in_features;
            let run_in_this_row = (config.in_features - in_feature).min(count - i);

            let mut batch_idx = 0;
            while batch_idx < config.batch {
                let output_idx = batch_idx * config.out_features + out_feature;
                let input_start = batch_idx * config.in_features + in_feature;

                let mut sum = 0.0;
                for j in 0..run_in_this_row {
                    sum += input[input_start + j];
                }
                output[output_idx] += value * sum;

                batch_idx += 1;
            }

            current_element += run_in_this_row;
            i += run_in_this_row;
        }
    }

    Ok(())
}

fn accumulate_fused_raw_fp16_chunk(
    input: &[f32],
    output: &mut [f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw FP16 stream for {weight_name} has odd length"
        )));
    }

    if config.batch == 1 {
        return accumulate_fused_raw_fp16_chunk_batch1(
            input,
            output,
            raw_bytes,
            element_start,
            config,
            weight_name,
        );
    }

    let weight_elements = raw_bytes.len() / 2;
    let mut local_idx = 0usize;
    let mut global_idx = element_start;

    const BLOCK_SIZE: usize = 128;
    let mut w_block = [0.0f32; BLOCK_SIZE];

    while local_idx < weight_elements {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);

        let mut row_idx = 0;
        while row_idx < row_len {
            let block_len = (row_len - row_idx).min(BLOCK_SIZE);
            let byte_start = (local_idx + row_idx) * 2;
            let block_bytes = &raw_bytes[byte_start..byte_start + block_len * 2];

            // Decode this block ONCE into the stack array
            let mut i = 0;
            while i + 4 <= block_len {
                let b = &block_bytes[i * 2..i * 2 + 8];
                w_block[i] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                w_block[i + 1] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[2], b[3]]));
                w_block[i + 2] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[4], b[5]]));
                w_block[i + 3] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[6], b[7]]));
                i += 4;
            }
            while i < block_len {
                let b = &block_bytes[i * 2..i * 2 + 2];
                w_block[i] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                i += 1;
            }
            let w_slice = &w_block[..block_len];

            // Same 8-wide batch unrolling as accumulate_weight_chunk
            let mut batch_idx = 0usize;
            while batch_idx + 8 <= config.batch {
                let out_idx0 = batch_idx * config.out_features + out_feature;
                let out_idx1 = (batch_idx + 1) * config.out_features + out_feature;
                let out_idx2 = (batch_idx + 2) * config.out_features + out_feature;
                let out_idx3 = (batch_idx + 3) * config.out_features + out_feature;
                let out_idx4 = (batch_idx + 4) * config.out_features + out_feature;
                let out_idx5 = (batch_idx + 5) * config.out_features + out_feature;
                let out_idx6 = (batch_idx + 6) * config.out_features + out_feature;
                let out_idx7 = (batch_idx + 7) * config.out_features + out_feature;

                let mut acc0 = output[out_idx0];
                let mut acc1 = output[out_idx1];
                let mut acc2 = output[out_idx2];
                let mut acc3 = output[out_idx3];
                let mut acc4 = output[out_idx4];
                let mut acc5 = output[out_idx5];
                let mut acc6 = output[out_idx6];
                let mut acc7 = output[out_idx7];

                let in_start0 = batch_idx * config.in_features + in_feature + row_idx;
                let in_start1 = (batch_idx + 1) * config.in_features + in_feature + row_idx;
                let in_start2 = (batch_idx + 2) * config.in_features + in_feature + row_idx;
                let in_start3 = (batch_idx + 3) * config.in_features + in_feature + row_idx;
                let in_start4 = (batch_idx + 4) * config.in_features + in_feature + row_idx;
                let in_start5 = (batch_idx + 5) * config.in_features + in_feature + row_idx;
                let in_start6 = (batch_idx + 6) * config.in_features + in_feature + row_idx;
                let in_start7 = (batch_idx + 7) * config.in_features + in_feature + row_idx;

                let mut idx = 0;
                while idx + 4 <= block_len {
                    let w = &w_slice[idx..idx + 4];
                    let i0 = &input[in_start0 + idx..in_start0 + idx + 4];
                    let i1 = &input[in_start1 + idx..in_start1 + idx + 4];
                    let i2 = &input[in_start2 + idx..in_start2 + idx + 4];
                    let i3 = &input[in_start3 + idx..in_start3 + idx + 4];
                    let i4 = &input[in_start4 + idx..in_start4 + idx + 4];
                    let i5 = &input[in_start5 + idx..in_start5 + idx + 4];
                    let i6 = &input[in_start6 + idx..in_start6 + idx + 4];
                    let i7 = &input[in_start7 + idx..in_start7 + idx + 4];

                    acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                    acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                    acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                    acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                    acc4 += w[0] * i4[0] + w[1] * i4[1] + w[2] * i4[2] + w[3] * i4[3];
                    acc5 += w[0] * i5[0] + w[1] * i5[1] + w[2] * i5[2] + w[3] * i5[3];
                    acc6 += w[0] * i6[0] + w[1] * i6[1] + w[2] * i6[2] + w[3] * i6[3];
                    acc7 += w[0] * i7[0] + w[1] * i7[1] + w[2] * i7[2] + w[3] * i7[3];
                    idx += 4;
                }
                while idx < block_len {
                    let weight = w_slice[idx];
                    acc0 += input[in_start0 + idx] * weight;
                    acc1 += input[in_start1 + idx] * weight;
                    acc2 += input[in_start2 + idx] * weight;
                    acc3 += input[in_start3 + idx] * weight;
                    acc4 += input[in_start4 + idx] * weight;
                    acc5 += input[in_start5 + idx] * weight;
                    acc6 += input[in_start6 + idx] * weight;
                    acc7 += input[in_start7 + idx] * weight;
                    idx += 1;
                }

                output[out_idx0] = acc0;
                output[out_idx1] = acc1;
                output[out_idx2] = acc2;
                output[out_idx3] = acc3;
                output[out_idx4] = acc4;
                output[out_idx5] = acc5;
                output[out_idx6] = acc6;
                output[out_idx7] = acc7;
                batch_idx += 8;
            }

            while batch_idx + 4 <= config.batch {
                let out_idx0 = batch_idx * config.out_features + out_feature;
                let out_idx1 = (batch_idx + 1) * config.out_features + out_feature;
                let out_idx2 = (batch_idx + 2) * config.out_features + out_feature;
                let out_idx3 = (batch_idx + 3) * config.out_features + out_feature;

                let mut acc0 = output[out_idx0];
                let mut acc1 = output[out_idx1];
                let mut acc2 = output[out_idx2];
                let mut acc3 = output[out_idx3];

                let in_start0 = batch_idx * config.in_features + in_feature + row_idx;
                let in_start1 = (batch_idx + 1) * config.in_features + in_feature + row_idx;
                let in_start2 = (batch_idx + 2) * config.in_features + in_feature + row_idx;
                let in_start3 = (batch_idx + 3) * config.in_features + in_feature + row_idx;

                let mut idx = 0;
                while idx + 4 <= block_len {
                    let w = &w_slice[idx..idx + 4];
                    let i0 = &input[in_start0 + idx..in_start0 + idx + 4];
                    let i1 = &input[in_start1 + idx..in_start1 + idx + 4];
                    let i2 = &input[in_start2 + idx..in_start2 + idx + 4];
                    let i3 = &input[in_start3 + idx..in_start3 + idx + 4];

                    acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                    acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                    acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                    acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                    idx += 4;
                }
                while idx < block_len {
                    let weight = w_slice[idx];
                    acc0 += input[in_start0 + idx] * weight;
                    acc1 += input[in_start1 + idx] * weight;
                    acc2 += input[in_start2 + idx] * weight;
                    acc3 += input[in_start3 + idx] * weight;
                    idx += 1;
                }

                output[out_idx0] = acc0;
                output[out_idx1] = acc1;
                output[out_idx2] = acc2;
                output[out_idx3] = acc3;
                batch_idx += 4;
            }

            while batch_idx < config.batch {
                let out_idx = batch_idx * config.out_features + out_feature;
                let mut acc = output[out_idx];
                let in_start = batch_idx * config.in_features + in_feature + row_idx;

                let mut idx = 0;
                while idx + 4 <= block_len {
                    let w = &w_slice[idx..idx + 4];
                    let i0 = &input[in_start + idx..in_start + idx + 4];
                    acc += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                    idx += 4;
                }
                while idx < block_len {
                    acc += w_slice[idx] * input[in_start + idx];
                    idx += 1;
                }
                output[out_idx] = acc;
                batch_idx += 1;
            }

            row_idx += block_len;
        }

        local_idx += row_len;
        global_idx += row_len;
    }

    Ok(())
}

fn accumulate_fused_raw_fp16_chunk_batch1(
    input: &[f32],
    output: &mut [f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    accumulate_fused_raw_fp16_chunk_batch1_row_blocked(
        input,
        output,
        raw_bytes,
        element_start,
        config,
        weight_name,
    )
}

fn accumulate_fused_raw_fp16_chunk_batch1_row_blocked(
    input: &[f32],
    output: &mut [f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw FP16 stream for {weight_name} has odd length"
        )));
    }

    if config.batch != 1 {
        return Err(RuntimeError::Shape(format!(
            "raw FP16 batch1 row-block kernel requires batch=1, got {}",
            config.batch
        )));
    }

    let expected_weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    let weight_elements = raw_bytes.len() / 2;
    let element_end = element_start
        .checked_add(weight_elements)
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > expected_weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {expected_weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;

    while local_idx < weight_elements && !global_idx.is_multiple_of(config.in_features) {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);
        let mut acc = output[out_feature];
        acc += raw_fp16_dot_segment(input, raw_bytes, local_idx, in_feature, row_len)?;
        output[out_feature] = acc;
        local_idx += row_len;
        global_idx += row_len;
    }

    let row_block_elements = config
        .in_features
        .checked_mul(4)
        .ok_or_else(|| RuntimeError::Shape("row block element count overflow".to_string()))?;
    while local_idx + row_block_elements <= weight_elements {
        let out_feature = global_idx / config.in_features;
        if out_feature + 3 >= config.out_features {
            break;
        }

        let mut acc0 = output[out_feature];
        let mut acc1 = output[out_feature + 1];
        let mut acc2 = output[out_feature + 2];
        let mut acc3 = output[out_feature + 3];
        let row0_start = local_idx;
        let row1_start = local_idx + config.in_features;
        let row2_start = row1_start + config.in_features;
        let row3_start = row2_start + config.in_features;

        let mut idx = 0usize;
        while idx < config.in_features {
            let x = input[idx];
            acc0 += x * fp16_weight_at(raw_bytes, row0_start + idx);
            acc1 += x * fp16_weight_at(raw_bytes, row1_start + idx);
            acc2 += x * fp16_weight_at(raw_bytes, row2_start + idx);
            acc3 += x * fp16_weight_at(raw_bytes, row3_start + idx);
            idx += 1;
        }

        output[out_feature] = acc0;
        output[out_feature + 1] = acc1;
        output[out_feature + 2] = acc2;
        output[out_feature + 3] = acc3;
        local_idx += row_block_elements;
        global_idx += row_block_elements;
    }

    while local_idx < weight_elements {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);
        let mut acc = output[out_feature];
        acc += raw_fp16_dot_segment(input, raw_bytes, local_idx, in_feature, row_len)?;
        output[out_feature] = acc;
        local_idx += row_len;
        global_idx += row_len;
    }

    Ok(())
}

fn accumulate_fused_raw_bf16_chunk_batch1(
    input: &[f32],
    output: &mut [f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    accumulate_fused_raw_16bit_chunk_batch1_row_blocked(
        input,
        output,
        raw_bytes,
        element_start,
        config,
        rllm_container::DType::Bf16,
        weight_name,
    )
}

fn accumulate_fused_raw_16bit_chunk_batch1_row_blocked(
    input: &[f32],
    output: &mut [f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    weight_name: &str,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw 16-bit stream for {weight_name} has odd length"
        )));
    }
    if !matches!(
        dtype,
        rllm_container::DType::Fp16 | rllm_container::DType::Bf16
    ) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw 16-bit stream for {weight_name} has unsupported dtype {dtype:?}"
        )));
    }

    if config.batch != 1 {
        return Err(RuntimeError::Shape(format!(
            "raw 16-bit batch1 row-block kernel requires batch=1, got {}",
            config.batch
        )));
    }

    let expected_weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    let weight_elements = raw_bytes.len() / 2;
    let element_end = element_start
        .checked_add(weight_elements)
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > expected_weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {expected_weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;

    while local_idx < weight_elements && !global_idx.is_multiple_of(config.in_features) {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);
        let mut acc = output[out_feature];
        acc += raw_16bit_dot_segment(input, raw_bytes, local_idx, in_feature, row_len, dtype)?;
        output[out_feature] = acc;
        local_idx += row_len;
        global_idx += row_len;
    }

    let row_block_elements = config
        .in_features
        .checked_mul(4)
        .ok_or_else(|| RuntimeError::Shape("row block element count overflow".to_string()))?;
    while local_idx + row_block_elements <= weight_elements {
        let out_feature = global_idx / config.in_features;
        if out_feature + 3 >= config.out_features {
            break;
        }

        let mut acc0 = output[out_feature];
        let mut acc1 = output[out_feature + 1];
        let mut acc2 = output[out_feature + 2];
        let mut acc3 = output[out_feature + 3];
        let row0_start = local_idx;
        let row1_start = local_idx + config.in_features;
        let row2_start = row1_start + config.in_features;
        let row3_start = row2_start + config.in_features;

        let mut idx = 0usize;
        while idx < config.in_features {
            let x = input[idx];
            acc0 += x * raw_16bit_weight_at(raw_bytes, row0_start + idx, dtype);
            acc1 += x * raw_16bit_weight_at(raw_bytes, row1_start + idx, dtype);
            acc2 += x * raw_16bit_weight_at(raw_bytes, row2_start + idx, dtype);
            acc3 += x * raw_16bit_weight_at(raw_bytes, row3_start + idx, dtype);
            idx += 1;
        }

        output[out_feature] = acc0;
        output[out_feature + 1] = acc1;
        output[out_feature + 2] = acc2;
        output[out_feature + 3] = acc3;
        local_idx += row_block_elements;
        global_idx += row_block_elements;
    }

    while local_idx < weight_elements {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);
        let mut acc = output[out_feature];
        acc += raw_16bit_dot_segment(input, raw_bytes, local_idx, in_feature, row_len, dtype)?;
        output[out_feature] = acc;
        local_idx += row_len;
        global_idx += row_len;
    }

    Ok(())
}

fn raw_fp16_dot_segment(
    input: &[f32],
    raw_bytes: &[u8],
    local_idx: usize,
    in_feature: usize,
    row_len: usize,
) -> Result<f32> {
    let mut acc = 0.0f32;
    let mut idx = 0usize;
    while idx + 4 <= row_len {
        let byte_idx = (local_idx + idx) * 2;
        let bytes = &raw_bytes[byte_idx..byte_idx + 8];
        let w0 = crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        let w1 = crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
        let w2 = crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[4], bytes[5]]));
        let w3 = crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[6], bytes[7]]));
        let input_start = in_feature + idx;
        acc += input[input_start] * w0
            + input[input_start + 1] * w1
            + input[input_start + 2] * w2
            + input[input_start + 3] * w3;
        idx += 4;
    }
    while idx < row_len {
        acc += input[in_feature + idx] * fp16_weight_at(raw_bytes, local_idx + idx);
        idx += 1;
    }
    Ok(acc)
}

fn raw_16bit_dot_segment(
    input: &[f32],
    raw_bytes: &[u8],
    local_idx: usize,
    in_feature: usize,
    row_len: usize,
    dtype: rllm_container::DType,
) -> Result<f32> {
    let mut acc = 0.0f32;
    let mut idx = 0usize;
    while idx + 4 <= row_len {
        let byte_idx = (local_idx + idx) * 2;
        let bytes = &raw_bytes[byte_idx..byte_idx + 8];
        let input_start = in_feature + idx;
        acc += input[input_start]
            * raw_16bit_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]), dtype)
            + input[input_start + 1]
                * raw_16bit_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]), dtype)
            + input[input_start + 2]
                * raw_16bit_to_f32(u16::from_le_bytes([bytes[4], bytes[5]]), dtype)
            + input[input_start + 3]
                * raw_16bit_to_f32(u16::from_le_bytes([bytes[6], bytes[7]]), dtype);
        idx += 4;
    }
    while idx < row_len {
        acc += input[in_feature + idx] * raw_16bit_weight_at(raw_bytes, local_idx + idx, dtype);
        idx += 1;
    }
    Ok(acc)
}

fn accumulate_silu_gate_up_raw_16bit_chunk_batch1(
    input: &[f32],
    gate_raw_bytes: &[u8],
    up_raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    state: &mut SiluGateUpState<'_>,
    weight_name: &str,
) -> Result<()> {
    if !gate_raw_bytes.len().is_multiple_of(2) || !up_raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw FP16 stream for {weight_name} has odd length"
        )));
    }
    if gate_raw_bytes.len() != up_raw_bytes.len() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "gate/up raw chunk len mismatch for {weight_name}: gate={}, up={}",
            gate_raw_bytes.len(),
            up_raw_bytes.len()
        )));
    }
    if config.batch != 1 {
        return Err(RuntimeError::Shape(format!(
            "raw FP16 fused gate/up kernel requires batch=1, got {}",
            config.batch
        )));
    }

    let expected_weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    let weight_elements = gate_raw_bytes.len() / 2;
    let element_end = element_start
        .checked_add(weight_elements)
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > expected_weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {expected_weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;
    while local_idx < weight_elements {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);
        let (gate_delta, up_delta) = raw_16bit_dot_pair_segment(
            input,
            gate_raw_bytes,
            up_raw_bytes,
            local_idx,
            in_feature,
            row_len,
            dtype,
        )?;
        state.gate_acc += gate_delta;
        state.up_acc += up_delta;

        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }

    Ok(())
}

fn raw_16bit_dot_pair_segment(
    input: &[f32],
    first_raw_bytes: &[u8],
    second_raw_bytes: &[u8],
    local_idx: usize,
    in_feature: usize,
    row_len: usize,
    dtype: rllm_container::DType,
) -> Result<(f32, f32)> {
    let mut first_acc = 0.0f32;
    let mut second_acc = 0.0f32;
    let mut idx = 0usize;
    while idx + 4 <= row_len {
        let byte_idx = (local_idx + idx) * 2;
        let first = &first_raw_bytes[byte_idx..byte_idx + 8];
        let second = &second_raw_bytes[byte_idx..byte_idx + 8];
        let input_start = in_feature + idx;
        let x0 = input[input_start];
        let x1 = input[input_start + 1];
        let x2 = input[input_start + 2];
        let x3 = input[input_start + 3];

        first_acc += x0 * raw_16bit_to_f32(u16::from_le_bytes([first[0], first[1]]), dtype)
            + x1 * raw_16bit_to_f32(u16::from_le_bytes([first[2], first[3]]), dtype)
            + x2 * raw_16bit_to_f32(u16::from_le_bytes([first[4], first[5]]), dtype)
            + x3 * raw_16bit_to_f32(u16::from_le_bytes([first[6], first[7]]), dtype);
        second_acc += x0 * raw_16bit_to_f32(u16::from_le_bytes([second[0], second[1]]), dtype)
            + x1 * raw_16bit_to_f32(u16::from_le_bytes([second[2], second[3]]), dtype)
            + x2 * raw_16bit_to_f32(u16::from_le_bytes([second[4], second[5]]), dtype)
            + x3 * raw_16bit_to_f32(u16::from_le_bytes([second[6], second[7]]), dtype);
        idx += 4;
    }
    while idx < row_len {
        let input_value = input[in_feature + idx];
        first_acc += input_value * raw_16bit_weight_at(first_raw_bytes, local_idx + idx, dtype);
        second_acc += input_value * raw_16bit_weight_at(second_raw_bytes, local_idx + idx, dtype);
        idx += 1;
    }
    Ok((first_acc, second_acc))
}

#[inline(always)]
fn raw_16bit_weight_at(raw_bytes: &[u8], element_idx: usize, dtype: rllm_container::DType) -> f32 {
    let byte_idx = element_idx * 2;
    raw_16bit_to_f32(
        u16::from_le_bytes([raw_bytes[byte_idx], raw_bytes[byte_idx + 1]]),
        dtype,
    )
}

fn accumulate_sparse_raw_16bit_linear_chunk_batch1(
    input: &[f32],
    selected: &[usize],
    output: &mut [f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    weight_name: &str,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Sparse raw 16-bit stream for {weight_name} has odd length"
        )));
    }
    let weight_elements = raw_bytes.len() / 2;
    let element_end = element_start.checked_add(weight_elements).ok_or_else(|| {
        RuntimeError::Shape("sparse raw chunk element range overflow".to_string())
    })?;
    let expected = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("sparse weight element count overflow".to_string()))?;
    if element_end > expected {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} sparse chunk elements {element_start}..{element_end} exceed expected {expected}"
        )));
    }
    if weight_elements == 0 {
        return Ok(());
    }

    let first_row = element_start / config.in_features;
    let last_row = element_end.saturating_sub(1) / config.in_features;
    for (out_feature, out_value) in output
        .iter_mut()
        .enumerate()
        .take(last_row + 1)
        .skip(first_row)
    {
        let row_base = out_feature * config.in_features;
        let mut acc = *out_value;
        for &in_feature in selected {
            let global = row_base + in_feature;
            if global >= element_start && global < element_end {
                let local = global - element_start;
                acc += input[in_feature] * raw_16bit_weight_at(raw_bytes, local, dtype);
            }
        }
        *out_value = acc;
    }
    Ok(())
}

fn parallel_sparse_raw_16bit_linear_chunk_batch1(
    input: &[f32],
    selected: &[usize],
    output: &mut [f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    weight_name: &str,
    threads: usize,
) -> Result<()> {
    if !element_start.is_multiple_of(config.in_features)
        || !(raw_bytes.len() / 2).is_multiple_of(config.in_features)
    {
        return accumulate_sparse_raw_16bit_linear_chunk_batch1(
            input,
            selected,
            output,
            raw_bytes,
            element_start,
            config,
            dtype,
            weight_name,
        );
    }
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Sparse raw 16-bit stream for {weight_name} has odd length"
        )));
    }
    let weight_elements = raw_bytes.len() / 2;
    let element_end = element_start.checked_add(weight_elements).ok_or_else(|| {
        RuntimeError::Shape("parallel sparse raw chunk element range overflow".to_string())
    })?;
    let expected = config.out_features.checked_mul(config.in_features).ok_or_else(|| {
        RuntimeError::Shape("parallel sparse weight element count overflow".to_string())
    })?;
    if element_end > expected {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} parallel sparse chunk elements {element_start}..{element_end} exceed expected {expected}"
        )));
    }

    let first_row = element_start / config.in_features;
    let rows = weight_elements / config.in_features;
    if rows == 0 {
        return Ok(());
    }
    if first_row + rows > output.len() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} parallel sparse rows {}..{} exceed output len {}",
            first_row,
            first_row + rows,
            output.len()
        )));
    }
    let threads = effective_row_block_threads(rows, threads);
    if threads == 1 {
        return accumulate_sparse_raw_16bit_linear_chunk_batch1(
            input,
            selected,
            output,
            raw_bytes,
            element_start,
            config,
            dtype,
            weight_name,
        );
    }

    let rows_per_thread = rows.div_ceil(threads);
    let output_rows = &mut output[first_row..first_row + rows];
    std::thread::scope(|scope| {
        for (thread_idx, output_chunk) in output_rows.chunks_mut(rows_per_thread).enumerate() {
            let row_start = thread_idx * rows_per_thread;
            scope.spawn(move || {
                for (row_offset, out_value) in output_chunk.iter_mut().enumerate() {
                    let local_row_base = (row_start + row_offset) * config.in_features;
                    let mut acc = *out_value;
                    for &in_feature in selected {
                        acc += input[in_feature]
                            * raw_16bit_weight_at(raw_bytes, local_row_base + in_feature, dtype);
                    }
                    *out_value = acc;
                }
            });
        }
    });
    Ok(())
}

fn accumulate_sparse_silu_gate_up_raw_16bit_chunk_batch1(
    input: &[f32],
    selected: &[usize],
    gate_bytes: &[u8],
    up_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    state: &mut SiluGateUpState<'_>,
    weight_name: &str,
) -> Result<()> {
    if !gate_bytes.len().is_multiple_of(2) || !up_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Sparse raw gate/up stream for {weight_name} has odd length"
        )));
    }
    let weight_elements = gate_bytes.len() / 2;
    let element_end = element_start.checked_add(weight_elements).ok_or_else(|| {
        RuntimeError::Shape("sparse gate/up chunk element range overflow".to_string())
    })?;
    let expected = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("sparse gate/up element count overflow".to_string()))?;
    if element_end > expected {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} sparse gate/up chunk elements {element_start}..{element_end} exceed expected {expected}"
        )));
    }
    if weight_elements == 0 {
        return Ok(());
    }

    let first_row = element_start / config.in_features;
    let last_row = element_end.saturating_sub(1) / config.in_features;
    for out_feature in first_row..=last_row {
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic sparse row {out_feature}, current {}",
                state.current_out_feature
            )));
        }

        let row_base = out_feature * config.in_features;
        for &in_feature in selected {
            let global = row_base + in_feature;
            if global >= element_start && global < element_end {
                let local = global - element_start;
                let x = input[in_feature];
                state.gate_acc += x * raw_16bit_weight_at(gate_bytes, local, dtype);
                state.up_acc += x * raw_16bit_weight_at(up_bytes, local, dtype);
            }
        }

        if element_end >= row_base + config.in_features {
            state.finish_current(config, weight_name)?;
        }
    }
    Ok(())
}

fn parallel_sparse_silu_gate_up_raw_16bit_chunk_batch1(
    input: &[f32],
    selected: &[usize],
    gate_bytes: &[u8],
    up_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    output: &mut [f32],
    weight_name: &str,
    threads: usize,
) -> Result<()> {
    if !gate_bytes.len().is_multiple_of(2) || !up_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Parallel sparse raw gate/up stream for {weight_name} has odd length"
        )));
    }
    if gate_bytes.len() != up_bytes.len() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Parallel sparse gate/up stream for {weight_name} has mismatched byte lengths"
        )));
    }
    let weight_elements = gate_bytes.len() / 2;
    if !element_start.is_multiple_of(config.in_features)
        || !weight_elements.is_multiple_of(config.in_features)
    {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Parallel sparse gate/up for {weight_name} requires complete row-aligned chunks"
        )));
    }
    let element_end = element_start.checked_add(weight_elements).ok_or_else(|| {
        RuntimeError::Shape("parallel sparse gate/up chunk element range overflow".to_string())
    })?;
    let expected = config.out_features.checked_mul(config.in_features).ok_or_else(|| {
        RuntimeError::Shape("parallel sparse gate/up element count overflow".to_string())
    })?;
    if element_end > expected {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} parallel sparse gate/up chunk elements {element_start}..{element_end} exceed expected {expected}"
        )));
    }

    let first_row = element_start / config.in_features;
    let rows = weight_elements / config.in_features;
    if rows == 0 {
        return Ok(());
    }
    if first_row + rows > output.len() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} parallel sparse gate/up rows {}..{} exceed output len {}",
            first_row,
            first_row + rows,
            output.len()
        )));
    }
    let threads = effective_row_block_threads(rows, threads);
    if threads == 1 {
        let output_rows = &mut output[first_row..first_row + rows];
        for (row_offset, out_value) in output_rows.iter_mut().enumerate() {
            let local_row_base = row_offset * config.in_features;
            let mut gate_acc = 0.0f32;
            let mut up_acc = 0.0f32;
            for &in_feature in selected {
                let x = input[in_feature];
                gate_acc +=
                    x * raw_16bit_weight_at(gate_bytes, local_row_base + in_feature, dtype);
                up_acc += x * raw_16bit_weight_at(up_bytes, local_row_base + in_feature, dtype);
            }
            *out_value = crate::silu(gate_acc) * up_acc;
        }
        return Ok(());
    }

    let rows_per_thread = rows.div_ceil(threads);
    let output_rows = &mut output[first_row..first_row + rows];
    std::thread::scope(|scope| {
        for (thread_idx, output_chunk) in output_rows.chunks_mut(rows_per_thread).enumerate() {
            let row_start = thread_idx * rows_per_thread;
            scope.spawn(move || {
                for (row_offset, out_value) in output_chunk.iter_mut().enumerate() {
                    let local_row_base = (row_start + row_offset) * config.in_features;
                    let mut gate_acc = 0.0f32;
                    let mut up_acc = 0.0f32;
                    for &in_feature in selected {
                        let x = input[in_feature];
                        gate_acc += x
                            * raw_16bit_weight_at(gate_bytes, local_row_base + in_feature, dtype);
                        up_acc +=
                            x * raw_16bit_weight_at(up_bytes, local_row_base + in_feature, dtype);
                    }
                    *out_value = crate::silu(gate_acc) * up_acc;
                }
            });
        }
    });
    Ok(())
}

#[inline(always)]
fn raw_16bit_to_f32(bits: u16, dtype: rllm_container::DType) -> f32 {
    match dtype {
        rllm_container::DType::Fp16 => crate::tensor::fp16_to_f32(bits),
        rllm_container::DType::Bf16 => crate::tensor::bf16_to_f32(bits),
        _ => unreachable!("raw 16-bit kernel only supports FP16/BF16"),
    }
}

#[inline(always)]
fn fp16_weight_at(raw_bytes: &[u8], element_idx: usize) -> f32 {
    let byte_idx = element_idx * 2;
    crate::tensor::fp16_to_f32(u16::from_le_bytes([
        raw_bytes[byte_idx],
        raw_bytes[byte_idx + 1],
    ]))
}

fn accumulate_multiply_raw_fp16_chunk(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw FP16 stream for {weight_name} has odd length"
        )));
    }

    if config.batch == 1 {
        return accumulate_multiply_raw_fp16_chunk_batch1(
            input,
            raw_bytes,
            element_start,
            config,
            state,
            weight_name,
        );
    }

    let weight_elements = raw_bytes.len() / 2;
    let mut local_idx = 0usize;
    let mut global_idx = element_start;

    const BLOCK_SIZE: usize = 128;
    let mut w_block = [0.0f32; BLOCK_SIZE];

    while local_idx < weight_elements {
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);

        let mut row_idx = 0usize;
        while row_idx < row_len {
            let block_len = (row_len - row_idx).min(BLOCK_SIZE);
            let byte_start = (local_idx + row_idx) * 2;
            let block_bytes = &raw_bytes[byte_start..byte_start + block_len * 2];

            let mut idx = 0usize;
            while idx + 4 <= block_len {
                let bytes = &block_bytes[idx * 2..idx * 2 + 8];
                w_block[idx] = crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                w_block[idx + 1] =
                    crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
                w_block[idx + 2] =
                    crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[4], bytes[5]]));
                w_block[idx + 3] =
                    crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[6], bytes[7]]));
                idx += 4;
            }
            while idx < block_len {
                let bytes = &block_bytes[idx * 2..idx * 2 + 2];
                w_block[idx] = crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                idx += 1;
            }

            accumulate_weight_chunk_multiply_into(
                input,
                &w_block[..block_len],
                global_idx + row_idx,
                config,
                state,
                weight_name,
            )?;

            row_idx += block_len;
        }

        local_idx += row_len;
        global_idx += row_len;
    }

    Ok(())
}

fn accumulate_multiply_raw_fp16_chunk_batch1(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<()> {
    accumulate_multiply_raw_fp16_chunk_batch1_row_blocked(
        input,
        raw_bytes,
        element_start,
        config,
        state,
        weight_name,
    )
}

fn accumulate_multiply_raw_fp16_chunk_batch1_row_blocked(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw FP16 stream for {weight_name} has odd length"
        )));
    }

    if config.batch != 1 {
        return Err(RuntimeError::Shape(format!(
            "raw FP16 batch1 multiply row-block kernel requires batch=1, got {}",
            config.batch
        )));
    }

    let expected_weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    let weight_elements = raw_bytes.len() / 2;
    let element_end = element_start
        .checked_add(weight_elements)
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > expected_weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {expected_weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;

    while local_idx < weight_elements && !global_idx.is_multiple_of(config.in_features) {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);
        let mut acc = state.current_acc[0];
        acc += raw_fp16_dot_segment(input, raw_bytes, local_idx, in_feature, row_len)?;
        state.current_acc[0] = acc;
        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }

    let row_block_elements = config
        .in_features
        .checked_mul(4)
        .ok_or_else(|| RuntimeError::Shape("row block element count overflow".to_string()))?;
    while local_idx + row_block_elements <= weight_elements {
        let out_feature = global_idx / config.in_features;
        if out_feature + 3 >= config.out_features {
            break;
        }
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let mut acc0 = state.current_acc[0];
        let mut acc1 = state
            .bias
            .and_then(|values| values.get(out_feature + 1))
            .copied()
            .unwrap_or(0.0);
        let mut acc2 = state
            .bias
            .and_then(|values| values.get(out_feature + 2))
            .copied()
            .unwrap_or(0.0);
        let mut acc3 = state
            .bias
            .and_then(|values| values.get(out_feature + 3))
            .copied()
            .unwrap_or(0.0);
        let row0_start = local_idx;
        let row1_start = local_idx + config.in_features;
        let row2_start = row1_start + config.in_features;
        let row3_start = row2_start + config.in_features;

        let mut idx = 0usize;
        while idx < config.in_features {
            let x = input[idx];
            acc0 += x * fp16_weight_at(raw_bytes, row0_start + idx);
            acc1 += x * fp16_weight_at(raw_bytes, row1_start + idx);
            acc2 += x * fp16_weight_at(raw_bytes, row2_start + idx);
            acc3 += x * fp16_weight_at(raw_bytes, row3_start + idx);
            idx += 1;
        }

        state.target[out_feature] *= acc0;
        state.target[out_feature + 1] *= acc1;
        state.target[out_feature + 2] *= acc2;
        state.target[out_feature + 3] *= acc3;
        state.current_out_feature += 4;
        if state.current_out_feature < config.out_features {
            let next = state
                .bias
                .and_then(|values| values.get(state.current_out_feature))
                .copied()
                .unwrap_or(0.0);
            state.current_acc[0] = next;
        }
        local_idx += row_block_elements;
        global_idx += row_block_elements;
    }

    while local_idx < weight_elements {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);
        let mut acc = state.current_acc[0];
        acc += raw_fp16_dot_segment(input, raw_bytes, local_idx, in_feature, row_len)?;
        state.current_acc[0] = acc;
        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }

    Ok(())
}

#[cfg(all(test, target_arch = "aarch64"))]
mod r119_panel_tests {
    use super::*;

    // Build a deterministic Q8_0 weight chunk: out_rows × in_features, each
    // 32-block = 2-byte fp16 scale (0.125) + 32 i8.
    pub fn make_q8_pub(out_rows: usize, in_features: usize) -> Vec<u8> { make_q8(out_rows, in_features) }
    fn make_q8(out_rows: usize, in_features: usize) -> Vec<u8> {
        let bpr = in_features / 32;
        let mut bytes = Vec::new();
        for o in 0..out_rows {
            for b in 0..bpr {
                bytes.extend_from_slice(&crate::tensor::f32_to_fp16(0.125).to_le_bytes());
                for k in 0..32 {
                    let q = (((o * 7 + b * 5 + k * 3) as i16 % 17) - 8) as i8;
                    bytes.push(q as u8);
                }
            }
        }
        bytes
    }

    pub fn make_input_pub(batch: usize, in_features: usize) -> Vec<f32> { make_input(batch, in_features) }
    fn make_input(batch: usize, in_features: usize) -> Vec<f32> {
        (0..batch * in_features)
            .map(|i| (i as f32 % 91.0) * 0.00390625 - 0.17)
            .collect()
    }

    fn run_panel_vs_r111(batch: usize, in_features: usize, out_features: usize) {
        if !q8_i8mm_available() {
            return;
        }
        let q8 = make_q8(out_features, in_features);
        let input = make_input(batch, in_features);
        let config = StreamingLinearConfig {
            batch,
            in_features,
            out_features,
        };
        let we = out_features * in_features;

        let mut out_ref = vec![0.0f32; batch * out_features];
        accumulate_q8_0_chunk_int8_activation(&input, &mut out_ref, &q8, 0, config, we).unwrap();

        let mut out_panel = vec![0.0f32; batch * out_features];
        let used =
            accumulate_q8_0_chunk_panel_smmla(&input, &mut out_panel, &q8, 0, config).unwrap();
        assert!(used, "panel path should engage for batch={batch}");

        let mut max_diff = 0.0f32;
        let mut worst = (0, 0);
        for t in 0..batch {
            for o in 0..out_features {
                let d = (out_ref[t * out_features + o] - out_panel[t * out_features + o]).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (t, o);
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "panel vs r111 mismatch batch={batch} out={out_features}: max_diff={max_diff} at row {} col {} (ref={} panel={})",
            worst.0,
            worst.1,
            out_ref[worst.0 * out_features + worst.1],
            out_panel[worst.0 * out_features + worst.1],
        );
    }

    #[test]
    fn panel_matches_r111_even_batch_even_out() {
        run_panel_vs_r111(4, 64, 4);
    }

    #[test]
    fn panel_matches_r111_odd_batch() {
        run_panel_vs_r111(3, 64, 4);
    }

    #[test]
    fn panel_matches_r111_odd_out() {
        run_panel_vs_r111(4, 64, 3);
    }

    #[test]
    fn panel_matches_r111_odd_both() {
        run_panel_vs_r111(5, 64, 5);
    }

    #[test]
    fn panel_matches_r111_realistic_shape() {
        run_panel_vs_r111(55, 2048, 8);
    }

    // R124 octet boundaries: exercise output-octet + pair-remainder + odd-row
    // tails for both even and odd batch. out_features chosen to hit each split:
    // 8=1 octet; 10=octet+pair; 11=octet+pair+odd; 17=2 octets+odd; 22=2oct+3pair.
    #[test]
    fn octet_even_batch_boundaries() {
        for out in [8, 9, 10, 11, 12, 15, 16, 17, 22, 24] {
            run_panel_vs_r111(54, 256, out);
        }
    }

    #[test]
    fn octet_odd_batch_boundaries() {
        for out in [8, 9, 10, 11, 12, 15, 16, 17, 22, 24] {
            run_panel_vs_r111(53, 256, out);
        }
    }

    #[test]
    fn octet_realistic_multi_octet() {
        run_panel_vs_r111(53, 2048, 64);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod r119_panel_multichunk_tests {
    use super::*;
    use super::r119_panel_tests::*;

    // Process a full matmul as several row-chunks (each chunk = a sub-range of
    // output rows at its own element_start), comparing panel vs R111.
    fn run_multichunk(batch: usize, in_features: usize, out_features: usize, chunk_rows: usize) {
        if !q8_i8mm_available() {
            return;
        }
        let bpr = in_features / 32;
        let q8_full = make_q8_pub(out_features, in_features);
        let input = make_input_pub(batch, in_features);
        let config = StreamingLinearConfig { batch, in_features, out_features };

        let mut out_ref = vec![0.0f32; batch * out_features];
        let mut out_panel = vec![0.0f32; batch * out_features];

        let we = out_features * in_features;
        let mut row = 0;
        while row < out_features {
            let rows = chunk_rows.min(out_features - row);
            let elem_start = row * in_features;
            let byte_start = row * bpr * 34;
            let byte_end = (row + rows) * bpr * 34;
            let chunk = &q8_full[byte_start..byte_end];

            accumulate_q8_0_chunk_int8_activation(&input, &mut out_ref, chunk, elem_start, config, we).unwrap();
            let used = accumulate_q8_0_chunk_panel_smmla(&input, &mut out_panel, chunk, elem_start, config).unwrap();
            assert!(used, "panel should engage chunk at row {row}");
            row += rows;
        }

        let mut max_diff = 0.0f32;
        let mut worst = (0, 0);
        for t in 0..batch {
            for o in 0..out_features {
                let d = (out_ref[t * out_features + o] - out_panel[t * out_features + o]).abs();
                if d > max_diff { max_diff = d; worst = (t, o); }
            }
        }
        assert!(max_diff < 1e-3,
            "multichunk panel vs r111 mismatch b={batch} out={out_features} chunk_rows={chunk_rows}: max_diff={max_diff} at row {} col {} (ref={} panel={})",
            worst.0, worst.1, out_ref[worst.0*out_features+worst.1], out_panel[worst.0*out_features+worst.1]);
    }

    #[test]
    fn multichunk_even_chunk_rows() { run_multichunk(55, 2048, 8, 4); }
    #[test]
    fn multichunk_odd_chunk_rows() { run_multichunk(55, 2048, 8, 3); }
    #[test]
    fn multichunk_single_row_chunks() { run_multichunk(55, 2048, 6, 1); }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod r119_panel_realchunk_tests {
    use super::*;
    use super::r119_panel_tests::*;

    // Replicate the real q_proj chunk pattern: batch=53, in=2048, out=2048,
    // chunks of 481,481,481,481,124 output rows.
    fn run_pattern(batch: usize, in_features: usize, out_features: usize, chunk_pattern: &[usize]) {
        if !q8_i8mm_available() { return; }
        assert_eq!(chunk_pattern.iter().sum::<usize>(), out_features);
        let bpr = in_features / 32;
        let q8_full = make_q8_pub(out_features, in_features);
        let input = make_input_pub(batch, in_features);
        let config = StreamingLinearConfig { batch, in_features, out_features };
        let we = out_features * in_features;

        let mut out_ref = vec![0.0f32; batch * out_features];
        let mut out_panel = vec![0.0f32; batch * out_features];
        let mut row = 0;
        for &rows in chunk_pattern {
            let elem_start = row * in_features;
            let chunk = &q8_full[row * bpr * 34..(row + rows) * bpr * 34];
            accumulate_q8_0_chunk_int8_activation(&input, &mut out_ref, chunk, elem_start, config, we).unwrap();
            let used = accumulate_q8_0_chunk_panel_smmla(&input, &mut out_panel, chunk, elem_start, config).unwrap();
            assert!(used, "panel should engage chunk at row {row} rows {rows}");
            row += rows;
        }
        let mut max_diff = 0.0f32; let mut worst = (0usize, 0usize);
        for t in 0..batch { for o in 0..out_features {
            let d = (out_ref[t*out_features+o]-out_panel[t*out_features+o]).abs();
            if d > max_diff { max_diff = d; worst = (t,o); }
        }}
        assert!(max_diff < 1e-3,
            "REAL pattern mismatch b={batch} out={out_features}: max_diff={max_diff} at row {} col {} (ref={} panel={})",
            worst.0, worst.1, out_ref[worst.0*out_features+worst.1], out_panel[worst.0*out_features+worst.1]);
    }

    #[test]
    fn real_qproj_b53() { run_pattern(53, 2048, 2048, &[481,481,481,481,124]); }
    #[test]
    fn real_kvproj_b53() { run_pattern(53, 2048, 512, &[481, 31]); }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod r119_panel_in8192_tests {
    use super::*;
    use super::r119_panel_tests::*;

    fn run(batch: usize, in_features: usize, out_features: usize) {
        if !q8_i8mm_available() { return; }
        let q8 = make_q8_pub(out_features, in_features);
        let input = make_input_pub(batch, in_features);
        let config = StreamingLinearConfig { batch, in_features, out_features };
        let we = out_features * in_features;
        let mut out_ref = vec![0.0f32; batch * out_features];
        accumulate_q8_0_chunk_int8_activation(&input, &mut out_ref, &q8, 0, config, we).unwrap();
        let mut out_panel = vec![0.0f32; batch * out_features];
        let used = accumulate_q8_0_chunk_panel_smmla(&input, &mut out_panel, &q8, 0, config).unwrap();
        assert!(used);
        let mut md = 0.0f32; let mut worst=(0,0);
        for t in 0..batch { for o in 0..out_features {
            let d=(out_ref[t*out_features+o]-out_panel[t*out_features+o]).abs();
            if d>md { md=d; worst=(t,o); }
        }}
        assert!(md < 1e-3, "in={in_features} mismatch: max_diff={md} at row {} col {} (ref={} panel={})",
            worst.0, worst.1, out_ref[worst.0*out_features+worst.1], out_panel[worst.0*out_features+worst.1]);
    }

    #[test] fn down_in8192() { run(53, 8192, 4); }
    #[test] fn down_in8192_realistic() { run(53, 8192, 64); }
}

/// R121: the multiply-into fast path (`try_panel_multiply_into_up` +
/// `target *= up + bias`). Validates the two pieces R121 actually adds on top of
/// the already-tested panel kernel: (1) accumulating the panel into a scratch
/// buffer chunk-by-chunk with a threaded `element_start`, and (2) the caller's
/// `target *= up + bias` arithmetic — both against the whole-weight int8
/// reference so the comparison is tight (same int8 dot on both sides), not a
/// quant-tolerance check.
#[cfg(all(test, target_arch = "aarch64"))]
mod r121_multiply_into_tests {
    use super::*;
    use super::r119_panel_tests::*;

    fn run(batch: usize, in_features: usize, out_features: usize, chunk_rows: usize) {
        if !q8_i8mm_available() {
            return;
        }
        let q8 = make_q8_pub(out_features, in_features);
        let input = make_input_pub(batch, in_features);
        let bias: Vec<f32> = (0..out_features).map(|f| (f as f32) * 0.013 - 0.07).collect();
        let config = StreamingLinearConfig { batch, in_features, out_features };

        // Reference up over the full weight via the int8-activation path.
        let we = out_features * in_features;
        let mut up_ref = vec![0.0f32; batch * out_features];
        accumulate_q8_0_chunk_int8_activation(&input, &mut up_ref, &q8, 0, config, we).unwrap();

        // Panel up accumulated chunk-by-chunk, mirroring try_panel_multiply_into_up:
        // each chunk covers `chunk_rows` output rows, with element_start derived
        // from the running byte offset exactly like chunk_element_start_for_dtype.
        let blocks_per_row = in_features / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut up_panel = vec![0.0f32; batch * out_features];
        let mut row = 0usize;
        let mut byte_offset = 0usize;
        while row < out_features {
            let rows = chunk_rows.min(out_features - row);
            let start = row * bytes_per_row;
            let end = start + rows * bytes_per_row;
            let element_start = (byte_offset / 34) * 32;
            let used = accumulate_q8_0_chunk_panel_smmla(
                &input,
                &mut up_panel,
                &q8[start..end],
                element_start,
                config,
            )
            .unwrap();
            assert!(used, "panel should engage for chunk at row {row}");
            byte_offset += end - start;
            row += rows;
        }

        // Apply the caller's multiply-into on both, then compare end results.
        let init: Vec<f32> = (0..batch * out_features)
            .map(|i| (i as f32 % 13.0) * 0.1 + 0.3)
            .collect();
        let mut max = 0.0f32;
        let mut worst = (0, 0);
        for b in 0..batch {
            for f in 0..out_features {
                let idx = b * out_features + f;
                let tgt_ref = init[idx] * (up_ref[idx] + bias[f]);
                let tgt_panel = init[idx] * (up_panel[idx] + bias[f]);
                let d = (tgt_ref - tgt_panel).abs();
                if d > max {
                    max = d;
                    worst = (b, f);
                }
            }
        }
        assert!(
            max < 1e-2,
            "multiply-into panel vs ref max_diff={max} at row {} col {}",
            worst.0,
            worst.1
        );
    }

    #[test] fn multiply_into_even() { run(4, 64, 8, 4); }
    #[test] fn multiply_into_odd_out_and_chunks() { run(5, 64, 7, 3); }
    #[test] fn multiply_into_realistic_up() { run(53, 2048, 16, 8); }
    #[test] fn multiply_into_single_chunk_full() { run(53, 2048, 32, 32); }
}
