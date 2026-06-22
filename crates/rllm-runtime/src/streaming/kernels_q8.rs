// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

// Q8 int8 low-level primitives: activation / i8mm availability, per-segment quantize,
// int8 dot, activation & weight panel packing. matmul -> kernels_q8_matmul.rs,
// helpers+f32-fallback -> kernels_q8_support.rs (R168 split). include!d into mod.rs.

/// REEBORN-Q8 NEON fast path: int8×int8 `sdot` (Q8) / vectorized `vfmaq` (bf16)
/// dot instead of the scalar f32-dequant fallback. R171: **default ON** — ~9× faster
/// and token-parity-validated (q8 near-exact quant-only diff; bf16 argmax-preserving).
/// Opt OUT with `RLLM_Q8_ACTIVATION=0`/`false`/`no`/`off` to force the scalar path.
fn q8_activation_path_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        match std::env::var(Q8_ACTIVATION_ENV)
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
        {
            // Explicit opt-out only; unset / empty / any other value → fast path on.
            Some(v) => !matches!(v.as_str(), "0" | "false" | "no" | "off"),
            None => true,
        }
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
        // Interleave the two rows' 32 int8 weights into the panel at 8-byte
        // segment granularity: [r0 seg0, r1 seg0, r0 seg1, r1 seg1, ...]. NEON
        // moves each 8-byte segment in one load/store instead of 8 scalar bytes;
        // the byte values are identical (q8 is already int8, just reinterpreted).
        let pbase = b * 64;
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            // SAFETY: each (off + 2 + seg*8 + 8) <= off + 34 is in bounds of the
            // block, and pbase + 4*16 == (b+1)*64 <= panel.len() (2*in_features).
            unsafe {
                let src0 = q8_bytes.as_ptr().add(off0 + 2);
                let src1 = q8_bytes.as_ptr().add(off1 + 2);
                let dst = panel.as_mut_ptr().add(pbase) as *mut u8;
                for seg in 0..4 {
                    let v0 = vld1_u8(src0.add(seg * 8));
                    let v1 = vld1_u8(src1.add(seg * 8));
                    vst1_u8(dst.add(seg * 16), v0);
                    vst1_u8(dst.add(seg * 16 + 8), v1);
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for seg in 0..4 {
            let dst = pbase + seg * 16;
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

