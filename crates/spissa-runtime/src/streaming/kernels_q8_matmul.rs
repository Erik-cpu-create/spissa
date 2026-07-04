// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

// Q8 int8 / SMMLA matmul kernel family (accumulate_q8_0_chunk*: decode batch1, prefill
// panel, parallel, multiply-into, fused-argmax). Split out of kernels_q8.rs (R168); include!d into mod.rs.

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
    #[allow(clippy::needless_return)] // `return` required: the aarch64 block below is the fn tail
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
    #[allow(clippy::needless_range_loop)] // b drives raw pointer offsets across parallel buffers
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

    // batch=1 decode GEMV: REEPOOL persistent workers + work-stealing over row
    // chunks (R172). The old per-call `thread::scope` spawned OS threads ~once per
    // projection (~182/token) — that spawn dominated on mobile schedulers (measured:
    // single-thread BEAT 2-thread). Amortizing spawn via the persistent pool lets the
    // cores actually help (microbench 1.8 -> 4.7 GB/s). RLLM_THREADS=1 -> serial inline.
    let pool = decode_pool();
    if pool.size() <= 1 || out_features < 2 * MIN_ROWS_PER_PARALLEL_Q8_PREFILL {
        sdot_int8_batch1_rows_range(output, q8_bytes, &act_i8, &act_scales, 0, blocks_per_row);
        return Ok(());
    }
    // Oversubscribe ~4x the pool so fast P-cores steal more tasks than slow E-cores.
    let chunk_rows = (out_features / (pool.size() * 4))
        .max(MIN_ROWS_PER_PARALLEL_Q8_PREFILL)
        .max(1);
    let n_tasks = out_features.div_ceil(chunk_rows);
    let out_base = DisjointMut(output.as_mut_ptr());
    let act_i8 = &act_i8;
    let act_scales = &act_scales;
    pool.parallel_for(n_tasks, |t| {
        let base = t * chunk_rows;
        let rows = chunk_rows.min(out_features - base);
        // SAFETY: tasks own disjoint row ranges [base, base+rows) of `output`.
        let chunk = unsafe { std::slice::from_raw_parts_mut(out_base.at(base), rows) };
        sdot_int8_batch1_rows_range(chunk, q8_bytes, act_i8, act_scales, base, blocks_per_row);
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
    // The int8/panel kernels assume each 32-wide q8 block stays within one output row
    // (in_features % 32 == 0). For non-multiples a block straddles rows, which the
    // int8-activation path does not handle — route those to the scalar path below.
    // Real transformer projections are always %32==0; this only guards odd test/edge dims.
    if q8_activation_path_enabled() && config.in_features.is_multiple_of(32) {
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
    if !next_delta.is_multiple_of(32) {
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

