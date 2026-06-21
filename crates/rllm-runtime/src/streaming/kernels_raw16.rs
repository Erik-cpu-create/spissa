// Dense fused raw bf16/fp16/16-bit matmul (rle + chunk/batch1/row-blocked) + SiLU-gate-up
// + 16-bit dot-segment helpers. sparse -> kernels_raw16_sparse.rs, multiply-into ->
// kernels_raw16_multiply.rs (R169 split). include!d into mod.rs.

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
    // In --fast (int8-activation) mode, bf16 rows can use the NEON dot kernel
    // (exact (bits<<16) upcast + vectorized FMA — the R137 lm_head kernel). The
    // EXACT default keeps the scalar accumulation so its bit-for-bit tests hold;
    // the NEON path differs only in f32 accumulation order (a few ULPs).
    let fast_bf16 = q8_activation_path_enabled() && matches!(dtype, rllm_container::DType::Bf16);
    let bf16_act = fast_bf16.then(|| Bf16DotActivation::new(input));

    // R173: pool the full-row NEON bf16 dots across the persistent decode pool.
    // After the prologue `global_idx` is row-aligned, so [first_full, last_full) are
    // whole output rows fully contained in this chunk — each row is touched only by
    // the chunk that fully contains it, so the per-row writes are disjoint. Pooling
    // these recovers the multi-core throughput the bf16-resident (rANS/raw-bf16)
    // decode path was missing (it ran single-threaded). When it runs it consumes all
    // full rows; the main 4-row-block loop below then no-ops and the tail handles the
    // partial last row. Bit-identical to the serial path (per-row dot, no reorder).
    #[cfg(target_arch = "aarch64")]
    if let Some(act) = &bf16_act {
        let n = config.in_features;
        debug_assert_eq!(global_idx % n, 0, "prologue must leave global_idx row-aligned");
        let first_full = global_idx / n;
        let last_full = element_end / n;
        let full_rows = last_full.saturating_sub(first_full);
        let pool = decode_pool();
        if pool.size() > 1 && full_rows >= 2 * MIN_ROWS_PER_PARALLEL_ARGMAX {
            let chunk_rows = (full_rows / (pool.size() * 4)).max(MIN_ROWS_PER_PARALLEL_ARGMAX);
            let n_tasks = full_rows.div_ceil(chunk_rows);
            let out_base = DisjointMut(output.as_mut_ptr());
            let es = element_start;
            pool.parallel_for(n_tasks, |t| {
                let lo = first_full + t * chunk_rows;
                let hi = (lo + chunk_rows).min(last_full);
                for r in lo..hi {
                    let local = r * n - es;
                    let d = act.row_dot(&raw_bytes[local * 2..(local + n) * 2], n);
                    // SAFETY: row r is fully in this chunk and unique to this task's
                    // range, so this is the only write to output[r] (disjoint).
                    unsafe { *out_base.at(r) += d };
                }
            });
            let consumed = full_rows * n;
            local_idx += consumed;
            global_idx += consumed;
        }
    }

    while local_idx + row_block_elements <= weight_elements {
        let out_feature = global_idx / config.in_features;
        if out_feature + 3 >= config.out_features {
            break;
        }

        let row0_start = local_idx;
        let row1_start = local_idx + config.in_features;
        let row2_start = row1_start + config.in_features;
        let row3_start = row2_start + config.in_features;

        if let Some(act) = &bf16_act {
            let n = config.in_features;
            let dot = |rs: usize| act.row_dot(&raw_bytes[rs * 2..(rs + n) * 2], n);
            output[out_feature] += dot(row0_start);
            output[out_feature + 1] += dot(row1_start);
            output[out_feature + 2] += dot(row2_start);
            output[out_feature + 3] += dot(row3_start);
        } else {
            let mut acc0 = output[out_feature];
            let mut acc1 = output[out_feature + 1];
            let mut acc2 = output[out_feature + 2];
            let mut acc3 = output[out_feature + 3];
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
        }
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

