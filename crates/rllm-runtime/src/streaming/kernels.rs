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
    for local_row in 0..row_count {
        let out_feature = first_row + local_row;
        let mut acc = output[out_feature];
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
        output[out_feature] = acc;
    }
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
