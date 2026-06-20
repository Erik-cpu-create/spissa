// Sparse raw-16bit linear + sparse SiLU-gate-up kernels (batch1 + parallel).
// Split out of kernels_raw16.rs (R169); include!d into streaming/mod.rs.

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

