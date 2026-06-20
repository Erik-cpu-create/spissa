// Streaming linear with fused argmax (no full-logit materialization) + candidate-row
// kernels. Split out of linear.rs (R167); include!d into mod.rs.

/// Streaming single-row linear argmax without materializing full output logits.
///
/// This is intended for CLI/generation argmax sampling paths where callers only
/// need the best output row, not the full `[batch=1, out_features]` logits. It
/// preserves the same row-major accumulation order as `streaming_tile_linear_from_model`
/// for `batch=1`, including rows split across chunks/tiles.
pub fn streaming_tile_linear_argmax_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<usize> {
    streaming_tile_linear_argmax_with_rolling_from_model(
        model,
        weight_name,
        input,
        bias,
        config,
        budget,
        None,
    )
}

/// Experimental argmax over only the first `prefix_out_features` rows.
///
/// This is an approximate research path for LM-head shortlist experiments. It
/// keeps the stored weight tensor unchanged but streams only a row prefix.
pub fn streaming_tile_linear_argmax_prefix_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    prefix_out_features: usize,
    budget: &mut MemoryBudget,
) -> Result<usize> {
    validate_tile_linear_config(config)?;
    if prefix_out_features == 0 || prefix_out_features > config.linear.out_features {
        return Err(RuntimeError::Shape(format!(
            "streaming linear prefix argmax rows {prefix_out_features} must be in 1..={}",
            config.linear.out_features
        )));
    }
    if config.linear.batch != 1 {
        return Err(RuntimeError::Shape(format!(
            "streaming linear prefix argmax requires batch=1, got {}",
            config.linear.batch
        )));
    }

    let prefix_linear = StreamingLinearConfig {
        batch: config.linear.batch,
        in_features: config.linear.in_features,
        out_features: prefix_out_features,
    };
    validate_linear_shapes(input, bias, prefix_linear)?;

    let tensor = model.tensor(weight_name)?.clone();
    if tensor.shape.len() != 2 {
        return Err(RuntimeError::Shape(format!(
            "weight tensor {} must be rank-2 [out,in], got {:?}",
            tensor.name, tensor.shape
        )));
    }
    let tensor_out = usize::try_from(tensor.shape[0])
        .map_err(|_| RuntimeError::Shape("weight out_features overflows usize".to_string()))?;
    let tensor_in = usize::try_from(tensor.shape[1])
        .map_err(|_| RuntimeError::Shape("weight in_features overflows usize".to_string()))?;
    if tensor_out != config.linear.out_features || tensor_in != config.linear.in_features {
        return Err(RuntimeError::Shape(format!(
            "weight tensor {} shape {:?} does not match requested [{}, {}]",
            tensor.name, tensor.shape, config.linear.out_features, config.linear.in_features
        )));
    }

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    if chunks.is_empty() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} has no chunks"
        )));
    }

    let dtype_size = tensor.dtype.size_bytes();
    let prefix_weight_bytes = prefix_out_features
        .checked_mul(config.linear.in_features)
        .and_then(|elements| elements.checked_mul(dtype_size))
        .ok_or_else(|| RuntimeError::Shape("prefix weight byte size overflow".to_string()))?;
    let mut byte_offset = 0usize;
    let mut streamed_prefix_bytes = 0usize;
    let mut state = StreamingLinearArgmaxState::new(bias);

    for chunk in chunks {
        if streamed_prefix_bytes >= prefix_weight_bytes {
            break;
        }
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} prefix stream reached unaligned byte offset {byte_offset} for dtype size {dtype_size}"
            )));
        }
        let element_start = byte_offset / dtype_size;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;
        let remaining_prefix_bytes = prefix_weight_bytes - streamed_prefix_bytes;
        let bytes_to_stream = expected_chunk_bytes.min(remaining_prefix_bytes);

        if chunk.codec_id == "rtc-raw-v1"
            && matches!(
                tensor.dtype,
                rllm_container::DType::Fp16 | rllm_container::DType::Bf16
            )
        {
            model.with_raw_chunk(chunk.chunk_id, budget, |compressed_bytes, _budget| {
                if compressed_bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} raw byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        compressed_bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                accumulate_raw_16bit_chunk_argmax(
                    input,
                    &compressed_bytes[..bytes_to_stream],
                    element_start,
                    prefix_linear,
                    tensor.dtype,
                    &mut state,
                    weight_name,
                    None,
                )
            })?;
        } else {
            model.with_decoded_chunk(chunk.chunk_id, budget, |bytes, budget| {
                if bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} decoded byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                if bytes_to_stream % dtype_size != 0 {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "prefix byte len {bytes_to_stream} is not aligned to dtype size {dtype_size}"
                    )));
                }

                let elements_to_stream = bytes_to_stream / dtype_size;
                let scratch_bytes = elements_to_stream
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| RuntimeError::Shape("prefix f32 scratch overflow".to_string()))?;
                let scratch_label = format!(
                    "streaming prefix argmax f32 scratch chunk {} elements 0..{}",
                    chunk.chunk_id, elements_to_stream
                );
                budget.reserve(scratch_bytes, scratch_label.clone())?;
                let weights = match decode_to_f32(tensor.dtype, &bytes[..bytes_to_stream]) {
                    Ok(values) => values,
                    Err(err) => {
                        budget.release(scratch_bytes, scratch_label)?;
                        return Err(err);
                    }
                };
                let result = accumulate_weight_chunk_argmax(
                    input,
                    &weights,
                    element_start,
                    prefix_linear,
                    &mut state,
                    weight_name,
                );
                drop(weights);
                budget.release(scratch_bytes, scratch_label)?;
                result
            })?;
        }

        streamed_prefix_bytes = streamed_prefix_bytes
            .checked_add(bytes_to_stream)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData("prefix byte stream overflow".to_string())
            })?;
        byte_offset = byte_offset
            .checked_add(expected_chunk_bytes)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData("chunk byte offset overflow".to_string())
            })?;
    }

    if streamed_prefix_bytes != prefix_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {streamed_prefix_bytes} prefix bytes, expected {prefix_weight_bytes}"
        )));
    }

    state.finish(prefix_linear, weight_name)
}

pub fn streaming_tile_linear_argmax_candidate_rows_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    candidate_rows: &[usize],
    budget: &mut MemoryBudget,
) -> Result<Option<usize>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, bias, config.linear)?;
    if config.linear.batch != 1 || candidate_rows.is_empty() {
        return Ok(None);
    }

    let tensor = model.tensor(weight_name)?.clone();
    validate_weight_tensor(&tensor, config.linear)?;
    if !matches!(
        tensor.dtype,
        rllm_container::DType::Fp16 | rllm_container::DType::Bf16
    ) {
        return Ok(None);
    }

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    if chunks.is_empty() || chunks.iter().any(|chunk| chunk.codec_id != "rtc-raw-v1") {
        return Ok(None);
    }

    let mut rows = candidate_rows.to_vec();
    rows.sort_unstable();
    rows.dedup();
    for &row in &rows {
        if row >= config.linear.out_features {
            return Err(RuntimeError::Shape(format!(
                "candidate row {row} out of range for {} output features",
                config.linear.out_features
            )));
        }
    }

    let mut scores: Vec<(usize, f32)> = rows
        .iter()
        .map(|&row| {
            let initial = bias
                .and_then(|values| values.get(row))
                .copied()
                .unwrap_or(0.0);
            (row, initial)
        })
        .collect();
    let dtype_size = tensor.dtype.size_bytes();
    let expected_weight_bytes = config
        .linear
        .out_features
        .checked_mul(config.linear.in_features)
        .and_then(|elements| elements.checked_mul(dtype_size))
        .ok_or_else(|| {
            RuntimeError::Shape("candidate row weight byte size overflow".to_string())
        })?;

    let mut byte_offset = 0usize;
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} candidate row stream reached unaligned byte offset {byte_offset}"
            )));
        }
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;
        let element_start = byte_offset / dtype_size;
        let chunk_elements = expected_chunk_bytes / dtype_size;
        let element_end = element_start.checked_add(chunk_elements).ok_or_else(|| {
            RuntimeError::InvalidTensorData("candidate row chunk element end overflow".to_string())
        })?;

        model.with_raw_chunk(chunk.chunk_id, budget, |raw_bytes, _budget| {
            if raw_bytes.len() != expected_chunk_bytes {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "chunk {} raw byte len {} does not match metadata {}",
                    chunk.chunk_id,
                    raw_bytes.len(),
                    expected_chunk_bytes
                )));
            }
            accumulate_candidate_rows_raw_16bit_chunk(
                input,
                raw_bytes,
                element_start,
                element_end,
                config.linear,
                tensor.dtype,
                weight_name,
                &mut scores,
            )
        })?;

        byte_offset = byte_offset
            .checked_add(expected_chunk_bytes)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData(
                    "candidate row chunk byte offset overflow".to_string(),
                )
            })?;
    }

    if byte_offset != expected_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {byte_offset} candidate row bytes, expected {expected_weight_bytes}"
        )));
    }

    let mut best = scores.first().copied().ok_or_else(|| {
        RuntimeError::InvalidTensorData("empty candidate row score set".to_string())
    })?;
    for &(row, score) in scores.iter().skip(1) {
        if score > best.1 {
            best = (row, score);
        }
    }
    Ok(Some(best.0))
}

pub fn streaming_tile_linear_argmax_candidate_rows_range_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    candidate_rows: &[usize],
    budget: &mut MemoryBudget,
) -> Result<Option<usize>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, bias, config.linear)?;
    if config.linear.batch != 1 || candidate_rows.is_empty() {
        return Ok(None);
    }

    let tensor = model.tensor(weight_name)?.clone();
    validate_weight_tensor(&tensor, config.linear)?;
    if !matches!(
        tensor.dtype,
        rllm_container::DType::Fp16 | rllm_container::DType::Bf16
    ) {
        return Ok(None);
    }

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    if chunks.is_empty() {
        return Ok(None);
    }

    let mut rows = candidate_rows.to_vec();
    rows.sort_unstable();
    rows.dedup();
    for &row in &rows {
        if row >= config.linear.out_features {
            return Err(RuntimeError::Shape(format!(
                "candidate row {row} out of range for {} output features",
                config.linear.out_features
            )));
        }
    }

    let mut scores: Vec<(usize, f32)> = rows
        .iter()
        .map(|&row| {
            let initial = bias
                .and_then(|values| values.get(row))
                .copied()
                .unwrap_or(0.0);
            (row, initial)
        })
        .collect();
    let dtype_size = tensor.dtype.size_bytes();
    let expected_weight_bytes = config
        .linear
        .out_features
        .checked_mul(config.linear.in_features)
        .and_then(|elements| elements.checked_mul(dtype_size))
        .ok_or_else(|| {
            RuntimeError::Shape("candidate row weight byte size overflow".to_string())
        })?;

    let mut byte_offset = 0usize;
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} candidate row range stream reached unaligned byte offset {byte_offset}"
            )));
        }
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;
        if !expected_chunk_bytes.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "chunk {} byte len {} is not aligned to dtype size {}",
                chunk.chunk_id, expected_chunk_bytes, dtype_size
            )));
        }
        let element_start = byte_offset / dtype_size;
        let chunk_elements = expected_chunk_bytes / dtype_size;
        let element_end = element_start.checked_add(chunk_elements).ok_or_else(|| {
            RuntimeError::InvalidTensorData(
                "candidate row range chunk element end overflow".to_string(),
            )
        })?;

        for (row, score) in scores.iter_mut() {
            let row_start = row
                .checked_mul(config.linear.in_features)
                .ok_or_else(|| RuntimeError::Shape("candidate row start overflow".to_string()))?;
            let row_end = row_start
                .checked_add(config.linear.in_features)
                .ok_or_else(|| RuntimeError::Shape("candidate row end overflow".to_string()))?;
            let overlap_start = row_start.max(element_start);
            let overlap_end = row_end.min(element_end);
            if overlap_start >= overlap_end {
                continue;
            }

            let local_element_start = overlap_start - element_start;
            let range_byte_offset =
                local_element_start.checked_mul(dtype_size).ok_or_else(|| {
                    RuntimeError::Shape("candidate row range byte offset overflow".to_string())
                })?;
            let range_elements = overlap_end - overlap_start;
            let range_byte_len = range_elements.checked_mul(dtype_size).ok_or_else(|| {
                RuntimeError::Shape("candidate row range byte len overflow".to_string())
            })?;
            let input_start = overlap_start - row_start;
            model.with_decoded_chunk_range(
                chunk.chunk_id,
                range_byte_offset as u64,
                range_byte_len as u64,
                budget,
                |bytes, _budget| {
                    accumulate_candidate_row_range_16bit(
                        input,
                        input_start,
                        bytes,
                        tensor.dtype,
                        score,
                        weight_name,
                    )
                },
            )?;
        }

        byte_offset = byte_offset
            .checked_add(expected_chunk_bytes)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData(
                    "candidate row range chunk byte offset overflow".to_string(),
                )
            })?;
    }

    if byte_offset != expected_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {byte_offset} candidate row range bytes, expected {expected_weight_bytes}"
        )));
    }

    let mut best = scores.first().copied().ok_or_else(|| {
        RuntimeError::InvalidTensorData("empty candidate row score set".to_string())
    })?;
    for &(row, score) in scores.iter().skip(1) {
        if score > best.1 {
            best = (row, score);
        }
    }
    Ok(Some(best.0))
}

fn accumulate_candidate_row_range_16bit(
    input: &[f32],
    input_start: usize,
    raw_bytes: &[u8],
    dtype: rllm_container::DType,
    score: &mut f32,
    weight_name: &str,
) -> Result<()> {
    let dtype_size = dtype.size_bytes();
    if !raw_bytes.len().is_multiple_of(dtype_size) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "candidate row range for {weight_name} is not aligned to dtype size {dtype_size}"
        )));
    }
    let elements = raw_bytes.len() / dtype_size;
    if input_start + elements > input.len() {
        return Err(RuntimeError::Shape(format!(
            "candidate row range input span {}..{} exceeds input len {}",
            input_start,
            input_start + elements,
            input.len()
        )));
    }
    for local_idx in 0..elements {
        *score += input[input_start + local_idx] * raw_16bit_weight_at(raw_bytes, local_idx, dtype);
    }
    Ok(())
}

fn accumulate_candidate_rows_raw_16bit_chunk(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    element_end: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    weight_name: &str,
    scores: &mut [(usize, f32)],
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(dtype.size_bytes()) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Candidate row raw stream for {weight_name} is not aligned to dtype size {}",
            dtype.size_bytes()
        )));
    }
    let expected = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| {
            RuntimeError::Shape("candidate row weight element count overflow".to_string())
        })?;
    if element_end > expected {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} candidate row chunk elements {element_start}..{element_end} exceed expected {expected}"
        )));
    }

    for (row, score) in scores.iter_mut() {
        let row_start = row
            .checked_mul(config.in_features)
            .ok_or_else(|| RuntimeError::Shape("candidate row start overflow".to_string()))?;
        let row_end = row_start
            .checked_add(config.in_features)
            .ok_or_else(|| RuntimeError::Shape("candidate row end overflow".to_string()))?;
        let overlap_start = row_start.max(element_start);
        let overlap_end = row_end.min(element_end);
        if overlap_start >= overlap_end {
            continue;
        }

        let mut local_idx = overlap_start - element_start;
        let mut input_idx = overlap_start - row_start;
        while local_idx < overlap_end - element_start {
            *score += input[input_idx] * raw_16bit_weight_at(raw_bytes, local_idx, dtype);
            local_idx += 1;
            input_idx += 1;
        }
    }
    Ok(())
}

pub(crate) fn streaming_tile_linear_argmax_with_rolling_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    budget: &mut MemoryBudget,
    mut rolling: Option<&mut crate::rolling::RollingExecutor>,
) -> Result<usize> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, bias, config.linear)?;
    if config.linear.in_features == 0 || config.linear.out_features == 0 {
        return Err(RuntimeError::Shape(format!(
            "streaming linear argmax requires non-zero in/out features, got in_features={}, out_features={}",
            config.linear.in_features, config.linear.out_features
        )));
    }
    if config.linear.batch != 1 {
        return Err(RuntimeError::Shape(format!(
            "streaming linear argmax requires batch=1, got {}",
            config.linear.batch
        )));
    }
    let tensor = model.tensor(weight_name)?.clone();
    validate_weight_tensor(&tensor, config.linear)?;

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    if chunks.is_empty() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} has no chunks"
        )));
    }

    let mut byte_offset = 0usize;
    let mut state = StreamingLinearArgmaxState::new(bias);
    for chunk in chunks {
        let element_start = chunk_element_start_for_dtype(tensor.dtype, byte_offset, weight_name)?;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;

        if chunk.codec_id == "rtc-raw-v1"
            && matches!(
                tensor.dtype,
                rllm_container::DType::Fp16 | rllm_container::DType::Bf16
            )
        {
            model.with_raw_chunk(chunk.chunk_id, budget, |compressed_bytes, _budget| {
                if compressed_bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} raw byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        compressed_bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                accumulate_raw_16bit_chunk_argmax(
                    input,
                    compressed_bytes,
                    element_start,
                    config.linear,
                    tensor.dtype,
                    &mut state,
                    weight_name,
                    rolling.as_deref_mut(),
                )
            })?;
        } else if tensor.dtype == rllm_container::DType::Q8_0 {
            // R126: zero-copy raw q8 bytes for lm_head (skip per-call .to_vec()).
            let kernel = |bytes: &[u8], _budget: &mut MemoryBudget| -> Result<()> {
                if bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                accumulate_q8_0_chunk_argmax(
                    input,
                    bytes,
                    element_start,
                    config.linear,
                    &mut state,
                    weight_name,
                )
            };
            if chunk.codec_id == "rtc-raw-v1" {
                model.with_raw_chunk(chunk.chunk_id, budget, kernel)?;
            } else {
                model.with_decoded_chunk(chunk.chunk_id, budget, kernel)?;
            }
        } else if tensor.dtype.is_quantized() {
            model.with_decoded_chunk(chunk.chunk_id, budget, |bytes, budget| {
                if bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} decoded byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                let weights = dequantize_quantized_chunk(tensor.dtype, chunk.chunk_id, bytes)?;
                let scratch_bytes = weights
                    .len()
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        RuntimeError::Shape("quantized argmax f32 scratch overflow".to_string())
                    })?;
                let scratch_label = format!(
                    "streaming {:?} argmax f32 scratch chunk {} elements {}..{}",
                    tensor.dtype,
                    chunk.chunk_id,
                    element_start,
                    element_start + weights.len()
                );
                budget.reserve(scratch_bytes, scratch_label.clone())?;
                let result = accumulate_weight_chunk_argmax(
                    input,
                    &weights,
                    element_start,
                    config.linear,
                    &mut state,
                    weight_name,
                );
                budget.release(scratch_bytes, scratch_label)?;
                result
            })?;
        } else {
            let dtype_size = tensor.dtype.size_bytes();
            model.with_decoded_chunk(chunk.chunk_id, budget, |bytes, budget| {
                if bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} decoded byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                if bytes.len() % dtype_size != 0 {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} byte len {} is not aligned to dtype size {}",
                        chunk.chunk_id,
                        bytes.len(),
                        dtype_size
                    )));
                }

                let elements_in_chunk = bytes.len() / dtype_size;
                let mut local_element_start = 0usize;
                while local_element_start < elements_in_chunk {
                    let tile_len = config
                        .tile_elements
                        .min(elements_in_chunk - local_element_start);
                    let tile_byte_start =
                        local_element_start.checked_mul(dtype_size).ok_or_else(|| {
                            RuntimeError::Shape("tile byte start overflow".to_string())
                        })?;
                    let tile_byte_len = tile_len
                        .checked_mul(dtype_size)
                        .ok_or_else(|| RuntimeError::Shape("tile byte len overflow".to_string()))?;
                    let tile_byte_end = tile_byte_start
                        .checked_add(tile_byte_len)
                        .ok_or_else(|| RuntimeError::Shape("tile byte end overflow".to_string()))?;
                    let scratch_bytes = tile_len
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| {
                            RuntimeError::Shape("tile f32 scratch overflow".to_string())
                        })?;
                    let scratch_label = format!(
                        "streaming argmax tile f32 scratch chunk {} elements {}..{}",
                        chunk.chunk_id,
                        local_element_start,
                        local_element_start + tile_len
                    );

                    budget.reserve(scratch_bytes, scratch_label.clone())?;
                    let weights =
                        match decode_to_f32(tensor.dtype, &bytes[tile_byte_start..tile_byte_end]) {
                            Ok(values) => values,
                            Err(err) => {
                                budget.release(scratch_bytes, scratch_label)?;
                                return Err(err);
                            }
                        };
                    let result = accumulate_weight_chunk_argmax(
                        input,
                        &weights,
                        element_start + local_element_start,
                        config.linear,
                        &mut state,
                        weight_name,
                    );
                    drop(weights);
                    budget.release(scratch_bytes, scratch_label)?;
                    result?;
                    local_element_start += tile_len;
                }
                Ok(())
            })?;
        }

        byte_offset = byte_offset
            .checked_add(expected_chunk_bytes)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData("chunk byte offset overflow".to_string())
            })?;
    }

    let expected_weight_bytes = linear_weight_storage_bytes(tensor.dtype, config.linear)?;
    if byte_offset != expected_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {byte_offset} bytes, expected {expected_weight_bytes}"
        )));
    }

    state.finish(config.linear, weight_name)
}
