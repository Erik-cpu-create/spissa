use crate::{RamaExperimentalSpeedConfig, RamaExperimentalSpeedStats};

/// Low-RAM PyTorch-style linear layer over a chunked `.rllm` weight tensor.
///
/// Computes `input[batch,in] × weight[out,in]^T + bias[out]` while decoding
/// only one compressed weight chunk at a time. The caller owns activation and
/// output memory accounting; `budget` tracks transient compressed/decoded/f32
/// chunk scratch memory.
pub fn streaming_linear_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    validate_linear_shapes(input, bias, config)?;
    let tensor = model.tensor(weight_name)?.clone();
    validate_weight_tensor(&tensor, config)?;

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    if chunks.is_empty() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} has no chunks"
        )));
    }

    let mut output = vec![0.0f32; config.batch * config.out_features];
    if let Some(bias) = bias {
        for batch_idx in 0..config.batch {
            let row_start = batch_idx * config.out_features;
            output[row_start..row_start + config.out_features].copy_from_slice(bias);
        }
    }

    let dtype_size = tensor.dtype.size_bytes();
    let mut byte_offset = 0usize;
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} chunk stream reached unaligned byte offset {byte_offset} for dtype size {dtype_size}"
            )));
        }
        let element_start = byte_offset / dtype_size;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;

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

            let scratch_bytes = (bytes.len() / dtype_size) * std::mem::size_of::<f32>();
            let scratch_label = format!("streaming f32 scratch chunk {}", chunk.chunk_id);
            budget.reserve(scratch_bytes, scratch_label.clone())?;
            let weights = match decode_to_f32(tensor.dtype, bytes) {
                Ok(values) => values,
                Err(err) => {
                    budget.release(scratch_bytes, scratch_label)?;
                    return Err(err);
                }
            };

            let result = accumulate_weight_chunk(
                input,
                &mut output,
                &weights,
                element_start,
                config,
                weight_name,
            );
            drop(weights);
            budget.release(scratch_bytes, scratch_label)?;
            result
        })?;

        byte_offset = byte_offset
            .checked_add(expected_chunk_bytes)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData("chunk byte offset overflow".to_string())
            })?;
    }

    let expected_weight_bytes = config
        .out_features
        .checked_mul(config.in_features)
        .and_then(|elements| elements.checked_mul(dtype_size))
        .ok_or_else(|| RuntimeError::Shape("weight byte size overflow".to_string()))?;
    if byte_offset != expected_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {byte_offset} bytes, expected {expected_weight_bytes}"
        )));
    }

    Ok(output)
}

/// Phase 7 fused tile variant of `streaming_linear_from_model`.
///
/// Computes the same PyTorch-style linear layer but converts only
/// `tile_elements` weight values into f32 scratch at a time before immediately
/// accumulating them into the output. Current RTC codecs still require decoding
/// one compressed chunk to original bytes first; this removes the separate
/// full-f32-chunk scratch window and is the first verified step toward true
/// fused decode+matmul.
pub fn streaming_tile_linear_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, bias, config.linear)?;
    let tensor = model.tensor(weight_name)?.clone();
    validate_weight_tensor(&tensor, config.linear)?;

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    if chunks.is_empty() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} has no chunks"
        )));
    }

    let mut output = vec![0.0f32; config.linear.batch * config.linear.out_features];
    if let Some(bias) = bias {
        for batch_idx in 0..config.linear.batch {
            let row_start = batch_idx * config.linear.out_features;
            output[row_start..row_start + config.linear.out_features].copy_from_slice(bias);
        }
    }

    let dtype_size = tensor.dtype.size_bytes();
    let mut byte_offset = 0usize;
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} chunk stream reached unaligned byte offset {byte_offset} for dtype size {dtype_size}"
            )));
        }
        let element_start = byte_offset / dtype_size;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;

        if chunk.codec_id == "rtc-rle-v1" && tensor.dtype == rllm_container::DType::U8 {
            model.with_raw_chunk(chunk.chunk_id, budget, |compressed_bytes, _budget| {
                accumulate_fused_rle_chunk_u8(
                    input,
                    &mut output,
                    compressed_bytes,
                    element_start,
                    config.linear,
                    weight_name,
                )
            })?;
        } else if chunk.codec_id == "rtc-raw-v1" && tensor.dtype == rllm_container::DType::Fp16 {
            model.with_raw_chunk(chunk.chunk_id, budget, |compressed_bytes, _budget| {
                if compressed_bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} raw byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        compressed_bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                accumulate_fused_raw_fp16_chunk(
                    input,
                    &mut output,
                    compressed_bytes,
                    element_start,
                    config.linear,
                    weight_name,
                )
            })?;
        } else if chunk.codec_id == "rtc-raw-v1"
            && tensor.dtype == rllm_container::DType::Bf16
            && config.linear.batch == 1
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
                accumulate_fused_raw_bf16_chunk_batch1(
                    input,
                    &mut output,
                    compressed_bytes,
                    element_start,
                    config.linear,
                    weight_name,
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
                        "streaming fused tile f32 scratch chunk {} elements {}..{}",
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
                    let result = accumulate_weight_chunk(
                        input,
                        &mut output,
                        &weights,
                        element_start + local_element_start,
                        config.linear,
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

    let expected_weight_bytes = config
        .linear
        .out_features
        .checked_mul(config.linear.in_features)
        .and_then(|elements| elements.checked_mul(dtype_size))
        .ok_or_else(|| RuntimeError::Shape("weight byte size overflow".to_string()))?;
    if byte_offset != expected_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {byte_offset} bytes, expected {expected_weight_bytes}"
        )));
    }

    Ok(output)
}

/// Streaming single-pass linear projection multiplied into an existing output.
///
/// This computes `target *= Linear(input, weight, bias)` without materializing
/// the full linear output. It is intended for gated MLP decode paths such as
/// LLaMA `silu(gate_proj(x)) * up_proj(x)`, where the left-hand activation
/// already lives in `target`.
pub fn streaming_tile_linear_multiply_into_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    target: &mut [f32],
    config: StreamingTileLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<()> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, bias, config.linear)?;
    let target_len = config
        .linear
        .batch
        .checked_mul(config.linear.out_features)
        .ok_or_else(|| RuntimeError::Shape("target len overflow".to_string()))?;
    if target.len() != target_len {
        return Err(RuntimeError::Shape(format!(
            "target len {} does not match batch*out_features = {}",
            target.len(),
            target_len
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

    let dtype_size = tensor.dtype.size_bytes();
    let mut byte_offset = 0usize;
    let mut state = StreamingLinearMultiplyIntoState::new(target, bias, config.linear);
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} chunk stream reached unaligned byte offset {byte_offset} for dtype size {dtype_size}"
            )));
        }
        let element_start = byte_offset / dtype_size;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;

        if chunk.codec_id == "rtc-raw-v1" && tensor.dtype == rllm_container::DType::Fp16 {
            model.with_raw_chunk(chunk.chunk_id, budget, |compressed_bytes, _budget| {
                if compressed_bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} raw byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        compressed_bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                accumulate_multiply_raw_fp16_chunk(
                    input,
                    compressed_bytes,
                    element_start,
                    config.linear,
                    &mut state,
                    weight_name,
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
                        "streaming multiply tile f32 scratch chunk {} elements {}..{}",
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
                    let result = accumulate_weight_chunk_multiply_into(
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

    let expected_weight_bytes = config
        .linear
        .out_features
        .checked_mul(config.linear.in_features)
        .and_then(|elements| elements.checked_mul(dtype_size))
        .ok_or_else(|| RuntimeError::Shape("weight byte size overflow".to_string()))?;
    if byte_offset != expected_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {byte_offset} bytes, expected {expected_weight_bytes}"
        )));
    }

    state.finish(config.linear, weight_name)
}

/// Fused LLaMA-style gated MLP input projection for raw-FP16 batch-1 decode.
///
/// Computes `silu(gate_proj(input)) * up_proj(input)` without materializing
/// either projection separately. Returns `Ok(None)` when the model layout is
/// unsupported so callers can fall back to the generic streaming linear path.
pub fn streaming_silu_gate_up_from_model(
    model: &mut LazyRllmModel,
    gate_weight_name: &str,
    up_weight_name: &str,
    input: &[f32],
    config: StreamingTileLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, None, config.linear)?;
    if config.linear.batch != 1 {
        return Ok(None);
    }

    let gate_tensor = model.tensor(gate_weight_name)?.clone();
    let up_tensor = model.tensor(up_weight_name)?.clone();
    validate_weight_tensor(&gate_tensor, config.linear)?;
    validate_weight_tensor(&up_tensor, config.linear)?;
    let raw_dtype = gate_tensor.dtype;
    if gate_tensor.dtype != up_tensor.dtype
        || !matches!(
            raw_dtype,
            rllm_container::DType::Fp16 | rllm_container::DType::Bf16
        )
    {
        return Ok(None);
    }

    let gate_chunks: Vec<ChunkMeta> = model.chunks_for_tensor(gate_tensor.tensor_id).to_vec();
    let up_chunks: Vec<ChunkMeta> = model.chunks_for_tensor(up_tensor.tensor_id).to_vec();
    if gate_chunks.is_empty() || gate_chunks.len() != up_chunks.len() {
        return Ok(None);
    }
    for (gate_chunk, up_chunk) in gate_chunks.iter().zip(up_chunks.iter()) {
        if gate_chunk.codec_id != "rtc-raw-v1"
            || up_chunk.codec_id != "rtc-raw-v1"
            || gate_chunk.chunk_offset_in_tensor != up_chunk.chunk_offset_in_tensor
            || gate_chunk.uncompressed_size != up_chunk.uncompressed_size
            || gate_chunk.compressed_size != gate_chunk.uncompressed_size
            || up_chunk.compressed_size != up_chunk.uncompressed_size
        {
            return Ok(None);
        }
    }

    let mut output = vec![0.0f32; config.linear.out_features];
    {
        let mut state = SiluGateUpState::new(&mut output);
        let dtype_size = raw_dtype.size_bytes();
        let mut byte_offset = 0usize;
        for (gate_chunk, up_chunk) in gate_chunks.iter().zip(up_chunks.iter()) {
            if !byte_offset.is_multiple_of(dtype_size) {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "weight tensors {gate_weight_name}/{up_weight_name} chunk stream reached unaligned byte offset {byte_offset} for dtype size {dtype_size}"
                )));
            }
            let element_start = byte_offset / dtype_size;
            let expected_chunk_bytes =
                usize::try_from(gate_chunk.uncompressed_size).map_err(|_| {
                    RuntimeError::InvalidTensorData(format!(
                        "chunk {} uncompressed size does not fit usize",
                        gate_chunk.chunk_id
                    ))
                })?;

            model.with_two_raw_chunks(
                gate_chunk.chunk_id,
                up_chunk.chunk_id,
                budget,
                |gate_bytes, up_bytes, _budget| {
                    if gate_bytes.len() != expected_chunk_bytes {
                        return Err(RuntimeError::InvalidTensorData(format!(
                            "chunk {} raw byte len {} does not match metadata {}",
                            gate_chunk.chunk_id,
                            gate_bytes.len(),
                            expected_chunk_bytes
                        )));
                    }
                    if up_bytes.len() != expected_chunk_bytes {
                        return Err(RuntimeError::InvalidTensorData(format!(
                            "chunk {} raw byte len {} does not match metadata {}",
                            up_chunk.chunk_id,
                            up_bytes.len(),
                            expected_chunk_bytes
                        )));
                    }
                    accumulate_silu_gate_up_raw_16bit_chunk_batch1(
                        input,
                        gate_bytes,
                        up_bytes,
                        element_start,
                        config.linear,
                        raw_dtype,
                        &mut state,
                        gate_weight_name,
                    )
                },
            )?;

            byte_offset = byte_offset
                .checked_add(expected_chunk_bytes)
                .ok_or_else(|| {
                    RuntimeError::InvalidTensorData("chunk byte offset overflow".to_string())
                })?;
        }

        let expected_weight_bytes = config
            .linear
            .out_features
            .checked_mul(config.linear.in_features)
            .and_then(|elements| elements.checked_mul(dtype_size))
            .ok_or_else(|| RuntimeError::Shape("weight byte size overflow".to_string()))?;
        if byte_offset != expected_weight_bytes {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensors {gate_weight_name}/{up_weight_name} streamed {byte_offset} bytes, expected {expected_weight_bytes}"
            )));
        }
        state.finish(config.linear, gate_weight_name)?;
    }

    Ok(Some(output))
}

/// Experimental sparse batch-1 projection over raw 16-bit weights.
///
/// This is an opt-in research path used by RLLM experimental speed mode. It
/// keeps model weights unchanged and computes an approximate projection from
/// the top activation dimensions by absolute magnitude.
pub fn streaming_sparse_tile_linear_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    speed_config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, bias, config.linear)?;
    if !speed_config.enabled || config.linear.batch != 1 || config.linear.in_features == 0 {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let tensor = model.tensor(weight_name)?.clone();
    validate_weight_tensor(&tensor, config.linear)?;
    if !matches!(
        tensor.dtype,
        rllm_container::DType::Fp16 | rllm_container::DType::Bf16
    ) {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let selected = crate::select_top_abs_indices(
        input,
        speed_config.topk_for_input(config.linear.in_features, 256),
    );
    if selected.is_empty() {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    if chunks.is_empty() || chunks.iter().any(|chunk| chunk.codec_id != "rtc-raw-v1") {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let mut output = vec![0.0f32; config.linear.out_features];
    if let Some(bias) = bias {
        output.copy_from_slice(bias);
    }

    let dtype_size = tensor.dtype.size_bytes();
    let mut byte_offset = 0usize;
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} sparse stream reached unaligned byte offset {byte_offset}"
            )));
        }
        let element_start = byte_offset / dtype_size;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
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
            accumulate_sparse_raw_16bit_linear_chunk_batch1(
                input,
                &selected,
                &mut output,
                raw_bytes,
                element_start,
                config.linear,
                tensor.dtype,
                weight_name,
            )
        })?;

        byte_offset = byte_offset.checked_add(expected_chunk_bytes).ok_or_else(|| {
            RuntimeError::InvalidTensorData("sparse chunk byte offset overflow".to_string())
        })?;
    }

    stats.record_sparse_projection(
        selected.len(),
        config.linear.in_features,
        config.linear.out_features,
        1,
    );
    Ok(Some(output))
}

/// Experimental sparse LLaMA gated MLP input projection.
///
/// Computes `silu(gate_proj(input)) * up_proj(input)` from a deterministic
/// activation top-k subset. Unsupported layouts return `Ok(None)` so callers
/// can use the exact low-RAM path.
pub fn streaming_sparse_silu_gate_up_from_model(
    model: &mut LazyRllmModel,
    gate_weight_name: &str,
    up_weight_name: &str,
    input: &[f32],
    config: StreamingTileLinearConfig,
    speed_config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, None, config.linear)?;
    if !speed_config.enabled || config.linear.batch != 1 || config.linear.in_features == 0 {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let gate_tensor = model.tensor(gate_weight_name)?.clone();
    let up_tensor = model.tensor(up_weight_name)?.clone();
    validate_weight_tensor(&gate_tensor, config.linear)?;
    validate_weight_tensor(&up_tensor, config.linear)?;
    if gate_tensor.dtype != up_tensor.dtype
        || !matches!(
            gate_tensor.dtype,
            rllm_container::DType::Fp16 | rllm_container::DType::Bf16
        )
    {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let selected = crate::select_top_abs_indices(
        input,
        speed_config.topk_for_input(config.linear.in_features, 256),
    );
    if selected.is_empty() {
        stats.record_exact_fallback();
        return Ok(None);
    }

    let gate_chunks: Vec<ChunkMeta> = model.chunks_for_tensor(gate_tensor.tensor_id).to_vec();
    let up_chunks: Vec<ChunkMeta> = model.chunks_for_tensor(up_tensor.tensor_id).to_vec();
    if gate_chunks.is_empty() || gate_chunks.len() != up_chunks.len() {
        stats.record_exact_fallback();
        return Ok(None);
    }
    for (gate_chunk, up_chunk) in gate_chunks.iter().zip(up_chunks.iter()) {
        if gate_chunk.codec_id != "rtc-raw-v1"
            || up_chunk.codec_id != "rtc-raw-v1"
            || gate_chunk.chunk_offset_in_tensor != up_chunk.chunk_offset_in_tensor
            || gate_chunk.uncompressed_size != up_chunk.uncompressed_size
        {
            stats.record_exact_fallback();
            return Ok(None);
        }
    }

    let mut output = vec![0.0f32; config.linear.out_features];
    {
        let mut state = SiluGateUpState::new(&mut output);
        let dtype_size = gate_tensor.dtype.size_bytes();
        let mut byte_offset = 0usize;
        for (gate_chunk, up_chunk) in gate_chunks.iter().zip(up_chunks.iter()) {
            if !byte_offset.is_multiple_of(dtype_size) {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "weight tensors {gate_weight_name}/{up_weight_name} sparse stream reached unaligned byte offset {byte_offset}"
                )));
            }
            let element_start = byte_offset / dtype_size;
            let expected_chunk_bytes =
                usize::try_from(gate_chunk.uncompressed_size).map_err(|_| {
                    RuntimeError::InvalidTensorData(format!(
                        "chunk {} uncompressed size does not fit usize",
                        gate_chunk.chunk_id
                    ))
                })?;

            model.with_two_raw_chunks(
                gate_chunk.chunk_id,
                up_chunk.chunk_id,
                budget,
                |gate_bytes, up_bytes, _budget| {
                    if gate_bytes.len() != expected_chunk_bytes
                        || up_bytes.len() != expected_chunk_bytes
                    {
                        return Err(RuntimeError::InvalidTensorData(format!(
                            "sparse gate/up raw chunk len mismatch for chunks {}/{}",
                            gate_chunk.chunk_id, up_chunk.chunk_id
                        )));
                    }
                    accumulate_sparse_silu_gate_up_raw_16bit_chunk_batch1(
                        input,
                        &selected,
                        gate_bytes,
                        up_bytes,
                        element_start,
                        config.linear,
                        gate_tensor.dtype,
                        &mut state,
                        gate_weight_name,
                    )
                },
            )?;

            byte_offset = byte_offset.checked_add(expected_chunk_bytes).ok_or_else(|| {
                RuntimeError::InvalidTensorData("sparse gate/up byte offset overflow".to_string())
            })?;
        }
        state.finish(config.linear, gate_weight_name)?;
    }

    stats.record_sparse_projection(
        selected.len(),
        config.linear.in_features,
        config.linear.out_features,
        2,
    );
    Ok(Some(output))
}

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
            .ok_or_else(|| RuntimeError::InvalidTensorData("prefix byte stream overflow".to_string()))?;
        byte_offset = byte_offset
            .checked_add(expected_chunk_bytes)
            .ok_or_else(|| RuntimeError::InvalidTensorData("chunk byte offset overflow".to_string()))?;
    }

    if streamed_prefix_bytes != prefix_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {streamed_prefix_bytes} prefix bytes, expected {prefix_weight_bytes}"
        )));
    }

    state.finish(prefix_linear, weight_name)
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

    let dtype_size = tensor.dtype.size_bytes();
    let mut byte_offset = 0usize;
    let mut state = StreamingLinearArgmaxState::new(bias);
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} chunk stream reached unaligned byte offset {byte_offset} for dtype size {dtype_size}"
            )));
        }
        let element_start = byte_offset / dtype_size;
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

    let expected_weight_bytes = config
        .linear
        .out_features
        .checked_mul(config.linear.in_features)
        .and_then(|elements| elements.checked_mul(dtype_size))
        .ok_or_else(|| RuntimeError::Shape("weight byte size overflow".to_string()))?;
    if byte_offset != expected_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {byte_offset} bytes, expected {expected_weight_bytes}"
        )));
    }

    state.finish(config.linear, weight_name)
}
