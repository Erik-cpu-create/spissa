// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

// Streaming multiply-into linear (target *= Linear(x,W)) + panel multiply-into +
// the SiLU-gate-up fused MLP kernel. Split out of linear.rs (R171); include!d into mod.rs.

/// Streaming single-pass linear projection multiplied into an existing output.
///
/// This computes `target *= Linear(input, weight, bias)` without materializing
/// the full linear output. It is intended for gated MLP decode paths such as
/// LLaMA `silu(gate_proj(x)) * up_proj(x)`, where the left-hand activation
/// already lives in `target`.
pub fn streaming_tile_linear_multiply_into_from_model(
    model: &mut LazySpissaModel,
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

    // R133-style decode fast-path for the multiply-into projection (LLaMA up_proj
    // at batch=1). The regular tile-linear path got the whole-tensor int8 sdot in
    // R133, but multiply_into only had the batch>=2 panel path — so up_proj fell to
    // the scalar per-chunk path on DECODE (~107ms/token vs ~6ms for gate_proj on
    // Llama 1B). Compute the projection with the whole-tensor int8 sdot kernel into
    // a scratch, then apply `target *= up + bias`. Falls through if not contiguous-raw.
    if tensor.dtype == spissa_container::DType::Q8_0
        && config.linear.batch == 1
        && config.linear.in_features.is_multiple_of(32)
        && q8_activation_path_enabled()
    {
        let lin = config.linear;
        let out_features = lin.out_features;
        let mut up = vec![0.0f32; out_features];
        let handled = model.with_raw_tensor(tensor.tensor_id, |q8_bytes| {
            accumulate_q8_0_full_tensor_int8_batch1(input, &mut up, q8_bytes, lin)
        })?;
        if handled.is_some() {
            for (f, slot) in target.iter_mut().enumerate().take(out_features) {
                let bias_v = bias.map(|values| values[f]).unwrap_or(0.0);
                *slot *= up[f] + bias_v;
            }
            return Ok(());
        }
    }

    // R121: i8mm packed-panel fast path for the multiply-into projection
    // (LLaMA up_proj). Compute the full Q8_0 linear dot product into a scratch
    // buffer via the same panel kernel R119 uses for gate/down, then apply
    // `target *= up + bias`. This bypasses the fused row-state machine for
    // panel-eligible Q8_0 weights; otherwise we fall through to it unchanged.
    if tensor.dtype == spissa_container::DType::Q8_0
        && q8_activation_path_enabled()
        && config.linear.batch >= 2
        && config.linear.in_features.is_multiple_of(32)
    {
        if let Some(up) = try_panel_multiply_into_up(
            model,
            weight_name,
            input,
            tensor.dtype,
            &chunks,
            config.linear,
            budget,
        )? {
            let out_features = config.linear.out_features;
            for batch_idx in 0..config.linear.batch {
                let row = batch_idx * out_features;
                for f in 0..out_features {
                    let bias_v = bias.map(|values| values[f]).unwrap_or(0.0);
                    target[row + f] *= up[row + f] + bias_v;
                }
            }
            return Ok(());
        }
        // Not panel-eligible (e.g. odd shape / no i8mm): fall through.
    }

    let dtype_size = tensor.dtype.size_bytes();
    let mut byte_offset = 0usize;
    let mut state = StreamingLinearMultiplyIntoState::new(target, bias, config.linear);
    for chunk in chunks {
        let element_start = chunk_element_start_for_dtype(tensor.dtype, byte_offset, weight_name)?;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;

        if chunk.codec_id == "rtc-raw-v1" && tensor.dtype == spissa_container::DType::Fp16 {
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
        } else if tensor.dtype == spissa_container::DType::Q8_0 {
            // R126: zero-copy raw q8 bytes (skip per-call .to_vec()).
            let kernel = |bytes: &[u8], _budget: &mut MemoryBudget| -> Result<()> {
                if bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                accumulate_q8_0_chunk_multiply_into(
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

                let scratch_bytes = quantized_elements_for_bytes(tensor.dtype, bytes.len())?
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        RuntimeError::Shape("quantized f32 scratch overflow".to_string())
                    })?;
                let scratch_label = format!(
                    "streaming multiply {:?} f32 scratch chunk {}",
                    tensor.dtype, chunk.chunk_id
                );
                budget.reserve(scratch_bytes, scratch_label.clone())?;
                let weights = match dequantize_quantized_chunk(tensor.dtype, chunk.chunk_id, bytes)
                {
                    Ok(values) => values,
                    Err(err) => {
                        budget.release(scratch_bytes, scratch_label)?;
                        return Err(err);
                    }
                };
                let result = accumulate_weight_chunk_multiply_into(
                    input,
                    &weights,
                    element_start,
                    config.linear,
                    &mut state,
                    weight_name,
                );
                drop(weights);
                budget.release(scratch_bytes, scratch_label)?;
                result
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

    let expected_weight_bytes = linear_weight_storage_bytes(tensor.dtype, config.linear)?;
    if byte_offset != expected_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {byte_offset} bytes, expected {expected_weight_bytes}"
        )));
    }

    state.finish(config.linear, weight_name)
}

/// R121: compute a full Q8_0 linear projection into a fresh buffer using the
/// i8mm packed panel (the same kernel R119 uses for gate/down).
///
/// Returns `Ok(Some(up))` where `up` holds the raw linear dot products (no bias)
/// laid out as `[batch * out_features]`, or `Ok(None)` if any chunk is not
/// panel-eligible so the caller falls back to the fused state machine. On the
/// `None` path the scratch is discarded and the budget is restored, so a full
/// re-decode by the fallback is safe.
fn try_panel_multiply_into_up(
    model: &mut LazySpissaModel,
    weight_name: &str,
    input: &[f32],
    dtype: spissa_container::DType,
    chunks: &[ChunkMeta],
    config: StreamingLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    if dtype != spissa_container::DType::Q8_0 {
        return Ok(None);
    }
    let out_len = config
        .batch
        .checked_mul(config.out_features)
        .ok_or_else(|| RuntimeError::Shape("panel up scratch len overflow".to_string()))?;
    let scratch_bytes = out_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| RuntimeError::Shape("panel up scratch bytes overflow".to_string()))?;
    let scratch_label = format!("R121 panel up scratch {weight_name}");
    budget.reserve(scratch_bytes, scratch_label.clone())?;
    let mut up = vec![0.0f32; out_len];

    let mut byte_offset = 0usize;
    let mut all_paneled = true;
    for chunk in chunks {
        let element_start = chunk_element_start_for_dtype(dtype, byte_offset, weight_name)?;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;
        let up_ref = &mut up;
        let mut paneled = false;
        // R126: zero-copy raw q8 bytes (skip per-call .to_vec()).
        let kernel = |bytes: &[u8], _budget: &mut MemoryBudget| -> Result<()> {
            if bytes.len() != expected_chunk_bytes {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "chunk {} byte len {} does not match metadata {}",
                    chunk.chunk_id,
                    bytes.len(),
                    expected_chunk_bytes
                )));
            }
            paneled = accumulate_q8_0_chunk_panel_smmla(input, up_ref, bytes, element_start, config)?;
            Ok(())
        };
        if chunk.codec_id == "rtc-raw-v1" {
            model.with_raw_chunk(chunk.chunk_id, budget, kernel)?;
        } else {
            model.with_decoded_chunk(chunk.chunk_id, budget, kernel)?;
        }
        if !paneled {
            all_paneled = false;
            break;
        }
        byte_offset = byte_offset.checked_add(expected_chunk_bytes).ok_or_else(|| {
            RuntimeError::InvalidTensorData("chunk byte offset overflow".to_string())
        })?;
    }

    budget.release(scratch_bytes, scratch_label)?;

    if all_paneled {
        Ok(Some(up))
    } else {
        Ok(None)
    }
}

/// Fused LLaMA-style gated MLP input projection for raw-FP16 batch-1 decode.
///
/// Computes `silu(gate_proj(input)) * up_proj(input)` without materializing
/// either projection separately. Returns `Ok(None)` when the model layout is
/// unsupported so callers can fall back to the generic streaming linear path.
pub fn streaming_silu_gate_up_from_model(
    model: &mut LazySpissaModel,
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
            spissa_container::DType::Fp16 | spissa_container::DType::Bf16
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

