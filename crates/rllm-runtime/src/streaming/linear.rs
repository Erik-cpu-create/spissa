// Core dense streaming linear projection: shared helpers + streaming_linear_from_model
// + streaming_tile_linear_from_model. multiply-into + silu-gate-up -> linear_multiply.rs (R171).

pub fn input_tile_sidecar_weight_name(weight_name: &str) -> String {
    format!("{INPUT_TILE_SIDECAR_PREFIX}{weight_name}")
}

fn linear_weight_elements(config: StreamingLinearConfig) -> Result<usize> {
    config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))
}

fn linear_weight_storage_bytes(
    dtype: rllm_container::DType,
    config: StreamingLinearConfig,
) -> Result<usize> {
    Ok(dtype.byte_size_for_elements(linear_weight_elements(config)?))
}

fn chunk_element_start_for_dtype(
    dtype: rllm_container::DType,
    byte_offset: usize,
    weight_name: &str,
) -> Result<usize> {
    if let Some(block_bytes) = quantized_block_bytes(dtype) {
        if !byte_offset.is_multiple_of(block_bytes) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} {:?} chunk stream reached unaligned byte offset {byte_offset}",
                dtype
            )));
        }
        Ok((byte_offset / block_bytes) * 32)
    } else {
        let dtype_size = dtype.size_bytes();
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} chunk stream reached unaligned byte offset {byte_offset} for dtype size {dtype_size}"
            )));
        }
        Ok(byte_offset / dtype_size)
    }
}

fn quantized_block_bytes(dtype: rllm_container::DType) -> Option<usize> {
    match dtype {
        rllm_container::DType::Q4_0 => Some(18),
        rllm_container::DType::Q8_0 => Some(34),
        _ => None,
    }
}

fn quantized_elements_for_bytes(dtype: rllm_container::DType, bytes_len: usize) -> Result<usize> {
    let block_bytes = quantized_block_bytes(dtype).ok_or_else(|| {
        RuntimeError::InvalidTensorData(format!("{:?} is not a quantized block dtype", dtype))
    })?;
    if !bytes_len.is_multiple_of(block_bytes) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "{:?} byte len {} is not aligned to block size {}",
            dtype, bytes_len, block_bytes
        )));
    }
    Ok((bytes_len / block_bytes) * 32)
}

fn dequantize_quantized_chunk(
    dtype: rllm_container::DType,
    chunk_id: u64,
    bytes: &[u8],
) -> Result<Vec<f32>> {
    let elements = quantized_elements_for_bytes(dtype, bytes.len()).map_err(|err| {
        RuntimeError::InvalidTensorData(format!(
            "quantized chunk {chunk_id} invalid {:?} data: {err}",
            dtype
        ))
    })?;
    let mut weights = vec![0.0f32; elements];
    match dtype {
        rllm_container::DType::Q4_0 => crate::dequantize::dequantize_q4_0(bytes, &mut weights),
        rllm_container::DType::Q8_0 => crate::dequantize::dequantize_q8_0(bytes, &mut weights),
        _ => {
            return Err(RuntimeError::InvalidTensorData(format!(
                "{:?} is not a quantized block dtype",
                dtype
            )));
        }
    }
    Ok(weights)
}

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
        let element_start = chunk_element_start_for_dtype(tensor.dtype, byte_offset, weight_name)?;
        let expected_chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;

        if tensor.dtype == rllm_container::DType::Q8_0 {
            model.with_decoded_chunk(chunk.chunk_id, budget, |bytes, _budget| {
                if bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} decoded byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                accumulate_q8_0_chunk_parallel(
                    input,
                    &mut output,
                    bytes,
                    element_start,
                    config,
                    weight_name,
                )
            })?;
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
                    "streaming {:?} f32 scratch chunk {}",
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
        }

        byte_offset = byte_offset
            .checked_add(expected_chunk_bytes)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData("chunk byte offset overflow".to_string())
            })?;
    }

    let expected_weight_bytes = linear_weight_storage_bytes(tensor.dtype, config)?;
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

    // R133 decode fast-path: raw-codec Q8_0 + batch=1 (default on; opt out with
    // RLLM_Q8_ACTIVATION=0). View the whole tensor as one contiguous mmap slice and
    // run int8 sdot row-parallel — bypassing the per-chunk dispatch + scalar i8×f32.
    // Requires in_features%32==0 (the sdot block width); falls through to the
    // per-chunk path otherwise, or if the tensor isn't contiguous-raw.
    if tensor.dtype == rllm_container::DType::Q8_0
        && config.linear.batch == 1
        && config.linear.in_features % 32 == 0
        && q8_activation_path_enabled()
    {
        let lin = config.linear;
        let handled = model.with_raw_tensor(tensor.tensor_id, |q8_bytes| {
            accumulate_q8_0_full_tensor_int8_batch1(input, &mut output, q8_bytes, lin)
        })?;
        if handled.is_some() {
            return Ok(output);
        }
    }

    // R138 prefill fast-path: raw-codec Q8_0 + batch>=2 (gated on RLLM_Q8_ACTIVATION).
    // Same whole-tensor view as R133, but split the BATCH rows across workers ONCE
    // per projection (output is contiguous per batch row, so the split is sound) and
    // run the i8mm panel kernel per worker — avoiding the per-chunk thread-spawn that
    // made naive prefill parallelization slower than single-threaded.
    if tensor.dtype == rllm_container::DType::Q8_0
        && config.linear.batch >= 2
        && config.linear.in_features % 32 == 0
        && q8_activation_path_enabled()
    {
        let lin = config.linear;
        let weight = weight_name;
        let handled = model.with_raw_tensor(tensor.tensor_id, |q8_bytes| {
            accumulate_q8_0_full_tensor_panel_batch(input, &mut output, q8_bytes, lin, weight)
        })?;
        if handled.is_some() {
            return Ok(output);
        }
    }

    let dtype_size = tensor.dtype.size_bytes();
    let mut byte_offset = 0usize;
    for chunk in chunks {
        let element_start = chunk_element_start_for_dtype(tensor.dtype, byte_offset, weight_name)?;
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
        } else if tensor.dtype == rllm_container::DType::Bf16 && config.linear.batch == 1 {
            // bf16 + batch=1: run the fused bf16 kernel for BOTH raw and compressed
            // codecs. Raw chunks are read zero-copy from the mmap; compressed chunks
            // (rANS / bit-plane) are decoded to bf16 via `with_decoded_chunk` — which
            // caches the decoded bytes once when RLLM_DECODE_RESIDENT is set — and then
            // feed the SAME fused kernel, instead of materializing an f32 scratch and
            // running the generic f32 matmul (the old compressed-bf16 path). Lossless:
            // a compressed bf16 chunk decodes to the identical bf16 weights as the raw
            // chunk, so the output is bit-identical to the raw-bf16 path (same kernel,
            // same exact weights). This closes the rANS/bit-plane in-RAM speed gap to
            // the raw-bf16 ceiling — decoded bf16 IS bf16, so it equals (never beats) it.
            let kernel = |bf16_bytes: &[u8], _budget: &mut MemoryBudget| -> Result<()> {
                if bf16_bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} bf16 byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        bf16_bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                accumulate_fused_raw_bf16_chunk_batch1(
                    input,
                    &mut output,
                    bf16_bytes,
                    element_start,
                    config.linear,
                    weight_name,
                )
            };
            if chunk.codec_id == "rtc-raw-v1" {
                model.with_raw_chunk(chunk.chunk_id, budget, kernel)?;
            } else {
                model.with_decoded_chunk(chunk.chunk_id, budget, kernel)?;
            }
        } else if tensor.dtype == rllm_container::DType::Q8_0 {
            // R126: raw (identity-codec) chunks already hold the final q8 bytes, so
            // read them zero-copy from the mmap instead of paying a per-call
            // `.to_vec()` decode (which re-copies ~the whole model every token on
            // the decode path). Bytes are identical; compressed codecs still decode.
            let kernel = |quantized_bytes: &[u8], _budget: &mut MemoryBudget| -> Result<()> {
                if quantized_bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        quantized_bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                accumulate_q8_0_chunk_parallel(
                    input,
                    &mut output,
                    quantized_bytes,
                    element_start,
                    config.linear,
                    weight_name,
                )
            };
            if chunk.codec_id == "rtc-raw-v1" {
                model.with_raw_chunk(chunk.chunk_id, budget, kernel)?;
            } else {
                model.with_decoded_chunk(chunk.chunk_id, budget, kernel)?;
            }
        } else if tensor.dtype.is_quantized() {
            model.with_decoded_chunk(chunk.chunk_id, budget, |quantized_bytes, budget| {
                if quantized_bytes.len() != expected_chunk_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "chunk {} decoded byte len {} does not match metadata {}",
                        chunk.chunk_id,
                        quantized_bytes.len(),
                        expected_chunk_bytes
                    )));
                }
                let scratch_bytes =
                    quantized_elements_for_bytes(tensor.dtype, quantized_bytes.len())?
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| {
                            RuntimeError::Shape("quantized f32 scratch overflow".to_string())
                        })?;
                let scratch_label =
                    format!("streaming dequantized f32 scratch chunk {}", chunk.chunk_id);

                budget.reserve(scratch_bytes, scratch_label.clone())?;
                let weights =
                    match dequantize_quantized_chunk(tensor.dtype, chunk.chunk_id, quantized_bytes)
                    {
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
                    config.linear,
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

    let expected_weight_bytes = linear_weight_storage_bytes(tensor.dtype, config.linear)?;
    if byte_offset != expected_weight_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed {byte_offset} bytes, expected {expected_weight_bytes}"
        )));
    }

    Ok(output)
}

