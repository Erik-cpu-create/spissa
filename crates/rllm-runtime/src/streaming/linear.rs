// Core dense streaming linear matmul (streaming_linear / tile_linear / multiply_into)
// + SiLU-gate-up + shared helpers. Sparse -> linear_sparse.rs, argmax -> linear_argmax.rs (R167).

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

    // R133 decode fast-path: raw-codec Q8_0 + batch=1 (gated on RLLM_Q8_ACTIVATION).
    // View the whole tensor as one contiguous mmap slice and run int8 sdot
    // row-parallel — bypassing the per-chunk dispatch + scalar i8×f32 path.
    // Falls through to the per-chunk path if the tensor isn't contiguous-raw.
    if tensor.dtype == rllm_container::DType::Q8_0
        && config.linear.batch == 1
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

    // R133-style decode fast-path for the multiply-into projection (LLaMA up_proj
    // at batch=1). The regular tile-linear path got the whole-tensor int8 sdot in
    // R133, but multiply_into only had the batch>=2 panel path — so up_proj fell to
    // the scalar per-chunk path on DECODE (~107ms/token vs ~6ms for gate_proj on
    // Llama 1B). Compute the projection with the whole-tensor int8 sdot kernel into
    // a scratch, then apply `target *= up + bias`. Falls through if not contiguous-raw.
    if tensor.dtype == rllm_container::DType::Q8_0
        && config.linear.batch == 1
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
    if tensor.dtype == rllm_container::DType::Q8_0
        && q8_activation_path_enabled()
        && config.linear.batch >= 2
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
        } else if tensor.dtype == rllm_container::DType::Q8_0 {
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
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    dtype: rllm_container::DType,
    chunks: &[ChunkMeta],
    config: StreamingLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    if dtype != rllm_container::DType::Q8_0 {
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

