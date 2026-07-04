// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

// Input-tiled sparse linear + SiLU-gate-up kernels + InputTileColumnRange / input-tile
// column helpers. Split out of linear_sparse.rs (R170); include!d into streaming/mod.rs.

pub fn streaming_input_tiled_sparse_tile_linear_from_model(
    model: &mut LazySpissaModel,
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
    if !speed_config.enabled
        || !speed_config.aip_input_tiles
        || config.linear.batch != 1
        || config.linear.in_features == 0
    {
        return Ok(None);
    }

    let selected = crate::select_top_abs_indices(
        input,
        speed_config.topk_for_input(config.linear.in_features, 256),
    );
    streaming_input_tiled_sparse_tile_linear_selected_inner(
        model,
        weight_name,
        input,
        bias,
        config,
        &selected,
        stats,
        budget,
    )
}

pub fn streaming_input_tiled_sparse_tile_linear_selected_from_model(
    model: &mut LazySpissaModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    selected: &[usize],
    stats: &mut RamaExperimentalSpeedStats,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, bias, config.linear)?;
    if config.linear.batch != 1 || config.linear.in_features == 0 {
        return Ok(None);
    }
    streaming_input_tiled_sparse_tile_linear_selected_inner(
        model,
        weight_name,
        input,
        bias,
        config,
        selected,
        stats,
        budget,
    )
}

fn streaming_input_tiled_sparse_tile_linear_selected_inner(
    model: &mut LazySpissaModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    selected: &[usize],
    stats: &mut RamaExperimentalSpeedStats,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    if selected.is_empty()
        || selected
            .iter()
            .any(|in_feature| *in_feature >= config.linear.in_features)
    {
        return Ok(None);
    }

    let tensor = model.tensor(weight_name)?.clone();
    validate_weight_tensor(&tensor, config.linear)?;
    if !matches!(
        tensor.dtype,
        spissa_container::DType::Fp16 | spissa_container::DType::Bf16
    ) {
        return Ok(None);
    }
    let sidecar_name = input_tile_sidecar_weight_name(weight_name);
    let sidecar_tensor = match model.tensor(&sidecar_name) {
        Ok(tensor) => tensor.clone(),
        Err(RuntimeError::MissingTensor(_)) => return Ok(None),
        Err(err) => return Err(err),
    };
    if !input_tile_sidecar_tensor_matches(&sidecar_tensor, config.linear, tensor.dtype)? {
        return Ok(None);
    }

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(sidecar_tensor.tensor_id).to_vec();
    if chunks.is_empty() {
        return Ok(None);
    }

    let mut output = vec![0.0f32; config.linear.out_features];
    if let Some(bias) = bias {
        output.copy_from_slice(bias);
    }

    let dtype_size = tensor.dtype.size_bytes();
    let mut range_reads = 0usize;
    let mut range_bytes = 0usize;
    for &in_feature in selected {
        let Some(range) = input_tile_column_range(&chunks, in_feature, config.linear, dtype_size)?
        else {
            return Ok(None);
        };
        let x = input[in_feature];
        model.with_raw_chunk_range(
            range.chunk_id,
            range.byte_offset,
            range.byte_len,
            budget,
            |bytes, _budget| {
                accumulate_input_tile_column(
                    bytes,
                    x,
                    tensor.dtype,
                    &mut output,
                    weight_name,
                    config.linear,
                )
            },
        )?;
        range_reads = range_reads.saturating_add(1);
        range_bytes =
            range_bytes.saturating_add(usize::try_from(range.byte_len).unwrap_or(usize::MAX));
    }

    stats.record_sparse_projection(
        selected.len(),
        config.linear.in_features,
        config.linear.out_features,
        1,
    );
    stats.record_input_tile_ranges(range_reads, range_bytes);
    Ok(Some(output))
}

pub fn streaming_input_tiled_sparse_silu_gate_up_from_model(
    model: &mut LazySpissaModel,
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
    if !speed_config.enabled
        || !speed_config.aip_input_tiles
        || config.linear.batch != 1
        || config.linear.in_features == 0
    {
        return Ok(None);
    }

    let gate_tensor = model.tensor(gate_weight_name)?.clone();
    let up_tensor = model.tensor(up_weight_name)?.clone();
    validate_weight_tensor(&gate_tensor, config.linear)?;
    validate_weight_tensor(&up_tensor, config.linear)?;
    if gate_tensor.dtype != up_tensor.dtype
        || !matches!(
            gate_tensor.dtype,
            spissa_container::DType::Fp16 | spissa_container::DType::Bf16
        )
    {
        return Ok(None);
    }

    let gate_sidecar_name = input_tile_sidecar_weight_name(gate_weight_name);
    let up_sidecar_name = input_tile_sidecar_weight_name(up_weight_name);
    let gate_sidecar = match model.tensor(&gate_sidecar_name) {
        Ok(tensor) => tensor.clone(),
        Err(RuntimeError::MissingTensor(_)) => return Ok(None),
        Err(err) => return Err(err),
    };
    let up_sidecar = match model.tensor(&up_sidecar_name) {
        Ok(tensor) => tensor.clone(),
        Err(RuntimeError::MissingTensor(_)) => return Ok(None),
        Err(err) => return Err(err),
    };
    if !input_tile_sidecar_tensor_matches(&gate_sidecar, config.linear, gate_tensor.dtype)?
        || !input_tile_sidecar_tensor_matches(&up_sidecar, config.linear, up_tensor.dtype)?
    {
        return Ok(None);
    }

    let selected = crate::select_top_abs_indices(
        input,
        speed_config.topk_for_input(config.linear.in_features, 256),
    );
    if selected.is_empty() {
        return Ok(None);
    }

    let gate_chunks: Vec<ChunkMeta> = model.chunks_for_tensor(gate_sidecar.tensor_id).to_vec();
    let up_chunks: Vec<ChunkMeta> = model.chunks_for_tensor(up_sidecar.tensor_id).to_vec();
    if gate_chunks.is_empty() || up_chunks.is_empty() {
        return Ok(None);
    }

    let mut gate_acc = vec![0.0f32; config.linear.out_features];
    let mut up_acc = vec![0.0f32; config.linear.out_features];
    let dtype_size = gate_tensor.dtype.size_bytes();
    let mut range_reads = 0usize;
    let mut range_bytes = 0usize;
    for &in_feature in &selected {
        let Some(gate_range) =
            input_tile_column_range(&gate_chunks, in_feature, config.linear, dtype_size)?
        else {
            return Ok(None);
        };
        let Some(up_range) =
            input_tile_column_range(&up_chunks, in_feature, config.linear, dtype_size)?
        else {
            return Ok(None);
        };
        let x = input[in_feature];
        model.with_raw_chunk_range(
            gate_range.chunk_id,
            gate_range.byte_offset,
            gate_range.byte_len,
            budget,
            |bytes, _budget| {
                accumulate_input_tile_column(
                    bytes,
                    x,
                    gate_tensor.dtype,
                    &mut gate_acc,
                    gate_weight_name,
                    config.linear,
                )
            },
        )?;
        model.with_raw_chunk_range(
            up_range.chunk_id,
            up_range.byte_offset,
            up_range.byte_len,
            budget,
            |bytes, _budget| {
                accumulate_input_tile_column(
                    bytes,
                    x,
                    up_tensor.dtype,
                    &mut up_acc,
                    up_weight_name,
                    config.linear,
                )
            },
        )?;
        range_reads = range_reads.saturating_add(2);
        range_bytes = range_bytes
            .saturating_add(usize::try_from(gate_range.byte_len).unwrap_or(usize::MAX))
            .saturating_add(usize::try_from(up_range.byte_len).unwrap_or(usize::MAX));
    }

    let mut output = Vec::with_capacity(config.linear.out_features);
    for (gate, up) in gate_acc.into_iter().zip(up_acc) {
        output.push(crate::silu(gate) * up);
    }

    stats.record_sparse_projection(
        selected.len(),
        config.linear.in_features,
        config.linear.out_features,
        2,
    );
    stats.record_input_tile_ranges(range_reads, range_bytes);
    Ok(Some(output))
}

#[derive(Debug, Clone, Copy)]
struct InputTileColumnRange {
    chunk_id: u64,
    byte_offset: u64,
    byte_len: u64,
}

fn input_tile_sidecar_tensor_matches(
    tensor: &spissa_container::TensorMeta,
    config: StreamingLinearConfig,
    dtype: spissa_container::DType,
) -> Result<bool> {
    if tensor.dtype != dtype || tensor.shape.len() != 2 {
        return Ok(false);
    }
    let sidecar_in = usize::try_from(tensor.shape[0])
        .map_err(|_| RuntimeError::Shape("input-tile sidecar input dim overflow".to_string()))?;
    let sidecar_out = usize::try_from(tensor.shape[1])
        .map_err(|_| RuntimeError::Shape("input-tile sidecar output dim overflow".to_string()))?;
    if sidecar_in != config.in_features || sidecar_out != config.out_features {
        return Ok(false);
    }
    let expected_bytes = config
        .in_features
        .checked_mul(config.out_features)
        .and_then(|elements| elements.checked_mul(dtype.size_bytes()))
        .ok_or_else(|| RuntimeError::Shape("input-tile sidecar byte size overflow".to_string()))?;
    Ok(tensor.original_size_bytes == expected_bytes as u64)
}

fn input_tile_column_range(
    chunks: &[ChunkMeta],
    in_feature: usize,
    config: StreamingLinearConfig,
    dtype_size: usize,
) -> Result<Option<InputTileColumnRange>> {
    if in_feature >= config.in_features || dtype_size == 0 {
        return Ok(None);
    }
    let column_elements = config.out_features;
    let column_start = in_feature
        .checked_mul(config.out_features)
        .ok_or_else(|| RuntimeError::Shape("input-tile column start overflow".to_string()))?;
    let column_end = column_start
        .checked_add(column_elements)
        .ok_or_else(|| RuntimeError::Shape("input-tile column end overflow".to_string()))?;
    let column_bytes = column_elements
        .checked_mul(dtype_size)
        .ok_or_else(|| RuntimeError::Shape("input-tile column byte len overflow".to_string()))?;

    for chunk in chunks {
        if chunk.codec_id != "rtc-raw-v1"
            || !chunk.uncompressed_size.is_multiple_of(dtype_size as u64)
        {
            return Ok(None);
        }
        let chunk_start = usize::try_from(chunk.chunk_offset_in_tensor).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "input-tile chunk {} offset overflows usize",
                chunk.chunk_id
            ))
        })?;
        let chunk_elements = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "input-tile chunk {} size overflows usize",
                chunk.chunk_id
            ))
        })? / dtype_size;
        let chunk_end = chunk_start.checked_add(chunk_elements).ok_or_else(|| {
            RuntimeError::InvalidTensorData("input-tile chunk element end overflow".to_string())
        })?;
        if column_start >= chunk_start && column_end <= chunk_end {
            let byte_offset = (column_start - chunk_start)
                .checked_mul(dtype_size)
                .ok_or_else(|| {
                    RuntimeError::Shape("input-tile range byte offset overflow".to_string())
                })?;
            let byte_offset_u64 = byte_offset as u64;
            let column_bytes_u64 = column_bytes as u64;
            let has_range = chunk.range_checksums.iter().any(|range| {
                range.original_offset == byte_offset_u64 && range.original_size == column_bytes_u64
            });
            if !has_range {
                return Ok(None);
            }
            return Ok(Some(InputTileColumnRange {
                chunk_id: chunk.chunk_id,
                byte_offset: byte_offset_u64,
                byte_len: column_bytes_u64,
            }));
        }
    }

    Ok(None)
}

fn accumulate_input_tile_column(
    raw_bytes: &[u8],
    input_value: f32,
    dtype: spissa_container::DType,
    output: &mut [f32],
    weight_name: &str,
    config: StreamingLinearConfig,
) -> Result<()> {
    if raw_bytes.len() != config.out_features * dtype.size_bytes() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "input-tile range for {weight_name} decoded to {} bytes, expected {}",
            raw_bytes.len(),
            config.out_features * dtype.size_bytes()
        )));
    }
    if output.len() != config.out_features {
        return Err(RuntimeError::Shape(format!(
            "input-tile output len {} does not match out_features {}",
            output.len(),
            config.out_features
        )));
    }
    for (out_feature, out_value) in output.iter_mut().enumerate() {
        *out_value += input_value * raw_16bit_weight_at(raw_bytes, out_feature, dtype);
    }
    Ok(())
}

