// Sparse column-cache fill (ensure/fill) + plain sparse SiLU-gate-up + complete-rows
// helper. Split out of linear_sparse.rs (R170); include!d into streaming/mod.rs.

fn ensure_sparse_columns(
    model: &mut LazyRllmModel,
    weight_name: &str,
    tensor_id: u64,
    dtype: rllm_container::DType,
    config: StreamingLinearConfig,
    selected: &[usize],
    cache: &mut SparseColumnCache,
    budget: &mut MemoryBudget,
) -> Result<bool> {
    let mut existing = 0usize;
    let mut missing = Vec::new();
    for &in_feature in selected {
        if cache.has_column(weight_name, in_feature, config) {
            existing = existing.saturating_add(1);
        } else {
            missing.push(in_feature);
        }
    }
    cache.record_hits(existing);
    if missing.is_empty() {
        return Ok(true);
    }
    if !cache.can_insert(missing.len()) {
        return Ok(false);
    }

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor_id).to_vec();
    if chunks.is_empty() || chunks.iter().any(|chunk| chunk.codec_id != "rtc-raw-v1") {
        return Ok(false);
    }

    let mut new_columns: Vec<(usize, Vec<f32>)> = missing
        .iter()
        .map(|&in_feature| (in_feature, vec![0.0f32; config.out_features]))
        .collect();
    let dtype_size = dtype.size_bytes();
    let mut byte_offset = 0usize;
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} column cache reached unaligned byte offset {byte_offset}"
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
            fill_sparse_column_cache_chunk(
                raw_bytes,
                element_start,
                config,
                dtype,
                weight_name,
                &mut new_columns,
            )
        })?;

        byte_offset = byte_offset
            .checked_add(expected_chunk_bytes)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData("column cache byte offset overflow".to_string())
            })?;
    }

    for (in_feature, column) in new_columns {
        cache.insert_column(weight_name, in_feature, config, column);
    }
    Ok(true)
}

fn fill_sparse_column_cache_chunk(
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
    weight_name: &str,
    columns: &mut [(usize, Vec<f32>)],
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Column cache raw 16-bit stream for {weight_name} has odd length"
        )));
    }
    let weight_elements = raw_bytes.len() / 2;
    let element_end = element_start.checked_add(weight_elements).ok_or_else(|| {
        RuntimeError::Shape("column cache chunk element range overflow".to_string())
    })?;
    let expected = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| {
            RuntimeError::Shape("column cache weight element count overflow".to_string())
        })?;
    if element_end > expected {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} column cache chunk elements {element_start}..{element_end} exceed expected {expected}"
        )));
    }
    if weight_elements == 0 {
        return Ok(());
    }

    let first_row = element_start / config.in_features;
    let last_row = element_end.saturating_sub(1) / config.in_features;
    for out_feature in first_row..=last_row {
        let row_base = out_feature * config.in_features;
        for (in_feature, column) in columns.iter_mut() {
            let global = row_base + *in_feature;
            if global >= element_start && global < element_end {
                let local = global - element_start;
                column[out_feature] = raw_16bit_weight_at(raw_bytes, local, dtype);
            }
        }
    }
    Ok(())
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
    let dtype_size = gate_tensor.dtype.size_bytes();
    let worker_count = sparse_runtime_thread_count();
    let use_parallel_rows = worker_count > 1
        && sparse_chunks_are_complete_rows(&gate_chunks, config.linear.in_features, dtype_size)?;
    if use_parallel_rows {
        let mut byte_offset = 0usize;
        for (gate_chunk, up_chunk) in gate_chunks.iter().zip(up_chunks.iter()) {
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
                    parallel_sparse_silu_gate_up_raw_16bit_chunk_batch1(
                        input,
                        &selected,
                        gate_bytes,
                        up_bytes,
                        element_start,
                        config.linear,
                        gate_tensor.dtype,
                        &mut output,
                        gate_weight_name,
                        worker_count,
                    )
                },
            )?;

            byte_offset = byte_offset
                .checked_add(expected_chunk_bytes)
                .ok_or_else(|| {
                    RuntimeError::InvalidTensorData(
                        "sparse gate/up byte offset overflow".to_string(),
                    )
                })?;
        }
    } else {
        let mut state = SiluGateUpState::new(&mut output);
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

            byte_offset = byte_offset
                .checked_add(expected_chunk_bytes)
                .ok_or_else(|| {
                    RuntimeError::InvalidTensorData(
                        "sparse gate/up byte offset overflow".to_string(),
                    )
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

fn sparse_chunks_are_complete_rows(
    chunks: &[ChunkMeta],
    in_features: usize,
    dtype_size: usize,
) -> Result<bool> {
    if in_features == 0 || dtype_size == 0 {
        return Ok(false);
    }

    let mut byte_offset = 0usize;
    for chunk in chunks {
        if !byte_offset.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "sparse chunk stream reached unaligned byte offset {byte_offset}"
            )));
        }
        let chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;
        if !chunk_bytes.is_multiple_of(dtype_size) {
            return Ok(false);
        }

        let element_start = byte_offset / dtype_size;
        let chunk_elements = chunk_bytes / dtype_size;
        if !element_start.is_multiple_of(in_features) || !chunk_elements.is_multiple_of(in_features)
        {
            return Ok(false);
        }

        byte_offset = byte_offset.checked_add(chunk_bytes).ok_or_else(|| {
            RuntimeError::InvalidTensorData("sparse chunk byte offset overflow".to_string())
        })?;
    }

    Ok(true)
}

