// Sparse / column-cached / input-tiled streaming linear + sparse SiLU-gate-up kernels,
// plus the SparseColumnCache infrastructure. Split out of linear.rs (R167); include!d into mod.rs.

use crate::{RamaExperimentalSpeedConfig, RamaExperimentalSpeedStats};
use std::collections::HashMap;

const DEFAULT_SPARSE_COLUMN_CACHE_MAX_COLUMNS: usize = 8192;
const RLLM_AIP_COLUMN_CACHE_MAX_COLUMNS_ENV: &str = "RLLM_AIP_COLUMN_CACHE_MAX_COLUMNS";
const INPUT_TILE_SIDECAR_PREFIX: &str = "__rllm_aip_input_tiles.";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparseColumnCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub resident_columns: usize,
    pub resident_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SparseColumnKey {
    weight_name: String,
    in_feature: usize,
    in_features: usize,
    out_features: usize,
}

#[derive(Debug, Clone)]
pub struct SparseColumnCache {
    columns: HashMap<SparseColumnKey, Vec<f32>>,
    max_columns: usize,
    stats: SparseColumnCacheStats,
}

impl Default for SparseColumnCache {
    fn default() -> Self {
        Self::from_env()
    }
}

impl SparseColumnCache {
    pub fn from_env() -> Self {
        let max_columns = std::env::var(RLLM_AIP_COLUMN_CACHE_MAX_COLUMNS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_SPARSE_COLUMN_CACHE_MAX_COLUMNS);
        Self::with_max_columns(max_columns)
    }

    pub fn with_max_columns(max_columns: usize) -> Self {
        Self {
            columns: HashMap::new(),
            max_columns: max_columns.max(1),
            stats: SparseColumnCacheStats::default(),
        }
    }

    pub fn stats(&self) -> SparseColumnCacheStats {
        SparseColumnCacheStats {
            hits: self.stats.hits,
            misses: self.stats.misses,
            resident_columns: self.columns.len(),
            resident_bytes: self
                .columns
                .values()
                .map(|column| column.len() * std::mem::size_of::<f32>())
                .sum(),
        }
    }

    fn key(weight_name: &str, in_feature: usize, config: StreamingLinearConfig) -> SparseColumnKey {
        SparseColumnKey {
            weight_name: weight_name.to_string(),
            in_feature,
            in_features: config.in_features,
            out_features: config.out_features,
        }
    }

    fn can_insert(&self, count: usize) -> bool {
        self.columns.len().saturating_add(count) <= self.max_columns
    }

    fn has_column(
        &self,
        weight_name: &str,
        in_feature: usize,
        config: StreamingLinearConfig,
    ) -> bool {
        let key = Self::key(weight_name, in_feature, config);
        self.columns.contains_key(&key)
    }

    fn column_ref(
        &self,
        weight_name: &str,
        in_feature: usize,
        config: StreamingLinearConfig,
    ) -> Option<&[f32]> {
        let key = Self::key(weight_name, in_feature, config);
        self.columns.get(&key).map(Vec::as_slice)
    }

    fn record_hits(&mut self, hits: usize) {
        self.stats.hits = self.stats.hits.saturating_add(hits);
    }

    fn insert_column(
        &mut self,
        weight_name: &str,
        in_feature: usize,
        config: StreamingLinearConfig,
        column: Vec<f32>,
    ) {
        let key = Self::key(weight_name, in_feature, config);
        self.stats.misses = self.stats.misses.saturating_add(1);
        self.columns.insert(key, column);
    }
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
    let worker_count = sparse_runtime_thread_count();
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
            if worker_count > 1 {
                parallel_sparse_raw_16bit_linear_chunk_batch1(
                    input,
                    &selected,
                    &mut output,
                    raw_bytes,
                    element_start,
                    config.linear,
                    tensor.dtype,
                    weight_name,
                    worker_count,
                )
            } else {
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
            }
        })?;

        byte_offset = byte_offset
            .checked_add(expected_chunk_bytes)
            .ok_or_else(|| {
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

pub fn streaming_column_cached_sparse_tile_linear_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingTileLinearConfig,
    speed_config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
    cache: &mut SparseColumnCache,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, bias, config.linear)?;
    if !speed_config.enabled
        || !speed_config.aip_column_cache
        || config.linear.batch != 1
        || config.linear.in_features == 0
    {
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

    let selected = crate::select_top_abs_indices(
        input,
        speed_config.topk_for_input(config.linear.in_features, 256),
    );
    if selected.is_empty() {
        return Ok(None);
    }

    let before_cache = cache.stats();
    if !ensure_sparse_columns(
        model,
        weight_name,
        tensor.tensor_id,
        tensor.dtype,
        config.linear,
        &selected,
        cache,
        budget,
    )? {
        return Ok(None);
    }

    let mut output = vec![0.0f32; config.linear.out_features];
    if let Some(bias) = bias {
        output.copy_from_slice(bias);
    }
    for &in_feature in &selected {
        let Some(column) = cache.column_ref(weight_name, in_feature, config.linear) else {
            return Ok(None);
        };
        let x = input[in_feature];
        for (out, weight) in output.iter_mut().zip(column.iter()) {
            *out += x * *weight;
        }
    }

    stats.record_sparse_projection(
        selected.len(),
        config.linear.in_features,
        config.linear.out_features,
        1,
    );
    let after_cache = cache.stats();
    stats.record_column_cache(
        after_cache.hits.saturating_sub(before_cache.hits),
        after_cache.misses.saturating_sub(before_cache.misses),
        after_cache.resident_columns,
        after_cache.resident_bytes,
    );
    Ok(Some(output))
}

pub fn streaming_column_cached_sparse_silu_gate_up_from_model(
    model: &mut LazyRllmModel,
    gate_weight_name: &str,
    up_weight_name: &str,
    input: &[f32],
    config: StreamingTileLinearConfig,
    speed_config: RamaExperimentalSpeedConfig,
    stats: &mut RamaExperimentalSpeedStats,
    cache: &mut SparseColumnCache,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    validate_tile_linear_config(config)?;
    validate_linear_shapes(input, None, config.linear)?;
    if !speed_config.enabled
        || !speed_config.aip_column_cache
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
            rllm_container::DType::Fp16 | rllm_container::DType::Bf16
        )
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

    let before_cache = cache.stats();
    let gate_ready = ensure_sparse_columns(
        model,
        gate_weight_name,
        gate_tensor.tensor_id,
        gate_tensor.dtype,
        config.linear,
        &selected,
        cache,
        budget,
    )?;
    let up_ready = ensure_sparse_columns(
        model,
        up_weight_name,
        up_tensor.tensor_id,
        up_tensor.dtype,
        config.linear,
        &selected,
        cache,
        budget,
    )?;
    if !gate_ready || !up_ready {
        return Ok(None);
    }

    let mut gate_acc = vec![0.0f32; config.linear.out_features];
    let mut up_acc = vec![0.0f32; config.linear.out_features];
    for &in_feature in &selected {
        let Some(gate_column) = cache.column_ref(gate_weight_name, in_feature, config.linear)
        else {
            return Ok(None);
        };
        let Some(up_column) = cache.column_ref(up_weight_name, in_feature, config.linear) else {
            return Ok(None);
        };
        let x = input[in_feature];
        for ((gate, up), (gate_weight, up_weight)) in gate_acc
            .iter_mut()
            .zip(up_acc.iter_mut())
            .zip(gate_column.iter().zip(up_column.iter()))
        {
            *gate += x * *gate_weight;
            *up += x * *up_weight;
        }
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
    let after_cache = cache.stats();
    stats.record_column_cache(
        after_cache.hits.saturating_sub(before_cache.hits),
        after_cache.misses.saturating_sub(before_cache.misses),
        after_cache.resident_columns,
        after_cache.resident_bytes,
    );
    Ok(Some(output))
}

pub fn streaming_input_tiled_sparse_tile_linear_from_model(
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
    model: &mut LazyRllmModel,
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
    model: &mut LazyRllmModel,
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
        rllm_container::DType::Fp16 | rllm_container::DType::Bf16
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
            rllm_container::DType::Fp16 | rllm_container::DType::Bf16
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
    tensor: &rllm_container::TensorMeta,
    config: StreamingLinearConfig,
    dtype: rllm_container::DType,
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
    dtype: rllm_container::DType,
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

