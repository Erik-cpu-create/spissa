// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

// SparseColumnCache data structure + cache-management impl, plus basic and column-cached
// sparse tile-linear / SiLU kernels. input-tiled -> linear_sparse_input.rs, cache-fill +
// plain sparse silu -> linear_sparse_support.rs (R170 split). include!d into mod.rs.

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

