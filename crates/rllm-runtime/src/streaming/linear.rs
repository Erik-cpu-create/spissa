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
