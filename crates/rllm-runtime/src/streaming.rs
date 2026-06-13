use crate::tensor::decode_to_f32;
use crate::{
    apply_gpt_neox_rotary_inplace, scaled_dot_product_attention_with_cache, KvAttentionConfig,
    KvCache, LazyRllmModel, MemoryBudget, Result, RotaryEmbeddingConfig, RuntimeError,
};
use rllm_container::{ChunkMeta, TensorMeta};
use std::time::{Duration, Instant};

pub const DEFAULT_STREAMING_TILE_ELEMENTS: usize = 4096;

#[derive(Debug, Clone, Copy)]
pub struct StreamingLinearConfig {
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingTileLinearConfig {
    pub linear: StreamingLinearConfig,
    /// Maximum number of weight elements converted into f32 scratch at once.
    ///
    /// Current RTC codecs still decode one compressed chunk to original bytes;
    /// Phase 7 starts by fusing f32 conversion and matmul accumulation over
    /// bounded tiles instead of materializing a full f32 chunk.
    pub tile_elements: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingMlpConfig {
    pub batch: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingAttentionConfig {
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
}

#[derive(Debug)]
pub struct StreamingAttentionRuntime<'a> {
    pub rotary: Option<RotaryEmbeddingConfig>,
    pub kv_cache: Option<&'a mut KvCache>,
}

impl Default for StreamingAttentionRuntime<'_> {
    fn default() -> Self {
        Self {
            rotary: None,
            kv_cache: None,
        }
    }
}

impl StreamingAttentionConfig {
    fn hidden_size(self) -> Result<usize> {
        self.num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| RuntimeError::Shape("attention hidden_size overflow".to_string()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingBlockConfig {
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub causal: bool,
    pub layer_norm_eps: f32,
}

#[derive(Debug, Default)]
pub struct StreamingBlockRuntime<'a> {
    pub attention: StreamingAttentionRuntime<'a>,
    /// GPT-NeoX/Pythia can use parallel residual blocks:
    /// `x + attention(LN1(x)) + mlp(LN2(x))`.
    ///
    /// The default remains the older sequential pre-norm toy block:
    /// `x + attention(LN1(x)) -> LN2(residual) -> + mlp(...)`.
    pub parallel_residual: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamingBlockTiming {
    pub attention_norm_ns: u64,
    pub attention_ns: u64,
    pub attention_qkv_projection_ns: u64,
    pub attention_qkv_split_ns: u64,
    pub attention_rotary_ns: u64,
    pub attention_score_context_ns: u64,
    pub attention_output_projection_ns: u64,
    pub attention_kv_append_ns: u64,
    pub attention_residual_ns: u64,
    pub mlp_norm_ns: u64,
    pub mlp_ns: u64,
    pub mlp_input_projection_ns: u64,
    pub mlp_activation_ns: u64,
    pub mlp_output_projection_ns: u64,
    pub mlp_residual_ns: u64,
    pub attention_norm_calls: usize,
    pub attention_calls: usize,
    pub attention_qkv_projection_calls: usize,
    pub attention_qkv_split_calls: usize,
    pub attention_rotary_calls: usize,
    pub attention_score_context_calls: usize,
    pub attention_output_projection_calls: usize,
    pub attention_kv_append_calls: usize,
    pub attention_residual_calls: usize,
    pub mlp_norm_calls: usize,
    pub mlp_calls: usize,
    pub mlp_input_projection_calls: usize,
    pub mlp_activation_calls: usize,
    pub mlp_output_projection_calls: usize,
    pub mlp_residual_calls: usize,
}

impl StreamingBlockTiming {
    fn record_attention_norm(&mut self, elapsed: Duration) {
        self.attention_norm_ns = self
            .attention_norm_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_norm_calls = self.attention_norm_calls.saturating_add(1);
    }

    fn record_attention(&mut self, elapsed: Duration) {
        self.attention_ns = self.attention_ns.saturating_add(elapsed_ns_u64(elapsed));
        self.attention_calls = self.attention_calls.saturating_add(1);
    }

    fn record_attention_qkv_projection(&mut self, elapsed: Duration) {
        self.attention_qkv_projection_ns = self
            .attention_qkv_projection_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_qkv_projection_calls = self.attention_qkv_projection_calls.saturating_add(1);
    }

    fn record_attention_qkv_split(&mut self, elapsed: Duration) {
        self.attention_qkv_split_ns = self
            .attention_qkv_split_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_qkv_split_calls = self.attention_qkv_split_calls.saturating_add(1);
    }

    fn record_attention_rotary(&mut self, elapsed: Duration) {
        self.attention_rotary_ns = self
            .attention_rotary_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_rotary_calls = self.attention_rotary_calls.saturating_add(1);
    }

    fn record_attention_score_context(&mut self, elapsed: Duration) {
        self.attention_score_context_ns = self
            .attention_score_context_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_score_context_calls = self.attention_score_context_calls.saturating_add(1);
    }

    fn record_attention_output_projection(&mut self, elapsed: Duration) {
        self.attention_output_projection_ns = self
            .attention_output_projection_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_output_projection_calls =
            self.attention_output_projection_calls.saturating_add(1);
    }

    fn record_attention_kv_append(&mut self, elapsed: Duration) {
        self.attention_kv_append_ns = self
            .attention_kv_append_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_kv_append_calls = self.attention_kv_append_calls.saturating_add(1);
    }

    fn record_attention_residual(&mut self, elapsed: Duration) {
        self.attention_residual_ns = self
            .attention_residual_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_residual_calls = self.attention_residual_calls.saturating_add(1);
    }

    fn record_mlp_norm(&mut self, elapsed: Duration) {
        self.mlp_norm_ns = self.mlp_norm_ns.saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_norm_calls = self.mlp_norm_calls.saturating_add(1);
    }

    fn record_mlp(&mut self, elapsed: Duration) {
        self.mlp_ns = self.mlp_ns.saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_calls = self.mlp_calls.saturating_add(1);
    }

    fn record_mlp_input_projection(&mut self, elapsed: Duration) {
        self.mlp_input_projection_ns = self
            .mlp_input_projection_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_input_projection_calls = self.mlp_input_projection_calls.saturating_add(1);
    }

    fn record_mlp_activation(&mut self, elapsed: Duration) {
        self.mlp_activation_ns = self
            .mlp_activation_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_activation_calls = self.mlp_activation_calls.saturating_add(1);
    }

    fn record_mlp_output_projection(&mut self, elapsed: Duration) {
        self.mlp_output_projection_ns = self
            .mlp_output_projection_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_output_projection_calls = self.mlp_output_projection_calls.saturating_add(1);
    }

    fn record_mlp_residual(&mut self, elapsed: Duration) {
        self.mlp_residual_ns = self.mlp_residual_ns.saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_residual_calls = self.mlp_residual_calls.saturating_add(1);
    }
}

fn elapsed_ns_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl StreamingBlockConfig {
    fn hidden_size(self) -> Result<usize> {
        self.num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| RuntimeError::Shape("block hidden_size overflow".to_string()))
    }

    fn attention_config(self) -> StreamingAttentionConfig {
        StreamingAttentionConfig {
            seq_len: self.seq_len,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            causal: self.causal,
        }
    }

    fn mlp_config(self) -> Result<StreamingMlpConfig> {
        Ok(StreamingMlpConfig {
            batch: self.seq_len,
            hidden_size: self.hidden_size()?,
            intermediate_size: self.intermediate_size,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingBlockTensorNames<'a> {
    pub qkv_weight: &'a str,
    pub attention_out_weight: &'a str,
    pub mlp_in_weight: &'a str,
    pub mlp_out_weight: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingBlockParameters<'a> {
    pub input_layernorm_weight: &'a [f32],
    pub input_layernorm_bias: &'a [f32],
    pub qkv_bias: Option<&'a [f32]>,
    pub attention_out_bias: Option<&'a [f32]>,
    pub post_attention_layernorm_weight: &'a [f32],
    pub post_attention_layernorm_bias: &'a [f32],
    pub mlp_in_bias: Option<&'a [f32]>,
    pub mlp_out_bias: Option<&'a [f32]>,
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
        if byte_offset % dtype_size != 0 {
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
        if byte_offset % dtype_size != 0 {
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
                    let tile_byte_start = local_element_start
                        .checked_mul(dtype_size)
                        .ok_or_else(|| RuntimeError::Shape("tile byte start overflow".to_string()))?;
                    let tile_byte_len = tile_len
                        .checked_mul(dtype_size)
                        .ok_or_else(|| RuntimeError::Shape("tile byte len overflow".to_string()))?;
                    let tile_byte_end = tile_byte_start
                        .checked_add(tile_byte_len)
                        .ok_or_else(|| RuntimeError::Shape("tile byte end overflow".to_string()))?;
                    let scratch_bytes = tile_len
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| RuntimeError::Shape("tile f32 scratch overflow".to_string()))?;
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
        if byte_offset % dtype_size != 0 {
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

            let elements_in_chunk = bytes.len() / dtype_size;
            let mut local_element_start = 0usize;
            while local_element_start < elements_in_chunk {
                let tile_len = config
                    .tile_elements
                    .min(elements_in_chunk - local_element_start);
                let tile_byte_start = local_element_start
                    .checked_mul(dtype_size)
                    .ok_or_else(|| RuntimeError::Shape("tile byte start overflow".to_string()))?;
                let tile_byte_len = tile_len
                    .checked_mul(dtype_size)
                    .ok_or_else(|| RuntimeError::Shape("tile byte len overflow".to_string()))?;
                let tile_byte_end = tile_byte_start
                    .checked_add(tile_byte_len)
                    .ok_or_else(|| RuntimeError::Shape("tile byte end overflow".to_string()))?;
                let scratch_bytes = tile_len
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| RuntimeError::Shape("tile f32 scratch overflow".to_string()))?;
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

fn streaming_default_tile_linear_from_model(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_tile_linear_from_model(
        model,
        weight_name,
        input,
        bias,
        StreamingTileLinearConfig {
            linear: config,
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        },
        budget,
    )
}

/// Low-RAM two-layer MLP block over chunked `.rllm` weight tensors.
///
/// Computes `Linear(input, w_in, b_in) -> GELU -> Linear(hidden, w_out, b_out)`.
/// The intermediate activation is reserved in `budget` for the duration of the
/// second linear pass, while each weight chunk is still decoded/released one at
/// a time through the default Phase 7 tiled linear path.
pub fn streaming_mlp_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    w_in_name: &str,
    b_in: Option<&[f32]>,
    w_out_name: &str,
    b_out: Option<&[f32]>,
    config: StreamingMlpConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_mlp_with_timing_from_model(
        model, input, w_in_name, b_in, w_out_name, b_out, config, budget, None,
    )
}

fn streaming_mlp_with_timing_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    w_in_name: &str,
    b_in: Option<&[f32]>,
    w_out_name: &str,
    b_out: Option<&[f32]>,
    config: StreamingMlpConfig,
    budget: &mut MemoryBudget,
    mut timing: Option<&mut StreamingBlockTiming>,
) -> Result<Vec<f32>> {
    validate_mlp_shapes(input, b_in, b_out, config)?;

    let intermediate_bytes = activation_bytes(
        config.batch,
        config.intermediate_size,
        "streaming MLP intermediate",
    )?;
    let intermediate_label = "streaming MLP intermediate activation".to_string();
    budget.reserve(intermediate_bytes, intermediate_label.clone())?;

    let input_projection_started = Instant::now();
    let mut hidden = match streaming_default_tile_linear_from_model(
        model,
        w_in_name,
        input,
        b_in,
        StreamingLinearConfig {
            batch: config.batch,
            in_features: config.hidden_size,
            out_features: config.intermediate_size,
        },
        budget,
    ) {
        Ok(hidden) => hidden,
        Err(err) => {
            budget.release(intermediate_bytes, intermediate_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp_input_projection(input_projection_started.elapsed());
    }

    let activation_started = Instant::now();
    crate::ops::gelu_inplace(&mut hidden);
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp_activation(activation_started.elapsed());
    }

    let output_projection_started = Instant::now();
    let output = match streaming_default_tile_linear_from_model(
        model,
        w_out_name,
        &hidden,
        b_out,
        StreamingLinearConfig {
            batch: config.batch,
            in_features: config.intermediate_size,
            out_features: config.hidden_size,
        },
        budget,
    ) {
        Ok(output) => output,
        Err(err) => {
            budget.release(intermediate_bytes, intermediate_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp_output_projection(output_projection_started.elapsed());
    }

    drop(hidden);
    budget.release(intermediate_bytes, intermediate_label)?;
    Ok(output)
}

/// Low-RAM attention sub-block over chunked `.rllm` QKV and output weights.
///
/// This implements the non-rotary toy baseline used by Phase 5 tests:
/// `tiled QKV linear -> split Q/K/V -> scaled dot-product attention ->
/// tiled output projection`. Real GPT-NeoX rotary embeddings are layered on
/// through `StreamingAttentionRuntime`.
pub fn streaming_attention_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    qkv_weight_name: &str,
    qkv_bias: Option<&[f32]>,
    out_weight_name: &str,
    out_bias: Option<&[f32]>,
    config: StreamingAttentionConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_attention_with_runtime_from_model(
        model,
        input,
        qkv_weight_name,
        qkv_bias,
        out_weight_name,
        out_bias,
        config,
        StreamingAttentionRuntime::default(),
        budget,
    )
}

pub fn streaming_attention_with_runtime_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    qkv_weight_name: &str,
    qkv_bias: Option<&[f32]>,
    out_weight_name: &str,
    out_bias: Option<&[f32]>,
    config: StreamingAttentionConfig,
    runtime: StreamingAttentionRuntime<'_>,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_attention_with_runtime_and_timing_from_model(
        model,
        input,
        qkv_weight_name,
        qkv_bias,
        out_weight_name,
        out_bias,
        config,
        runtime,
        budget,
        None,
    )
}

fn streaming_attention_with_runtime_and_timing_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    qkv_weight_name: &str,
    qkv_bias: Option<&[f32]>,
    out_weight_name: &str,
    out_bias: Option<&[f32]>,
    config: StreamingAttentionConfig,
    mut runtime: StreamingAttentionRuntime<'_>,
    budget: &mut MemoryBudget,
    mut timing: Option<&mut StreamingBlockTiming>,
) -> Result<Vec<f32>> {
    validate_attention_shapes(input, qkv_bias, out_bias, config)?;
    let hidden_size = config.hidden_size()?;
    let qkv_features = hidden_size
        .checked_mul(3)
        .ok_or_else(|| RuntimeError::Shape("QKV feature count overflow".to_string()))?;

    let qkv_bytes = activation_bytes(config.seq_len, qkv_features, "streaming attention QKV")?;
    let qkv_label = "streaming attention fused QKV activation".to_string();
    budget.reserve(qkv_bytes, qkv_label.clone())?;

    let qkv_projection_started = Instant::now();
    let fused_qkv = match streaming_default_tile_linear_from_model(
        model,
        qkv_weight_name,
        input,
        qkv_bias,
        StreamingLinearConfig {
            batch: config.seq_len,
            in_features: hidden_size,
            out_features: qkv_features,
        },
        budget,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(qkv_bytes, qkv_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_qkv_projection(qkv_projection_started.elapsed());
    }

    let split_label = "streaming attention split QKV activation".to_string();
    if let Err(err) = budget.reserve(qkv_bytes, split_label.clone()) {
        budget.release(qkv_bytes, qkv_label)?;
        return Err(err);
    }
    let qkv_split_started = Instant::now();
    let (mut q, mut k, v) = match split_fused_qkv(&fused_qkv, config) {
        Ok(split) => split,
        Err(err) => {
            budget.release(qkv_bytes, split_label)?;
            budget.release(qkv_bytes, qkv_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_qkv_split(qkv_split_started.elapsed());
    }
    drop(fused_qkv);
    budget.release(qkv_bytes, qkv_label)?;

    if let Some(rotary) = runtime.rotary {
        let rotary_started = Instant::now();
        if let Err(err) = apply_gpt_neox_rotary_inplace(&mut q, &mut k, rotary) {
            budget.release(qkv_bytes, split_label)?;
            return Err(err);
        }
        if let Some(timing) = timing.as_deref_mut() {
            timing.record_attention_rotary(rotary_started.elapsed());
        }
    }

    let cache_is_active = runtime.kv_cache.is_some();

    let attention_bytes = activation_bytes(
        config.seq_len,
        hidden_size,
        "streaming attention output activation",
    )?;
    let attention_label = "streaming attention output activation".to_string();
    if let Err(err) = budget.reserve(attention_bytes, attention_label.clone()) {
        budget.release(qkv_bytes, split_label)?;
        return Err(err);
    }
    let cache_view = runtime.kv_cache.as_ref().map(|cache| &**cache);
    let score_context_started = Instant::now();
    let attended = match scaled_dot_product_attention_with_cache(
        &q,
        &k,
        &v,
        cache_view,
        KvAttentionConfig {
            query_len: config.seq_len,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            causal: config.causal,
        },
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(attention_bytes, attention_label)?;
            budget.release(qkv_bytes, split_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_score_context(score_context_started.elapsed());
    }
    if !cache_is_active {
        budget.release(qkv_bytes, split_label.clone())?;
    }

    let output_projection_started = Instant::now();
    let output = match streaming_default_tile_linear_from_model(
        model,
        out_weight_name,
        &attended,
        out_bias,
        StreamingLinearConfig {
            batch: config.seq_len,
            in_features: hidden_size,
            out_features: hidden_size,
        },
        budget,
    ) {
        Ok(output) => output,
        Err(err) => {
            budget.release(attention_bytes, attention_label)?;
            if cache_is_active {
                budget.release(qkv_bytes, split_label)?;
            }
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_output_projection(output_projection_started.elapsed());
    }

    if let Some(cache) = runtime.kv_cache.as_deref_mut() {
        let kv_append_started = Instant::now();
        if let Err(err) = cache.append(&k, &v, config.seq_len) {
            budget.release(attention_bytes, attention_label)?;
            budget.release(qkv_bytes, split_label)?;
            return Err(err);
        }
        if let Some(timing) = timing.as_deref_mut() {
            timing.record_attention_kv_append(kv_append_started.elapsed());
        }
        budget.release(qkv_bytes, split_label)?;
    }

    drop((q, k, v));
    drop(attended);
    budget.release(attention_bytes, attention_label)?;
    Ok(output)
}

/// Low-RAM pre-norm transformer block skeleton over chunked `.rllm` weights.
///
/// Computes the Phase 5 toy GPT-NeoX/Pythia-shaped block:
/// `LN -> streaming attention -> residual -> LN -> streaming MLP -> residual`.
/// Rotary embeddings, KV-cache reuse, and tokenizer/generation wiring are
/// intentionally out of scope for this primitive.
pub fn streaming_transformer_block_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: StreamingBlockTensorNames<'_>,
    params: StreamingBlockParameters<'_>,
    config: StreamingBlockConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_transformer_block_with_runtime_from_model(
        model,
        input,
        names,
        params,
        config,
        StreamingBlockRuntime::default(),
        budget,
    )
}

pub fn streaming_transformer_block_with_runtime_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: StreamingBlockTensorNames<'_>,
    params: StreamingBlockParameters<'_>,
    config: StreamingBlockConfig,
    runtime: StreamingBlockRuntime<'_>,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_transformer_block_with_runtime_and_timing_from_model(
        model, input, names, params, config, runtime, budget, None,
    )
}

pub fn streaming_transformer_block_with_runtime_and_timing_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: StreamingBlockTensorNames<'_>,
    params: StreamingBlockParameters<'_>,
    config: StreamingBlockConfig,
    runtime: StreamingBlockRuntime<'_>,
    budget: &mut MemoryBudget,
    mut timing: Option<&mut StreamingBlockTiming>,
) -> Result<Vec<f32>> {
    validate_block_shapes(input, params, config)?;
    let hidden_size = config.hidden_size()?;
    let hidden_bytes = activation_bytes(
        config.seq_len,
        hidden_size,
        "streaming block hidden activation",
    )?;

    let mut residual = input.to_vec();

    let attention_input_label = "streaming block input layernorm activation".to_string();
    budget.reserve(hidden_bytes, attention_input_label.clone())?;
    let attention_norm_started = Instant::now();
    let attention_input = match crate::ops::layer_norm(
        input,
        params.input_layernorm_weight,
        params.input_layernorm_bias,
        config.seq_len,
        hidden_size,
        config.layer_norm_eps,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, attention_input_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_norm(attention_norm_started.elapsed());
    }

    let attention_output_label = "streaming block attention output activation".to_string();
    if let Err(err) = budget.reserve(hidden_bytes, attention_output_label.clone()) {
        budget.release(hidden_bytes, attention_input_label)?;
        return Err(err);
    }
    let attention_started = Instant::now();
    let attention_output = match streaming_attention_with_runtime_and_timing_from_model(
        model,
        &attention_input,
        names.qkv_weight,
        params.qkv_bias,
        names.attention_out_weight,
        params.attention_out_bias,
        config.attention_config(),
        runtime.attention,
        budget,
        timing.as_deref_mut(),
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, attention_output_label)?;
            budget.release(hidden_bytes, attention_input_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention(attention_started.elapsed());
    }
    let attention_residual_started = Instant::now();
    drop(attention_input);
    budget.release(hidden_bytes, attention_input_label)?;
    if let Err(err) = crate::ops::add_inplace(&mut residual, &attention_output) {
        budget.release(hidden_bytes, attention_output_label)?;
        return Err(err);
    }
    drop(attention_output);
    budget.release(hidden_bytes, attention_output_label)?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_residual(attention_residual_started.elapsed());
    }

    let mlp_input_label = "streaming block post-attention layernorm activation".to_string();
    budget.reserve(hidden_bytes, mlp_input_label.clone())?;
    let mlp_input_source = if runtime.parallel_residual {
        input
    } else {
        residual.as_slice()
    };
    let mlp_norm_started = Instant::now();
    let mlp_input = match crate::ops::layer_norm(
        mlp_input_source,
        params.post_attention_layernorm_weight,
        params.post_attention_layernorm_bias,
        config.seq_len,
        hidden_size,
        config.layer_norm_eps,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, mlp_input_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp_norm(mlp_norm_started.elapsed());
    }

    let mlp_output_label = "streaming block MLP output activation".to_string();
    if let Err(err) = budget.reserve(hidden_bytes, mlp_output_label.clone()) {
        budget.release(hidden_bytes, mlp_input_label)?;
        return Err(err);
    }
    let mlp_started = Instant::now();
    let mlp_output = match streaming_mlp_with_timing_from_model(
        model,
        &mlp_input,
        names.mlp_in_weight,
        params.mlp_in_bias,
        names.mlp_out_weight,
        params.mlp_out_bias,
        config.mlp_config()?,
        budget,
        timing.as_deref_mut(),
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, mlp_output_label)?;
            budget.release(hidden_bytes, mlp_input_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp(mlp_started.elapsed());
    }
    let mlp_residual_started = Instant::now();
    drop(mlp_input);
    budget.release(hidden_bytes, mlp_input_label)?;
    if let Err(err) = crate::ops::add_inplace(&mut residual, &mlp_output) {
        budget.release(hidden_bytes, mlp_output_label)?;
        return Err(err);
    }
    drop(mlp_output);
    budget.release(hidden_bytes, mlp_output_label)?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp_residual(mlp_residual_started.elapsed());
    }

    Ok(residual)
}

fn validate_block_shapes(
    input: &[f32],
    params: StreamingBlockParameters<'_>,
    config: StreamingBlockConfig,
) -> Result<()> {
    let hidden_size = config.hidden_size()?;
    let input_len = config
        .seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| RuntimeError::Shape("block input length overflow".to_string()))?;
    if input.len() != input_len {
        return Err(RuntimeError::Shape(format!(
            "block input len {} does not match seq_len*hidden_size = {}",
            input.len(),
            input_len
        )));
    }
    validate_norm_params(
        "input layernorm",
        params.input_layernorm_weight,
        params.input_layernorm_bias,
        hidden_size,
    )?;
    validate_norm_params(
        "post-attention layernorm",
        params.post_attention_layernorm_weight,
        params.post_attention_layernorm_bias,
        hidden_size,
    )?;

    let qkv_features = hidden_size
        .checked_mul(3)
        .ok_or_else(|| RuntimeError::Shape("block QKV feature overflow".to_string()))?;
    if let Some(bias) = params.qkv_bias {
        if bias.len() != qkv_features {
            return Err(RuntimeError::Shape(format!(
                "block QKV bias len {} does not match 3*hidden_size {}",
                bias.len(),
                qkv_features
            )));
        }
    }
    if let Some(bias) = params.attention_out_bias {
        if bias.len() != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "block attention output bias len {} does not match hidden_size {}",
                bias.len(),
                hidden_size
            )));
        }
    }
    if let Some(bias) = params.mlp_in_bias {
        if bias.len() != config.intermediate_size {
            return Err(RuntimeError::Shape(format!(
                "block MLP input bias len {} does not match intermediate_size {}",
                bias.len(),
                config.intermediate_size
            )));
        }
    }
    if let Some(bias) = params.mlp_out_bias {
        if bias.len() != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "block MLP output bias len {} does not match hidden_size {}",
                bias.len(),
                hidden_size
            )));
        }
    }
    if !config.layer_norm_eps.is_finite() || config.layer_norm_eps < 0.0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "layer_norm_eps must be finite and non-negative, got {}",
            config.layer_norm_eps
        )));
    }
    Ok(())
}

fn validate_norm_params(
    name: &str,
    weight: &[f32],
    bias: &[f32],
    hidden_size: usize,
) -> Result<()> {
    if weight.len() != hidden_size || bias.len() != hidden_size {
        return Err(RuntimeError::Shape(format!(
            "{name} params must match hidden_size {hidden_size}: weight={}, bias={}",
            weight.len(),
            bias.len()
        )));
    }
    Ok(())
}

fn validate_attention_shapes(
    input: &[f32],
    qkv_bias: Option<&[f32]>,
    out_bias: Option<&[f32]>,
    config: StreamingAttentionConfig,
) -> Result<()> {
    let hidden_size = config.hidden_size()?;
    let input_len = config
        .seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| RuntimeError::Shape("attention input length overflow".to_string()))?;
    if input.len() != input_len {
        return Err(RuntimeError::Shape(format!(
            "attention input len {} does not match seq_len*hidden_size = {}",
            input.len(),
            input_len
        )));
    }
    let qkv_features = hidden_size
        .checked_mul(3)
        .ok_or_else(|| RuntimeError::Shape("QKV feature count overflow".to_string()))?;
    if let Some(bias) = qkv_bias {
        if bias.len() != qkv_features {
            return Err(RuntimeError::Shape(format!(
                "QKV bias len {} does not match 3*hidden_size {}",
                bias.len(),
                qkv_features
            )));
        }
    }
    if let Some(bias) = out_bias {
        if bias.len() != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "attention output bias len {} does not match hidden_size {}",
                bias.len(),
                hidden_size
            )));
        }
    }
    Ok(())
}

fn split_fused_qkv(
    fused: &[f32],
    config: StreamingAttentionConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let hidden_size = config.hidden_size()?;
    let expected = config
        .seq_len
        .checked_mul(hidden_size)
        .and_then(|values| values.checked_mul(3))
        .ok_or_else(|| RuntimeError::Shape("fused QKV length overflow".to_string()))?;
    if fused.len() != expected {
        return Err(RuntimeError::Shape(format!(
            "fused QKV len {} does not match seq_len*3*hidden_size = {}",
            fused.len(),
            expected
        )));
    }

    let values_per_stream = config
        .seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| RuntimeError::Shape("QKV split length overflow".to_string()))?;
    let mut q = vec![0.0f32; values_per_stream];
    let mut k = vec![0.0f32; values_per_stream];
    let mut v = vec![0.0f32; values_per_stream];
    for pos in 0..config.seq_len {
        let fused_row = pos * hidden_size * 3;
        let out_row = pos * hidden_size;
        for head in 0..config.num_heads {
            let fused_head = fused_row + head * config.head_dim * 3;
            let out_head = out_row + head * config.head_dim;
            q[out_head..out_head + config.head_dim]
                .copy_from_slice(&fused[fused_head..fused_head + config.head_dim]);
            k[out_head..out_head + config.head_dim].copy_from_slice(
                &fused[fused_head + config.head_dim..fused_head + 2 * config.head_dim],
            );
            v[out_head..out_head + config.head_dim].copy_from_slice(
                &fused[fused_head + 2 * config.head_dim..fused_head + 3 * config.head_dim],
            );
        }
    }
    Ok((q, k, v))
}

fn validate_mlp_shapes(
    input: &[f32],
    b_in: Option<&[f32]>,
    b_out: Option<&[f32]>,
    config: StreamingMlpConfig,
) -> Result<()> {
    if input.len() != config.batch * config.hidden_size {
        return Err(RuntimeError::Shape(format!(
            "MLP input len {} does not match batch*hidden_size = {}",
            input.len(),
            config.batch * config.hidden_size
        )));
    }
    if let Some(bias) = b_in {
        if bias.len() != config.intermediate_size {
            return Err(RuntimeError::Shape(format!(
                "MLP input bias len {} does not match intermediate_size {}",
                bias.len(),
                config.intermediate_size
            )));
        }
    }
    if let Some(bias) = b_out {
        if bias.len() != config.hidden_size {
            return Err(RuntimeError::Shape(format!(
                "MLP output bias len {} does not match hidden_size {}",
                bias.len(),
                config.hidden_size
            )));
        }
    }
    Ok(())
}

fn activation_bytes(batch: usize, features: usize, label: &str) -> Result<usize> {
    batch
        .checked_mul(features)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| RuntimeError::Shape(format!("{label} byte size overflow")))
}

fn validate_linear_shapes(
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingLinearConfig,
) -> Result<()> {
    if input.len() != config.batch * config.in_features {
        return Err(RuntimeError::Shape(format!(
            "input len {} does not match batch*in_features = {}",
            input.len(),
            config.batch * config.in_features
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != config.out_features {
            return Err(RuntimeError::Shape(format!(
                "bias len {} does not match out_features {}",
                bias.len(),
                config.out_features
            )));
        }
    }
    Ok(())
}

fn validate_tile_linear_config(config: StreamingTileLinearConfig) -> Result<()> {
    if config.tile_elements == 0 {
        return Err(RuntimeError::Shape(
            "tile_elements must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn validate_weight_tensor(tensor: &TensorMeta, config: StreamingLinearConfig) -> Result<()> {
    if tensor.shape.len() != 2 {
        return Err(RuntimeError::Shape(format!(
            "weight tensor {} must be rank-2 [out,in], got {:?}",
            tensor.name, tensor.shape
        )));
    }
    let out = usize::try_from(tensor.shape[0])
        .map_err(|_| RuntimeError::Shape("weight out_features overflows usize".to_string()))?;
    let input = usize::try_from(tensor.shape[1])
        .map_err(|_| RuntimeError::Shape("weight in_features overflows usize".to_string()))?;
    if out != config.out_features || input != config.in_features {
        return Err(RuntimeError::Shape(format!(
            "weight tensor {} shape {:?} does not match requested [{}, {}]",
            tensor.name, tensor.shape, config.out_features, config.in_features
        )));
    }

    let expected_bytes = config
        .out_features
        .checked_mul(config.in_features)
        .and_then(|elements| elements.checked_mul(tensor.dtype.size_bytes()))
        .ok_or_else(|| RuntimeError::Shape("weight byte size overflow".to_string()))?;
    if tensor.original_size_bytes != expected_bytes as u64 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {} original_size_bytes={} does not match shape/dtype bytes {}",
            tensor.name, tensor.original_size_bytes, expected_bytes
        )));
    }
    Ok(())
}

struct StreamingLinearArgmaxState<'a> {
    bias: Option<&'a [f32]>,
    current_out_feature: usize,
    current_acc: f32,
    best_index: usize,
    best_value: f32,
    seen: bool,
}

impl<'a> StreamingLinearArgmaxState<'a> {
    fn new(bias: Option<&'a [f32]>) -> Self {
        Self {
            bias,
            current_out_feature: 0,
            current_acc: bias
                .and_then(|values| values.first())
                .copied()
                .unwrap_or(0.0),
            best_index: 0,
            best_value: f32::NEG_INFINITY,
            seen: false,
        }
    }

    fn finish_current(&mut self, config: StreamingLinearConfig, weight_name: &str) -> Result<()> {
        if self.current_out_feature >= config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed more rows than expected {}",
                config.out_features
            )));
        }
        if !self.seen || self.current_acc > self.best_value {
            self.best_index = self.current_out_feature;
            self.best_value = self.current_acc;
            self.seen = true;
        }
        self.current_out_feature += 1;
        if self.current_out_feature < config.out_features {
            self.current_acc = self
                .bias
                .and_then(|values| values.get(self.current_out_feature))
                .copied()
                .unwrap_or(0.0);
        }
        Ok(())
    }

    fn finish(self, config: StreamingLinearConfig, weight_name: &str) -> Result<usize> {
        if self.current_out_feature != config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed {} complete rows, expected {}",
                self.current_out_feature, config.out_features
            )));
        }
        if !self.seen {
            return Err(RuntimeError::InvalidTensorData(
                "cannot argmax empty streaming linear output".to_string(),
            ));
        }
        Ok(self.best_index)
    }
}

fn accumulate_weight_chunk_argmax(
    input: &[f32],
    weights: &[f32],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearArgmaxState<'_>,
    weight_name: &str,
) -> Result<()> {
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    let element_end = element_start
        .checked_add(weights.len())
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;
    while local_idx < weights.len() {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let row_len = (config.in_features - in_feature).min(weights.len() - local_idx);
        let weight_row = &weights[local_idx..local_idx + row_len];
        let input_row = &input[in_feature..in_feature + row_len];
        let mut idx = 0;
        while idx + 4 <= row_len {
            let w = &weight_row[idx..idx + 4];
            let i_row = &input_row[idx..idx + 4];
            state.current_acc += w[0] * i_row[0] + w[1] * i_row[1] + w[2] * i_row[2] + w[3] * i_row[3];
            idx += 4;
        }
        while idx < row_len {
            state.current_acc += input_row[idx] * weight_row[idx];
            idx += 1;
        }

        local_idx += row_len;
        global_idx += row_len;
        if global_idx % config.in_features == 0 {
            state.finish_current(config, weight_name)?;
        }
    }
    Ok(())
}

fn accumulate_weight_chunk(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;

    let element_end = element_start
        .checked_add(weights.len())
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;
    while local_idx < weights.len() {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weights.len() - local_idx);
        let weight_row = &weights[local_idx..local_idx + row_len];

        let mut batch_idx = 0usize;
        while batch_idx + 8 <= config.batch {
            let output_idx0 = batch_idx * config.out_features + out_feature;
            let output_idx1 = (batch_idx + 1) * config.out_features + out_feature;
            let output_idx2 = (batch_idx + 2) * config.out_features + out_feature;
            let output_idx3 = (batch_idx + 3) * config.out_features + out_feature;
            let output_idx4 = (batch_idx + 4) * config.out_features + out_feature;
            let output_idx5 = (batch_idx + 5) * config.out_features + out_feature;
            let output_idx6 = (batch_idx + 6) * config.out_features + out_feature;
            let output_idx7 = (batch_idx + 7) * config.out_features + out_feature;
            let mut acc0 = output[output_idx0];
            let mut acc1 = output[output_idx1];
            let mut acc2 = output[output_idx2];
            let mut acc3 = output[output_idx3];
            let mut acc4 = output[output_idx4];
            let mut acc5 = output[output_idx5];
            let mut acc6 = output[output_idx6];
            let mut acc7 = output[output_idx7];
            let input_start0 = batch_idx * config.in_features + in_feature;
            let input_start1 = (batch_idx + 1) * config.in_features + in_feature;
            let input_start2 = (batch_idx + 2) * config.in_features + in_feature;
            let input_start3 = (batch_idx + 3) * config.in_features + in_feature;
            let input_start4 = (batch_idx + 4) * config.in_features + in_feature;
            let input_start5 = (batch_idx + 5) * config.in_features + in_feature;
            let input_start6 = (batch_idx + 6) * config.in_features + in_feature;
            let input_start7 = (batch_idx + 7) * config.in_features + in_feature;
            let mut idx = 0;
            while idx + 4 <= row_len {
                let w = &weight_row[idx..idx + 4];
                let i0 = &input[input_start0 + idx..input_start0 + idx + 4];
                let i1 = &input[input_start1 + idx..input_start1 + idx + 4];
                let i2 = &input[input_start2 + idx..input_start2 + idx + 4];
                let i3 = &input[input_start3 + idx..input_start3 + idx + 4];
                let i4 = &input[input_start4 + idx..input_start4 + idx + 4];
                let i5 = &input[input_start5 + idx..input_start5 + idx + 4];
                let i6 = &input[input_start6 + idx..input_start6 + idx + 4];
                let i7 = &input[input_start7 + idx..input_start7 + idx + 4];

                acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                acc4 += w[0] * i4[0] + w[1] * i4[1] + w[2] * i4[2] + w[3] * i4[3];
                acc5 += w[0] * i5[0] + w[1] * i5[1] + w[2] * i5[2] + w[3] * i5[3];
                acc6 += w[0] * i6[0] + w[1] * i6[1] + w[2] * i6[2] + w[3] * i6[3];
                acc7 += w[0] * i7[0] + w[1] * i7[1] + w[2] * i7[2] + w[3] * i7[3];

                idx += 4;
            }
            while idx < row_len {
                let weight = weight_row[idx];
                acc0 += input[input_start0 + idx] * weight;
                acc1 += input[input_start1 + idx] * weight;
                acc2 += input[input_start2 + idx] * weight;
                acc3 += input[input_start3 + idx] * weight;
                acc4 += input[input_start4 + idx] * weight;
                acc5 += input[input_start5 + idx] * weight;
                acc6 += input[input_start6 + idx] * weight;
                acc7 += input[input_start7 + idx] * weight;
                idx += 1;
            }
            output[output_idx0] = acc0;
            output[output_idx1] = acc1;
            output[output_idx2] = acc2;
            output[output_idx3] = acc3;
            output[output_idx4] = acc4;
            output[output_idx5] = acc5;
            output[output_idx6] = acc6;
            output[output_idx7] = acc7;
            batch_idx += 8;
        }
        while batch_idx + 4 <= config.batch {
            let output_idx0 = batch_idx * config.out_features + out_feature;
            let output_idx1 = (batch_idx + 1) * config.out_features + out_feature;
            let output_idx2 = (batch_idx + 2) * config.out_features + out_feature;
            let output_idx3 = (batch_idx + 3) * config.out_features + out_feature;
            let mut acc0 = output[output_idx0];
            let mut acc1 = output[output_idx1];
            let mut acc2 = output[output_idx2];
            let mut acc3 = output[output_idx3];
            let input_start0 = batch_idx * config.in_features + in_feature;
            let input_start1 = (batch_idx + 1) * config.in_features + in_feature;
            let input_start2 = (batch_idx + 2) * config.in_features + in_feature;
            let input_start3 = (batch_idx + 3) * config.in_features + in_feature;
            let mut idx = 0;
            while idx + 4 <= row_len {
                let w = &weight_row[idx..idx + 4];
                let i0 = &input[input_start0 + idx..input_start0 + idx + 4];
                let i1 = &input[input_start1 + idx..input_start1 + idx + 4];
                let i2 = &input[input_start2 + idx..input_start2 + idx + 4];
                let i3 = &input[input_start3 + idx..input_start3 + idx + 4];

                acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                idx += 4;
            }
            while idx < row_len {
                let weight = weight_row[idx];
                acc0 += input[input_start0 + idx] * weight;
                acc1 += input[input_start1 + idx] * weight;
                acc2 += input[input_start2 + idx] * weight;
                acc3 += input[input_start3 + idx] * weight;
                idx += 1;
            }
            output[output_idx0] = acc0;
            output[output_idx1] = acc1;
            output[output_idx2] = acc2;
            output[output_idx3] = acc3;
            batch_idx += 4;
        }
        while batch_idx < config.batch {
            let input_start = batch_idx * config.in_features + in_feature;
            let input_row = &input[input_start..input_start + row_len];
            let output_idx = batch_idx * config.out_features + out_feature;
            let mut acc = output[output_idx];
            let mut idx = 0;
            while idx + 4 <= row_len {
                let w = &weight_row[idx..idx + 4];
                let i0 = &input_row[idx..idx + 4];
                acc += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                idx += 4;
            }
            while idx < row_len {
                acc += input_row[idx] * weight_row[idx];
                idx += 1;
            }
            output[output_idx] = acc;
            batch_idx += 1;
        }

        local_idx += row_len;
        global_idx += row_len;
    }
    Ok(())
}

fn accumulate_fused_rle_chunk_u8(
    input: &[f32],
    output: &mut [f32],
    rle_stream: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    if rle_stream.len() % 2 != 0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "RLE stream for {weight_name} has odd length"
        )));
    }

    let mut current_element = element_start;
    for chunk in rle_stream.chunks_exact(2) {
        let count = chunk[0] as usize;
        let value = chunk[1] as f32;

        let mut i = 0;
        while i < count {
            let out_feature = current_element / config.in_features;
            let in_feature = current_element % config.in_features;
            let run_in_this_row = (config.in_features - in_feature).min(count - i);
            
            let mut batch_idx = 0;
            while batch_idx < config.batch {
                let output_idx = batch_idx * config.out_features + out_feature;
                let input_start = batch_idx * config.in_features + in_feature;
                
                let mut sum = 0.0;
                for j in 0..run_in_this_row {
                    sum += input[input_start + j];
                }
                output[output_idx] += value * sum;
                
                batch_idx += 1;
            }
            
            current_element += run_in_this_row;
            i += run_in_this_row;
        }
    }
    
    Ok(())
}

fn accumulate_fused_raw_fp16_chunk(
    input: &[f32],
    output: &mut [f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    if raw_bytes.len() % 2 != 0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw FP16 stream for {weight_name} has odd length"
        )));
    }

    let weight_elements = raw_bytes.len() / 2;
    let mut local_idx = 0usize;
    let mut global_idx = element_start;

    const BLOCK_SIZE: usize = 128;
    let mut w_block = [0.0f32; BLOCK_SIZE];

    while local_idx < weight_elements {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);

        let mut row_idx = 0;
        while row_idx < row_len {
            let block_len = (row_len - row_idx).min(BLOCK_SIZE);
            let byte_start = (local_idx + row_idx) * 2;
            let block_bytes = &raw_bytes[byte_start..byte_start + block_len * 2];

            // Decode this block ONCE into the stack array
            let mut i = 0;
            while i + 4 <= block_len {
                let b = &block_bytes[i * 2..i * 2 + 8];
                w_block[i] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                w_block[i + 1] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[2], b[3]]));
                w_block[i + 2] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[4], b[5]]));
                w_block[i + 3] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[6], b[7]]));
                i += 4;
            }
            while i < block_len {
                let b = &block_bytes[i * 2..i * 2 + 2];
                w_block[i] = crate::tensor::fp16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                i += 1;
            }
            let w_slice = &w_block[..block_len];

            // Same 8-wide batch unrolling as accumulate_weight_chunk
            let mut batch_idx = 0usize;
            while batch_idx + 8 <= config.batch {
                let out_idx0 = batch_idx * config.out_features + out_feature;
                let out_idx1 = (batch_idx + 1) * config.out_features + out_feature;
                let out_idx2 = (batch_idx + 2) * config.out_features + out_feature;
                let out_idx3 = (batch_idx + 3) * config.out_features + out_feature;
                let out_idx4 = (batch_idx + 4) * config.out_features + out_feature;
                let out_idx5 = (batch_idx + 5) * config.out_features + out_feature;
                let out_idx6 = (batch_idx + 6) * config.out_features + out_feature;
                let out_idx7 = (batch_idx + 7) * config.out_features + out_feature;

                let mut acc0 = output[out_idx0];
                let mut acc1 = output[out_idx1];
                let mut acc2 = output[out_idx2];
                let mut acc3 = output[out_idx3];
                let mut acc4 = output[out_idx4];
                let mut acc5 = output[out_idx5];
                let mut acc6 = output[out_idx6];
                let mut acc7 = output[out_idx7];

                let in_start0 = batch_idx * config.in_features + in_feature + row_idx;
                let in_start1 = (batch_idx + 1) * config.in_features + in_feature + row_idx;
                let in_start2 = (batch_idx + 2) * config.in_features + in_feature + row_idx;
                let in_start3 = (batch_idx + 3) * config.in_features + in_feature + row_idx;
                let in_start4 = (batch_idx + 4) * config.in_features + in_feature + row_idx;
                let in_start5 = (batch_idx + 5) * config.in_features + in_feature + row_idx;
                let in_start6 = (batch_idx + 6) * config.in_features + in_feature + row_idx;
                let in_start7 = (batch_idx + 7) * config.in_features + in_feature + row_idx;

                let mut idx = 0;
                while idx + 4 <= block_len {
                    let w = &w_slice[idx..idx + 4];
                    let i0 = &input[in_start0 + idx..in_start0 + idx + 4];
                    let i1 = &input[in_start1 + idx..in_start1 + idx + 4];
                    let i2 = &input[in_start2 + idx..in_start2 + idx + 4];
                    let i3 = &input[in_start3 + idx..in_start3 + idx + 4];
                    let i4 = &input[in_start4 + idx..in_start4 + idx + 4];
                    let i5 = &input[in_start5 + idx..in_start5 + idx + 4];
                    let i6 = &input[in_start6 + idx..in_start6 + idx + 4];
                    let i7 = &input[in_start7 + idx..in_start7 + idx + 4];

                    acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                    acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                    acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                    acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                    acc4 += w[0] * i4[0] + w[1] * i4[1] + w[2] * i4[2] + w[3] * i4[3];
                    acc5 += w[0] * i5[0] + w[1] * i5[1] + w[2] * i5[2] + w[3] * i5[3];
                    acc6 += w[0] * i6[0] + w[1] * i6[1] + w[2] * i6[2] + w[3] * i6[3];
                    acc7 += w[0] * i7[0] + w[1] * i7[1] + w[2] * i7[2] + w[3] * i7[3];
                    idx += 4;
                }
                while idx < block_len {
                    let weight = w_slice[idx];
                    acc0 += input[in_start0 + idx] * weight;
                    acc1 += input[in_start1 + idx] * weight;
                    acc2 += input[in_start2 + idx] * weight;
                    acc3 += input[in_start3 + idx] * weight;
                    acc4 += input[in_start4 + idx] * weight;
                    acc5 += input[in_start5 + idx] * weight;
                    acc6 += input[in_start6 + idx] * weight;
                    acc7 += input[in_start7 + idx] * weight;
                    idx += 1;
                }

                output[out_idx0] = acc0;
                output[out_idx1] = acc1;
                output[out_idx2] = acc2;
                output[out_idx3] = acc3;
                output[out_idx4] = acc4;
                output[out_idx5] = acc5;
                output[out_idx6] = acc6;
                output[out_idx7] = acc7;
                batch_idx += 8;
            }

            while batch_idx + 4 <= config.batch {
                let out_idx0 = batch_idx * config.out_features + out_feature;
                let out_idx1 = (batch_idx + 1) * config.out_features + out_feature;
                let out_idx2 = (batch_idx + 2) * config.out_features + out_feature;
                let out_idx3 = (batch_idx + 3) * config.out_features + out_feature;

                let mut acc0 = output[out_idx0];
                let mut acc1 = output[out_idx1];
                let mut acc2 = output[out_idx2];
                let mut acc3 = output[out_idx3];

                let in_start0 = batch_idx * config.in_features + in_feature + row_idx;
                let in_start1 = (batch_idx + 1) * config.in_features + in_feature + row_idx;
                let in_start2 = (batch_idx + 2) * config.in_features + in_feature + row_idx;
                let in_start3 = (batch_idx + 3) * config.in_features + in_feature + row_idx;

                let mut idx = 0;
                while idx + 4 <= block_len {
                    let w = &w_slice[idx..idx + 4];
                    let i0 = &input[in_start0 + idx..in_start0 + idx + 4];
                    let i1 = &input[in_start1 + idx..in_start1 + idx + 4];
                    let i2 = &input[in_start2 + idx..in_start2 + idx + 4];
                    let i3 = &input[in_start3 + idx..in_start3 + idx + 4];

                    acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                    acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                    acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                    acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                    idx += 4;
                }
                while idx < block_len {
                    let weight = w_slice[idx];
                    acc0 += input[in_start0 + idx] * weight;
                    acc1 += input[in_start1 + idx] * weight;
                    acc2 += input[in_start2 + idx] * weight;
                    acc3 += input[in_start3 + idx] * weight;
                    idx += 1;
                }

                output[out_idx0] = acc0;
                output[out_idx1] = acc1;
                output[out_idx2] = acc2;
                output[out_idx3] = acc3;
                batch_idx += 4;
            }

            while batch_idx < config.batch {
                let out_idx = batch_idx * config.out_features + out_feature;
                let mut acc = output[out_idx];
                let in_start = batch_idx * config.in_features + in_feature + row_idx;

                let mut idx = 0;
                while idx + 4 <= block_len {
                    let w = &w_slice[idx..idx + 4];
                    let i0 = &input[in_start + idx..in_start + idx + 4];
                    acc += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                    idx += 4;
                }
                while idx < block_len {
                    acc += w_slice[idx] * input[in_start + idx];
                    idx += 1;
                }
                output[out_idx] = acc;
                batch_idx += 1;
            }

            row_idx += block_len;
        }

        local_idx += row_len;
        global_idx += row_len;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linear, sample_argmax};
    use rllm_container::{DType, GlobalMetadata, RllmWriter, TensorMeta};
    use rtc_codec::{EncodeMeta, RleCodec, TensorCodec};
    use sha2::{Digest, Sha256};

    fn sha256_array(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rllm-streaming-{name}-{}.rllm", std::process::id()))
    }

    fn write_chunked_weight(path: &std::path::Path) {
        let weight = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // [out=2, in=3]
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "linear.weight".to_string(),
            shape: vec![2, 3],
            dtype: DType::Fp32,
            original_size_bytes: weight.len() as u64,
            compressed_size_bytes: weight.len() as u64,
            original_sha256: sha256_array(&weight),
            chunk_count: 2,
            chunk_start_index: 0,
        });

        // Split in the middle of row 1. Streaming must reconstruct global element
        // positions from cumulative decoded size, not from chunk_offset_in_tensor.
        writer
            .write_chunk(0, "rtc-raw-v1", &weight[..16], &weight[..16], 0)
            .unwrap();
        writer
            .write_chunk(0, "rtc-raw-v1", &weight[16..], &weight[16..], 1)
            .unwrap();
        writer.finalize().unwrap();
    }

    fn add_rle_zero_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        out_features: usize,
        in_features: usize,
    ) {
        let values = vec![0.0f32; out_features * in_features];
        let bytes = f32_bytes(&values);
        let encoded = RleCodec
            .encode(
                &bytes,
                &EncodeMeta {
                    name: name.to_string(),
                    shape: vec![out_features as u64, in_features as u64],
                    dtype: "F32".to_string(),
                },
            )
            .unwrap();
        assert!(encoded.data.len() < bytes.len() / 8);

        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape: vec![out_features as u64, in_features as u64],
            dtype: DType::Fp32,
            original_size_bytes: bytes.len() as u64,
            compressed_size_bytes: encoded.data.len() as u64,
            original_sha256: sha256_array(&bytes),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(tensor_id, "rtc-rle-v1", &encoded.data, &bytes, 0)
            .unwrap();
    }

    fn write_rle_zero_weight(path: &std::path::Path, out_features: usize, in_features: usize) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_rle_zero_tensor(
            &mut writer,
            0,
            "linear.zero.weight",
            out_features,
            in_features,
        );
        writer.finalize().unwrap();
    }

    #[test]
    fn streaming_linear_matches_full_decode_linear_across_chunk_boundary() {
        let path = temp_path("linear");
        write_chunked_weight(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![10.0, 20.0, 30.0, -1.0, 2.0, -3.0]; // [batch=2, in=3]
        let bias = vec![1.0, -1.0];
        let expected = linear(
            &input,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(&bias),
            2,
            3,
            2,
        )
        .unwrap();
        let mut budget = MemoryBudget::new(256);

        let actual = streaming_linear_from_model(
            &mut model,
            "linear.weight",
            &input,
            Some(&bias),
            StreamingLinearConfig {
                batch: 2,
                in_features: 3,
                out_features: 2,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 64,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_linear_matches_full_decode_with_eight_and_four_batch_fast_paths_and_tail() {
        let path = temp_path("linear-batch-fast-path-tail");
        write_chunked_weight(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input: Vec<f32> = (0..39).map(|idx| idx as f32 * 0.25 - 1.0).collect(); // [batch=13, in=3]
        let bias = vec![1.0, -1.0];
        let expected = linear(
            &input,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(&bias),
            13,
            3,
            2,
        )
        .unwrap();
        let mut budget = MemoryBudget::new(512);

        let actual = streaming_linear_from_model(
            &mut model,
            "linear.weight",
            &input,
            Some(&bias),
            StreamingLinearConfig {
                batch: 13,
                in_features: 3,
                out_features: 2,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_linear_rejects_too_small_transient_budget_without_leaking() {
        let path = temp_path("budget");
        write_chunked_weight(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0, 2.0, 3.0];
        let mut budget = MemoryBudget::new(31);

        let err = streaming_linear_from_model(
            &mut model,
            "linear.weight",
            &input,
            None,
            StreamingLinearConfig {
                batch: 1,
                in_features: 3,
                out_features: 2,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_matches_full_decode_with_smaller_scratch_budget() {
        let path = temp_path("tile-linear");
        write_rle_zero_weight(&path, 32, 32);
        let mut standard_model = LazyRllmModel::open(&path).unwrap();
        let mut tile_model = LazyRllmModel::open(&path).unwrap();
        let input: Vec<f32> = (0..64).map(|idx| (idx as f32) * 0.01 - 0.25).collect();
        let bias: Vec<f32> = (0..32).map(|idx| idx as f32 * 0.125).collect();
        let config = StreamingLinearConfig {
            batch: 2,
            in_features: 32,
            out_features: 32,
        };

        let mut standard_budget = MemoryBudget::new(5_000);
        let standard_err = streaming_linear_from_model(
            &mut standard_model,
            "linear.zero.weight",
            &input,
            Some(&bias),
            config,
            &mut standard_budget,
        )
        .unwrap_err();
        assert!(matches!(
            standard_err,
            RuntimeError::MemoryBudgetExceeded { .. }
        ));
        assert_eq!(standard_budget.current_bytes(), 0);

        let mut tile_budget = MemoryBudget::new(5_000);
        let actual = streaming_tile_linear_from_model(
            &mut tile_model,
            "linear.zero.weight",
            &input,
            Some(&bias),
            StreamingTileLinearConfig {
                linear: config,
                tile_elements: 16,
            },
            &mut tile_budget,
        )
        .unwrap();

        let expected = linear(&input, &vec![0.0; 32 * 32], Some(&bias), 2, 32, 32).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(tile_budget.current_bytes(), 0);
        assert!(
            tile_budget.peak_bytes() < 5_000,
            "peak was {}",
            tile_budget.peak_bytes()
        );
        assert!(standard_budget.peak_bytes() < 5_000);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_rejects_too_small_tile_scratch_without_leaking() {
        let path = temp_path("tile-linear-budget");
        write_rle_zero_weight(&path, 32, 32);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 32];
        let mut budget = MemoryBudget::new(4_140);

        let err = streaming_tile_linear_from_model(
            &mut model,
            "linear.zero.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 32,
                    out_features: 32,
                },
                tile_elements: 16,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    fn add_f32_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        shape: Vec<u64>,
        values: &[f32],
        split_at: usize,
    ) {
        let bytes = f32_bytes(values);
        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape,
            dtype: DType::Fp32,
            original_size_bytes: bytes.len() as u64,
            compressed_size_bytes: bytes.len() as u64,
            original_sha256: sha256_array(&bytes),
            chunk_count: 2,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(
                tensor_id,
                "rtc-raw-v1",
                &bytes[..split_at],
                &bytes[..split_at],
                0,
            )
            .unwrap();
        writer
            .write_chunk(
                tensor_id,
                "rtc-raw-v1",
                &bytes[split_at..],
                &bytes[split_at..],
                1,
            )
            .unwrap();
    }

    #[test]
    fn streaming_tile_linear_argmax_matches_full_logits_across_split_rows() {
        let path = temp_path("tile-linear-argmax");
        let weight = vec![
            0.5, -1.0, 2.0, -2.0, 0.25, 0.5, 1.0, 1.0, -1.0, 0.0, -0.5, 0.75,
        ];
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "linear.argmax.weight",
            vec![4, 3],
            &weight,
            20,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let bias = vec![0.0, 0.5, -1.0, 4.0];
        let config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: 3,
                out_features: 4,
            },
            tile_elements: 2,
        };
        let mut logits_model = LazyRllmModel::open(&path).unwrap();
        let mut argmax_model = LazyRllmModel::open(&path).unwrap();
        let mut logits_budget = MemoryBudget::new(256);
        let mut argmax_budget = MemoryBudget::new(256);

        let logits = streaming_tile_linear_from_model(
            &mut logits_model,
            "linear.argmax.weight",
            &input,
            Some(&bias),
            config,
            &mut logits_budget,
        )
        .unwrap();
        let expected = sample_argmax(&logits).unwrap();
        let actual = streaming_tile_linear_argmax_from_model(
            &mut argmax_model,
            "linear.argmax.weight",
            &input,
            Some(&bias),
            config,
            &mut argmax_budget,
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(logits_budget.current_bytes(), 0);
        assert_eq!(argmax_budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    fn write_chunked_mlp(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "mlp.dense_h_to_4h.weight",
            vec![3, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, -1.0],
            12,
        );
        add_f32_tensor(
            &mut writer,
            1,
            "mlp.dense_4h_to_h.weight",
            vec![2, 3],
            &[1.0, 2.0, 3.0, -1.0, 0.5, 0.25],
            16,
        );
        writer.finalize().unwrap();
    }

    fn write_rle_zero_mlp(path: &std::path::Path, hidden_size: usize, intermediate_size: usize) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_rle_zero_tensor(
            &mut writer,
            0,
            "mlp.zero.dense_h_to_4h.weight",
            intermediate_size,
            hidden_size,
        );
        add_rle_zero_tensor(
            &mut writer,
            1,
            "mlp.zero.dense_4h_to_h.weight",
            hidden_size,
            intermediate_size,
        );
        writer.finalize().unwrap();
    }

    fn assert_close_vec(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= eps,
                "idx={idx}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn streaming_mlp_matches_full_decode_mlp_and_releases_intermediate_budget() {
        let path = temp_path("mlp");
        write_chunked_mlp(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 2.0, 0.25]; // [batch=2, hidden=2]
        let b_in = vec![0.1, -0.2, 0.3];
        let b_out = vec![0.05, -0.05];
        let expected = crate::mlp(
            &input,
            &[1.0, 0.0, 0.0, 1.0, 1.0, -1.0],
            Some(&b_in),
            &[1.0, 2.0, 3.0, -1.0, 0.5, 0.25],
            Some(&b_out),
            2,
            2,
            3,
        )
        .unwrap();
        let mut budget = MemoryBudget::new(160);

        let actual = streaming_mlp_from_model(
            &mut model,
            &input,
            "mlp.dense_h_to_4h.weight",
            Some(&b_in),
            "mlp.dense_4h_to_h.weight",
            Some(&b_out),
            StreamingMlpConfig {
                batch: 2,
                hidden_size: 2,
                intermediate_size: 3,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 80,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_mlp_uses_tiled_linear_to_fit_below_full_chunk_scratch_budget() {
        let path = temp_path("mlp-tiled-budget");
        write_rle_zero_mlp(&path, 128, 128);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 128];
        let mut budget = MemoryBudget::new(100_000);

        let actual = streaming_mlp_from_model(
            &mut model,
            &input,
            "mlp.zero.dense_h_to_4h.weight",
            None,
            "mlp.zero.dense_4h_to_h.weight",
            None,
            StreamingMlpConfig {
                batch: 1,
                hidden_size: 128,
                intermediate_size: 128,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, vec![0.0; 128]);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() < 100_000,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_mlp_rejects_too_small_intermediate_budget_without_leaking() {
        let path = temp_path("mlp-budget");
        write_chunked_mlp(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0];
        let mut budget = MemoryBudget::new(11);

        let err = streaming_mlp_from_model(
            &mut model,
            &input,
            "mlp.dense_h_to_4h.weight",
            None,
            "mlp.dense_4h_to_h.weight",
            None,
            StreamingMlpConfig {
                batch: 1,
                hidden_size: 2,
                intermediate_size: 3,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    fn write_chunked_attention(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "attention.query_key_value.weight",
            vec![6, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0],
            20,
        );
        add_f32_tensor(
            &mut writer,
            1,
            "attention.dense.weight",
            vec![2, 2],
            &[1.0, 0.5, -0.25, 1.0],
            8,
        );
        writer.finalize().unwrap();
    }

    fn write_rle_zero_attention(path: &std::path::Path, hidden_size: usize) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_rle_zero_tensor(
            &mut writer,
            0,
            "attention.zero.query_key_value.weight",
            3 * hidden_size,
            hidden_size,
        );
        add_rle_zero_tensor(
            &mut writer,
            1,
            "attention.zero.dense.weight",
            hidden_size,
            hidden_size,
        );
        writer.finalize().unwrap();
    }

    fn identity_qkv_weight(hidden_size: usize) -> Vec<f32> {
        let mut weight = vec![0.0f32; 3 * hidden_size * hidden_size];
        for block in 0..3 {
            for dim in 0..hidden_size {
                weight[(block * hidden_size + dim) * hidden_size + dim] = 1.0;
            }
        }
        weight
    }

    fn identity_weight(hidden_size: usize) -> Vec<f32> {
        let mut weight = vec![0.0f32; hidden_size * hidden_size];
        for dim in 0..hidden_size {
            weight[dim * hidden_size + dim] = 1.0;
        }
        weight
    }

    fn write_chunked_rotary_attention(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        let qkv_weight = identity_qkv_weight(4);
        let out_weight = identity_weight(4);
        add_f32_tensor(
            &mut writer,
            0,
            "attention.rotary_qkv.weight",
            vec![12, 4],
            &qkv_weight,
            96,
        );
        add_f32_tensor(
            &mut writer,
            1,
            "attention.rotary_dense.weight",
            vec![4, 4],
            &out_weight,
            32,
        );
        writer.finalize().unwrap();
    }

    fn split_fused_qkv_for_test(
        fused: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hidden = num_heads * head_dim;
        let mut q = vec![0.0f32; seq_len * hidden];
        let mut k = vec![0.0f32; seq_len * hidden];
        let mut v = vec![0.0f32; seq_len * hidden];
        for pos in 0..seq_len {
            let fused_row = pos * hidden * 3;
            let out_row = pos * hidden;
            for head in 0..num_heads {
                let fused_head = fused_row + head * head_dim * 3;
                let out_head = out_row + head * head_dim;
                q[out_head..out_head + head_dim]
                    .copy_from_slice(&fused[fused_head..fused_head + head_dim]);
                k[out_head..out_head + head_dim]
                    .copy_from_slice(&fused[fused_head + head_dim..fused_head + 2 * head_dim]);
                v[out_head..out_head + head_dim]
                    .copy_from_slice(&fused[fused_head + 2 * head_dim..fused_head + 3 * head_dim]);
            }
        }
        (q, k, v)
    }

    #[test]
    fn split_fused_qkv_uses_gpt_neox_per_head_layout() {
        let fused: Vec<f32> = (1..=24).map(|value| value as f32).collect();
        let (q, k, v) = split_fused_qkv(
            &fused,
            StreamingAttentionConfig {
                seq_len: 2,
                num_heads: 2,
                head_dim: 2,
                causal: true,
            },
        )
        .unwrap();

        assert_eq!(q, vec![1.0, 2.0, 7.0, 8.0, 13.0, 14.0, 19.0, 20.0]);
        assert_eq!(k, vec![3.0, 4.0, 9.0, 10.0, 15.0, 16.0, 21.0, 22.0]);
        assert_eq!(v, vec![5.0, 6.0, 11.0, 12.0, 17.0, 18.0, 23.0, 24.0]);
    }

    #[test]
    fn streaming_attention_matches_full_decode_qkv_attention_and_releases_budget() {
        let path = temp_path("attention");
        write_chunked_attention(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 1.25, 0.75]; // [seq=2, hidden=2]
        let qkv_bias = vec![0.1, -0.2, 0.0, 0.3, -0.1, 0.2];
        let out_bias = vec![0.05, -0.05];
        let qkv_weight = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
        let out_weight = [1.0, 0.5, -0.25, 1.0];
        let fused = linear(&input, &qkv_weight, Some(&qkv_bias), 2, 2, 6).unwrap();
        let (q, k, v) = split_fused_qkv_for_test(&fused, 2, 1, 2);
        let attended = crate::scaled_dot_product_attention(&q, &k, &v, 2, 1, 2, true).unwrap();
        let expected = linear(&attended, &out_weight, Some(&out_bias), 2, 2, 2).unwrap();
        let mut budget = MemoryBudget::new(256);

        let actual = streaming_attention_from_model(
            &mut model,
            &input,
            "attention.query_key_value.weight",
            Some(&qkv_bias),
            "attention.dense.weight",
            Some(&out_bias),
            StreamingAttentionConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 2,
                causal: true,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 128,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_attention_uses_tiled_linear_to_fit_below_full_chunk_scratch_budget() {
        let path = temp_path("attention-tiled-budget");
        write_rle_zero_attention(&path, 80);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 80];
        let mut budget = MemoryBudget::new(112_000);

        let actual = streaming_attention_from_model(
            &mut model,
            &input,
            "attention.zero.query_key_value.weight",
            None,
            "attention.zero.dense.weight",
            None,
            StreamingAttentionConfig {
                seq_len: 1,
                num_heads: 1,
                head_dim: 80,
                causal: true,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, vec![0.0; 80]);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() < 112_000,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_attention_rejects_too_small_qkv_budget_without_leaking() {
        let path = temp_path("attention-budget");
        write_chunked_attention(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0];
        let mut budget = MemoryBudget::new(23);

        let err = streaming_attention_from_model(
            &mut model,
            &input,
            "attention.query_key_value.weight",
            None,
            "attention.dense.weight",
            None,
            StreamingAttentionConfig {
                seq_len: 1,
                num_heads: 1,
                head_dim: 2,
                causal: true,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_attention_with_rotary_and_kv_cache_matches_full_decode_last_token() {
        let path = temp_path("rotary-attention");
        write_chunked_rotary_attention(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let qkv_weight = identity_qkv_weight(4);
        let out_weight = identity_weight(4);
        let previous_input = [0.4, -0.2, 0.1, 0.3];
        let current_input = [0.7, 0.5, -0.4, 0.2];
        let mut full_input = Vec::new();
        full_input.extend_from_slice(&previous_input);
        full_input.extend_from_slice(&current_input);
        let fused = linear(&full_input, &qkv_weight, None, 2, 4, 12).unwrap();
        let (mut q, mut k, v) = split_fused_qkv_for_test(&fused, 2, 1, 4);
        crate::apply_gpt_neox_rotary_inplace(
            &mut q,
            &mut k,
            crate::RotaryEmbeddingConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 4,
                rotary_dim: 4,
                base: 10_000.0,
                position_offset: 0,
            },
        )
        .unwrap();
        let full_attended = crate::scaled_dot_product_attention(&q, &k, &v, 2, 1, 4, true).unwrap();
        let expected = linear(&full_attended[4..8], &out_weight, None, 1, 4, 4).unwrap();

        let mut cache = crate::KvCache::new(1, 4, 4).unwrap();
        cache.append(&k[..4], &v[..4], 1).unwrap();
        let mut budget = MemoryBudget::new(1024);
        let mut timing = StreamingBlockTiming::default();

        let actual = streaming_attention_with_runtime_and_timing_from_model(
            &mut model,
            &current_input,
            "attention.rotary_qkv.weight",
            None,
            "attention.rotary_dense.weight",
            None,
            StreamingAttentionConfig {
                seq_len: 1,
                num_heads: 1,
                head_dim: 4,
                causal: true,
            },
            StreamingAttentionRuntime {
                rotary: Some(crate::RotaryEmbeddingConfig {
                    seq_len: 1,
                    num_heads: 1,
                    head_dim: 4,
                    rotary_dim: 4,
                    base: 10_000.0,
                    position_offset: 1,
                }),
                kv_cache: Some(&mut cache),
            },
            &mut budget,
            Some(&mut timing),
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(cache.len(), 2);
        assert_close_vec(cache.keys(), &k[..8], 1e-6);
        assert_close_vec(cache.values(), &v[..8], 1e-6);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(timing.attention_qkv_projection_calls, 1);
        assert_eq!(timing.attention_qkv_split_calls, 1);
        assert_eq!(timing.attention_rotary_calls, 1);
        assert_eq!(timing.attention_score_context_calls, 1);
        assert_eq!(timing.attention_output_projection_calls, 1);
        assert_eq!(timing.attention_kv_append_calls, 1);
        assert!(timing.attention_qkv_projection_ns > 0);
        assert!(timing.attention_qkv_split_ns > 0);
        assert!(timing.attention_rotary_ns > 0);
        assert!(timing.attention_score_context_ns > 0);
        assert!(timing.attention_output_projection_ns > 0);
        assert!(timing.attention_kv_append_ns > 0);

        std::fs::remove_file(&path).ok();
    }

    fn write_chunked_block(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "gpt_neox.layers.0.attention.query_key_value.weight",
            vec![6, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0],
            20,
        );
        add_f32_tensor(
            &mut writer,
            1,
            "gpt_neox.layers.0.attention.dense.weight",
            vec![2, 2],
            &[1.0, 0.5, -0.25, 1.0],
            8,
        );
        add_f32_tensor(
            &mut writer,
            2,
            "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
            vec![3, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, -1.0],
            12,
        );
        add_f32_tensor(
            &mut writer,
            3,
            "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
            vec![2, 3],
            &[1.0, 2.0, 3.0, -1.0, 0.5, 0.25],
            16,
        );
        writer.finalize().unwrap();
    }

    fn full_decode_block_baseline(input: &[f32], parallel_residual: bool) -> Vec<f32> {
        let ln1_weight = [1.1, 0.9];
        let ln1_bias = [0.05, -0.05];
        let qkv_bias = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2];
        let attention_out_bias = [0.05, -0.05];
        let ln2_weight = [0.8, 1.2];
        let ln2_bias = [-0.02, 0.04];
        let mlp_in_bias = [0.1, -0.2, 0.3];
        let mlp_out_bias = [0.05, -0.05];
        let qkv_weight = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
        let attention_out_weight = [1.0, 0.5, -0.25, 1.0];
        let mlp_in_weight = [1.0, 0.0, 0.0, 1.0, 1.0, -1.0];
        let mlp_out_weight = [1.0, 2.0, 3.0, -1.0, 0.5, 0.25];

        let attention_input = crate::layer_norm(input, &ln1_weight, &ln1_bias, 2, 2, 1e-5).unwrap();
        let fused = linear(&attention_input, &qkv_weight, Some(&qkv_bias), 2, 2, 6).unwrap();
        let (q, k, v) = split_fused_qkv_for_test(&fused, 2, 1, 2);
        let attended = crate::scaled_dot_product_attention(&q, &k, &v, 2, 1, 2, true).unwrap();
        let attention_out = linear(
            &attended,
            &attention_out_weight,
            Some(&attention_out_bias),
            2,
            2,
            2,
        )
        .unwrap();
        let mut residual = input.to_vec();
        crate::add_inplace(&mut residual, &attention_out).unwrap();

        let mlp_input_source = if parallel_residual {
            input
        } else {
            residual.as_slice()
        };
        let mlp_input =
            crate::layer_norm(mlp_input_source, &ln2_weight, &ln2_bias, 2, 2, 1e-5).unwrap();
        let mlp_out = crate::mlp(
            &mlp_input,
            &mlp_in_weight,
            Some(&mlp_in_bias),
            &mlp_out_weight,
            Some(&mlp_out_bias),
            2,
            2,
            3,
        )
        .unwrap();
        crate::add_inplace(&mut residual, &mlp_out).unwrap();
        residual
    }

    fn block_params_for_test<'a>() -> StreamingBlockParameters<'a> {
        static LN1_WEIGHT: [f32; 2] = [1.1, 0.9];
        static LN1_BIAS: [f32; 2] = [0.05, -0.05];
        static QKV_BIAS: [f32; 6] = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2];
        static ATTENTION_OUT_BIAS: [f32; 2] = [0.05, -0.05];
        static LN2_WEIGHT: [f32; 2] = [0.8, 1.2];
        static LN2_BIAS: [f32; 2] = [-0.02, 0.04];
        static MLP_IN_BIAS: [f32; 3] = [0.1, -0.2, 0.3];
        static MLP_OUT_BIAS: [f32; 2] = [0.05, -0.05];

        StreamingBlockParameters {
            input_layernorm_weight: &LN1_WEIGHT,
            input_layernorm_bias: &LN1_BIAS,
            qkv_bias: Some(&QKV_BIAS),
            attention_out_bias: Some(&ATTENTION_OUT_BIAS),
            post_attention_layernorm_weight: &LN2_WEIGHT,
            post_attention_layernorm_bias: &LN2_BIAS,
            mlp_in_bias: Some(&MLP_IN_BIAS),
            mlp_out_bias: Some(&MLP_OUT_BIAS),
        }
    }

    fn block_names_for_test<'a>() -> StreamingBlockTensorNames<'a> {
        StreamingBlockTensorNames {
            qkv_weight: "gpt_neox.layers.0.attention.query_key_value.weight",
            attention_out_weight: "gpt_neox.layers.0.attention.dense.weight",
            mlp_in_weight: "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
            mlp_out_weight: "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
        }
    }

    #[test]
    fn streaming_transformer_block_matches_full_decode_baseline_and_releases_budget() {
        let path = temp_path("block");
        write_chunked_block(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 1.25, 0.75];
        let expected = full_decode_block_baseline(&input, false);
        let mut budget = MemoryBudget::new(512);

        let actual = streaming_transformer_block_from_model(
            &mut model,
            &input,
            block_names_for_test(),
            block_params_for_test(),
            StreamingBlockConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 2,
                intermediate_size: 3,
                causal: true,
                layer_norm_eps: 1e-5,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-5);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 256,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_transformer_block_supports_parallel_residual_baseline() {
        let path = temp_path("block_parallel_residual");
        write_chunked_block(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 1.25, 0.75];
        let expected = full_decode_block_baseline(&input, true);
        let mut budget = MemoryBudget::new(512);

        let actual = streaming_transformer_block_with_runtime_from_model(
            &mut model,
            &input,
            block_names_for_test(),
            block_params_for_test(),
            StreamingBlockConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 2,
                intermediate_size: 3,
                causal: true,
                layer_norm_eps: 1e-5,
            },
            StreamingBlockRuntime {
                attention: StreamingAttentionRuntime::default(),
                parallel_residual: true,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-5);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_transformer_block_timing_records_each_subphase_once() {
        let path = temp_path("block-timing");
        write_chunked_block(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 1.25, 0.75];
        let expected = full_decode_block_baseline(&input, false);
        let mut budget = MemoryBudget::new(512);
        let mut timing = StreamingBlockTiming::default();

        let actual = streaming_transformer_block_with_runtime_and_timing_from_model(
            &mut model,
            &input,
            block_names_for_test(),
            block_params_for_test(),
            StreamingBlockConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 2,
                intermediate_size: 3,
                causal: true,
                layer_norm_eps: 1e-5,
            },
            StreamingBlockRuntime::default(),
            &mut budget,
            Some(&mut timing),
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-5);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(timing.attention_norm_calls, 1);
        assert_eq!(timing.attention_calls, 1);
        assert_eq!(timing.attention_qkv_projection_calls, 1);
        assert_eq!(timing.attention_qkv_split_calls, 1);
        assert_eq!(timing.attention_score_context_calls, 1);
        assert_eq!(timing.attention_output_projection_calls, 1);
        assert_eq!(timing.attention_rotary_calls, 0);
        assert_eq!(timing.attention_kv_append_calls, 0);
        assert_eq!(timing.attention_residual_calls, 1);
        assert_eq!(timing.mlp_norm_calls, 1);
        assert_eq!(timing.mlp_calls, 1);
        assert_eq!(timing.mlp_input_projection_calls, 1);
        assert_eq!(timing.mlp_activation_calls, 1);
        assert_eq!(timing.mlp_output_projection_calls, 1);
        assert_eq!(timing.mlp_residual_calls, 1);
        assert!(timing.attention_qkv_projection_ns > 0);
        assert!(timing.attention_qkv_split_ns > 0);
        assert!(timing.attention_score_context_ns > 0);
        assert!(timing.attention_output_projection_ns > 0);
        assert!(timing.mlp_input_projection_ns > 0);
        assert!(timing.mlp_activation_ns > 0);
        assert!(timing.mlp_output_projection_ns > 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_transformer_block_rejects_too_small_attention_budget_without_leaking() {
        let path = temp_path("block-budget");
        write_chunked_block(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0];
        let mut budget = MemoryBudget::new(31);

        let err = streaming_transformer_block_from_model(
            &mut model,
            &input,
            block_names_for_test(),
            block_params_for_test(),
            StreamingBlockConfig {
                seq_len: 1,
                num_heads: 1,
                head_dim: 2,
                intermediate_size: 3,
                causal: true,
                layer_norm_eps: 1e-5,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }
}
