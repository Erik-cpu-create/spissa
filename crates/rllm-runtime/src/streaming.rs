use crate::tensor::decode_to_f32;
use crate::{
    apply_gpt_neox_rotary_inplace, scaled_dot_product_attention_with_cache, KvAttentionConfig,
    KvCache, LazyRllmModel, MemoryBudget, Result, RotaryEmbeddingConfig, RuntimeError,
};
use rllm_container::{ChunkMeta, TensorMeta};

#[derive(Debug, Clone, Copy)]
pub struct StreamingLinearConfig {
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
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

/// Low-RAM two-layer MLP block over chunked `.rllm` weight tensors.
///
/// Computes `Linear(input, w_in, b_in) -> GELU -> Linear(hidden, w_out, b_out)`.
/// The intermediate activation is reserved in `budget` for the duration of the
/// second linear pass, while each weight chunk is still decoded/released one at
/// a time by `streaming_linear_from_model`.
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
    validate_mlp_shapes(input, b_in, b_out, config)?;

    let intermediate_bytes = activation_bytes(
        config.batch,
        config.intermediate_size,
        "streaming MLP intermediate",
    )?;
    let intermediate_label = "streaming MLP intermediate activation".to_string();
    budget.reserve(intermediate_bytes, intermediate_label.clone())?;

    let mut hidden = match streaming_linear_from_model(
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

    crate::ops::gelu_inplace(&mut hidden);

    let output = match streaming_linear_from_model(
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

    drop(hidden);
    budget.release(intermediate_bytes, intermediate_label)?;
    Ok(output)
}

/// Low-RAM attention sub-block over chunked `.rllm` QKV and output weights.
///
/// This implements the non-rotary toy baseline used by Phase 5 tests:
/// `streaming QKV linear -> split Q/K/V -> scaled dot-product attention ->
/// streaming output projection`. Real GPT-NeoX rotary embeddings are still a
/// later block-level concern.
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
    mut runtime: StreamingAttentionRuntime<'_>,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    validate_attention_shapes(input, qkv_bias, out_bias, config)?;
    let hidden_size = config.hidden_size()?;
    let qkv_features = hidden_size
        .checked_mul(3)
        .ok_or_else(|| RuntimeError::Shape("QKV feature count overflow".to_string()))?;

    let qkv_bytes = activation_bytes(config.seq_len, qkv_features, "streaming attention QKV")?;
    let qkv_label = "streaming attention fused QKV activation".to_string();
    budget.reserve(qkv_bytes, qkv_label.clone())?;

    let fused_qkv = match streaming_linear_from_model(
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

    let split_label = "streaming attention split QKV activation".to_string();
    if let Err(err) = budget.reserve(qkv_bytes, split_label.clone()) {
        budget.release(qkv_bytes, qkv_label)?;
        return Err(err);
    }
    let (mut q, mut k, v) = match split_fused_qkv(&fused_qkv, config) {
        Ok(split) => split,
        Err(err) => {
            budget.release(qkv_bytes, split_label)?;
            budget.release(qkv_bytes, qkv_label)?;
            return Err(err);
        }
    };
    drop(fused_qkv);
    budget.release(qkv_bytes, qkv_label)?;

    if let Some(rotary) = runtime.rotary {
        if let Err(err) = apply_gpt_neox_rotary_inplace(&mut q, &mut k, rotary) {
            budget.release(qkv_bytes, split_label)?;
            return Err(err);
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
    if !cache_is_active {
        budget.release(qkv_bytes, split_label.clone())?;
    }

    let output = match streaming_linear_from_model(
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

    if let Some(cache) = runtime.kv_cache.as_deref_mut() {
        if let Err(err) = cache.append(&k, &v, config.seq_len) {
            budget.release(attention_bytes, attention_label)?;
            budget.release(qkv_bytes, split_label)?;
            return Err(err);
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

    let attention_output_label = "streaming block attention output activation".to_string();
    if let Err(err) = budget.reserve(hidden_bytes, attention_output_label.clone()) {
        budget.release(hidden_bytes, attention_input_label)?;
        return Err(err);
    }
    let attention_output = match streaming_attention_with_runtime_from_model(
        model,
        &attention_input,
        names.qkv_weight,
        params.qkv_bias,
        names.attention_out_weight,
        params.attention_out_bias,
        config.attention_config(),
        runtime.attention,
        budget,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, attention_output_label)?;
            budget.release(hidden_bytes, attention_input_label)?;
            return Err(err);
        }
    };
    drop(attention_input);
    budget.release(hidden_bytes, attention_input_label)?;
    if let Err(err) = crate::ops::add_inplace(&mut residual, &attention_output) {
        budget.release(hidden_bytes, attention_output_label)?;
        return Err(err);
    }
    drop(attention_output);
    budget.release(hidden_bytes, attention_output_label)?;

    let mlp_input_label = "streaming block post-attention layernorm activation".to_string();
    budget.reserve(hidden_bytes, mlp_input_label.clone())?;
    let mlp_input = match crate::ops::layer_norm(
        &residual,
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

    let mlp_output_label = "streaming block MLP output activation".to_string();
    if let Err(err) = budget.reserve(hidden_bytes, mlp_output_label.clone()) {
        budget.release(hidden_bytes, mlp_input_label)?;
        return Err(err);
    }
    let mlp_output = match streaming_mlp_from_model(
        model,
        &mlp_input,
        names.mlp_in_weight,
        params.mlp_in_bias,
        names.mlp_out_weight,
        params.mlp_out_bias,
        config.mlp_config()?,
        budget,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, mlp_output_label)?;
            budget.release(hidden_bytes, mlp_input_label)?;
            return Err(err);
        }
    };
    drop(mlp_input);
    budget.release(hidden_bytes, mlp_input_label)?;
    if let Err(err) = crate::ops::add_inplace(&mut residual, &mlp_output) {
        budget.release(hidden_bytes, mlp_output_label)?;
        return Err(err);
    }
    drop(mlp_output);
    budget.release(hidden_bytes, mlp_output_label)?;

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
        q[out_row..out_row + hidden_size]
            .copy_from_slice(&fused[fused_row..fused_row + hidden_size]);
        k[out_row..out_row + hidden_size]
            .copy_from_slice(&fused[fused_row + hidden_size..fused_row + 2 * hidden_size]);
        v[out_row..out_row + hidden_size]
            .copy_from_slice(&fused[fused_row + 2 * hidden_size..fused_row + 3 * hidden_size]);
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

    for (local_idx, &weight) in weights.iter().enumerate() {
        let global_idx = element_start + local_idx;
        if global_idx >= weight_elements {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} chunk element {global_idx} exceeds expected {weight_elements}"
            )));
        }
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        for batch_idx in 0..config.batch {
            let input_idx = batch_idx * config.in_features + in_feature;
            let output_idx = batch_idx * config.out_features + out_feature;
            output[output_idx] += input[input_idx] * weight;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear;
    use rllm_container::{DType, GlobalMetadata, RllmWriter, TensorMeta};
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
            q[out_row..out_row + hidden].copy_from_slice(&fused[fused_row..fused_row + hidden]);
            k[out_row..out_row + hidden]
                .copy_from_slice(&fused[fused_row + hidden..fused_row + 2 * hidden]);
            v[out_row..out_row + hidden]
                .copy_from_slice(&fused[fused_row + 2 * hidden..fused_row + 3 * hidden]);
        }
        (q, k, v)
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

        let actual = streaming_attention_with_runtime_from_model(
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
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(cache.len(), 2);
        assert_close_vec(cache.keys(), &k[..8], 1e-6);
        assert_close_vec(cache.values(), &v[..8], 1e-6);
        assert_eq!(budget.current_bytes(), 0);

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

    fn full_decode_block_baseline(input: &[f32]) -> Vec<f32> {
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

        let mlp_input = crate::layer_norm(&residual, &ln2_weight, &ln2_bias, 2, 2, 1e-5).unwrap();
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
        let expected = full_decode_block_baseline(&input);
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
