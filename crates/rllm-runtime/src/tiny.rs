use crate::tensor::decode_to_f32;
use crate::{
    layer_norm, sample_argmax, sample_top_p, streaming_tile_linear_from_model,
    streaming_transformer_block_with_runtime_from_model, KvCache, LazyRllmModel, MemoryBudget,
    Result, RotaryEmbeddingConfig, RuntimeError, StreamingAttentionRuntime, StreamingBlockConfig,
    StreamingBlockParameters, StreamingBlockRuntime, StreamingBlockTensorNames,
    StreamingLinearConfig, StreamingTileLinearConfig, DEFAULT_STREAMING_TILE_ELEMENTS,
};
use rllm_container::{ChunkMeta, TensorMeta};

#[derive(Debug, Clone, Copy)]
pub struct StreamingEmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

#[derive(Debug, Clone)]
struct EmbeddingChunkWindow {
    chunk: ChunkMeta,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Debug, Clone)]
struct EmbeddingCopySpan {
    range_start_in_chunk: usize,
    range_len_bytes: usize,
    output_start: usize,
}

#[derive(Debug, Clone)]
struct EmbeddingChunkRequest {
    chunk: ChunkMeta,
    range_start_in_chunk: usize,
    range_len_bytes: usize,
    copies: Vec<EmbeddingCopySpan>,
}

#[derive(Debug, Clone, Copy)]
pub enum StreamingSamplingConfig {
    Argmax,
    TopP {
        temperature: f32,
        top_p: f32,
        seed: u64,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingTinyTransformerConfig {
    pub seq_len: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub causal: bool,
    pub layer_norm_eps: f32,
    pub sampling: StreamingSamplingConfig,
}

impl StreamingTinyTransformerConfig {
    fn hidden_size(self) -> Result<usize> {
        self.num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| RuntimeError::Shape("tiny transformer hidden_size overflow".to_string()))
    }

    fn embedding_config(self) -> Result<StreamingEmbeddingConfig> {
        Ok(StreamingEmbeddingConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size()?,
        })
    }

    fn block_config(self) -> StreamingBlockConfig {
        StreamingBlockConfig {
            seq_len: self.seq_len,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            causal: self.causal,
            layer_norm_eps: self.layer_norm_eps,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingTinyTransformerTensorNames<'a> {
    pub embedding_weight: &'a str,
    pub block: StreamingBlockTensorNames<'a>,
    pub lm_head_weight: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingTinyTransformerParameters<'a> {
    pub block: StreamingBlockParameters<'a>,
    pub final_layernorm_weight: &'a [f32],
    pub final_layernorm_bias: &'a [f32],
    pub lm_head_bias: Option<&'a [f32]>,
}

#[derive(Debug, Clone)]
pub struct StreamingNextTokenResult {
    pub logits: Vec<f32>,
    pub token_id: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingTinyRotaryConfig {
    pub rotary_dim: usize,
    pub base: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingTinyGenerationConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub causal: bool,
    pub layer_norm_eps: f32,
    pub sampling: StreamingSamplingConfig,
    pub rotary: Option<StreamingTinyRotaryConfig>,
}

impl StreamingTinyGenerationConfig {
    fn hidden_size(self) -> Result<usize> {
        self.num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| RuntimeError::Shape("tiny generation hidden_size overflow".to_string()))
    }

    fn tiny_config(self, seq_len: usize) -> StreamingTinyTransformerConfig {
        StreamingTinyTransformerConfig {
            seq_len,
            vocab_size: self.vocab_size,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            causal: self.causal,
            layer_norm_eps: self.layer_norm_eps,
            sampling: self.sampling,
        }
    }

    fn rotary_config(
        self,
        seq_len: usize,
        position_offset: usize,
    ) -> Option<RotaryEmbeddingConfig> {
        self.rotary.map(|rotary| RotaryEmbeddingConfig {
            seq_len,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            rotary_dim: rotary.rotary_dim,
            base: rotary.base,
            position_offset,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ContextEchoState {
    block_kv_caches: Vec<KvCache>,
}

impl ContextEchoState {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(RuntimeError::Shape(
                "ContextEchoState must contain at least one layer cache".to_string(),
            ));
        }
        let mut block_kv_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            block_kv_caches.push(KvCache::new(num_heads, head_dim, max_seq_len)?);
        }
        Ok(Self { block_kv_caches })
    }

    pub fn layer_count(&self) -> usize {
        self.block_kv_caches.len()
    }

    pub fn cache_len(&self, layer_idx: usize) -> Result<usize> {
        self.block_kv_caches
            .get(layer_idx)
            .map(KvCache::len)
            .ok_or_else(|| RuntimeError::Shape(format!("context echo layer {layer_idx} missing")))
    }

    pub fn cache_shape(&self, layer_idx: usize) -> Result<(usize, usize, usize)> {
        self.block_kv_caches
            .get(layer_idx)
            .map(|cache| (cache.num_heads(), cache.head_dim(), cache.max_seq_len()))
            .ok_or_else(|| RuntimeError::Shape(format!("context echo layer {layer_idx} missing")))
    }

    pub fn resident_bytes(&self) -> usize {
        self.block_kv_caches
            .iter()
            .map(|cache| {
                cache
                    .keys()
                    .len()
                    .saturating_add(cache.values().len())
                    .saturating_mul(std::mem::size_of::<f32>())
            })
            .sum()
    }

    pub(crate) fn block_cache_mut(&mut self, layer_idx: usize) -> Result<&mut KvCache> {
        self.block_kv_caches
            .get_mut(layer_idx)
            .ok_or_else(|| RuntimeError::Shape(format!("context echo layer {layer_idx} missing")))
    }
}

#[derive(Debug, Clone)]
pub struct StreamingTinyGenerationResult {
    pub token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub step_logits: Vec<Vec<f32>>,
    pub context_echo_state: ContextEchoState,
    pub context_echo_bytes: usize,
}

/// Low-RAM embedding lookup over a chunked `.rllm` embedding tensor.
///
/// The returned activation is caller-owned. `budget` tracks only transient
/// compressed/decoded/f32 scratch for the streamed chunks.
pub fn streaming_embedding_lookup_from_model(
    model: &mut LazyRllmModel,
    embedding_name: &str,
    token_ids: &[usize],
    config: StreamingEmbeddingConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    validate_embedding_inputs(token_ids, config)?;
    let tensor = model.tensor(embedding_name)?.clone();
    validate_embedding_tensor(&tensor, config)?;

    let chunks: Vec<ChunkMeta> = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    if chunks.is_empty() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "embedding tensor {embedding_name} has no chunks"
        )));
    }

    let mut output = vec![0.0f32; token_ids.len() * config.hidden_size];
    let dtype_size = tensor.dtype.size_bytes();
    let expected_bytes = config
        .vocab_size
        .checked_mul(config.hidden_size)
        .and_then(|elements| elements.checked_mul(dtype_size))
        .ok_or_else(|| RuntimeError::Shape("embedding byte size overflow".to_string()))?;
    let row_byte_len = config
        .hidden_size
        .checked_mul(dtype_size)
        .ok_or_else(|| RuntimeError::Shape("embedding row byte size overflow".to_string()))?;
    let chunk_windows =
        embedding_chunk_windows(&chunks, dtype_size, expected_bytes, embedding_name)?;
    let requests = build_embedding_row_requests(
        token_ids,
        config,
        &chunk_windows,
        row_byte_len,
        dtype_size,
        embedding_name,
    )?;

    for request in requests {
        model.with_decoded_chunk_range(
            request.chunk.chunk_id,
            request.range_start_in_chunk as u64,
            request.range_len_bytes as u64,
            budget,
            |bytes, budget| {
                if bytes.len() != request.range_len_bytes {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "embedding chunk {} range decoded to {} bytes, expected {}",
                        request.chunk.chunk_id,
                        bytes.len(),
                        request.range_len_bytes
                    )));
                }
                if bytes.len() % dtype_size != 0 {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "embedding chunk {} range byte len {} is not aligned to dtype size {}",
                        request.chunk.chunk_id,
                        bytes.len(),
                        dtype_size
                    )));
                }

                let scratch_bytes = (bytes.len() / dtype_size)
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        RuntimeError::Shape("embedding f32 scratch overflow".to_string())
                    })?;
                let scratch_label = format!(
                    "streaming embedding f32 scratch chunk {} range [{}..{})",
                    request.chunk.chunk_id,
                    request.range_start_in_chunk,
                    request.range_start_in_chunk + request.range_len_bytes
                );
                budget.reserve(scratch_bytes, scratch_label.clone())?;
                let values = match decode_to_f32(tensor.dtype, bytes) {
                    Ok(values) => values,
                    Err(err) => {
                        budget.release(scratch_bytes, scratch_label)?;
                        return Err(err);
                    }
                };

                let result = copy_embedding_request_values(
                    &values,
                    &request,
                    &mut output,
                    dtype_size,
                    config,
                    embedding_name,
                );
                drop(values);
                budget.release(scratch_bytes, scratch_label)?;
                result
            },
        )?;
    }

    Ok(output)
}

/// Tiny end-to-end next-token smoke path over one streaming transformer block.
///
/// This is a Phase 5 correctness primitive, not real model generation: it uses
/// caller-supplied norm/bias parameters, one block, final layer norm, and a
/// chunk-streamed LM head. Tokenizer, rotary embeddings, and KV-cache reuse are
/// intentionally out of scope.
pub fn streaming_tiny_transformer_next_token_from_model(
    model: &mut LazyRllmModel,
    token_ids: &[usize],
    names: StreamingTinyTransformerTensorNames<'_>,
    params: StreamingTinyTransformerParameters<'_>,
    config: StreamingTinyTransformerConfig,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    streaming_tiny_transformer_next_token_with_runtime_from_model(
        model,
        token_ids,
        names,
        params,
        config,
        StreamingBlockRuntime::default(),
        budget,
    )
}

pub fn streaming_tiny_transformer_next_token_with_runtime_from_model(
    model: &mut LazyRllmModel,
    token_ids: &[usize],
    names: StreamingTinyTransformerTensorNames<'_>,
    params: StreamingTinyTransformerParameters<'_>,
    config: StreamingTinyTransformerConfig,
    runtime: StreamingBlockRuntime<'_>,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    validate_tiny_inputs(token_ids, params, config)?;
    let hidden_size = config.hidden_size()?;
    let sequence_bytes = activation_bytes(config.seq_len, hidden_size, "tiny sequence activation")?;

    let embedding_label = "tiny transformer embedding activation".to_string();
    budget.reserve(sequence_bytes, embedding_label.clone())?;
    let embeddings = match streaming_embedding_lookup_from_model(
        model,
        names.embedding_weight,
        token_ids,
        config.embedding_config()?,
        budget,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(sequence_bytes, embedding_label)?;
            return Err(err);
        }
    };

    let block_output_label = "tiny transformer block output activation".to_string();
    if let Err(err) = budget.reserve(sequence_bytes, block_output_label.clone()) {
        budget.release(sequence_bytes, embedding_label)?;
        return Err(err);
    }
    let block_output = match streaming_transformer_block_with_runtime_from_model(
        model,
        &embeddings,
        names.block,
        params.block,
        config.block_config(),
        runtime,
        budget,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(sequence_bytes, block_output_label)?;
            budget.release(sequence_bytes, embedding_label)?;
            return Err(err);
        }
    };
    drop(embeddings);
    budget.release(sequence_bytes, embedding_label)?;

    let final_norm_label = "tiny transformer final layernorm activation".to_string();
    if let Err(err) = budget.reserve(sequence_bytes, final_norm_label.clone()) {
        budget.release(sequence_bytes, block_output_label)?;
        return Err(err);
    }
    let final_norm = match layer_norm(
        &block_output,
        params.final_layernorm_weight,
        params.final_layernorm_bias,
        config.seq_len,
        hidden_size,
        config.layer_norm_eps,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(sequence_bytes, final_norm_label)?;
            budget.release(sequence_bytes, block_output_label)?;
            return Err(err);
        }
    };
    drop(block_output);
    budget.release(sequence_bytes, block_output_label)?;

    let last_hidden_start = (config.seq_len - 1)
        .checked_mul(hidden_size)
        .ok_or_else(|| RuntimeError::Shape("last hidden offset overflow".to_string()))?;
    let last_hidden = &final_norm[last_hidden_start..last_hidden_start + hidden_size];
    let logits = match streaming_tile_linear_from_model(
        model,
        names.lm_head_weight,
        last_hidden,
        params.lm_head_bias,
        StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: hidden_size,
                out_features: config.vocab_size,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        },
        budget,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(sequence_bytes, final_norm_label)?;
            return Err(err);
        }
    };
    drop(final_norm);
    budget.release(sequence_bytes, final_norm_label)?;

    let token_id = sample_logits(&logits, config.sampling)?;
    Ok(StreamingNextTokenResult { logits, token_id })
}

pub fn streaming_tiny_transformer_prefill_from_model(
    model: &mut LazyRllmModel,
    prompt_token_ids: &[usize],
    names: StreamingTinyTransformerTensorNames<'_>,
    params: StreamingTinyTransformerParameters<'_>,
    config: StreamingTinyGenerationConfig,
    state: &mut ContextEchoState,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    validate_generation_inputs(prompt_token_ids, config, state)?;
    if state.cache_len(0)? != 0 {
        return Err(RuntimeError::InvalidTensorData(
            "prefill requires an empty context echo cache".to_string(),
        ));
    }
    streaming_tiny_transformer_generation_step_from_model(
        model,
        prompt_token_ids,
        0,
        names,
        params,
        config,
        state,
        budget,
    )
}

pub fn streaming_tiny_transformer_decode_step_from_model(
    model: &mut LazyRllmModel,
    token_id: usize,
    position_offset: usize,
    names: StreamingTinyTransformerTensorNames<'_>,
    params: StreamingTinyTransformerParameters<'_>,
    config: StreamingTinyGenerationConfig,
    state: &mut ContextEchoState,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    validate_generation_config(config)?;
    validate_embedding_inputs(&[token_id], config.embedding_config()?)?;
    validate_context_echo_state(config, state)?;
    let current_cache_len = state.cache_len(0)?;
    if current_cache_len != position_offset {
        return Err(RuntimeError::Shape(format!(
            "decode position_offset {position_offset} must match context echo cache len {current_cache_len}"
        )));
    }
    streaming_tiny_transformer_generation_step_from_model(
        model,
        &[token_id],
        position_offset,
        names,
        params,
        config,
        state,
        budget,
    )
}

pub fn streaming_tiny_transformer_generate_from_model(
    model: &mut LazyRllmModel,
    prompt_token_ids: &[usize],
    names: StreamingTinyTransformerTensorNames<'_>,
    params: StreamingTinyTransformerParameters<'_>,
    config: StreamingTinyGenerationConfig,
    budget: &mut MemoryBudget,
) -> Result<StreamingTinyGenerationResult> {
    validate_generation_inputs(
        prompt_token_ids,
        config,
        &ContextEchoState::new(1, config.num_heads, config.head_dim, config.max_seq_len)?,
    )?;
    let mut state =
        ContextEchoState::new(1, config.num_heads, config.head_dim, config.max_seq_len)?;
    let mut token_ids = prompt_token_ids.to_vec();
    let mut generated_token_ids = Vec::with_capacity(config.max_new_tokens);
    let mut step_logits = Vec::with_capacity(config.max_new_tokens);

    let prefill = streaming_tiny_transformer_prefill_from_model(
        model,
        prompt_token_ids,
        names,
        params,
        config,
        &mut state,
        budget,
    )?;
    generated_token_ids.push(prefill.token_id);
    token_ids.push(prefill.token_id);
    step_logits.push(prefill.logits);

    while generated_token_ids.len() < config.max_new_tokens {
        let input_token = *generated_token_ids.last().ok_or_else(|| {
            RuntimeError::InvalidTensorData("missing generated token".to_string())
        })?;
        let position_offset = token_ids.len() - 1;
        let step = streaming_tiny_transformer_decode_step_from_model(
            model,
            input_token,
            position_offset,
            names,
            params,
            config,
            &mut state,
            budget,
        )?;
        generated_token_ids.push(step.token_id);
        token_ids.push(step.token_id);
        step_logits.push(step.logits);
    }

    let context_echo_bytes = state.resident_bytes();
    Ok(StreamingTinyGenerationResult {
        token_ids,
        generated_token_ids,
        step_logits,
        context_echo_state: state,
        context_echo_bytes,
    })
}

fn streaming_tiny_transformer_generation_step_from_model(
    model: &mut LazyRllmModel,
    token_ids: &[usize],
    position_offset: usize,
    names: StreamingTinyTransformerTensorNames<'_>,
    params: StreamingTinyTransformerParameters<'_>,
    config: StreamingTinyGenerationConfig,
    state: &mut ContextEchoState,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    let seq_len = token_ids.len();
    let rotary = config.rotary_config(seq_len, position_offset);
    let cache = state.block_cache_mut(0)?;
    streaming_tiny_transformer_next_token_with_runtime_from_model(
        model,
        token_ids,
        names,
        params,
        config.tiny_config(seq_len),
        StreamingBlockRuntime {
            attention: StreamingAttentionRuntime {
                rotary,
                kv_cache: Some(cache),
            },
            parallel_residual: false,
        },
        budget,
    )
}

fn validate_generation_inputs(
    prompt_token_ids: &[usize],
    config: StreamingTinyGenerationConfig,
    state: &ContextEchoState,
) -> Result<()> {
    validate_generation_config(config)?;
    if prompt_token_ids.is_empty() {
        return Err(RuntimeError::Shape(
            "generation prompt must contain at least one token".to_string(),
        ));
    }
    validate_embedding_inputs(prompt_token_ids, config.embedding_config()?)?;
    validate_context_echo_state(config, state)?;
    let required_sequence = prompt_token_ids
        .len()
        .checked_add(config.max_new_tokens)
        .ok_or_else(|| RuntimeError::Shape("generation sequence length overflow".to_string()))?;
    if required_sequence > config.max_seq_len {
        return Err(RuntimeError::Shape(format!(
            "prompt len {} + max_new_tokens {} exceeds max_seq_len {}",
            prompt_token_ids.len(),
            config.max_new_tokens,
            config.max_seq_len
        )));
    }
    Ok(())
}

fn validate_generation_config(config: StreamingTinyGenerationConfig) -> Result<()> {
    if config.max_new_tokens == 0 {
        return Err(RuntimeError::Shape(
            "max_new_tokens must be greater than zero".to_string(),
        ));
    }
    if config.max_seq_len == 0
        || config.vocab_size == 0
        || config.num_heads == 0
        || config.head_dim == 0
        || config.intermediate_size == 0
    {
        return Err(RuntimeError::Shape(format!(
            "generation dimensions must be non-zero: max_seq_len={}, vocab_size={}, num_heads={}, head_dim={}, intermediate_size={}",
            config.max_seq_len,
            config.vocab_size,
            config.num_heads,
            config.head_dim,
            config.intermediate_size
        )));
    }
    let _hidden_size = config.hidden_size()?;
    if !config.layer_norm_eps.is_finite() || config.layer_norm_eps < 0.0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "layer_norm_eps must be finite and non-negative, got {}",
            config.layer_norm_eps
        )));
    }
    if let Some(rotary) = config.rotary {
        if rotary.rotary_dim == 0
            || rotary.rotary_dim > config.head_dim
            || rotary.rotary_dim % 2 != 0
        {
            return Err(RuntimeError::Shape(format!(
                "rotary_dim must be even and in 1..=head_dim, got rotary_dim={}, head_dim={}",
                rotary.rotary_dim, config.head_dim
            )));
        }
        if !rotary.base.is_finite() || rotary.base <= 0.0 {
            return Err(RuntimeError::InvalidTensorData(format!(
                "rotary base must be finite and positive, got {}",
                rotary.base
            )));
        }
    }
    Ok(())
}

fn validate_context_echo_state(
    config: StreamingTinyGenerationConfig,
    state: &ContextEchoState,
) -> Result<()> {
    if state.layer_count() != 1 {
        return Err(RuntimeError::Shape(format!(
            "tiny generation expects exactly one context echo layer, got {}",
            state.layer_count()
        )));
    }
    let cache = state
        .block_kv_caches
        .first()
        .ok_or_else(|| RuntimeError::Shape("missing context echo layer 0".to_string()))?;
    if cache.num_heads() != config.num_heads || cache.head_dim() != config.head_dim {
        return Err(RuntimeError::Shape(format!(
            "context echo cache shape {}/{} does not match generation config {}/{}",
            cache.num_heads(),
            cache.head_dim(),
            config.num_heads,
            config.head_dim
        )));
    }
    if cache.max_seq_len() != config.max_seq_len {
        return Err(RuntimeError::Shape(format!(
            "context echo cache max_seq_len {} does not match generation config {}",
            cache.max_seq_len(),
            config.max_seq_len
        )));
    }
    Ok(())
}

impl StreamingTinyGenerationConfig {
    fn embedding_config(self) -> Result<StreamingEmbeddingConfig> {
        Ok(StreamingEmbeddingConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size()?,
        })
    }
}

fn sample_logits(logits: &[f32], sampling: StreamingSamplingConfig) -> Result<usize> {
    match sampling {
        StreamingSamplingConfig::Argmax => sample_argmax(logits),
        StreamingSamplingConfig::TopP {
            temperature,
            top_p,
            seed,
        } => sample_top_p(logits, temperature, top_p, seed),
    }
}

fn validate_embedding_inputs(token_ids: &[usize], config: StreamingEmbeddingConfig) -> Result<()> {
    if config.vocab_size == 0 || config.hidden_size == 0 {
        return Err(RuntimeError::Shape(format!(
            "embedding config must be non-zero, got vocab_size={}, hidden_size={}",
            config.vocab_size, config.hidden_size
        )));
    }
    for &token_id in token_ids {
        if token_id >= config.vocab_size {
            return Err(RuntimeError::Shape(format!(
                "token id {token_id} out of range for vocab size {}",
                config.vocab_size
            )));
        }
    }
    Ok(())
}

fn validate_embedding_tensor(tensor: &TensorMeta, config: StreamingEmbeddingConfig) -> Result<()> {
    if tensor.shape.len() != 2 {
        return Err(RuntimeError::Shape(format!(
            "embedding tensor {} must be rank-2 [vocab, hidden], got {:?}",
            tensor.name, tensor.shape
        )));
    }
    let vocab = usize::try_from(tensor.shape[0])
        .map_err(|_| RuntimeError::Shape("embedding vocab size overflows usize".to_string()))?;
    let hidden = usize::try_from(tensor.shape[1])
        .map_err(|_| RuntimeError::Shape("embedding hidden size overflows usize".to_string()))?;
    if vocab != config.vocab_size || hidden != config.hidden_size {
        return Err(RuntimeError::Shape(format!(
            "embedding tensor {} shape {:?} does not match requested [{}, {}]",
            tensor.name, tensor.shape, config.vocab_size, config.hidden_size
        )));
    }
    let expected_bytes = config
        .vocab_size
        .checked_mul(config.hidden_size)
        .and_then(|elements| elements.checked_mul(tensor.dtype.size_bytes()))
        .ok_or_else(|| RuntimeError::Shape("embedding byte size overflow".to_string()))?;
    if tensor.original_size_bytes != expected_bytes as u64 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "embedding tensor {} original_size_bytes={} does not match shape/dtype bytes {}",
            tensor.name, tensor.original_size_bytes, expected_bytes
        )));
    }
    Ok(())
}

fn embedding_chunk_windows(
    chunks: &[ChunkMeta],
    dtype_size: usize,
    expected_bytes: usize,
    embedding_name: &str,
) -> Result<Vec<EmbeddingChunkWindow>> {
    let mut windows = Vec::with_capacity(chunks.len());
    let mut byte_offset = 0usize;
    for chunk in chunks {
        if byte_offset % dtype_size != 0 {
            return Err(RuntimeError::InvalidTensorData(format!(
                "embedding tensor {embedding_name} chunk stream reached unaligned byte offset {byte_offset} for dtype size {dtype_size}"
            )));
        }
        let chunk_bytes = usize::try_from(chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                chunk.chunk_id
            ))
        })?;
        if chunk_bytes % dtype_size != 0 {
            return Err(RuntimeError::InvalidTensorData(format!(
                "embedding tensor {embedding_name} chunk {} byte len {} is not aligned to dtype size {}",
                chunk.chunk_id, chunk_bytes, dtype_size
            )));
        }
        let end_byte = byte_offset.checked_add(chunk_bytes).ok_or_else(|| {
            RuntimeError::InvalidTensorData("embedding chunk byte offset overflow".to_string())
        })?;
        windows.push(EmbeddingChunkWindow {
            chunk: chunk.clone(),
            start_byte: byte_offset,
            end_byte,
        });
        byte_offset = end_byte;
    }

    if byte_offset != expected_bytes {
        return Err(RuntimeError::InvalidTensorData(format!(
            "embedding tensor {embedding_name} chunk stream covers {byte_offset} bytes, expected {expected_bytes}"
        )));
    }

    Ok(windows)
}

fn build_embedding_row_requests(
    token_ids: &[usize],
    config: StreamingEmbeddingConfig,
    windows: &[EmbeddingChunkWindow],
    row_byte_len: usize,
    dtype_size: usize,
    embedding_name: &str,
) -> Result<Vec<EmbeddingChunkRequest>> {
    let mut requests = Vec::new();
    for window in windows {
        let mut copies = Vec::new();
        let mut request_start = usize::MAX;
        let mut request_end = 0usize;

        for (seq_idx, &token_id) in token_ids.iter().enumerate() {
            let row_start = token_id.checked_mul(row_byte_len).ok_or_else(|| {
                RuntimeError::Shape("embedding row byte offset overflow".to_string())
            })?;
            let row_end = row_start.checked_add(row_byte_len).ok_or_else(|| {
                RuntimeError::Shape("embedding row byte end overflow".to_string())
            })?;
            let overlap_start = row_start.max(window.start_byte);
            let overlap_end = row_end.min(window.end_byte);
            if overlap_start >= overlap_end {
                continue;
            }
            let range_start_in_chunk = overlap_start - window.start_byte;
            let range_len_bytes = overlap_end - overlap_start;
            if range_start_in_chunk % dtype_size != 0 || range_len_bytes % dtype_size != 0 {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "embedding tensor {embedding_name} token row {token_id} overlaps chunk {} on unaligned byte range [{}..{}) for dtype size {}",
                    window.chunk.chunk_id,
                    range_start_in_chunk,
                    range_start_in_chunk + range_len_bytes,
                    dtype_size
                )));
            }
            let output_col_start = (overlap_start - row_start) / dtype_size;
            let output_start = seq_idx
                .checked_mul(config.hidden_size)
                .and_then(|base| base.checked_add(output_col_start))
                .ok_or_else(|| {
                    RuntimeError::Shape("embedding output offset overflow".to_string())
                })?;
            copies.push(EmbeddingCopySpan {
                range_start_in_chunk,
                range_len_bytes,
                output_start,
            });
            request_start = request_start.min(range_start_in_chunk);
            request_end = request_end.max(range_start_in_chunk + range_len_bytes);
        }

        if !copies.is_empty() {
            requests.push(EmbeddingChunkRequest {
                chunk: window.chunk.clone(),
                range_start_in_chunk: request_start,
                range_len_bytes: request_end - request_start,
                copies,
            });
        }
    }

    Ok(requests)
}

fn copy_embedding_request_values(
    values: &[f32],
    request: &EmbeddingChunkRequest,
    output: &mut [f32],
    dtype_size: usize,
    config: StreamingEmbeddingConfig,
    embedding_name: &str,
) -> Result<()> {
    for copy in &request.copies {
        let relative_byte_start = copy
            .range_start_in_chunk
            .checked_sub(request.range_start_in_chunk)
            .ok_or_else(|| RuntimeError::Shape("embedding relative range underflow".to_string()))?;
        if relative_byte_start % dtype_size != 0 || copy.range_len_bytes % dtype_size != 0 {
            return Err(RuntimeError::InvalidTensorData(format!(
                "embedding tensor {embedding_name} copy range is not aligned to dtype size {dtype_size}"
            )));
        }
        let source_start = relative_byte_start / dtype_size;
        let source_len = copy.range_len_bytes / dtype_size;
        let source_end = source_start
            .checked_add(source_len)
            .ok_or_else(|| RuntimeError::Shape("embedding source range overflow".to_string()))?;
        let output_end = copy
            .output_start
            .checked_add(source_len)
            .ok_or_else(|| RuntimeError::Shape("embedding output range overflow".to_string()))?;
        if source_end > values.len() || output_end > output.len() || source_len > config.hidden_size
        {
            return Err(RuntimeError::InvalidTensorData(format!(
                "embedding tensor {embedding_name} copy range out of bounds: source [{}..{}) / {}, output [{}..{}) / {}",
                source_start,
                source_end,
                values.len(),
                copy.output_start,
                output_end,
                output.len()
            )));
        }
        output[copy.output_start..output_end].copy_from_slice(&values[source_start..source_end]);
    }
    Ok(())
}

fn validate_tiny_inputs(
    token_ids: &[usize],
    params: StreamingTinyTransformerParameters<'_>,
    config: StreamingTinyTransformerConfig,
) -> Result<()> {
    if config.seq_len == 0 {
        return Err(RuntimeError::Shape(
            "tiny transformer seq_len must be greater than zero".to_string(),
        ));
    }
    if token_ids.len() != config.seq_len {
        return Err(RuntimeError::Shape(format!(
            "token_ids len {} does not match seq_len {}",
            token_ids.len(),
            config.seq_len
        )));
    }
    validate_embedding_inputs(token_ids, config.embedding_config()?)?;
    let hidden_size = config.hidden_size()?;
    validate_norm_params(
        "final layernorm",
        params.final_layernorm_weight,
        params.final_layernorm_bias,
        hidden_size,
    )?;
    if let Some(bias) = params.lm_head_bias {
        if bias.len() != config.vocab_size {
            return Err(RuntimeError::Shape(format!(
                "LM head bias len {} does not match vocab_size {}",
                bias.len(),
                config.vocab_size
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

fn activation_bytes(rows: usize, cols: usize, label: &str) -> Result<usize> {
    rows.checked_mul(cols)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| RuntimeError::Shape(format!("{label} byte size overflow")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        embedding_lookup, layer_norm, linear, mlp, sample_argmax, scaled_dot_product_attention,
        LazyRllmModel, MemoryBudget, RuntimeError, StreamingBlockParameters,
        StreamingBlockTensorNames,
    };
    use rllm_container::{DType, GlobalMetadata, RllmWriter, TensorMeta};
    use sha2::{Digest, Sha256};

    const VOCAB_SIZE: usize = 3;
    const HIDDEN_SIZE: usize = 2;
    const SEQ_LEN: usize = 2;
    const NUM_HEADS: usize = 1;
    const HEAD_DIM: usize = 2;
    const INTERMEDIATE_SIZE: usize = 3;

    const EMBEDDING_WEIGHT: [f32; 6] = [0.5, -1.0, 1.25, 0.75, -0.5, 0.25];
    const QKV_WEIGHT: [f32; 12] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
    const ATTENTION_OUT_WEIGHT: [f32; 4] = [1.0, 0.5, -0.25, 1.0];
    const MLP_IN_WEIGHT: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 1.0, -1.0];
    const MLP_OUT_WEIGHT: [f32; 6] = [1.0, 2.0, 3.0, -1.0, 0.5, 0.25];
    const LM_HEAD_WEIGHT: [f32; 6] = [1.0, 0.0, 0.0, 1.0, -0.5, 0.75];

    const LN1_WEIGHT: [f32; 2] = [1.1, 0.9];
    const LN1_BIAS: [f32; 2] = [0.05, -0.05];
    const QKV_BIAS: [f32; 6] = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2];
    const ATTENTION_OUT_BIAS: [f32; 2] = [0.05, -0.05];
    const LN2_WEIGHT: [f32; 2] = [0.8, 1.2];
    const LN2_BIAS: [f32; 2] = [-0.02, 0.04];
    const MLP_IN_BIAS: [f32; 3] = [0.1, -0.2, 0.3];
    const MLP_OUT_BIAS: [f32; 2] = [0.05, -0.05];
    const FINAL_LN_WEIGHT: [f32; 2] = [1.0, 1.0];
    const FINAL_LN_BIAS: [f32; 2] = [0.0, 0.0];
    const LM_HEAD_BIAS: [f32; 3] = [0.01, 0.02, -0.01];

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
        std::env::temp_dir().join(format!("rllm-tiny-{name}-{}.rllm", std::process::id()))
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

    fn write_tiny_model(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "embed.weight",
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
            &EMBEDDING_WEIGHT,
            16,
        );
        add_f32_tensor(
            &mut writer,
            1,
            "gpt_neox.layers.0.attention.query_key_value.weight",
            vec![(3 * HIDDEN_SIZE) as u64, HIDDEN_SIZE as u64],
            &QKV_WEIGHT,
            20,
        );
        add_f32_tensor(
            &mut writer,
            2,
            "gpt_neox.layers.0.attention.dense.weight",
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            &ATTENTION_OUT_WEIGHT,
            8,
        );
        add_f32_tensor(
            &mut writer,
            3,
            "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
            vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            &MLP_IN_WEIGHT,
            12,
        );
        add_f32_tensor(
            &mut writer,
            4,
            "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
            vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
            &MLP_OUT_WEIGHT,
            16,
        );
        add_f32_tensor(
            &mut writer,
            5,
            "lm_head.weight",
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
            &LM_HEAD_WEIGHT,
            12,
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

    fn split_qkv(fused: &[f32], seq_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hidden = HIDDEN_SIZE;
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

    fn full_decode_block_baseline_with_rotary(
        input: &[f32],
        seq_len: usize,
        rotary: Option<StreamingTinyRotaryConfig>,
    ) -> Vec<f32> {
        let attention_input =
            layer_norm(input, &LN1_WEIGHT, &LN1_BIAS, seq_len, HIDDEN_SIZE, 1e-5).unwrap();
        let fused = linear(
            &attention_input,
            &QKV_WEIGHT,
            Some(&QKV_BIAS),
            seq_len,
            HIDDEN_SIZE,
            3 * HIDDEN_SIZE,
        )
        .unwrap();
        let (mut q, mut k, v) = split_qkv(&fused, seq_len);
        if let Some(rotary) = rotary {
            crate::apply_gpt_neox_rotary_inplace(
                &mut q,
                &mut k,
                crate::RotaryEmbeddingConfig {
                    seq_len,
                    num_heads: NUM_HEADS,
                    head_dim: HEAD_DIM,
                    rotary_dim: rotary.rotary_dim,
                    base: rotary.base,
                    position_offset: 0,
                },
            )
            .unwrap();
        }
        let attended =
            scaled_dot_product_attention(&q, &k, &v, seq_len, NUM_HEADS, HEAD_DIM, true).unwrap();
        let attention_out = linear(
            &attended,
            &ATTENTION_OUT_WEIGHT,
            Some(&ATTENTION_OUT_BIAS),
            seq_len,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )
        .unwrap();
        let mut residual = input.to_vec();
        crate::add_inplace(&mut residual, &attention_out).unwrap();

        let mlp_input = layer_norm(
            &residual,
            &LN2_WEIGHT,
            &LN2_BIAS,
            seq_len,
            HIDDEN_SIZE,
            1e-5,
        )
        .unwrap();
        let mlp_out = mlp(
            &mlp_input,
            &MLP_IN_WEIGHT,
            Some(&MLP_IN_BIAS),
            &MLP_OUT_WEIGHT,
            Some(&MLP_OUT_BIAS),
            seq_len,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
        )
        .unwrap();
        crate::add_inplace(&mut residual, &mlp_out).unwrap();
        residual
    }

    fn full_decode_tiny_next_token_baseline(token_ids: &[usize]) -> (Vec<f32>, usize) {
        full_decode_tiny_next_token_baseline_with_rotary(token_ids, None)
    }

    fn full_decode_tiny_next_token_baseline_with_rotary(
        token_ids: &[usize],
        rotary: Option<StreamingTinyRotaryConfig>,
    ) -> (Vec<f32>, usize) {
        let embeddings =
            embedding_lookup(&EMBEDDING_WEIGHT, VOCAB_SIZE, HIDDEN_SIZE, token_ids).unwrap();
        let block_out =
            full_decode_block_baseline_with_rotary(&embeddings, token_ids.len(), rotary);
        let final_norm = layer_norm(
            &block_out,
            &FINAL_LN_WEIGHT,
            &FINAL_LN_BIAS,
            token_ids.len(),
            HIDDEN_SIZE,
            1e-5,
        )
        .unwrap();
        let last_hidden =
            &final_norm[(token_ids.len() - 1) * HIDDEN_SIZE..token_ids.len() * HIDDEN_SIZE];
        let logits = linear(
            last_hidden,
            &LM_HEAD_WEIGHT,
            Some(&LM_HEAD_BIAS),
            1,
            HIDDEN_SIZE,
            VOCAB_SIZE,
        )
        .unwrap();
        let token_id = sample_argmax(&logits).unwrap();
        (logits, token_id)
    }

    fn block_names_for_test<'a>() -> StreamingBlockTensorNames<'a> {
        StreamingBlockTensorNames {
            qkv_weight: "gpt_neox.layers.0.attention.query_key_value.weight",
            attention_out_weight: "gpt_neox.layers.0.attention.dense.weight",
            mlp_in_weight: "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
            mlp_out_weight: "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
        }
    }

    fn block_params_for_test<'a>() -> StreamingBlockParameters<'a> {
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

    fn tiny_names_for_test<'a>() -> StreamingTinyTransformerTensorNames<'a> {
        StreamingTinyTransformerTensorNames {
            embedding_weight: "embed.weight",
            block: block_names_for_test(),
            lm_head_weight: "lm_head.weight",
        }
    }

    fn tiny_params_for_test<'a>() -> StreamingTinyTransformerParameters<'a> {
        StreamingTinyTransformerParameters {
            block: block_params_for_test(),
            final_layernorm_weight: &FINAL_LN_WEIGHT,
            final_layernorm_bias: &FINAL_LN_BIAS,
            lm_head_bias: Some(&LM_HEAD_BIAS),
        }
    }

    fn tiny_config_for_test(sampling: StreamingSamplingConfig) -> StreamingTinyTransformerConfig {
        StreamingTinyTransformerConfig {
            seq_len: SEQ_LEN,
            vocab_size: VOCAB_SIZE,
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: INTERMEDIATE_SIZE,
            causal: true,
            layer_norm_eps: 1e-5,
            sampling,
        }
    }

    fn generation_config_for_test(max_new_tokens: usize) -> StreamingTinyGenerationConfig {
        StreamingTinyGenerationConfig {
            max_new_tokens,
            max_seq_len: 8,
            vocab_size: VOCAB_SIZE,
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: INTERMEDIATE_SIZE,
            causal: true,
            layer_norm_eps: 1e-5,
            sampling: StreamingSamplingConfig::Argmax,
            rotary: Some(StreamingTinyRotaryConfig {
                rotary_dim: HEAD_DIM,
                base: 10_000.0,
            }),
        }
    }

    fn full_decode_generate_with_rotary(
        prompt_token_ids: &[usize],
        max_new_tokens: usize,
        rotary: Option<StreamingTinyRotaryConfig>,
    ) -> (Vec<usize>, Vec<Vec<f32>>) {
        let mut token_ids = prompt_token_ids.to_vec();
        let mut step_logits = Vec::new();
        for _ in 0..max_new_tokens {
            let (logits, token_id) =
                full_decode_tiny_next_token_baseline_with_rotary(&token_ids, rotary);
            step_logits.push(logits);
            token_ids.push(token_id);
        }
        (token_ids[prompt_token_ids.len()..].to_vec(), step_logits)
    }

    #[test]
    fn streaming_embedding_lookup_matches_full_decode_and_releases_budget() {
        let path = temp_path("embedding");
        write_tiny_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let token_ids = [2, 0, 2];
        let expected =
            embedding_lookup(&EMBEDDING_WEIGHT, VOCAB_SIZE, HIDDEN_SIZE, &token_ids).unwrap();
        let mut budget = MemoryBudget::new(128);

        let actual = streaming_embedding_lookup_from_model(
            &mut model,
            "embed.weight",
            &token_ids,
            StreamingEmbeddingConfig {
                vocab_size: VOCAB_SIZE,
                hidden_size: HIDDEN_SIZE,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_embedding_lookup_recalls_only_touched_row_chunks() {
        let path = temp_path("embedding-row-recall");
        write_tiny_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let token_ids = [2];
        let expected =
            embedding_lookup(&EMBEDDING_WEIGHT, VOCAB_SIZE, HIDDEN_SIZE, &token_ids).unwrap();
        let mut budget = MemoryBudget::new(24);

        let actual = streaming_embedding_lookup_from_model(
            &mut model,
            "embed.weight",
            &token_ids,
            StreamingEmbeddingConfig {
                vocab_size: VOCAB_SIZE,
                hidden_size: HIDDEN_SIZE,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 24,
            "selective row recall should fit under the budget that full embedding chunk scan exceeds; peak={} bytes",
            budget.peak_bytes()
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tiny_transformer_next_token_matches_full_decode_argmax_smoke() {
        let path = temp_path("next-token");
        write_tiny_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let token_ids = [0, 2];
        let (expected_logits, expected_token_id) = full_decode_tiny_next_token_baseline(&token_ids);
        let mut budget = MemoryBudget::new(1024);

        let actual = streaming_tiny_transformer_next_token_from_model(
            &mut model,
            &token_ids,
            tiny_names_for_test(),
            tiny_params_for_test(),
            tiny_config_for_test(StreamingSamplingConfig::Argmax),
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual.logits, &expected_logits, 1e-5);
        assert_eq!(actual.token_id, expected_token_id);
        assert_eq!(budget.current_bytes(), 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn echo_prefill_and_decode_step_match_full_context_with_rotary_context_echo() {
        let path = temp_path("echo-prefill-decode");
        write_tiny_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prompt = [0, 2];
        let generation_config = generation_config_for_test(2);
        let mut state =
            ContextEchoState::new(1, NUM_HEADS, HEAD_DIM, generation_config.max_seq_len).unwrap();
        let mut budget = MemoryBudget::new(4096);

        let prefill = streaming_tiny_transformer_prefill_from_model(
            &mut model,
            &prompt,
            tiny_names_for_test(),
            tiny_params_for_test(),
            generation_config,
            &mut state,
            &mut budget,
        )
        .unwrap();
        let (expected_prefill_logits, expected_prefill_token) =
            full_decode_tiny_next_token_baseline_with_rotary(&prompt, generation_config.rotary);

        assert_close_vec(&prefill.logits, &expected_prefill_logits, 1e-5);
        assert_eq!(prefill.token_id, expected_prefill_token);
        assert_eq!(state.cache_len(0).unwrap(), prompt.len());
        assert_eq!(budget.current_bytes(), 0);

        let mut context = prompt.to_vec();
        context.push(prefill.token_id);
        let decode = streaming_tiny_transformer_decode_step_from_model(
            &mut model,
            prefill.token_id,
            prompt.len(),
            tiny_names_for_test(),
            tiny_params_for_test(),
            generation_config,
            &mut state,
            &mut budget,
        )
        .unwrap();
        let (expected_decode_logits, expected_decode_token) =
            full_decode_tiny_next_token_baseline_with_rotary(&context, generation_config.rotary);

        assert_close_vec(&decode.logits, &expected_decode_logits, 1e-5);
        assert_eq!(decode.token_id, expected_decode_token);
        assert_eq!(state.cache_len(0).unwrap(), prompt.len() + 1);
        assert!(state.resident_bytes() > 0);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn echo_generation_loop_matches_full_context_recompute_for_each_step() {
        let path = temp_path("echo-generate");
        write_tiny_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prompt = [0, 2];
        let generation_config = generation_config_for_test(3);
        let (expected_generated, expected_logits) = full_decode_generate_with_rotary(
            &prompt,
            generation_config.max_new_tokens,
            generation_config.rotary,
        );
        let mut budget = MemoryBudget::new(8192);

        let actual = streaming_tiny_transformer_generate_from_model(
            &mut model,
            &prompt,
            tiny_names_for_test(),
            tiny_params_for_test(),
            generation_config,
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual.generated_token_ids, expected_generated);
        assert_eq!(
            actual.token_ids,
            [prompt.as_slice(), expected_generated.as_slice()].concat()
        );
        assert_eq!(actual.step_logits.len(), expected_logits.len());
        for (actual_logits, expected_logits) in actual.step_logits.iter().zip(&expected_logits) {
            assert_close_vec(actual_logits, expected_logits, 1e-5);
        }
        assert_eq!(
            actual.context_echo_state.cache_len(0).unwrap(),
            prompt.len() + generation_config.max_new_tokens - 1
        );
        assert_eq!(
            actual.context_echo_bytes,
            actual.context_echo_state.resident_bytes()
        );
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tiny_transformer_rejects_too_small_embedding_budget_without_leaking() {
        let path = temp_path("budget");
        write_tiny_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let token_ids = [0, 2];
        let mut budget = MemoryBudget::new(15);

        let err = streaming_tiny_transformer_next_token_from_model(
            &mut model,
            &token_ids,
            tiny_names_for_test(),
            tiny_params_for_test(),
            tiny_config_for_test(StreamingSamplingConfig::Argmax),
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);
        std::fs::remove_file(&path).ok();
    }
}
