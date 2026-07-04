// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

use crate::{
    layer_norm, sample_argmax, sample_top_p, streaming_embedding_lookup_from_model,
    streaming_tile_linear_from_model, streaming_transformer_block_with_runtime_from_model,
    ContextEchoState, LazySpissaModel, MemoryBudget, Result, RotaryEmbeddingConfig, RuntimeError,
    StreamingAttentionRuntime, StreamingBlockConfig, StreamingBlockParameters,
    StreamingBlockRuntime, StreamingBlockTensorNames, StreamingEmbeddingConfig,
    StreamingLinearConfig, StreamingNextTokenResult, StreamingSamplingConfig,
    StreamingTileLinearConfig, StreamingTinyRotaryConfig, DEFAULT_STREAMING_TILE_ELEMENTS,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct StreamingEchoTransformerConfig {
    pub num_layers: usize,
    pub max_new_tokens: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub causal: bool,
    pub layer_norm_eps: f32,
    pub use_parallel_residual: bool,
    pub sampling: StreamingSamplingConfig,
    pub rotary: Option<StreamingTinyRotaryConfig>,
}

impl StreamingEchoTransformerConfig {
    fn hidden_size(self) -> Result<usize> {
        self.num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| RuntimeError::Shape("echo transformer hidden_size overflow".to_string()))
    }

    fn embedding_config(self) -> Result<StreamingEmbeddingConfig> {
        Ok(StreamingEmbeddingConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size()?,
        })
    }

    fn block_config(self, seq_len: usize) -> StreamingBlockConfig {
        StreamingBlockConfig {
            seq_len,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            causal: self.causal,
            layer_norm_eps: self.layer_norm_eps,
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

#[derive(Debug, Clone, Copy)]
pub struct StreamingEchoTransformerTensorNames<'a> {
    pub embedding_weight: &'a str,
    pub layers: &'a [StreamingBlockTensorNames<'a>],
    pub lm_head_weight: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingEchoTransformerParameters<'a> {
    pub layers: &'a [StreamingBlockParameters<'a>],
    pub final_layernorm_weight: &'a [f32],
    pub final_layernorm_bias: &'a [f32],
    pub lm_head_bias: Option<&'a [f32]>,
}

#[derive(Debug, Clone)]
pub struct StreamingEchoGenerationResult {
    pub token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub step_logits: Vec<Vec<f32>>,
    pub context_echo_state: ContextEchoState,
    pub context_echo_bytes: usize,
    pub timing: Option<RamaGenerationTiming>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct RamaGenerationTiming {
    pub prefill_ns: u64,
    pub decode_ns: u64,
    pub final_norm_ns: u64,
    pub lm_head_ns: u64,
    pub sampling_ns: u64,
    pub prefill_embedding_ns: u64,
    pub prefill_layer_params_ns: u64,
    pub prefill_attention_norm_ns: u64,
    pub prefill_attention_ns: u64,
    pub prefill_attention_qkv_projection_ns: u64,
    pub prefill_attention_qkv_split_ns: u64,
    pub prefill_attention_rotary_ns: u64,
    pub prefill_attention_score_context_ns: u64,
    pub prefill_attention_output_projection_ns: u64,
    pub prefill_attention_kv_append_ns: u64,
    pub prefill_attention_residual_ns: u64,
    pub prefill_mlp_norm_ns: u64,
    pub prefill_mlp_ns: u64,
    pub prefill_mlp_input_projection_ns: u64,
    pub prefill_mlp_activation_ns: u64,
    pub prefill_mlp_output_projection_ns: u64,
    pub prefill_mlp_residual_ns: u64,
    pub prefill_chunks: usize,
    pub decode_steps: usize,
    pub max_prefill_chunk_tokens: usize,
    pub prefill_timed_blocks: usize,
}

impl RamaGenerationTiming {
    pub fn record_prefill_chunk(&mut self, chunk_tokens: usize, elapsed_ns: u64) {
        self.prefill_ns = self.prefill_ns.saturating_add(elapsed_ns);
        self.prefill_chunks = self.prefill_chunks.saturating_add(1);
        self.max_prefill_chunk_tokens = self.max_prefill_chunk_tokens.max(chunk_tokens);
    }

    pub fn record_prefill_embedding(&mut self, elapsed_ns: u64) {
        self.prefill_embedding_ns = self.prefill_embedding_ns.saturating_add(elapsed_ns);
    }

    pub fn record_prefill_layer_params(&mut self, elapsed_ns: u64) {
        self.prefill_layer_params_ns = self.prefill_layer_params_ns.saturating_add(elapsed_ns);
    }

    pub fn record_prefill_block_timing(
        &mut self,
        attention_norm_ns: u64,
        attention_ns: u64,
        attention_qkv_projection_ns: u64,
        attention_qkv_split_ns: u64,
        attention_rotary_ns: u64,
        attention_score_context_ns: u64,
        attention_output_projection_ns: u64,
        attention_kv_append_ns: u64,
        attention_residual_ns: u64,
        mlp_norm_ns: u64,
        mlp_ns: u64,
        mlp_input_projection_ns: u64,
        mlp_activation_ns: u64,
        mlp_output_projection_ns: u64,
        mlp_residual_ns: u64,
    ) {
        self.prefill_attention_norm_ns = self
            .prefill_attention_norm_ns
            .saturating_add(attention_norm_ns);
        self.prefill_attention_ns = self.prefill_attention_ns.saturating_add(attention_ns);
        self.prefill_attention_qkv_projection_ns = self
            .prefill_attention_qkv_projection_ns
            .saturating_add(attention_qkv_projection_ns);
        self.prefill_attention_qkv_split_ns = self
            .prefill_attention_qkv_split_ns
            .saturating_add(attention_qkv_split_ns);
        self.prefill_attention_rotary_ns = self
            .prefill_attention_rotary_ns
            .saturating_add(attention_rotary_ns);
        self.prefill_attention_score_context_ns = self
            .prefill_attention_score_context_ns
            .saturating_add(attention_score_context_ns);
        self.prefill_attention_output_projection_ns = self
            .prefill_attention_output_projection_ns
            .saturating_add(attention_output_projection_ns);
        self.prefill_attention_kv_append_ns = self
            .prefill_attention_kv_append_ns
            .saturating_add(attention_kv_append_ns);
        self.prefill_attention_residual_ns = self
            .prefill_attention_residual_ns
            .saturating_add(attention_residual_ns);
        self.prefill_mlp_norm_ns = self.prefill_mlp_norm_ns.saturating_add(mlp_norm_ns);
        self.prefill_mlp_ns = self.prefill_mlp_ns.saturating_add(mlp_ns);
        self.prefill_mlp_input_projection_ns = self
            .prefill_mlp_input_projection_ns
            .saturating_add(mlp_input_projection_ns);
        self.prefill_mlp_activation_ns = self
            .prefill_mlp_activation_ns
            .saturating_add(mlp_activation_ns);
        self.prefill_mlp_output_projection_ns = self
            .prefill_mlp_output_projection_ns
            .saturating_add(mlp_output_projection_ns);
        self.prefill_mlp_residual_ns = self.prefill_mlp_residual_ns.saturating_add(mlp_residual_ns);
        self.prefill_timed_blocks = self.prefill_timed_blocks.saturating_add(1);
    }

    pub fn record_decode_step(&mut self, elapsed_ns: u64) {
        self.decode_ns = self.decode_ns.saturating_add(elapsed_ns);
        self.decode_steps = self.decode_steps.saturating_add(1);
    }

    pub fn record_final_norm(&mut self, elapsed_ns: u64) {
        self.final_norm_ns = self.final_norm_ns.saturating_add(elapsed_ns);
    }

    pub fn record_lm_head(&mut self, elapsed_ns: u64) {
        self.lm_head_ns = self.lm_head_ns.saturating_add(elapsed_ns);
    }

    pub fn record_sampling(&mut self, elapsed_ns: u64) {
        self.sampling_ns = self.sampling_ns.saturating_add(elapsed_ns);
    }
}

impl StreamingEchoGenerationResult {
    pub fn context_memory_bytes(&self) -> usize {
        self.context_echo_bytes
    }

    pub fn context_state(&self) -> &ContextEchoState {
        &self.context_echo_state
    }

    pub fn timing(&self) -> Option<&RamaGenerationTiming> {
        self.timing.as_ref()
    }
}

pub type RamaContextState = ContextEchoState;
pub type StreamingRamaTransformerConfig = StreamingEchoTransformerConfig;
pub type StreamingRamaTransformerTensorNames<'a> = StreamingEchoTransformerTensorNames<'a>;
pub type StreamingRamaTransformerParameters<'a> = StreamingEchoTransformerParameters<'a>;
pub type StreamingRamaGenerationResult = StreamingEchoGenerationResult;

/// Multi-layer token-ID RAMA loop over chunked `.spsa` weights.
///
/// This is the first all-layer RAMA orchestration primitive: callers still
/// provide token IDs and model-specific norm/bias parameters, but each layer owns
/// its own context-memory KV-cache and the stack can prefill/decode/generate across
/// all configured layers.
pub fn streaming_echo_transformer_prefill_from_model(
    model: &mut LazySpissaModel,
    prompt_token_ids: &[usize],
    names: StreamingEchoTransformerTensorNames<'_>,
    params: StreamingEchoTransformerParameters<'_>,
    config: StreamingEchoTransformerConfig,
    state: &mut ContextEchoState,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    validate_generation_inputs(prompt_token_ids, names, params, config, state)?;
    for layer_idx in 0..config.num_layers {
        let len = state.cache_len(layer_idx)?;
        if len != 0 {
            return Err(RuntimeError::InvalidTensorData(format!(
                "prefill requires empty context echo cache at layer {layer_idx}, got len {len}"
            )));
        }
    }
    streaming_echo_transformer_step_from_model(
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

pub fn streaming_echo_transformer_decode_step_from_model(
    model: &mut LazySpissaModel,
    token_id: usize,
    position_offset: usize,
    names: StreamingEchoTransformerTensorNames<'_>,
    params: StreamingEchoTransformerParameters<'_>,
    config: StreamingEchoTransformerConfig,
    state: &mut ContextEchoState,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    validate_generation_config(config)?;
    validate_layer_contract(names, params, config)?;
    validate_token_ids(&[token_id], config)?;
    validate_context_echo_state(config, state)?;
    if position_offset >= config.max_seq_len {
        return Err(RuntimeError::Shape(format!(
            "decode position_offset {position_offset} must be less than max_seq_len {}",
            config.max_seq_len
        )));
    }
    for layer_idx in 0..config.num_layers {
        let len = state.cache_len(layer_idx)?;
        if len != position_offset {
            return Err(RuntimeError::Shape(format!(
                "decode position_offset {position_offset} must match context echo cache len {len} at layer {layer_idx}"
            )));
        }
    }
    streaming_echo_transformer_step_from_model(
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

pub fn streaming_echo_transformer_generate_from_model(
    model: &mut LazySpissaModel,
    prompt_token_ids: &[usize],
    names: StreamingEchoTransformerTensorNames<'_>,
    params: StreamingEchoTransformerParameters<'_>,
    config: StreamingEchoTransformerConfig,
    budget: &mut MemoryBudget,
) -> Result<StreamingEchoGenerationResult> {
    let mut state = ContextEchoState::new(
        config.num_layers,
        config.num_heads,
        config.head_dim,
        config.max_seq_len,
    )?;
    validate_generation_inputs(prompt_token_ids, names, params, config, &state)?;

    let mut token_ids = prompt_token_ids.to_vec();
    let mut generated_token_ids = Vec::with_capacity(config.max_new_tokens);
    let mut step_logits = Vec::with_capacity(config.max_new_tokens);

    let prefill = streaming_echo_transformer_prefill_from_model(
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
        let step = streaming_echo_transformer_decode_step_from_model(
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
    Ok(StreamingEchoGenerationResult {
        token_ids,
        generated_token_ids,
        step_logits,
        context_echo_state: state,
        context_echo_bytes,
        timing: None,
    })
}

pub fn streaming_rama_transformer_prefill_from_model(
    model: &mut LazySpissaModel,
    prompt_token_ids: &[usize],
    names: StreamingRamaTransformerTensorNames<'_>,
    params: StreamingRamaTransformerParameters<'_>,
    config: StreamingRamaTransformerConfig,
    state: &mut RamaContextState,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    streaming_echo_transformer_prefill_from_model(
        model,
        prompt_token_ids,
        names,
        params,
        config,
        state,
        budget,
    )
}

pub fn streaming_rama_transformer_decode_step_from_model(
    model: &mut LazySpissaModel,
    token_id: usize,
    position_offset: usize,
    names: StreamingRamaTransformerTensorNames<'_>,
    params: StreamingRamaTransformerParameters<'_>,
    config: StreamingRamaTransformerConfig,
    state: &mut RamaContextState,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    streaming_echo_transformer_decode_step_from_model(
        model,
        token_id,
        position_offset,
        names,
        params,
        config,
        state,
        budget,
    )
}

pub fn streaming_rama_transformer_generate_from_model(
    model: &mut LazySpissaModel,
    prompt_token_ids: &[usize],
    names: StreamingRamaTransformerTensorNames<'_>,
    params: StreamingRamaTransformerParameters<'_>,
    config: StreamingRamaTransformerConfig,
    budget: &mut MemoryBudget,
) -> Result<StreamingRamaGenerationResult> {
    streaming_echo_transformer_generate_from_model(
        model,
        prompt_token_ids,
        names,
        params,
        config,
        budget,
    )
}

fn streaming_echo_transformer_step_from_model(
    model: &mut LazySpissaModel,
    token_ids: &[usize],
    position_offset: usize,
    names: StreamingEchoTransformerTensorNames<'_>,
    params: StreamingEchoTransformerParameters<'_>,
    config: StreamingEchoTransformerConfig,
    state: &mut ContextEchoState,
    budget: &mut MemoryBudget,
) -> Result<StreamingNextTokenResult> {
    validate_step_inputs(token_ids, names, params, config, state)?;
    let seq_len = token_ids.len();
    let hidden_size = config.hidden_size()?;
    let sequence_bytes = activation_bytes(seq_len, hidden_size, "echo sequence activation")?;

    let mut current_label = "echo transformer embedding activation".to_string();
    budget.reserve(sequence_bytes, current_label.clone())?;
    let mut current_hidden = match streaming_embedding_lookup_from_model(
        model,
        names.embedding_weight,
        token_ids,
        config.embedding_config()?,
        budget,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(sequence_bytes, current_label)?;
            return Err(err);
        }
    };

    for layer_idx in 0..config.num_layers {
        let next_label = format!("echo transformer layer {layer_idx} output activation");
        if let Err(err) = budget.reserve(sequence_bytes, next_label.clone()) {
            budget.release(sequence_bytes, current_label)?;
            return Err(err);
        }
        let cache = match state.block_cache_mut(layer_idx) {
            Ok(cache) => cache,
            Err(err) => {
                budget.release(sequence_bytes, next_label)?;
                budget.release(sequence_bytes, current_label)?;
                return Err(err);
            }
        };
        let next_hidden = match streaming_transformer_block_with_runtime_from_model(
            model,
            &current_hidden,
            names.layers[layer_idx],
            params.layers[layer_idx],
            config.block_config(seq_len),
            StreamingBlockRuntime {
                attention: StreamingAttentionRuntime {
                    rotary: config.rotary_config(seq_len, position_offset),
                    kv_cache: Some(cache),
                },
                parallel_residual: config.use_parallel_residual,
            },
            budget,
        ) {
            Ok(values) => values,
            Err(err) => {
                budget.release(sequence_bytes, next_label)?;
                budget.release(sequence_bytes, current_label)?;
                return Err(err);
            }
        };
        drop(current_hidden);
        budget.release(sequence_bytes, current_label)?;
        current_hidden = next_hidden;
        current_label = next_label;
    }

    let final_norm_label = "echo transformer final layernorm activation".to_string();
    if let Err(err) = budget.reserve(sequence_bytes, final_norm_label.clone()) {
        budget.release(sequence_bytes, current_label)?;
        return Err(err);
    }
    let final_norm = match layer_norm(
        &current_hidden,
        params.final_layernorm_weight,
        params.final_layernorm_bias,
        seq_len,
        hidden_size,
        config.layer_norm_eps,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(sequence_bytes, final_norm_label)?;
            budget.release(sequence_bytes, current_label)?;
            return Err(err);
        }
    };
    drop(current_hidden);
    budget.release(sequence_bytes, current_label)?;

    let last_hidden_start = (seq_len - 1)
        .checked_mul(hidden_size)
        .ok_or_else(|| RuntimeError::Shape("echo last hidden offset overflow".to_string()))?;
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

fn validate_generation_inputs(
    prompt_token_ids: &[usize],
    names: StreamingEchoTransformerTensorNames<'_>,
    params: StreamingEchoTransformerParameters<'_>,
    config: StreamingEchoTransformerConfig,
    state: &ContextEchoState,
) -> Result<()> {
    validate_step_inputs(prompt_token_ids, names, params, config, state)?;
    if config.max_new_tokens == 0 {
        return Err(RuntimeError::Shape(
            "max_new_tokens must be greater than zero".to_string(),
        ));
    }
    let required_sequence = prompt_token_ids
        .len()
        .checked_add(config.max_new_tokens)
        .ok_or_else(|| {
            RuntimeError::Shape("echo generation sequence length overflow".to_string())
        })?;
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

fn validate_step_inputs(
    token_ids: &[usize],
    names: StreamingEchoTransformerTensorNames<'_>,
    params: StreamingEchoTransformerParameters<'_>,
    config: StreamingEchoTransformerConfig,
    state: &ContextEchoState,
) -> Result<()> {
    validate_generation_config(config)?;
    validate_layer_contract(names, params, config)?;
    validate_token_ids(token_ids, config)?;
    validate_context_echo_state(config, state)?;
    validate_final_params(params, config)?;
    Ok(())
}

fn validate_generation_config(config: StreamingEchoTransformerConfig) -> Result<()> {
    if config.num_layers == 0
        || config.max_seq_len == 0
        || config.vocab_size == 0
        || config.num_heads == 0
        || config.head_dim == 0
        || config.intermediate_size == 0
    {
        return Err(RuntimeError::Shape(format!(
            "echo dimensions must be non-zero: num_layers={}, max_seq_len={}, vocab_size={}, num_heads={}, head_dim={}, intermediate_size={}",
            config.num_layers,
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

fn validate_layer_contract(
    names: StreamingEchoTransformerTensorNames<'_>,
    params: StreamingEchoTransformerParameters<'_>,
    config: StreamingEchoTransformerConfig,
) -> Result<()> {
    if names.layers.len() != config.num_layers || params.layers.len() != config.num_layers {
        return Err(RuntimeError::Shape(format!(
            "echo layer contract mismatch: config.num_layers={}, names.layers={}, params.layers={}",
            config.num_layers,
            names.layers.len(),
            params.layers.len()
        )));
    }
    validate_final_params(params, config)
}

fn validate_final_params(
    params: StreamingEchoTransformerParameters<'_>,
    config: StreamingEchoTransformerConfig,
) -> Result<()> {
    let hidden_size = config.hidden_size()?;
    if params.final_layernorm_weight.len() != hidden_size
        || params.final_layernorm_bias.len() != hidden_size
    {
        return Err(RuntimeError::Shape(format!(
            "echo final layernorm params must match hidden_size {hidden_size}: weight={}, bias={}",
            params.final_layernorm_weight.len(),
            params.final_layernorm_bias.len()
        )));
    }
    if let Some(bias) = params.lm_head_bias {
        if bias.len() != config.vocab_size {
            return Err(RuntimeError::Shape(format!(
                "echo LM head bias len {} does not match vocab_size {}",
                bias.len(),
                config.vocab_size
            )));
        }
    }
    Ok(())
}

fn validate_token_ids(token_ids: &[usize], config: StreamingEchoTransformerConfig) -> Result<()> {
    if token_ids.is_empty() {
        return Err(RuntimeError::Shape(
            "echo token input must contain at least one token".to_string(),
        ));
    }
    if token_ids.len() > config.max_seq_len {
        return Err(RuntimeError::Shape(format!(
            "token_ids len {} exceeds max_seq_len {}",
            token_ids.len(),
            config.max_seq_len
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

fn validate_context_echo_state(
    config: StreamingEchoTransformerConfig,
    state: &ContextEchoState,
) -> Result<()> {
    if state.layer_count() != config.num_layers {
        return Err(RuntimeError::Shape(format!(
            "echo generation expects {} context echo layers, got {}",
            config.num_layers,
            state.layer_count()
        )));
    }
    for layer_idx in 0..config.num_layers {
        let (num_heads, head_dim, max_seq_len) = state.cache_shape(layer_idx)?;
        if num_heads != config.num_heads || head_dim != config.head_dim {
            return Err(RuntimeError::Shape(format!(
                "context echo layer {layer_idx} cache shape {num_heads}/{head_dim} does not match generation config {}/{}",
                config.num_heads, config.head_dim
            )));
        }
        if max_seq_len != config.max_seq_len {
            return Err(RuntimeError::Shape(format!(
                "context echo layer {layer_idx} max_seq_len {max_seq_len} does not match generation config {}",
                config.max_seq_len
            )));
        }
    }
    Ok(())
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
        ContextEchoState, LazySpissaModel, MemoryBudget, StreamingBlockParameters,
        StreamingBlockTensorNames, StreamingSamplingConfig, StreamingTinyRotaryConfig,
    };
    use sha2::{Digest, Sha256};
    use spissa_container::{DType, GlobalMetadata, SpissaWriter, TensorMeta};

    const NUM_LAYERS: usize = 2;
    const VOCAB_SIZE: usize = 3;
    const HIDDEN_SIZE: usize = 2;
    const NUM_HEADS: usize = 1;
    const HEAD_DIM: usize = 2;
    const INTERMEDIATE_SIZE: usize = 3;

    const EMBEDDING_WEIGHT: [f32; 6] = [0.5, -1.0, 1.25, 0.75, -0.5, 0.25];
    const QKV_WEIGHT_L0: [f32; 12] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
    const QKV_WEIGHT_L1: [f32; 12] = [
        0.75, -0.25, 0.5, 1.25, -1.0, 0.5, 0.25, 0.75, 1.5, -0.5, -0.75, 1.0,
    ];
    const ATTENTION_OUT_WEIGHT_L0: [f32; 4] = [1.0, 0.5, -0.25, 1.0];
    const ATTENTION_OUT_WEIGHT_L1: [f32; 4] = [0.6, -0.4, 0.8, 0.9];
    const MLP_IN_WEIGHT_L0: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 1.0, -1.0];
    const MLP_IN_WEIGHT_L1: [f32; 6] = [0.5, 0.25, -0.75, 1.0, 1.25, 0.5];
    const MLP_OUT_WEIGHT_L0: [f32; 6] = [1.0, 2.0, 3.0, -1.0, 0.5, 0.25];
    const MLP_OUT_WEIGHT_L1: [f32; 6] = [0.75, -0.5, 1.0, 0.25, 1.5, -0.25];
    const LM_HEAD_WEIGHT: [f32; 6] = [1.0, 0.0, 0.0, 1.0, -0.5, 0.75];

    const LN1_WEIGHT_L0: [f32; 2] = [1.1, 0.9];
    const LN1_BIAS_L0: [f32; 2] = [0.05, -0.05];
    const QKV_BIAS_L0: [f32; 6] = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2];
    const ATTENTION_OUT_BIAS_L0: [f32; 2] = [0.05, -0.05];
    const LN2_WEIGHT_L0: [f32; 2] = [0.8, 1.2];
    const LN2_BIAS_L0: [f32; 2] = [-0.02, 0.04];
    const MLP_IN_BIAS_L0: [f32; 3] = [0.1, -0.2, 0.3];
    const MLP_OUT_BIAS_L0: [f32; 2] = [0.05, -0.05];

    const LN1_WEIGHT_L1: [f32; 2] = [0.95, 1.05];
    const LN1_BIAS_L1: [f32; 2] = [-0.03, 0.07];
    const QKV_BIAS_L1: [f32; 6] = [-0.05, 0.15, 0.2, -0.1, 0.05, -0.25];
    const ATTENTION_OUT_BIAS_L1: [f32; 2] = [-0.02, 0.03];
    const LN2_WEIGHT_L1: [f32; 2] = [1.15, 0.85];
    const LN2_BIAS_L1: [f32; 2] = [0.01, -0.06];
    const MLP_IN_BIAS_L1: [f32; 3] = [-0.15, 0.05, 0.2];
    const MLP_OUT_BIAS_L1: [f32; 2] = [-0.04, 0.08];

    const FINAL_LN_WEIGHT: [f32; 2] = [1.0, 1.0];
    const FINAL_LN_BIAS: [f32; 2] = [0.0, 0.0];
    const LM_HEAD_BIAS: [f32; 3] = [0.01, 0.02, -0.01];

    #[test]
    fn rama_public_aliases_cover_legacy_echo_generation_surface() {
        let state = RamaContextState::new(1, NUM_HEADS, HEAD_DIM, 8).unwrap();
        let config: StreamingRamaTransformerConfig = StreamingEchoTransformerConfig {
            num_layers: 1,
            max_new_tokens: 1,
            max_seq_len: 8,
            vocab_size: VOCAB_SIZE,
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: INTERMEDIATE_SIZE,
            causal: true,
            layer_norm_eps: 1e-5,
            use_parallel_residual: false,
            sampling: StreamingSamplingConfig::Argmax,
            rotary: None,
        };
        let _names: Option<StreamingRamaTransformerTensorNames<'_>> = None;
        let _params: Option<StreamingRamaTransformerParameters<'_>> = None;
        let _result: Option<StreamingRamaGenerationResult> = None;
        assert_eq!(state.layer_count(), config.num_layers);
        assert_eq!(state.cache_len(0).unwrap(), 0);
        assert_eq!(state.resident_bytes(), 0);
    }

    struct FullBlockRefs<'a> {
        qkv_weight: &'a [f32],
        attention_out_weight: &'a [f32],
        mlp_in_weight: &'a [f32],
        mlp_out_weight: &'a [f32],
        input_layernorm_weight: &'a [f32],
        input_layernorm_bias: &'a [f32],
        qkv_bias: &'a [f32],
        attention_out_bias: &'a [f32],
        post_attention_layernorm_weight: &'a [f32],
        post_attention_layernorm_bias: &'a [f32],
        mlp_in_bias: &'a [f32],
        mlp_out_bias: &'a [f32],
    }

    fn full_block_refs(layer_idx: usize) -> FullBlockRefs<'static> {
        match layer_idx {
            0 => FullBlockRefs {
                qkv_weight: &QKV_WEIGHT_L0,
                attention_out_weight: &ATTENTION_OUT_WEIGHT_L0,
                mlp_in_weight: &MLP_IN_WEIGHT_L0,
                mlp_out_weight: &MLP_OUT_WEIGHT_L0,
                input_layernorm_weight: &LN1_WEIGHT_L0,
                input_layernorm_bias: &LN1_BIAS_L0,
                qkv_bias: &QKV_BIAS_L0,
                attention_out_bias: &ATTENTION_OUT_BIAS_L0,
                post_attention_layernorm_weight: &LN2_WEIGHT_L0,
                post_attention_layernorm_bias: &LN2_BIAS_L0,
                mlp_in_bias: &MLP_IN_BIAS_L0,
                mlp_out_bias: &MLP_OUT_BIAS_L0,
            },
            1 => FullBlockRefs {
                qkv_weight: &QKV_WEIGHT_L1,
                attention_out_weight: &ATTENTION_OUT_WEIGHT_L1,
                mlp_in_weight: &MLP_IN_WEIGHT_L1,
                mlp_out_weight: &MLP_OUT_WEIGHT_L1,
                input_layernorm_weight: &LN1_WEIGHT_L1,
                input_layernorm_bias: &LN1_BIAS_L1,
                qkv_bias: &QKV_BIAS_L1,
                attention_out_bias: &ATTENTION_OUT_BIAS_L1,
                post_attention_layernorm_weight: &LN2_WEIGHT_L1,
                post_attention_layernorm_bias: &LN2_BIAS_L1,
                mlp_in_bias: &MLP_IN_BIAS_L1,
                mlp_out_bias: &MLP_OUT_BIAS_L1,
            },
            _ => panic!("unexpected layer {layer_idx}"),
        }
    }

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
        std::env::temp_dir().join(format!("rllm-echo-{name}-{}.spsa", std::process::id()))
    }

    fn add_f32_tensor(
        writer: &mut SpissaWriter,
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

    fn write_echo_stack_model(path: &std::path::Path) {
        let mut writer = SpissaWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "embed.weight",
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
            &EMBEDDING_WEIGHT,
            16,
        );
        for layer_idx in 0..NUM_LAYERS {
            let refs = full_block_refs(layer_idx);
            let base_id = 1 + (layer_idx as u64 * 4);
            add_f32_tensor(
                &mut writer,
                base_id,
                &format!("gpt_neox.layers.{layer_idx}.attention.query_key_value.weight"),
                vec![(3 * HIDDEN_SIZE) as u64, HIDDEN_SIZE as u64],
                refs.qkv_weight,
                20,
            );
            add_f32_tensor(
                &mut writer,
                base_id + 1,
                &format!("gpt_neox.layers.{layer_idx}.attention.dense.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
                refs.attention_out_weight,
                8,
            );
            add_f32_tensor(
                &mut writer,
                base_id + 2,
                &format!("gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h.weight"),
                vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
                refs.mlp_in_weight,
                12,
            );
            add_f32_tensor(
                &mut writer,
                base_id + 3,
                &format!("gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight"),
                vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
                refs.mlp_out_weight,
                16,
            );
        }
        add_f32_tensor(
            &mut writer,
            9,
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
        let mut q = vec![0.0f32; seq_len * HIDDEN_SIZE];
        let mut k = vec![0.0f32; seq_len * HIDDEN_SIZE];
        let mut v = vec![0.0f32; seq_len * HIDDEN_SIZE];
        for pos in 0..seq_len {
            let fused_row = pos * HIDDEN_SIZE * 3;
            let out_row = pos * HIDDEN_SIZE;
            q[out_row..out_row + HIDDEN_SIZE]
                .copy_from_slice(&fused[fused_row..fused_row + HIDDEN_SIZE]);
            k[out_row..out_row + HIDDEN_SIZE]
                .copy_from_slice(&fused[fused_row + HIDDEN_SIZE..fused_row + 2 * HIDDEN_SIZE]);
            v[out_row..out_row + HIDDEN_SIZE]
                .copy_from_slice(&fused[fused_row + 2 * HIDDEN_SIZE..fused_row + 3 * HIDDEN_SIZE]);
        }
        (q, k, v)
    }

    fn full_decode_block_baseline_with_rotary(
        input: &[f32],
        seq_len: usize,
        layer_idx: usize,
        rotary: Option<StreamingTinyRotaryConfig>,
    ) -> Vec<f32> {
        let refs = full_block_refs(layer_idx);
        let attention_input = layer_norm(
            input,
            refs.input_layernorm_weight,
            refs.input_layernorm_bias,
            seq_len,
            HIDDEN_SIZE,
            1e-5,
        )
        .unwrap();
        let fused = linear(
            &attention_input,
            refs.qkv_weight,
            Some(refs.qkv_bias),
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
            refs.attention_out_weight,
            Some(refs.attention_out_bias),
            seq_len,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )
        .unwrap();
        let mut residual = input.to_vec();
        crate::add_inplace(&mut residual, &attention_out).unwrap();

        let mlp_input = layer_norm(
            &residual,
            refs.post_attention_layernorm_weight,
            refs.post_attention_layernorm_bias,
            seq_len,
            HIDDEN_SIZE,
            1e-5,
        )
        .unwrap();
        let mlp_out = mlp(
            &mlp_input,
            refs.mlp_in_weight,
            Some(refs.mlp_in_bias),
            refs.mlp_out_weight,
            Some(refs.mlp_out_bias),
            seq_len,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
        )
        .unwrap();
        crate::add_inplace(&mut residual, &mlp_out).unwrap();
        residual
    }

    fn full_decode_stack_next_token_baseline(
        token_ids: &[usize],
        rotary: Option<StreamingTinyRotaryConfig>,
    ) -> (Vec<f32>, usize) {
        let mut hidden =
            embedding_lookup(&EMBEDDING_WEIGHT, VOCAB_SIZE, HIDDEN_SIZE, token_ids).unwrap();
        for layer_idx in 0..NUM_LAYERS {
            hidden =
                full_decode_block_baseline_with_rotary(&hidden, token_ids.len(), layer_idx, rotary);
        }
        let final_norm = layer_norm(
            &hidden,
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

    fn full_decode_stack_generate(
        prompt_token_ids: &[usize],
        max_new_tokens: usize,
        rotary: Option<StreamingTinyRotaryConfig>,
    ) -> (Vec<usize>, Vec<Vec<f32>>) {
        let mut token_ids = prompt_token_ids.to_vec();
        let mut step_logits = Vec::new();
        for _ in 0..max_new_tokens {
            let (logits, token_id) = full_decode_stack_next_token_baseline(&token_ids, rotary);
            step_logits.push(logits);
            token_ids.push(token_id);
        }
        (token_ids[prompt_token_ids.len()..].to_vec(), step_logits)
    }

    fn block_names_for_test<'a>() -> [StreamingBlockTensorNames<'a>; NUM_LAYERS] {
        [
            StreamingBlockTensorNames {
                qkv_weight: "gpt_neox.layers.0.attention.query_key_value.weight",
                attention_out_weight: "gpt_neox.layers.0.attention.dense.weight",
                mlp_in_weight: "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
                mlp_out_weight: "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
            },
            StreamingBlockTensorNames {
                qkv_weight: "gpt_neox.layers.1.attention.query_key_value.weight",
                attention_out_weight: "gpt_neox.layers.1.attention.dense.weight",
                mlp_in_weight: "gpt_neox.layers.1.mlp.dense_h_to_4h.weight",
                mlp_out_weight: "gpt_neox.layers.1.mlp.dense_4h_to_h.weight",
            },
        ]
    }

    fn block_params_for_test<'a>() -> [StreamingBlockParameters<'a>; NUM_LAYERS] {
        [
            StreamingBlockParameters {
                input_layernorm_weight: &LN1_WEIGHT_L0,
                input_layernorm_bias: &LN1_BIAS_L0,
                qkv_bias: Some(&QKV_BIAS_L0),
                attention_out_bias: Some(&ATTENTION_OUT_BIAS_L0),
                post_attention_layernorm_weight: &LN2_WEIGHT_L0,
                post_attention_layernorm_bias: &LN2_BIAS_L0,
                mlp_in_bias: Some(&MLP_IN_BIAS_L0),
                mlp_out_bias: Some(&MLP_OUT_BIAS_L0),
            },
            StreamingBlockParameters {
                input_layernorm_weight: &LN1_WEIGHT_L1,
                input_layernorm_bias: &LN1_BIAS_L1,
                qkv_bias: Some(&QKV_BIAS_L1),
                attention_out_bias: Some(&ATTENTION_OUT_BIAS_L1),
                post_attention_layernorm_weight: &LN2_WEIGHT_L1,
                post_attention_layernorm_bias: &LN2_BIAS_L1,
                mlp_in_bias: Some(&MLP_IN_BIAS_L1),
                mlp_out_bias: Some(&MLP_OUT_BIAS_L1),
            },
        ]
    }

    fn echo_names_for_test<'a>(
        layers: &'a [StreamingBlockTensorNames<'a>],
    ) -> StreamingEchoTransformerTensorNames<'a> {
        StreamingEchoTransformerTensorNames {
            embedding_weight: "embed.weight",
            layers,
            lm_head_weight: "lm_head.weight",
        }
    }

    fn echo_params_for_test<'a>(
        layers: &'a [StreamingBlockParameters<'a>],
    ) -> StreamingEchoTransformerParameters<'a> {
        StreamingEchoTransformerParameters {
            layers,
            final_layernorm_weight: &FINAL_LN_WEIGHT,
            final_layernorm_bias: &FINAL_LN_BIAS,
            lm_head_bias: Some(&LM_HEAD_BIAS),
        }
    }

    fn echo_config_for_test(max_new_tokens: usize) -> StreamingEchoTransformerConfig {
        StreamingEchoTransformerConfig {
            num_layers: NUM_LAYERS,
            max_new_tokens,
            max_seq_len: 8,
            vocab_size: VOCAB_SIZE,
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: INTERMEDIATE_SIZE,
            causal: true,
            layer_norm_eps: 1e-5,
            use_parallel_residual: false,
            sampling: StreamingSamplingConfig::Argmax,
            rotary: Some(StreamingTinyRotaryConfig {
                rotary_dim: HEAD_DIM,
                base: 10_000.0,
            }),
        }
    }

    #[test]
    fn echo_stack_prefill_and_decode_step_match_full_context_per_layer_cache() {
        let path = temp_path("prefill-decode");
        write_echo_stack_model(&path);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let names = block_names_for_test();
        let params = block_params_for_test();
        let prompt = [0, 2];
        let config = echo_config_for_test(2);
        let mut state =
            ContextEchoState::new(NUM_LAYERS, NUM_HEADS, HEAD_DIM, config.max_seq_len).unwrap();
        let mut budget = MemoryBudget::new(8192);

        let prefill = streaming_echo_transformer_prefill_from_model(
            &mut model,
            &prompt,
            echo_names_for_test(&names),
            echo_params_for_test(&params),
            config,
            &mut state,
            &mut budget,
        )
        .unwrap();
        let (expected_prefill_logits, expected_prefill_token) =
            full_decode_stack_next_token_baseline(&prompt, config.rotary);

        assert_close_vec(&prefill.logits, &expected_prefill_logits, 1e-5);
        assert_eq!(prefill.token_id, expected_prefill_token);
        for layer_idx in 0..NUM_LAYERS {
            assert_eq!(state.cache_len(layer_idx).unwrap(), prompt.len());
        }
        assert_eq!(budget.current_bytes(), 0);

        let mut context = prompt.to_vec();
        context.push(prefill.token_id);
        let decode = streaming_echo_transformer_decode_step_from_model(
            &mut model,
            prefill.token_id,
            prompt.len(),
            echo_names_for_test(&names),
            echo_params_for_test(&params),
            config,
            &mut state,
            &mut budget,
        )
        .unwrap();
        let (expected_decode_logits, expected_decode_token) =
            full_decode_stack_next_token_baseline(&context, config.rotary);

        assert_close_vec(&decode.logits, &expected_decode_logits, 1e-5);
        assert_eq!(decode.token_id, expected_decode_token);
        for layer_idx in 0..NUM_LAYERS {
            assert_eq!(state.cache_len(layer_idx).unwrap(), prompt.len() + 1);
        }
        assert_eq!(
            state.resident_bytes(),
            NUM_LAYERS * (prompt.len() + 1) * HIDDEN_SIZE * 2 * std::mem::size_of::<f32>()
        );
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn echo_stack_generation_matches_full_context_recompute_across_layers() {
        let path = temp_path("generate");
        write_echo_stack_model(&path);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let names = block_names_for_test();
        let params = block_params_for_test();
        let prompt = [0, 2];
        let config = echo_config_for_test(3);
        let (expected_generated, expected_logits) =
            full_decode_stack_generate(&prompt, config.max_new_tokens, config.rotary);
        let mut budget = MemoryBudget::new(16384);

        let actual = streaming_echo_transformer_generate_from_model(
            &mut model,
            &prompt,
            echo_names_for_test(&names),
            echo_params_for_test(&params),
            config,
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
        for layer_idx in 0..NUM_LAYERS {
            assert_eq!(
                actual.context_echo_state.cache_len(layer_idx).unwrap(),
                prompt.len() + config.max_new_tokens - 1
            );
        }
        assert_eq!(
            actual.context_echo_bytes,
            actual.context_echo_state.resident_bytes()
        );
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }
}
