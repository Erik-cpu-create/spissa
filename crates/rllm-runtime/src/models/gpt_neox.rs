use crate::RamaGenerationTiming;
use crate::{
    gpt_neox_rotary_dim, layer_norm, sample_argmax, sample_top_p,
    streaming_echo_transformer_generate_from_model, streaming_embedding_lookup_from_model,
    streaming_tile_linear_argmax_from_model, streaming_tile_linear_from_model,
    streaming_transformer_block_with_runtime_and_timing_from_model, LazyRllmModel, MemoryBudget,
    RamaContextState, Result, RllmTokenizer, RuntimeError, StreamingAttentionRuntime,
    StreamingBlockConfig, StreamingBlockParameters, StreamingBlockRuntime,
    StreamingBlockTensorNames, StreamingBlockTiming, StreamingEchoGenerationResult,
    StreamingEchoTransformerConfig, StreamingEchoTransformerParameters,
    StreamingEchoTransformerTensorNames, StreamingEmbeddingConfig, StreamingLinearConfig,
    StreamingNextTokenResult, StreamingRamaGenerationResult, StreamingSamplingConfig,
    StreamingTileLinearConfig, StreamingTinyRotaryConfig, DEFAULT_STREAMING_TILE_ELEMENTS,
};
use rllm_container::{GlobalMetadata, ModelConfigMetadata};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct GptNeoxEchoBuildConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: Option<usize>,
    pub num_heads: usize,
    pub rotary_pct: f32,
    pub rotary_base: f32,
    pub causal: bool,
    pub layer_norm_eps: f32,
    pub use_parallel_residual: bool,
    pub sampling: StreamingSamplingConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct GptNeoxEchoGenerationConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: Option<usize>,
    pub causal: bool,
    pub sampling: StreamingSamplingConfig,
}

pub type GptNeoxRamaBuildConfig = GptNeoxEchoBuildConfig;
pub type GptNeoxRamaGenerationConfig = GptNeoxEchoGenerationConfig;
pub type PreparedGptNeoxRamaTransformer = PreparedGptNeoxEchoTransformer;

const RAMA_PREFILL_BASE_CHUNK_TOKENS: usize = 32;
const RAMA_PREFILL_BASE_HIDDEN_SIZE: usize = 512;
const RAMA_PREFILL_BASE_LAYER_COUNT: usize = 6;
const RAMA_PREFILL_LOW_RAM_MAX_CHUNK_TOKENS: usize = 128;
const RAMA_PREFILL_SPEED_MAX_CHUNK_TOKENS: usize = 256;
const RAMA_PREFILL_ESTIMATE_HIDDEN_MULTIPLIER: usize = 7;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RamaPrefillPolicy {
    /// Prefer the measured low-RAM-safe window for the model shape.
    #[default]
    LowRam,
    /// Prefer a larger speed-biased window, still downshifted by an explicit transient budget.
    Speed,
}

impl RamaPrefillPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LowRam => "low-ram",
            Self::Speed => "speed",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GptNeoxRamaGenerationOptions {
    /// Collect low-overhead aggregate phase timings without buffering per-chunk trace events.
    pub timing: bool,
    /// Optional maximum number of prompt tokens processed per prefill recall window.
    ///
    /// This keeps active activations bounded for real long prompts. Intermediate
    /// chunks update layer KV caches and deliberately skip final-norm/lm-head
    /// projection because only the last prompt token produces the first generated
    /// token.
    pub prefill_chunk_tokens: Option<usize>,
    /// Preserve full per-step logits for parity/debug output.
    ///
    /// CLI argmax generation can set this false to stream the lm-head argmax
    /// without materializing/storing full vocabulary logits. Top-p sampling and
    /// `--logits-out` callers still require full logits.
    pub collect_logits: bool,
}

impl Default for GptNeoxRamaGenerationOptions {
    fn default() -> Self {
        Self {
            timing: false,
            prefill_chunk_tokens: None,
            collect_logits: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OwnedStreamingBlockTensorNames {
    pub qkv_weight: String,
    pub attention_out_weight: String,
    pub mlp_in_weight: String,
    pub mlp_out_weight: String,
}

impl OwnedStreamingBlockTensorNames {
    fn as_borrowed(&self) -> StreamingBlockTensorNames<'_> {
        StreamingBlockTensorNames {
            qkv_weight: self.qkv_weight.as_str(),
            attention_out_weight: self.attention_out_weight.as_str(),
            mlp_in_weight: self.mlp_in_weight.as_str(),
            mlp_out_weight: self.mlp_out_weight.as_str(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OwnedStreamingBlockParameters {
    pub input_layernorm_weight: Vec<f32>,
    pub input_layernorm_bias: Vec<f32>,
    pub qkv_bias: Option<Vec<f32>>,
    pub attention_out_bias: Option<Vec<f32>>,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_bias: Vec<f32>,
    pub mlp_in_bias: Option<Vec<f32>>,
    pub mlp_out_bias: Option<Vec<f32>>,
}

impl OwnedStreamingBlockParameters {
    fn as_borrowed(&self) -> StreamingBlockParameters<'_> {
        StreamingBlockParameters {
            input_layernorm_weight: &self.input_layernorm_weight,
            input_layernorm_bias: &self.input_layernorm_bias,
            qkv_bias: self.qkv_bias.as_deref(),
            attention_out_bias: self.attention_out_bias.as_deref(),
            post_attention_layernorm_weight: &self.post_attention_layernorm_weight,
            post_attention_layernorm_bias: &self.post_attention_layernorm_bias,
            mlp_in_bias: self.mlp_in_bias.as_deref(),
            mlp_out_bias: self.mlp_out_bias.as_deref(),
        }
    }

    fn resident_bytes(&self) -> usize {
        let required = self.input_layernorm_weight.len()
            + self.input_layernorm_bias.len()
            + self.post_attention_layernorm_weight.len()
            + self.post_attention_layernorm_bias.len();
        let optional = self.qkv_bias.as_ref().map(Vec::len).unwrap_or(0)
            + self.attention_out_bias.as_ref().map(Vec::len).unwrap_or(0)
            + self.mlp_in_bias.as_ref().map(Vec::len).unwrap_or(0)
            + self.mlp_out_bias.as_ref().map(Vec::len).unwrap_or(0);
        required
            .saturating_add(optional)
            .saturating_mul(std::mem::size_of::<f32>())
    }
}

#[derive(Debug, Clone)]
pub struct PreparedGptNeoxEchoTransformer {
    pub config: StreamingEchoTransformerConfig,
    pub embedding_weight: String,
    pub layers: Vec<OwnedStreamingBlockTensorNames>,
    pub lm_head_weight: String,
    pub layer_params: Vec<OwnedStreamingBlockParameters>,
    pub final_layernorm_weight: Vec<f32>,
    pub final_layernorm_bias: Vec<f32>,
    pub lm_head_bias: Option<Vec<f32>>,
    pub resident_parameter_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct LayerDecodedGptNeoxRamaTransformer {
    pub config: StreamingEchoTransformerConfig,
    pub embedding_weight: String,
    pub layers: Vec<OwnedStreamingBlockTensorNames>,
    pub lm_head_weight: String,
    pub final_layernorm_weight: Vec<f32>,
    pub final_layernorm_bias: Vec<f32>,
    pub lm_head_bias: Option<Vec<f32>>,
    pub pinned_lm_head_weight: Option<Vec<f32>>,
    pub resident_parameter_bytes: usize,
    pub max_layer_parameter_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct GptNeoxTextGenerationResult {
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub token_ids: Vec<usize>,
    pub text: String,
    pub generated_text: String,
    pub context_echo_bytes: usize,
}

impl GptNeoxTextGenerationResult {
    pub fn context_memory_bytes(&self) -> usize {
        self.context_echo_bytes
    }
}

impl PreparedGptNeoxEchoTransformer {
    pub fn generate_from_model(
        &self,
        model: &mut LazyRllmModel,
        prompt_token_ids: &[usize],
        budget: &mut MemoryBudget,
    ) -> Result<StreamingEchoGenerationResult> {
        let layer_names: Vec<StreamingBlockTensorNames<'_>> = self
            .layers
            .iter()
            .map(|names| names.as_borrowed())
            .collect();
        let layer_params: Vec<StreamingBlockParameters<'_>> = self
            .layer_params
            .iter()
            .map(|params| params.as_borrowed())
            .collect();
        streaming_echo_transformer_generate_from_model(
            model,
            prompt_token_ids,
            StreamingEchoTransformerTensorNames {
                embedding_weight: self.embedding_weight.as_str(),
                layers: &layer_names,
                lm_head_weight: self.lm_head_weight.as_str(),
            },
            StreamingEchoTransformerParameters {
                layers: &layer_params,
                final_layernorm_weight: &self.final_layernorm_weight,
                final_layernorm_bias: &self.final_layernorm_bias,
                lm_head_bias: self.lm_head_bias.as_deref(),
            },
            self.config,
            budget,
        )
    }

    pub fn generate_text_from_model(
        &self,
        model: &mut LazyRllmModel,
        tokenizer: &RllmTokenizer,
        prompt_text: &str,
        budget: &mut MemoryBudget,
    ) -> Result<GptNeoxTextGenerationResult> {
        let prompt_token_ids = tokenizer.encode(prompt_text)?;
        let generation = self.generate_from_model(model, &prompt_token_ids, budget)?;
        let text = tokenizer.decode(&generation.token_ids)?;
        let generated_text = tokenizer.decode(&generation.generated_token_ids)?;
        Ok(GptNeoxTextGenerationResult {
            prompt_token_ids,
            generated_token_ids: generation.generated_token_ids,
            token_ids: generation.token_ids,
            text,
            generated_text,
            context_echo_bytes: generation.context_echo_bytes,
        })
    }
}

impl LayerDecodedGptNeoxRamaTransformer {
    pub fn pin_lm_head(&mut self, model: &mut LazyRllmModel, budget: &mut MemoryBudget) {
        if self.pinned_lm_head_weight.is_some() {
            return;
        }
        if let Ok(tensor) = model.tensor(&self.lm_head_weight) {
            if let Ok(runtime_bytes) = crate::lazy::runtime_f32_bytes_for_tensor(tensor) {
                if budget
                    .reserve(runtime_bytes, format!("pinned {}", self.lm_head_weight))
                    .is_ok()
                {
                    let expected_shape = [
                        self.config.vocab_size,
                        self.config.head_dim * self.config.num_heads,
                    ];
                    if let Ok(data) =
                        decode_matrix_tensor(model, &self.lm_head_weight, &expected_shape)
                    {
                        self.pinned_lm_head_weight = Some(data);
                    } else {
                        budget
                            .release(runtime_bytes, format!("pinned {}", self.lm_head_weight))
                            .ok();
                    }
                }
            }
        }
    }

    pub fn generate_from_model(
        &self,
        model: &mut LazyRllmModel,
        prompt_token_ids: &[usize],
        budget: &mut MemoryBudget,
    ) -> Result<StreamingRamaGenerationResult> {
        self.generate_from_model_with_options(
            model,
            prompt_token_ids,
            budget,
            GptNeoxRamaGenerationOptions::default(),
        )
    }

    pub fn generate_from_model_with_options(
        &self,
        model: &mut LazyRllmModel,
        prompt_token_ids: &[usize],
        budget: &mut MemoryBudget,
        options: GptNeoxRamaGenerationOptions,
    ) -> Result<StreamingRamaGenerationResult> {
        let mut state = RamaContextState::new(
            self.config.num_layers,
            self.config.num_heads,
            self.config.head_dim,
            self.config.max_seq_len,
        )?;
        validate_layer_decode_generation_inputs(prompt_token_ids, self.config, &state)?;
        let prefill_chunk_tokens =
            validated_prefill_chunk_tokens(options.prefill_chunk_tokens, prompt_token_ids.len())?;
        let mut timing = options.timing.then(RamaGenerationTiming::default);

        let mut token_ids = prompt_token_ids.to_vec();
        let mut generated_token_ids = Vec::with_capacity(self.config.max_new_tokens);
        let mut step_logits = if options.collect_logits {
            Vec::with_capacity(self.config.max_new_tokens)
        } else {
            Vec::new()
        };

        let mut prefill_position = 0usize;
        let mut prefill = None;
        for chunk in prompt_token_ids.chunks(prefill_chunk_tokens) {
            let is_last_prompt_chunk = prefill_position + chunk.len() == prompt_token_ids.len();
            let started = Instant::now();
            let step = self.step_from_model_inner(
                model,
                chunk,
                prefill_position,
                &mut state,
                budget,
                is_last_prompt_chunk,
                options.collect_logits,
                true,
                timing.as_mut(),
            )?;
            if let Some(timing) = timing.as_mut() {
                timing.record_prefill_chunk(chunk.len(), elapsed_ns_u64(started.elapsed()));
            }
            if is_last_prompt_chunk {
                prefill = step;
            }
            prefill_position += chunk.len();
        }
        let prefill = prefill.ok_or_else(|| {
            RuntimeError::InvalidTensorData("RAMA prefill produced no next token".to_string())
        })?;
        generated_token_ids.push(prefill.token_id);
        token_ids.push(prefill.token_id);
        if options.collect_logits {
            step_logits.push(prefill.logits);
        }

        while generated_token_ids.len() < self.config.max_new_tokens {
            let input_token = *generated_token_ids.last().ok_or_else(|| {
                RuntimeError::InvalidTensorData("missing generated token".to_string())
            })?;
            let position_offset = token_ids.len() - 1;
            let started = Instant::now();
            let step = self
                .step_from_model_inner(
                    model,
                    &[input_token],
                    position_offset,
                    &mut state,
                    budget,
                    true,
                    options.collect_logits,
                    false,
                    timing.as_mut(),
                )?
                .ok_or_else(|| {
                    RuntimeError::InvalidTensorData(
                        "RAMA decode step produced no logits".to_string(),
                    )
                })?;
            if let Some(timing) = timing.as_mut() {
                timing.record_decode_step(elapsed_ns_u64(started.elapsed()));
            }
            generated_token_ids.push(step.token_id);
            token_ids.push(step.token_id);
            if options.collect_logits {
                step_logits.push(step.logits);
            }
        }

        let context_echo_bytes = state.resident_bytes();
        Ok(StreamingRamaGenerationResult {
            token_ids,
            generated_token_ids,
            step_logits,
            context_echo_state: state,
            context_echo_bytes,
            timing,
        })
    }

    pub fn generate_text_from_model(
        &self,
        model: &mut LazyRllmModel,
        tokenizer: &RllmTokenizer,
        prompt_text: &str,
        budget: &mut MemoryBudget,
    ) -> Result<GptNeoxTextGenerationResult> {
        let prompt_token_ids = tokenizer.encode(prompt_text)?;
        let generation = self.generate_from_model(model, &prompt_token_ids, budget)?;
        let text = tokenizer.decode(&generation.token_ids)?;
        let generated_text = tokenizer.decode(&generation.generated_token_ids)?;
        let context_echo_bytes = generation.context_memory_bytes();
        Ok(GptNeoxTextGenerationResult {
            prompt_token_ids,
            generated_token_ids: generation.generated_token_ids,
            token_ids: generation.token_ids,
            text,
            generated_text,
            context_echo_bytes,
        })
    }

    fn step_from_model_inner(
        &self,
        model: &mut LazyRllmModel,
        token_ids: &[usize],
        position_offset: usize,
        state: &mut RamaContextState,
        budget: &mut MemoryBudget,
        emit_logits: bool,
        collect_logits: bool,
        record_prefill_detail: bool,
        mut timing: Option<&mut RamaGenerationTiming>,
    ) -> Result<Option<StreamingNextTokenResult>> {
        validate_layer_decode_step_inputs(token_ids, position_offset, self.config, state)?;
        let seq_len = token_ids.len();
        let hidden_size = layer_decode_hidden_size(self.config)?;
        let sequence_bytes = activation_bytes(
            seq_len,
            hidden_size,
            "gpt-neox RAMA layer-decode sequence activation",
        )?;

        let mut current_label = "gpt-neox RAMA embedding activation".to_string();
        budget.reserve(sequence_bytes, current_label.clone())?;
        let embedding_started = Instant::now();
        let mut current_hidden = match streaming_embedding_lookup_from_model(
            model,
            &self.embedding_weight,
            token_ids,
            StreamingEmbeddingConfig {
                vocab_size: self.config.vocab_size,
                hidden_size,
            },
            budget,
        ) {
            Ok(values) => values,
            Err(err) => {
                budget.release(sequence_bytes, current_label)?;
                return Err(err);
            }
        };
        if record_prefill_detail {
            if let Some(timing) = timing.as_deref_mut() {
                timing.record_prefill_embedding(elapsed_ns_u64(embedding_started.elapsed()));
            }
        }

        for layer_idx in 0..self.config.num_layers {
            let next_label = format!("gpt-neox RAMA layer {layer_idx} output activation");
            if let Err(err) = budget.reserve(sequence_bytes, next_label.clone()) {
                budget.release(sequence_bytes, current_label)?;
                return Err(err);
            }

            let layer_params_started = Instant::now();
            let params = match self.decode_layer_params_from_model(model, layer_idx) {
                Ok(params) => params,
                Err(err) => {
                    budget.release(sequence_bytes, next_label)?;
                    budget.release(sequence_bytes, current_label)?;
                    return Err(err);
                }
            };
            if record_prefill_detail {
                if let Some(timing) = timing.as_deref_mut() {
                    timing.record_prefill_layer_params(elapsed_ns_u64(
                        layer_params_started.elapsed(),
                    ));
                }
            }
            let param_bytes = params.resident_bytes();
            let param_label = format!("gpt-neox RAMA layer {layer_idx} active params");
            if let Err(err) = budget.reserve(param_bytes, param_label.clone()) {
                budget.release(sequence_bytes, next_label)?;
                budget.release(sequence_bytes, current_label)?;
                return Err(err);
            }

            let cache = match state.block_cache_mut(layer_idx) {
                Ok(cache) => cache,
                Err(err) => {
                    budget.release(param_bytes, param_label)?;
                    budget.release(sequence_bytes, next_label)?;
                    budget.release(sequence_bytes, current_label)?;
                    return Err(err);
                }
            };
            let mut block_timing = StreamingBlockTiming::default();
            let block_timing_ref = if record_prefill_detail {
                Some(&mut block_timing)
            } else {
                None
            };
            let next_hidden = match streaming_transformer_block_with_runtime_and_timing_from_model(
                model,
                &current_hidden,
                self.layers[layer_idx].as_borrowed(),
                params.as_borrowed(),
                StreamingBlockConfig {
                    seq_len,
                    num_heads: self.config.num_heads,
                    head_dim: self.config.head_dim,
                    intermediate_size: self.config.intermediate_size,
                    causal: self.config.causal,
                    layer_norm_eps: self.config.layer_norm_eps,
                },
                StreamingBlockRuntime {
                    attention: StreamingAttentionRuntime {
                        rotary: layer_decode_rotary_config(self.config, seq_len, position_offset),
                        kv_cache: Some(cache),
                    },
                    parallel_residual: self.config.use_parallel_residual,
                },
                budget,
                block_timing_ref,
            ) {
                Ok(values) => values,
                Err(err) => {
                    budget.release(param_bytes, param_label)?;
                    budget.release(sequence_bytes, next_label)?;
                    budget.release(sequence_bytes, current_label)?;
                    return Err(err);
                }
            };
            if record_prefill_detail {
                if let Some(timing) = timing.as_deref_mut() {
                    timing.record_prefill_block_timing(
                        block_timing.attention_norm_ns,
                        block_timing.attention_ns,
                        block_timing.attention_qkv_projection_ns,
                        block_timing.attention_qkv_split_ns,
                        block_timing.attention_rotary_ns,
                        block_timing.attention_score_context_ns,
                        block_timing.attention_output_projection_ns,
                        block_timing.attention_kv_append_ns,
                        block_timing.attention_residual_ns,
                        block_timing.mlp_norm_ns,
                        block_timing.mlp_ns,
                        block_timing.mlp_input_projection_ns,
                        block_timing.mlp_activation_ns,
                        block_timing.mlp_output_projection_ns,
                        block_timing.mlp_residual_ns,
                    );
                }
            }
            budget.release(param_bytes, param_label)?;
            drop(params);
            drop(current_hidden);
            budget.release(sequence_bytes, current_label)?;
            current_hidden = next_hidden;
            current_label = next_label;
        }

        if !emit_logits {
            drop(current_hidden);
            budget.release(sequence_bytes, current_label)?;
            return Ok(None);
        }

        let final_norm_label = "gpt-neox RAMA final layernorm activation".to_string();
        if let Err(err) = budget.reserve(sequence_bytes, final_norm_label.clone()) {
            budget.release(sequence_bytes, current_label)?;
            return Err(err);
        }
        let final_norm_started = Instant::now();
        let final_norm = match layer_norm(
            &current_hidden,
            &self.final_layernorm_weight,
            &self.final_layernorm_bias,
            seq_len,
            hidden_size,
            self.config.layer_norm_eps,
        ) {
            Ok(values) => values,
            Err(err) => {
                budget.release(sequence_bytes, final_norm_label)?;
                budget.release(sequence_bytes, current_label)?;
                return Err(err);
            }
        };
        if let Some(timing) = timing.as_deref_mut() {
            timing.record_final_norm(elapsed_ns_u64(final_norm_started.elapsed()));
        }
        drop(current_hidden);
        budget.release(sequence_bytes, current_label)?;

        let last_hidden_start = (seq_len - 1).checked_mul(hidden_size).ok_or_else(|| {
            RuntimeError::Shape("gpt-neox RAMA last hidden offset overflow".to_string())
        })?;
        let last_hidden = &final_norm[last_hidden_start..last_hidden_start + hidden_size];
        let lm_head_config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: hidden_size,
                out_features: self.config.vocab_size,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        };
        let lm_head_started = Instant::now();
        let (logits, token_id) =
            if !collect_logits && matches!(self.config.sampling, StreamingSamplingConfig::Argmax) {
                let token_id = if let Some(pinned) = &self.pinned_lm_head_weight {
                    let logits = match crate::ops::linear(
                        last_hidden,
                        pinned,
                        self.lm_head_bias.as_deref(),
                        1,
                        hidden_size,
                        self.config.vocab_size,
                    ) {
                        Ok(l) => l,
                        Err(err) => {
                            budget.release(sequence_bytes, final_norm_label)?;
                            return Err(err);
                        }
                    };
                    if let Some(timing) = timing.as_deref_mut() {
                        timing.record_lm_head(elapsed_ns_u64(lm_head_started.elapsed()));
                    }
                    let mut max_id = 0;
                    let mut max_val = logits[0];
                    for (i, &v) in logits.iter().enumerate() {
                        if v > max_val {
                            max_val = v;
                            max_id = i;
                        }
                    }
                    max_id
                } else {
                    let tid = match streaming_tile_linear_argmax_from_model(
                        model,
                        &self.lm_head_weight,
                        last_hidden,
                        self.lm_head_bias.as_deref(),
                        lm_head_config,
                        budget,
                    ) {
                        Ok(token_id) => token_id,
                        Err(err) => {
                            budget.release(sequence_bytes, final_norm_label)?;
                            return Err(err);
                        }
                    };
                    if let Some(timing) = timing.as_deref_mut() {
                        timing.record_lm_head(elapsed_ns_u64(lm_head_started.elapsed()));
                    }
                    tid
                };
                (Vec::new(), token_id)
            } else {
                let logits = if let Some(pinned) = &self.pinned_lm_head_weight {
                    match crate::ops::linear(
                        last_hidden,
                        pinned,
                        self.lm_head_bias.as_deref(),
                        1,
                        hidden_size,
                        self.config.vocab_size,
                    ) {
                        Ok(l) => l,
                        Err(err) => {
                            budget.release(sequence_bytes, final_norm_label)?;
                            return Err(err);
                        }
                    }
                } else {
                    match streaming_tile_linear_from_model(
                        model,
                        &self.lm_head_weight,
                        last_hidden,
                        self.lm_head_bias.as_deref(),
                        lm_head_config,
                        budget,
                    ) {
                        Ok(values) => values,
                        Err(err) => {
                            budget.release(sequence_bytes, final_norm_label)?;
                            return Err(err);
                        }
                    }
                };
                if let Some(timing) = timing.as_deref_mut() {
                    timing.record_lm_head(elapsed_ns_u64(lm_head_started.elapsed()));
                }

                let sampling_started = Instant::now();
                let token_id = match sample_gpt_neox_logits(&logits, self.config.sampling) {
                    Ok(token_id) => token_id,
                    Err(err) => {
                        budget.release(sequence_bytes, final_norm_label)?;
                        return Err(err);
                    }
                };
                if let Some(timing) = timing {
                    timing.record_sampling(elapsed_ns_u64(sampling_started.elapsed()));
                }
                let logits = if collect_logits { logits } else { Vec::new() };
                (logits, token_id)
            };
        drop(final_norm);
        budget.release(sequence_bytes, final_norm_label)?;
        Ok(Some(StreamingNextTokenResult { logits, token_id }))
    }

    fn decode_layer_params_from_model(
        &self,
        model: &mut LazyRllmModel,
        layer_idx: usize,
    ) -> Result<OwnedStreamingBlockParameters> {
        if layer_idx >= self.config.num_layers {
            return Err(RuntimeError::Shape(format!(
                "gpt-neox RAMA layer index {layer_idx} out of range for {} layers",
                self.config.num_layers
            )));
        }
        let hidden_size = layer_decode_hidden_size(self.config)?;
        let prefix = format!("gpt_neox.layers.{layer_idx}");
        Ok(OwnedStreamingBlockParameters {
            input_layernorm_weight: decode_vector_tensor(
                model,
                &format!("{prefix}.input_layernorm.weight"),
                hidden_size,
            )?,
            input_layernorm_bias: decode_vector_tensor(
                model,
                &format!("{prefix}.input_layernorm.bias"),
                hidden_size,
            )?,
            qkv_bias: decode_optional_vector_tensor(
                model,
                &format!("{prefix}.attention.query_key_value.bias"),
                3 * hidden_size,
            )?,
            attention_out_bias: decode_optional_vector_tensor(
                model,
                &format!("{prefix}.attention.dense.bias"),
                hidden_size,
            )?,
            post_attention_layernorm_weight: decode_vector_tensor(
                model,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden_size,
            )?,
            post_attention_layernorm_bias: decode_vector_tensor(
                model,
                &format!("{prefix}.post_attention_layernorm.bias"),
                hidden_size,
            )?,
            mlp_in_bias: decode_optional_vector_tensor(
                model,
                &format!("{prefix}.mlp.dense_h_to_4h.bias"),
                self.config.intermediate_size,
            )?,
            mlp_out_bias: decode_optional_vector_tensor(
                model,
                &format!("{prefix}.mlp.dense_4h_to_h.bias"),
                hidden_size,
            )?,
        })
    }
}

fn validated_prefill_chunk_tokens(requested: Option<usize>, prompt_len: usize) -> Result<usize> {
    match requested {
        Some(0) => Err(RuntimeError::Shape(
            "RAMA prefill chunk size must be greater than zero".to_string(),
        )),
        Some(chunk_tokens) => Ok(chunk_tokens.min(prompt_len)),
        None => Ok(prompt_len),
    }
}

pub fn recommend_rama_prefill_chunk_tokens(
    config: StreamingEchoTransformerConfig,
    policy: RamaPrefillPolicy,
    prompt_len: usize,
    memory_budget_bytes: Option<usize>,
) -> Result<usize> {
    validate_layer_decode_config(config)?;
    if prompt_len == 0 {
        return Err(RuntimeError::Shape(
            "RAMA prefill policy requires at least one prompt token".to_string(),
        ));
    }

    let mut chunk_tokens = shape_aware_prefill_chunk_tokens(config, policy)?.min(prompt_len);
    if let Some(limit) = memory_budget_bytes.filter(|&limit| limit != usize::MAX) {
        while chunk_tokens > 1 && estimated_prefill_window_peak_bytes(config, chunk_tokens)? > limit
        {
            chunk_tokens = (chunk_tokens / 2).max(1);
        }
    }
    Ok(chunk_tokens)
}

fn shape_aware_prefill_chunk_tokens(
    config: StreamingEchoTransformerConfig,
    policy: RamaPrefillPolicy,
) -> Result<usize> {
    let hidden_size = layer_decode_hidden_size(config)?;
    let hidden_scaled = scaled_prefill_window(hidden_size, RAMA_PREFILL_BASE_HIDDEN_SIZE)?;
    let layer_scaled = scaled_prefill_window(config.num_layers, RAMA_PREFILL_BASE_LAYER_COUNT)?;
    let raw = hidden_scaled
        .max(layer_scaled)
        .max(RAMA_PREFILL_BASE_CHUNK_TOKENS);
    let low_ram = next_power_of_two_at_least(raw)?.clamp(
        RAMA_PREFILL_BASE_CHUNK_TOKENS,
        RAMA_PREFILL_LOW_RAM_MAX_CHUNK_TOKENS,
    );

    Ok(match policy {
        RamaPrefillPolicy::LowRam => low_ram,
        RamaPrefillPolicy::Speed => low_ram
            .saturating_mul(2)
            .min(RAMA_PREFILL_SPEED_MAX_CHUNK_TOKENS),
    })
}

fn scaled_prefill_window(value: usize, baseline: usize) -> Result<usize> {
    let scaled = value
        .checked_mul(RAMA_PREFILL_BASE_CHUNK_TOKENS)
        .ok_or_else(|| RuntimeError::Shape("RAMA prefill policy scale overflow".to_string()))?;
    Ok(div_ceil_usize(scaled, baseline))
}

fn div_ceil_usize(numerator: usize, denominator: usize) -> usize {
    numerator / denominator + usize::from(!numerator.is_multiple_of(denominator))
}

fn next_power_of_two_at_least(value: usize) -> Result<usize> {
    let mut power = 1usize;
    while power < value {
        power = power.checked_mul(2).ok_or_else(|| {
            RuntimeError::Shape("RAMA prefill policy power-of-two overflow".to_string())
        })?;
    }
    Ok(power)
}

fn estimated_prefill_window_peak_bytes(
    config: StreamingEchoTransformerConfig,
    chunk_tokens: usize,
) -> Result<usize> {
    let hidden_size = layer_decode_hidden_size(config)?;
    let hidden_terms = hidden_size
        .checked_mul(RAMA_PREFILL_ESTIMATE_HIDDEN_MULTIPLIER)
        .ok_or_else(|| RuntimeError::Shape("RAMA prefill hidden estimate overflow".to_string()))?;
    let values_per_token = config
        .intermediate_size
        .checked_add(hidden_terms)
        .ok_or_else(|| {
            RuntimeError::Shape("RAMA prefill per-token estimate overflow".to_string())
        })?;
    chunk_tokens
        .checked_mul(values_per_token)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| RuntimeError::Shape("RAMA prefill byte estimate overflow".to_string()))
}

fn elapsed_ns_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

pub fn prepare_gpt_neox_echo_transformer_from_model(
    model: &mut LazyRllmModel,
    build: GptNeoxEchoBuildConfig,
) -> Result<PreparedGptNeoxEchoTransformer> {
    validate_build_config(build)?;
    let embedding_weight =
        choose_existing_tensor(model, &["gpt_neox.embed_in.weight", "embed.weight"])?;
    let lm_head_weight = choose_existing_tensor(
        model,
        &[
            "embed_out.weight",
            "lm_head.weight",
            "gpt_neox.embed_out.weight",
        ],
    )?;
    let embedding_shape = tensor_shape(model, &embedding_weight)?;
    if embedding_shape.len() != 2 {
        return Err(RuntimeError::Shape(format!(
            "GPT-NeoX embedding tensor {embedding_weight} must be rank-2 [vocab, hidden], got {:?}",
            embedding_shape
        )));
    }
    let vocab_size = embedding_shape[0];
    let hidden_size = embedding_shape[1];
    if hidden_size % build.num_heads != 0 {
        return Err(RuntimeError::Shape(format!(
            "hidden_size {hidden_size} must be divisible by num_heads {}",
            build.num_heads
        )));
    }
    validate_weight_shape(model, &lm_head_weight, &[vocab_size, hidden_size])?;

    let num_layers = detect_contiguous_layer_count(model)?;
    let head_dim = hidden_size / build.num_heads;
    let rotary_dim = gpt_neox_rotary_dim(head_dim, build.rotary_pct)?;
    let max_seq_len = build
        .max_seq_len
        .unwrap_or_else(|| model.metadata().default_context_length as usize);

    let mut layers = Vec::with_capacity(num_layers);
    let mut layer_params = Vec::with_capacity(num_layers);
    let mut intermediate_size = None;
    let mut resident_parameter_bytes = 0usize;

    for layer_idx in 0..num_layers {
        let prefix = format!("gpt_neox.layers.{layer_idx}");
        let qkv_weight = format!("{prefix}.attention.query_key_value.weight");
        let attention_out_weight = format!("{prefix}.attention.dense.weight");
        let mlp_in_weight = format!("{prefix}.mlp.dense_h_to_4h.weight");
        let mlp_out_weight = format!("{prefix}.mlp.dense_4h_to_h.weight");

        validate_weight_shape(model, &qkv_weight, &[3 * hidden_size, hidden_size])?;
        validate_weight_shape(model, &attention_out_weight, &[hidden_size, hidden_size])?;
        let mlp_in_shape = tensor_shape(model, &mlp_in_weight)?;
        if mlp_in_shape.len() != 2 || mlp_in_shape[1] != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "GPT-NeoX MLP input weight {mlp_in_weight} must have shape [intermediate, {hidden_size}], got {:?}",
                mlp_in_shape
            )));
        }
        let layer_intermediate = mlp_in_shape[0];
        match intermediate_size {
            Some(existing) if existing != layer_intermediate => {
                return Err(RuntimeError::Shape(format!(
                    "layer {layer_idx} intermediate_size {layer_intermediate} does not match previous {existing}"
                )));
            }
            None => intermediate_size = Some(layer_intermediate),
            _ => {}
        }
        validate_weight_shape(model, &mlp_out_weight, &[hidden_size, layer_intermediate])?;

        let params = OwnedStreamingBlockParameters {
            input_layernorm_weight: decode_vector_tensor(
                model,
                &format!("{prefix}.input_layernorm.weight"),
                hidden_size,
            )?,
            input_layernorm_bias: decode_vector_tensor(
                model,
                &format!("{prefix}.input_layernorm.bias"),
                hidden_size,
            )?,
            qkv_bias: decode_optional_vector_tensor(
                model,
                &format!("{prefix}.attention.query_key_value.bias"),
                3 * hidden_size,
            )?,
            attention_out_bias: decode_optional_vector_tensor(
                model,
                &format!("{prefix}.attention.dense.bias"),
                hidden_size,
            )?,
            post_attention_layernorm_weight: decode_vector_tensor(
                model,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden_size,
            )?,
            post_attention_layernorm_bias: decode_vector_tensor(
                model,
                &format!("{prefix}.post_attention_layernorm.bias"),
                hidden_size,
            )?,
            mlp_in_bias: decode_optional_vector_tensor(
                model,
                &format!("{prefix}.mlp.dense_h_to_4h.bias"),
                layer_intermediate,
            )?,
            mlp_out_bias: decode_optional_vector_tensor(
                model,
                &format!("{prefix}.mlp.dense_4h_to_h.bias"),
                hidden_size,
            )?,
        };
        resident_parameter_bytes = resident_parameter_bytes.saturating_add(params.resident_bytes());

        layers.push(OwnedStreamingBlockTensorNames {
            qkv_weight,
            attention_out_weight,
            mlp_in_weight,
            mlp_out_weight,
        });
        layer_params.push(params);
    }

    let final_layernorm_weight = decode_vector_tensor(
        model,
        &choose_existing_tensor(
            model,
            &[
                "gpt_neox.final_layer_norm.weight",
                "final_layer_norm.weight",
                "final_layernorm.weight",
            ],
        )?,
        hidden_size,
    )?;
    let final_layernorm_bias = decode_vector_tensor(
        model,
        &choose_existing_tensor(
            model,
            &[
                "gpt_neox.final_layer_norm.bias",
                "final_layer_norm.bias",
                "final_layernorm.bias",
            ],
        )?,
        hidden_size,
    )?;
    resident_parameter_bytes = resident_parameter_bytes
        .saturating_add(final_layernorm_weight.len() * std::mem::size_of::<f32>())
        .saturating_add(final_layernorm_bias.len() * std::mem::size_of::<f32>());

    let lm_head_bias = decode_first_optional_vector_tensor(
        model,
        &["embed_out.bias", "lm_head.bias", "gpt_neox.embed_out.bias"],
        vocab_size,
    )?;
    if let Some(bias) = &lm_head_bias {
        resident_parameter_bytes =
            resident_parameter_bytes.saturating_add(bias.len() * std::mem::size_of::<f32>());
    }

    Ok(PreparedGptNeoxEchoTransformer {
        config: StreamingEchoTransformerConfig {
            num_layers,
            max_new_tokens: build.max_new_tokens,
            max_seq_len,
            vocab_size,
            num_heads: build.num_heads,
            head_dim,
            intermediate_size: intermediate_size.ok_or_else(|| {
                RuntimeError::Shape("GPT-NeoX stack has no MLP layers".to_string())
            })?,
            causal: build.causal,
            layer_norm_eps: build.layer_norm_eps,
            use_parallel_residual: build.use_parallel_residual,
            sampling: build.sampling,
            rotary: Some(StreamingTinyRotaryConfig {
                rotary_dim,
                base: build.rotary_base,
            }),
        },
        embedding_weight,
        layers,
        lm_head_weight,
        layer_params,
        final_layernorm_weight,
        final_layernorm_bias,
        lm_head_bias,
        resident_parameter_bytes,
    })
}

pub fn prepare_gpt_neox_echo_transformer_from_metadata(
    model: &mut LazyRllmModel,
    generation: GptNeoxEchoGenerationConfig,
) -> Result<PreparedGptNeoxEchoTransformer> {
    let build = gpt_neox_build_config_from_metadata(model.metadata(), generation)?;
    prepare_gpt_neox_echo_transformer_from_model(model, build)
}

pub fn prepare_gpt_neox_rama_transformer_from_model(
    model: &mut LazyRllmModel,
    build: GptNeoxRamaBuildConfig,
) -> Result<PreparedGptNeoxRamaTransformer> {
    prepare_gpt_neox_echo_transformer_from_model(model, build)
}

pub fn prepare_gpt_neox_rama_transformer_from_metadata(
    model: &mut LazyRllmModel,
    generation: GptNeoxRamaGenerationConfig,
) -> Result<PreparedGptNeoxRamaTransformer> {
    prepare_gpt_neox_echo_transformer_from_metadata(model, generation)
}

pub fn prepare_gpt_neox_rama_layer_decode_transformer_from_model(
    model: &mut LazyRllmModel,
    build: GptNeoxRamaBuildConfig,
) -> Result<LayerDecodedGptNeoxRamaTransformer> {
    validate_build_config(build)?;
    let embedding_weight =
        choose_existing_tensor(model, &["gpt_neox.embed_in.weight", "embed.weight"])?;
    let lm_head_weight = choose_existing_tensor(
        model,
        &[
            "embed_out.weight",
            "lm_head.weight",
            "gpt_neox.embed_out.weight",
        ],
    )?;
    let embedding_shape = tensor_shape(model, &embedding_weight)?;
    if embedding_shape.len() != 2 {
        return Err(RuntimeError::Shape(format!(
            "GPT-NeoX embedding tensor {embedding_weight} must be rank-2 [vocab, hidden], got {:?}",
            embedding_shape
        )));
    }
    let vocab_size = embedding_shape[0];
    let hidden_size = embedding_shape[1];
    if hidden_size % build.num_heads != 0 {
        return Err(RuntimeError::Shape(format!(
            "hidden_size {hidden_size} must be divisible by num_heads {}",
            build.num_heads
        )));
    }
    validate_weight_shape(model, &lm_head_weight, &[vocab_size, hidden_size])?;

    let num_layers = detect_contiguous_layer_count(model)?;
    let head_dim = hidden_size / build.num_heads;
    let rotary_dim = gpt_neox_rotary_dim(head_dim, build.rotary_pct)?;
    let max_seq_len = build
        .max_seq_len
        .unwrap_or_else(|| model.metadata().default_context_length as usize);

    let mut layers = Vec::with_capacity(num_layers);
    let mut intermediate_size = None;
    let mut max_layer_parameter_bytes = 0usize;

    for layer_idx in 0..num_layers {
        let prefix = format!("gpt_neox.layers.{layer_idx}");
        let qkv_weight = format!("{prefix}.attention.query_key_value.weight");
        let attention_out_weight = format!("{prefix}.attention.dense.weight");
        let mlp_in_weight = format!("{prefix}.mlp.dense_h_to_4h.weight");
        let mlp_out_weight = format!("{prefix}.mlp.dense_4h_to_h.weight");

        validate_weight_shape(model, &qkv_weight, &[3 * hidden_size, hidden_size])?;
        validate_weight_shape(model, &attention_out_weight, &[hidden_size, hidden_size])?;
        let mlp_in_shape = tensor_shape(model, &mlp_in_weight)?;
        if mlp_in_shape.len() != 2 || mlp_in_shape[1] != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "GPT-NeoX MLP input weight {mlp_in_weight} must have shape [intermediate, {hidden_size}], got {:?}",
                mlp_in_shape
            )));
        }
        let layer_intermediate = mlp_in_shape[0];
        match intermediate_size {
            Some(existing) if existing != layer_intermediate => {
                return Err(RuntimeError::Shape(format!(
                    "layer {layer_idx} intermediate_size {layer_intermediate} does not match previous {existing}"
                )));
            }
            None => intermediate_size = Some(layer_intermediate),
            _ => {}
        }
        validate_weight_shape(model, &mlp_out_weight, &[hidden_size, layer_intermediate])?;
        let layer_param_bytes =
            layer_decode_parameter_bytes(model, &prefix, hidden_size, layer_intermediate)?;
        max_layer_parameter_bytes = max_layer_parameter_bytes.max(layer_param_bytes);

        layers.push(OwnedStreamingBlockTensorNames {
            qkv_weight,
            attention_out_weight,
            mlp_in_weight,
            mlp_out_weight,
        });
    }

    let final_layernorm_weight = decode_vector_tensor(
        model,
        &choose_existing_tensor(
            model,
            &[
                "gpt_neox.final_layer_norm.weight",
                "final_layer_norm.weight",
                "final_layernorm.weight",
            ],
        )?,
        hidden_size,
    )?;
    let final_layernorm_bias = decode_vector_tensor(
        model,
        &choose_existing_tensor(
            model,
            &[
                "gpt_neox.final_layer_norm.bias",
                "final_layer_norm.bias",
                "final_layernorm.bias",
            ],
        )?,
        hidden_size,
    )?;
    let mut resident_parameter_bytes = final_layernorm_weight
        .len()
        .saturating_add(final_layernorm_bias.len())
        .saturating_mul(std::mem::size_of::<f32>());

    let lm_head_bias = decode_first_optional_vector_tensor(
        model,
        &["embed_out.bias", "lm_head.bias", "gpt_neox.embed_out.bias"],
        vocab_size,
    )?;
    if let Some(bias) = &lm_head_bias {
        resident_parameter_bytes =
            resident_parameter_bytes.saturating_add(bias.len() * std::mem::size_of::<f32>());
    }

    Ok(LayerDecodedGptNeoxRamaTransformer {
        config: StreamingEchoTransformerConfig {
            num_layers,
            max_new_tokens: build.max_new_tokens,
            max_seq_len,
            vocab_size,
            num_heads: build.num_heads,
            head_dim,
            intermediate_size: intermediate_size.ok_or_else(|| {
                RuntimeError::Shape("GPT-NeoX stack has no MLP layers".to_string())
            })?,
            causal: build.causal,
            layer_norm_eps: build.layer_norm_eps,
            use_parallel_residual: build.use_parallel_residual,
            sampling: build.sampling,
            rotary: Some(StreamingTinyRotaryConfig {
                rotary_dim,
                base: build.rotary_base,
            }),
        },
        embedding_weight,
        layers,
        lm_head_weight,
        final_layernorm_weight,
        final_layernorm_bias,
        lm_head_bias,
        pinned_lm_head_weight: None,
        resident_parameter_bytes,
        max_layer_parameter_bytes,
    })
}

pub fn prepare_gpt_neox_rama_layer_decode_transformer_from_metadata(
    model: &mut LazyRllmModel,
    generation: GptNeoxRamaGenerationConfig,
) -> Result<LayerDecodedGptNeoxRamaTransformer> {
    let build = gpt_neox_build_config_from_metadata(model.metadata(), generation)?;
    prepare_gpt_neox_rama_layer_decode_transformer_from_model(model, build)
}

fn gpt_neox_build_config_from_metadata(
    metadata: &GlobalMetadata,
    generation: GptNeoxEchoGenerationConfig,
) -> Result<GptNeoxEchoBuildConfig> {
    let model_config = metadata.model_config.as_ref().ok_or_else(|| {
        RuntimeError::InvalidTensorData(
            "GPT-NeoX metadata preparation requires model_config from original config.json"
                .to_string(),
        )
    })?;
    validate_gpt_neox_architecture(metadata, model_config)?;
    let num_heads = required_usize_config_field(
        model_config.num_attention_heads,
        "model_config.num_attention_heads",
    )?;
    let max_seq_len = generation
        .max_seq_len
        .map(Ok)
        .unwrap_or_else(|| infer_max_seq_len(metadata, model_config))?;
    if max_seq_len == 0 {
        return Err(RuntimeError::Shape(
            "GPT-NeoX max_seq_len inferred from metadata must be greater than zero".to_string(),
        ));
    }

    Ok(GptNeoxEchoBuildConfig {
        max_new_tokens: generation.max_new_tokens,
        max_seq_len: Some(max_seq_len),
        num_heads,
        rotary_pct: model_config.rotary_pct.unwrap_or(1.0),
        rotary_base: model_config.rotary_emb_base.unwrap_or(10_000.0),
        causal: generation.causal,
        layer_norm_eps: model_config.layer_norm_eps.unwrap_or(1e-5),
        use_parallel_residual: model_config.use_parallel_residual.unwrap_or(false),
        sampling: generation.sampling,
    })
}

fn validate_gpt_neox_architecture(
    metadata: &GlobalMetadata,
    model_config: &ModelConfigMetadata,
) -> Result<()> {
    let architecture = model_config
        .architecture_type
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or(metadata.architecture.as_str())
        .to_ascii_lowercase();
    if architecture != "gpt_neox" && !architecture.contains("neox") {
        return Err(RuntimeError::InvalidTensorData(format!(
            "expected GPT-NeoX model_config architecture, got {architecture}"
        )));
    }
    Ok(())
}

fn infer_max_seq_len(
    metadata: &GlobalMetadata,
    model_config: &ModelConfigMetadata,
) -> Result<usize> {
    if let Some(max_position_embeddings) = model_config.max_position_embeddings {
        return usize::try_from(max_position_embeddings).map_err(|_| {
            RuntimeError::Shape(format!(
                "model_config.max_position_embeddings {max_position_embeddings} overflows usize"
            ))
        });
    }
    usize::try_from(metadata.default_context_length).map_err(|_| {
        RuntimeError::Shape(format!(
            "default_context_length {} overflows usize",
            metadata.default_context_length
        ))
    })
}

fn required_usize_config_field(value: Option<u64>, field: &str) -> Result<usize> {
    let value = value.ok_or_else(|| {
        RuntimeError::InvalidTensorData(format!("GPT-NeoX metadata preparation requires {field}"))
    })?;
    usize::try_from(value)
        .map_err(|_| RuntimeError::Shape(format!("{field} value {value} overflows usize")))
}

fn validate_build_config(build: GptNeoxEchoBuildConfig) -> Result<()> {
    if build.max_new_tokens == 0 || build.num_heads == 0 {
        return Err(RuntimeError::Shape(format!(
            "GPT-NeoX build config must have non-zero max_new_tokens and num_heads, got max_new_tokens={}, num_heads={}",
            build.max_new_tokens, build.num_heads
        )));
    }
    if let Some(max_seq_len) = build.max_seq_len {
        if max_seq_len == 0 {
            return Err(RuntimeError::Shape(
                "GPT-NeoX max_seq_len override must be greater than zero".to_string(),
            ));
        }
    }
    if !build.layer_norm_eps.is_finite() || build.layer_norm_eps < 0.0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "layer_norm_eps must be finite and non-negative, got {}",
            build.layer_norm_eps
        )));
    }
    if !build.rotary_base.is_finite() || build.rotary_base <= 0.0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "rotary_base must be finite and positive, got {}",
            build.rotary_base
        )));
    }
    Ok(())
}

fn choose_existing_tensor(model: &LazyRllmModel, candidates: &[&str]) -> Result<String> {
    for name in candidates {
        if model.tensor(name).is_ok() {
            return Ok((*name).to_string());
        }
    }
    Err(RuntimeError::MissingTensor(candidates.join(" or ")))
}

fn tensor_shape(model: &LazyRllmModel, name: &str) -> Result<Vec<usize>> {
    model
        .tensor(name)?
        .shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                RuntimeError::Shape(format!("tensor {name} dimension {dim} overflows usize"))
            })
        })
        .collect()
}

fn validate_weight_shape(model: &LazyRllmModel, name: &str, expected: &[usize]) -> Result<()> {
    let actual = tensor_shape(model, name)?;
    if actual != expected {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected {:?}",
            actual, expected
        )));
    }
    Ok(())
}

fn detect_contiguous_layer_count(model: &LazyRllmModel) -> Result<usize> {
    let mut layers = BTreeSet::new();
    for tensor in model.tensors() {
        if tensor.name.ends_with(".attention.query_key_value.weight") {
            if let Some(layer_idx) = parse_layer_id(&tensor.name) {
                layers.insert(layer_idx);
            }
        }
    }
    if layers.is_empty() {
        return Err(RuntimeError::MissingTensor(
            "gpt_neox.layers.N.attention.query_key_value.weight".to_string(),
        ));
    }
    for (expected, actual) in layers.iter().enumerate() {
        if expected != *actual {
            return Err(RuntimeError::Shape(format!(
                "GPT-NeoX layers must be contiguous from 0, expected layer {expected}, found {actual}"
            )));
        }
    }
    Ok(layers.len())
}

fn parse_layer_id(name: &str) -> Option<usize> {
    let marker = ".layers.";
    let start = name.find(marker)? + marker.len();
    let rest = &name[start..];
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

fn decode_vector_tensor(
    model: &mut LazyRllmModel,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>> {
    let mut budget = MemoryBudget::unbounded();
    let tensor = model.decode_tensor(name, &mut budget)?;
    let runtime_bytes = tensor.runtime_size_bytes();
    budget.release(runtime_bytes, format!("prepared param release: {name}"))?;
    if tensor.shape != [expected_len] {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected [{expected_len}]",
            tensor.shape
        )));
    }
    Ok(tensor.data)
}

fn decode_matrix_tensor(
    model: &mut LazyRllmModel,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    let mut budget = MemoryBudget::unbounded();
    let tensor = model.decode_tensor(name, &mut budget)?;
    let runtime_bytes = tensor.runtime_size_bytes();
    budget.release(runtime_bytes, format!("prepared param release: {name}"))?;
    if tensor.shape != expected_shape {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected {:?}",
            tensor.shape, expected_shape
        )));
    }
    Ok(tensor.data)
}
fn decode_optional_vector_tensor(
    model: &mut LazyRllmModel,
    name: &str,
    expected_len: usize,
) -> Result<Option<Vec<f32>>> {
    if model.tensor(name).is_ok() {
        decode_vector_tensor(model, name, expected_len).map(Some)
    } else {
        Ok(None)
    }
}

fn decode_first_optional_vector_tensor(
    model: &mut LazyRllmModel,
    candidates: &[&str],
    expected_len: usize,
) -> Result<Option<Vec<f32>>> {
    for name in candidates {
        if model.tensor(name).is_ok() {
            return decode_vector_tensor(model, name, expected_len).map(Some);
        }
    }
    Ok(None)
}

fn layer_decode_hidden_size(config: StreamingEchoTransformerConfig) -> Result<usize> {
    config
        .num_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| RuntimeError::Shape("gpt-neox RAMA hidden_size overflow".to_string()))
}

fn layer_decode_rotary_config(
    config: StreamingEchoTransformerConfig,
    seq_len: usize,
    position_offset: usize,
) -> Option<crate::RotaryEmbeddingConfig> {
    config.rotary.map(|rotary| crate::RotaryEmbeddingConfig {
        seq_len,
        num_heads: config.num_heads,
        head_dim: config.head_dim,
        rotary_dim: rotary.rotary_dim,
        base: rotary.base,
        position_offset,
    })
}

fn sample_gpt_neox_logits(logits: &[f32], sampling: StreamingSamplingConfig) -> Result<usize> {
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
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| RuntimeError::Shape(format!("{label} byte size overflow")))
}

fn layer_decode_parameter_bytes(
    model: &LazyRllmModel,
    prefix: &str,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<usize> {
    let mut values = 0usize;
    values = values
        .checked_add(required_vector_len(
            model,
            &format!("{prefix}.input_layernorm.weight"),
            hidden_size,
        )?)
        .ok_or_else(|| RuntimeError::Shape("layer parameter size overflow".to_string()))?;
    values = values
        .checked_add(required_vector_len(
            model,
            &format!("{prefix}.input_layernorm.bias"),
            hidden_size,
        )?)
        .ok_or_else(|| RuntimeError::Shape("layer parameter size overflow".to_string()))?;
    values = values
        .checked_add(optional_vector_len(
            model,
            &format!("{prefix}.attention.query_key_value.bias"),
            3 * hidden_size,
        )?)
        .ok_or_else(|| RuntimeError::Shape("layer parameter size overflow".to_string()))?;
    values = values
        .checked_add(optional_vector_len(
            model,
            &format!("{prefix}.attention.dense.bias"),
            hidden_size,
        )?)
        .ok_or_else(|| RuntimeError::Shape("layer parameter size overflow".to_string()))?;
    values = values
        .checked_add(required_vector_len(
            model,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden_size,
        )?)
        .ok_or_else(|| RuntimeError::Shape("layer parameter size overflow".to_string()))?;
    values = values
        .checked_add(required_vector_len(
            model,
            &format!("{prefix}.post_attention_layernorm.bias"),
            hidden_size,
        )?)
        .ok_or_else(|| RuntimeError::Shape("layer parameter size overflow".to_string()))?;
    values = values
        .checked_add(optional_vector_len(
            model,
            &format!("{prefix}.mlp.dense_h_to_4h.bias"),
            intermediate_size,
        )?)
        .ok_or_else(|| RuntimeError::Shape("layer parameter size overflow".to_string()))?;
    values = values
        .checked_add(optional_vector_len(
            model,
            &format!("{prefix}.mlp.dense_4h_to_h.bias"),
            hidden_size,
        )?)
        .ok_or_else(|| RuntimeError::Shape("layer parameter size overflow".to_string()))?;

    values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| RuntimeError::Shape("layer parameter byte size overflow".to_string()))
}

fn required_vector_len(model: &LazyRllmModel, name: &str, expected_len: usize) -> Result<usize> {
    validate_vector_shape(model, name, expected_len)?;
    Ok(expected_len)
}

fn optional_vector_len(model: &LazyRllmModel, name: &str, expected_len: usize) -> Result<usize> {
    if model.tensor(name).is_ok() {
        validate_vector_shape(model, name, expected_len)?;
        Ok(expected_len)
    } else {
        Ok(0)
    }
}

fn validate_vector_shape(model: &LazyRllmModel, name: &str, expected_len: usize) -> Result<()> {
    let actual = tensor_shape(model, name)?;
    if actual != [expected_len] {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected [{expected_len}]",
            actual
        )));
    }
    Ok(())
}

fn validate_layer_decode_generation_inputs(
    prompt_token_ids: &[usize],
    config: StreamingEchoTransformerConfig,
    state: &RamaContextState,
) -> Result<()> {
    validate_layer_decode_step_inputs(prompt_token_ids, 0, config, state)?;
    if config.max_new_tokens == 0 {
        return Err(RuntimeError::Shape(
            "max_new_tokens must be greater than zero".to_string(),
        ));
    }
    let required_sequence = prompt_token_ids
        .len()
        .checked_add(config.max_new_tokens)
        .ok_or_else(|| RuntimeError::Shape("RAMA layer-decode length overflow".to_string()))?;
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

fn validate_layer_decode_step_inputs(
    token_ids: &[usize],
    position_offset: usize,
    config: StreamingEchoTransformerConfig,
    state: &RamaContextState,
) -> Result<()> {
    validate_layer_decode_config(config)?;
    validate_layer_decode_token_ids(token_ids, config)?;
    let end_pos = position_offset
        .checked_add(token_ids.len())
        .ok_or_else(|| RuntimeError::Shape("RAMA layer-decode position overflow".to_string()))?;
    if end_pos > config.max_seq_len {
        return Err(RuntimeError::Shape(format!(
            "position_offset {position_offset} + token len {} exceeds max_seq_len {}",
            token_ids.len(),
            config.max_seq_len
        )));
    }
    validate_layer_decode_state(config, state)?;
    for layer_idx in 0..config.num_layers {
        let len = state.cache_len(layer_idx)?;
        if len != position_offset {
            return Err(RuntimeError::Shape(format!(
                "decode position_offset {position_offset} must match RAMA layer cache len {len} at layer {layer_idx}"
            )));
        }
    }
    Ok(())
}

fn validate_layer_decode_config(config: StreamingEchoTransformerConfig) -> Result<()> {
    if config.num_layers == 0
        || config.max_seq_len == 0
        || config.vocab_size == 0
        || config.num_heads == 0
        || config.head_dim == 0
        || config.intermediate_size == 0
    {
        return Err(RuntimeError::Shape(format!(
            "RAMA layer-decode dimensions must be non-zero: num_layers={}, max_seq_len={}, vocab_size={}, num_heads={}, head_dim={}, intermediate_size={}",
            config.num_layers,
            config.max_seq_len,
            config.vocab_size,
            config.num_heads,
            config.head_dim,
            config.intermediate_size
        )));
    }
    let _hidden_size = layer_decode_hidden_size(config)?;
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

fn validate_layer_decode_token_ids(
    token_ids: &[usize],
    config: StreamingEchoTransformerConfig,
) -> Result<()> {
    if token_ids.is_empty() {
        return Err(RuntimeError::Shape(
            "RAMA layer-decode step requires at least one token".to_string(),
        ));
    }
    for &token_id in token_ids {
        if token_id >= config.vocab_size {
            return Err(RuntimeError::Shape(format!(
                "token id {token_id} out of vocab range 0..{}",
                config.vocab_size
            )));
        }
    }
    Ok(())
}

fn validate_layer_decode_state(
    config: StreamingEchoTransformerConfig,
    state: &RamaContextState,
) -> Result<()> {
    if state.layer_count() != config.num_layers {
        return Err(RuntimeError::Shape(format!(
            "RAMA layer-decode state layer_count {} does not match config {}",
            state.layer_count(),
            config.num_layers
        )));
    }
    for layer_idx in 0..config.num_layers {
        let shape = state.cache_shape(layer_idx)?;
        if shape != (config.num_heads, config.head_dim, config.max_seq_len) {
            return Err(RuntimeError::Shape(format!(
                "RAMA layer {layer_idx} cache shape {:?} does not match expected ({}, {}, {})",
                shape, config.num_heads, config.head_dim, config.max_seq_len
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        streaming_echo_transformer_generate_from_model, LazyRllmModel, MemoryBudget, RllmTokenizer,
        StreamingBlockParameters, StreamingBlockTensorNames, StreamingEchoTransformerParameters,
        StreamingEchoTransformerTensorNames, StreamingSamplingConfig,
    };
    use rllm_container::{
        DType, GlobalMetadata, ModelConfigMetadata, RllmWriter, TensorMeta, TokenizerMetadata,
    };
    use sha2::{Digest, Sha256};

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
        std::env::temp_dir().join(format!("rllm-gpt-neox-{name}-{}.rllm", std::process::id()))
    }

    fn add_f32_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        shape: Vec<u64>,
        values: &[f32],
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
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(tensor_id, "rtc-raw-v1", &bytes, &bytes, 0)
            .unwrap();
    }

    fn write_gpt_neox_stack_model(path: &std::path::Path) {
        let mut meta = GlobalMetadata::new_test();
        meta.model_name = "pythia-ish-test".to_string();
        meta.architecture = "gpt_neox".to_string();
        meta.default_context_length = 8;
        meta.model_config = Some(ModelConfigMetadata {
            architecture_type: Some("gpt_neox".to_string()),
            num_hidden_layers: Some(NUM_LAYERS as u64),
            hidden_size: Some(HIDDEN_SIZE as u64),
            intermediate_size: Some(INTERMEDIATE_SIZE as u64),
            num_attention_heads: Some(NUM_HEADS as u64),
            max_position_embeddings: Some(8),
            rotary_pct: Some(1.0),
            rotary_emb_base: Some(10_000.0),
            layer_norm_eps: Some(1e-5),
            use_parallel_residual: Some(true),
            vocab_size: Some(VOCAB_SIZE as u64),
            num_key_value_heads: None,
            rms_norm_eps: None,
            rope_theta: None,
            tie_word_embeddings: None,
            ..Default::default()
        });
        let mut writer = RllmWriter::new(path, meta).unwrap();
        let mut tensor_id = 0u64;

        add_f32_tensor(
            &mut writer,
            tensor_id,
            "gpt_neox.embed_in.weight",
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
            &EMBEDDING_WEIGHT,
        );
        tensor_id += 1;

        for layer_idx in 0..NUM_LAYERS {
            let refs = full_block_refs(layer_idx);
            let prefix = format!("gpt_neox.layers.{layer_idx}");
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.input_layernorm.weight"),
                vec![HIDDEN_SIZE as u64],
                refs.input_layernorm_weight,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.input_layernorm.bias"),
                vec![HIDDEN_SIZE as u64],
                refs.input_layernorm_bias,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.attention.query_key_value.weight"),
                vec![(3 * HIDDEN_SIZE) as u64, HIDDEN_SIZE as u64],
                refs.qkv_weight,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.attention.query_key_value.bias"),
                vec![(3 * HIDDEN_SIZE) as u64],
                refs.qkv_bias,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.attention.dense.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
                refs.attention_out_weight,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.attention.dense.bias"),
                vec![HIDDEN_SIZE as u64],
                refs.attention_out_bias,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.post_attention_layernorm.weight"),
                vec![HIDDEN_SIZE as u64],
                refs.post_attention_layernorm_weight,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.post_attention_layernorm.bias"),
                vec![HIDDEN_SIZE as u64],
                refs.post_attention_layernorm_bias,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.dense_h_to_4h.weight"),
                vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
                refs.mlp_in_weight,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.dense_h_to_4h.bias"),
                vec![INTERMEDIATE_SIZE as u64],
                refs.mlp_in_bias,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.dense_4h_to_h.weight"),
                vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
                refs.mlp_out_weight,
            );
            tensor_id += 1;
            add_f32_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.dense_4h_to_h.bias"),
                vec![HIDDEN_SIZE as u64],
                refs.mlp_out_bias,
            );
            tensor_id += 1;
        }

        add_f32_tensor(
            &mut writer,
            tensor_id,
            "gpt_neox.final_layer_norm.weight",
            vec![HIDDEN_SIZE as u64],
            &FINAL_LN_WEIGHT,
        );
        tensor_id += 1;
        add_f32_tensor(
            &mut writer,
            tensor_id,
            "gpt_neox.final_layer_norm.bias",
            vec![HIDDEN_SIZE as u64],
            &FINAL_LN_BIAS,
        );
        tensor_id += 1;
        add_f32_tensor(
            &mut writer,
            tensor_id,
            "embed_out.weight",
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
            &LM_HEAD_WEIGHT,
        );
        tensor_id += 1;
        add_f32_tensor(
            &mut writer,
            tensor_id,
            "embed_out.bias",
            vec![VOCAB_SIZE as u64],
            &LM_HEAD_BIAS,
        );

        writer.finalize().unwrap();
    }

    fn explicit_layer_names<'a>() -> [StreamingBlockTensorNames<'a>; NUM_LAYERS] {
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

    fn explicit_layer_params<'a>() -> [StreamingBlockParameters<'a>; NUM_LAYERS] {
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

    fn build_config(max_new_tokens: usize) -> GptNeoxEchoBuildConfig {
        GptNeoxEchoBuildConfig {
            max_new_tokens,
            max_seq_len: Some(8),
            num_heads: NUM_HEADS,
            rotary_pct: 1.0,
            rotary_base: 10_000.0,
            causal: true,
            layer_norm_eps: 1e-5,
            use_parallel_residual: false,
            sampling: StreamingSamplingConfig::Argmax,
        }
    }

    fn policy_config(
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> StreamingEchoTransformerConfig {
        StreamingEchoTransformerConfig {
            num_layers,
            max_new_tokens: 16,
            max_seq_len: 2048,
            vocab_size: 50_304,
            num_heads,
            head_dim: hidden_size / num_heads,
            intermediate_size,
            causal: true,
            layer_norm_eps: 1e-5,
            use_parallel_residual: true,
            sampling: StreamingSamplingConfig::Argmax,
            rotary: Some(StreamingTinyRotaryConfig {
                rotary_dim: hidden_size / num_heads,
                base: 10_000.0,
            }),
        }
    }

    #[test]
    fn recommended_rama_prefill_policy_selects_shape_aware_windows() {
        let pythia_70m_like = policy_config(6, 512, 8, 2048);
        let pythia_160m_like = policy_config(12, 768, 12, 3072);
        let budget_100mb = 100 * 1024 * 1024;

        assert_eq!(
            recommend_rama_prefill_chunk_tokens(
                pythia_70m_like,
                RamaPrefillPolicy::LowRam,
                1024,
                Some(budget_100mb),
            )
            .unwrap(),
            32
        );
        assert_eq!(
            recommend_rama_prefill_chunk_tokens(
                pythia_160m_like,
                RamaPrefillPolicy::LowRam,
                1024,
                Some(budget_100mb),
            )
            .unwrap(),
            64
        );
        assert_eq!(
            recommend_rama_prefill_chunk_tokens(
                pythia_160m_like,
                RamaPrefillPolicy::Speed,
                1024,
                Some(budget_100mb),
            )
            .unwrap(),
            128
        );
    }

    #[test]
    fn recommended_rama_prefill_policy_respects_prompt_len_and_budget() {
        let pythia_160m_like = policy_config(12, 768, 12, 3072);

        assert_eq!(
            recommend_rama_prefill_chunk_tokens(
                pythia_160m_like,
                RamaPrefillPolicy::LowRam,
                48,
                Some(100 * 1024 * 1024),
            )
            .unwrap(),
            48
        );
        assert_eq!(
            recommend_rama_prefill_chunk_tokens(
                pythia_160m_like,
                RamaPrefillPolicy::Speed,
                1024,
                Some(2 * 1024 * 1024),
            )
            .unwrap(),
            32
        );
        assert!(recommend_rama_prefill_chunk_tokens(
            pythia_160m_like,
            RamaPrefillPolicy::LowRam,
            0,
            Some(100 * 1024 * 1024),
        )
        .is_err());
    }

    #[test]
    fn prepares_gpt_neox_echo_stack_from_metadata_and_decodes_small_params() {
        let path = temp_path("prepare");
        write_gpt_neox_stack_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();

        let prepared =
            prepare_gpt_neox_echo_transformer_from_model(&mut model, build_config(2)).unwrap();

        assert_eq!(prepared.config.num_layers, NUM_LAYERS);
        assert_eq!(prepared.config.vocab_size, VOCAB_SIZE);
        assert_eq!(prepared.config.num_heads, NUM_HEADS);
        assert_eq!(prepared.config.head_dim, HEAD_DIM);
        assert_eq!(prepared.config.intermediate_size, INTERMEDIATE_SIZE);
        assert_eq!(prepared.config.max_seq_len, 8);
        assert_eq!(prepared.config.rotary.unwrap().rotary_dim, HEAD_DIM);
        assert_eq!(prepared.embedding_weight, "gpt_neox.embed_in.weight");
        assert_eq!(prepared.lm_head_weight, "embed_out.weight");
        assert_eq!(prepared.layers.len(), NUM_LAYERS);
        assert_eq!(
            prepared.layers[1].qkv_weight,
            "gpt_neox.layers.1.attention.query_key_value.weight"
        );
        assert_eq!(
            prepared.layer_params[0].input_layernorm_weight,
            LN1_WEIGHT_L0
        );
        assert_eq!(
            prepared.layer_params[1].qkv_bias.as_deref(),
            Some(QKV_BIAS_L1.as_slice())
        );
        assert_eq!(prepared.final_layernorm_weight, FINAL_LN_WEIGHT);
        assert_eq!(
            prepared.lm_head_bias.as_deref(),
            Some(LM_HEAD_BIAS.as_slice())
        );
        assert!(prepared.resident_parameter_bytes > 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prepares_gpt_neox_echo_stack_from_persisted_model_config_metadata() {
        let path = temp_path("metadata-config");
        write_gpt_neox_stack_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();

        let prepared = prepare_gpt_neox_echo_transformer_from_metadata(
            &mut model,
            GptNeoxEchoGenerationConfig {
                max_new_tokens: 2,
                max_seq_len: None,
                causal: true,
                sampling: StreamingSamplingConfig::Argmax,
            },
        )
        .unwrap();

        assert_eq!(prepared.config.num_heads, NUM_HEADS);
        assert_eq!(prepared.config.head_dim, HEAD_DIM);
        assert_eq!(prepared.config.max_seq_len, 8);
        assert_eq!(prepared.config.intermediate_size, INTERMEDIATE_SIZE);
        assert_eq!(prepared.config.layer_norm_eps, 1e-5);
        assert_eq!(prepared.config.rotary.unwrap().rotary_dim, HEAD_DIM);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prepares_gpt_neox_rama_layer_decode_stack_without_resident_layer_params() {
        let path = temp_path("layer-decode-prepare");
        write_gpt_neox_stack_model(&path);
        let mut prepared_model = LazyRllmModel::open(&path).unwrap();
        let mut layer_decode_model = LazyRllmModel::open(&path).unwrap();

        let prepared =
            prepare_gpt_neox_rama_transformer_from_model(&mut prepared_model, build_config(2))
                .unwrap();
        let layer_decoded = prepare_gpt_neox_rama_layer_decode_transformer_from_model(
            &mut layer_decode_model,
            build_config(2),
        )
        .unwrap();

        assert_eq!(layer_decoded.config.num_layers, prepared.config.num_layers);
        assert_eq!(layer_decoded.config.vocab_size, prepared.config.vocab_size);
        assert_eq!(layer_decoded.config.head_dim, prepared.config.head_dim);
        assert_eq!(layer_decoded.embedding_weight, prepared.embedding_weight);
        assert_eq!(layer_decoded.lm_head_weight, prepared.lm_head_weight);
        assert_eq!(layer_decoded.layers.len(), NUM_LAYERS);
        assert_eq!(
            layer_decoded.layers[1].qkv_weight,
            "gpt_neox.layers.1.attention.query_key_value.weight"
        );
        assert!(layer_decoded.max_layer_parameter_bytes > 0);
        assert!(layer_decoded.resident_parameter_bytes > 0);
        assert!(layer_decoded.resident_parameter_bytes < prepared.resident_parameter_bytes);
        assert_eq!(prepared.layer_params.len(), NUM_LAYERS);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn layer_decoded_gpt_neox_rama_generation_matches_prepared_stack() {
        let path = temp_path("layer-decode-generate");
        write_gpt_neox_stack_model(&path);
        let mut prepared_model = LazyRllmModel::open(&path).unwrap();
        let mut layer_decode_model = LazyRllmModel::open(&path).unwrap();
        let prompt = [0, 2];
        let build = build_config(3);
        let prepared =
            prepare_gpt_neox_rama_transformer_from_model(&mut prepared_model, build).unwrap();
        let layer_decoded = prepare_gpt_neox_rama_layer_decode_transformer_from_model(
            &mut layer_decode_model,
            build,
        )
        .unwrap();
        let mut prepared_budget = MemoryBudget::new(16384);
        let mut layer_decode_budget = MemoryBudget::new(16384);

        let expected = prepared
            .generate_from_model(&mut prepared_model, &prompt, &mut prepared_budget)
            .unwrap();
        let actual = layer_decoded
            .generate_from_model(&mut layer_decode_model, &prompt, &mut layer_decode_budget)
            .unwrap();

        assert_eq!(actual.generated_token_ids, expected.generated_token_ids);
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.step_logits, expected.step_logits);
        assert_eq!(
            actual.context_memory_bytes(),
            expected.context_memory_bytes()
        );
        assert_eq!(prepared_budget.current_bytes(), 0);
        assert_eq!(layer_decode_budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn layer_decoded_gpt_neox_can_skip_logit_collection_for_argmax_generation() {
        let path = temp_path("layer-decode-no-logits");
        write_gpt_neox_stack_model(&path);
        let mut logits_model = LazyRllmModel::open(&path).unwrap();
        let mut no_logits_model = LazyRllmModel::open(&path).unwrap();
        let prompt = [0, 2, 1];
        let build = build_config(3);
        let logits_transformer =
            prepare_gpt_neox_rama_layer_decode_transformer_from_model(&mut logits_model, build)
                .unwrap();
        let no_logits_transformer =
            prepare_gpt_neox_rama_layer_decode_transformer_from_model(&mut no_logits_model, build)
                .unwrap();
        let mut logits_budget = MemoryBudget::new(32768);
        let mut no_logits_budget = MemoryBudget::new(32768);

        let expected = logits_transformer
            .generate_from_model(&mut logits_model, &prompt, &mut logits_budget)
            .unwrap();
        let actual = no_logits_transformer
            .generate_from_model_with_options(
                &mut no_logits_model,
                &prompt,
                &mut no_logits_budget,
                GptNeoxRamaGenerationOptions {
                    timing: true,
                    prefill_chunk_tokens: Some(2),
                    collect_logits: false,
                },
            )
            .unwrap();

        assert_eq!(actual.generated_token_ids, expected.generated_token_ids);
        assert_eq!(actual.token_ids, expected.token_ids);
        assert!(actual.step_logits.is_empty());
        assert_eq!(
            actual.context_memory_bytes(),
            expected.context_memory_bytes()
        );
        assert_eq!(logits_budget.current_bytes(), 0);
        assert_eq!(no_logits_budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn layer_decoded_gpt_neox_chunked_prefill_matches_full_prefill() {
        let path = temp_path("layer-decode-chunked-prefill");
        write_gpt_neox_stack_model(&path);
        let mut full_model = LazyRllmModel::open(&path).unwrap();
        let mut chunked_model = LazyRllmModel::open(&path).unwrap();
        let prompt = [0, 2, 1, 2];
        let build = build_config(3);
        let full =
            prepare_gpt_neox_rama_layer_decode_transformer_from_model(&mut full_model, build)
                .unwrap();
        let chunked =
            prepare_gpt_neox_rama_layer_decode_transformer_from_model(&mut chunked_model, build)
                .unwrap();
        let mut full_budget = MemoryBudget::new(32768);
        let mut chunked_budget = MemoryBudget::new(32768);

        let expected = full
            .generate_from_model(&mut full_model, &prompt, &mut full_budget)
            .unwrap();
        let actual = chunked
            .generate_from_model_with_options(
                &mut chunked_model,
                &prompt,
                &mut chunked_budget,
                GptNeoxRamaGenerationOptions {
                    timing: true,
                    prefill_chunk_tokens: Some(2),
                    collect_logits: true,
                },
            )
            .unwrap();

        assert_eq!(actual.generated_token_ids, expected.generated_token_ids);
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.step_logits, expected.step_logits);
        assert_eq!(
            actual.context_memory_bytes(),
            expected.context_memory_bytes()
        );
        let timing = actual
            .timing()
            .expect("chunked prefill should collect timing");
        assert_eq!(timing.prefill_chunks, 2);
        assert_eq!(timing.decode_steps, build.max_new_tokens - 1);
        assert_eq!(timing.max_prefill_chunk_tokens, 2);
        assert_eq!(
            timing.prefill_timed_blocks,
            timing.prefill_chunks * NUM_LAYERS
        );
        assert!(timing.prefill_embedding_ns > 0);
        assert!(timing.prefill_layer_params_ns > 0);
        assert!(timing.prefill_attention_ns > 0);
        assert!(timing.prefill_attention_qkv_projection_ns > 0);
        assert!(timing.prefill_attention_qkv_split_ns > 0);
        assert!(timing.prefill_attention_rotary_ns > 0);
        assert!(timing.prefill_attention_score_context_ns > 0);
        assert!(timing.prefill_attention_output_projection_ns > 0);
        assert!(timing.prefill_attention_kv_append_ns > 0);
        assert!(timing.prefill_mlp_ns > 0);
        assert!(timing.prefill_mlp_input_projection_ns > 0);
        assert!(timing.prefill_mlp_activation_ns > 0);
        assert!(timing.prefill_mlp_output_projection_ns > 0);
        assert_eq!(full_budget.current_bytes(), 0);
        assert_eq!(chunked_budget.current_bytes(), 0);
        assert!(
            chunked_budget.peak_bytes() <= full_budget.peak_bytes(),
            "chunked prefill should not increase transient peak: chunked={} full={}",
            chunked_budget.peak_bytes(),
            full_budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn layer_decoded_gpt_neox_rejects_zero_prefill_chunk_size() {
        let path = temp_path("layer-decode-zero-prefill-chunk");
        write_gpt_neox_stack_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prompt = [0, 2];
        let transformer =
            prepare_gpt_neox_rama_layer_decode_transformer_from_model(&mut model, build_config(1))
                .unwrap();
        let mut budget = MemoryBudget::new(16384);

        let err = transformer
            .generate_from_model_with_options(
                &mut model,
                &prompt,
                &mut budget,
                GptNeoxRamaGenerationOptions {
                    timing: false,
                    prefill_chunk_tokens: Some(0),
                    collect_logits: true,
                },
            )
            .unwrap_err();

        assert!(matches!(err, RuntimeError::Shape(_)));
        assert_eq!(budget.current_bytes(), 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prepared_gpt_neox_echo_generation_matches_explicit_stack_contract() {
        let path = temp_path("generate");
        write_gpt_neox_stack_model(&path);
        let mut explicit_model = LazyRllmModel::open(&path).unwrap();
        let mut prepared_model = LazyRllmModel::open(&path).unwrap();
        let layer_names = explicit_layer_names();
        let layer_params = explicit_layer_params();
        let prompt = [0, 2];
        let build = build_config(3);
        let prepared =
            prepare_gpt_neox_echo_transformer_from_model(&mut prepared_model, build).unwrap();
        let mut explicit_budget = MemoryBudget::new(16384);
        let mut prepared_budget = MemoryBudget::new(16384);

        let explicit = streaming_echo_transformer_generate_from_model(
            &mut explicit_model,
            &prompt,
            StreamingEchoTransformerTensorNames {
                embedding_weight: "gpt_neox.embed_in.weight",
                layers: &layer_names,
                lm_head_weight: "embed_out.weight",
            },
            StreamingEchoTransformerParameters {
                layers: &layer_params,
                final_layernorm_weight: &FINAL_LN_WEIGHT,
                final_layernorm_bias: &FINAL_LN_BIAS,
                lm_head_bias: Some(&LM_HEAD_BIAS),
            },
            prepared.config,
            &mut explicit_budget,
        )
        .unwrap();

        let actual = prepared
            .generate_from_model(&mut prepared_model, &prompt, &mut prepared_budget)
            .unwrap();

        assert_eq!(actual.generated_token_ids, explicit.generated_token_ids);
        assert_eq!(actual.token_ids, explicit.token_ids);
        assert_eq!(actual.step_logits, explicit.step_logits);
        assert_eq!(actual.context_echo_bytes, explicit.context_echo_bytes);
        assert_eq!(explicit_budget.current_bytes(), 0);
        assert_eq!(prepared_budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prepared_gpt_neox_echo_text_generation_uses_tokenizer_boundary() {
        let path = temp_path("text-generate");
        write_gpt_neox_stack_model(&path);
        let mut expected_model = LazyRllmModel::open(&path).unwrap();
        let mut text_model = LazyRllmModel::open(&path).unwrap();
        let tokenizer = RllmTokenizer::from_metadata(&TokenizerMetadata {
            tokenizer_type: Some("hf-wordlevel".to_string()),
            id_to_token: vec!["A".to_string(), " B".to_string(), " C".to_string()],
            bpe_merges: Vec::new(),
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
        })
        .unwrap();
        let prompt = [0, 2];
        let prepared =
            prepare_gpt_neox_echo_transformer_from_model(&mut text_model, build_config(3)).unwrap();
        let mut expected_budget = MemoryBudget::new(16384);
        let expected = prepared
            .generate_from_model(&mut expected_model, &prompt, &mut expected_budget)
            .unwrap();
        let mut text_budget = MemoryBudget::new(16384);

        let actual = prepared
            .generate_text_from_model(&mut text_model, &tokenizer, "A C", &mut text_budget)
            .unwrap();

        assert_eq!(actual.prompt_token_ids, prompt);
        assert_eq!(actual.generated_token_ids, expected.generated_token_ids);
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.text, tokenizer.decode(&expected.token_ids).unwrap());
        assert_eq!(
            actual.generated_text,
            tokenizer.decode(&expected.generated_token_ids).unwrap()
        );
        assert_eq!(expected_budget.current_bytes(), 0);
        assert_eq!(text_budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }
}
