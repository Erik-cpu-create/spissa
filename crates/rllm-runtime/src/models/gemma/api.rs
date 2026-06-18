use crate::models::gemma::generate::{streaming_gemma_transformer_block, GemmaBlockRuntime};
use crate::models::gemma::model::*;
use crate::rotary::KvCache;
use crate::{
    ops::{embedding_lookup, rms_norm, sample_argmax, sample_top_p},
    LazyRllmModel, MemoryBudget, Result, RuntimeError, StreamingSamplingConfig,
};
use rllm_container::ModelConfigMetadata;

/// Gemma 3 fixed family default for the global-layer linear RoPE divisor
/// (`rope_scaling.factor`). Used only when the packed metadata predates the
/// `rope_scaling_factor` field; current packs carry the value explicitly.
const GEMMA3_DEFAULT_ROPE_SCALING_FACTOR: f32 = 8.0;

#[derive(Debug, Clone, Copy)]
pub struct GemmaGenerationConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: Option<usize>,
    pub causal: bool,
    pub sampling: StreamingSamplingConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct GemmaGenerationOptions {
    pub collect_logits: bool,
}

impl Default for GemmaGenerationOptions {
    fn default() -> Self {
        Self {
            collect_logits: true,
        }
    }
}

fn require_config<'a>(model: &'a LazyRllmModel) -> Result<&'a ModelConfigMetadata> {
    model.metadata().model_config.as_ref().ok_or_else(|| {
        RuntimeError::InvalidTensorData(
            "gemma generation requires persisted model_config metadata; repack with --config <config.json>"
                .to_string(),
        )
    })
}

fn require_usize(field: &str, value: Option<u64>) -> Result<usize> {
    let value = value.ok_or_else(|| {
        RuntimeError::InvalidTensorData(format!("gemma model_config is missing required field {field}"))
    })?;
    usize::try_from(value)
        .map_err(|_| RuntimeError::Shape(format!("gemma model_config field {field}={value} overflows usize")))
}

/// Decode an RMSNorm weight and pre-bake Gemma's `(1 + weight)` convention so it
/// can drive the standard `rms_norm` directly.
fn decode_norm_1plus(
    model: &mut LazyRllmModel,
    name: &str,
    expected_len: usize,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let tensor = model.decode_tensor(name, budget)?;
    if tensor.shape != [expected_len] {
        return Err(RuntimeError::Shape(format!(
            "gemma norm {name} shape {:?} does not match expected [{expected_len}]",
            tensor.shape
        )));
    }
    let mut data = tensor.data;
    for value in data.iter_mut() {
        *value += 1.0;
    }
    Ok(data)
}

pub fn prepare_gemma_transformer_from_metadata(
    model: &mut LazyRllmModel,
    generation: GemmaGenerationConfig,
) -> Result<PreparedGemmaTransformer> {
    let (build, num_layers) = {
        let config = require_config(model)?;
        let hidden_size = require_usize("hidden_size", config.hidden_size)?;
        let num_heads = require_usize("num_attention_heads", config.num_attention_heads)?;
        let num_key_value_heads = config
            .num_key_value_heads
            .map(|v| require_usize("num_key_value_heads", Some(v)))
            .transpose()?
            .unwrap_or(num_heads);
        // Gemma sets head_dim explicitly (256), independent of hidden/heads.
        let head_dim = config
            .head_dim
            .map(|v| require_usize("head_dim", Some(v)))
            .transpose()?
            .unwrap_or(hidden_size / num_heads);
        let query_pre_attn_scalar = config.query_pre_attn_scalar.unwrap_or(head_dim as f32);
        let max_seq_len = generation
            .max_seq_len
            .or_else(|| config.max_position_embeddings.and_then(|v| usize::try_from(v).ok()))
            .unwrap_or(2048);
        let num_layers = require_usize("num_hidden_layers", config.num_hidden_layers)?;
        let build = GemmaBuildConfig {
            max_new_tokens: generation.max_new_tokens,
            max_seq_len,
            num_heads,
            num_key_value_heads,
            head_dim,
            hidden_size,
            intermediate_size: require_usize("intermediate_size", config.intermediate_size)?,
            rms_norm_eps: config.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: config.rope_theta.unwrap_or(1_000_000.0),
            rope_local_base_freq: config.rope_local_base_freq.unwrap_or(10_000.0),
            rope_scaling_factor: config
                .rope_scaling_factor
                .unwrap_or(GEMMA3_DEFAULT_ROPE_SCALING_FACTOR),
            sliding_window_pattern: config
                .sliding_window_pattern
                .map(|v| v as usize)
                .unwrap_or(1),
            attn_scale: 1.0 / query_pre_attn_scalar.sqrt(),
            embed_scale: (hidden_size as f32).sqrt(),
            causal: generation.causal,
            sampling: generation.sampling,
        };
        (build, num_layers)
    };

    let embedding_weight = "model.embed_tokens.weight".to_string();
    let lm_head_weight = if model.tensor("lm_head.weight").is_ok() {
        "lm_head.weight".to_string()
    } else {
        embedding_weight.clone()
    };

    let final_layernorm =
        decode_norm_1plus(model, "model.norm.weight", build.hidden_size, &mut MemoryBudget::unbounded())?;

    let mut layers = Vec::with_capacity(num_layers);
    let mut layer_norms = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        layers.push(GemmaBlockTensorNames {
            q_weight: format!("model.layers.{i}.self_attn.q_proj.weight"),
            k_weight: format!("model.layers.{i}.self_attn.k_proj.weight"),
            v_weight: format!("model.layers.{i}.self_attn.v_proj.weight"),
            o_weight: format!("model.layers.{i}.self_attn.o_proj.weight"),
            gate_weight: format!("model.layers.{i}.mlp.gate_proj.weight"),
            up_weight: format!("model.layers.{i}.mlp.up_proj.weight"),
            down_weight: format!("model.layers.{i}.mlp.down_proj.weight"),
        });
        layer_norms.push(decode_layer_norms(model, i, &build)?);
    }
    if layers.is_empty() {
        return Err(RuntimeError::Shape("gemma model requires at least one layer".to_string()));
    }

    Ok(PreparedGemmaTransformer {
        config: build,
        embedding_weight,
        lm_head_weight,
        layers,
        layer_norms,
        final_layernorm,
    })
}

fn decode_layer_norms(
    model: &mut LazyRllmModel,
    layer: usize,
    build: &GemmaBuildConfig,
) -> Result<GemmaLayerNorms> {
    let hidden = build.hidden_size;
    let head_dim = build.head_dim;
    let mut budget = MemoryBudget::unbounded();
    let mut norm = |model: &mut LazyRllmModel, suffix: &str, len: usize| -> Result<Vec<f32>> {
        decode_norm_1plus(model, &format!("model.layers.{layer}.{suffix}"), len, &mut budget)
    };
    Ok(GemmaLayerNorms {
        input_layernorm: norm(model, "input_layernorm.weight", hidden)?,
        post_attention_layernorm: norm(model, "post_attention_layernorm.weight", hidden)?,
        pre_feedforward_layernorm: norm(model, "pre_feedforward_layernorm.weight", hidden)?,
        post_feedforward_layernorm: norm(model, "post_feedforward_layernorm.weight", hidden)?,
        q_norm: norm(model, "self_attn.q_norm.weight", head_dim)?,
        k_norm: norm(model, "self_attn.k_norm.weight", head_dim)?,
    })
}

pub fn gemma_generate_from_model(
    model: &mut LazyRllmModel,
    prepared: &PreparedGemmaTransformer,
    prompt_token_ids: &[usize],
    budget: &mut MemoryBudget,
    options: GemmaGenerationOptions,
    on_token: &mut dyn FnMut(usize) -> bool,
) -> Result<GemmaTextGenerationResult> {
    let build = &prepared.config;
    let hidden = build.hidden_size;
    let mut token_ids = prompt_token_ids.to_vec();
    let mut generated_token_ids = Vec::new();

    let mut caches = Vec::with_capacity(prepared.layers.len());
    for _ in 0..prepared.layers.len() {
        caches.push(KvCache::new(build.num_key_value_heads, build.head_dim, build.max_seq_len)?);
    }

    // Tied embeddings: decode embed_tokens once and reuse the same f32 buffer
    // for input lookup (scaled by sqrt(hidden)) and the LM head (unscaled).
    let embedding = model.decode_tensor(&prepared.embedding_weight, budget)?.data;
    let vocab_size = embedding.len() / hidden;

    // For parity we keep the first decode step's logits (the prefill → first
    // predicted token over the raw prompt), which is the most directly
    // comparable to a single HF forward.
    let mut first_step_logits = None;
    for step in 0..build.max_new_tokens {
        let current_tokens: &[usize] = if step == 0 {
            prompt_token_ids
        } else {
            &generated_token_ids[generated_token_ids.len() - 1..]
        };
        let seq_len = current_tokens.len();
        let position_offset = token_ids.len() - seq_len;

        let mut hidden_states = embedding_lookup(&embedding, vocab_size, hidden, current_tokens)?;
        for value in hidden_states.iter_mut() {
            *value *= build.embed_scale;
        }

        for (i, names) in prepared.layers.iter().enumerate() {
            hidden_states = streaming_gemma_transformer_block(
                model,
                &hidden_states,
                names,
                &prepared.layer_norms[i],
                build,
                GemmaBlockRuntime {
                    seq_len,
                    position_offset,
                    layer_index: i,
                },
                budget,
                Some(&mut caches[i]),
            )?;
        }

        hidden_states = rms_norm(&hidden_states, &prepared.final_layernorm, seq_len, hidden, build.rms_norm_eps)?;

        let last_hidden = &hidden_states[(seq_len - 1) * hidden..];
        let logits = lm_head_logits(last_hidden, &embedding, vocab_size, hidden);

        let next_token = match build.sampling {
            StreamingSamplingConfig::Argmax => sample_argmax(&logits)?,
            StreamingSamplingConfig::TopP { temperature, top_p, seed } => {
                sample_top_p(&logits, temperature, top_p, seed)?
            }
        };
        token_ids.push(next_token);
        generated_token_ids.push(next_token);
        if step == 0 && options.collect_logits {
            first_step_logits = Some(logits);
        }
        if !on_token(next_token) {
            break;
        }
    }

    let context_echo_bytes = caches.iter().map(KvCache::resident_bytes).sum();
    Ok(GemmaTextGenerationResult {
        prompt_token_ids: prompt_token_ids.to_vec(),
        generated_token_ids,
        token_ids,
        context_echo_bytes,
        logits: first_step_logits,
    })
}

fn lm_head_logits(last_hidden: &[f32], weight: &[f32], vocab_size: usize, hidden: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab_size];
    for (v, logit) in logits.iter_mut().enumerate() {
        let row = &weight[v * hidden..v * hidden + hidden];
        let mut sum = 0.0f32;
        for (h, w) in row.iter().enumerate() {
            sum += last_hidden[h] * *w;
        }
        *logit = sum;
    }
    logits
}
