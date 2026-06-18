use crate::models::gemma::generate::{streaming_gemma_transformer_block, GemmaBlockRuntime};
use crate::models::gemma::model::*;
use crate::rotary::KvCache;
use crate::{
    bf16_to_f32, lm_head_logits_parallel, lm_head_logits_parallel_bf16,
    ops::{embedding_lookup, rms_norm, sample_argmax, sample_top_p},
    LazyRllmModel, MemoryBudget, Result, RuntimeError, StreamingSamplingConfig,
};
use rllm_container::{DType, ModelConfigMetadata};

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

    // Tied embeddings. Materializing the 262208×2560 table as f32 costs 2.68 GB
    // resident — which evicts the q8 weights on an 8 GB device (the decode/RAM
    // bottleneck). Instead read the bf16 table DIRECTLY from the mmap (zero-copy)
    // and dequant-on-the-fly for both the input lookup and the LM head. Fall back
    // to a one-time f32 decode only if the embedding isn't a contiguous bf16 raw
    // tensor (e.g. a different codec/dtype).
    let embed_meta = model.tensor(&prepared.embedding_weight)?.clone();
    let vocab_size = embed_meta.shape.first().copied().unwrap_or(0) as usize;
    let embed_id = embed_meta.tensor_id;
    let bf16_direct = embed_meta.dtype == DType::Bf16
        && model.with_raw_tensor(embed_id, |_| Ok::<(), RuntimeError>(()))?.is_some();
    let embedding_f32: Option<Vec<f32>> = if bf16_direct {
        None
    } else {
        Some(model.decode_tensor(&prepared.embedding_weight, budget)?.data)
    };

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

        let mut hidden_states = gemma_embed_input(
            model,
            embedding_f32.as_deref(),
            embed_id,
            current_tokens,
            vocab_size,
            hidden,
            build.embed_scale,
        )?;

        let layers_started = crate::q8_kernel_profile_enabled().then(std::time::Instant::now);
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
        if let Some(t) = layers_started {
            eprintln!(
                "[gemma-profile] step {step} seq_len={seq_len}: {} layers {:.0}ms",
                prepared.layers.len(),
                t.elapsed().as_secs_f64() * 1000.0
            );
        }

        hidden_states = rms_norm(&hidden_states, &prepared.final_layernorm, seq_len, hidden, build.rms_norm_eps)?;

        let last_hidden = &hidden_states[(seq_len - 1) * hidden..];
        // R131: parallel LM-head GEMV over the 262k-row vocabulary (bf16-direct
        // from mmap, or f32 fallback).
        let logits = gemma_lm_head(
            model,
            embedding_f32.as_deref(),
            embed_id,
            last_hidden,
            vocab_size,
            hidden,
        )?;

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

/// Input embedding lookup scaled by `embed_scale` — from a resident f32 table
/// (fallback) or bf16-direct from the mmap (default; no 2.68 GB f32 alloc).
fn gemma_embed_input(
    model: &mut LazyRllmModel,
    embedding_f32: Option<&[f32]>,
    embed_id: u64,
    token_ids: &[usize],
    vocab_size: usize,
    hidden: usize,
    embed_scale: f32,
) -> Result<Vec<f32>> {
    if let Some(emb) = embedding_f32 {
        let mut h = embedding_lookup(emb, vocab_size, hidden, token_ids)?;
        for value in h.iter_mut() {
            *value *= embed_scale;
        }
        return Ok(h);
    }
    model
        .with_raw_tensor(embed_id, |bf16| {
            gemma_embed_lookup_bf16(bf16, token_ids, hidden, vocab_size, embed_scale)
        })?
        .ok_or_else(|| {
            RuntimeError::InvalidTensorData("bf16 embedding became non-contiguous".to_string())
        })
}

/// Tied LM head — f32 table (fallback) or bf16-direct from the mmap (default).
fn gemma_lm_head(
    model: &mut LazyRllmModel,
    embedding_f32: Option<&[f32]>,
    embed_id: u64,
    last_hidden: &[f32],
    vocab_size: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    if let Some(emb) = embedding_f32 {
        return Ok(lm_head_logits_parallel(last_hidden, emb, vocab_size, hidden));
    }
    model
        .with_raw_tensor(embed_id, |bf16| {
            Ok::<_, RuntimeError>(lm_head_logits_parallel_bf16(
                last_hidden,
                bf16,
                vocab_size,
                hidden,
            ))
        })?
        .ok_or_else(|| {
            RuntimeError::InvalidTensorData("bf16 embedding became non-contiguous".to_string())
        })
}

/// Gather `token_ids` rows from the bf16 embedding table (mmap bytes), dequant
/// bf16→f32, scaled by `embed_scale`. Row-major, 2 bytes/element.
fn gemma_embed_lookup_bf16(
    bf16: &[u8],
    token_ids: &[usize],
    hidden: usize,
    vocab_size: usize,
    embed_scale: f32,
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(token_ids.len() * hidden);
    for &token in token_ids {
        if token >= vocab_size {
            return Err(RuntimeError::Shape(format!(
                "token id {token} out of range for vocab {vocab_size}"
            )));
        }
        let row_base = token * hidden * 2;
        for h in 0..hidden {
            let off = row_base + h * 2;
            let bits = u16::from_le_bytes([bf16[off], bf16[off + 1]]);
            out.push(bf16_to_f32(bits) * embed_scale);
        }
    }
    Ok(out)
}

