//! Qwen3.5 text adapter: build a `PreparedQwenTransformer` from `.rllm` metadata and
//! run greedy/top-p generation with heterogeneous per-layer dispatch.

use crate::models::qwen::generate::validate_prepared;
use crate::models::qwen::model::{
    PreparedQwenTransformer, QwenBuildConfig, QwenLayerKind, QwenLayerParams, QwenLayerTensors,
};
use crate::models::qwen::session::QwenSession;
use crate::ops::embedding_lookup;
use crate::{LazyRllmModel, MemoryBudget, Result, RuntimeError, StreamingSamplingConfig};
use rllm_container::ModelConfigMetadata;

#[derive(Debug, Clone, Copy)]
pub struct QwenGenerationConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: Option<usize>,
    pub causal: bool,
    pub sampling: StreamingSamplingConfig,
}

fn require_config<'a>(model: &'a LazyRllmModel) -> Result<&'a ModelConfigMetadata> {
    model.metadata().model_config.as_ref().ok_or_else(|| {
        RuntimeError::InvalidTensorData(
            "qwen generation requires persisted model_config metadata; repack with --config <config.json>".to_string(),
        )
    })
}

fn req_usize(name: &str, value: Option<u64>) -> Result<usize> {
    let v = value.ok_or_else(|| {
        RuntimeError::InvalidTensorData(format!("qwen model_config is missing required field {name}"))
    })?;
    usize::try_from(v)
        .map_err(|_| RuntimeError::Shape(format!("qwen model_config field {name}={v} overflows usize")))
}

fn decode_vec(model: &mut LazyRllmModel, name: &str, expected: usize) -> Result<Vec<f32>> {
    let mut budget = MemoryBudget::unbounded();
    let tensor = model.decode_tensor(name, &mut budget)?;
    if tensor.data.len() != expected {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} has {} elements, expected {expected}",
            tensor.data.len()
        )));
    }
    Ok(tensor.data)
}

/// Decode a `Qwen3_5RMSNorm` weight and pre-bake the `(1 + weight)` convention into
/// it (Qwen3.5/Gemma style: `out = _norm(x) * (1 + weight)`), so the standard
/// `ops::rms_norm` (which multiplies by the weight directly) is exact. NOTE: this is
/// NOT applied to the gated DeltaNet norm (`Qwen3_5RMSNormGated`), which uses the
/// weight directly.
fn decode_norm_1plus(model: &mut LazyRllmModel, name: &str, expected: usize) -> Result<Vec<f32>> {
    let mut v = decode_vec(model, name, expected)?;
    v.iter_mut().for_each(|x| *x += 1.0);
    Ok(v)
}

/// Build the per-layer operator schedule from `layer_types`, falling back to the
/// `full_attention_interval` rule (`idx % interval == interval-1`).
fn layer_kinds(config: &ModelConfigMetadata, num_layers: usize) -> Vec<QwenLayerKind> {
    if let Some(types) = config.layer_types.as_ref() {
        if types.len() == num_layers {
            return types
                .iter()
                .map(|t| {
                    if t == "full_attention" {
                        QwenLayerKind::FullAttention
                    } else {
                        QwenLayerKind::LinearAttention
                    }
                })
                .collect();
        }
    }
    let interval = config
        .full_attention_interval
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .unwrap_or(4);
    (0..num_layers)
        .map(|i| {
            if i % interval == interval - 1 {
                QwenLayerKind::FullAttention
            } else {
                QwenLayerKind::LinearAttention
            }
        })
        .collect()
}

/// Resolve a `QwenBuildConfig` from persisted `config.json` metadata (with Qwen3.5
/// defaults for any field a non-Qwen checkpoint omits).
fn build_qwen_config(
    config: &ModelConfigMetadata,
    generation: QwenGenerationConfig,
) -> Result<QwenBuildConfig> {
    let hidden_size = req_usize("hidden_size", config.hidden_size)?;
    let intermediate_size = req_usize("intermediate_size", config.intermediate_size)?;
    let num_heads = req_usize("num_attention_heads", config.num_attention_heads)?;
    let num_kv_heads = config
        .num_key_value_heads
        .map(|v| req_usize("num_key_value_heads", Some(v)))
        .transpose()?
        .unwrap_or(num_heads);
    let head_dim = config
        .head_dim
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(hidden_size / num_heads.max(1));
    let partial = config.partial_rotary_factor.unwrap_or(1.0);
    let mut rotary_dim = ((head_dim as f32) * partial) as usize;
    if !rotary_dim.is_multiple_of(2) {
        rotary_dim -= 1;
    }
    let linear_num_heads = config
        .linear_num_key_heads
        .or(config.linear_num_value_heads)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(num_heads);
    let linear_key_dim = config
        .linear_key_head_dim
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(head_dim);
    let linear_value_dim = config
        .linear_value_head_dim
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(linear_key_dim);
    let conv_kernel = config
        .linear_conv_kernel_dim
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(4);
    let max_seq_len = generation
        .max_seq_len
        .or_else(|| config.max_position_embeddings.and_then(|v| usize::try_from(v).ok()))
        .unwrap_or(4096);

    Ok(QwenBuildConfig {
        max_new_tokens: generation.max_new_tokens,
        max_seq_len,
        hidden_size,
        intermediate_size,
        rms_norm_eps: config.rms_norm_eps.unwrap_or(1e-6),
        causal: generation.causal,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        rope_theta: config.rope_theta.unwrap_or(10_000.0),
        attn_output_gate: config.attn_output_gate.unwrap_or(false),
        linear_num_heads,
        linear_key_dim,
        linear_value_dim,
        conv_kernel,
        sampling: generation.sampling,
    })
}

pub fn prepare_qwen_transformer_from_metadata(
    model: &mut LazyRllmModel,
    generation: QwenGenerationConfig,
) -> Result<PreparedQwenTransformer> {
    let config = require_config(model)?.clone();
    let build = build_qwen_config(&config, generation)?;
    let hidden_size = build.hidden_size;
    let head_dim = build.head_dim;
    let linear_num_heads = build.linear_num_heads;
    let linear_value_dim = build.linear_value_dim;
    let conv_kernel = build.conv_kernel;

    // Count layers by probing the per-layer input layernorm.
    let mut num_layers = 0;
    while model
        .tensor(&format!("model.layers.{num_layers}.input_layernorm.weight"))
        .is_ok()
    {
        num_layers += 1;
    }
    if num_layers == 0 {
        return Err(RuntimeError::Shape(
            "qwen model requires at least one layer".to_string(),
        ));
    }
    let kinds = layer_kinds(&config, num_layers);

    let embedding_weight = "model.embed_tokens.weight".to_string();
    // tie_word_embeddings: lm_head reuses the embedding table.
    let lm_head_weight = if model.tensor("lm_head.weight").is_ok() {
        "lm_head.weight".to_string()
    } else {
        embedding_weight.clone()
    };
    let final_norm = decode_norm_1plus(model, "model.norm.weight", hidden_size)?;

    let mut layers = Vec::with_capacity(num_layers);
    let mut layer_params = Vec::with_capacity(num_layers);
    let conv_channels = build.linear_conv_channels();
    for i in 0..num_layers {
        let p = format!("model.layers.{i}");
        let mut tensors = QwenLayerTensors {
            gate_proj: format!("{p}.mlp.gate_proj.weight"),
            up_proj: format!("{p}.mlp.up_proj.weight"),
            down_proj: format!("{p}.mlp.down_proj.weight"),
            ..Default::default()
        };
        let mut params = QwenLayerParams {
            kind_full_attention: kinds[i] == QwenLayerKind::FullAttention,
            input_layernorm: decode_norm_1plus(
                model,
                &format!("{p}.input_layernorm.weight"),
                hidden_size,
            )?,
            post_attention_layernorm: decode_norm_1plus(
                model,
                &format!("{p}.post_attention_layernorm.weight"),
                hidden_size,
            )?,
            ..Default::default()
        };
        match kinds[i] {
            QwenLayerKind::FullAttention => {
                tensors.q_proj = format!("{p}.self_attn.q_proj.weight");
                tensors.k_proj = format!("{p}.self_attn.k_proj.weight");
                tensors.v_proj = format!("{p}.self_attn.v_proj.weight");
                tensors.o_proj = format!("{p}.self_attn.o_proj.weight");
                params.q_norm =
                    decode_norm_1plus(model, &format!("{p}.self_attn.q_norm.weight"), head_dim)?;
                params.k_norm =
                    decode_norm_1plus(model, &format!("{p}.self_attn.k_norm.weight"), head_dim)?;
            }
            QwenLayerKind::LinearAttention => {
                tensors.in_proj_qkv = format!("{p}.linear_attn.in_proj_qkv.weight");
                tensors.in_proj_a = format!("{p}.linear_attn.in_proj_a.weight");
                tensors.in_proj_b = format!("{p}.linear_attn.in_proj_b.weight");
                tensors.in_proj_z = format!("{p}.linear_attn.in_proj_z.weight");
                tensors.out_proj = format!("{p}.linear_attn.out_proj.weight");
                params.a_log = decode_vec(model, &format!("{p}.linear_attn.A_log"), linear_num_heads)?;
                params.dt_bias =
                    decode_vec(model, &format!("{p}.linear_attn.dt_bias"), linear_num_heads)?;
                params.conv1d = decode_vec(
                    model,
                    &format!("{p}.linear_attn.conv1d.weight"),
                    conv_channels * conv_kernel,
                )?;
                params.linear_norm =
                    decode_vec(model, &format!("{p}.linear_attn.norm.weight"), linear_value_dim)?;
            }
        }
        layers.push(tensors);
        layer_params.push(params);
    }

    let prepared = PreparedQwenTransformer {
        config: build,
        embedding_weight,
        lm_head_weight,
        final_norm,
        layers,
        layer_params,
    };
    validate_prepared(&prepared)?;
    Ok(prepared)
}

/// Gather `tokens` rows from the (bf16) embedding table directly from the mmap,
/// converting bf16→f32, instead of decoding the whole 248k×2048 table to f32 (~1 GB)
/// just to index a few rows per step. Falls back to a full decode for a non-raw
/// (compressed) embedding tensor.
pub(crate) fn gather_embedding_rows(
    model: &mut LazyRllmModel,
    name: &str,
    hidden: usize,
    tokens: &[usize],
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let tensor = model.tensor(name)?.clone();
    if tensor.dtype == rllm_container::DType::Bf16 {
        let mut out = vec![0.0f32; tokens.len() * hidden];
        let handled = model.with_raw_tensor(tensor.tensor_id, |bytes| {
            for (i, &tok) in tokens.iter().enumerate() {
                let row = tok * hidden * 2;
                if row + hidden * 2 > bytes.len() {
                    return Err(RuntimeError::Shape(format!(
                        "embedding row {tok} out of range ({} rows)",
                        bytes.len() / (hidden * 2)
                    )));
                }
                for h in 0..hidden {
                    let bits = u16::from_le_bytes([bytes[row + h * 2], bytes[row + h * 2 + 1]]);
                    out[i * hidden + h] = crate::bf16_to_f32(bits);
                }
            }
            Ok(())
        })?;
        if handled.is_some() {
            return Ok(out);
        }
    }
    // Fallback: non-raw (compressed) embedding — decode once and gather.
    let full = model.decode_tensor(name, budget)?.data;
    let vocab = full.len() / hidden;
    embedding_lookup(&full, vocab, hidden, tokens)
}

/// Greedy/top-p generation. Calls `on_token(token_id) -> continue?` for each new token.
pub fn qwen_generate_from_model(
    model: &mut LazyRllmModel,
    prepared: &PreparedQwenTransformer,
    prompt_token_ids: &[usize],
    budget: &mut MemoryBudget,
    on_token: &mut dyn FnMut(usize) -> bool,
) -> Result<Vec<usize>> {
    // One-shot generation = a fresh session generating once. `prepared` is cloned so the
    // session can own it (callers that want a persistent multi-turn session build a
    // `QwenSession` directly).
    let max_new = prepared.config.max_new_tokens;
    let sampling = prepared.config.sampling;
    let mut session = QwenSession::new(model, prepared.clone())?;
    session.generate(model, prompt_token_ids, max_new, sampling, &[], budget, on_token)
}
