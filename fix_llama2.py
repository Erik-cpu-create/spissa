import sys

# 1. Re-generate generate.rs
with open("crates/rllm-runtime/src/llama/generate.rs", "w") as f:
    f.write("""use crate::llama::model::{OwnedLlamaStreamingBlockParameters, OwnedLlamaStreamingBlockTensorNames};
use crate::rotary::{apply_llama_rotary_inplace, KvAttentionConfig, KvCache, RotaryEmbeddingConfig};
use crate::{
    ops::{add_inplace, rms_norm, silu_inplace},
    scaled_dot_product_attention_with_cache, streaming_tile_linear_from_model, LazyRllmModel,
    MemoryBudget, Result, RuntimeError, StreamingTileLinearConfig, StreamingLinearConfig, DEFAULT_STREAMING_TILE_ELEMENTS
};
use std::time::Instant;

pub struct LlamaStreamingBlockConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub causal: bool,
    pub position_offset: usize,
}

pub fn streaming_llama_transformer_block(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: &OwnedLlamaStreamingBlockTensorNames,
    params: &OwnedLlamaStreamingBlockParameters,
    config: LlamaStreamingBlockConfig,
    budget: &mut MemoryBudget,
    cache: Option<&mut KvCache>,
) -> Result<Vec<f32>> {
    let mut residual = input.to_vec();

    let attention_input = rms_norm(
        input,
        &params.input_layernorm_weight,
        config.seq_len,
        config.hidden_size,
        config.rms_norm_eps,
    )?;

    let q_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.hidden_size,
            out_features: config.q_heads * config.head_dim,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let kv_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.hidden_size,
            out_features: config.kv_heads * config.head_dim,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };

    let mut q = streaming_tile_linear_from_model(model, &names.q_weight, &attention_input, None, q_config, budget)?;
    let mut k = streaming_tile_linear_from_model(model, &names.k_weight, &attention_input, None, kv_config, budget)?;
    let mut v = streaming_tile_linear_from_model(model, &names.v_weight, &attention_input, None, kv_config, budget)?;

    let rope_config = RotaryEmbeddingConfig {
        seq_len: config.seq_len,
        num_heads: config.q_heads,
        head_dim: config.head_dim,
        rotary_dim: config.head_dim,
        base: config.rope_theta,
        position_offset: config.position_offset,
    };
    apply_llama_rotary_inplace(&mut q, &mut k, config.q_heads, config.kv_heads, rope_config)?;

    if let Some(c) = cache.as_deref_mut() {
        c.append(&k, &v, config.seq_len)?;
    }

    let k_broadcasted = if config.q_heads != config.kv_heads {
        broadcast_gqa(&k, config.seq_len, config.kv_heads, config.q_heads, config.head_dim)?
    } else {
        k
    };
    let v_broadcasted = if config.q_heads != config.kv_heads {
        broadcast_gqa(&v, config.seq_len, config.kv_heads, config.q_heads, config.head_dim)?
    } else {
        v
    };

    let cache_broadcasted = if let Some(c) = cache.as_ref() {
        if config.q_heads != config.kv_heads {
            let b_k = broadcast_gqa(c.keys(), c.len(), config.kv_heads, config.q_heads, config.head_dim)?;
            let b_v = broadcast_gqa(c.values(), c.len(), config.kv_heads, config.q_heads, config.head_dim)?;
            let mut new_c = KvCache::new(config.q_heads, config.head_dim, c.max_seq_len())?;
            new_c.append(&b_k, &b_v, c.len())?;
            Some(new_c)
        } else {
            None
        }
    } else {
        None
    };

    let attn_config = KvAttentionConfig {
        query_len: config.seq_len,
        num_heads: config.q_heads,
        head_dim: config.head_dim,
        causal: config.causal,
    };

    let cache_ref = cache_broadcasted.as_ref().or_else(|| {
        if config.q_heads == config.kv_heads {
            None
        } else {
            None
        }
    });

    let attn_out = if let Some(c) = cache_ref {
        scaled_dot_product_attention_with_cache(&q, &k_broadcasted, &v_broadcasted, Some(c), attn_config)?
    } else if config.q_heads == config.kv_heads && cache.is_some() {
        scaled_dot_product_attention_with_cache(&q, &k_broadcasted, &v_broadcasted, cache.as_deref(), attn_config)?
    } else {
        scaled_dot_product_attention_with_cache(&q, &k_broadcasted, &v_broadcasted, None, attn_config)?
    };

    let o_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.q_heads * config.head_dim,
            out_features: config.hidden_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let o = streaming_tile_linear_from_model(model, &names.o_weight, &attn_out, None, o_config, budget)?;

    add_inplace(&mut residual, &o)?;

    let mlp_input = rms_norm(
        &residual,
        &params.post_attention_layernorm_weight,
        config.seq_len,
        config.hidden_size,
        config.rms_norm_eps,
    )?;

    let mlp_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.hidden_size,
            out_features: config.intermediate_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let mut gate = streaming_tile_linear_from_model(model, &names.gate_weight, &mlp_input, None, mlp_config, budget)?;
    silu_inplace(&mut gate);
    let up = streaming_tile_linear_from_model(model, &names.up_weight, &mlp_input, None, mlp_config, budget)?;
    
    for (g, u) in gate.iter_mut().zip(up.iter()) {
        *g *= *u;
    }

    let down_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.intermediate_size,
            out_features: config.hidden_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let down = streaming_tile_linear_from_model(model, &names.down_weight, &gate, None, down_config, budget)?;

    add_inplace(&mut residual, &down)?;

    Ok(residual)
}

fn broadcast_gqa(
    tensor: &[f32],
    seq_len: usize,
    kv_heads: usize,
    q_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if q_heads % kv_heads != 0 {
        return Err(RuntimeError::Shape(format!("q_heads {} must be a multiple of kv_heads {}", q_heads, kv_heads)));
    }
    let repeats = q_heads / kv_heads;
    let expected_in = seq_len * kv_heads * head_dim;
    if tensor.len() != expected_in {
        return Err(RuntimeError::Shape(format!("broadcast shape mismatch: expected {}, got {}", expected_in, tensor.len())));
    }
    let expected_out = seq_len * q_heads * head_dim;
    let mut out = Vec::with_capacity(expected_out);
    
    for pos in 0..seq_len {
        for kv_head in 0..kv_heads {
            let start = (pos * kv_heads + kv_head) * head_dim;
            let head_slice = &tensor[start..start + head_dim];
            for _ in 0..repeats {
                out.extend_from_slice(head_slice);
            }
        }
    }
    Ok(out)
}
""")

# 2. Re-generate api.rs
with open("crates/rllm-runtime/src/llama/api.rs", "w") as f:
    f.write("""use crate::llama::model::*;
use crate::llama::generate::{streaming_llama_transformer_block, LlamaStreamingBlockConfig};
use crate::rotary::KvCache;
use crate::{
    ops::{embedding_lookup, rms_norm, sample_argmax, sample_top_p},
    streaming_tile_linear_from_model, LazyRllmModel, MemoryBudget, Result, RuntimeError,
    StreamingTileLinearConfig, StreamingLinearConfig, DEFAULT_STREAMING_TILE_ELEMENTS
};
use rllm_container::GlobalMetadata;
use std::time::Instant;

fn decode_vector_tensor(
    model: &mut LazyRllmModel,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>> {
    let mut budget = MemoryBudget::unbounded();
    let tensor = model.decode_tensor(name, &mut budget)?;
    if tensor.shape != [expected_len] {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected [{expected_len}]",
            tensor.shape
        )));
    }
    Ok(tensor.data)
}

pub fn prepare_llama_rama_layer_decode_transformer_from_metadata(
    model: &mut LazyRllmModel,
    generation: LlamaRamaGenerationConfig,
) -> Result<LayerDecodedLlamaRamaTransformer> {
    let metadata = model.metadata();
    let model_config = metadata.model_config.as_ref().unwrap();
    let num_heads = model_config.num_attention_heads.unwrap() as usize;
    let num_key_value_heads = model_config.num_key_value_heads.unwrap_or(num_heads as u64) as usize;
    let hidden_size = model_config.hidden_size.unwrap() as usize;
    let intermediate_size = model_config.intermediate_size.unwrap() as usize;
    let head_dim = hidden_size / num_heads;
    let max_seq_len = generation.max_seq_len.unwrap_or(2048);

    let build = LlamaEchoBuildConfig {
        max_new_tokens: generation.max_new_tokens,
        max_seq_len: Some(max_seq_len),
        num_heads,
        num_key_value_heads,
        causal: generation.causal,
        rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-5),
        rope_theta: model_config.rope_theta.unwrap_or(10000.0),
        sampling: generation.sampling,
    };

    let embedding_weight = "model.embed_tokens.weight".to_string();
    let lm_head_weight = "lm_head.weight".to_string();
    
    let final_layernorm_weight = decode_vector_tensor(model, "model.norm.weight", hidden_size)?;
    
    let mut num_layers = 0;
    while model.tensor(&format!("model.layers.{num_layers}.input_layernorm.weight")).is_ok() {
        num_layers += 1;
    }

    let mut layers = Vec::new();
    for i in 0..num_layers {
        layers.push(OwnedLlamaStreamingBlockTensorNames {
            q_weight: format!("model.layers.{i}.self_attn.q_proj.weight"),
            k_weight: format!("model.layers.{i}.self_attn.k_proj.weight"),
            v_weight: format!("model.layers.{i}.self_attn.v_proj.weight"),
            o_weight: format!("model.layers.{i}.self_attn.o_proj.weight"),
            gate_weight: format!("model.layers.{i}.mlp.gate_proj.weight"),
            up_weight: format!("model.layers.{i}.mlp.up_proj.weight"),
            down_weight: format!("model.layers.{i}.mlp.down_proj.weight"),
        });
    }

    Ok(LayerDecodedLlamaRamaTransformer {
        config: build,
        embedding_weight,
        layers,
        lm_head_weight,
        final_layernorm_weight,
        pinned_lm_head_weight: None,
        resident_parameter_bytes: 0,
        max_layer_parameter_bytes: 0,
    })
}

pub fn rama_layer_decoded_llama_transformer_generate_from_model(
    model: &mut LazyRllmModel,
    prepared: &LayerDecodedLlamaRamaTransformer,
    prompt_token_ids: &[usize],
    budget: &mut MemoryBudget,
    options: LlamaRamaGenerationOptions,
) -> Result<LlamaTextGenerationResult> {
    let mut token_ids = prompt_token_ids.to_vec();
    let mut generated_token_ids = Vec::new();

    let hidden_size = model.metadata().model_config.as_ref().unwrap().hidden_size.unwrap() as usize;
    let intermediate_size = model.metadata().model_config.as_ref().unwrap().intermediate_size.unwrap() as usize;
    let head_dim = hidden_size / prepared.config.num_heads;
    
    let mut caches = Vec::new();
    for _ in 0..prepared.layers.len() {
        caches.push(KvCache::new(prepared.config.num_key_value_heads, head_dim, prepared.config.max_seq_len.unwrap())?);
    }

    for step in 0..prepared.config.max_new_tokens {
        let current_tokens = if step == 0 { prompt_token_ids } else { &generated_token_ids[generated_token_ids.len() - 1..] };
        let seq_len = current_tokens.len();
        
        let position_offset = token_ids.len() - seq_len;
        
        // 1. Embedding
        let embedding_data = model.decode_tensor(&prepared.embedding_weight, budget)?.data;
        let vocab_size = embedding_data.len() / hidden_size;
        let mut hidden = embedding_lookup(&embedding_data, vocab_size, hidden_size, current_tokens)?;
        
        // 2. Layers
        for (i, layer_names) in prepared.layers.iter().enumerate() {
            let input_layernorm_weight = decode_vector_tensor(model, &format!("model.layers.{i}.input_layernorm.weight"), hidden_size)?;
            let post_attention_layernorm_weight = decode_vector_tensor(model, &format!("model.layers.{i}.post_attention_layernorm.weight"), hidden_size)?;
            let params = OwnedLlamaStreamingBlockParameters {
                input_layernorm_weight,
                post_attention_layernorm_weight,
            };
            let config = LlamaStreamingBlockConfig {
                seq_len,
                hidden_size,
                q_heads: prepared.config.num_heads,
                kv_heads: prepared.config.num_key_value_heads,
                head_dim,
                intermediate_size,
                rms_norm_eps: prepared.config.rms_norm_eps,
                rope_theta: prepared.config.rope_theta,
                causal: prepared.config.causal,
                position_offset,
            };
            hidden = streaming_llama_transformer_block(model, &hidden, layer_names, &params, config, budget, Some(&mut caches[i]))?;
        }

        // 3. Final norm
        hidden = rms_norm(&hidden, &prepared.final_layernorm_weight, seq_len, hidden_size, prepared.config.rms_norm_eps)?;

        // 4. LM Head (only on last token for generation)
        let last_hidden = &hidden[(seq_len - 1) * hidden_size..];
        let lm_config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: hidden_size,
                out_features: vocab_size,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        };
        let logits = streaming_tile_linear_from_model(model, &prepared.lm_head_weight, last_hidden, None, lm_config, budget)?;

        // 5. Sample
        let next_token = match prepared.config.sampling {
            crate::StreamingSamplingConfig::Argmax => sample_argmax(&logits)?,
            crate::StreamingSamplingConfig::TopP { temperature, top_p, seed } => sample_top_p(&logits, temperature, top_p, seed)?,
        };

        token_ids.push(next_token);
        generated_token_ids.push(next_token);
    }

    Ok(LlamaTextGenerationResult {
        prompt_token_ids: prompt_token_ids.to_vec(),
        generated_token_ids,
        token_ids,
        text: String::new(),
        generated_text: String::new(),
        context_echo_bytes: 0,
    })
}
""")
