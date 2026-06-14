use crate::models::llama::model::{
    OwnedLlamaStreamingBlockParameters, OwnedLlamaStreamingBlockTensorNames,
};
use crate::rotary::{
    apply_llama_rotary_inplace, KvAttentionConfig, KvCache, RotaryEmbeddingConfig,
};
use crate::{
    ops::{add_inplace, rms_norm, silu_inplace},
    scaled_dot_product_attention_with_cache, streaming_tile_linear_from_model, LazyRllmModel,
    MemoryBudget, RamaTransformerPhaseTimings, Result, StreamingLinearConfig,
    StreamingTileLinearConfig, DEFAULT_STREAMING_TILE_ELEMENTS,
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
    streaming_llama_transformer_block_with_timing(
        model, input, names, params, config, budget, cache, None,
    )
}

pub fn streaming_llama_transformer_block_with_timing(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: &OwnedLlamaStreamingBlockTensorNames,
    params: &OwnedLlamaStreamingBlockParameters,
    config: LlamaStreamingBlockConfig,
    budget: &mut MemoryBudget,
    cache: Option<&mut KvCache>,
    mut timing: Option<&mut RamaTransformerPhaseTimings>,
) -> Result<Vec<f32>> {
    if let Some(timing) = timing.as_deref_mut() {
        timing.profiled_layers = timing.profiled_layers.saturating_add(1);
    }
    let mut residual = input.to_vec();

    let started = Instant::now();
    let attention_input = rms_norm(
        input,
        &params.input_layernorm_weight,
        config.seq_len,
        config.hidden_size,
        config.rms_norm_eps,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.attention_norm_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

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

    let started = Instant::now();
    let mut q = streaming_tile_linear_from_model(
        model,
        &names.q_weight,
        &attention_input,
        None,
        q_config,
        budget,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.q_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    let mut k = streaming_tile_linear_from_model(
        model,
        &names.k_weight,
        &attention_input,
        None,
        kv_config,
        budget,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.k_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    let v = streaming_tile_linear_from_model(
        model,
        &names.v_weight,
        &attention_input,
        None,
        kv_config,
        budget,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.v_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let rope_config = RotaryEmbeddingConfig {
        seq_len: config.seq_len,
        num_heads: config.q_heads,
        head_dim: config.head_dim,
        rotary_dim: config.head_dim,
        base: config.rope_theta,
        position_offset: config.position_offset,
    };
    let started = Instant::now();
    apply_llama_rotary_inplace(&mut q, &mut k, config.q_heads, config.kv_heads, rope_config)?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.rotary_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let attn_config = KvAttentionConfig {
        query_len: config.seq_len,
        num_heads: config.q_heads,
        kv_heads: config.kv_heads,
        head_dim: config.head_dim,
        causal: config.causal,
    };

    let started = Instant::now();
    let attn_out =
        scaled_dot_product_attention_with_cache(&q, &k, &v, cache.as_deref(), attn_config)?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.attention_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    if let Some(c) = cache {
        let started = Instant::now();
        c.append(&k, &v, config.seq_len)?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.kv_append_ms += started.elapsed().as_secs_f64() * 1000.0;
        }
    }

    let o_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.q_heads * config.head_dim,
            out_features: config.hidden_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let started = Instant::now();
    let o = streaming_tile_linear_from_model(
        model,
        &names.o_weight,
        &attn_out,
        None,
        o_config,
        budget,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.o_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    add_inplace(&mut residual, &o)?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.attention_residual_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    let mlp_input = rms_norm(
        &residual,
        &params.post_attention_layernorm_weight,
        config.seq_len,
        config.hidden_size,
        config.rms_norm_eps,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.mlp_norm_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let mlp_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.hidden_size,
            out_features: config.intermediate_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let started = Instant::now();
    let mut gate = streaming_tile_linear_from_model(
        model,
        &names.gate_weight,
        &mlp_input,
        None,
        mlp_config,
        budget,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.gate_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    silu_inplace(&mut gate);
    if let Some(timing) = timing.as_deref_mut() {
        timing.activation_multiply_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    let up = streaming_tile_linear_from_model(
        model,
        &names.up_weight,
        &mlp_input,
        None,
        mlp_config,
        budget,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.up_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    for (g, u) in gate.iter_mut().zip(up.iter()) {
        *g *= *u;
    }
    if let Some(timing) = timing.as_deref_mut() {
        timing.activation_multiply_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let down_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.intermediate_size,
            out_features: config.hidden_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let started = Instant::now();
    let down = streaming_tile_linear_from_model(
        model,
        &names.down_weight,
        &gate,
        None,
        down_config,
        budget,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.down_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    add_inplace(&mut residual, &down)?;
    if let Some(timing) = timing {
        timing.mlp_residual_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    Ok(residual)
}
