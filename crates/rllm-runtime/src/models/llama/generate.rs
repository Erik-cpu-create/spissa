use crate::models::llama::model::{
    OwnedLlamaStreamingBlockParameters, OwnedLlamaStreamingBlockTensorNames,
};
use crate::rotary::{
    apply_llama_rotary_inplace, KvAttentionConfig, KvCache, RotaryEmbeddingConfig,
};
use crate::{
    ops::{add_inplace, rms_norm, silu_inplace},
    scaled_dot_product_attention_with_cache, streaming_tile_linear_from_model, LazyRllmModel,
    MemoryBudget, Result, StreamingLinearConfig, StreamingTileLinearConfig,
    DEFAULT_STREAMING_TILE_ELEMENTS,
};

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

    let mut q = streaming_tile_linear_from_model(
        model,
        &names.q_weight,
        &attention_input,
        None,
        q_config,
        budget,
    )?;
    let mut k = streaming_tile_linear_from_model(
        model,
        &names.k_weight,
        &attention_input,
        None,
        kv_config,
        budget,
    )?;
    let v = streaming_tile_linear_from_model(
        model,
        &names.v_weight,
        &attention_input,
        None,
        kv_config,
        budget,
    )?;

    let rope_config = RotaryEmbeddingConfig {
        seq_len: config.seq_len,
        num_heads: config.q_heads,
        head_dim: config.head_dim,
        rotary_dim: config.head_dim,
        base: config.rope_theta,
        position_offset: config.position_offset,
    };
    apply_llama_rotary_inplace(&mut q, &mut k, config.q_heads, config.kv_heads, rope_config)?;

    let attn_config = KvAttentionConfig {
        query_len: config.seq_len,
        num_heads: config.q_heads,
        kv_heads: config.kv_heads,
        head_dim: config.head_dim,
        causal: config.causal,
    };

    let attn_out =
        scaled_dot_product_attention_with_cache(&q, &k, &v, cache.as_deref(), attn_config)?;

    if let Some(c) = cache {
        c.append(&k, &v, config.seq_len)?;
    }

    let o_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.q_heads * config.head_dim,
            out_features: config.hidden_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let o = streaming_tile_linear_from_model(
        model,
        &names.o_weight,
        &attn_out,
        None,
        o_config,
        budget,
    )?;

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
    let mut gate = streaming_tile_linear_from_model(
        model,
        &names.gate_weight,
        &mlp_input,
        None,
        mlp_config,
        budget,
    )?;
    silu_inplace(&mut gate);
    let up = streaming_tile_linear_from_model(
        model,
        &names.up_weight,
        &mlp_input,
        None,
        mlp_config,
        budget,
    )?;

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
    let down = streaming_tile_linear_from_model(
        model,
        &names.down_weight,
        &gate,
        None,
        down_config,
        budget,
    )?;

    add_inplace(&mut residual, &down)?;

    Ok(residual)
}
