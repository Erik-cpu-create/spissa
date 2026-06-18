use crate::models::gemma::model::{
    is_global_layer, GemmaBlockTensorNames, GemmaBuildConfig, GemmaLayerNorms,
};
use crate::rotary::{apply_gemma_rotary_inplace, KvAttentionConfig, KvCache, RotaryEmbeddingConfig};
use crate::{
    ops::{add_inplace, gelu_inplace, rms_norm},
    scaled_dot_product_attention_with_cache, streaming_tile_linear_from_model, LazyRllmModel,
    MemoryBudget, Result, StreamingLinearConfig, StreamingTileLinearConfig,
    DEFAULT_STREAMING_TILE_ELEMENTS,
};

/// Per-call dynamic state for a single Gemma block forward.
#[derive(Debug, Clone, Copy)]
pub struct GemmaBlockRuntime {
    pub seq_len: usize,
    pub position_offset: usize,
    pub layer_index: usize,
}

/// Stream a single dense projection `input[batch, in] · weight[out, in]^T`
/// from the model, dispatching to the fast q8 / raw tile kernels.
fn project(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_tile_linear_from_model(
        model,
        weight_name,
        input,
        None,
        StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch,
                in_features,
                out_features,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        },
        budget,
    )
}

/// One Gemma 3 decoder layer with the sandwich-norm residual structure:
///
/// ```text
/// h = x + post_attention_layernorm(attn(input_layernorm(x)))
/// out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
/// ```
///
/// All RMSNorm weights in `norms` are pre-baked with Gemma's `(1 + weight)`
/// convention so the standard [`crate::ops::rms_norm`] applies directly.
pub fn streaming_gemma_transformer_block(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: &GemmaBlockTensorNames,
    norms: &GemmaLayerNorms,
    build: &GemmaBuildConfig,
    runtime: GemmaBlockRuntime,
    budget: &mut MemoryBudget,
    cache: Option<&mut KvCache>,
) -> Result<Vec<f32>> {
    let mut residual = input.to_vec();
    let attn_delta =
        gemma_attention_sublayer(model, input, names, norms, build, runtime, budget, cache)?;
    add_inplace(&mut residual, &attn_delta)?;
    let mlp_delta = gemma_mlp_sublayer(model, &residual, names, norms, build, runtime, budget)?;
    add_inplace(&mut residual, &mlp_delta)?;
    Ok(residual)
}

/// `post_attention_layernorm(o_proj(attn(input_layernorm(x))))`, the value
/// added back to the residual stream. Applies per-head QK-norm before RoPE,
/// dual RoPE, and the `1/sqrt(query_pre_attn_scalar)` attention scale.
#[allow(clippy::too_many_arguments)]
fn gemma_attention_sublayer(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: &GemmaBlockTensorNames,
    norms: &GemmaLayerNorms,
    build: &GemmaBuildConfig,
    runtime: GemmaBlockRuntime,
    budget: &mut MemoryBudget,
    cache: Option<&mut KvCache>,
) -> Result<Vec<f32>> {
    let seq_len = runtime.seq_len;
    let hidden = build.hidden_size;
    let head_dim = build.head_dim;
    let q_heads = build.num_heads;
    let kv_heads = build.num_key_value_heads;

    let attn_input = rms_norm(input, &norms.input_layernorm, seq_len, hidden, build.rms_norm_eps)?;

    let mut q = project(model, &names.q_weight, &attn_input, seq_len, hidden, q_heads * head_dim, budget)?;
    let mut k = project(model, &names.k_weight, &attn_input, seq_len, hidden, kv_heads * head_dim, budget)?;
    let v = project(model, &names.v_weight, &attn_input, seq_len, hidden, kv_heads * head_dim, budget)?;

    // Per-head QK-norm over head_dim. Q/K are laid out [seq, heads, head_dim],
    // i.e. (seq*heads) rows of head_dim — exactly what rms_norm normalizes.
    q = rms_norm(&q, &norms.q_norm, seq_len * q_heads, head_dim, build.rms_norm_eps)?;
    k = rms_norm(&k, &norms.k_norm, seq_len * kv_heads, head_dim, build.rms_norm_eps)?;

    // Dual RoPE: global layers use rope_theta scaled by rope_scaling_factor,
    // local layers use rope_local_base_freq unscaled.
    let (rope_base, position_divisor) =
        if is_global_layer(runtime.layer_index, build.sliding_window_pattern) {
            (build.rope_theta, build.rope_scaling_factor)
        } else {
            (build.rope_local_base_freq, 1.0)
        };
    let rope_config = RotaryEmbeddingConfig {
        seq_len,
        num_heads: q_heads,
        head_dim,
        rotary_dim: head_dim,
        base: rope_base,
        position_offset: runtime.position_offset,
    };
    apply_gemma_rotary_inplace(&mut q, &mut k, q_heads, kv_heads, rope_config, position_divisor)?;

    // Attention scale is 1/sqrt(query_pre_attn_scalar). The shared SDPA bakes in
    // 1/sqrt(head_dim), so fold the residual factor into Q (exactly ×1.0 when
    // query_pre_attn_scalar == head_dim, as on Gemma 3 4B).
    let q_prescale = build.attn_scale * (head_dim as f32).sqrt();
    for value in q.iter_mut() {
        *value *= q_prescale;
    }

    let attn_out = scaled_dot_product_attention_with_cache(
        &q,
        &k,
        &v,
        cache.as_deref(),
        KvAttentionConfig {
            query_len: seq_len,
            num_heads: q_heads,
            kv_heads,
            head_dim,
            causal: build.causal,
        },
    )?;
    if let Some(c) = cache {
        c.append(&k, &v, seq_len)?;
    }

    let attn_proj = project(model, &names.o_weight, &attn_out, seq_len, q_heads * head_dim, hidden, budget)?;
    rms_norm(
        &attn_proj,
        &norms.post_attention_layernorm,
        seq_len,
        hidden,
        build.rms_norm_eps,
    )
}

/// `post_feedforward_layernorm(down_proj(geglu(pre_feedforward_layernorm(h))))`,
/// the value added back to the residual stream. GeGLU uses `gelu_pytorch_tanh`.
fn gemma_mlp_sublayer(
    model: &mut LazyRllmModel,
    residual: &[f32],
    names: &GemmaBlockTensorNames,
    norms: &GemmaLayerNorms,
    build: &GemmaBuildConfig,
    runtime: GemmaBlockRuntime,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let seq_len = runtime.seq_len;
    let hidden = build.hidden_size;
    let intermediate = build.intermediate_size;

    let mlp_input = rms_norm(
        residual,
        &norms.pre_feedforward_layernorm,
        seq_len,
        hidden,
        build.rms_norm_eps,
    )?;
    let mut gate = project(model, &names.gate_weight, &mlp_input, seq_len, hidden, intermediate, budget)?;
    gelu_inplace(&mut gate); // gelu_pytorch_tanh
    let up = project(model, &names.up_weight, &mlp_input, seq_len, hidden, intermediate, budget)?;
    for (g, u) in gate.iter_mut().zip(&up) {
        *g *= *u;
    }
    let down = project(model, &names.down_weight, &gate, seq_len, intermediate, hidden, budget)?;
    rms_norm(
        &down,
        &norms.post_feedforward_layernorm,
        seq_len,
        hidden,
        build.rms_norm_eps,
    )
}
