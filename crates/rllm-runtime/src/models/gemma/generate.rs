// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

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

/// Capacity-bound (RLLM_STREAM_EMBEDDING) tied LM head: compute logits by streaming
/// the embedding `[vocab, hidden]` as an output projection (`logits = embedding ·
/// last_hidden`) WITHOUT holding the 604 MB bf16 table resident. Reuses the same
/// fused-bf16 streaming kernel as the body projections (R161), so an rANS/bit-plane
/// embedding decodes per chunk and never materializes — peak resident stays near the
/// compressed size. Bit-identical weights to the resident path (lossless).
pub(super) fn gemma_lm_head_streaming(
    model: &mut LazyRllmModel,
    embedding_weight: &str,
    last_hidden: &[f32],
    vocab_size: usize,
    hidden: usize,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    project(model, embedding_weight, last_hidden, 1, hidden, vocab_size, budget)
}

/// Capacity-bound input embedding: gather `token_ids` rows from the tied embedding by
/// decoding ONLY the chunk(s) that contain them (peak transient = one chunk), then
/// dequant bf16→f32 scaled by `embed_scale`. No resident table — the companion of
/// [`gemma_lm_head_streaming`]. Rows never straddle a chunk boundary (the packer pads
/// each chunk to a whole number of rows), so a row lives entirely in one chunk.
pub(super) fn gemma_embed_input_streaming(
    model: &mut LazyRllmModel,
    embedding_weight: &str,
    token_ids: &[usize],
    hidden: usize,
    embed_scale: f32,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let tensor = model.tensor(embedding_weight)?.clone();
    let row_bytes = hidden * 2; // bf16 = 2 bytes/element
    let mut out = vec![0.0f32; token_ids.len() * hidden];

    let mut chunks = model.chunks_for_tensor(tensor.tensor_id).to_vec();
    chunks.sort_by_key(|chunk| chunk.chunk_offset_in_tensor);
    let mut byte_offset = 0usize;
    for chunk in chunks {
        let chunk_start = byte_offset;
        let chunk_len = chunk.uncompressed_size as usize;
        byte_offset += chunk_len;
        let chunk_end = chunk_start + chunk_len;

        // (output index, byte offset within this chunk) for each requested row here.
        let hits: Vec<(usize, usize)> = token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &row)| {
                let row_start = row.checked_mul(row_bytes)?;
                (row_start >= chunk_start && row_start + row_bytes <= chunk_end)
                    .then_some((i, row_start - chunk_start))
            })
            .collect();
        if hits.is_empty() {
            continue; // skip the decode entirely — only touch chunks we need
        }

        model.with_decoded_chunk(chunk.chunk_id, budget, |bytes, _budget| {
            for (out_idx, local) in &hits {
                let dst = &mut out[out_idx * hidden..(out_idx + 1) * hidden];
                for (h, value) in dst.iter_mut().enumerate() {
                    let lo = bytes[local + 2 * h];
                    let hi = bytes[local + 2 * h + 1];
                    let bits = (u16::from_le_bytes([lo, hi]) as u32) << 16;
                    *value = f32::from_bits(bits) * embed_scale;
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
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
    // Profile (gated): time attention vs MLP on a representative layer (0) to
    // locate the decode cost without 34 lines/token of noise.
    let profile = crate::q8_kernel_profile_enabled() && runtime.layer_index == 0;
    let attn_t = profile.then(std::time::Instant::now);
    let attn_delta =
        gemma_attention_sublayer(model, input, names, norms, build, runtime, budget, cache)?;
    let attn_ms = attn_t.map(|t| t.elapsed().as_secs_f64() * 1000.0);
    add_inplace(&mut residual, &attn_delta)?;
    let mlp_t = profile.then(std::time::Instant::now);
    let mlp_delta = gemma_mlp_sublayer(model, &residual, names, norms, build, runtime, budget)?;
    if let (Some(a), Some(t)) = (attn_ms, mlp_t) {
        eprintln!(
            "[gemma-profile] layer0 attn {a:.1}ms mlp {:.1}ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }
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
