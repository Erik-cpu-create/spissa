//! Qwen3.5 per-layer forward blocks.
//!
//! - `qwen_gated_attention_block` (REEGATE): GQA softmax attention with QK-RMSNorm,
//!   partial RoPE, and a per-head sigmoid output gate split per-head from `q_proj`.
//! - `qwen_gated_deltanet_block` (REEDELTA): the Gated-DeltaNet linear-attention
//!   recurrence (depthwise short conv + SiLU, L2-normed + scaled q/k, delta-rule scan
//!   with a per-head decay gate, gated RMSNorm with the z gate). Sequential over tokens,
//!   so one code path serves prefill (seq_len>1) and decode (seq_len==1); the recurrent
//!   `state` persists across calls.
//!
//! Math is grounded verbatim in HF `modeling_qwen3_5.py` (`Qwen3_5GatedDeltaNet`,
//! `Qwen3_5RMSNormGated`, `torch_recurrent_gated_delta_rule`) and validated to produce
//! logits matching the HF reference (top-token + ordering) for the 2B checkpoint.

use crate::models::qwen::model::{
    GatedDeltaNetState, PreparedQwenTransformer, QwenBuildConfig, QwenLayerParams, QwenLayerTensors,
};
use crate::ops::{add_inplace, rms_norm, silu_inplace};
use crate::rotary::{apply_llama_rotary_inplace, KvAttentionConfig, KvCache, RotaryEmbeddingConfig};
use crate::{
    scaled_dot_product_attention_with_cache, streaming_tile_linear_from_model, LazyRllmModel,
    MemoryBudget, Result, RuntimeError, StreamingLinearConfig, StreamingTileLinearConfig,
    DEFAULT_STREAMING_TILE_ELEMENTS,
};

// ----- small scalar math (f32) -----

#[inline]
fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn siluf(x: f32) -> f32 {
    x * sigmoidf(x)
}

#[inline]
fn softplusf(x: f32) -> f32 {
    // ln(1 + e^x), numerically stable for large x.
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// In-place L2 normalization of a `dim`-length slice: `x /= sqrt(sum(x^2) + eps)`.
#[inline]
fn l2norm_inplace(x: &mut [f32], eps: f32) {
    let sumsq: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sumsq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Stream a weight `[out_features, in_features]` from the container and compute
/// `input[batch, in_features] · Wᵀ -> [batch, out_features]` (no bias).
fn project(
    model: &mut LazyRllmModel,
    name: &str,
    input: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let cfg = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch,
            in_features,
            out_features,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    streaming_tile_linear_from_model(model, name, input, None, cfg, budget)
}

/// Dense SwiGLU FFN: `down(silu(gate(x)) * up(x))`.
fn qwen_swiglu_ffn(
    model: &mut LazyRllmModel,
    tensors: &QwenLayerTensors,
    x: &[f32],
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let mut gate = project(
        model,
        &tensors.gate_proj,
        x,
        seq_len,
        hidden_size,
        intermediate_size,
        budget,
    )?;
    let up = project(
        model,
        &tensors.up_proj,
        x,
        seq_len,
        hidden_size,
        intermediate_size,
        budget,
    )?;
    silu_inplace(&mut gate);
    for (g, u) in gate.iter_mut().zip(up.iter()) {
        *g *= *u;
    }
    project(
        model,
        &tensors.down_proj,
        &gate,
        seq_len,
        intermediate_size,
        hidden_size,
        budget,
    )
}

/// One Gated-DeltaNet recurrence step for a single head: decay, delta-rule write,
/// readout. `s` is the persistent `[k_dim, v_dim]` state (`s[k*v_dim + v]`); `q`/`k`
/// are already L2-normed (and `q` scaled by `1/sqrt(k_dim)`); writes the readout to `o`.
#[allow(clippy::too_many_arguments)]
fn delta_rule_head_step(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    g: f32,
    k_dim: usize,
    v_dim: usize,
    o: &mut [f32],
) {
    for val in s.iter_mut() {
        *val *= g; // decay
    }
    // kv_mem[v] = sum_k s[k,v] * k[k]
    let mut kv_mem = vec![0.0f32; v_dim];
    for kk in 0..k_dim {
        let kval = k[kk];
        if kval == 0.0 {
            continue;
        }
        let base = kk * v_dim;
        for vv in 0..v_dim {
            kv_mem[vv] += s[base + vv] * kval;
        }
    }
    // delta[v] = (v[v] - kv_mem[v]) * beta ; s[k,v] += k[k] * delta[v]
    let mut delta = vec![0.0f32; v_dim];
    for vv in 0..v_dim {
        delta[vv] = (v[vv] - kv_mem[vv]) * beta;
    }
    for kk in 0..k_dim {
        let kval = k[kk];
        if kval == 0.0 {
            continue;
        }
        let base = kk * v_dim;
        for vv in 0..v_dim {
            s[base + vv] += kval * delta[vv];
        }
    }
    // o[v] = sum_k s[k,v] * q[k]
    for val in o.iter_mut() {
        *val = 0.0;
    }
    for kk in 0..k_dim {
        let qval = q[kk];
        if qval == 0.0 {
            continue;
        }
        let base = kk * v_dim;
        for vv in 0..v_dim {
            o[vv] += s[base + vv] * qval;
        }
    }
}

/// Gated full-attention layer (full attn at `idx % full_attention_interval == 3`).
#[allow(clippy::too_many_arguments)]
pub fn qwen_gated_attention_block(
    model: &mut LazyRllmModel,
    input: &[f32],
    tensors: &QwenLayerTensors,
    params: &QwenLayerParams,
    cfg: &QwenBuildConfig,
    seq_len: usize,
    position_offset: usize,
    budget: &mut MemoryBudget,
    cache: &mut KvCache,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let q_out = nh * hd;
    let kv_out = nkv * hd;

    let mut residual = input.to_vec();
    let x = rms_norm(
        input,
        &params.input_layernorm,
        seq_len,
        hidden,
        cfg.rms_norm_eps,
    )?;

    // q_proj emits a PER-HEAD [query(head_dim) ‖ gate(head_dim)] split when
    // attn_output_gate is set: view(.., num_heads, 2*head_dim).chunk(2, dim=-1). Repack
    // into contiguous query/gate matching the [h0, h1, ...] attention layout.
    let qg_out = if cfg.attn_output_gate { 2 * q_out } else { q_out };
    let qg = project(model, &tensors.q_proj, &x, seq_len, hidden, qg_out, budget)?;
    let mut query = vec![0.0f32; seq_len * q_out];
    let mut gate = vec![0.0f32; seq_len * q_out];
    if cfg.attn_output_gate {
        for t in 0..seq_len {
            for h in 0..nh {
                let src = t * qg_out + h * 2 * hd;
                let dst = t * q_out + h * hd;
                query[dst..dst + hd].copy_from_slice(&qg[src..src + hd]);
                gate[dst..dst + hd].copy_from_slice(&qg[src + hd..src + 2 * hd]);
            }
        }
    } else {
        query.copy_from_slice(&qg);
    }

    let mut k = project(model, &tensors.k_proj, &x, seq_len, hidden, kv_out, budget)?;
    let v = project(model, &tensors.v_proj, &x, seq_len, hidden, kv_out, budget)?;

    // QK-RMSNorm over head_dim, then partial RoPE (rotate first rotary_dim dims).
    query = rms_norm(&query, &params.q_norm, seq_len * nh, hd, cfg.rms_norm_eps)?;
    k = rms_norm(&k, &params.k_norm, seq_len * nkv, hd, cfg.rms_norm_eps)?;
    let rope = RotaryEmbeddingConfig {
        seq_len,
        num_heads: nh,
        head_dim: hd,
        rotary_dim: cfg.rotary_dim,
        base: cfg.rope_theta,
        position_offset,
    };
    apply_llama_rotary_inplace(&mut query, &mut k, nh, nkv, rope)?;

    let attn_cfg = KvAttentionConfig {
        query_len: seq_len,
        num_heads: nh,
        kv_heads: nkv,
        head_dim: hd,
        causal: cfg.causal,
    };
    let mut attn = scaled_dot_product_attention_with_cache(&query, &k, &v, Some(cache), attn_cfg)?;
    cache.append(&k, &v, seq_len)?;

    // Per-head output gate: attn *= sigmoid(gate).
    if cfg.attn_output_gate {
        for (a, g) in attn.iter_mut().zip(gate.iter()) {
            *a *= sigmoidf(*g);
        }
    }

    let o = project(model, &tensors.o_proj, &attn, seq_len, q_out, hidden, budget)?;
    add_inplace(&mut residual, &o)?;

    let mlp_in = rms_norm(
        &residual,
        &params.post_attention_layernorm,
        seq_len,
        hidden,
        cfg.rms_norm_eps,
    )?;
    let ffn = qwen_swiglu_ffn(
        model,
        tensors,
        &mlp_in,
        seq_len,
        hidden,
        cfg.intermediate_size,
        budget,
    )?;
    add_inplace(&mut residual, &ffn)?;
    Ok(residual)
}

/// Apply the depthwise causal short conv (kernel `kernel`) + SiLU to one token's
/// `q‖k‖v` stream, using/advancing the persistent conv history.
fn conv_silu_step(
    params: &QwenLayerParams,
    state: &mut GatedDeltaNetState,
    qkv_t: &[f32],
    channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let conv_hist = kernel.saturating_sub(1);
    let mut conv_out = vec![0.0f32; channels];
    for c in 0..channels {
        let mut acc = 0.0f32;
        for j in 0..conv_hist {
            acc += params.conv1d[c * kernel + j] * state.conv[j * channels + c];
        }
        acc += params.conv1d[c * kernel + conv_hist] * qkv_t[c];
        conv_out[c] = siluf(acc);
    }
    // Advance conv history with the RAW qkv input (conv operates on the input stream).
    if conv_hist > 0 {
        for j in 0..conv_hist - 1 {
            for c in 0..channels {
                state.conv[j * channels + c] = state.conv[(j + 1) * channels + c];
            }
        }
        let last = conv_hist - 1;
        state.conv[last * channels..(last + 1) * channels].copy_from_slice(qkv_t);
    }
    conv_out
}

/// Gated-DeltaNet linear-attention layer (REEDELTA). Processes tokens sequentially,
/// updating the persistent recurrent `state` (constant size in context length).
#[allow(clippy::too_many_arguments)]
pub fn qwen_gated_deltanet_block(
    model: &mut LazyRllmModel,
    input: &[f32],
    tensors: &QwenLayerTensors,
    params: &QwenLayerParams,
    cfg: &QwenBuildConfig,
    seq_len: usize,
    budget: &mut MemoryBudget,
    state: &mut GatedDeltaNetState,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let heads = cfg.linear_num_heads;
    let kd = cfg.linear_key_dim;
    let vd = cfg.linear_value_dim;
    let channels = cfg.linear_conv_channels(); // q‖k‖v width = heads*(2kd+vd)
    let qk_w = cfg.linear_qk_width(); // heads*kd
    let kernel = cfg.conv_kernel;
    let eps = cfg.rms_norm_eps;
    let q_scale = 1.0 / (kd as f32).sqrt();

    let mut residual = input.to_vec();
    let x = rms_norm(input, &params.input_layernorm, seq_len, hidden, eps)?;

    // Projections (whole sequence at once).
    let qkv = project(model, &tensors.in_proj_qkv, &x, seq_len, hidden, channels, budget)?;
    let a = project(model, &tensors.in_proj_a, &x, seq_len, hidden, heads, budget)?;
    let b = project(model, &tensors.in_proj_b, &x, seq_len, hidden, heads, budget)?;
    let z = project(model, &tensors.in_proj_z, &x, seq_len, hidden, heads * vd, budget)?;

    let mut out_heads = vec![0.0f32; seq_len * heads * vd];
    let mut o = vec![0.0f32; vd];

    for t in 0..seq_len {
        let qkv_t = &qkv[t * channels..(t + 1) * channels];
        let conv_out = conv_silu_step(params, state, qkv_t, channels, kernel);

        for h in 0..heads {
            let mut q_h = conv_out[h * kd..(h + 1) * kd].to_vec();
            let mut k_h = conv_out[qk_w + h * kd..qk_w + (h + 1) * kd].to_vec();
            let v_h = &conv_out[2 * qk_w + h * vd..2 * qk_w + (h + 1) * vd];
            l2norm_inplace(&mut q_h, 1e-6);
            l2norm_inplace(&mut k_h, 1e-6);
            // HF scales the L2-normed query by 1/sqrt(key_dim) before the readout; this
            // sets the readout magnitude so the gated-RMSNorm epsilon interacts the same
            // (matters for strong-decay heads where o ~ eps).
            for val in q_h.iter_mut() {
                *val *= q_scale;
            }

            let beta = sigmoidf(b[t * heads + h]);
            let g = (-params.a_log[h].exp() * softplusf(a[t * heads + h] + params.dt_bias[h])).exp();
            let s = &mut state.s[h * kd * vd..(h + 1) * kd * vd];
            delta_rule_head_step(s, &q_h, &k_h, v_h, beta, g, kd, vd, &mut o);

            // Gated RMSNorm: norm first (variance from o), ×weight, then ×silu(z).
            let variance = o.iter().map(|v| v * v).sum::<f32>() / vd as f32;
            let inv = 1.0 / (variance + eps).sqrt();
            let z_h = &z[t * heads * vd + h * vd..t * heads * vd + (h + 1) * vd];
            let out_base = t * heads * vd + h * vd;
            for vv in 0..vd {
                out_heads[out_base + vv] = o[vv] * inv * params.linear_norm[vv] * siluf(z_h[vv]);
            }
        }
    }

    let out = project(
        model,
        &tensors.out_proj,
        &out_heads,
        seq_len,
        heads * vd,
        hidden,
        budget,
    )?;
    add_inplace(&mut residual, &out)?;

    let mlp_in = rms_norm(
        &residual,
        &params.post_attention_layernorm,
        seq_len,
        hidden,
        eps,
    )?;
    let ffn = qwen_swiglu_ffn(
        model,
        tensors,
        &mlp_in,
        seq_len,
        hidden,
        cfg.intermediate_size,
        budget,
    )?;
    add_inplace(&mut residual, &ffn)?;
    Ok(residual)
}

/// Guard: ensure the prepared transformer's dims are internally consistent.
pub(crate) fn validate_prepared(prepared: &PreparedQwenTransformer) -> Result<()> {
    let cfg = &prepared.config;
    if cfg.num_heads == 0 || cfg.num_kv_heads == 0 || !cfg.num_heads.is_multiple_of(cfg.num_kv_heads)
    {
        return Err(RuntimeError::Shape(format!(
            "qwen attention heads invalid: num_heads={}, num_kv_heads={}",
            cfg.num_heads, cfg.num_kv_heads
        )));
    }
    if cfg.rotary_dim > cfg.head_dim || !cfg.rotary_dim.is_multiple_of(2) {
        return Err(RuntimeError::Shape(format!(
            "qwen rotary_dim {} must be even and <= head_dim {}",
            cfg.rotary_dim, cfg.head_dim
        )));
    }
    if prepared.layers.len() != prepared.layer_params.len() {
        return Err(RuntimeError::Shape(
            "qwen layers and layer_params length mismatch".to_string(),
        ));
    }
    Ok(())
}
