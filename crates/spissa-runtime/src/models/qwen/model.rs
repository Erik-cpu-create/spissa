// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

//! Qwen3.5-2B (qwen3_5 / Qwen3-Next-style) text-only model types.
//!
//! Hybrid decoder: a `LLLF` schedule of 18 Gated-DeltaNet linear-attention layers
//! and 6 gated full-attention layers (full attn at `idx % full_attention_interval == 3`),
//! dense SwiGLU FFN, tied lm_head. See `docs/qwen3_5-adapter-design.md`.

use crate::rotary::KvCache;
use crate::StreamingSamplingConfig;

/// Which operator a decoder layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenLayerKind {
    /// Gated DeltaNet (linear attention) — persistent recurrent state, no growing KV.
    LinearAttention,
    /// Gated full attention — GQA softmax attention with a per-head output gate.
    FullAttention,
}

/// Static build/runtime config for the Qwen text decoder.
#[derive(Debug, Clone, Copy)]
pub struct QwenBuildConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub causal: bool,
    // --- gated full attention ---
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Number of leading head-dim entries RoPE rotates (partial rotary; Qwen3.5: 64).
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub attn_output_gate: bool,
    // --- gated DeltaNet (linear attention) ---
    pub linear_num_heads: usize,
    pub linear_key_dim: usize,
    pub linear_value_dim: usize,
    pub conv_kernel: usize,
    pub sampling: StreamingSamplingConfig,
}

impl QwenBuildConfig {
    /// Total channels carried through the linear-attn short conv = `q‖k‖v` width
    /// = `heads * (key_dim + key_dim + value_dim)` (Qwen3.5: 16·384 = 6144).
    pub fn linear_conv_channels(&self) -> usize {
        self.linear_num_heads * (2 * self.linear_key_dim + self.linear_value_dim)
    }

    /// Width of the `q`/`k` halves of the conv stream (`heads * key_dim`).
    pub fn linear_qk_width(&self) -> usize {
        self.linear_num_heads * self.linear_key_dim
    }

    /// Width of the `v` half of the conv stream (`heads * value_dim`).
    pub fn linear_v_width(&self) -> usize {
        self.linear_num_heads * self.linear_value_dim
    }
}

/// Streamed (decoded-on-demand) projection weight names for one decoder layer.
/// Only the subset matching the layer kind is populated/used.
#[derive(Debug, Clone, Default)]
pub struct QwenLayerTensors {
    // gated full attention
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    // gated DeltaNet linear attention
    pub in_proj_qkv: String,
    pub in_proj_a: String,
    pub in_proj_b: String,
    pub in_proj_z: String,
    pub out_proj: String,
    // dense SwiGLU FFN (every layer)
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

/// Small per-layer parameters decoded once and pinned in f32.
#[derive(Debug, Clone, Default)]
pub struct QwenLayerParams {
    pub kind_full_attention: bool,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    // full attention: per-head-dim QK RMSNorm
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    // linear attention: decay / gate / short-conv / gated-rmsnorm params
    pub a_log: Vec<f32>,       // [linear_num_heads]
    pub dt_bias: Vec<f32>,     // [linear_num_heads]
    pub conv1d: Vec<f32>,      // [conv_channels * conv_kernel], depthwise
    pub linear_norm: Vec<f32>, // [linear_value_dim] gated-RMSNorm weight
}

impl QwenLayerParams {
    pub fn kind(&self) -> QwenLayerKind {
        if self.kind_full_attention {
            QwenLayerKind::FullAttention
        } else {
            QwenLayerKind::LinearAttention
        }
    }
}

/// A fully prepared Qwen text transformer (names for streamed weights + pinned
/// small params), ready to drive greedy generation.
#[derive(Debug, Clone)]
pub struct PreparedQwenTransformer {
    pub config: QwenBuildConfig,
    pub embedding_weight: String,
    /// Tied to `embed_tokens` (Qwen3.5 sets `tie_word_embeddings = true`).
    pub lm_head_weight: String,
    pub final_norm: Vec<f32>,
    pub layers: Vec<QwenLayerTensors>,
    pub layer_params: Vec<QwenLayerParams>,
}

/// Persistent recurrent state for one Gated-DeltaNet layer. Size is CONSTANT in
/// context length: the `s` matrix plus a short conv history.
#[derive(Debug, Clone)]
pub struct GatedDeltaNetState {
    pub heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    pub conv_channels: usize,
    /// Recurrent state, `[heads * k_dim * v_dim]`, indexed `s[(h*k_dim + k)*v_dim + v]`.
    pub s: Vec<f32>,
    /// Last `conv_kernel-1` conv inputs, `[(conv_kernel-1) * conv_channels]`,
    /// oldest frame first.
    pub conv: Vec<f32>,
}

impl GatedDeltaNetState {
    pub fn new(
        heads: usize,
        k_dim: usize,
        v_dim: usize,
        conv_kernel: usize,
        conv_channels: usize,
    ) -> Self {
        Self {
            heads,
            k_dim,
            v_dim,
            conv_kernel,
            conv_channels,
            s: vec![0.0; heads * k_dim * v_dim],
            conv: vec![0.0; conv_kernel.saturating_sub(1) * conv_channels],
        }
    }

    pub fn resident_bytes(&self) -> usize {
        (self.s.len() + self.conv.len()) * std::mem::size_of::<f32>()
    }
}

/// Per-layer mixing state: a growing KV cache for full-attn layers, a constant-size
/// recurrent state for linear-attn layers.
#[derive(Debug, Clone)]
pub enum QwenLayerCache {
    Attn(KvCache),
    Linear(GatedDeltaNetState),
}

impl QwenLayerCache {
    pub fn resident_bytes(&self) -> usize {
        match self {
            QwenLayerCache::Attn(c) => c
                .len()
                .saturating_mul(c.num_heads())
                .saturating_mul(c.head_dim())
                .saturating_mul(2)
                .saturating_mul(std::mem::size_of::<f32>()),
            QwenLayerCache::Linear(s) => s.resident_bytes(),
        }
    }
}
