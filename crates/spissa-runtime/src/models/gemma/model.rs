// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

use crate::StreamingSamplingConfig;

/// Static build/run configuration for the Gemma-family text decoder.
///
/// Gemma 3 differs from LLaMA in a handful of fixed ways captured here:
/// - `head_dim` is explicit (256), independent of `hidden_size / num_heads`;
/// - attention is scaled by `1/sqrt(query_pre_attn_scalar)`, not `1/sqrt(head_dim)`;
/// - embeddings are scaled by `sqrt(hidden_size)` on input;
/// - RoPE is dual: global (non-sliding) layers use `rope_theta` with a linear
///   position divisor `rope_scaling_factor`; local (sliding) layers use
///   `rope_local_base_freq` unscaled. See [`is_global_layer`].
#[derive(Debug, Clone, Copy)]
pub struct GemmaBuildConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: usize,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    /// Global (non-sliding) RoPE base (`rope_theta`, Gemma 3 = 1e6).
    pub rope_theta: f32,
    /// Local (sliding) RoPE base (`rope_local_base_freq`, Gemma 3 = 1e4).
    pub rope_local_base_freq: f32,
    /// Linear RoPE position divisor for global layers (Gemma 3 = 8.0).
    pub rope_scaling_factor: f32,
    /// Period of the local:global interleave (Gemma 3 = 6 → 5 local : 1 global).
    pub sliding_window_pattern: usize,
    /// Attention logit scale, `1/sqrt(query_pre_attn_scalar)` (Gemma 3 = 1/16).
    pub attn_scale: f32,
    /// Input embedding scale, `sqrt(hidden_size)` (Gemma 3 ≈ 50.6).
    pub embed_scale: f32,
    pub causal: bool,
    pub sampling: StreamingSamplingConfig,
}

/// A Gemma layer is *global* (full attention + global RoPE) when
/// `(layer_idx + 1) % sliding_window_pattern == 0`, otherwise *local* (sliding
/// window + local RoPE). This mirrors HF `Gemma3Attention.is_sliding =
/// bool((layer_idx + 1) % sliding_window_pattern)`. A pattern of 0 or 1 means
/// every layer is global.
pub fn is_global_layer(layer_idx: usize, sliding_window_pattern: usize) -> bool {
    if sliding_window_pattern <= 1 {
        return true;
    }
    (layer_idx + 1).is_multiple_of(sliding_window_pattern)
}

/// Per-layer projection-weight tensor names (the large matrices streamed from
/// the model on every forward). QK-norm / layernorm vectors are decoded and
/// pinned separately in [`GemmaLayerNorms`].
#[derive(Debug, Clone)]
pub struct GemmaBlockTensorNames {
    pub q_weight: String,
    pub k_weight: String,
    pub v_weight: String,
    pub o_weight: String,
    pub gate_weight: String,
    pub up_weight: String,
    pub down_weight: String,
}

/// Per-layer RMSNorm weights, already decoded to f32 and pre-baked with Gemma's
/// `(1 + weight)` convention so the standard [`crate::ops::rms_norm`] can be
/// reused verbatim. `q_norm`/`k_norm` are length `head_dim`; the rest are
/// length `hidden_size`.
#[derive(Debug, Clone)]
pub struct GemmaLayerNorms {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub pre_feedforward_layernorm: Vec<f32>,
    pub post_feedforward_layernorm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// A fully prepared Gemma transformer: projection-weight names plus all
/// small norm vectors decoded once and pinned for the generation loop. The
/// embedding matrix doubles as the (tied) LM head.
#[derive(Debug, Clone)]
pub struct PreparedGemmaTransformer {
    pub config: GemmaBuildConfig,
    pub embedding_weight: String,
    pub lm_head_weight: String,
    pub layers: Vec<GemmaBlockTensorNames>,
    pub layer_norms: Vec<GemmaLayerNorms>,
    pub final_layernorm: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct GemmaTextGenerationResult {
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub token_ids: Vec<usize>,
    pub context_echo_bytes: usize,
    pub logits: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::is_global_layer;

    #[test]
    fn gemma3_global_layers_are_every_sixth_one_indexed() {
        // sliding_window_pattern = 6 → 5 local : 1 global. Global layers are
        // those where (idx + 1) % 6 == 0: 5, 11, 17, 23, 29.
        let pattern = 6;
        let globals: Vec<usize> = (0..34).filter(|&i| is_global_layer(i, pattern)).collect();
        assert_eq!(globals, vec![5, 11, 17, 23, 29]);
        // Layer 33 (the last of 34) is local, not global.
        assert!(!is_global_layer(33, pattern));
    }

    #[test]
    fn degenerate_patterns_make_every_layer_global() {
        for pattern in [0, 1] {
            assert!(is_global_layer(0, pattern));
            assert!(is_global_layer(7, pattern));
        }
    }
}
