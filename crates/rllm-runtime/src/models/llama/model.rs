// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use crate::StreamingSamplingConfig;

#[derive(Debug, Clone, Copy)]
pub struct LlamaEchoBuildConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: Option<usize>,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub causal: bool,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sampling: StreamingSamplingConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct LlamaEchoGenerationConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: Option<usize>,
    pub causal: bool,
    pub sampling: StreamingSamplingConfig,
}

pub type LlamaRamaBuildConfig = LlamaEchoBuildConfig;
pub type LlamaRamaGenerationConfig = LlamaEchoGenerationConfig;

#[derive(Debug, Clone, Copy)]
pub struct LlamaRamaGenerationOptions {
    pub timing: bool,
    pub prefill_chunk_tokens: Option<usize>,
    pub collect_logits: bool,
}

impl Default for LlamaRamaGenerationOptions {
    fn default() -> Self {
        Self {
            timing: false,
            prefill_chunk_tokens: None,
            collect_logits: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OwnedLlamaStreamingBlockTensorNames {
    pub q_weight: String,
    pub k_weight: String,
    pub v_weight: String,
    pub o_weight: String,
    pub gate_weight: String,
    pub up_weight: String,
    pub down_weight: String,
}

#[derive(Debug, Clone)]
pub struct OwnedLlamaStreamingBlockParameters {
    pub input_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct PreparedLlamaEchoTransformer {
    pub config: LlamaEchoBuildConfig,
    pub embedding_weight: String,
    pub layers: Vec<OwnedLlamaStreamingBlockTensorNames>,
    pub lm_head_weight: String,
    pub layer_params: Vec<OwnedLlamaStreamingBlockParameters>,
    pub final_layernorm_weight: Vec<f32>,
    pub resident_parameter_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct LayerDecodedLlamaRamaTransformer {
    pub config: LlamaEchoBuildConfig,
    pub embedding_weight: String,
    pub layers: Vec<OwnedLlamaStreamingBlockTensorNames>,
    pub lm_head_weight: String,
    pub final_layernorm_weight: Vec<f32>,
    pub pinned_lm_head_weight: Option<Vec<f32>>,
    pub resident_parameter_bytes: usize,
    pub max_layer_parameter_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct LlamaTextGenerationResult {
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub token_ids: Vec<usize>,
    pub text: String,
    pub generated_text: String,
    pub context_echo_bytes: usize,
    pub logits: Option<Vec<f32>>,
}
