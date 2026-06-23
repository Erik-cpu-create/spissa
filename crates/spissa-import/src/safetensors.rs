// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! Safetensors format parser
//!
//! Safetensors is a simple format for storing tensors safely and efficiently.
//! Format:
//! - 8 bytes: header length (u64, little-endian)
//! - N bytes: JSON header (metadata about tensors)
//! - Rest: raw tensor data

use spissa_container::{DType, ModelConfigMetadata, TensorMeta};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SafetensorsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid header length: {0}")]
    InvalidHeaderLength(u64),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Invalid tokenizer metadata: {0}")]
    InvalidTokenizer(String),
}

pub type Result<T> = std::result::Result<T, SafetensorsError>;

#[derive(Debug, Clone, Deserialize)]
struct HuggingFaceModelConfig {
    architectures: Option<Vec<String>>,
    model_type: Option<String>,
    #[serde(alias = "n_layer")]
    num_hidden_layers: Option<u64>,
    #[serde(alias = "n_embd")]
    hidden_size: Option<u64>,
    intermediate_size: Option<u64>,
    #[serde(alias = "n_head")]
    num_attention_heads: Option<u64>,
    #[serde(alias = "n_positions")]
    max_position_embeddings: Option<u64>,
    rotary_pct: Option<f32>,
    rotary_emb_base: Option<f32>,
    layer_norm_eps: Option<f32>,
    use_parallel_residual: Option<bool>,
    vocab_size: Option<u64>,
    rms_norm_eps: Option<f32>,
    num_key_value_heads: Option<u64>,
    rope_theta: Option<f32>,
    tie_word_embeddings: Option<bool>,
    // Gemma-family fields
    head_dim: Option<u64>,
    query_pre_attn_scalar: Option<f32>,
    sliding_window: Option<u64>,
    sliding_window_pattern: Option<u64>,
    rope_local_base_freq: Option<f32>,
    hidden_activation: Option<String>,
    /// Linear RoPE-scaling for global layers, e.g. Gemma 3
    /// `{"factor": 8.0, "rope_type": "linear"}`.
    rope_scaling: Option<RopeScalingConfig>,
    /// Multimodal checkpoints (e.g. Gemma 3 4B = `Gemma3ForConditionalGeneration`)
    /// nest the language-model architecture params under `text_config`.
    text_config: Option<Box<HuggingFaceModelConfig>>,
    // --- Qwen3.5 / Qwen3-Next hybrid linear-attention fields ---
    /// Per-layer operator schedule, e.g. `["linear_attention", …, "full_attention"]`.
    layer_types: Option<Vec<String>>,
    /// Period of the full-attention interleave (Qwen3.5: every 4th layer is full attn).
    full_attention_interval: Option<u64>,
    /// Gated-DeltaNet (linear attention) head dims / counts and depthwise short-conv kernel.
    linear_key_head_dim: Option<u64>,
    linear_value_head_dim: Option<u64>,
    linear_num_key_heads: Option<u64>,
    linear_num_value_heads: Option<u64>,
    linear_conv_kernel_dim: Option<u64>,
    /// Gated full-attention emits a per-head output gate from `q_proj` (doubles its width).
    attn_output_gate: Option<bool>,
    /// Fraction of `head_dim` that RoPE rotates (Qwen3.5: 0.25 → 64 of 256). May be flat
    /// or nested under `rope_parameters`.
    partial_rotary_factor: Option<f32>,
    /// Qwen nests rope params (theta, partial rotary, mrope sections) under `rope_parameters`.
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeScalingConfig {
    factor: Option<f32>,
    rope_type: Option<String>,
    /// Phi-3 LongRoPE per-dimension factors.
    short_factor: Option<Vec<f32>>,
    long_factor: Option<Vec<f32>>,
}

/// Qwen3.5 `rope_parameters` block: rope theta, partial-rotary fraction, and the
/// multimodal-RoPE section split (which collapses to ordinary RoPE for text-only).
#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    rope_theta: Option<f32>,
    partial_rotary_factor: Option<f32>,
    mrope_section: Option<Vec<u64>>,
    #[allow(dead_code)]
    mrope_interleaved: Option<bool>,
}

pub fn read_model_config_metadata(path: impl AsRef<Path>) -> Result<ModelConfigMetadata> {
    let json = fs::read_to_string(path)?;
    model_config_metadata_from_json_str(&json)
}

pub fn model_config_metadata_from_json_str(json: &str) -> Result<ModelConfigMetadata> {
    let config: HuggingFaceModelConfig = serde_json::from_str(json)?;
    // Architecture detection uses the TOP-level model_type/architectures (e.g.
    // "gemma3" / "Gemma3ForConditionalGeneration" for the multimodal wrapper).
    let architecture_type = config
        .model_type
        .as_deref()
        .or_else(|| config.architectures.as_ref()?.first().map(String::as_str))
        .map(normalize_architecture_type);
    // Multimodal checkpoints nest the language-model architecture under
    // `text_config`; read dimensions/params from there when present.
    let dims: &HuggingFaceModelConfig = config.text_config.as_deref().unwrap_or(&config);

    Ok(ModelConfigMetadata {
        architecture_type,
        num_hidden_layers: dims.num_hidden_layers,
        hidden_size: dims.hidden_size,
        intermediate_size: dims.intermediate_size,
        num_attention_heads: dims.num_attention_heads,
        max_position_embeddings: dims.max_position_embeddings,
        rotary_pct: dims.rotary_pct,
        rotary_emb_base: dims.rotary_emb_base,
        layer_norm_eps: dims.layer_norm_eps,
        use_parallel_residual: dims.use_parallel_residual,
        vocab_size: dims.vocab_size,
        rms_norm_eps: dims.rms_norm_eps,
        num_key_value_heads: dims.num_key_value_heads,
        rope_theta: dims
            .rope_theta
            .or_else(|| dims.rope_parameters.as_ref().and_then(|r| r.rope_theta)),
        tie_word_embeddings: dims.tie_word_embeddings.or(config.tie_word_embeddings),
        head_dim: dims.head_dim,
        query_pre_attn_scalar: dims.query_pre_attn_scalar,
        sliding_window: dims.sliding_window,
        sliding_window_pattern: dims.sliding_window_pattern,
        rope_local_base_freq: dims.rope_local_base_freq,
        hidden_activation: dims.hidden_activation.clone(),
        // Only linear scaling is captured (Gemma 3 family); other rope_type
        // values are left as `None` so the runtime falls back to unscaled rope.
        rope_scaling_factor: dims
            .rope_scaling
            .as_ref()
            .filter(|scaling| {
                scaling
                    .rope_type
                    .as_deref()
                    .map(|kind| kind.eq_ignore_ascii_case("linear"))
                    .unwrap_or(false)
            })
            .and_then(|scaling| scaling.factor),
        // Phi-3 LongRoPE short factor (applies within the original window — short-context chat).
        rope_short_factor: dims
            .rope_scaling
            .as_ref()
            .and_then(|s| s.short_factor.clone()),
        // --- Qwen3.5 / Qwen3-Next hybrid linear-attention fields ---
        layer_types: dims.layer_types.clone(),
        full_attention_interval: dims.full_attention_interval,
        linear_key_head_dim: dims.linear_key_head_dim,
        linear_value_head_dim: dims.linear_value_head_dim,
        linear_num_key_heads: dims.linear_num_key_heads,
        linear_num_value_heads: dims.linear_num_value_heads,
        linear_conv_kernel_dim: dims.linear_conv_kernel_dim,
        attn_output_gate: dims.attn_output_gate,
        partial_rotary_factor: dims
            .partial_rotary_factor
            .or_else(|| dims.rope_parameters.as_ref().and_then(|r| r.partial_rotary_factor)),
        mrope_section: dims
            .rope_parameters
            .as_ref()
            .and_then(|r| r.mrope_section.clone()),
    })
}

fn normalize_architecture_type(value: &str) -> String {
    let normalized = value.to_ascii_lowercase().replace('-', "_");
    if normalized == "gpt_neox" || normalized.contains("gptneox") {
        "gpt_neox".to_string()
    } else if normalized == "llamaforcausallm" || normalized.contains("llama") {
        "llama".to_string()
    } else if normalized.contains("gemma3") {
        // "gemma3", "gemma3_text", "gemma3forconditionalgeneration"
        "gemma3".to_string()
    } else if normalized.contains("gemma2") {
        "gemma2".to_string()
    } else if normalized.contains("gemma") {
        "gemma".to_string()
    } else if normalized.contains("qwen3") {
        // "qwen3_5", "qwen3_next", "qwen3_5forconditionalgeneration" → the hybrid
        // Gated-DeltaNet + gated-attention adapter.
        "qwen3".to_string()
    } else if normalized.contains("qwen") {
        "qwen".to_string()
    } else if normalized.contains("phi3") || normalized.contains("phi4") {
        // Phi-3 / Phi-4-mini: fused qkv/gate_up are split at import → reuses the LLaMA decode with
        // partial RoPE + LongRoPE short factor. Matches both "phi3" and "Phi3ForCausalLM".
        "phi3".to_string()
    } else {
        normalized
    }
}

/// Metadata for a single tensor in safetensors format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsTensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Header of a safetensors file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsHeader {
    #[serde(flatten)]
    pub tensors: HashMap<String, SafetensorsTensorMeta>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub __metadata__: Option<HashMap<String, String>>,
}

/// Reader for safetensors files
pub struct SafetensorsReader {
    file: BufReader<File>,
    header: SafetensorsHeader,
    data_offset: u64,
}

impl SafetensorsReader {
    /// Open and parse a safetensors file
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Read header length (8 bytes, u64, little-endian)
        let mut header_len_bytes = [0u8; 8];
        reader.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes);

        // Sanity check
        if header_len > 100_000_000 {
            // 100MB header is unreasonable
            return Err(SafetensorsError::InvalidHeaderLength(header_len));
        }

        // Read header JSON
        let mut header_bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut header_bytes)?;
        let header: SafetensorsHeader = serde_json::from_slice(&header_bytes)?;

        // Data starts after header length + header
        let data_offset = 8 + header_len;

        Ok(Self {
            file: reader,
            header,
            data_offset,
        })
    }

    /// Get the header
    pub fn header(&self) -> &SafetensorsHeader {
        &self.header
    }

    /// List all tensor names
    pub fn list_tensors(&self) -> Vec<&str> {
        self.header.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor metadata by name
    pub fn get_tensor_meta(&self, name: &str) -> Option<&SafetensorsTensorMeta> {
        self.header.tensors.get(name)
    }

    /// Read tensor data by name
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let meta = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        let [start, end] = meta.data_offsets;
        let size = end - start;

        self.file
            .seek(SeekFrom::Start(self.data_offset + start as u64))?;
        let mut data = vec![0u8; size];
        self.file.read_exact(&mut data)?;

        Ok(data)
    }

    /// Convert safetensors dtype to RLLM DType
    pub fn convert_dtype(dtype: &str) -> Result<DType> {
        match dtype {
            "F16" => Ok(DType::Fp16),
            "BF16" => Ok(DType::Bf16),
            "F32" => Ok(DType::Fp32),
            "F64" => Ok(DType::Fp64),
            "I8" => Ok(DType::I8),
            "I16" => Ok(DType::I16),
            "I32" => Ok(DType::I32),
            "I64" => Ok(DType::I64),
            "U8" => Ok(DType::U8),
            "U16" => Ok(DType::U16),
            "U32" => Ok(DType::U32),
            "U64" => Ok(DType::U64),
            _ => Err(SafetensorsError::UnsupportedDtype(dtype.to_string())),
        }
    }

    /// Convert tensor metadata to RLLM TensorMeta
    pub fn to_rllm_meta(&mut self, name: &str) -> Result<TensorMeta> {
        let meta = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        let dtype = Self::convert_dtype(&meta.dtype)?;
        let shape: Vec<u64> = meta.shape.iter().map(|&x| x as u64).collect();
        let size = meta.data_offsets[1] - meta.data_offsets[0];

        // Compute hash (we'll need to read the data)
        let data = self.read_tensor(name)?;
        let hash: [u8; 32] = Sha256::digest(&data).into();

        Ok(TensorMeta {
            tensor_id: 0, // Will be set by writer
            name: name.to_string(),
            shape,
            dtype,
            original_size_bytes: size as u64,
            compressed_size_bytes: 0, // Will be computed by writer
            original_sha256: hash,
            chunk_count: 0,       // Will be computed by writer
            chunk_start_index: 0, // Will be computed by writer
        })
    }
}

/// `model.safetensors.index.json`: maps each tensor to the shard file holding it.
#[derive(Debug, Clone, Deserialize)]
pub struct SafetensorsIndex {
    pub weight_map: HashMap<String, String>,
}

/// Reader over a sharded safetensors checkpoint (`*.index.json` + N shard files).
/// Presents the same `list_tensors` / `read_tensor` / `to_rllm_meta` surface as
/// `SafetensorsReader`, dispatching each tensor to the shard that holds it.
pub struct ShardedSafetensorsReader {
    shards: Vec<SafetensorsReader>,
    name_to_shard: HashMap<String, usize>,
}

impl ShardedSafetensorsReader {
    /// Open a sharded checkpoint from its `*.index.json` path. Shard files are
    /// resolved relative to the index file's directory.
    pub fn open_index(index_path: impl AsRef<Path>) -> Result<Self> {
        let index_path = index_path.as_ref();
        let dir = index_path.parent().unwrap_or_else(|| Path::new("."));
        let json = fs::read_to_string(index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&json)?;

        // Unique shard files, deterministic order.
        let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
        shard_files.sort();
        shard_files.dedup();

        let mut shards = Vec::with_capacity(shard_files.len());
        let mut file_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, f) in shard_files.iter().enumerate() {
            shards.push(SafetensorsReader::open(dir.join(f))?);
            file_to_idx.insert(f.clone(), i);
        }
        let name_to_shard = index
            .weight_map
            .into_iter()
            .map(|(name, f)| (name, file_to_idx[&f]))
            .collect();

        Ok(Self {
            shards,
            name_to_shard,
        })
    }

    /// List all tensor names across every shard.
    pub fn list_tensors(&self) -> Vec<String> {
        self.name_to_shard.keys().cloned().collect()
    }

    /// Read tensor data, dispatching to the shard that holds it.
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let idx = *self
            .name_to_shard
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;
        self.shards[idx].read_tensor(name)
    }

    /// Convert a tensor's metadata to RLLM `TensorMeta` (hashes the data).
    pub fn to_rllm_meta(&mut self, name: &str) -> Result<TensorMeta> {
        let idx = *self
            .name_to_shard
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;
        self.shards[idx].to_rllm_meta(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer_metadata_from_json_str;

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(
            SafetensorsReader::convert_dtype("F16").unwrap(),
            DType::Fp16
        );
        assert_eq!(
            SafetensorsReader::convert_dtype("BF16").unwrap(),
            DType::Bf16
        );
        assert_eq!(
            SafetensorsReader::convert_dtype("F32").unwrap(),
            DType::Fp32
        );
        assert!(SafetensorsReader::convert_dtype("INVALID").is_err());
    }

    #[test]
    fn parses_gpt_neox_config_json_into_model_config_metadata() {
        let json = r#"{
            "architectures": ["GPTNeoXForCausalLM"],
            "model_type": "gpt_neox",
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "max_position_embeddings": 4096,
            "rotary_pct": 0.25,
            "rotary_emb_base": 10000,
            "layer_norm_eps": 0.00001,
            "use_parallel_residual": true,
            "vocab_size": 50432
        }"#;

        let config = model_config_metadata_from_json_str(json).unwrap();

        assert_eq!(config.architecture_type.as_deref(), Some("gpt_neox"));
        assert_eq!(config.num_hidden_layers, Some(2));
        assert_eq!(config.hidden_size, Some(128));
        assert_eq!(config.intermediate_size, Some(512));
        assert_eq!(config.num_attention_heads, Some(4));
        assert_eq!(config.max_position_embeddings, Some(4096));
        assert_eq!(config.rotary_pct, Some(0.25));
        assert_eq!(config.rotary_emb_base, Some(10_000.0));
        assert_eq!(config.layer_norm_eps, Some(1e-5));
        assert_eq!(config.use_parallel_residual, Some(true));
        assert_eq!(config.vocab_size, Some(50_432));
    }

    #[test]
    fn parses_huggingface_tokenizer_json_into_tokenizer_metadata() {
        let json = r#"{
            "model": {
                "type": "WordLevel",
                "unk_token": "<unk>",
                "vocab": {
                    "Hello": 0,
                    " world": 1,
                    "<unk>": 2
                },
                "merges": ["H ello", [" wor", "ld"]]
            },
            "added_tokens": [
                {"id": 3, "content": "<|endoftext|>", "special": true}
            ],
            "eos_token": "<|endoftext|>"
        }"#;

        let tokenizer = tokenizer_metadata_from_json_str(json).unwrap();

        assert_eq!(tokenizer.tokenizer_type.as_deref(), Some("hf-wordlevel"));
        assert_eq!(
            tokenizer.id_to_token,
            ["Hello", " world", "<unk>", "<|endoftext|>"]
        );
        assert_eq!(
            tokenizer.bpe_merges,
            [
                ("H".to_string(), "ello".to_string()),
                (" wor".to_string(), "ld".to_string())
            ]
        );
        assert_eq!(tokenizer.unk_token_id, Some(2));
        assert_eq!(tokenizer.eos_token_id, Some(3));
    }

    /// Integration check against the real downloaded Gemma 3 4B sharded
    /// checkpoint. Ignored by default (depends on a local 8.6GB download).
    /// Run: `cargo test -p spissa-import --release sharded_reads_real_gemma -- --ignored`
    #[test]
    #[ignore]
    fn sharded_reads_real_gemma() {
        let idx = "../../models/gemma-3-4b-it/model.safetensors.index.json";
        let mut r = ShardedSafetensorsReader::open_index(idx).expect("open index");
        let names = r.list_tensors();
        assert!(names.len() > 800, "expected ~883 tensors, got {}", names.len());
        // text decoder is nested under language_model.*; embeddings are [vocab, hidden]
        let meta = r
            .to_rllm_meta("language_model.model.embed_tokens.weight")
            .expect("embed_tokens meta");
        assert_eq!(meta.shape, vec![262208, 2560], "gemma3 embed shape");
        // a layer-0 q_proj: head_dim 256 * 8 heads = 2048 out, 2560 in
        let q = r
            .to_rllm_meta("language_model.model.layers.0.self_attn.q_proj.weight")
            .expect("q_proj meta");
        assert_eq!(q.shape, vec![2048, 2560], "gemma3 q_proj shape (8*256, 2560)");
        // the Gemma-specific QK-norm tensor exists
        assert!(r
            .to_rllm_meta("language_model.model.layers.0.self_attn.q_norm.weight")
            .is_ok());
        // reading actual bytes works (cross-shard dispatch)
        let data = r
            .read_tensor("language_model.model.norm.weight")
            .expect("final norm bytes");
        assert_eq!(data.len(), 2560 * 2, "bf16 final norm = hidden*2 bytes");
    }

    /// Parse the real multimodal Gemma 3 config.json: architecture from the top,
    /// dimensions + Gemma fields from `text_config`. Ignored (needs the download).
    #[test]
    #[ignore]
    fn parses_real_gemma3_multimodal_config() {
        let json = std::fs::read_to_string("../../models/gemma-3-4b-it/config.json")
            .expect("read gemma config");
        let m = model_config_metadata_from_json_str(&json).expect("parse config");
        assert_eq!(m.architecture_type.as_deref(), Some("gemma3"));
        // dims come from text_config (NOT the multimodal top level)
        assert_eq!(m.hidden_size, Some(2560));
        assert_eq!(m.num_hidden_layers, Some(34));
        assert_eq!(m.num_attention_heads, Some(8));
        assert_eq!(m.num_key_value_heads, Some(4));
        assert_eq!(m.head_dim, Some(256)); // explicit, != 2560/8
        assert_eq!(m.intermediate_size, Some(10240));
        assert_eq!(m.vocab_size, Some(262208));
        assert_eq!(m.hidden_activation.as_deref(), Some("gelu_pytorch_tanh"));
        assert_eq!(m.sliding_window, Some(1024));
        assert_eq!(m.sliding_window_pattern, Some(6));
        assert_eq!(m.rope_theta, Some(1_000_000.0));
        assert_eq!(m.rope_local_base_freq, Some(10_000.0));
        assert_eq!(m.query_pre_attn_scalar, Some(256.0));
        assert_eq!(m.rms_norm_eps, Some(1e-6));
        // linear rope-scaling factor for the global layers
        assert_eq!(m.rope_scaling_factor, Some(8.0));
    }

    #[test]
    fn parses_linear_rope_scaling_factor_and_ignores_non_linear() {
        let linear = model_config_metadata_from_json_str(
            r#"{"model_type":"gemma3","rope_scaling":{"factor":8.0,"rope_type":"linear"}}"#,
        )
        .unwrap();
        assert_eq!(linear.rope_scaling_factor, Some(8.0));

        // A non-linear rope_type (e.g. yarn/llama3) must not be captured as a
        // plain linear divisor — leave it None so the runtime stays unscaled.
        let non_linear = model_config_metadata_from_json_str(
            r#"{"model_type":"llama","rope_scaling":{"factor":8.0,"rope_type":"llama3"}}"#,
        )
        .unwrap();
        assert_eq!(non_linear.rope_scaling_factor, None);

        let absent =
            model_config_metadata_from_json_str(r#"{"model_type":"llama"}"#).unwrap();
        assert_eq!(absent.rope_scaling_factor, None);
    }
}
