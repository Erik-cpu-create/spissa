//! Tensor and chunk metadata structures

use serde::{Deserialize, Serialize};

/// Data type of a tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    /// 16-bit floating point (IEEE 754)
    Fp16,
    /// 16-bit brain floating point
    Bf16,
    /// 32-bit floating point
    Fp32,
    /// 64-bit floating point
    Fp64,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
}

impl DType {
    /// Size in bytes of a single element of this dtype
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Fp16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Fp32 | DType::I32 | DType::U32 => 4,
            DType::Fp64 | DType::I64 | DType::U64 => 8,
            DType::I8 | DType::U8 => 1,
        }
    }
}

/// Metadata for a single tensor in the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Unique tensor identifier
    pub tensor_id: u64,
    /// Tensor name (e.g., "layers.0.mlp.down_proj.weight")
    pub name: String,
    /// Tensor shape (dimensions)
    pub shape: Vec<u64>,
    /// Data type
    pub dtype: DType,
    /// Original uncompressed size in bytes
    pub original_size_bytes: u64,
    /// Compressed size in bytes (sum of all chunks)
    pub compressed_size_bytes: u64,
    /// SHA-256 hash of original tensor bytes
    pub original_sha256: [u8; 32],
    /// Number of chunks this tensor is split into
    pub chunk_count: u32,
    /// Index of first chunk in the chunk directory
    pub chunk_start_index: u64,
}

/// Metadata for a single compressed chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMeta {
    /// Unique chunk identifier
    pub chunk_id: u64,
    /// ID of the tensor this chunk belongs to
    pub tensor_id: u64,
    /// Offset of this chunk within the tensor (in elements)
    pub chunk_offset_in_tensor: u64,
    /// Uncompressed size in bytes
    pub uncompressed_size: u64,
    /// Compressed size in bytes
    pub compressed_size: u64,
    /// File offset where compressed data starts
    pub file_offset: u64,
    /// Codec ID used to compress this chunk (e.g., "rtc-raw-v1")
    pub codec_id: String,
    /// SHA-256 hash of original (uncompressed) chunk bytes
    pub chunk_sha256_original: [u8; 32],
    /// SHA-256 hash of compressed chunk bytes
    pub chunk_sha256_compressed: [u8; 32],
}

/// Architecture/config fields imported from an original model `config.json`.
///
/// This stays optional so older `.rllm` files continue to deserialize. Runtime
/// adapters should still validate tensor shapes because config metadata is a
/// hint/contract, not a replacement for container truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModelConfigMetadata {
    /// Normalized architecture/model type, e.g. `gpt_neox`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_hidden_layers: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intermediate_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_attention_heads: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_position_embeddings: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotary_pct: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotary_emb_base: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_norm_eps: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<u64>,
}

/// Minimal tokenizer vocabulary/config metadata persisted in `.rllm` global metadata.
///
/// This intentionally stores a runtime-ready `id_to_token` table instead of a
/// full HuggingFace tokenizer graph. Phase 5E uses it for a narrow text smoke
/// boundary over token-ID generation; richer BPE/normalizer fidelity can remain a
/// later tokenizer crate concern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TokenizerMetadata {
    /// Normalized tokenizer source/type, e.g. `hf-bpe` or `hf-wordlevel`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_type: Option<String>,
    /// Token strings indexed by token ID.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub id_to_token: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unk_token_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_token_id: Option<u64>,
}

/// Global metadata for the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetadata {
    /// Model name
    pub model_name: String,
    /// Architecture (e.g., "decoder-only-transformer")
    pub architecture: String,
    /// Source format (e.g., "safetensors")
    pub source_format: String,
    /// Whether compression is lossless
    pub lossless: bool,
    /// Default context length
    pub default_context_length: u64,
    /// Tokenizer type
    pub tokenizer_type: String,
    /// Tool that created this file
    pub created_by: String,
    /// Codec used for compression
    pub codec: String,
    /// Optional original architecture config fields needed by runtime adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_config: Option<ModelConfigMetadata>,
    /// Optional tokenizer vocabulary/config fields needed by text-boundary runtime adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerMetadata>,
}

impl GlobalMetadata {
    /// Create default metadata for testing
    pub fn new_test() -> Self {
        Self {
            model_name: "test-model".to_string(),
            architecture: "test".to_string(),
            source_format: "test".to_string(),
            lossless: true,
            default_context_length: 2048,
            tokenizer_type: "none".to_string(),
            created_by: "rllm-pack".to_string(),
            codec: "rtc-raw-v1".to_string(),
            model_config: None,
            tokenizer: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Fp16.size_bytes(), 2);
        assert_eq!(DType::Bf16.size_bytes(), 2);
        assert_eq!(DType::Fp32.size_bytes(), 4);
        assert_eq!(DType::Fp64.size_bytes(), 8);
        assert_eq!(DType::I8.size_bytes(), 1);
        assert_eq!(DType::U32.size_bytes(), 4);
    }

    #[test]
    fn test_tensor_meta_serialization() {
        let meta = TensorMeta {
            tensor_id: 42,
            name: "test.weight".to_string(),
            shape: vec![4096, 11008],
            dtype: DType::Bf16,
            original_size_bytes: 90177536,
            compressed_size_bytes: 73400320,
            original_sha256: [0u8; 32],
            chunk_count: 256,
            chunk_start_index: 1024,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let decoded: TensorMeta = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.tensor_id, 42);
        assert_eq!(decoded.name, "test.weight");
        assert_eq!(decoded.shape, vec![4096, 11008]);
        assert_eq!(decoded.dtype, DType::Bf16);
    }

    #[test]
    fn test_global_metadata_serialization() {
        let meta = GlobalMetadata::new_test();
        let json = serde_json::to_string(&meta).unwrap();
        let decoded: GlobalMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.model_name, "test-model");
        assert!(decoded.lossless);
    }

    #[test]
    fn test_global_metadata_preserves_optional_model_config_and_reads_legacy_json() {
        let legacy_json = r#"{
            "model_name": "legacy",
            "architecture": "gpt_neox",
            "source_format": "safetensors",
            "lossless": true,
            "default_context_length": 2048,
            "tokenizer_type": "none",
            "created_by": "rllm-cli",
            "codec": "auto"
        }"#;
        let legacy: GlobalMetadata = serde_json::from_str(legacy_json).unwrap();
        assert!(legacy.model_config.is_none());

        let mut meta = GlobalMetadata::new_test();
        meta.model_config = Some(ModelConfigMetadata {
            architecture_type: Some("gpt_neox".to_string()),
            num_hidden_layers: Some(2),
            hidden_size: Some(128),
            intermediate_size: Some(512),
            num_attention_heads: Some(4),
            max_position_embeddings: Some(4096),
            rotary_pct: Some(0.25),
            rotary_emb_base: Some(10_000.0),
            layer_norm_eps: Some(1e-5),
            vocab_size: Some(50_432),
        });

        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("model_config"));
        let decoded: GlobalMetadata = serde_json::from_str(&json).unwrap();
        let decoded_config = decoded.model_config.unwrap();
        assert_eq!(
            decoded_config.architecture_type.as_deref(),
            Some("gpt_neox")
        );
        assert_eq!(decoded_config.num_attention_heads, Some(4));
        assert_eq!(decoded_config.rotary_pct, Some(0.25));
    }

    #[test]
    fn test_global_metadata_preserves_optional_tokenizer_metadata_and_reads_legacy_json() {
        let legacy_json = r#"{
            "model_name": "legacy",
            "architecture": "gpt_neox",
            "source_format": "safetensors",
            "lossless": true,
            "default_context_length": 2048,
            "tokenizer_type": "none",
            "created_by": "rllm-cli",
            "codec": "auto"
        }"#;
        let legacy: GlobalMetadata = serde_json::from_str(legacy_json).unwrap();
        assert!(legacy.tokenizer.is_none());

        let mut meta = GlobalMetadata::new_test();
        meta.tokenizer_type = "hf-wordlevel".to_string();
        meta.tokenizer = Some(TokenizerMetadata {
            tokenizer_type: Some("hf-wordlevel".to_string()),
            id_to_token: vec!["A".to_string(), " B".to_string(), "<unk>".to_string()],
            unk_token_id: Some(2),
            bos_token_id: None,
            eos_token_id: None,
        });

        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("tokenizer"));
        let decoded: GlobalMetadata = serde_json::from_str(&json).unwrap();
        let tokenizer = decoded.tokenizer.unwrap();
        assert_eq!(tokenizer.tokenizer_type.as_deref(), Some("hf-wordlevel"));
        assert_eq!(tokenizer.id_to_token[1], " B");
        assert_eq!(tokenizer.unk_token_id, Some(2));
    }
}
