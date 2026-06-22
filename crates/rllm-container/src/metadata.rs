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
    /// 4-bit block-quantized type (32 elements per block)
    Q4_0,
    /// 8-bit block-quantized type (32 elements per block)
    Q8_0,
}

impl DType {
    /// Size in bytes of a single element of this dtype
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Fp16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Fp32 | DType::I32 | DType::U32 => 4,
            DType::Fp64 | DType::I64 | DType::U64 => 8,
            DType::I8 | DType::U8 | DType::Q4_0 | DType::Q8_0 => 1,
        }
    }

    /// Whether this dtype is quantized
    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Q4_0 | DType::Q8_0)
    }

    /// Calculate total bytes for a given number of elements of this dtype
    pub fn byte_size_for_elements(&self, count: usize) -> usize {
        match self {
            DType::Q4_0 => {
                let blocks = (count + 31) / 32;
                blocks * 18
            }
            DType::Q8_0 => {
                let blocks = (count + 31) / 32;
                blocks * 34
            }
            _ => count * self.size_bytes(),
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

/// Integrity metadata for a byte range inside one chunk.
///
/// Range offsets are relative to the start of the chunk payload, not the whole
/// tensor/file. `original_*` describes the decoded/original byte range;
/// `compressed_*` describes the corresponding encoded byte range. For raw
/// identity chunks these spans are identical. For future independently-compressed
/// tile blocks they may differ.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRangeMeta {
    /// Stable ordinal within the parent chunk.
    pub range_id: u32,
    /// Offset in decoded/original chunk bytes.
    pub original_offset: u64,
    /// Length in decoded/original chunk bytes.
    pub original_size: u64,
    /// Offset in compressed chunk bytes.
    pub compressed_offset: u64,
    /// Length in compressed chunk bytes.
    pub compressed_size: u64,
    /// SHA-256 hash of this decoded/original byte range.
    pub sha256_original: [u8; 32],
    /// SHA-256 hash of this compressed byte range.
    pub sha256_compressed: [u8; 32],
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
    /// Optional per-range integrity metadata for verified partial reads.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub range_checksums: Vec<ChunkRangeMeta>,
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
    pub use_parallel_residual: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms_norm_eps: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_key_value_heads: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tie_word_embeddings: Option<bool>,
    // --- Gemma-family fields (also usable by other models that set them) ---
    /// Explicit attention head dimension. Gemma sets this independent of
    /// `hidden_size / num_attention_heads` (e.g. Gemma 3 4B uses 256).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<u64>,
    /// Gemma attention query pre-scale base; attention scaled by
    /// `1/sqrt(query_pre_attn_scalar)` instead of `1/sqrt(head_dim)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f32>,
    /// Local sliding-window size (Gemma 2/3 interleave local + global attention).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<u64>,
    /// Period of the local:global interleave (Gemma 3: 6 → 5 local : 1 global).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_pattern: Option<u64>,
    /// RoPE base for local (sliding-window) layers; global layers use `rope_theta`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_local_base_freq: Option<f32>,
    /// MLP activation (e.g. `gelu_pytorch_tanh` for Gemma, `silu` for LLaMA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_activation: Option<String>,
    /// Linear RoPE-scaling factor for the global (non-sliding) layers
    /// (Gemma 3: `rope_scaling.factor = 8.0`, `rope_type = "linear"` → positions
    /// divided by this factor before computing the global rotary angle). Local
    /// layers are unscaled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling_factor: Option<f32>,
    // --- Qwen3.5 / Qwen3-Next hybrid linear-attention fields ---
    /// Per-layer operator schedule (`"linear_attention"` / `"full_attention"`). When
    /// absent, the interleave can be derived from `full_attention_interval`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Period of the full-attention interleave (Qwen3.5: 4 → every 4th layer is full attn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_attention_interval: Option<u64>,
    /// Gated-DeltaNet key/value head dim (Qwen3.5: 128 each).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_key_head_dim: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_value_head_dim: Option<u64>,
    /// Gated-DeltaNet key/value head counts (Qwen3.5: 16 each).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_num_key_heads: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_num_value_heads: Option<u64>,
    /// Depthwise causal short-conv kernel over q‖k‖v (Qwen3.5: 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_conv_kernel_dim: Option<u64>,
    /// Whether gated full-attention emits a per-head output gate from `q_proj`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_output_gate: Option<bool>,
    /// Fraction of `head_dim` that RoPE rotates (Qwen3.5: 0.25 → 64 of 256).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f32>,
    /// Multimodal-RoPE section split (collapses to ordinary RoPE for text-only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mrope_section: Option<Vec<u64>>,
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
    /// Optional HuggingFace BPE merge rules, in rank order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bpe_merges: Vec<(String, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unk_token_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_token_id: Option<u64>,
    /// Pre-tokenization scheme. `"byte_level"` (GPT-2 `Ġ`/`Ċ`, the default for
    /// back-compat) or `"metaspace"` (SentencePiece `▁` spaces, used by Gemma).
    /// Drives both encode pre-tokenization and decode surface rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_tokenizer: Option<String>,
    /// Whether `encode` prepends `bos_token_id` (SentencePiece-style models such
    /// as Gemma set `add_bos_token = true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_bos_token: Option<bool>,
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
    fn q4_0_dtype_uses_block_byte_size() {
        assert!(DType::Q4_0.is_quantized());
        assert_eq!(DType::Q4_0.byte_size_for_elements(0), 0);
        assert_eq!(DType::Q4_0.byte_size_for_elements(1), 18);
        assert_eq!(DType::Q4_0.byte_size_for_elements(32), 18);
        assert_eq!(DType::Q4_0.byte_size_for_elements(33), 36);
    }

    #[test]
    fn q8_0_dtype_uses_block_byte_size() {
        assert!(DType::Q8_0.is_quantized());
        assert_eq!(DType::Q8_0.byte_size_for_elements(0), 0);
        assert_eq!(DType::Q8_0.byte_size_for_elements(1), 34);
        assert_eq!(DType::Q8_0.byte_size_for_elements(32), 34);
        assert_eq!(DType::Q8_0.byte_size_for_elements(33), 68);
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
    fn test_chunk_meta_preserves_optional_range_checksums_and_reads_legacy_json() {
        let legacy_json = r#"{
            "chunk_id": 7,
            "tensor_id": 3,
            "chunk_offset_in_tensor": 0,
            "uncompressed_size": 16,
            "compressed_size": 16,
            "file_offset": 44,
            "codec_id": "rtc-raw-v1",
            "chunk_sha256_original": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            "chunk_sha256_compressed": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        }"#;
        let legacy: ChunkMeta = serde_json::from_str(legacy_json).unwrap();
        assert!(legacy.range_checksums.is_empty());

        let mut meta = legacy;
        meta.range_checksums.push(ChunkRangeMeta {
            range_id: 0,
            original_offset: 4,
            original_size: 4,
            compressed_offset: 4,
            compressed_size: 4,
            sha256_original: [1u8; 32],
            sha256_compressed: [2u8; 32],
        });
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("range_checksums"));
        let decoded: ChunkMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.range_checksums.len(), 1);
        assert_eq!(decoded.range_checksums[0].original_offset, 4);
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
            use_parallel_residual: Some(true),
            vocab_size: Some(50_432),
            num_key_value_heads: None,
            rms_norm_eps: None,
            rope_theta: None,
            tie_word_embeddings: None,
            ..Default::default()
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
        assert_eq!(decoded_config.use_parallel_residual, Some(true));
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
            bpe_merges: Vec::new(),
            unk_token_id: Some(2),
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
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
