//! TensorCodec trait definition

use crate::error::Result;

/// Metadata about a tensor being encoded
#[derive(Debug, Clone)]
pub struct EncodeMeta {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<u64>,
    /// Data type string (e.g., "fp16", "bf16", "fp32")
    pub dtype: String,
}

/// Metadata about a compressed chunk needed for decoding
#[derive(Debug, Clone)]
pub struct DecodeMeta {
    /// Codec ID that was used to encode
    pub codec_id: String,
    /// Original uncompressed size
    pub uncompressed_size: u64,
}

/// Result of encoding a chunk
#[derive(Debug, Clone)]
pub struct EncodedChunk {
    /// Codec ID used
    pub codec_id: String,
    /// Compressed bytes
    pub data: Vec<u8>,
    /// Original uncompressed size
    pub original_size: u64,
}

/// Trait for lossless tensor codecs
///
/// # Contract
///
/// For any valid input:
/// ```text
/// let encoded = codec.encode(input, meta)?;
/// let decoded = codec.decode(&encoded.data, &decode_meta)?;
/// assert_eq!(decoded, input);
/// ```
pub trait TensorCodec: Send + Sync {
    /// Unique identifier for this codec (e.g., "rtc-raw-v1")
    fn id(&self) -> &'static str;

    /// Encode raw tensor bytes into compressed form
    fn encode(&self, input: &[u8], meta: &EncodeMeta) -> Result<EncodedChunk>;

    /// Decode compressed bytes back to original tensor bytes
    fn decode(&self, encoded: &[u8], meta: &DecodeMeta) -> Result<Vec<u8>>;

    /// Verify that decode(encode(input)) == input
    ///
    /// This is the fundamental correctness test for any codec.
    fn verify_roundtrip(&self, input: &[u8], meta: &EncodeMeta) -> Result<bool> {
        let encoded = self.encode(input, meta)?;
        let decode_meta = DecodeMeta {
            codec_id: encoded.codec_id.clone(),
            uncompressed_size: encoded.original_size,
        };
        let decoded = self.decode(&encoded.data, &decode_meta)?;
        Ok(decoded == input)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Test helper: verify round-trip for any codec
    pub fn assert_roundtrip(codec: &dyn TensorCodec, data: &[u8], name: &str) {
        let meta = EncodeMeta {
            name: name.to_string(),
            shape: vec![data.len() as u64],
            dtype: "u8".to_string(),
        };

        let result = codec.verify_roundtrip(data, &meta);
        assert!(result.is_ok(), "Round-trip failed: {:?}", result.err());
        assert!(result.unwrap(), "Round-trip returned false");
    }
}
