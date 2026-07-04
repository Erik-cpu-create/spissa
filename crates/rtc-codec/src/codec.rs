// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

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

/// Byte range requested from the decoded/original chunk payload.
///
/// Ranges are expressed in uncompressed bytes. Codec implementations that can
/// map this range directly to encoded bytes may override `decode_range` without
/// materializing the full chunk; other codecs use the default full-decode +
/// slice fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRange {
    /// Start offset in the decoded/original chunk bytes.
    pub offset: u64,
    /// Number of decoded/original bytes requested.
    pub len: u64,
}

impl DecodeRange {
    pub fn new(offset: u64, len: u64) -> Self {
        Self { offset, len }
    }

    pub fn end(self) -> Result<u64> {
        self.offset.checked_add(self.len).ok_or_else(|| {
            crate::error::CodecError::InvalidData("decode range end overflow".to_string())
        })
    }

    pub fn validate(self, uncompressed_size: u64) -> Result<()> {
        let end = self.end()?;
        if end > uncompressed_size {
            return Err(crate::error::CodecError::InvalidData(format!(
                "decode range [{}, {}) exceeds uncompressed size {}",
                self.offset, end, uncompressed_size
            )));
        }
        Ok(())
    }

    pub fn slice(self, decoded: &[u8]) -> Result<Vec<u8>> {
        self.validate(decoded.len() as u64)?;
        let start = self.offset as usize;
        let end = self.end()? as usize;
        Ok(decoded[start..end].to_vec())
    }
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

    /// Whether `decode_range` avoids materializing the full decoded chunk.
    ///
    /// Runtime memory accounting should only reserve the requested range when
    /// this returns true. Otherwise callers should assume the implementation may
    /// allocate a full decoded chunk internally.
    fn supports_native_range_decode(&self) -> bool {
        false
    }

    /// Decode a byte range from the original chunk payload.
    ///
    /// The default implementation preserves correctness for every codec by
    /// decoding the full chunk and slicing the requested range. Codecs with a
    /// direct encoded-range mapping should override this method and return true
    /// from `supports_native_range_decode`.
    fn decode_range(
        &self,
        encoded: &[u8],
        meta: &DecodeMeta,
        range: DecodeRange,
    ) -> Result<Vec<u8>> {
        range.validate(meta.uncompressed_size)?;
        let decoded = self.decode(encoded, meta)?;
        range.slice(&decoded)
    }

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

    #[test]
    fn decode_range_validates_bounds_and_slices() {
        let data = b"abcdefghijkl";
        let range = DecodeRange::new(3, 4);
        assert_eq!(range.end().unwrap(), 7);
        assert_eq!(range.slice(data).unwrap(), b"defg");

        let err = DecodeRange::new(10, 3).slice(data).unwrap_err();
        assert!(format!("{err}").contains("exceeds uncompressed size"));
    }
}
