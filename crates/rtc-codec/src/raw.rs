// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! rtc-raw-v1: Identity codec (no compression)
//!
//! This codec passes data through unchanged. It serves as:
//! 1. A fallback when other codecs make data larger
//! 2. A baseline for testing the codec framework
//! 3. A reference implementation for the TensorCodec trait

use crate::codec::{DecodeMeta, DecodeRange, EncodeMeta, EncodedChunk, TensorCodec};
use crate::error::{CodecError, Result};
use crate::CODEC_RAW_V1;

/// The raw (identity) codec - no compression
pub struct RawCodec;

impl RawCodec {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RawCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorCodec for RawCodec {
    fn id(&self) -> &'static str {
        CODEC_RAW_V1
    }

    fn encode(&self, input: &[u8], _meta: &EncodeMeta) -> Result<EncodedChunk> {
        Ok(EncodedChunk {
            codec_id: CODEC_RAW_V1.to_string(),
            data: input.to_vec(),
            original_size: input.len() as u64,
        })
    }

    fn decode(&self, encoded: &[u8], meta: &DecodeMeta) -> Result<Vec<u8>> {
        if encoded.len() as u64 != meta.uncompressed_size {
            return Err(CodecError::InvalidData(format!(
                "Size mismatch: encoded={} expected={}",
                encoded.len(),
                meta.uncompressed_size
            )));
        }
        Ok(encoded.to_vec())
    }

    fn supports_native_range_decode(&self) -> bool {
        true
    }

    fn decode_range(
        &self,
        encoded: &[u8],
        meta: &DecodeMeta,
        range: DecodeRange,
    ) -> Result<Vec<u8>> {
        if encoded.len() as u64 != meta.uncompressed_size {
            return Err(CodecError::InvalidData(format!(
                "Size mismatch: encoded={} expected={}",
                encoded.len(),
                meta.uncompressed_size
            )));
        }
        range.slice(encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::tests::assert_roundtrip;

    #[test]
    fn test_raw_empty() {
        let codec = RawCodec::new();
        assert_roundtrip(&codec, b"", "empty");
    }

    #[test]
    fn test_raw_small() {
        let codec = RawCodec::new();
        assert_roundtrip(&codec, b"hello world", "small");
    }

    #[test]
    fn test_raw_zeros() {
        let codec = RawCodec::new();
        let data = vec![0u8; 1024];
        assert_roundtrip(&codec, &data, "zeros");
    }

    #[test]
    fn test_raw_ones() {
        let codec = RawCodec::new();
        let data = vec![0xFFu8; 1024];
        assert_roundtrip(&codec, &data, "ones");
    }

    #[test]
    fn test_raw_random() {
        let codec = RawCodec::new();
        // Pseudo-random data
        let data: Vec<u8> = (0..4096).map(|i| ((i * 7 + 13) % 256) as u8).collect();
        assert_roundtrip(&codec, &data, "random");
    }

    #[test]
    fn test_raw_encode_preserves_data() {
        let codec = RawCodec::new();
        let input = b"test data 12345";
        let meta = EncodeMeta {
            name: "test".to_string(),
            shape: vec![input.len() as u64],
            dtype: "u8".to_string(),
        };

        let encoded = codec.encode(input, &meta).unwrap();
        assert_eq!(encoded.data, input);
        assert_eq!(encoded.codec_id, CODEC_RAW_V1);
    }

    #[test]
    fn test_raw_decode_size_check() {
        let codec = RawCodec::new();
        let data = vec![1u8; 100];
        let meta = DecodeMeta {
            codec_id: CODEC_RAW_V1.to_string(),
            uncompressed_size: 50, // wrong size
        };

        let result = codec.decode(&data, &meta);
        assert!(result.is_err());
    }

    #[test]
    fn test_raw_decode_range_slices_without_full_decode_contract() {
        let codec = RawCodec::new();
        let data = b"0123456789abcdef";
        let meta = DecodeMeta {
            codec_id: CODEC_RAW_V1.to_string(),
            uncompressed_size: data.len() as u64,
        };

        assert!(codec.supports_native_range_decode());
        let decoded = codec
            .decode_range(data, &meta, DecodeRange::new(4, 6))
            .unwrap();
        assert_eq!(decoded, b"456789");
    }
}
