// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! rtc-rle-v1: Run-Length Encoding codec
//!
//! Encodes runs of repeated bytes as (count, value) pairs.
//! Effective for data with many consecutive identical bytes.

use crate::codec::{DecodeMeta, EncodeMeta, EncodedChunk, TensorCodec};
use crate::error::{CodecError, Result};
use crate::CODEC_RLE_V1;

/// RLE codec - run-length encoding
pub struct RleCodec;

impl RleCodec {
    pub fn new() -> Self {
        Self
    }

    /// Encode data using RLE
    ///
    /// Format: [count: u8, value: u8, count: u8, value: u8, ...]
    /// Max run length: 255 bytes (u8 max)
    fn encode_rle(&self, input: &[u8]) -> Vec<u8> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::new();
        let mut i = 0;

        while i < input.len() {
            let value = input[i];
            let mut count = 1u8;

            // Count consecutive identical bytes
            while i + (count as usize) < input.len()
                && input[i + count as usize] == value
                && count < 255
            {
                count += 1;
            }

            output.push(count);
            output.push(value);
            i += count as usize;
        }

        output
    }

    /// Decode RLE data
    fn decode_rle(&self, encoded: &[u8]) -> Result<Vec<u8>> {
        if !encoded.len().is_multiple_of(2) {
            return Err(CodecError::InvalidData(
                "RLE encoded data must have even length".to_string(),
            ));
        }

        let mut output = Vec::new();

        for i in (0..encoded.len()).step_by(2) {
            let count = encoded[i];
            let value = encoded[i + 1];

            for _ in 0..count {
                output.push(value);
            }
        }

        Ok(output)
    }
}

impl Default for RleCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorCodec for RleCodec {
    fn id(&self) -> &'static str {
        CODEC_RLE_V1
    }

    fn encode(&self, input: &[u8], _meta: &EncodeMeta) -> Result<EncodedChunk> {
        let encoded = self.encode_rle(input);

        Ok(EncodedChunk {
            codec_id: CODEC_RLE_V1.to_string(),
            data: encoded,
            original_size: input.len() as u64,
        })
    }

    fn decode(&self, encoded: &[u8], meta: &DecodeMeta) -> Result<Vec<u8>> {
        let decoded = self.decode_rle(encoded)?;

        if decoded.len() as u64 != meta.uncompressed_size {
            return Err(CodecError::InvalidData(format!(
                "Decoded size mismatch: expected {}, got {}",
                meta.uncompressed_size,
                decoded.len()
            )));
        }

        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::tests::assert_roundtrip;

    #[test]
    fn test_rle_empty() {
        let codec = RleCodec::new();
        assert_roundtrip(&codec, b"", "empty");
    }

    #[test]
    fn test_rle_all_zeros() {
        let codec = RleCodec::new();
        let data = vec![0u8; 1000];
        assert_roundtrip(&codec, &data, "zeros");

        // Should compress well
        let meta = EncodeMeta {
            name: "test".to_string(),
            shape: vec![1000],
            dtype: "u8".to_string(),
        };
        let encoded = codec.encode(&data, &meta).unwrap();
        assert!(encoded.data.len() < data.len() / 10);
    }

    #[test]
    fn test_rle_all_ones() {
        let codec = RleCodec::new();
        let data = vec![0xFFu8; 1000];
        assert_roundtrip(&codec, &data, "ones");

        // Should compress well
        let meta = EncodeMeta {
            name: "test".to_string(),
            shape: vec![1000],
            dtype: "u8".to_string(),
        };
        let encoded = codec.encode(&data, &meta).unwrap();
        assert!(encoded.data.len() < data.len() / 10);
    }

    #[test]
    fn test_rle_repeating_pattern() {
        let codec = RleCodec::new();
        // Pattern: 10 zeros, 10 ones, repeat
        let mut data = Vec::new();
        for _ in 0..50 {
            data.extend(vec![0u8; 10]);
            data.extend(vec![1u8; 10]);
        }
        assert_roundtrip(&codec, &data, "repeating");

        // Should compress well
        let meta = EncodeMeta {
            name: "test".to_string(),
            shape: vec![data.len() as u64],
            dtype: "u8".to_string(),
        };
        let encoded = codec.encode(&data, &meta).unwrap();
        assert!(encoded.data.len() <= data.len() / 5);
    }

    #[test]
    fn test_rle_random_data() {
        let codec = RleCodec::new();
        // Random data won't compress well with RLE
        let data: Vec<u8> = (0..1000).map(|i| ((i * 7 + 13) % 256) as u8).collect();
        assert_roundtrip(&codec, &data, "random");

        // Should expand (RLE overhead)
        let meta = EncodeMeta {
            name: "test".to_string(),
            shape: vec![1000],
            dtype: "u8".to_string(),
        };
        let encoded = codec.encode(&data, &meta).unwrap();
        assert!(encoded.data.len() > data.len());
    }

    #[test]
    fn test_rle_single_byte() {
        let codec = RleCodec::new();
        assert_roundtrip(&codec, b"x", "single");
    }

    #[test]
    fn test_rle_max_run() {
        let codec = RleCodec::new();
        // Test run of exactly 255 bytes (max for u8)
        let data = vec![42u8; 255];
        assert_roundtrip(&codec, &data, "max_run");

        // Test run of 256 bytes (should split into two runs)
        let data = vec![42u8; 256];
        assert_roundtrip(&codec, &data, "over_max_run");
    }

    #[test]
    fn test_rle_decode_invalid_length() {
        let codec = RleCodec::new();
        let meta = DecodeMeta {
            codec_id: CODEC_RLE_V1.to_string(),
            uncompressed_size: 10,
        };

        // Odd length is invalid
        let result = codec.decode(&[1, 2, 3], &meta);
        assert!(result.is_err());
    }
}
