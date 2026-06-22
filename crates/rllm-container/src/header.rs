// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! RLLM file header structures

use serde::{Deserialize, Serialize};

/// The file header for .spsa files
///
/// Layout (44 bytes):
/// - magic: 4 bytes ("RLLM")
/// - version: 4 bytes (u32, little-endian)
/// - endian: 1 byte (0 = little-endian)
/// - reserved: 3 bytes
/// - metadata_offset: 8 bytes (u64, file offset to global metadata)
/// - tensor_dir_offset: 8 bytes (u64, file offset to tensor directory)
/// - chunk_dir_offset: 8 bytes (u64, file offset to chunk directory)
/// - data_start_offset: 8 bytes (u64, file offset where chunk data starts)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RllmHeader {
    /// Magic bytes: must be "RLLM"
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Endianness: 0 = little-endian
    pub endian: u8,
    /// Reserved bytes (must be zero)
    pub reserved: [u8; 3],
    /// Offset to global metadata block
    pub metadata_offset: u64,
    /// Offset to tensor directory
    pub tensor_dir_offset: u64,
    /// Offset to chunk directory
    pub chunk_dir_offset: u64,
    /// Offset where chunk data starts
    pub data_start_offset: u64,
}

impl RllmHeader {
    /// Size of the header in bytes
    pub const SIZE: usize = 44;

    /// Create a new header with default values
    pub fn new() -> Self {
        Self {
            magic: *crate::RLLM_MAGIC,
            version: crate::RLLM_VERSION,
            endian: 0, // little-endian
            reserved: [0; 3],
            metadata_offset: 0,
            tensor_dir_offset: 0,
            chunk_dir_offset: 0,
            data_start_offset: 0,
        }
    }

    /// Validate the header
    pub fn validate(&self) -> crate::error::Result<()> {
        if &self.magic != crate::RLLM_MAGIC {
            return Err(crate::error::ContainerError::InvalidMagic);
        }
        if self.version != crate::RLLM_VERSION {
            return Err(crate::error::ContainerError::UnsupportedVersion(
                self.version,
            ));
        }
        Ok(())
    }

    /// Serialize header to bytes (little-endian)
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8] = self.endian;
        buf[9..12].copy_from_slice(&self.reserved);
        buf[12..20].copy_from_slice(&self.metadata_offset.to_le_bytes());
        buf[20..28].copy_from_slice(&self.tensor_dir_offset.to_le_bytes());
        buf[28..36].copy_from_slice(&self.chunk_dir_offset.to_le_bytes());
        buf[36..44].copy_from_slice(&self.data_start_offset.to_le_bytes());
        buf
    }

    /// Deserialize header from bytes (little-endian)
    pub fn from_bytes(buf: &[u8; Self::SIZE]) -> Self {
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);

        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let endian = buf[8];
        let mut reserved = [0u8; 3];
        reserved.copy_from_slice(&buf[9..12]);
        let metadata_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let tensor_dir_offset = u64::from_le_bytes(buf[20..28].try_into().unwrap());
        let chunk_dir_offset = u64::from_le_bytes(buf[28..36].try_into().unwrap());
        let data_start_offset = u64::from_le_bytes(buf[36..44].try_into().unwrap());

        Self {
            magic,
            version,
            endian,
            reserved,
            metadata_offset,
            tensor_dir_offset,
            chunk_dir_offset,
            data_start_offset,
        }
    }
}

impl Default for RllmHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let mut header = RllmHeader::new();
        header.metadata_offset = 1024;
        header.tensor_dir_offset = 2048;
        header.chunk_dir_offset = 4096;
        header.data_start_offset = 44;

        let bytes = header.to_bytes();
        let decoded = RllmHeader::from_bytes(&bytes);

        assert_eq!(decoded.magic, *crate::RLLM_MAGIC);
        assert_eq!(decoded.version, crate::RLLM_VERSION);
        assert_eq!(decoded.endian, 0);
        assert_eq!(decoded.metadata_offset, 1024);
        assert_eq!(decoded.tensor_dir_offset, 2048);
        assert_eq!(decoded.chunk_dir_offset, 4096);
        assert_eq!(decoded.data_start_offset, 44);
    }

    #[test]
    fn test_header_validation() {
        let header = RllmHeader::new();
        assert!(header.validate().is_ok());

        let mut bad = header.clone();
        bad.magic = *b"XXXX";
        assert!(bad.validate().is_err());

        let mut bad_version = header;
        bad_version.version = 999;
        assert!(bad_version.validate().is_err());
    }
}
