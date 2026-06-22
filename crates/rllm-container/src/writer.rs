// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! RLLM file writer

use crate::error::{ContainerError, Result};
use crate::header::RllmHeader;
use crate::metadata::{ChunkMeta, ChunkRangeMeta, GlobalMetadata, TensorMeta};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// Writer for .spsa files
///
/// # Usage
///
/// See the integration tests in `tests/` for complete examples.
pub struct RllmWriter {
    file: BufWriter<File>,
    tensors: Vec<TensorMeta>,
    chunks: Vec<ChunkMeta>,
    metadata: GlobalMetadata,
    current_offset: u64,
}

/// Byte-range mapping used to build persisted per-range chunk checksums.
///
/// Offsets are relative to the parent chunk's original/compressed payloads.
/// This deliberately stays codec-agnostic: raw identity chunks can use matching
/// spans, while future independently-compressed tile blocks can provide smaller
/// compressed spans for each original tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRangeSpec {
    pub original_offset: u64,
    pub original_size: u64,
    pub compressed_offset: u64,
    pub compressed_size: u64,
}

impl RllmWriter {
    /// Create a new writer for the given output path
    pub fn new(path: impl AsRef<Path>, metadata: GlobalMetadata) -> Result<Self> {
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);

        // Write placeholder header (will be updated in finalize)
        let header = RllmHeader::new();
        writer.write_all(&header.to_bytes())?;

        Ok(Self {
            file: writer,
            tensors: Vec::new(),
            chunks: Vec::new(),
            metadata,
            current_offset: RllmHeader::SIZE as u64,
        })
    }

    /// Add a tensor metadata entry and return its ID
    pub fn add_tensor(&mut self, meta: TensorMeta) -> u64 {
        let id = meta.tensor_id;
        self.tensors.push(meta);
        id
    }

    /// Write a compressed chunk and return its ID.
    ///
    /// The chunk data is written immediately to the file.
    /// `data` is the compressed bytes, `original_data` is the uncompressed bytes
    /// (used for checksum verification).
    pub fn write_chunk(
        &mut self,
        tensor_id: u64,
        codec_id: &str,
        data: &[u8],
        original_data: &[u8],
        chunk_offset_in_tensor: u64,
    ) -> Result<u64> {
        self.write_chunk_with_range_specs(
            tensor_id,
            codec_id,
            data,
            original_data,
            chunk_offset_in_tensor,
            &[],
        )
    }

    /// Write a compressed chunk and attach explicit per-range checksum metadata.
    pub fn write_chunk_with_range_specs(
        &mut self,
        tensor_id: u64,
        codec_id: &str,
        data: &[u8],
        original_data: &[u8],
        chunk_offset_in_tensor: u64,
        ranges: &[ChunkRangeSpec],
    ) -> Result<u64> {
        let chunk_id = self.chunks.len() as u64;
        let file_offset = self.current_offset;

        // Compute checksums
        let original_hash: [u8; 32] = Sha256::digest(original_data).into();
        let compressed_hash: [u8; 32] = Sha256::digest(data).into();
        let range_checksums = build_range_checksums(data, original_data, ranges)?;

        let chunk_meta = ChunkMeta {
            chunk_id,
            tensor_id,
            chunk_offset_in_tensor,
            uncompressed_size: original_data.len() as u64,
            compressed_size: data.len() as u64,
            file_offset,
            codec_id: codec_id.to_string(),
            chunk_sha256_original: original_hash,
            chunk_sha256_compressed: compressed_hash,
            range_checksums,
        };

        // Write compressed data
        self.file.write_all(data)?;
        self.current_offset += data.len() as u64;

        self.chunks.push(chunk_meta);
        Ok(chunk_id)
    }

    /// Write an identity-mapped chunk with fixed-size per-range checksums.
    ///
    /// This helper is valid only when compressed bytes have the same byte layout
    /// as original bytes (for example `rtc-raw-v1`). It is useful as the first
    /// verified metadata path before pack-time tile-compressed blocks exist.
    pub fn write_chunk_with_identity_range_checksums(
        &mut self,
        tensor_id: u64,
        codec_id: &str,
        data: &[u8],
        original_data: &[u8],
        chunk_offset_in_tensor: u64,
        range_size_bytes: u64,
    ) -> Result<u64> {
        if data.len() != original_data.len() {
            return Err(ContainerError::InvalidRange {
                context: "identity chunk range checksums require equal compressed/original sizes"
                    .to_string(),
                offset: data.len() as u64,
                len: original_data.len() as u64,
                size: data.len().max(original_data.len()) as u64,
            });
        }
        if range_size_bytes == 0 {
            return Err(ContainerError::InvalidRange {
                context: "identity chunk range size".to_string(),
                offset: 0,
                len: 0,
                size: data.len() as u64,
            });
        }

        let total = data.len() as u64;
        let mut ranges = Vec::new();
        let mut offset = 0u64;
        while offset < total {
            let len = range_size_bytes.min(total - offset);
            ranges.push(ChunkRangeSpec {
                original_offset: offset,
                original_size: len,
                compressed_offset: offset,
                compressed_size: len,
            });
            offset += len;
        }

        self.write_chunk_with_range_specs(
            tensor_id,
            codec_id,
            data,
            original_data,
            chunk_offset_in_tensor,
            &ranges,
        )
    }

    /// Finalize the file
    ///
    /// This writes:
    /// 1. Global metadata (JSON, length-prefixed)
    /// 2. Tensor directory (JSON, length-prefixed)
    /// 3. Chunk directory (JSON, length-prefixed)
    /// 4. Updates header with all offsets
    pub fn finalize(mut self) -> Result<()> {
        // Update tensor metadata from actual chunks
        for tensor in &mut self.tensors {
            let tensor_chunks: Vec<_> = self
                .chunks
                .iter()
                .filter(|c| c.tensor_id == tensor.tensor_id)
                .collect();
            tensor.chunk_count = tensor_chunks.len() as u32;
            tensor.compressed_size_bytes = tensor_chunks.iter().map(|c| c.compressed_size).sum();
            if !tensor_chunks.is_empty() {
                tensor.chunk_start_index = tensor_chunks[0].chunk_id;
            }
        }

        // 1. Write global metadata
        let metadata_offset = self.current_offset;
        let json = serde_json::to_string_pretty(&self.metadata)?;
        let bytes = json.as_bytes();
        self.file.write_all(&(bytes.len() as u64).to_le_bytes())?;
        self.file.write_all(bytes)?;
        self.current_offset += 8 + bytes.len() as u64;

        // 2. Write tensor directory
        let tensor_dir_offset = self.current_offset;
        let tensor_json = serde_json::to_string_pretty(&self.tensors)?;
        let tensor_bytes = tensor_json.as_bytes();
        self.file
            .write_all(&(tensor_bytes.len() as u64).to_le_bytes())?;
        self.file.write_all(tensor_bytes)?;
        self.current_offset += 8 + tensor_bytes.len() as u64;

        // 3. Write chunk directory
        let chunk_dir_offset = self.current_offset;
        let chunk_json = serde_json::to_string_pretty(&self.chunks)?;
        let chunk_bytes = chunk_json.as_bytes();
        self.file
            .write_all(&(chunk_bytes.len() as u64).to_le_bytes())?;
        self.file.write_all(chunk_bytes)?;
        self.current_offset += 8 + chunk_bytes.len() as u64;

        // 4. Update header with actual offsets
        let mut header = RllmHeader::new();
        header.metadata_offset = metadata_offset;
        header.tensor_dir_offset = tensor_dir_offset;
        header.chunk_dir_offset = chunk_dir_offset;
        header.data_start_offset = RllmHeader::SIZE as u64;

        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header.to_bytes())?;
        self.file.flush()?;

        Ok(())
    }
}

fn build_range_checksums(
    data: &[u8],
    original_data: &[u8],
    ranges: &[ChunkRangeSpec],
) -> Result<Vec<ChunkRangeMeta>> {
    ranges
        .iter()
        .enumerate()
        .map(|(range_id, spec)| {
            let original = slice_checked(
                original_data,
                spec.original_offset,
                spec.original_size,
                "original chunk range",
            )?;
            let compressed = slice_checked(
                data,
                spec.compressed_offset,
                spec.compressed_size,
                "compressed chunk range",
            )?;
            Ok(ChunkRangeMeta {
                range_id: u32::try_from(range_id).map_err(|_| ContainerError::InvalidRange {
                    context: "too many chunk ranges".to_string(),
                    offset: range_id as u64,
                    len: 1,
                    size: u32::MAX as u64,
                })?,
                original_offset: spec.original_offset,
                original_size: spec.original_size,
                compressed_offset: spec.compressed_offset,
                compressed_size: spec.compressed_size,
                sha256_original: Sha256::digest(original).into(),
                sha256_compressed: Sha256::digest(compressed).into(),
            })
        })
        .collect()
}

fn slice_checked<'a>(data: &'a [u8], offset: u64, len: u64, context: &str) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| ContainerError::InvalidRange {
            context: context.to_string(),
            offset,
            len,
            size: data.len() as u64,
        })?;
    if end > data.len() as u64 {
        return Err(ContainerError::InvalidRange {
            context: context.to_string(),
            offset,
            len,
            size: data.len() as u64,
        });
    }
    let start = usize::try_from(offset).map_err(|_| ContainerError::InvalidRange {
        context: context.to_string(),
        offset,
        len,
        size: data.len() as u64,
    })?;
    let end = usize::try_from(end).map_err(|_| ContainerError::InvalidRange {
        context: context.to_string(),
        offset,
        len,
        size: data.len() as u64,
    })?;
    Ok(&data[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::DType;
    use std::io::Read;

    #[test]
    fn test_writer_creates_file() {
        let temp = std::env::temp_dir().join("test_writer.spsa");
        let metadata = GlobalMetadata::new_test();
        let writer = RllmWriter::new(&temp, metadata).unwrap();
        writer.finalize().unwrap();

        assert!(temp.exists());
        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn test_writer_with_tensor() {
        let temp = std::env::temp_dir().join("test_writer_tensor.spsa");
        let metadata = GlobalMetadata::new_test();
        let mut writer = RllmWriter::new(&temp, metadata).unwrap();

        let tensor_meta = TensorMeta {
            tensor_id: 0,
            name: "test.weight".to_string(),
            shape: vec![4, 4],
            dtype: DType::Fp32,
            original_size_bytes: 64,
            compressed_size_bytes: 64,
            original_sha256: [0u8; 32],
            chunk_count: 1,
            chunk_start_index: 0,
        };
        writer.add_tensor(tensor_meta);

        let data = vec![1u8; 64];
        writer
            .write_chunk(0, "rtc-raw-v1", &data, &data, 0)
            .unwrap();

        writer.finalize().unwrap();

        let mut file = File::open(&temp).unwrap();
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"RLLM");

        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn test_writer_persists_identity_range_checksums() {
        let temp = std::env::temp_dir().join("test_writer_range_checksums.spsa");
        let metadata = GlobalMetadata::new_test();
        let mut writer = RllmWriter::new(&temp, metadata).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "test.weight".to_string(),
            shape: vec![10],
            dtype: DType::U8,
            original_size_bytes: 10,
            compressed_size_bytes: 10,
            original_sha256: [0u8; 32],
            chunk_count: 1,
            chunk_start_index: 0,
        });

        let data: Vec<u8> = (0..10).collect();
        writer
            .write_chunk_with_identity_range_checksums(0, "rtc-raw-v1", &data, &data, 0, 4)
            .unwrap();
        writer.finalize().unwrap();

        let reader = crate::reader::RllmReader::open(&temp).unwrap();
        let chunk = &reader.list_chunks()[0];
        assert_eq!(chunk.range_checksums.len(), 3);
        assert_eq!(chunk.range_checksums[0].original_offset, 0);
        assert_eq!(chunk.range_checksums[0].original_size, 4);
        assert_eq!(chunk.range_checksums[1].original_offset, 4);
        assert_eq!(chunk.range_checksums[1].original_size, 4);
        assert_eq!(chunk.range_checksums[2].original_offset, 8);
        assert_eq!(chunk.range_checksums[2].original_size, 2);
        let first_original_hash: [u8; 32] = Sha256::digest(&data[0..4]).into();
        let last_compressed_hash: [u8; 32] = Sha256::digest(&data[8..10]).into();
        assert_eq!(
            chunk.range_checksums[0].sha256_original,
            first_original_hash
        );
        assert_eq!(
            chunk.range_checksums[2].sha256_compressed,
            last_compressed_hash
        );

        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn test_writer_rejects_out_of_bounds_range_spec() {
        let temp = std::env::temp_dir().join("test_writer_bad_range.spsa");
        let metadata = GlobalMetadata::new_test();
        let mut writer = RllmWriter::new(&temp, metadata).unwrap();
        let data = vec![1u8; 8];
        let err = writer
            .write_chunk_with_range_specs(
                0,
                "rtc-raw-v1",
                &data,
                &data,
                0,
                &[ChunkRangeSpec {
                    original_offset: 6,
                    original_size: 4,
                    compressed_offset: 0,
                    compressed_size: 4,
                }],
            )
            .unwrap_err();
        assert!(matches!(err, ContainerError::InvalidRange { .. }));
        std::fs::remove_file(&temp).ok();
    }
}
