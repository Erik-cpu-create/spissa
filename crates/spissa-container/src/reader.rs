// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

//! Spissa file reader

use crate::error::{ContainerError, Result};
use crate::header::SpissaHeader;
use crate::metadata::{ChunkMeta, GlobalMetadata, TensorMeta};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Reader for .spsa files
///
/// Uses memory-mapped I/O for zero-copy chunk reads at runtime.
/// The file is initially parsed via buffered I/O (for headers/metadata),
/// then the entire file is memory-mapped for zero-copy chunk access.
pub struct SpissaReader {
    mmap: Mmap,
    #[allow(dead_code)]
    header: SpissaHeader,
    metadata: GlobalMetadata,
    tensors: Vec<TensorMeta>,
    chunks: Vec<ChunkMeta>,
}

impl SpissaReader {
    /// Open and parse a .spsa file
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header_bytes = [0u8; SpissaHeader::SIZE];
        reader.read_exact(&mut header_bytes)?;
        let header = SpissaHeader::from_bytes(&header_bytes);
        header.validate()?;

        // Read metadata
        reader.seek(SeekFrom::Start(header.metadata_offset))?;
        let mut metadata_len_bytes = [0u8; 8];
        reader.read_exact(&mut metadata_len_bytes)?;
        let metadata_len = u64::from_le_bytes(metadata_len_bytes);

        let mut metadata_bytes = vec![0u8; metadata_len as usize];
        reader.read_exact(&mut metadata_bytes)?;
        let metadata: GlobalMetadata = serde_json::from_slice(&metadata_bytes)?;

        // Read tensor directory
        reader.seek(SeekFrom::Start(header.tensor_dir_offset))?;
        let mut tensor_dir_len_bytes = [0u8; 8];
        reader.read_exact(&mut tensor_dir_len_bytes)?;
        let tensor_dir_len = u64::from_le_bytes(tensor_dir_len_bytes);

        let mut tensor_dir_bytes = vec![0u8; tensor_dir_len as usize];
        reader.read_exact(&mut tensor_dir_bytes)?;
        let tensors: Vec<TensorMeta> = serde_json::from_slice(&tensor_dir_bytes)?;

        // Read chunk directory
        reader.seek(SeekFrom::Start(header.chunk_dir_offset))?;
        let mut chunk_dir_len_bytes = [0u8; 8];
        reader.read_exact(&mut chunk_dir_len_bytes)?;
        let chunk_dir_len = u64::from_le_bytes(chunk_dir_len_bytes);

        let mut chunk_dir_bytes = vec![0u8; chunk_dir_len as usize];
        reader.read_exact(&mut chunk_dir_bytes)?;
        let chunks: Vec<ChunkMeta> = serde_json::from_slice(&chunk_dir_bytes)?;

        // Drop the BufReader, re-open file for mmap
        drop(reader);
        let file = File::open(path.as_ref())?;
        // SAFETY: The file is opened read-only. We hold the Mmap for the lifetime of SpissaReader.
        // The file must not be modified externally while the reader is alive.
        let mmap = unsafe { Mmap::map(&file)? };

        // Residency hints. A plain mmap faults pages in lazily and lets the OS
        // evict them under pressure, so a decode loop that re-reads the whole
        // weight set every token keeps re-faulting from disk (memory-bound on
        // machines where the model nearly fits RAM). MADV_WILLNEED asks the
        // kernel to read the mapping ahead so the weights land resident up
        // front; it is a pure hint with no correctness impact, so we always
        // issue it and ignore failures (unsupported platforms / large maps).
        // `advise`/`Advice` are `#[cfg(unix)]` in memmap2; on other platforms
        // (e.g. Windows) the hint is simply skipped.
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::WillNeed);

        // Opt-in hard residency: mlock pins the whole mapping so the OS cannot
        // evict it. This guarantees the model stays in RAM (matching the
        // resident behaviour of llama.cpp's --mlock) at the risk of OOM when
        // the working set exceeds physical RAM, so it is gated behind an env
        // flag and failures are tolerated rather than fatal.
        // `Mmap::lock` is `#[cfg(unix)]`; on non-unix targets we report the
        // unsupported request and continue (the mapping still works, just
        // without hard residency).
        if std::env::var("SPISSA_MLOCK")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            #[cfg(unix)]
            match mmap.lock() {
                Ok(()) => {}
                Err(e) => eprintln!("[rllm] SPISSA_MLOCK=1 requested but mlock failed: {e}"),
            }
            #[cfg(not(unix))]
            eprintln!(
                "[rllm] SPISSA_MLOCK=1 requested but mlock is not supported on this platform; ignoring"
            );
        }

        Ok(Self {
            mmap,
            header,
            metadata,
            tensors,
            chunks,
        })
    }

    /// Get the file header
    pub fn header(&self) -> &SpissaHeader {
        &self.header
    }

    /// Get global metadata
    pub fn metadata(&self) -> &GlobalMetadata {
        &self.metadata
    }

    /// List all tensors
    pub fn list_tensors(&self) -> &[TensorMeta] {
        &self.tensors
    }

    /// Get tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// R160b: advise the kernel it can drop the mmap pages covering `[offset, offset+len)`
    /// (MADV_DONTNEED). Used after decode-once caching so the compressed bytes don't stay
    /// resident on top of the decoded cache. Only whole pages fully inside the range are
    /// dropped (16 KB-aligned, conservative for Apple Silicon); best-effort, errors ignored.
    pub fn release_range(&self, offset: usize, len: usize) {
        const PAGE: usize = 16384;
        let start = (offset + PAGE - 1) & !(PAGE - 1);
        let end = (offset + len) & !(PAGE - 1);
        if end > start {
            // memmap2 gates MADV_DONTNEED to Linux; call libc directly so it works on
            // Darwin too (read-only mmap, so dropping pages just re-faults on re-access).
            #[cfg(unix)]
            unsafe {
                libc::madvise(
                    self.mmap.as_ptr().add(start) as *mut libc::c_void,
                    end - start,
                    libc::MADV_DONTNEED,
                );
            }
        }
    }

    /// Read a chunk's compressed data as a zero-copy slice from the mmap.
    ///
    /// This is the primary high-performance path. No allocation, no memcpy.
    pub fn read_chunk_slice(&self, chunk_id: u64) -> Result<&[u8]> {
        let chunk = self
            .chunks
            .iter()
            .find(|c| c.chunk_id == chunk_id)
            .ok_or(ContainerError::ChunkNotFound(chunk_id))?;

        let offset = chunk.file_offset as usize;
        let len = chunk.compressed_size as usize;
        let end = offset
            .checked_add(len)
            .ok_or(ContainerError::InvalidRange {
                context: format!("chunk {chunk_id}"),
                offset: chunk.file_offset,
                len: chunk.compressed_size,
                size: self.mmap.len() as u64,
            })?;

        if end > self.mmap.len() {
            return Err(ContainerError::TruncatedFile {
                expected: end as u64,
                actual: self.mmap.len() as u64,
            });
        }

        Ok(&self.mmap[offset..end])
    }

    /// Read a chunk's compressed data (allocating copy, for backward compatibility).
    pub fn read_chunk(&self, chunk_id: u64) -> Result<Vec<u8>> {
        Ok(self.read_chunk_slice(chunk_id)?.to_vec())
    }

    /// The entire memory-mapped file as a zero-copy slice. Used for bulk
    /// operations that index many spans at once across threads (e.g. parallel
    /// integrity verification), where a single shared `&[u8]` is cheaper than a
    /// borrow-per-chunk through `read_span`.
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    /// Read an arbitrary file span as a zero-copy mmap slice. Used by the q8
    /// decode fast-path to view a whole tensor's contiguous raw chunks as one
    /// slice (skipping per-chunk dispatch).
    pub fn read_span(&self, offset: u64, len: u64) -> Result<&[u8]> {
        let offset = offset as usize;
        let end = offset
            .checked_add(len as usize)
            .ok_or(ContainerError::InvalidRange {
                context: "read_span".to_string(),
                offset: offset as u64,
                len,
                size: self.mmap.len() as u64,
            })?;
        if end > self.mmap.len() {
            return Err(ContainerError::TruncatedFile {
                expected: end as u64,
                actual: self.mmap.len() as u64,
            });
        }
        Ok(&self.mmap[offset..end])
    }

    /// Read a byte range from a chunk's compressed payload as a zero-copy slice.
    pub fn read_chunk_range_slice(
        &self,
        chunk_id: u64,
        byte_offset: u64,
        byte_len: u64,
    ) -> Result<&[u8]> {
        let chunk = self
            .chunks
            .iter()
            .find(|c| c.chunk_id == chunk_id)
            .ok_or(ContainerError::ChunkNotFound(chunk_id))?;
        let end =
            byte_offset
                .checked_add(byte_len)
                .ok_or_else(|| ContainerError::InvalidRange {
                    context: format!("chunk {chunk_id}"),
                    offset: byte_offset,
                    len: byte_len,
                    size: chunk.compressed_size,
                })?;
        if end > chunk.compressed_size {
            return Err(ContainerError::InvalidRange {
                context: format!("chunk {chunk_id}"),
                offset: byte_offset,
                len: byte_len,
                size: chunk.compressed_size,
            });
        }

        let abs_offset = (chunk.file_offset + byte_offset) as usize;
        let abs_end = (chunk.file_offset + end) as usize;

        if abs_end > self.mmap.len() {
            return Err(ContainerError::TruncatedFile {
                expected: abs_end as u64,
                actual: self.mmap.len() as u64,
            });
        }

        Ok(&self.mmap[abs_offset..abs_end])
    }

    /// Read a byte range from a chunk's compressed payload (allocating copy).
    pub fn read_chunk_range(
        &self,
        chunk_id: u64,
        byte_offset: u64,
        byte_len: u64,
    ) -> Result<Vec<u8>> {
        Ok(self
            .read_chunk_range_slice(chunk_id, byte_offset, byte_len)?
            .to_vec())
    }

    /// Get all chunks for a tensor
    pub fn get_tensor_chunks(&self, tensor_id: u64) -> Vec<&ChunkMeta> {
        self.chunks
            .iter()
            .filter(|c| c.tensor_id == tensor_id)
            .collect()
    }

    /// Get all chunks
    pub fn list_chunks(&self) -> &[ChunkMeta] {
        &self.chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::DType;
    use crate::writer::SpissaWriter;

    #[test]
    fn test_reader_writer_roundtrip() {
        let temp = std::env::temp_dir().join("test_roundtrip.spsa");

        // Write
        let metadata = GlobalMetadata::new_test();
        let mut writer = SpissaWriter::new(&temp, metadata).unwrap();

        let tensor_meta = TensorMeta {
            tensor_id: 0,
            name: "test.weight".to_string(),
            shape: vec![2, 2],
            dtype: DType::Fp32,
            original_size_bytes: 16,
            compressed_size_bytes: 16,
            original_sha256: [0u8; 32],
            chunk_count: 1,
            chunk_start_index: 0,
        };
        writer.add_tensor(tensor_meta);

        let data = vec![42u8; 16];
        writer
            .write_chunk(0, "rtc-raw-v1", &data, &data, 0)
            .unwrap();

        writer.finalize().unwrap();

        // Read
        let reader = SpissaReader::open(&temp).unwrap();
        assert_eq!(reader.header().version, crate::SPISSA_VERSION);
        assert_eq!(reader.metadata().model_name, "test-model");
        assert_eq!(reader.list_tensors().len(), 1);
        assert_eq!(reader.list_tensors()[0].name, "test.weight");
        assert_eq!(reader.list_chunks().len(), 1);

        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn test_reader_chunk_data() {
        let temp = std::env::temp_dir().join("test_chunk_data.spsa");

        // Write
        let metadata = GlobalMetadata::new_test();
        let mut writer = SpissaWriter::new(&temp, metadata).unwrap();

        let tensor_meta = TensorMeta {
            tensor_id: 0,
            name: "test".to_string(),
            shape: vec![4],
            dtype: DType::U8,
            original_size_bytes: 4,
            compressed_size_bytes: 4,
            original_sha256: [0u8; 32],
            chunk_count: 1,
            chunk_start_index: 0,
        };
        writer.add_tensor(tensor_meta);

        let data = vec![1, 2, 3, 4];
        writer
            .write_chunk(0, "rtc-raw-v1", &data, &data, 0)
            .unwrap();

        writer.finalize().unwrap();

        // Read and verify chunk data
        let reader = SpissaReader::open(&temp).unwrap();
        let chunk_data = reader.read_chunk(0).unwrap();
        assert_eq!(chunk_data, vec![1, 2, 3, 4]);

        // Also verify zero-copy slice
        let chunk_slice = reader.read_chunk_slice(0).unwrap();
        assert_eq!(chunk_slice, &[1, 2, 3, 4]);

        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn test_reader_chunk_range_data() {
        let temp = std::env::temp_dir().join("test_chunk_range_data.spsa");

        let metadata = GlobalMetadata::new_test();
        let mut writer = SpissaWriter::new(&temp, metadata).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "test".to_string(),
            shape: vec![8],
            dtype: DType::U8,
            original_size_bytes: 8,
            compressed_size_bytes: 8,
            original_sha256: [0u8; 32],
            chunk_count: 1,
            chunk_start_index: 0,
        });

        let data = vec![10, 11, 12, 13, 14, 15, 16, 17];
        writer
            .write_chunk(0, "rtc-raw-v1", &data, &data, 0)
            .unwrap();
        writer.finalize().unwrap();

        let reader = SpissaReader::open(&temp).unwrap();
        assert_eq!(
            reader.read_chunk_range(0, 2, 4).unwrap(),
            vec![12, 13, 14, 15]
        );
        assert!(matches!(
            reader.read_chunk_range(0, 6, 3),
            Err(ContainerError::InvalidRange { .. })
        ));

        std::fs::remove_file(&temp).ok();
    }
}
