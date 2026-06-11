//! RLLM file writer

use crate::error::Result;
use crate::header::RllmHeader;
use crate::metadata::{ChunkMeta, GlobalMetadata, TensorMeta};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// Writer for .rllm files
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

    /// Write a compressed chunk and return its ID
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
        let chunk_id = self.chunks.len() as u64;
        let file_offset = self.current_offset;

        // Compute checksums
        let original_hash: [u8; 32] = Sha256::digest(original_data).into();
        let compressed_hash: [u8; 32] = Sha256::digest(data).into();

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
        };

        // Write compressed data
        self.file.write_all(data)?;
        self.current_offset += data.len() as u64;

        self.chunks.push(chunk_meta);
        Ok(chunk_id)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::DType;
    use std::io::Read;

    #[test]
    fn test_writer_creates_file() {
        let temp = std::env::temp_dir().join("test_writer.rllm");
        let metadata = GlobalMetadata::new_test();
        let writer = RllmWriter::new(&temp, metadata).unwrap();
        writer.finalize().unwrap();

        assert!(temp.exists());
        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn test_writer_with_tensor() {
        let temp = std::env::temp_dir().join("test_writer_tensor.rllm");
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
}
