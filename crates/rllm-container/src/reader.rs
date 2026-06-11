//! RLLM file reader

use crate::error::{ContainerError, Result};
use crate::header::RllmHeader;
use crate::metadata::{ChunkMeta, GlobalMetadata, TensorMeta};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Reader for .rllm files
pub struct RllmReader {
    file: BufReader<File>,
    header: RllmHeader,
    metadata: GlobalMetadata,
    tensors: Vec<TensorMeta>,
    chunks: Vec<ChunkMeta>,
}

impl RllmReader {
    /// Open and parse a .rllm file
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header_bytes = [0u8; RllmHeader::SIZE];
        reader.read_exact(&mut header_bytes)?;
        let header = RllmHeader::from_bytes(&header_bytes);
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

        Ok(Self {
            file: reader,
            header,
            metadata,
            tensors,
            chunks,
        })
    }

    /// Get the file header
    pub fn header(&self) -> &RllmHeader {
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

    /// Read a chunk's compressed data
    pub fn read_chunk(&mut self, chunk_id: u64) -> Result<Vec<u8>> {
        let chunk = self
            .chunks
            .iter()
            .find(|c| c.chunk_id == chunk_id)
            .ok_or(ContainerError::ChunkNotFound(chunk_id))?;

        self.file.seek(SeekFrom::Start(chunk.file_offset))?;
        let mut data = vec![0u8; chunk.compressed_size as usize];
        self.file.read_exact(&mut data)?;

        Ok(data)
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
    use crate::writer::RllmWriter;

    #[test]
    fn test_reader_writer_roundtrip() {
        let temp = std::env::temp_dir().join("test_roundtrip.rllm");

        // Write
        let metadata = GlobalMetadata::new_test();
        let mut writer = RllmWriter::new(&temp, metadata).unwrap();

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
        let reader = RllmReader::open(&temp).unwrap();
        assert_eq!(reader.header().version, crate::RLLM_VERSION);
        assert_eq!(reader.metadata().model_name, "test-model");
        assert_eq!(reader.list_tensors().len(), 1);
        assert_eq!(reader.list_tensors()[0].name, "test.weight");
        assert_eq!(reader.list_chunks().len(), 1);

        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn test_reader_chunk_data() {
        let temp = std::env::temp_dir().join("test_chunk_data.rllm");

        // Write
        let metadata = GlobalMetadata::new_test();
        let mut writer = RllmWriter::new(&temp, metadata).unwrap();

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
        let mut reader = RllmReader::open(&temp).unwrap();
        let chunk_data = reader.read_chunk(0).unwrap();
        assert_eq!(chunk_data, vec![1, 2, 3, 4]);

        std::fs::remove_file(&temp).ok();
    }
}
