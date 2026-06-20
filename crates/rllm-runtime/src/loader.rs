use crate::{Result, RuntimeError, Tensor};
use rllm_container::{ChunkMeta, ChunkRangeMeta, GlobalMetadata, RllmReader, TensorMeta};
use rtc_codec::{BitplaneCodec, DecodeMeta, HuffmanCodec, RansCodec, RawCodec, RleCodec, TensorCodec};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FullDecodeStats {
    pub tensor_count: usize,
    pub total_original_bytes: u64,
    pub total_runtime_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct FullDecodeModel {
    pub metadata: GlobalMetadata,
    pub tensors: HashMap<String, Tensor>,
    pub stats: FullDecodeStats,
}

impl FullDecodeModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let reader = RllmReader::open(path)?;
        let metadata = reader.metadata().clone();
        let tensor_metas: Vec<TensorMeta> = reader.list_tensors().to_vec();

        let mut tensors = HashMap::with_capacity(tensor_metas.len());
        let mut total_original_bytes = 0u64;
        let mut total_runtime_bytes = 0usize;

        for tensor_meta in tensor_metas {
            let tensor_bytes = decode_tensor_bytes(&reader, &tensor_meta)?;
            verify_tensor_checksum(&tensor_meta, &tensor_bytes)?;

            let tensor = Tensor::from_bytes(
                tensor_meta.name.clone(),
                tensor_meta.shape.clone(),
                tensor_meta.dtype,
                &tensor_bytes,
            )?;
            total_original_bytes += tensor_meta.original_size_bytes;
            total_runtime_bytes += tensor.runtime_size_bytes();
            tensors.insert(tensor_meta.name, tensor);
        }

        Ok(Self {
            metadata,
            stats: FullDecodeStats {
                tensor_count: tensors.len(),
                total_original_bytes,
                total_runtime_bytes,
            },
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| RuntimeError::MissingTensor(name.to_string()))
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }
}

pub(crate) fn decode_tensor_bytes(
    reader: &RllmReader,
    tensor_meta: &TensorMeta,
) -> Result<Vec<u8>> {
    let mut chunks: Vec<ChunkMeta> = reader
        .get_tensor_chunks(tensor_meta.tensor_id)
        .into_iter()
        .cloned()
        .collect();
    chunks.sort_by_key(|chunk| chunk.chunk_offset_in_tensor);

    let mut tensor_bytes = Vec::with_capacity(tensor_meta.original_size_bytes as usize);

    for chunk in chunks {
        let compressed = reader.read_chunk_slice(chunk.chunk_id)?;
        verify_compressed_chunk_checksum(&chunk, compressed)?;

        let codec = codec_for_id(&chunk.codec_id)?;
        let decoded = codec.decode(
            compressed,
            &DecodeMeta {
                codec_id: chunk.codec_id.clone(),
                uncompressed_size: chunk.uncompressed_size,
            },
        )?;

        if decoded.len() != chunk.uncompressed_size as usize {
            return Err(RuntimeError::InvalidTensorData(format!(
                "chunk {} decoded to {} bytes, expected {}",
                chunk.chunk_id,
                decoded.len(),
                chunk.uncompressed_size
            )));
        }
        verify_original_chunk_checksum(&chunk, &decoded)?;
        tensor_bytes.extend_from_slice(&decoded);
    }

    if tensor_bytes.len() != tensor_meta.original_size_bytes as usize {
        return Err(RuntimeError::InvalidTensorData(format!(
            "tensor {} decoded to {} bytes, expected {}",
            tensor_meta.name,
            tensor_bytes.len(),
            tensor_meta.original_size_bytes
        )));
    }

    Ok(tensor_bytes)
}

pub(crate) fn codec_for_id(codec_id: &str) -> Result<Box<dyn TensorCodec>> {
    match codec_id {
        "rtc-raw-v1" => Ok(Box::new(RawCodec)),
        "rtc-rle-v1" => Ok(Box::new(RleCodec)),
        "rtc-huff-v1" => Ok(Box::new(HuffmanCodec)),
        "rtc-rans-v1" => Ok(Box::new(RansCodec)),
        "rtc-bitplane-v1" => Ok(Box::new(BitplaneCodec)),
        _ => Err(RuntimeError::UnknownCodec(codec_id.to_string())),
    }
}

pub(crate) fn verify_compressed_chunk_checksum(chunk: &ChunkMeta, compressed: &[u8]) -> Result<()> {
    let computed = Sha256::digest(compressed);
    if computed.as_slice() != chunk.chunk_sha256_compressed {
        return Err(RuntimeError::ChecksumMismatch(format!(
            "compressed chunk {} checksum mismatch",
            chunk.chunk_id
        )));
    }
    Ok(())
}

pub(crate) fn verify_original_chunk_checksum(chunk: &ChunkMeta, decoded: &[u8]) -> Result<()> {
    let computed = Sha256::digest(decoded);
    if computed.as_slice() != chunk.chunk_sha256_original {
        return Err(RuntimeError::ChecksumMismatch(format!(
            "decoded chunk {} checksum mismatch",
            chunk.chunk_id
        )));
    }
    Ok(())
}

#[allow(dead_code)]
// Phase 7.8C foundation: used by tests now, production routing starts after
// pack-time tile/block alignment gives every partial read an integrity record.
pub(crate) fn chunk_range_for_original_bytes(
    chunk: &ChunkMeta,
    byte_offset: u64,
    byte_len: u64,
) -> Result<&ChunkRangeMeta> {
    let end = byte_offset.checked_add(byte_len).ok_or_else(|| {
        RuntimeError::InvalidTensorData(format!(
            "chunk {} range [{byte_offset}, +{byte_len}) overflows",
            chunk.chunk_id
        ))
    })?;
    chunk
        .range_checksums
        .iter()
        .find(|range| range.original_offset == byte_offset && range.original_size == byte_len)
        .ok_or_else(|| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} has no checksum metadata for original byte range [{byte_offset}, {end})",
                chunk.chunk_id
            ))
        })
}

#[allow(dead_code)]
// Phase 7.8C foundation: used by tests now, production routing starts after
// pack-time tile/block alignment gives every partial read an integrity record.
pub(crate) fn verify_original_chunk_range_checksum(
    range: &ChunkRangeMeta,
    decoded_range: &[u8],
) -> Result<()> {
    if decoded_range.len() as u64 != range.original_size {
        return Err(RuntimeError::InvalidTensorData(format!(
            "chunk range {} decoded len {} does not match metadata {}",
            range.range_id,
            decoded_range.len(),
            range.original_size
        )));
    }
    let computed = Sha256::digest(decoded_range);
    if computed.as_slice() != range.sha256_original {
        return Err(RuntimeError::ChecksumMismatch(format!(
            "decoded chunk range {} checksum mismatch",
            range.range_id
        )));
    }
    Ok(())
}

#[allow(dead_code)]
// Phase 7.8C foundation: used by tests now, production routing starts after
// pack-time tile/block alignment gives every partial read an integrity record.
pub(crate) fn verify_compressed_chunk_range_checksum(
    range: &ChunkRangeMeta,
    compressed_range: &[u8],
) -> Result<()> {
    if compressed_range.len() as u64 != range.compressed_size {
        return Err(RuntimeError::InvalidTensorData(format!(
            "chunk range {} compressed len {} does not match metadata {}",
            range.range_id,
            compressed_range.len(),
            range.compressed_size
        )));
    }
    let computed = Sha256::digest(compressed_range);
    if computed.as_slice() != range.sha256_compressed {
        return Err(RuntimeError::ChecksumMismatch(format!(
            "compressed chunk range {} checksum mismatch",
            range.range_id
        )));
    }
    Ok(())
}

pub(crate) fn verify_tensor_checksum(tensor_meta: &TensorMeta, decoded: &[u8]) -> Result<()> {
    let computed = Sha256::digest(decoded);
    if computed.as_slice() != tensor_meta.original_sha256 {
        return Err(RuntimeError::ChecksumMismatch(format!(
            "tensor {} checksum mismatch",
            tensor_meta.name
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rllm_container::{DType, RllmWriter};
    use rtc_codec::{EncodeMeta, RleCodec};

    fn sha256_array(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rllm-runtime-{name}-{}.rllm", std::process::id()))
    }

    #[test]
    fn full_decode_loads_raw_f32_tensor() {
        let path = temp_path("raw");
        let data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "weight".to_string(),
            shape: vec![2, 2],
            dtype: DType::Fp32,
            original_size_bytes: data.len() as u64,
            compressed_size_bytes: data.len() as u64,
            original_sha256: sha256_array(&data),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(0, "rtc-raw-v1", &data, &data, 0)
            .unwrap();
        writer.finalize().unwrap();

        let model = FullDecodeModel::load(&path).unwrap();
        let tensor = model.get("weight").unwrap();
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(model.stats.tensor_count, 1);
        assert_eq!(model.stats.total_original_bytes, data.len() as u64);
        assert_eq!(
            model.stats.total_runtime_bytes,
            4 * std::mem::size_of::<f32>()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn full_decode_loads_rle_u8_tensor() {
        let path = temp_path("rle");
        let data = vec![7u8; 128];
        let codec = RleCodec;
        let encoded = codec
            .encode(
                &data,
                &EncodeMeta {
                    name: "ids".to_string(),
                    shape: vec![data.len() as u64],
                    dtype: "u8".to_string(),
                },
            )
            .unwrap();

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "ids".to_string(),
            shape: vec![128],
            dtype: DType::U8,
            original_size_bytes: data.len() as u64,
            compressed_size_bytes: encoded.data.len() as u64,
            original_sha256: sha256_array(&data),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(0, encoded.codec_id.as_str(), &encoded.data, &data, 0)
            .unwrap();
        writer.finalize().unwrap();

        let model = FullDecodeModel::load(&path).unwrap();
        let tensor = model.get("ids").unwrap();
        assert_eq!(tensor.data.len(), 128);
        assert!(tensor.data.iter().all(|&v| v == 7.0));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn verifies_chunk_range_checksum_metadata() {
        let path = temp_path("range-checksums");
        let data: Vec<u8> = (0..12).collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "ids".to_string(),
            shape: vec![12],
            dtype: DType::U8,
            original_size_bytes: data.len() as u64,
            compressed_size_bytes: data.len() as u64,
            original_sha256: sha256_array(&data),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk_with_identity_range_checksums(0, "rtc-raw-v1", &data, &data, 0, 4)
            .unwrap();
        writer.finalize().unwrap();

        let reader = RllmReader::open(&path).unwrap();
        let chunk = reader.list_chunks()[0].clone();
        let range = chunk_range_for_original_bytes(&chunk, 4, 4).unwrap();
        let compressed_range = reader
            .read_chunk_range(
                chunk.chunk_id,
                range.compressed_offset,
                range.compressed_size,
            )
            .unwrap();

        verify_compressed_chunk_range_checksum(range, &compressed_range).unwrap();
        verify_original_chunk_range_checksum(range, &data[4..8]).unwrap();
        assert!(chunk_range_for_original_bytes(&chunk, 5, 4).is_err());

        let mut corrupted = compressed_range.clone();
        corrupted[0] ^= 0xFF;
        assert!(matches!(
            verify_compressed_chunk_range_checksum(range, &corrupted),
            Err(RuntimeError::ChecksumMismatch(_))
        ));

        std::fs::remove_file(&path).ok();
    }
}
