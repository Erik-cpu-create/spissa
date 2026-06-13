use crate::loader::{
    codec_for_id, verify_compressed_chunk_checksum, verify_original_chunk_checksum,
    verify_tensor_checksum,
};
use crate::{MemoryBudget, RamaTrace, RamaTraceEventInput, Result, RuntimeError, Tensor};
use rllm_container::{ChunkMeta, GlobalMetadata, RllmReader, TensorMeta};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamaIntegrityMode {
    /// Verify compressed and decoded chunk checksums every time a chunk is recalled.
    Strict,
    /// Verify each chunk checksum once per process, then trust the already-verified chunk.
    VerifyOnce,
}

impl Default for RamaIntegrityMode {
    fn default() -> Self {
        Self::Strict
    }
}

#[derive(Debug, Clone)]
pub struct LazyModelStats {
    pub file_size_bytes: u64,
    pub tensor_count: usize,
    pub chunk_count: usize,
    pub total_original_bytes: u64,
    pub total_compressed_chunk_bytes: u64,
    pub full_decode_runtime_bytes: usize,
}

pub struct LazyRllmModel {
    path: PathBuf,
    reader: RllmReader,
    metadata: GlobalMetadata,
    tensors_by_name: HashMap<String, TensorMeta>,
    tensors_by_id: HashMap<u64, TensorMeta>,
    chunks_by_tensor: HashMap<u64, Vec<ChunkMeta>>,
    chunks_by_id: HashMap<u64, ChunkMeta>,
    stats: LazyModelStats,
    rama_trace: Option<RamaTrace>,
    integrity_mode: RamaIntegrityMode,
    verified_compressed_chunks: HashSet<u64>,
    verified_original_chunks: HashSet<u64>,
}

impl LazyRllmModel {
    /// Open a `.rllm` file without decoding tensor payloads.
    ///
    /// This reads only the container header, global metadata, tensor directory,
    /// and chunk directory. It is the entry point for low-RAM execution modes.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let file_size_bytes = std::fs::metadata(&path_buf)?.len();
        let reader = RllmReader::open(&path_buf)?;
        let metadata = reader.metadata().clone();
        let tensor_metas: Vec<TensorMeta> = reader.list_tensors().to_vec();
        let chunk_metas: Vec<ChunkMeta> = reader.list_chunks().to_vec();

        let mut tensors_by_name = HashMap::with_capacity(tensor_metas.len());
        let mut tensors_by_id = HashMap::with_capacity(tensor_metas.len());
        let mut total_original_bytes = 0u64;
        let mut full_decode_runtime_bytes = 0usize;

        for tensor in tensor_metas {
            total_original_bytes = total_original_bytes.saturating_add(tensor.original_size_bytes);
            full_decode_runtime_bytes = full_decode_runtime_bytes
                .saturating_add(runtime_f32_bytes_for_tensor(&tensor).unwrap_or(usize::MAX / 4));
            tensors_by_name.insert(tensor.name.clone(), tensor.clone());
            tensors_by_id.insert(tensor.tensor_id, tensor);
        }

        let mut chunks_by_tensor: HashMap<u64, Vec<ChunkMeta>> = HashMap::new();
        let mut chunks_by_id = HashMap::with_capacity(chunk_metas.len());
        let mut total_compressed_chunk_bytes = 0u64;
        for chunk in chunk_metas {
            total_compressed_chunk_bytes =
                total_compressed_chunk_bytes.saturating_add(chunk.compressed_size);
            chunks_by_tensor
                .entry(chunk.tensor_id)
                .or_default()
                .push(chunk.clone());
            chunks_by_id.insert(chunk.chunk_id, chunk);
        }

        for chunks in chunks_by_tensor.values_mut() {
            chunks.sort_by_key(|chunk| chunk.chunk_offset_in_tensor);
        }

        let stats = LazyModelStats {
            file_size_bytes,
            tensor_count: tensors_by_name.len(),
            chunk_count: chunks_by_id.len(),
            total_original_bytes,
            total_compressed_chunk_bytes,
            full_decode_runtime_bytes,
        };

        Ok(Self {
            path: path_buf,
            reader,
            metadata,
            tensors_by_name,
            tensors_by_id,
            chunks_by_tensor,
            chunks_by_id,
            stats,
            rama_trace: None,
            integrity_mode: RamaIntegrityMode::Strict,
            verified_compressed_chunks: HashSet::new(),
            verified_original_chunks: HashSet::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn metadata(&self) -> &GlobalMetadata {
        &self.metadata
    }

    pub fn stats(&self) -> &LazyModelStats {
        &self.stats
    }

    pub fn enable_rama_trace(&mut self) {
        self.rama_trace = Some(RamaTrace::new(
            self.metadata.model_name.clone(),
            self.metadata.architecture.clone(),
        ));
    }

    pub fn take_rama_trace(&mut self) -> Option<RamaTrace> {
        self.rama_trace.take()
    }

    pub fn set_rama_integrity_mode(&mut self, mode: RamaIntegrityMode) {
        if self.integrity_mode != mode {
            self.verified_compressed_chunks.clear();
            self.verified_original_chunks.clear();
        }
        self.integrity_mode = mode;
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorMeta> {
        self.tensors_by_name
            .get(name)
            .ok_or_else(|| RuntimeError::MissingTensor(name.to_string()))
    }

    pub fn tensor_by_id(&self, tensor_id: u64) -> Result<&TensorMeta> {
        self.tensors_by_id
            .get(&tensor_id)
            .ok_or_else(|| RuntimeError::MissingTensor(format!("tensor_id={tensor_id}")))
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors_by_name.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    pub fn tensors(&self) -> impl Iterator<Item = &TensorMeta> {
        self.tensors_by_id.values()
    }

    pub fn chunks(&self) -> impl Iterator<Item = &ChunkMeta> {
        self.chunks_by_id.values()
    }

    pub fn chunks_for_tensor(&self, tensor_id: u64) -> &[ChunkMeta] {
        self.chunks_by_tensor
            .get(&tensor_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn record_rama_chunk_event(
        &mut self,
        phase: &str,
        label: String,
        chunk: &ChunkMeta,
        tensor_name: Option<&str>,
        start: Instant,
        duration: Duration,
        budget: &MemoryBudget,
    ) {
        if let Some(trace) = self.rama_trace.as_mut() {
            let start_ns = trace.elapsed_ns_since_start(start);
            trace.record(RamaTraceEventInput {
                phase: phase.to_string(),
                label,
                tensor_name: tensor_name.map(ToOwned::to_owned),
                tensor_id: Some(chunk.tensor_id),
                chunk_id: Some(chunk.chunk_id),
                codec_id: Some(chunk.codec_id.clone()),
                compressed_bytes: Some(chunk.compressed_size),
                decoded_bytes: Some(chunk.uncompressed_size),
                start_ns,
                duration_ns: crate::trace::saturating_duration_nanos(duration),
                budget_current_bytes: budget.current_bytes(),
                budget_peak_bytes: budget.peak_bytes(),
            });
        }
    }

    fn verify_compressed_chunk_if_needed(
        &mut self,
        chunk: &ChunkMeta,
        compressed: &[u8],
    ) -> Result<bool> {
        if self.integrity_mode == RamaIntegrityMode::VerifyOnce
            && self.verified_compressed_chunks.contains(&chunk.chunk_id)
        {
            return Ok(false);
        }
        verify_compressed_chunk_checksum(chunk, compressed)?;
        if self.integrity_mode == RamaIntegrityMode::VerifyOnce {
            self.verified_compressed_chunks.insert(chunk.chunk_id);
        }
        Ok(true)
    }

    fn verify_original_chunk_if_needed(
        &mut self,
        chunk: &ChunkMeta,
        decoded: &[u8],
    ) -> Result<bool> {
        if self.integrity_mode == RamaIntegrityMode::VerifyOnce
            && self.verified_original_chunks.contains(&chunk.chunk_id)
        {
            return Ok(false);
        }
        verify_original_chunk_checksum(chunk, decoded)?;
        if self.integrity_mode == RamaIntegrityMode::VerifyOnce {
            self.verified_original_chunks.insert(chunk.chunk_id);
        }
        Ok(true)
    }

    /// Decode one tensor under a caller-provided memory budget.
    ///
    /// This is still tensor-level materialization. Tile-stream code should use
    /// `with_decoded_chunk` to bound the decode window to one chunk/tile.
    pub fn decode_tensor(&mut self, name: &str, budget: &mut MemoryBudget) -> Result<Tensor> {
        let tensor_meta = self.tensor(name)?.clone();
        budget.reserve(
            tensor_meta.original_size_bytes as usize,
            format!("tensor raw bytes: {name}"),
        )?;
        let raw_bytes = match crate::loader::decode_tensor_bytes(&mut self.reader, &tensor_meta) {
            Ok(bytes) => bytes,
            Err(err) => {
                budget.release(
                    tensor_meta.original_size_bytes as usize,
                    format!("tensor raw bytes rollback: {name}"),
                )?;
                return Err(err);
            }
        };

        let runtime_bytes = runtime_f32_bytes_for_tensor(&tensor_meta)?;
        budget.reserve(runtime_bytes, format!("tensor f32 runtime: {name}"))?;
        let tensor = match Tensor::from_bytes(
            tensor_meta.name.clone(),
            tensor_meta.shape.clone(),
            tensor_meta.dtype,
            &raw_bytes,
        ) {
            Ok(tensor) => tensor,
            Err(err) => {
                budget.release(runtime_bytes, format!("tensor f32 rollback: {name}"))?;
                budget.release(
                    tensor_meta.original_size_bytes as usize,
                    format!("tensor raw rollback: {name}"),
                )?;
                return Err(err);
            }
        };
        budget.release(
            tensor_meta.original_size_bytes as usize,
            format!("tensor raw bytes consumed: {name}"),
        )?;
        Ok(tensor)
    }

    /// Decode a single compressed chunk, run a closure with the decoded bytes,
    /// and release both compressed and decoded buffers before returning.
    pub fn with_decoded_chunk<R>(
        &mut self,
        chunk_id: u64,
        budget: &mut MemoryBudget,
        f: impl FnOnce(&[u8], &mut MemoryBudget) -> Result<R>,
    ) -> Result<R> {
        let chunk = self
            .chunks_by_id
            .get(&chunk_id)
            .ok_or_else(|| RuntimeError::InvalidTensorData(format!("missing chunk {chunk_id}")))?
            .clone();
        let tensor_name = self
            .tensors_by_id
            .get(&chunk.tensor_id)
            .map(|tensor| tensor.name.clone());

        let compressed_label = format!("compressed chunk {}", chunk.chunk_id);
        budget.reserve(chunk.compressed_size as usize, compressed_label.clone())?;
        let read_start = Instant::now();
        let compressed = match self.reader.read_chunk(chunk.chunk_id) {
            Ok(bytes) => bytes,
            Err(err) => {
                budget.release(chunk.compressed_size as usize, compressed_label)?;
                return Err(err.into());
            }
        };
        self.record_rama_chunk_event(
            "chunk_read",
            format!("read chunk {}", chunk.chunk_id),
            &chunk,
            tensor_name.as_deref(),
            read_start,
            read_start.elapsed(),
            budget,
        );

        let compressed_checksum_start = Instant::now();
        let compressed_verified = match self.verify_compressed_chunk_if_needed(&chunk, &compressed)
        {
            Ok(verified) => verified,
            Err(err) => {
                budget.release(chunk.compressed_size as usize, compressed_label)?;
                return Err(err);
            }
        };
        if compressed_verified {
            self.record_rama_chunk_event(
                "chunk_compressed_checksum",
                format!("verify compressed chunk {}", chunk.chunk_id),
                &chunk,
                tensor_name.as_deref(),
                compressed_checksum_start,
                compressed_checksum_start.elapsed(),
                budget,
            );
        }

        let decoded_label = format!("decoded chunk {}", chunk.chunk_id);
        if let Err(err) = budget.reserve(chunk.uncompressed_size as usize, decoded_label.clone()) {
            budget.release(chunk.compressed_size as usize, compressed_label)?;
            return Err(err);
        }

        let codec = match codec_for_id(&chunk.codec_id) {
            Ok(codec) => codec,
            Err(err) => {
                budget.release(chunk.uncompressed_size as usize, decoded_label)?;
                budget.release(chunk.compressed_size as usize, compressed_label)?;
                return Err(err);
            }
        };
        let decode_start = Instant::now();
        let decoded = match codec.decode(
            &compressed,
            &rtc_codec::DecodeMeta {
                codec_id: chunk.codec_id.clone(),
                uncompressed_size: chunk.uncompressed_size,
            },
        ) {
            Ok(bytes) => bytes,
            Err(err) => {
                budget.release(chunk.uncompressed_size as usize, decoded_label)?;
                budget.release(chunk.compressed_size as usize, compressed_label)?;
                return Err(err.into());
            }
        };
        self.record_rama_chunk_event(
            "chunk_decode",
            format!("decode chunk {}", chunk.chunk_id),
            &chunk,
            tensor_name.as_deref(),
            decode_start,
            decode_start.elapsed(),
            budget,
        );
        budget.release(chunk.compressed_size as usize, compressed_label)?;

        if decoded.len() != chunk.uncompressed_size as usize {
            budget.release(chunk.uncompressed_size as usize, decoded_label)?;
            return Err(RuntimeError::InvalidTensorData(format!(
                "chunk {} decoded to {} bytes, expected {}",
                chunk.chunk_id,
                decoded.len(),
                chunk.uncompressed_size
            )));
        }
        let original_checksum_start = Instant::now();
        let original_verified = match self.verify_original_chunk_if_needed(&chunk, &decoded) {
            Ok(verified) => verified,
            Err(err) => {
                budget.release(chunk.uncompressed_size as usize, decoded_label)?;
                return Err(err);
            }
        };
        if original_verified {
            self.record_rama_chunk_event(
                "chunk_original_checksum",
                format!("verify original chunk {}", chunk.chunk_id),
                &chunk,
                tensor_name.as_deref(),
                original_checksum_start,
                original_checksum_start.elapsed(),
                budget,
            );
        }

        let compute_start = Instant::now();
        let result = f(&decoded, budget);
        self.record_rama_chunk_event(
            "chunk_compute_closure",
            format!("compute with decoded chunk {}", chunk.chunk_id),
            &chunk,
            tensor_name.as_deref(),
            compute_start,
            compute_start.elapsed(),
            budget,
        );
        budget.release(chunk.uncompressed_size as usize, decoded_label)?;
        result
    }

    /// Decode a byte range from a compressed chunk when the codec supports it,
    /// otherwise fall back to the full-chunk path.
    ///
    /// This is the Phase 7.8 foundation for tile-aligned decode. Memory accounting
    /// reserves only the requested decoded range for native range codecs. For
    /// codecs without native range support, it deliberately falls back to
    /// `with_decoded_chunk` so the budget still reflects full decoded materialization.
    pub fn with_decoded_chunk_range<R>(
        &mut self,
        chunk_id: u64,
        byte_offset: u64,
        byte_len: u64,
        budget: &mut MemoryBudget,
        f: impl FnOnce(&[u8], &mut MemoryBudget) -> Result<R>,
    ) -> Result<R> {
        let chunk = self
            .chunks_by_id
            .get(&chunk_id)
            .ok_or_else(|| RuntimeError::InvalidTensorData(format!("missing chunk {chunk_id}")))?
            .clone();
        let range = rtc_codec::DecodeRange::new(byte_offset, byte_len);
        range.validate(chunk.uncompressed_size)?;

        let codec = codec_for_id(&chunk.codec_id)?;
        if !codec.supports_native_range_decode() {
            return self.with_decoded_chunk(chunk_id, budget, |decoded, budget| {
                let start = usize::try_from(byte_offset).map_err(|_| {
                    RuntimeError::InvalidTensorData(format!(
                        "chunk {chunk_id} range offset {byte_offset} overflows usize"
                    ))
                })?;
                let end = usize::try_from(range.end()?).map_err(|_| {
                    RuntimeError::InvalidTensorData(format!(
                        "chunk {chunk_id} range end overflows usize"
                    ))
                })?;
                f(&decoded[start..end], budget)
            });
        }

        let compressed_label = format!("compressed chunk {}", chunk.chunk_id);
        budget.reserve(chunk.compressed_size as usize, compressed_label.clone())?;
        let compressed = match self.reader.read_chunk(chunk.chunk_id) {
            Ok(bytes) => bytes,
            Err(err) => {
                budget.release(chunk.compressed_size as usize, compressed_label)?;
                return Err(err.into());
            }
        };

        if let Err(err) = self.verify_compressed_chunk_if_needed(&chunk, &compressed) {
            budget.release(chunk.compressed_size as usize, compressed_label)?;
            return Err(err);
        }

        let decoded_range_bytes = usize::try_from(byte_len).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} range len {} overflows usize",
                chunk.chunk_id, byte_len
            ))
        })?;
        let decoded_label = format!(
            "decoded chunk {} byte range [{}..{})",
            chunk.chunk_id,
            byte_offset,
            range.end()?
        );
        if let Err(err) = budget.reserve(decoded_range_bytes, decoded_label.clone()) {
            budget.release(chunk.compressed_size as usize, compressed_label)?;
            return Err(err);
        }

        let decoded = match codec.decode_range(
            &compressed,
            &rtc_codec::DecodeMeta {
                codec_id: chunk.codec_id.clone(),
                uncompressed_size: chunk.uncompressed_size,
            },
            range,
        ) {
            Ok(bytes) => bytes,
            Err(err) => {
                budget.release(decoded_range_bytes, decoded_label)?;
                budget.release(chunk.compressed_size as usize, compressed_label)?;
                return Err(err.into());
            }
        };
        budget.release(chunk.compressed_size as usize, compressed_label)?;

        if decoded.len() != decoded_range_bytes {
            budget.release(decoded_range_bytes, decoded_label)?;
            return Err(RuntimeError::InvalidTensorData(format!(
                "chunk {} range decoded to {} bytes, expected {}",
                chunk.chunk_id,
                decoded.len(),
                decoded_range_bytes
            )));
        }

        let result = f(&decoded, budget);
        budget.release(decoded_range_bytes, decoded_label)?;
        result
    }

    /// Decode one full tensor, verify it, and release raw bytes before returning.
    /// Intended for tests and transitional runtime paths.
    pub fn verify_tensor_roundtrip(&mut self, name: &str, budget: &mut MemoryBudget) -> Result<()> {
        let tensor_meta = self.tensor(name)?.clone();
        budget.reserve(
            tensor_meta.original_size_bytes as usize,
            format!("roundtrip tensor raw: {name}"),
        )?;
        let raw_bytes = crate::loader::decode_tensor_bytes(&mut self.reader, &tensor_meta)?;
        verify_tensor_checksum(&tensor_meta, &raw_bytes)?;
        budget.release(
            tensor_meta.original_size_bytes as usize,
            format!("roundtrip tensor raw release: {name}"),
        )?;
        Ok(())
    }
}

pub(crate) fn runtime_f32_bytes_for_tensor(tensor: &TensorMeta) -> Result<usize> {
    let dtype_size = tensor.dtype.size_bytes() as u64;
    if dtype_size == 0 || tensor.original_size_bytes % dtype_size != 0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "tensor {} has original_size_bytes={} not divisible by dtype size {}",
            tensor.name, tensor.original_size_bytes, dtype_size
        )));
    }
    let elements = tensor.original_size_bytes / dtype_size;
    elements
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| {
            RuntimeError::InvalidTensorData(format!(
                "tensor {} runtime f32 size overflows usize",
                tensor.name
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rllm_container::{DType, RllmWriter};
    use sha2::{Digest, Sha256};

    fn sha256_array(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rllm-lazy-{name}-{}.rllm", std::process::id()))
    }

    #[test]
    fn open_reads_metadata_without_decoding_payloads() {
        let path = temp_path("metadata");
        let data = vec![1u8, 2, 3, 4];
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 7,
            name: "tiny.weight".to_string(),
            shape: vec![4],
            dtype: DType::U8,
            original_size_bytes: data.len() as u64,
            compressed_size_bytes: data.len() as u64,
            original_sha256: sha256_array(&data),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(7, "rtc-raw-v1", &data, &data, 0)
            .unwrap();
        writer.finalize().unwrap();

        let model = LazyRllmModel::open(&path).unwrap();
        assert_eq!(model.stats().tensor_count, 1);
        assert_eq!(model.stats().chunk_count, 1);
        assert_eq!(model.tensor("tiny.weight").unwrap().tensor_id, 7);
        assert_eq!(model.chunks_for_tensor(7).len(), 1);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn with_decoded_chunk_releases_budget_after_compute() {
        let path = temp_path("chunk");
        let data = vec![9u8; 16];
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "tiny.weight".to_string(),
            shape: vec![16],
            dtype: DType::U8,
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

        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(64);
        let sum = model
            .with_decoded_chunk(0, &mut budget, |bytes, _budget| {
                Ok(bytes.iter().map(|b| *b as u32).sum::<u32>())
            })
            .unwrap();

        assert_eq!(sum, 9 * 16);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(budget.peak_bytes(), 32);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn with_decoded_chunk_records_rama_trace_events() {
        let path = temp_path("chunk-trace");
        let data: Vec<u8> = (0..16).collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "tiny.weight".to_string(),
            shape: vec![16],
            dtype: DType::U8,
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

        let mut model = LazyRllmModel::open(&path).unwrap();
        model.enable_rama_trace();
        let mut budget = MemoryBudget::new(64);
        model
            .with_decoded_chunk(0, &mut budget, |bytes, _budget| Ok(bytes.len()))
            .unwrap();
        let trace = model.take_rama_trace().expect("trace should be enabled");
        let phases: Vec<&str> = trace
            .events
            .iter()
            .map(|event| event.phase.as_str())
            .collect();

        assert_eq!(
            phases,
            vec![
                "chunk_read",
                "chunk_compressed_checksum",
                "chunk_decode",
                "chunk_original_checksum",
                "chunk_compute_closure",
            ]
        );
        assert!(trace
            .events
            .iter()
            .all(|event| event.tensor_name.as_deref() == Some("tiny.weight")));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn verify_once_integrity_records_checksums_only_on_first_chunk_access() {
        let path = temp_path("chunk-verify-once");
        let data: Vec<u8> = (0..16).collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "tiny.weight".to_string(),
            shape: vec![16],
            dtype: DType::U8,
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

        let mut model = LazyRllmModel::open(&path).unwrap();
        model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
        model.enable_rama_trace();
        let mut budget = MemoryBudget::new(64);
        for _ in 0..2 {
            model
                .with_decoded_chunk(0, &mut budget, |bytes, _budget| Ok(bytes.len()))
                .unwrap();
        }
        let trace = model.take_rama_trace().expect("trace should be enabled");
        let compressed_checksum_events = trace
            .events
            .iter()
            .filter(|event| event.phase == "chunk_compressed_checksum")
            .count();
        let original_checksum_events = trace
            .events
            .iter()
            .filter(|event| event.phase == "chunk_original_checksum")
            .count();
        let read_events = trace
            .events
            .iter()
            .filter(|event| event.phase == "chunk_read")
            .count();

        assert_eq!(compressed_checksum_events, 1);
        assert_eq!(original_checksum_events, 1);
        assert_eq!(read_events, 2);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn with_decoded_chunk_range_uses_native_raw_range_budget() {
        let path = temp_path("chunk-range-raw");
        let data: Vec<u8> = (0..16).collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "tiny.weight".to_string(),
            shape: vec![16],
            dtype: DType::U8,
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

        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(64);
        let range = model
            .with_decoded_chunk_range(0, 4, 6, &mut budget, |bytes, _budget| Ok(bytes.to_vec()))
            .unwrap();

        assert_eq!(range, vec![4, 5, 6, 7, 8, 9]);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(budget.peak_bytes(), 22);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn with_decoded_chunk_range_falls_back_to_full_decode_budget_for_rle() {
        let path = temp_path("chunk-range-rle");
        let data = vec![7u8; 16];
        let compressed = vec![16u8, 7u8];
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "tiny.weight".to_string(),
            shape: vec![16],
            dtype: DType::U8,
            original_size_bytes: data.len() as u64,
            compressed_size_bytes: compressed.len() as u64,
            original_sha256: sha256_array(&data),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(0, "rtc-rle-v1", &compressed, &data, 0)
            .unwrap();
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(64);
        let range = model
            .with_decoded_chunk_range(0, 4, 6, &mut budget, |bytes, _budget| Ok(bytes.to_vec()))
            .unwrap();

        assert_eq!(range, vec![7u8; 6]);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(budget.peak_bytes(), 18);

        std::fs::remove_file(&path).ok();
    }
}
