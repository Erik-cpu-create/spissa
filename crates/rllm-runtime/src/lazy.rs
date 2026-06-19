use crate::loader::{
    chunk_range_for_original_bytes, codec_for_id, verify_compressed_chunk_checksum,
    verify_compressed_chunk_range_checksum, verify_original_chunk_checksum, verify_tensor_checksum,
};
use crate::{MemoryBudget, RamaTrace, RamaTraceEventInput, Result, RuntimeError, Tensor};
use rllm_container::{ChunkMeta, GlobalMetadata, RllmReader, TensorMeta};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RamaIntegrityMode {
    /// Verify compressed and decoded chunk checksums every time a chunk is recalled.
    #[default]
    Strict,
    /// Verify each chunk checksum once per process, then trust the already-verified chunk.
    VerifyOnce,
    /// Trust local artifact bytes without runtime checksum verification.
    Unchecked,
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
    verified_compressed_chunk_ranges: HashSet<(u64, u64, u64)>,
    verified_original_chunks: HashSet<u64>,
    verified_tensors: HashSet<u64>,
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
            verified_compressed_chunk_ranges: HashSet::new(),
            verified_original_chunks: HashSet::new(),
            verified_tensors: HashSet::new(),
        })
    }

    /// Expose a whole tensor's raw bytes as ONE contiguous zero-copy mmap slice,
    /// for the q8 decode fast-path that bypasses per-chunk dispatch.
    ///
    /// Returns `Ok(None)` (caller falls back to the chunk path) unless the tensor
    /// is stored as a contiguous run of identity-codec (`rtc-raw-v1`) chunks. The
    /// slice is integrity-checked once against the tensor checksum (honoring the
    /// integrity mode); a mismatch also yields `None` so the verified chunk path
    /// takes over rather than failing the call.
    pub fn with_raw_tensor<R>(
        &mut self,
        tensor_id: u64,
        f: impl FnOnce(&[u8]) -> Result<R>,
    ) -> Result<Option<R>> {
        let (first, total, all_chunks_verified) = {
            let chunks = match self.chunks_by_tensor.get(&tensor_id) {
                Some(chunks) if !chunks.is_empty() => chunks,
                _ => return Ok(None),
            };
            let first = chunks[0].file_offset;
            let mut cursor = first;
            // Bridge to the chunk-level integrity record: this whole-tensor view
            // is exactly the tensor's rtc-raw-v1 chunks concatenated, and for
            // that codec the raw bytes ARE the original bytes. So if every chunk
            // was already SHA-verified via the chunk path (e.g. during prefill),
            // the tensor's bytes are proven intact and re-hashing the whole
            // tensor here is pure redundant work — the cost that showed up as a
            // ~6s "warmup" on the first decode token.
            let mut all_chunks_verified = true;
            for chunk in chunks {
                if chunk.codec_id != "rtc-raw-v1"
                    || chunk.compressed_size != chunk.uncompressed_size
                    || chunk.file_offset != cursor
                {
                    return Ok(None);
                }
                if !self.verified_compressed_chunks.contains(&chunk.chunk_id) {
                    all_chunks_verified = false;
                }
                cursor = cursor.saturating_add(chunk.compressed_size);
            }
            (first, cursor - first, all_chunks_verified)
        };

        // In VerifyOnce, skip the redundant whole-tensor hash when the chunks are
        // already verified. Strict still re-verifies every call (never skips);
        // Unchecked never verifies.
        let bridge_skip =
            self.integrity_mode == RamaIntegrityMode::VerifyOnce && all_chunks_verified;
        let verify = self.integrity_mode != RamaIntegrityMode::Unchecked
            && !self.verified_tensors.contains(&tensor_id)
            && !bridge_skip;

        let result = {
            let slice = self.reader.read_span(first, total)?;
            if verify {
                let tensor_meta = match self.tensors_by_id.get(&tensor_id) {
                    Some(meta) => meta,
                    None => return Ok(None),
                };
                if verify_tensor_checksum(tensor_meta, slice).is_err() {
                    return Ok(None);
                }
            }
            f(slice)?
        };
        if self.integrity_mode == RamaIntegrityMode::VerifyOnce {
            self.verified_tensors.insert(tensor_id);
        }
        Ok(Some(result))
    }

    /// Verify every not-yet-verified chunk's SHA-256 up front, in parallel.
    ///
    /// In VerifyOnce, the per-chunk integrity pass would otherwise run serially
    /// inline during the first prefill (a multi-second stall for a multi-GB
    /// model), and the decode fast-path's whole-tensor hash is skipped once the
    /// chunks are verified (see the bridge in `with_raw_tensor`). SHA over
    /// independent chunks is embarrassingly parallel, so front-loading it across
    /// cores turns that serial stall into a brief startup cost. Returns the
    /// number of chunks verified. No-op for Strict (which must re-verify on
    /// every access) and Unchecked (which never verifies).
    pub fn prewarm_chunk_integrity(&mut self) -> Result<usize> {
        if self.integrity_mode != RamaIntegrityMode::VerifyOnce {
            return Ok(0);
        }
        let pending: Vec<ChunkMeta> = self
            .chunks_by_id
            .values()
            .filter(|chunk| !self.verified_compressed_chunks.contains(&chunk.chunk_id))
            .cloned()
            .collect();
        if pending.is_empty() {
            return Ok(0);
        }

        let threads = prewarm_thread_count().min(pending.len()).max(1);
        let shard_len = pending.len().div_ceil(threads);
        let bytes: &[u8] = self.reader.as_slice();

        // Each thread verifies a disjoint shard of chunks against the shared,
        // read-only mmap and returns the ids it confirmed. Failures propagate
        // (a corrupt chunk fails the whole prewarm, same as inline verification).
        let verified: Vec<u64> = std::thread::scope(|scope| -> Result<Vec<u64>> {
            let mut handles = Vec::new();
            for shard in pending.chunks(shard_len) {
                handles.push(scope.spawn(move || -> Result<Vec<u64>> {
                    let mut ok = Vec::with_capacity(shard.len());
                    for chunk in shard {
                        let start = chunk.file_offset as usize;
                        let end = start.checked_add(chunk.compressed_size as usize).ok_or_else(
                            || {
                                RuntimeError::InvalidTensorData(format!(
                                    "chunk {} span overflow",
                                    chunk.chunk_id
                                ))
                            },
                        )?;
                        let slice = bytes.get(start..end).ok_or_else(|| {
                            RuntimeError::InvalidTensorData(format!(
                                "chunk {} out of file bounds",
                                chunk.chunk_id
                            ))
                        })?;
                        verify_compressed_chunk_checksum(chunk, slice)?;
                        ok.push(chunk.chunk_id);
                    }
                    Ok(ok)
                }));
            }
            let mut all = Vec::with_capacity(pending.len());
            for handle in handles {
                let ids = handle.join().map_err(|_| {
                    RuntimeError::InvalidTensorData("integrity prewarm thread panicked".to_string())
                })??;
                all.extend(ids);
            }
            Ok(all)
        })?;

        let count = verified.len();
        for id in verified {
            self.verified_compressed_chunks.insert(id);
        }
        Ok(count)
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
            self.verified_compressed_chunk_ranges.clear();
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
        if self.integrity_mode == RamaIntegrityMode::Unchecked {
            return Ok(false);
        }
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

    fn verify_compressed_chunk_range_if_needed(
        &mut self,
        chunk: &ChunkMeta,
        byte_offset: u64,
        byte_len: u64,
        compressed_range: &[u8],
    ) -> Result<bool> {
        if self.integrity_mode == RamaIntegrityMode::Unchecked {
            return Ok(false);
        }
        let range = chunk_range_for_original_bytes(chunk, byte_offset, byte_len)?;
        let key = (
            chunk.chunk_id,
            range.compressed_offset,
            range.compressed_size,
        );
        if self.integrity_mode == RamaIntegrityMode::VerifyOnce
            && self.verified_compressed_chunk_ranges.contains(&key)
        {
            return Ok(false);
        }
        verify_compressed_chunk_range_checksum(range, compressed_range)?;
        if self.integrity_mode == RamaIntegrityMode::VerifyOnce {
            self.verified_compressed_chunk_ranges.insert(key);
        }
        Ok(true)
    }

    fn verify_original_chunk_if_needed(
        &mut self,
        chunk: &ChunkMeta,
        decoded: &[u8],
    ) -> Result<bool> {
        if self.integrity_mode == RamaIntegrityMode::Unchecked {
            return Ok(false);
        }
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
        let raw_bytes = match crate::loader::decode_tensor_bytes(&self.reader, &tensor_meta) {
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

    /// Expose the raw compressed chunk bytes directly without decoding.
    /// Used for specialized kernels like Fused Decode-MatMul for specific codecs.
    pub fn with_raw_chunk<R>(
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

        let is_bounded = budget.limit_bytes() != usize::MAX;
        let compressed_label = if is_bounded {
            format!("compressed chunk {}", chunk.chunk_id)
        } else {
            String::new()
        };

        if is_bounded {
            budget.reserve(chunk.compressed_size as usize, compressed_label.clone())?;
        }

        let read_start = Instant::now();
        let compressed = match self.reader.read_chunk_slice(chunk.chunk_id) {
            Ok(bytes) => unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) },
            Err(err) => {
                if is_bounded {
                    budget.release(chunk.compressed_size as usize, compressed_label)?;
                }
                return Err(err.into());
            }
        };

        if self.rama_trace.is_some() {
            let tensor_name = self
                .tensors_by_id
                .get(&chunk.tensor_id)
                .map(|t| t.name.clone());
            self.record_rama_chunk_event(
                "chunk_read",
                format!("read chunk {}", chunk.chunk_id),
                &chunk,
                tensor_name.as_deref(),
                read_start,
                read_start.elapsed(),
                budget,
            );
        }

        let compressed_checksum_start = Instant::now();
        let compressed_verified = match self.verify_compressed_chunk_if_needed(&chunk, compressed) {
            Ok(verified) => verified,
            Err(err) => {
                if is_bounded {
                    budget.release(chunk.compressed_size as usize, compressed_label)?;
                }
                return Err(err);
            }
        };

        if compressed_verified && self.rama_trace.is_some() {
            let tensor_name = self
                .tensors_by_id
                .get(&chunk.tensor_id)
                .map(|t| t.name.clone());
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

        let result = f(compressed, budget);
        if is_bounded {
            budget.release(chunk.compressed_size as usize, compressed_label)?;
        }
        result
    }

    /// Expose a raw identity chunk byte range without touching the full chunk.
    ///
    /// The requested range must have persisted range checksum metadata. This is
    /// the runtime primitive for input-major sidecar tensors where one selected
    /// activation feature maps to one contiguous weight range.
    pub fn with_raw_chunk_range<R>(
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
        if chunk.codec_id != "rtc-raw-v1" {
            return Err(RuntimeError::InvalidTensorData(format!(
                "raw chunk range requires rtc-raw-v1, got {} for chunk {}",
                chunk.codec_id, chunk.chunk_id
            )));
        }

        let end = byte_offset.checked_add(byte_len).ok_or_else(|| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} raw range [{byte_offset}, +{byte_len}) overflows",
                chunk.chunk_id
            ))
        })?;
        if end > chunk.compressed_size {
            return Err(RuntimeError::InvalidTensorData(format!(
                "chunk {} raw range [{byte_offset}, {end}) exceeds compressed size {}",
                chunk.chunk_id, chunk.compressed_size
            )));
        }

        let is_bounded = budget.limit_bytes() != usize::MAX;
        let range_bytes = usize::try_from(byte_len).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} raw range len {byte_len} overflows usize",
                chunk.chunk_id
            ))
        })?;
        let range_label = if is_bounded {
            format!(
                "raw chunk {} byte range [{}..{})",
                chunk.chunk_id, byte_offset, end
            )
        } else {
            String::new()
        };
        if is_bounded {
            budget.reserve(range_bytes, range_label.clone())?;
        }

        let read_start = Instant::now();
        let compressed_range =
            match self
                .reader
                .read_chunk_range_slice(chunk.chunk_id, byte_offset, byte_len)
            {
                Ok(bytes) => unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) },
                Err(err) => {
                    if is_bounded {
                        budget.release(range_bytes, range_label)?;
                    }
                    return Err(err.into());
                }
            };

        if self.rama_trace.is_some() {
            let tensor_name = self
                .tensors_by_id
                .get(&chunk.tensor_id)
                .map(|t| t.name.clone());
            self.record_rama_chunk_event(
                "chunk_range_read",
                format!(
                    "read chunk {} range {}..{}",
                    chunk.chunk_id, byte_offset, end
                ),
                &chunk,
                tensor_name.as_deref(),
                read_start,
                read_start.elapsed(),
                budget,
            );
        }

        let checksum_start = Instant::now();
        let compressed_verified = match self.verify_compressed_chunk_range_if_needed(
            &chunk,
            byte_offset,
            byte_len,
            compressed_range,
        ) {
            Ok(verified) => verified,
            Err(err) => {
                if is_bounded {
                    budget.release(range_bytes, range_label)?;
                }
                return Err(err);
            }
        };
        if compressed_verified && self.rama_trace.is_some() {
            let tensor_name = self
                .tensors_by_id
                .get(&chunk.tensor_id)
                .map(|t| t.name.clone());
            self.record_rama_chunk_event(
                "chunk_range_checksum",
                format!(
                    "verify chunk {} range {}..{}",
                    chunk.chunk_id, byte_offset, end
                ),
                &chunk,
                tensor_name.as_deref(),
                checksum_start,
                checksum_start.elapsed(),
                budget,
            );
        }

        let result = f(compressed_range, budget);
        if is_bounded {
            budget.release(range_bytes, range_label)?;
        }
        result
    }

    /// Expose two raw compressed chunks at once for fused kernels that combine
    /// compatible tensors without materializing either full intermediate.
    pub fn with_two_raw_chunks<R>(
        &mut self,
        first_chunk_id: u64,
        second_chunk_id: u64,
        budget: &mut MemoryBudget,
        f: impl FnOnce(&[u8], &[u8], &mut MemoryBudget) -> Result<R>,
    ) -> Result<R> {
        let first_chunk = self
            .chunks_by_id
            .get(&first_chunk_id)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData(format!("missing chunk {first_chunk_id}"))
            })?
            .clone();
        let second_chunk = self
            .chunks_by_id
            .get(&second_chunk_id)
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData(format!("missing chunk {second_chunk_id}"))
            })?
            .clone();

        let is_bounded = budget.limit_bytes() != usize::MAX;
        let first_label = if is_bounded {
            format!("compressed chunk {}", first_chunk.chunk_id)
        } else {
            String::new()
        };
        let second_label = if is_bounded {
            format!("compressed chunk {}", second_chunk.chunk_id)
        } else {
            String::new()
        };

        if is_bounded {
            budget.reserve(first_chunk.compressed_size as usize, first_label.clone())?;
            if let Err(err) =
                budget.reserve(second_chunk.compressed_size as usize, second_label.clone())
            {
                budget.release(first_chunk.compressed_size as usize, first_label)?;
                return Err(err);
            }
        }

        let first_raw = match self.reader.read_chunk_slice(first_chunk.chunk_id) {
            Ok(bytes) => unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) },
            Err(err) => {
                if is_bounded {
                    budget.release(second_chunk.compressed_size as usize, second_label)?;
                    budget.release(first_chunk.compressed_size as usize, first_label)?;
                }
                return Err(err.into());
            }
        };
        let second_raw = match self.reader.read_chunk_slice(second_chunk.chunk_id) {
            Ok(bytes) => unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) },
            Err(err) => {
                if is_bounded {
                    budget.release(second_chunk.compressed_size as usize, second_label)?;
                    budget.release(first_chunk.compressed_size as usize, first_label)?;
                }
                return Err(err.into());
            }
        };

        let first_verified = match self.verify_compressed_chunk_if_needed(&first_chunk, first_raw) {
            Ok(verified) => verified,
            Err(err) => {
                if is_bounded {
                    budget.release(second_chunk.compressed_size as usize, second_label)?;
                    budget.release(first_chunk.compressed_size as usize, first_label)?;
                }
                return Err(err);
            }
        };
        let second_verified =
            match self.verify_compressed_chunk_if_needed(&second_chunk, second_raw) {
                Ok(verified) => verified,
                Err(err) => {
                    if is_bounded {
                        budget.release(second_chunk.compressed_size as usize, second_label)?;
                        budget.release(first_chunk.compressed_size as usize, first_label)?;
                    }
                    return Err(err);
                }
            };

        if self.rama_trace.is_some() {
            if first_verified {
                let now = Instant::now();
                let tensor_name = self
                    .tensors_by_id
                    .get(&first_chunk.tensor_id)
                    .map(|t| t.name.clone());
                self.record_rama_chunk_event(
                    "chunk_compressed_checksum",
                    format!("verify compressed chunk {}", first_chunk.chunk_id),
                    &first_chunk,
                    tensor_name.as_deref(),
                    now,
                    now.elapsed(),
                    budget,
                );
            }
            if second_verified {
                let now = Instant::now();
                let tensor_name = self
                    .tensors_by_id
                    .get(&second_chunk.tensor_id)
                    .map(|t| t.name.clone());
                self.record_rama_chunk_event(
                    "chunk_compressed_checksum",
                    format!("verify compressed chunk {}", second_chunk.chunk_id),
                    &second_chunk,
                    tensor_name.as_deref(),
                    now,
                    now.elapsed(),
                    budget,
                );
            }
        }

        let result = f(first_raw, second_raw, budget);
        if is_bounded {
            budget.release(second_chunk.compressed_size as usize, second_label)?;
            budget.release(first_chunk.compressed_size as usize, first_label)?;
        }
        result
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

        let is_bounded = budget.limit_bytes() != usize::MAX;
        let compressed_label = if is_bounded {
            format!("compressed chunk {}", chunk.chunk_id)
        } else {
            String::new()
        };

        if is_bounded {
            budget.reserve(chunk.compressed_size as usize, compressed_label.clone())?;
        }

        let read_start = Instant::now();
        let compressed = match self.reader.read_chunk_slice(chunk.chunk_id) {
            Ok(bytes) => unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) },
            Err(err) => {
                if is_bounded {
                    budget.release(chunk.compressed_size as usize, compressed_label)?;
                }
                return Err(err.into());
            }
        };

        if self.rama_trace.is_some() {
            let tensor_name = self
                .tensors_by_id
                .get(&chunk.tensor_id)
                .map(|t| t.name.clone());
            self.record_rama_chunk_event(
                "chunk_read",
                format!("read chunk {}", chunk.chunk_id),
                &chunk,
                tensor_name.as_deref(),
                read_start,
                read_start.elapsed(),
                budget,
            );
        }

        let compressed_checksum_start = Instant::now();
        let compressed_verified = match self.verify_compressed_chunk_if_needed(&chunk, compressed) {
            Ok(verified) => verified,
            Err(err) => {
                if is_bounded {
                    budget.release(chunk.compressed_size as usize, compressed_label)?;
                }
                return Err(err);
            }
        };
        if compressed_verified && self.rama_trace.is_some() {
            let tensor_name = self
                .tensors_by_id
                .get(&chunk.tensor_id)
                .map(|t| t.name.clone());
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

        let decoded_label = if is_bounded {
            format!("decoded chunk {}", chunk.chunk_id)
        } else {
            String::new()
        };

        if is_bounded {
            if let Err(err) =
                budget.reserve(chunk.uncompressed_size as usize, decoded_label.clone())
            {
                budget.release(chunk.compressed_size as usize, compressed_label)?;
                return Err(err);
            }
        }

        let codec = match codec_for_id(&chunk.codec_id) {
            Ok(codec) => codec,
            Err(err) => {
                if is_bounded {
                    budget.release(chunk.uncompressed_size as usize, decoded_label)?;
                    budget.release(chunk.compressed_size as usize, compressed_label)?;
                }
                return Err(err);
            }
        };
        let decode_start = Instant::now();
        let decoded = match codec.decode(
            compressed,
            &rtc_codec::DecodeMeta {
                codec_id: chunk.codec_id.clone(),
                uncompressed_size: chunk.uncompressed_size,
            },
        ) {
            Ok(bytes) => bytes,
            Err(err) => {
                if is_bounded {
                    budget.release(chunk.uncompressed_size as usize, decoded_label)?;
                    budget.release(chunk.compressed_size as usize, compressed_label)?;
                }
                return Err(err.into());
            }
        };
        if self.rama_trace.is_some() {
            let tensor_name = self
                .tensors_by_id
                .get(&chunk.tensor_id)
                .map(|t| t.name.clone());
            self.record_rama_chunk_event(
                "chunk_decode",
                format!("decode chunk {}", chunk.chunk_id),
                &chunk,
                tensor_name.as_deref(),
                decode_start,
                decode_start.elapsed(),
                budget,
            );
        }
        if is_bounded {
            budget.release(chunk.compressed_size as usize, compressed_label)?;
        }

        if decoded.len() != chunk.uncompressed_size as usize {
            if is_bounded {
                budget.release(chunk.uncompressed_size as usize, decoded_label)?;
            }
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
                if is_bounded {
                    budget.release(chunk.uncompressed_size as usize, decoded_label)?;
                }
                return Err(err);
            }
        };
        if original_verified && self.rama_trace.is_some() {
            let tensor_name = self
                .tensors_by_id
                .get(&chunk.tensor_id)
                .map(|t| t.name.clone());
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
        if self.rama_trace.is_some() {
            let tensor_name = self
                .tensors_by_id
                .get(&chunk.tensor_id)
                .map(|t| t.name.clone());
            self.record_rama_chunk_event(
                "chunk_compute_closure",
                format!("compute with decoded chunk {}", chunk.chunk_id),
                &chunk,
                tensor_name.as_deref(),
                compute_start,
                compute_start.elapsed(),
                budget,
            );
        }
        if is_bounded {
            budget.release(chunk.uncompressed_size as usize, decoded_label)?;
        }
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

        let is_bounded = budget.limit_bytes() != usize::MAX;
        let compressed_label = if is_bounded {
            format!("compressed chunk {}", chunk.chunk_id)
        } else {
            String::new()
        };

        if is_bounded {
            budget.reserve(chunk.compressed_size as usize, compressed_label.clone())?;
        }
        let compressed = match self.reader.read_chunk_slice(chunk.chunk_id) {
            Ok(bytes) => unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) },
            Err(err) => {
                if is_bounded {
                    budget.release(chunk.compressed_size as usize, compressed_label)?;
                }
                return Err(err.into());
            }
        };

        if let Err(err) = self.verify_compressed_chunk_if_needed(&chunk, compressed) {
            if is_bounded {
                budget.release(chunk.compressed_size as usize, compressed_label)?;
            }
            return Err(err);
        }

        let decoded_range_bytes = usize::try_from(byte_len).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} range len {} overflows usize",
                chunk.chunk_id, byte_len
            ))
        })?;
        let decoded_label = if is_bounded {
            format!(
                "decoded chunk {} byte range [{}..{})",
                chunk.chunk_id,
                byte_offset,
                range.end()?
            )
        } else {
            String::new()
        };
        if is_bounded {
            if let Err(err) = budget.reserve(decoded_range_bytes, decoded_label.clone()) {
                budget.release(chunk.compressed_size as usize, compressed_label)?;
                return Err(err);
            }
        }

        let decoded = match codec.decode_range(
            compressed,
            &rtc_codec::DecodeMeta {
                codec_id: chunk.codec_id.clone(),
                uncompressed_size: chunk.uncompressed_size,
            },
            range,
        ) {
            Ok(bytes) => bytes,
            Err(err) => {
                if is_bounded {
                    budget.release(decoded_range_bytes, decoded_label)?;
                    budget.release(chunk.compressed_size as usize, compressed_label)?;
                }
                return Err(err.into());
            }
        };
        if is_bounded {
            budget.release(chunk.compressed_size as usize, compressed_label)?;
        }

        if decoded.len() != decoded_range_bytes {
            if is_bounded {
                budget.release(decoded_range_bytes, decoded_label)?;
            }
            return Err(RuntimeError::InvalidTensorData(format!(
                "chunk {} range decoded to {} bytes, expected {}",
                chunk.chunk_id,
                decoded.len(),
                decoded_range_bytes
            )));
        }

        let result = f(&decoded, budget);
        if is_bounded {
            budget.release(decoded_range_bytes, decoded_label)?;
        }
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
        let raw_bytes = crate::loader::decode_tensor_bytes(&self.reader, &tensor_meta)?;
        verify_tensor_checksum(&tensor_meta, &raw_bytes)?;
        budget.release(
            tensor_meta.original_size_bytes as usize,
            format!("roundtrip tensor raw release: {name}"),
        )?;
        Ok(())
    }
}

/// Thread count for the parallel integrity prewarm. Honors `RLLM_THREADS`
/// (the same knob the q8 kernels use) and otherwise uses the available
/// parallelism, falling back to single-threaded.
fn prewarm_thread_count() -> usize {
    if let Ok(raw) = std::env::var("RLLM_THREADS") {
        if let Ok(n) = raw.trim().parse::<usize>() {
            if n >= 1 {
                return n;
            }
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

pub(crate) fn runtime_f32_bytes_for_tensor(tensor: &TensorMeta) -> Result<usize> {
    let elements = if tensor.dtype.is_quantized() {
        tensor.shape.iter().product::<u64>()
    } else {
        let dtype_size = tensor.dtype.size_bytes() as u64;
        if dtype_size == 0 || !tensor.original_size_bytes.is_multiple_of(dtype_size) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "tensor {} has original_size_bytes={} not divisible by dtype size {}",
                tensor.name, tensor.original_size_bytes, dtype_size
            )));
        }
        tensor.original_size_bytes / dtype_size
    };
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

    /// Dump the bf16 tied embedding of the raw Llama 1B model to /tmp for the
    /// rtc-codec feasibility measurement. Needs the local artifact.
    /// Run: `cargo test -p rllm-runtime --release dump_bf16_embedding_sample -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_bf16_embedding_sample() {
        // Dump the bf16 tied embedding of the raw Llama 1B model to /tmp for the
        // rtc-codec feasibility measurement. Needs the local artifact.
        let path = "../../models/Llama-3.2-1B-Instruct-raw.rllm";
        let mut m = LazyRllmModel::open(path).unwrap();
        let name = "model.embed_tokens.weight";
        let meta = m.tensor(name).unwrap().clone();
        assert_eq!(format!("{:?}", meta.dtype), "Bf16");
        // raw bf16 bytes straight from the mmap (one contiguous tensor).
        let bytes = m
            .with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec()))
            .unwrap()
            .expect("embedding is contiguous-raw");
        std::fs::write("/tmp/rllm-bf16-sample.bin", &bytes).unwrap();
        eprintln!("wrote {} bf16 bytes to /tmp/rllm-bf16-sample.bin", bytes.len());
    }

    /// Measure the ACTUAL q8_0 quantization error of a packed model against its
    /// bf16 original, one tensor at a time (decode pair -> compare -> drop, so the
    /// transient is bounded to ~2 tensors). Needs both local artifacts. Run:
    /// `cargo test -p rllm-runtime --release q8_vs_bf16_quantization_error -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn q8_vs_bf16_quantization_error() {
        let q8_path = "../../models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm";
        let raw_path = "../../models/Llama-3.2-1B-Instruct-raw.rllm";
        let mut q8 = LazyRllmModel::open(q8_path).expect("open q8");
        let mut raw = LazyRllmModel::open(raw_path).expect("open raw bf16");

        // Representative transformer weights across the stack (these are Q8_0 in
        // the keep-io pack, Bf16 in the raw pack).
        let names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.7.mlp.up_proj.weight",
            "model.layers.15.self_attn.o_proj.weight",
            "model.layers.15.mlp.down_proj.weight",
        ];

        let mut tot_sq_err = 0.0f64;
        let mut tot_sq_ref = 0.0f64;
        let mut tot_n = 0u64;
        let mut global_max = 0.0f32;

        for name in names {
            let dq = {
                let mut b = MemoryBudget::unbounded();
                q8.decode_tensor(name, &mut b).expect("decode q8").data
            };
            let bf = {
                let mut b = MemoryBudget::unbounded();
                raw.decode_tensor(name, &mut b).expect("decode raw").data
            };
            assert_eq!(dq.len(), bf.len(), "{name} length mismatch");

            let mut sq_err = 0.0f64;
            let mut sq_ref = 0.0f64;
            let mut max_abs = 0.0f32;
            for (x, y) in dq.iter().zip(bf.iter()) {
                let d = (x - y).abs();
                sq_err += (d as f64) * (d as f64);
                sq_ref += (*y as f64) * (*y as f64);
                if d > max_abs {
                    max_abs = d;
                }
            }
            let n = dq.len() as f64;
            let rms_err = (sq_err / n).sqrt();
            let rms_ref = (sq_ref / n).sqrt();
            eprintln!(
                "{name:55} rel_rms={:.4}%  max_abs={:.6}  (rms_ref={:.6})",
                100.0 * rms_err / rms_ref,
                max_abs,
                rms_ref
            );
            tot_sq_err += sq_err;
            tot_sq_ref += sq_ref;
            tot_n += dq.len() as u64;
            if max_abs > global_max {
                global_max = max_abs;
            }
            // dq + bf drop here -> next pair reuses the memory.
        }

        let overall_rel =
            100.0 * (tot_sq_err / tot_n as f64).sqrt() / (tot_sq_ref / tot_n as f64).sqrt();
        eprintln!(
            "\n=== OVERALL q8_0 vs bf16: rel_rms_error = {:.4}%  global_max_abs = {:.6}  (weights={}) ===",
            overall_rel, global_max, tot_n
        );
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
    fn with_raw_tensor_skips_whole_tensor_hash_once_chunks_are_verified() {
        // Tensor written with a DELIBERATELY WRONG tensor-level checksum but a
        // CORRECT chunk checksum. A whole-tensor verify would fail; a chunk
        // verify passes. This lets us observe whether with_raw_tensor skips the
        // redundant whole-tensor hash once the chunks are already verified.
        let path = temp_path("raw-tensor-bridge");
        let data: Vec<u8> = (0..32).collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "bridge.weight".to_string(),
            shape: vec![32],
            dtype: DType::U8,
            original_size_bytes: data.len() as u64,
            compressed_size_bytes: data.len() as u64,
            original_sha256: [0xAB; 32], // wrong on purpose
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer.write_chunk(0, "rtc-raw-v1", &data, &data, 0).unwrap();
        writer.finalize().unwrap();

        // (a) VerifyOnce, chunks NOT pre-verified: with_raw_tensor runs the
        // whole-tensor verify, which catches the bad checksum → declines.
        {
            let mut model = LazyRllmModel::open(&path).unwrap();
            model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
            let got = model.with_raw_tensor(0, |bytes| Ok(bytes.len())).unwrap();
            assert_eq!(got, None, "bad tensor checksum must be caught when chunks are unverified");
        }

        // (b) VerifyOnce, chunk pre-verified (as prefill does): the bytes are
        // already proven intact, so with_raw_tensor skips the redundant hash and
        // succeeds despite the bad tensor-level checksum.
        {
            let mut model = LazyRllmModel::open(&path).unwrap();
            model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
            let mut budget = MemoryBudget::new(usize::MAX);
            model.with_raw_chunk(0, &mut budget, |bytes, _b| Ok(bytes.len())).unwrap();
            let got = model.with_raw_tensor(0, |bytes| Ok(bytes.len())).unwrap();
            assert_eq!(
                got,
                Some(data.len()),
                "verified chunks must let with_raw_tensor skip the redundant whole-tensor hash"
            );
        }

        // (c) Strict never skips: even after the chunk is verified, the bad
        // tensor checksum is re-checked on every call and caught.
        {
            let mut model = LazyRllmModel::open(&path).unwrap();
            model.set_rama_integrity_mode(RamaIntegrityMode::Strict);
            let mut budget = MemoryBudget::new(usize::MAX);
            model.with_raw_chunk(0, &mut budget, |bytes, _b| Ok(bytes.len())).unwrap();
            let got = model.with_raw_tensor(0, |bytes| Ok(bytes.len())).unwrap();
            assert_eq!(got, None, "Strict must re-verify the whole tensor on every call");
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prewarm_chunk_integrity_verifies_chunks_and_enables_the_bridge() {
        // Tensor with a WRONG whole-tensor checksum but correct chunk checksums:
        // only a working chunk->tensor bridge (fed by the prewarm) lets
        // with_raw_tensor succeed.
        let path = temp_path("prewarm-integrity");
        let data: Vec<u8> = (0..48).collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "w".to_string(),
            shape: vec![48],
            dtype: DType::U8,
            original_size_bytes: 48,
            compressed_size_bytes: 48,
            original_sha256: [0xCD; 32], // wrong on purpose
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer.write_chunk(0, "rtc-raw-v1", &data, &data, 0).unwrap();
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
        assert_eq!(model.prewarm_chunk_integrity().unwrap(), 1, "verifies the one chunk");
        // Idempotent: nothing left to verify on a second call.
        assert_eq!(model.prewarm_chunk_integrity().unwrap(), 0);
        // Bridge active: with_raw_tensor skips the (wrong) whole-tensor hash.
        assert_eq!(model.with_raw_tensor(0, |b| Ok(b.len())).unwrap(), Some(48));

        // No-op outside VerifyOnce.
        for mode in [RamaIntegrityMode::Unchecked, RamaIntegrityMode::Strict] {
            let mut other = LazyRllmModel::open(&path).unwrap();
            other.set_rama_integrity_mode(mode);
            assert_eq!(other.prewarm_chunk_integrity().unwrap(), 0);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn unchecked_integrity_records_no_checksum_events() {
        let path = temp_path("chunk-unchecked");
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
        model.set_rama_integrity_mode(RamaIntegrityMode::Unchecked);
        model.enable_rama_trace();
        let mut budget = MemoryBudget::new(64);
        model
            .with_decoded_chunk(0, &mut budget, |bytes, _budget| Ok(bytes.len()))
            .unwrap();

        let trace = model.take_rama_trace().expect("trace should be enabled");
        assert!(trace
            .events
            .iter()
            .all(|event| event.phase != "chunk_compressed_checksum"
                && event.phase != "chunk_original_checksum"));
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
