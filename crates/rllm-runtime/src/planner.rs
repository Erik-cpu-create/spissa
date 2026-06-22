// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use crate::lazy::runtime_f32_bytes_for_tensor;
use crate::{LazyRllmModel, Result, RuntimeError};
use rllm_container::{ChunkMeta, TensorMeta};
use std::collections::{BTreeMap, HashMap};
use std::str::FromStr;

const DEFAULT_TILE_STREAM_ELEMENTS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    FullDecode,
    LayerStream,
    TileStream,
}

impl RuntimeMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            RuntimeMode::FullDecode => "full-decode",
            RuntimeMode::LayerStream => "layer-stream",
            RuntimeMode::TileStream => "tile-stream",
        }
    }
}

impl FromStr for RuntimeMode {
    type Err = RuntimeError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "full" | "full-decode" | "fulldecode" => Ok(RuntimeMode::FullDecode),
            "layer" | "layer-stream" | "layer-decode" | "layerstream" => {
                Ok(RuntimeMode::LayerStream)
            }
            "tile" | "tile-stream" | "tile-decode" | "tilestream" => Ok(RuntimeMode::TileStream),
            other => Err(RuntimeError::InvalidRuntimeMode(other.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimePlanConfig {
    pub mode: RuntimeMode,
    pub context_length: usize,
    pub memory_budget_bytes: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ModelShapeHints {
    pub hidden_size: Option<usize>,
    pub num_layers: usize,
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RuntimePlan {
    pub model_name: String,
    pub architecture: String,
    pub mode: RuntimeMode,
    pub context_length: usize,
    pub memory_budget_bytes: Option<usize>,
    pub file_size_bytes: u64,
    pub compressed_chunk_bytes: u64,
    pub tensor_count: usize,
    pub chunk_count: usize,
    pub total_original_bytes: u64,
    pub full_decode_runtime_bytes: usize,
    pub metadata_index_bytes_estimate: usize,
    pub activation_window_bytes: usize,
    pub kv_cache_bytes_estimate: usize,
    pub planned_peak_bytes: usize,
    pub largest_step: PlanStep,
    pub shape_hints: ModelShapeHints,
    pub status: PlanStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanStatus {
    Ok,
    OverBudget { over_by_bytes: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanStep {
    pub label: String,
    pub tensor_name: Option<String>,
    pub chunk_id: Option<u64>,
    pub bytes: usize,
}

impl PlanStep {
    fn zero(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            tensor_name: None,
            chunk_id: None,
            bytes: 0,
        }
    }
}

impl RuntimePlan {
    pub fn within_budget(&self) -> bool {
        matches!(self.status, PlanStatus::Ok)
    }
}

pub fn build_runtime_plan(model: &LazyRllmModel, config: RuntimePlanConfig) -> Result<RuntimePlan> {
    let shape_hints = infer_shape_hints(model);
    let metadata_index_bytes_estimate = estimate_metadata_index_bytes(model);
    let activation_window_bytes =
        estimate_activation_window_bytes(&shape_hints, config.context_length);
    let kv_cache_bytes_estimate = estimate_kv_cache_bytes(&shape_hints, config.context_length);
    let runtime_state_bytes = metadata_index_bytes_estimate
        .saturating_add(activation_window_bytes)
        .saturating_add(kv_cache_bytes_estimate);

    let largest_step = match config.mode {
        RuntimeMode::FullDecode => full_decode_step(model, runtime_state_bytes),
        RuntimeMode::LayerStream => layer_stream_step(model, runtime_state_bytes)?,
        RuntimeMode::TileStream => tile_stream_step(model, runtime_state_bytes)?,
    };

    let planned_peak_bytes = largest_step.bytes;
    let status = match config.memory_budget_bytes {
        Some(limit) if planned_peak_bytes > limit => PlanStatus::OverBudget {
            over_by_bytes: planned_peak_bytes - limit,
        },
        _ => PlanStatus::Ok,
    };

    Ok(RuntimePlan {
        model_name: model.metadata().model_name.clone(),
        architecture: model.metadata().architecture.clone(),
        mode: config.mode,
        context_length: config.context_length,
        memory_budget_bytes: config.memory_budget_bytes,
        file_size_bytes: model.stats().file_size_bytes,
        compressed_chunk_bytes: model.stats().total_compressed_chunk_bytes,
        tensor_count: model.stats().tensor_count,
        chunk_count: model.stats().chunk_count,
        total_original_bytes: model.stats().total_original_bytes,
        full_decode_runtime_bytes: model.stats().full_decode_runtime_bytes,
        metadata_index_bytes_estimate,
        activation_window_bytes,
        kv_cache_bytes_estimate,
        planned_peak_bytes,
        largest_step,
        shape_hints,
        status,
    })
}

fn full_decode_step(model: &LazyRllmModel, runtime_state_bytes: usize) -> PlanStep {
    let largest_raw_tensor = model
        .tensors()
        .map(|tensor| tensor.original_size_bytes as usize)
        .max()
        .unwrap_or(0);
    PlanStep {
        label: "full-decode all tensors to f32".to_string(),
        tensor_name: None,
        chunk_id: None,
        bytes: runtime_state_bytes
            .saturating_add(model.stats().full_decode_runtime_bytes)
            .saturating_add(largest_raw_tensor),
    }
}

fn layer_stream_step(model: &LazyRllmModel, runtime_state_bytes: usize) -> Result<PlanStep> {
    let mut layer_bytes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut non_layer_bytes = 0usize;

    for tensor in model.tensors() {
        let runtime_bytes = runtime_f32_bytes_for_tensor(tensor)?;
        let decode_window = runtime_bytes.saturating_add(tensor.original_size_bytes as usize);
        if let Some(layer_id) = parse_layer_id(&tensor.name) {
            *layer_bytes.entry(layer_id).or_default() = layer_bytes
                .get(&layer_id)
                .copied()
                .unwrap_or(0)
                .saturating_add(decode_window);
        } else {
            non_layer_bytes = non_layer_bytes.max(decode_window);
        }
    }

    let (layer_id, bytes) = layer_bytes
        .iter()
        .max_by_key(|(_, bytes)| *bytes)
        .map(|(layer_id, bytes)| (*layer_id, *bytes))
        .unwrap_or((0, 0));

    if bytes >= non_layer_bytes {
        Ok(PlanStep {
            label: format!("layer-stream decode layer {layer_id}"),
            tensor_name: None,
            chunk_id: None,
            bytes: runtime_state_bytes.saturating_add(bytes),
        })
    } else {
        Ok(PlanStep {
            label: "layer-stream non-layer tensor window".to_string(),
            tensor_name: None,
            chunk_id: None,
            bytes: runtime_state_bytes.saturating_add(non_layer_bytes),
        })
    }
}

fn tile_stream_step(model: &LazyRllmModel, runtime_state_bytes: usize) -> Result<PlanStep> {
    let mut largest = PlanStep::zero("tile-stream no chunks");
    let tensors_by_id: HashMap<u64, &TensorMeta> = model
        .tensors()
        .map(|tensor| (tensor.tensor_id, tensor))
        .collect();

    for chunk in model.chunks() {
        let Some(tensor) = tensors_by_id.get(&chunk.tensor_id) else {
            continue;
        };
        let scratch_bytes = runtime_f32_tile_scratch_bytes_for_chunk(chunk, tensor)?;
        let step_bytes = runtime_state_bytes
            .saturating_add(chunk.compressed_size as usize)
            .saturating_add(chunk.uncompressed_size as usize)
            .saturating_add(scratch_bytes);
        if step_bytes > largest.bytes {
            largest = PlanStep {
                label: "tile-stream fused tile decode+matmul chunk".to_string(),
                tensor_name: Some(tensor.name.clone()),
                chunk_id: Some(chunk.chunk_id),
                bytes: step_bytes,
            };
        }
    }

    Ok(largest)
}

#[cfg(test)]
fn runtime_f32_bytes_for_chunk(chunk: &ChunkMeta, tensor: &TensorMeta) -> Result<usize> {
    let dtype_size = tensor.dtype.size_bytes() as u64;
    if dtype_size == 0 || !chunk.uncompressed_size.is_multiple_of(dtype_size) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "chunk {} for tensor {} has uncompressed_size={} not divisible by dtype size {}",
            chunk.chunk_id, tensor.name, chunk.uncompressed_size, dtype_size
        )));
    }
    let elements = chunk.uncompressed_size / dtype_size;
    elements
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} runtime f32 scratch size overflows usize",
                chunk.chunk_id
            ))
        })
}

fn runtime_f32_tile_scratch_bytes_for_chunk(
    chunk: &ChunkMeta,
    tensor: &TensorMeta,
) -> Result<usize> {
    let dtype_size = tensor.dtype.size_bytes() as u64;
    if dtype_size == 0 || !chunk.uncompressed_size.is_multiple_of(dtype_size) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "chunk {} for tensor {} has uncompressed_size={} not divisible by dtype size {}",
            chunk.chunk_id, tensor.name, chunk.uncompressed_size, dtype_size
        )));
    }
    let elements = usize::try_from(chunk.uncompressed_size / dtype_size).map_err(|_| {
        RuntimeError::InvalidTensorData(format!(
            "chunk {} element count overflows usize",
            chunk.chunk_id
        ))
    })?;
    elements
        .min(DEFAULT_TILE_STREAM_ELEMENTS)
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} tile f32 scratch size overflows usize",
                chunk.chunk_id
            ))
        })
}

fn estimate_activation_window_bytes(hints: &ModelShapeHints, ctx: usize) -> usize {
    hints
        .hidden_size
        .map(|hidden| {
            ctx.saturating_mul(hidden)
                .saturating_mul(std::mem::size_of::<f32>())
        })
        .unwrap_or(0)
}

fn estimate_kv_cache_bytes(hints: &ModelShapeHints, ctx: usize) -> usize {
    // Low-RAM plan assumes KV cache is retained in fp16/bf16-sized storage.
    hints
        .hidden_size
        .map(|hidden| {
            hints
                .num_layers
                .saturating_mul(ctx)
                .saturating_mul(hidden)
                .saturating_mul(2) // K + V
                .saturating_mul(2) // fp16/bf16 bytes
        })
        .unwrap_or(0)
}

fn estimate_metadata_index_bytes(model: &LazyRllmModel) -> usize {
    let name_bytes: usize = model.tensors().map(|tensor| tensor.name.len()).sum();
    let codec_bytes: usize = model.chunks().map(|chunk| chunk.codec_id.len()).sum();
    model
        .stats()
        .tensor_count
        .saturating_mul(256)
        .saturating_add(model.stats().chunk_count.saturating_mul(192))
        .saturating_add(name_bytes)
        .saturating_add(codec_bytes)
}

fn infer_shape_hints(model: &LazyRllmModel) -> ModelShapeHints {
    let mut hidden_size = None;
    let mut vocab_size = None;
    let mut max_position_embeddings = Some(model.metadata().default_context_length as usize);
    let mut max_layer_id = None;

    for tensor in model.tensors() {
        if (tensor.name.contains("embed_in.weight") || tensor.name == "tok_embeddings.weight")
            && tensor.shape.len() == 2
        {
            vocab_size = usize::try_from(tensor.shape[0]).ok();
            hidden_size = usize::try_from(tensor.shape[1]).ok();
        }
        if (tensor.name.contains("embed_out.weight") || tensor.name == "lm_head.weight")
            && tensor.shape.len() == 2
        {
            vocab_size.get_or_insert(usize::try_from(tensor.shape[0]).unwrap_or(0));
            hidden_size.get_or_insert(usize::try_from(tensor.shape[1]).unwrap_or(0));
        }
        if tensor.name.contains("attention.bias") && tensor.shape.len() == 4 {
            max_position_embeddings = usize::try_from(tensor.shape[2]).ok();
        }
        if let Some(layer_id) = parse_layer_id(&tensor.name) {
            max_layer_id = Some(max_layer_id.map_or(layer_id, |max: usize| max.max(layer_id)));
        }
    }

    ModelShapeHints {
        hidden_size,
        num_layers: max_layer_id.map(|id| id + 1).unwrap_or(0),
        vocab_size,
        max_position_embeddings,
    }
}

fn parse_layer_id(name: &str) -> Option<usize> {
    let marker = ".layers.";
    let start = name.find(marker)? + marker.len();
    let rest = &name[start..];
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rllm_container::{DType, GlobalMetadata, RllmWriter};
    use sha2::{Digest, Sha256};

    fn sha256_array(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rllm-plan-{name}-{}.spsa", std::process::id()))
    }

    fn write_test_model(path: &std::path::Path) {
        let mut meta = GlobalMetadata::new_test();
        meta.model_name = "plan-test".to_string();
        meta.default_context_length = 8;
        let mut writer = RllmWriter::new(path, meta).unwrap();

        let embed = vec![0u8; 16 * 2]; // 16 fp16 values
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "gpt_neox.embed_in.weight".to_string(),
            shape: vec![4, 4],
            dtype: DType::Fp16,
            original_size_bytes: embed.len() as u64,
            compressed_size_bytes: embed.len() as u64,
            original_sha256: sha256_array(&embed),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(0, "rtc-raw-v1", &embed, &embed, 0)
            .unwrap();

        let layer = vec![1u8; 32 * 2]; // 32 fp16 values
        writer.add_tensor(TensorMeta {
            tensor_id: 1,
            name: "gpt_neox.layers.0.attention.dense.weight".to_string(),
            shape: vec![8, 4],
            dtype: DType::Fp16,
            original_size_bytes: layer.len() as u64,
            compressed_size_bytes: layer.len() as u64,
            original_sha256: sha256_array(&layer),
            chunk_count: 1,
            chunk_start_index: 1,
        });
        writer
            .write_chunk(1, "rtc-raw-v1", &layer, &layer, 0)
            .unwrap();
        writer.finalize().unwrap();
    }

    #[test]
    fn parses_runtime_modes() {
        assert_eq!(
            "full-decode".parse::<RuntimeMode>().unwrap(),
            RuntimeMode::FullDecode
        );
        assert_eq!(
            "layer-stream".parse::<RuntimeMode>().unwrap(),
            RuntimeMode::LayerStream
        );
        assert_eq!(
            "tile-stream".parse::<RuntimeMode>().unwrap(),
            RuntimeMode::TileStream
        );
        assert!("fast-mode".parse::<RuntimeMode>().is_err());
    }

    #[test]
    fn tile_stream_plan_reports_largest_chunk_window_and_budget_status() {
        let path = temp_path("tile");
        write_test_model(&path);
        let model = LazyRllmModel::open(&path).unwrap();
        let plan = build_runtime_plan(
            &model,
            RuntimePlanConfig {
                mode: RuntimeMode::TileStream,
                context_length: 4,
                memory_budget_bytes: Some(1024 * 1024),
            },
        )
        .unwrap();

        assert!(plan.within_budget());
        assert_eq!(plan.shape_hints.hidden_size, Some(4));
        assert_eq!(plan.shape_hints.num_layers, 1);
        assert_eq!(plan.largest_step.chunk_id, Some(1));
        assert_eq!(
            plan.largest_step.label,
            "tile-stream fused tile decode+matmul chunk"
        );
        assert!(plan.planned_peak_bytes >= plan.activation_window_bytes);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn tile_stream_plan_caps_f32_scratch_to_tile_window() {
        let tensor = TensorMeta {
            tensor_id: 9,
            name: "large.weight".to_string(),
            shape: vec![(DEFAULT_TILE_STREAM_ELEMENTS * 2) as u64],
            dtype: DType::Fp16,
            original_size_bytes: (DEFAULT_TILE_STREAM_ELEMENTS * 2 * 2) as u64,
            compressed_size_bytes: 1,
            original_sha256: [0u8; 32],
            chunk_count: 1,
            chunk_start_index: 0,
        };
        let chunk = ChunkMeta {
            chunk_id: 11,
            tensor_id: 9,
            chunk_offset_in_tensor: 0,
            uncompressed_size: tensor.original_size_bytes,
            compressed_size: 1,
            file_offset: 0,
            codec_id: "rtc-raw-v1".to_string(),
            chunk_sha256_original: [0u8; 32],
            chunk_sha256_compressed: [0u8; 32],
            range_checksums: Vec::new(),
        };

        assert_eq!(
            runtime_f32_bytes_for_chunk(&chunk, &tensor).unwrap(),
            DEFAULT_TILE_STREAM_ELEMENTS * 2 * std::mem::size_of::<f32>()
        );
        assert_eq!(
            runtime_f32_tile_scratch_bytes_for_chunk(&chunk, &tensor).unwrap(),
            DEFAULT_TILE_STREAM_ELEMENTS * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn plan_marks_over_budget_when_peak_exceeds_limit() {
        let path = temp_path("over");
        write_test_model(&path);
        let model = LazyRllmModel::open(&path).unwrap();
        let plan = build_runtime_plan(
            &model,
            RuntimePlanConfig {
                mode: RuntimeMode::TileStream,
                context_length: 4,
                memory_budget_bytes: Some(1),
            },
        )
        .unwrap();

        assert!(matches!(plan.status, PlanStatus::OverBudget { .. }));

        std::fs::remove_file(&path).ok();
    }
}
