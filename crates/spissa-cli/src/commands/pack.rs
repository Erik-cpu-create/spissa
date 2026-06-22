// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use crate::commands::common::parse_size;
use crate::progress::{bar, print_pack_result, Spinner};
use anyhow::{Context, Result};
use std::time::Instant;
use spissa_container::{ChunkRangeSpec, DType, GlobalMetadata, SpissaWriter, TensorMeta};
use spissa_import::{
    read_model_config_metadata, read_tokenizer_metadata, SafetensorsReader,
    ShardedSafetensorsReader,
};
use rtc_codec::{
    BitplaneCodec, EncodeMeta, HuffmanCodec, RansCodec, RawCodec, RleCodec, TensorCodec,
    CODEC_BITPLANE_V1, CODEC_HUFF_V1, CODEC_RANS_V1, CODEC_RAW_V1, CODEC_RLE_V1,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackCodecPolicy {
    Auto,
    Raw,
    Rle,
    Huff,
    Rans,
    Bitplane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackQuantizePolicy {
    None,
    Raw,
    Q4_0,
    Q4_0KeepIo,
    Q4_0MlpOnly,
    Q4_0AttentionOnly,
    Q4AttentionQ8MlpKeepIo,
    Q8TransformerKeepIo,
}

fn sha256_array(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

impl PackCodecPolicy {
    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "raw" | CODEC_RAW_V1 => Ok(Self::Raw),
            "rle" | CODEC_RLE_V1 => Ok(Self::Rle),
            "huff" | "huffman" | CODEC_HUFF_V1 => Ok(Self::Huff),
            "rans" | CODEC_RANS_V1 => Ok(Self::Rans),
            "bitplane" | CODEC_BITPLANE_V1 => Ok(Self::Bitplane),
            other => anyhow::bail!(
                "unsupported --codec {other:?}; expected one of: auto, raw, rle, huff, rans, bitplane"
            ),
        }
    }

    fn metadata_label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Raw => CODEC_RAW_V1,
            Self::Rle => CODEC_RLE_V1,
            Self::Huff => CODEC_HUFF_V1,
            Self::Rans => CODEC_RANS_V1,
            Self::Bitplane => CODEC_BITPLANE_V1,
        }
    }

    fn codecs(self) -> Vec<Box<dyn TensorCodec>> {
        match self {
            Self::Auto => vec![
                Box::new(RawCodec),
                Box::new(RleCodec),
                Box::new(HuffmanCodec),
            ],
            Self::Raw => vec![Box::new(RawCodec)],
            Self::Rle => vec![Box::new(RleCodec)],
            Self::Huff => vec![Box::new(HuffmanCodec)],
            // rANS for the bf16 weights, raw fallback for non-bf16 chunks: the codec
            // emits FLAG_RAW internally, but also keep RawCodec in the try-list so the
            // smallest-lossless picker can choose raw when rANS doesn't help.
            Self::Rans => vec![Box::new(RansCodec), Box::new(RawCodec)],
            // bit-plane for bf16 (fast NEON decode), raw fallback for non-bf16 chunks.
            Self::Bitplane => vec![Box::new(BitplaneCodec), Box::new(RawCodec)],
        }
    }
}

impl PackQuantizePolicy {
    fn parse(raw: Option<&str>) -> Result<Self> {
        match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            None | Some("") => Ok(Self::None),
            Some("raw") => Ok(Self::Raw),
            Some("q4_0" | "q4-0") => Ok(Self::Q4_0),
            Some("q4_0_keep_io" | "q4-0-keep-io") => Ok(Self::Q4_0KeepIo),
            Some("q4_0_mlp_only" | "q4-0-mlp-only") => Ok(Self::Q4_0MlpOnly),
            Some("q4_0_attention_only" | "q4-0-attention-only") => Ok(Self::Q4_0AttentionOnly),
            Some("q4_attn_q8_mlp_keep_io" | "q4-attn-q8-mlp-keep-io") => {
                Ok(Self::Q4AttentionQ8MlpKeepIo)
            }
            Some("q8_transformer_keep_io" | "q8-transformer-keep-io") => {
                Ok(Self::Q8TransformerKeepIo)
            }
            Some(other) => {
                anyhow::bail!(
                    "unsupported --quantize {other:?}; expected one of: raw, q4_0, q4_0_keep_io, q4_0_mlp_only, q4_0_attention_only, q4_attn_q8_mlp_keep_io, q8_transformer_keep_io"
                )
            }
        }
    }

    fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q4_0KeepIo
                | Self::Q4_0MlpOnly
                | Self::Q4_0AttentionOnly
                | Self::Q4AttentionQ8MlpKeepIo
                | Self::Q8TransformerKeepIo
        )
    }

    fn allows_input_tile_sidecars(self) -> bool {
        !self.is_quantized()
    }

    #[cfg(test)]
    fn should_quantize_tensor(self, tensor_name: &str, shape: &[u64], dtype: DType) -> bool {
        self.quantized_dtype_for_tensor(tensor_name, shape, dtype)
            .is_some()
    }

    fn quantized_dtype_for_tensor(
        self,
        tensor_name: &str,
        shape: &[u64],
        dtype: DType,
    ) -> Option<DType> {
        if !self.is_quantized() || !is_quantizable_weight_tensor(tensor_name, shape, dtype) {
            return None;
        }
        if self == Self::Q4AttentionQ8MlpKeepIo {
            if is_llama_attention_projection_weight(tensor_name) {
                return Some(DType::Q4_0);
            }
            if is_llama_mlp_projection_weight(tensor_name) {
                return Some(DType::Q8_0);
            }
            return None;
        }
        if self == Self::Q8TransformerKeepIo {
            return (is_llama_attention_projection_weight(tensor_name)
                || is_llama_mlp_projection_weight(tensor_name)
                || is_qwen_linear_attn_projection_weight(tensor_name))
            .then_some(DType::Q8_0);
        }
        if self == Self::Q4_0MlpOnly {
            return is_llama_mlp_projection_weight(tensor_name).then_some(DType::Q4_0);
        }
        if self == Self::Q4_0AttentionOnly {
            return is_llama_attention_projection_weight(tensor_name).then_some(DType::Q4_0);
        }
        if self == Self::Q4_0KeepIo && is_llama_lm_head_weight(tensor_name) {
            return None;
        }
        Some(DType::Q4_0)
    }
}

fn is_quantizable_weight_tensor(tensor_name: &str, shape: &[u64], dtype: DType) -> bool {
    tensor_name.contains(".weight")
        && shape.len() >= 2
        && shape.iter().product::<u64>() >= 128
        && matches!(dtype, DType::Fp16 | DType::Bf16 | DType::Fp32)
}

/// A safetensors source: either a single file or a sharded checkpoint
/// (`*.index.json` + N shards). Presents one read surface to the pack loop.
enum TensorSource {
    Single(SafetensorsReader),
    Sharded(ShardedSafetensorsReader),
}

impl TensorSource {
    /// Open a single `.safetensors`, a `*.index.json`, or a directory that
    /// contains a `model.safetensors.index.json` (sharded) or `model.safetensors`.
    fn open(path: &Path) -> Result<Self> {
        let is_index = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".index.json"));
        if is_index {
            return Ok(Self::Sharded(ShardedSafetensorsReader::open_index(path)?));
        }
        if path.is_dir() {
            let index = path.join("model.safetensors.index.json");
            if index.exists() {
                return Ok(Self::Sharded(ShardedSafetensorsReader::open_index(index)?));
            }
            let single = path.join("model.safetensors");
            return Ok(Self::Single(SafetensorsReader::open(single)?));
        }
        Ok(Self::Single(SafetensorsReader::open(path)?))
    }

    fn list_tensors(&self) -> Vec<String> {
        match self {
            Self::Single(r) => r.list_tensors().into_iter().map(String::from).collect(),
            Self::Sharded(r) => r.list_tensors(),
        }
    }

    fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        match self {
            Self::Single(r) => Ok(r.read_tensor(name)?),
            Self::Sharded(r) => Ok(r.read_tensor(name)?),
        }
    }

    fn to_rllm_meta(&mut self, name: &str) -> Result<TensorMeta> {
        match self {
            Self::Single(r) => Ok(r.to_rllm_meta(name)?),
            Self::Sharded(r) => Ok(r.to_rllm_meta(name)?),
        }
    }
}

/// Map raw checkpoint tensor names to the names stored in the `.spsa`, returning
/// `(rllm_name, source_name)` pairs. For multimodal Gemma checkpoints
/// (`Gemma3ForConditionalGeneration`) this keeps only the language model
/// (`language_model.*`), strips that prefix so tensors match the standard
/// `model.layers.*` convention (so the existing quant matchers + runtime naming
/// apply), and drops the vision tower / multimodal projector. Other architectures
/// pass through unchanged.
fn map_tensor_names(raw: &[String], architecture: &str) -> Vec<(String, String)> {
    const PREFIX: &str = "language_model.";
    // Multimodal Gemma (`Gemma3ForConditionalGeneration`, e.g. 4B) wraps the LM
    // under `language_model.`; text-only Gemma (`Gemma3ForCausalLM`, e.g. 1B) uses
    // bare `model.*`. Only strip/filter the prefix when it is actually present —
    // otherwise a text-only Gemma checkpoint would be filtered down to zero tensors.
    if architecture.starts_with("gemma") && raw.iter().any(|t| t.starts_with(PREFIX)) {
        let mut mapped: Vec<(String, String)> = raw
            .iter()
            .filter(|t| t.starts_with(PREFIX))
            .map(|t| (t[PREFIX.len()..].to_string(), t.clone()))
            .collect();
        mapped.sort();
        mapped
    } else if architecture.starts_with("qwen") {
        // Qwen3.5 (`Qwen3_5ForConditionalGeneration`) nests the text decoder under
        // `model.language_model.*`. Keep only that, rewrite the prefix to the standard
        // `model.*` convention, and DROP the vision tower (`model.visual.*`) and the
        // multi-token-prediction head (`mtp.*`) — this is the text-only adapter.
        const QPREFIX: &str = "model.language_model.";
        let mut mapped: Vec<(String, String)> = raw
            .iter()
            .filter(|t| t.starts_with(QPREFIX))
            .map(|t| (format!("model.{}", &t[QPREFIX.len()..]), t.clone()))
            .collect();
        mapped.sort();
        mapped
    } else {
        raw.iter().map(|t| (t.clone(), t.clone())).collect()
    }
}

pub fn run(
    input: &str,
    output: &str,
    chunk_size: &str,
    codec_policy: &str,
    range_checksum_size: Option<&str>,
    tile_block_elements: Option<usize>,
    llama_mlp_input_tiles: bool,
    llama_attention_input_tiles: bool,
    llama_lm_head_input_tiles: bool,
    input_tile_features: usize,
    config: Option<&str>,
    tokenizer: Option<&str>,
    no_tokenizer: bool,
    quantize: Option<&str>,
    verbose: bool,
) -> Result<()> {
    let started = Instant::now();
    let chunk_size_bytes = parse_size(chunk_size)?;
    let codec_policy = PackCodecPolicy::parse(codec_policy)?;
    let quantize_policy = PackQuantizePolicy::parse(quantize)?;
    let range_checksum_size_bytes = match range_checksum_size.map(parse_size).transpose()? {
        Some(0) => anyhow::bail!("--range-checksum-size must be greater than zero"),
        other => other,
    };
    if matches!(tile_block_elements, Some(0)) {
        anyhow::bail!("--tile-block-elements must be greater than zero");
    }
    if input_tile_features == 0 {
        anyhow::bail!("--input-tile-features must be greater than zero");
    }
    let input_path = Path::new(input);

    if !input_path.exists() {
        anyhow::bail!("Input path does not exist: {}", input);
    }

    // `verbose` (global -v) keeps the old line-by-line log; otherwise a single live spinner
    // owns the terminal and we finish with the result box.
    let mut spinner = if verbose {
        println!("Reading safetensors from: {}", input);
        None
    } else {
        Some(Spinner::start("Reading model …"))
    };

    // Open the source (single file, a `*.index.json`, or a directory holding a
    // sharded `model.safetensors.index.json`).
    let mut source = TensorSource::open(input_path)
        .with_context(|| format!("Failed to open safetensors source: {}", input))?;
    let raw_names = source.list_tensors();

    // Create metadata
    let model_name = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let config_path = resolve_model_config_path(input_path, config);
    let model_config = match config_path {
        Some(path) => Some(
            read_model_config_metadata(&path)
                .with_context(|| format!("Failed to read model config: {}", path.display()))?,
        ),
        None => None,
    };
    let architecture = model_config
        .as_ref()
        .and_then(|config| config.architecture_type.clone())
        .unwrap_or_else(|| "unknown".to_string());
    // Map raw checkpoint names to `.spsa` names (filters vision + strips the
    // `language_model.` prefix for multimodal Gemma; identity otherwise).
    let tensors = map_tensor_names(&raw_names, &architecture);
    if verbose {
        println!(
            "Found {} tensors ({} packed for architecture '{}')",
            raw_names.len(),
            tensors.len(),
            architecture
        );
    }
    let default_context_length = model_config
        .as_ref()
        .and_then(|config| config.max_position_embeddings)
        .unwrap_or(0);
    let tokenizer_path = resolve_tokenizer_path(input_path, tokenizer, no_tokenizer);
    let tokenizer = match tokenizer_path {
        Some(path) => Some(
            read_tokenizer_metadata(&path)
                .with_context(|| format!("Failed to read tokenizer: {}", path.display()))?,
        ),
        None => None,
    };
    let tokenizer_type = tokenizer
        .as_ref()
        .and_then(|tokenizer| tokenizer.tokenizer_type.clone())
        .unwrap_or_else(|| "none".to_string());

    let metadata = GlobalMetadata {
        model_name: model_name.clone(),
        architecture,
        source_format: "safetensors".to_string(),
        lossless: !quantize_policy.is_quantized(),
        default_context_length,
        tokenizer_type,
        created_by: "spissa-cli".to_string(),
        codec: codec_policy.metadata_label().to_string(),
        model_config,
        tokenizer,
    };

    let mut writer = SpissaWriter::new(output, metadata)?;

    // Codecs to try. `auto` preserves the original smallest-lossless behavior;
    // forced policies create runtime-layout artifacts for measured RAMA trade-offs.
    let codecs = codec_policy.codecs();

    let mut total_original = 0;
    let mut total_compressed = 0;
    let mut chunk_count = 0;
    let mut range_checksum_count = 0usize;
    let mut range_checksum_skipped_chunks = 0usize;
    let mut tile_block_aligned_tensors = 0usize;
    let mut input_tile_sidecar_tensors = 0usize;
    let mut input_tile_sidecar_ranges = 0usize;
    let mut input_tile_sidecar_chunks = 0usize;
    let mut next_tensor_id = 0u64;

    // Process each tensor
    let total_tensors = tensors.len();
    for (tensor_idx, (tensor_name, src_name)) in tensors.iter().enumerate() {
        if verbose {
            println!(
                "Processing tensor: {} ({}/{})",
                tensor_name,
                tensor_idx + 1,
                total_tensors
            );
        } else if let Some(sp) = &spinner {
            let frac = tensor_idx as f64 / total_tensors.max(1) as f64;
            sp.set(format!(
                "Packing  {}  {}/{} tensors",
                bar(frac, 22),
                tensor_idx + 1,
                total_tensors
            ));
        }

        // Read from the source (original) name; store under the `.spsa` (mapped)
        // name so quant matchers + the runtime see standard `model.layers.*`.
        let mut tensor_data = source.read_tensor(src_name)?;
        let mut meta = source.to_rllm_meta(src_name)?;
        meta.name = tensor_name.clone();
        let tensor_id = next_tensor_id;
        next_tensor_id = next_tensor_id.saturating_add(1);
        meta.tensor_id = tensor_id;

        // Apply Q4_0 quantization if requested and applicable
        let quantized_dtype =
            quantize_policy.quantized_dtype_for_tensor(tensor_name, &meta.shape, meta.dtype);

        if let Some(target_dtype) = quantized_dtype {
            if verbose {
                println!("  Quantizing {} to {:?}...", tensor_name, target_dtype);
            }
            let quantized = match target_dtype {
                spissa_container::DType::Q4_0 => {
                    quantize_to_q4_0(&tensor_data, meta.dtype, &meta.shape)?
                }
                spissa_container::DType::Q8_0 => {
                    quantize_to_q8_0(&tensor_data, meta.dtype, &meta.shape)?
                }
                other => anyhow::bail!("unsupported quantized target dtype {:?}", other),
            };
            meta.dtype = target_dtype;
            meta.original_size_bytes = quantized.len() as u64;
            meta.original_sha256 = sha256_array(&quantized);
            tensor_data = quantized;
        }

        let effective_chunk_size_bytes = effective_chunk_size_bytes_for_tensor(
            meta.dtype,
            &meta.shape,
            chunk_size_bytes,
            tile_block_elements,
            tensor_name,
        )?;
        if effective_chunk_size_bytes == 0 {
            anyhow::bail!("effective chunk size for tensor {} is zero", tensor_name);
        }
        if tile_block_elements.is_some() {
            tile_block_aligned_tensors += 1;
        }

        total_original += meta.original_size_bytes;

        // Add tensor to writer
        writer.add_tensor(meta.clone());

        // Encode chunks
        let encode_meta = EncodeMeta {
            name: tensor_name.to_string(),
            shape: vec![tensor_data.len() as u64],
            dtype: "u8".to_string(),
        };

        for (i, chunk) in tensor_data.chunks(effective_chunk_size_bytes).enumerate() {
            let mut best_encoded = None;
            let mut best_size = usize::MAX;

            for codec in &codecs {
                let encoded = codec.encode(chunk, &encode_meta)?;

                if !codec.verify_roundtrip(chunk, &encode_meta)? {
                    continue;
                }

                if encoded.data.len() < best_size {
                    best_size = encoded.data.len();
                    best_encoded = Some((encoded, codec.id().to_string()));
                }
            }

            let (encoded, codec_id) = best_encoded.ok_or_else(|| {
                anyhow::anyhow!(
                    "No codec succeeded for chunk {} of tensor {}",
                    i,
                    tensor_name
                )
            })?;

            if let Some(range_size) = range_checksum_size_bytes {
                if codec_id == CODEC_RAW_V1 && encoded.data.len() == chunk.len() {
                    writer.write_chunk_with_identity_range_checksums(
                        tensor_id,
                        &codec_id,
                        &encoded.data,
                        chunk,
                        i as u64,
                        range_size as u64,
                    )?;
                    range_checksum_count += chunk.len().div_ceil(range_size);
                } else {
                    writer.write_chunk(tensor_id, &codec_id, &encoded.data, chunk, i as u64)?;
                    range_checksum_skipped_chunks += 1;
                }
            } else {
                writer.write_chunk(tensor_id, &codec_id, &encoded.data, chunk, i as u64)?;
            }

            chunk_count += 1;
            total_compressed += encoded.data.len();
        }

        let should_write_input_tile_sidecar = quantize_policy.allows_input_tile_sidecars()
            && ((llama_mlp_input_tiles && is_llama_mlp_projection_weight(tensor_name))
                || (llama_attention_input_tiles
                    && is_llama_attention_projection_weight(tensor_name))
                || (llama_lm_head_input_tiles && is_llama_lm_head_weight(tensor_name)));
        if should_write_input_tile_sidecar {
            let sidecar_tensor_id = next_tensor_id;
            next_tensor_id = next_tensor_id.saturating_add(1);
            let sidecar_stats = write_input_tile_sidecar_tensor(
                &mut writer,
                sidecar_tensor_id,
                tensor_name,
                &meta,
                &tensor_data,
                input_tile_features,
            )
            .with_context(|| format!("failed to write input-tile sidecar for {tensor_name}"))?;
            input_tile_sidecar_tensors += 1;
            input_tile_sidecar_ranges += sidecar_stats.range_count;
            input_tile_sidecar_chunks += sidecar_stats.chunk_count;
            chunk_count += sidecar_stats.chunk_count;
            total_original += sidecar_stats.original_bytes as u64;
            total_compressed += sidecar_stats.compressed_bytes;
        }
    }

    if let Some(sp) = &spinner {
        sp.set("Finalizing container …");
    }

    if verbose {
        println!("\nEncoded {} chunks total", chunk_count);
        println!("Codec policy: {}", codec_policy.metadata_label());
        if range_checksum_size_bytes.is_some() {
            println!("Range checksums emitted: {}", range_checksum_count);
            if range_checksum_skipped_chunks > 0 {
                println!(
                    "Range checksums skipped for {} non-identity compressed chunks",
                    range_checksum_skipped_chunks
                );
            }
        }
        if let Some(tile_elements) = tile_block_elements {
            println!(
                "Tile-block packing: {} tensor(s), {} element(s) per chunk/block",
                tile_block_aligned_tensors, tile_elements
            );
        }
        if quantize_policy.allows_input_tile_sidecars()
            && (llama_mlp_input_tiles || llama_attention_input_tiles || llama_lm_head_input_tiles)
        {
            println!(
                "Input-tile sidecars: {} tensor(s), {} chunk(s), {} feature range(s), {} feature(s) per chunk",
                input_tile_sidecar_tensors,
                input_tile_sidecar_chunks,
                input_tile_sidecar_ranges,
                input_tile_features
            );
        }
        println!("Original size: {} bytes", total_original);
        println!("Compressed size: {} bytes", total_compressed);
        if total_original > 0 {
            let ratio = total_compressed as f64 / total_original as f64 * 100.0;
            println!("Compression ratio: {:.1}%", ratio);
        }
    }

    writer.finalize()?;

    if verbose {
        println!("Written to: {}", output);
    } else {
        if let Some(sp) = spinner.take() {
            sp.clear();
        }
        let filename = std::path::Path::new(output)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(output);
        print_pack_result(
            filename,
            &codec_policy.metadata_label(),
            total_tensors,
            chunk_count,
            total_original,
            total_compressed as u64,
            output,
            started.elapsed().as_secs_f64(),
        );
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
struct InputTileSidecarWriteStats {
    original_bytes: usize,
    compressed_bytes: usize,
    chunk_count: usize,
    range_count: usize,
}

fn is_llama_mlp_projection_weight(tensor_name: &str) -> bool {
    tensor_name.starts_with("model.layers.")
        && tensor_name.contains(".mlp.")
        && (tensor_name.ends_with(".gate_proj.weight")
            || tensor_name.ends_with(".up_proj.weight")
            || tensor_name.ends_with(".down_proj.weight"))
}

fn is_llama_attention_projection_weight(tensor_name: &str) -> bool {
    tensor_name.starts_with("model.layers.")
        && tensor_name.contains(".self_attn.")
        && (tensor_name.ends_with(".q_proj.weight")
            || tensor_name.ends_with(".k_proj.weight")
            || tensor_name.ends_with(".v_proj.weight")
            || tensor_name.ends_with(".o_proj.weight"))
}

fn is_llama_lm_head_weight(tensor_name: &str) -> bool {
    tensor_name == "lm_head.weight" || tensor_name == "model.embed_tokens.weight"
}

/// Qwen3.5 Gated-DeltaNet large 2-D projections (the per-token-streamed weights that
/// dominate decode cost). The small `in_proj_a`/`in_proj_b` (per-head decay/beta) and
/// the depthwise `conv1d` are intentionally left bf16 — they feed the sensitive
/// recurrence and are tiny.
fn is_qwen_linear_attn_projection_weight(tensor_name: &str) -> bool {
    tensor_name.starts_with("model.layers.")
        && tensor_name.contains(".linear_attn.")
        && (tensor_name.ends_with(".in_proj_qkv.weight")
            || tensor_name.ends_with(".in_proj_z.weight")
            || tensor_name.ends_with(".out_proj.weight"))
}

fn input_tile_sidecar_bytes(
    row_major: &[u8],
    out_features: usize,
    in_features: usize,
    dtype_size: usize,
) -> Result<Vec<u8>> {
    if dtype_size == 0 || !row_major.len().is_multiple_of(dtype_size) {
        anyhow::bail!(
            "input-tile sidecar source byte len {} is not aligned to dtype size {}",
            row_major.len(),
            dtype_size
        );
    }
    let expected = out_features
        .checked_mul(in_features)
        .and_then(|elements| elements.checked_mul(dtype_size))
        .ok_or_else(|| anyhow::anyhow!("input-tile sidecar source byte size overflow"))?;
    if row_major.len() != expected {
        anyhow::bail!(
            "input-tile sidecar source byte len {} does not match expected {}",
            row_major.len(),
            expected
        );
    }

    let mut input_major = vec![0u8; row_major.len()];
    for in_feature in 0..in_features {
        for out_feature in 0..out_features {
            let source = (out_feature * in_features + in_feature) * dtype_size;
            let dest = (in_feature * out_features + out_feature) * dtype_size;
            input_major[dest..dest + dtype_size]
                .copy_from_slice(&row_major[source..source + dtype_size]);
        }
    }
    Ok(input_major)
}

fn input_tile_range_specs(
    feature_count: usize,
    out_features: usize,
    dtype_size: usize,
) -> Vec<ChunkRangeSpec> {
    let column_bytes = (out_features * dtype_size) as u64;
    (0..feature_count)
        .map(|feature_offset| {
            let offset = feature_offset as u64 * column_bytes;
            ChunkRangeSpec {
                original_offset: offset,
                original_size: column_bytes,
                compressed_offset: offset,
                compressed_size: column_bytes,
            }
        })
        .collect()
}

fn effective_chunk_size_bytes_for_tensor(
    dtype: spissa_container::DType,
    shape: &[u64],
    chunk_size_bytes: usize,
    tile_block_elements: Option<usize>,
    tensor_name: &str,
) -> Result<usize> {
    if let Some(block_bytes) = quantized_block_bytes(dtype) {
        let block_aligned = (chunk_size_bytes / block_bytes) * block_bytes;
        if let Some(row_bytes) = quantized_row_bytes(dtype, shape) {
            if row_bytes <= chunk_size_bytes {
                return Ok((chunk_size_bytes / row_bytes) * row_bytes);
            }
        }
        return Ok(block_aligned);
    }

    if let Some(tile_elements) = tile_block_elements {
        let dtype_size = dtype.size_bytes();
        return tile_elements.checked_mul(dtype_size).ok_or_else(|| {
            anyhow::anyhow!(
                "--tile-block-elements overflow for tensor {} with dtype size {}",
                tensor_name,
                dtype_size
            )
        });
    }

    Ok(chunk_size_bytes)
}

fn quantized_block_bytes(dtype: spissa_container::DType) -> Option<usize> {
    match dtype {
        spissa_container::DType::Q4_0 => Some(18),
        spissa_container::DType::Q8_0 => Some(34),
        _ => None,
    }
}

fn quantized_row_bytes(dtype: spissa_container::DType, shape: &[u64]) -> Option<usize> {
    if shape.len() != 2 {
        return None;
    }
    let in_features = usize::try_from(shape[1]).ok()?;
    if in_features == 0 || !in_features.is_multiple_of(32) {
        return None;
    }
    let block_bytes = quantized_block_bytes(dtype)?;
    Some((in_features / 32) * block_bytes)
}

fn write_input_tile_sidecar_tensor(
    writer: &mut SpissaWriter,
    tensor_id: u64,
    source_name: &str,
    source_meta: &TensorMeta,
    source_bytes: &[u8],
    input_tile_features: usize,
) -> Result<InputTileSidecarWriteStats> {
    if source_meta.shape.len() != 2 {
        anyhow::bail!(
            "input-tile sidecar source tensor {} must be rank-2, got {:?}",
            source_name,
            source_meta.shape
        );
    }
    if !matches!(source_meta.dtype, DType::Fp16 | DType::Bf16) {
        anyhow::bail!(
            "input-tile sidecar source tensor {} must be FP16/BF16, got {:?}",
            source_name,
            source_meta.dtype
        );
    }
    let out_features = usize::try_from(source_meta.shape[0])
        .map_err(|_| anyhow::anyhow!("input-tile out_features overflow"))?;
    let in_features = usize::try_from(source_meta.shape[1])
        .map_err(|_| anyhow::anyhow!("input-tile in_features overflow"))?;
    let dtype_size = source_meta.dtype.size_bytes();
    let sidecar_bytes =
        input_tile_sidecar_bytes(source_bytes, out_features, in_features, dtype_size)?;
    let sidecar_name = spissa_runtime::input_tile_sidecar_weight_name(source_name);

    writer.add_tensor(TensorMeta {
        tensor_id,
        name: sidecar_name,
        shape: vec![in_features as u64, out_features as u64],
        dtype: source_meta.dtype,
        original_size_bytes: sidecar_bytes.len() as u64,
        compressed_size_bytes: sidecar_bytes.len() as u64,
        original_sha256: sha256_array(&sidecar_bytes),
        chunk_count: 0,
        chunk_start_index: 0,
    });

    let features_per_chunk = input_tile_features.max(1);
    let mut stats = InputTileSidecarWriteStats {
        original_bytes: sidecar_bytes.len(),
        compressed_bytes: 0,
        chunk_count: 0,
        range_count: 0,
    };
    for feature_start in (0..in_features).step_by(features_per_chunk) {
        let feature_end = (feature_start + features_per_chunk).min(in_features);
        let feature_count = feature_end - feature_start;
        let byte_start = feature_start
            .checked_mul(out_features)
            .and_then(|elements| elements.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("input-tile chunk byte start overflow"))?;
        let byte_end = feature_end
            .checked_mul(out_features)
            .and_then(|elements| elements.checked_mul(dtype_size))
            .ok_or_else(|| anyhow::anyhow!("input-tile chunk byte end overflow"))?;
        let chunk = &sidecar_bytes[byte_start..byte_end];
        let ranges = input_tile_range_specs(feature_count, out_features, dtype_size);
        writer.write_chunk_with_range_specs(
            tensor_id,
            CODEC_RAW_V1,
            chunk,
            chunk,
            (feature_start * out_features) as u64,
            &ranges,
        )?;
        stats.compressed_bytes += chunk.len();
        stats.chunk_count += 1;
        stats.range_count += ranges.len();
    }
    Ok(stats)
}

/// Directory holding the checkpoint's sidecar files: for a directory input that's
/// the directory itself; for a file/index input it's the file's parent. Lets pack
/// auto-resolve `config.json`/`tokenizer.json` whether the input is a folder or a file.
fn checkpoint_sidecar_dir(input_path: &Path) -> Option<PathBuf> {
    if input_path.is_dir() {
        Some(input_path.to_path_buf())
    } else {
        input_path.parent().map(Path::to_path_buf)
    }
}

fn resolve_model_config_path(input_path: &Path, explicit_config: Option<&str>) -> Option<PathBuf> {
    if let Some(config) = explicit_config {
        return Some(PathBuf::from(config));
    }
    let sibling = checkpoint_sidecar_dir(input_path)?.join("config.json");
    sibling.exists().then_some(sibling)
}

fn resolve_tokenizer_path(
    input_path: &Path,
    explicit_tokenizer: Option<&str>,
    no_tokenizer: bool,
) -> Option<PathBuf> {
    if no_tokenizer {
        return None;
    }
    if let Some(tokenizer) = explicit_tokenizer {
        return Some(PathBuf::from(tokenizer));
    }
    let sibling = checkpoint_sidecar_dir(input_path)?.join("tokenizer.json");
    sibling.exists().then_some(sibling)
}

fn quantize_to_q4_0(
    raw_data: &[u8],
    dtype: spissa_container::DType,
    shape: &[u64],
) -> Result<Vec<u8>> {
    spissa_runtime::quantize_to_q4_0(raw_data, dtype, shape).map_err(|e| anyhow::anyhow!(e))
}

fn quantize_to_q8_0(
    raw_data: &[u8],
    dtype: spissa_container::DType,
    shape: &[u64],
) -> Result<Vec<u8>> {
    spissa_runtime::quantize_to_q8_0(raw_data, dtype, shape).map_err(|e| anyhow::anyhow!(e))
}

#[cfg(test)]
mod tests {
    use super::{
        effective_chunk_size_bytes_for_tensor, input_tile_range_specs, input_tile_sidecar_bytes,
        is_llama_attention_projection_weight, is_llama_lm_head_weight,
        is_llama_mlp_projection_weight, map_tensor_names, PackCodecPolicy, PackQuantizePolicy,
    };

    #[test]
    fn map_tensor_names_keeps_text_only_gemma() {
        // Text-only Gemma (Gemma3ForCausalLM, e.g. 1B): bare `model.*` names, no
        // `language_model.` prefix -> identity passthrough (regression: was filtered
        // to zero tensors).
        let raw = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.norm.weight".to_string(),
        ];
        let mapped = map_tensor_names(&raw, "gemma3");
        assert_eq!(mapped.len(), 3, "text-only gemma must keep all tensors");
        assert!(mapped.iter().all(|(dst, src)| dst == src));
    }

    #[test]
    fn map_tensor_names_strips_multimodal_gemma_prefix() {
        // Multimodal Gemma (Gemma3ForConditionalGeneration, e.g. 4B): strip the
        // `language_model.` prefix and drop the vision tower.
        let raw = vec![
            "language_model.model.layers.0.self_attn.q_proj.weight".to_string(),
            "vision_tower.encoder.layer.0.weight".to_string(),
        ];
        let mapped = map_tensor_names(&raw, "gemma3");
        assert_eq!(mapped.len(), 1, "vision tower dropped, LM kept");
        assert_eq!(mapped[0].0, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped[0].1, "language_model.model.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn map_tensor_names_qwen_text_only_strips_prefix_drops_vision_and_mtp() {
        // Qwen3.5 (Qwen3_5ForConditionalGeneration): the text decoder lives under
        // `model.language_model.*`. Keep only that, rewrite to the canonical `model.*`
        // convention, and drop the vision tower (`model.visual.*`) and MTP head (`mtp.*`).
        let raw = vec![
            "model.language_model.embed_tokens.weight".to_string(),
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            "model.language_model.layers.3.self_attn.q_proj.weight".to_string(),
            "model.language_model.norm.weight".to_string(),
            "model.visual.blocks.0.attn.qkv.weight".to_string(),
            "mtp.layers.0.self_attn.q_proj.weight".to_string(),
        ];
        let mapped = map_tensor_names(&raw, "qwen3");
        assert_eq!(mapped.len(), 4, "vision + mtp dropped, text decoder kept");
        assert!(
            mapped.iter().all(|(dst, _)| dst.starts_with("model.")
                && !dst.contains("language_model")
                && !dst.contains("visual")
                && !dst.contains("mtp")),
            "names canonicalized to model.* with no language_model/visual/mtp"
        );
        let qkv = mapped
            .iter()
            .find(|(_, src)| src.ends_with("layers.0.linear_attn.in_proj_qkv.weight"))
            .expect("linear_attn tensor kept");
        assert_eq!(qkv.0, "model.layers.0.linear_attn.in_proj_qkv.weight");
        assert_eq!(qkv.1, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
    }

    #[test]
    fn pack_codec_policy_accepts_short_and_codec_ids() {
        assert_eq!(
            PackCodecPolicy::parse("auto").unwrap(),
            PackCodecPolicy::Auto
        );
        assert_eq!(PackCodecPolicy::parse("raw").unwrap(), PackCodecPolicy::Raw);
        assert_eq!(
            PackCodecPolicy::parse("rtc-raw-v1").unwrap(),
            PackCodecPolicy::Raw
        );
        assert_eq!(PackCodecPolicy::parse("rle").unwrap(), PackCodecPolicy::Rle);
        assert_eq!(
            PackCodecPolicy::parse("huffman").unwrap(),
            PackCodecPolicy::Huff
        );
        assert_eq!(
            PackCodecPolicy::parse("rtc-huff-v1").unwrap(),
            PackCodecPolicy::Huff
        );
    }

    #[test]
    fn pack_codec_policy_rejects_unknown_values() {
        assert!(PackCodecPolicy::parse("zstd").is_err());
    }

    #[test]
    fn pack_quantize_policy_accepts_raw_and_q4_0_aliases() {
        assert_eq!(
            PackQuantizePolicy::parse(None).unwrap(),
            PackQuantizePolicy::None
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("")).unwrap(),
            PackQuantizePolicy::None
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("raw")).unwrap(),
            PackQuantizePolicy::Raw
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q4_0")).unwrap(),
            PackQuantizePolicy::Q4_0
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q4-0")).unwrap(),
            PackQuantizePolicy::Q4_0
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q4_0_keep_io")).unwrap(),
            PackQuantizePolicy::Q4_0KeepIo
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q4-0-keep-io")).unwrap(),
            PackQuantizePolicy::Q4_0KeepIo
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q4_0_mlp_only")).unwrap(),
            PackQuantizePolicy::Q4_0MlpOnly
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q4-0-mlp-only")).unwrap(),
            PackQuantizePolicy::Q4_0MlpOnly
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q4_0_attention_only")).unwrap(),
            PackQuantizePolicy::Q4_0AttentionOnly
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q4-0-attention-only")).unwrap(),
            PackQuantizePolicy::Q4_0AttentionOnly
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q4_attn_q8_mlp_keep_io")).unwrap(),
            PackQuantizePolicy::Q4AttentionQ8MlpKeepIo
        );
        assert_eq!(
            PackQuantizePolicy::parse(Some("q8_transformer_keep_io")).unwrap(),
            PackQuantizePolicy::Q8TransformerKeepIo
        );
    }

    #[test]
    fn pack_quantize_policy_rejects_unknown_values() {
        assert!(PackQuantizePolicy::parse(Some("int4")).is_err());
    }

    #[test]
    fn q4_0_keep_io_preserves_embedding_and_lm_head_weights() {
        assert!(PackQuantizePolicy::Q4_0.should_quantize_tensor(
            "model.embed_tokens.weight",
            &[49152, 576],
            spissa_container::DType::Bf16
        ));
        assert!(PackQuantizePolicy::Q4_0.should_quantize_tensor(
            "lm_head.weight",
            &[49152, 576],
            spissa_container::DType::Bf16
        ));
        assert!(!PackQuantizePolicy::Q4_0KeepIo.should_quantize_tensor(
            "model.embed_tokens.weight",
            &[49152, 576],
            spissa_container::DType::Bf16
        ));
        assert!(!PackQuantizePolicy::Q4_0KeepIo.should_quantize_tensor(
            "lm_head.weight",
            &[49152, 576],
            spissa_container::DType::Bf16
        ));
        assert!(PackQuantizePolicy::Q4_0KeepIo.should_quantize_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &[1536, 576],
            spissa_container::DType::Bf16
        ));
    }

    #[test]
    fn q4_0_mlp_only_quantizes_mlp_and_preserves_attention_and_io_weights() {
        assert!(!PackQuantizePolicy::Q4_0MlpOnly.should_quantize_tensor(
            "model.embed_tokens.weight",
            &[49152, 576],
            spissa_container::DType::Bf16
        ));
        assert!(!PackQuantizePolicy::Q4_0MlpOnly.should_quantize_tensor(
            "lm_head.weight",
            &[49152, 576],
            spissa_container::DType::Bf16
        ));
        assert!(!PackQuantizePolicy::Q4_0MlpOnly.should_quantize_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &[576, 576],
            spissa_container::DType::Bf16
        ));
        assert!(PackQuantizePolicy::Q4_0MlpOnly.should_quantize_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            &[1536, 576],
            spissa_container::DType::Bf16
        ));
        assert!(PackQuantizePolicy::Q4_0MlpOnly.should_quantize_tensor(
            "model.layers.0.mlp.up_proj.weight",
            &[1536, 576],
            spissa_container::DType::Bf16
        ));
        assert!(PackQuantizePolicy::Q4_0MlpOnly.should_quantize_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &[576, 1536],
            spissa_container::DType::Bf16
        ));
    }

    #[test]
    fn q4_0_attention_only_quantizes_attention_and_preserves_mlp_and_io_weights() {
        assert!(
            !PackQuantizePolicy::Q4_0AttentionOnly.should_quantize_tensor(
                "model.embed_tokens.weight",
                &[49152, 576],
                spissa_container::DType::Bf16
            )
        );
        assert!(
            !PackQuantizePolicy::Q4_0AttentionOnly.should_quantize_tensor(
                "model.layers.0.mlp.gate_proj.weight",
                &[1536, 576],
                spissa_container::DType::Bf16
            )
        );
        assert!(
            PackQuantizePolicy::Q4_0AttentionOnly.should_quantize_tensor(
                "model.layers.0.self_attn.q_proj.weight",
                &[576, 576],
                spissa_container::DType::Bf16
            )
        );
        assert!(
            PackQuantizePolicy::Q4_0AttentionOnly.should_quantize_tensor(
                "model.layers.0.self_attn.k_proj.weight",
                &[192, 576],
                spissa_container::DType::Bf16
            )
        );
        assert!(
            PackQuantizePolicy::Q4_0AttentionOnly.should_quantize_tensor(
                "model.layers.0.self_attn.v_proj.weight",
                &[192, 576],
                spissa_container::DType::Bf16
            )
        );
        assert!(
            PackQuantizePolicy::Q4_0AttentionOnly.should_quantize_tensor(
                "model.layers.0.self_attn.o_proj.weight",
                &[576, 576],
                spissa_container::DType::Bf16
            )
        );
    }

    #[test]
    fn q4_attention_q8_mlp_keep_io_uses_mixed_quantization() {
        assert_eq!(
            PackQuantizePolicy::Q4AttentionQ8MlpKeepIo.quantized_dtype_for_tensor(
                "model.embed_tokens.weight",
                &[49152, 576],
                spissa_container::DType::Bf16
            ),
            None
        );
        assert_eq!(
            PackQuantizePolicy::Q4AttentionQ8MlpKeepIo.quantized_dtype_for_tensor(
                "model.layers.0.self_attn.q_proj.weight",
                &[576, 576],
                spissa_container::DType::Bf16
            ),
            Some(spissa_container::DType::Q4_0)
        );
        assert_eq!(
            PackQuantizePolicy::Q4AttentionQ8MlpKeepIo.quantized_dtype_for_tensor(
                "model.layers.0.mlp.gate_proj.weight",
                &[1536, 576],
                spissa_container::DType::Bf16
            ),
            Some(spissa_container::DType::Q8_0)
        );
    }

    #[test]
    fn q8_transformer_keep_io_quantizes_attention_and_mlp_to_q8() {
        assert_eq!(
            PackQuantizePolicy::Q8TransformerKeepIo.quantized_dtype_for_tensor(
                "model.embed_tokens.weight",
                &[49152, 576],
                spissa_container::DType::Bf16
            ),
            None
        );
        assert_eq!(
            PackQuantizePolicy::Q8TransformerKeepIo.quantized_dtype_for_tensor(
                "model.layers.0.self_attn.q_proj.weight",
                &[576, 576],
                spissa_container::DType::Bf16
            ),
            Some(spissa_container::DType::Q8_0)
        );
        assert_eq!(
            PackQuantizePolicy::Q8TransformerKeepIo.quantized_dtype_for_tensor(
                "model.layers.0.mlp.down_proj.weight",
                &[576, 1536],
                spissa_container::DType::Bf16
            ),
            Some(spissa_container::DType::Q8_0)
        );
    }

    #[test]
    fn quantized_2d_chunk_size_aligns_to_row_bytes_for_q8_fast_path() {
        let chunk_size = effective_chunk_size_bytes_for_tensor(
            spissa_container::DType::Q8_0,
            &[8192, 2048],
            1_048_576,
            None,
            "model.layers.0.mlp.gate_proj.weight",
        )
        .unwrap();

        let q8_row_bytes = (2048 / 32) * 34;
        assert_eq!(q8_row_bytes, 2176);
        assert_eq!(chunk_size % q8_row_bytes, 0);
        assert_eq!(chunk_size, 1_046_656);
    }

    #[test]
    fn llama_mlp_projection_selector_is_scoped_to_weight_tensors() {
        assert!(is_llama_mlp_projection_weight(
            "model.layers.0.mlp.gate_proj.weight"
        ));
        assert!(is_llama_mlp_projection_weight(
            "model.layers.15.mlp.down_proj.weight"
        ));
        assert!(!is_llama_mlp_projection_weight(
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!is_llama_mlp_projection_weight(
            "model.layers.0.mlp.gate_proj.bias"
        ));
    }

    #[test]
    fn llama_attention_projection_selector_is_scoped_to_weight_tensors() {
        assert!(is_llama_attention_projection_weight(
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(is_llama_attention_projection_weight(
            "model.layers.15.self_attn.o_proj.weight"
        ));
        assert!(!is_llama_attention_projection_weight(
            "model.layers.0.mlp.gate_proj.weight"
        ));
        assert!(!is_llama_attention_projection_weight(
            "model.layers.0.self_attn.q_proj.bias"
        ));
    }

    #[test]
    fn llama_lm_head_selector_is_scoped_to_weight_tensor() {
        assert!(is_llama_lm_head_weight("lm_head.weight"));
        assert!(is_llama_lm_head_weight("model.embed_tokens.weight"));
        assert!(!is_llama_lm_head_weight("lm_head.bias"));
    }

    #[test]
    fn input_tile_sidecar_bytes_transpose_row_major_16bit_matrix() {
        let row_major: Vec<u8> = [1u16, 2, 3, 4, 5, 6]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let sidecar = input_tile_sidecar_bytes(&row_major, 2, 3, 2).unwrap();
        let values: Vec<u16> = sidecar
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();

        assert_eq!(values, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn input_tile_range_specs_cover_each_feature_column_inside_chunk() {
        let ranges = input_tile_range_specs(2, 4, 2);

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].original_offset, 0);
        assert_eq!(ranges[0].original_size, 8);
        assert_eq!(ranges[1].original_offset, 8);
        assert_eq!(ranges[1].original_size, 8);
    }
}
