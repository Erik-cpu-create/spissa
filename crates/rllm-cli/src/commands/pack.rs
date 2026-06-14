use crate::commands::common::parse_size;
use anyhow::{Context, Result};
use rllm_container::{ChunkRangeSpec, DType, GlobalMetadata, RllmWriter, TensorMeta};
use rllm_import::{read_model_config_metadata, read_tokenizer_metadata, SafetensorsReader};
use rtc_codec::{
    EncodeMeta, HuffmanCodec, RawCodec, RleCodec, TensorCodec, CODEC_HUFF_V1, CODEC_RAW_V1,
    CODEC_RLE_V1,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackCodecPolicy {
    Auto,
    Raw,
    Rle,
    Huff,
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
            other => anyhow::bail!(
                "unsupported --codec {other:?}; expected one of: auto, raw, rle, huff"
            ),
        }
    }

    fn metadata_label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Raw => CODEC_RAW_V1,
            Self::Rle => CODEC_RLE_V1,
            Self::Huff => CODEC_HUFF_V1,
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
        }
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
) -> Result<()> {
    let chunk_size_bytes = parse_size(chunk_size)?;
    let codec_policy = PackCodecPolicy::parse(codec_policy)?;
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

    println!("Reading safetensors from: {}", input);

    // Open safetensors file
    let mut reader = SafetensorsReader::open(input_path)
        .with_context(|| format!("Failed to open safetensors file: {}", input))?;

    let tensor_names: Vec<String> = reader
        .list_tensors()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    println!("Found {} tensors", tensor_names.len());

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
        lossless: true,
        default_context_length,
        tokenizer_type,
        created_by: "rllm-cli".to_string(),
        codec: codec_policy.metadata_label().to_string(),
        model_config,
        tokenizer,
    };

    let mut writer = RllmWriter::new(output, metadata)?;

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
    for (tensor_idx, tensor_name) in tensor_names.iter().enumerate() {
        println!(
            "Processing tensor: {} ({}/{})",
            tensor_name,
            tensor_idx + 1,
            tensor_names.len()
        );

        // Read tensor data
        let tensor_data = reader.read_tensor(tensor_name)?;
        let tensor_meta = reader.to_rllm_meta(tensor_name)?;

        // Update tensor ID
        let mut meta = tensor_meta;
        let tensor_id = next_tensor_id;
        next_tensor_id = next_tensor_id.saturating_add(1);
        meta.tensor_id = tensor_id;
        let effective_chunk_size_bytes = if let Some(tile_elements) = tile_block_elements {
            let dtype_size = meta.dtype.size_bytes();
            tile_elements.checked_mul(dtype_size).ok_or_else(|| {
                anyhow::anyhow!(
                    "--tile-block-elements overflow for tensor {} with dtype size {}",
                    tensor_name,
                    dtype_size
                )
            })?
        } else {
            chunk_size_bytes
        };
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

        let should_write_input_tile_sidecar = (llama_mlp_input_tiles
            && is_llama_mlp_projection_weight(tensor_name))
            || (llama_attention_input_tiles && is_llama_attention_projection_weight(tensor_name));
        let should_write_input_tile_sidecar = should_write_input_tile_sidecar
            || (llama_lm_head_input_tiles && is_llama_lm_head_weight(tensor_name));
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
    if llama_mlp_input_tiles || llama_attention_input_tiles || llama_lm_head_input_tiles {
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

    writer.finalize()?;

    println!("Written to: {}", output);

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

fn write_input_tile_sidecar_tensor(
    writer: &mut RllmWriter,
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
    let sidecar_name = rllm_runtime::input_tile_sidecar_weight_name(source_name);

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

fn resolve_model_config_path(input_path: &Path, explicit_config: Option<&str>) -> Option<PathBuf> {
    if let Some(config) = explicit_config {
        return Some(PathBuf::from(config));
    }
    let sibling = input_path.parent()?.join("config.json");
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
    let sibling = input_path.parent()?.join("tokenizer.json");
    sibling.exists().then_some(sibling)
}

#[cfg(test)]
mod tests {
    use super::{
        input_tile_range_specs, input_tile_sidecar_bytes, is_llama_attention_projection_weight,
        is_llama_lm_head_weight, is_llama_mlp_projection_weight, PackCodecPolicy,
    };

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
