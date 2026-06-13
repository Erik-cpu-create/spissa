use crate::commands::common::parse_size;
use anyhow::{Context, Result};
use rllm_container::{GlobalMetadata, RllmWriter};
use rllm_import::{read_model_config_metadata, read_tokenizer_metadata, SafetensorsReader};
use rtc_codec::{
    EncodeMeta, HuffmanCodec, RawCodec, RleCodec, TensorCodec, CODEC_HUFF_V1, CODEC_RAW_V1,
    CODEC_RLE_V1,
};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackCodecPolicy {
    Auto,
    Raw,
    Rle,
    Huff,
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

    // Process each tensor
    for (tensor_id, tensor_name) in tensor_names.iter().enumerate() {
        println!(
            "Processing tensor: {} ({}/{})",
            tensor_name,
            tensor_id + 1,
            tensor_names.len()
        );

        // Read tensor data
        let tensor_data = reader.read_tensor(tensor_name)?;
        let tensor_meta = reader.to_rllm_meta(tensor_name)?;

        // Update tensor ID
        let mut meta = tensor_meta;
        meta.tensor_id = tensor_id as u64;
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
        writer.add_tensor(meta);

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
                        tensor_id as u64,
                        &codec_id,
                        &encoded.data,
                        chunk,
                        i as u64,
                        range_size as u64,
                    )?;
                    range_checksum_count += chunk.len().div_ceil(range_size);
                } else {
                    writer.write_chunk(
                        tensor_id as u64,
                        &codec_id,
                        &encoded.data,
                        chunk,
                        i as u64,
                    )?;
                    range_checksum_skipped_chunks += 1;
                }
            } else {
                writer.write_chunk(tensor_id as u64, &codec_id, &encoded.data, chunk, i as u64)?;
            }

            chunk_count += 1;
            total_compressed += encoded.data.len();
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
    use super::PackCodecPolicy;

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
}
