use crate::commands::common::parse_size;
use anyhow::{Context, Result};
use rllm_container::{GlobalMetadata, RllmWriter};
use rllm_import::{read_model_config_metadata, read_tokenizer_metadata, SafetensorsReader};
use rtc_codec::{EncodeMeta, HuffmanCodec, RawCodec, RleCodec, TensorCodec};
use std::path::{Path, PathBuf};

pub fn run(
    input: &str,
    output: &str,
    chunk_size: &str,
    config: Option<&str>,
    tokenizer: Option<&str>,
    no_tokenizer: bool,
) -> Result<()> {
    let chunk_size_bytes = parse_size(chunk_size)?;
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
        codec: "auto".to_string(),
        model_config,
        tokenizer,
    };

    let mut writer = RllmWriter::new(output, metadata)?;

    // Codecs to try
    let codecs: Vec<Box<dyn TensorCodec>> = vec![
        Box::new(RawCodec),
        Box::new(RleCodec),
        Box::new(HuffmanCodec),
    ];

    let mut total_original = 0;
    let mut total_compressed = 0;
    let mut chunk_count = 0;

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

        total_original += meta.original_size_bytes;

        // Add tensor to writer
        writer.add_tensor(meta);

        // Encode chunks
        let encode_meta = EncodeMeta {
            name: tensor_name.to_string(),
            shape: vec![tensor_data.len() as u64],
            dtype: "u8".to_string(),
        };

        for (i, chunk) in tensor_data.chunks(chunk_size_bytes).enumerate() {
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

            writer.write_chunk(tensor_id as u64, &codec_id, &encoded.data, chunk, i as u64)?;

            chunk_count += 1;
            total_compressed += encoded.data.len();
        }
    }

    println!("\nEncoded {} chunks total", chunk_count);
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
