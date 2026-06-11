use anyhow::{Context, Result};
use rllm_container::RllmReader;
use std::path::Path;

pub fn run(file: &str) -> Result<()> {
    let path = Path::new(file);

    if !path.exists() {
        anyhow::bail!("File does not exist: {}", file);
    }

    let reader =
        RllmReader::open(path).with_context(|| format!("Failed to open file: {}", file))?;

    let header = reader.header();
    let metadata = reader.metadata();

    println!("RLLM File: {}", file);
    println!("Format version: {}", header.version);
    println!("Model name: {}", metadata.model_name);
    println!("Architecture: {}", metadata.architecture);
    println!("Source format: {}", metadata.source_format);
    println!("Lossless: {}", metadata.lossless);
    println!("Created by: {}", metadata.created_by);
    println!("Codec: {}", metadata.codec);
    println!();

    let tensors = reader.list_tensors();
    println!("Tensors: {}", tensors.len());

    let mut total_original = 0;
    let mut total_compressed = 0;

    for tensor in tensors {
        println!(
            "  - {} ({:?}): {:?}",
            tensor.name, tensor.dtype, tensor.shape
        );
        println!("    Original: {} bytes", tensor.original_size_bytes);
        println!("    Compressed: {} bytes", tensor.compressed_size_bytes);
        println!("    Chunks: {}", tensor.chunk_count);

        total_original += tensor.original_size_bytes;
        total_compressed += tensor.compressed_size_bytes;
    }

    println!();
    println!("Total original: {} bytes", total_original);
    println!("Total compressed: {} bytes", total_compressed);

    if total_original > 0 {
        let ratio = total_compressed as f64 / total_original as f64;
        println!("Compression ratio: {:.2}%", ratio * 100.0);
    }

    Ok(())
}
