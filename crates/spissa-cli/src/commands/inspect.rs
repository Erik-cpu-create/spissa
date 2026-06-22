// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use anyhow::{Context, Result};
use spissa_container::SpissaReader;
use std::path::Path;

pub fn run(file: &str) -> Result<()> {
    let path = Path::new(file);

    if !path.exists() {
        anyhow::bail!("File does not exist: {}", file);
    }

    let reader =
        SpissaReader::open(path).with_context(|| format!("Failed to open file: {}", file))?;

    let header = reader.header();
    let metadata = reader.metadata();

    println!("spissa file: {}", file);
    println!("Format version: {}", header.version);
    println!("Model name: {}", metadata.model_name);
    println!("Architecture: {}", metadata.architecture);
    println!("Source format: {}", metadata.source_format);
    println!("Lossless: {}", metadata.lossless);
    println!("Created by: {}", metadata.created_by);
    println!("Codec: {}", metadata.codec);
    if let Some(config) = &metadata.model_config {
        println!("Model config:");
        if let Some(architecture_type) = &config.architecture_type {
            println!("  Architecture type: {}", architecture_type);
        }
        if let Some(layers) = config.num_hidden_layers {
            println!("  Layers: {}", layers);
        }
        if let Some(hidden_size) = config.hidden_size {
            println!("  Hidden size: {}", hidden_size);
        }
        if let Some(num_heads) = config.num_attention_heads {
            println!("  Attention heads: {}", num_heads);
        }
        if let Some(intermediate_size) = config.intermediate_size {
            println!("  Intermediate size: {}", intermediate_size);
        }
        if let Some(max_positions) = config.max_position_embeddings {
            println!("  Max positions: {}", max_positions);
        }
        if let Some(rotary_pct) = config.rotary_pct {
            println!("  Rotary pct: {}", rotary_pct);
        }
        if let Some(rotary_base) = config.rotary_emb_base {
            println!("  Rotary base: {}", rotary_base);
        }
        if let Some(layer_norm_eps) = config.layer_norm_eps {
            println!("  LayerNorm eps: {}", layer_norm_eps);
        }
        if let Some(use_parallel_residual) = config.use_parallel_residual {
            println!("  Parallel residual: {}", use_parallel_residual);
        }
        if let Some(vocab_size) = config.vocab_size {
            println!("  Vocab size: {}", vocab_size);
        }
    }
    if let Some(tokenizer) = &metadata.tokenizer {
        println!("Tokenizer:");
        if let Some(tokenizer_type) = &tokenizer.tokenizer_type {
            println!("  Type: {}", tokenizer_type);
        }
        println!("  Vocab size: {}", tokenizer.id_to_token.len());
        if let Some(unk_token_id) = tokenizer.unk_token_id {
            println!("  UNK token id: {}", unk_token_id);
        }
        if let Some(bos_token_id) = tokenizer.bos_token_id {
            println!("  BOS token id: {}", bos_token_id);
        }
        if let Some(eos_token_id) = tokenizer.eos_token_id {
            println!("  EOS token id: {}", eos_token_id);
        }
    }
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
        let range_checksum_count: usize = reader
            .list_chunks()
            .iter()
            .filter(|chunk| chunk.tensor_id == tensor.tensor_id)
            .map(|chunk| chunk.range_checksums.len())
            .sum();
        if range_checksum_count > 0 {
            println!("    Range checksums: {}", range_checksum_count);
        }

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
