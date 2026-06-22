// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use anyhow::{Context, Result};
use spissa_container::SpissaReader;
use spissa_import::SafetensorsReader;
use rtc_codec::{BitplaneCodec, DecodeMeta, HuffmanCodec, RansCodec, RawCodec, RleCodec, TensorCodec};
use sha2::{Digest, Sha256};
use std::path::Path;

fn get_codec(codec_id: &str) -> Result<Box<dyn TensorCodec>> {
    match codec_id {
        "rtc-raw-v1" => Ok(Box::new(RawCodec)),
        "rtc-rle-v1" => Ok(Box::new(RleCodec)),
        "rtc-huff-v1" => Ok(Box::new(HuffmanCodec)),
        "rtc-rans-v1" => Ok(Box::new(RansCodec)),
        "rtc-bitplane-v1" => Ok(Box::new(BitplaneCodec)),
        _ => anyhow::bail!("Unknown codec: {}", codec_id),
    }
}

pub fn run(original: &str, compressed: &str) -> Result<()> {
    let original_path = Path::new(original);
    let compressed_path = Path::new(compressed);

    if !original_path.exists() {
        anyhow::bail!("Original file does not exist: {}", original);
    }
    if !compressed_path.exists() {
        anyhow::bail!("Compressed file does not exist: {}", compressed);
    }

    println!("Verifying: {} against {}", compressed, original);

    // Open safetensors reader
    let mut safetensors = SafetensorsReader::open(original_path)
        .with_context(|| format!("Failed to open original file: {}", original))?;

    // Open RLLM reader
    let reader = SpissaReader::open(compressed_path)
        .with_context(|| format!("Failed to open compressed file: {}", compressed))?;

    // Collect tensor info first
    let tensor_info: Vec<_> = reader
        .list_tensors()
        .iter()
        .map(|t| (t.tensor_id, t.name.clone(), t.original_sha256))
        .collect();
    let tensor_count = tensor_info.len();

    println!("Found {} tensors", tensor_count);

    let mut total_verified = 0;

    // Verify each tensor
    for (tensor_id, tensor_name, expected_hash) in tensor_info {
        println!("Verifying tensor: {}", tensor_name);

        // Read original tensor data
        let original_data = safetensors.read_tensor(&tensor_name)?;
        let original_hash = Sha256::digest(&original_data);

        // Check hash matches
        if expected_hash != original_hash.as_slice() {
            let orig_hex = original_hash.iter().fold(String::new(), |mut s, b| {
                s.push_str(&format!("{:02x}", b));
                s
            });
            let stored_hex = expected_hash.iter().fold(String::new(), |mut s, b| {
                s.push_str(&format!("{:02x}", b));
                s
            });
            anyhow::bail!(
                "Hash mismatch for tensor {}!\n  Original: {}\n  Stored:   {}",
                tensor_name,
                orig_hex,
                stored_hex
            );
        }

        // Get chunks for this tensor
        let chunks = reader.get_tensor_chunks(tensor_id);
        let chunk_data: Vec<_> = chunks
            .iter()
            .map(|c| {
                (
                    c.chunk_id,
                    c.codec_id.clone(),
                    c.uncompressed_size,
                    c.chunk_sha256_original,
                )
            })
            .collect();

        let mut decoded_data = Vec::new();

        for (chunk_id, codec_id, uncompressed_size, chunk_hash) in chunk_data {
            let codec = get_codec(&codec_id)?;
            let compressed_data = reader.read_chunk(chunk_id)?;

            let decode_meta = DecodeMeta {
                codec_id: codec_id.clone(),
                uncompressed_size,
            };

            let decoded = codec.decode(&compressed_data, &decode_meta)?;

            let computed_hash = Sha256::digest(&decoded);
            if computed_hash.as_slice() != chunk_hash {
                anyhow::bail!(
                    "Chunk {} hash mismatch for tensor {}",
                    chunk_id,
                    tensor_name
                );
            }

            decoded_data.extend(decoded);
        }

        if decoded_data != original_data {
            anyhow::bail!(
                "Decoded data does not match original for tensor {}!",
                tensor_name
            );
        }

        println!("  [OK] {} bytes verified", decoded_data.len());
        total_verified += decoded_data.len();
    }

    println!(
        "\n[OK] Verified {} tensors, {} bytes total",
        tensor_count, total_verified
    );
    println!("[OK] LOSSLESS VERIFIED");

    Ok(())
}
