use anyhow::{Context, Result};
use rllm_container::RllmReader;
use rtc_codec::{DecodeMeta, HuffmanCodec, RawCodec, RleCodec, TensorCodec};
use std::fs;
use std::path::Path;

fn get_codec(codec_id: &str) -> Result<Box<dyn TensorCodec>> {
    match codec_id {
        "rtc-raw-v1" => Ok(Box::new(RawCodec)),
        "rtc-rle-v1" => Ok(Box::new(RleCodec)),
        "rtc-huff-v1" => Ok(Box::new(HuffmanCodec)),
        _ => anyhow::bail!("Unknown codec: {}", codec_id),
    }
}

pub fn run(file: &str, out: &str) -> Result<()> {
    let path = Path::new(file);

    if !path.exists() {
        anyhow::bail!("File does not exist: {}", file);
    }

    println!("Unpacking: {}", file);

    let mut reader =
        RllmReader::open(path).with_context(|| format!("Failed to open file: {}", file))?;

    let tensors = reader.list_tensors();
    if tensors.len() != 1 {
        anyhow::bail!("Expected 1 tensor, found {}", tensors.len());
    }

    let chunks = reader.get_tensor_chunks(0);
    let chunk_data: Vec<_> = chunks
        .iter()
        .map(|c| (c.chunk_id, c.codec_id.clone(), c.uncompressed_size))
        .collect();

    let mut decoded_data = Vec::new();

    for (chunk_id, codec_id, uncompressed_size) in chunk_data {
        let codec = get_codec(&codec_id)?;
        let compressed_data = reader.read_chunk(chunk_id)?;

        let decode_meta = DecodeMeta {
            codec_id,
            uncompressed_size,
        };

        let decoded = codec.decode(&compressed_data, &decode_meta)?;
        decoded_data.extend(decoded);
    }

    fs::write(out, &decoded_data)
        .with_context(|| format!("Failed to write output file: {}", out))?;

    println!("Unpacked {} bytes to: {}", decoded_data.len(), out);

    Ok(())
}
