# Spissa Format v1 Specification

## Overview

The `.spsa` format is a single-file binary container for storing compressed LLM model tensors. It supports:

- Lossless compression via chunked tensor storage
- Random access to individual tensors and chunks
- Integrity verification via SHA-256 checksums
- Multiple codec support
- Memory-mapped loading

## File Layout

```
┌─────────────────────────────────┐
│ Header (44 bytes)               │
├─────────────────────────────────┤
│ Compressed Chunk Data (variable)│
├─────────────────────────────────┤
│ Global Metadata (u64 len + JSON)│
├─────────────────────────────────┤
│ Tensor Directory (u64 len + JSON)│
├─────────────────────────────────┤
│ Chunk Directory (u64 len + JSON)│
└─────────────────────────────────┘
```

## Header

Size: 44 bytes, little-endian

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | "SPSA" (0x53505341) |
| 4 | 4 | version | Format version (u32, currently 1) |
| 8 | 1 | endian | 0 = little-endian |
| 9 | 3 | reserved | Must be zero |
| 12 | 8 | metadata_offset | File offset to global metadata (u64) |
| 20 | 8 | tensor_dir_offset | File offset to tensor directory (u64) |
| 28 | 8 | chunk_dir_offset | File offset to chunk directory (u64) |
| 36 | 8 | data_start_offset | File offset where compressed chunk data starts (u64) |

## Global Metadata

Length-prefixed JSON, located at `metadata_offset`. The first 8 bytes are a
little-endian `u64` JSON byte length, followed by the JSON payload.

```json
{
  "model_name": "example-12b",
  "architecture": "decoder-only-transformer",
  "source_format": "safetensors",
  "lossless": true,
  "default_context_length": 2048,
  "tokenizer_type": "sentencepiece",
  "created_by": "spissa-pack",
  "codec": "rtc-lossless-v1",
  "tokenizer": {
    "tokenizer_type": "hf-bpe",
    "id_to_token": ["<|endoftext|>", "Hello", "Ġworld"],
    "eos_token_id": 0
  }
}
```

`tokenizer` is optional and additive. Existing `.spsa` files without tokenizer
metadata remain valid. Phase 5E stores a runtime-ready `id_to_token` table for a
narrow text-generation smoke path; full production BPE/normalizer fidelity is a
future tokenizer-runtime concern.

## Tensor Directory

Array of tensor metadata entries. Each entry:

```json
{
  "tensor_id": 42,
  "name": "layers.0.mlp.down_proj.weight",
  "shape": [4096, 11008],
  "dtype": "bf16",
  "original_size_bytes": 90177536,
  "compressed_size_bytes": 73400320,
  "original_sha256": "abcdef...",
  "chunk_count": 256,
  "chunk_start_index": 1024
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| tensor_id | u64 | Unique identifier |
| name | string | Tensor name (dot-separated path) |
| shape | u64[] | Tensor dimensions |
| dtype | string | Data type (fp16, bf16, fp32, etc.) |
| original_size_bytes | u64 | Uncompressed size |
| compressed_size_bytes | u64 | Compressed size (sum of chunks) |
| original_sha256 | bytes[32] | SHA-256 of original tensor |
| chunk_count | u32 | Number of chunks |
| chunk_start_index | u64 | First chunk index in chunk directory |

## Chunk Directory

Array of chunk metadata entries. Each entry:

```json
{
  "chunk_id": 1024,
  "tensor_id": 42,
  "chunk_offset_in_tensor": 0,
  "uncompressed_size": 262144,
  "compressed_size": 213712,
  "file_offset": 987654321,
  "codec_id": "rtc-lossless-v1",
  "chunk_sha256_original": "...",
  "chunk_sha256_compressed": "..."
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| chunk_id | u64 | Unique identifier |
| tensor_id | u64 | Parent tensor |
| chunk_offset_in_tensor | u64 | Offset within tensor (elements) |
| uncompressed_size | u64 | Original chunk size |
| compressed_size | u64 | Compressed size |
| file_offset | u64 | Absolute file offset to compressed data |
| codec_id | string | Codec used (e.g., "rtc-raw-v1") |
| chunk_sha256_original | bytes[32] | SHA-256 of uncompressed chunk |
| chunk_sha256_compressed | bytes[32] | SHA-256 of compressed chunk |

## Chunk Size

Default: 1 MB uncompressed per chunk.

Trade-offs:
- **Small chunks** (256KB): Better random access, more metadata overhead
- **Large chunks** (4MB): Better compression ratio, worse random access

## Integrity

Spissa v1 stores SHA-256 integrity metadata per tensor and per chunk. Optional
range checksums may also be present for verified partial reads. There is no
footer checksum in the current writer implementation.

## Versioning

The format version is stored in the header. When making breaking changes:

1. Bump the version number
2. Maintain backward compatibility when possible
3. Document migration path

## Design Decisions

1. **Single file** — easier to distribute, checksum, and memory-map
2. **JSON metadata** — human-readable, easy to debug
3. **SHA-256 checksums** — strong integrity verification
4. **Chunk-based** — enables random access and partial decoding
5. **Codec per chunk** — allows mixing codecs for different data types
