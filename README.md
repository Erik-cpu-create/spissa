# RLLM — Runtime-compressed Local LLM

RLLM is an experimental local LLM runtime built around **lossless compressed model storage**. It stores model tensors in a chunked compressed container (`.rllm`) and aims to run inference by decoding only the tensor blocks needed at runtime.

**Core positioning:**

> Run local LLMs from compressed weights without changing the model.

## What RLLM Does

- Stores model tensors in a chunked compressed container format (`.rllm`)
- Uses **RTC (Rama Tensor Codec)** — a library of lossless tensor compression codecs
- Verifies that decoded weights are **bit-identical** to the originals
- Supports runtime modes that decode only what's needed (layer-wise, tile-wise)
- Provides honest compression metrics — no magic claims

## What RLLM Does NOT Do

- ❌ Claim magical compression (e.g., "7.6GB → 500MB with same quality")
- ❌ Perform lossy quantization (unless explicitly labeled as `lossy-optimize`)
- ❌ Wrap Ollama, llama.cpp, or any existing runtime
- ❌ Change model weights in any way

## Quick Start

```bash
# Build
cargo build --release

# Check system
cargo run -- doctor

# Pack a model (Phase 1+)
cargo run -- pack ./model_dir --out model.rllm --codec rtc-lossless-v1

# Inspect
cargo run -- inspect model.rllm

# Verify lossless
cargo run -- verify ./model_dir model.rllm

# Unpack
cargo run -- unpack model.rllm --out ./restored_model

# Run inference (Phase 5+)
cargo run -- run model.rllm --prompt "Hello"
```

## Architecture

```
rllm/
├── crates/
│   ├── rllm-cli/        # CLI binary
│   ├── rllm-container/  # .rllm binary format parser/writer
│   └── rtc-codec/       # Lossless tensor compression codecs
├── docs/
│   ├── format-rllm-v1.md
│   ├── codec-rtc-v1.md
│   └── roadmap.md
└── tests/
```

### Components

| Crate | Purpose |
|-------|---------|
| `rllm-cli` | Command-line interface (`rllm` binary) |
| `rllm-container` | `.rllm` file format: header, metadata, tensor/chunk directories |
| `rtc-codec` | Lossless tensor codecs: `rtc-raw-v1`, `rtc-rle-v1`, and more |

## File Format

The `.rllm` format is a single-file container with:

- **Magic header** — "RLLM" + version + endian marker
- **Global metadata** — model name, architecture, codec, etc.
- **Tensor directory** — metadata for each tensor (name, shape, dtype, checksums)
- **Chunk directory** — metadata for each compressed chunk (offsets, sizes, codec ID)
- **Compressed data** — tensor chunks encoded with RTC codecs
- **Footer checksum** — integrity verification

See [docs/format-rllm-v1.md](docs/format-rllm-v1.md) for the full specification.

## Codecs

RTC (Rama Tensor Codec) provides lossless compression codecs:

| Codec | Description | Status |
|-------|-------------|--------|
| `rtc-raw-v1` | Identity (no compression) — fallback/baseline | ✅ Implemented |
| `rtc-rle-v1` | Run-length encoding | ✅ Implemented |
| `rtc-huff-v1` | In-house byte-level Huffman entropy codec | ✅ Implemented |
| `rtc-delta-v1` | Delta encoding | 🔜 Future |
| `rtc-bitplane-v1` | Bitplane packing | 🔜 Future |
| `rtc-entropy-v1` | Advanced entropy coding beyond Huffman | 🔜 Future |

Every codec must satisfy: `decode(encode(input)) == input` (bit-identical).

See [docs/codec-rtc-v1.md](docs/codec-rtc-v1.md) for codec design details.

## Runtime Modes

| Mode | Description | Status |
|------|-------------|--------|
| `full-decode` | Decode all tensors to RAM | 🔜 Phase 5 |
| `layer-decode` | Decode layer-by-layer, release after use | 🔜 Phase 6 |
| `tile-decode` | Decode tiles during matmul | 🔜 Phase 7 |
| `fused-decode-matmul` | Decode + multiply in one step | 🔜 Future |

## Roadmap

### Phase 0 — Project Skeleton ✅
- [x] Cargo workspace
- [x] CLI skeleton with clap
- [x] Basic structs (Header, TensorMeta, ChunkMeta)
- [x] Unit test setup
- [x] `rtc-raw-v1` codec with round-trip verification

### Phase 1 — RLLM Container v1
- [ ] Write .rllm header
- [ ] Write metadata (JSON)
- [ ] Write tensor directory
- [ ] Write chunk directory
- [ ] Write raw chunks
- [ ] Read .rllm back
- [ ] `rllm inspect` command

### Phase 2 — RTC Codec v1
- [x] `rtc-rle-v1` codec
- [x] `rtc-huff-v1` byte-level Huffman codec
- [x] Codec selection (try multiple, pick best)
- [x] Per-chunk verification

### Phase 3 — Pack/Unpack/Verify
- [ ] `rllm pack` with real tensor input
- [ ] `rllm unpack`
- [ ] `rllm verify` with bit-identical check

### Phase 4 — Safetensors Import
- [ ] Read safetensors metadata
- [ ] Read tensor bytes
- [ ] Pack/unpack safetensors losslessly

### Phase 5 — Toy Inference Runtime
- [ ] Minimal tensor operations
- [ ] Embedding, linear, attention, MLP
- [ ] Sampling and streaming output

### Phase 6-7 — Layer/Tile Decode
- [ ] Layer-wise decode runtime
- [ ] Tile-wise decode with fused matmul
- [ ] Memory tracking and cache

### Phase 8 — Real Model Support
- [ ] Small real transformer model
- [ ] Tokenizer support
- [ ] Logit comparison with reference

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Run with verbose logging
cargo run -- --verbose doctor

# Format code (when rustfmt is available)
cargo fmt

# Lint
cargo clippy
```

## Design Principles

1. **Lossless by default** — decoded weights must be bit-identical to originals
2. **Honest metrics** — report actual compression ratios, never overclaim
3. **From scratch** — no wrapping Ollama/llama.cpp
4. **Incremental** — build phase by phase, verify each step
5. **Test everything** — round-trip tests for every codec, checksums everywhere

## License

MIT
