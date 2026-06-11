# Runtime-compressed Local LLM (RLLM)

Brain-inspired compressed runtime for local LLMs.

RLLM is an experimental local LLM runtime built around **lossless compressed model storage**. It stores model tensors in a chunked compressed container (`.rllm`) and aims to run inference by decoding only the tensor blocks needed at runtime.

> Run local LLMs from compressed weights without changing model weights.

## What RLLM Does

- Stores model tensors in a chunked compressed container format (`.rllm`)
- Uses **RTC (Rama Tensor Codec)** — in-house lossless tensor compression codecs
- Verifies decoded weights are **bit-identical** to originals
- Imports safetensors model files into `.rllm`
- Reports honest compression metrics — no magic claims

## What RLLM Does NOT Do

- ❌ Claim magical compression (for example, “7.6GB → 500MB with same quality”)
- ❌ Perform lossy quantization unless explicitly labeled as a lossy optimization
- ❌ Wrap Ollama, llama.cpp, or another existing runtime
- ❌ Change model weights in any way
- ❌ Use external generic compression libraries by default; RTC codecs are custom/in-house unless explicitly approved

## Current Status

Implemented:

- Cargo workspace with four crates
- `.rllm` v1 container reader/writer
- Safetensors import
- CLI commands: `pack`, `inspect`, `verify`, `doctor`
- Stubbed future commands: `run`, `import`, `benchmark`
- RTC codecs:
  - `rtc-raw-v1`
  - `rtc-rle-v1`
  - `rtc-huff-v1` (custom byte-level Huffman)
- Per-chunk SHA-256 verification
- Multi-tensor safetensors verification

Not yet implemented:

- Real inference runtime
- Layer/tile decode runtime
- Multi-tensor unpack back to safetensors layout
- Tokenizer/model execution

## Quick Start

```bash
# Build
cargo build

# Run tests
cargo test

# Check CLI/system info
cargo run -- doctor

# Pack a safetensors model into .rllm
cargo run -- pack ./models/pythia-70m/model.safetensors \
  --out ./models/pythia-70m.rllm \
  --chunk-size 32mb

# Inspect metadata and compression stats
cargo run -- inspect ./models/pythia-70m.rllm

# Verify lossless round-trip against the original safetensors file
cargo run -- verify ./models/pythia-70m/model.safetensors ./models/pythia-70m.rllm
```

`models/` is ignored because model files and `.rllm` outputs are large reproducible local artifacts.

## Architecture

```text
rllm/
├── crates/
│   ├── rllm-cli/        # CLI binary (`rllm`)
│   ├── rllm-container/  # .rllm binary format parser/writer
│   ├── rllm-import/     # Safetensors import
│   └── rtc-codec/       # In-house lossless tensor compression codecs
├── docs/
│   ├── format-rllm-v1.md
│   ├── codec-rtc-v1.md
│   └── roadmap.md
└── rllm_ai_agent_spec.md
```

### Components

| Crate | Purpose |
|-------|---------|
| `rllm-cli` | Command-line interface (`rllm` binary) |
| `rllm-container` | `.rllm` file format: header, metadata, tensor/chunk directories |
| `rllm-import` | External format import, currently safetensors |
| `rtc-codec` | Lossless tensor codecs: raw, RLE, Huffman |

## File Format

The `.rllm` format is a single-file container with:

- **Header** — `RLLM` magic + version + endian marker + directory offsets
- **Compressed chunk data** — tensor chunks encoded with RTC codecs
- **Global metadata** — model name, architecture, codec, tokenizer type
- **Tensor directory** — tensor metadata: name, shape, dtype, checksums
- **Chunk directory** — chunk metadata: offsets, sizes, codec ID, checksums

See [docs/format-rllm-v1.md](docs/format-rllm-v1.md) for the full specification.

## Codecs

Every RTC codec must satisfy:

```text
decode(encode(input)) == input
```

| Codec | Description | Status |
|-------|-------------|--------|
| `rtc-raw-v1` | Identity/no compression; baseline fallback | ✅ Implemented |
| `rtc-rle-v1` | Run-length encoding | ✅ Implemented |
| `rtc-huff-v1` | In-house byte-level Huffman entropy codec | ✅ Implemented |
| `rtc-delta-v1` | Delta encoding | 🔜 Future |
| `rtc-bitplane-v1` | Bitplane packing | 🔜 Future |
| `rtc-entropy-v1` | Advanced entropy coding beyond Huffman | 🔜 Future |

See [docs/codec-rtc-v1.md](docs/codec-rtc-v1.md) for codec design details.

## Verified Example: Pythia-70M

Local verification with EleutherAI Pythia-70M safetensors:

- 94 tensors imported
- 166,019,180 tensor bytes verified
- Best local output with `raw + rle + huff`, `--chunk-size 32mb`:
  - original safetensors file: 166,029,852 bytes / 158.34 MiB
  - `.rllm`: 126,456,271 bytes / 120.60 MiB
  - ratio: 76.16% of original file size
- `rllm verify`: `[OK] LOSSLESS VERIFIED`

The model files are not committed to this repository.

## Runtime Modes

| Mode | Description | Status |
|------|-------------|--------|
| `full-decode` | Decode all tensors to RAM | 🔜 Phase 5 |
| `layer-decode` | Decode layer-by-layer, release after use | 🔜 Phase 6 |
| `tile-decode` | Decode tiles during matmul | 🔜 Phase 7 |
| `fused-decode-matmul` | Decode + multiply in one step | 🔜 Future |

## Development

```bash
# Format
cargo fmt --all

# Build
cargo build

# Test
cargo test

# CLI help
cargo run -- --help
cargo run -- pack --help
```

Before committing:

```bash
git diff --check
cargo test
```

## Design Principles

1. **Lossless by default** — decoded weights must be bit-identical to originals
2. **Honest metrics** — report actual compression ratios, never overclaim
3. **From scratch** — no wrapping Ollama/llama.cpp
4. **Custom codecs** — compression codecs are in-house RTC codecs unless explicitly approved otherwise
5. **Incremental** — build phase by phase, verify each step
6. **Test everything** — round-trip tests for every codec, checksums everywhere

## License

CC0-1.0. See [LICENSE](LICENSE).
