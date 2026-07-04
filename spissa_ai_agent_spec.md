# Spissa / RTC — Lossless Runtime-Compressed Local LLM Tool

> Project brief for an AI agent / coding agent.
> Goal: build a local tool similar to Ollama, but **from scratch**, with one core feature: **LLM models can be stored in a dedicated compressed format and executed by our own runtime without altering the model weights in any lossy way**.

---

## 0. Idea Summary

We want to build a new tool for running LLMs offline/locally. This tool must not be a mere wrapper around Ollama, llama.cpp, or an existing format. It must have its own identity:

- Its own CLI.
- Its own model file format.
- Its own tensor compression library.
- Its own inference runtime.
- Core focus: **lossless compressed model storage + a runtime that can read compressed weights block-wise/tile-wise**.

Working names:

```text
Spissa = Runtime-compressed Local LLM
RTC  = Rama Tensor Codec
RAMA = Rama Active Memory Architecture
ERIK = Episodic Recall Inference Kernel (reserved future subsystem)
REE  = Rama Erik Esprada kernel lineage
```

Core positioning:

```text
Run local LLMs from compressed weights without changing the model.
```

---

## 1. Core Principles That Must Not Be Violated

### 1.1 Do not claim magical compression

This tool **must not** promise:

```text
LLM 7.6GB -> 500MB with the exact same quality
```

Lossless compression has limits. If the model data is already very dense, the compression gain may be small.

Claims that are allowed:

```text
Spissa stores model weights in a lossless compressed format.
Spissa can verify that decoded weights are identical to the original weights.
Spissa can reduce model storage size when the weights still contain redundancy.
Spissa can reduce peak RAM through block-wise/tile-wise decoding, depending on the runtime mode.
```

### 1.2 “Without reducing quality” means lossless

In this project, the phrase **without reducing quality** must be interpreted technically as:

```text
Decoded model weights must be bit-identical to the original weights.
```

Not:

```text
Benchmarks look similar.
Outputs feel similar.
Users don't notice the quality drop.
```

Lossless criterion:

```text
original tensor bytes == decoded tensor bytes
```

If the weights are identical, then the model being run is the same model.

### 1.3 Do not apply extra quantization for lossless mode

Extra quantization such as FP16 -> INT4 is lossy, unless the input weights are already INT4 and we merely re-pack the bits without changing the values.

Allowed modes:

```text
lossless-pack      = safe, weights unchanged
lossy-optimize     = optional in the future, must be clearly labeled
```

This project's default is:

```text
lossless only
```

### 1.4 Do not depend on Ollama/llama.cpp as the core runtime

Learning general ideas from other tools is fine, but the project's implementation must be from scratch.

Not allowed:

```text
spissa merely calls ollama run
spissa merely wraps llama.cpp
spissa merely renames GGUF into its own format
```

Allowed:

```text
studying general transformer concepts
using standard libraries for CLI, checksum, mmap, filesystem
using small/toy models for early validation
```

### 1.5 Runtime kernels must have REE versioning

Every Spissa execution kernel that becomes a serious candidate must have an REE
lineage name before it is reported, benchmarked, or merged.

Rule:

```text
REE = Rama Erik Esprada kernel lineage
```

Example names:

```text
REEDOT-LAB    = dot-product microbench lab, not yet used by the runtime
REEBORN-Q8    = the first proven Q8 kernel, promoted into the runtime
REETHINK-Q8   = kernel redesign after the previous direction failed
REEFUSE-Q8    = fused kernel, e.g. gate/up or matmul+scale
REELITE-Q8    = kernel dedicated to low-end CPU / IoT profiles
```

The agent must not create or merge generically named changes such as
`fast path`, `candidate kernel`, or `optimized kernel` without recording their
REE name in the plan and benchmark report. Failed trials must still record their
REE name so that negative evidence stays traceable.

---

## 2. Product Target

### 2.1 Main CLI

Target commands:

```bash
spissa import ./model_dir
spissa pack ./model_dir --out model.spsa --codec rtc-lossless-v1
spissa inspect model.spsa
spissa verify ./model_dir model.spsa
spissa run model.spsa
```

Additional commands:

```bash
spissa unpack model.spsa --out ./restored_model
spissa benchmark model.spsa
spissa doctor
```

### 2.2 MVP Features

The MVP must have:

1. The `spissa` CLI.
2. The `.spsa` file format.
3. The `rtc` library for lossless tensor encode/decode.
4. Minimal import from a simple tensor format.
5. Packing tensors into `.spsa`.
6. Unpacking `.spsa` back to the original tensors.
7. Bit-identical verification.
8. A toy inference runtime for small models.

The MVP is not yet required to support large models such as Gemma 12B.

### 2.3 Post-MVP Features

Once the MVP is stable:

1. `safetensors` support.
2. Tokenizer support.
3. Minimal transformer architecture support.
4. Layer-by-layer inference.
5. Block-wise decoding.
6. Tile-wise decoding.
7. Memory-mapped compressed loading.
8. Fused decode + matmul.
9. Automatic low-RAM runtime mode.

---

## 3. System Architecture

```text
spissa CLI
├── import module
│   ├── load model metadata
│   ├── load tokenizer
│   └── load tensors
│
├── pack module
│   ├── split tensors into chunks
│   ├── encode chunks with RTC
│   ├── write .spsa container
│   └── write checksums
│
├── inspect module
│   ├── read header
│   ├── list tensors
│   ├── list codecs
│   ├── show compression ratio
│   └── estimate memory
│
├── verify module
│   ├── decode all chunks
│   ├── compare tensor hash
│   ├── compare shape/dtype
│   └── report bit-identical status
│
└── runtime module
    ├── tokenizer
    ├── transformer graph
    ├── compressed tensor reader
    ├── decode cache
    ├── matmul kernels
    ├── attention
    ├── sampling
    └── streaming output
```

---

## 4. Format File `.spsa`

### 4.1 Format goals

The `.spsa` format must:

- Be single-file where possible.
- Be able to store model metadata.
- Be able to store the tokenizer.
- Be able to store tensors in chunks.
- Support random access to chunks.
- Support per-tensor and per-chunk checksums.
- Support multiple codecs in the future.
- Support memory mapping.
- Be readable without decompressing the entire model.

### 4.2 Initial layout

```text
.spsa file
├── magic header
├── format version
├── global metadata length
├── global metadata
├── tokenizer block
├── architecture block
├── tensor directory
├── codec directory
├── chunk directory
├── compressed tensor chunks
└── footer checksum
```

### 4.3 Header

Example header fields:

```text
magic:      "SPSA"
version:    1
endian:     little
created_by: spissa
container:  spissa-v1
```

### 4.4 Global metadata

Metadata is stored in an easily readable format, e.g. JSON/CBOR/MessagePack.

Example:

```json
{
  "model_name": "example-12b",
  "architecture": "decoder-only-transformer",
  "source_format": "safetensors",
  "lossless": true,
  "default_context_length": 2048,
  "tokenizer_type": "sentencepiece-or-bpe",
  "created_by": "spissa-pack",
  "codec": "rtc-lossless-v1"
}
```

### 4.5 Tensor directory

Every tensor must have metadata:

```text
tensor_id
name
shape
dtype
original_size_bytes
compressed_size_bytes
original_sha256
chunk_count
chunk_start_index
```

Example:

```json
{
  "tensor_id": 42,
  "name": "layers.0.mlp.down_proj.weight",
  "shape": [4096, 11008],
  "dtype": "bf16",
  "original_size_bytes": 90177536,
  "compressed_size_bytes": 73400320,
  "original_sha256": "...",
  "chunk_count": 256,
  "chunk_start_index": 1024
}
```

### 4.6 Chunk directory

Every chunk must have metadata:

```text
chunk_id
tensor_id
chunk_offset_in_tensor
uncompressed_size
compressed_size
file_offset
codec_id
chunk_sha256_original
chunk_sha256_compressed
```

Example:

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

### 4.7 Chunk size

Recommended initial chunk size:

```text
256KB to 4MB uncompressed per chunk
```

Trade-off:

```text
small chunk  = good random access, larger metadata
large chunk  = better compression, worse random access
```

The MVP can start from:

```text
1MB per chunk
```

---

## 5. RTC: Rama Tensor Codec

### 5.1 RTC goals

RTC is a lossless codec for LLM tensors.

Input:

```text
raw tensor bytes + dtype + shape
```

Output:

```text
compressed chunk bytes + codec metadata
```

Decoding must produce exactly the same bytes.

### 5.2 Initial codec modes

MVP codecs:

```text
rtc-raw-v1
rtc-rle-v1
rtc-delta-v1
rtc-bitplane-v1
rtc-entropy-v1
```

For the earliest MVP, implement at minimum:

```text
rtc-raw-v1
rtc-rle-v1
```

`rtc-raw-v1` does not compress, but is important as a fallback.

### 5.3 Codec selection

The packer must try several codecs on a small chunk and pick the best one.

Pseudo-flow:

```text
for each chunk:
    candidates = []
    candidates.append(encode_raw(chunk))
    candidates.append(encode_rle(chunk))
    candidates.append(encode_delta(chunk))
    candidates.append(encode_bitplane(chunk))
    choose smallest candidate that decodes exactly
```

If a codec fails to shrink the data, use raw.

### 5.4 Dtype-aware compression

RTC must be dtype-aware.

For `fp16` / `bf16`:

```text
split bits into sign/exponent/mantissa candidates
try compress exponent stream
try compress repeated patterns
try bitplane packing
fallback raw mantissa if random
```

For quantized int weights:

```text
respect original packed representation
never change values
try bit-level repacking only if reversible
compress scales separately
compress zero-points separately
```

For metadata/scale tensors:

```text
try delta coding
try RLE
try entropy coding
```

### 5.5 Lossless verification per chunk

Every encode must immediately run a self-test:

```text
encoded = encode(chunk)
decoded = decode(encoded)
assert decoded == chunk
```

If they differ, the candidate codec is rejected.

---

## 6. Runtime Design

### 6.1 Runtime goals

The runtime must be able to run a model from `.spsa` without fully unpacking it to the original file.

Runtime modes:

```text
full-decode mode       = decode all tensors into RAM, the simplest
layer-decode mode      = decode a layer when needed
tile-decode mode       = decode a tensor slice during matmul
fused-decode-matmul    = decode and multiply directly, the long-term target
```

### 6.2 MVP runtime

The MVP runtime may be simple:

```text
load .spsa
full-decode a small model's tensors into RAM
run toy transformer inference
verify deterministic output
```

Then move up to:

```text
layer-by-layer decode
```

### 6.3 Low-RAM runtime target

Low-RAM mode must avoid decoding the entire model at once.

Flow:

```text
for each transformer layer:
    decode layer weights
    run attention + mlp
    release layer weights
```

For large matmuls:

```text
for each tile in weight matrix:
    decode tile
    multiply activation x tile
    accumulate output
    release tile
```

### 6.4 Cache

The runtime needs a decode cache:

```text
LRU cache for decoded chunks
configurable max memory
```

CLI example:

```bash
spissa run model.spsa --cache 512mb
spissa run model.spsa --mode layer-decode
spissa run model.spsa --mode tile-decode --cache 256mb
```

### 6.5 Expected trade-off

The documentation must be honest:

```text
Compressed runtime can reduce storage and peak RAM.
It may reduce speed because decoding costs CPU cycles.
```

Do not claim it is always faster.

---

## 7. CLI Specification

### 7.1 `spissa pack`

```bash
spissa pack ./model_dir --out model.spsa --codec rtc-lossless-v1
```

Example output:

```text
Reading model: ./model_dir
Tensors: 291
Original size: 7.60 GB
Compressed size: 6.92 GB
Ratio: 91.0%
Lossless verification: passed
Output: model.spsa
```

Options:

```bash
--codec rtc-lossless-v1
--chunk-size 1mb
--verify
--no-tokenizer
--metadata metadata.json
--compression-level 1..9
```

### 7.2 `spissa inspect`

```bash
spissa inspect model.spsa
```

Output:

```text
File: model.spsa
Format: Spissa v1
Lossless: true
Architecture: decoder-only-transformer
Tensors: 291
Original size: 7.60 GB
Compressed size: 6.92 GB
Compression ratio: 91.0%
Chunk count: 7781
Codec: rtc-lossless-v1
Tokenizer: included
```

### 7.3 `spissa verify`

```bash
spissa verify ./model_dir model.spsa
```

Output:

```text
Verifying tensors...
[OK] shapes match
[OK] dtypes match
[OK] tensor hashes match
[OK] decoded bytes are bit-identical
Status: LOSSLESS VERIFIED
```

### 7.4 `spissa unpack`

```bash
spissa unpack model.spsa --out ./restored_model
```

Output:

```text
Decoded model written to ./restored_model
Verification: passed
```

### 7.5 `spissa run`

```bash
spissa run model.spsa
```

Options:

```bash
--prompt "Hello"
--ctx 2048
--max-tokens 128
--temperature 0.7
--top-p 0.9
--mode full-decode|layer-decode|tile-decode
--cache 512mb
--threads 4
```

---

## 8. Repository Structure

Suggested monorepo:

```text
spissa/
├── README.md
├── docs/
│   ├── format-spissa-v1.md
│   ├── codec-rtc-v1.md
│   ├── runtime-design.md
│   └── roadmap.md
│
├── crates/
│   ├── spissa-cli/
│   ├── spissa-container/
│   ├── rtc-codec/
│   ├── spissa-runtime/
│   ├── spissa-tokenizer/
│   └── spissa-kernels/
│
├── python/
│   └── spissa/
│
├── tests/
│   ├── golden/
│   ├── codec/
│   ├── container/
│   └── runtime/
│
├── examples/
│   ├── tiny-model/
│   └── toy-transformer/
│
└── benchmarks/
    ├── compression/
    ├── decode-speed/
    └── inference-memory/
```

Recommended language:

```text
Rust for CLI/container/codec safety
C/C++ or Rust SIMD for kernels later
Python for experiments/tests only
```

---

## 9. Implementation Phases

### Phase 0 — Project skeleton

Deliverables:

```text
cargo workspace
CLI skeleton
unit test setup
README
basic docs
```

Commands should exist:

```bash
spissa --help
spissa pack --help
spissa inspect --help
spissa verify --help
```

Acceptance criteria:

```text
Project builds.
CLI help works.
No model support needed yet.
```

---

### Phase 1 — Spissa container v1

Deliverables:

```text
write .spsa header
write metadata
write tensor directory
write chunk directory
write raw chunks
read .spsa back
inspect command
```

Use fake tensor input first.

Acceptance criteria:

```text
Can create .spsa file from sample tensors.
Can inspect file.
Can list tensors and chunks.
Can decode raw chunks.
```

---

### Phase 2 — RTC lossless codec v1

Deliverables:

```text
rtc-raw-v1
rtc-rle-v1
codec trait/interface
encode/decode test
per-chunk verification
```

Acceptance criteria:

```text
For every test tensor:
decode(encode(tensor)) == tensor
```

Golden tests:

```text
all-zero tensor
random tensor
repeated pattern tensor
small fp16 tensor
small bf16-like bytes
small int4-packed-like bytes
```

---

### Phase 3 — Pack/unpack/verify

Deliverables:

```text
spissa pack sample_tensor_dir --out model.spsa
spissa unpack model.spsa --out restored_tensor_dir
spissa verify original restored/model.spsa
```

Acceptance criteria:

```text
Original files and unpacked files are byte-identical.
SHA256 matches.
```

---

### Phase 4 — Safetensors import

Deliverables:

```text
read safetensors metadata
read tensor bytes
pack safetensors into .spsa
unpack back to safetensors-compatible layout
verify hashes
```

Acceptance criteria:

```text
Small safetensors model can be packed/unpacked losslessly.
```

---

### Phase 5 — Toy inference runtime

Deliverables:

```text
minimal tensor operations
embedding lookup
linear layer
RMSNorm or LayerNorm
simple attention
MLP
sampling
```

Start with a tiny custom transformer, not a production LLM.

Acceptance criteria:

```text
Same prompt + same seed produces same tokens in original runtime and Spissa runtime.
```

---

### Phase 6 — Layer decode runtime

Deliverables:

```text
runtime reads compressed .spsa
only decodes needed layer weights
releases weights after layer
tracks peak memory
```

Acceptance criteria:

```text
Peak memory lower than full decode mode for supported toy model.
Output identical to full decode mode.
```

---

### Phase 7 — Tile decode runtime

Current status:

```text
partial complete: fused tile-linear primitive bounds f32 scratch to a tile while preserving chunk-level decode correctness; streaming MLP/attention/transformer blocks and tiny/RAMA/GPT-NeoX generation projections now route through that tiled linear path; local Pythia-70M release benchmark matrix completed with 88.62–94.62 MiB max RSS under a 100mb internal budget; fixed-token HF/PyTorch logits comparison now passes top-1/top-10 on tested prompts after GPT-NeoX parallel residual metadata and per-head QKV split fixes; Phase 7.8A/B/C/D range-decode, per-range checksum metadata, opt-in raw pack range metadata, and pack-time tile/block chunk alignment foundation is implemented; real local Pythia-70M tile-block artifact benchmark completed at 18.39–22.64 MiB RSS with 386 KiB tracked transient peak while preserving tested HF logits parity; Phase 7.9A RAMA trace profiler is implemented with `spissa run --rama-trace`, and measured one-token Pythia trace showed chunk decode dominated recorded time (~3716 ms) while disk read was small (~32 ms); Phase 7.9B selective embedding row recall is implemented, reducing `gpt_neox.embed_in.weight` trace events from 1,965 to 5 and improving the 12-row tile-block matrix from 5.07 to 2.93 average seconds/token (~1.73× average speedup) with max RSS 22.28 MiB; Phase 7.9C low-ram-fast raw/tile-block profile is implemented with explicit `spissa pack --codec`, `spissa run --rama-integrity strict|verify-once`, and a reproducible benchmark harness; local Pythia-70M raw/tile-block artifact verifies losslessly, preserves tested HF top-1 parity, and reaches 0.35 average seconds/token / 3.26 average tok/s / 4.35 best tok/s while RSS stays 19.17–23.36 MiB for short prompts; Phase 7.9D real long-prompt benchmark is implemented with deterministic `--token-ids` input lengths 1/128/512/1024 under `ctx=2048`, showing short prompt + 16 generated tokens at 4.301 tok/s / 20.67 MiB RSS but 512-token and 1024-token prompts dropping to 0.300 tok/s / 44.98 MiB RSS and 0.148 tok/s / 70.84 MiB RSS; long-prompt HF parity passes top-1/top-10 for 128-token and 512-token fixed prompts; Phase 7.9E RAMA chunked prefill/timing is implemented with `spissa run --rama-timing` and `--rama-prefill-chunk-tokens`, improving 512-token + 16 generated from 56.43s / 0.284 tok/s / 46.22 MiB RSS to 35.20s / 0.455 tok/s / 34.05 MiB RSS and 1024-token + 16 generated from 110.29s / 0.145 tok/s / 70.55 MiB RSS to 63.84s / 0.251 tok/s / 44.98 MiB RSS while preserving tested 512-token HF top-1/top-10 parity; Phase 7.10A row-span linear accumulation optimizes the tiled-linear hot loop without changing format or logits semantics, improving short prompt + 16 generated from 4.39s / 3.647 tok/s to 2.25s / 7.101 tok/s, 512-token chunked prefill from 35.20s / 0.455 tok/s to 7.08s / 2.259 tok/s, and 1024-token chunked prefill from 63.84s / 0.251 tok/s to 12.76s / 1.254 tok/s while preserving tested 512-token HF top-1/top-10 parity; Phase 7.10B RAMA prefill homeostasis completes the broader post-rowspan matrix and sets the measured 32-token prefill window as the CLI default, reaching 9.756 tok/s for short prompt + 16 generated, 2.3495 tok/s / 32.77 MiB RSS for 512-token + 16, and 1.1653 tok/s / 44.91 MiB RSS for 1024-token + 16; Phase 7.10C deep prefill timing is implemented and shows the next measured bottleneck is RAMA prefill MLP compute (57.7–61.4% of prefill) with attention second (33.8–39.8%) and layer-param recall tiny (~0.2%); Phase 7.10D splits MLP into input projection/GELU/output projection, rejects measured regressions from larger MLP tiles and single-row dot unroll, and accepts four-prompt-row accumulation reuse, improving 512-token + 16 from 7.56s / 2.116 tok/s to 5.25s / 3.048 tok/s and 1024-token + 16 from 13.95s / 1.147 tok/s to 9.12s / 1.754 tok/s while preserving saved default-32 Spissa logits exactly; Phase 7.10E splits attention into QKV projection, QKV split, rotary, score/context, output projection, and KV append, rejects measured in-place softmax regression, and accepts K/V row-slice score/context optimization, improving 512-token + 16 from 4.95s / 3.232 tok/s to 4.56s / 3.509 tok/s and 1024-token + 16 from 8.63s / 1.854 tok/s to 7.12s / 2.247 tok/s while keeping RSS effectively bounded; Phase 7.11A validates the same generic GPT-NeoX/Pythia path on Pythia-160M without model-specific code: raw/tile-block pack emits 184 tensors / 3366 chunks / 367 MiB, verify passes 374,977,752 bytes losslessly, token `[12092]` generates `Hello!`, HF/PyTorch top-k parity passes (`top1_match=true`, top-10 overlap 10/10, max abs diff 0.02246094), and the 1/128/512/1024 × 1/4/16 matrix completes with 1024 + 16 at 31.08s / 0.515 tok/s / 99.47 MiB RSS / 1.04 MiB tracked transient; Phase 7.11B sweeps Pythia-160M prefill windows 8/16/32/64/128/256 and transient budget thresholds: for 1024 + 16, chunk=64 improves over default-32 from 31.23s / 0.512 tok/s to 28.22s / 0.567 tok/s at 99.02 MiB RSS, chunk=128 reaches 26.65s / 0.600 tok/s at ~100.06 MiB RSS but needs just under 4 MiB tracked transient, and chunk=256 barely improves speed while jumping to 107.20 MiB RSS
remaining: generic shape/budget-aware prefill policy is implemented in Phase 7.12A, and generic eight-row projection reuse is implemented in Phase 7.12B; next either pursue another measured dense-projection slice if fresh timing identifies a safe generic candidate, or start Phase 8 LLaMA-family architecture expansion with a new adapter rather than model-size-specific hacks; keep further MLP/QKV/decode/lm-head work evidence-driven; consider low-RAM parallel row-span accumulation only if short-prompt decode/lm-head becomes the priority; evaluate true intra-chunk compressed range decode only when measured trade-offs justify it
```

Deliverables:

```text
matrix weight chunks aligned to matmul tiles
decode tile
multiply
accumulate
release tile
```

Acceptance criteria:

```text
Output numerically identical to full decode mode for same dtype path.
Peak memory lower than layer decode mode.
```

---

### Phase 8 — Real model support

Deliverables:

```text
support one small real architecture first
load tokenizer
load config
run prompt
stream tokens
```

Recommended first real targets:

```text
very small GPT-style model
TinyLlama-class model only after toy runtime works
```

Acceptance criteria:

```text
Can run a real small model from .spsa.
Can verify decoded weights.
Can compare logits with reference implementation within exact/numeric tolerance.
```

---

## 10. Testing Requirements

### 10.1 Codec tests

Every codec must pass:

```text
empty input
small input
large input
random input
repeated input
structured tensor input
corrupted input handling
wrong checksum handling
```

### 10.2 Container tests

Must test:

```text
bad magic header
unsupported version
truncated file
wrong chunk offset
wrong checksum
metadata parse failure
unknown codec id
```

### 10.3 Verify tests

Must test:

```text
same model -> pass
changed byte -> fail
changed shape -> fail
changed dtype -> fail
missing tensor -> fail
extra tensor -> fail or warning depending mode
```

### 10.4 Runtime tests

Must test:

```text
full decode output
layer decode output
tile decode output
deterministic sampling with fixed seed
memory tracking
cache eviction
```

---

## 11. Performance Metrics

Track these metrics:

```text
original_size_bytes
compressed_size_bytes
compression_ratio
pack_time_seconds
unpack_time_seconds
decode_speed_mb_per_sec
peak_ram_full_decode
peak_ram_layer_decode
peak_ram_tile_decode
tokens_per_second_full_decode
tokens_per_second_layer_decode
tokens_per_second_tile_decode
```

Example benchmark output:

```text
Model: example-small
Original size: 1024 MB
Compressed size: 870 MB
Compression ratio: 84.9%
Pack time: 22.4s
Decode speed: 610 MB/s
Full decode peak RAM: 1450 MB
Layer decode peak RAM: 820 MB
Tile decode peak RAM: 540 MB
```

---

## 12. README Positioning

The README must be honest and strong.

Suggested README intro:

```markdown
# Spissa

Spissa is an experimental local LLM runtime built around lossless compressed model storage.
It stores model tensors in a chunked compressed container and aims to run inference by decoding only the tensor blocks needed at runtime.

Spissa does not claim magical compression. It preserves model weights exactly. If a model is already highly compressed or quantized, storage gains may be small. The long-term goal is lower peak memory through block-wise and tile-wise decoding.
```

Forbidden README claims:

```text
compress any LLM 10x without quality loss
run 70B on any phone with same quality
beats all existing runtimes
```

Allowed README claims:

```text
lossless tensor compression
bit-identical verification
chunked compressed model container
runtime-oriented compressed weights
experimental low-RAM inference modes
```

---

## 13. Developer Notes for AI Agent

The AI agent should follow these rules:

1. Build incrementally.
2. Prefer correctness over speed.
3. Never skip lossless verification.
4. Never silently use lossy compression.
5. Label all experimental features clearly.
6. Implement tests before optimizing.
7. Keep format versioned.
8. Add checksum everywhere important.
9. Avoid hidden dependency on Ollama/llama.cpp.
10. If compression makes a chunk larger, store it raw.
11. Keep CLI outputs human-readable.
12. Keep internal APIs documented.

---

## 14. Suggested Rust Interfaces

### 14.1 Codec trait

```rust
pub trait TensorCodec {
    fn id(&self) -> &'static str;
    fn encode(&self, input: &[u8], meta: &TensorMeta) -> Result<EncodedChunk>;
    fn decode(&self, encoded: &[u8], meta: &ChunkMeta) -> Result<Vec<u8>>;
}
```

### 14.2 Tensor metadata

```rust
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub original_size_bytes: u64,
    pub sha256: [u8; 32],
}
```

### 14.3 Chunk metadata

```rust
pub struct ChunkMeta {
    pub chunk_id: u64,
    pub tensor_id: u64,
    pub codec_id: String,
    pub file_offset: u64,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
    pub original_sha256: [u8; 32],
    pub compressed_sha256: [u8; 32],
}
```

### 14.4 Container reader

```rust
pub struct RllmReader {
    path: PathBuf,
    header: RllmHeader,
    tensors: Vec<TensorMeta>,
    chunks: Vec<ChunkMeta>,
}

impl RllmReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self>;
    pub fn list_tensors(&self) -> &[TensorMeta];
    pub fn read_chunk(&self, chunk_id: u64) -> Result<Vec<u8>>;
    pub fn decode_tensor(&self, tensor_name: &str) -> Result<Vec<u8>>;
}
```

---

## 15. Pseudocode: Pack

```text
function pack_model(input_model, output_file):
    model = read_model(input_model)
    writer = RllmWriter(output_file)

    writer.write_header(version=1)
    writer.write_global_metadata(model.metadata)
    writer.reserve_tensor_directory()
    writer.reserve_chunk_directory()

    for tensor in model.tensors:
        tensor_id = writer.add_tensor_metadata(tensor)
        chunks = split_into_chunks(tensor.bytes, chunk_size)

        for chunk in chunks:
            best = choose_best_lossless_codec(chunk, tensor.dtype)

            decoded = decode(best)
            assert decoded == chunk.bytes

            writer.write_chunk(
                tensor_id=tensor_id,
                codec_id=best.codec_id,
                compressed_bytes=best.bytes,
                original_hash=sha256(chunk.bytes),
                compressed_hash=sha256(best.bytes)
            )

    writer.finalize_directories()
    writer.write_footer_checksum()
```

---

## 16. Pseudocode: Verify

```text
function verify(original_model, compressed_file):
    original = read_model(original_model)
    spissa = RllmReader.open(compressed_file)

    for original_tensor in original.tensors:
        decoded = spissa.decode_tensor(original_tensor.name)

        if decoded.bytes != original_tensor.bytes:
            return FAIL

        if sha256(decoded.bytes) != original_tensor.sha256:
            return FAIL

    return LOSSLESS_VERIFIED
```

---

## 17. Pseudocode: Layer Decode Runtime

```text
function run(prompt):
    tokens = tokenizer.encode(prompt)
    state = init_state(tokens)

    for step in generation_steps:
        hidden = embed(tokens)

        for layer_id in model.layers:
            layer_weights = decode_layer_weights(layer_id)
            hidden = run_layer(hidden, layer_weights)
            release(layer_weights)

        logits = lm_head(hidden)
        next_token = sample(logits)
        tokens.append(next_token)

    return tokenizer.decode(tokens)
```

---

## 18. Pseudocode: Tile Decode Matmul

```text
function compressed_matmul(input_vector, compressed_weight_matrix):
    output = zeros(out_dim)

    for tile in compressed_weight_matrix.tiles:
        weight_tile = decode_tile(tile)
        partial = matmul(input_vector_slice(tile), weight_tile)
        output.accumulate(partial)
        release(weight_tile)

    return output
```

---

## 19. Important Technical Risks

### 19.1 Compression ratio may be small

If model is already quantized/compressed, lossless gains may be limited.

Mitigation:

```text
always show honest compression ratio
fallback to raw chunks
focus on runtime memory advantage, not only storage size
```

### 19.2 Runtime may be slower

Decoding during inference costs CPU cycles.

Mitigation:

```text
cache decoded chunks
layer decode mode
SIMD codec later
fused decode + matmul later
```

### 19.3 Large model support is hard

Do not start with 12B model. Start tiny.

Mitigation:

```text
toy tensors -> toy transformer -> small real model -> larger models
```

### 19.4 Exact output comparison can be tricky

Even with same weights, different kernels can produce slightly different floating point results.

Mitigation:

```text
first verify weights bit-identical
then compare logits with tolerance
then compare deterministic token generation in controlled runtime
```

---

## 20. Success Criteria

### Technical success v1

Project is successful if:

```text
1. It packs model tensors into .spsa.
2. It unpacks them exactly.
3. It verifies bit-identical weights.
4. It shows honest compression metrics.
5. It runs a tiny model from .spsa.
6. It supports at least full-decode runtime.
```

### Technical success v2

Project is stronger if:

```text
1. It supports safetensors.
2. It supports layer-wise decode.
3. It reduces peak RAM compared to full decode.
4. It can run a small real transformer model.
```

### Technical success v3

Project becomes genuinely unique if:

```text
1. It supports tile-wise compressed inference.
2. It avoids full decompression of large tensors.
3. It has fused decode + matmul kernels.
4. It can run useful local LLMs with lower peak RAM.
```

---

## 21. First Task for AI Agent

Start with this exact task:

```text
Create a Rust workspace for Spissa with these crates:
- spissa-cli
- spissa-container
- rtc-codec

Implement:
1. CLI skeleton with clap.
2. `spissa pack`, `spissa inspect`, `spissa unpack`, `spissa verify` command stubs.
3. Spissa header struct.
4. TensorMeta and ChunkMeta structs.
5. rtc-raw-v1 codec.
6. Unit test proving decode(encode(bytes)) == bytes.
7. A small fake tensor pack/unpack flow.

Do not implement real LLM inference yet.
Do not use Ollama or llama.cpp.
Focus only on correctness and file format skeleton.
```

---

## 22. Founder Intent

The founder wants a tool that feels different from existing local LLM runners.

The soul of the project:

```text
Not just another model runner.
Not just another quantizer.
Not just a wrapper.

A from-scratch local LLM runtime designed around compressed weights.
```

The project should be ambitious, but honest.
