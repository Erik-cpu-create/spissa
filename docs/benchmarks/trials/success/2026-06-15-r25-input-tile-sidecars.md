# Trial: R25 Input-Tile Sidecars

Date: 2026-06-15
Owner: RLLM
Status: success with quality limitation
Folder: success

## Hypothesis

R20-R22 showed that sparse AIP reduces arithmetic but still loses time to
row-major weight traffic, resident f32 sidecars, and LM-head scans. R25 tests
input-major sidecar tensors on disk. The sidecar keeps the model values
unchanged, but stores selected input features as contiguous ranges so runtime
can read only the feature columns used by AIP.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Source model: `models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Input-tile gate: `RLLM_AIP_INPUT_TILES=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Sidecar coverage: MLP projections, attention projections, and tied LM head
- Sidecar layout: raw BF16 input-major `[in_features, out_features]`
- Sidecar chunking: `--input-tile-features 16`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin rllm --bin llama-test
```

Pack artifact:

```bash
target/release/rllm pack \
  models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors \
  --out models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
  --codec raw \
  --chunk-size 1mb \
  --config models/downloads/llama-3.2-1b-instruct-unsloth/config.json \
  --tokenizer models/downloads/llama-3.2-1b-instruct-unsloth/tokenizer.json \
  --llama-mlp-input-tiles \
  --llama-attention-input-tiles \
  --llama-lm-head-input-tiles \
  --input-tile-features 16
```

Pack result:

- input-tile sidecars: 113 tensors
- sidecar chunks: 20608
- feature ranges: 329728
- artifact original/compressed size: 4943122432 bytes
- pack time: 44.66s real
- pack max RSS: 1268695040 bytes

Benchmark command shape:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=<k> \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens <n> \
    --profile-phases
```

## Results

| artifact | variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | max top-k | input-tile reads | input-tile bytes | repetition ratio | unique tokens | RLLM peak transient | max RSS | peak footprint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw R22 baseline | AIP speed top-k 128 | 16 | 13.61s | 0.28 | 0.24 | 128 | 0 | 0 | 0.93 | 2/16 | 1050689536 | 2460532736 | 1620905248 |
| R23 MLP sidecar | top-k 128 | 16 | 12.57s | 1.84 | 0.77 | 128 | 92160 | 1132462080 | 0.93 | 2/16 | 1050689536 | 2465792000 | 1942457800 |
| R23 MLP sidecar | top-k 32 | 16 | 13.20s | 3.12 | 0.89 | 32 | 23040 | 283115520 | 0.93 | 2/16 | 1050689536 | 2107539456 | 1944851112 |
| R24 MLP+attention sidecar | top-k 4, lm-head prefix 512 | 16 | 11.33s | 46.70 | 1.37 | 4 | 6720 | 45219840 | 0.93 | 2/16 | 1050689536 | 2054389760 | 2156057064 |
| R25 all sidecars full vocab | top-k 4 | 16 | 11.44s | 23.37 | 1.32 | 4 | 6784 | 61636608 | 0.40 | 5/16 | 1050689536 | 2682290176 | 2157989416 |
| R25 all sidecars full vocab | top-k 3 | 16 | 11.40s | 29.95 | 1.34 | 3 | 5088 | 46227456 | 0.60 | 5/16 | 1050689536 | 2115141632 | 2156613400 |
| R25 all sidecars full vocab | top-k 3 | 64 | 11.50s | 44.89 | 4.96 | 3 | 21360 | 191692800 | 0.76 | 11/64 | 1050689536 | 1994129408 | 2164741496 |
| R25 all sidecars full vocab | top-k 4 | 64 | 11.25s | 37.13 | 4.94 | 4 | 28480 | 255590400 | 0.62 | 10/64 | 1050689536 | 2251915264 | 2158579936 |

Representative R25 top-k 4, 64-token profile:

| decode phase | time |
|---|---:|
| decode total | 1696.90ms |
| transformer | 1652.69ms |
| attention total | 713.69ms |
| MLP total | 938.06ms |
| LM-head | 43.40ms |
| q projection | 201.68ms |
| k projection | 157.62ms |
| v projection | 150.70ms |
| gate projection | 512.26ms |
| down projection | 423.66ms |

## Analysis

The speed gate is reached in sustained decode. R25 top-k 3 and top-k 4 both
exceed 30 tok/s over 64 generated tokens while keeping RLLM tracked transient
peak at `1050689536` bytes. The final verification rerun measured 37.13 tok/s;
an earlier run of the same artifact measured 54.18 tok/s, so follow-up paper
numbers should include repeated samples rather than a single best case.

The key change is that MLP, attention, and tied LM-head projections no longer
scan row-major weight chunks for selected dimensions. Runtime reads exact
feature ranges from input-major sidecars through range checksum metadata.

The tied LM-head sidecar is important. Without it, top-k 4 full-vocab decode
was only 8.30 tok/s because exact LM-head still dominated. With the tied
`model.embed_tokens.weight` sidecar, top-k 4 full-vocab reached 23.37 tok/s on
16 tokens and 37.13 tok/s on 64 tokens in the final verification rerun.

Quality is not solved. The measured text is repetitive and not chat-quality:

- top-k 4, 64 tokens repeats short fragments such as `mir` and `.swing`
- top-k 3, 64 tokens has repetition ratio 0.76
- top-k 2 reaches 70.00 tok/s but collapses to repeated `sob`

This means R25 is a speed-layout success, not a final user-facing inference
quality solution.

## Decision

success with quality limitation

Reason: R25 exceeds the requested 30-40 tok/s speed range for Llama 3.2 1B
Instruct in CPU-only experimental mode, using full-vocab sparse LM-head rather
than LM-head prefix. However, output quality remains unacceptable for chat.

Paper value:
- positive evidence that input-major sidecar layout can cross 30 tok/s on
  Llama 3.2 1B CPU-only
- positive evidence that full-vocab sparse LM-head removes the LM-head ceiling
  without prefix-vocab truncation
- limitation evidence that extreme activation top-k policies break language
  quality
- useful data point that low transient RAM can coexist with larger disk layout
  sidecars

## Next Experiment

R26 should attack quality while preserving the R25 layout:

- separate top-k per projection type, for example attention top-k 8, MLP top-k
  4, LM-head top-k 16
- adaptive top-k based on activation entropy or residual norm
- exact edge layers plus sparse middle layers
- full-vocab sparse LM-head with larger top-k than transformer projections
- compare generated token match or perplexity proxy against exact mode before
  calling the path chat-ready
