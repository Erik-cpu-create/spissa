# Runtime-compressed Local LLM (RLLM)

Brain-inspired compressed runtime for local LLMs, guided by RAMA: Rama Active Memory Architecture.

RLLM is an experimental local LLM runtime built around **lossless compressed model storage**. It stores model tensors in a chunked compressed container (`.rllm`) and aims to run inference by decoding only the tensor blocks needed at runtime.

> Run local LLMs from compressed weights without changing model weights.

## Architecture Direction

RLLM uses **RAMA**: **Rama Active Memory Architecture**.

RAMA frames RLLM as a memory-first runtime:

- compressed `.rllm` tensors are dormant long-term memory
- chunk/layer/tile decode is selective recall
- `MemoryBudget` is bounded working memory
- KV-cache is short-term context memory
- streaming inference is the active thought loop

The future focused recall subsystem name is reserved as **ERIK**: **Episodic Recall Inference Kernel**.

See [`docs/rllm-rama-architecture.md`](docs/rllm-rama-architecture.md) for the Phase 5D.5 architecture boundary, originality doctrine, and RAMA/ERIK naming contract. The previous ECHO/EMBER doc is retained only as a superseded compatibility pointer.

## What RLLM Does

- Stores model tensors in a chunked compressed container format (`.rllm`)
- Uses **RTC (RLLM Tensor Codec)** — in-house lossless tensor compression codecs
- Verifies decoded weights are **bit-identical** to originals
- Imports safetensors model files into `.rllm`
- Reports honest compression metrics — no magic claims

## What RLLM Does NOT Do

- ❌ Claim magical compression (for example, “7.6GB → 500MB with same quality”)
- ❌ Perform lossy quantization unless explicitly labeled as a lossy optimization
- ❌ Wrap Ollama, llama.cpp, or another existing runtime
- ❌ Change model weights in any way
- ❌ Use external generic compression libraries by default; RTC codecs are custom/in-house unless explicitly approved
- ❌ Claim to simulate a biological brain, consciousness, or self-learning cognition

## Current Status

Implemented:

- Cargo workspace with five crates
- `.rllm` v1 container reader/writer
- Safetensors import
- CLI commands: `pack`, `inspect`, `verify`, `run`, `doctor`
- Stubbed future commands: `import`, `benchmark`
- RTC codecs:
  - `rtc-raw-v1`
  - `rtc-rle-v1`
  - `rtc-huff-v1` (custom byte-level Huffman)
- Per-chunk SHA-256 verification
- Multi-tensor safetensors verification
- Phase 5A runtime foundation:
  - full-decode `.rllm` loader
  - fp16/bf16/f32/int runtime conversion to f32
  - basic tensor ops: embedding lookup, matmul, linear, layernorm, RMSNorm, GELU, softmax, attention, MLP, argmax/top-p sampling
- Phase 5B low-memory runtime planning:
  - metadata-only `.rllm` open path
  - memory budget accounting
  - layer-stream/tile-stream dry-run planner
  - `rllm run --memory-budget --ctx --mode tile-stream`
  - chunk-scoped streaming linear kernel (`decode chunk → f32 scratch → accumulate → release`)
  - streaming MLP sub-block (`linear → GELU → linear`) with budgeted intermediate activation
  - streaming attention/QKV sub-block (`QKV projection → split heads → attention → output projection`)
  - streaming pre-norm transformer block skeleton (`LN → attention → residual → LN → MLP → residual`)
  - tiny end-to-end next-token smoke path (`embedding → one block → final LN → lm_head → sample`)
  - GPT-NeoX/Pythia-style rotary embeddings and KV-cache attention primitives
  - RAMA architecture spec for memory-first, brain-inspired runtime direction
  - first executable cached generation loop: tiny multi-step token-ID generation with explicit prefill/decode-step paths and ContextEcho KV-cache state
  - multi-layer token-ID stack: per-layer ContextEcho KV caches, prefill/decode/generate over all configured streaming blocks, and logits checked against full-context recomputation
  - GPT-NeoX/Pythia adapter that infers standard tensor names/shapes from `.rllm` metadata and decodes small norm/bias tensors into an owned prepared stack
  - optional original `config.json` metadata persistence for GPT-NeoX/Pythia fields (`num_attention_heads`, rotary settings, layer norm eps, context length) plus runtime auto-prepare from that metadata
  - optional tokenizer vocabulary/config metadata persistence from `tokenizer.json`, plus tokenizer-backed text smoke generation via `rllm run --prompt ...`
  - Phase 6 RAMA layer-decode GPT-NeoX/Pythia path: keeps only names + final params resident, decodes per-layer norm/bias vectors just-in-time, budgets active layer params, and matches the prepared stack baseline
  - Phase 7 RAMA tiled runtime routing: streaming MLP/attention/transformer blocks and tiny/RAMA/GPT-NeoX generation heads route through the fused tile-linear path, converting only bounded weight tiles into f32 scratch while preserving logits/token baselines
  - Phase 7.6 release benchmark matrix for local Pythia-70M with persisted `config.json` + `tokenizer.json`: `ctx=128/512/1024`, `max-new-tokens=1/4/8/16`, all runs pass under `--memory-budget 100mb`, measured RSS 88.62–94.62 MiB with ~4.47–5.29s/token after Phase 7.7 fidelity fixes
  - Phase 7.7 fixed-token HF/PyTorch logits comparison: RLLM top-1/top-10 match on tested Pythia-70M prompts after persisting `use_parallel_residual` and fixing GPT-NeoX per-head QKV split
  - Phase 7.8 tile-block artifact benchmark: local Pythia-70M repacked with `--tile-block-elements 65536` runs the same `ctx=128/512/1024` × `max-new-tokens=1/4/8/16` matrix at 18.39–22.64 MiB RSS with 386 KiB tracked transient peak while preserving tested HF logits parity
  - Phase 7.9A RAMA trace profiler: `rllm run --rama-trace <path>` emits chunk recall timing JSON; real Pythia-70M one-token trace shows `chunk_decode` dominates recorded time (~3716 ms of 4505 ms), especially `gpt_neox.embed_in.weight` and `embed_out.weight`, while disk read is only ~32 ms
  - Phase 7.9B RAMA embedding row recall: embedding lookup now recalls only touched token rows/chunks; the same 12-run tile-block matrix improved from 5.07 to 2.93 average seconds/token (~1.73× average speedup) while max RSS stayed ~22 MiB and HF logits parity remained intact
  - Phase 7.9C RAMA low-ram-fast runtime layout: `rllm pack --codec raw --tile-block-elements 65536` builds a compute-ready raw/tile-block artifact; `rllm run --rama-integrity verify-once` verifies chunks once per process. The same 12-run matrix reached 0.35 average seconds/token / 3.26 average tok/s / 4.35 best tok/s while RSS stayed 19.17–23.36 MiB and HF parity remained intact.
  - Phase 7.9D real long-prompt benchmark: actual `--token-ids` prompt lengths `1/128/512/1024` with `ctx=2048` expose the next bottleneck. Short prompt remains 4.30 tok/s at 20.67 MiB RSS, but 512-token prompt + 16 generated tokens drops to 0.30 tok/s at 44.98 MiB RSS, and 1024-token prompt + 16 generated tokens drops to 0.15 tok/s at 70.84 MiB RSS; 128-token and 512-token HF logits parity still pass top-1/top-10.
  - Phase 7.9E RAMA chunked prefill: `rllm run --rama-timing <path>` adds aggregate timing and `--rama-prefill-chunk-tokens 64` bounds long-prompt prefill windows. It improved 512-token + 16 generated from 56.43s / 0.284 tok/s / 46.22 MiB RSS to 35.20s / 0.455 tok/s / 34.05 MiB RSS, and 1024-token + 16 generated from 110.29s / 0.145 tok/s / 70.55 MiB RSS to 63.84s / 0.251 tok/s / 44.98 MiB RSS.
  - Phase 7.10A row-span linear accumulation: the tiled-linear hot loop now accumulates contiguous row spans instead of doing per-weight division/modulo. Short prompt + 16 generated improved from 3.65 to 7.10 tok/s at ~20.66 MiB RSS; 512-token chunked prefill improved to 2.26 tok/s at ~32.98 MiB RSS; 1024-token chunked prefill improved to 1.25 tok/s at ~44.80 MiB RSS while 512-token HF parity remained top-1/top-10 matched.
  - Phase 7.10B RAMA prefill homeostasis: broader post-7.10A matrix reaches 9.756 tok/s for 1-token prompt + 16 generated at 20.33 MiB RSS; measured prefill chunk sweep chooses 32 real input tokens as the default CLI prefill window. Best swept long-prompt rows: 512-token + 16 at 2.3495 tok/s / 32.77 MiB RSS and 1024-token + 16 at 1.1653 tok/s / 44.91 MiB RSS.
  - Phase 7.12A generic shape/budget-aware prefill policy: `rllm run` now defaults to auto low-RAM prefill selection from GPT-NeoX/Pythia shape and explicit transient budget, preserving Pythia-70M-like 32-token defaults while selecting 64 for Pythia-160M-like low-RAM runs; `--rama-prefill-policy speed` selects the speed-biased larger window when budget allows.
  - Phase 7.12B generic eight-row projection reuse: the shared tiled-linear hot loop now reuses each decoded weight row fragment across 8 prompt-token rows before falling back to the existing 4-row/scalar tails, improving the measured Pythia-160M 512-token speed-policy row from 13.21s to 12.34s wall time while keeping tracked transient memory unchanged at 3.79 MiB.
  - R57 edge attention locality cache for the Llama experimental-speed path: optional per-layer recent-index caches can reuse a tiny number of previous edge-attention input features for sparse Q/K/V. The retained `window=8, extra=1` preset preserved the 30 tok/s floor on Llama 3.2 1B Instruct and slightly improved cheap quality counters, while the wider `window=16, extra=4` probe was rejected.


Not yet implemented:

- Production-grade tokenizer/normalizer fidelity beyond the current runtime-ready vocabulary metadata
- Multi-tensor unpack back to safetensors layout
- True intra-chunk compressed range decode/routing for non-identity codecs
- Comfortable 512/1024-token chat latency; Phase 7.12B improves generic projection throughput, but 512/1024-token prompts are still not interactive

## Quick Start

```bash
# Build
cargo build

# Run tests
cargo test

# Check CLI/system info
cargo run -- doctor

# Interactive LLaMA-family token-native session
cargo run --release -p rllm-cli --bin llama-test -- \
  --model models/SmolLM2-135M-raw.rllm \
  --ctx 2048 \
  --max-new-tokens 64

# Llama exact/raw baseline. This is intentionally slow on CPU-only and is used
# as the quality/reference path, not the 30-40 tok/s experimental speed path.
cargo run --release -p rllm-cli --bin llama-test -- \
  --model models/Llama-3.2-1B-Instruct-raw.rllm \
  --ctx 2048 \
  --max-new-tokens 64

# Experimental speed artifact for Llama 3.2 1B. Use this artifact and the env
# flags below when testing the sparse research path; running the raw artifact
# without these flags will stay on the slow exact path.
cargo build --release -p rllm-cli --bin rllm --bin llama-test
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
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64

# Current R57 experimental speed-floor preset. This builds on the R43 sparse
# path with a tiny edge-attention locality cache. It is still a research path:
# output quality remains a limitation and should not be treated as chat-ready.
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI=100 \
  RLLM_AIP_ATTENTION_LOCALITY_WINDOW=8 \
  RLLM_AIP_ATTENTION_LOCALITY_EXTRA=1 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64

# Optional projection-specific AIP knobs for R31/R32 experiments:
# RLLM_AIP_ATTENTION_TOPK, RLLM_AIP_MLP_TOPK, RLLM_AIP_DOWN_TOPK,
# RLLM_AIP_LM_HEAD_TOPK.
# Optional R32 layer-edge probes:
# RLLM_AIP_EDGE_LAYERS, RLLM_AIP_EDGE_TOPK.
# Optional R33 repeat-margin controller:
# RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=500
# Optional R35 adaptive repeat-margin controller:
# RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=50
# RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1
# Optional R36 phrase novelty controller:
# RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75
# RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1
# RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4
# Optional R37 confidence-gated novelty:
# RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100
# Optional R48 exact prompt prefill probe:
# RLLM_AIP_EXACT_PREFILL=1
# Optional R50 periodic exact LM-head diagnostic probe. This is expensive and
# rejected as a speed preset; use only when measuring sparse-vs-exact token
# selection drift.
# RLLM_AIP_LM_HEAD_EXACT_EVERY=8
# Optional R53/R54 confidence-gated candidate rescore probes. R54 preserves
# repeat/novelty controllers, but this path is still rejected as chat-ready.
# RLLM_AIP_LM_HEAD_RESCORE=2
# RLLM_AIP_LM_HEAD_RESCORE_GAP_MILLI=250
# Optional R55 exact edge-layer hidden-state calibration probe. It improves
# diversity/repetition but is rejected as a speed preset.
# RLLM_AIP_EXACT_EDGE_LAYERS=1
# Optional R56 projection-filtered exact edge probe:
# RLLM_AIP_EXACT_EDGE_PROJECTION=attention
# Accepted values: all, attention, attn, mlp-gate-up, gate-up, gateup,
# mlp-down, down.
# Optional R57 edge attention locality cache. Retained preset:
# RLLM_AIP_ATTENTION_LOCALITY_WINDOW=8
# RLLM_AIP_ATTENTION_LOCALITY_EXTRA=1
# Rejected wide probe:
# RLLM_AIP_ATTENTION_LOCALITY_WINDOW=16
# RLLM_AIP_ATTENTION_LOCALITY_EXTRA=4

# Force the runtime worker count for CPU-only benchmarks.
RLLM_THREADS=1 cargo run --release -p rllm-cli --bin llama-test -- \
  --model models/SmolLM2-135M-raw.rllm \
  --ctx 2048 \
  --max-new-tokens 16

# Pack a safetensors model into .rllm
cargo run -- pack ./models/pythia-70m/model.safetensors \
  --out ./models/pythia-70m.rllm \
  --chunk-size 32mb

# Inspect metadata and compression stats
cargo run -- inspect ./models/pythia-70m.rllm

# Verify lossless round-trip against the original safetensors file
cargo run -- verify ./models/pythia-70m/model.safetensors ./models/pythia-70m.rllm

# Full-decode runtime smoke test (loads tensors into f32 runtime memory)
cargo run -- run ./models/pythia-70m.rllm

# Low-RAM tile-stream planning (metadata only; no full tensor decode)
cargo run -- run ./models/pythia-70m.rllm \
  --memory-budget 100mb \
  --ctx 1024 \
  --mode tile-stream

# Token generation + external RSS measurement on macOS release binary
cargo build --release
/usr/bin/time -l target/release/rllm run ./models/pythia-70m.rllm \
  --prompt 'Hello' \
  --max-new-tokens 1 \
  --ctx 128 \
  --memory-budget 100mb

# Repeatable Phase 7.6 release RSS benchmark matrix
python3 scripts/phase76_release_rss_benchmark.py \
  --tokens 1,4,8,16 \
  --ctx 128,512,1024 \
  --memory-budget 100mb

# Phase 7.7 fixed-token HF/PyTorch logits comparison
uv run --with torch --with transformers --with safetensors \
  scripts/phase77_compare_logits.py \
  --token-ids 12092,13 \
  --ctx 128 \
  --memory-budget 100mb

# Phase 7.9A RAMA trace profiler for generation bottlenecks
target/release/rllm run ./models/pythia-70m-phase78d-tileblocks.rllm \
  --token-ids 12092 \
  --max-new-tokens 1 \
  --ctx 128 \
  --memory-budget 100mb \
  --rama-trace target/phase79a/rama_trace.json

# Phase 7.9C low-ram-fast raw/tile-block benchmark profile
target/release/rllm pack ./models/pythia-70m/model.safetensors \
  --out ./models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.rllm \
  --codec raw \
  --tile-block-elements 65536 \
  --config ./models/pythia-70m/config.json \
  --tokenizer ./models/pythia-70m/tokenizer.json
python3 scripts/phase79c_low_ram_fast_benchmark.py \
  --skip-pack \
  --skip-verify \
  --rama-integrity verify-once \
  --tokens 1,4,8,16 \
  --ctx 128,512,1024 \
  --memory-budget 100mb

# Phase 7.9D real long-prompt benchmark (actual input token counts)
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 1,128,512,1024 \
  --max-new-tokens 1,4,16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --memory-budget 100mb

# Phase 7.9E RAMA chunked prefill timing/optimization benchmark
python3 scripts/phase79e_prefill_timing_benchmark.py \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --prefill-chunks full,64 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --memory-budget 100mb

# Phase 7.10B RAMA prefill homeostasis / post-rowspan matrix
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 1,128,512,1024 \
  --max-new-tokens 1,4,16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --memory-budget 100mb

# Phase 7.10C deep prefill timing benchmark
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --out-dir target/phase710c-deep-prefill-timing \
  --memory-budget 100mb

# Phase 7.10D MLP prefill row-reuse benchmark
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --out-dir target/phase710d-four-batch-optimized \
  --memory-budget 100mb

# Phase 7.10E attention row-slice timing/optimization benchmark
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --out-dir target/phase710e-attention-row-slice-optimized \
  --memory-budget 100mb

# Phase 7.11A Pythia-160M GPT-NeoX/Pythia-family scale validation
hf download EleutherAI/pythia-160m \
  model.safetensors config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
  --local-dir models/pythia-160m
target/release/rllm pack models/pythia-160m/model.safetensors \
  --out models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.rllm \
  --codec raw \
  --tile-block-elements 65536 \
  --config models/pythia-160m/config.json \
  --tokenizer models/pythia-160m/tokenizer.json
target/release/rllm verify \
  models/pythia-160m/model.safetensors \
  models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.rllm
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --artifact models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.rllm \
  --input-tokens 1,128,512,1024 \
  --max-new-tokens 1,4,16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --memory-budget 100mb

# Phase 7.11B Pythia-160M prefill window / memory-budget sweep
python3 scripts/phase79e_prefill_timing_benchmark.py \
  --bin target/release/rllm \
  --artifact models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.rllm \
  --out-dir target/phase711b-pythia160m/chunk-sweep \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --prefill-chunks 8,16,32,64,128,256 \
  --ctx 2048 \
  --memory-budget 100mb \
  --rama-integrity verify-once
```

`models/` is ignored because model files and `.rllm` outputs are large reproducible local artifacts.
See [docs/phase76-release-rss-benchmark.md](docs/phase76-release-rss-benchmark.md) for the measured matrix.
See [docs/phase77-hf-logits-comparison.md](docs/phase77-hf-logits-comparison.md) for the HF/PyTorch reference comparison.
See [docs/phase78-range-decode-foundation.md](docs/phase78-range-decode-foundation.md) for the range-decode/checksum foundation.
See [docs/phase78-tileblock-rss-benchmark.md](docs/phase78-tileblock-rss-benchmark.md) for the measured tile-block RSS matrix.
See [docs/phase79-rama-trace-profiler.md](docs/phase79-rama-trace-profiler.md) for the opt-in RAMA trace profiler and bottleneck evidence.
See [docs/phase79-embedding-row-recall.md](docs/phase79-embedding-row-recall.md) for the measured selective embedding row recall speedup.
See [docs/phase79-low-ram-fast-runtime-layout.md](docs/phase79-low-ram-fast-runtime-layout.md) for the measured raw/tile-block low-ram-fast profile and verify-once integrity benchmark.
See [docs/phase79-long-prompt-benchmark.md](docs/phase79-long-prompt-benchmark.md) for the real long-prompt prefill/context benchmark and next bottleneck evidence.
See [docs/phase79e-rama-prefill-chunking.md](docs/phase79e-rama-prefill-chunking.md) for chunked prefill timing and long-prompt improvements.
See [docs/phase710a-row-span-linear-accumulation.md](docs/phase710a-row-span-linear-accumulation.md) for row-span tiled-linear optimization.
See [docs/phase710b-rama-prefill-homeostasis.md](docs/phase710b-rama-prefill-homeostasis.md) for the post-rowspan matrix and measured 32-token prefill default.
See [docs/phase710c-deep-prefill-timing.md](docs/phase710c-deep-prefill-timing.md) for deep prefill sub-phase timing and the measured next bottleneck.
See [docs/phase710d-mlp-prefill-row-reuse.md](docs/phase710d-mlp-prefill-row-reuse.md) for MLP prefill split timing, rejected optimizations, and the accepted four-row accumulation reuse speedup.
See [docs/phase710e-attention-row-slice.md](docs/phase710e-attention-row-slice.md) for attention split timing, rejected softmax optimization, and the accepted K/V row-slice score/context speedup.
See [docs/phase711a-pythia160m-scale-validation.md](docs/phase711a-pythia160m-scale-validation.md) for GPT-NeoX/Pythia-family scale validation on Pythia-160M.
See [docs/phase711b-pythia160m-prefill-window-sweep.md](docs/phase711b-pythia160m-prefill-window-sweep.md) for the Pythia-160M prefill window and transient memory-budget sweep.
See [docs/phase712a-shape-budget-prefill-policy.md](docs/phase712a-shape-budget-prefill-policy.md) for the generic shape/budget-aware RAMA prefill policy.
See [docs/phase712b-eight-row-projection-reuse.md](docs/phase712b-eight-row-projection-reuse.md) for the generic eight-row projection reuse optimization.


## Architecture

```text
rllm/
├── crates/
│   ├── rllm-cli/        # CLI binary (`rllm`)
│   ├── rllm-container/  # .rllm binary format parser/writer
│   ├── rllm-import/     # Safetensors import
│   ├── rllm-runtime/    # Lazy/streaming runtime, session cache, tensor ops
│   │   └── src/streaming/
│   │       ├── linear.rs      # streaming/tiled linear projections
│   │       ├── argmax.rs      # LM-head argmax + CPU-aware row parallelism
│   │       ├── kernels.rs     # raw fp16/bf16 row kernels
│   │       ├── mlp.rs         # streaming MLP helpers
│   │       ├── attention.rs   # streaming attention helpers
│   │       ├── block.rs       # transformer block orchestration
│   │       └── validation.rs  # shape/config validation
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
| `rllm-runtime` | Lazy/runtime loader, memory-budgeted planner, streaming kernels, session cache, tensor ops |
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
| `layer-decode` | Decode GPT-NeoX/Pythia layer params just-in-time, release after each layer | ✅ Phase 6 partial |
| `tile-decode` | Bounded tile f32 scratch routed through MLP/attention/RAMA generation; codec-level range decode still pending | ✅ Phase 7 partial |
| `fused-decode-matmul` | Codec-level range decode + multiply in one step | 🔜 Future |

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
