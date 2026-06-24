# Spissa — Runtime-compressed Local LLM

> **compressed · local · yours**

Brain-inspired compressed runtime for local LLMs, guided by RAMA: Rama Active Memory Architecture.

Spissa is a from-scratch local LLM runtime built around **runtime-compressed model storage** — **lossless by default** (rANS / bit-plane), with optional lossy quantization (q8 / q4). It stores model tensors in a chunked compressed container (`.spsa`) and runs inference by decoding only the tensor blocks needed at runtime. One self-contained binary, no dependencies, runs on any device.

> Run local LLMs from compressed weights without changing model weights.

*(Spissa is Latin for "dense / packed" — the project was formerly named RLLM. The `spissa` CLI replaces the old `rllm` command; the `.spsa` container format is unchanged.)*

## Architecture Direction

Spissa uses **RAMA**: **Rama Active Memory Architecture**.

RAMA frames Spissa as a memory-first runtime:

- compressed `.spsa` tensors are dormant long-term memory
- chunk/layer/tile decode is selective recall
- `MemoryBudget` is bounded working memory
- KV-cache is short-term context memory
- streaming inference is the active thought loop

The future focused recall subsystem name is reserved as **ERIK**: **Episodic Recall Inference Kernel**.

See [`docs/rllm-rama-architecture.md`](docs/rllm-rama-architecture.md) for the Phase 5D.5 architecture boundary, originality doctrine, and RAMA/ERIK naming contract. The previous ECHO/EMBER doc is retained only as a superseded compatibility pointer.

## What Spissa Does

- Stores model tensors in a chunked compressed container format (`.spsa`)
- Uses **RTC (Rama Tensor Codec)** — in-house lossless tensor compression codecs
- Verifies decoded weights are **bit-identical** to originals
- Imports safetensors model files into `.spsa`
- Reports honest compression metrics — no magic claims

## What Spissa Does NOT Do

- ❌ Claim magical compression (for example, “7.6GB → 500MB with same quality”)
- ❌ Perform lossy quantization unless explicitly labeled as a lossy optimization
- ❌ Wrap Ollama, llama.cpp, or another existing runtime
- ❌ Change model weights in any way
- ❌ Use external generic compression libraries by default; RTC codecs are custom/in-house unless explicitly approved
- ❌ Claim to simulate a biological brain, consciousness, or self-learning cognition

## Chat

Spissa runs an **interactive, multi-turn chat** on CPU. It is **codec-agnostic** — run
fully **lossless** weights (rANS / bit-plane) or lossy **q8** (fastest); the model loads
once and keeps a **resident KV cache** across turns, so each message prefills only the
new text. Architecture (Gemma / Llama) is auto-detected from the packed metadata.

```bash
# canonical command — any packed model, any codec:
./target/release/spissa chat models/gemma-3-1b-it-q8-raw.spsa --fast   # q8, fastest
./target/release/spissa chat models/gemma-3-1b-it-rans.spsa            # rANS, LOSSLESS
./target/release/spissa chat <model.spsa> --low-ram                    # stream embedding (>RAM regime)

# convenience wrappers (Gemma 4B / Llama 1B|3B):
./try-gemma.sh chat
./try-llama.sh chat
./try-llama.sh -m 3b chat

# one-shot
./try-gemma.sh "What is the capital of Australia?"
```

In chat: just type. `/reset` starts a new conversation, `/exit` (or `exit`/`quit`) ends it.
`--fast` is q8 turbo (see below). On big.LITTLE ARM (Apple Silicon, phones) the decode
GEMV **auto-pins to the performance cores** (~2× vs all-cores; override with
`RLLM_THREADS`). Note: lossless rANS holds up better over long multi-turn than lossy q8,
which can degrade into empty replies on a tiny model.

### `--fast` mode

The wrappers always pass `--fast`, which turns on two levers that only pay off
together (so it is opt-in, not the default):

- **residency** — `mlock` the weight mmap so the OS cannot evict it (also issues
  `MADV_WILLNEED`); without this the int8 kernels stall on page faults.
- **int8-activation kernels** — quantize the activation to int8 and use NEON
  `sdot`/`i8mm` (near-exact, quant-only diff vs the exact scalar path, the same
  approach llama.cpp uses for q8). The integrity SHA-256 pass is front-loaded in
  parallel at startup.

Everything stays **lossless**: weights are read exactly, only the f32 activation
is quantized, and per-byte SHA-256 integrity is still verified once per run
(something Ollama/llama.cpp do not do).

### Supported models

| Model | Pack | Notes |
|---|---|---|
| Gemma 3 1B Instruct | `gemma-3-1b-it-{q8-raw,rans}.spsa` | q8 (fast) or rANS (lossless); best for phones |
| Gemma 3 4B Instruct | `gemma-3-4b-it-q8.spsa` | tied bf16 embedding, sandwich-norm, dual-RoPE |
| Llama 3.2 1B / 3B Instruct | `Llama-3.2-{1B,3B}-Instruct-q8….spsa` | tied bf16 embedding |

Each codec (`--codec raw` for q8, `--codec rans`/`bitplane` for lossless bf16) packs the
same model; pick by the size/speed/lossless trade-off you want.

### Performance (Apple A18 Pro, CPU, vs Ollama CPU same chip)

Honest measured numbers in `--fast` mode (decode is steady-state):

| | Gemma 3 4B | Llama 3.2 1B | Llama 3.2 3B |
|---|---|---|---|
| Prefill | ~36 tok/s (≈ Ollama parity) | ~0.5s/turn | ~1.4s/turn |
| Decode | ~8 tok/s | ~17 tok/s | ~7–8 tok/s |
| RAM | ~5 GB | ~1.05 GB (< Ollama 1.8 GB) | ~1.6 GB |

Output is coherent and on-par with Ollama (no hallucination). Remaining decode gap
vs Ollama is the 2-performance-core hardware ceiling, not kernel quality.

### Devices (universal)

Spissa is built to run on **all devices, not just Apple Silicon**. The `.spsa` file format
is platform-independent (copy it as-is), and the kernels are `cfg`-gated with fallbacks:

| Device | Status |
|---|---|
| ARM laptop (Apple Silicon) | ✅ fully optimized — NEON / `sdot` / `smmla` |
| Android (Snapdragon / MediaTek) | ✅ same ARM path; big.LITTLE perf-cores auto-detected |
| ARM SBC / server | ✅ optimized |
| x86 laptop (Intel / AMD) | ⚠️ compiles & runs, but **scalar** (AVX kernels are future work) |

The fast paths are portable ARM (NEON/i8mm), never an Apple-only library, so the same
optimizations carry from a Mac to a phone. To run on **Android**, build natively in
Termux — see [docs/android-termux.md](docs/android-termux.md). Use a 1B model on phones
(`gemma-3-1b-it-q8-raw.spsa` 1.38 GB, or `…-rans.spsa` 1.36 GB lossless); the 4B is too
big for most phones' usable RAM.

### Packing your own model

```bash
# Download a HF checkpoint (safetensors + config.json + tokenizer.json), then:
./target/release/spissa pack <model.safetensors | *.index.json | model-dir> \
  --out models/<name>-q8.spsa \
  --quantize q8_transformer_keep_io \
  --codec raw          # raw (rtc-raw-v1) is required for the zero-copy fast path
```

## Why Spissa

What makes Spissa different from just running a quantized model in Ollama or
llama.cpp:

- **Lossless by default.** The transformer weights are stored bit-identically to
  the original safetensors — `decode(encode(w)) == w`, checked per byte. Other
  runtimes ship lossy quantized weights (q4/q6/q8 K-quants that approximate the
  model). Spissa only ever quantizes the *f32 activation* at runtime in `--fast`
  (a quant-only diff, opt-in and labeled), never the stored weights. You get the
  real model, not an approximation of it.
- **Custom in-house codecs (RTC).** Compression is done by **RTC (Rama Tensor
  Codec)** — our own lossless tensor codecs, not a generic library like
  zstd/gzip. See [RTC below](#rtc-rama-tensor-codec).
- **Integrity every run.** Every weight chunk carries a SHA-256 and is verified
  (once per process in `verify-once`, prewarmed in parallel under `--fast`).
  Ollama/llama.cpp do not verify weights at load — Spissa proves the bytes are
  intact before using them.
- **Memory-first, runtime-compressed.** Weights are dormant compressed memory;
  only the chunks/tiles needed are decoded, under a bounded `MemoryBudget`. On
  small models this shows as a real RAM win (Llama 3.2 1B: ~1.05 GB vs Ollama's
  ~1.8 GB on the same machine).
- **Original, not a wrapper.** Spissa does not embed or shell out to Ollama or
  llama.cpp. The container, codecs, and CPU kernels are written from scratch — yet
  reach **prefill parity** with llama.cpp/Ollama on the same chip (and within
  ~1.5–2.5× on decode, a hardware-core ceiling, not a kernel-quality gap).
- **Honest metrics.** No "10× smaller, same quality" claims. Compression ratios,
  tok/s, and RAM are measured and reported as-is, including the limitations.

### RTC (Rama Tensor Codec)

RTC is Spissa's family of **lossless** tensor codecs. Each must satisfy
`decode(encode(input)) == input` exactly — that is the contract that keeps the
model bit-identical. The codec is chosen *per chunk* at pack time, which lets one
`.spsa` trade storage size against runtime speed:

- **`rtc-raw-v1`** — identity layout. No size win, but its bytes are the final
  weight bytes, so the runtime reads them **zero-copy straight from the mmap** —
  this is what the `--fast` q8 kernels need (whole-tensor `sdot`/`i8mm` with no
  per-token decode). Pack q8 models with `--codec raw`.
- **`rtc-rle-v1`** — run-length encoding for repetitive regions.
- **`rtc-huff-v1`** — in-house **byte-level Huffman** entropy codec for real
  lossless size reduction on disk (e.g. Pythia-70M packs to ~76% of the original
  safetensors, bit-exact).
- **`rtc-rans-v1`** — interleaved **rANS** codec on the bf16 exponent — lossless at the
  measured entropy floor (~10.5 bits/weight, ~34% smaller than raw bf16). Smallest
  lossless `.spsa`; decode-once at load runs at bf16 speed (`pack --codec rans`).
- **`rtc-bitplane-v1`** — fixed-width **bit-plane** layout with branchless NEON
  `tbl`-gather decode — lossless, the fastest-decoding lossless codec (`pack --codec bitplane`).
- planned: `rtc-delta-v1`, `rtc-entropy-v1`, AVX decode for x86.

The key design point: RTC separates **storage compression** (entropy codecs like
Huffman, smaller on disk) from **runtime residency** (raw, zero-copy, fast). A
generic compressor would force you to decompress the whole model into RAM before
inference; RTC lets the runtime decode only the tiles it needs — or skip decode
entirely on the raw fast path — while still being able to verify every byte.
Details: [docs/codec-rtc-v1.md](docs/codec-rtc-v1.md).

## Current Status

Implemented:

- Cargo workspace with five crates
- `.spsa` v1 container reader/writer
- Safetensors import
- CLI commands: `pack`, `unpack`, `inspect`, `verify`, `run`, `chat` (interactive, codec-agnostic), `bench`, `doctor`
- Stubbed future command: `import`
- RTC codecs:
  - `rtc-raw-v1` (zero-copy identity layout)
  - `rtc-rle-v1`
  - `rtc-huff-v1` (custom byte-level Huffman)
  - `rtc-rans-v1` (interleaved rANS exponent codec — **lossless**, ~10.5 bits/weight, at the bf16 entropy floor)
  - `rtc-bitplane-v1` (fixed-width bit-plane — **lossless**, fast NEON `tbl`-gather decode)
- Per-chunk SHA-256 verification
- Multi-tensor safetensors verification
- Phase 5A runtime foundation:
  - full-decode `.spsa` loader
  - fp16/bf16/f32/int runtime conversion to f32
  - basic tensor ops: embedding lookup, matmul, linear, layernorm, RMSNorm, GELU, softmax, attention, MLP, argmax/top-p sampling
- Phase 5B low-memory runtime planning:
  - metadata-only `.spsa` open path
  - memory budget accounting
  - layer-stream/tile-stream dry-run planner
  - `spissa run --memory-budget --ctx --mode tile-stream`
  - chunk-scoped streaming linear kernel (`decode chunk → f32 scratch → accumulate → release`)
  - streaming MLP sub-block (`linear → GELU → linear`) with budgeted intermediate activation
  - streaming attention/QKV sub-block (`QKV projection → split heads → attention → output projection`)
  - streaming pre-norm transformer block skeleton (`LN → attention → residual → LN → MLP → residual`)
  - tiny end-to-end next-token smoke path (`embedding → one block → final LN → lm_head → sample`)
  - GPT-NeoX/Pythia-style rotary embeddings and KV-cache attention primitives
  - RAMA architecture spec for memory-first, brain-inspired runtime direction
  - first executable cached generation loop: tiny multi-step token-ID generation with explicit prefill/decode-step paths and ContextEcho KV-cache state
  - multi-layer token-ID stack: per-layer ContextEcho KV caches, prefill/decode/generate over all configured streaming blocks, and logits checked against full-context recomputation
  - GPT-NeoX/Pythia adapter that infers standard tensor names/shapes from `.spsa` metadata and decodes small norm/bias tensors into an owned prepared stack
  - optional original `config.json` metadata persistence for GPT-NeoX/Pythia fields (`num_attention_heads`, rotary settings, layer norm eps, context length) plus runtime auto-prepare from that metadata
  - optional tokenizer vocabulary/config metadata persistence from `tokenizer.json`, plus tokenizer-backed text smoke generation via `spissa run --prompt ...`
  - Phase 6 RAMA layer-decode GPT-NeoX/Pythia path: keeps only names + final params resident, decodes per-layer norm/bias vectors just-in-time, budgets active layer params, and matches the prepared stack baseline
  - Phase 7 RAMA tiled runtime routing: streaming MLP/attention/transformer blocks and tiny/RAMA/GPT-NeoX generation heads route through the fused tile-linear path, converting only bounded weight tiles into f32 scratch while preserving logits/token baselines
  - Phase 7.6 release benchmark matrix for local Pythia-70M with persisted `config.json` + `tokenizer.json`: `ctx=128/512/1024`, `max-new-tokens=1/4/8/16`, all runs pass under `--memory-budget 100mb`, measured RSS 88.62–94.62 MiB with ~4.47–5.29s/token after Phase 7.7 fidelity fixes
  - Phase 7.7 fixed-token HF/PyTorch logits comparison: Spissa top-1/top-10 match on tested Pythia-70M prompts after persisting `use_parallel_residual` and fixing GPT-NeoX per-head QKV split
  - Phase 7.8 tile-block artifact benchmark: local Pythia-70M repacked with `--tile-block-elements 65536` runs the same `ctx=128/512/1024` × `max-new-tokens=1/4/8/16` matrix at 18.39–22.64 MiB RSS with 386 KiB tracked transient peak while preserving tested HF logits parity
  - Phase 7.9A RAMA trace profiler: `spissa run --rama-trace <path>` emits chunk recall timing JSON; real Pythia-70M one-token trace shows `chunk_decode` dominates recorded time (~3716 ms of 4505 ms), especially `gpt_neox.embed_in.weight` and `embed_out.weight`, while disk read is only ~32 ms
  - Phase 7.9B RAMA embedding row recall: embedding lookup now recalls only touched token rows/chunks; the same 12-run tile-block matrix improved from 5.07 to 2.93 average seconds/token (~1.73× average speedup) while max RSS stayed ~22 MiB and HF logits parity remained intact
  - Phase 7.9C RAMA low-ram-fast runtime layout: `spissa pack --codec raw --tile-block-elements 65536` builds a compute-ready raw/tile-block artifact; `spissa run --rama-integrity verify-once` verifies chunks once per process. The same 12-run matrix reached 0.35 average seconds/token / 3.26 average tok/s / 4.35 best tok/s while RSS stayed 19.17–23.36 MiB and HF parity remained intact.
  - Phase 7.9D real long-prompt benchmark: actual `--token-ids` prompt lengths `1/128/512/1024` with `ctx=2048` expose the next bottleneck. Short prompt remains 4.30 tok/s at 20.67 MiB RSS, but 512-token prompt + 16 generated tokens drops to 0.30 tok/s at 44.98 MiB RSS, and 1024-token prompt + 16 generated tokens drops to 0.15 tok/s at 70.84 MiB RSS; 128-token and 512-token HF logits parity still pass top-1/top-10.
  - Phase 7.9E RAMA chunked prefill: `spissa run --rama-timing <path>` adds aggregate timing and `--rama-prefill-chunk-tokens 64` bounds long-prompt prefill windows. It improved 512-token + 16 generated from 56.43s / 0.284 tok/s / 46.22 MiB RSS to 35.20s / 0.455 tok/s / 34.05 MiB RSS, and 1024-token + 16 generated from 110.29s / 0.145 tok/s / 70.55 MiB RSS to 63.84s / 0.251 tok/s / 44.98 MiB RSS.
  - Phase 7.10A row-span linear accumulation: the tiled-linear hot loop now accumulates contiguous row spans instead of doing per-weight division/modulo. Short prompt + 16 generated improved from 3.65 to 7.10 tok/s at ~20.66 MiB RSS; 512-token chunked prefill improved to 2.26 tok/s at ~32.98 MiB RSS; 1024-token chunked prefill improved to 1.25 tok/s at ~44.80 MiB RSS while 512-token HF parity remained top-1/top-10 matched.
  - Phase 7.10B RAMA prefill homeostasis: broader post-7.10A matrix reaches 9.756 tok/s for 1-token prompt + 16 generated at 20.33 MiB RSS; measured prefill chunk sweep chooses 32 real input tokens as the default CLI prefill window. Best swept long-prompt rows: 512-token + 16 at 2.3495 tok/s / 32.77 MiB RSS and 1024-token + 16 at 1.1653 tok/s / 44.91 MiB RSS.
  - Phase 7.12A generic shape/budget-aware prefill policy: `spissa run` now defaults to auto low-RAM prefill selection from GPT-NeoX/Pythia shape and explicit transient budget, preserving Pythia-70M-like 32-token defaults while selecting 64 for Pythia-160M-like low-RAM runs; `--rama-prefill-policy speed` selects the speed-biased larger window when budget allows.
  - Phase 7.12B generic eight-row projection reuse: the shared tiled-linear hot loop now reuses each decoded weight row fragment across 8 prompt-token rows before falling back to the existing 4-row/scalar tails, improving the measured Pythia-160M 512-token speed-policy row from 13.21s to 12.34s wall time while keeping tracked transient memory unchanged at 3.79 MiB.
  - R57 edge attention locality cache for the Llama experimental-speed path: optional per-layer recent-index caches can reuse a tiny number of previous edge-attention input features for sparse Q/K/V. The retained `window=8, extra=1` preset preserved the 30 tok/s floor on Llama 3.2 1B Instruct and slightly improved cheap quality counters, while the wider `window=16, extra=4` probe was rejected.
  - R58 Llama3 chat-template baseline: `llama-test --chat-template llama3` formats Llama 3.x Instruct prompts with BOS/header/EOT tokens and uses raw special-token stop fallbacks when older `.spsa` metadata lacks `eos_token_id`. Exact mode now has a coherent chat baseline before sparse AIP quality is judged.
  - q8 `--fast` runtime + interactive chat (see the [Chat](#chat) section). R134–R139 made q8 decode/prefill production-usable on CPU: `mlock`/`MADV_WILLNEED` residency, int8 `sdot`/`i8mm` decode, row-parallel prefill, NEON `i8mm` weight packing (prefill reaches llama.cpp/Ollama CPU parity on the same chip), parallel SHA-256 integrity prewarm, and a NEON bf16 LM-head. Gemma 3 4B and Llama 3.2 1B/3B run multi-turn chat with a resident KV cache (`GemmaChatSession` / the Llama token-native session), bit-identical byte-level UTF-8 decode (emoji/accents), and `./try-gemma.sh` / `./try-llama.sh` runners. Lossless preserved: weights read exactly, per-byte SHA-256 verified once per run; only the f32 activation is int8-quantized in `--fast`.


Not yet implemented:

- Production-grade tokenizer/normalizer fidelity beyond the current runtime-ready vocabulary metadata
- Multi-tensor unpack back to safetensors layout
- True intra-chunk compressed range decode/routing for non-identity codecs
- Comfortable 512/1024-token chat latency; Phase 7.12B improves generic projection throughput, but 512/1024-token prompts are still not interactive

## Quick Start

```bash
# Build
cargo build --release

# Run tests
cargo test

# Interactive chat — codec-agnostic (rANS/q8/bf16), auto-detects Gemma/Llama
SPISSA_INTEGRITY=unchecked ./target/release/spissa chat models/gemma-3-1b-it-rans.spsa           # lossless
SPISSA_INTEGRITY=unchecked ./target/release/spissa chat models/gemma-3-1b-it-q8-raw.spsa --fast  # q8, fastest
#   add --low-ram to stream the embedding (>RAM regime)
# convenience wrappers (Gemma 4B / Llama 1B|3B):  ./try-gemma.sh chat   |   ./try-llama.sh chat

# Pack a HF checkpoint into .spsa (choose a codec by the trade-off you want)
./target/release/spissa pack <model.safetensors | *.index.json | model-dir> \
  --out models/<name>.spsa \
  --config <config.json> --tokenizer <tokenizer.json> \
  --codec rans            # rans / bitplane = lossless bf16 ; raw = zero-copy (use with --quantize)
#   --quantize q8_transformer_keep_io   # lossy q8 (pair with `--codec raw` for the fast path)

# Inspect metadata + compression stats
./target/release/spissa inspect models/<name>.spsa

# Verify lossless round-trip vs the original safetensors
./target/release/spissa verify <model.safetensors> models/<name>.spsa

# Measure quantization loss (q8/q4 vs lossless) on the real weights
./target/release/quant-error models/<name>.spsa

# Low-RAM streaming / memory planning (metadata only; no full decode)
./target/release/spissa run models/<name>.spsa --memory-budget 100mb --ctx 1024 --mode tile-stream

# System / CLI info
./target/release/spissa doctor
```

### Useful environment variables

- `SPISSA_INTEGRITY=unchecked|verify-once|strict` — skip / once-per-process / per-recall SHA-256 verification.
- `RLLM_THREADS=N` — cap worker threads (decode auto-pins to performance cores on big.LITTLE; see [docs/android-termux.md](docs/android-termux.md)).
- `SPISSA_DECODE_RESIDENT=1` — decode weights once at load and cache (bf16-class steady speed; the default inside `spissa chat`).
- `SPISSA_STREAM_EMBEDDING=1` — stream the tied embedding instead of holding it resident (resident ≈ compressed size, for the >RAM regime).
- `--fast` (chat) — q8 turbo: `mlock` residency + int8-activation kernels.

> The historical Pythia/RAMA phase-7x benchmark recipes and the experimental sparse-path
> (AIP) env flags now live under [docs/archive/](docs/archive/).

## Architecture

```text
spissa/
├── crates/
│   ├── spissa-cli/        # CLI binary (`spissa`)
│   ├── spissa-container/  # .spsa binary format parser/writer
│   ├── spissa-import/     # Safetensors import
│   ├── spissa-runtime/    # Lazy/streaming runtime, session cache, tensor ops
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
| `spissa-cli` | Command-line interface (`spissa` binary) |
| `spissa-container` | `.spsa` file format: header, metadata, tensor/chunk directories |
| `spissa-import` | External format import, currently safetensors |
| `spissa-runtime` | Lazy/runtime loader, memory-budgeted planner, streaming kernels, session cache, tensor ops |
| `rtc-codec` | Lossless tensor codecs: raw, RLE, Huffman |

## File Format

The `.spsa` format is a single-file container with:

- **Header** — `SPSA` magic + version + endian marker + directory offsets
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
| `rtc-rans-v1` | Interleaved rANS exponent codec (lossless, ~10.5 bits/weight, entropy floor) | ✅ Implemented |
| `rtc-bitplane-v1` | Fixed-width bit-plane, NEON `tbl`-gather decode (lossless, fastest) | ✅ Implemented |
| `rtc-delta-v1` | Delta encoding | 🔜 Future |
| `rtc-entropy-v1` | Advanced entropy coding beyond Huffman | 🔜 Future |

See [docs/codec-rtc-v1.md](docs/codec-rtc-v1.md) for codec design details.

## Verified Example: Pythia-70M

Local verification with EleutherAI Pythia-70M safetensors:

- 94 tensors imported
- 166,019,180 tensor bytes verified
- Best local output with `raw + rle + huff`, `--chunk-size 32mb`:
  - original safetensors file: 166,029,852 bytes / 158.34 MiB
  - `.spsa`: 126,456,271 bytes / 120.60 MiB
  - ratio: 76.16% of original file size
- `spissa verify`: `[OK] LOSSLESS VERIFIED`

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

Proprietary — All Rights Reserved. No license is granted; see [LICENSE](LICENSE).
