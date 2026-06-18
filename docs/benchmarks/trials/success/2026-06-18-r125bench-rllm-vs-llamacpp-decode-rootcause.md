# Trial: RLLM vs llama.cpp/Ollama competitive baseline + decode root-cause

Date: 2026-06-18
Owner: RLLM
Status: accepted (diagnostic)
Folder: success

## Hypothesis

After R124 (octet ILP, prefill 1.24s), measure RLLM honestly against the
reference CPU engine (llama.cpp, which Ollama wraps) on the *same* model, *same*
single-thread CPU-only conditions, to locate the real remaining gaps (prefill,
decode, RAM) and decide whether the low-RAM thesis still holds.

## Scope

- Mode: exact-lowram (diagnostic / competitive baseline)
- REE kernel: REEFUSE-Q8-I8MM-PANEL (R124 state)
- Model/artifact: RLLM `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
  vs llama.cpp GGUF `llama3.2:1b` Q8_0 (Ollama blob `74701a8c…`, same weights)
- Architecture: LLaMA 3.2 1B Instruct, Q8_0
- Target device/profile: Apple A18 Pro, single-thread, CPU-only
- Bottleneck tag: memory bandwidth / allocation (decode)

## Setup

Commands:

```bash
# llama.cpp built CPU-only (Metal off), same GGUF
cmake -B build -DGGML_METAL=OFF -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
./build/bin/llama-bench -m <gguf> -t 1 -ngl 0 -p 64 -n 32      # pp/tg tok/s
/usr/bin/time -l ./build/bin/llama-cli -m <gguf> -t 1 -ngl 0 -no-cnv -st -p "…" -n 8   # RSS
# RLLM
RLLM_THREADS=1 RLLM_Q8_ACTIVATION=1 llama-test --model <rllm> --max-new-tokens 16
/usr/bin/time -l … llama-test …    # RSS
```

Runtime context:

- build profile: release
- CPU: Apple A18 Pro (2 perf + 4 eff), single-thread (`-t 1` / `RLLM_THREADS=1`)
- OS: macOS (darwin 25.5)
- relevant env/config: CPU-only (llama.cpp `-ngl 0`, RLLM is CPU-only); Ollama
  `num_thread:1` is NOT honored for prompt processing (it used all cores → its
  1260 tok/s prefill was multi-threaded and is excluded; `llama-bench -t 1` is
  authoritative).

## Results

| run | input tok | gen tok | prefill | decode tok/s | RSS | notes |
|---|---:|---:|---:|---:|---:|---|
| llama.cpp `-t 1` CPU | 64 (pp) | 32 (tg) | 123 tok/s | 36.3 | 3.30 GB | repacks q8 → 2 resident copies |
| RLLM R124 | 54 | 16 | ~44 tok/s | 1.4 | 1.33 GB | streaming, per-token repack |
| **ratio** | | | **2.8x slower** | **~26x slower** | **2.5x LESS** | |

llama.cpp RAM breakdown (load log): CPU model 266 MiB + **CPU_REPACK 1252 MiB** +
KV 128 MiB + compute 68 MiB, plus the mmap'd 1.3 GB → ~3.3 GB peak RSS.

## Analysis

- **RAM thesis holds, decisively.** RLLM runs the model lossless in **1.33 GB**;
  llama.cpp's fast CPU path needs **3.30 GB** because on aarch64 it keeps the
  mmap weights AND a repacked int8 copy (the `CPU_REPACK` buffer). RLLM owns the
  low-RAM end of the RAM↔speed Pareto frontier.
- **Prefill is nearly competitive** (~2.8x), thanks to R119–R124 panel/octet work.
- **Decode is the real gap (~26x).** Literature: decode is weight-memory-bound
  ("weight memory ≈ 98.8% of decode memory ops"). The single-core memory floor
  for reading ~1.3 GB resident q8 weights is ~28 ms/token ≈ 36 tok/s —
  **llama.cpp sits exactly at that floor**, RLLM at 700 ms/token is **25x above
  it**. So RLLM's slow decode is NOT the price of low RAM; it is removable
  per-token CPU overhead.
- **Root cause (verified, corrects an earlier wrong guess).** The per-token
  `.to_vec()` decode copy was suspected but disproven (R126: zero-copy gave no
  speedup). The real cost is the batch=1 int8 kernel
  (`accumulate_q8_0_chunk_int8_activation`): it re-quantizes each input segment
  **once per output row** (e.g. 8192× redundant for gate) — the panel path caches
  this for prefill (batch≥2), the batch=1 decode path does not. The int8 dot is
  already SIMD `sdot`.
- **Lossless paths to low-RAM + fast decode together** (literature-backed):
  (1) cache the batch1 activation quant (R127), (2) store weights pre-packed in
  the `.rllm` at pack-time (1 copy, mmap → less RAM than llama.cpp AND no
  per-token repack), (3) compute-fused lossless decompression (ZipServ-style;
  RLLM already owns the RTC codec). PowerInfer / LLM-in-a-flash reach speed via
  **activation sparsity**, which requires ReLU-sparsified models (lossy, a
  different checkpoint) — incompatible with RLLM's bit-exact doctrine and with
  dense SiLU LLaMA 3.2.

## Decision

accepted (diagnostic)

Reason: measured, apples-to-apples, single-thread CPU-only — RLLM uses **2.5x
less RAM** (1.33 vs 3.30 GB) lossless, is **2.8x** behind on prefill and **26x**
behind on decode. The decode gap is overhead (batch1 re-quantization), not a RAM
tradeoff, and not memcpy — so it is closable losslessly.

Paper value:

- use as positive evidence (RAM): RLLM holds the lowest-RAM lossless point;
  llama.cpp pays ~2.5x RAM for its decode speed via weight repacking.
- use as limitation (decode): 26x decode gap, root-caused to batch1 redundant
  activation quantization.

## Next Experiment

R127: cache the batch1 activation quantization (quantize the input once per
matmul, reuse across all output rows; accumulate per output row into one sdot
register, scale per block) — the panel already does this for prefill. Then
re-measure decode against the ~36 tok/s memory floor. Sources: arXiv 2312.11514
(LLM in a flash), 2603.17435 (ZipServ), 2412.20185 (DecDEC), 2508.06753 (AI-PC).
