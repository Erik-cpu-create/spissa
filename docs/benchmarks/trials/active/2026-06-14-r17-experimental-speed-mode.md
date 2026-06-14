# Trial: R17 Experimental Speed Mode

Date: 2026-06-14
Owner: RLLM
Status: active
Folder: active

## Hypothesis

Turbo Sparse Decode can improve Llama 3.2 1B Instruct CPU-only decode speed by
reducing raw BF16 MLP projection work without changing model weights or default
exact-lowram behavior.

## Scope

- Mode: experimental-speed
- Models/artifacts: `models/SmolLM2-135M-raw.rllm`, `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Bottleneck tag: sparse MLP projection
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Top-k: `RLLM_TURBO_TOPK=256`

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=256 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | sparse calls | fallbacks | max top-k | skipped madds | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Analysis

Pending measurement.

## Decision

active

Reason: measurement pending.

Paper value:

- pending
