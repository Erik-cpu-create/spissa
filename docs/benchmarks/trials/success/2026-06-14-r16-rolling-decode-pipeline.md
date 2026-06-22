# Trial: R16 Rolling Decode Pipeline

Date: 2026-06-14
Owner: RLLM
Status: success with limitation
Folder: success

## Hypothesis

An opt-in rolling executor can reduce decode friction by reusing scheduling
policy and measuring worker activity without duplicating model tensors, KV
cache, or full logits buffers.

## Scope

- Mode: exact-lowram
- Models/artifacts: `models/SmolLM2-135M-raw.spsa`, `models/Llama-3.2-1B-Instruct-raw.spsa`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Bottleneck tag: rolling decode pipeline
- Runtime gate: `RLLM_ROLLING=1`

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  RLLM_THREADS=1 /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_ROLLING=1 /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

The Llama 3.2 1B comparison used `--max-new-tokens 4`. A 16-token baseline was
attempted first, but it took `99.76s` wall time and produced a very noisy
`0.18 tok/s` decode reading, so the fair R16 pair was reduced to 4 tokens.

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | rolling tasks | rolling fallbacks | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | baseline `RLLM_THREADS=1` | 16 | 1.42 s | 20.58 | 7.43 | 0 | 0 | 458833920 bytes | 188842752 bytes | 113262592 bytes |
| SmolLM2-135M | `RLLM_ROLLING=1` | 16 | 0.98 s | 23.11 | 9.80 | 5184 | 0 | 459636736 bytes | 189661976 bytes | 113262592 bytes |
| Llama-3.2-1B-Instruct | baseline `RLLM_THREADS=1` | 4 | 13.89 s | 0.22 | 0.14 | 0 | 0 | 2252423168 bytes | 1621429536 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | `RLLM_ROLLING=1` | 4 | 13.34 s | 0.33 | 0.18 | 12024 | 0 | 2430009344 bytes | 1619594048 bytes | 1050689536 bytes |

## Analysis

R16 improved decode speed on both measured artifacts while keeping RLLM's
tracked transient memory flat. SmolLM2 improved from `20.58` to `23.11 tok/s`,
about `1.12x`. Llama 3.2 1B improved from `0.22` to `0.33 tok/s` in the fair
4-token comparison, about `1.50x`.

The rolling counters prove that the opt-in path was active: SmolLM2 submitted
`5184` rolling tasks and Llama 1B submitted `12024` rolling tasks. The fallback
count stayed at zero for these LM-head shapes.

The limitation is important: this R16 implementation is not yet a true
persistent worker pool. It introduces a session-gated rolling executor and
telemetry around the raw 16-bit LM-head argmax path, but the underlying row work
still uses scoped workers. This validates that an opt-in rolling scheduling
policy can move speed without increasing RLLM peak transient memory, but it does
not fully prove the flywheel/persistent-worker version of the idea.

RSS remains noisy on macOS. Llama's max RSS increased in the rolling run, while
peak footprint and RLLM peak transient memory stayed flat or slightly lower.
For RLLM's low-RAM claim, the stronger signal is unchanged RLLM peak transient
memory; RSS should still be monitored in later longer runs.

## Decision

success with limitation

Reason: rolling opt-in improved decode tok/s on both tested artifacts and kept
RLLM peak transient memory unchanged, but the implementation is still limited to
LM-head argmax scheduling and does not yet provide persistent worker reuse.

Paper value:

- useful positive evidence: CPU-only opt-in scheduling can improve decode speed
  without duplicating model tensors or increasing RLLM peak transient memory
- useful limitation: the first rolling executor is not enough for the 30-40
  tok/s target and does not yet validate persistent worker/flywheel execution

## Next Experiment

R17 should turn the rolling executor into a true persistent worker pool or move
the rolling idea into the heavier transformer projection path. The next
candidate should avoid scoped worker spawn inside small calls and measure
whether larger-grain scheduling improves SmolLM2 beyond `23 tok/s` without
raising the Llama 1B memory footprint.
