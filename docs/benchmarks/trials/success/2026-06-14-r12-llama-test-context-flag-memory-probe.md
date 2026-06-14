# Trial: R12 llama-test Context Flag Memory Probe

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

The interactive `llama-test` binary should expose context length and generation
length controls so RAM probes can compare 2K, 4K, and 8K context settings
without changing source code.

## Scope

- Mode: exact-lowram
- Models/artifacts: `models/SmolLM2-135M-raw.rllm`, `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Bottleneck tag: context capacity

## Setup

Commands:

```bash
printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 1

printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 1
```

The same command was repeated with `--ctx 4096` and `--ctx 8192`.

## Results

| model | ctx | generated | TTFT/prefill | max RSS | peak memory footprint | RLLM peak transient |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | 2048 | 1 | 1.43 s | 459309056 bytes | 189334272 bytes | 113262592 bytes |
| SmolLM2-135M | 4096 | 1 | 1.04 s | 458866688 bytes | 188957536 bytes | 113262592 bytes |
| SmolLM2-135M | 8192 | 1 | 1.07 s | 458932224 bytes | 189137928 bytes | 113262592 bytes |
| Llama-3.2-1B-Instruct | 2048 | 1 | 13.59 s | 2453110784 bytes | 1621478712 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | 4096 | 1 | 13.05 s | 2018410496 bytes | 1620528512 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | 8192 | 1 | 13.87 s | 2087714816 bytes | 1620659800 bytes | 1050689536 bytes |

## Analysis

R12 adds `--ctx` and `--max-new-tokens` to `llama-test`. The short prompt
probe shows that increasing configured context from 2K to 8K does not
immediately increase macOS peak footprint for this workload.

This does not prove full 8K long-context RAM usage. RLLM's KV cache reports
resident context memory from tokens actually appended to the session, and Rust
`Vec::with_capacity` capacity may not show up as resident memory until pages
are touched.

## Decision

success

Reason: the CLI now supports context-controlled RAM probes, and the short
prompt probe confirms that context capacity alone does not spike live memory on
macOS for the tested artifacts.

Paper value:

- use as tooling evidence
- use as context-memory caveat

## Next Experiment

Run filled-context probes at 512, 1024, 2048, and 4096 real prompt tokens to
measure resident KV growth under actual long-context use.
