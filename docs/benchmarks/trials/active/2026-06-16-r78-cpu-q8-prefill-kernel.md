# Trial: R78 CPU Q8 Prefill Kernel

Date: 2026-06-16
Owner: RLLM
Status: running
Folder: active

## Hypothesis

Llama 3.2 1B Q8 exact-lowram prefill is dominated by CPU Q8 MLP projections.
Adding a Q8_0 complete-row fast path for `batch > 1` should reduce prefill time
without changing generated text, peak transient memory, or CPU-only semantics.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Architecture: Llama 3.2 1B Instruct
- Target device/profile: local CPU-only RLLM release build
- Expected bottleneck: Q8 MLP projection prefill
- Bottleneck tag: CPU arithmetic

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf '%s\nquit\n' 'Answer yes or no: is fire cold?' \
  | target/release/llama-test \
      --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm \
      --chat-template llama3 \
      --max-new-tokens 4 \
      --profile-phases
```

Runtime context:

- build profile: release
- OS: macOS
- GPU: not used by RLLM
- relevant config: `--chat-template llama3`, `--profile-phases`

## Results

| run | prompt/input tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | | | | | | | | |
| trial | | | | | | | | |

## Analysis

Fill this after baseline and trial runs. Include the phase-profile breakdown for
prefill and decode, especially `attention_total`, `mlp_total`, `gate`, `up`,
`down`, and `lm_head`.

## Decision

needs follow-up

Reason: waiting for before/after measurements.

Paper value:

- not paper-worthy yet

## Next Experiment

Decide after the measured R78 result.
