# Trial: R11 llama-test Persistent Session SmolLM2

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

The interactive `llama-test` binary should use the same persistent token-native
session path as the benchmark runner. Replaying the full accumulated text
history each turn causes TTFT/prefill to grow with chat length and hides the
actual R10 session behavior.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Bottleneck tag: full-history replay

## Setup

Command:

```bash
printf 'good morning\nhalo\nwho are you ?\nexit\n' | \
  cargo run --release -p rllm-cli --bin llama-test -- \
    --model models/SmolLM2-135M-raw.rllm
```

## Results

| turn | TTFT/prefill | decode tok/s | end-to-end tok/s | generated | context | peak transient |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.47 s | 18.70 | 13.22 | 64 | 66 | 113262592 bytes |
| 2 | 0.17 s | 20.06 | 19.35 | 64 | 133 | 113262592 bytes |
| 3 | 0.20 s | 19.58 | 18.73 | 64 | 202 | 113262592 bytes |

## Analysis

Before R11, `llama-test` appended user input and generated text to a single
`conversation_history` string, tokenized the full string every turn, and ran
the full prompt through the non-session generation path. This made later
turns appear slower even though the optimized session path was available.

R11 changes `llama-test` to call `RamaChatSession::generate_turn` with only the
current token-native user turn suffix. TTFT no longer increases with the
visible accumulated context in this smoke test.

## Decision

success

Reason: the user-facing interactive binary now exercises the persistent
session path and reproduces the R10 memory profile instead of the old
full-replay behavior.

Paper value:

- use as CLI correctness and bottleneck attribution evidence

## Next Experiment

Separate generation quality from runtime speed by adding chat-template aware
prompting for instruct models while preserving the persistent KV/session path.
