# Trial: R1 Persistent Chat Session SmolLM2 Text Transcript

Date: 2026-06-14
Owner: RLLM
Status: inconclusive
Folder: inconclusive

## Hypothesis

Keeping KV-cache alive across turns should reduce turn 2 prefill latency because only the new transcript suffix is appended.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: full-history replay and memory bandwidth
- Bottleneck tag: tokenizer

## Setup

Commands:

```bash
cargo build --release -p rllm-cli
printf 'Hello\nContinue\nexit\n' | /usr/bin/time -l target/release/llama-test --model models/SmolLM2-135M-raw.spsa
/usr/bin/time -l cargo run --release -p rllm-cli -- chat-session models/SmolLM2-135M-raw.spsa --turn 'Hello' --turn 'Continue' --max-new-tokens 64 --ctx 2048 --out docs/benchmarks/trials/active/2026-06-14-r1-session-smollm2.md
```

Runtime context:

- build profile: release
- CPU: Apple A18 Pro
- RAM: 8589934592 bytes
- OS: macOS 26.5.1 build 25F80
- architecture: arm64
- relevant env/config: `RamaIntegrityMode::VerifyOnce`, argmax sampling, `ctx=2048`, `max_new_tokens=64`

## Results

| run | prompt behavior | input tokens | generated tokens | TTFT/prefill | decode tok/s | process wall | RSS | peak transient | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline turn 1 (`llama-test`) | full `conversation_history` text replay | not captured | 64 | 2130.00 ms | 7.52 | 21.32 s for two-turn command | 578404352 bytes | not captured | old path tokenizes and runs full text history |
| baseline turn 2 (`llama-test`) | full `conversation_history` text replay | not captured | 64 | 1990.00 ms | 7.16 | 21.32 s for two-turn command | 578404352 bytes | not captured | second turn includes prior decoded reply text |
| old R1 text run | independently encoded user turns | 1 each turn | 64 each turn | turn 2 307.86 ms | turn 2 8.81 | 16.80 s for two-turn command | 579387392 bytes | 226508800 bytes | invalid comparison: token stream was not proven equal to full text replay |
| strict R1 text run | full transcript suffix validation | failed before report write | failed before turn 2 | not accepted | not accepted | 17.45 s failed command | 575864832 bytes | not captured | `chat-session token history does not match full transcript tokenization` |

Raw timing summary:

```text
baseline llama-test:
  21.32 real, 20.22 user, 0.30 sys
  578404352 maximum resident set size
  308544328 peak memory footprint

old R1 text run before strict validation:
  16.80 real, 15.93 user, 0.18 sys
  579387392 maximum resident set size
  308986696 peak memory footprint

strict R1 text run after validation:
  17.45 real, 10.10 user, 0.31 sys
  575864832 maximum resident set size
  305529672 peak memory footprint
```

## Analysis

The first R1 text benchmark produced a strong apparent turn 2 speedup, but it encoded each user turn independently. That does not prove equivalence with the baseline `llama-test` path, which decodes generated tokens to text, appends a newline, and re-encodes the full `conversation_history`.

The CLI now validates text transcript tokenization before continuing. On this SmolLM2 artifact, the first generated assistant token sequence does not re-encode to the same token history under the current greedy tokenizer. Because the cached session context cannot be proven equivalent to full text replay, the previous timing comparison is not valid evidence.

This does not reject the runtime session design. It identifies a tokenizer/transcript benchmark limitation: text-mode chat evidence needs a boundary format that is proven incrementally tokenizable, or the benchmark must be token-native.

## Decision

inconclusive

Reason: strict validation rejected the text transcript before turn 2, so the benchmark cannot support an accepted speedup claim.

Paper value:

- use as limitation

## Next Experiment

Create a token-native full-replay baseline and compare it against persistent `RamaChatSession` using the exact same token IDs. Keep text transcript benchmarks gated behind `session.token_history() == tokenizer.encode(full_transcript)` validation.
