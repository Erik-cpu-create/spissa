# Trial: R1 Persistent Chat Session SmolLM2

Date: 2026-06-14
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

Keeping KV-cache alive across turns reduces turn 2 prefill latency because only new user tokens are appended.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: full-history replay and memory bandwidth
- Bottleneck tag: cache locality

## Setup

Commands:

```bash
cargo build --release -p rllm-cli
printf 'Hello\nContinue\nexit\n' | /usr/bin/time -l target/release/llama-test --model models/SmolLM2-135M-raw.rllm
/usr/bin/time -l target/release/rllm chat-session models/SmolLM2-135M-raw.rllm --turn 'Hello' --turn 'Continue' --max-new-tokens 64 --ctx 2048 --out docs/benchmarks/trials/active/2026-06-14-r1-session-smollm2.md
```

The measured R1 command first wrote to `active/`; this report was moved to
`success/` after the baseline comparison accepted the result. For replaying the
final report location, use the same command with:

```bash
--out docs/benchmarks/trials/success/2026-06-14-r1-session-smollm2.md
```

Runtime context:

- build profile: release
- CPU: Apple A18 Pro
- RAM: 8589934592 bytes
- OS: macOS 26.5.1 build 25F80
- architecture: arm64
- relevant env/config: `RamaIntegrityMode::VerifyOnce`, argmax sampling, `ctx=2048`, `max_new_tokens=64`

## Results

| run | prompt behavior | input tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | process wall | RSS | peak transient | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline turn 1 (`llama-test`) | full `conversation_history` replay | not captured | 64 | 2130.00 ms | 7.52 | not captured | 21.32 s for two-turn command | 578404352 bytes | not captured | old path tokenizes and runs full `conversation_history`; `/usr/bin/time -l` peak footprint was 308544328 bytes |
| baseline turn 2 (`llama-test`) | full `conversation_history` replay | not captured | 64 | 1990.00 ms | 7.16 | not captured | 21.32 s for two-turn command | 578404352 bytes | not captured | second turn includes prior generated reply in prompt |
| R1 turn 1 (`chat-session`) | append new turn only | 1 | 64 | 1122.00 ms | 8.90 | 7.80 | 16.80 s for two-turn command | 579387392 bytes | 226508800 bytes | replayed_tokens=0 flushed_pending_tokens=0 context_bytes=2949120 |
| R1 turn 2 (`chat-session`) | append new turn only | 1 | 64 | 307.86 ms | 8.81 | 8.58 | 16.80 s for two-turn command | 579387392 bytes | 226508800 bytes | replayed_tokens=0 flushed_pending_tokens=1 context_bytes=5944320 |

Raw timing summary:

```text
baseline llama-test:
  21.32 real, 20.22 user, 0.30 sys
  578404352 maximum resident set size
  308544328 peak memory footprint

R1 chat-session:
  16.80 real, 15.93 user, 0.18 sys
  579387392 maximum resident set size
  308986696 peak memory footprint
```

## Analysis

R1 validates the session hypothesis for this two-turn SmolLM2 run. The baseline `llama-test` path appends generated text into `conversation_history`, tokenizes the entire history again, and calls the non-session generation path for each turn. The R1 path keeps the adapter KV-cache resident, flushes the previous uncached assistant tail once, then appends only the new user turn.

Turn 2 prefill/TTFT improved from 1990.00 ms to 307.86 ms, a 6.46x speedup and about 84.5% reduction. Decode speed also improved from 7.16 tok/s to 8.81 tok/s. End-to-end wall time for the scripted two-turn command improved from 21.32 s to 16.80 s, while RSS stayed effectively flat at about 579 MB.

The important correctness signal is that R1 turn 2 reports `replayed_tokens=0` and `flushed_pending_tokens=1`. That means the previous assistant tail token was cached before the new turn, and the old history was not replayed.

## Decision

accepted

Reason: measured turn 2 TTFT/prefill speedup is large, replay count is zero, and resident memory did not materially increase versus the baseline.

Paper value:

- use as positive evidence

## Next Experiment

Run the same persistent-session benchmark on a larger LLaMA-family artifact and add a longer multi-turn trial to see when KV-cache memory growth becomes the next bottleneck.
