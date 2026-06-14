# Trial: R3 Decode Hot-Path Pre-Optimization Profile

Date: 2026-06-14
Owner: RLLM
Status: running
Folder: active

## Hypothesis

A coarse decode subphase profile should identify whether R3B should target LM head/sampling or a deeper transformer-layer bottleneck.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: per-token decode hot path after R2 removed later-turn replay overhead
- Bottleneck tag: transformer hot path

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.rllm' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r3-decode-hotpath-smollm2-preopt.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 1945.81 ms | 903.12 ms | 1389.02 | 1334.91 | 3334.83 | 2238.04 | 10.80 | 11.24 | 4.80 | 7.15 | match | match | embedding=0.03ms transformer=2030.51ms final_norm=0.01ms lm_head=207.46ms profiled_total=2238.00ms | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=226508800 |
| 2 | 18 | 1 | 16 | 16 | 804.71 ms | 165.79 ms | 1364.42 | 1359.16 | 2169.14 | 1524.95 | 10.99 | 11.04 | 7.38 | 10.49 | match | match | embedding=0.01ms transformer=1321.72ms final_norm=0.01ms lm_head=203.18ms profiled_total=1524.93ms | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=226508800 |

## Analysis

Baseline and session token streams matched for every measured turn.

R3 phase timing is aggregated from LLaMA session adapter append calls for the measured turn. Treat it as coarse wall-clock evidence for choosing the next hot-path target, not cycle-level profiling.

The LM-head gate did not pass. Turn 1 LM head was 207.46 ms / 2238.00 ms = 9.3% of profiled session time. Turn 2 LM head was 203.18 ms / 1524.93 ms = 13.3%. Both are below the 15% Task 5 threshold, while transformer layers dominate at 2030.51 ms on turn 1 and 1321.72 ms on turn 2.

## Decision

inconclusive

Reason: token equality is valid and profiling evidence is useful, but the planned LM-head optimization gate did not pass. R3B should not claim an optimization win from this pre-opt run.

Paper value:

- useful negative/triage evidence for paper appendix: LM head is not the first bottleneck on this SmolLM2 token-native CPU run

## Next Experiment

Target transformer-layer matmul/projection and memory-bandwidth behavior next. Do not run the argmax/no-full-logits optimization until LM head reaches a defensible share of token time or a different model shows that profile.
