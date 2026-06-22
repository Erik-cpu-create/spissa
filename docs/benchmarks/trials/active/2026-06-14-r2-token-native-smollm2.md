# Trial: R2 Token-Native Chat Session

Date: 2026-06-14
Owner: RLLM
Status: running
Folder: active

## Hypothesis

A persistent KV-cache session should reduce later-turn TTFT when compared against full token-history replay for the exact same token stream.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: full-history replay and memory bandwidth
- Bottleneck tag: cache locality

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r2-token-native-smollm2.md'
```

Runtime context:

- build profile: release
- command wrapper: `/usr/bin/time -l`
- relevant env/config: `RamaIntegrityMode::VerifyOnce`, argmax sampling, `ctx=2048`, `max_new_tokens=16`
- note: this benchmark opens two model handles in one process, so process RSS is harness cost, not session-only production RSS

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 1584.77 ms | 953.20 ms | 1498.00 | 1454.48 | 3082.77 | 2407.67 | 10.01 | 10.31 | 5.19 | 6.65 | match | match | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=226508800 |
| 2 | 18 | 1 | 16 | 16 | 871.27 ms | 179.35 ms | 1467.77 | 1488.45 | 2339.04 | 1667.80 | 10.22 | 10.08 | 6.84 | 9.59 | match | match | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=226508800 |

## Analysis

Baseline and session token streams matched for every measured turn. This makes
the R2 timing evidence valid for the exact token stream, unlike the earlier R1
text-transcript trial.

Turn 2 TTFT improved from 871.27 ms in the full-replay baseline to 179.35 ms in
the persistent session, a 4.86x speedup and about 79.4% reduction. Decode
throughput did not materially improve: baseline was 10.22 tok/s and session was
10.08 tok/s on turn 2. That points R3 away from session replay and toward the
decode hot path, especially matmul/projection and memory bandwidth.

Raw process timing:

```text
13.38 real, 9.97 user, 0.17 sys
1073299456 maximum resident set size
534168848 peak memory footprint
```

## Decision

needs follow-up

Reason: token-native equivalence is proven, and the session removes most turn 2
TTFT/replay cost, but this active report still needs one review before moving to
success.

Paper value:

- use as positive evidence after review

## Next Experiment

Use R3 to attack decode throughput. The session path reduced replay/TTFT, but
the ~10 tok/s decode rate remains essentially unchanged.
