# Trial: R7 Llama 3.2 1B Model Comparison

Date: 2026-06-14
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

Switching from the SmolLM2-135M raw artifact to the downloaded Llama 3.2 1B Instruct raw artifact should clarify whether the current speed ceiling is mainly caused by the tested model or by the RLLM runtime hot path.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: model shape and projection bandwidth
- Bottleneck tag: model shape

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/Llama-3.2-1B-Instruct-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r7-llama32-1b-model-comparison.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 15456.79 ms | 7239.23 ms | 27675.23 | 12224.12 | 43132.01 | 19463.36 | 0.54 | 1.23 | 0.37 | 0.82 | match | match | embedding=14.44ms transformer=16900.01ms final_norm=0.04ms lm_head=2548.74ms profiled_total=19463.23ms layers=256 attention_total=3968.78ms attention_norm=0.84ms q=1572.54ms k=398.06ms v=389.31ms rotary=3.24ms attention=3.07ms kv_append=0.20ms o=1601.35ms attention_residual=0.16ms mlp_total=12930.72ms mlp_norm=0.66ms gate=6531.04ms up=0.00ms activation_multiply=0.00ms down=6398.84ms mlp_residual=0.18ms | session_replayed=0 flushed=0 baseline_peak=2626682880 session_peak=2101362688 |
| 2 | 18 | 1 | 16 | 16 | 16909.65 ms | 1884.55 ms | 20349.73 | 12386.80 | 37259.37 | 14271.36 | 0.74 | 1.21 | 0.43 | 1.12 | match | match | embedding=6.42ms transformer=11580.13ms final_norm=0.05ms lm_head=2684.62ms profiled_total=14271.22ms layers=272 attention_total=2655.62ms attention_norm=0.76ms q=1059.11ms k=263.70ms v=265.30ms rotary=3.14ms attention=7.99ms kv_append=0.28ms o=1055.16ms attention_residual=0.18ms mlp_total=8924.00ms mlp_norm=0.70ms gate=4568.15ms up=0.00ms activation_multiply=0.00ms down=4354.96ms mlp_residual=0.18ms | session_replayed=0 flushed=1 baseline_peak=2626682880 session_peak=2101362688 |

## Analysis

Baseline and session token streams matched for every measured turn.

Llama 3.2 1B Instruct is much slower than the R7 SmolLM2-135M benchmark on this CPU-only RLLM path. Session decode throughput is `1.23 tok/s` on turn 1 and `1.21 tok/s` on turn 2, versus R7 SmolLM2 repeat throughput around `10.98/10.85 tok/s`.

The model shape explains the regression: this artifact has hidden size `2048`, intermediate size `8192`, vocab size `128256`, and 16 layers. Although it has fewer layers than SmolLM2-135M, each projection is substantially larger. The session phase timing shows turn 2 transformer time at `11580.13ms`, MLP total at `8924.00ms`, LM head at `2684.62ms`, and session peak transient memory at `2101362688` bytes.

This result supports the current bottleneck theory: model choice matters, but switching to a larger Llama-family model does not solve the speed target. RLLM still needs runtime-side BF16 projection acceleration, memory-traffic reduction, and likely packed/SIMD kernels before 1B-class models can approach the target chat speed.

## Decision

failed

Reason: the token stream is valid, but the model comparison is much slower than SmolLM2 and far below the 30-40 tok/s target.

Paper value:

- useful negative result

## Next Experiment

Continue R8 on the runtime path, not model switching: add raw BF16 direct projection coverage for non-fused projections such as down projection, q/o projection, and LM head.
