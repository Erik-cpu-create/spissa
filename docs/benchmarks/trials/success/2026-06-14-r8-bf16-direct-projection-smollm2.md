# Trial: R8 Raw BF16 Direct Projection SmolLM2

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

Extending the raw batch-1 projection path from FP16-only to BF16 should reduce projection overhead on the real SmolLM2 raw artifact, because the artifact stores LLaMA-family weights as BF16.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: BF16 projection decode and memory bandwidth
- Bottleneck tag: raw BF16 projection

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.rllm' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r8-bf16-direct-projection-smollm2.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 1736.98 ms | 625.83 ms | 1106.92 | 842.45 | 2843.91 | 1468.28 | 13.55 | 17.81 | 5.63 | 10.90 | match | match | embedding=0.01ms transformer=1231.61ms final_norm=0.02ms lm_head=236.58ms profiled_total=1468.22ms layers=480 attention_total=286.99ms attention_norm=0.40ms q=106.69ms k=36.23ms v=35.65ms rotary=1.54ms attention=1.75ms kv_append=0.14ms o=104.55ms attention_residual=0.03ms mlp_total=944.18ms mlp_norm=0.30ms gate=647.99ms up=0.00ms activation_multiply=0.00ms down=295.84ms mlp_residual=0.04ms | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=226492416 |
| 2 | 18 | 1 | 16 | 16 | 1538.70 ms | 99.15 ms | 849.24 | 850.52 | 2387.94 | 949.67 | 17.66 | 17.64 | 6.70 | 16.85 | match | match | embedding=0.01ms transformer=718.94ms final_norm=0.01ms lm_head=230.65ms profiled_total=949.62ms layers=510 attention_total=162.75ms attention_norm=0.38ms q=57.70ms k=20.30ms v=19.98ms rotary=1.63ms attention=4.88ms kv_append=0.14ms o=57.71ms attention_residual=0.03ms mlp_total=555.76ms mlp_norm=0.32ms gate=390.61ms up=0.00ms activation_multiply=0.00ms down=164.79ms mlp_residual=0.04ms | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=226492416 |

## Analysis

Baseline and session token streams matched for every measured turn.

R8 is a strong positive result. Session decode throughput improved from R7 repeat `10.98/10.85 tok/s` to R8 primary `17.81/17.64 tok/s`, with repeat `17.34/17.76 tok/s`. Token and history streams matched in both measured turns.

The gain comes from routing raw BF16 batch-1 projections through direct byte-to-dot accumulation instead of the decoded/tiled f32 fallback. On turn 2, transformer time is `718.94ms`, attention total is `162.75ms`, MLP total is `555.76ms`, and down projection is `164.79ms`. Compared with R7 turn 2, down projection dropped from about `440.52ms` to `164.79ms`, while q/o projection buckets also dropped from about `163ms` each to about `58ms` each. LM head remains high at about `230ms`, so it is now one of the next obvious targets.

This supports the runtime bottleneck theory: model choice matters, but raw BF16 projection handling in RLLM was a major limiter. R8 still does not reach the 30-40 tok/s target, but it closes a large part of the gap without changing model size or compressing weights.

## Decision

success

Reason: token equality is valid and the throughput improvement is repeatable over R7.

Paper value:

- useful positive evidence

## Next Experiment

R9 should target remaining per-token projection overhead. The highest-value candidates are a raw BF16 argmax/LM-head path, unifying the gate/up fused kernel with the new row-blocked BF16 direct kernel, and then SIMD/packed BF16 dot kernels.
