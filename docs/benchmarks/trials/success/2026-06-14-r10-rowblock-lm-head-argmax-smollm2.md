# Trial: R10 Row-Blocked LM Head Argmax SmolLM2

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

Row-blocking the raw BF16 LM-head argmax kernel should recover speed from R9 while keeping the lower memory footprint from streaming argmax.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: scalar LM-head argmax row overhead
- Bottleneck tag: row-blocked LM-head argmax

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.rllm' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r10-rowblock-lm-head-argmax-smollm2.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 1631.79 ms | 607.73 ms | 908.19 | 788.94 | 2539.98 | 1396.67 | 16.52 | 19.01 | 6.30 | 11.46 | match | match | embedding=0.01ms transformer=1043.87ms final_norm=0.01ms lm_head=352.75ms profiled_total=1396.64ms layers=480 attention_total=240.64ms attention_norm=0.30ms q=88.37ms k=30.15ms v=29.80ms rotary=1.42ms attention=1.55ms kv_append=0.11ms o=88.91ms attention_residual=0.03ms mlp_total=802.85ms mlp_norm=0.27ms gate=556.31ms up=0.00ms activation_multiply=0.00ms down=246.24ms mlp_residual=0.03ms | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=113246208 |
| 2 | 18 | 1 | 16 | 16 | 1282.75 ms | 90.97 ms | 735.64 | 782.50 | 2018.39 | 873.47 | 20.39 | 19.17 | 7.93 | 18.32 | match | match | embedding=0.01ms transformer=637.11ms final_norm=0.01ms lm_head=236.32ms profiled_total=873.44ms layers=510 attention_total=140.94ms attention_norm=0.33ms q=50.21ms k=17.42ms v=17.26ms rotary=1.44ms attention=4.10ms kv_append=0.14ms o=50.01ms attention_residual=0.03ms mlp_total=495.75ms mlp_norm=0.28ms gate=350.09ms up=0.00ms activation_multiply=0.00ms down=145.35ms mlp_residual=0.03ms | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=113246208 |

## Analysis

Baseline and session token streams matched for every measured turn.

R10 is a positive speed recovery over R9 while preserving the low-RAM profile. SmolLM2 R9 measured `17.15/16.86 tok/s` with repeat `16.74/17.07 tok/s`; R10 measures `19.01/19.17 tok/s`, with repeat turn 2 at `19.03 tok/s`. Token and history streams still match.

The session peak remains `113246208` bytes, the same low-RAM footprint as R9 and about half of R8's `226492416` bytes. Turn 2 LM-head time drops from R9's about `273.93ms` to R10's `236.32ms`, close to the R8 resident-f32 LM-head bucket while retaining the streaming memory behavior.

This is still below the 30-40 tok/s target, but it is the first trial that improves speed beyond R8 while preserving the R9 memory reduction.

## Decision

success

Reason: token equality is valid, turn-2 throughput improves over R9 and R8, and session peak memory remains low.

Paper value:

- useful positive speed and memory evidence

## Next Experiment

R11 should target the remaining large transformer buckets: fused gate/up row-blocking, q/o projection SIMD, or a packed BF16 dot layout.
