# Trial: R8 Llama 3.2 1B BF16 Direct Projection

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

The raw BF16 batch-1 projection path should also improve the downloaded Llama 3.2 1B Instruct artifact, even if the model remains much slower than SmolLM2 because its projection shapes are larger.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: model shape and raw BF16 projection bandwidth
- Bottleneck tag: raw BF16 projection

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/Llama-3.2-1B-Instruct-raw.rllm' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r8-llama32-1b-bf16-direct-projection.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 15345.82 ms | 6743.82 ms | 60979.71 | 8921.54 | 76325.53 | 15665.37 | 0.25 | 1.68 | 0.21 | 1.02 | match | match | embedding=18.40ms transformer=12175.45ms final_norm=0.05ms lm_head=3470.10ms profiled_total=15663.99ms layers=256 attention_total=1924.57ms attention_norm=0.83ms q=773.60ms k=187.02ms v=188.62ms rotary=3.08ms attention=2.90ms kv_append=0.28ms o=768.07ms attention_residual=0.18ms mlp_total=10250.33ms mlp_norm=0.70ms gate=7124.77ms up=0.00ms activation_multiply=0.00ms down=3124.68ms mlp_residual=0.18ms | session_replayed=0 flushed=0 baseline_peak=2626682880 session_peak=2101346304 |
| 2 | 18 | 1 | 16 | 16 | 21620.38 ms | 3835.63 ms | 54866.99 | 10267.85 | 76487.37 | 14103.49 | 0.27 | 1.46 | 0.21 | 1.13 | match | match | embedding=2.75ms transformer=11044.60ms final_norm=0.05ms lm_head=3055.97ms profiled_total=14103.38ms layers=272 attention_total=1659.37ms attention_norm=0.97ms q=674.77ms k=153.04ms v=151.61ms rotary=3.61ms attention=8.91ms kv_append=0.42ms o=665.81ms attention_residual=0.24ms mlp_total=9384.59ms mlp_norm=0.83ms gate=6683.85ms up=0.00ms activation_multiply=0.00ms down=2699.68ms mlp_residual=0.23ms | session_replayed=0 flushed=1 baseline_peak=2626682880 session_peak=2101346304 |

## Analysis

Baseline and session token streams matched for every measured turn.

R8 improves Llama 3.2 1B Instruct from the R7 model-comparison result, but it remains far below the 30-40 tok/s target. R7 measured Llama 1B at `1.23/1.21 tok/s`; R8 measures `1.68/1.46 tok/s` with token and history streams still matching.

The remaining cost is dominated by model shape. Turn 2 still spends `11044.60ms` in transformer work, with `9384.59ms` in MLP, `6683.85ms` in fused gate/up, `2699.68ms` in down projection, and `3055.97ms` in LM head. The new BF16 direct path reduces down/q/k/v/o projection overhead, but the fused gate/up bucket and LM head are still too expensive on scalar CPU code for a 1B-class Llama shape.

This is useful model-scale evidence: R8 is a real runtime improvement, not a SmolLM-only artifact, but larger dense models need the next runtime step before they can be credible low-end CPU targets.

## Decision

success

Reason: the token stream is valid and throughput improves over the R7 Llama 1B comparison, while the report clearly records the remaining target gap.

Paper value:

- useful positive evidence with limitation

## Next Experiment

R9 should target the remaining large buckets for both SmolLM2 and Llama 1B: LM head argmax/logit projection, fused gate/up row-blocking, and SIMD/packed BF16 dot kernels.
