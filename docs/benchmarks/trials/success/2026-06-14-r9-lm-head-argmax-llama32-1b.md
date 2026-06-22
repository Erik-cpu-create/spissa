# Trial: R9 LM Head Streaming Argmax Llama 3.2 1B

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

Streaming LM-head argmax should reduce the downloaded Llama 3.2 1B session memory footprint significantly, because its tied vocabulary projection is large.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: resident LM-head memory and model shape
- Bottleneck tag: LM-head argmax memory

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/Llama-3.2-1B-Instruct-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r9-lm-head-argmax-llama32-1b.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 13834.24 ms | 8305.70 ms | 53189.45 | 9104.27 | 67023.69 | 17409.98 | 0.28 | 1.65 | 0.24 | 0.92 | match | match | embedding=3.18ms transformer=12815.70ms final_norm=0.05ms lm_head=4590.58ms profiled_total=17409.51ms layers=256 attention_total=2042.13ms attention_norm=0.73ms q=822.72ms k=193.82ms v=200.08ms rotary=3.20ms attention=3.03ms kv_append=0.32ms o=818.05ms attention_residual=0.18ms mlp_total=10773.03ms mlp_norm=0.80ms gate=7514.78ms up=0.00ms activation_multiply=0.00ms down=3257.22ms mlp_residual=0.23ms | session_replayed=0 flushed=0 baseline_peak=2626682880 session_peak=1050673152 |
| 2 | 18 | 1 | 16 | 16 | 19116.10 ms | 4578.45 ms | 83641.27 | 8786.50 | 102757.37 | 13364.96 | 0.18 | 1.71 | 0.16 | 1.20 | match | match | embedding=0.09ms transformer=9562.58ms final_norm=0.05ms lm_head=3802.14ms profiled_total=13364.87ms layers=272 attention_total=1547.27ms attention_norm=0.70ms q=625.06ms k=139.34ms v=139.09ms rotary=3.16ms attention=8.02ms kv_append=0.47ms o=631.27ms attention_residual=0.16ms mlp_total=8014.78ms mlp_norm=0.78ms gate=5527.46ms up=0.00ms activation_multiply=0.00ms down=2486.34ms mlp_residual=0.20ms | session_replayed=0 flushed=1 baseline_peak=2626682880 session_peak=1050673152 |

## Analysis

Baseline and session token streams matched for every measured turn.

R9 is a large memory win on Llama 3.2 1B. Session peak transient memory drops from R8's `2101346304` bytes to `1050673152` bytes, roughly a 50% reduction. Token and history streams still match.

Throughput is mixed but not worse overall than R8: R8 measured `1.68/1.46 tok/s`, while R9 measures `1.65/1.71 tok/s`. Turn 2 improves, but the LM-head bucket remains very large at `3802.14ms`, and the model is still far below the 30-40 tok/s target.

This result matters for the RLLM research goal because it cuts about 1GB from the Llama 1B session footprint without compressing the model. It does not solve speed; it establishes the next target as faster low-RAM LM-head argmax and packed/SIMD projection kernels.

## Decision

success

Reason: RAM reduction is large, token equality is valid, and Llama 1B turn 2 throughput improves over R8.

Paper value:

- useful positive memory evidence with speed limitation

## Next Experiment

R10 should target row-blocked/SIMD raw BF16 argmax and a hybrid LM-head policy that can choose resident f32, streaming BF16, or packed quantized layout based on RAM budget.
