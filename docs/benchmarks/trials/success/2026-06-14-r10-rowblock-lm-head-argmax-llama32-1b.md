# Trial: R10 Row-Blocked LM Head Argmax Llama 3.2 1B

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

Row-blocking the raw BF16 LM-head argmax kernel should improve the Llama 3.2 1B low-RAM session path over R9, especially on the later turn where the cached chat path is most representative.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: model shape and scalar LM-head argmax row overhead
- Bottleneck tag: row-blocked LM-head argmax

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/Llama-3.2-1B-Instruct-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r10-rowblock-lm-head-argmax-llama32-1b.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 13348.45 ms | 6764.64 ms | 8694.28 | 9954.34 | 22042.72 | 16719.15 | 1.73 | 1.51 | 0.73 | 0.96 | match | match | embedding=1.47ms transformer=11675.81ms final_norm=0.04ms lm_head=5041.58ms profiled_total=16718.90ms layers=256 attention_total=2020.91ms attention_norm=0.88ms q=830.10ms k=161.94ms v=165.01ms rotary=2.85ms attention=2.69ms kv_append=0.28ms o=857.00ms attention_residual=0.16ms mlp_total=9654.43ms mlp_norm=0.54ms gate=6066.48ms up=0.00ms activation_multiply=0.00ms down=3587.26ms mlp_residual=0.15ms | session_replayed=0 flushed=0 baseline_peak=2626682880 session_peak=1050673152 |
| 2 | 18 | 1 | 16 | 16 | 15664.16 ms | 3056.27 ms | 19728.34 | 7786.41 | 35392.50 | 10842.69 | 0.76 | 1.93 | 0.45 | 1.48 | match | match | embedding=0.08ms transformer=7546.96ms final_norm=0.04ms lm_head=3295.56ms profiled_total=10842.63ms layers=272 attention_total=1227.10ms attention_norm=0.72ms q=510.36ms k=96.41ms v=98.26ms rotary=2.74ms attention=6.79ms kv_append=0.31ms o=511.37ms attention_residual=0.15ms mlp_total=6319.41ms mlp_norm=0.55ms gate=4080.06ms up=0.00ms activation_multiply=0.00ms down=2238.65ms mlp_residual=0.14ms | session_replayed=0 flushed=1 baseline_peak=2626682880 session_peak=1050673152 |

## Analysis

Baseline and session token streams matched for every measured turn.

R10 improves the Llama 3.2 1B turn-2 chat path while preserving R9's memory reduction. R9 measured `1.65/1.71 tok/s`; R10 measures `1.51/1.93 tok/s`. Turn 1 is noisier and lower than R9, but turn 2 improves over both R9 and R8. Token and history streams still match.

The session peak remains `1050673152` bytes, about half of R8's `2101346304` bytes. Turn 2 LM-head time drops from R9's `3802.14ms` to R10's `3295.56ms`, but the model is still dominated by large transformer/MLP buckets and remains far from the 30-40 tok/s target.

This is useful model-scale evidence that row-blocking helps the low-RAM argmax path, but larger Llama-class dense models need deeper projection/SIMD work.

## Decision

success

Reason: token equality is valid, turn-2 throughput improves over R9, and the low-RAM peak is preserved.

Paper value:

- useful positive speed and memory evidence with model-scale limitation

## Next Experiment

R11 should target fused gate/up row-blocking or packed/SIMD BF16 projection kernels, because LM-head-only work cannot close the remaining model-scale gap.
