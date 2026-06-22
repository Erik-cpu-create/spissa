# Trial: R9 LM Head Streaming Argmax SmolLM2

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

Replacing resident f32 LLaMA session LM-head logits with raw BF16 streaming argmax should reduce peak RAM substantially while preserving exact token output.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: resident LM-head memory and logit materialization
- Bottleneck tag: LM-head argmax memory

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r9-lm-head-argmax-smollm2.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 1483.92 ms | 708.15 ms | 1287.28 | 874.78 | 2771.20 | 1582.93 | 11.65 | 17.15 | 5.77 | 10.11 | match | match | embedding=0.02ms transformer=1187.76ms final_norm=0.01ms lm_head=395.09ms profiled_total=1582.88ms layers=480 attention_total=279.57ms attention_norm=0.38ms q=102.66ms k=35.05ms v=34.63ms rotary=1.52ms attention=1.64ms kv_append=0.13ms o=103.53ms attention_residual=0.03ms mlp_total=907.77ms mlp_norm=0.30ms gate=626.83ms up=0.00ms activation_multiply=0.00ms down=280.60ms mlp_residual=0.04ms | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=113246208 |
| 2 | 18 | 1 | 16 | 16 | 1480.73 ms | 98.40 ms | 822.65 | 889.63 | 2303.38 | 988.03 | 18.23 | 16.86 | 6.95 | 16.19 | match | match | embedding=0.02ms transformer=714.04ms final_norm=0.01ms lm_head=273.93ms profiled_total=987.99ms layers=510 attention_total=162.54ms attention_norm=0.37ms q=57.81ms k=20.21ms v=20.15ms rotary=1.68ms attention=4.76ms kv_append=0.13ms o=57.42ms attention_residual=0.03ms mlp_total=551.06ms mlp_norm=0.31ms gate=387.15ms up=0.00ms activation_multiply=0.00ms down=163.56ms mlp_residual=0.03ms | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=113246208 |

## Analysis

Baseline and session token streams matched for every measured turn.

R9 is a memory success and a speed tradeoff on SmolLM2. Session peak transient memory drops from R8's `226492416` bytes to `113246208` bytes, roughly a 50% reduction. Token and history streams still match.

Throughput is slightly below R8: R8 primary/repeat measured `17.81/17.64 tok/s` and `17.34/17.76 tok/s`; R9 measures `17.15/16.86 tok/s`, with repeat `16.74/17.07 tok/s`. The LM-head bucket rises from about `230ms` in R8 turn 2 to about `274ms` in R9 turn 2 because streaming raw BF16 reads less resident RAM but rereads the compressed/raw weight chunks per token.

This is useful low-RAM evidence, not a speed breakthrough. It shows RLLM can cut resident/session peak memory sharply without changing model size, but the 30-40 tok/s path still needs faster packed/SIMD argmax or a hybrid policy that pins LM head only when enough RAM is available.

## Decision

success

Reason: RAM reduction is large and token equality is valid, even though speed is slightly worse than R8.

Paper value:

- useful positive memory evidence with speed limitation

## Next Experiment

R10 should target speed recovery for the low-RAM path: row-blocked/SIMD raw BF16 argmax, hybrid resident-vs-streaming LM-head policy, or packed quantized LM-head layout.
