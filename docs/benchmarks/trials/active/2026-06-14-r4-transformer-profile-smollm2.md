# Trial: R4 Transformer Subphase Profile

Date: 2026-06-14
Owner: RLLM
Status: running
Folder: active

## Hypothesis

Deep transformer subphase timing should identify the next decode optimization target after R3 showed the transformer block dominates token time.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: transformer projection and memory bandwidth
- Bottleneck tag: transformer hot path

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.rllm' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r4-transformer-profile-smollm2.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 1974.67 ms | 923.46 ms | 1451.73 | 1394.80 | 3426.40 | 2318.27 | 10.33 | 10.75 | 4.67 | 6.90 | match | match | embedding=0.02ms transformer=2108.64ms final_norm=0.01ms lm_head=209.55ms profiled_total=2318.23ms layers=480 attention_total=532.62ms attention_norm=0.30ms q=199.20ms k=66.95ms v=66.60ms rotary=1.35ms attention=1.44ms kv_append=0.14ms o=196.62ms attention_residual=0.03ms mlp_total=1575.56ms mlp_norm=0.30ms gate=522.03ms up=525.68ms activation_multiply=1.08ms down=526.43ms mlp_residual=0.03ms | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=226508800 |
| 2 | 18 | 1 | 16 | 16 | 841.24 ms | 173.69 ms | 1415.17 | 1641.92 | 2256.41 | 1815.61 | 10.60 | 9.14 | 7.09 | 8.81 | match | match | embedding=0.01ms transformer=1577.18ms final_norm=0.01ms lm_head=238.36ms profiled_total=1815.56ms layers=510 attention_total=398.49ms attention_norm=0.37ms q=145.67ms k=49.98ms v=48.92ms rotary=1.88ms attention=4.82ms kv_append=0.19ms o=146.61ms attention_residual=0.05ms mlp_total=1178.03ms mlp_norm=0.46ms gate=391.66ms up=393.33ms activation_multiply=1.38ms down=391.13ms mlp_residual=0.07ms | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=226508800 |

## Analysis

Baseline and session token streams matched for every measured turn.

R4 phase timing is aggregated from LLaMA session adapter append calls for the measured turn. Treat it as coarse wall-clock evidence for choosing the next hot-path target, not cycle-level profiling. Deep transformer detail is enabled only by this benchmark command so normal session use does not pay the full profiler overhead.

The dominant bucket is MLP projection. Turn 1 reports MLP total 1575.56 ms versus attention total 532.62 ms. Turn 2 reports MLP total 1178.03 ms versus attention total 398.49 ms. Inside MLP, the main costs are gate/up/down projections: turn 1 gate=522.03 ms, up=525.68 ms, down=526.43 ms; turn 2 gate=391.66 ms, up=393.33 ms, down=391.13 ms.

The second-tier attention costs are q/o projections: turn 1 q=199.20 ms and o=196.62 ms; turn 2 q=145.67 ms and o=146.61 ms. Rotary, attention score/context, KV append, norms, activation multiply, and residual adds are all small compared with projection time.

## Decision

success

Reason: token equality stayed valid and the deep profile identifies MLP gate/up/down projection as the next defensible optimization target.

Paper value:

- useful bottleneck evidence: for this SmolLM2 CPU-only token-native run, projection-heavy MLP dominates transformer time more than attention score/context or KV append

## Next Experiment

Implement R5 against the projection hot path. First target MLP gate/up/down matvec locality and allocation behavior, then compare against q/o projection if MLP optimization does not move decode speed enough.
