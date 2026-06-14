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
| 1 | 1 | 1 | 16 | 16 | 2168.61 ms | 1393.28 ms | 1573.73 | 2316.34 | 3742.33 | 3709.64 | 9.53 | 6.48 | 4.28 | 4.31 | match | match | embedding=0.06ms transformer=3395.73ms final_norm=0.02ms lm_head=313.72ms profiled_total=3709.52ms layers=480 attention_total=848.92ms attention_norm=1.08ms q=308.45ms k=117.57ms v=102.74ms rotary=2.64ms attention=2.37ms kv_append=0.19ms o=313.81ms attention_residual=0.08ms mlp_total=2545.84ms mlp_norm=0.59ms gate=826.70ms up=826.90ms activation_multiply=1.89ms down=889.66ms mlp_residual=0.11ms | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=226508800 |
| 2 | 18 | 1 | 16 | 16 | 1030.94 ms | 292.54 ms | 2115.74 | 1895.99 | 3146.68 | 2188.53 | 7.09 | 7.91 | 5.08 | 7.31 | match | match | embedding=0.14ms transformer=1890.75ms final_norm=0.01ms lm_head=297.49ms profiled_total=2188.39ms layers=510 attention_total=476.59ms attention_norm=0.53ms q=173.21ms k=60.03ms v=60.63ms rotary=2.50ms attention=6.08ms kv_append=0.22ms o=173.33ms attention_residual=0.07ms mlp_total=1413.35ms mlp_norm=0.45ms gate=467.34ms up=476.05ms activation_multiply=1.71ms down=467.68ms mlp_residual=0.11ms | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=226508800 |

## Analysis

Baseline and session token streams matched for every measured turn.

R4 phase timing is aggregated from LLaMA session adapter append calls for the measured turn. Treat it as coarse wall-clock evidence for choosing the next hot-path target, not cycle-level profiling. The extra instrumentation adds measurable overhead, so the detailed rows are for bottleneck attribution rather than raw speed comparison against R3.

The dominant bucket is MLP projection. Turn 1 reports MLP total 2545.84 ms versus attention total 848.92 ms. Turn 2 reports MLP total 1413.35 ms versus attention total 476.59 ms. Inside MLP, the main costs are gate/up/down projections: turn 1 gate=826.70 ms, up=826.90 ms, down=889.66 ms; turn 2 gate=467.34 ms, up=476.05 ms, down=467.68 ms.

The second-tier attention costs are q/o projections: turn 1 q=308.45 ms and o=313.81 ms; turn 2 q=173.21 ms and o=173.33 ms. Rotary, attention score/context, KV append, norms, activation multiply, and residual adds are all small compared with projection time.

## Decision

success

Reason: token equality stayed valid and the deep profile identifies MLP gate/up/down projection as the next defensible optimization target.

Paper value:

- useful bottleneck evidence: for this SmolLM2 CPU-only token-native run, projection-heavy MLP dominates transformer time more than attention score/context or KV append

## Next Experiment

Implement R5 against the projection hot path. First target MLP gate/up/down matvec locality and allocation behavior, then compare against q/o projection if MLP optimization does not move decode speed enough.
