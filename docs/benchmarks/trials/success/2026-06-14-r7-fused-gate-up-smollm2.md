# Trial: R7 Fused BF16 Gate/Up MLP Projection

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

Fusing LLaMA `gate_proj` and `up_proj` for raw-BF16 batch-1 decode should reduce MLP projection time by computing `silu(gate) * up` in one streaming pass over aligned gate/up chunks.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: BF16 MLP gate/up projection passes
- Bottleneck tag: fused gate/up projection

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r7-fused-gate-up-smollm2.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 2409.43 ms | 878.03 ms | 1607.99 | 1387.80 | 4017.43 | 2265.83 | 9.33 | 10.81 | 3.98 | 7.06 | match | match | embedding=0.03ms transformer=2022.28ms final_norm=0.01ms lm_head=243.44ms profiled_total=2265.77ms layers=480 attention_total=680.58ms attention_norm=0.40ms q=252.06ms k=85.56ms v=85.53ms rotary=1.72ms attention=1.94ms kv_append=0.14ms o=253.19ms attention_residual=0.04ms mlp_total=1341.23ms mlp_norm=0.37ms gate=664.69ms up=0.00ms activation_multiply=0.00ms down=676.13ms mlp_residual=0.05ms | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=226508800 |
| 2 | 18 | 1 | 16 | 16 | 1385.11 ms | 171.61 ms | 1389.14 | 1387.82 | 2774.24 | 1559.43 | 10.80 | 10.81 | 5.77 | 10.26 | match | match | embedding=0.01ms transformer=1319.08ms final_norm=0.01ms lm_head=240.27ms profiled_total=1559.38ms layers=510 attention_total=444.59ms attention_norm=0.42ms q=163.28ms k=55.47ms v=55.32ms rotary=1.84ms attention=5.38ms kv_append=0.14ms o=162.70ms attention_residual=0.04ms mlp_total=873.99ms mlp_norm=0.41ms gate=433.01ms up=0.00ms activation_multiply=0.00ms down=440.52ms mlp_residual=0.05ms | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=226508800 |

## Analysis

Baseline and session token streams matched for every measured turn.

R7 preserved correctness and produced the first clear post-R4 projection improvement. The primary run measured 10.81 tok/s for both measured turns. A repeat run measured 10.98 tok/s for turn 1 and 10.85 tok/s for turn 2.

The fused BF16 path was active: `up=0.00ms` and `activation_multiply=0.00ms` in the detailed timing because the combined work is charged into the `gate` bucket. R7 turn 2 reports `mlp_total=873.99ms`, down from R6 turn 2 `mlp_total=1102.14ms` and R5 turn 2 `mlp_total=1122.80ms`.

This is still not enough for the project target. The measured decode speed is around 10.8 tok/s, not 30-40 tok/s. Remaining dominant costs are fused gate/up, down projection, attention q/o projections, and LM head. The next speed step needs to apply the same raw-BF16 direct strategy to more projection paths or introduce SIMD/packed layouts.

## Decision

success

Reason: token equality stayed valid and the BF16 fused gate/up path reduced MLP projection time with repeatable tok/s improvement over R5/R6.

Paper value:

- useful positive evidence: raw-BF16 fused multi-projection streaming works and moves CPU-only decode speed in the right direction

## Next Experiment

R8 should extend raw-BF16 direct/fused projection coverage beyond gate/up. Candidate targets are down projection, q/o projection, and LM head. A stronger path is a packed or SIMD BF16 dot kernel that keeps the low-RAM streaming contract while reducing scalar conversion/multiply overhead.
