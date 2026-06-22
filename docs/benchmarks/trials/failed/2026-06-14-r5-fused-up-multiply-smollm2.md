# Trial: R5 Fused Up Multiply + Raw FP16 Batch-1 Projection

Date: 2026-06-14
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

Avoiding materialization of the LLaMA MLP `up_proj` output and adding a direct raw-FP16 batch-1 projection path should reduce decode time for the projection-heavy transformer hot path identified in R4.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: MLP gate/up/down projection memory bandwidth and per-token matvec overhead
- Bottleneck tag: transformer MLP projection

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r5-fused-up-multiply-smollm2.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 2461.80 ms | 984.51 ms | 1574.93 | 1509.09 | 4036.73 | 2493.60 | 9.52 | 9.94 | 3.96 | 6.42 | match | match | embedding=0.03ms transformer=2268.85ms final_norm=0.01ms lm_head=224.66ms profiled_total=2493.55ms layers=480 attention_total=560.89ms attention_norm=0.33ms q=210.61ms k=70.33ms v=70.01ms rotary=1.39ms attention=1.67ms kv_append=0.12ms o=206.41ms attention_residual=0.03ms mlp_total=1707.55ms mlp_norm=0.33ms gate=554.38ms up=590.48ms activation_multiply=1.07ms down=561.24ms mlp_residual=0.04ms | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=226508800 |
| 2 | 18 | 1 | 16 | 16 | 979.56 ms | 185.82 ms | 1495.65 | 1517.54 | 2475.21 | 1703.36 | 10.03 | 9.88 | 6.46 | 9.39 | match | match | embedding=0.01ms transformer=1493.04ms final_norm=0.01ms lm_head=210.26ms profiled_total=1703.32ms layers=510 attention_total=369.77ms attention_norm=0.34ms q=135.49ms k=46.09ms v=45.78ms rotary=1.53ms attention=4.56ms kv_append=0.13ms o=135.81ms attention_residual=0.04ms mlp_total=1122.80ms mlp_norm=0.31ms gate=360.33ms up=396.53ms activation_multiply=1.12ms down=364.46ms mlp_residual=0.04ms | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=226508800 |

## Analysis

Baseline and session token streams matched for every measured turn.

R5 preserved correctness: baseline and session token streams still matched for both measured turns, and session replay stayed at zero.

The speed hypothesis did not hold. R4 measured session decode at 10.75 tok/s for turn 1 and 9.14 tok/s for turn 2. R5 measured 9.94 tok/s for turn 1 and 9.88 tok/s for turn 2. That is not a reliable net decode-speed improvement, and it remains far below the 30-40 tok/s project target.

The fused `up_proj` path removed the standalone activation multiply pass, but `up` timing did not fall: turn 1 reported `up=590.48ms` versus R4 `up=525.68ms`, and turn 2 reported `up=396.53ms` versus R4 `up=393.33ms`. The likely cause is that the new multiply-into accumulator adds control overhead while the matvec itself remains dominated by weight streaming and scalar FP16-to-F32 multiply-accumulate work.

Raw-FP16 batch-1 direct accumulation also did not produce enough projection speed movement to change the overall decode profile. The evidence points away from allocation-only changes and toward a stronger projection kernel change, such as multi-output row blocking, SIMD-friendly dot products, or a packed/tiled runtime weight layout.

## Decision

failed

Reason: correctness was preserved, but the optimization did not reliably improve decode token/s and did not move RLLM closer to the 30-40 tok/s target.

Paper value:

- useful negative result: allocation-only/fused-multiply changes are insufficient for this CPU-only SmolLM2 LLaMA decode profile

## Next Experiment

R6 should target the projection kernel itself, not just intermediate allocation. The next candidate is a batch-1 raw-FP16 dot kernel that computes multiple output rows per input scan or introduces a packed row-block layout during import so gate/up/down and q/o projection spend less time streaming scalar rows independently.
