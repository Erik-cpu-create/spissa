# Trial: R6 Raw-FP16 Batch-1 Row-Block Projection

Date: 2026-06-14
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

A row-blocked raw-FP16 batch-1 projection kernel should reduce LLaMA decode projection time by reusing each input value across multiple output rows and cutting row-by-row scalar loop overhead.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: transformer projection scalar dot products
- Bottleneck tag: projection row blocking

## Setup

Commands:

```bash
cargo run --release -p rllm-cli -- chat-session-token 'models/SmolLM2-135M-raw.spsa' --turn-ids 1 --turn-ids 2 --max-new-tokens 16 --ctx 2048 --out 'docs/benchmarks/trials/active/2026-06-14-r6-rowblock-fp16-projection-smollm2.md'
```

## Results

| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 2369.03 ms | 979.65 ms | 1583.02 | 1488.76 | 3952.05 | 2468.41 | 9.48 | 10.08 | 4.05 | 6.48 | match | match | embedding=0.03ms transformer=2250.95ms final_norm=0.01ms lm_head=217.37ms profiled_total=2468.36ms layers=480 attention_total=557.46ms attention_norm=0.37ms q=208.45ms k=69.74ms v=69.13ms rotary=1.43ms attention=1.61ms kv_append=0.12ms o=206.57ms attention_residual=0.03ms mlp_total=1693.03ms mlp_norm=0.30ms gate=550.91ms up=587.07ms activation_multiply=1.02ms down=553.69ms mlp_residual=0.03ms | session_replayed=0 flushed=0 baseline_peak=283115520 session_peak=226508800 |
| 2 | 18 | 1 | 16 | 16 | 988.19 ms | 186.07 ms | 1498.46 | 1501.67 | 2486.65 | 1687.73 | 10.01 | 9.99 | 6.43 | 9.48 | match | match | embedding=0.01ms transformer=1463.47ms final_norm=0.01ms lm_head=224.20ms profiled_total=1687.70ms layers=510 attention_total=360.88ms attention_norm=0.33ms q=132.45ms k=44.57ms v=44.57ms rotary=1.54ms attention=4.29ms kv_append=0.12ms o=132.97ms attention_residual=0.03ms mlp_total=1102.14ms mlp_norm=0.30ms gate=352.11ms up=391.32ms activation_multiply=1.07ms down=357.32ms mlp_residual=0.03ms | session_replayed=0 flushed=1 baseline_peak=283115520 session_peak=226508800 |

## Analysis

Baseline and session token streams matched for every measured turn.

R6 preserved correctness: token streams matched for both measured turns and session replay stayed at zero.

The measured speed-up is too small and noisy to count as progress toward the 30-40 tok/s target. The primary run measured 10.08 tok/s for turn 1 and 9.99 tok/s for turn 2. A repeat run with the same command measured 10.02 tok/s for turn 1 and 9.87 tok/s for turn 2.

Compared with R5, the first R6 run was only a small improvement: R5 measured 9.94 tok/s and 9.88 tok/s. Compared with R4, R6 improved turn 2 over 9.14 tok/s but did not beat R4 turn 1 at 10.75 tok/s. This is not enough to justify row-block scalar FP16 as the next main path.

Projection subphase timings remain dominated by MLP and attention projections. R6 turn 2 reports `mlp_total=1102.14ms`, with `gate=352.11ms`, `up=391.32ms`, and `down=357.32ms`; attention q/o projections remain `q=132.45ms` and `o=132.97ms`. The row-block scalar kernel slightly changes loop structure but still streams FP16 rows and converts/scalar-multiplies every weight independently.

## Decision

failed

Reason: correctness stayed valid, but scalar row-block projection produced only noise-level token/s changes and remains far below the 30-40 tok/s goal.

Paper value:

- useful negative result: scalar row blocking alone is insufficient; RLLM needs a stronger projection strategy

## Next Experiment

R7 should move beyond scalar row blocking. The next stronger candidates are SIMD dot products for raw FP16-to-F32 accumulation, a packed/tiled import layout that stores projection rows in row blocks, or a multi-projection MLP layout that reads `gate` and `up` weights together.
