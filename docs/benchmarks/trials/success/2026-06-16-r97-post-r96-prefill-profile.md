# R97: Post-R96 Prefill Profile

Date: 2026-06-16
Owner: RLLM
Status: accepted diagnostic
Folder: success

## Hypothesis

R96 improved Q8 prefill with the aarch64 `REEVEC-Q8-NEON-BATCH4` kernel. R97
checks where the remaining prefill time goes before starting another kernel.

## Scope

- Mode: exact-lowram diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Runtime state: after R96 `REEVEC-Q8-NEON-BATCH4`
- Target device/profile: CPU-only, single-thread benchmark
- Bottleneck tag: post-NEON Q8 prefill attribution

R97 makes no runtime code change.

## Setup

Normal controls:

```bash
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r97-control${i}.txt" 2> "target/r97-control${i}.time"
done
```

Q8 profiled trials:

```bash
for i in 1 2; do
  RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r97-profile${i}.txt" 2> "target/r97-profile${i}.time"
done
```

## Results

All runs kept the answer correct:

```text
No
```

Normal controls:

| run | output | context tokens | prefill | decode | MLP total | attention total | lm_head | gate | up | down | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control 1 | No | 55 | 9.85s | 0.50 tok/s | 7455.54ms | 1449.05ms | 943.80ms | 3038.01ms | 2008.19ms | 2398.86ms | 1,050,673,152 | 1,737,015,296 | 16.29s |
| control 2 | No | 55 | 9.31s | 0.90 tok/s | 6823.51ms | 1387.79ms | 1098.73ms | 2871.94ms | 1749.54ms | 2190.61ms | 1,050,673,152 | 1,644,036,096 | 13.20s |
| control 3 | No | 55 | 9.60s | 0.40 tok/s | 6884.27ms | 1471.13ms | 1243.68ms | 2990.78ms | 1727.72ms | 2154.29ms | 1,050,673,152 | 1,639,956,480 | 15.33s |

Profiled trials:

| run | output | context tokens | prefill | decode | MLP total | attention total | lm_head | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| profile 1 | No | 55 | 12.98s | 0.88 tok/s | 9742.45ms | 2014.93ms | 1221.67ms | 1,050,673,152 | 1,630,715,904 | 17.11s |
| profile 2 | No | 55 | 12.94s | 0.72 tok/s | 9509.07ms | 2015.27ms | 1418.12ms | 1,050,673,152 | 1,722,793,984 | 17.32s |

Q8 profile rows:

| run | top path | calls | blocks | rows | batch items | elapsed |
|---|---|---:|---:|---:|---:|---:|
| profile 1 | `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 0 | 1,611,661,312 | 5845.56ms |
| profile 1 | `batch1_complete_linear` | 800 | 22,020,096 | 245,760 | 800 | 277.62ms |
| profile 1 | `batch1_complete_multiply` | 288 | 8,388,608 | 131,072 | 288 | 105.23ms |
| profile 2 | `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 0 | 1,611,661,312 | 6138.07ms |
| profile 2 | `batch1_complete_linear` | 800 | 22,020,096 | 245,760 | 800 | 282.49ms |
| profile 2 | `batch1_complete_multiply` | 288 | 8,388,608 | 131,072 | 288 | 114.42ms |

## Analysis

R96 remains a real prefill improvement. R97 normal controls stayed around
`9.31-9.85s` prefill, matching the R96 accepted range.

The remaining bottleneck is still prefill MLP work:

- normal MLP total: `6823.51-7455.54ms`
- normal attention total: `1387.79-1471.13ms`
- normal lm_head: `943.80-1243.68ms`

The Q8 branch profiler also shows that `batch_gt1_scaled` is still the dominant
Q8 path after R96, at `5845.56-6138.07ms`. Batch1 complete-row paths remain
small by comparison and should not be the next target.

Within the normal MLP breakdown, gate and down are the largest remaining
projection buckets:

- gate: `2871.94-3038.01ms`
- down: `2154.29-2398.86ms`
- up: `1727.72-2008.19ms`

Since R96 already optimized the f32 batch4 accumulator, the likely remaining
cost inside `batch_gt1_scaled` is scalar Q8 block scale/dequant work in
`q8_0_scaled_block`, plus memory traffic around the scaled block. R94 showed
that scalar fusion is slower, so the next credible candidate should be a
NEON scale/dequant helper for the existing scaled-block design, not another
scalar loop rewrite.

## Decision

accepted diagnostic

Reason: R97 produced complete post-R96 phase and Q8 branch attribution, with
correct output and unchanged internal peak transient memory.

Paper value:

- useful evidence that R96 did not shift the bottleneck away from Q8 prefill
- useful evidence that batch1 decode helpers remain the wrong target
- useful planning evidence for R98: target NEON Q8 scale/dequant or another
  measured part of `batch_gt1_scaled`

## Next Experiment

R98 should target `q8_0_scaled_block` with an aarch64 NEON scale/dequant lab
candidate behind the same portable-fallback discipline used by R96.

Do not start with batch1 decode, scalar fusion, or manual scalar unrolling.
