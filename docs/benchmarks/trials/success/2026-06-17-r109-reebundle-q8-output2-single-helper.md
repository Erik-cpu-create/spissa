# R109: REEBUNDLE Q8 Output2 Single Helper

Date: 2026-06-17
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

R108 accepted output-feature bundling but used a conservative helper that called
the existing batch4 helper twice. R109 tests whether a single output2 helper can
reduce helper overhead and RSS noise while preserving exact output math.

## Scope

- Mode: exact-lowram runtime gate
- REE kernel lineage: `REEBUNDLE-Q8-OUTPUT2-SINGLE`
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Prompt: `Answer yes or no: is fire cold?`
- Threading: `RLLM_THREADS=1`
- Integrity: `--rama-integrity unchecked`
- Diagnostics: `--profile-phases`, optional `RLLM_Q8_KERNEL_PROFILE=1`
- Bottleneck tag: CPU arithmetic / Q8 output-feature helper overhead

R109 keeps R108's pair detection and safety gate. It only replaces the output2
batch4 helper body with a single-loop scalar fallback plus aarch64 NEON helper.

## Setup

Same-turn pre-control on R108 code:

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r109-pre-control.txt 2> target/r109-pre-control.time
```

Candidate:

```bash
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r109-run${i}.txt" 2> "target/r109-run${i}.time"
done
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r109-profile.txt 2> target/r109-profile.time
```

Runtime context:

- build profile: release
- target benchmark: single-thread CPU-only
- relevant env/config: `RLLM_THREADS=1`, `--rama-integrity unchecked`,
  `--chat-template llama3`, `--profile-phases`
- architecture fast path: `target_arch = "aarch64"`
- fallback: portable scalar output2 helper on non-aarch64

## Runtime Results

All runs kept the visible answer correct:

```text
No
```

| run | output | context tokens | generated tokens | TTFT / prefill | decode tok/s | E2E tok/s | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre-control | No | 55 | 2 | 9.64s | 0.39 | 0.16 | 7227.58ms | 2977.20ms | 1832.12ms | 2408.08ms | 1,050,673,152 | 1,639,284,736 | 16.31s |
| R109 1 | No | 55 | 2 | 9.13s | 0.58 | 0.18 | 6678.13ms | 2649.57ms | 1892.71ms | 2125.59ms | 1,050,673,152 | 1,626,767,360 | 14.87s |
| R109 2 | No | 55 | 2 | 8.03s | 0.66 | 0.21 | 5783.37ms | 2330.80ms | 1605.08ms | 1836.88ms | 1,050,673,152 | 1,607,057,408 | 12.32s |
| R109 3 | No | 55 | 2 | 7.76s | 0.80 | 0.22 | 5523.60ms | 2264.71ms | 1498.62ms | 1749.97ms | 1,050,673,152 | 1,672,904,704 | 11.67s |

Best candidate prefill improved from `9.64s` to `7.76s`, a `1.24x` speedup or
`19.50%` TTFT/prefill reduction. Internal RLLM peak transient stayed unchanged.
Max RSS stayed far below the R108 worst observed `2,496,315,392 bytes`.

## Profile Results

Candidate profile:

| path | calls | blocks | rows | batch items | elapsed |
|---|---:|---:|---:|---:|---:|
| `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 0 | 1,611,661,312 | 7361.68ms |
| `batch_gt1_normal_output2_batch4` | 142,930,944 | 142,930,944 | 285,861,888 | 571,723,776 | 3320.06ms |
| `batch_gt1_multiply_batch4` | 109,051,904 | 109,051,904 | 0 | 436,207,616 | 860.53ms |
| `batch_gt1_normal_tail` | 22,020,096 | 22,020,096 | 0 | 22,020,096 | 323.93ms |
| `batch1_complete_linear` | 800 | 22,020,096 | 245,760 | 800 | 239.85ms |
| `batch_gt1_normal_scale` | 22,020,096 | 22,020,096 | 0 | 0 | 217.67ms |

R108 candidate profile measured `batch_gt1_normal_output2_batch4` at
`4532.71ms`. R109 reduces that row to `3320.06ms`, a `26.76%` profile-row
reduction, while preserving the same call/block counts.

## Analysis

R109 passes. The single output2 helper keeps the R108 runtime routing but avoids
two separate helper calls for adjacent output rows. The benchmark shows better
prefill, lower MLP time, unchanged internal peak transient memory, and no repeat
of R108's high RSS spike.

This is still a single-machine aarch64 runtime result. The portable fallback is
kept for other CPUs, but measured speedup is only proven on this CPU.

## Decision

accepted

Reason: `REEBUNDLE-Q8-OUTPUT2-SINGLE` improved same-turn prefill from `9.64s`
to best `7.76s`, kept output `No`, kept RLLM peak transient memory unchanged,
and reduced the output2 profile row from R108's `4532.71ms` to `3320.06ms`.

Paper value:

- useful positive evidence that RLLM can reduce exact low-RAM Q8 prefill time by
  bundling adjacent output features into one CPU helper
- stronger than R108 because it also reduces the measured output2 profile row
  and avoids the prior RSS spike
- limitation: aarch64 speedup still needs x86_64/portable-SIMD follow-up

## Next Experiment

R110 should remove the per-chunk `Vec<bool>` allocation used to mark consumed
second blocks. A row/block arithmetic skip strategy should keep R109's math and
profile row while reducing allocation/RSS risk further.
