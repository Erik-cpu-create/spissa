# R105: REEINLINE Q8 Batch4 Callsite Runtime Gate

Date: 2026-06-17
Owner: RLLM
Status: rejected
Folder: failed

## Hypothesis

R104 showed that optimizing the 3-row scalar tail in isolation did not survive
the full runtime profile. R105 tested a narrower callsite-overhead hypothesis:
force the existing Q8 normal batch4 wrapper and aarch64 NEON implementation to
inline, without changing the algorithm, data layout, model format, output math,
or RAM footprint.

## Scope

- Mode: exact-lowram runtime gate
- REE kernel lineage: `REEINLINE-Q8-BATCH4-CALLSITE`
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Prompt: `Answer yes or no: is fire cold?`
- Threading: `RLLM_THREADS=1`
- Integrity: `--rama-integrity unchecked`
- Diagnostics: `--profile-phases`, optional `RLLM_Q8_KERNEL_PROFILE=1`
- Bottleneck tag: CPU arithmetic / Q8 batch4 callsite

The runtime candidate was reverted because the targeted profile row regressed.

## Setup

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r105-pre-control.txt 2> target/r105-pre-control.time
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r105-pre-profile.txt 2> target/r105-pre-profile.time
cargo fmt
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r105-run${i}.txt" 2> "target/r105-run${i}.time"; done
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r105-profile.txt 2> target/r105-profile.time
```

Runtime context:

- build profile: release for `llama-test`
- CPU: Apple A18 Pro
- RAM: 8 GiB
- OS: macOS
- relevant env/config: `RLLM_THREADS=1`, `--rama-integrity unchecked`,
  `--chat-template llama3`, `--profile-phases`

## Runtime Results

All candidate runs kept the visible answer correct:

```text
No
```

| run | context tokens | generated tokens | TTFT/prefill | decode tok/s | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre-control | 55 | 2 | 9.61s | 1.29 | 7282.18ms | 2915.61ms | 1825.00ms | 2531.96ms | 1,050,673,152 | 1,647,984,640 | 14.73s |
| candidate 1 | 55 | 2 | 9.38s | 0.47 | 6712.09ms | 2745.31ms | 1612.82ms | 2343.81ms | 1,050,673,152 | 1,624,113,152 | 15.13s |
| candidate 2 | 55 | 2 | 9.37s | 1.57 | 7067.06ms | 2959.98ms | 1649.75ms | 2446.29ms | 1,050,673,152 | 1,652,654,080 | 12.98s |
| candidate 3 | 55 | 2 | 9.65s | 1.53 | 7114.66ms | 2939.83ms | 1655.64ms | 2508.26ms | 1,050,673,152 | 1,699,020,800 | 13.10s |

The best no-profile candidate prefill, `9.37s`, was slightly better than the
same-turn pre-control `9.61s`. That was not enough to accept the change because
the targeted Q8 profile row regressed.

## Detail Profile

| path | R103 profile | same-turn pre-profile | R105 candidate profile | decision |
|---|---:|---:|---:|---|
| `batch_gt1_scaled` | 10589.93ms | 10914.93ms | 11403.76ms | failed |
| `batch_gt1_normal_batch4` | 3551.82ms | 3727.99ms | 3867.66ms | failed |
| `batch_gt1_normal_tail` | 1030.26ms | 1043.31ms | 1096.10ms | failed |
| `batch_gt1_multiply_batch4` | 881.84ms | 918.71ms | 960.12ms | failed |
| `batch_gt1_normal_scale` | 507.11ms | 525.24ms | 565.35ms | failed |

The optional scaled-block wrapper retry was skipped because the batch4-only
candidate clearly failed the profile gate rather than producing a neutral or
mixed result.

## Analysis

The result rejects the hypothesis that plain `#[inline(always)]` on the current
batch4 wrapper is enough to lower the real runtime hot path. The measured
profile moved in the wrong direction across the main target row and related
Q8 rows. The slight no-profile prefill improvement is within run variance and
does not justify keeping a source change that fails the direct profile gate.

This is not a correctness failure. The visible output stayed `No`, repetition
metrics stayed clean, and peak transient memory stayed unchanged. It is a
performance-gate failure, so the runtime source change was reverted.

## Decision

rejected

Reason: `batch_gt1_normal_batch4` increased from the same-turn baseline
`3727.99ms` to `3867.66ms`, and also stayed worse than R103 `3551.82ms`.

Paper value:

- use as negative evidence
- useful warning that compiler hinting alone is not enough for the remaining Q8
  streaming bottleneck

## Next Experiment

R106 should stop testing callsite hints. The next credible target is a real
loop-level reduction: either process multiple output features per loaded input
tile, or add a profiler that separates per-block loop overhead from NEON dot
work before changing the kernel again.
