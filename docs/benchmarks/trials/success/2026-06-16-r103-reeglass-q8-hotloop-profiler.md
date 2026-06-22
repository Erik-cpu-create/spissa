# R103: REEGLASS-Q8 Hotloop Profiler

Date: 2026-06-16
Owner: RLLM
Status: accepted diagnostic
Folder: success

## Hypothesis

R99 identified `batch_gt1_scaled` as the remaining first-order Q8 prefill path.
R100-R102 showed that batch8 widening, block64 pairing, and pre-scaled f32
sidecars do not beat R98's `REECAST` path. R103 adds detail attribution inside
the Q8 hot loop so R104 can target the correct remaining cost.

## Scope

- Mode: exact-lowram diagnostic
- REE kernel lineage: `REEGLASS-Q8-HOTLOOP-PROFILER`
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Prompt: `Answer yes or no: is fire cold?`
- Threading: `RLLM_THREADS=1`
- Integrity: `--rama-integrity unchecked`
- Diagnostics: `--profile-phases`, `RLLM_Q8_KERNEL_PROFILE=1`
- Bottleneck tag: Q8 streaming hot-loop attribution

Default runtime behavior is unchanged. Detail rows are emitted only when
`RLLM_Q8_KERNEL_PROFILE=1`.

## Setup

```bash
cargo test -p rllm-runtime q8_profile -- --nocapture
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
cargo test -p rllm-cli q8_kernel_profile_suffix -- --nocapture
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r103-control.txt 2> target/r103-control.time
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r103-profile-top8.txt 2> target/r103-profile-top8.time
```

## Runtime Results

Both runs kept the answer correct:

```text
No
```

| run | output | context tokens | prefill | decode | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | No | 55 | 9.89s | 1.42 tok/s | 7652.19ms | 3018.65ms | 1934.16ms | 2689.48ms | 1,050,673,152 | 1,739,784,192 | 14.92s |
| profile | No | 55 | 12.29s | 1.68 tok/s | 9930.09ms | 3777.33ms | 2796.73ms | 3346.20ms | 1,050,673,152 | 1,794,310,144 | 16.14s |

The profiled run is expected to be slower because it times hot-loop segments.

## Detail Profile

`Q8KernelProfile` top rows from the profiled run:

| path | calls | blocks | batch items | elapsed |
|---|---:|---:|---:|---:|
| `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 1,611,661,312 | 10589.93ms |
| `batch_gt1_normal_batch4` | 286,261,248 | 286,261,248 | 1,145,044,992 | 3551.82ms |
| `batch_gt1_normal_tail` | 22,020,096 | 22,020,096 | 22,020,096 | 1030.26ms |
| `batch_gt1_multiply_batch4` | 109,051,904 | 109,051,904 | 436,207,616 | 881.84ms |
| `batch_gt1_normal_scale` | 22,020,096 | 22,020,096 | 0 | 507.11ms |
| `batch1_complete_linear` | 800 | 22,020,096 | 800 | 277.76ms |
| `batch_gt1_multiply_tail` | 8,388,608 | 8,388,608 | 8,388,608 | 201.47ms |
| `batch_gt1_multiply_advance` | 8,388,608 | 8,388,608 | 0 | 167.75ms |

Control run did not include `Q8KernelProfile`, confirming the profiler remains
opt-in.

## Analysis

R103 rules out the simple idea that Q8 scale/dequant is the main remaining
problem. `batch_gt1_normal_scale` is only `507.11ms`, while normal-path batch4
dot work is `3551.82ms` and normal scalar tail is `1030.26ms`.

This also explains why R102 failed: pre-scaling weights into f32 removes a
smaller cost while increasing memory traffic. The hot cost is still the normal
linear batch4/tail accumulation used by gate/down.

The scalar tail is surprisingly large. With batch 55, every block has a
3-token remainder after batch4 groups. The profiler reports `22,020,096` tail
calls and `1030.26ms`, making this a credible R104 target. A batch remainder
specialization for the 3-token tail could be lower risk than another full
kernel rewrite.

## Decision

accepted diagnostic

Reason: R103 produced actionable hot-loop attribution without changing default
runtime behavior, correctness, or peak transient memory.

Paper value:

- useful profiling evidence after multiple failed micro-kernels
- shows why f32 sidecars and scale/dequant work were the wrong target
- identifies normal-path batch4 and batch remainder handling as the next target

## Next Experiment

R104 should target `batch_gt1_normal_tail` first with a small NEON/scalar
specialization for the 3-token remainder in batch 55. Keep it lab-gated and
runtime-gated; do not touch container, packer, or model format.
