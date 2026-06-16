# R93: REETHINK-Q8 Runtime Shape Profiler

Date: 2026-06-16
Owner: RLLM
Status: accepted diagnostic
Folder: success

## Hypothesis

R91 and R92 showed that synthetic microbench wins can fail when promoted to the
full runtime path. R93 tests a narrower claim: before adding another Q8 kernel,
RLLM needs opt-in runtime attribution for the exact Q8 branches used by
`llama-test`.

## Scope

- Mode: exact-lowram diagnostic, single-thread CPU-only
- REE kernel lineage: `REETHINK-Q8-SHAPE-PROFILER`
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Architecture: Q8_0 streaming linear, multiply-into, and argmax paths
- Target device/profile: CPU-only, single-thread benchmark
- Bottleneck tag: CPU arithmetic / runtime branch attribution

R93 does not add a speed kernel. It adds disabled-by-default profiler state,
branch instrumentation, and a compact `Q8KernelProfile` metrics suffix when
`RLLM_Q8_KERNEL_PROFILE=1` is enabled.

## Setup

Validation:

```bash
cargo test -p rllm-runtime q8_profile -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-cli --bin llama-test q8_kernel_profile -- --nocapture
cargo build --release -p rllm-cli --bin llama-test
```

Control:

```bash
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r93-control.txt 2> target/r93-control.time
```

Profiled:

```bash
for i in 1 2; do
  RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r93-q8-profile-run${i}.txt" 2> "target/r93-q8-profile-run${i}.time"
done
```

Runtime context:

- build profile: release
- relevant env/config: `RLLM_THREADS=1`, `--rama-integrity unchecked`, `--profile-phases`
- profiler env for diagnostic runs: `RLLM_Q8_KERNEL_PROFILE=1`

## Results

All runs kept the answer correct:

```text
No
```

Control without Q8 profile:

| run | output | context tokens | prefill | decode | MLP total | peak transient | max RSS | elapsed | Q8 profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| control | No | 55 | 12.70s | 0.48 tok/s | 9842.23ms | 1,050,673,152 | 1,643,626,496 | 19.14s | absent |

Profiled runs:

| run | output | context tokens | prefill | decode | MLP total | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| profile 1 | No | 55 | 14.96s | 0.75 tok/s | 11327.71ms | 1,050,673,152 | 1,674,051,584 | 18.81s |
| profile 2 | No | 55 | 15.16s | 0.74 tok/s | 11531.29ms | 1,050,673,152 | 1,640,726,528 | 19.29s |

Q8 profile rows:

| run | top path | calls | blocks | rows | batch items | elapsed |
|---|---|---:|---:|---:|---:|---:|
| profile 1 | `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 0 | 1,611,661,312 | 9931.51ms |
| profile 1 | `batch1_complete_linear` | 800 | 22,020,096 | 245,760 | 800 | 264.79ms |
| profile 1 | `batch1_complete_multiply` | 288 | 8,388,608 | 131,072 | 288 | 100.80ms |
| profile 2 | `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 0 | 1,611,661,312 | 10717.01ms |
| profile 2 | `batch1_complete_linear` | 800 | 22,020,096 | 245,760 | 800 | 270.04ms |
| profile 2 | `batch1_complete_multiply` | 288 | 8,388,608 | 131,072 | 288 | 104.93ms |

The default control metrics line did not contain `Q8KernelProfile`. Profiled
runs did contain `Q8KernelProfile: kernel=REETHINK-Q8-SHAPE-PROFILER`.

## Analysis

R93 confirms that the runtime branch to target next is `batch_gt1_scaled`, not
the batch1 complete-row helpers. In both profiled runs, `batch_gt1_scaled`
dominates Q8 branch elapsed time by roughly two orders of magnitude over the
batch1 helper paths.

This also explains why R92 failed: optimizing a batch1 decode-shaped path did
not touch the dominant prefill work for the current prompt/model shape. The hot
prefill workload is repeatedly scaling Q8 blocks and applying batch>1 dot32
work across projection chunks.

R93 should not be interpreted as a speedup. The accepted value is attribution:
it preserves output correctness, keeps peak transient memory unchanged, keeps
profile output opt-in, and identifies the next kernel target with measured
runtime evidence.

## Decision

accepted diagnostic

Reason: `REETHINK-Q8-SHAPE-PROFILER` produced opt-in Q8 branch attribution,
kept the output correct, and did not change default output formatting when the
profiler env was absent.

Paper value:

- useful process evidence that runtime branch attribution is required before
  kernel promotion
- useful bottleneck evidence that the next kernel should target
  `batch_gt1_scaled` Q8 prefill work
- not a speed result

## Next Experiment

R94 should target the `batch_gt1_scaled` branch directly. The next candidate
should avoid batch1-only decode kernels and instead reduce the cost of repeated
Q8 block scaling plus batch dot32 work in the prefill path while preserving the
current RAM ceiling and exact output.
