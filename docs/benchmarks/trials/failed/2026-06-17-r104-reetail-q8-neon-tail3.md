# R104: REETAIL Q8 NEON Tail3 Runtime Gate

Date: 2026-06-17
Owner: RLLM
Status: rejected
Folder: failed

## Hypothesis

R103 showed `batch_gt1_normal_tail` at `1030.26ms` under
`RLLM_Q8_KERNEL_PROFILE=1`. With batch 55, the normal Q8 path processes 13
batch4 groups plus a 3-row remainder per full block. R104 tested whether a
small NEON batch3 tail accumulator, `REETAIL-Q8-NEON-TAIL3-LAB`, could remove
that scalar remainder cost without changing exact math.

## Scope

- Mode: exact-lowram lab/runtime gate
- REE kernel lineage: `REETAIL-Q8-NEON-TAIL3-LAB`
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Architecture: Llama 3.2 1B Instruct Q8 transformer keep-IO row chunks
- Target device/profile: Apple A18 Pro, 8 GiB RAM, CPU-only single-thread benchmark
- Expected bottleneck: Q8 normal-path scalar tail after batch4 groups
- Bottleneck tag: CPU arithmetic / Q8 streaming hot-loop tail

The runtime candidate was tried after the lab gate passed, but was reverted
because the runtime profile did not confirm the target improvement.

## Setup

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench --json target/r104-reetail-lab.json --markdown target/r104-reetail-lab.md --iters 2000 --batch 55
target/release/q8-microbench --json target/r104-reetail-lab-long.json --markdown target/r104-reetail-lab-long.md --iters 10000 --batch 55
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r104-run${i}.txt" 2> "target/r104-run${i}.time"; done
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r104-profile.txt 2> target/r104-profile.time
```

Runtime context:

- build profile: release for benchmark binaries
- CPU: Apple A18 Pro
- RAM: 8 GiB
- OS: macOS
- relevant env/config: `RLLM_THREADS=1`, `--rama-integrity unchecked`,
  `--chat-template llama3`, `--profile-phases`, optional
  `RLLM_Q8_KERNEL_PROFILE=1`

## Lab Results

| lab | baseline | prior best | R104 candidate | max abs diff | decision |
|---|---:|---:|---:|---:|---|
| 2000 iters | `baseline_i8_dot32_batch4` 103751250ns | `reecast_neon_scale_batch4` 21697584ns | `reetail_neon_tail3_batch4` 20345375ns | 0.00000000 | lab pass |
| 10000 iters | `baseline_i8_dot32_batch4` 490368250ns | `reecast_neon_scale_batch4` 88968458ns | `reetail_neon_tail3_batch4` 85603583ns | 0.00000000 | lab pass |

The synthetic lab supported trying the runtime candidate.

## Runtime Results

All candidate runs kept the visible answer correct:

```text
No
```

| run | context tokens | generated tokens | TTFT/prefill | decode tok/s | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate 1 | 55 | 2 | 9.68s | 1.67 | 7974.82ms | 3424.61ms | 1952.15ms | 2588.35ms | 1,050,673,152 | 1,672,642,560 | 14.20s |
| candidate 2 | 55 | 2 | 9.02s | 1.61 | 6755.06ms | 3036.38ms | 1541.21ms | 2166.22ms | 1,050,673,152 | 1,883,619,328 | 12.28s |
| candidate 3 | 55 | 2 | 9.26s | 1.62 | 7469.27ms | 3340.86ms | 1708.39ms | 2409.37ms | 1,050,673,152 | 1,853,603,840 | 12.48s |

The no-profile runs were not enough to accept the change because the targeted
profile row regressed.

## Detail Profile

| path | R103 profile | R104 candidate profile | R104 inline retry profile | decision |
|---|---:|---:|---:|---|
| `batch_gt1_normal_tail` | 1030.26ms | 1213.80ms | 1226.25ms | failed |
| `batch_gt1_normal_batch4` | 3551.82ms | 3598.28ms | 3629.85ms | no improvement |
| `batch_gt1_normal_scale` | 507.11ms | 493.91ms | 507.13ms | neutral |
| `batch_gt1_scaled` | 10589.93ms | 11289.83ms | 11431.71ms | no improvement |

The inline retry added `#[inline(always)]` to the runtime helper and rebuilt
`llama-test`, but the profile still showed `batch_gt1_normal_tail` above R103.
The runtime helper was therefore removed.

## Analysis

The lab benchmark was too optimistic for this runtime shape. It isolates a
single row-like kernel and shows that a direct batch3 NEON remainder can beat
`REECAST` there, but the full streaming path measures millions of tiny block
calls with profiler boundaries, output-strided writes, and the larger MLP
pipeline around it. In that real path, the tail specialization did not reduce
the measured tail bucket.

This is not a quality failure: output stayed `No`, repetition stayed clean, and
peak transient memory stayed unchanged. It is a performance gate failure. The
runtime path was reverted to avoid merging a kernel that only wins in lab.

## Decision

rejected

Reason: `REETAIL-Q8-NEON-TAIL3-LAB` passed microbench with exact output, but the
runtime candidate failed the targeted profile gate. `batch_gt1_normal_tail`
increased from R103 `1030.26ms` to `1213.80ms` and `1226.25ms` in the retry.

Paper value:

- use as negative evidence
- useful warning that tiny tail microbench wins do not necessarily survive the
  full streaming runtime profile

## Next Experiment

R105 should stop optimizing the three-row remainder in isolation. The next
credible target is the larger `batch_gt1_normal_batch4` bucket or a streaming
loop-level change that reduces per-block call/profiling overhead without adding
sidecars or extra RAM.
