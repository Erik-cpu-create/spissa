# R96: REEVEC-Q8 NEON Batch4 Kernel

Date: 2026-06-16
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

R94 and R95 showed that scalar reshapes of the Q8 batch4 prefill hot path were
slower than the current scaled-block runtime shape. R96 tested whether an
aarch64 NEON f32x4 accumulator could improve the R93 `batch_gt1_scaled` branch
while preserving the portable scalar fallback for non-aarch64 CPUs.

## Scope

- Mode: exact-lowram lab plus runtime gate
- REE kernel lineage: `REEVEC-Q8-NEON-BATCH4`
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Architecture: aarch64 NEON fast path with portable fallback
- Target branch: R93 `batch_gt1_scaled`
- Bottleneck tag: CPU arithmetic / Q8 NEON batch4 prefill

R96 does not change the `.spsa` format, Q8 format, tokenizer, prompt template,
sampling policy, or memory-budget logic.

## Setup

Lab:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r96-reevec-lab.json \
  --markdown target/r96-reevec-lab.md \
  --iters 2000 \
  --batch 55
```

Runtime pre-control:

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r96-pre-control.txt 2> target/r96-pre-control.time
```

Runtime post-change:

```bash
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r96-run${i}.txt" 2> "target/r96-run${i}.time"
done
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r96-profile.txt 2> target/r96-profile.time
```

Runtime context:

- build profile: release
- relevant env/config: `RLLM_THREADS=1`, `--rama-integrity unchecked`, `--profile-phases`
- architecture fast path: `target_arch = "aarch64"`
- fallback: portable scalar helper on non-aarch64

## Lab Results

| variant | elapsed ns | speedup vs baseline | max abs diff | checksum |
|---|---:|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 99158792 | 1.000x | 0.00000000 | -15.000977 |
| `scaled_f32_dot32_batch4` | 38024209 | 2.608x | 0.00000000 | -15.000977 |
| `scaled_f32_dot32_batch4_runtime` | 33595042 | 2.952x | 0.00000000 | -15.000977 |
| `reelane_f32_dot32_batch4` | 49054166 | 2.021x | 0.00000000 | -15.000977 |
| `reeflow_i8_scaled_batch4` | 45528709 | 2.178x | 0.00000000 | -15.000977 |
| `unrolled_i8_dot32_batch4` | 89722583 | 1.105x | 0.00000000 | -15.000977 |
| `reevec_neon_f32_dot32_batch4` | 17933958 | 5.529x | 0.00000000 | -15.000977 |

The NEON variant was exact and beat the runtime-shaped scalar baseline by
`1.873x` in the lab.

## Runtime Results

All runtime runs kept the answer correct:

```text
No
```

| run | output | context tokens | prefill | decode | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre-control | No | 55 | 13.85s | 0.42 tok/s | 10436.33ms | 3533.50ms | 3323.09ms | 3566.16ms | 1,050,673,152 | 1,633,746,944 | 20.80s |
| R96 1 | No | 55 | 9.78s | 0.95 tok/s | 7802.20ms | 3195.58ms | 2066.99ms | 2528.46ms | 1,050,673,152 | 1,713,094,656 | 15.13s |
| R96 2 | No | 55 | 9.03s | 1.40 tok/s | 6734.20ms | 2861.48ms | 1709.09ms | 2151.82ms | 1,050,673,152 | 1,867,874,304 | 13.18s |
| R96 3 | No | 55 | 9.19s | 1.44 tok/s | 7135.61ms | 3083.76ms | 1810.60ms | 2228.90ms | 1,050,673,152 | 1,709,801,472 | 13.24s |

Best prefill improved from `13.85s` to `9.03s`, a `1.53x` prefill speedup in
this run. Internal peak transient memory stayed unchanged at
`1,050,673,152 bytes`. Process max RSS varied upward in two runs, so RSS should
continue to be watched, but the RLLM memory-budgeted transient peak did not
grow.

Profiled attribution after the runtime change:

| path | calls | blocks | batch items | elapsed |
|---|---:|---:|---:|---:|
| `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 1,611,661,312 | 6129.00ms |
| `batch1_complete_linear` | 800 | 22,020,096 | 800 | 314.01ms |
| `batch1_complete_multiply` | 288 | 8,388,608 | 288 | 118.62ms |

R93 profiled runs measured `batch_gt1_scaled` at `9931.51-10717.01ms`, so R96
reduced the measured hot-branch elapsed time materially while keeping the same
branch attribution.

## Analysis

R96 is the first post-R93 kernel promotion that clears both lab and runtime
gates. The key difference from R94/R95 is that it does not try to outsmart the
compiler with scalar loop reshaping. It keeps the current scaled f32 block and
uses aarch64 NEON to perform f32x4 multiply-add across the 32-element block for
four batch lanes.

The portable scalar path remains in the codebase and is used on non-aarch64
targets. Therefore this is not an Apple-only runtime design, but the measured
speedup is only proven on the current aarch64 machine.

## Decision

accepted

Reason: `REEVEC-Q8-NEON-BATCH4` passed lab with exact output and improved the
single-thread Llama 3.2 1B Q8 prefill from `13.85s` to best `9.03s`, with
correct answer and unchanged internal peak transient memory.

Paper value:

- useful positive evidence that RLLM can improve exact low-RAM CPU-only Q8
  prefill with architecture-aware kernels
- useful process evidence that lab gate plus runtime gate prevents bad scalar
  promotions
- limitation: speedup is measured on aarch64 NEON; other CPU architectures still
  need their own fast path or portable SIMD strategy

## Next Experiment

R97 should either:

- add an x86_64 SIMD equivalent behind the same wrapper pattern, or
- target another measured hot path after rerunning the R93 profiler with R96 in
  place.

Do not return to scalar Q8 batch4 loop reshaping unless a new profile shows a
different scalar bottleneck.
