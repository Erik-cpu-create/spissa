# R98: REECAST-Q8 NEON Scale Kernel

Date: 2026-06-16
Owner: RLLM
Status: accepted with marginal runtime signal
Folder: success

## Hypothesis

R97 showed that after R96 the remaining prefill bottleneck still sits in Q8
`batch_gt1_scaled`. R98 tested whether replacing scalar Q8 scale/dequant with
an aarch64 NEON helper could improve the existing R96 scaled-block plus NEON
batch4 accumulator path.

## Scope

- Mode: exact-lowram lab plus runtime gate
- REE kernel lineage: `REECAST-Q8-NEON-SCALE`
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Architecture: aarch64 NEON scale/dequant helper with portable scalar fallback
- Target branch: R93/R97 `batch_gt1_scaled`
- Bottleneck tag: CPU arithmetic / Q8 NEON scale-dequant prefill

R98 does not change `.rllm`, Q8 format, prompt formatting, sampling, tokenizer,
or memory-budget logic.

## Setup

Lab:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r98-reecast-lab.json \
  --markdown target/r98-reecast-lab.md \
  --iters 2000 \
  --batch 55
```

Runtime:

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r98-pre-control.txt 2> target/r98-pre-control.time
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r98-run${i}.txt" 2> "target/r98-run${i}.time"
done
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r98-profile.txt 2> target/r98-profile.time
```

## Lab Results

| variant | elapsed ns | speedup vs baseline | max abs diff | checksum |
|---|---:|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 92000708 | 1.000x | 0.00000000 | -15.000977 |
| `scaled_f32_dot32_batch4` | 41284959 | 2.228x | 0.00000000 | -15.000977 |
| `scaled_f32_dot32_batch4_runtime` | 34562291 | 2.662x | 0.00000000 | -15.000977 |
| `reelane_f32_dot32_batch4` | 47913250 | 1.920x | 0.00000000 | -15.000977 |
| `reeflow_i8_scaled_batch4` | 46350333 | 1.985x | 0.00000000 | -15.000977 |
| `unrolled_i8_dot32_batch4` | 91637667 | 1.004x | 0.00000000 | -15.000977 |
| `reevec_neon_f32_dot32_batch4` | 18401625 | 5.000x | 0.00000000 | -15.000977 |
| `reecast_neon_scale_batch4` | 17416916 | 5.282x | 0.00000000 | -15.000977 |

The NEON scale/dequant helper was exact and beat the R96 lab winner by `5.35%`
in this lab run.

## Runtime Results

All runtime runs kept the answer correct:

```text
No
```

| run | output | context tokens | prefill | decode | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre-control | No | 55 | 10.30s | 0.30 tok/s | 7790.16ms | 3184.22ms | 2081.62ms | 2512.94ms | 1,050,673,152 | 1,565,736,960 | 18.16s |
| R98 1 | No | 55 | 10.62s | 0.39 tok/s | 8206.30ms | 3321.64ms | 2197.96ms | 2675.19ms | 1,050,673,152 | 1,949,007,872 | 17.15s |
| R98 2 | No | 55 | 10.30s | 0.66 tok/s | 7600.66ms | 3160.52ms | 1944.26ms | 2484.23ms | 1,050,673,152 | 1,626,865,664 | 15.19s |
| R98 3 | No | 55 | 9.28s | 1.10 tok/s | 6782.23ms | 2886.93ms | 1671.18ms | 2212.64ms | 1,050,673,152 | 1,653,047,296 | 13.56s |

Profiled attribution:

| path | calls | blocks | batch items | elapsed |
|---|---:|---:|---:|---:|
| `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 1,611,661,312 | 6206.66ms |
| `batch1_complete_linear` | 800 | 22,020,096 | 800 | 278.98ms |
| `batch1_complete_multiply` | 288 | 8,388,608 | 288 | 105.63ms |

## Analysis

R98 passed the lab gate and did not regress correctness or internal peak
transient memory. Runtime best prefill improved over the immediate R98
pre-control from `10.30s` to `9.28s`.

The runtime signal is marginal when compared to R97's best normal control
(`9.31s`). This means R98 should be treated as a small accepted kernel cleanup,
not a major new speed milestone. The profiled `batch_gt1_scaled` elapsed
(`6206.66ms`) is also within the R97 profiled range (`5845.56-6138.07ms`) plus
normal run variance, so the profiler does not prove a large branch-level win.

The value of R98 is that it removes scalar Q8 scale/dequant from the aarch64 hot
path while preserving exact output and portable fallback behavior. The remaining
prefill bottleneck is still MLP/Q8 `batch_gt1_scaled`.

## Decision

accepted with marginal runtime signal

Reason: lab passed with exact output and a small win over R96's lab winner;
runtime kept output correct and internal peak transient unchanged, with best
prefill beating the immediate pre-control.

Paper value:

- useful small positive evidence for aarch64 NEON scale/dequant
- not a headline speedup
- confirms that after R96/R98 the remaining problem is still Q8 prefill MLP

## Next Experiment

R99 should not spend more time on 32-element scalar micro-shapes. The next useful
stage should either:

- profile layer/projection-specific prefill cost to see whether gate/down should
  get specialized scheduling, or
- explore a bounded layout/sidecar that reduces repeated per-block work without
  increasing resident RAM beyond the current budget.
