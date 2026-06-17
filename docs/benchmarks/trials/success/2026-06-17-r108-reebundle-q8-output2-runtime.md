# R108: REEBUNDLE Q8 Output2 Runtime Gate

Date: 2026-06-17
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

R107 proved in the lab that bundling two adjacent Q8 output features for the
same input block can reduce loop-level work while preserving exact output.
R108 tests whether a conservative runtime gate for that shape improves the real
Llama 3.2 1B Q8 prefill path.

## Scope

- Mode: exact-lowram runtime gate
- REE kernel lineage: `REEBUNDLE-Q8-OUTPUT2`
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Prompt: `Answer yes or no: is fire cold?`
- Threading: `RLLM_THREADS=1`
- Integrity: `--rama-integrity unchecked`
- Diagnostics: `--profile-phases`, optional `RLLM_Q8_KERNEL_PROFILE=1`
- Bottleneck tag: CPU arithmetic / Q8 output-feature bundling

The change is limited to normal batch>1 Q8 linear accumulation. It does not
change `.rllm` format, Q8 format, tokenizer, prompt template, sampling policy,
memory budget logic, Q8 multiply-into, Q8 argmax, or batch1 row fast paths.

## Setup

Pre-control, before runtime source change:

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r108-pre-control.txt 2> target/r108-pre-control.time
```

Candidate:

```bash
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r108-run${i}.txt" 2> "target/r108-run${i}.time"
done
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r108-profile.txt 2> target/r108-profile.time
```

Profile control was also run from a detached R107 worktree at commit `2dd3b91`
to preserve the pre-R108 runtime code:

```bash
git worktree add --detach /tmp/rllm-r108-profile-control 2dd3b91
cd /tmp/rllm-r108-profile-control
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model /Users/deansanbhnanwr/Projects/rllm/models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > /Users/deansanbhnanwr/Projects/rllm/target/r108-pre-profile-control-warm.txt 2> /Users/deansanbhnanwr/Projects/rllm/target/r108-pre-profile-control-warm.time
```

Runtime context:

- build profile: release
- target benchmark: single-thread CPU-only
- relevant env/config: `RLLM_THREADS=1`, `--rama-integrity unchecked`,
  `--chat-template llama3`, `--profile-phases`
- architecture fast path: existing `target_arch = "aarch64"` helpers
- fallback: existing portable scalar helper chain

## Runtime Results

All runs kept the visible answer correct:

```text
No
```

| run | output | context tokens | generated tokens | TTFT / prefill | decode tok/s | E2E tok/s | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre-control | No | 55 | 2 | 9.28s | 1.69 | 0.20 | 7592.12ms | 3127.93ms | 1914.57ms | 2539.61ms | 1,050,673,152 | 1,844,117,504 | 13.74s |
| R108 1 | No | 55 | 2 | 8.24s | 1.10 | 0.22 | 6099.64ms | 2559.89ms | 1469.71ms | 2059.58ms | 1,050,673,152 | 2,130,460,672 | 12.27s |
| R108 2 | No | 55 | 2 | 7.55s | 1.47 | 0.24 | 6234.80ms | 2610.73ms | 1491.10ms | 2122.15ms | 1,050,673,152 | 2,496,315,392 | 10.68s |
| R108 3 | No | 55 | 2 | 8.78s | 0.69 | 0.20 | 6528.98ms | 2668.15ms | 1569.21ms | 2280.61ms | 1,050,673,152 | 2,410,708,992 | 12.80s |

Best candidate prefill improved from `9.28s` to `7.55s`, a `1.23x` speedup
or `18.64%` reduction in TTFT/prefill. Internal RLLM peak transient memory stayed
unchanged at `1,050,673,152 bytes`.

Process max RSS increased in the candidate runs. The internal memory-budgeted
transient peak did not grow, but the RSS movement is a limitation and should be
watched in the next runtime slice.

## Profile Results

Candidate profile:

| path | calls | blocks | rows | batch items | elapsed |
|---|---:|---:|---:|---:|---:|
| `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 0 | 1,611,661,312 | 8520.42ms |
| `batch_gt1_normal_output2_batch4` | 142,930,944 | 142,930,944 | 285,861,888 | 571,723,776 | 4532.71ms |
| `batch_gt1_multiply_batch4` | 109,051,904 | 109,051,904 | 0 | 436,207,616 | 858.53ms |
| `batch_gt1_normal_tail` | 22,020,096 | 22,020,096 | 0 | 22,020,096 | 310.26ms |
| `batch1_complete_linear` | 800 | 22,020,096 | 245,760 | 800 | 271.47ms |
| `batch_gt1_normal_scale` | 22,020,096 | 22,020,096 | 0 | 0 | 223.37ms |

Warm R107 profile control from detached worktree:

| path | calls | blocks | rows | batch items | elapsed |
|---|---:|---:|---:|---:|---:|
| `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 0 | 1,611,661,312 | 29334.37ms |
| `batch_gt1_normal_batch4` | 286,261,248 | 286,261,248 | 0 | 1,145,044,992 | 24174.49ms |
| `batch_gt1_normal_batch4_kernel` | 286,261,248 | 286,261,248 | 0 | 1,145,044,992 | 5781.06ms |
| `batch_gt1_normal_batch4_setup` | 286,261,248 | 286,261,248 | 0 | 0 | 5061.13ms |
| `batch_gt1_multiply_batch4` | 109,051,904 | 109,051,904 | 0 | 436,207,616 | 896.29ms |
| `batch_gt1_normal_tail` | 22,020,096 | 22,020,096 | 0 | 22,020,096 | 471.23ms |
| `batch_gt1_normal_scale` | 22,020,096 | 22,020,096 | 0 | 0 | 429.27ms |

The control profile is much heavier because the pre-R108 path records per-call
batch4 setup and kernel timings. The useful signal is that the new
`batch_gt1_normal_output2_batch4` row is non-zero and replaces the old normal
batch4 profile shape in the candidate run.

## Analysis

R108 clears the runtime gate. The candidate keeps exact visible output for the
benchmark prompt, keeps internal peak transient memory unchanged, and improves
same-turn no-profile prefill by more than the required `5%` threshold.

The implementation is intentionally conservative. It detects two adjacent output
rows that share the same 32-element input block, scales both Q8 blocks once, and
accumulates both output features for the same batch4 input lanes. Unsafe shapes
fall back to the previous single-output path.

This is not yet the final output2 kernel. The current runtime helper still calls
the existing batch4 helper twice. R107's lab showed a stronger single-helper
shape, so R109 should reduce the remaining helper overhead and address RSS noise
without changing output math.

## Decision

accepted

Reason: `REEBUNDLE-Q8-OUTPUT2` improved same-turn no-profile prefill from
`9.28s` to best `7.55s` with output `No` and unchanged RLLM peak transient
memory.

Paper value:

- useful positive evidence for exact low-RAM Q8 output-feature bundling
- useful runtime proof that R107's lab direction survives real Llama 3.2 1B Q8
  prefill
- limitation: process max RSS increased in this run and needs follow-up

## Next Experiment

R109 should keep the accepted output2 routing but replace the conservative
two-helper wrapper with a single output2 NEON helper modeled after R107's lab
kernel. It should also remove or reduce any avoidable per-chunk allocation and
rerun the same RSS-sensitive benchmark gate.
