# R106: REEGLASS Q8 Batch4 Split Profiler

Date: 2026-06-17
Owner: RLLM
Status: accepted diagnostic
Folder: success

## Hypothesis

R104 and R105 both rejected plausible-looking tiny runtime changes. R106 tests a
diagnostic hypothesis instead: split the large `batch_gt1_normal_batch4` bucket
into setup and kernel timing so R107 targets the measured source of cost instead
of guessing.

## Scope

- Mode: exact-lowram diagnostic
- REE kernel lineage: `REEGLASS-Q8-BATCH4-SPLIT-PROFILER`
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Prompt: `Answer yes or no: is fire cold?`
- Threading: `RLLM_THREADS=1`
- Integrity: `--rama-integrity unchecked`
- Diagnostics: `--profile-phases`, optional `RLLM_Q8_KERNEL_PROFILE=1`
- Target device/profile: Apple A18 Pro, 8 GiB RAM, CPU-only single-thread benchmark
- Bottleneck tag: CPU arithmetic / Q8 streaming hot-loop attribution

Default runtime behavior is unchanged. The new detail rows are emitted only
when `RLLM_Q8_KERNEL_PROFILE=1`.

## Setup

```bash
cargo test -p rllm-runtime q8_profile -- --nocapture
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
cargo test -p rllm-cli --bin llama-test q8_kernel_profile_suffix -- --nocapture
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r106-control.txt 2> target/r106-control.time
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r106-profile.txt 2> target/r106-profile.time
```

## Runtime Results

Both runs kept the visible answer correct:

```text
No
```

| run | output | context tokens | generated tokens | TTFT/prefill | decode tok/s | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | No | 55 | 2 | 9.58s | 1.44 | 7674.52ms | 3181.31ms | 1940.74ms | 2542.11ms | 1,050,673,152 | 1,644,658,688 | 14.45s |
| profile | No | 55 | 2 | 33.60s | 0.42 | 25194.43ms | 10970.78ms | 3103.42ms | 11108.86ms | 1,050,673,152 | 1,640,235,008 | 38.95s |

The profiled run is intentionally much slower because it now calls timers inside
the batch4 inner loop. This is acceptable for R106 because the change is
diagnostic and opt-in.

## Detail Profile

`Q8KernelProfile` top rows from the profiled run:

| path | calls | blocks | batch items | elapsed |
|---|---:|---:|---:|---:|
| `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 1,611,661,312 | 30519.36ms |
| `batch_gt1_normal_batch4` | 286,261,248 | 286,261,248 | 1,145,044,992 | 25008.82ms |
| `batch_gt1_normal_batch4_kernel` | 286,261,248 | 286,261,248 | 1,145,044,992 | 6015.19ms |
| `batch_gt1_normal_batch4_setup` | 286,261,248 | 286,261,248 | 0 | 5241.44ms |
| `batch_gt1_multiply_batch4` | 109,051,904 | 109,051,904 | 436,207,616 | 1014.57ms |
| `batch_gt1_normal_tail` | 22,020,096 | 22,020,096 | 22,020,096 | 491.96ms |
| `batch_gt1_normal_scale` | 22,020,096 | 22,020,096 | 0 | 471.82ms |
| `batch1_complete_linear` | 800 | 22,020,096 | 800 | 228.66ms |

Computed batch4 residual:

```text
25008.82ms - 5241.44ms - 6015.19ms = 13752.19ms
```

The residual is not pure business logic cost. It includes loop overhead plus
the cost of per-iteration instrumentation itself. The important signal is that
the full batch4 aggregate cannot be explained by NEON kernel math alone.

## Analysis

R106 confirms that the remaining Q8 batch4 problem is not a single obvious
inner-kernel fix. With fine-grained timing enabled, the existing helper call
accounts for `6015.19ms`, setup accounts for `5241.44ms`, and the residual is
`13752.19ms`. The profiler itself is now heavy enough that its absolute numbers
should not be compared as speed measurements against R103-R105.

The useful result is directional: there is meaningful cost outside the NEON dot
helper. R107 should not start by rewriting the dot32 math again. It should
either reduce the number of batch4 loop iterations by processing more work per
loaded/scaled Q8 block, or first add a coarser profiler that measures groups of
batch4 calls without one timer pair per iteration.

## Decision

accepted diagnostic

Reason: R106 added opt-in attribution rows, preserved default behavior, kept the
answer correct, kept peak transient memory unchanged, and identified that the
batch4 aggregate is dominated by loop/instrumentation/setup around the helper,
not just kernel math.

Paper value:

- useful diagnostic evidence after R104/R105 failed runtime gates
- shows why another tiny kernel hint is unlikely to be enough
- supports a loop-level or coarser-profile R107 design

## Next Experiment

R107 should target loop-level work, not another scalar hint. Recommended name:
`REEBUNDLE-Q8-OUTPUT2-LAB`. The first design should test whether processing two
adjacent output features per scaled input block can reduce loop/setup overhead
without adding persistent sidecars or increasing peak transient memory.
