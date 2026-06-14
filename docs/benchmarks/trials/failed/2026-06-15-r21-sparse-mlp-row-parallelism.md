# Trial: R21 Sparse MLP Row Parallelism

Date: 2026-06-15
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

R20 showed that Llama 3.2 1B Instruct decode is dominated by MLP projection
work. R21 tests whether sparse MLP row work can scale across available CPU
threads without changing the `.rllm` model format.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Expected bottleneck: sparse MLP row accumulation
- Bottleneck tag: sparse MLP row parallelism
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Parallel sparse gate: `RLLM_SPARSE_PARALLEL=1`
- Thread control: `RLLM_THREADS`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Top-k: `RLLM_AIP_TOPK=128`

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 8

printf 'good morning\nexit\n' | \
  RLLM_SPARSE_PARALLEL=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 8
```

Runtime context:

- build profile: release
- default sparse path remains sequential unless `RLLM_SPARSE_PARALLEL=1`
- parallel kernel only runs for row-aligned complete-row raw chunks; unsupported
  chunk layouts keep the sequential path

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | AIP calls | fallbacks | max top-k | repeated ratio | max run | unique tokens | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | AIP speed top-k 128, default sparse sequential | 8 | 13.34s | 0.32 | 0.23 | 224 | 32 | 128 | 0.86 | 7 | 2/8 | 2446016512 | 1620855736 | 1050689536 |
| Llama-3.2-1B-Instruct | AIP speed top-k 128, `RLLM_SPARSE_PARALLEL=1` | 8 | 11.65s | 0.26 | 0.21 | 224 | 32 | 128 | 0.86 | 7 | 2/8 | 2382987264 | 1620872552 | 1050689536 |

Additional 16-token observation:

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | AIP speed top-k 128, default sparse sequential | 16 | 14.00s | 0.35 | 0.28 | 2476900352 | 1620938040 | 1050689536 |
| Llama-3.2-1B-Instruct | AIP speed top-k 128, pre-gate auto sparse parallel observation | 16 | 13.08s | 0.36 | 0.29 | 2470051840 | 1621020008 | 1050689536 |

## Analysis

The parallel sparse kernels are functionally correct in unit tests, but the
runtime experiment does not improve speed. On the fair 8-token comparison,
opt-in sparse parallelism is slower (`0.26 tok/s`) than the default sequential
sparse path (`0.32 tok/s`).

The most likely reason is scheduling and memory-traffic overhead. R20 showed
that sparse AIP still touches row-major weight chunks. R21 adds thread work on
top of the same chunk traffic, so it cannot solve the deeper layout problem.
The `/usr/bin/time` counters also showed high page-fault counts during these
runs, so this result should be treated as a negative signal for this strategy,
not as a stable final speed baseline for RLLM overall.

RLLM tracked peak transient memory stayed flat at `1050689536` bytes.

## Decision

failed

Reason: sparse MLP row parallelism did not improve decode speed and remains far
below the 30-40 tok/s target.

Paper value:
- useful negative result for thread-per-chunk sparse row parallelism
- useful evidence that CPU parallelism alone does not solve row-major sparse
  memory traffic
- useful implementation guard: failed strategy is opt-in via
  `RLLM_SPARSE_PARALLEL=1`, not default behavior

## Next Experiment

R22 should target the layout problem directly. A likely direction is an
experimental activation-column sidecar or input-tile projection layout so sparse
AIP can read selected input dimensions without scanning full row-major MLP
weights.
