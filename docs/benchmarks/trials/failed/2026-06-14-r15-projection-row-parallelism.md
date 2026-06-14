# Trial: R15 CPU-Aware Projection Row Parallelism

Date: 2026-06-14
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

Parallelizing complete rows in the raw 16-bit batch-1 projection kernels should
improve CPU-only decode throughput without increasing model RAM, because
transformer projections remain a larger bottleneck than LM-head argmax.

## Scope

- Mode: exact-lowram
- Models/artifacts: `models/SmolLM2-135M-raw.rllm`, `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: single CPU package, low RAM, multiple CPU threads when available
- Bottleneck tag: transformer projection row parallelism

## Implementation Attempted

- Added a CPU-aware raw projection row splitter for complete rows.
- Routed raw FP16 and raw BF16 batch-1 projection chunks through the parallel
  path when more than one runtime thread was available.
- Kept partial rows on the existing sequential path.
- Tested both uncapped auto parallelism and a capped auto policy that limited
  default projection workers to two threads unless `RLLM_THREADS` was set.

The attempted code was rejected and removed after measurement. This report keeps
the evidence only.

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  RLLM_THREADS=1 /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

The Llama 3.2 1B runs used the same pattern with
`models/Llama-3.2-1B-Instruct-raw.rllm`.

## Results

Host logical CPUs: `6`.

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | baseline, `RLLM_THREADS=1` | 16 | 1.44 s | 20.16 | 7.33 | 459800576 bytes | 189793024 bytes | 113262592 bytes |
| SmolLM2-135M | uncapped auto projection parallelism | 16 | 0.99 s | 19.28 | 9.05 | 459685888 bytes | 189678336 bytes | 113262592 bytes |
| SmolLM2-135M | manual `RLLM_THREADS=2` | 16 | 1.34 s | 20.82 | 7.75 | 459636736 bytes | 189629184 bytes | 113262592 bytes |
| SmolLM2-135M | capped auto projection parallelism | 16 | 1.41 s | 12.84 | 6.20 | 458997760 bytes | 188990208 bytes | 113262592 bytes |
| Llama-3.2-1B-Instruct | baseline, `RLLM_THREADS=1` | 16 | 12.24 s | 1.47 | 0.71 | 2543550464 bytes | 1620413728 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | uncapped auto projection parallelism | 16 | 10.23 s | 1.35 | 0.75 | 2499297280 bytes | 1620708640 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | manual `RLLM_THREADS=2` | 16 | 11.83 s | 1.55 | 0.74 | 2623537152 bytes | 1620577568 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | capped auto projection parallelism | 16 | 11.48 s | 1.41 | 0.72 | 2552381440 bytes | 1620675896 bytes | 1050689536 bytes |

## Analysis

The RAM objective held: RLLM peak transient memory stayed flat for both models.
That part is useful, but the speed objective failed.

Decode throughput did not improve reliably. SmolLM2 uncapped auto fell from
`20.16` to `19.28 tok/s`, and capped auto fell sharply to `12.84 tok/s`. Llama
3.2 1B uncapped auto fell from `1.47` to `1.35 tok/s`, and capped auto stayed
below baseline at `1.41 tok/s`.

Manual two-thread runs were noisy: SmolLM2 reached `20.82 tok/s`, and Llama 1B
reached `1.55 tok/s`, but those gains were not stable enough to justify a
default runtime change. The capped auto result is the stronger decision signal
because it represents the safe default policy RLLM would expose to users.

The likely bottleneck is not thread count alone. Scoped per-call projection
parallelism creates overhead at a granularity that is too small for these
matvec calls, and it can add cache pressure and memory bandwidth contention.
End-to-end numbers sometimes look better because TTFT/prefill timing is noisy,
but decode tok/s is the R15 acceptance metric.

## Decision

failed

Reason: runtime peak memory stayed flat, but default projection row parallelism
regressed decode speed and did not move RLLM toward the 30-40 tok/s target.

Paper value:

- useful negative result: naive projection row threading is insufficient for
  RLLM's low-RAM CPU-only path
- useful design constraint: CPU auto-scaling needs larger-grain scheduling or
  persistent workers, not scoped thread spawning inside small projection calls

## Next Experiment

Do not merge per-call projection row parallelism. R16 should target one of the
heavier projection paths with a lower-overhead design: persistent worker reuse,
larger-grain layer/tensor scheduling, SIMD raw BF16 dot kernels, or a packed
projection layout that reduces scalar conversion and memory traffic before
adding more threads.
