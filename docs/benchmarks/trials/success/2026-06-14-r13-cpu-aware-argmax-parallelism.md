# Trial: R13 CPU-Aware Argmax Parallelism

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

RLLM can improve CPU-only decode throughput by parallelizing row-independent
LM-head argmax work while keeping RAM roughly flat. The experiment should avoid
duplicating model tensors, KV cache, or full logits buffers.

## Scope

- Mode: exact-lowram
- Models/artifacts: `models/SmolLM2-135M-raw.spsa`, `models/Llama-3.2-1B-Instruct-raw.spsa`
- Architecture: llama
- Target device/profile: single CPU package, low RAM, multiple CPU threads when available
- Bottleneck tag: CPU row parallelism

## Implementation

- Added `RLLM_THREADS` runtime override.
- Added auto CPU detection via `std::thread::available_parallelism()`.
- Parallelized complete raw 16-bit LM-head argmax rows with scoped worker threads.
- Kept partial chunk rows on the sequential path.
- Added an adaptive auto policy: large-vocab argmax defaults to a 2-thread cap
  unless `RLLM_THREADS` explicitly overrides it.

The design keeps per-worker state to a local best `(token, value)` candidate and
does not allocate a full logits vector per worker.

## Setup

Commands:

```bash
printf 'good morning\nexit\n' | \
  RLLM_THREADS=1 /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

The Llama 3.2 1B runs used the same pattern with `--max-new-tokens 4`.

## Results

Host logical CPUs: `6`.

| model | threads | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | max RSS | peak footprint | RLLM peak transient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | 1 | 16 | 1.38 s | 19.25 | 7.42 | 459472896 bytes | 189465344 bytes | 113262592 bytes |
| SmolLM2-135M | auto | 16 | 1.12 s | 20.61 | 8.66 | 459898880 bytes | 189891328 bytes | 113262592 bytes |
| Llama-3.2-1B-Instruct | 1 | 4 | 10.99 s | 0.59 | 0.25 | 2557640704 bytes | 1620512032 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | auto | 4 | 11.96 s | 0.70 | 0.25 | 2494660608 bytes | 1620479264 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | 6 manual | 4 | 10.91 s | 0.64 | 0.26 | 2494578688 bytes | 1620626792 bytes | 1050689536 bytes |

## Analysis

R13 improves decode throughput without increasing RLLM's tracked transient
memory. SmolLM2 improves from `19.25` to `20.61 tok/s`, about `1.07x`. Llama
3.2 1B improves from `0.59` to `0.70 tok/s`, about `1.19x`, but end-to-end
throughput remains dominated by prefill and broader transformer cost.

The first uncapped auto trial showed that using all available workers can be
unstable for large-vocab argmax because thread overhead, page behavior, and
memory bandwidth can erase the benefit. The adaptive cap keeps the default
closer to the best observed Llama 1B result while preserving manual override
for future experiments.

## Decision

success with limitation

Reason: decode speed improved on both tested artifacts and RAM stayed flat, but
the gain is too small to change the overall Llama 1B performance profile.

Paper value:

- use as positive CPU-only evidence
- use as limitation: naive all-core scaling is not automatically optimal

## Next Experiment

Move from scoped per-call worker spawning to persistent worker reuse or
parallelize a heavier transformer projection bucket. The current R13 result
shows that CPU parallelism can help, but LM-head-only parallelism is not enough
to reach the 30-40 tok/s target.
