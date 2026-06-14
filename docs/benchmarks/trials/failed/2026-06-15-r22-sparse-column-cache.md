# Trial: R22 Sparse Column Cache

Date: 2026-06-15
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

R20 showed that MLP projection streaming dominates Llama 3.2 1B decode. R21
showed that row-parallel sparse MLP is not enough because it still scans
row-major weight chunks. R22 tests an activation-column cache: extract selected
input columns on first use, then reuse contiguous cached columns for later
tokens.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Expected bottleneck: sparse MLP row-major weight traffic
- Bottleneck tag: sparse column cache
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Column cache gate: `RLLM_AIP_COLUMN_CACHE=1`
- Cache cap: `RLLM_AIP_COLUMN_CACHE_MAX_COLUMNS`, default `8192`
- AIP policy: `RLLM_AIP_POLICY=speed`

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  RLLM_AIP_COLUMN_CACHE=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\ngood morning\nexit\n' | \
  RLLM_AIP_COLUMN_CACHE=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 8

printf 'good morning\nexit\n' | \
  RLLM_AIP_COLUMN_CACHE=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=32 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 8

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=32 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 8
```

Runtime context:

- build profile: release
- default sparse path remains unchanged unless `RLLM_AIP_COLUMN_CACHE=1`
- cache stores f32 columns and is capped by resident column count
- high page-fault counts were observed during these runs, so compare variants
  run in the same session rather than treating absolute speed as stable

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | max top-k | column cache hits | column cache misses | resident cache | repeated ratio | max run | unique tokens | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | column cache top-k 128 | 16 | 13.97s | 0.15 | 0.14 | 128 | 649 | 8183 | 8183 cols / 198418432 bytes | 0.93 | 15 | 2/16 | 2303033344 | 1729629568 | 1050689536 |
| Llama-3.2-1B-Instruct | column cache top-k 128, turn 1 | 8 | 13.43s | 0.13 | 0.12 | 128 | 649 | 8183 | 8183 cols / 198418432 bytes | 0.86 | 7 | 2/8 | 2313928704 | 1649249616 | 1050689536 |
| Llama-3.2-1B-Instruct | column cache top-k 128, turn 2 | 8 | 12.08s | 0.12 | 0.12 | 128 | 0 | 0 | cache cap hit, row-major fallback | 0.86 | 7 | 2/8 | 2313928704 | 1649249616 | 1050689536 |
| Llama-3.2-1B-Instruct | column cache top-k 32 | 8 | 13.81s | 0.31 | 0.22 | 32 | 6153 | 4599 | 4599 cols / 97837056 bytes | 0.86 | 7 | 2/8 | 2370994176 | 1619790848 | 1050689536 |
| Llama-3.2-1B-Instruct | no column cache top-k 32 | 8 | 12.27s | 0.48 | 0.30 | 32 | 0 | 0 | none | 0.86 | 7 | 2/8 | 2452094976 | 1620692280 | 1050689536 |

## Analysis

The column-cache implementation is correct in unit tests and telemetry confirms
the path is active, but it does not improve runtime speed.

Top-k 128 fills the default cache cap almost immediately:

- `8183` resident columns
- about `198 MB` f32 column cache
- only `649` hits before the cap is reached

The second turn then falls back to the row-major sparse path because new
selected columns cannot be inserted under the cap. Increasing the cap would let
more columns stay resident, but it directly conflicts with the low-RAM goal and
would still pay a heavy first-use extraction cost.

Top-k 32 reduces the resident cache to about `98 MB`, but it is still slower
than the no-cache sparse path in the same session (`0.31 tok/s` vs
`0.48 tok/s`). This means a naive f32 column cache is not the right layout
strategy.

RLLM tracked peak transient memory stayed flat at `1050689536` bytes, but
process footprint increased when the resident column cache was populated.

## Decision

failed

Reason: activation-column f32 caching increases memory pressure and does not
improve decode speed for Llama 3.2 1B Instruct.

Paper value:
- useful negative result for resident f32 sparse-column caching
- useful evidence that selected activation dimensions change enough to exhaust
  a bounded cache quickly
- useful evidence that the next layout experiment must avoid large f32 resident
  sidecars

## Next Experiment

R23 should move from resident f32 columns to a compact on-disk or chunk-local
input-tile layout. The next candidate is to repack only MLP weights into small
input-feature tiles so AIP can read selected feature groups without scanning
full row-major rows and without materializing large f32 sidecars.
