# Phase 7.12B Generic Eight-Row Projection Reuse

Phase 7.12B follows the Phase 7.12A policy work. The goal is to optimize the remaining measured Pythia-160M MLP/QKV projection bottlenecks without adding model-specific branches, changing the `.spsa` format, or using sparse/hot-cold activation routing.

## Scope

Accepted optimization:

```text
accumulate_weight_chunk: 4 prompt-token rows per decoded weight row -> 8 prompt-token rows per decoded weight row, with the existing 4-row and scalar tail fallbacks preserved.
```

This applies generically to every caller of the shared tiled-linear primitive:

```text
MLP input projection
MLP output projection
attention QKV projection
attention output projection
lm_head / final projections where batch size allows it
```

Out of scope:

```text
model-name-specific Pythia-160M branches
format changes
new dependencies
PowerInfer-style hot/cold neuron or activation-locality routing
parallel execution
SIMD intrinsics
```

## Why this target

Phase 7.11B showed Pythia-160M still spends most long-prompt prefill time in dense projections after the prefill policy is tuned. A current Phase 7.12A speed-policy baseline on the local Pythia-160M raw/tile-block artifact confirms the bottleneck:

```text
artifact: models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.spsa
input tokens: 512
max new tokens: 16
ctx: 2048
policy: --rama-prefill-policy speed
prefill window: 128
memory budget: 100mb
integrity: verify-once
```

Baseline timing:

| metric | ms |
|---|---:|
| prefill | 9174.13 |
| MLP total | 5601.07 |
| MLP input projection | 2773.15 |
| MLP output projection | 2784.70 |
| attention total | 3202.66 |
| attention QKV projection | 2071.98 |
| attention output projection | 700.68 |
| attention score/context | 420.98 |
| decode | 3512.38 |
| lm_head | 1405.39 |

Interpretation: MLP input/output projections and QKV projection dominate the target row, while score/context is already much smaller after Phase 7.10E.

## Implementation

Before:

```text
for each decoded weight row fragment:
    process prompt-token rows in groups of 4
    process remaining rows one at a time
```

After:

```text
for each decoded weight row fragment:
    process prompt-token rows in groups of 8
    process remaining rows in groups of 4
    process remaining rows one at a time
```

The accumulation order for each individual token/output feature remains the same over input features. The change only reuses a decoded weight scalar across more independent prompt-token rows before moving to the next weight scalar.

## Correctness checks

Unit/regression coverage:

```text
streaming_linear_matches_full_decode_with_eight_and_four_batch_fast_paths_and_tail
streaming_tile_linear_matches_full_decode_with_smaller_scratch_budget
streaming_attention_with_rotary_and_kv_cache_matches_full_decode_last_token
layer_decoded_gpt_neox_chunked_prefill_matches_full_prefill
```

The renamed linear test uses batch=13 to cover:

```text
8-row fast path + 4-row fallback + 1-row tail
```

## Measured result

Same local Pythia-160M 512-token speed-policy command before/after:

| metric | before | after | delta |
|---|---:|---:|---:|
| wall time | 13.21s | 12.34s | -6.6% |
| prefill | 9174.13 ms | 8268.24 ms | -9.9% |
| MLP total | 5601.07 ms | 4939.77 ms | -11.8% |
| MLP input projection | 2773.15 ms | 2445.28 ms | -11.8% |
| MLP output projection | 2784.70 ms | 2447.01 ms | -12.1% |
| attention total | 3202.66 ms | 2939.40 ms | -8.2% |
| attention QKV projection | 2071.98 ms | 1845.34 ms | -10.9% |
| attention output projection | 700.68 ms | 629.20 ms | -10.2% |
| max RSS | 63.33 MiB | 63.45 MiB | effectively flat |
| tracked transient peak | 3.79 MiB | 3.79 MiB | unchanged |

1024-token speed-policy confirmation after the optimization:

| metric | value |
|---|---:|
| wall time | 20.13s |
| generated tok/s | 0.795 tok/s |
| max RSS | 98.92 MiB |
| prefill | 16372.47 ms |
| MLP total | 9456.13 ms |
| MLP input projection | 4700.70 ms |
| MLP output projection | 4659.26 ms |
| attention total | 6507.16 ms |
| attention QKV projection | 3520.09 ms |
| attention score/context | 1768.90 ms |
| tracked transient peak | 3.79 MiB |

For context, the Phase 7.11B documented 1024-token speed-biased chunk=128 row was 26.65s / 0.600 tok/s / ~100.06 MiB RSS. The exact current before/after run was only captured for 512 tokens, so the 1024 number should be read as a confirmation row rather than a strict same-session delta.

## Remaining bottleneck

After eight-row projection reuse, the Pythia-160M 1024-token row still shows:

```text
MLP projections remain the largest prefill bucket.
QKV projection remains the largest attention projection bucket.
score/context is no longer the dominant attention problem, but grows with context length.
lm_head/decode remain relevant for short prompts.
```

Recommended next choices:

```text
Phase 7.12C: another generic dense-projection optimization, only if a new measurement identifies a safe candidate.
Phase 8: start LLaMA-family adapter work if architecture breadth is now higher priority than more GPT-NeoX projection tuning.
```
