# Phase 7.10A Row-Span Linear Accumulation

Phase 7.10A optimizes the core RAMA tiled-linear accumulation loop without changing the `.spsa` format, model artifact, codec policy, logits semantics, or memory budget contract.

This is an original RLLM/RAMA compute-path optimization. It does **not** use hot/cold neurons, activation-locality prediction, GPU residency scheduling, sparse neuron routing, or any PowerInfer-style mechanism.

## Change

Before Phase 7.10A, `accumulate_weight_chunk` handled every decoded weight element independently:

```text
for each weight element:
  global_idx -> out_feature = global_idx / in_features
  global_idx -> in_feature  = global_idx % in_features
  for each batch:
    output[out_feature] += input[in_feature] * weight
```

That was simple and correct, but expensive in the hot path because it performed division/modulo and output indexing for every weight element.

Phase 7.10A keeps the same row-major arithmetic order but accumulates contiguous row spans:

```text
for each contiguous row span inside the decoded tile:
  compute out_feature/in_feature once
  dot(input row slice, weight row slice)
  accumulate into output row once
```

The per-output accumulation order remains increasing `in_feature`, so tested logits stay unchanged.

## Scope

Touched runtime code:

```text
crates/rllm-runtime/src/streaming.rs
```

Specifically:

```text
accumulate_weight_chunk(...)
```

No CLI flag, no file format change, no model repack, no new dependency.

## Short-prompt benchmark

Artifact/runtime:

```text
artifact: models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa
runtime integrity: verify-once
input tokens: 1
new tokens: 16
ctx: 2048
memory budget: 100mb
prefill chunk: full
```

Before this loop change, same local row:

```text
elapsed:   4.39s
tok/s:     3.647
RSS:       20.61 MiB
prefill:   549.53 ms
decode:    3121.57 ms
lm_head:   2094.48 ms
```

After Phase 7.10A:

```text
elapsed:   2.25s
tok/s:     7.101
RSS:       20.66 MiB
prefill:   417.80 ms
decode:    1258.83 ms
lm_head:   938.89 ms
```

Delta:

```text
tok/s speedup:      1.95x
elapsed reduction:  48.7%
RSS change:         effectively unchanged
```

## Long-prompt validation

The same accumulation loop is used by attention/MLP/lm-head tile-linear calls, so long prompts also improved substantially.

### 512-token prompt, chunked prefill

Phase 7.9E best row:

```text
input tokens: 512
new tokens:   16
prefill chunk: 64
elapsed:      35.20s
tok/s:        0.455
RSS:          34.05 MiB
prefill:      31.28s
```

After Phase 7.10A:

```text
elapsed:      7.08s
tok/s:        2.259
RSS:          32.98 MiB
prefill:      5.38s
decode:       1.67s
lm_head:      1.22s
```

Delta:

```text
tok/s speedup:      4.96x
elapsed reduction:  79.9%
```

### 1024-token prompt, chunked prefill

Phase 7.9E best row:

```text
input tokens: 1024
new tokens:   16
prefill chunk: 64
elapsed:      63.84s
tok/s:        0.251
RSS:          44.98 MiB
prefill:      59.95s
```

After Phase 7.10A:

```text
elapsed:      12.76s
tok/s:        1.254
RSS:          44.80 MiB
prefill:      11.28s
decode:       1.46s
lm_head:      1.05s
```

Delta:

```text
tok/s speedup:      5.00x
elapsed reduction:  80.0%
```

## Correctness

Targeted tiled-linear tests passed after the change:

```bash
cargo test -p rllm-runtime streaming_tile_linear -- --nocapture
```

Result:

```text
2 passed; 0 failed
```

Real Pythia-70M 512-token HF/PyTorch logits comparison was run with chunked prefill after this optimization:

```bash
uv run --with torch --with transformers --with safetensors \
  scripts/phase77_compare_logits.py \
  --rllm-artifact models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa \
  --out-dir target/phase710a-logits-512-c64 \
  --token-ids <512 deterministic ids> \
  --ctx 2048 \
  --memory-budget 100mb \
  --rama-integrity verify-once \
  --rama-prefill-chunk-tokens 64
```

Result:

```text
top1_match: true
top5_overlap: 5/5
top10_overlap: 10/10
max_abs_diff: 0.02746582
mean_abs_diff: 0.01679823
RLLM top1: 12092
HF top1: 12092
```

The metrics match the previous 512-token parity check, so Phase 7.10A preserves tested logits.

## Interpretation

Phase 7.10A is a CPU hot-loop improvement, not a new architecture trick. The main insight is that RAMA's chunk/tile recall path was already correct, but the per-element accumulation loop left easy row-major locality on the table.

The updated status is materially better:

```text
short prompt + 16 generated: 7.10 tok/s at ~20.66 MiB RSS
512 prompt + 16 generated:   2.26 tok/s at ~32.98 MiB RSS
1024 prompt + 16 generated:  1.25 tok/s at ~44.80 MiB RSS
```

Follow-up Phase 7.10B runs the broader post-rowspan matrix and tunes the RAMA prefill window to 32 real input tokens by default. The best swept rows are 9.756 tok/s for short prompt + 16 generated, 2.3495 tok/s / 32.77 MiB RSS for 512-token + 16, and 1.1653 tok/s / 44.91 MiB RSS for 1024-token + 16. See [`phase710b-rama-prefill-homeostasis.md`](phase710b-rama-prefill-homeostasis.md).

## Next bottleneck

Measured `lm_head_ms` improved, and Phase 7.10B shows long prompts remain prefill-dominant. The next target should still be measurement-led:

1. Add deeper timing inside attention/MLP/layer-param recall for 512/1024-token prefill.
2. Optimize the measured dominant prefill sub-phase.
3. Consider exact low-RAM parallel row-span accumulation only if short-prompt decode/lm-head becomes the priority.

Keep the RAMA boundary: chunk/tile/context-memory optimization, not neuron activation prediction.
