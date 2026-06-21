# Phase 7.10E RAMA Attention Row-Slice Timing and Optimization

Phase 7.10E follows the Phase 7.10D evidence: after MLP prefill row reuse, long-prompt attention became comparable to the optimized MLP bucket. This phase splits attention timing into measured sub-phases, rejects a measured regression, and accepts a RAMA-native score/context optimization that preserves bounded active memory and tested logits semantics.

## Scope

Artifact:

```text
models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.rllm
```

Benchmark shape:

```text
input_tokens = 512,1024
max_new_tokens = 16
ctx = 2048
rama_integrity = verify-once
rama_prefill_chunk_tokens = default 32
memory_budget = 100mb
```

Command shape:

```bash
cargo build --release
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --memory-budget 100mb
```

## Added attention timing split

`--rama-timing` now reports these prefill attention sub-buckets:

```text
prefill_attention_qkv_projection_ms
prefill_attention_qkv_split_ms
prefill_attention_rotary_ms
prefill_attention_score_context_ms
prefill_attention_output_projection_ms
prefill_attention_kv_append_ms
```

The existing broader buckets remain:

```text
prefill_attention_norm_ms
prefill_attention_ms
prefill_attention_residual_ms
prefill_mlp_*_ms
prefill_timed_blocks
```

## Split-timing baseline

Output directory:

```text
target/phase710e-attention-split-baseline
```

| input tokens | real seconds | gen tok/sec | max RSS MiB | prefill ms | attention ms | MLP ms |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4.95 | 3.2323 | 32.64 | 3154.67 | 1235.03 | 1619.31 |
| 1024 | 8.63 | 1.8540 | 44.78 | 7159.02 | 3565.79 | 3297.72 |

Attention split:

| input tokens | QKV projection | score/context | output projection | rotary | KV append |
|---:|---:|---:|---:|---:|---:|
| 512 | 578.28 ms / 46.8% attention | 448.48 ms / 36.3% | 204.36 ms / 16.5% | 1.78 ms / 0.1% | 1.29 ms / 0.1% |
| 1024 | 1169.17 ms / 32.8% attention | 1982.70 ms / 55.6% | 405.69 ms / 11.4% | 3.64 ms / 0.1% | 2.63 ms / 0.1% |

Interpretation:

- At 512 tokens, QKV projection is the largest attention sub-phase.
- At 1024 tokens, score/context is the clear dominant attention sub-phase.
- Rotary, KV append, QKV split, norms, and residuals are not meaningful bottlenecks.

## Rejected candidate: in-place softmax

A first candidate replaced the score/context path's per-query `softmax_rows` allocation with an in-place softmax over the reusable score buffer. It preserved correctness tests but regressed measured throughput:

Output directory:

```text
target/phase710e-attention-softmax-optimized
```

| input tokens | real seconds delta | tok/sec delta | prefill delta | attention delta | score/context delta |
|---:|---:|---:|---:|---:|---:|
| 512 | +2.4% slower | -2.4% | +1.3% | +2.9% | +3.5% |
| 1024 | +3.4% slower | -3.3% | +2.8% | +2.4% | +2.3% |

Decision: rejected and reverted. Allocation removal alone did not improve this Rust release path on the measured local Pythia artifact.

## Accepted optimization: K/V row-slice score/context path

The accepted optimization changes `scaled_dot_product_attention_with_cache` to take contiguous K/V row slices once per key position instead of calling a scalar `kv_value()` helper per element.

Before:

```text
dot loop:   per key × per dim calls kv_value(cache/current branch + index)
value loop: per dim × per key calls kv_value(cache/current branch + index)
```

After:

```text
dot loop:   per key gets one key row slice, then dot over dims
value loop: per key gets one value row slice, then updates all dims
```

This keeps the same mathematical path and preserves per-output-dimension accumulation order over key positions:

```text
for each output dim, accumulation still follows key_pos = 0..key_len
```

It is not a sparse predictor, not a hot/cold neuron split, and not a copied external architecture. It is a local row-slice/data-access optimization inside RAMA's existing attention score/context primitive.

Output directory:

```text
target/phase710e-attention-row-slice-optimized
```

## Optimized benchmark result

| input tokens | real seconds | gen tok/sec | max RSS MiB | prefill ms | attention ms | score/context ms | MLP ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4.56 | 3.5088 | 32.88 | 2790.08 | 893.84 | 129.65 | 1623.38 |
| 1024 | 7.12 | 2.2472 | 44.97 | 5749.73 | 2147.87 | 548.02 | 3309.76 |

Delta vs split baseline:

| input tokens | real seconds | tok/sec | prefill | attention | score/context |
|---:|---:|---:|---:|---:|---:|
| 512 | -7.9% | +8.6% | -11.6% | -27.6% | -71.1% |
| 1024 | -17.5% | +21.2% | -19.7% | -39.8% | -72.4% |

RSS stayed effectively bounded and comparable:

```text
512:  32.64 -> 32.88 MiB
1024: 44.78 -> 44.97 MiB
```

Tracked transient budget stayed at:

```text
794 KiB
```

## Post-optimization bottleneck

After row-slice optimization, the 1024-token + 16 row shows:

```text
prefill_mlp_ms                         3309.76 ms
prefill_attention_ms                   2147.87 ms
prefill_attention_qkv_projection_ms    1188.90 ms
prefill_attention_score_context_ms      548.02 ms
prefill_attention_output_projection_ms  402.73 ms
lm_head_ms                             1003.67 ms
decode_ms                              1353.18 ms
```

So the next measured long-prompt prefill target is no longer score/context. Remaining meaningful targets are:

1. MLP projections again (largest prefill component)
2. QKV projection inside attention
3. Decode/lm-head path for short-prompt throughput

## Verification

Targeted checks run before the accepted benchmark:

```bash
cargo fmt --all
cargo test -p rllm-runtime cached_attention_for_next_token_matches_full_causal_attention_last_row -- --nocapture
cargo test -p rllm-runtime streaming_attention_with_rotary_and_kv_cache_matches_full_decode_last_token -- --nocapture
cargo test -p rllm-runtime streaming_transformer_block_timing_records_each_subphase_once -- --nocapture
cargo test -p rllm-runtime layer_decoded_gpt_neox_chunked_prefill_matches_full_prefill -- --nocapture
```

All targeted checks passed.

Full-suite and logits smoke are tracked in the final Phase 7.10E verification step.

## Recommendation

Phase 7.10E completes the immediate RAMA long-prompt attention optimization loop. The next best step is scale validation, not another blind micro-optimization:

```text
Phase 7.11 — Pythia-160M scale validation
```

Suggested acceptance:

- pack local/downloaded Pythia-160M into raw/tile-block `.rllm`
- verify lossless
- run short prompt and fixed token-ID generation
- run 128/512/1024-token timing matrix if runtime is acceptable
- compare RSS/tok/sec against the Pythia-70M Phase 7.10E baseline
- only after Pythia-160M, start LLaMA-family architecture expansion
