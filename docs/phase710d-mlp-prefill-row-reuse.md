# Phase 7.10D RAMA Prefill MLP Row Reuse

Phase 7.10D follows the Phase 7.10C evidence: long-prompt prefill is dominated by MLP compute, not layer-param recall. This phase first splits the MLP timing bucket, then applies one RAMA-native hot-loop optimization that preserves exact tested logits and bounded active memory.

No external architecture or code was copied. The accepted optimization is a local tiled-linear accumulation change: reuse each decoded weight row across four prompt-token rows at a time.

## Scope

Goals:

- Split `prefill_mlp_ms` into:
  - `prefill_mlp_input_projection_ms`
  - `prefill_mlp_activation_ms`
  - `prefill_mlp_output_projection_ms`
- Identify the dominant MLP sub-phase.
- Apply only a measured safe optimization.
- Preserve row-major per-token accumulation order and tested logits.

Out of scope:

- sparse neuron prediction;
- hot/cold neuron routing;
- PowerInfer-style activation locality;
- changing file format or model artifact layout.

## MLP split benchmark

Command:

```bash
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --out-dir target/phase710d-mlp-split-baseline \
  --timeout-seconds 900
```

Baseline after split instrumentation:

| input tokens | real seconds | tok/sec | max RSS MiB | prefill ms | MLP ms | MLP input ms | GELU ms | MLP output ms | attention ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 7.56 | 2.1164 | 33.20 | 5688.43 | 3461.01 | 1492.28 | 44.70 | 1923.96 | 1946.08 |
| 1024 | 13.95 | 1.1470 | 44.92 | 12433.89 | 7170.17 | 3089.80 | 96.01 | 3984.22 | 4962.43 |

Interpretation:

```text
MLP output projection is the largest MLP sub-phase.
MLP input projection is second.
GELU is negligible (~1.3% of MLP).
```

## Rejected optimization attempts

Two measured attempts were rejected:

1. **Larger MLP projection tile window**
   - Intended to process more full rows per tile in `dense_4h_to_h`.
   - Result: ~1% slower on both 512 and 1024 prompts.
   - Decision: reverted.

2. **Single-row dot-product unroll**
   - Intended to reduce loop overhead in the dot product.
   - Result: much slower on the real benchmark (`1024+16` rose to 16.78s).
   - Decision: reverted.

These rejections are important: they confirm we are not accepting plausible-looking optimizations without benchmark evidence.

## Accepted optimization

The accepted change is inside `accumulate_weight_chunk`:

```text
Before:
  For each output row fragment, loop over every prompt-token row separately.
  Each token rereads the same decoded weight row fragment.

After:
  Process four prompt-token rows at a time.
  Load each decoded weight once and update four accumulators.
  Preserve each token's accumulation order over input features.
```

This is RAMA-native because it only changes the active compute scheduling over already-recalled tile data. It does not change logits semantics, storage format, recall policy, or memory ownership.

## Optimized benchmark

Command:

```bash
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --out-dir target/phase710d-four-batch-optimized \
  --timeout-seconds 900
```

Results:

| input tokens | real seconds | tok/sec | max RSS MiB | prefill ms | MLP ms | MLP input ms | GELU ms | MLP output ms | attention ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 5.25 | 3.0476 | 32.53 | 3288.53 | 1689.54 | 820.96 | 49.55 | 818.98 | 1289.11 |
| 1024 | 9.12 | 1.7544 | 44.67 | 7564.89 | 3502.78 | 1699.96 | 105.58 | 1697.12 | 3744.92 |

Delta vs split baseline:

| input tokens | real seconds | tok/sec | prefill ms | MLP ms | MLP input | MLP output | RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | -30.6% | +44.0% | -42.2% | -51.2% | -45.0% | -57.4% | -2.0% |
| 1024 | -34.6% | +53.0% | -39.2% | -51.1% | -45.0% | -57.4% | -0.6% |

The same accumulation optimization also helps attention because the QKV and attention-output projections use the same tiled-linear hot loop.

## Correctness check

Direct HF rerun is still blocked in this shell because `uv`/`torch` are unavailable. Instead, the optimized RLLM logits were compared against the prior saved default-32 RLLM logits that had already matched the HF-validated path.

512-token prompt comparison:

```text
logits len:      50304
max_abs_diff:   0.0
mean_abs_diff:  0.0
generated IDs:  [12092] vs [12092]
top1:           12092 vs 12092
top10 overlap:  10/10
```

## Next target

After Phase 7.10D, the dominant long-prompt prefill target shifts toward attention:

```text
1024-token + 16 after optimization:
attention: 3744.92 ms
MLP:       3502.78 ms
```

The next phase should be Phase 7.10E: split attention timing into QKV projection, rotary/KV append, attention score/context, and output projection before optimizing attention.
