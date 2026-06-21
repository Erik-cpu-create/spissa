# Phase 7.10C Deep Prefill Timing

Phase 7.10C adds low-overhead sub-phase timing inside RAMA long-prompt prefill. The goal is to stop guessing about the next optimization target after Phase 7.10B made 32-token prefill windows the measured default.

This remains an original RAMA/RLLM measurement path: bounded activation windows, deterministic layer recall, and explicit working-memory release. It does **not** introduce hot/cold neuron prediction, activation-locality routing, or PowerInfer-style scheduling.

## Implementation

`RamaGenerationTiming` now records aggregate prefill sub-phases:

- `prefill_embedding_ms`
- `prefill_layer_params_ms`
- `prefill_attention_norm_ms`
- `prefill_attention_ms`
- `prefill_attention_residual_ms`
- `prefill_mlp_norm_ms`
- `prefill_mlp_ms`
- `prefill_mlp_residual_ms`
- `prefill_timed_blocks`

The block-level instrumentation lives beside the existing streaming transformer block path and is only enabled by the RAMA timing path. Existing untimed block callers keep the previous API through a wrapper.

## Benchmark command

```bash
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --out-dir target/phase710c-deep-prefill-timing \
  --timeout-seconds 900
```

Artifact:

```text
models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.rllm
```

Runtime behavior:

```text
CLI default RAMA prefill chunk: 32 real input tokens
Integrity: verify-once
Memory budget: 100mb
Generated tokens: 16
```

## Results

| input tokens | real seconds | generated tok/sec | max RSS MiB | peak transient | prefill ms | decode ms | lm_head ms | timed blocks |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 512 | 7.16 | 2.2346 | 33.22 | 794.00 KiB | 5377.55 | 1306.87 | 961.82 | 96 |
| 1024 | 13.33 | 1.2003 | 44.92 | 794.00 KiB | 11888.26 | 1423.17 | 1027.59 | 192 |

The new deep buckets explain the prefill cost:

| input tokens | embedding | layer params | attention | MLP | attention share | MLP share |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 6.63 ms | 11.05 ms | 1818.26 ms | 3300.11 ms | 33.8% | 61.4% |
| 1024 | 10.24 ms | 19.61 ms | 4731.65 ms | 6863.69 ms | 39.8% | 57.7% |

Other measured sub-phases are effectively noise at this scale:

```text
attention norm/residual: <0.1% of prefill
MLP norm/residual:       <0.1% of prefill
embedding recall:        ~0.1% of prefill
layer-param recall:      ~0.2% of prefill
```

## Interpretation

Phase 7.10C resolves the next optimization target:

```text
Primary bottleneck:   RAMA prefill MLP compute
Secondary bottleneck: RAMA prefill attention compute
Not the bottleneck:   layer-param recall, embedding recall, norms, residual adds
```

That is important because it prevents a wrong next step. Optimizing layer-param recall would be architecturally neat but measured impact is tiny for this artifact. The next meaningful speed work should target the MLP prefill path while preserving RAMA's bounded active memory model and exact logits semantics.

## Next phase recommendation

Phase 7.10D should optimize RAMA prefill MLP without copying external sparse-neuron or predictor designs.

Safe RAMA-native candidates:

1. **MLP prefill row-span reuse / batching review**
   - inspect whether chunked prefill repeatedly pays avoidable per-token setup in `dense_h_to_4h` and `dense_4h_to_h` paths;
   - keep row-major accumulation order stable for tested logits.

2. **Low-RAM MLP tile scheduling**
   - preserve bounded f32 tile scratch;
   - evaluate larger tile windows only if RSS/transient budget stays controlled.

3. **Prefill-only MLP timing split**
   - if needed, split `prefill_mlp_ms` into MLP input projection vs GELU vs output projection before optimizing.

Avoid for now:

- optimizing layer-param recall first;
- adding predictor/sparsity systems;
- parallel row-span accumulation unless short-prompt decode/lm-head becomes the priority.

## Verification

Targeted verification run before this benchmark:

```bash
cargo fmt --all
python3 -m py_compile scripts/*.py
cargo test -p rllm-runtime streaming_transformer_block_timing_records_each_subphase_once -- --nocapture
cargo test -p rllm-runtime layer_decoded_gpt_neox_chunked_prefill_matches_full_prefill -- --nocapture
cargo test -p rllm-cli -- --nocapture
cargo build --release
```

Observed targeted results:

```text
streaming_transformer_block_timing_records_each_subphase_once ... ok
layer_decoded_gpt_neox_chunked_prefill_matches_full_prefill ... ok
rllm-cli tests: 9 passed
release build: finished
```

Full workspace verification is still required after documentation updates.
