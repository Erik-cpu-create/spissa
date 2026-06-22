# Phase 7.12C Streamed LM-Head Argmax / Logit Collection Gate

Phase 7.12C adds a narrow RAMA runtime slice for argmax generation:

- CLI generation without `--logits-out` now sets `collect_logits=false`.
- GPT-NeoX/Pythia RAMA generation keeps full per-step logits by default for API/tests/parity tooling.
- When `collect_logits=false` and sampling is `Argmax`, the lm-head path streams the best output row directly via `streaming_tile_linear_argmax_from_model` instead of materializing and storing full vocabulary logits.
- Top-p sampling and `--logits-out` still use the full logits path.

This is an original RLLM/RAMA low-memory runtime hygiene slice. It does **not** use hot/cold neurons, activation-locality prediction, GPU residency scheduling, sparse neuron routing, or any PowerInfer-style mechanism.

## Why this slice was tried

The Phase 7.12C baseline showed short prompts spending noticeable time in lm-head/final generation overhead, while long prompts remained prefill-bound:

| run | input tokens | baseline tok/s | baseline RSS | baseline lm_head ms |
|---|---:|---:|---:|---:|
| Pythia-70M low-ram | 1 | 10.390 | 19.98 MiB | 852.15 |
| Pythia-70M low-ram | 1024 | 2.862 | 44.42 MiB | 924.78 |
| Pythia-160M low-ram | 1 | 3.548 | 25.92 MiB | 1380.20 |
| Pythia-160M low-ram | 1024 | 0.716 | 98.20 MiB | 1632.94 |
| Pythia-160M speed window 128 | 1024 | 0.728 | 99.02 MiB | 1593.86 |

The hypothesis was that default CLI argmax does not need to retain full logits, so skipping full-logit materialization/storage could reduce memory and possibly improve short-prompt speed.

## Implementation

Touched runtime boundary only:

- `crates/rllm-runtime/src/streaming.rs`
  - Added `streaming_tile_linear_argmax_from_model`.
  - The helper is generic over streamed row-major linear weights and supports `batch=1`.
  - It preserves row-major accumulation order and handles rows split across chunks/tiles.
- `crates/rllm-runtime/src/gpt_neox.rs`
  - Added `GptNeoxRamaGenerationOptions::collect_logits`.
  - Default remains `collect_logits=true` for compatibility with existing runtime callers and logits parity tests.
  - `collect_logits=false` uses streamed lm-head argmax only for `StreamingSamplingConfig::Argmax`.
- `crates/rllm-cli/src/commands/run.rs`
  - CLI passes `collect_logits = logits_out.is_some()`.
  - `--logits-out` therefore keeps the old full-logits path.

No `.spsa` format changes, no codec changes, no model-specific branches.

## Correctness checks

Commands run:

```bash
cargo test -p rllm-runtime argmax
cargo fmt --check
cargo test -p rllm-runtime
cargo test -p rllm-cli
cargo build --release
```

Observed results:

- Focused argmax tests: 4 passed.
- Runtime crate tests: 75 passed.
- CLI crate tests: 11 passed.
- Release build completed successfully.

New guards:

- `streaming_tile_linear_argmax_matches_full_logits_across_split_rows`
  - Verifies streamed argmax matches full tiled-linear logits + `sample_argmax` across a chunk split inside a weight row.
- `layer_decoded_gpt_neox_can_skip_logit_collection_for_argmax_generation`
  - Verifies GPT-NeoX RAMA `collect_logits=false` produces the same token IDs as the full-logits path and returns empty `step_logits`.

## Measurement

Benchmark command shape:

```bash
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --artifact <pythia artifact> \
  --input-tokens 1,1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --memory-budget 100mb \
  --rama-integrity verify-once \
  --rama-timing-dir target/phase712c-argmax/<run>-timing \
  --out-dir target/phase712c-argmax/<run> \
  --timeout-seconds 900
```

For speed-policy rows the script used explicit windows because the older benchmark harness does not accept `--rama-prefill-policy`:

- Pythia-70M speed: `--rama-prefill-chunk-tokens 64`
- Pythia-160M speed: `--rama-prefill-chunk-tokens 128`

### New vs baseline

| run | input tokens | baseline tok/s | 7.12C tok/s | delta | baseline RSS | 7.12C RSS | baseline lm_head ms | 7.12C lm_head ms | sampling ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70M low-ram | 1 | 10.390 | 7.583 | -27.0% | 19.98 | 19.44 | 852.15 | 850.24 | 0.00 |
| 70M low-ram | 1024 | 2.862 | 2.909 | +1.6% | 44.42 | 44.44 | 924.78 | 898.29 | 0.00 |
| 70M speed64 | 1 | 10.127 | 10.000 | -1.3% | 19.98 | 19.97 | 894.29 | 899.64 | 0.00 |
| 70M speed64 | 1024 | 2.852 | 2.925 | +2.6% | 44.28 | 44.55 | 998.40 | 946.25 | 0.00 |
| 160M low-ram | 1 | 3.548 | 3.404 | -4.0% | 25.92 | 24.66 | 1380.20 | 1443.23 | 0.00 |
| 160M low-ram | 1024 | 0.716 | 0.727 | +1.6% | 98.20 | 97.50 | 1632.94 | 1530.08 | 0.00 |
| 160M speed128 | 1 | 3.292 | 3.279 | -0.4% | 25.84 | 24.53 | 1498.26 | 1501.54 | 0.00 |
| 160M speed128 | 1024 | 0.728 | 0.746 | +2.4% | 99.02 | 98.88 | 1593.86 | 1531.39 | 0.00 |

## Conclusion

Phase 7.12C is useful as runtime hygiene and modest memory/logit-storage reduction, but it is **not** the major token-speed lever.

What the measurement proves:

- Sampling time drops to effectively zero in timing JSON because argmax is fused into the streamed lm-head pass.
- Full-vocab logits are no longer retained for default CLI argmax generation.
- RSS is slightly lower on some short 160M rows (~1.3 MiB), but not enough to matter for the 10-20 tok/s target.
- lm-head dot-product work remains, so short-prompt speed does not materially improve and may move within noise/regression range.
- Long-prompt rows improve slightly (+1.6% to +2.6%), but those rows are still dominated by prefill MLP/QKV/attention, not sampling/logit retention.

Next speed-relevant work should return to measured generic projection/prefill kernels instead of spending more time on logit-storage elimination.

## Current target status

- Pythia-70M short prompt can still hit roughly 10 tok/s in the speed64 row, but the low-ram one-row result is noisy and should not be overclaimed.
- Pythia-160M remains far from 10-20 tok/s: ~3.3-3.4 tok/s short prompt and ~0.73-0.75 tok/s for 1024-token prompt + 16 generated.
- The honest bottleneck remains dense projection work in RAMA prefill/decode, especially MLP and QKV/attention paths.
