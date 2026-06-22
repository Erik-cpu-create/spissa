# Phase 7.9B RAMA Embedding Row Recall

Phase 7.9B reduces repeated codec decode/recall cost for the input embedding table by changing `streaming_embedding_lookup_from_model` from a full embedding chunk scan into selective row recall.

This is a RAMA-native optimization:

```text
Before: recall every embedding chunk, then gather matching rows.
After: compute requested token row byte ranges, recall only overlapping chunks/ranges, copy only requested rows.
```

It stays distinct from PowerInfer-style activation locality:

```text
Unit of recall: tensor row/chunk byte ranges in `.spsa`
Not used: hot/cold neurons, activation prediction, GPU/CPU neuron partitioning
```

## Implementation

Touched runtime boundary:

```text
crates/rllm-runtime/src/tiny.rs
```

`streaming_embedding_lookup_from_model` now:

1. Validates embedding tensor shape and chunk coverage.
2. Computes row byte spans for each requested token id.
3. Groups row spans by overlapping chunk.
4. Calls `LazyRllmModel::with_decoded_chunk_range` only for touched chunks/ranges.
5. Decodes only the requested range into f32 scratch and copies it into the output activation.

For native range codecs such as `rtc-raw-v1`, only the requested decoded byte range is reserved. For non-native range codecs such as current Huffman, `with_decoded_chunk_range` intentionally falls back to full decode of the touched chunk; this still avoids all untouched embedding chunks.

## Unit guard

Added targeted test:

```text
tiny::tests::streaming_embedding_lookup_recalls_only_touched_row_chunks
```

The tiny embedding fixture has two row-aligned chunks. The test requests token `[2]` and uses a 24-byte budget. The old full-scan embedding path would exceed that budget when it tried to decode untouched chunk 0. The new selective row path succeeds and releases budget back to zero.

## Real Pythia one-token smoke

Command:

```bash
/usr/bin/time -l target/release/rllm run models/pythia-70m-phase78d-tileblocks.spsa \
  --token-ids 12092 \
  --max-new-tokens 1 \
  --ctx 128 \
  --memory-budget 100mb
```

Result:

```text
Generated token IDs: [13]
Full text: Hello,
Peak transient budget: 291.68 KiB
2.69 real / 2.64 user / 0.02 sys
maximum resident set size: 19,054,592 bytes
```

Previous comparable Phase 7.8E no-trace one-token row:

```text
4.31 real, 18.39 MiB RSS, 386.00 KiB tracked transient
```

## Trace comparison

Commands compared:

```bash
# Phase 7.9A baseline trace
/usr/bin/time -l target/release/rllm run models/pythia-70m-phase78d-tileblocks.spsa \
  --token-ids 12092 --max-new-tokens 1 --ctx 128 --memory-budget 100mb \
  --rama-trace target/phase79a/rama_trace_timed.json

# Phase 7.9B trace
/usr/bin/time -l target/release/rllm run models/pythia-70m-phase78d-tileblocks.spsa \
  --token-ids 12092 --max-new-tokens 1 --ctx 128 --memory-budget 100mb \
  --rama-trace target/phase79b/rama_trace_timed.json
```

Trace-mode wall/RSS:

| Phase | real s | user s | max RSS bytes | peak transient |
|---|---:|---:|---:|---:|
| 7.9A full embedding scan | 4.48 | 4.38 | 41,631,744 | 386.00 KiB |
| 7.9B row recall | 3.22 | 2.66 | 30,785,536 | 291.68 KiB |

Recorded events and phase totals:

| Phase | 7.9A events | 7.9B events | 7.9A ms | 7.9B ms | Saved |
|---|---:|---:|---:|---:|---:|
| `chunk_decode` | 1,074 | 682 | 3685.829 | 2155.544 | 1530.285 ms |
| `chunk_original_checksum` | 1,074 | 682 | 260.886 | 166.417 | 94.469 ms |
| `chunk_compute_closure` | 1,074 | 682 | 250.435 | 201.001 | 49.435 ms |
| `chunk_compressed_checksum` | 1,074 | 682 | 235.662 | 145.803 | 89.859 ms |
| `chunk_read` | 1,074 | 682 | 14.389 | 22.133 | -7.744 ms |

Embedding-specific tensor delta:

| Tensor | 7.9A events | 7.9B events | 7.9A ms | 7.9B ms | Saved |
|---|---:|---:|---:|---:|---:|
| `gpt_neox.embed_in.weight` | 1,965 | 5 | 1653.052 | 6.629 | 1646.423 ms |
| `embed_out.weight` | 1,965 | 1,965 | 1523.508 | 1468.309 | 55.199 ms |

Interpretation:

```text
Input embedding recall is effectively fixed for one-token prompts: it no longer scans/decode all embedding chunks. The dominant remaining recurrent hotspot is `embed_out.weight` / lm_head, which still must compute full-vocab logits for exact argmax/top-p correctness.
```

## Full 12-row benchmark matrix

Command:

```bash
python3 scripts/phase76_release_rss_benchmark.py \
  --skip-build \
  --artifact models/pythia-70m-phase78d-tileblocks.spsa \
  --out-dir target/phase79b-embedding-row-bench \
  --tokens 1,4,8,16 \
  --ctx 128,512,1024 \
  --memory-budget 100mb \
  --timeout-seconds 900
```

Result:

```text
12/12 succeeded
seconds/token range: 2.68–3.23
average seconds/token: 2.93
max RSS range: 18.95–22.28 MiB
tracked transient peak: 291.68 KiB
```

Compared with Phase 7.8E tile-block artifact baseline:

| Metric | Phase 7.8E | Phase 7.9B | Delta |
|---|---:|---:|---:|
| avg seconds/token | 5.07 | 2.93 | 42.34% lower |
| speedup range | — | 1.61×–1.80× | — |
| avg speedup | — | 1.73× | — |
| max RSS | 22.64 MiB | 22.28 MiB | -0.36 MiB |
| tracked transient peak | 386.00 KiB | 291.68 KiB | -94.32 KiB |

Per-row comparison:

| ctx | tokens | 7.8E s/token | 7.9B s/token | speedup | 7.8E RSS | 7.9B RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 4.31 | 2.68 | 1.61× | 18.39 | 18.95 |
| 128 | 4 | 4.56 | 2.70 | 1.69× | 20.23 | 19.66 |
| 128 | 8 | 4.99 | 2.80 | 1.78× | 20.72 | 20.53 |
| 128 | 16 | 5.20 | 2.89 | 1.80× | 21.62 | 21.20 |
| 512 | 1 | 5.25 | 2.93 | 1.79× | 19.62 | 19.50 |
| 512 | 4 | 5.22 | 2.91 | 1.79× | 19.94 | 19.75 |
| 512 | 8 | 5.18 | 2.92 | 1.77× | 19.80 | 20.55 |
| 512 | 16 | 5.23 | 2.94 | 1.78× | 22.64 | 22.28 |
| 1024 | 1 | 5.24 | 2.97 | 1.76× | 19.77 | 19.70 |
| 1024 | 4 | 5.23 | 3.03 | 1.73× | 18.77 | 19.58 |
| 1024 | 8 | 5.24 | 3.11 | 1.68× | 20.02 | 19.69 |
| 1024 | 16 | 5.24 | 3.23 | 1.62× | 20.45 | 21.09 |

## HF/PyTorch parity

Command:

```bash
uv run --with torch --with transformers --with safetensors \
  scripts/phase77_compare_logits.py \
  --rllm-artifact models/pythia-70m-phase78d-tileblocks.spsa \
  --out-dir target/phase79b-logits \
  --token-ids 12092,13 \
  --ctx 128 \
  --memory-budget 100mb
```

Result:

```text
top1_match=True
max_abs_diff=0.00769043
mean_abs_diff=0.00231486
rllm_top1=309
hf_top1=309
```

## Follow-up RAMA-native target

Phase 7.9C followed this Phase 7.9B result with a raw/tile-block low-ram-fast profile and verify-once integrity mode. After that slice, Huffman decode is no longer dominant; the remaining hotspot is compute, especially `embed_out.weight` / lm_head:

```text
Phase 7.9B: embed_out.weight: 1,965 events / ~1468 ms in one-token trace
Phase 7.9C verify-once: embed_out.weight: 19,650 events / ~2099 ms in 16-token trace
```

Next candidates:

1. Keep exact full-vocab logits baseline unchanged.
2. Split `chunk_compute_closure` into conversion vs matmul timing if needed.
3. Parallelize/tile `embed_out.weight` / lm-head compute under the same low-RAM envelope.
4. Avoid PowerInfer-style hot/cold neuron prediction; keep the unit of optimization as verified `.spsa` tensor chunks/ranges.

## Status

[PRODUCTION-READY]

- Selective embedding row recall is covered by unit tests and real Pythia parity.
- It preserves deterministic output and HF top-1/top-10 parity on the tested fixed-token prompt.
- It improves measured 12-row benchmark speed without increasing max RSS.

[EXPERIMENTAL]

- Range-native benefits currently apply only to codecs that support native range decode (`rtc-raw-v1`). Current Huffman still full-decodes touched chunks.

[NOT DOING]

- No hot/cold neuron predictor.
- No activation locality model.
- No GPU/CPU neuron partitioning.
