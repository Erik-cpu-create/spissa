# Phase 7.8E Tile-Block Pythia RSS Benchmark

Phase 7.8E preliminarily benchmarks a real local Pythia-70M artifact packed with chunk-aligned tile blocks:

```bash
cargo run --quiet -- pack models/pythia-70m/model.safetensors \
  --out models/pythia-70m-phase78d-tileblocks.spsa \
  --chunk-size 16mb \
  --tile-block-elements 65536 \
  --config models/pythia-70m/config.json \
  --tokenizer models/pythia-70m/tokenizer.json
```

The runtime path is still the existing chunk-verified tiled RAMA path. The improvement comes from pack-time chunk alignment: each tile/block is stored as an independently verified `.spsa` chunk, so the existing chunk decode path no longer materializes 16 MiB chunks for every tiled projection.

## Artifact

```text
Artifact: models/pythia-70m-phase78d-tileblocks.spsa
Original size: 166,019,180 bytes
Compressed size: 127,330,065 bytes
Compression ratio: 76.7%
Chunks: 1,520
Tile-block elements: 65,536
Config/tokenizer metadata: persisted
```

Trade-off: this artifact has many more chunks than the earlier 16 MiB-chunk artifact and a larger compressed payload. The payoff is much lower runtime RSS and transient decode scratch.

## Release RSS matrix

Command:

```bash
python3 scripts/phase76_release_rss_benchmark.py \
  --skip-build \
  --artifact models/pythia-70m-phase78d-tileblocks.spsa \
  --out-dir target/phase78e-tileblock-bench \
  --tokens 1,4,8,16 \
  --ctx 128,512,1024 \
  --memory-budget 100mb \
  --timeout-seconds 900
```

Results:

| ctx | max new tokens | exit | real s | s/token | max RSS MiB | peak footprint MiB | peak transient | generated text |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 1 | 0 | 4.31 | 4.31 | 18.39 | 16.98 | 386.00 KiB | `,` |
| 128 | 4 | 0 | 18.24 | 4.56 | 20.23 | 18.83 | 386.00 KiB | `, I'm trying` |
| 128 | 8 | 0 | 39.90 | 4.99 | 20.72 | 19.31 | 386.00 KiB | `, I'm trying to get the name` |
| 128 | 16 | 0 | 83.17 | 5.20 | 21.62 | 19.39 | 386.00 KiB | `, I'm trying to get the name of the phone number in the phone number` |
| 512 | 1 | 0 | 5.25 | 5.25 | 19.62 | 18.22 | 386.00 KiB | `,` |
| 512 | 4 | 0 | 20.87 | 5.22 | 19.94 | 18.53 | 386.00 KiB | `, I'm trying` |
| 512 | 8 | 0 | 41.46 | 5.18 | 19.80 | 18.39 | 386.00 KiB | `, I'm trying to get the name` |
| 512 | 16 | 0 | 83.70 | 5.23 | 22.64 | 21.23 | 386.00 KiB | `, I'm trying to get the name of the phone number in the phone number` |
| 1024 | 1 | 0 | 5.24 | 5.24 | 19.77 | 18.36 | 386.00 KiB | `,` |
| 1024 | 4 | 0 | 20.91 | 5.23 | 18.77 | 17.38 | 386.00 KiB | `, I'm trying` |
| 1024 | 8 | 0 | 41.92 | 5.24 | 20.02 | 18.61 | 386.00 KiB | `, I'm trying to get the name` |
| 1024 | 16 | 0 | 83.77 | 5.24 | 20.45 | 19.05 | 386.00 KiB | `, I'm trying to get the name of the phone number in the phone number` |

Summary:

```text
Rows: 12/12 succeeded
RSS range: 18.39–22.64 MiB
Peak RSS run: ctx=512, max_new_tokens=16, 22.64 MiB
Throughput: ~4.31–5.25 seconds/token in release
Tracked transient peak: 386.00 KiB across all runs
```

## Comparison to the Phase 7.6/7.7 16 MiB-chunk artifact

```text
16 MiB chunk artifact RSS range:      88.62–94.62 MiB
Tile-block artifact RSS range:        18.39–22.64 MiB
Max-RSS reduction vs 16 MiB artifact: 76.07%
Tracked transient reduction:          48.00 MiB → 386.00 KiB (~99.21%)
```

Against the full-decode baseline recorded in Phase 7.6:

```text
Full-decode baseline:     364.66 MiB
Tile-block max RSS:        22.64 MiB
Reduction vs full decode:  93.79%
Full-decode/RSS:           16.11x
```

Speed did not materially improve; it stayed in the same ~4–5s/token band. This phase is a memory architecture win, not a token-throughput optimization.

## HF/PyTorch logits parity

Command:

```bash
uv run --with torch --with transformers --with safetensors \
  scripts/phase77_compare_logits.py \
  --rllm-artifact models/pythia-70m-phase78d-tileblocks.spsa \
  --out-dir target/phase78e-tileblock-logits \
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

So chunk-aligned tile-block packing preserves the Phase 7.7 HF/PyTorch fixed-token logits parity for the tested prompt.

## Caveats

- This is one local Pythia-70M artifact on macOS with `/usr/bin/time -l`; it is a real artifact benchmark, not a cross-platform guarantee.
- The artifact has more chunks and a larger compressed payload than coarse chunking.
- The runtime still uses chunk-level verification; true intra-chunk partial compressed reads for non-identity codecs remain future work.
- Tokenizer/normalizer fidelity beyond the current runtime-ready vocabulary metadata remains out of scope for this measurement.
