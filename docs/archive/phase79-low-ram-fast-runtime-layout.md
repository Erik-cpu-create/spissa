# Phase 7.9C RAMA Low-RAM-Fast Runtime Layout

Phase 7.9C tests the user's target direction explicitly:

```text
low RAM and still fast
```

The measured answer after Phase 7.9B was clear: the ultra-low-RAM Huffman/tile-block path was memory-efficient, but repeated token-loop Huffman decode was too slow. Phase 7.9C adds a compute-ready raw/tile-block path and an opt-in verify-once integrity mode so the runtime can avoid repeated codec decode and repeated SHA work while keeping the active working set bounded.

This is still RAMA-native:

```text
compressed or raw .spsa storage = long-term memory
raw tile-block layout = consolidated fast pathway
per-chunk/tile recall = bounded active working memory
verify-once = process-local trust after first verified recall
```

It is not a PowerInfer-style hot/cold neuron predictor. The optimization unit remains deterministic `.spsa` tensor chunks/tiles, not predicted active neurons.

## What changed

### Pack-time codec policy

`rllm pack` now accepts an explicit codec policy:

```bash
rllm pack <input.safetensors> \
  --out <output.spsa> \
  --codec auto|raw|rle|huff \
  --tile-block-elements <n>
```

Policy behavior:

| Policy | Meaning |
|---|---|
| `auto` | Existing behavior: try raw/RLE/Huffman and pick smallest verified chunk |
| `raw` | Force `rtc-raw-v1`; larger artifact, cheap decode/range path |
| `rle` | Force `rtc-rle-v1` |
| `huff` | Force `rtc-huff-v1` |

The default remains `auto`, so existing compressed artifacts are unchanged unless the user opts in.

### Runtime integrity policy

`rllm run` now accepts:

```bash
--rama-integrity strict|verify-once
```

| Mode | Default? | Behavior |
|---|---:|---|
| `strict` | yes | Verify compressed and decoded chunk SHA-256 on every recall |
| `verify-once` | no | Verify each chunk once per process, then trust the already-verified chunk on subsequent recalls |

`verify-once` is meant for low-ram-fast repeated generation over a pre-verified artifact. It does not disable artifact lossless verification; the Phase 7.9C raw artifact was separately checked with `rllm verify`.

### Benchmark harness

Added:

```text
scripts/phase79c_low_ram_fast_benchmark.py
```

It reproducibly:

1. builds release `rllm`,
2. packs the raw/tile-block artifact,
3. optionally runs `rllm verify`,
4. runs the same Phase 7.6/7.8/7.9B RSS matrix,
5. writes CSV/Markdown summaries,
6. compares against the Phase 7.9B baseline CSV.

The generic Phase 7.6 benchmark script also accepts repeated extra run arguments:

```bash
python3 scripts/phase76_release_rss_benchmark.py \
  --run-arg --rama-integrity \
  --run-arg verify-once
```

## Artifact

Real local Pythia-70M was repacked as a raw/tile-block `.spsa` artifact:

```bash
target/release/rllm pack models/pythia-70m/model.safetensors \
  --out models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa \
  --codec raw \
  --tile-block-elements 65536 \
  --config models/pythia-70m/config.json \
  --tokenizer models/pythia-70m/tokenizer.json
```

Measured artifact size:

```text
models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa
160.60 MiB
```

This is larger than the Huffman tile-block artifact, but it removes Huffman decode from the token loop.

Lossless verification passed:

```text
[OK] Verified 94 tensors, 166019180 bytes total
[OK] LOSSLESS VERIFIED
```

## Benchmarks

All benchmark rows used:

```text
artifact: models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa
prompt: Hello
ctx: 128,512,1024
max-new-tokens: 1,4,8,16
memory budget: 100mb
measurement: /usr/bin/time -l target/release/rllm run ...
host: Apple A18 Pro, 6 CPU cores
```

### Summary by phase

| Phase / profile | Avg seconds/token | Avg tok/s | Best tok/s | Max RSS | Notes |
|---|---:|---:|---:|---:|---|
| Phase 7.9B Huffman tile-block + embedding row recall | 2.93 | 0.343 | 0.373 | 22.28 MiB | Ultra-low-RAM compressed path |
| Phase 7.9C raw tile-block, `strict` integrity | 0.58 | 1.717 | 1.852 | 22.23 MiB | Removes Huffman decode; checksums every recall |
| Phase 7.9C raw tile-block, `verify-once` integrity | 0.35 | 3.262 | 4.348 | 23.36 MiB | Verifies each chunk once per process |

### Phase 7.9C strict matrix

Command:

```bash
python3 scripts/phase79c_low_ram_fast_benchmark.py \
  --skip-build \
  --skip-pack \
  --skip-verify \
  --tokens 1,4,8,16 \
  --ctx 128,512,1024 \
  --timeout-seconds 900
```

Result summary:

```text
12/12 passed
seconds/token: 0.54–0.61; avg 0.58
tokens/second: 1.639–1.852; avg 1.717
max RSS: 19.55–22.23 MiB; avg 20.37 MiB
average speedup vs Phase 7.9B: 5.02×
```

### Phase 7.9C verify-once matrix

Command:

```bash
python3 scripts/phase79c_low_ram_fast_benchmark.py \
  --skip-build \
  --skip-pack \
  --skip-verify \
  --tokens 1,4,8,16 \
  --ctx 128,512,1024 \
  --rama-integrity verify-once \
  --out-dir target/phase79c-low-ram-fast-verify-once \
  --timeout-seconds 900
```

Result summary:

```text
12/12 passed
seconds/token: 0.23–0.60; avg 0.35
tokens/second: 1.667–4.348; avg 3.262
max RSS: 19.17–23.36 MiB; avg 20.50 MiB
average speedup vs Phase 7.9B: 9.58×
```

Paired comparison versus Phase 7.9B:

| ctx | tokens | 7.9B s/token | 7.9C verify-once s/token | speedup | 7.9B RSS | 7.9C RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 2.68 | 0.55 | 4.87× | 18.95 | 19.55 |
| 128 | 4 | 2.70 | 0.29 | 9.31× | 19.66 | 19.80 |
| 128 | 8 | 2.80 | 0.25 | 11.20× | 20.53 | 20.72 |
| 128 | 16 | 2.89 | 0.23 | 12.57× | 21.20 | 22.31 |
| 512 | 1 | 2.93 | 0.59 | 4.97× | 19.50 | 19.64 |
| 512 | 4 | 2.91 | 0.31 | 9.39× | 19.75 | 19.94 |
| 512 | 8 | 2.92 | 0.26 | 11.23× | 20.55 | 20.59 |
| 512 | 16 | 2.94 | 0.24 | 12.25× | 22.28 | 23.36 |
| 1024 | 1 | 2.97 | 0.60 | 4.95× | 19.70 | 19.17 |
| 1024 | 4 | 3.03 | 0.31 | 9.77× | 19.58 | 19.72 |
| 1024 | 8 | 3.11 | 0.27 | 11.52× | 19.69 | 19.81 |
| 1024 | 16 | 3.23 | 0.25 | 12.92× | 21.09 | 21.44 |

## Trace findings after 7.9C

### Raw/tile-block strict, one-token trace

```text
events: 3405
chunk_decode:                 1.276 ms total
chunk_read:                   8.723 ms total
chunk_compressed_checksum:  159.068 ms total
chunk_original_checksum:    160.144 ms total
chunk_compute_closure:      195.853 ms total
```

Interpretation:

```text
Huffman decode was removed from the token loop.
The next bottlenecks became SHA verification and compute, especially lm_head.
```

Top tensor:

```text
embed_out.weight: 301.639 ms / 1965 events in one-token trace
```

### Raw/tile-block verify-once, 16-token trace

```text
events: 34050
chunk_read:                10896 events / 138.815 ms
chunk_decode:              10896 events /  24.268 ms
chunk_compute_closure:     10896 events / 3144.776 ms
chunk_compressed_checksum:   681 events / 161.264 ms
chunk_original_checksum:     681 events / 160.098 ms
```

`verify-once` worked as intended: checksum events are per unique chunk instead of per recall. The remaining dominant phase is now compute closure.

Top tensor:

```text
embed_out.weight: 2098.899 ms / 19650 events in the 16-token trace
```

Trace-mode RSS was much higher than normal (`~127 MiB` for the 16-token trace) because it buffered 34,050 JSON events. Normal non-trace verify-once benchmark RSS stayed around 19–23 MiB.

## HF/PyTorch parity

Raw/tile-block artifact still preserved the tested HF parity:

```bash
uv run --with torch --with transformers --with safetensors \
  scripts/phase77_compare_logits.py \
  --rllm-artifact models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa \
  --out-dir target/phase79c-logits \
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

## Did this hit 10 tok/s?

No.

Best measured non-trace row so far:

```text
4.348 tok/s
0.23 seconds/token
RSS: 22.31 MiB
```

Gap to 10 tok/s:

```text
10 / 4.348 = ~2.30× more speed needed
```

The important result is that Phase 7.9C proves the low-RAM + faster direction is viable:

```text
0.34 tok/s → 4.35 tok/s while staying around 20–23 MiB RSS
```

But reaching 10 tok/s likely requires compute-side optimization, not more Huffman/codec tuning.

Follow-up Phase 7.9D showed that these Phase 7.9C numbers are short-prompt/decode-loop evidence, not long-prompt evidence. With actual deterministic `--token-ids` prompt lengths under `ctx=2048`, the same raw/tile-block artifact kept the 1-token prompt + 16 generated-token row at 4.301 tok/s / 20.67 MiB RSS, but 512-token and 1024-token prompts dropped to 0.300 tok/s / 44.98 MiB RSS and 0.148 tok/s / 70.84 MiB RSS. See [`phase79-long-prompt-benchmark.md`](phase79-long-prompt-benchmark.md).

## Next target

After Phase 7.9D, the short-prompt token loop is no longer dominated by Huffman decode, but real long prompts first need better prefill/KV/context timing and optimization. Full-vocab `embed_out.weight` / lm_head projection remains a likely decode-step bottleneck, but it should be re-profiled after prefill/decode timing is split.

Next RAMA-native candidates:

1. Split request timing into prefill, decode step, lm_head, and sampling.
2. Profile 512-token and 1024-token prompts without buffering huge trace JSON when possible.
3. Optimize RAMA prefill/KV/context memory growth and repeated per-token work.
4. Then add row-parallel/tile-parallel matmul for `streaming_tile_linear_from_model` / low-RAM exact `lm_head` if decode-step timing still shows it dominates.

Target for next slice:

```text
Phase 7.9E — Long-prompt prefill/context timing split and optimization
Goal: recover practical long-prompt latency while preserving the short-prompt low-RAM envelope and exact logits parity.
```

## Status

```text
[PRODUCTION-READY]
- default strict integrity remains unchanged
- raw forced codec packs losslessly and verifies against original safetensors
- benchmark harness is reproducible
- HF top-1 parity preserved on tested prompt

[EXPERIMENTAL]
- --rama-integrity verify-once is opt-in speed mode for pre-verified artifacts
- raw/tile-block low-ram-fast profile trades disk size for runtime speed

[NOT YET MET]
- 10 tok/s target; best measured row is 4.35 tok/s
- practical 512/1024-token prompt latency; Phase 7.9D exposes prefill/context bottlenecks
```
