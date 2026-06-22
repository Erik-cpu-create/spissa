# Phase 7.9D Real Long-Prompt Benchmark

Phase 7.9D validates the Phase 7.9C low-ram-fast raw/tile-block runtime against *actual* long prompts, not just larger `--ctx` capacities.

Previous Phase 7.9C chat-speed numbers used the short prompt `Hello` while sweeping context capacity. That proved the short-prompt decode loop could reach up to ~4 tok/s with ~20–23 MiB RSS, but it did not prove long prefill behavior.

## Artifact and runtime

```text
artifact: models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa
codec/layout: raw tile-block, --tile-block-elements 65536
runtime integrity: verify-once
memory budget: 100mb
ctx: 2048
```

Harness:

```bash
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 1,128,512,1024 \
  --max-new-tokens 1,4,16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --out-dir target/phase79d-long-prompt \
  --timeout-seconds 1800
```

The harness passes deterministic fixed token IDs via `rllm run --token-ids`, so tokenizer behavior is not part of this benchmark. The reported generated-token throughput is end-to-end CLI request throughput: process startup, artifact open, real prompt/prefill processing, decode, sampling, output printing, and `/usr/bin/time -l` measurement.

## Benchmark result

```text
rows: 12/12 successful
seconds/generated-token: 0.23–102.63; avg 17.73
generated tokens/sec:    0.010–4.301; avg 0.997
max RSS:                 19.86–74.14 MiB; avg 40.47 MiB
```

| input tokens | new tokens | real seconds | sec/gen token | gen tok/sec | max RSS MiB | peak footprint MiB | peak transient |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.54 | 0.54 | 1.852 | 20.59 | 19.14 | 298.00 KiB |
| 1 | 4 | 1.17 | 0.29 | 3.419 | 19.86 | 18.41 | 298.00 KiB |
| 1 | 16 | 3.72 | 0.23 | 4.301 | 20.67 | 19.22 | 298.00 KiB |
| 128 | 1 | 8.79 | 8.79 | 0.114 | 22.81 | 21.36 | 2.53 MiB |
| 128 | 4 | 9.71 | 2.43 | 0.412 | 23.00 | 21.55 | 2.53 MiB |
| 128 | 16 | 12.65 | 0.79 | 1.265 | 23.41 | 21.95 | 2.53 MiB |
| 512 | 1 | 48.42 | 48.42 | 0.021 | 50.23 | 44.66 | 10.03 MiB |
| 512 | 4 | 49.86 | 12.46 | 0.080 | 44.83 | 41.75 | 10.03 MiB |
| 512 | 16 | 53.37 | 3.34 | 0.300 | 44.98 | 41.59 | 10.03 MiB |
| 1024 | 1 | 102.63 | 102.63 | 0.010 | 74.14 | 56.58 | 20.03 MiB |
| 1024 | 4 | 104.53 | 26.13 | 0.038 | 70.30 | 56.23 | 20.03 MiB |
| 1024 | 16 | 108.05 | 6.75 | 0.148 | 70.84 | 56.73 | 20.03 MiB |

## Interpretation

Phase 7.9C remains valid for short-prompt chat/decode:

```text
1 input token + 16 generated tokens: 4.30 tok/s, 20.67 MiB RSS
```

But real long prompts expose the next bottleneck:

```text
128 input tokens + 16 generated tokens: 1.27 tok/s, 23.41 MiB RSS
512 input tokens + 16 generated tokens: 0.30 tok/s, 44.98 MiB RSS
1024 input tokens + 16 generated tokens: 0.15 tok/s, 70.84 MiB RSS
```

This is not a codec bottleneck. The raw/tile-block artifact removes Huffman decode from the fast path. The cliff is from the current prefill/context implementation: real prompt length grows KV/context memory and performs expensive per-token work before generation can amortize decode speed.

Phase 7.9E adds an opt-in RAMA chunked prefill path that partially fixes this cliff; see [`phase79e-rama-prefill-chunking.md`](phase79e-rama-prefill-chunking.md).

## Scientific parity checks

Long fixed-token HF/PyTorch logits comparisons were run against the same raw/tile-block artifact.

### 128-token prompt

```bash
uv run --with torch --with transformers --with safetensors \
  scripts/phase77_compare_logits.py \
  --rllm-artifact models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa \
  --out-dir target/phase79d-long-prompt-logits-128 \
  --token-ids <128 deterministic ids> \
  --ctx 2048 \
  --memory-budget 100mb
```

```text
top1_match: true
top5_overlap: 5/5
top10_overlap: 10/10
max_abs_diff: 0.01708984
mean_abs_diff: 0.00942427
RLLM top1: 12092
HF top1: 12092
```

### 512-token prompt

```bash
uv run --with torch --with transformers --with safetensors \
  scripts/phase77_compare_logits.py \
  --rllm-artifact models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa \
  --out-dir target/phase79d-long-prompt-logits-512 \
  --token-ids <512 deterministic ids> \
  --ctx 2048 \
  --memory-budget 100mb
```

```text
top1_match: true
top5_overlap: 5/5
top10_overlap: 10/10
max_abs_diff: 0.02746582
mean_abs_diff: 0.01679823
RLLM top1: 12092
HF top1: 12092
```

## Status

[PRODUCTION-READY]

- Short-prompt Phase 7.9C raw/tile-block verify-once benchmark remains a valid low-RAM chat/decode result.
- Fixed-token long-prompt logits parity passes for 128-token and 512-token deterministic prompts.

[EXPERIMENTAL]

- Long-prompt request latency and RSS are not yet good enough for practical long-context chat.
- 512-token prompts approach/exceed the previous 50 MiB preferred RAM target depending on generated length.
- 1024-token prompts remain under the 100 MiB internal budget but climb to ~70–74 MiB RSS and are slow.

[NOT DONE]

- The CLI benchmark does not split prefill and decode timings directly.
- Cold-cache vs warm-cache measurement is still pending.
- 1024-token HF parity was not run in this slice because the 512-token comparison already validates the failing/cliff region and 1024-token end-to-end runs take ~100s each.

## Next implementation target

The next Phase 7 slice should optimize real long-prompt prefill/context behavior before further short-prompt speed tuning.

Recommended order:

1. Add timing split: prefill vs decode-step vs lm-head.
2. Profile a 512-token prompt with `--rama-trace` or lighter non-buffered timing counters.
3. Optimize prefill/KV/context path while preserving exact full-vocab logits.
4. Re-run this 1/128/512/1024 input-token matrix.

Current best next label:

```text
Phase 7.9E: RAMA long-prompt prefill/context profiling and optimization
```
