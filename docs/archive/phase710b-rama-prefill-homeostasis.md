# Phase 7.10B RAMA Prefill Homeostasis

Phase 7.10B validates the post-7.10A runtime across a broader real-prompt matrix, then tunes the RAMA prefill activation window from measurement instead of guessing.

This remains an original RLLM/RAMA memory-first optimization. It does **not** use hot/cold neurons, activation-locality predictors, sparse neuron routing, GPU residency scheduling, or any PowerInfer-style mechanism. The tuned unit is the deterministic prompt prefill recall window: how many real prompt tokens are allowed to be active in one prefill pass.

## Scope

Touched code:

```text
crates/rllm-cli/src/main.rs
crates/rllm-cli/src/commands/run.rs
scripts/phase79d_long_prompt_benchmark.py
```

Runtime API semantics remain unchanged: `GptNeoxRamaGenerationOptions { prefill_chunk_tokens: None }` still means one full prompt prefill pass for in-crate callers.

CLI generation now defaults to the measured RAMA prefill window:

```text
DEFAULT_RAMA_PREFILL_CHUNK_TOKENS = 32
```

Users can override or disable it:

```bash
# Override the window
rllm run model.spsa --token-ids ... --rama-prefill-chunk-tokens 64

# Disable chunked prefill and process the full prompt in one pass
rllm run model.spsa --token-ids ... --no-rama-prefill-chunking
```

`--no-rama-prefill-chunking` and `--rama-prefill-chunk-tokens` are intentionally rejected together to avoid ambiguous CLI behavior.

## Artifact / runtime

```text
artifact: models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa
codec/layout: raw tile-block, --tile-block-elements 65536
runtime integrity: verify-once
ctx: 2048
memory budget: 100mb
hardware/OS measurement: local macOS /usr/bin/time -l process max RSS
```

## Broader post-7.10A matrix

Command shape:

```bash
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --input-tokens 1,128,512,1024 \
  --max-new-tokens 1,4,16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-prefill-chunk-tokens 64 \
  --rama-timing-dir timing \
  --out-dir target/phase710b-post-rowspan-matrix \
  --timeout-seconds 900
```

Result: `12/12` rows passed.

| input tokens | new tokens | real seconds | gen tok/s | max RSS MiB | prefill ms | decode ms | lm_head ms | peak transient |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.44 | 2.273 | 19.86 | 421.49 | 0.00 | 233.11 | 298 KiB |
| 1 | 4 | 0.66 | 6.061 | 19.88 | 405.45 | 242.54 | 367.55 | 298 KiB |
| 1 | 16 | 1.64 | 9.756 | 20.33 | 403.18 | 1219.94 | 922.87 | 298 KiB |
| 128 | 1 | 1.54 | 0.649 | 23.11 | 1528.57 | 0.00 | 233.66 | 1.28 MiB |
| 128 | 4 | 1.84 | 2.174 | 23.12 | 1571.97 | 258.87 | 406.02 | 1.28 MiB |
| 128 | 16 | 2.87 | 5.575 | 23.80 | 1524.75 | 1329.03 | 990.35 | 1.28 MiB |
| 512 | 1 | 5.54 | 0.181 | 32.59 | 5523.57 | 0.00 | 248.84 | 1.28 MiB |
| 512 | 4 | 5.93 | 0.675 | 32.72 | 5628.91 | 281.84 | 412.60 | 1.28 MiB |
| 512 | 16 | 7.21 | 2.219 | 33.92 | 5716.00 | 1482.11 | 1095.99 | 1.28 MiB |
| 1024 | 1 | 12.38 | 0.081 | 44.64 | 12362.20 | 0.00 | 268.74 | 1.28 MiB |
| 1024 | 4 | 13.11 | 0.305 | 44.97 | 12792.69 | 298.15 | 429.02 | 1.28 MiB |
| 1024 | 16 | 14.33 | 1.117 | 45.14 | 12749.54 | 1561.78 | 1115.68 | 1.28 MiB |

For 16-token rows, the timing split showed the active bottleneck clearly:

```text
input=1:    prefill 24.8%, decode 75.2%, lm_head ~=75.6% of decode
input=128:  prefill 53.4%, decode 46.6%, lm_head ~=74.5% of decode
input=512:  prefill 79.4%, decode 20.6%, lm_head ~=73.9% of decode
input=1024: prefill 89.1%, decode 10.9%, lm_head ~=71.4% of decode
```

So after Phase 7.10A, short prompts are now near 10 tok/s, while long prompts remain prefill-dominant.

## Prefill chunk-size sweep

Command shape:

```bash
for chunk in 32 64 128 256 512; do
  python3 scripts/phase79d_long_prompt_benchmark.py \
    --skip-build \
    --input-tokens 512,1024 \
    --max-new-tokens 16 \
    --ctx 2048 \
    --rama-integrity verify-once \
    --rama-prefill-chunk-tokens "$chunk" \
    --rama-timing-dir timing \
    --out-dir "target/phase710b-prefill-sweep/chunk-${chunk}" \
    --timeout-seconds 900
done
```

| prefill chunk | input tokens | real seconds | gen tok/s | max RSS MiB | prefill ms | decode ms | lm_head ms | peak transient |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 512 | 6.81 | 2.3495 | 32.77 | 5441.06 | 1343.15 | 1029.72 | 794 KiB |
| 64 | 512 | 7.42 | 2.1563 | 32.92 | 5905.05 | 1496.64 | 1100.72 | 1.28 MiB |
| 128 | 512 | 7.85 | 2.0382 | 33.38 | 6217.03 | 1619.81 | 1188.86 | 2.53 MiB |
| 256 | 512 | 7.77 | 2.0592 | 36.73 | 6184.25 | 1569.41 | 1169.68 | 5.03 MiB |
| 512 | 512 | 7.57 | 2.1136 | 47.69 | 6029.62 | 1526.15 | 1135.78 | 10.03 MiB |
| 32 | 1024 | 13.73 | 1.1653 | 44.91 | 12223.92 | 1495.63 | 1079.03 | 794 KiB |
| 64 | 1024 | 14.33 | 1.1165 | 45.19 | 12776.49 | 1535.19 | 1103.03 | 1.28 MiB |
| 128 | 1024 | 15.66 | 1.0217 | 45.39 | 13799.12 | 1842.96 | 1330.55 | 2.53 MiB |
| 256 | 1024 | 14.22 | 1.1252 | 49.30 | 12611.69 | 1595.87 | 1155.01 | 5.03 MiB |
| 512 | 1024 | 14.37 | 1.1134 | 59.88 | 12769.20 | 1585.03 | 1153.31 | 10.03 MiB |

Measured winner for both tested long prompts:

```text
prefill chunk = 32 real input tokens
```

Compared with the previous Phase 7.9E/7.10A working value of `64`:

```text
512-token +16:  2.1563 -> 2.3495 tok/s, RSS 32.92 -> 32.77 MiB, prefill 5905.05 -> 5441.06 ms
1024-token +16: 1.1165 -> 1.1653 tok/s, RSS 45.19 -> 44.91 MiB, prefill 12776.49 -> 12223.92 ms
```

This is not a huge speedup, but it is a clean RAMA-homeostasis improvement: lower active transient memory and slightly better long-prompt throughput.

## Default-path smoke

After switching the CLI generation default to the measured 32-token window (now superseded by Phase 7.12A's shape/budget-aware policy for larger shapes), a 512-token + 16 generated smoke without an explicit `--rama-prefill-chunk-tokens` produced timing JSON with:

```json
{
  "max_prefill_chunk_tokens": 32,
  "prefill_chunks": 16
}
```

The same default-32 logits were compared against the prior HF-validated 512-token chunk-64 RLLM logits:

```text
logit length: 50304
max abs diff: 0.0
mean abs diff: 0.0
generated token IDs: [12092] vs [12092]
top-1: 12092 vs 12092
top-10 overlap: 10/10
```

Direct HF/PyTorch dependencies were unavailable in this shell (`uv` and Python `torch` were missing), so Phase 7.10B used this local-equivalence check against the already HF-validated Phase 7.10A 512-token output. Earlier Phase 7.10A HF parity remains the scientific reference for the same prompt/artifact path.

## Interpretation

Phase 7.10B moves the default CLI behavior from explicit experimental chunking to measured RAMA memory homeostasis:

```text
short prompt + 16:  up to 9.756 tok/s at ~20.33 MiB RSS
512 prompt + 16:    best swept row 2.3495 tok/s at ~32.77 MiB RSS
1024 prompt + 16:   best swept row 1.1653 tok/s at ~44.91 MiB RSS
```

Remaining bottleneck by prompt class:

```text
short prompts: decode/lm_head still matter
long prompts: prefill dominates strongly
```

The next optimization should not be another blind chunk-size sweep. Recommended next measured target:

1. add deeper timing inside prefill for attention vs MLP vs layer-param recall,
2. identify whether repeated layer-param recall, attention score work, or MLP projection dominates 512/1024 prompts,
3. only then implement the next low-RAM optimization.

## Caveats

- Benchmarks are local macOS process RSS via `/usr/bin/time -l`; OS page-cache effects are not isolated here.
- The benchmark is still end-to-end CLI request timing, not an always-on chat server loop.
- The fast path is the raw/tile-block low-ram-fast artifact, not Huffman-compressed fast execution.
- Current validation is still Pythia-70M-specific.
- No production-grade latency claim: 1024-token prompts are improved but still not comfortable interactive chat.
