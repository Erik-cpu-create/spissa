# Phase 7.11B Pythia-160M Prefill Window and Memory-Budget Sweep

Phase 7.11B follows Phase 7.11A's scale validation. The goal is to tune the generic RAMA prefill window behavior for the larger GPT-NeoX/Pythia shape without adding model-specific Pythia-160M code.

Result: larger prefill windows transfer cleanly to Pythia-160M. `--rama-prefill-chunk-tokens 64` is the best low-RAM-safe recommendation under an approximate 100 MiB RSS target. `128` is the speed-biased setting if slightly above/around 100 MiB RSS is acceptable. `256` adds little speed over `128` but increases RSS and transient budget materially.

## Scope

Artifact:

```text
models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.rllm
```

Benchmark scope:

```text
input_tokens: 512, 1024
max_new_tokens: 16
ctx: 2048
integrity: verify-once
memory_budget: 100mb for chunk sweep
prefill_chunk_tokens: 8,16,32,64,128,256
```

The benchmark used the existing timing harness:

```bash
python3 scripts/phase79e_prefill_timing_benchmark.py \
  --bin target/release/rllm \
  --artifact models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.rllm \
  --out-dir target/phase711b-pythia160m/chunk-sweep \
  --input-tokens 512,1024 \
  --max-new-tokens 16 \
  --prefill-chunks 8,16,32,64,128,256 \
  --ctx 2048 \
  --memory-budget 100mb \
  --rama-integrity verify-once \
  --timeout-seconds 1800
```

Memory-budget threshold rows used 1024-token input, 16 generated tokens, and `--rama-prefill-chunk-tokens 128`.

## Chunk/window sweep

| input | new | chunk | elapsed s | tok/s | RSS MiB | context MiB | transient KiB | prefill ms | prefill chunks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 16 | 8 | 21.63 | 0.740 | 62.78 | 37.05 | 487.00 | 17208.70 | 64 |
| 512 | 16 | 16 | 18.31 | 0.874 | 63.27 | 37.05 | 679.00 | 14249.49 | 32 |
| 512 | 16 | 32 | 16.53 | 0.968 | 62.67 | 37.05 | 1064.96 | 12421.35 | 16 |
| 512 | 16 | 64 | 15.68 | 1.020 | 63.12 | 37.05 | 1955.84 | 11598.73 | 8 |
| 512 | 16 | 128 | 15.26 | 1.048 | 64.75 | 37.05 | 3880.96 | 11161.25 | 4 |
| 512 | 16 | 256 | 15.14 | 1.057 | 70.58 | 37.05 | 7720.96 | 11021.66 | 2 |
| 1024 | 16 | 8 | 43.35 | 0.369 | 99.05 | 73.05 | 487.00 | 39062.95 | 128 |
| 1024 | 16 | 16 | 35.42 | 0.452 | 98.47 | 73.05 | 679.00 | 30878.18 | 64 |
| 1024 | 16 | 32 | 31.23 | 0.512 | 93.75 | 73.05 | 1064.96 | 26493.96 | 32 |
| 1024 | 16 | 64 | 28.22 | 0.567 | 99.02 | 73.05 | 1955.84 | 24014.05 | 16 |
| 1024 | 16 | 128 | 26.65 | 0.600 | 100.06 | 73.05 | 3880.96 | 22479.29 | 8 |
| 1024 | 16 | 256 | 26.29 | 0.609 | 107.20 | 73.05 | 7720.96 | 22104.59 | 4 |

## Relative to chunk=32

| input | chunk | tok/s delta vs 32 | RSS delta vs 32 |
|---:|---:|---:|---:|
| 512 | 8 | -23.59% | +0.11 MiB |
| 512 | 16 | -9.72% | +0.59 MiB |
| 512 | 32 | baseline | baseline |
| 512 | 64 | +5.39% | +0.45 MiB |
| 512 | 128 | +8.28% | +2.08 MiB |
| 512 | 256 | +9.17% | +7.91 MiB |
| 1024 | 8 | -27.94% | +5.30 MiB |
| 1024 | 16 | -11.82% | +4.72 MiB |
| 1024 | 32 | baseline | baseline |
| 1024 | 64 | +10.70% | +5.27 MiB |
| 1024 | 128 | +17.19% | +6.31 MiB |
| 1024 | 256 | +18.80% | +13.45 MiB |

## Memory-budget sweep

For `input_tokens=1024`, `new_tokens=16`, `chunk=128`:

| budget | exit | elapsed s | tok/s | RSS MiB | context MiB | peak transient MiB | current transient MiB | prefill ms | prefill chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512kb | 1 | 0.024 | n/a | 22.23 | n/a | n/a | n/a | n/a | n/a |
| 768kb | 1 | 0.025 | n/a | 22.80 | n/a | n/a | n/a | n/a | n/a |
| 1mb | 1 | 0.026 | n/a | 23.58 | n/a | n/a | n/a | n/a | n/a |
| 2mb | 1 | 0.026 | n/a | 23.08 | n/a | n/a | n/a | n/a | n/a |
| 4mb | 0 | 25.34 | 0.632 | 100.44 | 73.05 | 3.79 | 0.0 | 21085.22 | 8 |
| 100mb | 0 | 27.32 | 0.586 | 101.36 | 73.05 | 3.79 | 0.0 | 22984.18 | 8 |

Narrow threshold rows:

| budget | exit | elapsed s | tok/s | RSS MiB | peak transient MiB |
|---|---:|---:|---:|---:|---:|
| 3072kb | 1 | 0.08 | n/a | 23.27 | n/a |
| 3584kb | 1 | 0.08 | n/a | 23.30 | n/a |
| 3840kb | 1 | 0.08 | n/a | 23.19 | n/a |
| 3968kb | 0 | 25.65 | 0.624 | 100.80 | 3.79 |
| 4096kb | 0 | 27.49 | 0.582 | 101.70 | 3.79 |

Representative failure:

```text
Error: memory budget exceeded while reserving 1179648 bytes for streaming attention split QKV activation: current=2792448, limit=3932160
```

Interpretation: `--memory-budget` guards the runtime's tracked transient decode/activation allocations. It is not a full-process RSS cap. Context/KV memory still scales with prompt/context shape and is reported separately as `Context memory bytes`.

## Recommendation

Use three modes rather than one universal constant:

```text
Pythia-70M low-RAM default: chunk=32 (existing measured default)
Pythia-160M low-RAM-safe override: chunk=64
Pythia-160M speed-biased override: chunk=128 with >=4 MiB transient budget and tolerance for ~100-102 MiB RSS
Avoid chunk=256 unless RSS >107 MiB and ~7.5 MiB transient are acceptable.
```

Phase 7.12A resolves the global-default question by using a shape/budget-aware policy instead of changing the whole CLI to one fixed number:

- Pythia-70M-like low-RAM runs still select `32`.
- Pythia-160M-like low-RAM runs select `64`.
- `--rama-prefill-policy speed` selects the larger speed-biased window, downshifted if an explicit transient budget is too small.
- Explicit `--rama-prefill-chunk-tokens <n>` still wins for fresh sweeps/repro runs.

A safe user-facing 160M command is:

```bash
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --artifact models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.rllm \
  --input-tokens 1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-prefill-chunk-tokens 64 \
  --memory-budget 100mb
```

A speed-biased command is:

```bash
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --artifact models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.rllm \
  --input-tokens 1024 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-prefill-chunk-tokens 128 \
  --memory-budget 4mb
```

## Next decision

Phase 7.11B answered the chunk/window question, and Phase 7.12A implemented the generic shape/budget-aware default. The next evidence-driven implementation target is now:

```text
Phase 7.12B — optimize generic MLP/QKV projection bottlenecks
```

Keep it generic and timing-driven: no Pythia-160M-specific branches, and no PowerInfer-style hot/cold activation routing.

If the priority is raw token speed, optimize the measured Pythia-160M compute bottlenecks: MLP projection and QKV projection.
