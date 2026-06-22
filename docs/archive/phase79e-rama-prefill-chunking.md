# Phase 7.9E RAMA Chunked Prefill Timing and Optimization

Phase 7.9E adds two RAMA-native long-prompt improvements:

1. `rllm run --rama-timing <path>` — low-overhead aggregate timing JSON for prefill/decode/final-norm/lm-head/sampling without buffering the large per-chunk trace used by `--rama-trace`.
2. `rllm run --rama-prefill-chunk-tokens <n>` — opt-in bounded prompt prefill windows. Intermediate prompt chunks update layer KV caches but skip final-layernorm/lm-head/sampling because only the final prompt token produces the first generated token.

This is an original RLLM/RAMA memory-first optimization. It does **not** use hot/cold neurons, activation locality prediction, GPU residency scheduling, or PowerInfer-style sparse neuron routing.

## Artifact and runtime

```text
artifact: models/pythia-70m-phase79c-low-ram-fast-raw-tileblocks.spsa
codec/layout: raw tile-block, --tile-block-elements 65536
runtime integrity: verify-once
memory budget: 100mb
ctx: 2048
max-new-tokens: 16
```

Benchmark harness:

```bash
python3 scripts/phase79e_prefill_timing_benchmark.py \
  --input-tokens 512 \
  --max-new-tokens 16 \
  --prefill-chunks full,64,128,256 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --memory-budget 100mb
```

## 512-token prompt sweep

| input tokens | new tokens | prefill chunk | elapsed | gen tok/s | RSS | context memory | peak transient | prefill time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 16 | full | 56.43s | 0.284 | 46.22 MiB | 12.35 MiB | 10.03 MiB | 52.08s |
| 512 | 16 | 32 | 41.89s | 0.382 | 32.66 MiB | 12.35 MiB | 794 KiB | 38.15s |
| 512 | 16 | 64 | 35.20s | 0.455 | 34.05 MiB | 12.35 MiB | 1.28 MiB | 31.28s |
| 512 | 16 | 128 | 44.09s | 0.363 | 33.69 MiB | 12.35 MiB | 2.53 MiB | 40.18s |
| 512 | 16 | 256 | 59.94s | 0.267 | 36.14 MiB | 12.35 MiB | 5.03 MiB | 56.05s |

Best speed in this sweep: `--rama-prefill-chunk-tokens 64`.

Compared with full prefill at 512 input tokens:

```text
elapsed reduction: 37.63%
speedup:           1.60x
RSS reduction:     26.34% / 12.17 MiB
transient peak:    10.03 MiB -> 1.28 MiB
```

## 1024-token validation

| input tokens | new tokens | prefill chunk | elapsed | gen tok/s | RSS | context memory | peak transient | prefill time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | full | 110.29s | 0.145 | 70.55 MiB | 24.35 MiB | 20.03 MiB | 106.46s |
| 1024 | 16 | 32 | 80.12s | 0.200 | 44.89 MiB | 24.35 MiB | 794 KiB | 76.28s |
| 1024 | 16 | 64 | 63.84s | 0.251 | 44.98 MiB | 24.35 MiB | 1.28 MiB | 59.95s |

Compared with full prefill at 1024 input tokens:

```text
elapsed reduction: 42.12%
speedup:           1.73x
RSS reduction:     36.23% / 25.56 MiB
transient peak:    20.03 MiB -> 1.28 MiB
```

## Short/medium prompt smoke

| input tokens | new tokens | prefill chunk | elapsed | gen tok/s | RSS | prefill time |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | full | 3.85s | 4.158 | 20.17 MiB | 0.54s |
| 1 | 16 | 64 | 3.77s | 4.245 | 20.39 MiB | 0.53s |
| 128 | 16 | 64 | 10.82s | 1.478 | 23.88 MiB | 7.28s |

Short-prompt behavior remains effectively unchanged. The chunked path mainly helps when real input length is much larger than the chunk window.

## Timing split

Representative aggregate timing JSON:

```text
512 full:  prefill 52.08s, decode 3.83s, lm_head 2.58s, prefill_chunks=1
512 c64:   prefill 31.28s, decode 3.89s, lm_head 2.63s, prefill_chunks=8
1024 full: prefill 106.46s, decode 3.82s, lm_head 2.55s, prefill_chunks=1
1024 c64:  prefill 59.95s, decode 3.86s, lm_head 2.58s, prefill_chunks=16
```

The long-prompt bottleneck is confirmed as prefill/context work. Decode and lm-head are relatively stable across prompt length; lm-head remains a short-prompt bottleneck candidate, but long-prompt latency is dominated by prefill.

## Correctness

Runtime unit coverage:

```bash
cargo test -p rllm-runtime gpt_neox -- --nocapture
```

Result:

```text
10 passed; 0 failed
```

The new tests verify that chunked prefill:

- matches full prefill generated token IDs,
- matches the full token sequence,
- matches step logits exactly on the tiny GPT-NeoX fixture,
- preserves context memory bytes,
- collects timing metadata,
- rejects zero chunk size without leaking transient budget.

Real Pythia-70M HF/PyTorch logits parity was also run with `--rama-prefill-chunk-tokens 64` on a deterministic 512-token prompt:

```text
top1_match: true
top5_overlap: 5/5
top10_overlap: 10/10
max_abs_diff: 0.02746582
mean_abs_diff: 0.01679823
```

These match the previous full-prefill 512-token parity metrics, so chunking does not change the tested logits.

## Interpretation

Phase 7.9D exposed that real long prompts were not yet practical:

```text
512 tokens + 16 generated: 0.300 tok/s / 44.98 MiB RSS
1024 tokens + 16 generated: 0.148 tok/s / 70.84 MiB RSS
```

Phase 7.9E improves both speed and memory for the same prompt class:

```text
512 tokens + 16 generated, chunk=64:  0.455 tok/s / 34.05 MiB RSS
1024 tokens + 16 generated, chunk=64: 0.251 tok/s / 44.98 MiB RSS
```

This is still not fast enough for comfortable long-document chat, but it is a real RAMA prefill/context improvement and gives a concrete tuning knob.

Follow-up Phase 7.10A keeps the same chunked prefill mechanism but optimizes the shared tiled-linear accumulation hot loop. With `--rama-prefill-chunk-tokens 64`, the 512-token + 16 generated row improved further from 35.20s / 0.455 tok/s to 7.08s / 2.259 tok/s, and the 1024-token + 16 generated row improved from 63.84s / 0.251 tok/s to 12.76s / 1.254 tok/s. Phase 7.10B then sweeps the post-rowspan prefill window and chooses 32 tokens as the measured Pythia-70M-like default; Phase 7.12A supersedes the fixed CLI default with a generic shape/budget-aware policy.

## Current recommendation

Phase 7.12A supersedes the earlier fixed-window recommendation. The current CLI default is the auto low-RAM policy:

```bash
--rama-prefill-policy low-ram
--rama-integrity verify-once
```

For Pythia-70M-like shapes this still selects 32 tokens; for Pythia-160M-like low-RAM runs it selects 64 tokens. Use `--rama-prefill-policy speed` for the larger speed-biased window when budget/RSS tolerance allows. Override with `--rama-prefill-chunk-tokens <n>` only when running a fresh sweep for a specific artifact/machine, or use `--no-rama-prefill-chunking` to reproduce full-prompt prefill.

## Next bottleneck

After chunked prefill, Phase 7.10A row-span accumulation, and Phase 7.10B homeostasis, the next measured work is:

1. add deeper timing inside attention/MLP/layer-param recall for real long-prompt prefill,
2. optimize the measured dominant prefill sub-phase,
3. consider low-RAM parallel row-span accumulation only if short-prompt decode/lm-head becomes the priority.

Do not pursue PowerInfer-like neuron predictors or hot/cold activation routing. The RAMA path remains chunk/tile/context-memory oriented.
