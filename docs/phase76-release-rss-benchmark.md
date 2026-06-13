# Phase 7.6 Release RSS Benchmark

This benchmark turns the Phase 7.6 smoke test into a repeatable release-build matrix. The table below reflects the current runtime after the Phase 7.7 GPT-NeoX fidelity fixes (`use_parallel_residual` metadata + per-head QKV split).

## Setup

Artifact:

```text
models/pythia-70m-phase76-16mb.rllm
```

Artifact properties:

```text
Pythia-70M
16MiB chunks
120.46 MiB compressed
persisted gpt_neox config metadata
persisted use_parallel_residual metadata
persisted hf-bpe tokenizer metadata
```

Benchmark harness:

```bash
python3 scripts/phase76_release_rss_benchmark.py \
  --tokens 1,4,8,16 \
  --ctx 128,512,1024 \
  --memory-budget 100mb
```

The script builds `target/release/rllm`, then measures the release binary directly with macOS `/usr/bin/time -l` so RSS does not include Cargo wrapper overhead. Generated CSV/Markdown outputs are written to ignored `target/phase76-bench/`.

## Results

Prompt:

```text
Hello
```

Post-Phase 7.7 generation behavior is deterministic across the matrix and starts with HF-aligned token id `13`, decoding to `,`.

| ctx | max new tokens | exit | real s | s/token | max RSS MiB | peak footprint MiB | peak transient | generated text |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 1 | 0 | 4.47 | 4.47 | 88.62 | 87.23 | 48.00 MiB | `,` |
| 128 | 4 | 0 | 18.11 | 4.53 | 89.55 | 88.16 | 48.00 MiB | `, I'm trying` |
| 128 | 8 | 0 | 41.85 | 5.23 | 89.95 | 92.53 | 48.00 MiB | `, I'm trying to get the name` |
| 128 | 16 | 0 | 81.88 | 5.12 | 94.62 | 93.23 | 48.00 MiB | `, I'm trying to get the name of the phone number in the phone number` |
| 512 | 1 | 0 | 5.12 | 5.12 | 89.55 | 88.16 | 48.00 MiB | `,` |
| 512 | 4 | 0 | 20.05 | 5.01 | 90.59 | 89.20 | 48.00 MiB | `, I'm trying` |
| 512 | 8 | 0 | 41.51 | 5.19 | 89.91 | 88.52 | 48.00 MiB | `, I'm trying to get the name` |
| 512 | 16 | 0 | 83.41 | 5.21 | 93.77 | 92.38 | 48.00 MiB | `, I'm trying to get the name of the phone number in the phone number` |
| 1024 | 1 | 0 | 5.29 | 5.29 | 88.66 | 87.28 | 48.00 MiB | `,` |
| 1024 | 4 | 0 | 20.92 | 5.23 | 88.89 | 87.52 | 48.00 MiB | `, I'm trying` |
| 1024 | 8 | 0 | 41.24 | 5.16 | 91.88 | 90.30 | 48.00 MiB | `, I'm trying to get the name` |
| 1024 | 16 | 0 | 83.16 | 5.20 | 90.91 | 87.45 | 48.00 MiB | `, I'm trying to get the name of the phone number in the phone number` |

## Summary

```text
Rows: 12/12 succeeded
RSS range: 88.62–94.62 MiB
Peak RSS run: ctx=128, max_new_tokens=16, 94.62 MiB
Longest run: ctx=512, max_new_tokens=16, 83.41s
Throughput: ~4.47–5.29 seconds/token in release
Tracked transient peak: 48.00 MiB across all runs
```

Compared with the earlier debug one-token smoke:

```text
Debug 1-token direct binary:   34.58s, 90.28 MiB RSS
Release 1-token matrix run:     4.47s, 88.62 MiB RSS
Release speedup:                ~7.74x for the one-token case
```

Compared with the full-decode planning baseline:

```text
Full-decode baseline: 364.66 MiB
Max release RSS:       94.62 MiB
Reduction:             74.05%
Full-decode/RSS:       3.85x
```

Planner comparison:

```text
16MiB tile-stream planner peak: 44.77 MiB
Max measured release RSS:       94.62 MiB
RSS/planner:                    2.11x
```

The gap is expected because `MemoryBudget` tracks RLLM internal transient/tensor buffers while process RSS also includes binary/code pages, allocator overhead, metadata/tokenizer structures, stack, and other runtime allocations.

## Interpretation

[PRODUCTION-READY]

- The release binary can run actual local Pythia-70M token generation under a `100mb` internal budget.
- The measured OS RSS stays under 100 MiB for this benchmark matrix.
- Token generation is deterministic for the tested prompt and sampling config.

[EXPERIMENTAL]

- Speed is still slow: roughly 4.5–5.3 seconds/token on this machine.
- The runtime still decodes chunk-level original bytes before tile conversion; it is not yet true codec-level range decode.
- The tokenizer is a simplified runtime tokenizer, not full HuggingFace BPE/normalizer fidelity.

[VERIFIED ELSEWHERE]

- HF/PyTorch fixed-token logits parity is covered separately in [`phase77-hf-logits-comparison.md`](phase77-hf-logits-comparison.md).

[NOT VERIFIED]

- Longer prompt sweeps and larger batch/context stress cases are not yet benchmarked.
- RSS is empirical for this macOS environment and may differ across OS/allocator/build settings.

## Next steps

1. Implement pack-time tile alignment.
2. Implement codec-level range decode so the runtime can decode only the tile/range needed for matmul.
3. Add broader tokenizer/text parity and longer-prompt reference sweeps.
4. Add a process-RSS estimator layer above `MemoryBudget` so reports can show both internal planned peak and expected process RSS.
