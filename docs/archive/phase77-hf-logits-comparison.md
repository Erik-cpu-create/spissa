# Phase 7.7 HF/PyTorch Logits Comparison

Phase 7.7 adds a fixed-token-ID external reference comparison for the local Pythia-70M artifact. Fixed token IDs intentionally avoid tokenizer fidelity as a confounder: RLLM and HuggingFace/PyTorch receive the same IDs and compare the next-token logits vector.

## Setup

Artifact:

```text
models/pythia-70m-phase76-16mb.spsa
```

Reference model:

```text
models/pythia-70m
```

RLLM command surface:

```bash
target/release/rllm run models/pythia-70m-phase76-16mb.spsa \
  --token-ids 12092,13 \
  --max-new-tokens 1 \
  --ctx 128 \
  --memory-budget 100mb \
  --logits-out target/phase77/rllm_logits.json
```

Comparison harness:

```bash
uv run --with torch --with transformers --with safetensors \
  scripts/phase77_compare_logits.py \
  --token-ids 12092,13 \
  --ctx 128 \
  --memory-budget 100mb
```

## Runtime fixes required by the comparison

The first HF comparison failed with a large mismatch:

```text
prompt token IDs: [12092]
RLLM top-1: 39091
HF top-1:   13
max abs diff: 634.57830811
mean abs diff: 594.88293457
```

That failure exposed real GPT-NeoX architecture fidelity gaps rather than tokenizer issues:

1. `use_parallel_residual` from `config.json` was not persisted/read by the runtime metadata path.
2. Fused GPT-NeoX QKV projection was split as grouped `[all_q, all_k, all_v]`; GPT-NeoX uses per-head layout `[q_head, k_head, v_head]` for each head.

After fixing those, RLLM's tiled RAMA layer-decode path matches HuggingFace top-k ordering on fixed token IDs.

## Measured comparisons

### Case A — one prompt token

```text
Prompt token IDs: [12092]
RLLM generated token: 13
HF top-1 token: 13
Top-1 match: true
Max abs diff: 0.00451660
Mean abs diff: 0.00144225
```

### Case B — two prompt tokens

```text
Prompt token IDs: [12092, 13]
RLLM generated token: 309
HF top-1 token: 309
Top-1 match: true
Max abs diff: 0.00769043
Mean abs diff: 0.00231486
RMS abs diff: 0.00264794
Top-5 overlap: 5/5
Top-10 overlap: 10/10
```

Top-10 logits for Case B:

| rank | RLLM token/logit | HF token/logit |
|---:|---|---|
| 1 | 309 / 1079.166504 | 309 / 1079.168823 |
| 2 | 187 / 1077.502075 | 187 / 1077.501343 |
| 3 | 368 / 1077.334717 | 368 / 1077.336914 |
| 4 | 352 / 1076.886963 | 352 / 1076.889893 |
| 5 | 359 / 1076.812988 | 359 / 1076.816162 |
| 6 | 42 / 1076.790649 | 42 / 1076.793091 |
| 7 | 253 / 1076.713501 | 253 / 1076.717041 |
| 8 | 849 / 1076.606689 | 849 / 1076.609863 |
| 9 | 619 / 1076.593750 | 619 / 1076.597046 |
| 10 | 627 / 1076.553955 | 627 / 1076.555542 |

## Interpretation

[PRODUCTION-READY]

For fixed token IDs on local Pythia-70M, the RLLM Phase 7 tiled RAMA layer-decode path now matches HuggingFace/PyTorch top-1 and top-10 logits ordering for tested short contexts. Absolute differences are small and consistent with fp16 weights plus implementation/math-order differences.

[EXPERIMENTAL]

This is not a full tokenizer/text-generation parity suite. It proves runtime architecture/logits fidelity for fixed IDs, not full HuggingFace tokenizer normalizer/BPE behavior.

[NOT DONE]

- Longer prompt sweeps are not yet automated.
- Full tokenizer parity is not yet implemented.
- Codec-level range decode and pack-time tile alignment are still future Phase 7 work.
