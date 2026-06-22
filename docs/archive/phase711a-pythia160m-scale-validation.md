# Phase 7.11A Pythia-160M Scale Validation

Phase 7.11A validates that the RAMA GPT-NeoX/Pythia runtime path is family-generic beyond the local Pythia-70M baseline. The goal is not to add model-specific adaptation for Pythia-160M; any runtime fix would need to apply generically to the GPT-NeoX/Pythia adapter or shared RAMA kernels.

Result: Pythia-160M packs, verifies, runs, matches HF/PyTorch top-k on a fixed-token smoke prompt, and completes the conservative timing/RSS matrix without model-specific code changes.

## Scope

Model source:

```text
EleutherAI/pythia-160m
```

Downloaded local files:

```text
models/pythia-160m/model.safetensors      374,998,696 bytes
models/pythia-160m/config.json            569 bytes
models/pythia-160m/tokenizer.json         2,113,710 bytes
models/pythia-160m/tokenizer_config.json  396 bytes
models/pythia-160m/special_tokens_map.json 99 bytes
```

Config confirms the same GPT-NeoX/Pythia family path:

```text
architecture: GPTNeoXForCausalLM
model_type: gpt_neox
hidden_size: 768
intermediate_size: 3072
num_hidden_layers: 12
num_attention_heads: 12
max_position_embeddings: 2048
rotary_pct: 0.25
rotary_emb_base: 10000
use_parallel_residual: true
vocab_size: 50304
```

## Artifact

Pack command:

```bash
target/release/rllm pack models/pythia-160m/model.safetensors \
  --out models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.spsa \
  --codec raw \
  --tile-block-elements 65536 \
  --config models/pythia-160m/config.json \
  --tokenizer models/pythia-160m/tokenizer.json
```

Pack result:

```text
Encoded 3366 chunks total
Codec policy: rtc-raw-v1
Tile-block packing: 184 tensor(s), 65536 element(s) per chunk/block
Original size: 374,977,752 bytes
Compressed size: 374,977,752 bytes
Compression ratio: 100.0%
Artifact size on disk: 367 MiB
```

Inspect result:

```text
Format version: 1
Architecture: gpt_neox
Lossless: true
Codec: rtc-raw-v1
Model config present: yes
Tokenizer present: yes
Tensors: 184
Layers: 12
Hidden size: 768
Attention heads: 12
Intermediate size: 3072
Parallel residual: true
```

## Lossless verification

Verify command:

```bash
target/release/rllm verify \
  models/pythia-160m/model.safetensors \
  models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.spsa
```

Result:

```text
[OK] Verified 184 tensors, 374,977,752 bytes total
[OK] LOSSLESS VERIFIED
```

## Runtime smoke

Smoke command shape:

```bash
/usr/bin/time -l target/release/rllm run \
  models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.spsa \
  --token-ids 12092 \
  --ctx 128 \
  --max-new-tokens 1 \
  --memory-budget 100mb \
  --rama-integrity verify-once \
  --rama-timing target/phase711a-pythia160m/smoke/rama_timing_one_token.json \
  --logits-out target/phase711a-pythia160m/smoke/rllm_logits_one_token.json
```

Result:

```text
Prompt token IDs: [12092]
Generated token IDs: [2]
Generated text: !
Full text: Hello!
Resident non-layer params: 6.00 KiB
Max active layer params: 39.00 KiB
Context memory bytes: 72.00 KiB
Peak transient budget: 319.00 KiB
Current transient budget: 0 B
real: 1.38s
max RSS: 25,772,032 bytes (~24.58 MiB)
```

Repeatability check:

```text
generated_a: [2]
generated_b: [2]
logits length: 50304
max_abs_diff: 0.0
mean_abs_diff: 0.0
top1_a_b: 2 / 2
top10_overlap: 10/10
```

## HF/PyTorch logits comparison

Command:

```bash
uv run --with torch --with transformers --with safetensors \
  scripts/phase77_compare_logits.py \
  --model-dir models/pythia-160m \
  --rllm-artifact models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.spsa \
  --token-ids 12092 \
  --ctx 128 \
  --memory-budget 100mb \
  --rama-integrity verify-once \
  --rama-prefill-chunk-tokens 32 \
  --out-dir target/phase711a-pythia160m/hf-logits \
  --timeout-seconds 900
```

Result:

```text
top1_match: true
top5_overlap: 5/5
top10_overlap: 10/10
max_abs_diff: 0.02246094
mean_abs_diff: 0.01319405
RMS abs diff: 0.01338230
RLLM top-1 id: 2
HF top-1 id: 2
```

Interpretation: the existing GPT-NeoX/Pythia adapter remains numerically faithful enough on the Pythia-160M fixed-token smoke prompt. The absolute diff is larger than the earlier 70M smoke but top-k ordering is preserved for the tested prompt.

## Benchmark matrix

Benchmark command shape:

```bash
python3 scripts/phase79d_long_prompt_benchmark.py \
  --skip-build \
  --artifact models/pythia-160m-phase711a-low-ram-fast-raw-tileblocks.spsa \
  --input-tokens 1,128,512,1024 \
  --max-new-tokens 1,4,16 \
  --ctx 2048 \
  --rama-integrity verify-once \
  --rama-timing-dir timing \
  --memory-budget 100mb
```

The matrix was run in two chunks:

```text
target/phase711a-pythia160m/benchmark-1-128-512
target/phase711a-pythia160m/benchmark-1024
```

All rows exited successfully.

| input tokens | new tokens | real seconds | gen tok/sec | max RSS MiB | peak transient | prefill ms | attention ms | QKV ms | score/context ms | MLP ms | lm_head ms |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.17 | 0.855 | 23.92 | 319.00 KiB | 1150.51 | 265.61 | 198.18 | 0.02 | 534.56 | 346.70 |
| 1 | 4 | 1.85 | 2.162 | 23.84 | 319.00 KiB | 1131.74 | 277.41 | 214.26 | 0.02 | 507.26 | 562.65 |
| 1 | 16 | 4.98 | 3.213 | 25.88 | 319.00 KiB | 1157.66 | 270.55 | 198.85 | 0.03 | 526.46 | 1553.01 |
| 128 | 1 | 3.76 | 0.266 | 32.92 | 1.04 MiB | 3742.26 | 1131.39 | 825.94 | 32.58 | 2197.23 | 396.79 |
| 128 | 4 | 4.70 | 0.851 | 33.20 | 1.04 MiB | 3841.65 | 1153.95 | 846.19 | 30.59 | 2272.85 | 649.08 |
| 128 | 16 | 7.99 | 2.003 | 35.91 | 1.04 MiB | 3869.23 | 1163.79 | 829.41 | 32.25 | 2285.43 | 1652.24 |
| 512 | 1 | 12.84 | 0.078 | 59.91 | 1.04 MiB | 12822.97 | 4366.73 | 2893.15 | 473.65 | 7984.20 | 412.65 |
| 512 | 4 | 13.97 | 0.286 | 60.78 | 1.04 MiB | 13066.17 | 4427.15 | 2939.43 | 478.44 | 8154.93 | 707.73 |
| 512 | 16 | 17.73 | 0.902 | 62.72 | 1.04 MiB | 13341.90 | 4530.49 | 3004.50 | 483.67 | 8323.12 | 1743.84 |
| 1024 | 1 | 25.16 | 0.040 | 96.69 | 1.04 MiB | 25140.34 | 9415.46 | 5515.18 | 1925.96 | 15183.04 | 424.68 |
| 1024 | 4 | 27.33 | 0.146 | 97.00 | 1.04 MiB | 26424.03 | 9805.48 | 5758.57 | 1962.25 | 16069.56 | 683.52 |
| 1024 | 16 | 31.08 | 0.515 | 99.47 | 1.04 MiB | 26606.40 | 9881.80 | 5833.58 | 1979.05 | 16181.24 | 1755.75 |

## Comparison to Pythia-70M Phase 7.10E

Matching 16-token rows:

| row | metric | Pythia-70M 7.10E | Pythia-160M 7.11A | ratio |
|---|---|---:|---:|---:|
| 512 + 16 | real seconds | 4.56 | 17.73 | 3.89× slower |
| 512 + 16 | gen tok/sec | 3.5088 | 0.9024 | 0.26× |
| 512 + 16 | max RSS MiB | 32.88 | 62.72 | 1.91× |
| 512 + 16 | prefill ms | 2790.08 | 13341.90 | 4.78× |
| 512 + 16 | attention ms | 893.84 | 4530.49 | 5.07× |
| 512 + 16 | MLP ms | 1623.38 | 8323.12 | 5.13× |
| 512 + 16 | lm_head ms | 991.41 | 1743.84 | 1.76× |
| 1024 + 16 | real seconds | 7.12 | 31.08 | 4.37× slower |
| 1024 + 16 | gen tok/sec | 2.2472 | 0.5148 | 0.23× |
| 1024 + 16 | max RSS MiB | 44.97 | 99.47 | 2.21× |
| 1024 + 16 | prefill ms | 5749.73 | 26606.40 | 4.63× |
| 1024 + 16 | attention ms | 2147.87 | 9881.80 | 4.60× |
| 1024 + 16 | MLP ms | 3309.76 | 16181.24 | 4.89× |
| 1024 + 16 | lm_head ms | 1003.67 | 1755.75 | 1.75× |

## Interpretation

Pythia-160M validates the intended architecture boundary:

- Same `.spsa` format.
- Same GPT-NeoX/Pythia adapter.
- Same RAMA tiled linear, chunked prefill, verify-once, MLP row reuse, and attention row-slice kernels.
- No model-specific Pythia-160M code path was required.
- Lossless pack/verify and HF top-k parity pass on the fixed-token smoke prompt.

The main scaling cost is prefill compute, especially MLP and QKV projection:

```text
1024 + 16 prefill: 26606.40 ms
  MLP:             16181.24 ms (~60.8% of prefill)
  attention:        9881.80 ms (~37.1% of prefill)
    QKV projection: 5833.58 ms
    score/context:  1979.05 ms
    output proj:    2043.54 ms
```

Layer-param recall remains tiny relative to compute:

```text
1024 + 16 layer-param recall: 69.61 ms (~0.26% of prefill)
```

Memory scaling is acceptable but near the current `100mb` internal/RSS target at 1024-token prompts:

```text
1024 + 16 RSS: 99.47 MiB
tracked transient: 1.04 MiB
context memory: 73.05 MiB
current transient after run: 0 B
```

So the next evidence-driven target is not layer loading. It is either:

1. keep 160M as the validated scale baseline and proceed toward broader model-family work, or
2. optimize generic MLP/QKV projection kernels for larger GPT-NeoX shapes, or
3. run prefill chunk/window and memory-budget sweeps on 160M to tune default heuristics before LLaMA-family expansion.

## Decision gate

Recommended next slice:

```text
Phase 7.11B — Pythia-160M prefill-chunk / memory-budget sweep
```

Reason: 1024-token Pythia-160M succeeds but lands near 100 MiB RSS. Before adding more kernel complexity, measure whether chunk window and context/RSS behavior can be tuned generically for the larger family shape.

Alternative next slice:

```text
Phase 7.12 — Generic MLP/QKV projection optimization for larger GPT-NeoX shapes
```

Only choose this if token speed is the immediate priority. The 7.11A data shows the likely performance target is MLP/QKV projection, not score/context or layer-param recall.
