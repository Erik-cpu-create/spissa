# Trial: R66 Layer-Specific Top-K Budget Overrides

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: failed

## Hypothesis

In R65, we saw that exact layer resets cannot recover a corrupted hidden state. We hypothesize that rather than attempting to reset the drift via periodic exact layers, we can prevent drift from accumulating in the first place by allocating a higher top-k budget (e.g., `topk=16` instead of `topk=4`) to the early sparse layers (layers 2-6) immediately following the exact prefix layers (layers 0-1).

## Scope

- Mode: experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: hidden-state calibration across layers
- Bottleneck tag: hidden-state calibration

## Setup

Implementation details:

- Added `RLLM_AIP_LAYER_TOPK_OVERRIDES` environment variable to `RamaExperimentalSpeedConfig`.
- Configured overrides as `layer:topk` comma-separated pairs, parsed into a `[u16; 128]` array.
- Overrides bypass default global/edge top-k configurations in `aip_decision_for_projection`.

Commands:

```bash
# Variant 1: Edge Override (First three layers exact edge override)
printf 'good morning\nexit\n' | /usr/bin/time -l env \
  RLLM_AIP_INPUT_TILES=1 \
  RLLM_EXPERIMENTAL_SPEED=1 \
  RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 \
  RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI=100 \
  RLLM_AIP_ATTENTION_LOCALITY_WINDOW=8 \
  RLLM_AIP_ATTENTION_LOCALITY_EXTRA=1 \
  RLLM_AIP_EXACT_PREFILL=1 \
  RLLM_AIP_LAYER_DRIFT_PROBE=1 \
  RLLM_AIP_LAYER_TOPK_OVERRIDES="0:16,1:16,2:16,3:8,4:8,5:8" \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'

# Variant 2: Combined Prefix N=2 + Layer-specific top-k overrides
printf 'good morning\nexit\n' | /usr/bin/time -l env \
  RLLM_AIP_INPUT_TILES=1 \
  RLLM_EXPERIMENTAL_SPEED=1 \
  RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 \
  RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI=100 \
  RLLM_AIP_ATTENTION_LOCALITY_WINDOW=8 \
  RLLM_AIP_ATTENTION_LOCALITY_EXTRA=1 \
  RLLM_AIP_EXACT_PREFILL=1 \
  RLLM_AIP_LAYER_DRIFT_PROBE=1 \
  RLLM_AIP_EXACT_PREFIX_LAYERS=2 \
  RLLM_AIP_LAYER_TOPK_OVERRIDES="2:16,3:16,4:16,5:16,6:16" \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

## Results

| variant | token output | mismatch layers | first mismatch | pre-mismatch L2/cos | max L2/cos | finding |
|---|---|---:|---:|---:|---:|---|
| Overrides Only | `Howngx` | 15/16 | 2 | 1340 / 425 | 15174 / 971 | layer 1 with topk=16 still drifts and triggers mismatch at layer 2 |
| Prefix N=2 + Overrides | `Howose` | 10/16 | 7 | 4146 / 596 | 15224 / 913 | increasing sparse layers 2-6 budget to 16 slightly increases drift compared to default topk=4 |

Representative suffixes:

```text
RLLM_AIP_LAYER_TOPK_OVERRIDES="0:16,1:16,2:16,3:8,4:8,5:8"
Howngx
layer_drift_probe=1 layers=16 mismatch_layers=15 first_mismatch_layer=2 pre_mismatch_max_l2_milli=1340 pre_mismatch_max_cosine_gap_milli=425 max_l2_milli=15174 max_cosine_gap_milli=971

RLLM_AIP_EXACT_PREFIX_LAYERS=2 RLLM_AIP_LAYER_TOPK_OVERRIDES="2:16,3:16,4:16,5:16,6:16"
Howose
layer_drift_probe=1 layers=16 mismatch_layers=10 first_mismatch_layer=7 pre_mismatch_max_l2_milli=4146 pre_mismatch_max_cosine_gap_milli=596 max_l2_milli=15224 max_cosine_gap_milli=913
```

## Analysis

The layer-specific top-k overrides policy failed to recover the target token `How can`.

The empirical results show a surprising and counter-intuitive behavior:
1. **Low top-k values do not guarantee faster drift; sometimes they act as filters:** When we increased the top-k budget from `4` to `16` for layers 2-6, the pre-mismatch L2 drift actually increased from **3722** to **4146** compared to the prefix N=2 base case. This suggests that keeping more low-magnitude activations (which might contain minor noise or errors) can propagate and amplify drift across layers compared to a strict `topk=4` filter which only retains the strongest signals.
2. **First layer drift is extremely high:** Even at layer 0 (index 0) with `topk=16`, the L2 drift reached **1340**. This indicates that the sparse approximations in the MLP gate/up projections are inherently lossy, and a small absolute top-k (even 16) is not enough to preserve numerical alignment in the absence of exact prefix layers.

## Decision

failed

Reason: The layer-specific overrides failed to recover the exact output. In fact, increasing top-k from 4 to 16 marginally accelerated drift.

## Next Experiment

R67 should reconsider the sparse approximation logic itself or try adjusting other structural variables:
- Look into whether the MLP down projection (which is also sparse) is the main contributor to the drift when MLPs are active. We could test making only the down projection exact (`RLLM_AIP_EXACT_LAYER_PROJECTION=down`).
- Explore using a higher top-k for Attention instead of MLPs.
- Investigate if there is a way to dynamically scale the top-k threshold based on activation distribution rather than static values.
