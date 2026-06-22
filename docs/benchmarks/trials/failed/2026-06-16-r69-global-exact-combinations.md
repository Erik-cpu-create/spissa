# Trial: R69 Global Exact Projection Combinations

Date: 2026-06-16
Owner: RLLM
Status: reviewed
Folder: failed

## Hypothesis

In R68, making individual projection types (Attention, Gate-Up, or Down) globally exact failed to prevent the top-1 hidden-state mismatch at Layer 2. We hypothesize that combining multiple projection types in exact mode (specifically, making the entire MLP block exact with `mlp`, or making Attention and Gate-Up exact with `attention-gate-up`) will delay or prevent the hidden-state numerical drift.

## Scope

- Mode: experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: hidden-state calibration across layers
- Bottleneck tag: hidden-state calibration

## Setup

Implementation details:
- Extended `RamaAipProjectionKind` with new variants: `Mlp`, `AttentionGateUp`, `AttentionDown`, and `All`.
- Added a `matches()` method on `RamaAipProjectionKind` and updated `exact_layer_projection` and `exact_edge_projection` to support combination exact matching.
- Ran Trial 1 with `RLLM_AIP_EXACT_LAYER_PROJECTION=mlp` (entire MLP exact, Attention sparse).
- Ran Trial 2 with `RLLM_AIP_EXACT_LAYER_PROJECTION=attention-gate-up` (Attention and Gate-Up exact, Down sparse).
- Ran Control Trial with `RLLM_AIP_EXACT_LAYER_PROJECTION=all` (entire transformer exact, LM-head sparse).

Command for Trial 1:
```bash
printf 'good morning\nexit\n' | env \
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
  RLLM_AIP_EXACT_LAYER_PROJECTION=mlp \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

Command for Trial 2:
```bash
printf 'good morning\nexit\n' | env \
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
  RLLM_AIP_EXACT_LAYER_PROJECTION=attention-gate-up \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

Command for Control Trial:
```bash
printf 'good morning\nexit\n' | env \
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
  RLLM_AIP_EXACT_LAYER_PROJECTION=all \
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
| Global Exact MLP | `How115` | 13/16 | 4 | 3132 / 586 | 20812 / 943 | Mismatch delayed from Layer 2 to Layer 4 |
| Global Exact Attn+Gate-Up | `Howlon` | 15/16 | 2 | 1097 / 257 | 16955 / 832 | Mismatch still triggers at Layer 2 |
| Control: Global Exact All | `How bro` | 0/16 | 0 | 0 / 0 | 0 / 0 | Sanity check: 0 mismatch, transformer fully exact |

## Analysis

1. **MLP Exactness is Highly Effective**: Making the entire MLP block exact (`gateup` and `down`) delayed the first mismatch from Layer 2 to Layer 4. This is the first time we have delayed the mismatch beyond Layer 2 without making prefix layers exact. This indicates that the MLP projections (which run twice per layer and have high activation variance) are a primary contributor to drift accumulation.
2. **Attention+Gate-Up is Insufficient**: When only MLP Down is left sparse, it still triggers a mismatch immediately at Layer 2. This shows that the MLP Down projection approximation introduces enough noise to trigger mismatch on its own if not forced exact.
3. **Sparse LM-Head in Control**: The control trial (`all`) achieved 0 mismatch layers within the transformer, returning `How bro` (due to the sparse LM-head repeat/novelty constraints). This confirms the validity of the global projection framework.

## Decision

failed

Reason: Although global exact MLP delayed the mismatch to Layer 4, it did not completely prevent the top-1 mismatch.

## Next Experiment

Given that making the entire MLP block exact delayed the first mismatch to Layer 4:
- We could investigate coupling **Exact Prefix Layers (N=2 or N=4)** with **Global Exact MLP**. This might push the mismatch much further downstream or eliminate it entirely.
- We should also look into how to recover the target token `How can` by relaxing the LM-head novelty penalty, or testing if `all` exact plus exact LM-head gives `How can`.
