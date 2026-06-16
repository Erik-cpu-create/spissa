# Trial: R67 Global Exact Down-Projection

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: failed

## Hypothesis

In R66, layer-specific top-k overrides failed to prevent drift and actually marginally accelerated it when higher top-k budgets were allocated to sparse early layers. We hypothesize that the sparse MLP Down projection might be a significant contributor to numerical drift. By making the Down projection exact across all layers (`RLLM_AIP_EXACT_LAYER_PROJECTION=down`), we can isolate its impact and see if it delays or prevents the hidden-state corruption.

## Scope

- Mode: experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: hidden-state calibration across layers
- Bottleneck tag: hidden-state calibration

## Setup

Implementation details:
- Modified `exact_layer_projection` in `speed.rs` to apply the projection override globally to all layers when `aip_exact_layer` is `None`. This allows `RLLM_AIP_EXACT_LAYER_PROJECTION=down` to make all Down projections exact.
- Verified parsing and global mapping via unit tests.

Command:

```bash
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
  RLLM_AIP_EXACT_LAYER_PROJECTION=down \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

## Results

| variant | token output | mismatch layers | first mismatch | pre-mismatch L2/cos | max L2/cos | finding |
|---|---|---:|---:|---:|---:|---|
| Global Exact Down | `How mir` | 15/16 | 2 | 1463 / 542 | 18916 / 1014 | Global exact Down projection has no effect on delaying the mismatch |

Representative suffix:

```text
> How mir
[... AIP: policy=speed calls=81 fallbacks=0 max_topk=4 ... layer_drift_probe=1 layers=16 mismatch_layers=15 first_mismatch_layer=2 pre_mismatch_max_l2_milli=1463 pre_mismatch_max_cosine_gap_milli=542 max_l2_milli=18916 max_cosine_gap_milli=1014 ... ]
```

## Analysis

Making the MLP Down projection exact globally failed to recover the target token `How can`. The output was `How mir`, identically matching the fully sparse baseline observed in earlier experiments (like R60/R62).

The exact Down projection did not delay the first top-1 mismatch, which still occurred at layer 2. The pre-mismatch L2 drift at layer 1 was **1463**, slightly worse than the R66 baseline variant (**1340**). This decisively proves that the Down projection is not the sole or primary driver of the hidden-state collapse. The drift introduced by sparse Attention and sparse Gate-Up projections is entirely sufficient to corrupt the hidden state before the exact Down projection can even act on it.

## Decision

failed

Reason: Global exact Down projection does not mitigate numerical drift or improve the token output.

## Next Experiment

Given that isolated adjustments to layer-specific top-k budgets (R66) and the MLP Down projection (R67) do not heal or prevent drift effectively:
- We must evaluate if a global exact projection policy for another projection type, specifically **Attention**, can prevent the drift (`RLLM_AIP_EXACT_LAYER_PROJECTION=attention`).
- Or consider rethinking the `aip_policy` strategy completely to introduce dynamic magnitude-based gating rather than static top-k selection.
