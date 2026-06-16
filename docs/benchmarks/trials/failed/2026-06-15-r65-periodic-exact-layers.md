# Trial: R65 Periodic Exact Layers Policy

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: failed

## Hypothesis

In R64, we observed that early exact layers (prefix) delay the top-1 mismatch, but hidden-state drift still accumulates silently before the mismatch occurs. We hypothesize that interspersing exact layers periodically (e.g., every $K$ layers) will act as a "numerical anchor" to reset or suppress the accumulated drift, further delaying or preventing the top-1 mismatch.

## Scope

- Mode: experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: hidden-state calibration across layers
- Bottleneck tag: hidden-state calibration

## Setup

Implementation details:

- Added `RLLM_AIP_EXACT_PERIODIC_LAYERS` environment variable to `RamaExperimentalSpeedConfig`.
- If `aip_exact_periodic_layers = Some(K)`, layers where `layer_index % K == 0` are evaluated with fully exact projections.

Commands:

```bash
# Variant 1: K=4 (Periodic Reset every 4 layers)
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
  RLLM_AIP_EXACT_PERIODIC_LAYERS=4 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'

# Variant 2: K=2 (Periodic Reset every 2 layers)
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
  RLLM_AIP_EXACT_PERIODIC_LAYERS=2 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'

# Variant 3: Combined (Prefix N=2, Periodic K=2)
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
  RLLM_AIP_EXACT_PERIODIC_LAYERS=2 \
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
| Periodic K=4 | `How gl` | 15/16 | 2 | 0 / 0 | 15785 / 899 | fails immediately at layer 2 since layer 1 remains sparse |
| Periodic K=2 | `How accident` | 13/16 | 2 | 0 / 0 | 17754 / 911 | exact layer at index 2 (layer 3) cannot recover from layer 1 mismatch |
| Prefix N=2 + Periodic K=2 | `How ga` | 10/16 | 7 | 3281 / 387 | 17642 / 874 | slightly reduces pre-mismatch drift but first mismatch is still stuck at layer 7 |

Representative suffixes:

```text
RLLM_AIP_EXACT_PERIODIC_LAYERS=4
How gl
layer_drift_probe=1 layers=16 mismatch_layers=15 first_mismatch_layer=2 max_l2_milli=15785 max_cosine_gap_milli=899 max_exact_margin_milli=5268

RLLM_AIP_EXACT_PERIODIC_LAYERS=2
How accident
layer_drift_probe=1 layers=16 mismatch_layers=13 first_mismatch_layer=2 max_l2_milli=17754 max_cosine_gap_milli=911 max_exact_margin_milli=5268

RLLM_AIP_EXACT_PREFIX_LAYERS=2 RLLM_AIP_EXACT_PERIODIC_LAYERS=2
How ga
layer_drift_probe=1 layers=16 mismatch_layers=10 first_mismatch_layer=7 pre_mismatch_max_l2_milli=3281 pre_mismatch_max_cosine_gap_milli=387 max_l2_milli=17642 max_cosine_gap_milli=874
```

## Analysis

The periodic exactness policy failed to recover the target token `How can`.
Looking closely at the three variants reveals critical numerical behaviors:

1. **Exact layers do not correct corrupted hidden states:** In `K=2`, layer 2 is exact, layer 1 is sparse. A top-1 mismatch occurs immediately at layer 2 (index 1) because layer 1 is sparse. When layer 2 (index 2, exact) runs, it does not "pull" the drifted vector back to the reference manifold; it simply calculates exact transformations on a drifted vector, propagating or magnifying the error.
2. **Periodic exactness reduces drift magnitude slightly:** In the combined prefix (N=2) + periodic (K=2) run, the pre-mismatch drift before layer 7 dropped to L2=3281 (from 3722 in N=2 prefix-only). This proves periodic layers do slow down drift accumulation slightly. However, they do not prevent the eventual divergence at layer 7.

This suggests that interspersing fully exact layers is insufficient if the intermediate sparse layers are still causing rapid drift. 

## Decision

failed

Reason: The periodic exact layers did not recover the exact target output. Interspersing exact layers slows down drift accumulation marginally but cannot recover once divergence begins.

## Next Experiment

R66 should look into the root cause of the rapid drift in intermediate sparse layers. Since even frequent exact resets (K=2) fail to stop the degradation, we may need to:
- Investigate if the MLP Gate-Up top-k budget (currently `topk=4`) is too low for early layers.
- Check if specific layers (e.g., layers 3-6) require a larger top-k budget rather than a hard binary exact/sparse policy.
- Test a policy that dynamically scales the top-k budget instead of relying on exact overrides.
