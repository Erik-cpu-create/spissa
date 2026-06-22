# Trial: R64 Exact Prefix Layers Policy

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: failed

## Hypothesis

R63 showed that setting only layer 2 to be exact was not enough to recover the target token, because hidden drift had already begun accumulating before the first mismatch layer.
We hypothesize that a "prefix-style" exact policy, specifically making the first N layers fully exact (e.g., layers 1 and 2), will better suppress early hidden drift and delay the first top-1 mismatch. Furthermore, we expect to measure non-trivial `pre_mismatch` drift, proving that hidden states diverge long before the top-1 prediction flips.

## Scope

- Mode: experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: hidden-state calibration across layers
- Bottleneck tag: hidden-state calibration

## Setup

Implementation details:

- Added `RLLM_AIP_EXACT_PREFIX_LAYERS` environment variable to `RamaExperimentalSpeedConfig`.
- Added `exact_prefix_projection` to force all projections in layers `< N` to be exact.
- Extended `RamaLayerDriftProbeStats` to track `pre_mismatch_max_l2_milli` and `pre_mismatch_max_cosine_gap_milli`. This captures the max distance of hidden vectors *before* the first layer that suffers a top-1 mismatch.

Commands:

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
  RLLM_AIP_EXACT_PREFIX_LAYERS=2 \
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
| prefix N=2 exact | `Howose` | 10/16 | 7 | 3722 / 564 | 15219 / 927 | exact prefix delays mismatch to layer 7 but hidden drift still rapidly accumulates |

Representative suffix:

```text
RLLM_AIP_EXACT_PREFIX_LAYERS=2
Howose
layer_drift_probe=1 layers=16 mismatch_layers=10 first_mismatch_layer=7 pre_mismatch_max_l2_milli=3722 pre_mismatch_max_cosine_gap_milli=564 max_l2_milli=15219 max_cosine_gap_milli=927 max_exact_margin_milli=5268 layer_attribution_probe=1 attribution_layer=7 attention_l2_milli=1821 attention_cosine_gap_milli=979 gate_up_l2_milli=3553 gate_up_cosine_gap_milli=862 down_l2_milli=2815 down_cosine_gap_milli=855
```

## Analysis

The results confirm the hypothesis that early exact layers push back the top-1 flip boundary. In the sparse baseline (and R63), the first mismatch occurred at layer 2. By making layers 1 and 2 exact (`EXACT_PREFIX_LAYERS=2`), the first mismatch is successfully delayed until layer 7.

However, the token output shifted from `How mir` to `Howose` rather than the exact target `How can`.
The newly implemented pre-mismatch metrics (`pre_mismatch_max_l2_milli=3722`, `pre_mismatch_max_cosine_gap_milli=564`) provide the missing link: even though layers 3-6 maintain the same top-1 prediction as the exact reference, their hidden state vectors are silently drifting. By the time layer 7 executes, the accumulated error is so large that the sparse layer's output diverges in its top-1 prediction. 

This confirms that waiting for a top-1 mismatch is too late to apply a corrective exact layer.

## Decision

failed

Reason: Although the prefix policy delays the first mismatch and the new metrics effectively track pre-mismatch drift, the exact target output is not recovered. Hidden drift accumulates silently under matching top-1 predictions.

Paper value:

- Strong quantitative evidence that top-1 probing masks severe hidden-state drift.
- Introduction of `pre_mismatch` drift tracking, providing a direct measurement of "silent" drift.
- Demonstrated that early exactness (prefix policy) delays semantic degradation but does not cure the underlying accumulation of numerical error in later sparse layers.

## Next Experiment

R65 should likely explore a more systemic drift mitigation strategy. Since error accumulates steadily, a periodic exactness policy (e.g. making every Nth layer exact, or applying exactness when L2 drift exceeds a threshold) might be required, or we must investigate the root cause of the rapid error inflation in the sparse MLPs.
