# Trial: R68 Global Exact Projections (Attention and Gate-Up)

Date: 2026-06-16
Owner: RLLM
Status: reviewed
Folder: failed

## Hypothesis

In R67, making the MLP Down projection globally exact failed to prevent the top-1 hidden-state mismatch at layer 2. We hypothesize that either global exact Attention (`attention`) or global exact Gate-Up (`gateup`) projections could delay or prevent this numerical drift. Running these diagnostics will isolate which specific projection type carries the primary driver of hidden-state collapse.

## Scope

- Mode: experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: hidden-state calibration across layers
- Bottleneck tag: hidden-state calibration

## Setup

Implementation details:
- Used the global exact projection routing capability from R67.
- Ran Trial 1 with `RLLM_AIP_EXACT_LAYER_PROJECTION=attention`.
- Ran Trial 2 with `RLLM_AIP_EXACT_LAYER_PROJECTION=gateup`.

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
  RLLM_AIP_EXACT_LAYER_PROJECTION=attention \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
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
  RLLM_AIP_EXACT_LAYER_PROJECTION=gateup \
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
| Global Exact Attention | `How táº¥n` | 15/16 | 2 | 1269 / 367 | 20326 / 962 | Mismatch still triggers at Layer 2 |
| Global Exact Gate-Up | `Howstood` | 15/16 | 2 | 1393 / 451 | 16997 / 984 | Mismatch still triggers at Layer 2 |
| Global Exact Down (R67) | `How mir` | 15/16 | 2 | 1463 / 542 | 18916 / 1014 | Mismatch still triggers at Layer 2 |

## Analysis

Neither global exact Attention nor global exact Gate-Up projections succeeded in delaying the first top-1 mismatch, which stubbornly occurred at Layer 2 for all cases. The mismatch layers metric remained unchanged at 15 out of 16 layers.

This provides strong empirical evidence that:
1. **Accumulated Drift**: The hidden-state drift is not caused by any single projection type in isolation. Rather, it is the cumulative result of approximation error across all sparse components in a layer (Attention, Gate-Up, and Down projections).
2. **Coupled Errors**: Isolating one component is insufficient to keep the hidden state inside the exact argmax basin at Layer 2, because the remaining sparse components introduce enough error to trigger the mismatch.

## Decision

failed

Reason: Individually forcing Attention or Gate-Up projections to be exact does not prevent or delay hidden-state drift.

## Next Experiment

Since forcing individual projection types to be exact globally is insufficient to prevent the top-1 mismatch at Layer 2, we must look into:
- Supporting combinations of global exact projections (e.g. making all MLP projections exact: Gate-Up + Down, or making Attention + Gate-Up exact).
- Alternatively, we should move away from a static Top-K policy and explore a dynamic magnitude-based gating strategy that adaptively selects top-k columns based on active activation values.
