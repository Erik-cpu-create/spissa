# Trial: R62 Layer-2 Projection Attribution

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: success

## Hypothesis

R61 localized the first exact-vs-sparse decode disagreement to layer 2. A
targeted attribution probe should show whether the layer-2 drift is concentrated
in attention, fused gate/up, or down projection outputs.

## Scope

- Mode: experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: layer-2 sparse projection drift
- Bottleneck tag: CPU arithmetic

## Setup

Implementation details:

- Reuses `RLLM_AIP_LAYER_DRIFT_PROBE=1`.
- After the layer drift probe finds `first_mismatch_layer`, the runtime reruns
  exact and sparse shadow passes to that layer only.
- The transformer block exposes debug-only captures for:
  - attention output after the output projection
  - fused `silu(gate) * up` output
  - down projection output
- Captured outputs are compared with L2 milli and cosine-gap milli metrics.

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

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
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'

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
  RLLM_AIP_LM_HEAD_EXACT_EVERY=1 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

## Results

| variant | token output | decode tok/s | layer probe | attribution layer | attention L2/cos | gate-up L2/cos | down L2/cos | finding |
|---|---|---:|---|---:|---:|---:|---:|---|
| sparse decode + attribution | `How mir` | 0.14 | 15/16 mismatch, first layer 2 | 2 | 1046 / 1263 | 1676 / 965 | 1416 / 965 | all three groups drift; attention direction drift is largest |
| exact final LM-head + attribution | `How how` | 0.13 | 15/16 mismatch, first layer 2 | 2 | 1046 / 1263 | 1676 / 965 | 1416 / 965 | attribution is unchanged by final LM-head exactness |

Representative suffix:

```text
layer_attribution_probe=1 attribution_layer=2 attention_l2_milli=1046 attention_cosine_gap_milli=1263 gate_up_l2_milli=1676 gate_up_cosine_gap_milli=965 down_l2_milli=1416 down_cosine_gap_milli=965
```

## Analysis

R62 confirms that layer-2 drift is not isolated to one cheap projection output.
All three captured groups differ materially from exact execution.

The strongest directional signal is attention: `attention_cosine_gap_milli=1263`
is larger than both gate/up and down. The largest absolute L2 signal is fused
gate/up at `1676`, with down close behind at `1416`. That means a retained fix
should start with layer-2 attention exactness or attention widening, but should
expect MLP-side residual drift to remain.

The exact final LM-head control reproduces the same attribution metrics while
changing only the final token from `mir` to `how`. That keeps R61's conclusion
intact: final LM-head exactness can alter the chosen token, but it does not
repair the sparse hidden-state trajectory.

## Decision

success with diagnostic limitation

Reason: R62 adds projection-family attribution for the first mismatch layer and
identifies attention as the largest direction drift inside layer 2. It does not
attempt a retained quality fix.

Paper value:

- positive evidence that first-mismatch attribution is measurable in the runtime
- diagnostic evidence that layer-2 attention has the largest directional drift
- limitation evidence that MLP drift is also present and likely coupled

## Next Experiment

R63 should test a retained policy rather than another observer: make layer-2
attention exact or wider while keeping other sparse settings fixed, then compare
whether token 2 recovers from `mir` toward the exact `can` without collapsing the
speed floor.
