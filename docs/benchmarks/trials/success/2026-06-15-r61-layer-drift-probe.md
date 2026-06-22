# Trial: R61 Layer Drift Probe

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: success

## Hypothesis

R60 showed that Llama3 sparse decode drift is already present before final
LM-head selection. A debug-only per-layer shadow probe should identify where
exact and sparse decode states first disagree, without changing the normal
generation path unless explicitly enabled.

## Scope

- Mode: experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: sparse transformer hidden-state drift
- Bottleneck tag: CPU arithmetic

## Setup

Implementation gates:

- `RLLM_AIP_LAYER_DRIFT_PROBE=1`
- decode-only probe: active only when `emit_logits`, experimental speed, and
  decode-step conditions are all true
- shadow state: cloned KV cache, local sparse column cache, local attention
  locality cache, local memory budget
- reported metrics: sampled probe passes, compared layers, top-1 mismatch
  layers, first mismatch layer, max hidden L2 milli, max cosine-gap milli, and
  max exact margin milli

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
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
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
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

Runtime context:

- build profile: release
- prompt: `good morning`
- chat formatting: `--chat-template llama3`
- system prompt: `You are a concise assistant.`
- decode window: 2 generated tokens

## Results

| variant | token output | decode tok/s | AIP calls | layer probe | mismatch layers | first mismatch layer | max L2 milli | max cosine-gap milli | max exact margin milli | finding |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exact prefill + sparse decode + drift probe | `How mir` | 0.14 | 97 | 1 pass, 16 layers | 15/16 | 2 | 18415 | 1013 | 5268 | transformer drift begins immediately after layer 1 |
| exact prefill + sparse decode + exact LM-head + drift probe | `How how` | 0.16 | 97 | 1 pass, 16 layers | 15/16 | 2 | 18415 | 1013 | 5268 | final exact LM-head changes token 2 but does not remove hidden drift |

Representative suffix:

```text
layer_drift_probe=1 layers=16 mismatch_layers=15 first_mismatch_layer=2 max_l2_milli=18415 max_cosine_gap_milli=1013 max_exact_margin_milli=5268
```

## Analysis

R61 succeeds as instrumentation. The normal sparse output remains `How mir`,
matching R60's sparse continuation failure, but the new probe now identifies the
first exact-vs-sparse layer top-1 disagreement at layer 2.

The exact-LM-head control still outputs `How how`, and the layer probe reports
the same hidden-state drift numbers. This confirms that final LM-head exactness
is downstream of the main failure; the sparse hidden state has already diverged
before final token selection.

The margin signal is also useful: max exact margin reaches `5268` milli, so the
observed layer-level top-1 disagreements are not only near-tie sampling noise.
The max cosine-gap milli value exceeds `1000`, which means at least one sparse
hidden vector is close to orthogonal or slightly opposite relative to its exact
counterpart.

## Decision

success with diagnostic limitation

Reason: R61 adds a debug-only probe that localizes decode drift to layer 2 for
the R25 Llama3 sparse path. It intentionally does not fix sparse quality.

Paper value:

- positive evidence that hidden-state drift begins very early in decode
- direct aggregate metric for exact-vs-sparse layer top-1 disagreement
- negative evidence against final LM-head-only recovery

## Next Experiment

R62 should focus on the first drift window instead of global LM-head controls.
The most direct next probe is a layer-2 projection attribution path: compare
exact-vs-sparse outputs around attention, gate/up, and down inside layer 2, then
test a retained exactness policy for the smallest projection group that prevents
the layer-2 top-1 flip.
