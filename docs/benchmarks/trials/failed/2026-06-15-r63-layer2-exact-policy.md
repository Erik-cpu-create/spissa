# Trial: R63 Layer-2 Exact Policy

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: failed

## Hypothesis

R62 showed the first sparse-vs-exact top-1 disagreement at layer 2 and the
largest directional attribution inside layer-2 attention. Making layer-2
attention exact, or the whole layer exact, should recover the second token from
the sparse `mir` toward the exact `can` while keeping the rest of the sparse
policy intact.

## Scope

- Mode: experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: layer-2 sparse hidden-state drift
- Bottleneck tag: hidden-state calibration

## Setup

Implementation details:

- Adds `RLLM_AIP_EXACT_LAYER` as a one-based layer override.
- Adds `RLLM_AIP_EXACT_LAYER_PROJECTION` with the same projection names used by
  `RLLM_AIP_EXACT_EDGE_PROJECTION`: `attention`, `gateup`, and `down`.
- The override is applied before exact-edge and speed/quality policy routing in
  `RamaExperimentalSpeedConfig::aip_decision_for_projection`.
- `RLLM_AIP_EXACT_LAYER=2` without a projection filter makes all sparse
  projections in layer 2 exact.

Commands:

```bash
cargo test -p rllm-runtime exact_layer -- --nocapture
cargo test -p rllm-runtime aip_exact -- --nocapture
cargo test -p rllm-runtime speed_policy -- --nocapture
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
  RLLM_AIP_EXACT_LAYER=2 \
  RLLM_AIP_EXACT_LAYER_PROJECTION=attention \
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
  RLLM_AIP_EXACT_LAYER=2 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

## Results

| variant | token output | decode tok/s | AIP calls | layer probe | attribution layer | attention L2/cos | gate-up L2/cos | down L2/cos | finding |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| layer-2 attention exact | `How.swing` | 0.14 | 93 | 15/16 mismatch, first layer 2 | 2 | 1335 / 885 | 1690 / 996 | 1427 / 1030 | does not recover token 2; output regresses from sparse baseline |
| layer-2 full exact | `How mir` | 0.14 | 91 | 12/16 mismatch, first layer 2 | 2 | 1335 / 885 | 3453 / 837 | 2759 / 768 | reduces later mismatch count but keeps the same bad token as sparse baseline |

Representative suffixes:

```text
RLLM_AIP_EXACT_LAYER=2 RLLM_AIP_EXACT_LAYER_PROJECTION=attention
How.swing
layer_drift_probe=1 layers=16 mismatch_layers=15 first_mismatch_layer=2 max_l2_milli=16324 max_cosine_gap_milli=993 max_exact_margin_milli=5268 layer_attribution_probe=1 attribution_layer=2 attention_l2_milli=1335 attention_cosine_gap_milli=885 gate_up_l2_milli=1690 gate_up_cosine_gap_milli=996 down_l2_milli=1427 down_cosine_gap_milli=1030

RLLM_AIP_EXACT_LAYER=2
How mir
layer_drift_probe=1 layers=16 mismatch_layers=12 first_mismatch_layer=2 max_l2_milli=21505 max_cosine_gap_milli=1015 max_exact_margin_milli=5268 layer_attribution_probe=1 attribution_layer=2 attention_l2_milli=1335 attention_cosine_gap_milli=885 gate_up_l2_milli=3453 gate_up_cosine_gap_milli=837 down_l2_milli=2759 down_cosine_gap_milli=768
```

## Analysis

R63 rejects the simple reading of R62. Layer-2 attention has a large directional
drift, but layer-2-only exactness is not enough to recover the decode trajectory.

The important clarification is that `first_mismatch_layer=2` is based on the
per-layer top-1 token probe, not on exact hidden equality. Layer 1 can already
carry hidden-state drift while still producing the same probed top-1 token. By
the time layer 2 runs, exact layer-2 math receives a different input vector, so
its output can still differ from the fully exact shadow pass.

Full exact layer 2 is still useful diagnostic evidence: mismatch layers fall
from 15/16 to 12/16, but the emitted two-token output remains `How mir`. That
means the next retained fix should target the drift before or across the layer-2
boundary, not only one projection family inside layer 2.

## Decision

failed

Reason: the retained exact-layer policy is implemented and testable, but the
measured layer-2 attention and full-layer exact variants do not recover the
exact reference token `can`.

Paper value:

- useful negative evidence that top-1 first-mismatch attribution is not the same
  as first hidden-state drift
- practical policy tooling for future exact-layer ablations
- diagnostic support for prefix or hidden-distance calibration before another
  projection-family trial

## Next Experiment

R64 should probe hidden-vector drift before the first top-1 mismatch and test a
prefix-style retained policy, for example exact layer 1 plus layer 2 or exact
first two layers, before spending more time on layer-2-only projection filters.
