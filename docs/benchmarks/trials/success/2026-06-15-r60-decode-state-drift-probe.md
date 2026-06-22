# Trial: R60 Decode-State Drift Probe

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: success

## Hypothesis

R59 showed that exact prefill recovers the first assistant token under the
Llama3 template, but sparse decode immediately collapses on the continuation.
A two-token probe should separate LM-head drift from transformer hidden-state
drift by forcing exactness at one side of the decode boundary at a time.

## Scope

- Mode: exact-lowram and experimental-speed diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: decode-state approximation drift
- Bottleneck tag: CPU arithmetic

## Setup

Commands:

```bash
printf 'good morning\nexit\n' | target/release/llama-test \
  --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
  --ctx 2048 \
  --max-new-tokens 2 \
  --chat-template llama3 \
  --system-prompt 'You are a concise assistant.'

printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI=100 \
  RLLM_AIP_ATTENTION_LOCALITY_WINDOW=8 \
  RLLM_AIP_ATTENTION_LOCALITY_EXTRA=1 \
  RLLM_AIP_EXACT_PREFILL=1 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'

printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI=100 \
  RLLM_AIP_ATTENTION_LOCALITY_WINDOW=8 \
  RLLM_AIP_ATTENTION_LOCALITY_EXTRA=1 \
  RLLM_AIP_EXACT_PREFILL=1 \
  RLLM_AIP_LM_HEAD_EXACT_EVERY=1 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'

printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI=100 \
  RLLM_AIP_ATTENTION_LOCALITY_WINDOW=8 \
  RLLM_AIP_ATTENTION_LOCALITY_EXTRA=1 \
  RLLM_AIP_EXACT_PREFILL=1 \
  RLLM_AIP_EXACT_EDGE_LAYERS=16 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'

printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI=100 \
  RLLM_AIP_ATTENTION_LOCALITY_WINDOW=8 \
  RLLM_AIP_ATTENTION_LOCALITY_EXTRA=1 \
  RLLM_AIP_EXACT_PREFILL=1 \
  RLLM_AIP_EXACT_EDGE_LAYERS=16 \
  RLLM_AIP_LM_HEAD_EXACT_EVERY=1 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 2 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

Projection-localization commands repeated the same sparse environment with
`RLLM_AIP_EXACT_EDGE_LAYERS=16`, `RLLM_AIP_LM_HEAD_EXACT_EVERY=1`, and one of:

```bash
RLLM_AIP_EXACT_EDGE_PROJECTION=attention
RLLM_AIP_EXACT_EDGE_PROJECTION=gateup
RLLM_AIP_EXACT_EDGE_PROJECTION=down
```

Runtime context:

- build profile: release
- prompt: `good morning`
- chat formatting: `--chat-template llama3`
- system prompt: `You are a concise assistant.`
- decode window: 2 generated tokens

## Results

| variant | exact transformer? | exact LM-head on token 2? | token output | TTFT/prefill | decode tok/s | AIP calls | peak bytes | finding |
|---|---|---|---|---:|---:|---:|---:|---|
| exact reference | yes | yes | `How can` | 22.47s | 0.15 | 0 | 1050689536 | target continuation |
| exact prefill + sparse decode | no | no | `How mir` | 25.70s | 5.35 | 97 | 1050689536 | sparse continuation drift |
| sparse transformer + exact LM-head | no | yes | `How how` | 24.49s | 3.19 | 97 | 1050689536 | hidden-state drift remains under exact LM-head |
| exact transformer + sparse LM-head | yes | no | `How bro` | 24.93s | 0.18 | 1 | 1050689536 | LM-head sparse projection also drifts |
| exact transformer + exact LM-head | yes | yes | `How can` | 25.24s | 0.15 | 1 | 1050689536 | control returns target continuation |
| exact attention + exact LM-head | attention only | yes | `Howreetings` | 25.08s | 1.61 | 33 | 1050689536 | attention exactness alone is insufficient |
| exact gate/up + exact LM-head | gate/up only | yes | `How Mag` | 25.90s | 0.20 | 81 | 1050689536 | gate/up exactness alone is insufficient |
| exact down + exact LM-head | down only | yes | `How how` | 26.20s | 0.66 | 81 | 1050689536 | down exactness alone is insufficient |

## Analysis

R60 confirms that the immediate Llama3 sparse collapse is not a simple
post-sampling bug. Exact prefill makes token 1 match the exact baseline
(`How`), but the sparse decode continuation selects `mir` instead of `can`.

Forcing an exact LM-head over the sparse hidden state changes token 2 from
`mir` to `how`, but still misses the exact reference. That isolates a
transformer hidden-state drift: the sparse decode hidden vector is already on
the wrong side of the decision boundary before LM-head approximation is applied.

The opposite control also fails: when all transformer projections are exact and
only the LM-head remains sparse, token 2 becomes `bro`. That means sparse
LM-head top-k over an otherwise exact hidden vector is independently unsafe for
chat continuation.

The full exact control returns `How can`, proving the environment and template
path are sound. Projection-filtered exactness does not isolate one sufficient
transformer family: exact attention gives `Howreetings`, exact gate/up gives
`How Mag`, and exact down gives `How how`. The drift is coupled across the
transformer sparse path and the sparse LM-head projection, not a single
projection-family bug that can be fixed by one existing exactness flag.

## Decision

success with diagnostic limitation

Reason: R60 successfully localizes the failure class. The next improvement
should not be another cheap LM-head controller. It needs either a hidden-state
agreement metric or a mixed exact/sparse projection design that measures
per-layer contribution before selecting where to spend exact compute.

Paper value:

- positive diagnostic evidence that exact prefill alone only repairs token 1
- negative evidence against LM-head-only fixes for Llama3 chat recovery
- limitation evidence that sparse transformer and sparse LM-head drift are
  coupled under the current R25 input-tiled artifact

## Next Experiment

R61 should add hidden-state or logit-margin instrumentation around the second
decode token. The minimum useful probe is a debug-only path that records exact
versus sparse top token and logit margin after each transformer layer, then
identifies the first layer where the exact next token leaves the sparse
candidate set.
