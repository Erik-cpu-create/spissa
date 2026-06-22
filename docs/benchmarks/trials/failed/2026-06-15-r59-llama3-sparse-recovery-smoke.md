# Trial: R59 Llama3 Sparse Recovery Smoke

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: failed

## Hypothesis

After R58 fixed the Llama3 chat-template boundary, sparse AIP quality might be
recoverable with existing low-cost controls: exact prefill, LM-head widening,
or exact edge-attention diagnostics. The expected positive signal is retaining
the exact chat baseline's opening while improving sparse diversity without
falling further below the speed floor.

## Scope

- Mode: exact-lowram and experimental-speed smoke
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: sparse decode-state quality
- Bottleneck tag: CPU arithmetic

## Setup

Commands:

```bash
printf 'good morning\nexit\n' | target/release/llama-test \
  --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
  --ctx 2048 \
  --max-new-tokens 16 \
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
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 16 \
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
    --max-new-tokens 16 \
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
  RLLM_AIP_LM_HEAD_TOPK=8 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 16 \
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
  RLLM_AIP_EXACT_EDGE_LAYERS=1 \
  RLLM_AIP_EXACT_EDGE_PROJECTION=attention \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 16 \
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
  RLLM_AIP_EXACT_EDGE_LAYERS=1 \
  RLLM_AIP_EXACT_EDGE_PROJECTION=attention \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 16 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

Runtime context:

- build profile: release
- relevant env/config: `--chat-template llama3`, concise system prompt, 16-token
  sparse smoke window

## Results

| variant | output sample | TTFT/prefill | decode tok/s | generated | context | peak bytes | repetition | unique |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| exact template reference | `How can I assist you today?` | 22.56s | 0.13 | 8 | 61 | 1050689536 | 0.00 | 8/8 |
| R57 retained preset + template | `Spl mir.swing mir.swing.swing...` | 19.91s | 23.46 | 16 | 69 | 1050689536 | 0.20 | 3/16 |
| exact prefill + R57 preset | `How mir.swing mir.swing.swing...` | 22.48s | 27.15 | 16 | 69 | 1050689536 | 0.20 | 3/16 |
| exact prefill + LM-head top-k 8 | `How.swinglaughter.swing...` | 22.15s | 21.93 | 16 | 69 | 1050689536 | 0.07 | 4/16 |
| exact edge attention diagnostic | `Splalmessen calruaponang...` | 21.34s | 14.11 | 16 | 69 | 1050689536 | 0.00 | 14/16 |
| exact prefill + exact edge attention | `Howuzzalmaimettiuy...` | 22.47s | 15.05 | 16 | 69 | 1050689536 | 0.07 | 14/16 |

## Analysis

R59 rejects the idea that existing low-cost sparse controllers recover Llama3
chat quality after R58. The exact template reference still produces the
coherent assistant-style answer, but the retained R57 sparse preset remains
collapsed and now falls below the 30 tok/s floor for the 16-token chat-template
smoke.

Exact prefill is the most useful diagnostic signal: it changes the first sparse
token from `Spl` to `How`, matching the exact baseline's opening direction.
That means the prompt boundary and initial exact state matter, but sparse decode
state corrupts the continuation immediately after the first generated token.

LM-head widening does not solve the issue. It keeps the `How` start but remains
fragmentary, raises unique tokens only from 3/16 to 4/16, and slows decode to
21.93 tok/s.

Exact edge attention improves cheap diversity counters much more strongly,
reaching 14/16 unique tokens and near-zero repetition, but it falls to roughly
14-15 tok/s and still does not produce semantic chat text. This makes full
edge-attention exactness useful as a diagnostic, not a retained R59 path.

## Decision

failed

Reason: no tested recovery path preserves the sparse speed floor and restores
chat quality. Exact prefill partially recovers the first token but not the
decode continuation; exact edge attention recovers diversity but is too slow
and still semantically weak.

Paper value:

- use as negative evidence that chat-template correctness alone does not make
  sparse AIP chat-ready
- use as limitation evidence that cheap LM-head controllers cannot repair
  decode-state drift after the first assistant token

## Next Experiment

R60 should measure and localize decode-state drift after the exact first token.
Start with a profiler that compares exact-prefill plus first-token sparse decode
against exact decode hidden-state/LM-head agreement before adding another
controller. The target is to identify which transformer projection introduces
the immediate `How -> mir.swing` collapse under the Llama3 template.
