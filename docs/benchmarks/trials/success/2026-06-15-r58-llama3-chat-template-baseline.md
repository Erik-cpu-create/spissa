# Trial: R58 Llama3 Chat Template Baseline

Date: 2026-06-15
Owner: RLLM
Status: reviewed
Folder: success

## Hypothesis

Llama 3.2 1B Instruct needs its chat template before runtime quality can be
judged. Raw user text is not a valid instruction prompt boundary, so exact mode
must be tested with Llama3 headers, EOT tokens, and an assistant generation
prompt before sparse AIP quality is blamed.

## Scope

- Mode: exact-lowram and experimental-speed smoke
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: single CPU, low RAM
- Expected bottleneck: tokenizer/chat-template boundary
- Bottleneck tag: tokenizer

## Setup

Commands:

```bash
printf 'good morning\nexit\n' | target/release/llama-test \
  --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
  --ctx 2048 \
  --max-new-tokens 8

printf 'good morning\nexit\n' | target/release/llama-test \
  --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
  --ctx 2048 \
  --max-new-tokens 8 \
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
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 16 \
    --chat-template llama3 \
    --system-prompt 'You are a concise assistant.'
```

## Results

| variant | prompt | output sample | TTFT/prefill | decode tok/s | total tokens | context | peak bytes | repetition | unique |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exact raw | `good morning` | `. I'm glad you're feeling well` | 13.58s | 0.15 | 8 | 10 | 1050689536 | 0.00 | 8/8 |
| exact llama3 template | `good morning` | `How can I assist you today?` | 22.20s | 0.16 | 8 | 61 | 1050689536 | 0.00 | 8/8 |
| R57 AIP + llama3 template | `good morning` | `Spl mir.swing mir.swing.swing...` | 21.08s | 26.65 | 16 | 69 | 1050689536 | 0.20 | 3/16 |

## Analysis

The exact/raw result proves the old prompt boundary was not a reliable chat
baseline. With the Llama3 chat template, exact mode produces a coherent
assistant-style first response. R58 therefore fixes the evaluation boundary for
Llama 3.2 1B Instruct without changing weights, quantizing, or copying another
runtime.

The experimental-speed smoke remains not chat-ready after the template. It
still collapses into sparse-token artifacts and, for this 16-token smoke, falls
below the 30 tok/s floor. That points R59 at sparse quality recovery under the
now-correct chat template, not at further template work.

## Decision

success with speed-mode limitation

Reason: exact Llama3 template output is coherent enough to become the new
quality baseline, but the sparse AIP path still fails chat quality and needs a
separate recovery stage.

Paper value:

- positive evidence that chat-template correctness is a first-order runtime
  evaluation variable
- limitation evidence that sparse speed shortcuts must be recalibrated after
  prompt formatting is fixed

## Next Experiment

R59 should run sparse quality recovery against `--chat-template llama3`, using
the exact template output as the baseline and spending extra compute only when
the sparse path shows repetition or low-confidence artifacts.
