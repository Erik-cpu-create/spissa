# Trial: R39 Selected-Token Retention

Date: 2026-06-15
Owner: RLLM
Status: inconclusive
Folder: inconclusive

## Hypothesis

R38 showed that reordering phrase-novelty fallback candidates does not improve
exact agreement. R39 tests the opposite pressure: when sparse top-1 repeats a
recent phrase, keep that selected token unless the best non-repeating fallback
is close enough by sparse-logit gap.

The goal is to preserve the 30-40 tok/s Llama 3.2 1B Instruct band while
recovering selected-token agreement versus R37.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Input-tile gate: `RLLM_AIP_INPUT_TILES=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Base transformer top-k: `RLLM_AIP_TOPK=4`
- Repeat guard: `RLLM_AIP_REPEAT_RUN_LIMIT=2`
- Adaptive margin: `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75`,
  `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1`
- Phrase novelty: `RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4`
- Confidence gate: `RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100`
- New retention control:
  `RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Profile benchmark:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI=<threshold> \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

Agreement profiler:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_RETENTION_MILLI=100 \
  RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | phrase novelty switches/checks | gap skips | retentions | selected exact match | raw sparse exact match | exact in sparse top-k | RLLM peak transient | max RSS | peak footprint | transformer time | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R37 control current sweep | 64 | 12.84s | 26.82 | 4.21 | 0.11 | 2 | 17/64 | 5/64 | 15 | N/A | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2272.75ms | 75.38ms |
| R39 retention 50 sweep | 64 | 13.40s | 33.75 | 4.19 | 0.14 | 2 | 14/64 | 8/64 | 19 | 4 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1791.83ms | 74.11ms |
| R39 retention 100 sweep | 64 | 13.00s | 38.77 | 4.38 | 0.13 | 2 | 15/64 | 5/64 | 15 | 1 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1561.36ms | 62.45ms |
| R39 retention 150 sweep | 64 | 13.51s | 35.32 | 4.18 | 0.13 | 2 | 15/64 | 5/64 | 15 | 1 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1712.25ms | 70.12ms |
| R39 retention 100 run 1 | 64 | 12.58s | 30.26 | 4.36 | 0.13 | 2 | 15/64 | 5/64 | 15 | 1 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2017.54ms | 63.27ms |
| R39 retention 100 run 2 | 64 | 13.45s | 47.70 | 4.33 | 0.13 | 2 | 15/64 | 5/64 | 15 | 1 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1257.22ms | 62.64ms |
| R39 retention 100 run 3 | 64 | 13.15s | 29.94 | 4.20 | 0.13 | 2 | 15/64 | 5/64 | 15 | 1 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2029.90ms | 73.17ms |
| R39 retention 100 agreement profiler | 64 | 16.18s | 5.27 | 2.28 | 0.13 | 2 | 15/64 | 5/64 | 15 | 1 | 8/64 | 28/64 | 41/64 | 1050689536 | 1785888768 | 2158350728 | 2151.15ms | 9800.05ms |
| R39 retention 100 cooldown 1 | 64 | 13.58s | 28.43 | 4.05 | 0.13 | 2 | 15/64 | 5/64 | 15 | 1 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2137.08ms | 77.24ms |
| R39 retention 100 cooldown 2 | 64 | 13.35s | 28.08 | 4.11 | 0.13 | 2 | 15/64 | 5/64 | 15 | 1 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2163.89ms | 78.28ms |
| R37 chat no-profile current | 64 | 12.77s | 27.77 | 4.26 | 0.11 | 2 | 17/64 | 5/64 | 15 | N/A | N/A | N/A | N/A | 1050689536 | N/A | N/A | N/A | N/A |
| R39 retention 100 chat no-profile | 64 | 13.59s | 46.59 | 4.28 | 0.13 | 2 | 15/64 | 5/64 | 15 | 1 | N/A | N/A | N/A | 1050689536 | N/A | N/A | N/A | N/A |

## Analysis

R39 produces the intended retention signal. The selected-token retention counter
fires once on the benchmark prompt, and the agreement profiler improves selected
exact match from the R37 reference value of 7/64 to 8/64. Exact-in-top-k also
improves from 40/64 to 41/64. Raw sparse exact stays flat at 28/64.

The tradeoff is diversity. Retention 100 and 150 reduce unique tokens from the
R37 reference 17/64 to 15/64 and increase repetition ratio from 0.11 to 0.13.
Retention 50 is worse for this prompt: 14/64 unique tokens with more novelty
switches.

Speed is not yet trustworthy enough for a success decision. Retention 100
records profile runs at 30.26, 47.70, and 29.94 tok/s, then cooldown profile
runs at 28.43 and 28.08 tok/s. The current R37 control also measures below the
historical R37 run at 26.82 tok/s, which points to machine/load variance. In
chat no-profile mode, R39 retention 100 reaches 46.59 tok/s while the adjacent
R37 control reaches 27.77 tok/s, so this trial cannot cleanly attribute speed
movement to the controller.

## Decision

inconclusive

Reason: R39 selected-token retention gives a small exact-agreement improvement,
but it reduces diversity and does not prove stable 30-40 tok/s behavior under
the current benchmark conditions.

Paper value:

- positive signal: selected exact match improves from 7/64 to 8/64 with no
  additional model IO
- negative signal: diversity regresses from 17/64 to 15/64 unique tokens
- benchmark caveat: current speed variance is large enough that future claims
  need an automated repeated-run harness before promotion to success

## Next Experiment

R40 should be a benchmark harness rather than another token heuristic:

- run control and candidate in alternating order
- collect at least three profile runs and one no-profile chat run per variant
- reject windows where the control is outside the accepted historical band
- emit a markdown-ready result table for the benchmark folders
