# Trial: R36 Phrase Novelty Controller

Date: 2026-06-15
Owner: RLLM
Status: success with exact-agreement limitation
Folder: success

## Hypothesis

R35 improved adjacent-token control and kept Llama 3.2 1B Instruct inside the
30-40 tok/s target, but output still repeated short fragments. R36 tests a
phrase-level novelty controller that only uses recent generated token IDs and
the already-computed sparse LM-head logits.

The controller keeps a small session-local output window. If the selected token
would complete a repeated 2-4 token phrase inside that window, it picks the next
highest sparse candidate that does not repeat the phrase. This should improve
short phrase diversity without exact LM-head rescoring or transformer top-k
widening.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Input-tile gate: `RLLM_AIP_INPUT_TILES=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Base transformer top-k: `RLLM_AIP_TOPK=4`
- Repeat guard: `RLLM_AIP_REPEAT_RUN_LIMIT=2`
- Adaptive margin gate: `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1`
- Best R36 slice: `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75`,
  `RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin rllm --bin llama-test
```

Production profile benchmark:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
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
  RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | repeat-margin switches/checks | phrase novelty switches/checks | selected exact match | raw sparse exact match | exact in sparse top-k | RLLM peak transient | max RSS | peak footprint | transformer time | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R35 margin 50 control rerun | 64 | 12.96s | 33.00 | 4.31 | 0.17 | 2 | 11/64 | 18/29 | N/A | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1841.15ms | 67.06ms |
| R36 margin 50 window 4 run 1 | 64 | 14.40s | 30.76 | 3.89 | 0.11 | 2 | 15/64 | 12/21 | 7/64 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1980.62ms | 66.76ms |
| R36 margin 50 window 4 run 2 | 64 | 13.63s | 32.10 | 4.10 | 0.11 | 2 | 15/64 | 12/21 | 7/64 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1897.26ms | 64.38ms |
| R36 margin 50 window 4 run 3 | 64 | 12.89s | 31.92 | 4.31 | 0.11 | 2 | 15/64 | 12/21 | 7/64 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1909.08ms | 63.80ms |
| R36 margin 75 window 4 run 1 | 64 | 11.99s | 36.14 | 4.66 | 0.10 | 2 | 15/64 | 13/21 | 7/64 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1690.26ms | 52.46ms |
| R36 margin 75 window 4 run 2 | 64 | 12.79s | 49.32 | 4.55 | 0.10 | 2 | 15/64 | 13/21 | 7/64 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1225.05ms | 51.74ms |
| R36 margin 75 window 4 run 3 | 64 | 12.23s | 36.91 | 4.59 | 0.10 | 2 | 15/64 | 13/21 | 7/64 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1653.84ms | 52.29ms |
| R36 margin 75 window 4 agreement profiler | 64 | 13.78s | 6.76 | 2.77 | 0.10 | 2 | 15/64 | 13/21 | 7/64 | 6/64 | 26/64 | 36/64 | 1050689536 | 1715077120 | 2157137040 | 1692.78ms | 7626.05ms |

Additional sweep notes:

- `RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=16` reached 35.22 tok/s in one pass and
  unique 25/64, but repeated validation dropped to 24-25 tok/s.
- `RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=6`, `8`, `10`, and `12` improved diversity
  but missed the speed target in the validation sweep.
- The first implementation used full-vocab sort for fallback candidates. That
  was replaced with a one-pass small top-k selector, reducing LM-head time for
  window 16 from roughly 145ms to roughly 70ms, though that slice still missed
  the speed target due token-path transformer time.

## Analysis

R36 margin 75 with novelty window 4 is the best speed-safe slice. It preserves
the 30+ tok/s target with a measured floor of 36.14 tok/s across the three
profile runs, improves repetition ratio from R35's 0.17 to 0.10, and improves
unique tokens from 11/64 to 15/64.

The exact-agreement profiler is worse than R35. Selected exact match falls from
13/64 in R35 to 6/64 in R36, raw sparse exact falls from 42/64 to 26/64, and
exact-in-top-k falls from 50/64 to 36/64. The novelty controller improves
surface diversity but pushes token choices farther away from exact LM-head
behavior.

The output is still not chat-quality. R36 is a useful speed-safe phrase-collapse
checkpoint, not the final semantic fix.

## Decision

success with exact-agreement limitation

Reason: R36 margin 75 + novelty window 4 preserves the target speed for Llama
3.2 1B Instruct and improves short-fragment diversity versus R35, but it
regresses exact LM-head agreement.

Paper value:

- positive speed evidence: three R36 margin 75/window 4 profile runs stayed
  above 30 tok/s
- positive diversity evidence: repetition ratio improved from 0.17 to 0.10 and
  unique tokens improved from 11/64 to 15/64
- limitation evidence: exact agreement regressed, proving phrase novelty alone
  is not enough for semantic quality

## Next Experiment

R37 should combine R35 and R36 signals instead of treating novelty as a hard
override:

- keep R36 margin 75/window 4 as the speed-safe diversity control
- only apply novelty when sparse top-1/top-2 gap is small
- do not apply novelty when the candidate also appears in the sparse exact
  agreement-friendly path
- keep exact agreement profiling diagnostic-only, not in production hot path
