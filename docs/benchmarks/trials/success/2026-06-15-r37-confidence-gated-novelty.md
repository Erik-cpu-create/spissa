# Trial: R37 Confidence-Gated Phrase Novelty

Date: 2026-06-15
Owner: RLLM
Status: success with quality limitation
Folder: success

## Hypothesis

R36 improved short-fragment diversity but pushed selected tokens farther away
from exact LM-head behavior. R37 tests a confidence gate for phrase novelty: if
the selected sparse candidate is much stronger than the next candidate, do not
override it even when it repeats a recent phrase.

The goal is to preserve the 30-40 tok/s Llama 3.2 1B Instruct band while
recovering some exact-agreement signal and keeping the diversity gain from R36.

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
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
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
  RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | phrase novelty switches/checks | gap skips | max novelty gap milli | selected exact match | raw sparse exact match | exact in sparse top-k | RLLM peak transient | max RSS | peak footprint | transformer time | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R36 no-gap control rerun | 64 | 13.14s | 34.07 | 4.27 | 0.10 | 2 | 15/64 | 7/64 | 0 | N/A | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1791.58ms | 56.83ms |
| R37 gap 100 run 1 | 64 | 12.71s | 32.27 | 4.36 | 0.11 | 2 | 17/64 | 5/64 | 15 | 201 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1882.65ms | 68.94ms |
| R37 gap 100 run 2 | 64 | 13.11s | 32.83 | 4.26 | 0.11 | 2 | 17/64 | 5/64 | 15 | 201 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1853.03ms | 65.09ms |
| R37 gap 100 run 3 | 64 | 13.02s | 31.29 | 4.26 | 0.11 | 2 | 17/64 | 5/64 | 15 | 201 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1946.04ms | 66.57ms |
| R37 gap 100 agreement profiler | 64 | 14.36s | 6.14 | 2.60 | 0.11 | 2 | 17/64 | 5/64 | 15 | 201 | 7/64 | 28/64 | 40/64 | 1050689536 | 1779449856 | 2157448672 | 2045.95ms | 8214.04ms |

Additional sweep:

- gap 100: 30.77 tok/s first-pass sweep, 17/64 unique, 15 gap skips
- gap 250: 31.76 tok/s, 15/64 unique, effectively same switch pattern as R36
- gap 500: 27.81 tok/s in current sweep, below target
- gap 750: 32.70 tok/s, same switch pattern as R36

## Analysis

R37 gap 100 is speed-stable in the validation sweep: 32.27, 32.83, and
31.29 tok/s. Compared with the R36 no-gap control rerun, it reduces novelty
switches from 7/64 to 5/64, records 15 confidence-gate skips, and improves
unique tokens from 15/64 to 17/64 while staying inside the 30-40 tok/s target.

The exact-agreement signal improves slightly versus R36: selected exact match
goes from 6/64 to 7/64, raw sparse exact from 26/64 to 28/64, and exact-in-top-k
from 36/64 to 40/64. This is a modest recovery, not a semantic-quality fix.

The output remains fragmentary and not chat-quality. R37 is useful because it
shows that gating novelty by sparse confidence can improve the tradeoff without
extra model IO.

## Decision

success with quality limitation

Reason: R37 keeps Llama 3.2 1B Instruct inside the 30-40 tok/s target and
improves both diversity and exact-agreement signals versus R36, but the semantic
quality is still insufficient.

Paper value:

- positive speed evidence: three R37 gap 100 profile runs stayed above 30 tok/s
- positive diversity evidence: unique tokens improved from 15/64 to 17/64
- positive control evidence: confidence gate skipped 15 phrase overrides
- limitation evidence: exact agreement improved only slightly, so confidence
  gating alone is not enough

## Next Experiment

R38 should make the novelty gate softer instead of binary:

- retain R37 gap 100 as the control
- use the confidence gap to rank fallback candidates, not only allow/deny
- prefer candidates that are both novel and close to the sparse selected token
- keep the fallback candidate scan to top-4 so LM-head cost stays bounded
