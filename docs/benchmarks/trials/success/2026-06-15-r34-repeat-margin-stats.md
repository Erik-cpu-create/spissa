# Trial: R34 Repeat-Margin Controller Stats

Date: 2026-06-15
Owner: RLLM
Status: success with diagnostic limitation
Folder: success

## Hypothesis

R33 showed that the repeat-margin controller can remove adjacent repeats while
staying inside the 30-40 tok/s target, but it also moves selected tokens away
from exact LM-head behavior. R34 adds cheap controller stats so future adaptive
experiments can measure how often the controller changes token selection.

The stats should add no model IO, preserve the R33/R30 speed path, and report
controller activity in CLI benchmark output.

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
- Repeat-margin gate: `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=500`
- New stats:
  - `lm_head_repeat_margin=<switches>/<checks>`
  - `max_gap_milli=<largest observed top-1/top-2 gap in milli-logit units>`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin rllm --bin llama-test
```

Production benchmark:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=500 \
  /usr/bin/time -l target/release/llama-test \
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
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=500 RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | repeat-margin switches/checks | max gap milli | selected exact match | raw sparse exact match | exact in sparse top-k | RLLM peak transient | max RSS | peak footprint | transformer time | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R30 repeat-run limit 2 control | 64 | 12.16s | 37.34 | 4.62 | 0.32 | 2 | 10/64 | N/A | N/A | N/A | N/A | N/A | 1050689536 | 1915584512 | 2158775584 | 1636.71ms | 49.35ms |
| R34 margin 500 stats | 64 | 11.91s | 36.88 | 4.70 | 0.00 | 1 | 12/64 | 30/30 | 453 | N/A | N/A | N/A | 1050689536 | 1661927424 | 2157120728 | 1660.35ms | 46.67ms |
| R34 margin 500 agreement profiler | 64 | 16.50s | 5.76 | 2.33 | 0.00 | 1 | 12/64 | 30/30 | 453 | 0/64 | 30/64 | 43/64 | 1050689536 | 1492254720 | 2156956744 | 2031.02ms | 8898.44ms |

## Analysis

The stats path works and does not materially slow the production mode. R34
margin 500 measured 36.88 tok/s with `lm_head_repeat_margin=30/30` and
`max_gap_milli=453`, while the R30 control measured 37.34 tok/s. The measured
decode delta is small enough to treat the instrumentation overhead as
negligible for this benchmark.

The controller stats explain the R33 behavior: on this prompt, every time the
repeat-margin condition was checked, it switched away from the repeating top-1
candidate. That produces repetition ratio 0.00 and max run 1, but the agreement
profiler still reports selected exact match at 0/64. Raw sparse exact match is
30/64 and exact-in-top-k is 43/64, so useful exact candidates are often nearby,
but the current controller is too aggressive for semantic fidelity.

## Decision

success with diagnostic limitation

Reason: R34 adds low-cost controller observability and preserves the 30-40 tok/s
speed target. It confirms that margin 500 is a strong collapse controller but
not a semantic quality fix.

Paper value:

- positive evidence that repeat-margin controller activity can be measured
  without extra model IO
- positive speed evidence: 36.88 tok/s with controller stats enabled
- limitation evidence: 30/30 controller switches correlates with 0/64 selected
  exact agreement

## Next Experiment

R35 should use these counters to make the controller adaptive rather than fixed:

- reduce switching when the controller has already switched too often in the
  recent window
- keep the R30/R33 speed path and avoid exact rescoring
- add a small rolling-window state in the Llama session, not global runtime
  state, so multi-turn behavior remains inspectable
