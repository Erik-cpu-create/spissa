# Trial: R32 Edge-Layer Top-k Override

Date: 2026-06-15
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

R31 showed that LM-head top-k 8 can improve diversity, but it was not stable
inside the 30-40 tok/s range. R32 tests whether raising sparse top-k only on
transformer edge layers can recover hidden-state quality at lower cost than
making every layer wider.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Input-tile gate: `RLLM_AIP_INPUT_TILES=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Base transformer top-k: `RLLM_AIP_TOPK=4`
- LM-head top-k: `RLLM_AIP_LM_HEAD_TOPK=8`
- Repeat guard: `RLLM_AIP_REPEAT_RUN_LIMIT=2`
- New edge controls:
  - `RLLM_AIP_EDGE_LAYERS`
  - `RLLM_AIP_EDGE_TOPK`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin rllm --bin llama-test
```

Benchmark shape:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_LM_HEAD_TOPK=8 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_EDGE_LAYERS=<n> RLLM_AIP_EDGE_TOPK=<k> \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

Agreement profiler:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_LM_HEAD_TOPK=8 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  RLLM_AIP_EDGE_LAYERS=1 RLLM_AIP_EDGE_TOPK=8 RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | selected exact match | raw sparse exact match | exact in sparse top-k | RLLM peak transient | max RSS | peak footprint | transformer time | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R30 repeat-run limit 2 stable control | 64 | 11.62s | 36.94 | 4.80 | 0.32 | 2 | 10/64 | N/A | N/A | N/A | 1050689536 | 1658699776 | 1859504192 | 1655.39ms | 48.64ms |
| R31 LM-head top-k 8 first-pass candidate | 64 | 12.51s | 34.23 | 4.46 | 0.29 | 2 | 19/64 | 20/64 in profiler | 38/64 in profiler | 50/64 in profiler | 1050689536 | 1431666688 | 2156285408 | 1745.84ms | 93.05ms |
| edge layers 1, edge top-k 8 | 64 | 11.46s | 22.38 | 4.48 | 0.21 | 2 | 25/64 | 13/64 in profiler | 23/64 in profiler | 38/64 in profiler | 1050689536 | 1683718144 | 2157809024 | 2722.22ms | 91.40ms |
| edge layers 2, edge top-k 8 | 64 | 11.54s | 22.58 | 4.47 | 0.24 | 2 | 20/64 | N/A | N/A | N/A | 1050689536 | 1724956672 | 1768802320 | 2678.48ms | 110.48ms |
| edge layers 1, edge top-k 16 | 64 | 11.61s | 24.33 | 4.51 | 0.24 | 2 | 17/64 | N/A | N/A | N/A | 1050689536 | 1782169600 | 2157252112 | 2489.34ms | 99.03ms |

## Analysis

The edge-layer override works mechanically, but the hypothesis fails. Relative
to the stable R30 control, edge top-k increases diversity in the best case from
10/64 unique tokens to 25/64, yet decode drops to 22.38 tok/s, well below the
30 tok/s floor. The profiler also shows worse agreement than the R31 first-pass
candidate: selected exact match falls from 20/64 to 13/64, and exact-in-top-k
falls from 50/64 to 38/64.

The cost appears in transformer time. The R30 stable control spent 1655.39ms in
the transformer over the 64-token decode, while edge layers 1/top-k 8 spent
2722.22ms. The extra sparse range reads and wider edge-layer work are too
expensive for the current speed target.

The output still remains fragmentary, so the diversity gain is not enough to
justify the speed regression. R32 should stay opt-in for research because it
can test layer-group hypotheses, but it should not replace the R30 speed preset.

## Decision

failed

Reason: edge-layer top-k override improves diversity but misses the 30-40 tok/s
target and worsens selected-token agreement.

Paper value:

- negative evidence that wider edge-layer sparse routing is too expensive at
  the current target
- limitation evidence that diversity alone can move away from exact-token
  behavior
- positive tooling evidence: the runtime can now isolate layer-edge top-k
  experiments without changing the artifact format

## Next Experiment

R33 should avoid widening transformer layers. A better next probe is a cheap
token-selection controller around the existing R31 candidate set:

- keep `RLLM_AIP_TOPK=4`
- avoid `RLLM_AIP_LM_HEAD_TOPK=8` as a required default until it is made stable
- keep `RLLM_AIP_REPEAT_RUN_LIMIT=2`
- use sparse-logit gap, token repetition, or candidate-rank history to select
  among the existing LM-head top-k candidates without exact rescoring
