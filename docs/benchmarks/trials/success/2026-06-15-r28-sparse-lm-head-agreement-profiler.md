# Trial: R28 Sparse LM-Head Agreement Profiler

Date: 2026-06-15
Owner: RLLM
Status: success with diagnostic limitation
Folder: success

## Hypothesis

R26 reaches the 30 tok/s gate but still produces poor text. R27 showed that
row-major exact candidate rescoring is too slow. R28 adds an opt-in profiler to
measure whether the sparse LM-head is at least proposing the exact LM-head token
inside its top-k candidates.

The goal is diagnostic: identify whether the next quality experiment should
target LM-head candidate scoring or deeper sparse transformer routing.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Input-tile gate: `RLLM_AIP_INPUT_TILES=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Base transformer top-k: `RLLM_AIP_TOPK=4`
- Profiler gate: `RLLM_AIP_LM_HEAD_AGREEMENT=1`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin rllm --bin llama-test
```

Agreement profiler:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_NO_REPEAT_LAST=1 RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

Production guardrail rerun without profiler:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_NO_REPEAT_LAST=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | selected exact match | raw sparse exact match | exact in sparse top-k | repetition ratio | max run | unique tokens | RLLM peak transient | max RSS | peak footprint | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R26 no-repeat guardrail, profiler off | 64 | 12.96s | 31.00 | 4.27 | N/A | N/A | N/A | 0.00 | 1 | 12/64 | 1050689536 | 1683013632 | 2158038208 | 74.31ms |
| R28 profiler, no-repeat on | 64 | 15.21s | 5.39 | 2.38 | 0/64 | 30/64 | 43/64 | 0.00 | 1 | 12/64 | 1050689536 | 1985282048 | 2158644488 | 9584.75ms |
| R28 profiler, no-repeat off | 64 | 15.41s | 5.46 | 2.37 | 39/64 | 39/64 | 44/64 | 0.62 | 18 | 10/64 | 1050689536 | 1607532544 | 2157169976 | 9394.24ms |

## Analysis

The profiler worked and did not change default behavior when disabled. The
post-patch R26 guardrail still reaches 31.00 tok/s, with the same low transient
memory profile and no adjacent repeats.

The profiler run is intentionally slow because it performs a full exact
LM-head argmax for each generated token. That cost appears in LM-head time:
74.31ms with the profiler off versus roughly 9.4-9.6s with the profiler on.

The useful signal is agreement:

- without the no-repeat guard, sparse LM-head selected the exact LM-head token
  39/64 times and had the exact token inside sparse top-4 44/64 times
- with the no-repeat guard, raw sparse argmax still matched exact 30/64 times,
  but final selected tokens matched exact 0/64 because the guard forces a
  different token whenever the top sparse token repeats the previous decode
  token

This means sparse LM-head top-4 is not random: it often contains the exact
candidate. The current no-repeat guard fixes adjacent collapse but also moves
the final choice away from exact LM-head behavior on this sparse hidden state.

## Decision

success with diagnostic limitation

Reason: R28 produced actionable agreement evidence while preserving the
non-profiler R26 speed gate. It is not a production mode because the exact
agreement scan is too slow.

Paper value:

- positive evidence that sparse LM-head top-4 often contains the exact LM-head
  argmax token
- limitation evidence that no-repeat is a collapse guard, not a quality
  recovery strategy
- negative speed evidence for full exact LM-head agreement scans inside decode
- methodology evidence for separating raw sparse argmax, selected token, and
  exact-in-top-k agreement

## Next Experiment

R29 should turn the R28 signal into a faster candidate-selection path:

- avoid full row-major chunk reads from R27
- try exact candidate scoring using raw row-range reads or a row-access sidecar
- only rescore sparse top-k LM-head candidates
- keep R26 as the floor: a candidate-quality mode must preserve about 30 tok/s
  or clearly justify a quality/speed trade-off
