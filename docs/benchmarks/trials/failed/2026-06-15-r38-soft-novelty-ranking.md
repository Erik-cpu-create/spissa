# Trial: R38 Soft Phrase Novelty Ranking

Date: 2026-06-15
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

R37 made phrase novelty binary: if the sparse selected token repeated a recent
phrase and the confidence gap was small enough, switch to the first top-k
candidate that did not repeat that phrase. R38 tests a softer ranking: keep the
same bounded top-4 scan, but score fallback candidates by sparse-logit gap plus
an optional repeat penalty.

The goal is to preserve the 30-40 tok/s Llama 3.2 1B Instruct band while
recovering exact-agreement signal or output diversity versus R37.

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
- Adaptive margin: `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=75`,
  `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1`
- Phrase novelty: `RLLM_AIP_LM_HEAD_NOVELTY_WINDOW=4`
- Confidence gate: `RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100`
- New soft ranking control:
  `RLLM_AIP_LM_HEAD_NOVELTY_REPEAT_PENALTY_MILLI`

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
  RLLM_AIP_LM_HEAD_NOVELTY_REPEAT_PENALTY_MILLI=<penalty> \
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
  RLLM_AIP_LM_HEAD_NOVELTY_GAP_MILLI=100 \
  RLLM_AIP_LM_HEAD_NOVELTY_REPEAT_PENALTY_MILLI=300 \
  RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | phrase novelty switches/checks | gap skips | soft choices | selected exact match | raw sparse exact match | exact in sparse top-k | RLLM peak transient | max RSS | peak footprint | transformer time | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R37 control rerun | 64 | 11.98s | 30.09 | 4.55 | 0.11 | 2 | 17/64 | 5/64 | 15 | N/A | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2027.41ms | 65.22ms |
| R38 penalty 75 sweep | 64 | 12.50s | 33.01 | 4.44 | 0.13 | 2 | 15/64 | 5/64 | 16 | 5 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1839.49ms | 68.30ms |
| R38 penalty 150 sweep | 64 | 11.98s | 49.52 | 4.83 | 0.13 | 2 | 15/64 | 5/64 | 16 | 5 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1216.16ms | 55.06ms |
| R38 penalty 300 sweep | 64 | 12.59s | 31.62 | 4.39 | 0.11 | 2 | 17/64 | 5/64 | 15 | 5 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1913.70ms | 77.80ms |
| R38 penalty 300 run 1 | 64 | 12.11s | 28.80 | 4.48 | 0.11 | 2 | 17/64 | 5/64 | 15 | 5 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2119.39ms | 67.28ms |
| R38 penalty 300 run 2 | 64 | 12.77s | 33.29 | 4.37 | 0.11 | 2 | 17/64 | 5/64 | 15 | 5 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1823.35ms | 67.61ms |
| R38 penalty 300 run 3 | 64 | 12.39s | 15.35 | 3.88 | 0.11 | 2 | 17/64 | 5/64 | 15 | 5 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 4019.99ms | 83.12ms |
| R38 penalty 300 agreement profiler | 64 | 18.18s | 4.97 | 2.07 | 0.11 | 2 | 17/64 | 5/64 | 15 | 5 | 7/64 | 28/64 | 40/64 | 1050689536 | 1239089152 | 2156989584 | 2395.09ms | 10284.03ms |
| R37 post-load control rerun | 64 | 12.65s | 29.70 | 4.33 | 0.11 | 2 | 17/64 | 5/64 | 15 | N/A | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2052.57ms | 67.17ms |

## Analysis

The feature works mechanically: the new `soft_choices` metric records five
soft-ranked fallback choices in the tested prompt, and the controller remains
bounded to the existing top-4 sparse candidate set.

The hypothesis fails. Penalties 75 and 150 stay fast in the first sweep but
reduce unique tokens from 17/64 to 15/64. Penalty 300 preserves the R37 diversity
shape, but the three validation runs are not stable inside the 30-40 tok/s
band: 28.80, 33.29, and 15.35 tok/s. A post-load R37 control also measured
29.70 tok/s, so part of the slowdown is likely machine/load variance, but R38
still does not prove a stable improvement.

The agreement profiler also does not improve the R37 quality signal. R38
penalty 300 records selected exact 7/64, raw sparse exact 28/64, and exact in
top-k 40/64, matching the R37 profiler values. The output remains fragmentary.

## Decision

failed

Reason: R38 soft phrase novelty ranking is functional and opt-in, but it does
not improve exact agreement versus R37 and does not prove stable 30-40 tok/s
performance in this validation run.

Paper value:

- negative evidence that soft novelty ranking alone does not recover semantic
  quality
- useful instrumentation evidence: `soft_choices` separates soft-ranked
  fallback decisions from hard novelty switches
- benchmark caveat: current machine/load variance can move both R37 and R38
  below 30 tok/s, so future speed claims should include repeated control runs

## Next Experiment

R39 should stop trying to improve quality by changing only the novelty fallback
order. Better next probes:

- add a deterministic benchmark harness that records repeated control/candidate
  runs and rejects thermally unstable windows before accepting speed claims
- try a selected-token retention score so exact sparse top-1 can survive weak
  novelty pressure
- keep all new quality controllers opt-in until they beat R37 on both speed and
  exact-agreement evidence
