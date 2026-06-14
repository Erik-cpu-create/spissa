# Trial: R29 LM-Head Range Candidate Rescore

Date: 2026-06-15
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

R27 exact candidate rescoring was too slow because it read original row-major
LM-head chunks while scoring only a few candidate rows. R28 showed that the
exact LM-head token is often present in sparse top-4 candidates. R29 replaces
the R27 full-chunk candidate scorer with a row-range scorer that reads only the
overlapping byte ranges for selected candidate rows.

The goal was to improve candidate choice while staying near the R26 30 tok/s
floor.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Input-tile gate: `RLLM_AIP_INPUT_TILES=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Base transformer top-k: `RLLM_AIP_TOPK=4`
- Repeat guard: `RLLM_AIP_NO_REPEAT_LAST=1`
- Candidate rescore gate: `RLLM_AIP_LM_HEAD_RESCORE=<n>`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin rllm --bin llama-test
```

Benchmark:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_NO_REPEAT_LAST=1 RLLM_AIP_LM_HEAD_RESCORE=<n> \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | RLLM peak transient | max RSS | peak footprint | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R26 guardrail, profiler/rescore off | 64 | 12.77s | 29.56 | 4.30 | 0.00 | 1 | 12/64 | 1050689536 | 1711636480 | 2157104272 | 80.33ms |
| R29 range rescore 2 | 64 | 14.09s | 23.77 | 3.82 | 0.00 | 1 | 12/64 | 1050689536 | 1775042560 | 2157038688 | 282.74ms |
| R29 range rescore 4 | 64 | 14.08s | 20.83 | 3.74 | 0.00 | 1 | 19/64 | 1050689536 | 1958379520 | 2157628584 | 330.02ms |

## Analysis

R29 improved the R27 candidate-rescore implementation but did not meet the
speed gate. R27 selective rescore 4 measured about 14 tok/s and more than 2.2s
of LM-head time. R29 range rescore 4 reduces LM-head time to 330.02ms and raises
decode to 20.83 tok/s, so the range scorer is a real improvement over the
full-chunk candidate scan.

The improvement is still not enough. Rescore 2 reaches 23.77 tok/s but does not
increase unique token count over the guardrail. Rescore 4 increases unique
tokens to 19/64, but decode remains far below the 30 tok/s target.

The likely cost is per-token row-range overhead: sparse collapse hits the
rescore condition often, and each candidate row can touch different row-major
chunks. Even though row-range scoring avoids full candidate chunk scans, it
still adds enough LM-head work and range-call overhead to miss the target.

## Decision

failed

Reason: R29 is faster than R27, but range candidate rescoring still drops Llama
3.2 1B Instruct below the 30 tok/s speed floor.

Paper value:

- positive evidence that row-range candidate scoring is materially faster than
  full-chunk candidate scoring
- negative evidence that row-range exact rescoring is still too expensive when
  triggered frequently
- limitation evidence that exact LM-head candidate scoring does not solve
  quality at the current speed target

## Next Experiment

R30 should avoid exact LM-head rescoring in the hot path. A better next probe is
a cheaper repetition policy:

- keep sparse LM-head top-k 4 and the R26 speed path
- replace strict no-repeat with a small repeat-run limit, so one exact-like
  repeat can pass but long collapse is blocked
- measure whether this keeps speed near 30 tok/s while improving token
  agreement and avoiding the R25 max-run collapse
