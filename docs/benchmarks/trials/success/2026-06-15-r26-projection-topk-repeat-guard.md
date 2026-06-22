# Trial: R26 Projection Top-K and Repeat Guard

Date: 2026-06-15
Owner: RLLM
Status: success with quality limitation
Folder: success

## Hypothesis

R25 crossed the Llama 3.2 1B speed target, but output collapsed into repeated
tokens. R26 tests two low-cost quality recovery controls without changing the
R25 sidecar artifact:

- projection-specific AIP top-k overrides
- an opt-in no-repeat-last argmax guard for sparse LM-head decode steps

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Input-tile gate: `RLLM_AIP_INPUT_TILES=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Base transformer top-k: `RLLM_AIP_TOPK=4`
- Repeat guard: `RLLM_AIP_NO_REPEAT_LAST=1`
- New projection knobs:
  - `RLLM_AIP_ATTENTION_TOPK`
  - `RLLM_AIP_MLP_TOPK`
  - `RLLM_AIP_DOWN_TOPK`
  - `RLLM_AIP_LM_HEAD_TOPK`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin rllm --bin llama-test
```

Primary benchmark:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_NO_REPEAT_LAST=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | max top-k | input-tile reads | input-tile bytes | repetition ratio | max run | unique tokens | RLLM peak transient | max RSS | peak footprint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R25 baseline top-k 4 | 64 | 11.25s | 37.13 | 4.94 | 4 | 28480 | 255590400 | 0.62 | 18 | 10/64 | 1050689536 | 2251915264 | 2158579936 |
| global top-k 8 | 32 | 12.56s | 12.67 | 2.13 | 8 | 28032 | 252575744 | 0.39 | 5 | 13/32 | 1050689536 | 1890729984 | 2157612752 |
| quality policy top-k 4 | 16 | 16.63s | 1.81 | 0.64 | 4 | 4864 | 41975808 | 0.13 | 3 | 12/16 | 1050689536 | 1471135744 | 2157219080 |
| LM-head top-k 16 | 64 | 15.91s | 27.31 | 3.51 | 16 | 29248 | 452591616 | 0.79 | 48 | 10/64 | 1050689536 | 1943388160 | 2158202144 |
| no-repeat top-k 4 | 64 | 12.93s | 30.06 | 4.26 | 4 | 28480 | 255590400 | 0.00 | 1 | 12/64 | 1050689536 | 2012823552 | 2157563072 |
| no-repeat top-k 3 | 64 | 13.61s | 26.13 | 4.00 | 3 | 21360 | 191692800 | 0.00 | 1 | 20/64 | 1050689536 | 1676197888 | 2157333696 |

Representative R26 no-repeat top-k 4 profile:

| decode phase | time |
|---|---:|
| decode total | 2096.05ms |
| transformer | 2019.68ms |
| attention total | 905.11ms |
| MLP total | 1113.36ms |
| LM-head | 75.07ms |
| q projection | 253.82ms |
| k projection | 202.54ms |
| v projection | 186.88ms |
| gate projection | 595.29ms |
| down projection | 515.52ms |

## Analysis

Global top-k increases are too expensive. Top-k 8 improved the repetition ratio
on a 32-token run, but decode fell to 12.67 tok/s. The existing quality policy
is not a speed candidate because exact MLP down dominates and decode falls to
1.81 tok/s.

Increasing only LM-head top-k also failed. It raised LM-head traffic and made
the visible collapse worse in this prompt, reaching only 27.31 tok/s with a
0.79 repetition ratio.

The no-repeat-last guard is the best R26 slice. It keeps the R25 transformer and
LM-head top-k at 4, but when sparse LM-head argmax would repeat the immediately
previous decode token, it selects the next-best token from the already
materialized sparse logits. This adds negligible algorithmic complexity and
keeps the run at 30.06 tok/s while dropping adjacent repetition to 0.00.

The output is still not chat-quality. The guard prevents same-token collapse,
but the text remains fragmentary and semantically poor. R26 should be treated as
a quality recovery step, not as final inference quality.

## Decision

success with quality limitation

Reason: R26 preserves the 30-40 tok/s speed gate on Llama 3.2 1B Instruct in
CPU-only experimental mode and removes adjacent repeated-token runs for the
measured prompt. It does not yet solve semantic quality.

Paper value:

- positive evidence that repeat-collapse can be reduced without increasing RAM
- positive evidence that projection-specific top-k controls are available for
  follow-up ablations without repacking
- negative evidence that global top-k increases and LM-head-only top-k increases
  are not enough
- limitation evidence that no-repeat guarding is not a substitute for better
  approximation quality

## Next Experiment

R27 should measure quality against exact-mode reference instead of relying only
on repetition:

- compare sparse generated tokens against exact generated tokens for fixed
  prompts
- add a small candidate-rescore mode: sparse path proposes a short candidate
  set, exact LM-head scores only those candidates
- test adaptive projection top-k by layer group while keeping the 30 tok/s gate
- report repeated samples because speed varied between 30 and 54 tok/s on the
  same artifact in R25/R26 runs
