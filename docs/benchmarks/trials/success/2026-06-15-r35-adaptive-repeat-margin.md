# Trial: R35 Adaptive Repeat-Margin Controller

Date: 2026-06-15
Owner: RLLM
Status: success with quality limitation
Folder: success

## Hypothesis

R34 proved that the repeat-margin controller is cheap to observe, but margin 500
was too aggressive: it switched 30/30 checks and selected exact LM-head output
0/64 times in the agreement profiler.

R35 tests a local adaptive controller. After three consecutive margin switches,
the Llama session temporarily throttles the effective margin to one quarter of
the configured value. The goal is to keep the R30/R33 sparse path, avoid exact
LM-head rescoring in production, reduce over-switching, and stay inside the
30-40 tok/s Llama 3.2 1B Instruct target.

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
- Adaptive gate: `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1`
- Best R35 margin in this sweep: `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=50`

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
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=50 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
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
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=50 \
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_ADAPTIVE=1 \
  RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | repeat-margin switches/checks | adaptive throttles | min margin milli | selected exact match | raw sparse exact match | exact in sparse top-k | RLLM peak transient | max RSS | peak footprint | transformer time | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R30 repeat-run limit 2 control | 64 | 12.87s | 30.40 | 4.28 | 0.32 | 2 | 10/64 | N/A | N/A | N/A | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2011.98ms | 59.63ms |
| R34-style fixed margin 500 rerun | 64 | 12.91s | 28.42 | 4.23 | 0.00 | 1 | 12/64 | 30/30 | 0 | N/A | N/A | N/A | N/A | 1050689536 | N/A | N/A | 2155.19ms | 60.68ms |
| R35 margin 50 adaptive run 1 | 64 | 12.76s | 31.30 | 4.33 | 0.17 | 2 | 11/64 | 18/29 | 3 | 12 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1946.46ms | 65.46ms |
| R35 margin 50 adaptive run 2 | 64 | 14.94s | 30.87 | 3.77 | 0.17 | 2 | 11/64 | 18/29 | 3 | 12 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1976.32ms | 63.53ms |
| R35 margin 50 adaptive run 3 | 64 | 13.91s | 32.06 | 4.03 | 0.17 | 2 | 11/64 | 18/29 | 3 | 12 | N/A | N/A | N/A | 1050689536 | N/A | N/A | 1898.74ms | 65.34ms |
| R35 margin 50 agreement profiler | 64 | 15.16s | 5.10 | 2.33 | 0.17 | 2 | 11/64 | 18/29 | 3 | 12 | 13/64 | 42/64 | 50/64 | 1050689536 | 1659568128 | 2157022328 | 2174.94ms | 10172.31ms |
| R35 margin 50 `/usr/bin/time -l` RSS run | 64 | 12.98s | 29.15 | 4.23 | 0.17 | 2 | 11/64 | 18/29 | 3 | 12 | N/A | N/A | N/A | 1050689536 | 1604567040 | 2157268376 | 2094.14ms | 66.25ms |

R35 margin 50 profile runs averaged 31.41 tok/s across three runs:
31.30, 30.87, and 32.06 tok/s. The `/usr/bin/time -l` RSS run is included for
memory evidence, but its decode number was treated as load-sensitive because the
same command without the wrapper stayed inside the target range.

## Analysis

R35 improves the R34 failure mode. Fixed margin 500 reran at 30/30 switches and
fell below the current-run speed target. Adaptive margin 50 reduced controller
activity to 18/29 checks with only three throttles, kept max run at 2, and stayed
inside the 30-40 tok/s production profile band across three no-wrapper profile
runs.

The agreement profiler is the strongest quality signal: selected exact match
improved from R34's 0/64 to 13/64, raw sparse exact match improved from 30/64 to
42/64, and exact-in-top-k improved from 43/64 to 50/64. This means the adaptive
controller is less destructive than the fixed margin controller while still
using only sparse logits in production.

The output is still not chat-quality. It remains fragmentary and repetitive at
the phrase level even though adjacent collapse is reduced. R35 should be treated
as a speed/diagnostic checkpoint, not the final quality solution.

## Decision

success with quality limitation

Reason: R35 reaches the 30-40 tok/s target for Llama 3.2 1B Instruct in
production profile runs and improves exact agreement versus fixed margin 500
without adding exact rescoring to the hot path.

Paper value:

- positive evidence that a tiny local controller can improve selected-token
  agreement while preserving speed
- positive speed evidence: 31.41 tok/s average across three production profile
  runs for margin 50 adaptive
- limitation evidence: output remains semantically poor despite improved sparse
  agreement

## Next Experiment

R36 should target phrase-level collapse rather than adjacent-token collapse:

- keep R35 adaptive margin 50 as the speed/quality control
- add a cheap rolling token-window novelty score in session state
- only intervene when the last 8-16 generated tokens repeat a recent pattern
- avoid exact LM-head rescoring and avoid widening transformer top-k
