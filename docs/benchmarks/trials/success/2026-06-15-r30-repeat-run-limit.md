# Trial: R30 Repeat-Run Limit

Date: 2026-06-15
Owner: RLLM
Status: success with quality limitation
Folder: success

## Hypothesis

R26 strict no-repeat reaches the 30 tok/s gate but moves selected tokens away
from exact LM-head behavior. R28 measured strict no-repeat selected exact match
at 0/64 on the benchmark prompt. R29 showed exact candidate rescoring is still
too expensive.

R30 tests a cheaper policy: allow a small same-token run, then block only when
the run reaches a configured limit. This avoids exact LM-head rescoring and
should keep speed close to the R26/R25 path while reducing long collapse.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Input-tile gate: `RLLM_AIP_INPUT_TILES=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Base transformer top-k: `RLLM_AIP_TOPK=4`
- New policy gate: `RLLM_AIP_REPEAT_RUN_LIMIT=<n>`

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
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | selected exact match | raw sparse exact match | exact in sparse top-k | repetition ratio | max run | unique tokens | RLLM peak transient | max RSS | peak footprint | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R25 no guard baseline | 64 | 11.25s | 37.13 | 4.94 | N/A | N/A | N/A | 0.62 | 18 | 10/64 | 1050689536 | 2251915264 | 2158579936 | N/A |
| R26 strict no-repeat reference | 64 | 12.93s | 30.06 | 4.26 | 0/64 in R28 profiler | 30/64 in R28 profiler | 43/64 in R28 profiler | 0.00 | 1 | 12/64 | 1050689536 | 2012823552 | 2157563072 | 75.07ms |
| R30 repeat-run limit 2 | 64 | 13.19s | 34.26 | 4.26 | N/A | N/A | N/A | 0.32 | 2 | 10/64 | 1050689536 | 2116780032 | 2158480720 | 60.30ms |
| R30 repeat-run limit 3 | 64 | 12.50s | 32.97 | 4.44 | N/A | N/A | N/A | 0.46 | 3 | 13/64 | 1050689536 | 1569210368 | 1743342808 | 50.06ms |
| R30 repeat-run limit 2 profiler | 64 | 14.15s | 6.59 | 2.70 | 20/64 | 37/64 | 45/64 | 0.32 | 2 | 10/64 | 1050689536 | 1683521536 | 2156973200 | 7831.32ms |

## Analysis

Repeat-run limit 2 is the best R30 slice. It keeps decode inside the 30-40 tok/s
target at 34.26 tok/s while capping collapse at a maximum run of 2. It is less
aggressive than strict no-repeat, so adjacent repetition returns, but it avoids
the R25 max-run 18 collapse.

The agreement profiler is intentionally slow because it runs exact LM-head
argmax per token. Its value is diagnostic: with limit 2, selected exact match is
20/64, compared with 0/64 for strict no-repeat in R28. This confirms that the
cheap run-limit policy preserves more exact-like token choices than the strict
guard.

The output is still not chat-quality. R30 improves the speed/quality trade-off
relative to strict no-repeat and exact candidate rescoring, but the generated
text is still fragmentary. The next work should target sparse hidden-state
quality or adaptive projection routing, not more exact LM-head rescoring.

## Decision

success with quality limitation

Reason: R30 restores a 30-40 tok/s production-mode result for Llama 3.2 1B
Instruct while reducing long repeat collapse and improving selected-token
agreement over strict no-repeat.

Paper value:

- positive evidence that a cheap repeat-run limit can stay in the speed target
- positive evidence that strict no-repeat overcorrects relative to exact LM-head
  behavior
- limitation evidence that collapse control alone is not enough for chat-quality
  semantics

## Next Experiment

R31 should target sparse hidden-state quality without exact LM-head work:

- adaptive projection top-k by layer group
- exact first/last layer or exact attention with sparse MLP
- keep `RLLM_AIP_REPEAT_RUN_LIMIT=2` as the current speed-gated collapse guard
