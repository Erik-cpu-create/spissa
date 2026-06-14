# Trial: R31 Projection Top-k Sweep

Date: 2026-06-15
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

R30 repeat-run limit 2 restored the 30-40 tok/s speed band but output still
collapsed into fragmentary text. R31 tests whether projection-specific top-k
controls can improve sparse output diversity while keeping Llama 3.2 1B
Instruct inside the 30 tok/s floor.

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
- Projection knobs:
  - `RLLM_AIP_ATTENTION_TOPK`
  - `RLLM_AIP_MLP_TOPK`
  - `RLLM_AIP_DOWN_TOPK`
  - `RLLM_AIP_LM_HEAD_TOPK`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin rllm --bin llama-test
```

Benchmark shape:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  <optional projection top-k env> \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

Fastest diversity slice observed during the first sweep:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_LM_HEAD_TOPK=8 RLLM_AIP_REPEAT_RUN_LIMIT=2 \
  target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm \
    --ctx 2048 \
    --max-new-tokens 64
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | selected exact match | raw sparse exact match | exact in sparse top-k | RLLM peak transient | max RSS | peak footprint | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R30 repeat-run limit 2 rerun | 64 | 12.04s | 36.50 | 4.65 | 0.32 | 2 | 10/64 | N/A | N/A | N/A | 1050689536 | 1507262464 | 2157710672 | 49.64ms |
| R30 repeat-run limit 2 post-patch control | 64 | 11.62s | 36.94 | 4.80 | 0.32 | 2 | 10/64 | N/A | N/A | N/A | 1050689536 | 1658699776 | 1859504192 | 48.64ms |
| attention top-k 8 | 64 | 12.07s | 32.16 | 4.56 | 0.32 | 2 | 9/64 | N/A | N/A | N/A | 1050689536 | 1417969664 | 2157366728 | 52.69ms |
| MLP top-k 6 | 64 | 11.74s | 31.41 | 4.66 | 0.32 | 2 | 9/64 | N/A | N/A | N/A | 1050689536 | 1633828864 | 2158087360 | 45.94ms |
| down top-k 8 | 64 | 11.92s | 32.25 | 4.61 | 0.35 | 2 | 12/64 | 22/64 in profiler | 41/64 in profiler | 50/64 in profiler | 1050689536 | 1649639424 | 2156203488 | 49.33ms |
| LM-head top-k 6 | 64 | 11.83s | 34.79 | 4.69 | 0.35 | 2 | 8/64 | N/A | N/A | N/A | 1050689536 | 1689665536 | 2156269000 | 79.35ms |
| LM-head top-k 8 first pass | 64 | 12.51s | 34.23 | 4.46 | 0.29 | 2 | 19/64 | 20/64 in profiler | 38/64 in profiler | 50/64 in profiler | 1050689536 | 1431666688 | 2156285408 | 93.05ms |
| LM-head top-k 8 post-patch sanity | 64 | 11.93s | 26.84 | 4.48 | 0.29 | 2 | 19/64 | N/A | N/A | N/A | 1050689536 | 1696366592 | 2156956840 | 116.43ms |
| LM-head top-k 8 post-patch rerun | 64 | 11.67s | 24.42 | 4.49 | 0.29 | 2 | 19/64 | N/A | N/A | N/A | 1050689536 | 1609662464 | 2156874872 | 133.58ms |
| LM-head top-k 8 no-profile post-patch | 64 | 11.52s | 26.39 | 4.60 | 0.29 | 2 | 19/64 | N/A | N/A | N/A | 1050689536 | 1624424448 | 2157120608 | N/A |

## Analysis

LM-head top-k 8 is the strongest R31 diversity slice, but it is not stable
enough to become the speed preset. The first pass reached 34.23 tok/s and
improved unique tokens from 10/64 to 19/64. Current post-patch reruns of the
same path measured 26.84, 24.42, and 26.39 tok/s, which is below the 30 tok/s
floor even without `--profile-phases`.

The quality limitation remains. The text is still fragmentary and not
chat-ready. The agreement profiler shows that LM-head top-k 8 increases exact
presence in the sparse candidate set to 50/64, but final selected exact match
stays at 20/64. This means the exact token is often nearby in sparse logit
space, but the cheap argmax + repeat guard still does not reliably select it.

Attention top-k 8, MLP top-k 6, and down top-k 8 all remain inside the 30 tok/s
floor in the initial sweep, but none produced a strong enough quality signal to
replace the R30 repeat-run limit baseline.

## Decision

failed

Reason: R31 found useful projection-specific signals, but the only variant with
a meaningful diversity improvement, `RLLM_AIP_LM_HEAD_TOPK=8`, did not
reliably stay inside the 30-40 tok/s target on rerun. The stable speed preset
remains R30 repeat-run limit 2 without LM-head widening.

Paper value:

- negative evidence that LM-head sparse candidate width can improve diversity
  but risks falling below the speed target
- limitation evidence that candidate presence is not enough; selected-token
  quality remains weak
- comparison evidence for projection-specific top-k cost differences

## Next Experiment

R32 should test adaptive hidden-state quality cheaply. The first probe is
edge-layer top-k override: keep middle layers at the R31 speed path but raise
top-k on the first and last layers to see whether hidden-state quality improves
without applying a global cost increase.
