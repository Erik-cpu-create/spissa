# Trial: R33 Repeat-Margin Token Controller

Date: 2026-06-15
Owner: RLLM
Status: success with quality limitation
Folder: success

## Hypothesis

R30 repeat-run limit 2 is the current speed-stable path for Llama 3.2 1B
Instruct, but it still allows adjacent token pairs and fragmentary loops.
R31/R32 showed that widening LM-head or transformer top-k can improve diversity
but costs too much or is unstable.

R33 tests a cheaper controller: keep the R30 sparse top-k path, then use the
already-computed sparse LM-head logits to skip a repeating top-1 token when the
top-1/top-2 gap is small. This should reduce repeat collapse without exact
LM-head rescoring or wider sparse projections.

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
- New controller gate: `RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=<n>`

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
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=<n> \
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
  RLLM_AIP_LM_HEAD_REPEAT_MARGIN_MILLI=<n> RLLM_AIP_LM_HEAD_AGREEMENT=1 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | repetition ratio | max run | unique tokens | selected exact match | raw sparse exact match | exact in sparse top-k | RLLM peak transient | max RSS | peak footprint | transformer time | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R30 repeat-run limit 2 rerun | 64 | 11.87s | 35.68 | 4.69 | 0.32 | 2 | 10/64 | N/A | N/A | N/A | 1050689536 | 1609760768 | 2158201976 | 1710.47ms | 53.94ms |
| margin 50 | 64 | 11.79s | 33.79 | 4.69 | 0.17 | 2 | 11/64 | N/A | N/A | N/A | 1050689536 | 1655406592 | 1740016640 | 1804.60ms | 58.18ms |
| margin 100 | 64 | 12.07s | 32.61 | 4.57 | 0.10 | 2 | 12/64 | 6/64 in profiler | 35/64 in profiler | 47/64 in profiler | 1050689536 | 1562296320 | 1963837600 | 1878.27ms | 52.13ms |
| margin 250 | 64 | 11.76s | 32.50 | 4.67 | 0.06 | 2 | 13/64 | 4/64 in profiler | 31/64 in profiler | 46/64 in profiler | 1050689536 | 1709309952 | 2158448120 | 1885.69ms | 51.43ms |
| margin 500 | 64 | 12.08s | 34.80 | 4.61 | 0.00 | 1 | 12/64 | 0/64 in profiler | 30/64 in profiler | 43/64 in profiler | 1050689536 | 1474560000 | 2157432312 | 1758.73ms | 49.37ms |

## Analysis

The controller works as intended: it adds no model IO and does not widen sparse
projections. All tested margins stay inside the 30-40 tok/s decode band.

Margin 500 is the strongest collapse-control slice. It reaches 34.80 tok/s,
keeps RLLM peak transient memory unchanged at 1050689536 bytes, and removes
adjacent repeats on the benchmark prompt: repetition ratio 0.00 and max run 1.
This is materially better than R30's max run 2 while staying much faster than
the older strict no-repeat reference from R26.

The limitation is semantic quality. The output is still fragmentary, and the
agreement profiler shows margin 500 selected exact match at 0/64. Lower margins
retain a small amount of selected exact agreement, but the best measured value
in this sweep is only 6/64 at margin 100. This confirms that repeat-margin is a
cheap collapse controller, not a true quality recovery mechanism.

## Decision

success with quality limitation

Reason: R33 achieves strict adjacent-repeat suppression at 34.80 tok/s without
widening sparse projections or adding exact rescoring. It improves collapse
control while preserving the speed target, but it does not produce chat-quality
text.

Paper value:

- positive evidence that token-level sparse-logit control can reduce collapse
  without extra model IO
- positive speed evidence: margin 500 stays inside the 30-40 tok/s target
- limitation evidence: reducing repeats can move selected tokens away from
  exact LM-head behavior

## Next Experiment

R34 should keep R33 as an opt-in collapse controller and target semantic quality
without exact rescoring. The next probe should use only existing sparse logits:

- candidate rank history across recent tokens
- sparse-logit confidence windows instead of a fixed repeat margin
- optional controller stats so benchmark reports can count how often the
  controller changes the selected token
