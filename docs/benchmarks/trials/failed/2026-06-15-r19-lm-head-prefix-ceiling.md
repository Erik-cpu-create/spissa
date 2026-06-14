# Trial: R19 LM-Head Prefix Ceiling

Date: 2026-06-15
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

RLLM can expose a radical experimental speed ceiling by scanning only a prefix
of the LM-head rows during argmax. If full-vocabulary LM-head work is the
dominant decode bottleneck, a small prefix should push Llama 3.2 1B Instruct
much closer to the 30-40 tok/s research target.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Expected bottleneck: LM-head full-vocabulary argmax
- Bottleneck tag: LM-head vocabulary scan
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Top-k: `RLLM_AIP_TOPK=128`
- LM-head prefix: `RLLM_AIP_LM_HEAD_ROWS`

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 RLLM_AIP_LM_HEAD_ROWS=512 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 RLLM_AIP_LM_HEAD_ROWS=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 RLLM_AIP_LM_HEAD_ROWS=512 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 64
```

Runtime context:

- build profile: release
- relevant env/config: `RLLM_EXPERIMENTAL_SPEED`, `RLLM_AIP_POLICY`, `RLLM_AIP_TOPK`, `RLLM_AIP_LM_HEAD_ROWS`
- default exact path remains unchanged unless `RLLM_AIP_LM_HEAD_ROWS` is set

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | AIP calls | fallbacks | max top-k | lm-head rows | repeated ratio | max run | unique tokens | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | exact baseline | 16 | 14.21s | 0.86 | 0.50 | 0 | 0 | 0 | full | 0.00 | 1 | 15/16 | 2477768704 | 1620741408 | 1050689536 |
| Llama-3.2-1B-Instruct | AIP speed, lm-head 512 | 16 | 11.40s | 5.28 | 1.12 | 480 | 32 | 128 | 512/128256 | 0.93 | 15 | 2/16 | 2119581696 | 1620675896 | 1050689536 |
| Llama-3.2-1B-Instruct | AIP speed, lm-head 128 | 16 | 9.06s | 3.62 | 1.21 | 480 | 32 | 128 | 128/128256 | 0.87 | 8 | 3/16 | 2200944640 | 1620593976 | 1050689536 |
| Llama-3.2-1B-Instruct | AIP speed, lm-head 512 | 64 | 10.66s | 7.31 | 3.32 | 2016 | 32 | 128 | 512/128256 | 0.98 | 63 | 2/64 | 2320908288 | 1620741432 | 1050689536 |

## Analysis

The experiment confirms that full-vocabulary LM-head work is a meaningful
decode cost, but it also rejects prefix-vocabulary argmax as a viable route to
the 30-40 tok/s target.

The best 64-token run reached `7.31 tok/s`, which is a large increase over the
short exact baseline but still far below the target. Output quality collapsed:
the `512` row run generated mostly repeated `v` tokens, with `ratio=0.98`,
`max_run=63`, and only `2/64` unique tokens. Reducing the prefix to `128` rows
did not improve speed, which suggests the remaining bottleneck is not only the
number of LM-head rows. The sparse MLP path, chunk streaming overhead, per-layer
fallbacks, attention, and session orchestration still dominate enough work to
cap speed.

RLLM tracked peak transient memory stayed flat at `1050689536` bytes, so the
experiment did not regress the low-RAM behavior.

## Decision

failed

Reason: prefix-vocabulary LM-head argmax does not reach the target speed and
produces unusable repeated output. It should remain an experimental ceiling
probe, not a default or quality path.

Paper value:
- useful negative result for vocabulary-prefix approximation
- useful evidence that LM-head cost matters but is not the only dominant cost
- useful evidence that repetition telemetry catches token-collapse modes
- useful low-RAM evidence because RLLM peak transient memory stayed unchanged

## Next Experiment

R20 should move away from prefix-vocabulary output. The next useful stage is a
phase-level decode profiler that records per-token time in attention, MLP,
LM-head, chunk decode, and session overhead. The profiler should tell us which
hot path must be attacked first before implementing another approximation.
