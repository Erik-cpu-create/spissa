# Trial: R78 CPU Q8 Prefill Kernel

Date: 2026-06-16
Owner: RLLM
Status: rejected
Folder: failed

## Hypothesis

Llama 3.2 1B Q8 exact-lowram prefill is dominated by CPU Q8 MLP projections.
Adding a Q8_0 complete-row fast path for `batch > 1` should reduce prefill time
without changing generated text, peak transient memory, or CPU-only semantics.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Architecture: Llama 3.2 1B Instruct
- Target device/profile: local CPU-only RLLM release build
- Expected bottleneck: Q8 MLP projection prefill
- Bottleneck tag: CPU arithmetic

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf '%s\nquit\n' 'Answer yes or no: is fire cold?' \
  | target/release/llama-test \
      --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm \
      --chat-template llama3 \
      --max-new-tokens 4 \
      --profile-phases
```

Runtime context:

- build profile: release
- OS: macOS
- GPU: not used by RLLM
- relevant config: `--chat-template llama3`, `--profile-phases`

## Results

| run | prompt/input tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 55 | 2 | 26.75s | 0.87 | 0.07 | 2233090048 | 1050673152 | output `No`; prefill transformer 24613.37ms, prefill MLP 20324.73ms, gate/up/down 6723.82/6745.49/6844.64ms |
| trial | 55 | 2 | 27.19s | 1.42 | 0.07 | 2443804672 | 1050673152 | output `No`; prefill transformer 25261.66ms, prefill MLP 20858.40ms, gate/up/down 6867.32/7009.06/6971.33ms |

## Analysis

Baseline confirms the current exact-Q8 Llama 1B prefill bottleneck is CPU MLP
projection arithmetic, not attention. Prefill took `26754.93ms`; transformer
time was `24613.37ms`, and MLP alone took `20324.73ms`. The three MLP
projections were evenly dominant: gate `6723.82ms`, up `6745.49ms`, and down
`6844.64ms`. Attention total was `4288.49ms`, and LM head was `2141.34ms`.

The trial output stayed correct (`No`) and peak transient memory stayed
unchanged, but the batch complete-row path did not reduce prefill. TTFT regressed
from `26.75s` to `27.19s`; prefill MLP increased from `20324.73ms` to
`20858.40ms`; RSS also increased from `2233090048` to `2443804672` bytes. The
decode tok/s number improved on this short two-token run, but this trial targeted
prefill and did not meet the 10% prefill improvement threshold.

The tested runtime commit was reverted after measurement so the failed kernel
does not remain in the runtime path.

## Decision

rejected

Reason: Q8 batch complete-row accumulation was not faster for Llama 1B prefill
and slightly regressed TTFT/prefill.

Paper value:

- useful negative evidence

## Next Experiment

Do not pursue this row-major batch loop shape. The next useful experiment should
profile chunk/IO overhead versus arithmetic inside the existing block-major Q8
generic path, or test a different CPU kernel layout that reuses each Q8 block
across the batch more effectively.
