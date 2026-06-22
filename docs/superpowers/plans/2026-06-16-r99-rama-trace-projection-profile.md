# R99 Implementation Plan: RAMA Trace Projection Profile

## Goal

R99 is a diagnostic stage after R98. The goal is to identify the next real
prefill bottleneck by projection bucket and layer before changing another CPU
kernel.

This stage must not claim a speedup. It is successful only if it produces
actionable evidence for R100.

## Context

R96 proved the `REEVEC-Q8-NEON-BATCH4` path can improve the Q8 batch-4 lab
shape and move runtime prefill from about 13.85s to about 9.03s in the best
observed run.

R98 added `REECAST-Q8-NEON-SCALE`; it produced a small lab win but only a noisy
runtime signal. The remaining `RLLM_Q8_KERNEL_PROFILE=1` hotspot is still the
batch-greater-than-one scaled Q8 path, so continuing with blind micro-kernels is
not enough.

## Architecture

Use the existing `llama-test` diagnostics:

- `--profile-phases` for prefill/decode phase timing.
- `--rama-trace <path>` for chunk-level trace JSON.
- `RLLM_Q8_KERNEL_PROFILE=1` for the runtime Q8 branch profile.

No runtime behavior changes are planned for R99. The trace JSON already records
`tensor_name`; R99 can derive layer/projection totals from that data with an
external parser.

## Commands

Build:

```sh
cargo build --release -p rllm-cli --bin llama-test
```

Trace run:

```sh
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace target/r99-rama-trace.json" > target/r99-trace.txt 2> target/r99-trace.time
```

Profiled trace run:

```sh
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace target/r99-rama-trace-profile.json" > target/r99-trace-profile.txt 2> target/r99-trace-profile.time
```

Parse:

- summarize `.summary.duration_by_tensor_bucket`
- summarize top layers from raw trace events by `model.layers.N.*.weight`
- summarize top layer/projection pairs
- capture `Q8KernelProfile` from the profiled run

## Acceptance Criteria

- `llama-test` builds in release mode.
- R99 traced run completes with output correctness unchanged for the control
  prompt. For this prompt, expected answer token is `No`.
- Report records:
  - wall-clock time and peak RSS
  - `--profile-phases` prefill/decode values
  - trace bucket totals
  - top layer/projection totals
  - Q8 profile from profiled trace run
  - recommended R100 target
- `docs/benchmarks/trials/index.md` links the R99 report.

## Non-goals

- No new kernel in R99.
- No quality tuning.
- No model repack.
- No multi-thread benchmark.
