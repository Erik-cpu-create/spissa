# Trial: R20 Phase Decode Profiler

Date: 2026-06-15
Owner: RLLM
Status: success
Folder: success

## Hypothesis

RLLM needs decode-phase profiling before the next radical speed experiment. R18
and R19 showed that sparse MLP and LM-head prefix tricks can move speed, but
they did not explain enough of the remaining bottleneck to justify another
approximation.

## Scope

- Mode: exact-lowram and experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-raw.spsa`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Expected bottleneck: per-token projection streaming
- Bottleneck tag: decode phase attribution
- CLI gate: `--profile-phases`
- Runtime behavior: default generation path remains unchanged unless the
  profiler flag is passed

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16 \
    --profile-phases

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16 \
    --profile-phases
```

Runtime context:

- build profile: release
- relevant env/config: `--profile-phases`, `RLLM_EXPERIMENTAL_SPEED`,
  `RLLM_AIP_POLICY`, `RLLM_AIP_TOPK`
- profiling output is wall-clock instrumentation and should be used for
  bottleneck ranking, not as the final speed score

## Results

| model | variant | generated | decode tok/s under profiler | decode wall | profiled total | overhead | transformer | attention total | MLP total | gate/fused gate-up | down | LM-head | profiled layers | repeated ratio | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | exact profiler | 16 | 0.15 | 97549.40ms | 97548.39ms | 1.01ms | 80082.65ms | 12768.33ms | 67313.24ms | 46392.87ms | 20917.33ms | 17465.47ms | 240 | 0.00 | 2263728128 | 1620364408 | 1050689536 |
| Llama-3.2-1B-Instruct | AIP speed top-k 128 profiler | 16 | 0.77 | 19573.69ms | 19573.36ms | 0.32ms | 14659.36ms | 3302.95ms | 11355.88ms | 6122.65ms | 5231.63ms | 4913.85ms | 240 | 0.93 | 2477834240 | 1620626744 | 1050689536 |

## Analysis

The profiler worked and produced a clear ranking. In exact mode, decode time is
dominated by the transformer block, especially MLP projection work:

- MLP total: `67313.24ms`
- gate/fused gate-up: `46392.87ms`
- down projection: `20917.33ms`
- LM-head: `17465.47ms`
- attention total: `12768.33ms`

The AIP speed profile confirms the same shape after sparse routing. MLP is
still larger than LM-head, even though AIP reduces arithmetic:

- MLP total: `11355.88ms`
- gate/fused gate-up: `6122.65ms`
- down projection: `5231.63ms`
- LM-head: `4913.85ms`
- attention total: `3302.95ms`

This explains why R19 prefix-vocabulary argmax did not reach the target. It
reduced one expensive phase, but MLP and projection streaming still dominated
enough of the decode loop to cap speed. It also explains why top-k arithmetic
alone is not enough: RLLM still touches row-major weight bytes in the sparse
path, so memory traffic and chunk streaming remain large.

RLLM tracked peak transient memory stayed flat at `1050689536` bytes.

## Decision

success

Reason: R20 produced decode-specific, reproducible phase attribution in
`llama-test` without changing default runtime behavior.

Paper value:
- useful profiling evidence for why the project should attack MLP projection
  streaming before more LM-head shortcuts
- useful limitation: sparse arithmetic does not automatically reduce row-major
  weight traffic
- useful methodology evidence: profiler separates decode wall time from
  prefill/TTFT

## Next Experiment

R21 should target sparse MLP memory traffic, not only sparse arithmetic. The
most promising path is an experimental activation-column or input-tile layout
that lets AIP read only selected input dimensions instead of scanning the full
row-major projection tensor.
