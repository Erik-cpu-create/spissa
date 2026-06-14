# Trial: R18 Quality-Aware AIP

Date: 2026-06-15
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

RLLM AIP quality policy can keep a useful part of the R17 sparse MLP speed gain
while reducing generated-token repetition by keeping early layers, final
layers, `mlp_down`, attention, and LM-head exact.

## Scope

- Mode: experimental-speed
- Models/artifacts: `models/SmolLM2-135M-raw.rllm`, `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Expected bottleneck: MLP projection arithmetic and memory access
- Bottleneck tag: CPU arithmetic
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- AIP policy sweep: `RLLM_AIP_POLICY=quality`, `RLLM_AIP_POLICY=speed`
- Top-k: `RLLM_AIP_TOPK=128`

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
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=quality RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=quality RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

Runtime context:

- build profile: release
- CPU: record from benchmark machine
- RAM: record from benchmark machine
- OS: record from benchmark machine
- relevant env/config: `RLLM_EXPERIMENTAL_SPEED`, `RLLM_AIP_POLICY`, `RLLM_AIP_TOPK`

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | AIP calls | fallbacks | max top-k | repeated ratio | max run | unique tokens | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | exact baseline | 16 | 10.99s | 1.43 | 0.74 | 0 | 0 | 0 | 0.00 | 1 | 15/16 | 2733178880 | 1621822776 | 1050689536 |
| Llama-3.2-1B-Instruct | AIP quality top-k 128 | 16 | 13.85s | 1.27 | 0.62 | 120 | 8 | 128 | 0.40 | 6 | 6/16 | 2517024768 | 1621740832 | 1050689536 |
| Llama-3.2-1B-Instruct | AIP speed top-k 128 | 16 | 10.66s | 3.60 | 1.08 | 480 | 32 | 128 | 0.93 | 15 | 2/16 | 2965798912 | 1621839184 | 1050689536 |
| SmolLM2-135M | AIP quality top-k 128 | 16 | 1.42s | 23.38 | 7.76 | 240 | 16 | 128 | 0.47 | 7 | 9/16 | 460144640 | 190120728 | 113262592 |

## Analysis

The telemetry and routing framework was successfully built and executed, but
the benchmark does not meet the R18 success criteria. The quality policy reduced
repetition versus the speed policy, but it did not keep enough speed.

The measured exact baseline was `1.43 tok/s`. AIP quality reached only
`1.27 tok/s` in this report, below the exact baseline and far below the required
2x improvement. A later rerun showed exact at `1.13 tok/s` and AIP quality at
`1.80 tok/s`, about `1.6x`; that is still below the R18 threshold.

The result is still useful evidence:

- **Speed policy**: highly repetitive loop, `ratio=0.93`, `unique=2/16`, and
  faster decode at `3.60 tok/s`.
- **Quality policy**: lower repetition, `ratio=0.40`, `unique=6/16`, but lower
  speed than required.
- RLLM tracked peak transient memory stayed flat at `1050689536` bytes for
  Llama 1B.

## Decision

failed

Reason: AIP quality did not reach the required 2x speedup over exact mode for
Llama 3.2 1B Instruct. The policy improves repetition compared to the raw speed
policy, but it is not fast enough.

Paper value:
- useful negative result for quality-aware sparse routing
- useful evidence that repetition telemetry catches collapse modes
- useful limitation: quality-preserving exact fallback can erase sparse speed
  gains

## Next Experiment

R19 should test balanced or adaptive AIP, especially `mlp_down` top-k sweeps and
runtime repetition-triggered fallback, instead of keeping `mlp_down` exact for
all quality-mode layers.
