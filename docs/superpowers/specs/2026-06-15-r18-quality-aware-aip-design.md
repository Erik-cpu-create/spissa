# R18 Quality-Aware AIP Design

Date: 2026-06-15
Owner: RLLM
Status: approved design

## Summary

R18 continues the RLLM experimental-speed path after R17 proved a positive
speed signal but exposed output degeneration. The next stage combines two
goals:

- rename and document the experimental sparse projection idea as
  **RLLM AIP: Activation-Indexed Projection**
- make the opt-in sparse path quality-aware before expanding the amount of
  approximate compute

R18 does not change the default exact-lowram runtime. Exact mode remains the
source of correctness. AIP remains a research mode under
`RLLM_EXPERIMENTAL_SPEED=1`.

## Background

R17 added an opt-in sparse MLP path for LLaMA-family decode. On Llama 3.2 1B
Instruct, top-k 128 improved decode throughput from `0.40 tok/s` to
`1.61 tok/s`, while RLLM tracked peak transient memory stayed flat. The same
run produced repetitive or low-quality text, so the result is useful as speed
evidence but not sufficient as a chat mode.

The practical R18 question is not whether sparse compute can move speed. R17
already answered that. R18 asks whether RLLM can keep part of that gain while
reducing the output degeneration enough to justify further work.

## Originality And Related Work

RLLM AIP must remain an original RLLM implementation.

The broad research area overlaps with sparse activation and sparse inference
work, including PowerInfer and TurboSparse. Those projects are related work,
not implementation sources.

RLLM AIP is distinct in target and implementation:

- RLLM AIP targets CPU-only low-RAM inference.
- RLLM AIP does not require GPU offload.
- RLLM AIP does not use a hot/cold neuron predictor cache.
- RLLM AIP keeps model weights unchanged by default.
- RLLM AIP currently selects activation dimensions directly from the live
  batch-1 projection input and uses RLLM-native streaming kernels.
- No source code, API shape, kernel layout, or benchmark harness is copied from
  llama.cpp, GGML, Ollama, PowerInfer, TurboSparse, or any other runtime.

The R18 docs should describe PowerInfer and TurboSparse only as related work in
the paper trail. Public references:

- PowerInfer paper: `https://arxiv.org/abs/2312.12456`
- PowerInfer repository: `https://github.com/SJTU-IPADS/PowerInfer`
- TurboSparse paper: `https://arxiv.org/abs/2406.05955`

## Goals

- Replace user-facing experimental terminology from "Turbo Sparse" toward
  "RLLM AIP" while keeping backwards-compatible env behavior for the first
  transition.
- Add an AIP policy layer that can choose exact or approximate projection by
  layer and projection type.
- Add a conservative `quality` policy that avoids sparse compute in the most
  quality-sensitive parts of the transformer.
- Add simple repetition telemetry so benchmark reports can compare output
  degeneration without relying only on subjective reading.
- Record R18 in the benchmark folder system with clear success or failure
  criteria.

## Non-Goals

- Do not make AIP default.
- Do not claim quality parity with exact mode.
- Do not add model compression or quantization.
- Do not add GPU, Metal, Accelerate, or external BLAS dependencies.
- Do not add sparse LM-head shortlist projection in R18 unless the design is
  revised. LM-head stays exact for token selection.
- Do not rewrite the `.rllm` artifact format or import pipeline in this stage.

## Mode Contract

The existing gate remains valid:

```bash
RLLM_EXPERIMENTAL_SPEED=1
```

R18 adds an AIP policy environment variable:

```bash
RLLM_AIP_POLICY=quality
```

Accepted values:

- `quality`: conservative quality-aware routing
- `speed`: R17-style aggressive AIP routing

If `RLLM_AIP_POLICY` is absent, R18 should default to `quality` when
`RLLM_EXPERIMENTAL_SPEED=1`. The old `RLLM_TURBO_TOPK` variable remains
accepted as a compatibility alias for the global top-k override, but new docs
should prefer:

```bash
RLLM_AIP_TOPK=128
```

Exact mode ignores all AIP settings when `RLLM_EXPERIMENTAL_SPEED` is off.

## AIP Policy

R18 introduces a small policy object owned by `speed.rs`. The policy answers
one question for each LLaMA projection:

Should this projection use exact compute or AIP compute for this layer and
projection kind?

Projection kinds:

- `mlp_gate_up`
- `mlp_down`

R18 keeps attention projections and LM-head exact.

Policy inputs:

- enabled flag
- policy kind: `quality` or `speed`
- layer index
- total layer count
- projection kind
- input width
- global top-k override, if provided

Policy output:

- exact
- AIP with selected top-k

## Quality Policy

The `quality` policy should be conservative:

- exact compute for early layers
- exact compute for final layers
- AIP only for middle layers
- AIP allowed for `mlp_gate_up`
- `mlp_down` should either stay exact or use a larger top-k than `mlp_gate_up`
  because R17 likely lost too much residual signal there

Initial layer window:

- exact first 25 percent of layers
- exact final 25 percent of layers
- AIP middle 50 percent of layers

Initial top-k:

- `mlp_gate_up`: `min(hidden_size, 128)` unless overridden
- `mlp_down`: exact by default in `quality`

This intentionally gives up part of R17's speed to measure whether quality
improves. If the benchmark is too slow but text is better, a follow-up stage can
enable AIP for `mlp_down` with a larger top-k sweep.

## Speed Policy

The `speed` policy preserves the R17 aggressive behavior under RLLM-native
names:

- AIP active for every eligible layer
- AIP active for `mlp_gate_up`
- AIP active for `mlp_down`
- top-k comes from `RLLM_AIP_TOPK`, `RLLM_TURBO_TOPK`, or the current default
  fallback

This gives R18 a direct R17 comparison mode and avoids hiding the previous
positive speed signal.

## Data Flow

1. `RamaExperimentalSpeedConfig::from_env()` reads
   `RLLM_EXPERIMENTAL_SPEED`, `RLLM_AIP_POLICY`, `RLLM_AIP_TOPK`, and the
   compatibility `RLLM_TURBO_TOPK` alias.
2. `LlamaRamaSessionAdapter` passes layer index and total layer count into each
   streaming transformer block.
3. `streaming_llama_transformer_block_with_timing` asks the AIP policy whether
   each MLP projection should use exact or AIP compute.
4. The existing sparse kernels remain the execution mechanism for AIP
   projections.
5. Unsupported dtype, shape, or chunk layout still falls back to exact compute.
6. CLI output reports AIP policy, sparse calls, fallbacks, max top-k, skipped
   multiply-add estimate, and repetition telemetry.

## Telemetry

R18 keeps the R17 sparse counters and renames the CLI label from
`ExperimentalSpeed` toward `AIP`.

Runtime telemetry:

- AIP policy name
- AIP projection calls
- exact fallbacks
- selected top-k sum
- max selected top-k
- estimated skipped multiply-add count
- peak scratch bytes

Generation quality telemetry:

- generated token count
- unique generated token count
- max repeated-token run
- repeated-token ratio

The repetition metrics live at session or CLI level, not inside the projection
kernel. They measure generated token IDs from a turn and should work for exact
and AIP modes.

## Error Handling

Unsupported AIP cases fall back to exact compute:

- non batch-1 decode
- unsupported dtype
- unsupported chunk layout
- empty activation vector
- invalid top-k
- policy exact decision

Shape corruption, tensor metadata corruption, impossible arithmetic overflow,
and invalid context growth remain hard errors.

## Testing

Unit tests:

- parse `RLLM_AIP_POLICY`
- parse `RLLM_AIP_TOPK`
- preserve `RLLM_TURBO_TOPK` compatibility
- policy selects exact early/final layers in `quality`
- policy selects AIP middle layers in `quality`
- policy selects AIP all eligible layers in `speed`
- repetition metrics detect unique tokens, repeated runs, and repeated ratio

Integration-style runtime tests:

- exact mode has no AIP activity
- `quality` mode produces fewer sparse calls than `speed` mode for the same
  toy LLaMA model
- existing sparse kernel tests still pass

Verification commands:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Benchmark Plan

Build once:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Primary Llama 3.2 1B runs:

```bash
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
```

SmolLM2 remains the small-model control:

```bash
printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=quality RLLM_AIP_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/SmolLM2-135M-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

The R18 report should compare:

- exact baseline
- R18 quality policy
- R18 speed policy
- R17 top-k 128 result as historical reference

## Success Criteria

R18 is a success if all are true:

- exact mode tests and behavior remain unchanged
- `quality` mode runs without increasing RLLM tracked peak transient memory
  beyond normal scratch variance
- Llama 3.2 1B `quality` mode remains at least 2x faster than the exact
  baseline in the same command shape
- repetition telemetry improves versus R17 top-k 128 on the same prompt
- CLI and docs use RLLM AIP terminology and clearly mark the path approximate

R18 is a strong success if Llama 3.2 1B `quality` mode reaches at least
`2.0 tok/s` while lowering repeated-token ratio against R17 top-k 128.

## Failure Criteria

R18 is failed or inconclusive if:

- exact mode changes
- AIP policy routing adds complexity without reducing repetition
- Llama 3.2 1B `quality` mode falls below 2x exact baseline
- memory rises enough to weaken the low-RAM claim
- telemetry is too vague to compare output degeneration

## Next Stages

If R18 succeeds:

- R19 can test AIP for `mlp_down` with larger top-k in quality mode.
- R20 can test sparse LM-head shortlist projection with strict fallback gates.
- A later stage can test packed AIP-friendly layout without changing default
  model artifacts.

If R18 fails:

- Keep the failure report.
- Keep `speed` policy only as a research comparison mode.
- Return to exact SIMD/packed raw 16-bit kernels or packed sparse layout as the
  next route.
