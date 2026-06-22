# R17 Experimental Speed Mode Design

Date: 2026-06-14
Owner: RLLM
Status: approved design

## Summary

R17 introduces an opt-in experimental speed mode for LLaMA-family chat decode.
The first strategy is **Turbo Sparse Decode**: keep the original model artifact
and weights intact, but reduce per-token CPU work by projecting from the most
dominant activation dimensions first.

This is not the default runtime path. The exact low-RAM path remains unchanged.
R17 is a research path for measuring whether activation-guided sparse compute
can move RLLM toward the 30-40 token/s target on CPU-only low-end devices.

## Research Intent

The current RLLM evidence shows that Llama 3.2 1B Instruct is dominated by raw
BF16 projection cost. R8, R10, R15, and R16 moved specific pieces, but the large
remaining buckets are still MLP gate/up, MLP down, and LM-head projection.
Thread count alone did not solve the problem because small per-call parallelism
adds scheduling and cache pressure.

R17 tests a different idea: for batch-1 decode, the activation vector is not
uniformly important. If a small set of activation dimensions contributes most
of the magnitude, RLLM can compute an approximate projection using only those
dimensions. This should reduce scalar BF16 conversion and multiply-add work
without changing the stored model.

## Originality Guard

R17 must be implemented from scratch inside RLLM.

- Do not copy source code from llama.cpp, GGML, Ollama, PowerInfer, or other
  runtimes.
- External work may be used only as conceptual background, not as code or
  API design.
- The implementation should use RLLM-native names, modules, benchmarks, and
  trial reports.
- The paper trail must label this path as RLLM experimental-speed research,
  distinct from exact-lowram inference.

## Mode Contract

Experimental speed mode is opt-in through an environment gate:

```bash
RLLM_EXPERIMENTAL_SPEED=1
```

Default behavior remains exact. When the gate is off, token output and memory
behavior must match the current exact-lowram path.

When the gate is on:

- RLLM may use approximate projection kernels.
- Generated tokens may differ from exact mode.
- Model weights must not be compressed, quantized, or rewritten by default.
- RAM usage must still be measured and reported.
- Trial reports must use mode label `experimental-speed`.

## Turbo Sparse Decode V1

Turbo Sparse Decode V1 targets LLaMA batch-1 decode.

For each token step, RLLM builds a compact activation index set:

1. Read the current projection input vector.
2. Select the top activation dimensions by absolute magnitude.
3. Use that index set to compute selected projection rows by summing only those
   dimensions.
4. Fall back to exact projection when shapes, dtypes, or chunk layout are not
   supported.

The selector must be deterministic for a fixed input vector and configuration.
Tie-breaking uses lower dimension index first.

## First Kernel Target

R17 starts with the MLP path because prior benchmarks show it is the largest
Llama 1B transformer bucket:

- fused `gate_proj` / `up_proj`
- `down_proj`

LM-head sparse/shortlist projection is intentionally left for a later R stage.
The first R17 question is whether sparse projection can cut transformer time
enough to justify expanding the technique.

## Configuration

Initial configuration is environment-based to keep CLI churn low:

```bash
RLLM_EXPERIMENTAL_SPEED=1
RLLM_TURBO_TOPK=256
```

If `RLLM_TURBO_TOPK` is absent or invalid, RLLM uses a conservative default
derived from hidden size:

- `min(hidden_size, 256)` for hidden-size inputs
- `min(intermediate_size, 512)` for intermediate-size inputs

The values are intentionally small for the first trial. Follow-up stages can
sweep top-k values after the kernel is measurable.

## Data Flow

The LLaMA session adapter owns the mode decision.

1. `LlamaRamaSessionAdapter` reads the experimental speed config at creation.
2. The config is passed into each streaming transformer block.
3. The block requests sparse MLP kernels when the mode is enabled.
4. Kernels return exact fallback markers when unsupported.
5. Session metrics expose sparse calls, fallbacks, selected top-k, and scratch
   memory.
6. The CLI prints those metrics in benchmark output.

This keeps the exact path independent and makes the experimental path easy to
remove or disable if a trial fails.

## Error Handling

Unsupported cases must fall back to exact kernels rather than panic:

- non batch-1 decode
- non raw BF16/FP16 tensors
- chunk layouts that split rows in unsupported ways
- top-k larger than input size
- empty activation vectors

Shape corruption, invalid tensor metadata, and impossible arithmetic overflow
remain hard errors.

## Telemetry

R17 adds experimental speed telemetry:

- enabled/disabled
- sparse projection calls
- exact fallbacks
- selected top-k
- estimated skipped multiply-add count
- peak sparse scratch bytes

The benchmark report must include the telemetry so failed trials still explain
why speed did or did not move.

## Benchmark Plan

Primary command shape:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=256 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.spsa \
    --ctx 2048 \
    --max-new-tokens 16
```

Control runs must include exact-lowram baseline without
`RLLM_EXPERIMENTAL_SPEED`.

R17 should record both SmolLM2-135M and Llama 3.2 1B Instruct, but the primary
decision target is Llama 3.2 1B Instruct.

## Success Criteria

R17 is a success if all are true:

- exact mode remains unchanged and tests pass
- experimental mode runs without increasing RLLM tracked peak transient memory
  by more than the sparse scratch budget
- Llama 3.2 1B Instruct improves decode speed by at least 2x over the exact
  baseline in the same command shape
- benchmark documentation clearly marks output as approximate

R17 is a strong success if Llama 3.2 1B reaches 5-10 token/s. The 30-40 token/s
target is not expected from V1 alone; it likely needs sparse MLP plus sparse
LM-head and a packed sparse access layout in later stages.

## Failure Criteria

R17 is failed or inconclusive if:

- exact mode behavior changes
- sparse mode increases memory enough to violate the low-RAM direction
- speed improves less than 2x on Llama 3.2 1B
- output degenerates into immediate invalid token loops in simple chat prompts
- the fallback path dominates so the sparse kernel is not actually measured

## Risks

Sparse projection may lose too much model signal and produce poor text. That is
acceptable for this stage if the trial is clearly labeled approximate, but it
should not be merged into default inference.

Sparse row access may reduce arithmetic but hurt cache locality. If this occurs,
the next stage should test packed sparse-friendly layouts rather than increasing
thread count.

The top-k selector itself must stay cheap. If selector time is visible in the
profile, R17 should reduce selector frequency or use a simpler threshold pass.

## Next Stages

If R17 produces a strong signal:

- R18: add sparse LM-head shortlist projection.
- R19: add packed sparse access layout during import or runtime preparation.
- R20: combine sparse compute with persistent worker scheduling.

If R17 fails:

- Keep the failure report.
- Return to exact SIMD/packed BF16 kernels as the next practical route.
