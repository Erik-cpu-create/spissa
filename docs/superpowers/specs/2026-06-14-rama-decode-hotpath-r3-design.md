# R3 Design: RAMA Decode Hot-Path Profiling and Targeted Optimization

Date: 2026-06-14
Status: proposed
Owner: RLLM/RAMA

## Objective

Run the first R3 improvement pass against the token-native session path by
measuring decode subphases and applying one small optimization only when the
profile identifies a clear low-risk target.

R3B is an improve phase, but it is still evidence-led. The goal is to move from
the R2 finding, where persistent sessions reduced later-turn TTFT but decode
remained around 10 tok/s, toward a concrete decode-speed bottleneck map.

## Problem

R2 proved that token-native persistent sessions can avoid later-turn full
history replay, but decode throughput stayed mostly unchanged. That means the
next limiting factor is probably inside the per-token decode path rather than
chat/session replay mechanics.

Optimizing without subphase timings would make the result hard to trust. R3B
must identify where the token time is going before changing hot-path code.

## Non-Goals

- Do not rewrite the transformer block, matmul kernels, KV-cache layout, model
  container format, or quantization format in R3B.
- Do not add a new model-family adapter.
- Do not copy behavior or source code from external runtimes.
- Do not claim universal 30-40 tok/s from this phase.
- Do not keep an optimization if token IDs diverge from the R2 token-native
  baseline.
- Do not broaden the benchmark beyond the existing token-native LLaMA path
  unless the profiling result requires a small report-only field.

## Design

R3B extends the existing LLaMA session decode path with optional timing
breakdowns. The benchmark continues to use the R2 token-native command so
baseline/session token equality remains the source of truth.

The first pass records timing around these coarse subphases:

- embedding lookup
- transformer layers as a total
- final RMSNorm
- LM head projection plus sampling

After one measured run, choose one optimization only if the timing shows a
clear target. The expected low-risk candidate is the LLaMA session LM-head path:
the current adapter can allocate and fill a full logits vector for every emitted
token even though `RamaChatSession` only needs the sampled token ID for the
R2/R3 benchmark path.

If the profile shows the LM head and sampling path is material, R3B may add an
argmax/no-full-logits path or a reusable logits scratch buffer while preserving
the existing adapter contract. If the transformer layers dominate, R3B records
that result and stops without unrelated micro-optimization.

## Data Flow

The benchmark run stays the same as R2:

```text
turn tokens -> full-replay baseline
turn tokens -> persistent RamaChatSession
assert generated tokens match
assert visible token history matches
record timing and memory evidence
```

R3B adds a second evidence layer inside the session side:

```text
append_tokens(...)
  measure embedding lookup
  measure transformer layers
  measure final norm when emitting a token
  measure lm_head + sampling when emitting a token
  return sampled token through RamaSessionStep
```

The report must make it clear whether timing is pre-optimization or
post-optimization.

## Optimization Boundary

The safe optimization boundary is `crates/rllm-runtime/src/models/llama/session.rs`.
Any R3B optimization must preserve:

- `RamaSessionAdapter::append_tokens` transactional behavior
- identical generated token IDs for argmax sampling
- unchanged visible token history
- existing `emit_logits=false` behavior
- successful rollback on adapter errors

The CLI may expose timing in the Markdown report, but it should not own model
hot-path logic.

## Metrics

Record per turn:

- all R2 token-native fields
- session decode subphase milliseconds
- subphase share of session decode time when available
- pre-optimization and post-optimization decode tok/s if an optimization is
  applied
- token match and history match status
- memory notes for any new scratch allocation or removed allocation

## Benchmark Classification

R3B reports start in `docs/benchmarks/trials/active/`.

A report can move to:

- `success` when token histories match and the targeted optimization improves
  measured decode throughput or removes meaningful transient allocation without
  slowing decode.
- `failed` when token histories match but the optimization slows decode or
  increases memory enough to be counterproductive.
- `inconclusive` when token histories diverge, timing is incomplete, or the
  profile does not identify a defensible optimization target.

## Acceptance Criteria

- The R3B report includes decode subphase timing for the LLaMA session path.
- The benchmark still validates generated token IDs and visible history against
  the full-replay baseline.
- If an optimization is implemented, it is limited to one targeted hot-path
  change selected from the timing evidence.
- If no optimization is implemented, the report states the measured bottleneck
  and why R3B stopped there.
- Tests cover the new timing/optimization behavior without requiring a large
  model artifact.
- Verification passes:
  - `cargo fmt --check`
  - `cargo check --workspace`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo run --quiet -- doctor`

## Risks

- Timing instrumentation can distort small measurements. R3B keeps the buckets
  coarse and uses them to choose direction, not to claim cycle-level precision.
- The LM-head optimization may not matter if transformer layers dominate the
  workload.
- A faster path that skips full logits is only valid for argmax/no-logits
  benchmark usage; top-p or logits-consuming paths must keep their existing
  behavior.
- Two model handles in the token-native benchmark still inflate process RSS,
  so report memory interpretation must separate benchmark harness overhead from
  runtime session memory.

## Next Step

Review and approve this R3B spec, then write the implementation plan before
touching runtime or CLI code.
