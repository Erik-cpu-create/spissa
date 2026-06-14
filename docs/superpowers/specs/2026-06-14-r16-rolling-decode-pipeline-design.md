# R16 Design: Rolling Decode Pipeline

Date: 2026-06-14
Status: proposed
Owner: RLLM/RAMA

## Objective

Turn the "wheel" idea into a valid runtime experiment: reduce CPU-only decode
friction by reusing the same execution machinery across repeated token work.

R16 should test whether RLLM can improve token/s by making hot-path work cyclic:
fixed buffers, stable work slots, and worker reuse where useful. The goal is to
move data and compute like a rolling mechanism instead of rebuilding scheduling
and scratch state for every small projection call.

## Problem

R15 showed that simply adding row-level projection threads is not enough. Peak
RLLM transient memory stayed flat, but default projection row parallelism
regressed decode speed. The likely cause is overhead at the wrong granularity:
scoped worker creation, synchronization, cache pressure, and memory bandwidth
contention can cost more than the parallel work saves.

The next experiment should keep the useful part of the idea, CPU work can be
distributed, but remove the main friction: repeated setup and unstable data
flow.

## Non-Goals

- Do not use the physical wheel analogy as proof of performance. It is only a
  design metaphor for cyclic reuse and reduced friction.
- Do not quantize, compress, prune, or change model weights in R16.
- Do not copy implementation from llama.cpp, PowerInfer, or other runtimes.
- Do not rewrite the full transformer block, model container, or importer.
- Do not add a new model-family adapter.
- Do not claim the 30-40 tok/s target unless the measured benchmark reaches it.
- Do not keep an optimization if token IDs diverge from the current baseline.

## Concept

The physical analogy maps to software in this way:

| physical mechanism | runtime mechanism |
|---|---|
| wheel reduces sliding friction | cache-friendly streaming reduces RAM traffic |
| bearing keeps rotation stable | persistent workers avoid repeated spawn cost |
| flywheel keeps momentum | reused scratch buffers avoid repeated allocation |
| gear changes force/speed tradeoff | chunk size, row blocks, and scheduling policy tune work granularity |
| load moves through repeated rotation | tensors move through fixed cyclic work slots |

The RLLM version is a rolling decode pipeline. Each emitted token repeats the
same kinds of work, so the runtime should reuse as much of the execution frame
as possible:

```text
prepare fixed work slots
prepare reusable scratch buffers
for each token:
  enqueue projection/chunk work into stable slots
  workers process slots without being recreated
  accumulators receive partial results
  slots and buffers rotate back for reuse
```

## Candidate Approaches

### A. Persistent Worker Wheel

Create a small reusable worker mechanism for one heavy hot path. Workers are
created once per session or execution context and receive repeated projection or
argmax work through fixed slots.

Tradeoff: this directly addresses the R15 failure mode, but it introduces
thread-lifetime and error-propagation complexity.

### B. Circular Scratch Arena

Start with single-threaded cyclic reuse: allocate fixed scratch buffers once and
rotate them through projection, MLP, attention, and LM-head work.

Tradeoff: lower risk and simpler correctness, but it may not improve speed if
the current code already avoids most repeated allocation.

### C. Packed Projection Wheel

Change import/runtime layout for selected projection weights so the hot path
reads cache-friendly row blocks repeatedly.

Tradeoff: potentially higher performance ceiling, but it touches model packing,
format compatibility, and migration scope.

## Recommended R16 Scope

R16 should start with Approach A as a narrow experiment, while keeping Approach B
as a fallback if thread lifetime proves too invasive.

The first implementation target should be one measurable path only:

- preferred target: raw BF16 LM-head argmax or projection row work that already
  has correctness tests
- fallback target: a reusable scratch arena around the existing streaming
  projection path

This keeps R16 small enough to accept or reject with benchmark evidence. It also
preserves the low-RAM contract: workers may own tiny local state, but must not
duplicate model tensors, KV cache, or full logits buffers.

## Architecture

Add a small runtime-owned component, tentatively named `RollingExecutor`, behind
the existing streaming/runtime boundary.

Responsibilities:

- detect a safe worker count from the existing CPU policy
- create workers once for the executor lifetime, not inside each projection call
- accept work in fixed-size task slots
- return deterministic partial results to the caller
- fall back to the current sequential path when the work is too small
- expose counters for tasks submitted, worker waits, sequential fallbacks, and
  peak scratch bytes

The first version should stay private to `rllm-runtime`. Public CLI behavior
does not change except for optional benchmark/report fields.

## Data Flow

Current rejected R15 shape:

```text
projection call
  split rows
  create scoped workers
  compute
  join workers
next projection call repeats setup
```

R16 target shape:

```text
session/runtime initializes RollingExecutor once
for each decode step:
  projection call
    fill stable work slots
    wake existing workers or run sequential fallback
    collect deterministic accumulators
    rotate slots back into reuse pool
```

The caller still owns model correctness. The executor only schedules row/chunk
work and returns the same numerical result as the existing path.

## Metrics

Record for SmolLM2-135M and Llama 3.2 1B Instruct:

- TTFT/prefill
- decode tok/s
- end-to-end tok/s
- max RSS from `/usr/bin/time -l`
- RLLM peak transient bytes
- peak footprint when available
- worker count and policy
- submitted rolling tasks
- sequential fallback count
- token equality against the baseline path

R16 must compare against the latest accepted default runtime, not against the
failed R15 experimental code.

## Benchmark Classification

R16 reports start in `docs/benchmarks/trials/active/`.

A report can move to:

- `success` when token IDs match, peak RLLM transient memory stays flat or
  improves, and decode tok/s improves over the current default on at least
  SmolLM2 without hurting Llama 1B.
- `failed` when token IDs match but decode regresses, memory grows materially,
  or worker overhead remains larger than the benefit.
- `inconclusive` when token IDs diverge, measurements are too noisy, or the
  implementation cannot isolate the rolling mechanism from unrelated changes.

## Acceptance Criteria

- R16 has a focused benchmark report with success, failed, or inconclusive
  classification.
- The implementation, if attempted, changes only one hot-path scheduling
  mechanism.
- Token IDs match the baseline for tested generation paths.
- RLLM does not duplicate model tensors, KV cache, or full logits per worker.
- The default path falls back to sequential execution for tiny work.
- Tests cover deterministic equality for the rolling path without requiring a
  large model artifact.
- Verification passes:
  - `cargo fmt --check`
  - `cargo check --workspace`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo run --quiet -- doctor`

## Risks

- Persistent workers can add synchronization overhead and make small models
  slower.
- A worker executor can complicate rollback/error behavior if it crosses the
  wrong module boundary.
- macOS scheduler behavior may differ from Linux low-end devices, so benchmark
  notes must record host assumptions.
- The largest Llama 1B bottleneck may remain scalar BF16 dot throughput rather
  than worker setup. If so, R16 should fail cleanly and point R17 toward SIMD or
  packed layouts.

## Next Step

Review and approve this R16 spec, then write the implementation plan before
touching runtime code.
