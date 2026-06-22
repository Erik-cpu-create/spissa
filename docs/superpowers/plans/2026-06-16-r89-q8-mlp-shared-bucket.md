# R89: Q8 MLP Shared Bucket Micro-Kernel

> **Constraint check:** continue with strict single-CPU intent. Focus solely on the most costly shared Q8 MLP operations during prefill, specifically targeting a micro-kernel approach rather than broad experiments.

## Goal

Design and implement a highly focused micro-kernel for the Q8 MLP projection to address the prefill bottleneck identified in R88. 

## Hypothesis

By consolidating the Q8 MLP dot-product and repacking work into a shared bucket micro-kernel, we can measurably decrease the prefill CPU time while preserving stable exact-lowram output semantics.

## Scope

- Mode: exact-lowram
- Component: Q8 MLP layers (`gate`, `up`, `down` projections)
- Target device/profile: Mac (CPU)
- Constraint: strictly avoid broad architectural refactoring; isolate the change to a new micro-kernel implementation for Q8 MLP.

## Task Checklist

- [ ] Implement the shared bucket micro-kernel for Q8 MLP in `crates/rllm-runtime/src/streaming/`.
- [ ] Validate implementation maintains baseline exactness (outputs `No`/`No.` as established in R88).
- [ ] Measure prefill latency with the new micro-kernel on `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`.
- [ ] Compare `prefill_total` and overall token processing speed against the R88 baseline.
- [ ] Add a dedicated trial doc in `docs/benchmarks/trials/active` and add a new row in `docs/benchmarks/trials/index.md`.
- [ ] Upon completion of the trial, move the doc from `active/` to either `docs/benchmarks/trials/success/` or `docs/benchmarks/trials/failed/` based on the gate results.
- [ ] Update the final status of the trial row in `docs/benchmarks/trials/index.md`.

## Deliverables

- `docs/benchmarks/trials/active/2026-06-16-r89-q8-mlp-shared-bucket.md` (initial)
- `docs/benchmarks/trials/success/` or `failed/` doc (final)
- Updated status row in `docs/benchmarks/trials/index.md`

## Gate

- **Must have:** A measurable reduction in prefill latency (`prefill_total`) while retaining correct output.
