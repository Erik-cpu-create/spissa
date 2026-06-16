# R88: RLLM vs Ollama Parity Closure Plan (CPU prefill focus)

> **Constraint check:** continue with strict single-CPU intent. This stage is about benchmark falsifiability and eliminating comparison artifacts before any invasive kernel redesign.

## Goal

Establish a reproducible, apples-to-apples prefill benchmark between RLLM (`q8_transformer_keepio-rowchunks`) and Ollama (`llama3.2:1b`) on identical input semantics, then either:

1) attribute the remaining latency gap to template/thread mismatch, or  
2) confirm a true arithmetic bottleneck in RLLM.

## Hypothesis

The perceived gap is currently inflated by runtime framing differences (template/token packing and CPU thread configuration).  
After alignment, measured latency should still show a gap, but the decision for further Q8 kernel work should use aligned metrics.

## Scope

- exact-lowram Q8 execution
- prefill-only emphasis (`context token` accounting and `prefill_total` in `--profile-phases`)
- single-process single-device experiments

## Task Checklist

- [x] Capture current best RLLM prefill baseline with `--chat-template llama3`, `--rama-integrity unchecked`, 3 runs.
- [x] Capture `RLLM_THREADS=1` sanity mode separately (same command set, 3 runs).
- [x] Capture Ollama `/api/chat` with `num_gpu:0`, `num_ctx:2048`, `temperature:0`, 3 runs; keep server config/threads from runner logs in trial note.
- [x] Compute apples-to-apples metric:
  - `prefill_ms / prompt_eval_count`
  - `prompt_eval_count` and `context token` side-by-side
- [x] Add a dedicated trial doc in `docs/benchmarks/trials/active` and update `docs/benchmarks/trials/index.md`.
- [x] If gap remains and is unchanged after this normalization, add one focused micro-kernel trial for Q8 shared MLP buckets only.

## Deliverables

- `docs/benchmarks/trials/active/2026-06-16-r88-rllm-vs-ollama-token-alignment.md`
- updated row in `docs/benchmarks/trials/index.md`

## Gate

- **Must have:** all commands, outputs, and computed normalization in docs.
- **Go/No-Go:** If normalized gap is not reduced enough to justify immediate kernel refactor, continue with trace-guided low-risk improvements.
