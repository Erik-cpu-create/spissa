# Research Trial Log (offline analysis)

Canonical home for REEBORN / REEFORM and other **information-theoretic and offline
analysis** experiments — entropy floors, quantization rate/quality tradeoffs, codec
prototypes — run as Python (numpy) measurements over model weights.

This is the **research/analysis** counterpart to the **runtime-benchmark** trial log at
`docs/benchmarks/trials/`. The two are deliberately separate because they measure
different things and must not be conflated:

| | this log (`research/trials/`) | runtime log (`docs/benchmarks/trials/`) |
|---|---|---|
| measures | bits/weight, entropy, SQNR, compression ratio | tok/s, TTFT/prefill, RSS, peak memory |
| runs on | model weights, offline | the spissa runtime, on host/device |
| language | Python + numpy (venv) | Rust (`cargo`, `target/release`) |
| id prefix | `eNN` (E0, E1, …) | `rNN` (R83, …) |

A measurement here is **not** a runtime claim. When an analysis result is promoted into a
real Rust codec/kernel, it must be benchmarked under `docs/benchmarks/trials/` with its REE
name before any runtime/speed/size claim is made about the shipped artifact.

## Folders

- `active/` — planned or running experiments.
- `success/` — accepted, with measured supporting evidence.
- `failed/` — rejected ideas, negative **or null** results (keep them — they prevent dead ends).
- `inconclusive/` — mixed or insufficient evidence.

## Naming

`YYYY-MM-DD-eNN-short-topic.md`, e.g. `2026-06-24-e0-entropy-floor-bf16-vs-quant.md`.

Use `templates/research-trial.md` as the starting shape. Update `index.md` on every add/move.

## Minimum evidence

- hypothesis, with the information-theoretic reasoning behind it
- REE codec lineage when relevant (`REEBORN-*`, `REEFORM-*`)
- model/artifact + **weight source path**, dtype, param count
- exact reproducible command (committed script under `research/<line>/` + the venv)
- runtime context (python / numpy version, host) — for reproducibility, NOT as a perf claim
- metrics: bits/weight, entropy (+ components), SQNR / quality, compression ratio
- **finding tag**: `information-theoretic limit` | `redundancy source` | `quantization tradeoff` | `null result` | `codec validation`
- results table, analysis, decision (accept/reject/inconclusive), paper value, next experiment

## Rules

- No compression claim without before/after numbers AND an explicit "lossless vs **what**" baseline.
- Always state the reference for "lossless": vs fp32/bf16, vs a q4/q3 checkpoint, or behavioural.
- Keep failed and **null** results — a measured dead end (e.g. zero spatial redundancy) is high value.
- Keep lossless-vs-fp results separate from lossless-vs-quantized results.
- Name every serious codec candidate with the REE lineage before promoting it to Rust.
- Cite external papers as prior art when they inform a hypothesis.
- Do not copy source code, formats, or implementation structure from other projects.
- Analysis scripts live under `research/<line>/` and must be re-runnable from the committed venv.
