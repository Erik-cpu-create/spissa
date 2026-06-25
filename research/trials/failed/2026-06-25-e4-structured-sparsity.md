# Research Trial: E4 — structured-sparsity / near-zero sub-population (filament #5)

Date: 2026-06-25
Owner: REEBORN
Status: rejected
Folder: failed

## Hypothesis

Order-0 already codes near-zero weights via their marginal frequency, so a near-zero
sub-population only beats order-0 if it is STRUCTURED — spatially clustered (runs) or whole
dead rows/cols — letting a sparse / run-length code describe the mask cheaply. Gamble: dense
Llama-1B has exploitable structured sparsity (e.g. dead neurons).

## Scope

- Experiment type: structure-hunt (sparsity)
- REE codec: REEBORN
- Lossless reference: vs fp/bf16, beating order-0
- Model/artifact + weight source: Llama-3.2-1B-Instruct (same as E0–E3)
- Finding tag: null result

## Method

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp4_sparsity.py
```

Near-zero = exponent below the ~3rd-percentile threshold (T=116). Measured: near-zero +
exact-zero fractions, spatial clustering ratio P(nz|left nz)/P(nz), mean near-zero run length
vs the i.i.d. geometric expectation, and dead rows/cols (≥90% near-zero).

## Results

| metric | value |
|---|---|
| near-zero fraction p | 2.258% |
| exact ±0 fraction | 0.0000% |
| P(nz \| left nz) | 2.438% (marginal 2.258%) |
| clustering ratio | **1.080** (1.0 = i.i.d.) |
| mean near-zero run length | 1.025 (i.i.d. = 1.023) |
| dead rows (≥90% nz) | **0 / 505,088** |
| dead cols (≥90% nz) | **0 / 329,728** |

## Analysis

Near-zero weights are scattered i.i.d.: clustering ratio 1.08 ≈ 1.0, run length matches the
i.i.d. geometric expectation, and there are **zero** dead rows or columns and **zero** exact
zeros. Llama-1B is densely trained with no pruning, so there is no structured sparsity. Order-0
is already optimal for this marginal → a sparse / run-length / mixture code adds ~0 bits.

## Decision

rejected

Reason: no structured sparsity in a dense model → mixture/sparse coding is null beyond order-0.

Paper value: negative/null evidence — dense LLM weights have no exploitable structured sparsity
(distinct from pruned/MoE models, where this filament might relight).

## Next Experiment

E5 — embedding rows split by token frequency (rare tokens ≈ near-init/random vs frequent ≈
trained → distinct sub-populations over 21% of params). Then permutation/BWT reorder.
