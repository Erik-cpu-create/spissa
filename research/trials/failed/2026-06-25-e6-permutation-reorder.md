# Research Trial: E6 — permutation / BWT-style reorder (filament #2)

Date: 2026-06-25
Owner: REEBORN
Status: rejected
Folder: failed

## Hypothesis

Order-0 entropy is permutation-invariant, so a reorder only helps paired with a context coder
AND if it manufactures local coherence the natural order lacks. Sorting rows/cols by mean
exponent should make neighbours similar → H(exp|neighbour) drops. Gamble: the drop exceeds the
cost of storing the permutation (log2(R!) bits).

## Scope

- Experiment type: structure-hunt (reorder)
- REE codec: REEBORN
- Lossless reference: vs fp/bf16, beating natural-order + the free-clustering alternative
- Model/artifact + weight source: Llama-3.2-1B-Instruct (same source)
- Finding tag: information-theoretic limit + null result

## Method

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp6_permutation.py
```

Sort rows by mean-exp (vertical neighbours) and cols by mean-exp (horizontal); measure
H(exp|neighbour) natural vs sorted, minus the permutation storage cost log2(R!)/N (resp. C).
Plus the fine-grain bound log2(N!)/N (sorting all weights).

## Results

H(exp) = 2.6305 b/w (order-0, permutation-invariant).

| scheme | H(exp\|nbr) | gain | perm_cost | NET | verdict |
|---|---:|---:|---:|---:|---|
| natural up | 2.6255 | 0.0050 | 0.0000 | 0.0050 | — |
| row-sorted up | 2.6206 | 0.0099 | 0.0049 | 0.0050 | = natural (zero benefit) |
| natural left | 2.6243 | 0.0062 | 0.0000 | 0.0062 | — |
| col-sorted left | 2.6201 | 0.0104 | 0.0028 | 0.0076 | +0.0014, dominated by E3 |

**Fine-grain bound:** sorting all 1236M weights costs log2(N!)/N = **28.76 b/w** — 11× the entire
2.63 b/w exponent budget.

## Analysis

- **Mechanism confirmed:** sorting roughly doubles the raw neighbour gain (row 0.0050→0.0099, col
  0.0062→0.0104) — reordering really does manufacture local coherence.
- **Economics dead:** the permutation storage cost eats it. Row-sort net (0.0050) equals the natural
  order (zero benefit); col-sort net (0.0076) is +0.0014 over natural — negligible, and both are
  **strictly dominated by E3's free per-layer-type clustering (~0.026 b/w, no permutation)**.
- **Fine-grain is mathematically hopeless:** element-level permutation (where large context gains
  would live) costs 28.76 b/w to store — 11× the exponent budget. You cannot manufacture exploitable
  order in i.i.d. data for less than you save. This is a clean impossibility bound.

## Decision

rejected

Reason: coarse permutation is dominated by free clustering; fine permutation is mathematically
impossible (storage cost 28.76 ≫ 2.63 budget). The mechanism is real but the economics never close.

Paper value: negative evidence + a crisp impossibility bound (log2(N!)/N = 28.76 b/w) for
element-level reordering of near-i.i.d. weights.

## Next Experiment

Standalone-lossless filament hunt is now thorough (E2–E6: one small win E3 +0.03 b/w, the rest
null/dominated). Bank E3's per-layer-type exponent tables into the REEBORN codec, and pivot to the
big-win axes: a different OBJECT (delta/base, REEFORM territory) or a different AXIS (decode speed,
>RAM streaming).
