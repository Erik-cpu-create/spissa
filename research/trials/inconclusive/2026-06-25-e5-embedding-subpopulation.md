# Research Trial: E5 — embedding row sub-populations (filament #3)

Date: 2026-06-25
Owner: REEBORN
Status: inconclusive
Folder: inconclusive

## Hypothesis

The token embedding (128256×2048 = 21% of params) is a mixture: rare/reserved tokens stay
near initialisation (~random), frequent tokens are heavily trained (distinct distribution).
Clustering rows by a trained-ness signal (row L2-norm, free) and coding each cluster with its
own table should beat the embedding's single-table order-0.

## Scope

- Experiment type: mixture-coding (embedding-specific)
- REE codec: REEBORN
- Lossless reference: vs fp/bf16, beating the embedding single-table floor
- Model/artifact + weight source: Llama-3.2-1B-Instruct embed_tokens (same source)
- Finding tag: redundancy source (real, negligible) + mechanism confirmation

## Method

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp5_embedding.py
```

Row L2-norm per token; 8 quantile clusters; H(exp|cluster) and H(u16|cluster) vs single table,
Miller-Madow + table cost + cluster-id cost (128256·log2(8) bits) subtracted. Also compares
reserved rows (id≥128000) vs BPE rows.

## Results

Embedding row-norm is tight: min 0.521, median 0.933, max 1.320 (max/median 1.4× — **not bimodal**).
Single-table floor: H(exp) 2.5849, H(u16) 10.4893 b/w.

| cluster (norm, K=8) | H_MM | gain | cost | NET (embed) | NET (overall) |
|---|---:|---:|---:|---:|---:|
| exp | 2.5739 | 0.0111 | 0.00148 | **+0.0096** | +0.0020 |
| u16 | 10.4765 | 0.0128 | 0.00394 | **+0.0089** | +0.0019 |

Reserved rows (id≥128000, 256 tokens, 0.20% of vocab): norm median **0.524 vs 0.933** (≈near-init),
H(exp) **2.84 vs 2.58** (more random ⇒ untrained). Hypothesis confirmed qualitatively.

## Analysis

- The mechanism is **real and confirmed**: untrained reserved-token rows are near-init with a
  distinct, higher-entropy distribution. But there are only 256 of them (0.04% of model params),
  so exploiting them saves nothing measurable.
- Among the 128000 BPE tokens, embedding norms are **uniformly trained** (tight, not bimodal), so
  row-norm clustering yields only +0.0096 b/w on the embedding = **+0.0020 b/w overall (0.02%)** —
  real (survives MM + table + id cost) but negligible.
- Same lever family as E3 (cluster→table); stacks an additional ~0.002 b/w. Would be materially
  larger on a model with many reserved/untrained tokens or heterogeneous embeddings (model-dependent).

## Decision

inconclusive

Reason: real sub-floor signal but negligible payoff on this densely-trained model. Mechanism
validated; magnitude under-powered (too few untrained rows).

Paper value: limitation + small positive (untrained-row = near-init mechanism confirmed; the
embedding-mixture win scales with the number of untrained/reserved tokens).

## Next Experiment

Filament #2 — learned permutation / BWT-style row reorder to manufacture local coherence the
natural order lacks. Or bank the exponent-clustering family (E3 + this) and reassess axes.
