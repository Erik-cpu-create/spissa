# Research Trial: E3 — mixture / clustered-table coding (filament #1)

Date: 2026-06-25
Owner: REEBORN
Status: accepted
Folder: success

## Hypothesis

REEFORM / R152 quoted the standalone lossless floor (~10.54 b/w) using ONE global frequency
table. Different layers have different weight-scale distributions, so coding each cluster
(tensor / layer-type / depth) with its OWN exponent table should go BELOW the global floor.
The gain = I(context; exponent), genuinely sub-floor, and an exponent table is ~free (few
active values per context). Nobody in-repo measured per-cluster tables.

## Scope

- Experiment type: structure-hunt / codec-design (mixture coding)
- REE codec: REEBORN
- Lossless reference: vs fp/bf16 (standalone), beating the GLOBAL-table order-0 floor
- Model/artifact + weight source: Llama-3.2-1B-Instruct (same as E0–E2)
- Finding tag: redundancy source (real, small)
- Prior art: REEFORM phase-0 + R152 rtc-rans-v1 (both global-table); this is the per-context refinement.

## Method

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp3_mixture.py
```

Per-tensor exponent + u16 histograms; aggregate to layer-type {embed, q,k,v,o, gate,up,down}
and to depth (layer index). H(exp|context) and H(u16|tensor), **Miller-Madow corrected AND
minus the per-context table storage cost** (16 bits/active entry). A surviving NET gain is real.

## Results

Global-table floor: H(exp) = 2.6305, H(u16) = 10.5483 b/w.

| context | #ctx | H_MM | gain | tbl_cost | NET gain | verdict |
|---|---:|---:|---:|---:|---:|---|
| exp \| tensor | 113 | 2.6009 | 0.0296 | 0.00004 | **+0.0296** | real win |
| exp \| layer-type | 8 | 2.6050 | 0.0255 | 0.00000 | **+0.0255** | real win |
| exp \| depth | 17 | 2.6270 | 0.0035 | 0.00001 | +0.0035 | null |
| u16 \| tensor | 113 | 10.5138 | 0.0345 | 0.0067 | **+0.0278** | win |

best NET exponent win = 0.0296 → new floor ≈ **10.5187 vs 10.5483** b/w.

## Analysis

- **A genuine sub-floor win.** Per-tensor / per-layer-type exponent tables beat REEFORM's
  global-table floor by ~0.03 b/w, surviving Miller-Madow AND model cost. Mechanistically sound:
  different layers have different weight-scale (exponent) distributions, so one global table is
  suboptimal. This is the first TRUE positive in the structure hunt (E2 was null / a false alarm).
- **Granularity saturates fast:** per-tensor (113 ctx) barely beats layer-type (8 ctx) — 0.0296 vs
  0.0255. So **layer-type (8 tables, zero meaningful cost) is the sweet spot** for the codec; finer
  context hits diminishing returns + bias (cf. E2 per-column).
- **depth is null** — weight scale is set by the projection role (q/k/v/o/gate/up/down/embed), not
  by layer depth.
- **Honest magnitude: ~0.3%** (10.548 → 10.519 b/w). Real but small — capped because the exponent is
  only 2.63 of 10.5 bits and we squeezed ~0.03 of it; the mantissa mass stays incompressible. Still a
  free keeper for the REEBORN codec.

## Decision

accepted

Reason: first confirmed lever that beats the global-table floor, bias+cost-corrected. Adopt
per-layer-type exponent frequency tables (8 tables, ~free) in the REEBORN codec → ~0.03 b/w below
REEFORM's standalone number, lossless.

Paper value: positive evidence (small but methodologically clean novel result — per-cluster tables
beat the single-global-table floor that prior in-repo measurements assumed).

## Next Experiment

Keep firing filaments (Edison): stack candidates and hunt for a brighter one —
- near-zero / magnitude sub-population split (mixture of a compressible tail + dense bulk),
- learned permutation / BWT-style reorder to manufacture local coherence,
- embedding rows split by token-frequency (21% of params, distinct sub-population).
Each is measured honestly; small wins stack, and a mantissa-structure find (unlikely but high-payoff)
would be the jackpot.
