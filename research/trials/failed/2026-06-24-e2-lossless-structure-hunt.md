# Research Trial: E2 — true-lossless structure hunt (the ratio gamble)

Date: 2026-06-24
Owner: REEBORN
Status: rejected
Folder: failed

## Hypothesis

REEFORM proved standalone raw-weight true-lossless is walled at order-0 ~10.54 b/w on
SmolLM2 + Qwen2.5, all structural levers null. Gamble: a lever REEFORM did NOT explicitly
try on raw weights — a 2-D joint exponent context, or a coarse bias-safe per-channel class —
might crack the wall on Llama-3.2-1B and let REEBORN beat REEFORM on standalone ratio.

## Scope

- Experiment type: entropy-floor / structure-hunt
- REE codec: REEBORN (design input)
- Lossless reference: vs fp/bf16 (standalone, no base)
- Model/artifact + weight source: Llama-3.2-1B-Instruct (same source as E0/E1)
- Finding tag: information-theoretic limit (null) + methodology
- Prior art: REEFORM phase-0 (docs/reeform-phase0-results.md); ZipNN, DFloat11 (mantissa wall), BitDelta (delta framing).

## Method

```bash
/tmp/reeborn-venv/bin/python research/reeborn/exp2_lossless_structure.py
```

Probes REEFORM did not explicitly run on raw weights: `exp|(left,up)` 2-D joint, `exp|colclass8`
and `exp|rowclass8` (coarse 8-bucket per-channel magnitude classes — the bias-safe form of
REEFORM's per-column lever), `mant|exp`, plus reproductions (`exp|left/up`, `mant|left`,
`sign|left`). **Every conditional entropy is Miller-Madow bias-corrected**; a gain counts only
if it survives. numpy 2.5.0 (venv); macOS arm64; ~5 min.

## Results

True order-0 floor reproduced: **H(u16) = 10.5483 b/w** (REEFORM got 10.5443 on SmolLM2 →
cross-architecture confirmation). Fields: sign 1.0000, exp 2.6305, mant 6.9644 (sum 10.5949).

| lever | kind | H_MM | marg | gain_MM | verdict |
|---|---|---:|---:|---:|---|
| exp \| left | cross | 2.6243 | 2.6305 | 0.0062 | null |
| exp \| up | cross | 2.6255 | 2.6305 | 0.0050 | null |
| exp \| (left,up) | cross | 2.6199 | 2.6305 | 0.0106 | suspect (high-context residual bias) |
| exp \| colclass8 | cross | 2.6236 | 2.6305 | 0.0069 | null |
| exp \| rowclass8 | cross | 2.6103 | 2.6305 | 0.0203 | real but tiny |
| mant \| exp | within | 6.9178 | 6.9644 | 0.0466 | SUBSUMED by u16 |
| mant \| left | cross | 6.9644 | 6.9644 | 0.0000 | null |
| sign \| left | cross | 1.0000 | 1.0000 | 0.0000 | null |

Best cross-weight gain below the u16 floor = **0.0203 b/w** → achievable 10.528 vs 10.548 = **0.19%**.

## Analysis

1. **Wall reproduced on a 3rd architecture.** H(u16) = 10.548 ≈ REEFORM's 10.544. The mantissa
   (6.96/7) is the irreducible barrier, confirmed for Llama-arch.
2. **The script's auto-verdict initially printed "CRACK FOUND" — a FALSE ALARM, caught.** It
   compared the field-separated sum (10.595) against the achievable, but the true floor is the
   joint-symbol H(u16) = 10.548. The `mant|exp` "gain" (0.0466) is exactly `I(exp;mant)` =
   field_sum − H(u16); coding the joint u16 symbol (which REEFORM/rANS do) already captures it.
   It is **not** a new lever. (This is the same finite-sample/baseline trap REEFORM documented;
   Miller-Madow + using the joint-symbol baseline caught it. Script verdict corrected.)
3. **Genuine sub-floor structure is negligible.** Only cross-weight exponent context goes below
   H(u16): the best is `exp|rowclass8` ≈ 0.020 b/w (caveat: classes fit in-sample; coarse
   8-bucket so MM-bias-safe, but worth a held-out check). `exp|(left,up)` 0.011 is suspect
   residual bias from 65536 contexts. Net achievable ≈ 0.19% below the floor — real but practically
   nil. 1-D `exp|left/up` (~0.005) match REEFORM's null.

## Decision

rejected

Reason: standalone true-lossless ratio is walled for Llama-arch too (3rd architecture). The
ratio gamble did not pay (~90% null as predicted). The single surviving lever (rowclass8,
0.02 b/w = 0.2%) does not let REEBORN meaningfully beat REEFORM on standalone ratio.

Paper value: strong negative/null evidence (cross-architecture wall confirmation) + methodology
note (joint-symbol baseline and Miller-Madow are both required to avoid phantom gains).

## Next Experiment

Pivot off the standalone-ratio axis (proven walled). Two live options for "beat REEFORM":
- **Speed/RAM axis** — a custom lossless codec hitting the 10.5 b/w floor with the fastest
  ARM/NEON decode + lowest RAM (REEFORM optimized ratio, never decode speed; this is spissa's
  edge mission). Profile REEFORM/rANS decode → REEBORN kernel design.
- **Delta / base-conditioned** — realize + extend REEFORM's base-exponent-conditioned delta
  coder (7.10 b/w), hunt a 2nd conditioning signal (needs a base+fine-tune pair).
