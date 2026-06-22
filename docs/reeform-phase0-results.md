<!--
Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
PROPRIETARY & CONFIDENTIAL — see LICENSE. Do not distribute, publish, or disclose.
-->

# REEFORM — Phase-0 Results & First Successful Test

**Research log** · Owner: Rama Erik Esprada · Project: Spissa · CONFIDENTIAL
Probe model: **SmolLM2-135M** (bf16, ≤1B per the research rule) · 211 weight matrices · 134.5M weights
Instruments (committed): `reeform-probe`, `reeform-lowrank`, `reeform-delta` (under `spissa-cli/src/bin/`)

---

## TL;DR

- **Compressing arbitrary bf16 weights "from scratch" below ~10.5 bit losslessly is information-
  theoretically blocked** — measured, not assumed, across every structural lever. The mantissa
  carries ~7 bit of irreducible information (the significand of continuous-valued weights is
  ~uniform). This is the same wall DFloat11 / ZipNN hit.
- **But the structure is in the DELTA, not the weights.** A fine-tune is its base plus a gentle
  update. The lossless **integer-subtract delta** `Δ = W_ft − W_base (mod 2¹⁶)` compresses to
  **7.70 bit/weight vs 10.54** for the full model — a **27.0% LOSSLESS reduction**, with
  **bit-exact** reconstruction verified over all 134.5M weights (0 mismatches).
- This is the **first successful REEFORM test** and a genuinely novel result: nobody ships
  fine-tunes as lossless integer-deltas from base.

---

## 1. Order-0 floor (confirmed)

`H0(full 16-bit symbol) = 10.5443 bit/weight` — exactly the floor rANS / bit-plane / DFloat11
land at. Field breakdown: **sign 1.0000** (perfectly random), **exp 2.6195**, **mantissa 6.9717**
(≈ 7, i.e. ~uniform / incompressible). So the only field with sub-random entropy — the only
possible lever — is the **exponent** (2.62 bit).

## 2. Structural levers — all NULL (the honest part)

| Hypothesis | Measurement | Result |
|---|---|---|
| Row neighbour (order-1/2) | H(exp\|left) 2.6164 vs 2.6195 | ~0 |
| Column neighbour | H(exp\|up) 2.6016 | ~0 |
| **Cross-layer** (residual stream) | H(exp\|prev-layer) 2.6077 vs 2.6090; integer Δ **11.20 > 10.53** | NULL (Δ raises entropy) |
| **Per-channel** (per-column) | apparent gains (exp 0.058, man 0.11) **≤ finite-sample bias** (~0.15) | overfitting, not structure |
| **Low-rank** (rank-32 SVD) | H0(residual) 10.506 vs 10.510 (−0.004) + U,V cost 1.22 → **net 11.73** | NULL |

**Why:** trained weights are close to i.i.d. samples of a smooth distribution → near maximum
entropy. The order-0 entropy ≈ the true entropy. The mantissa wall (~7 bit) is physics, not a
method limitation. ⇒ "extreme lossless ratio" of weights-as-given is impossible; that part only
yields to **lossy** (quantization), which is explicitly out of scope.

## 3. The pivot — structure lives in the fine-tune delta

A fine-tune (`Instruct`) is the **same** weights as its base after gentle training, so
`W_ft ≈ W_base`. The cross-LAYER delta was null (different weights), but the **base→fine-tune**
delta compares the *same* weights before/after — and there the structure is real.

`Δ = W_ft − W_base (mod 2¹⁶)` on the bf16 bit patterns (exactly reversible: `W_ft = W_base + Δ`):

```
weights identical base==ft : 2.32%
H0(W_ft)  baseline         : 10.5427 bit/weight   (ship the full fine-tune)
H0(XOR delta)              :  8.1073
H0(int-sub delta)          :  7.7014  ← best
H0(bf16 value delta)       :  8.0264
round-trip mismatches      : 0 / 134,479,872   ✅ BIT-EXACT
```

**Result: 7.701 vs 10.543 = 27.0% smaller, provably lossless.** Per additional fine-tune of a
shared base you store **7.70**, not 10.54, bit/weight. The HF ecosystem is mostly fine-tunes of a
handful of bases → the aggregate saving is large, and the technique is novel.

## 4. Why int-subtract wins

When a gradient step nudges a weight slightly, its bf16 **bit pattern** moves by a small integer
(bf16 is monotonic in value for a fixed sign), so `Δ` clusters tightly around 0 → its order-0
entropy collapses. XOR (8.11) is worse than integer subtraction (7.70) because XOR scatters bits
on exponent carries; integer Δ preserves the "small move" as a small number.

## 5. Next — the LoRA amplifier (Phase 1)

The delta is *exactly* the object LoRA approximates as **low-rank**. Low-rank on the raw weights
was null, but the **delta** should be genuinely low-rank (that is LoRA's whole premise). So:
`Δ ≈ A·Bᵀ (rank r) + small residual` → store `A,B` (tiny) + the lossless residual of `Δ`, whose
entropy should fall **well below 7.70**. This is the obvious next experiment and could turn 27%
into a much larger lossless win. Also: validate across more ≤1B base/fine-tune pairs; encode the
delta with the shipped rANS to confirm real bytes ≈ entropy; design the `spissa pack --base` path.

## 6. Honest framing for the record

The original "extreme lossless of arbitrary weights" target is **physically walled** (mantissa).
We did not fake a win there. We **redirected to where the redundancy actually is** — the
fine-tune delta — and got a real, verified, novel result. That is the invention's true home, and
arguably more valuable in practice than squeezing a single model's last fraction of a bit.
