<!--
Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
PROPRIETARY & CONFIDENTIAL — see LICENSE. Do not distribute, publish, or disclose.
-->

# REEFORM — Structural Lossless Codec for LLM Weights

**PRD / Research Charter** · v0.1 (draft) · Owner: **Rama Erik Esprada** · Project: **Spissa**
Status: **research, not yet started** · Classification: **CONFIDENTIAL — proprietary IP**

> `REEFORM` is a working codename (REE-lineage + *transform/reform*). Final name is decided
> if/when the method proves out.

---

## 0. One-sentence mission

> Invent a **brand-new lossless compression methodology for LLM weights** that compresses
> **below the entropy floor of every existing method** — by modeling structure that today's
> codecs are blind to — with **bit-identical reconstruction**, our own name, and our own paper.

This is a *from-scratch invention* effort, not an adaptation of an existing technique
(quantization, GPTQ/AWQ, rANS, bit-plane, DFloat11, ZipNN are all explicitly NOT what we are
doing — they are the bar we intend to beat).

---

## 1. The honest reframe — why this is open, not "impossible"

A naïve objection: *"Shannon says you can't compress below a source's entropy, and LLM bf16
weights sit at ≈10.6 bit/weight, so ≈10.6 bit is the wall."* **That objection is wrong, and
understanding why is the whole opportunity.**

- Shannon bounds compression below the **true entropy of the source under a given model of the
  data**. The entropy is **not a fixed physical constant** — it is a function of *how cleverly
  you model the data*. A better model ⇒ lower measured entropy ⇒ smaller lossless output.
- The ≈10.6 bit figure is the result of **order-0 entropy coding** — rANS / bit-plane / DFloat11
  / ZipNN all treat each weight (or each byte) as an **independent, identically-distributed**
  symbol. That is the *dumbest possible model* that still works. It throws away **all
  inter-weight structure**.
- Therefore ≈10.6 bit is **"as good as today's model of the data,"** not a law of nature.
  Inventing a **better model of LLM-weight structure** is exactly how the floor moves — and it
  is wide open, because **no lossless codec exploits LLM weight structure at all.**

We are not trying to violate information theory. We are trying to **find the better model** that
information theory says will exist *if the structure is there* — and in LLMs, **it is there**
(Section 3). Whether it is *rich enough* to break the floor by a useful margin is an **empirical
question we will measure** (Phase 0), not assume in either direction.

---

## 2. Hypothesis — exploitable structure in LLM weights

LLM weight tensors are **not** i.i.d. random. Three structural redundancies are *empirically
established* and **none are exploited by any lossless codec**:

1. **Low-rank-ness.** Pretrained weight matrices have rapidly-decaying singular-value spectra;
   the success of **LoRA** (low-rank *updates*) is direct evidence that the relevant signal lives
   in a low-rank subspace. Everyone uses low-rank for **lossy** adapters — **nobody uses it as a
   lossless predictor** with an exact residual.
2. **Cross-layer correlation.** The residual-stream architecture means adjacent layers transform
   *similar* representations; corresponding weight positions across layers are correlated and
   therefore **predictable from neighbors**.
3. **Local / structured correlation.** Within a tensor, neighboring weights (same neuron, adjacent
   channels, adjacent positions) are correlated; the bf16 **exponent field** is highly clustered
   (low entropy), and the **mantissa** may carry cross-weight correlation that order-0 coders miss.

If even one of these carries real signal, a predictor can turn the raw weight stream into a
**lower-entropy residual stream**, and the lossless size drops below 10.6 bit.

---

## 3. Concept — the predictive lossless codec (the methodology)

REEFORM is a **predict-then-encode** pipeline, not a raw entropy coder. For each weight tile:

```
ENCODE
  1. DECORRELATE (reversible, deterministic):
        prediction = P(low-rank basis, cross-layer neighbors, local context)
        residual   = W − prediction                # exact, in the value/bit domain
  2. MODEL:
        encode residual with a CONTEXT model (not order-0) +
        store predictor parameters (shared / cheap, amortized over many weights)

DECODE
  3. RECONSTRUCT:
        W = prediction + residual                  # bit-identical  ⇒  LOSSLESS
```

**Why it is lossless:** the predictor `P` is *deterministic* and the residual is stored
*exactly*; `prediction + residual` reproduces `W` to the bit. Compression comes **only** from the
residual having lower entropy than `W`, plus a cheap predictor description.

**Why it is novel:** the combination of (a) **low-rank prediction**, (b) **cross-layer
prediction**, and (c) a **context-modeled residual coder**, applied to **lossless** LLM-weight
compression, **does not exist** in literature or product. It is genuinely new IP.

**Candidate building blocks (to be selected by measurement, not assumption):**
- Decorrelating transforms: low-rank projection (`W ≈ U Σ Vᵀ` + exact residual), cross-layer
  delta (`W_L − αW_{L−1}`), reversible column/row **permutation** to cluster similar weights.
- Residual coder: context-mixing / higher-order entropy model over the residual bit-planes
  (beats the order-0 rANS we already ship).
- Everything must round-trip **bit-exact** and decode **fast enough for edge CPU** (ties into the
  separate fused-NEON-decode runtime track).

---

## 4. Success criteria

| # | Criterion | Bar |
|---|-----------|-----|
| S1 | **Bit-identical reconstruction** | byte-for-byte equality vs original (non-negotiable) |
| S2 | **Beat the floor** | **bits/weight < 10.6** on real LLM checkpoints (≤1B) |
| S3 | **Meaningful margin** | ideally ≤ 9.x bit lossless (would already beat DFloat11/ZipNN SOTA) |
| S4 | **Edge-decodable** | decode throughput compatible with the fused-NEON runtime goal |

Hitting **S1 + S2** with any positive margin is a publishable, novel result.

---

## 5. Research arc — research → trial/error → invention → success

| Phase | Goal | Deliverable / decision |
|-------|------|------------------------|
| **0 · PROBE** | Quantify how much structure actually exists (low-rank, cross-layer, residual entropy) in real ≤1B weights | go / no-go + *where* the signal is |
| **1 · PROTOTYPE** | Build the smallest predictor (e.g. low-rank + cross-layer) + exact residual coder; measure bits/weight; verify lossless | proof the floor moves |
| **2 · ITERATE** | Add building blocks the data demands; push the margin; harden round-trip | the codec |
| **3 · NAME + PAPER** | Final name, write-up, IP lock | the invention, on record |

A "no" at Phase 0 is **not failure** — it tells us precisely which structural assumption is weak,
so we redirect the attack. (Researchers measure; they do not quit, and do not over-promise.)

---

## 6. Experimental protocol

**Hard rule — research models ≤ 1B only.** Use ≤1B checkpoints for ALL probes/prototypes — they
are the **sweet spot**: fast iteration, fit comfortably in RAM, and still carry representative
transformer structure. Do **not** burn iteration time on 2B+ models during research.

Local ≤1B corpus (already on disk):
- `pythia-70m` (70M) · `pythia-160m` (160M) — fastest iteration
- `SmolLM2-135M` (135M)
- `Llama-3.2-1B-Instruct` (1B) · `gemma-3-1b-it` (1B) — upper-bound, most representative

Per-experiment metrics:
- **bits/weight** (vs the 10.6 baseline and vs our own rANS/bit-plane)
- **bit-exact verify** (the existing container magic + per-tensor sha256 round-trip)
- **singular-value spectrum** per weight matrix (low-rank-ness)
- **cross-layer predictability** (residual energy after layer-to-layer prediction)
- **residual entropy** vs **raw entropy** (the headline "did the floor move?" number)

---

## 7. Risks & open questions (honest)

- **R1 — structure may be thin.** Post-order-0, the mantissa could be near-random; the gain might
  be small. *Mitigation:* Phase 0 measures this **before** we build anything.
- **R2 — predictor cost.** A strong predictor/context-model may be slow to decode. *Mitigation:*
  co-design with the fused-NEON-decode runtime track; storage-only wins still count.
- **R3 — overfitting to one arch.** Validate across ≥3 of the ≤1B models.
- **R4 — "lossless" leaks.** Any float-domain transform must be provably reversible (watch −0.0,
  NaN, denormals). *Mitigation:* exhaustive round-trip tests, bit-domain residuals where needed.

---

## 8. IP & secrecy

- Proprietary to **Rama Erik Esprada**. Repo is private; license is All-Rights-Reserved.
- **No method, code, parameter, or result is ever sent to any external/web service.** All probes
  run **locally** on local weights. Web search (if ever used) stays on *generic public concepts*,
  never our specifics.
- Paper is drafted internally; publication timing is the Owner's decision (and a deliberate
  disclosure choice, since publishing forfeits trade-secret protection).

---

## 9. First step

**Phase 0 probe** on a ≤1B model (start with `SmolLM2-135M` or `pythia-160m` for speed):
measure singular-value spectra, cross-layer predictability, and residual-vs-raw entropy — to see,
with real numbers, **how far the floor can move**. The output of Phase 0 turns this charter from a
hypothesis into a measured plan of attack.
