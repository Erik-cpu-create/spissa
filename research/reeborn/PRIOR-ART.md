# REEBORN — Prior-Art Map (lossless weight / FP compression)

Date: 2026-06-25. Survey run via WebSearch from the main context (a 5-agent fan-out was
attempted but subagents were blocked from web access in their sandbox; lead ran it directly).
Purpose: know exactly what exists so REEBORN avoids reinventing/plagiarizing, and to locate any
genuinely-open gap.

## ⚠️ DECISIVE FINDING

**The proposed REEBORN design (raw 8-bit significand + entropy-coded exponent) is NOT original.**
It is **DFloat11** (arXiv 2504.11651, NeurIPS 2025) — Huffman-code the bf16 exponent, leave
sign+mantissa raw → ~11 bits/weight, ~30% smaller, lossless, on-the-fly decode. And **this repo
already implements it**: `crates/rtc-codec/src/dfloat.rs` = `rtc-dfloat-v1`, comment line 9:
"Original implementation (technique from DFloat11, arXiv 2504.11651)". Our E7 "REEBORN-PREFIX"
(10.64 b/w) re-derived the existing `rtc-dfloat-v1`.

## 1. Coder layer — ALL closed (any choice is commodity; some patented)

| coder | concept | year | note |
|---|---|---|---|
| Huffman / canonical Huffman | optimal prefix code; canonical = frequency-sorted, codebook = lengths only | 1952 | our "rank+static" idea = canonical Huffman |
| Shannon–Fano | top-down prefix | 1949 | suboptimal |
| Arithmetic / range coding | whole message → one fractional interval; fractional bits | 1976 | optimal |
| ANS / rANS / tANS | numeral-system state; optimal + fast tables | Duda 2014 | **core is effectively FREE** — Duda placed it in public domain; shipped openly in zstd/FSE, Apple LZFSE, DietGPU (MIT). Microsoft holds a patent on specific rANS *modifications* only (2022); Google's app abandoned 2018. ⇒ core rANS usable; verify a specific variant before shipping. |
| Golomb / Rice | unary quotient + binary remainder; geometric dists | 1966 | |
| Elias γ/δ/ω | universal codes for integers (small→short) | 1975 | |
| Exp-Golomb | unary prefix + binary | 1978 | used in H.264 |
| Tunstall | variable-to-fixed | 1967 | |
| unary / Fibonacci | building blocks | — | |

**No original optimal coder is possible** — the space is mathematically closed (Shannon limit +
these families). Genuine originality cannot live at the coding layer.

## 2. Lossless FP / weight codecs (standalone) — direct competitors

| work | mechanism | result | source |
|---|---|---|---|
| **DFloat11** | Huffman exponent, raw sign+mantissa, GPU decode | bf16 → ~11 bit, ~30%, lossless | arXiv 2504.11651 (NeurIPS'25) |
| **NeuZip** | ANS exponent via nvCOMP | lossless, slower inference | arXiv 2410.20650 |
| **ZipNN** | byte-shuffle exp/mantissa + zstd | 33% on bf16 | arXiv 2411.05239 |
| **Exponent Concentration** | α-stable weights → exp entropy 2–3 bit | theory + method | arXiv 2510.02676 |
| **NN components (weights/ckpt/KV)** | low-precision lossless | — | arXiv 2508.19263 |
| **Patent** | lossless exponent / lossy mantissa | granted | WO2021045810A1 |

Floor for ALL of these = **~10.5–10.6 b/w** on bf16 (our own measurement: 10.55). Matches.

## 3. General LZ / zstd on weights

zstd on bf16 ≈ **1.34× (~25%)**; generic LZ/gzip mostly fails because the mantissa is
high-entropy (no repeated byte strings). zstd = the de-facto baseline an original codec must beat.

## 4. Model / context-based (PPM, PAQ/cmix, neural NNCP)

PPM, context-mixing (PAQ/cmix), NNCP (online Transformer), Nacrith — built for *sequential*
data (text/genomic). On near-i.i.d. trained weights they give ~nothing (confirmed by our E2–E6
null results) and cost enormous compute. Not viable for weights. (NNLCB benchmark.)

## 5. Delta / base-relative — the ONLY path below the ~10.6 floor

BitDelta (1-bit delta from base, github.com/FasterDecoding/BitDelta); REEFORM (this project's
base-exponent-conditioned delta, 7.7 → 7.1 b/w). Reaches ~7.7 b/w but **requires a base model and
IS the delta object** (user has ruled delta out for REEBORN).

## 6. Edge / fast-decode lossless

DFloat11 (GPU), **DietGPU** (Meta, GPU rANS 250–410 GB/s), **RAS** (bit-exact rANS accelerator,
arXiv 2511.04684), **ZipServ** (hardware-aware lossless LLM inference, arXiv 2603.17435). All
**GPU/accelerator-centric**. CPU/ARM-edge lossless weight decode in the >RAM streaming regime
(this project's R144–R153) is comparatively **less crowded** — but it is an engineering/optimization
target, not a novel algorithm.

## VERDICT

1. **No original lossless ALGORITHM remains for standalone weights** — DFloat11/ZipNN/NeuZip +
   coding theory + this repo's own `rtc-dfloat-v1` and measurements have closed it. REEBORN-as-designed
   duplicates DFloat11.
2. **Genuine originality is only available at:** (a) an original *implementation* for the edge CPU/ARM
   fast-decode niche (engineering, fits spissa's mission); or (b) a *different object/problem* (delta —
   ruled out; or something outside standalone-weight lossless ratio entirely).
3. Patent caution (CORRECTED by the verified survey): **core rANS/ANS is effectively FREE** (public-domain
   intent; shipped in zstd/FSE, Apple LZFSE, DietGPU-MIT) — usable. Only a specific Microsoft *modification*
   (2022) is patented; verify any non-vanilla variant. **zfp / posits / MultiPosits are PATENT-FREE** (the
   "zfp patent" is a myth — that's Samplify/Altera US8959129B2, not LLNL). Real structured-method IP risks:
   **NVIDIA 2:4 sparse-DNN family (US10997496B2 …) and Tensor-Ring NN patents (Baidu US12236342B2, Rutgers
   WO2022251317A1)** — avoid those.

## VERIFIED UPDATE (web-recovered survey of delta / structured / edge-decode)

**The edge CPU/ARM fast-decode niche is VERIFIED genuinely open.** Ranked to the exact target
("lossless weight decode by GB/s in the model>RAM streaming regime on ARM"):
1. **DietGPU** (Meta, MIT) — closest *technique* (parallel rANS + code-the-exponent), wrong platform (**GPU**).
2. **DFloat11** — closest *goal* (lossless decode at inference), wrong platform (**GPU**).
3. **ZipNN / NeuZip / arXiv 2508.19263 (Intel)** — right platform (CPU) + lossless, but optimize
   **footprint/transfer**, NOT NEON decode-throughput in >RAM streaming.
4. **Giesen interleaved rANS (2014, ryg_rans, open) + Zstd/FSE (Collet, BSD)** — the enabling decode tech.
⇒ "**Multi-lane (8–32) SIMD interleaved rANS on NEON, fused into a streaming GEMV, lossless, benchmarked by
decode GB/s in the model>RAM edge regime**" is occupied by NO external work found. White space is real
(caveat: Jan-2026 cutoff — re-check live before any external novelty claim). Matches the repo's own
conclusion (`docs/idea-fast-lossless-decode-rram.md`).

**A second (speculative, unclaimed) lane:** no published scheme stores `W = U·V + LOSSLESS entropy-coded
residual R` for bit-exact reconstruction. For high-rank LLM weights `U·V` fails (matches REEFORM low-rank =
net 11.73, worse). Unclaimed twist: a lossless residual against an already-lossy q4/q3 baseline. UNVERIFIED
that it wins (residual may be the incompressible low bits) — needs an experiment.

**Delta landscape (confirms delta correctly avoided):** BitDelta (1-bit), DeltaZip (ETH, production
multi-tenant serving), Delta-CoMe (NeurIPS'24), GPT-Zip, Delta-DCT — all base-conditioned (need base at
decode); a different (adapter-serving) problem, not a self-contained codec.

## Sources

- DFloat11 — https://arxiv.org/abs/2504.11651 · https://github.com/LeanModels/DFloat11
- ZipNN — https://arxiv.org/html/2411.05239v2
- NeuZip — https://arxiv.org/abs/2410.20650
- Exponent Concentration — https://arxiv.org/html/2510.02676
- NN components lossless — https://arxiv.org/abs/2508.19263
- Patent WO2021045810A1 — https://patents.google.com/patent/WO2021045810A1/en
- ANS + patents — https://en.wikipedia.org/wiki/Asymmetric_numeral_systems
- Universal codes (Elias/Golomb) — https://en.wikipedia.org/wiki/Universal_code_(data_compression)
- DietGPU — https://github.com/facebookresearch/dietgpu
- BitDelta — https://github.com/FasterDecoding/BitDelta
- RAS rANS accelerator — https://arxiv.org/pdf/2511.04684
- ZipServ — https://arxiv.org/html/2603.17435v1
- Neural lossless benchmark (NNLCB) — https://fahaihi.github.io/NNLCB/
