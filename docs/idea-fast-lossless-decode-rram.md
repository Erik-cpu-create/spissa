# Idea (parked): small + lossless + fast-decode for the >RAM edge regime

Parked 2026-06-21 to work on other things first. This captures a technique set (from a
colleague of Erik's, well-read in Zstd/FSE, DFloat11, ZipServ) plus RLLM's **honest,
measured** assessment of each — so when we return we start from the real state, not from
scratch.

## The governing principle (correct — use this as the lens)

> Decode is **not a fixed tax.** Net cost = `bandwidth_saved − decode_compute_added`.
> Every technique below attacks the right-hand side.

This equation is exactly right and it **predicts RLLM's own measurements**: the sign of
`net` flips at the bandwidth/decode crossover.
- **fits-RAM (fast DRAM ~28 GB/s):** `bandwidth_saved` is small in time, `decode_compute`
  is non-trivial → **net NEGATIVE** (slower than bf16 zero-copy). Measured R144: fused
  decode = ~3× slower than reading bf16 straight from mmap. The fastest decode is *no
  decode*.
- **>RAM (slow flash/SSD ~1–2 GB/s):** `bandwidth_saved` is large in time, decode is
  hidden under I/O wait → **net POSITIVE** (faster). Measured R147/R148: ~1.32× faster
  than raw bf16 streaming.

**Conclusion: lossless-compressed decode wins ONLY where bf16 can't run (model > RAM).**
That is the edge/phone target — the right home for this whole toolkit.

## The 6 techniques vs RLLM's actual state

| # | Technique | RLLM status |
|---|-----------|-------------|
| 1 | **Interleaved rANS** (8/16/32 independent lanes, round-robin, SIMD — no cross-lane dependency) | ⚠️ PARTIAL — R152 has 4-lane interleave; R158c made it lane-parallel (threads). NOT yet 8–32-lane SIMD-within-lane. **Lever to push.** |
| 2 | **Tabled decode (tANS/FSE)** — build a table once, decode = gather + few ALU ops (L1/L2-resident, SCALE ≤ 4096 ≈ 4–8 KB) | ✅ DONE — `RansDecodeTables` (precomputed cum/slot, no per-step division). |
| 3 | **State-width tuned for branchless renorm** (31-bit state, 16/8-bit renorm, ≤1 byte pulled/symbol → vectorizable) | ⚠️ NOT TUNED — micro-lever, do alongside #1. |
| 4 | **Tile-granular decode + reuse** — decode a weight tile once, reuse across batch × N output columns; decode amortizes over O(batch×N) MAC → ~0 per token at large batch (DFloat11's "constant per-forward-pass overhead") | ⚠️ PARTIAL — R160 decode-once amortizes across tokens; cross-batch reuse in prefill not maximized. Ties to the prefill gap (see [prefill-gap-root-cause](../README.md)). |
| 5 | **Fused decode-GEMM** — decode into registers/SRAM inside the GEMM kernel; decompressed weights never touch DRAM (ZipServ's ZipGEMM) | ⚠️ TRIED, regime-dependent — see correction below. R144 (in-RAM NO-GO), R147/R148 (>RAM GO). |
| 6 | **Decode-asymmetric + targeted coding** — encode slow, decode fast; code only the high-entropy part (BF16 exponent ≈ 2.6 of 8 bits), leave mantissa raw | ✅ DONE — this is RTC-rANS's core design (R151 measured H(exp)≈2.6, mantissa≈white-noise/incompressible). |

**~4 of 6 already implemented.** RLLM is aligned with SOTA thinking; the genuinely-new
levers are #1 (more-lane SIMD rANS), #5 (fused, in the >RAM kernel), #3 (renorm tuning).

## Critical correction — point #5 is GPU-true, CPU-regime-specific

The colleague calls fused decode-GEMM "the real key" and says *"without it you're slower."*
That is **true on GPU, incomplete on CPU**:
- **GPU (ZipServ):** wins because of the **~600× compute-vs-HBM imbalance** — tensor cores
  starve for bandwidth, so decode-in-kernel (trade abundant compute for scarce bandwidth)
  wins always.
- **CPU:** that imbalance does **not** exist in fits-RAM — DRAM bandwidth ≈ or > NEON decode
  throughput, so fused decode in RAM is ~3× **slower** (R144). It only wins in **>RAM**,
  where slow flash supplies the imbalance (R147/R148).

So apply fused decode-GEMM to the **>RAM streaming path only** (R148 REESTREAM), never the
fits-RAM path.

## Throughput reality (decode is already fast enough for phones)

- rANS decode ≈ 3.8 GB/s; bit-plane NEON ≈ 12 GB/s (per core).
- Phone flash (UFS) ≈ 1–2 GB/s sequential → **decode already > flash** ⇒ the >RAM imbalance
  exists on phones today. So raw decode throughput is **not** the blocker for >RAM; the
  blocker is the **fused kernel** (#5, never materialize) + the streaming pipeline (have it,
  R148). #1 (more lanes) gives margin, not the unlock.

## Sequencing (decided 2026-06-21)

1. **Phone-first, 1B q8 (fits-RAM).** A 1B model FITS a phone → fits-RAM regime → q8/bf16
   win, this toolkit doesn't help yet. Prove RLLM runs on Android (Termux build + run +
   measure) BEFORE any of this. See [android-termux.md](android-termux.md).
2. **This toolkit, when outgrowing phone RAM.** When the target model EXCEEDS phone RAM
   (e.g. 4B on a 4–6 GB phone), the >RAM regime kicks in and this becomes the lever — and
   the novel contribution (CPU/ARM lossless fused-decode for edge; the papers are all GPU).

## Concrete TODO when we return (priority order)

1. **16/32-lane SIMD-within-lane rANS decode** (NEON), portable scalar fallback for x86
   (universal — see memory `universal-device-target`). Bench decode GB/s + parity.
2. **Fused rANS-decode → GEMM in the >RAM streaming kernel** (extend R148 REESTREAM; do
   NOT touch the fits-RAM path). Measure net vs raw bf16 streaming on a real >RAM model.
3. **Tile-reuse amortization in prefill** (decode tile once, reuse across batch columns) —
   also helps the prefill gap.
4. **Branchless renorm** state-width tuning (micro).

Keep RLLM's hard rules: lossless by default, honest metrics, portable kernels only (NEVER
Apple-only Accelerate/AMX). References: R144 (in-RAM fused NO-GO), R147/R148 (>RAM GO),
R151 (entropy floor), R152/R158c (interleaved rANS), R159 (bit-plane), R160 (decode-once).
