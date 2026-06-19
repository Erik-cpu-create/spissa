# Trial: R144 — REEFUSE-PLANE-DOT fused bit-plane decode→bfdot GEMV (Phase C)

Date: 2026-06-19
Owner: RLLM
Status: rejected (speed); concept (lossless + RAM) proven
Folder: failed

## Hypothesis

Fusing R143 bit-plane decode + R141 `bfdot` into one resident lm-head GEMV —
decode each weight row into an L1 scratch, `bfdot` it, never materialize bf16 to
DRAM — beats reading 16-bit bf16 from DRAM and `bfdot`-ing it, lossless, because
the bit-plane planes are ~19% smaller (13 vs 16 bits/weight). This is the e2e
proof for the R143 decode-throughput GO.

## Scope

- Mode: experimental (compressed-resident fused GEMV, Phase C)
- REE kernel: REEFUSE-PLANE-DOT (working name; Erik's final call before any paper use)
- Model/artifact: `Llama-3.2-1B-Instruct-raw.rllm` tied bf16 embedding (vocab 128256 × hidden 2048, 525 MB)
- Architecture: LLaMA 3.2 1B bf16 LM head
- Target device/profile: Apple A18 Pro (2 P + 4 E), macOS; release; single-core
- Expected bottleneck: per-row decode compute vs DRAM bandwidth
- Bottleneck tag: CPU arithmetic (decode) vs memory bandwidth — the decisive one

## Setup

Commands:

```bash
# Lossless parity (fused == plain bf16, bit-for-bit) + no-alloc decode wrapper:
cargo test -p rllm-runtime --lib fused_bitplane_gemv -- --nocapture
cargo test -p rtc-codec --lib decode_neon_w5_into -- --nocapture

# Real 525MB sample (reused):
cargo test -p rllm-runtime --release dump_bf16_embedding_sample -- --ignored --nocapture

# Feasibility (single-core plain-bf16 vs fused bit-plane GEMV, both bfdot):
cargo test -p rllm-runtime --release fused_bitplane_gemv_feasibility -- --ignored --nocapture
```

Runtime context:

- build profile: release
- CPU: Apple A18 Pro (2 P + 4 E); single-core measurement
- RAM: bf16 (525 MB) and bit-plane planes (427 MB) both resident in-process
- OS: macOS (Darwin 25.5.0)
- relevant env/config: `RLLM_Q8_ACTIVATION=1` + `RLLM_BF16_DOT=1` (both paths use bfdot)

## Results

One full lm-head GEMV (262.7M weights = 128256 rows × 2048), 5 warm iters, single-core:

| path | ms / token | resident | weight throughput | logits |
|---|---:|---:|---:|---|
| plain bf16 (read→bfdot) | **18.5** | 525 MB | ~14.2 Gweight/s (28 GB/s DRAM) | baseline |
| fused bit-plane (decode→bfdot) | **61.5** | 427 MB (19% less) | ~4.3 Gweight/s | **bit-identical** ✓ |

- **speedup: 0.30× → fused is 3.3× SLOWER.** VERDICT: 🔴 **NO-GO (speed).**
- **Lossless parity: OK** — fused logits equal the plain bf16 logits **bit-for-bit**
  (same exact weights, same bfdot kernel), on both the synthetic unit test and the
  real 525 MB embedding.
- All tests green: 290 rllm-runtime lib + 45 rtc-codec (incl. the no-alloc wrapper
  parity).

## Analysis

The fused kernel is correct and genuinely lossless — that part of the concept is
proven. But on speed it loses decisively, and the reason overturns R143's GO:

- **R143 measured decode throughput in isolation** (a tight loop over all weights,
  materializing) and compared its *aggregate* (×3.5 cores) 17.7 Gweight/s to a 12
  Gweight/s "budget." That was an optimistic proxy.
- **The e2e reality is different on two counts.** (1) Per-row `decode → bfdot` runs
  the decode and the dot *sequentially* — the ~43 ms of decode compute is
  **additive** to the dot, not overlapped with it. (2) The plain bf16 GEMV is
  **bandwidth-fast single-core**: 525 MB / 18.5 ms ≈ **28 GB/s ≈ 14.2 Gweight/s**.
  The NEON bit-plane decode runs at ~4–6 Gweight/s — **~2–3× slower than simply
  reading bf16 from DRAM.** So replacing a 14 Gweight/s read with a 6 Gweight/s
  decode (plus a 19% smaller read) makes the GEMV decode-compute-bound and slower.

This is the **Huff-LLM caveat made concrete** (noted in the speed thesis): the
compression win shrinks — here, inverts — as hardware bandwidth rises. The A18's
DRAM bandwidth is high enough that **reading uncompressed bf16 is faster than
decoding a compressed form**, even with a SIMD-optimal codec. Multi-threading
would let decode parallelize past the shared bus, narrowing but (by the bus
ceiling) not closing the gap — at best MARGINAL, never a clean speed win.

What is proven and stands:

- **Lossless compressed-resident is correct and real on CPU** — the fused path
  produces bit-identical logits from a 19%-smaller resident buffer. The **RAM win
  (525→427 MB, 19%) holds**; it simply costs ~3× speed.
- The frontier is now fully mapped (R142 + R143 + R144): for lossless bf16 on CPU,
  Huffman decodes too slow (serial), bit-plane decodes fast in isolation but its
  decode compute still loses to DRAM bandwidth when fused. **Compressed-resident
  on CPU buys RAM, not speed.**

## Decision

rejected on speed (NO-GO); the lossless + RAM-saving concept is proven.

Reason: The fused bit-plane decode→bfdot GEMV is bit-identical (lossless) and uses
19% less resident RAM, but is 3.3× slower single-core because NEON decode (~6
Gweight/s) cannot keep up with the bandwidth-fast bf16 DRAM read (~14 Gweight/s) —
and per-row decode time is additive to the dot. Do not wire it behind `--fast` for
speed; it would only be worth wiring as a **RAM-saving** mode (smaller resident
footprint, accept ~3× slower lm-head) for memory-constrained cases.

Paper value:

- use as negative evidence / limitation: closes the lossless-compressed-resident
  arc honestly. The complete result — Huffman (R142) too slow, bit-plane decode
  fast in isolation (R143) but **bandwidth-dominated when fused (R144)** — is a
  clean CPU-frontier finding: lossless weight compression on CPU saves RAM but not
  decode time, because CPU DRAM bandwidth already exceeds achievable decode
  throughput. Pairs with R141 (the bf16 LM head is bandwidth-bound).

## Next Experiment

- If RAM is the binding constraint (the mission's "run on cheap low-RAM CPU"
  angle): wire the fused path as an explicit **RAM-saving lm-head mode** (opt-in,
  19% smaller resident, ~3× slower) — its own small spec; honest about the
  trade. Otherwise compressed-resident-for-speed is closed on this hardware.
- The durable speed levers remain the q8 layers (decode) and prefill, not weight
  compression. Storage-compression (smaller `.rllm` files, decode-once-at-load)
  stays the practical use for the codecs.
