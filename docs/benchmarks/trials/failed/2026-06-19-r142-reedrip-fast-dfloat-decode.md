# Trial: R142 — REEDRIP fast `rtc-dfloat-v1` decode (feasibility gate)

Date: 2026-06-19
Owner: RLLM
Status: rejected
Folder: failed

## Hypothesis

Replacing the per-bit `BitReader` in the lossless `rtc-dfloat-v1` codec with a
buffered 64-bit-window reader (`BufferedBitReader`, shift/mask instead of div/mod
per bit) makes decode fast enough that compressed-resident inference — read ~10.6
bits/weight from RAM and decode on the fly — beats reading 16-bit bf16 straight
from RAM. The cheap lever (buffered reader) was expected to lift R140a's
0.02 GB/s naive decode by a large multiple toward the bandwidth budget.

## Scope

- Mode: experimental (compressed-resident feasibility; codec decode only, no kernel/runtime wiring)
- REE kernel: REEDRIP (working name; Erik's final call before any paper use)
- Model/artifact: `Llama-3.2-1B-Instruct-raw.rllm` tied bf16 embedding (`model.embed_tokens.weight`)
- Architecture: LLaMA 3.2 1B, bf16 embedding/LM head (hidden 2048, vocab 128256 → 262.7M weights, 525 MB)
- Target device/profile: Apple A18 Pro (2 P + 4 E), macOS; release build
- Expected bottleneck: per-bit Huffman bit extraction (div/mod per bit)
- Bottleneck tag: IO/decode (actual: per-symbol serial dependency chain + 64KB LUT, not bit extraction)

## Setup

Commands:

```bash
# Build the bit-identical fast decoder + parity tests (all green):
cargo test -p rtc-codec --lib decode_fast -- --nocapture

# Dump the real 525MB bf16 embedding sample:
cargo test -p rllm-runtime --release dump_bf16_embedding_sample -- --ignored --nocapture
# -> wrote 525336576 bf16 bytes to /tmp/rllm-bf16-sample.bin

# Run the feasibility bench (single-core decode throughput):
cargo test -p rtc-codec --release dfloat_fast_decode_feasibility -- --ignored --nocapture
```

Runtime context:

- build profile: release
- CPU: Apple A18 Pro (2 P + 4 E)
- RAM: encode + decode of the 525 MB sample in-process
- OS: macOS (Darwin 25.5.0)
- relevant env/config: none (pure codec micro-benchmark)

## Results

Single-core decode of the real 262.7M-weight bf16 embedding (5 warm iters):

| metric | naive `decode` | fast `decode_fast` | ratio |
|---|---:|---:|---:|
| throughput (Gweight/s) | 0.060 | 0.18 | **3.0×** |
| throughput (GB/s, bf16-out) | 0.12 | 0.37 | 3.0× |
| time / decode | ~4.4 s | 1.43 s | — |
| bits/weight (compression) | 10.626 | 10.626 | — (sanity ✓) |
| bit-identical to `decode`? | — | yes (parity tests green) | — |

Verdict computation (threshold from R141's bandwidth measurement):

| quantity | value |
|---|---:|
| fast single-core | 0.18 Gweight/s |
| aggregate (×3.5, A18 2P+4E) | **0.6 Gweight/s** |
| GO threshold (beat plain bf16) | ≥ 12 Gweight/s |
| MARGINAL threshold (RAM win) | ≥ 5 Gweight/s |
| **VERDICT** | **🔴 NO-GO** (≈20× below GO) |

All `rtc-codec` tests pass (39 + 2 ignored benches); `decode_fast` is bit-identical
to `decode` across skewed/single-exponent/random inputs and tail boundaries
(n = 0..33), and rejects the corrupt length table.

## Analysis

The buffered reader works and is bit-exact, but it is **3× faster, not 50–200×**,
and lands ~20× short of the GO threshold. The hypothesis was wrong about *where*
the cost is:

- **div/mod was not the bottleneck in release.** R140a's catastrophic 0.02 GB/s
  was a debug-build artifact; the release compiler already lowers `abs/8` and
  `abs%8` to a shift and a mask. The naive release decoder is ~0.06 Gweight/s
  (~4.4 s), not 26 s. So removing the per-bit arithmetic only bought 3×.
- **The real floor is the per-symbol serial dependency chain.** Each weight does
  `refill → peek → load lut.entries[window] → consume`, and the next `peek`
  depends on `consume`, which depends on the LUT load. That is a latency-bound
  chain with **no instruction-level parallelism** within a single bitstream. The
  decode LUT for a wide-exponent embedding is up to 2^15 × 2 B = 64 KB, so the
  per-symbol load sits on the critical path against L1. Measured ~5.4 ns/weight
  (~19 cycles) is consistent with that chain, not with bit extraction.

Consequence for the project thesis: the cheap lever (buffered single-stream
reader) is **necessary but nowhere near sufficient**. Closing a 20× gap would
require structural change the buffered reader cannot provide — per-row framing to
run many independent bitstreams interleaved for intra-core ILP (hiding the LUT
load latency), a two-level/smaller LUT, and multi-core (already counted in the
×3.5). Even an optimistic stack of those (multi-stream ILP ~4–6× × micro-opt
~1.5–2×) reaches only ~3–7 Gweight/s aggregate — **MARGINAL at best, not a clean
speed win**. The fused decode→bfdot kernel (Phase 2 proper) is therefore **not
worth building** on this codec: on CPU it cannot decode fast enough to beat plain
bf16, and at most it would break even on speed while saving RAM.

What still stands, honestly:
- **RAM-for-storage win is unaffected** — the codec compresses to 10.626
  bits/weight on disk (R140a result), independent of decode speed.
- **A real, retained deliverable:** `decode_fast` is a bit-identical, 3×-faster
  decoder — useful for `unpack`/`verify`/load-time decompression, not for
  per-token resident decode.

## Decision

rejected (NO-GO)

Reason: The buffered reader is correct and 3× faster but yields only 0.18
Gweight/s single-core (0.6 aggregate), ~20× below the ≥12 Gweight/s needed for
compressed-resident decode to beat plain bf16. The bottleneck is the per-symbol
serial Huffman dependency chain plus a 64 KB LUT, which the cheap lever cannot
remove. Do not build the Phase-2 fused decode→bfdot kernel on this codec.

Paper value:

- use as negative evidence / limitation: bounds the lossless-fast frontier on
  CPU. Combined with R141 (the bf16 LM head is bandwidth-bound, not
  compute-bound) and R140b (naive decode too slow), R142 shows the *decode side*
  of the same wall: a per-symbol-serial entropy codec cannot feed an exact-weight
  GEMV fast enough on CPU. The "compressed + lossless + fast simultaneously"
  target is not reachable with a canonical-Huffman codec decoded one stream at a
  time.

## Next Experiment

Do **not** proceed to the fused kernel. If the lossless-fast frontier is revisited,
the only levers with a chance are structural, and even success likely lands at
MARGINAL (RAM win, speed-neutral):

- **Per-row / per-tile framing** of the codec stream so N independent bitstreams
  decode interleaved within one core — the only way to get ILP past the
  per-symbol latency floor. Measure single-core Gweight/s with 4–8 interleaved
  streams before committing.
- **Smaller / two-level decode LUT** to relieve the 64 KB L1 pressure.
- A fundamentally different, SIMD-friendly lossless layout (e.g. fixed-width
  bit-planes per exponent group) rather than variable-length Huffman, if the
  RAM target justifies a new codec (own spec).
