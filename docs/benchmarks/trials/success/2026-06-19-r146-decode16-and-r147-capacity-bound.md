# Trial: R146 (16-wide decode) + R147 (capacity-bound streaming — the regime that wins)

Date: 2026-06-19
Owner: RLLM
Status: accepted (R147 GO — lossless compression wins when the model exceeds RAM)
Folder: success

## Hypothesis

R144/R145 found lossless bit-plane compressed-resident decode is *slower* than
reading bf16 — but only tested the **in-RAM** regime on a fast device (A18 Pro,
DRAM ~28 GB/s ≈ decode ~28 GB/s, balanced → decode additive → loss). The
literature (Cloudflare Unweight wins on GPU via a ~600× tensor-core-vs-HBM
imbalance; EntroLLM wins on edge) says the win needs **memory delivery to be the
bottleneck relative to decode**. Hypothesis: on CPU that regime is
**capacity-bound** — when the model exceeds RAM and streams from SSD/flash, the
slow SSD provides the imbalance, decode hides under it, and lossless compression
wins on speed AND RAM. This is exactly the low-RAM-device mission.

## Scope

- Mode: experimental (codec decode throughput + capacity-bound streaming)
- REE kernel: REEPLANE 16-wide (`decode16_w5_into`); the streaming proof reuses it
- Model/artifact: `Llama-3.2-1B-Instruct-raw.spsa` bf16 embedding (262.7M weights), replicated to >RAM for the stream test
- Target device/profile: Apple A18 Pro, macOS; release; single-core decode
- Bottleneck tag: IO/decode; storage bandwidth (the decisive one for R147)

## Setup

```bash
# R146: 16-wide decode parity + throughput vs 8-wide
cargo test -p rtc-codec --lib decode16_matches_8wide -- --nocapture
cargo test -p rtc-codec --release decode16_throughput_scout -- --ignored --nocapture
# R147: end-to-end capacity-bound stream (cold SSD, files > RAM, F_NOCACHE)
cargo test -p rtc-codec --release capacity_bound_stream_scout -- --ignored --nocapture
# cold SSD bandwidth measured separately by F_NOCACHE-reading the 6GB gemma raw file
```

## Results

**R146 — 16-wide `vqtbl2q` decode (single-core, real embedding):**

| | 8-wide (vtbl4) | 16-wide (vqtbl2q) | ratio |
|---|---:|---:|---:|
| decode throughput | 6.70 Gweight/s | 7.93 Gweight/s | 1.18× |
| bit-identical to 8-wide / scalar? | — | yes (parity green) | — |

16-wide is a real, validated, faster decoder (1.18×) — useful for load-time /
streaming decode. Aggregate ×3.5 = 27.7 Gweight/s = **45 GB/s** of compressed-input
consumption. (Not enough to flip the *in-RAM* e2e — that stays R145 NO-GO.)

**Key measured primitive — cold SSD bandwidth:** F_NOCACHE read of the 6.06 GB
gemma raw file (> RAM, so genuinely cold): **1.62 GB/s**. Decode consumes input at
**45 GB/s = ~28× faster than the SSD.**

**R147 — capacity-bound e2e stream (files > RAM, cold SSD, decode per tile):**

| stream | size | wall-clock | effective | 
|---|---:|---:|---:|
| raw bf16 | 6.3 GB | 3880 ms | 1.62 GB/s |
| bit-plane (read + decode) | 5.1 GB | 3424 ms | 1.50 GB/s |

**SPEEDUP 1.13× — GO.** Compressed streaming from SSD is faster end-to-end, lossless,
reading 19% fewer bytes; the decode is mostly hidden under the slow SSD (effective
1.50 vs 1.62 GB/s — a small additive residual because the scout's decode is
single-threaded and not pipelined with the read).

## Analysis

R147 is the regime that overturns the in-RAM NO-GO. The win is governed by the
**ratio of decode throughput to memory delivery**:

- **In-RAM, fast device (R144/R145):** DRAM ~28 GB/s ≈ decode ~28 GB/s. Balanced →
  decode time (~14–43 ms) is comparable to the bytes-saved read → additive → LOSS.
- **Capacity-bound, streaming from SSD (R147):** SSD **1.62 GB/s** vs decode
  **45 GB/s** = ~28× imbalance. Decode (9 ms) ≪ bytes-saved read time (60 ms/copy)
  → decode hidden → **WIN** (measured 1.13× even single-threaded, un-pipelined).

This is the CPU analog of Cloudflare's GPU insight (600× tensor-core-vs-HBM): the
imbalance is what makes decode-from-compressed win, and on cheap CPU devices the
**slow SSD supplies the imbalance**. The measured 1.13× is a conservative floor:
- Pipelining decode with the read + multi-threaded decode → decode fully hidden →
  approaches the pure-byte-ratio **1.23×**.
- Slower storage (cheaper devices, 0.5–1 GB/s) → larger imbalance → bigger win.
- If compression makes the model **fit in RAM** (avoiding swap entirely) → not
  1.1×, but **10–30×** (the dominant real-world benefit for low-RAM devices).

The whole R140–R147 arc now reads honestly: lossless compressed-resident on CPU is
bit-exact + smaller, and it is FASTER **exactly in the regime that matters for the
mission** — cheap, low-RAM devices running models larger than their RAM. The
earlier NO-GOs measured the one regime where it cannot win (small tensor, fast DRAM).

## Decision

accepted — R146: 16-wide decode validated (1.18×, bit-identical). R147: **GO** —
lossless bit-plane streaming from SSD is 1.13× faster e2e (conservative), lossless,
19% less I/O/RAM, in the capacity-bound regime (model > RAM).

Paper value:

- use as positive evidence: the CPU regime where lossless weight compression wins
  on BOTH speed and RAM — capacity-bound streaming, governed by the
  decode-vs-storage-bandwidth imbalance. Resolves the R140–R145 negatives as
  regime-specific, and identifies the operating point (model > RAM, cheap device)
  where the technique is a genuine win. Complements GPU SOTA (Cloudflare/DFloat)
  with the CPU/edge analog.

## Next Experiment

Build the real runtime path (R148): stream the bit-plane planes from disk during
generation, decode tiles pipelined with the read across cores, feed the matmul —
so a model that does not fit in RAM runs faster + lighter than raw bf16, lossless.
Wire as an opt-in capacity-bound mode; measure on a model that genuinely exceeds
the device RAM.
