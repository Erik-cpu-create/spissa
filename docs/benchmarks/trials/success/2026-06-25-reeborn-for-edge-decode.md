# Trial: REEBORN-FOR — coderless exponent decode for edge >RAM streaming

## Status

Success (the REEBORN edge thesis, validated as a microbench).

## Hypothesis

8-lane scalar rANS regressed (prior trial): more scalar lanes don't help, and rANS decode is the
wall in >RAM streaming (R153: e2e 1.05×, decode-bound). Hypothesis: a **coderless** frame-of-reference
fixed-width exponent code (no table, no state machine, no renorm) decodes far faster than rANS — fast
enough to keep the streaming path READ-bound (where compression wins) instead of decode-bound (where
it loses), even though its ratio is worse.

## Scope

- Mode: experimental, REEBORN-FOR (`rtc-reeborn-for`, `crates/rtc-codec/src/forcodec.rs`)
- REE kernel: coderless fixed-width FOR vs rtc-rans-v1 4-lane
- Artifact: synthetic 40M-symbol exponent stream (realistic bf16 peak; rANS lands 3.15 bits/sym)
- Target/profile: Apple Silicon arm64, `cargo test --release`, single core
- Bottleneck tag: codec decode throughput vs storage bandwidth

## Setup

```sh
cargo test -p rtc-codec reeborn_for_vs_rans_decode_bench -- --ignored --nocapture --release
```

FOR: base=104, width=5 (covers exp 104..135), MSB-first bit-pack. Round-trip verified bit-exact
(`for_roundtrip_fixed_width`).

## Results

| exponent codec | bits/exp | decode (Gweight/s/core) |
|---|---:|---:|
| REEBORN-FOR (coderless, 5-bit) | 5.00 | **1.797** |
| rANS (4-lane) | 3.15 | 0.291 |

→ FOR decodes **6.18× faster** at **1.59×** the exponent bits.

### >RAM streaming implication (cold read ~1.74 GB/s, from R150a)

Full weight = raw 8-bit significand + exponent. b/w: FOR = 8+5 = 13 (1.625 B/w); rANS = 8+3.15 ≈
11.15 (1.39 B/w); raw bf16 = 16 (2.0 B/w). Net Gw/s = min(read/bytes, decode):

| | B/weight | decode Gw/s | net in >RAM cold |
|---|---:|---:|---|
| raw bf16 | 2.00 | — | 0.87 (read-bound) |
| **REEBORN-FOR** | 1.625 | ~1.8 | **1.07 — READ-bound, 1.23× > raw** |
| rANS | 1.39 | 0.29 | **0.29 — DECODE-bound, 0.33× (slower than raw)** |

## Analysis

The coderless FOR design wins exactly where it counts. rANS has the better ratio but its ~0.29
Gw/s decode is below the read rate, so the path is decode-bound and compression makes streaming
*slower* than raw bf16 (R153's 1.05× problem, made concrete). FOR's 6.18×-faster decode keeps the
path read-bound, so its (smaller) byte savings translate directly into a 1.23× streaming speed-up.
This is the REEBORN edge contribution: in the model>RAM regime, a fast coderless decode beats a
better-ratio entropy coder — the niche the prior-art survey found unoccupied (DFloat11/DietGPU are
GPU; ZipNN/NeuZip optimize footprint, not ARM decode throughput).

## Decision

Accept REEBORN-FOR (coderless fixed-width exponent + raw significand) as the edge codec direction.

Paper value: positive — coderless decode flips the >RAM streaming path from decode-bound to
read-bound, where lossless compression actually buys speed.

## Caveats / next

- Single-core synthetic microbench on Apple Silicon; real per-tensor exponent ranges (and an escape
  path for rare out-of-range exponents) + real model + the ARM phone need measuring.
- This benches exponent decode in isolation; the significand is a raw copy. Real validation = fuse
  decode→bf16-reconstruct→streaming GEMV and measure e2e Gw/s in true >RAM cold on Mac then phone.
- Next: (1) NEON-vectorize the FOR unpack (branch-free bulk, expected even faster); (2) add a
  width+escape selector per tensor; (3) wire `rtc-reeborn-for` into the container + streaming GEMV;
  (4) e2e >RAM benchmark vs raw and vs rANS.
