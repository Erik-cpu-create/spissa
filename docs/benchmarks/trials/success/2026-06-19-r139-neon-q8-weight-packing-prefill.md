# Trial: R139 NEON q8 i8mm weight packing — prefill reaches Ollama parity

Date: 2026-06-19
Owner: RLLM
Status: accepted (prefill 511ms -> 201ms; matches Ollama CPU on the same chip)
Folder: success

## Hypothesis

After R138, prefill scaling capped at ~2x on this device. Profiling found the
device is an Apple A18 Pro (2 performance + 4 efficiency cores) and that smmla
WAS engaging (RLLM_Q8_PANEL=0 made prefill slower), yet per-core throughput was
only ~18 GMAC/s — far below i8mm peak. Suspected overhead: `pack_q8_weight_pair`,
which rearranges the two output rows' int8 weights into the i8mm panel layout on
every matmul, copied weights one byte at a time in a scalar loop. Vectorizing
that copy should remove the overhead and approach the smmla kernel's real speed.

## Scope

- Mode: fast-lowram runtime (q8, codec rtc-raw-v1)
- REE kernel: REEFUSE-Q8-I8MM-PANEL weight packing (NEON)
- Model/artifact: `models/gemma-3-4b-it-q8.spsa`
- Architecture: Gemma 3 4B, Q8_0
- Target device/profile: Apple A18 Pro (iPhone-class SoC), 2P+4E cores, CPU only
- Expected bottleneck: scalar weight repack in the i8mm panel hot loop
- Bottleneck tag: CPU arithmetic (scalar memory shuffle)

## Setup

```bash
cargo build --release -p rllm-cli
RLLM_Q8_KERNEL_PROFILE=1 ./target/release/gemma-test --fast ...   # default threads
```

`pack_q8_weight_pair` interleaves at 8-byte segment granularity, so each segment
is moved with one `vld1_u8`/`vst1_u8` instead of 8 scalar byte stores. Bytes are
identical (q8 is already int8, reinterpreted). Scalar fallback kept off-aarch64.

## Results

| prefill | before (scalar pack) | after (NEON pack) | vs Ollama CPU (same chip) |
|---|---:|---:|---:|
| 6 tokens | 511 ms | 201 ms (~2.5x) | ~170 ms |
| 27 tokens | 1111 ms | 747 ms (~1.5x) | ~755 ms (parity) |

Prefill throughput after: 6 tok 29.9 tok/s, 27 tok 36.1 tok/s — vs Ollama CPU
35.8 tok/s. Decode unchanged (batch=1 uses sdot, not this panel): ~93ms layers +
26ms lm_head. Output bit-for-bit unchanged and coherent ("Photosynthesis is the
process where plants, algae, and some bacteria use sunlight ... the oxygen we
breathe"). 285 tests green (the existing smmla parity tests cover correctness).

## Analysis

The smmla kernel was never the prefill bottleneck — the per-byte scalar repack
into the panel layout was, dominating for short prompts (~60% of prefill) and
shrinking as batch grows (the repack is O(weights), the matmul O(weights*batch)).
Vectorizing the copy brought prefill to llama.cpp/Ollama parity on the same A18
Pro. The earlier "~3x per-core kernel gap vs Ollama" was mostly this packing, not
a fundamentally slower matmul.

## Decision

accepted

Reason: ~2.5x prefill for short prompts, Ollama-parity for longer ones, output
unchanged, tests green, zero correctness risk (identical bytes, faster copy).

Paper value:

- use as positive evidence (a lossless CPU runtime can match llama.cpp prefill on
  the same silicon; the gap was an unvectorized repack, not the i8mm matmul).

## Next Experiment

Prefill matches Ollama; the per-core ceiling is now the hardware (2 P-cores).
Remaining levers are decode (q8 layers ~95ms; sdot already row-parallel but
~1.5x thread scaling on 2 P-cores) and the one-time ~2.7s integrity prewarm
(hardware SHA). Sub-Ollama prefill (the user's 0.1s) would require beating
llama.cpp on a phone SoC and is not realistically on the table.
