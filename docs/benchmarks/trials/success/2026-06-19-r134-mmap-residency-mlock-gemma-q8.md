# Trial: R134 mmap residency — MADV_WILLNEED + opt-in mlock

Date: 2026-06-19
Owner: RLLM
Status: accepted (2.9x decode, bit-identical output)
Folder: success

## Hypothesis

The Gemma q8 decode is memory-bound (proven in R133: cool-machine A/B showed
8-core sdot == scalar, decode is not compute-bound). The suspected cause is
that a plain `Mmap::map` faults weight pages in lazily and lets the OS evict
them under memory pressure. With a batch=1 decode loop re-reading the whole
weight set every token, evicted pages are re-faulted from disk each step, so
the runtime never reaches the resident-weights regime that lets llama.cpp hit
~12 tok/s on the same machine. Issuing `MADV_WILLNEED` (prefetch) and, where
the model fits RAM, `mlock` (pin) should keep the weights resident and remove
the per-token re-fault, giving a large decode speedup with byte-identical
output (no math changes — pure residency/IO).

This is the residency lever, the prerequisite identified by the
streaming-vs-resident thesis and the ggml-cpu-q8 methodology study. It is also
unblocked by the bf16-direct embedding (commit 0ebe80a), which dropped the
working set from 7.43 to 6.09 GB so a 4.5 GB pin can fit in 8 GB RAM.

## Scope

- Mode: exact-lowram runtime (q8, codec rtc-raw-v1 zero-copy)
- REE kernel: none — this is a residency/IO change (mmap advise + mlock), not
  an execution kernel, so the REE-kernel naming rule does not apply.
- Model/artifact: `models/gemma-3-4b-it-q8.rllm` (q8_transformer_keep_io, ~4.5 GB)
- Architecture: Gemma 3 4B, Q8_0
- Target device/profile: Apple Silicon, 8 GB RAM, CPU only
- Expected bottleneck: weight residency (page eviction / re-fault from disk)
- Bottleneck tag: memory bandwidth

## Setup

Commands:

```bash
cargo build --release -p rllm-cli

# baseline (WILLNEED only — mlock off)
/usr/bin/time -l ./target/release/gemma-test \
  --model models/gemma-3-4b-it-q8.rllm \
  --prompt "The capital of France is" --max-new-tokens 8 --ctx 256

# trial (mlock pin)
/usr/bin/time -l ./target/release/gemma-test \
  --model models/gemma-3-4b-it-q8.rllm \
  --prompt "The capital of France is" --mlock --max-new-tokens 8 --ctx 256
```

Runtime context:

- build profile: release
- CPU: Apple Silicon (ARM64), CPU-only
- RAM: 8 GB
- OS: macOS (Darwin 25.5.0), 16 KB pages
- relevant env/config: reader issues `MADV_WILLNEED` unconditionally;
  `RLLM_MLOCK=1` (set by `--mlock`) mlocks the whole mapping. Decode uses the
  default scalar q8 path (no `RLLM_Q8_ACTIVATION`) — residency only.

## Results

| run | prompt/input tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (WILLNEED only) | 6 | 8 | — | 0.15 | 0.15 | 4.69 GB | 0 | 1,731,729 page faults (model re-faulted ~6x over 8 tokens) |
| trial (+mlock) | 6 | 8 | — | 0.44 | 0.44 | 5.06 GB | 0 | 269,599 page faults (≈ model faulted ONCE then pinned); identical token ids |

Correctness: output bit-identical across both runs —
`[9079, 236761, 108, 50429, 563, 496, 4185, 3988]` ("Paris ..."). mlock changes
only page residency, not the math.

Tests: `cargo test -p rllm-container` green (17 passed).

## Analysis

mlock is the dominant residency lever. Page faults collapse 1,731,729 ->
269,599 (6.4x fewer). 269,599 ≈ 4.5 GB / 16 KB ≈ the model faulted in exactly
once and then held resident, versus the baseline re-faulting most of the model
every token. Decode wall-clock for 8 tokens drops 51.6 s -> 18.4 s and tok/s
rises 0.15 -> 0.44 (2.9x), with no change to output.

`MADV_WILLNEED` alone barely helped on this 8 GB machine (0.12 -> 0.15 tok/s):
prefetch reads the pages ahead but they are still evictable, so under 8 GB
pressure they get reclaimed and re-faulted anyway. The hint is kept on
unconditionally because it is free and helps prefault on machines with more
headroom; the residency win specifically needs the `mlock` pin.

The bf16-direct embedding (0ebe80a) was the enabler: it removed the 2.68 GB f32
embedding materialization, lowering the working set to 6.09 GB so a 4.5 GB pin
plus the rest fits inside 8 GB. Without it, mlock would not have fit.

This confirms the R133 conclusion (decode was memory-bound, not compute-bound):
the fix was residency, not a faster kernel. A follow-up A/B confirmed the int8
sdot fast-path adds no wall-clock on top of mlock (18.2 vs 18.4 s) — once
resident, decode is bandwidth-bound, so the fewer-instruction kernel does not
move wall-clock.

## Decision

accepted

Reason: 2.9x decode speedup with byte-identical output, validated end-to-end,
container tests green. Shipped as an opt-in `--mlock` flag (kept opt-in like
llama.cpp's `--mlock` because pinning a working set larger than RAM risks OOM).
`MADV_WILLNEED` shipped always-on.

Limitation: still 0.44 vs llama.cpp ~12 tok/s (~27x) on the same 8 GB machine.
With the model now resident, the remaining gap is the per-chunk dispatch
bandwidth (~2 GB/s effective vs the hardware ceiling), a separate lever — not
residency and not kernel arithmetic.

Paper value:

- use as positive evidence (residency is a first-order lever for memory-bound
  CPU decode) with limitation (still behind llama.cpp on raw bandwidth).

## Next Experiment

Per-chunk dispatch bandwidth: investigate why resident-weight reads run at
~2 GB/s instead of approaching the hardware ceiling (the ~3366
`with_raw_chunk` calls/token dispatch structure noted in R133). This is the
largest remaining gap (~27x). Lossless compressed-resident (EntroLLM-style
Huffman/ANS, ~1.3x + ~30% RAM on q8) is a later lever that only pays off once
decode is genuinely bandwidth-bound at the hardware level.
