# Spec: R148 — Pipelined streaming bit-plane GEMV (the capacity-bound runtime kernel)

Date: 2026-06-19
Status: design (approved to draft)
REE kernel (working name): **REESTREAM** (capacity-bound streaming decode→dot; Erik's final call before any report/paper use)

## Honest positioning (read first)

R147 proved the regime that wins: when the model exceeds RAM and streams cold from
SSD, lossless bit-plane is **1.13× faster** end-to-end (measured: raw bf16 6.3 GB
→ 3880 ms vs bit-plane 5.1 GB → 3424 ms), because cold SSD (1.62 GB/s) is ~28×
slower than decode (45 GB/s), so decode hides under the read. But the R147 scout
was **single-threaded and un-pipelined** — it read a whole block, *then* decoded
it, leaving a small additive residual (effective 1.50 vs 1.62 GB/s).

R148 builds the **production kernel**: a streaming GEMV that **overlaps the disk
read with the decode** (double-buffer: read block N+1 while decoding+dotting block
N) and **multi-threads the decode** across cores. The goal: hide the decode fully
under the SSD read so the stream runs at the compressed-byte SSD rate — pushing the
measured win from 1.13× toward the pure-byte-ratio **~1.23×**, lossless.

This is the heart of the capacity-bound runtime path. Full model wiring (pack a
model with the codec, register it in `codec_for_id`, stream the forward pass on a
model > RAM) is the explicit follow-on (R149+), gated on this kernel landing.

## Goal

A double-buffer pipelined streaming bit-plane GEMV kernel that reads block-framed
compressed weight planes sequentially from a file and decodes+dots each block while
the next block streams from disk — producing logits bit-identical to decoding the
same weights and dotting them. Benchmark it on a > RAM cold stream against (a) raw
bf16 streaming and (b) the R147 un-pipelined scout, and report the speedup.

## Design

### Block-framed streaming layout

The current bit-plane chunk is `[header][palette][index plane (all rows)][residual
plane (all rows)]` — the index and residual regions are separate, so a row-block's
bytes are not contiguous on disk. For sequential streaming, R148 uses a
**block-framed** layout: the weight matrix is cut into row-blocks of `B` rows, and
each block is written **contiguously** as `[block index bytes ++ block residual
bytes]`. A small block table (offset + byte length per block) precedes the blocks.
The reader then streams blocks sequentially with one read each, and each block
decodes independently (rows are byte-aligned: `hidden*5 % 8 == 0`). The palette is
shared (stored once in the header).

This is the per-row/per-tile framing flagged as future work since R141 — R148
introduces it for the streaming path. The non-streamed codec format is unchanged;
block-framing is a streaming-mode wrapper produced when packing for capacity-bound
streaming.

### Pipelined streaming GEMV `streaming_bitplane_gemv` (rllm-runtime)

New file `crates/rllm-runtime/src/streaming/bitplane_stream.rs` (`include!`d).
The dominant lever is **pipelining** (overlap the disk read with the decode), not
thread count: decode is ~28× faster than the cold SSD, so a *single* decoder
overlapped with the reader already keeps up with the disk and hides the decode.
So the core is a **double-buffer pipeline**, not a worker pool:

- **Reader thread:** reads block `N+1`'s bytes from the file into the spare buffer
  while the consumer decodes block `N` from the other buffer. Two block buffers
  alternate; a bounded `sync_channel(1)` (plus a free-buffer return channel) passes
  ownership so the reader never overwrites a buffer in use.
- **Consumer (main thread):** for each row in the received block, decode the row
  into a reused scratch (`rtc_codec::decode16_w5_into`, R146) and dot it
  (`bf16_row_dot_bf16`, R141, when FEAT_BF16; else the f32-upcast path), writing the
  block's disjoint output rows.
- **Overlap:** the reader's disk I/O for block `N+1` runs concurrently with the
  consumer's decode+dot of block `N`. Since decode ≪ the per-block read time,
  wall-clock ≈ the compressed-byte SSD read → the decode-additive residual the R147
  scout showed (1.50 vs 1.62 GB/s) collapses, pushing the win toward ~1.23×.

The activation is converted to bf16 once and shared (read-only).

Multi-threading the *decode* (a small worker pool draining a block) is a noted
optional refinement that only helps when storage is fast enough that decode
approaches the read rate; it is **out of scope** for R148 (the cold-SSD regime is
disk-bound, where one overlapped decoder suffices) and added later only if a faster
storage tier needs it.

### Benchmark (`#[ignore]`, > RAM cold stream)

Mirror R147: build a block-framed compressed file and a raw bf16 file, both
replicated to exceed RAM, F_NOCACHE-stream each cold. Compare:

- raw bf16 stream (read + dot),
- R147 un-pipelined scout (read whole block, then decode+dot, sequential),
- R148 `streaming_bitplane_gemv` (double-buffer pipeline: read next block while
  decoding+dotting current).

Report ms + effective GB/s + speedup vs raw and vs the scout. Confirm the
pipelined kernel's logits are bit-identical to a single-thread reference on a
small in-memory case.

## Non-goals

- Full model wiring: packing a model with the codec, `codec_for_id` registration,
  streaming the real forward pass, generation tok/s on a model > RAM. That is the
  R149+ follow-on, gated on this kernel.
- The in-RAM regime (R144/R145 NO-GO — not the target).
- General `(hidden, w)`; only `w=5`, `hidden=2048` (`hidden*5 % 8 == 0`).
- Compressing q8 layers; GPU; KV-cache.

## Testing

- **Lossless / bit-identical (hard rule):** the pipelined streaming GEMV produces
  logits bit-identical to a single-thread `decode + bf16_row_dot_bf16` reference on
  a small in-memory block-framed input. Multi-threading splits independent blocks,
  so order within a row is unchanged → exact equality.
- **Block-frame round-trip:** decoding all blocks of the block-framed layout
  reconstructs the same weights as the flat codec (a small parity test).
- **Honest metrics:** the bench reports raw-bf16 vs scout vs pipelined ms + the
  speedups on the > RAM cold stream — including if the pipelined win is only
  marginally above the scout (stated plainly).
- Existing rtc-codec + rllm-runtime tests stay green (additive).

## Originality & dependencies (doctrine)

- **Original code.** A double-buffered reader + decode-worker pipeline composing
  in-house REEPLANE decode (R146) + R141 bfdot, written from scratch. The
  block-framed layout is a thin streaming wrapper over the existing codec.
- **No new dependencies.** `std::thread` + `std::sync::mpsc` (or a bounded
  channel built from std), `std::fs`, `std::arch`. `cargo build` only.
- **Lossless by default** preserved; proven bit-identical by the parity test.

## Components / isolation

- Block-framed layout writer/reader (rtc-codec or rllm-runtime) — frames the
  existing planes into self-contained on-disk blocks + a block table.
- `streaming_bitplane_gemv` (rllm-runtime `bitplane_stream.rs`) — the pipelined
  reader + decode-worker pool; reuses `decode16_w5_into` + `bf16_row_dot_bf16`.
- Benchmark — `#[ignore]`, > RAM cold stream, pipelined vs scout vs raw.
- Full model wiring — R149+ follow-on spec, gated on this.

## Prior art (cited honestly)

Overlapping storage I/O with on-the-fly decompression is the standard
streaming-decompression pattern (and the principle behind Apple's "LLM in a flash"
for models exceeding DRAM). R148's distinct position is the CPU/ARM realization for
**lossless bf16** weights: block-framed bit-plane planes streamed from SSD, decoded
in parallel and overlapped with the read, feeding a native bf16 dot — the
capacity-bound operating point (model > RAM, cheap device) the R147 measurement
identified as the regime where lossless compression wins on CPU.
