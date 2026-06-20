# Spec: R149a — Streaming lm-head on a real model (Gemma 3 1B), correctness-first

Date: 2026-06-20
Status: design (approved to draft)
REE kernel: REESTREAM (R148, reused — no new kernel)

## Honest positioning (read first)

R148 proved the pipelined streaming bit-plane GEMV kernel works and wins **1.32×**
in the capacity-bound regime (model > RAM, cold SSD). R149a is the **first model
integration**: wire that kernel into a real model's LM head and prove it produces
**identical, lossless output** — on a small model (Gemma 3 1B IT) for fast,
correct iteration.

This phase is about **correctness, not speed**. Gemma 3 1B (1.5 GB) fits easily in
RAM, so streaming its lm-head from disk will NOT be faster than the resident path
(it re-reads per token instead of using the resident mmap) — that is expected and
fine. The speed win only manifests when the model exceeds RAM (a separate, later
phase: a > RAM model, or a cold-read measurement). R149a's job is to make the
streaming lm-head path **work and stay lossless** in a real model, as the
integration foundation for the rest of R149.

## Goal

1. A `pack_lmhead_bitplane` step that reads a model's tied bf16 embedding / LM head,
   encodes it with `rtc-bitplane-v1`, frames it into the R148 block layout, and
   writes a self-contained **sidecar file** (header + palette + blocks).
2. An **opt-in** path in the Gemma adapter that, when `RLLM_STREAM_LMHEAD=<sidecar>`
   is set, computes the LM-head logits by streaming the sidecar through
   `streaming_bitplane_gemv` instead of reading the resident bf16 embedding.
3. A correctness gate: Gemma 3 1B generates the **identical token sequence** with
   the streaming lm-head as with the resident path — proving the streaming
   integration is lossless end to end.

## Design

### Pre-flight: verify `w = 5` for the Gemma embedding

The R148 kernel (`decode16_w5_into`) is specialized to `w = 5` (≤ 32 distinct
exponents). Gemma 3 1B's embedding may differ from Llama's. **Task 1 first measures
the embedding's distinct-exponent count → `w`.** If `w = 5`, proceed. If `w ≠ 5`
(e.g. 6), that is a finding: R149a stops and records it, and the follow-up is
either generalizing the kernel to other widths or accepting the lm-head is not
w=5. (`hidden = 1152` already satisfies `hidden % 32 == 0` and `hidden*5 % 8 == 0`,
so the block geometry is valid; only the palette width is in question.)

### 1. `pack_lmhead_bitplane` (sidecar producer)

A function (exposed via a small `rllm` subcommand or an `#[ignore]` tool test for
R149a) that:

- Loads the model, reads the tied embedding tensor as bf16 (`vocab × hidden`).
- `BitplaneCodec::encode` → palette + index plane + residual plane; assert `w = 5`.
- Frames into R148 blocks (`block_rows` rows each, `[B×index ++ B×residual]`),
  matching `streaming_bitplane_gemv`'s layout.
- Writes a **sidecar file**: a small header (`magic`, `version`, `hidden`,
  `vocab`, `block_rows`, `palette_len`, palette bytes) followed by the framed
  blocks. The sidecar is self-contained and independent of the model's own codec.

`vocab` must be a multiple of `block_rows` (pad the last block with zero rows if
not; Gemma vocab 262144 is a multiple of 256, so no padding needed).

### 2. Gemma lm-head streaming (opt-in)

The Gemma adapter computes lm-head logits at `gemma/api.rs` via
`with_raw_tensor(embed_id, |bf16| lm_head_logits_parallel_bf16(...))`. Add a branch:
if `RLLM_STREAM_LMHEAD` names a sidecar path, parse its header, then call
`streaming_bitplane_gemv(sidecar, palette, hidden, block_rows, num_blocks,
last_hidden, &mut logits, nocache=false)` to fill the logits, and continue
(softmax/argmax) unchanged. `streaming_bitplane_gemv` becomes `pub(crate)`.

The header is read once per call (tiny); the blocks stream from the sidecar. Output
shape (`vocab` logits) is identical to the resident path, so nothing downstream
changes.

### 3. Correctness gate

Run Gemma 3 1B generation on a fixed prompt twice — resident (default) and with
`RLLM_STREAM_LMHEAD=<sidecar>` — and assert the **token id sequences are
identical**. Both paths decode the same exact bf16 weights and use the same f32
dot (`bf16_row_dot_f32`), so the logits are bit-identical → argmax identical →
identical output. This is the lossless end-to-end proof. (Reported via a CLI run
or an `#[ignore]` integration test that drives the model both ways.)

## Non-goals

- Speed: Gemma 1B fits in RAM, so streaming the lm-head is not faster here. The
  capacity-bound speed win is a later phase (> RAM model, or cold-read measurement).
- Streaming the transformer projections (only the lm-head). R149b+.
- Container reformatting / `codec_for_id` registration: the sidecar is a separate
  file, leaving the existing model format untouched.
- Generalizing `decode16_w5_into` beyond `w = 5` (only if the pre-flight finds w≠5).

## Testing

- **Pre-flight:** an `#[ignore]` measurement prints the Gemma 1B embedding's
  distinct-exponent count and `w`; the plan proceeds only if `w = 5`.
- **Sidecar round-trip:** decoding the sidecar's blocks reconstructs the same bf16
  embedding bytes as the model's tensor (a parity test on a small slice).
- **Lossless e2e (hard rule):** Gemma 3 1B produces the identical token sequence
  with the streaming lm-head as resident, on a fixed prompt.
- Existing tests stay green; the streaming path is opt-in (env-gated), default
  behavior unchanged.

## Originality & dependencies (doctrine)

- **Original code.** Reuses R148 `streaming_bitplane_gemv` + R143/R146 decode +
  `BitplaneCodec`; the sidecar producer + the Gemma opt-in branch are written from
  scratch. No external runtime.
- **No new dependencies.** `cargo build` only.
- **Lossless by default** preserved: the streaming path is opt-in and proven to
  produce identical output; the default resident path is untouched.

## Components / isolation

- `pack_lmhead_bitplane` (sidecar producer) — standalone; round-trip tested.
- Gemma lm-head opt-in branch (`gemma/api.rs`) — env-gated, additive, default path
  unchanged.
- `streaming_bitplane_gemv` → `pub(crate)` so the adapter can call it.
- Correctness gate — identical-token-sequence test, real Gemma 1B.

## Prior art (cited honestly)

Same lineage as R147/R148 (Apple "LLM in a flash" streaming-from-storage; the
GPU decompress-in-kernel of Cloudflare/DFloat). R149a is the integration step: the
proven streaming kernel wired into a real model's hot path, validated lossless on a
small model before the > RAM speed demonstration.
