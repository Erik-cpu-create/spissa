# Spec: R141 — `bfdot` exact-bf16 GEMV (fast lossless-weight compute)

Date: 2026-06-19
Status: design (approved to draft)
REE kernel (working name): **REEFLOW-BF16-DOT** (final call: Erik)

## Honest positioning (read first)

This closes the "fast + lossless" gap the project kept hitting — but only under a
precise definition of lossless, chosen deliberately:

> **Lossless = the model WEIGHTS are exact** (bit-for-bit bf16, never q8-rounded).
> Compute happens at the model's **native bf16 precision** via ARM `bfdot`.

This is NOT "bit-exact-equal-to-full-f32 output." `bfdot` multiplies bf16 inputs
and accumulates in f32, so its result differs from upcasting everything to f32 by a
small, bf16-input-precision amount. The trade we accept (Option A, user's call):

- **Weights:** 0% error (vs q8_0's measured 0.56% RMS). Strictly more faithful than q8.
- **Activations:** rounded f32→bf16 once per GEMV (more precise than the int8
  activation quant the q8 `--fast` path already uses).
- **Speed:** ~2× the current exact path (8 bf16/instr vs 4 f32-FMA after upcast).
  Still slower than q8 int8 `sdot` (16 int8/instr) — exact weights cost ~2× vs q8.
  `bfdot` does NOT beat q8 on raw speed; it makes the EXACT path fast enough to be
  usable. q8 `--fast` stays the fastest (lossy-weight) mode.

Why this matters for the project mission (CPU-only, low-RAM, low-energy local LLM,
no GPU required): it gives an **exact-weight** model that runs at a practical decode
speed on CPU — a capability llama.cpp/Ollama do not offer (they go fast only by
quantizing weights). Combined with Phase 2 (compressed-resident, below) the end
state is: exact weights, reduced RAM, computed at native bf16, all on CPU.

## The combined vision (two phases)

The two research directions discussed are complementary, not competing:

- **Phase 1 — `bfdot` compute (THIS spec, implementable detail).** Wins *speed* for
  exact-weight matmul. Independent of the codec. Target: the tied bf16
  embedding / LM head (already bf16 in the `keep_io` q8 models, memory/compute
  bound, easy to measure in isolation).
- **Phase 2 — codec fast-decode + compressed-resident (design-level here; its own
  spec later).** Wins *RAM*. The R140a `rtc-dfloat-v1` codec is lossless (10.626
  bits/weight) but decodes at 0.02 GB/s because the `BitReader` reads bit-by-bit and
  the stream is one monolithic blob per tensor (no parallel framing). Phase 2 fixes
  exactly that (buffered/SIMD bit-reader + per-row framing), then feeds decoded bf16
  tiles into the Phase 1 `bfdot` kernel. Designed below; gated behind Phase 1
  success and a fresh feasibility gate.

This spec implements Phase 1 fully and positions Phase 2.

## Goal

Replace the current exact-bf16 GEMV path (bf16→f32 upcast + `vfmaq_f32`) at the
LM-head / embedding with an ARM `bfdot` kernel: load bf16 weights directly (no
conversion, weights stay exact), dot against a once-converted bf16 activation
scratch, accumulate in f32. Gated on runtime `FEAT_BF16` detection with the existing
f32 path as fallback. Report decode speedup and the bf16-vs-f32 parity diff honestly.

## Design

### Key structural insight: convert the activation once

The LM head is a GEMV — one activation vector dotted against many weight rows. So the
`f32→bf16` activation conversion is done **once** into a reusable scratch buffer and
shared across all output rows (mirroring how the q8 path caches int8 activations via
`with_q8_panel_activations`). The per-row cost is then pure `bfdot`; activation
conversion is amortized to negligible.

### 1. Kernel `bf16_row_dot_bfdot` (`crates/rllm-runtime/src/streaming/mod.rs`)

- `#[cfg(target_arch = "aarch64")] #[target_feature(enable = "bf16")]`, inline asm
  `bfdot` — consistent with the existing fast kernels (`i8_dot32_sdot`,
  `smmla_*` are all inline asm; this avoids any dependence on bf16 intrinsic
  stabilization in stable Rust, keeping `cargo build` the only requirement).
- Loads weight row as `.8H` vectors directly from the bf16 weight bytes (little-endian
  u16, no conversion — weights remain exactly as stored).
- Loads activation `.8H` from the bf16 scratch.
- `bfdot Vacc.4s, Vw.8h, Va.8h` accumulating 8 bf16 products per instruction into a
  4-lane f32 accumulator; 4 independent accumulator chains for ILP (mirrors
  `batch1_x4_ilp`). Horizontal-add + scalar tail for the remainder.
- Returns f32 (same contract as `bf16_row_dot_f32`).

### 2. Activation conversion helper

A small `#[target_feature(enable = "bf16")]` routine converting an f32 slice to a
bf16 scratch `Vec`/buffer using `BFCVTN` (round-to-nearest-even), 4 f32 → 4 bf16 per
instruction, called once per GEMV. Scalar RNE fallback for the tail.

### 3. Runtime gate `bf16_dot_available()`

Mirrors `q8_sdot_available()` (`kernels.rs:255`): `is_aarch64_feature_detected!("bf16")`,
cached in a `OnceLock`. No compile-time restriction; binary runs everywhere and falls
back.

### 4. Dispatcher + override

`bf16_row_dot_f32` (the existing dispatcher at `mod.rs:199`) gains a branch: if
`bf16_dot_available()` and not disabled by env, use `bf16_row_dot_bfdot`; else the
current upcast→f32-FMA path. An env override `RLLM_BF16_DOT=0` forces the f32 path
(for parity testing and for users who want bit-identical-to-f32 behavior). Default:
bfdot when available.

### 5. Wiring

Engage the kernel where the LM-head / argmax bf16 dot is consumed:
`raw_16bit_argmax_rows_range` (`streaming/argmax.rs`, the R137 path). No change to the
embedding read path's exactness — only the dot kernel changes.

## Implementation phases (de-risk inside one plan)

1. **Feasibility gate (run FIRST, before wiring).** A micro-benchmark / `#[ignore]`
   measurement: (a) does the test device report `FEAT_BF16`? (b) does
   `bf16_row_dot_bfdot` match an f32 reference within bf16 tolerance on random +
   real rows? (c) ns/row of bfdot vs the f32-upcast kernel. **GO only if bf16 is
   present AND bfdot is faster.** If not, record the negative result honestly and
   stop — the fallback already preserves correctness. This is the honest gate, not
   optional.
2. **Kernel + activation-convert helper + gate + dispatcher.** Unit-tested in
   isolation.
3. **Wire into the LM-head/argmax path; measure end-to-end** decode tok/s + output
   parity (argmax match) vs the f32 path and vs q8 `--fast`.

## Phase 2 (design-level only — separate spec when Phase 1 lands)

Make `rtc-dfloat-v1` decode fast enough for compressed-resident:

- **Buffered bit-reader:** replace the per-bit `peek` loop (`dfloat.rs:84`) with a
  64-bit window + byte refill, so a Huffman symbol decode is one LUT lookup + one
  shift, not 15 div/mod/shift iterations. Expected the single biggest win.
- **Per-row (tile) framing:** the codec currently emits one monolithic exp-stream +
  residual blob per tensor (`dfloat.rs:318`). Add independent per-row bitstreams so
  rows decode in parallel across cores and `decode_range` can decode one tile for
  fusion.
- **Fused decode→bfdot:** decode a weight tile from the compressed stream into a
  small cache-resident bf16 scratch, then feed Phase 1's `bf16_row_dot_bfdot` in the
  same pass — exact weights, reduced RAM, native-bf16 compute.
- **Fresh feasibility gate:** measure decode GB/s after the buffered reader. Proceed
  to fused/resident only if decode no longer dominates the matmul. (R140a's gate
  measured 0.02 GB/s with the naive reader → that gate stays the baseline to beat.)

## Non-goals

- Beating q8 `--fast` on raw speed (exact weights cost ~2× vs q8 — stated, accepted).
- GPU. Sub-bf16-precision compute. Novel compression research.
- Phase 2 implementation in this plan (designed, not built here).
- KV-cache compression.

## Testing

- **Correctness:** `bf16_row_dot_bfdot` vs an f32 reference dot, within a documented
  bf16-input tolerance, on random and real weight rows.
- **Argmax parity:** on a real model's logits, bfdot vs f32 select the same top token
  on clear prompts (the lossless-weight contract is about weights; argmax stability is
  the user-visible check).
- **Honest metrics:** decode tok/s before→after; the bfdot-vs-f32 `max_abs_diff` +
  argmax-match reported as the accepted bf16-compute diff; the remaining gap vs q8
  `--fast` stated as a limitation.
- **Fallback:** with `RLLM_BF16_DOT=0` (and on non-bf16 / non-aarch64), the f32 path
  runs and existing tests stay bit-identical.
- Existing q8/bf16 tests stay green (the kernel is additive + gated).

## Originality & dependencies (doctrine)

- **Original code.** `bfdot`/`bfmmla` are ARM ISA instructions (FEAT_BF16); using them
  is using the hardware, exactly as using `sdot` is using the dotprod ISA. The kernel,
  the once-converted-activation scheme, the gate, and the dispatcher are written from
  scratch in this repo. No port of any external runtime.
- **No external dependencies.** Inline asm via `std::arch` (built in). No bf16 crate,
  no nightly intrinsics, no generic math library. `cargo build` stays the only
  requirement (the reason asm was chosen over intrinsics).

## Components / isolation

- `bf16_row_dot_bfdot` + activation-convert helper + `bf16_dot_available()` gate
  (rllm-runtime streaming) — testable standalone against the f32 reference.
- Dispatcher branch in `bf16_row_dot_f32` — additive, gated, env-overridable.
- LM-head/argmax wiring — the only call-site change.
- Phase 2 codec work — separate crate (`rtc-codec`) + separate spec.

## Prior art (cited honestly)

ARM `bfdot`/`bfmmla` are standard ISA; llama.cpp/ggml use bf16 and dotprod kernels on
CPU. RLLM's distinct position is the *combination*: exact (un-quantized) weights kept
losslessly, computed at native bf16 on CPU/ARM, with a path to compressed-resident —
the open CPU gap identified in the R140 work.
