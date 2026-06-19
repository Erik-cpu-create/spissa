# Spec: R145 — Tile-fused decode⟷bfdot lm-head GEMV (the optimized kernel)

Date: 2026-06-19
Status: design (approved to draft)
REE kernel (working name): **REEFUSE-PLANE-DOT v2** (Erik's final call before any report/paper use)

## Honest positioning (read first)

R144 measured the *naive* fused kernel (decode a whole row into an L1 scratch,
then `bfdot` the scratch) single-core and got 3.3× slower → NO-GO. That was one
implementation, not the verdict on the idea. The R145 multi-core **scout** then
showed the physics is favorable:

- Plain bf16 GEMV is **bandwidth-bound**: it plateaus at **~9.6 ms / token** at ≥2
  threads (bus saturated at ~55 GB/s). More cores don't help it.
- The fused path is **compute-bound** (decode): it keeps scaling with cores
  (60 → 19.9 ms from 1 → 6 threads). The single-core 3.3× gap narrowed to **2×**
  at 6 threads.
- The **compressed read floor is 7.8 ms** (427 MB at 55 GB/s) — **below** plain
  bf16's 9.6 ms. So if decode is fast enough AND overlapped with the read, the
  fused GEMV can be **~1.2× faster, lossless, at 19% less RAM**.

The target is concrete, from the scout: **decode ≥ ~34 Gweight/s aggregate AND
overlapped with the read.** Current naive fused is ~22 Gweight/s aggregate
(decode-bound portion) and *additive* (not overlapped). R145 attacks both gaps
with a properly engineered kernel. This is honest kernel engineering toward a
measured target — not a guaranteed win. The gate decides.

## Goal

Replace the naive per-row decode-then-bfdot with a **tile-fused** kernel: process
each weight row in 32-weight tiles, decoding each tile straight into NEON
registers and `bfdot`-accumulating it in the same pass — no L1 scratch round-trip,
and decode/dot overlap via the CPU's out-of-order engine. Decode 16-wide
(`vqtbl2q`). Multi-thread the GEMV. Benchmark vs plain bf16 multi-core (9.6 ms
floor) and report GO/MARGINAL/NO-GO. Lossless throughout.

## Design

### 1. Tile-fused row kernel `bitplane_row_dot_bfdot` (rllm-runtime)

`#[cfg(target_arch = "aarch64")] #[target_feature(enable = "bf16")]`, inline asm /
intrinsics, in `streaming/bitplane_gemv.rs`. Computes one logit = dot(activation,
decoded weight row) directly from the row's bit-plane planes:

```
unsafe fn bitplane_row_dot_bfdot(
    act_bf16: &[u16],      // activation, converted to bf16 once (shared across rows)
    pal_tbl: &Palette128,  // 32-entry palette preloaded into 2 uint8x16 regs (for vqtbl2q)
    idx_row: &[u8],        // open-ended slice at the row's index-plane start
    res_row: &[u8],        // open-ended slice at the row's residual-plane start
    hidden: usize,
) -> f32
```

- 4 independent f32 bfdot accumulator chains (mirrors R141 `bf16_row_dot_bf16`).
- Per 32-weight tile: decode 32 indices (32×5 = 160 bits = 20 bytes) into 4×
  `uint16x8` bf16 weight vectors — **16-wide unpack** (process 16 indices at a
  time via a 2-byte window + per-lane shift, gather exponents with `vqtbl2q_u8`
  over the 32-byte palette, zip exponent+residual into bf16); load the matching 4×
  activation `.8h`; issue 4 `bfdot`. Weights stay in registers — never stored to
  memory. The decode of tile N+1 and the `bfdot` of tile N occupy independent
  execution units, so the out-of-order engine overlaps them (strategy A).
- Horizontal-add the 4 chains; scalar tail for `hidden % 32` and for the final
  row whose 16-wide group load would read past the plane (the existing
  `simd_groups`-style guard).
- **Open-ended plane slices** (`&idx_plane[row*1280..]`, `&residuals[row*hidden..]`)
  so group loads stay in-bounds (same rule as R144).

### 2. GEMV wrapper `lm_head_logits_bitplane_fused`

Convert the activation to bf16 once (`Bf16DotActivation` / `convert_f32_to_bf16`),
preload the palette into the `vqtbl2q` register pair once, then loop rows calling
`bitplane_row_dot_bfdot`. A multi-threaded variant splits the vocab rows across
threads (reusing the `lm_head_logits_parallel_bf16` split pattern); each thread
shares the planes (`&[u8]`, Send+Sync) and owns a disjoint output slice.

### 3. Fallback (strategy B, only if A misses)

If the tight-loop/OOO kernel does not clear the 9.6 ms gate, add explicit software
pipelining: hold the decoded bf16 of tile N+1 in registers while issuing tile N's
`bfdot`, so the overlap is guaranteed rather than scheduler-dependent. Same
external interface; an internal change. Decided by the gate, not up front.

### Gate (from the scout, multi-core)

Both paths multi-threaded at the same thread count, on the real 525 MB embedding:

| fused ms/token (best thread count) | verdict |
|---|---|
| ≤ ~9.6 ms (beats plain bf16) | 🟢 **GO** — faster + 19% less RAM, lossless |
| ~9.6–12 ms | 🟡 **MARGINAL** — RAM win, ~par speed; decide per goal (try strategy B) |
| > ~12 ms | 🔴 **NO-GO** — decode still loses; the naive-fused NO-GO stands, frontier closed for speed |

The scout already provides the plain-bf16 baseline curve (9.6 ms at 6 threads) and
the naive-fused curve (19.9 ms) for direct before/after comparison.

## Non-goals

- `--fast` runtime wiring / `codec_for_id` registration / model packing — gated
  follow-up if GO.
- General `(hidden, w)`; only `w=5`, `hidden=2048` (the tied bf16 embedding).
- Beating ~1.2×/19% (the bf16 lossless ceiling). GPU. q8/q4 layers. KV-cache.

## Testing

- **Lossless parity (hard rule):** `bitplane_row_dot_bfdot` (and the GEMV wrapper)
  produce logits **bit-identical** to R144's `lm_head_logits_rows_bitplane` (which
  is already proven bit-identical to plain bf16) when it mirrors R141's 4-chain
  accumulation order exactly — the unit test asserts exact f32 equality vs the R144
  kernel on a small synthetic embedding. If an optimized reduction legitimately
  reorders the f32 accumulation (a few ULPs), fall back to the existing bf16
  convention used by the R137 lm-head tests — `max_abs_diff` within tolerance AND
  argmax preserved — and document it. The weights stay bit-exact either way; only
  f32 summation order may differ.
- **Decode-unpack parity:** the 16-wide tile unpack decodes the same indices as
  the scalar/8-wide path (covered transitively by the logit parity; an optional
  direct check on a tile is fine).
- **Honest metrics:** the multi-core bench reports plain-bf16 vs tile-fused
  ms/token across thread counts + the verdict — MARGINAL/NO-GO stated plainly.
- Existing rtc-codec + rllm-runtime tests stay green (additive kernel).

## Originality & dependencies (doctrine)

- **Original code.** A tile-fused decode⟷dot kernel composing in-house REEPLANE
  decode + R141 bfdot, written from scratch here. No external runtime.
- **No new dependencies.** NEON/bf16 via `std::arch` / inline asm; `cargo build`
  only.
- **Lossless by default** preserved — bit-exact weights, proven by the parity test.

## Components / isolation

- `bitplane_row_dot_bfdot` + `Palette128` preload (rllm-runtime
  `streaming/bitplane_gemv.rs`) — the tile-fused row kernel, parity-tested vs R144.
- `lm_head_logits_bitplane_fused` (+ multi-threaded variant) — the GEMV wrapper.
- Multi-core bench (from the scout, extended to include the new kernel) — `#[ignore]`,
  prints the before/after table + verdict.
- `--fast` wiring — separate follow-up spec, gated on the verdict.

## Prior art (cited honestly)

Fusing entropy/dictionary decode into the matmul inner loop so weights never
materialize is the GPU SOTA pattern (DFloat11, NeuZip, Cloudflare Unweight). R145
is the CPU/ARM realization: a tile decoded into NEON registers and consumed by a
native `bfdot` in the same pass, multi-threaded, lossless — the open CPU/edge gap
the R140 arc targets, now pursued to the optimized kernel the scout showed is
within physical reach.
