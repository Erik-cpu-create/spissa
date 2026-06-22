# Trial: R145 — REEFUSE-PLANE-DOT v2 tile-fused decode⟷bfdot GEMV

Date: 2026-06-19
Owner: RLLM
Status: rejected (speed); kernel correct + bit-identical
Folder: failed

## Hypothesis

R144's naive fused kernel (decode a whole row into an L1 scratch, then `bfdot` the
scratch) was 3.3× slower single-core. The multi-core scout then showed plain bf16
plateaus at ~9.6 ms (bus-bound) while the fused path scales with cores, and the
compressed read floor (7.8 ms) is below 9.6 ms — so a win looked physically
reachable IF (a) decode overlaps the read and (b) decode is fast enough. R145
attacks (a): a **tile-fused** kernel that decodes each 8-weight group straight into
NEON registers and `bfdot`s it in R141's 4-chain order — no L1 scratch round-trip,
decode meant to overlap the dot via out-of-order execution.

## Scope

- Mode: experimental (compressed-resident fused GEMV, optimized kernel)
- REE kernel: REEFUSE-PLANE-DOT v2 (working name; Erik's final call)
- Model/artifact: `Llama-3.2-1B-Instruct-raw.spsa` embedding (vocab 128256 × hidden 2048, 525 MB)
- Architecture: LLaMA 3.2 1B bf16 LM head
- Target device/profile: Apple A18 Pro (2 P + 4 E), macOS; release; 1–8 threads
- Expected bottleneck: per-row decode compute vs DRAM bandwidth
- Bottleneck tag: CPU arithmetic (decode)

## Setup

```bash
# Bit-identical parity (env-free decode+bfdot reference):
cargo test -p rllm-runtime --lib fused_kernel_matches_reference -- --nocapture
# Multi-core bench (plain bf16 vs R145 tile-fused, both bfdot):
cargo test -p rllm-runtime --release fused_kernel_multicore_bench -- --ignored --nocapture
```

Runtime context: release; Apple A18 Pro; bf16 (525 MB) + bit-plane planes (427 MB)
resident; `RLLM_BF16_DOT=1` + `RLLM_Q8_ACTIVATION=1` (both paths bfdot).

## Results

One full lm-head GEMV (262.7M weights), 3 warm iters, by thread count:

| threads | plain bf16 | R145 tile-fused | speedup |
|---:|---:|---:|---:|
| 1 | 24.4 ms | 58.7 ms | 0.42× |
| 2 | 10.9 ms | 36.7 ms | 0.30× |
| 4 | 10.0 ms | 24.3 ms | 0.41× |
| 6 | 9.7 ms | 22.3 ms | 0.44× |
| 8 | 9.7 ms | 21.6 ms | 0.45× |

- **Best fused 21.6 ms vs best plain 9.7 ms → 🔴 NO-GO (speed).**
- **Lossless parity: bit-identical** — the fused kernel equals a `decode_neon_w5` +
  `bf16_row_dot_bf16` reference byte-for-byte (env-free test, green). 291 lib tests pass.
- For reference: R144 naive-fused was 19.9 ms at 6 threads (scout). R145 tile-fused
  is 22.3 ms at 6 — **no improvement; marginally worse.**

## Analysis

The kernel is correct and bit-identical, but the optimization **did not work**, and
that is the informative result:

- **Tile-fusion + OOO overlap bought nothing.** Removing the L1 scratch round-trip
  and interleaving decode with `bfdot` left the time unchanged (22.3 vs R144's 19.9
  ms at 6 threads). The scratch was already L1-cheap, and **decode and `bfdot` do
  not overlap**: each group's `bfdot` is data-dependent on that group's decode, so
  the out-of-order engine cannot hide the decode under the dot. Strategy A's premise
  (overlap) does not hold here.
- **The decode compute is the entire wall.** Both R144 and R145 are decode-bound at
  ~6 Gweight/s/core (the 8-wide `vtbl`/window/shift/join per group), and `bfdot` is
  comparatively free. Fusion cannot speed up the decode itself.
- **Multi-core scales (as the scout showed) but cannot flip it.** Plain bf16 is
  bus-bound and plateaus at 9.7 ms by 2 threads; fused scales with cores but only to
  21.6 ms — still ~2.2× slower, because the decode is 2–3× too slow per weight and
  additive.

So the only untried lever is making the **decode itself** faster — the 16-wide
`vqtbl2q` path (gather 16 exponents/lookup vs 8). Honest assessment of its odds:
it might give ~1.5–2× on the decode (→ ~33–44 Gweight/s aggregate), which is near
the ~34 Gweight/s the scout said is needed. **But** R145 just showed decode does
**not** overlap the read (it is additive), so even a 2× faster decode would be
~6 ms of compute *plus* the ~7.8 ms read ≈ 13.8 ms sequential > 9.7 ms plain —
unless explicit software pipelining (strategy B) also makes the overlap real. So a
win now requires **both** 16-wide decode **and** working pipelining, and even then
lands near break-even. The odds dropped sharply from the scout's optimistic floor.

What stands: the kernel is correct, bit-identical, lossless; the RAM win (19%) is
real; and the measurement cleanly isolates that **decode throughput, not fusion or
overlap, is the sole remaining constraint** — and that it is additive, not hidden.

## Decision

rejected on speed (NO-GO); kernel correct + bit-identical + 19% less RAM.

Reason: tile-fusion + OOO overlap did not improve on R144 (22.3 vs 19.9 ms at 6
threads); plain bf16 stays at 9.7 ms. Decode compute is the entire wall and is
additive to the dot (no overlap). A speed win now needs both 16-wide `vqtbl2q`
decode AND explicit software pipelining, with the e2e win only near break-even —
a much weaker prospect than the scout suggested.

Paper value:

- use as negative evidence / limitation: refines the R140-R144 frontier finding.
  Not only is CPU DRAM bandwidth high enough to make compressed-resident slower
  (R144) — even an optimized fused kernel cannot overlap decode with the dot
  (data-dependent), so decode compute is purely additive. Lossless
  compressed-resident on CPU saves RAM; the decode-vs-bandwidth gap is structural.

## Next Experiment

- **Optional, lower-odds:** 16-wide `vqtbl2q` decode + explicit software pipelining
  (strategy B) — only worthwhile if pursuing the proof to its absolute limit; the
  e2e win is now near break-even at best (own spec, R146). Recommend a quick
  isolated decode-throughput microbench of the 16-wide path FIRST (does it reach
  ~34 Gweight/s?) before building the full pipelined kernel — gate cheaply.
- **Higher-value:** treat the RAM win as the deliverable (opt-in low-RAM lm-head
  mode, ~3× slower, 19% smaller) and redirect speed effort to the q8 layers /
  prefill, which are the real decode levers.
