# R117: REEFUSE Q8 i8mm Inline Lab

Date: 2026-06-17
Owner: RLLM
Status: rejected lab
Folder: failed

## Hypothesis

R116 found `smmla` correct but ~4.1x slower than tuned f32, blaming the per-block
`target_feature` helper call + memory round-trip. R117 tests the fix: one
`target_feature` function with the `smmla` asm emitted inline per block, the 2x2
int32 tile read into a `vreg` operand (no memory round-trip), and per-block scale +
f32 accumulation via NEON intrinsics with the output tile register-resident.

## Scope

- Mode: exact-lowram lab
- REE kernel lineage: `REEFUSE-Q8-I8MM-LAB` (inline structure)
- Same shape as R116: synthetic Q8 row pair, batch 55, in 2048, blocks 64
- Implementation: `reefuse_smmla_output2_inline` — `vld1q`/`vfmaq`/`vcvtq` intrinsics
  for the f32 accumulator, inline `asm!` with `out(vreg)` for the int32 tile, 8-byte
  `ld1 {v.d}[lane]` operand loads.

## Setup

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench --json target/r117.json --markdown target/r117.md --batch 55 --in-features 2048 --iters 3000
```

## Results

| variant | elapsed ns | speedup vs output2 baseline | max abs diff |
|---|---:|---:|---:|
| `baseline_i8_dot32_output2_batch4` (f32 ref) | 255334209 | 1.000x | 0.0 |
| `reebundle_neon_output2_batch4` (tuned f32) | 38378542 | 6.653x | 0.0 |
| `reefuse_smmla_output2` (R116 naive) | 156129916 | 1.635x | 0.00988 |
| **`reefuse_smmla_output2_inline` (R117)** | **132603708** | **1.926x** | 0.00988 |

Inlining improved `smmla` from `1.635x` to `1.926x` (~18%, the function-call +
round-trip overhead R116 identified) but it is still **~3.5x slower** than the
tuned f32 `reebundle` kernel (132.6ms vs 38.4ms). Correct (diff 0.0099).

## Analysis

R117 fails the lab gate but confirms the structural diagnosis and isolates the
remaining two costs:

1. **Lane loads.** The 2x8 `smmla` operands come from two non-contiguous rows
   (two token rows / two weight rows), so each is built with 8-byte
   `ld1 {v.d}[lane]` loads (16 per block) — much higher latency than the
   contiguous 16-byte `vld1q` loads the f32 kernel uses.
2. **Tiny tile.** A 2x2 output tile produces only 4 results per block of loads;
   `reebundle` produces 8 (output2 x batch4), so it amortizes loads better.

This is now the third int8 structure (R110/R112 sdot, R116 naive smmla, R117 inline
smmla) to lose to the tuned f32 NEON kernel. The conclusion is firm: beating f32
with i8mm requires a real **packed-panel GEMM** — pre-pack activation and weight
panels so the 2x8 operands are contiguous `vld1q` loads, and use a larger register
tile (e.g. 8x8 with multiple `smmla` accumulators) to amortize. That is a
substantial, dedicated kernel-engineering effort (the ggml/llama.cpp approach), not
an incremental lab variant.

## Decision

rejected lab

Reason: inline `smmla` (1.926x) is still ~3.5x slower than tuned f32 `reebundle`
(6.653x); the win needs packed panels + a larger tile, not a structure tweak.

Paper value:

- positive evidence that the inline (no-call, no-round-trip) structure is real
  (+18% over the naive helper) and that `out(vreg)` + intrinsics work on stable
- firm negative evidence (3rd int8 attempt) that incremental int8 kernels cannot
  beat tuned f32 NEON; only a packed-panel GEMM can

## Next Experiment

R118: a packed-panel i8mm Q8 GEMM — pack activations/weights into contiguous 2x8
(then 8x8) int8 panels at prep time, micro-kernel with `vld1q` loads + multiple
`smmla` accumulators per K step + register-resident f32 output, per-block scale via
intrinsics. Lab-gate vs `reebundle` before any runtime work. Banked exact win
remains R115 threading; extending it to MLP down/lm_head is the lower-risk parallel
track.
