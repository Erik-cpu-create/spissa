# R116: REEFUSE Q8 i8mm (smmla) Lab

Date: 2026-06-17
Owner: RLLM
Status: rejected lab
Folder: failed

## Hypothesis

R112 showed 4-wide `sdot` ≈ f32. R116 tests whether ARM **i8mm `smmla`** (an int8
2×8·2×8→2×2 matrix-multiply per instruction) beats the tuned f32 output2 kernel in
the microbench lab, before any runtime work.

## Scope

- Mode: exact-lowram lab
- REE kernel lineage: `REEFUSE-Q8-I8MM-LAB`
- Artifact shape: synthetic Llama 3.2 1B-like Q8 row pair (output2)
- Batch 55, in_features 2048, blocks_per_row 64
- Implementation: `reefuse_smmla_output2` — activations pre-quantized to int8
  (per-row scale); for each token pair and weight-row pair, a `target_feature(i8mm)`
  helper `smmla_block32` runs 4 `smmla` over a 32-element block (2×8 lane-loads),
  returns the 2×2 int32 tile, scaled per block in Rust.

Runtime code unchanged. i8mm confirmed available on A18 Pro (`is_aarch64_feature_detected!("i8mm") == true`); `smmla` inline asm compiles on stable.

## Setup

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench --json target/r116.json --markdown target/r116.md --batch 55 --in-features 2048 --iters 3000
```

## Results

| variant | elapsed ns | speedup vs output2 baseline | max abs diff |
|---|---:|---:|---:|
| `baseline_i8_dot32_output2_batch4` (f32 ref) | 235237542 | 1.000x | 0.00000000 |
| `reebundle_neon_output2_batch4` (tuned f32) | 35838458 | 6.564x | 0.00000000 |
| **`reefuse_smmla_output2` (i8mm)** | **146797250** | **1.602x** | **0.00988486** |

The `smmla` kernel is **~4.1x slower** than the tuned f32 `reebundle` kernel
(146.8ms vs 35.8ms), despite a correct result (diff 0.0099 = activation-quant
error only).

## Analysis

R116 fails the lab gate, but the `smmla` instruction itself is correct and fast —
the loss is structural, the same trap as R112's `sdot`:

- `smmla_block32` is a `#[target_feature]` helper called **once per 32-element
  block** (~1728 calls per matmul). On stable Rust, `#[target_feature]` functions
  cannot be `#[inline(always)]`, so every block pays a real function-call cost.
- Each call writes the 2×2 int32 tile to memory (`st1`) and Rust reads it back —
  a register→memory→register round-trip per block.
- Operands are built with 8-byte `ld1 {v.d}[lane]` lane-loads (higher latency than
  contiguous vector loads), because the two token rows / two weight rows are not
  contiguous in memory.

Effective throughput: `smmla` naive ~9 GFLOPS vs f32 `reebundle` ~38 GFLOPS.

A winning i8mm kernel needs the ggml/llama.cpp structure: the entire per-matmul
loop in ONE `target_feature` function (no per-block call), the int32 accumulator
and f32 output kept in registers across the K loop with per-block scale applied via
NEON `scvtf`/`fmul`/`fadd`, and ideally **packed activation/weight panels** so the
2×8 operands are contiguous vector loads. That is a real GEMM micro-kernel, not a
per-block helper.

## Decision

rejected lab

Reason: `reefuse_smmla_output2` is `4.1x` slower than the tuned f32 `reebundle`
kernel (1.602x vs 6.564x over the output2 baseline), with correctness preserved
(diff 0.0099).

Paper value:

- correctness evidence that `smmla` produces the right Q8 result on stable Rust
- negative evidence that a per-block `target_feature` helper structure cannot beat
  tuned f32 NEON — confirms (with R112) that the int8 win requires a fully-inlined,
  register-resident GEMM micro-kernel with packed panels, not a per-block wrapper

## Next Experiment

R117 should build a single-function `target_feature(i8mm)` Q8 GEMM micro-kernel:
whole K loop inlined, int32 tile + f32 output accumulators register-resident,
per-block scale via NEON intrinsics, and packed 2×8 (or 8×8) panels for contiguous
loads. Only then is i8mm worth a runtime gate. Meanwhile the banked, exact win is
R115 threading; extending it to MLP down/lm_head (R118) is lower-risk.
