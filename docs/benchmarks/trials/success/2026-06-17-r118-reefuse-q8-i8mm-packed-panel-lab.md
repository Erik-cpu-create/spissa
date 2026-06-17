# R118: REEFUSE Q8 i8mm Packed-Panel Lab

Date: 2026-06-17
Owner: RLLM
Status: accepted lab
Folder: success

## Hypothesis

R116/R117 isolated the remaining int8 GEMM cost to (1) `ld1 {v.d}[lane]` 8-byte
lane loads (non-contiguous 2x8 operands) and (2) the tiny 2x2 register tile. R118
tests the ggml/llama.cpp fix: pre-pack the weight-row pair and the activation-row
pairs into contiguous int8 panels so each kernel iteration uses 16-byte `vld1q`
loads, register-resident int32 accumulators, register-resident f32 output, and
per-block scale via `vfmaq_f32`.

## Scope

- Mode: exact-lowram lab
- REE kernel lineage: `REEFUSE-Q8-I8MM-LAB` (packed-panel)
- Artifact shape: synthetic Llama 3.2 1B-like Q8 row pair (output2)
- Batch 55, in_features 2048, blocks_per_row 64
- Implementation:
  - `pack_weight_panel_pair_lab` packs the two weight rows per K-block into
    4 segments of 16 contiguous bytes `[row0_K0..7 | row1_K0..7]` (skips fp16
    block-scale prefix; lab uses a single scalar `scale`, matching the other
    output2 variants).
  - `pack_act_panel_pair_lab` packs each consecutive pair of batch (token) rows
    the same way: 4 segments of `[t0_K0..7 | t1_K0..7]` per K-block.
  - `reefuse_smmla_panel_output2` (`#[target_feature(enable = "i8mm")]`) runs the
    whole K loop in one function. Per K-block, one inline `asm!` emits four
    `ld1 {v0.16b}, [{a}], #16` + `ld1 {v1.16b}, [{w}], #16` + `smmla {acc:v}.4s,
    v0.16b, v1.16b` substeps, producing one int32x4 tile per block. The tile is
    folded into a register-resident f32 accumulator via
    `vfmaq_f32(acc_f, vcvtq_f32_s32(tile_acc), scale_vec)`, where
    `scale_vec = [s_t0, s_t0, s_t1, s_t1] = scale * act_scales[...]`. Odd-batch
    tail (1 row) uses a scalar int8 dot over the raw `q8_pair` / `act_i8`.
- Packing: weight and activation panels are packed once **outside** the timed
  loop. This matches `quantize_rows_i8` / `reedot_i8_vdot` convention: at runtime
  the weight panel is pre-packed at prep time and the activation panel packs
  once per matmul, amortized across `out_features` (~thousands of inner kernel
  calls per pack).

## Setup

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench --json target/r118.json --markdown target/r118.md --batch 55 --in-features 2048 --iters 3000
```

## Results

| variant | elapsed ns | speedup vs output2 baseline | max abs diff |
|---|---:|---:|---:|
| `baseline_i8_dot32_output2_batch4` (f32 ref) | 247268291 | 1.000x | 0.0 |
| `reebundle_neon_output2_batch4` (tuned f32) | 37410125 | 6.610x | 0.0 |
| `reefuse_smmla_output2` (R116 naive) | 159992334 | 1.546x | 0.00988 |
| `reefuse_smmla_output2_inline` (R117 inline) | 136479750 | 1.812x | 0.00988 |
| **`reefuse_smmla_panel_output2` (R118)** | **7849750** | **31.500x** | **0.00988** |

R118 is the first int8 lab variant to beat the tuned f32 `reebundle` kernel:
**~4.77x faster than tuned f32** (7.85ms vs 37.41ms). The progression is
consistent with the analysis chain: naive helper 1.55x → inline structure 1.81x
→ packed-panel + contiguous loads 31.5x. Correctness matches the prior `smmla`
variants exactly (max abs diff 0.00988 = activation-quantization error only,
already R111-validated on the real model).

## Analysis

R118 passes the lab gate decisively. What the structure actually changed,
compared to R117:

- **Contiguous loads.** 4 × `vld1q` 16-byte loads per K-block replace 32 × 8-byte
  lane loads. `vld1q` is much lower latency and pairs better on Apple cores.
- **Bigger amortization per load.** Each pair of `vld1q` (one act, one weight)
  feeds one `smmla` that produces 4 int32 outputs (2 token rows × 2 weight rows).
  Per K-block: 8 loads → 4 smmla → 4 partial outputs × K=32 lanes = full output2
  contribution. R117 inline needed 16 lane-loads + 4 smmla for the same outputs.
- **Register-resident output accumulator.** The f32 output for the pair sits in
  one `float32x4_t` across the entire K loop. R117 wrote the int32 tile to memory
  every block (`out(vreg)` still required a write-back; the new design also gets
  to keep f32 in a register).

What R118 still does NOT model:

- Pack overhead for activations on every matmul. In the lab it is outside the
  timed loop (consistent with prior int8 variants). At runtime it is one shuffle
  pass over `batch × in_features` int8 bytes per matmul — amortized across
  hundreds of out_features chunks per pack, but not zero.
- Weight pre-packing at `.rllm` pack time. The `.rllm` container currently stores
  the Q8 format directly; a runtime promotion needs either an opt-in packed-panel
  layout in the container, or a one-shot pack on artifact load (memory cost).

## Decision

accepted lab

Reason: `reefuse_smmla_panel_output2` runs in `7.85ms` (31.5x over the output2
baseline, **4.77x over the tuned f32 `reebundle` kernel**), with correctness
preserved at the activation-quant tolerance (max abs diff 0.00988, same as prior
int8 variants).

Paper value:

- positive evidence that the ggml-style packed-panel + register-resident
  micro-kernel is the structure needed to beat tuned f32 NEON
- isolates the 3-attempt int8 plateau (R110/R112/R116/R117) to load-layout and
  tile-size, not the int8 instructions
- direct stepping stone to R119 runtime promotion: the kernel itself is proven;
  the open question is whether activation-pack + a runtime weight-panel layout
  carry the lab win through the prefill loop

## Next Experiment

R119: promote `REEFUSE-Q8-I8MM-PANEL` into the runtime under a same-turn f32
control, gated to batch>1 (prefill) so batch1 decode stays on the f32 fast path
(R111/R112 lesson). Open design choices: (a) pack the activation per matmul into
a scratch buffer reused across chunks, and (b) pack the weight per chunk at
recall time vs adding an opt-in pre-packed panel layout to `.rllm` at pack time.
R111 logit parity must hold; R115 threading should still apply. The bigger
8x8 / 4x4 tile variants and MLP down/lm_head extensions are R120+.
