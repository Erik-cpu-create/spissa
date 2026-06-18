# R119: REEFUSE Q8 i8mm Packed-Panel Runtime

Date: 2026-06-17
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

R118 proved the packed-panel `smmla` kernel is ~4.77x faster than the tuned f32
`reebundle` kernel in the lab. R119 promotes it into the real runtime Q8 matmul
(`accumulate_q8_0_chunk`, behind `RLLM_Q8_ACTIVATION`), gated to batch>1 prefill,
and checks that prefill speeds up while token/logit parity holds on the real
model.

## Scope

- Mode: exact-lowram runtime
- REE kernel lineage: `REEFUSE-Q8-I8MM-PANEL`
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Device: Apple A18 Pro, single-thread (`RLLM_THREADS=1`)
- Bottleneck tag: CPU arithmetic / Q8 i8mm GEMM
- Implementation (`accumulate_q8_0_chunk_panel_smmla`): engages when the CPU has
  i8mm, `batch >= 2`, `in_features % 32 == 0`, the chunk is output-row-aligned
  (`element_start % in_features == 0`, block count a multiple of blocks_per_row).
  Activations are quantized to int8 per 32-block and packed into pair-major panels
  **once per matmul**, cached thread-local by (ptr, len, shape, fingerprint) so the
  pack amortizes across all of a matmul's chunks. Each output-row pair packs its
  weight into a scratch panel; `smmla_accumulate_output_pair` runs the R118 kernel
  over all batch rows, with scalar int8 tails for the odd batch row and odd output
  row. Per-block weight + activation scales (not the lab's single scale) preserve
  the R111 parity convention. Non-eligible chunks / non-i8mm CPUs / batch1 decode
  fall back to the existing R111 naive int8 path; `RLLM_Q8_PANEL=0` forces the
  fallback.
- Engaged shapes on this model (instrumented): batch=53, in=2048 out∈{2048,512,8192}
  (attention q/k/v/o + MLP gate/up), and in=8192 out=2048 (MLP down).

## Setup

```bash
cargo build --release -p rllm-cli --bin llama-test
# control (f32) vs candidate (i8mm panel)
printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | RLLM_THREADS=1 target/release/llama-test --model <artifact> --chat-template llama3 --max-new-tokens 4 --rama-integrity unchecked
printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | RLLM_THREADS=1 RLLM_Q8_ACTIVATION=1 target/release/llama-test --model <artifact> --chat-template llama3 --max-new-tokens 4 --rama-integrity unchecked
```

## Results

Output stayed `No`; peak transient unchanged at `1,050,673,152 bytes`.

| config | prefill (best of 3) |
|---|---:|
| `RLLM_THREADS=1` (f32 control) | 6.83s |
| `RLLM_THREADS=1` + i8mm panel | **3.48s** |

- **~2.0x faster prefill** single-thread (6.83s → 3.48s).
- **Parity holds:** first-token full-vocab logits top-1 match, top-10 overlap
  10/10, max abs diff `0.2997` — identical to the R111 activation-quant error, so
  the panel adds no error beyond the already-validated int8 activation step.
- 12 new panel unit tests pass (even/odd batch, odd output row, multi-chunk,
  single-row chunks, real q/kv-proj chunk patterns, in=8192 down shape); 75
  streaming tests pass total.

## Analysis

R119 accepts: the R118 kernel carries a real ~2x prefill win into the runtime with
exact output. The runtime speedup (2x) is below the lab kernel speedup (4.77x)
because the matmul is only part of prefill — chunk decode, lm_head, attention
softmax, and the per-matmul activation pack are not accelerated by this kernel.

### Debugging note (the bug that cost the most)

The first runtime version produced wrong tokens (`No` → newline, logit diff 72)
while every isolated unit test passed. Root cause: the inner `asm!` wrote the
int32 tile to memory via `st1 ..., [{out}]` where `{out}` was passed as
`in(reg) tile.as_mut_ptr()`. Inline asm declared the pointer only as an input, so
the compiler did not know the pointed-to memory was clobbered and optimized
assuming `tile` was unchanged — an optimization-context-dependent heisenbug that
unit tests (different optimization context) could not catch. Fixed by reading the
accumulator through a proper `out(vreg) tile_acc: int32x4_t` operand and storing
to the array with `vst1q_s32` in Rust. Lesson: never write memory from inline asm
through a pointer passed as `in(reg)`; use a typed output operand.

## Decision

accepted

Reason: `RLLM_Q8_ACTIVATION=1` cut single-thread prefill from `6.83s` to `3.48s`
(~2.0x), output stayed `No`, first-token logits matched the f32 control (top-1
match, top-10 10/10, max abs diff 0.2997 = activation-quant only), and peak
transient is unchanged.

Paper value:

- first runtime kernel to beat tuned f32 Q8 prefill end-to-end (the int8 path
  finally pays off, via the packed-panel structure)
- exact within the activation-quant tolerance already validated by R111
- documents a real inline-asm UB class to avoid

## Next Experiment

R120: make the i8mm panel **stack with threading**. Currently panel single-thread
(~3.5s) ≈ R115 threading 6-core (~3.5s) but they do not stack (R115 splits batch
rows, fragmenting the batch the panel wants whole). Thread the panel over output
pairs instead (full batch, shared packed activation, disjoint output columns) to
target ~1–1.5s. Then reduce the per-matmul activation-pack overhead and extend the
kernel to lm_head / attention to close further on Ollama.
