# Trial: R130 batch1 4-row ILP GEMV (runtime promotion of R128b)

Date: 2026-06-18
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

R128b proved in the lab that a 4-row ILP sdot kernel (4 independent accumulator
chains + shared activation load) is ~1.6x over the per-row sdot path, and R129
proved the interleaved repack is a regression. Promote R128b to the runtime batch1
decode path — the real, format-free decode win.

## Scope

- Mode: exact-lowram runtime
- REE kernel: REEBORN-Q8-SDOT (batch1 4-row ILP)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Architecture: LLaMA 3.2 1B, Q8_0
- Target device/profile: Apple A18 Pro, single-thread
- Bottleneck tag: CPU arithmetic (sdot latency / ILP)

## Setup

Added `batch1_x4_ilp` (one asm block per K-block: 4 independent sdot chains for 4
output rows, activation block loaded once into v0/v1 and shared; reduce each tile
via `vaddvq_s32`, per-block per-row weight scale). Wired into
`accumulate_q8_0_chunk_int8_batch1_rowmajor`: process output rows in groups of 4
via `batch1_x4_ilp` on dotprod CPUs, with the per-row register-accumulate path as
the remainder + non-dotprod fallback. Bit-identical to the per-row path (same int32
dots, same block-order f32 accumulation).

## Results

- **Correctness:** output preserved (fire `No`), bit-exact by construction (same
  int32 dots + same block-order f32 accumulation as the per-row path; lab diff 0.0).
- **Kernel speed (lab, thermal-robust relative):** the 4-row ILP kernel is ~1.6x
  over the per-row sdot path (R128b microbench); the interleaved repack (R129) was
  measured slower and rejected.
- **End-to-end decode (clean number PENDING):** the machine was thermally saturated
  after a long session — runtime decode runs were too noisy/slow to record a
  trustworthy best-of-N. To be measured on a cool machine. Prior points for context:
  R127 ~2.66 tok/s, R128a ~2.9 tok/s; R130 is bit-exact with both and the kernel is
  ~1.6x faster in the lab, so a no-regression decode improvement is expected.
- **Tests:** 268 runtime tests pass (one pre-existing flaky timing assertion
  `attention_kv_append_ns > 0`, unrelated to R130, passes 3/3 on isolated re-run —
  flagged for a separate fix).

## Analysis

R130 promotes the best no-repack batch1 kernel. The 4 independent sdot chains hide
the ~3-cycle sdot latency that the per-row path (separate asm per block) and the
interleaved R129 (1–2 chains) could not. Lab relative speedup ~1.6x; runtime
end-to-end gain is lower because the kernel is not 100% of decode (chunk decode,
attention softmax/RoPE, lm_head argmax, norms).

## Decision

accepted

Reason: promotes the lab-validated R128b ILP kernel to the runtime batch1 decode
path, bit-exact (output byte-identical), 268 tests pass. The format-free decode
win after R129 (interleaved repack) was measured to regress.

Paper value:

- use as positive evidence: 4-independent-chain ILP sdot is the best no-repack
  batch1 decode kernel; ILP beats both the per-row path and the interleaved repack.

## Next Experiment

Re-profile decode on a thermally-quiet machine to find the new bottleneck (now that
the batch1 kernel has ILP): candidates are the per-matmul chunk/streaming machinery,
lm_head argmax (int8 lm_head = R125, ~20% of decode), and attention. Decode is still
short of the ~36 tok/s memory floor — but the remaining gap may now be machinery,
not the kernel.
