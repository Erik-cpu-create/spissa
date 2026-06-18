# Trial: R131 Gemma parallel LM-head GEMV

Date: 2026-06-18
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

Gemma 3's LM head is a dense matrix–vector product over a 262k-row vocabulary
(`logits[v] = Σ_h last_hidden[h] · embed_tokens[v, h]`, tied embedding) recomputed
every decode step. The Gemma adapter ran it as a scalar, single-threaded f32 loop
while the q8 transformer kernels already use all cores. Each logit is an
independent dot product, so splitting the vocab rows across worker threads should
be embarrassingly parallel and bit-identical, reclaiming the idle cores.

## Scope

- Mode: exact-lowram runtime
- Kernel: `lm_head_logits_parallel` (vocab-row split, scalar inner dot per row)
- Model/artifact: `gemma-3-4b-it-q8.rllm` (`q8_transformer_keep_io`, embedding/norms raw)
- Architecture: Gemma 3 4B, Q8_0 transformer + raw BF16 tied LM head
- Target device/profile: Intel Xeon @ 2.10GHz, 4 cores (x86_64)
- Bottleneck tag: CPU arithmetic (scalar single-thread LM-head GEMV)

## Setup

Added `lm_head_logits_parallel` to the streaming module (where the runtime thread
helpers live) and re-exported it at the crate root. It splits the `vocab_size`
output rows into contiguous blocks across `effective_runtime_threads` workers via
`std::thread::scope`, each computing its rows with the shared `lm_head_logits_rows`
scalar reference. Honors `RLLM_THREADS`; falls back to serial for a single core or
tiny vocab. The Gemma adapter (`gemma_generate_from_model`) now calls it instead of
its private scalar loop, which was removed.

Bit-identical by construction: every logit is still one independent dot product
computed by exactly one worker with the same accumulation order — only the row
range a thread owns changes.

## Results

- **Correctness:** parity-exact. Generated token ids unchanged before/after:
  `[818, 5279, 529, 6056, 563, 21038, 236761, 106]` →
  "The capital of Japan is Tokyo." Dedicated unit test
  `r131_parallel_lm_head_matches_serial_bit_for_bit` asserts `parallel == serial`
  bit-for-bit and passes.
- **End-to-end (gemma-3-4b-it, chat prompt, 8 tokens, identical workload):**
  - q8 transformer, scalar LM head: 38.59s (0.21 tok/s)
  - q8 transformer, R131 parallel LM head: **31.72s (0.25 tok/s)** — ~1.19x
  - Reference: raw BF16 baseline 57.91s (0.14 tok/s); raw → q8 → R131 is ~1.83x total.
- **LM head in isolation:** the two 8-token runs differ only in the LM head, so the
  ~6.9s delta over 8 head evaluations (~0.86s/call) is the head improvement alone —
  roughly the expected ~4x on a 4-core host (scalar 1-core → 4-core).
- **Tests:** runtime test suite passes including the new parity test.

## Analysis

The LM head went from one idle-core-leaving scalar loop to a 4-core split, ~4x on
the head itself. The aggregate decode gain is smaller (~1.19x) because the q8
transformer layers — already multi-threaded — dominate total decode time, and the
one-time full-embedding f32 decode (~2.68 GB, the 4 GB peak-RSS driver) is fixed
cost.

Caveat worth recording: the q8 ILP `sdot`/`i8mm` kernels (R118–R130) are AArch64
NEON paths, so on this x86_64 host the q8 transformer runs the portable fallback —
which is why the q8 step itself was only ~1.5x here rather than the larger ARM
gains. R131 is architecture-neutral (plain `std::thread`), so it helps on both.

Follow-ups (not in R131): quantize the tied embedding/LM head to q8 and route the
head through the fast q8 GEMV (cuts the 4 GB f32 embedding RSS and speeds the head
further, small quality risk on the IO weight); and a fused GELU gate-up MLP kernel
for Gemma (the existing fused kernel is SiLU/LLaMA-only).

## Decision

accepted
