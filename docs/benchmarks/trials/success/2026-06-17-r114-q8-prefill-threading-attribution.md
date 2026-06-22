# R114: Q8 Prefill Threading Attribution

Date: 2026-06-17
Owner: RLLM
Status: accepted diagnostic
Folder: success

## Hypothesis

After R112 rejected the int8 `sdot` kernel (within noise of tuned f32), the
remaining large lever for the ~170x single-thread Ollama prefill gap is suspected
to be threading: Ollama uses all CPU cores, RLLM is benchmarked at
`RLLM_THREADS=1`. R114 measures whether RLLM's Q8 prefill scales with cores and,
if not, attributes why.

## Scope

- Mode: exact-lowram diagnostic
- REE kernel: none (attribution only)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Device: Apple A18 Pro (6 cores: 2 perf + 4 eff), 8 GB
- Bottleneck tag: scheduler / CPU parallelism

## Setup

```bash
# same prompt, vary RLLM_THREADS
for T in 1 6; do
  printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | RLLM_THREADS=$T target/release/llama-test --model <artifact> --chat-template llama3 --max-new-tokens 4 --rama-integrity unchecked
done
# code audit: grep thread::scope / thread::spawn in the streaming runtime
```

## Results

| RLLM_THREADS | prefill | speedup vs 1 |
|---|---:|---:|
| 1 | 7.55s | 1.00x |
| 6 (all cores) | 6.74s | 1.10x |
| auto | 7.77s | 0.97x (decode also worse) |

Code audit (`crates/rllm-runtime/src/streaming/`):

- `thread::scope` exists only in `argmax.rs` (lm_head argmax) and
  `kernels.rs::parallel_sparse_raw_16bit_linear_chunk_batch1` (experimental-speed
  sparse, batch1).
- The exact Q8 prefill matmul kernels — `accumulate_q8_0_chunk` (attention
  q/k/v/o + MLP gate/up via tile-linear) and `accumulate_q8_0_chunk_multiply_into`
  (MLP down) — contain **no** `thread::scope`/`thread::spawn`. They are fully
  single-threaded.

Prefill phase split (control, single-thread): MLP ~5.7s, attention ~1.0s,
lm_head ~1.1s of ~7.8s.

## Analysis

R114 explains the ~1.1x core scaling: only the lm_head argmax (~14% of prefill)
is parallelized, so 6 cores save ~7% overall while the ~82% MLP + attention exact
Q8 matmul never leaves one core. R15 (projection row parallelism) was rejected
earlier, but that was **per-chunk** threading on the **raw** path with an
auto-thread default that regressed — not evidence that the Q8 prefill matmul
cannot be parallelized.

The decoded chunk granularity makes per-chunk-internal parallelism viable: gate/up
weights are `[8192, 2048]` in 18 chunks (~455 output rows/chunk), so each already
decoded `q8_bytes` chunk carries hundreds of rows of independent work. The
existing sparse path already parallelizes *inside* a decoded chunk via
`thread::scope`, so the model `&mut`/budget thread-safety issue is avoidable.

## Decision

accepted diagnostic

Reason: measured 6-core prefill scaling is only ~1.10x; code audit confirms the
exact Q8 prefill matmul is single-threaded and only lm_head argmax / sparse paths
use threads.

Paper value:

- high-value attribution: the dominant untapped lever for the Ollama prefill gap
  is parallelizing the exact Q8 matmul, not (only) the int8 kernel
- corrects the framing that single-thread kernel tuning (R78–R112) can close the
  gap on its own

## Next Experiment

R115 should parallelize `accumulate_q8_0_chunk` by **batch-row range** inside each
already-decoded chunk via `thread::scope` (each worker owns a contiguous output
row-slice through `split_at_mut`, shares `q8_bytes`/`input` read-only), gated to
batch>1 (prefill) so batch1 decode stays single-thread and avoids the R112 decode
regression. Compare same-turn vs the single-thread control and keep output `No`.
