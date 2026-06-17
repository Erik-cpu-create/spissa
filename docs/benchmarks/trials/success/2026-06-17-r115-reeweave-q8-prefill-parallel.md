# R115: REEWEAVE Q8 Prefill Row-Parallel

Date: 2026-06-17
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

R114 showed the exact Q8 prefill matmul is single-threaded (6 cores → ~1.1x) and
that decoded chunks carry hundreds of independent output rows. R115 tests whether
parallelizing one already-decoded Q8 chunk across CPU cores — by splitting the
batch (prompt token) rows — speeds up prefill while keeping output exact.

## Scope

- Mode: exact-lowram runtime
- REE kernel lineage: `REEWEAVE-Q8-PREFILL`
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Device: Apple A18 Pro (2 perf + 4 eff cores), 8 GB
- Bottleneck tag: scheduler / CPU parallelism
- Implementation: `accumulate_q8_0_chunk_parallel` wraps the existing kernel.
  For batch ≥ 8 it splits the batch rows across `RLLM_THREADS` workers via
  `std::thread::scope`; each worker owns a contiguous output row-slice
  (`split_at_mut`) and a contiguous input row-slice, shares the decoded `q8_bytes`
  read-only, and runs the unchanged single-thread kernel on its rows. Batch1
  decode falls through to sequential, so decode is unaffected.
- Coverage: the matmuls routed through `accumulate_q8_0_chunk` — attention q/k/v/o
  and MLP gate/up (via `streaming_(tile_)linear_from_model`). MLP down
  (`multiply_into`) and lm_head (`argmax`) are not yet covered.

`RLLM_THREADS` is the only knob: `=1` is the sequential control, `=6` is the
candidate. No new flag.

## Setup

```bash
cargo build --release -p rllm-cli --bin llama-test
for T in 1 6; do
  printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | RLLM_THREADS=$T target/release/llama-test --model <artifact> --chat-template llama3 --max-new-tokens 4 --rama-integrity unchecked
done
```

## Results

Output stayed `No`; peak transient unchanged at `1,050,673,152 bytes`.

| RLLM_THREADS | prefill (best of 3) | range | decode tok/s |
|---|---:|---:|---:|
| 1 (sequential control) | 6.56s | 6.56–7.62s | 1.18 |
| 6 (REEWEAVE parallel) | 4.08s | 4.08–4.20s | 1.68 |

- Best-of-3 speedup ~`1.6x`; against slower control runs (9.52s) the same parallel
  run (3.70s) was ~`2.6x`. The parallel runs were also more consistent (4.08–4.20s).
- **Bit-identical output:** T=1 vs T=6 first-token full-vocab logits `max_abs_diff
  = 0.0` (identical). This is the f32 path split by row — no quantization, no
  reordered reductions — so it is exact, not just close.
- 68 streaming unit tests pass.

## Analysis

R115 passes and is the first change to move prefill beyond single-thread kernel
noise. Splitting batch rows is clean in the streaming architecture because the
chunk is already decoded to `q8_bytes` (shared read-only) and the row-major
input/output split into contiguous, disjoint slices — no model `&mut`/budget
thread-safety problem and no output contention.

The speedup is below the core count because (a) A18 Pro is 2 perf + 4 eff cores
(eff ≈ ⅓ speed → ~3.3x effective ceiling), (b) lm_head argmax and MLP down are not
yet row-parallel, and (c) per-chunk `thread::scope` spawn overhead. All are
addressable.

## Decision

accepted

Reason: `RLLM_THREADS=6` cut prefill from `6.56s` to `4.08s` best-of-3 (~1.6x, up
to ~2.6x vs slower control runs), output stayed `No`, first-token logits are
bit-identical to the sequential path, and peak transient is unchanged.

Paper value:

- positive evidence that batch-row parallelism over already-decoded Q8 chunks is
  exact and gives a real CPU-scaling win the single-thread kernel work (R78–R112)
  could not — and it stacks with future kernel/i8mm work
- corrects R15's implication: per-chunk-internal row parallelism works where the
  earlier per-chunk-spawn raw-path attempt regressed

## Next Experiment

R116 should extend REEWEAVE to the uncovered hot kernels —
`accumulate_q8_0_chunk_multiply_into` (MLP down) and lm_head — and consider a
persistent worker pool (vs per-chunk `thread::scope` spawn) and perf-core
weighting to push past the current ceiling. The deferred i8mm kernel (R113 plan)
stacks on top once threading is maxed.
