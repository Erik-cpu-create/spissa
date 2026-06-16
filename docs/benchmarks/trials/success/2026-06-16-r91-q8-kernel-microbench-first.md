# R91: Q8 Kernel Microbench First

Date: 2026-06-16
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

Before another Q8 runtime hot-path change, RLLM should prove candidate dot
helpers in a cheap deterministic microbenchmark. This avoids repeating R89/R90,
where full runtime changes preserved output but regressed prefill.

## Scope

- Mode: exact-lowram lab
- REE kernel: `REEDOT-LAB`
- Model/artifact: Llama 3.2 1B-like Q8 dimensions, synthetic deterministic row
- Architecture: Q8_0-style 32-element blocks, representative prefill batch
- Target device/profile: CPU-only, low-end oriented scalar portable baseline
- Expected bottleneck: Q8 MLP arithmetic
- Bottleneck tag: CPU arithmetic / micro-kernel

## Setup

Commands:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r91-q8-microbench.json \
  --markdown target/r91-q8-microbench.md \
  --iters 2000
```

Runtime context:

- build profile: release for benchmark binary
- CPU: Apple A18 Pro
- RAM: 8589934592 bytes
- OS: Darwin 25.5.0 arm64
- relevant env/config: default R91 config, batch 55, in_features 2048, out_features context 8192, iters 2000

## Results

| variant | elapsed ns | speedup vs baseline | max abs diff | checksum |
|---|---:|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 112889458 | 1.000x | 0.00000000 | -15.000977 |
| `scaled_f32_dot32_batch4` | 47816750 | 2.361x | 0.00000000 | -15.000977 |
| `unrolled_i8_dot32_batch4` | 102209958 | 1.104x | 0.00000000 | -15.000977 |

R91 gate result:

- required variants present: yes
- every candidate `max_abs_diff <= 0.0001`: yes
- at least one candidate `>= 1.50x`: yes, `scaled_f32_dot32_batch4` at `2.361x`

## Analysis

The lab result confirms the direction that won earlier runtime work: convert a
Q8 block to scaled `f32` once, then reuse it across prompt rows. The measured
`scaled_f32_dot32_batch4` helper is `2.361x` faster than repeated signed-byte to
`f32` conversion in this isolated setup with identical checksum and zero diff.

The unrolled integer-dot variant is only `1.104x`, so simple scalar unrolling is
not a strong next runtime candidate by itself.

Important limitation: this benchmark is a lab microbenchmark, not an end-to-end
prefill measurement. It validates candidate arithmetic shape and gives RLLM a
repeatable REE kernel gate, but it does not claim R91 improved runtime prefill.

## Decision

accepted

Reason: `REEDOT-LAB` meets the R91 success gate and prevents anonymous kernel
experiments from being promoted without isolated evidence.

Paper value:

- use as process evidence for benchmark-gated original kernel development
- use as support for the accepted scaled-block Q8 arithmetic direction

## Next Experiment

R92 should not repeat the old anonymous runtime integration. It should name the
current accepted scaled-block family as the first promotable REE runtime lineage
or build a new `REEBORN-Q8` candidate on top of `scaled_f32_dot32_batch4`.

Any R92 runtime change must benchmark against the R88/R83 unchecked prefill
baseline and must keep output, peak transient memory, and exact Q8 behavior
unchanged.
