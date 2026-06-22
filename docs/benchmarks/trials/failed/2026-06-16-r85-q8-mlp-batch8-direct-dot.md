# R85: Q8 MLP Batch8 Direct Dot

## Status

Failed.

## Hypothesis

RLLM prefill is dominated by shared Q8 MLP compute. Processing eight prompt
rows per Q8_0 block directly, without materializing `[f32; 32]` for every
block, should reduce gate/up/down time while keeping exact output and low RAM.

## Baseline

- R83 best unchecked prefill: `11.45s`
- R84 measured unchecked prefill: `13.94s`
- R84 MLP total: `10703.88ms`
- R84 peak transient: `1050673152 bytes`
- R84 output: `No`

## Commands

TDD red test:

```sh
cargo test -p rllm-runtime q8_0_dot_32_batch8 -- --nocapture
```

Expected red result:

- compile failed because `accumulate_q8_0_dot_32_batch8` did not exist
- compile failed because `accumulate_q8_0_dot_32_batch8_into` did not exist

Local green tests after implementation:

```sh
cargo test -p rllm-runtime q8_0_dot_32_batch8 -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-runtime multiply_into -- --nocapture
```

Build:

```sh
cargo build --release --bin llama-test
```

Benchmark command, repeated three times:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
```

Post-revert verification:

```sh
cargo test -p rllm-runtime q8_0 -- --nocapture
```

## Results

Implementation verification:

- TDD red failed for the expected missing helper symbols.
- `cargo test -p rllm-runtime q8_0_dot_32_batch8 -- --nocapture`: `2 passed`
- `cargo test -p rllm-runtime q8_0 -- --nocapture`: `11 passed`
- `cargo test -p rllm-runtime multiply_into -- --nocapture`: `3 passed`
- `cargo build --release --bin llama-test`: succeeded

Benchmark run 1:

- output: `No`
- TTFT/prefill: `14.17s`
- decode: `1.30 tok/s`
- end-to-end: `0.13 tok/s`
- context: `55 tokens`
- internal peak transient: `1050673152 bytes`
- MLP total: `11576.30ms`
- gate/up/down: `3894.97ms / 3655.84ms / 4011.86ms`
- `/usr/bin/time -l` max RSS: `2108735488 bytes`
- `/usr/bin/time -l` real: `19.61s`

Benchmark run 2:

- output: `No`
- TTFT/prefill: `12.68s`
- decode: `0.47 tok/s`
- end-to-end: `0.14 tok/s`
- context: `55 tokens`
- internal peak transient: `1050673152 bytes`
- MLP total: `9953.61ms`
- gate/up/down: `3342.07ms / 3118.49ms / 3480.00ms`
- `/usr/bin/time -l` max RSS: `2827321344 bytes`
- `/usr/bin/time -l` real: `17.50s`

Benchmark run 3:

- output: `No`
- TTFT/prefill: `13.59s`
- decode: `1.07 tok/s`
- end-to-end: `0.14 tok/s`
- context: `55 tokens`
- internal peak transient: `1050673152 bytes`
- MLP total: `11093.59ms`
- gate/up/down: `3699.34ms / 3515.13ms / 3865.35ms`
- `/usr/bin/time -l` max RSS: `2024767488 bytes`
- `/usr/bin/time -l` real: `17.39s`

Best R85 candidate result:

- best prefill: `12.68s`
- best MLP total: `9953.61ms`
- output remained correct: `No`
- internal peak transient remained unchanged: `1050673152 bytes`

Post-revert verification:

- runtime source diff was cleared
- `cargo test -p rllm-runtime q8_0 -- --nocapture`: `9 passed`

## Decision

Rejected and runtime changes reverted. The batch8 direct-dot implementation did
not beat the strict R83 best prefill gate of `11.45s`; best observed prefill was
`12.68s`. It also did not consistently reduce MLP time versus the R84 measured
`10703.88ms`.

The negative result is useful: simply widening the portable Rust scalar batch
loop is not enough. R86 should target a different shape: either a measured
block-level repack/tile design with a bounded scratch budget or architecture
specific dot-product/SIMD support, with an explicit RAM ceiling before coding.
