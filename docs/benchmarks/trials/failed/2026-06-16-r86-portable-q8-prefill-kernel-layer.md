# R86: Portable Q8 Prefill Kernel Layer

## Status

Failed.

## Hypothesis

RLLM prefill is dominated by shared Q8 MLP compute. A portable Q8 kernel layer
lets RLLM keep scalar correctness on all CPUs while enabling CPU-specific
optimized dot paths for the same call-sites. The first optimized backend should
reduce gate/up/down time without changing the model format or RAM invariant.

## Baseline

- R83 best unchecked prefill: `11.45s`
- R84 measured unchecked prefill: `13.94s`
- R84 MLP total: `10703.88ms`
- R85 best unchecked prefill: `12.68s` but rejected
- R84/R85 peak transient: `1050673152 bytes`
- Baseline output: `No`

## Commands

TDD red test:

```sh
cargo test -p rllm-runtime q8_kernel -- --nocapture
```

Expected red result:

- missing `q8_kernel::dot32`
- missing `q8_kernel::accumulate_dot32_batch4`
- missing `q8_kernel::accumulate_dot32_batch4_into`

Targeted tests after scalar interface, call-site wiring, and aarch64 NEON backend:

```sh
cargo test -p rllm-runtime q8_kernel -- --nocapture
cargo test -p rllm-runtime q8_0 -- --nocapture
cargo test -p rllm-runtime multiply_into -- --nocapture
```

Build:

```sh
cargo build --release --bin llama-test
```

Benchmark command, repeated three times:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
```

Post-revert verification:

```sh
cargo test -p rllm-runtime q8_0 -- --nocapture
```

## Results

Implementation verification:

- TDD red failed for the expected missing kernel wrapper symbols.
- `cargo test -p rllm-runtime q8_kernel -- --nocapture`: `3 passed`
- `cargo test -p rllm-runtime q8_0 -- --nocapture`: `9 passed`
- `cargo test -p rllm-runtime multiply_into -- --nocapture`: `3 passed`
- `cargo build --release --bin llama-test`: succeeded after removing release warnings from test-only helpers.

Benchmark run 1:

- output: `No`
- TTFT/prefill: `13.45s`
- decode: `0.84 tok/s`
- end-to-end: `0.14 tok/s`
- context: `55 tokens`
- internal peak transient: `1050673152 bytes`
- MLP total: `10776.28ms`
- gate/up/down: `4969.74ms / 1931.28ms / 3862.34ms`
- `/usr/bin/time -l` max RSS: `1842544640 bytes`
- `/usr/bin/time -l` real: `18.60s`

Benchmark run 2:

- output: `No`
- TTFT/prefill: `12.53s`
- decode: `1.34 tok/s`
- end-to-end: `0.15 tok/s`
- context: `55 tokens`
- internal peak transient: `1050673152 bytes`
- MLP total: `9556.78ms`
- gate/up/down: `4503.48ms / 1554.83ms / 3487.61ms`
- `/usr/bin/time -l` max RSS: `1647673344 bytes`
- `/usr/bin/time -l` real: `15.99s`

Benchmark run 3:

- output: `No`
- TTFT/prefill: `12.17s`
- decode: `1.31 tok/s`
- end-to-end: `0.15 tok/s`
- context: `55 tokens`
- internal peak transient: `1050673152 bytes`
- MLP total: `9446.64ms`
- gate/up/down: `4487.93ms / 1496.88ms / 3449.99ms`
- `/usr/bin/time -l` max RSS: `1647755264 bytes`
- `/usr/bin/time -l` real: `15.97s`

Best R86 candidate result:

- best prefill: `12.17s`
- best MLP total: `9446.64ms`
- output remained correct: `No`
- internal peak transient remained unchanged: `1050673152 bytes`

Post-revert verification:

- runtime source diff was cleared
- `cargo test -p rllm-runtime q8_0 -- --nocapture`: `9 passed`

## Decision

Rejected and runtime changes reverted. The portable kernel boundary was the
right architectural direction, but this first aarch64 NEON FP32 dot backend did
not beat the strict R83 best prefill gate of `11.45s`; best observed prefill was
`12.17s`.

The result is still useful. It shows that per-block FP32 NEON dot alone is not
enough. R87 should avoid another small dot-wrapper experiment and instead target
bounded tile/repack for MLP with an explicit scratch/RAM ceiling, or compare a
more llama.cpp-like row/tile packing design before coding.
