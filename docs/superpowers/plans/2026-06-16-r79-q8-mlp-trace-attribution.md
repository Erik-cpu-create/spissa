# R79 Plan: Q8 MLP Trace Attribution

## Goal

Find the dominant cause of the slow Llama 3.2 1B Q8 exact-lowram prefill path before attempting another optimization.

R78 proved that the naive Q8 batch-row fast path did not improve prefill. The next step is diagnostic: produce a trace from the same `llama-test` benchmark harness and attribute Q8 MLP time to chunk read, checksum, decode, and compute events grouped by projection.

## Constraints

- Keep the runtime exact: no sparse activation, no approximate math, no quality tradeoff.
- Keep the benchmark command aligned with `docs/benchmarks/README.md`.
- Use the existing `RamaTrace` runtime event model instead of adding a second profiler.
- Scope changes to CLI trace exposure and benchmark evidence. Do not optimize the math kernel until the trace identifies the bottleneck.

## Files

- Modify `crates/rllm-cli/src/bin/llama-test.rs`
  - Add `--rama-trace <PATH>`.
  - Enable `LazyRllmModel::enable_rama_trace()` before preparing the Llama adapter when the path is provided.
  - After the interactive loop exits, call `take_rama_trace()` and write JSON to the requested path.
  - Keep existing stdout metrics unchanged so old benchmark parsing remains valid.

- Modify or add a small shared trace output helper in `crates/rllm-cli/src`
  - Prefer extracting the existing JSON writer shape from `crates/rllm-cli/src/commands/run.rs` if module structure allows it cleanly.
  - The JSON must include:
    - raw trace events
    - total recorded event count
    - totals by `phase`
    - totals by coarse tensor bucket for `chunk_compute_closure` events
  - Tensor buckets:
    - `mlp.gate_proj`
    - `mlp.up_proj`
    - `mlp.down_proj`
    - `attention.q_proj`
    - `attention.k_proj`
    - `attention.v_proj`
    - `attention.o_proj`
    - `lm_head`
    - `other`

- Modify `docs/benchmarks/trials/active/2026-06-16-r79-q8-mlp-trace-attribution.md`
  - Record command, model path, output, timing, memory, and trace summary.

- Modify `docs/benchmarks/trials/index.md`
  - Add the R79 row after measurement.

## Tests

1. Add/extend `llama-test` argument parsing tests:
   - default `rama_trace` is `None`
   - `--rama-trace /tmp/trace.json` parses the path

2. Add a unit test for trace summary generation if the helper is factored out:
   - synthetic events for `chunk_read`, `chunk_decode`, and `chunk_compute`
   - assert phase totals and MLP bucket totals are present

3. Run:

```sh
cargo test -p rllm-cli llama_test --bins
```

If the package/test filter does not match this repo layout, run the narrowest available equivalent and record the exact command in the benchmark report.

## Benchmark Command

Run the same one-turn Llama 3.2 1B sanity prompt used in R78, with trace enabled:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-trace target/r79-q8-mlp-trace.json"
```

Expected quality check: answer should remain `No`.

Expected metric capture:

- `TTFT/Prefill`
- `Decode tok/s`
- `E2E tok/s`
- `Total tokens`
- `Context tokens`
- `Peak transient bytes`
- `/usr/bin/time -l` peak RSS
- `PrefillProfile` subphase totals
- trace `duration_by_phase`
- trace `duration_by_tensor_bucket`

## Decision Rule

R79 is a successful diagnostic if it identifies which category dominates the measured Q8 prefill path:

- `chunk_compute_closure` dominates: optimize arithmetic/layout/kernel next.
- `chunk_decode` dominates: optimize Q8 unpack/dequant path next.
- `chunk_read` dominates: optimize chunk locality/prefetch/cache next.
- checksum phases dominate: revisit integrity mode/verification lifecycle.

R79 is inconclusive only if trace overhead materially changes the prefill profile or the trace lacks enough tensor names to attribute MLP projection cost.

## Implementation Steps

1. Add the active benchmark report skeleton.
2. Add `--rama-trace` to `llama-test` args and tests.
3. Implement or extract trace JSON writer with phase and tensor-bucket summaries.
4. Run focused tests.
5. Build release `llama-test` if needed.
6. Run the benchmark command above.
7. Fill the benchmark report with actual numbers and trace attribution.
8. Update `docs/benchmarks/trials/index.md`.
9. Commit the plan, code, and benchmark evidence together if all verification is complete.
