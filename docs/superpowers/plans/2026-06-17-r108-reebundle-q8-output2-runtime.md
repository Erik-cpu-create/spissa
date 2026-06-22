# R108 REEBUNDLE Q8 Output2 Runtime Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Promote the R107 output-feature bundling idea into a narrowly gated runtime prototype and accept it only if it improves same-turn Llama 3.2 1B Q8 prefill without changing output or peak transient memory.

**Architecture:** R108 touches only the normal batch>1 Q8 linear accumulation path in `streaming/kernels.rs`. The candidate detects two adjacent Q8 output rows for the same 32-element input block, scales both blocks once, accumulates both outputs for the same batch4 input lanes, and falls back to the existing single-output path for all unsafe shapes.

**Tech Stack:** Rust, `rllm-runtime`, `rllm-cli` `llama-test`, aarch64 NEON guarded by existing runtime helper patterns, benchmark reports under `docs/benchmarks/trials/`.

---

## Evidence Inputs

R106 showed the runtime hot path is broader than one NEON dot helper:

- `batch_gt1_normal_batch4`: `25008.82ms` under heavy split instrumentation
- `batch_gt1_normal_batch4_kernel`: `6015.19ms`
- `batch_gt1_normal_batch4_setup`: `5241.44ms`
- residual loop/instrumentation overhead: `13752.19ms`

R107 showed output2 bundling is exact and promising in lab:

- long lab baseline `baseline_i8_dot32_output2_batch4`: `1035911084ns`
- long lab candidate `reebundle_neon_output2_batch4`: `163494375ns`
- speedup: `6.336x`
- max abs diff: `0.00000000`

R108 must not claim runtime speedup from R107 alone. The acceptance evidence must come from the real Llama 3.2 1B Q8 runtime benchmark in this plan.

## Boundary

- Runtime owner: `crates/rllm-runtime/src/streaming/kernels.rs`
- Profile owner: `crates/rllm-runtime/src/q8_profile.rs`
- Tests owner: `crates/rllm-runtime/src/streaming/tests.rs`
- Benchmark docs owner: `docs/benchmarks/trials/`

Do not change the `.spsa` container format, Q8 block format, tokenizer, prompt template, memory budget logic, Q8 multiply-into path, Q8 argmax path, or batch1 complete-row fast path.

## Files

- Modify: `crates/rllm-runtime/src/q8_profile.rs`
  - Add a profile row for `batch_gt1_normal_output2_batch4`.
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add a private output2 batch4 helper.
  - Add a private gating helper that proves two adjacent row blocks are safe to bundle.
  - Route only safe block pairs through the bundled path.
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
  - Add exactness tests for adjacent-row bundling and declined unsafe shapes.
- Create on success: `docs/benchmarks/trials/success/2026-06-17-r108-reebundle-q8-output2-runtime.md`
- Create on failure: `docs/benchmarks/trials/failed/2026-06-17-r108-reebundle-q8-output2-runtime.md`
- Modify: `docs/benchmarks/trials/index.md`

## Gates

Correctness gates:

- `cargo test -p rllm-runtime q8_0_output2 -- --nocapture`
- `cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture`
- `cargo test -p rllm-runtime q8_profile_records_sorts_and_resets_rows -- --nocapture`
- `cargo test -p rllm-runtime q8_kernel_lab -- --nocapture`
- Runtime output for the benchmark prompt remains exactly `No`.

Build gates:

- `cargo fmt --check`
- `cargo build --release -p rllm-cli --bin llama-test`
- `git diff --check`

Runtime benchmark gates:

- Use `RLLM_THREADS=1`.
- Use `--rama-integrity unchecked` to match the current trusted local benchmark convention.
- Use `--chat-template llama3`.
- Use `--profile-phases`.
- Use the same prompt as R96-R107: `Answer yes or no: is fire cold?`
- Record TTFT/prefill, decode tok/s, end-to-end elapsed, MLP/gate/up/down timing, max RSS, and RLLM peak transient.
- Run a same-turn pre-control before the code change.
- Run three no-profile candidate trials after the code change.
- Run one `RLLM_Q8_KERNEL_PROFILE=1` candidate profile after the code change.

Acceptance:

- Output remains `No`.
- RLLM peak transient memory stays at or below the same-turn control.
- Best candidate prefill is better than same-turn control by at least `5%`.
- `batch_gt1_normal_output2_batch4` appears in the profile with non-zero calls.
- Existing `batch_gt1_normal_batch4 + batch_gt1_normal_output2_batch4` does not exceed same-turn profile evidence in a way that clearly regresses total Q8 runtime.

Rejection:

- If output changes, peak transient increases, or best prefill fails the `5%` gate, revert the runtime source change and write a failed report.
- If profile instrumentation shows the new path is unused, revert the runtime source change and write a failed report.

## Task 1: Add RED Runtime Tests

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`

- [x] **Step 1: Add adjacent output2 exactness test**

Add a test that writes two adjacent Q8 rows with two 32-element blocks each, uses `batch=4`, and verifies the output equals the existing scalar expectation:

```rust
#[test]
fn q8_0_output2_runtime_path_accumulates_adjacent_rows_exactly() {
    let mut row0_block0 = [0i8; 32];
    let mut row0_block1 = [0i8; 32];
    let mut row1_block0 = [0i8; 32];
    let mut row1_block1 = [0i8; 32];
    row0_block0.fill(1);
    row0_block1.fill(2);
    row1_block0.fill(3);
    row1_block1.fill(4);

    let mut q8 = Vec::new();
    q8.extend_from_slice(&q8_0_block_bytes(0.5, &row0_block0));
    q8.extend_from_slice(&q8_0_block_bytes(0.5, &row0_block1));
    q8.extend_from_slice(&q8_0_block_bytes(0.5, &row1_block0));
    q8.extend_from_slice(&q8_0_block_bytes(0.5, &row1_block1));

    let mut input = Vec::new();
    for batch in 0..4 {
        input.extend(std::iter::repeat((batch + 1) as f32).take(32));
        input.extend(std::iter::repeat((batch + 2) as f32).take(32));
    }
    let mut output = vec![0.0f32; 8];
    let config = StreamingLinearConfig {
        batch: 4,
        in_features: 64,
        out_features: 2,
    };

    accumulate_q8_0_chunk(&input, &mut output, &q8, 0, config, "linear.q8.output2.weight")
        .unwrap();

    assert_eq!(
        output,
        vec![
            80.0, 176.0,
            128.0, 288.0,
            176.0, 400.0,
            224.0, 512.0,
        ]
    );
}
```

- [x] **Step 2: Add declined unsafe-shape test**

Add a test that starts at row0 block1, proving the helper declines non-matching `in_feature` pairs and still produces exact output through fallback:

```rust
#[test]
fn q8_0_output2_runtime_path_declines_non_matching_input_blocks() {
    let mut row0_block1 = [0i8; 32];
    let mut row1_block0 = [0i8; 32];
    row0_block1.fill(2);
    row1_block0.fill(3);

    let mut q8 = Vec::new();
    q8.extend_from_slice(&q8_0_block_bytes(0.5, &row0_block1));
    q8.extend_from_slice(&q8_0_block_bytes(0.5, &row1_block0));

    let input = vec![1.0f32; 4 * 64];
    let mut output = vec![0.0f32; 8];
    let config = StreamingLinearConfig {
        batch: 4,
        in_features: 64,
        out_features: 2,
    };

    accumulate_q8_0_chunk(&input, &mut output, &q8, 32, config, "linear.q8.decline.weight")
        .unwrap();

    assert_eq!(output, vec![32.0, 48.0, 32.0, 48.0, 32.0, 48.0, 32.0, 48.0]);
}
```

- [x] **Step 3: Verify RED**

Run:

```bash
cargo test -p rllm-runtime q8_0_output2 -- --nocapture
```

Expected: the adjacent exactness test should initially pass through fallback or fail to prove no profile row exists. If it passes through fallback, keep it and add the profile-row RED in Task 2 before implementation.

## Task 2: Add Profile Row RED

**Files:**
- Modify: `crates/rllm-runtime/src/q8_profile.rs`

- [x] **Step 1: Add expected row to test before enum support**

Inside `q8_profile_records_sorts_and_resets_rows`, add an assertion that expects the new row after recording:

```rust
assert!(snapshot
    .rows
    .iter()
    .any(|row| row.path == "batch_gt1_normal_output2_batch4"));
```

- [x] **Step 2: Verify RED**

Run:

```bash
cargo test -p rllm-runtime q8_profile_records_sorts_and_resets_rows -- --nocapture
```

Expected: FAIL because no `Q8KernelPath::BatchGt1NormalOutput2Batch4` variant exists yet.

- [x] **Step 3: Add profile enum support**

Add enum variant:

```rust
BatchGt1NormalOutput2Batch4,
```

Add string mapping:

```rust
Self::BatchGt1NormalOutput2Batch4 => "batch_gt1_normal_output2_batch4",
```

Record it in the test:

```rust
record_q8_kernel_path(
    Q8KernelPath::BatchGt1NormalOutput2Batch4,
    2,
    2,
    2,
    8,
    Duration::from_nanos(550),
);
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
cargo test -p rllm-runtime q8_profile_records_sorts_and_resets_rows -- --nocapture
```

Expected: PASS.

## Task 3: Implement Runtime Output2 Gate

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [x] **Step 1: Add safety helper**

Add a private helper near `accumulate_q8_0_chunk`:

```rust
fn q8_output2_pair_offset(
    block_idx: usize,
    q8_block_count: usize,
    element_start: usize,
    weight_elements: usize,
    config: StreamingLinearConfig,
) -> Option<(usize, usize)> {
    let block_global_start = element_start.checked_add(block_idx.checked_mul(32)?)?;
    let out_feature = block_global_start / config.in_features;
    let in_feature = block_global_start % config.in_features;
    if in_feature + 32 > config.in_features {
        return None;
    }
    if out_feature + 1 >= config.out_features {
        return None;
    }
    let next_global_start = (out_feature + 1)
        .checked_mul(config.in_features)?
        .checked_add(in_feature)?;
    if next_global_start + 32 > weight_elements {
        return None;
    }
    if next_global_start < element_start {
        return None;
    }
    let next_delta = next_global_start - element_start;
    if next_delta % 32 != 0 {
        return None;
    }
    let next_block_idx = next_delta / 32;
    if next_block_idx >= q8_block_count {
        return None;
    }
    Some((out_feature, next_block_idx))
}
```

- [x] **Step 2: Add output2 batch4 helper**

Add a helper that accumulates two scaled blocks into two output features for four batch rows:

```rust
fn accumulate_f32_dot_32_output2_batch4_reebundle(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
) {
    accumulate_f32_dot_32_batch4_reevec(
        first,
        input,
        input_stride,
        output,
        output_stride,
        first_out_feature,
    );
    accumulate_f32_dot_32_batch4_reevec(
        second,
        input,
        input_stride,
        output,
        output_stride,
        first_out_feature + 1,
    );
}
```

This is deliberately conservative: it proves safe runtime routing first. If the first runtime gate passes but profile still shows helper overhead, a later R109 can replace this wrapper with a single NEON helper mirroring the R107 lab.

- [x] **Step 3: Route safe pairs through output2 path**

Inside `accumulate_q8_0_chunk`, maintain a `skip_blocks` vector or `Vec<bool>` sized to `q8_bytes.len() / 34`. When a block is consumed as the second row of a pair, skip it on its later loop iteration.

Use this shape before the existing single-output branch:

```rust
let q8_block_count = q8_bytes.len() / 34;
let mut consumed_as_output2_second = vec![false; q8_block_count];

for block_idx in 0..q8_block_count {
    if consumed_as_output2_second[block_idx] {
        continue;
    }
    ...
    if config.batch > 1 && block_len == 32 && in_feature + block_len <= config.in_features {
        if let Some((first_out_feature, second_block_idx)) = q8_output2_pair_offset(
            block_idx,
            q8_block_count,
            element_start,
            weight_elements,
            config,
        ) {
            if second_block_idx != block_idx {
                let second_offset = second_block_idx * 34;
                let second_scale = q8_0_block_scale(q8_bytes, second_offset);
                let second_qs = &q8_bytes[second_offset + 2..second_offset + 34];
                let first_scaled = q8_0_scaled_block_reecast(qs, scale);
                let second_scaled = q8_0_scaled_block_reecast(second_qs, second_scale);
                consumed_as_output2_second[second_block_idx] = true;
                // batch4 + tail accumulation goes here
                continue;
            }
        }
        // existing single-output branch stays here
    }
}
```

- [x] **Step 4: Record output2 profile row**

Add local counters:

```rust
let mut normal_output2_batch4_elapsed = std::time::Duration::ZERO;
let mut normal_output2_batch4_calls = 0u64;
let mut normal_output2_batch4_items = 0u64;
```

Record after the loop:

```rust
record_q8_kernel_path(
    Q8KernelPath::BatchGt1NormalOutput2Batch4,
    normal_output2_batch4_calls,
    normal_output2_batch4_calls,
    normal_output2_batch4_calls * 2,
    normal_output2_batch4_items,
    normal_output2_batch4_elapsed,
);
```

- [x] **Step 5: Verify GREEN**

Run:

```bash
cargo fmt
cargo test -p rllm-runtime q8_0_output2 -- --nocapture
cargo test -p rllm-runtime streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch -- --nocapture
cargo test -p rllm-runtime q8_profile_records_sorts_and_resets_rows -- --nocapture
```

Expected: PASS.

## Task 4: Run Runtime Control and Candidate Benchmarks

**Files:**
- Create candidate benchmark outputs under `target/r108-*`

- [x] **Step 1: Build release binary**

Run:

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: PASS.

- [x] **Step 2: Run same-turn pre-control before runtime code change if not already captured**

Run this before applying Task 3 runtime routing, or reset to `HEAD` temporarily if needed:

```bash
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r108-pre-control.txt 2> target/r108-pre-control.time
```

Expected:

- visible output includes exactly `No`
- phase profile includes TTFT/prefill and MLP timings
- stderr includes max RSS from `/usr/bin/time -l`

- [x] **Step 3: Run candidate no-profile trials**

Run:

```bash
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r108-run${i}.txt" 2> "target/r108-run${i}.time"
done
```

Expected:

- all runs output exactly `No`
- record TTFT/prefill, decode tok/s, MLP total, gate/up/down, peak transient, max RSS, elapsed

- [x] **Step 4: Run candidate Q8 profile**

Run:

```bash
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > target/r108-profile.txt 2> target/r108-profile.time
```

Expected:

- output exactly `No`
- profile output includes `batch_gt1_normal_output2_batch4`
- profile output still includes enough existing Q8 rows to compare against R106/R107 context

## Task 5: Decide, Document, and Revert if Needed

**Files:**
- Create: `docs/benchmarks/trials/success/2026-06-17-r108-reebundle-q8-output2-runtime.md` or `docs/benchmarks/trials/failed/2026-06-17-r108-reebundle-q8-output2-runtime.md`
- Modify: `docs/benchmarks/trials/index.md`

- [x] **Step 1: Decide acceptance**

Accept only if all gates pass:

- output `No`
- peak transient unchanged or lower than same-turn control
- best candidate prefill at least `5%` faster than same-turn control
- `batch_gt1_normal_output2_batch4` profile row non-zero

Reject and revert runtime code if any gate fails.

- [x] **Step 2: Write benchmark report**

Use the benchmark template requirements:

- hypothesis
- REE kernel lineage: `REEBUNDLE-Q8-OUTPUT2`
- artifact/model path and shape
- exact commands
- runtime context
- result table
- profile table
- decision
- next experiment

- [x] **Step 3: Update index**

Add one row to `docs/benchmarks/trials/index.md` with the correct folder and decision.

- [x] **Step 4: Scan for unfinished markers**

Run:

```bash
rg -n "T[B]D|T[O]DO|^- \\[ \\]" docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r108-reebundle-q8-output2-runtime.md docs/benchmarks/trials/failed/2026-06-17-r108-reebundle-q8-output2-runtime.md docs/superpowers/plans/2026-06-17-r108-reebundle-q8-output2-runtime.md
```

Expected: no unfinished markers except the non-existent alternate success/failed report path.

## Task 6: Final Verification and Commit

**Files:**
- All modified R108 files

- [x] **Step 1: Run final checks**

Run:

```bash
cargo fmt --check
cargo test -p rllm-runtime q8_0_output2 -- --nocapture
cargo test -p rllm-runtime q8_profile_records_sorts_and_resets_rows -- --nocapture
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-cli --bin llama-test
git diff --check
```

- [x] **Step 2: Review final diff**

Run:

```bash
git diff --stat
git diff -- crates/rllm-runtime/src/q8_profile.rs crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r108-reebundle-q8-output2-runtime.md docs/benchmarks/trials/failed/2026-06-17-r108-reebundle-q8-output2-runtime.md docs/superpowers/plans/2026-06-17-r108-reebundle-q8-output2-runtime.md
```

- [x] **Step 3: Commit**

If accepted:

```bash
git add crates/rllm-runtime/src/q8_profile.rs crates/rllm-runtime/src/streaming/kernels.rs crates/rllm-runtime/src/streaming/tests.rs docs/benchmarks/trials/index.md docs/benchmarks/trials/success/2026-06-17-r108-reebundle-q8-output2-runtime.md docs/superpowers/plans/2026-06-17-r108-reebundle-q8-output2-runtime.md
git commit -m "bench(runtime): gate reebundle q8 output2 runtime"
```

If rejected and runtime source was reverted:

```bash
git add docs/benchmarks/trials/index.md docs/benchmarks/trials/failed/2026-06-17-r108-reebundle-q8-output2-runtime.md docs/superpowers/plans/2026-06-17-r108-reebundle-q8-output2-runtime.md
git commit -m "bench(runtime): reject reebundle q8 output2 runtime"
```
