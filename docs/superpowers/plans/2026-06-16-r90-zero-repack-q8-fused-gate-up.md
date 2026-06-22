# R90 Zero-Repack Q8 Fused Gate/Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Llama 3.2 1B Q8 prefill MLP time without repeating R89's repack/allocation overhead.

**Architecture:** Add a Q8_0 fused gate/up path modeled after the existing raw-FP16 fused gate/up path, but support batch prefill and do not repack activations. The new path streams aligned `gate_proj` and `up_proj` chunks together, accumulates `gate_acc` and `up_acc` in small per-batch state, and writes `silu(gate_acc) * up_acc` directly to the existing gate/up output buffer.

**Tech Stack:** Rust, `rllm-runtime` streaming kernels, Q8_0 chunk metadata, `llama-test`, benchmark docs under `docs/benchmarks`.

---

## Why R90 Follows R89

R89 failed because its shared bucket micro-kernel added memory passes:

- repacked `mlp_input`
- allocated/filled intermediate `up_output`
- multiplied `gate_up_output * up_output` as a separate pass
- best prefill regressed to `12.23s` against R88 baseline `10.24s`

R90 must therefore obey this rule: **no full activation repack and no extra full-size MLP intermediate buffer**.

## Files

- Delete: `crates/rllm-runtime/src/streaming/q8_shared_bucket.rs`
  - R89 residue; not referenced by the module tree and should not remain as a misleading failed kernel.
- Modify: `docs/benchmarks/trials/failed/2026-06-16-r89-q8-mlp-shared-bucket.md`
  - Change `## Status` from `Active.` to `Failed.`
- Modify: `crates/rllm-runtime/src/streaming/argmax.rs`
  - Add a batch-capable `SiluGateUpBatchState`.
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add `accumulate_q8_0_silu_gate_up_chunk`.
  - Add a paired Q8 dot helper that accumulates gate/up using the same input slice without repacking.
- Modify: `crates/rllm-runtime/src/streaming/linear.rs`
  - Route Q8_0 aligned gate/up tensors through the fused path in `streaming_silu_gate_up_from_model`.
  - Keep the existing raw-FP16/BF16 path unchanged.
  - Return `Ok(None)` for unsupported Q8 layouts so the current exact fallback remains available.
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
  - Add deterministic Q8 fused gate/up correctness and fallback tests.
- Create: `docs/benchmarks/trials/active/2026-06-16-r90-zero-repack-q8-fused-gate-up.md`
  - Record tests, commands, three-run benchmark table, and decision.
- Modify: `docs/benchmarks/trials/index.md`
  - Add the R90 row after measurement.

## Success Gate

- Output for `Answer yes or no: is fire cold?` remains `No`.
- Peak transient remains `1,050,673,152 bytes` or lower.
- Best of three unchecked R90 prefill runs is `< 10.24s`.
- Best R90 MLP total is `< 8,380ms`.
- No new full-size `Vec<f32>` allocation is introduced beyond the existing output vector and two `batch`-length accumulator vectors.
- If any gate fails, revert runtime changes, keep docs as failed evidence, and do not keep dead kernel files.

## Task 1: Clean R89 Residue

**Files:**
- Delete: `crates/rllm-runtime/src/streaming/q8_shared_bucket.rs`
- Modify: `docs/benchmarks/trials/failed/2026-06-16-r89-q8-mlp-shared-bucket.md`

- [ ] **Step 1: Delete the unused failed kernel file**

```bash
rm crates/rllm-runtime/src/streaming/q8_shared_bucket.rs
```

Expected: `rg -n "q8_shared_bucket|shared_bucket" crates/rllm-runtime/src` prints nothing.

- [ ] **Step 2: Mark the R89 report status as failed**

Change:

```markdown
## Status

Active.
```

To:

```markdown
## Status

Failed.
```

- [ ] **Step 3: Verify no failed R89 code remains wired**

Run:

```bash
rg -n "q8_shared_bucket|shared_bucket" crates/rllm-runtime/src docs/benchmarks/trials/failed/2026-06-16-r89-q8-mlp-shared-bucket.md
```

Expected: only the R89 report may mention `shared bucket`; runtime source returns no matches.

## Task 2: Add Red Tests for Q8 Fused Gate/Up

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`

- [ ] **Step 1: Add a deterministic Q8 fused gate/up correctness test**

Add this test near the existing Q8 tests:

```rust
#[test]
fn q8_0_silu_gate_up_batch4_matches_materialized_projection() {
    let config = StreamingLinearConfig {
        batch: 4,
        in_features: 64,
        out_features: 3,
    };
    let input: Vec<f32> = (0..config.batch * config.in_features)
        .map(|idx| (idx as f32 % 17.0) * 0.03125 - 0.25)
        .collect();

    let mut gate_bytes = Vec::new();
    let mut up_bytes = Vec::new();
    for row in 0..config.out_features {
        for block in 0..2 {
            let mut gate_q = [0i8; 32];
            let mut up_q = [0i8; 32];
            for idx in 0..32 {
                gate_q[idx] = ((row * 5 + block * 3 + idx) as i8 % 15) - 7;
                up_q[idx] = ((row * 7 + block * 2 + idx) as i8 % 13) - 6;
            }
            gate_bytes.extend(q8_0_block_bytes(0.125, &gate_q));
            up_bytes.extend(q8_0_block_bytes(0.25, &up_q));
        }
    }

    let mut gate = vec![0.0f32; config.batch * config.out_features];
    let mut up = vec![0.0f32; config.batch * config.out_features];
    accumulate_q8_0_chunk(&input, &mut gate, &gate_bytes, 0, config, "gate").unwrap();
    accumulate_q8_0_chunk(&input, &mut up, &up_bytes, 0, config, "up").unwrap();
    for (gate_value, up_value) in gate.iter_mut().zip(up.iter()) {
        *gate_value = crate::ops::silu(*gate_value) * *up_value;
    }

    let mut actual = vec![0.0f32; config.batch * config.out_features];
    let mut state = SiluGateUpBatchState::new(&mut actual, config.batch);
    accumulate_q8_0_silu_gate_up_chunk(
        &input,
        &gate_bytes,
        &up_bytes,
        0,
        config,
        &mut state,
        "gate/up.q8",
    )
    .unwrap();
    state.finish(config, "gate/up.q8").unwrap();

    for (idx, (actual_value, expected_value)) in actual.iter().zip(gate.iter()).enumerate() {
        assert!(
            (actual_value - expected_value).abs() <= 1.0e-4,
            "idx={idx} actual={actual_value} expected={expected_value}"
        );
    }
}
```

- [ ] **Step 2: Add a fallback-shape test**

```rust
#[test]
fn q8_0_silu_gate_up_rejects_mismatched_chunk_lengths() {
    let config = StreamingLinearConfig {
        batch: 2,
        in_features: 32,
        out_features: 1,
    };
    let input = vec![0.25f32; config.batch * config.in_features];
    let q = [1i8; 32];
    let gate_bytes = q8_0_block_bytes(0.5, &q);
    let mut up_bytes = q8_0_block_bytes(0.5, &q);
    up_bytes.push(0);
    let mut output = vec![0.0f32; config.batch * config.out_features];
    let mut state = SiluGateUpBatchState::new(&mut output, config.batch);

    let err = accumulate_q8_0_silu_gate_up_chunk(
        &input,
        &gate_bytes,
        &up_bytes,
        0,
        config,
        &mut state,
        "gate/up.bad",
    )
    .unwrap_err();

    assert!(
        err.to_string().contains("Q8_0 gate/up chunk len mismatch"),
        "unexpected error: {err}"
    );
}
```

- [ ] **Step 3: Run the red tests**

Run:

```bash
cargo test -p rllm-runtime q8_0_silu_gate_up -- --nocapture
```

Expected: compile fails because `SiluGateUpBatchState` and `accumulate_q8_0_silu_gate_up_chunk` do not exist yet.

## Task 3: Add Batch Gate/Up State

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/argmax.rs`

- [ ] **Step 1: Add the batch state after `SiluGateUpState`**

```rust
struct SiluGateUpBatchState<'a> {
    output: &'a mut [f32],
    current_out_feature: usize,
    gate_acc: Vec<f32>,
    up_acc: Vec<f32>,
}

impl<'a> SiluGateUpBatchState<'a> {
    fn new(output: &'a mut [f32], batch: usize) -> Self {
        Self {
            output,
            current_out_feature: 0,
            gate_acc: vec![0.0; batch],
            up_acc: vec![0.0; batch],
        }
    }

    fn finish_current(&mut self, config: StreamingLinearConfig, weight_name: &str) -> Result<()> {
        if self.current_out_feature >= config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed more rows than expected {}",
                config.out_features
            )));
        }
        for batch_idx in 0..config.batch {
            self.output[batch_idx * config.out_features + self.current_out_feature] =
                crate::ops::silu(self.gate_acc[batch_idx]) * self.up_acc[batch_idx];
        }
        self.current_out_feature += 1;
        self.gate_acc.fill(0.0);
        self.up_acc.fill(0.0);
        Ok(())
    }

    fn finish(self, config: StreamingLinearConfig, weight_name: &str) -> Result<()> {
        if self.current_out_feature != config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed {} complete rows, expected {}",
                self.current_out_feature, config.out_features
            )));
        }
        Ok(())
    }
}
```

- [ ] **Step 2: Run the red tests again**

Run:

```bash
cargo test -p rllm-runtime q8_0_silu_gate_up -- --nocapture
```

Expected: compile still fails because `accumulate_q8_0_silu_gate_up_chunk` does not exist yet.

## Task 4: Implement Zero-Repack Q8 Fused Kernel

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`

- [ ] **Step 1: Add paired Q8 dot helper near `q8_0_dot_i8_f32`**

```rust
fn q8_0_dot_pair_i8_f32(
    gate_qs: &[u8],
    gate_scale: f32,
    up_qs: &[u8],
    up_scale: f32,
    input: &[f32],
    len: usize,
) -> (f32, f32) {
    let mut gate_acc = 0.0f32;
    let mut up_acc = 0.0f32;
    let mut idx = 0usize;
    while idx + 4 <= len {
        let x0 = input[idx];
        let x1 = input[idx + 1];
        let x2 = input[idx + 2];
        let x3 = input[idx + 3];
        gate_acc += gate_scale
            * ((gate_qs[idx] as i8) as f32 * x0
                + (gate_qs[idx + 1] as i8) as f32 * x1
                + (gate_qs[idx + 2] as i8) as f32 * x2
                + (gate_qs[idx + 3] as i8) as f32 * x3);
        up_acc += up_scale
            * ((up_qs[idx] as i8) as f32 * x0
                + (up_qs[idx + 1] as i8) as f32 * x1
                + (up_qs[idx + 2] as i8) as f32 * x2
                + (up_qs[idx + 3] as i8) as f32 * x3);
        idx += 4;
    }
    while idx < len {
        let x = input[idx];
        gate_acc += gate_scale * (gate_qs[idx] as i8) as f32 * x;
        up_acc += up_scale * (up_qs[idx] as i8) as f32 * x;
        idx += 1;
    }
    (gate_acc, up_acc)
}
```

- [ ] **Step 2: Add fused Q8 chunk accumulator**

```rust
fn accumulate_q8_0_silu_gate_up_chunk(
    input: &[f32],
    gate_q8_bytes: &[u8],
    up_q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut SiluGateUpBatchState<'_>,
    weight_name: &str,
) -> Result<()> {
    if gate_q8_bytes.len() != up_q8_bytes.len() {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Q8_0 gate/up chunk len mismatch for {weight_name}: gate={}, up={}",
            gate_q8_bytes.len(),
            up_q8_bytes.len()
        )));
    }
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("Q8_0 gate/up weight element count overflow".to_string()))?;
    validate_q8_0_chunk(gate_q8_bytes, element_start, weight_elements, weight_name)?;
    validate_q8_0_chunk(up_q8_bytes, element_start, weight_elements, weight_name)?;

    for block_idx in 0..gate_q8_bytes.len() / 34 {
        let block_global_start = element_start + block_idx * 32;
        if block_global_start >= weight_elements {
            break;
        }
        let block_len = (weight_elements - block_global_start).min(32);
        let out_feature = block_global_start / config.in_features;
        let in_feature = block_global_start % config.in_features;
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let gate_block_offset = block_idx * 34;
        let up_block_offset = block_idx * 34;
        let gate_scale = q8_0_block_scale(gate_q8_bytes, gate_block_offset);
        let up_scale = q8_0_block_scale(up_q8_bytes, up_block_offset);
        let gate_qs = &gate_q8_bytes[gate_block_offset + 2..gate_block_offset + 34];
        let up_qs = &up_q8_bytes[up_block_offset + 2..up_block_offset + 34];

        if in_feature + block_len <= config.in_features {
            for batch_idx in 0..config.batch {
                let input_start = batch_idx * config.in_features + in_feature;
                let (gate_delta, up_delta) = q8_0_dot_pair_i8_f32(
                    gate_qs,
                    gate_scale,
                    up_qs,
                    up_scale,
                    &input[input_start..],
                    block_len,
                );
                state.gate_acc[batch_idx] += gate_delta;
                state.up_acc[batch_idx] += up_delta;
            }
            if in_feature + block_len == config.in_features {
                state.finish_current(config, weight_name)?;
            }
        } else {
            for idx in 0..block_len {
                let global_idx = block_global_start + idx;
                let out_feature = global_idx / config.in_features;
                let in_feature = global_idx % config.in_features;
                while state.current_out_feature < out_feature {
                    state.finish_current(config, weight_name)?;
                }
                let gate_weight = gate_scale * (gate_qs[idx] as i8) as f32;
                let up_weight = up_scale * (up_qs[idx] as i8) as f32;
                for batch_idx in 0..config.batch {
                    let x = input[batch_idx * config.in_features + in_feature];
                    state.gate_acc[batch_idx] += x * gate_weight;
                    state.up_acc[batch_idx] += x * up_weight;
                }
                if in_feature + 1 == config.in_features {
                    state.finish_current(config, weight_name)?;
                }
            }
        }
    }
    Ok(())
}
```

- [ ] **Step 3: Run Q8 fused tests**

Run:

```bash
cargo test -p rllm-runtime q8_0_silu_gate_up -- --nocapture
```

Expected: both new tests pass.

## Task 5: Wire Q8 Fused Gate/Up Into Streaming Linear

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/linear.rs`

- [ ] **Step 1: Extend `streaming_silu_gate_up_from_model` dtype gate**

Change the dtype gate so Q8_0 is accepted in addition to raw FP16/BF16:

```rust
if gate_tensor.dtype != up_tensor.dtype
    || !matches!(
        raw_dtype,
        rllm_container::DType::Fp16 | rllm_container::DType::Bf16 | rllm_container::DType::Q8_0
    )
{
    return Ok(None);
}
```

- [ ] **Step 2: Keep the existing raw path unchanged**

Keep the current `rtc-raw-v1` checks and `accumulate_silu_gate_up_raw_16bit_chunk_batch1` call under:

```rust
if matches!(raw_dtype, rllm_container::DType::Fp16 | rllm_container::DType::Bf16) {
    if config.linear.batch != 1 {
        return Ok(None);
    }
    /* existing raw implementation */
}
```

- [ ] **Step 3: Add Q8_0 paired chunk path**

Add a Q8 branch before falling back:

```rust
if raw_dtype == rllm_container::DType::Q8_0 {
    if gate_chunks.is_empty() || gate_chunks.len() != up_chunks.len() {
        return Ok(None);
    }
    for (gate_chunk, up_chunk) in gate_chunks.iter().zip(up_chunks.iter()) {
        if gate_chunk.codec_id != up_chunk.codec_id
            || gate_chunk.chunk_offset_in_tensor != up_chunk.chunk_offset_in_tensor
            || gate_chunk.uncompressed_size != up_chunk.uncompressed_size
        {
            return Ok(None);
        }
    }

    let mut output = vec![0.0f32; config.linear.batch * config.linear.out_features];
    let mut state = SiluGateUpBatchState::new(&mut output, config.linear.batch);
    let mut byte_offset = 0usize;
    for (gate_chunk, up_chunk) in gate_chunks.iter().zip(up_chunks.iter()) {
        let element_start =
            quantized_elements_for_bytes(rllm_container::DType::Q8_0, byte_offset)?;
        let expected_chunk_bytes = usize::try_from(gate_chunk.uncompressed_size).map_err(|_| {
            RuntimeError::InvalidTensorData(format!(
                "chunk {} uncompressed size does not fit usize",
                gate_chunk.chunk_id
            ))
        })?;
        model.with_two_raw_chunks(gate_chunk.chunk_id, up_chunk.chunk_id, budget, |gate_bytes, up_bytes, _budget| {
            if gate_bytes.len() != expected_chunk_bytes || up_bytes.len() != expected_chunk_bytes {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "Q8_0 gate/up chunk byte len mismatch for {gate_weight_name}/{up_weight_name}"
                )));
            }
            accumulate_q8_0_silu_gate_up_chunk(
                input,
                gate_bytes,
                up_bytes,
                element_start,
                config.linear,
                &mut state,
                gate_weight_name,
            )
        })?;
        byte_offset = byte_offset.checked_add(expected_chunk_bytes).ok_or_else(|| {
            RuntimeError::InvalidTensorData("Q8_0 gate/up chunk byte offset overflow".to_string())
        })?;
    }
    state.finish(config.linear, gate_weight_name)?;
    return Ok(Some(output));
}
```

- [ ] **Step 4: Run focused streaming tests**

Run:

```bash
cargo test -p rllm-runtime streaming_silu_gate_up q8_0_silu_gate_up -- --nocapture
```

Expected: raw fused tests still pass and Q8 fused tests pass.

## Task 6: Full Regression Check

**Files:**
- No additional files.

- [ ] **Step 1: Run runtime Q8 tests**

```bash
cargo test -p rllm-runtime q8_0 -- --nocapture
```

Expected: all Q8 tests pass.

- [ ] **Step 2: Run llama-test CLI tests**

```bash
cargo test -p rllm-cli --bin llama-test
```

Expected: all llama-test tests pass.

- [ ] **Step 3: Build release benchmark binary**

```bash
cargo build --release -p rllm-cli --bin llama-test
```

Expected: release binary builds successfully.

## Task 7: Benchmark and Decide

**Files:**
- Create: `docs/benchmarks/trials/active/2026-06-16-r90-zero-repack-q8-fused-gate-up.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Create active trial report**

```markdown
# R90: Zero-Repack Q8 Fused Gate/Up

## Status

Active.

## Hypothesis

Fusing Q8_0 gate/up accumulation without repacking activations should reduce MLP prefill time by removing one projection pass boundary and one activation memory pass while preserving exact-lowram output.

## Baseline

- R88 best unchecked prefill: `10.24s`
- R88 best MLP total: `8380ms`
- R89 failed best prefill: `12.23s`
- Required output: `No`
- Required peak transient ceiling: `1,050,673,152 bytes`

## Commands

```bash
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do
  /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa \
    --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" \
    > /tmp/r90-run${i}.txt 2> /tmp/r90-run${i}.time
done
```

## Results

After the three benchmark runs complete, add one measured table with these columns:

```markdown
| run | output | context tokens | prefill | decode | MLP total | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
```

Fill exactly three rows from `/tmp/r90-run1.txt`, `/tmp/r90-run2.txt`, `/tmp/r90-run3.txt`, and their matching `.time` files. Do not leave empty cells.

## Decision

Set this to `accepted` only if the success gate passes. Otherwise set it to `failed` and move the report to `docs/benchmarks/trials/failed/`.
```

- [ ] **Step 2: Run three benchmark runs**

Run the commands from the report exactly.

Expected:

- each output starts with `No`
- peak transient does not exceed `1,050,673,152 bytes`

- [ ] **Step 3: Apply decision rule**

If best prefill is `< 10.24s` and best MLP total is `< 8380ms`, move the report to `docs/benchmarks/trials/success/`.

If the gate fails, revert runtime changes and move the report to `docs/benchmarks/trials/failed/`.

- [ ] **Step 4: Update index**

Add this row, replacing the result/status fields with measured data:

```markdown
| 2026-06-16 | 2026-06-16-r90-zero-repack-q8-fused-gate-up.md | active | Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa | exact-lowram | CPU arithmetic / Q8 fused gate-up | R88 best prefill 10.24s, MLP 8380ms, output `No`; R89 failed best 12.23s | pending R90 measurement | active | validates whether zero-repack Q8 gate/up fusion can beat repack-based failed kernels |
```

## Self-Review

- Spec coverage: R90 directly addresses R89's measured failure by forbidding repack and full-size intermediate buffers.
- Placeholder scan: The only empty table cells are benchmark result slots to be filled after running the exact commands.
- Type consistency: New function names are `SiluGateUpBatchState`, `q8_0_dot_pair_i8_f32`, and `accumulate_q8_0_silu_gate_up_chunk`; later tasks use the same names.
- Risk: This may still fail because Q8 gate/up fusion does not reduce the dominant multiply count. If it fails, keep the failed report and do not keep runtime code.
