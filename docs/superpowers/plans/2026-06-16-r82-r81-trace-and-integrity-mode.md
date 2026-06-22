# R82 R81 Trace and Integrity Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-profile the accepted R81 runtime and, only if evidence supports it, add an explicit benchmarkable integrity mode for trusted local artifacts.

**Architecture:** First run the existing `llama-test --rama-trace` path to see whether checksum time has become a first-order cost after R80/R81. If checksum is material, add an explicit `RamaIntegrityMode::Unchecked` and `llama-test --rama-integrity unchecked`, leaving the current default `verify-once` unchanged.

**Tech Stack:** Rust, RLLM lazy chunk loader, `llama-test`, existing benchmark docs.

---

## Files

- Modify `crates/rllm-runtime/src/lazy.rs`
  - Add `RamaIntegrityMode::Unchecked`.
  - Make checksum verification helpers return `Ok(false)` immediately in unchecked mode.
  - Keep default `Strict` and current `llama-test` default `VerifyOnce` behavior unchanged.

- Modify `crates/rllm-cli/src/bin/llama-test.rs`
  - Add `--rama-integrity <strict|verify-once|unchecked>`.
  - Default remains `verify-once`.
  - Parse invalid values as errors.

- Add `docs/benchmarks/trials/active/2026-06-16-r82-r81-trace-and-integrity-mode.md`
  - Record the R81 trace attribution and any unchecked benchmark result.

- Modify `docs/benchmarks/trials/index.md`
  - Add R82 row after measurement.

## Task 1: Trace R81 Before Code Changes

- [ ] Build current release:

```sh
cargo build --release --bin llama-test
```

- [ ] Run traced R81:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-trace target/r82-r81-trace.json"
```

- [ ] Summarize trace:

```sh
jq '.summary' target/r82-r81-trace.json
```

Expected quality: output remains `No`.

## Task 2: Red Tests for Explicit Integrity Mode

Only perform this task if Task 1 shows checksum verification is a material cost.

**Files:**
- Modify `crates/rllm-runtime/src/lazy.rs`
- Modify `crates/rllm-cli/src/bin/llama-test.rs`

- [ ] Add runtime test in `crates/rllm-runtime/src/lazy.rs`:

```rust
#[test]
fn unchecked_integrity_records_no_checksum_events() {
    let path = temp_model_path("unchecked-integrity");
    write_test_model(&path);
    let mut model = LazyRllmModel::open(&path).unwrap();
    model.set_rama_integrity_mode(RamaIntegrityMode::Unchecked);
    model.enable_rama_trace();
    let mut budget = MemoryBudget::unbounded();

    model
        .with_decoded_chunk(0, &mut budget, |bytes, _budget| Ok(bytes.len()))
        .unwrap();

    let trace = model.take_rama_trace().expect("trace should be enabled");
    assert!(trace
        .events
        .iter()
        .all(|event| event.phase != "chunk_compressed_checksum"
            && event.phase != "chunk_original_checksum"));
    std::fs::remove_file(path).ok();
}
```

- [ ] Add CLI parse test in `crates/rllm-cli/src/bin/llama-test.rs`:

```rust
#[test]
fn args_default_to_verify_once_integrity_and_accept_unchecked() {
    let default_args = Args::parse_from(["llama-test", "--model", "model.spsa"]);
    assert_eq!(default_args.rama_integrity, "verify-once");

    let unchecked_args = Args::parse_from([
        "llama-test",
        "--model",
        "model.spsa",
        "--rama-integrity",
        "unchecked",
    ]);
    assert_eq!(unchecked_args.rama_integrity, "unchecked");
}
```

- [ ] Run:

```sh
cargo test -p rllm-runtime unchecked_integrity_records_no_checksum_events
cargo test -p rllm-cli --bin llama-test args_default_to_verify_once_integrity_and_accept_unchecked
```

Expected: fail because `Unchecked` and `rama_integrity` do not exist yet.

## Task 3: Green Integrity Implementation

**Files:**
- Modify `crates/rllm-runtime/src/lazy.rs`
- Modify `crates/rllm-cli/src/bin/llama-test.rs`

- [ ] Add enum variant:

```rust
/// Trust local artifact bytes without runtime checksum verification.
Unchecked,
```

- [ ] At the top of each verification helper, add:

```rust
if self.integrity_mode == RamaIntegrityMode::Unchecked {
    return Ok(false);
}
```

- [ ] Add `parse_rama_integrity_mode` in `llama-test.rs` matching:

```rust
fn parse_rama_integrity_mode(raw: &str) -> Result<RamaIntegrityMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "strict" => Ok(RamaIntegrityMode::Strict),
        "verify-once" | "verify_once" | "once" => Ok(RamaIntegrityMode::VerifyOnce),
        "unchecked" | "none" | "trusted" => Ok(RamaIntegrityMode::Unchecked),
        other => anyhow::bail!(
            "unsupported --rama-integrity {other:?}; expected strict, verify-once, or unchecked"
        ),
    }
}
```

- [ ] Replace hardcoded `VerifyOnce` assignment with parsed mode.

## Task 4: Verification and Benchmark

Run:

```sh
cargo test -p rllm-runtime unchecked_integrity_records_no_checksum_events
cargo test -p rllm-cli --bin llama-test
cargo build --release --bin llama-test
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
```

Expected quality: output remains `No`.

Compare against:

- R78 baseline prefill: 26.75 s
- R80 best prefill: 22.06 s
- R81 best prefill: 21.41 s

## Task 5: Decision

- If unchecked materially improves prefill and output remains correct, record R82 as success with the explicit trust tradeoff.
- If checksum is not material or unchecked does not improve non-trace prefill, do not keep runtime changes; record diagnostic failure or inconclusive evidence.

## Self-Review

- Spec coverage: R82 measures first, then only changes integrity behavior if trace evidence supports it.
- Placeholder scan: No placeholder instructions remain.
- Type consistency: Function names and enum variants match existing RLLM code style.
