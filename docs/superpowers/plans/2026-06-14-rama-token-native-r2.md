# RAMA Token-Native R2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mini R2 benchmark command that compares full-replay LLaMA generation against persistent `RamaChatSession` generation using the exact same token IDs.

**Architecture:** Add a new `chat-session-token` CLI command in `rllm-cli`. It opens two model handles: one for the full-replay baseline and one for the persistent session, then validates per-turn generated tokens and visible histories before writing an active benchmark report.

**Tech Stack:** Rust 2021, Clap, existing LLaMA RAMA one-shot generation path, existing `RamaChatSession`, Markdown benchmark reports.

---

## File Structure

- Create `crates/rllm-cli/src/commands/chat_session_token.rs`
  - Own token-native parsing, full-replay/session benchmark orchestration, token match validation, and R2 Markdown report generation.
- Modify `crates/rllm-cli/src/commands/mod.rs`
  - Register the new command module.
- Modify `crates/rllm-cli/src/main.rs`
  - Add the `chat-session-token` subcommand and route it.
- Create `docs/benchmarks/trials/active/2026-06-14-r2-token-native-smollm2.md`
  - Store the first mini R2 trial output when the local benchmark runs.
- Modify `docs/benchmarks/trials/index.md`
  - Add the R2 trial row after measurement.

## Task 1: Token Parsing And Report Path Guard

**Files:**
- Create: `crates/rllm-cli/src/commands/chat_session_token.rs`

- [ ] **Step 1: Write failing tests for token parsing and output guard**

Add these tests in the new file:

```rust
#[cfg(test)]
mod tests {
    use super::{parse_token_turns, validate_report_output_path};

    #[test]
    fn parse_token_turns_accepts_comma_separated_turns() {
        let turns = vec!["1, 2,3".to_string(), "4".to_string()];

        assert_eq!(parse_token_turns(&turns).unwrap(), vec![vec![1, 2, 3], vec![4]]);
    }

    #[test]
    fn parse_token_turns_rejects_empty_and_invalid_values() {
        assert!(parse_token_turns(&[]).unwrap_err().to_string().contains("at least one"));
        assert!(parse_token_turns(&["".to_string()]).unwrap_err().to_string().contains("empty"));
        assert!(parse_token_turns(&["1,,2".to_string()]).unwrap_err().to_string().contains("empty token id"));
        assert!(parse_token_turns(&["1,nope".to_string()]).unwrap_err().to_string().contains("invalid token id"));
    }

    #[test]
    fn token_report_output_rejects_reviewed_trial_folders() {
        validate_report_output_path("docs/benchmarks/trials/active/r2.md").unwrap();

        for path in [
            "docs/benchmarks/trials/success/r2.md",
            "docs/benchmarks/trials/active/../failed/r2.md",
            "docs/benchmarks/trials/Inconclusive/r2.md",
        ] {
            assert!(validate_report_output_path(path)
                .unwrap_err()
                .to_string()
                .contains("active reports"));
        }
    }
}
```

- [ ] **Step 2: Run the focused failing tests**

```bash
cargo test -p rllm-cli chat_session_token -- --nocapture
```

Expected: compile fails because the file/module and helpers are not wired yet.

- [ ] **Step 3: Implement the parsing and path helpers**

Implement:

```rust
use anyhow::{Context, Result};

fn parse_token_turns(turns: &[String]) -> Result<Vec<Vec<usize>>> {
    if turns.is_empty() {
        anyhow::bail!("chat-session-token requires at least one --turn-ids");
    }
    turns
        .iter()
        .enumerate()
        .map(|(turn_idx, raw)| parse_token_turn(turn_idx + 1, raw))
        .collect()
}

fn parse_token_turn(turn_idx: usize, raw: &str) -> Result<Vec<usize>> {
    if raw.trim().is_empty() {
        anyhow::bail!("chat-session-token turn {turn_idx} must not be empty");
    }
    let mut ids = Vec::new();
    for (item_idx, part) in raw.split(',').enumerate() {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            anyhow::bail!(
                "chat-session-token turn {turn_idx} has empty token id at position {}",
                item_idx + 1
            );
        }
        ids.push(trimmed.parse::<usize>().with_context(|| {
            format!(
                "invalid token id in chat-session-token turn {turn_idx} at position {}",
                item_idx + 1
            )
        })?);
    }
    Ok(ids)
}

fn validate_report_output_path(out: &str) -> Result<()> {
    let components = normalized_path_components(out);
    for window in components.windows(4) {
        if window[0] == "docs"
            && window[1] == "benchmarks"
            && window[2] == "trials"
            && matches!(window[3].as_str(), "success" | "failed" | "inconclusive")
        {
            anyhow::bail!(
                "chat-session-token writes active reports; use docs/benchmarks/trials/active/ and move the report after review"
            );
        }
    }
    Ok(())
}

fn normalized_path_components(path: &str) -> Vec<String> {
    let mut components = Vec::new();
    let normalized = path.replace('\\', "/");
    for part in normalized.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if components
                    .last()
                    .is_some_and(|component: &String| component != "..")
                {
                    components.pop();
                } else {
                    components.push(part.to_string());
                }
            }
            _ => components.push(part.to_ascii_lowercase()),
        }
    }
    components
}
```

- [ ] **Step 4: Verify focused tests pass**

```bash
cargo test -p rllm-cli chat_session_token -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add crates/rllm-cli/src/commands/chat_session_token.rs
git commit -m "feat(cli): add token-native chat benchmark helpers"
```

## Task 2: CLI Registration

**Files:**
- Modify: `crates/rllm-cli/src/commands/mod.rs`
- Modify: `crates/rllm-cli/src/main.rs`

- [ ] **Step 1: Add the command module**

In `crates/rllm-cli/src/commands/mod.rs`, add:

```rust
pub mod chat_session_token;
```

- [ ] **Step 2: Add the Clap command**

In `crates/rllm-cli/src/main.rs`, add:

```rust
    /// Run a token-native full-replay vs persistent chat-session benchmark
    ChatSessionToken {
        /// Path to .rllm file
        file: String,

        /// Comma-separated token IDs for one user turn; pass this flag more than once
        #[arg(long = "turn-ids", required = true)]
        turns: Vec<String>,

        /// Maximum assistant tokens per turn
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,

        /// Maximum context length
        #[arg(long, default_value_t = 2048)]
        ctx: usize,

        /// Markdown report output path
        #[arg(long)]
        out: String,
    },
```

Route it in the match:

```rust
        Commands::ChatSessionToken {
            file,
            turns,
            max_new_tokens,
            ctx,
            out,
        } => commands::chat_session_token::run(&file, &turns, max_new_tokens, ctx, &out),
```

- [ ] **Step 3: Run help command**

```bash
cargo run -p rllm-cli -- chat-session-token --help
```

Expected: help lists `--turn-ids`, `--max-new-tokens`, `--ctx`, and `--out`.

- [ ] **Step 4: Commit**

```bash
git add crates/rllm-cli/src/commands/mod.rs crates/rllm-cli/src/main.rs
git commit -m "feat(cli): register token-native chat benchmark"
```

## Task 3: Full-Replay Vs Persistent Session Runner

**Files:**
- Modify: `crates/rllm-cli/src/commands/chat_session_token.rs`

- [ ] **Step 1: Write failing tests for token match helpers**

Add helper tests:

```rust
#[test]
fn token_match_summary_reports_match_and_mismatch() {
    let matched = token_match_summary(&[1, 2], &[1, 2]);
    assert_eq!(matched.matched, true);
    assert_eq!(matched.note, "match");

    let mismatched = token_match_summary(&[1, 2], &[1, 3]);
    assert_eq!(mismatched.matched, false);
    assert!(mismatched.note.contains("baseline=[1, 2] session=[1, 3]"));
}
```

- [ ] **Step 2: Run the focused failing test**

```bash
cargo test -p rllm-cli token_match_summary -- --nocapture
```

Expected: compile fails until helper structs/functions exist.

- [ ] **Step 3: Implement timing/result structs and runner**

Implement:

```rust
use rllm_runtime::{
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata,
        rama_layer_decoded_llama_transformer_generate_from_model, LlamaRamaGenerationConfig,
        LlamaRamaGenerationOptions, LlamaRamaSessionAdapter,
    },
    LazyRllmModel, MemoryBudget, RamaChatSession, RamaIntegrityMode, StreamingSamplingConfig,
};
use std::time::Instant;

#[derive(Debug, Clone)]
struct GenerationTiming {
    generated_tokens: usize,
    ttft_ms: f64,
    decode_ms: f64,
    end_to_end_ms: f64,
    decode_tok_s: f64,
    end_to_end_tok_s: f64,
}

#[derive(Debug, Clone)]
struct TokenMatchSummary {
    matched: bool,
    note: String,
}

fn token_match_summary(baseline: &[usize], session: &[usize]) -> TokenMatchSummary {
    if baseline == session {
        TokenMatchSummary {
            matched: true,
            note: "match".to_string(),
        }
    } else {
        TokenMatchSummary {
            matched: false,
            note: format!("mismatch baseline={baseline:?} session={session:?}"),
        }
    }
}
```

Then implement `pub fn run(...) -> Result<()>`:

```rust
pub fn run(
    file: &str,
    turns: &[String],
    max_new_tokens: usize,
    ctx: usize,
    out: &str,
) -> Result<()> {
    validate_report_output_path(out)?;
    if max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    if ctx == 0 {
        anyhow::bail!("--ctx must be greater than zero");
    }
    let token_turns = parse_token_turns(turns)?;

    let mut baseline_model =
        LazyRllmModel::open(file).with_context(|| format!("failed to open {file}"))?;
    baseline_model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
    let baseline_prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(
        &mut baseline_model,
        generation_config(max_new_tokens, ctx),
    )?;

    let mut session_model =
        LazyRllmModel::open(file).with_context(|| format!("failed to open {file}"))?;
    session_model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
    let session_prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(
        &mut session_model,
        generation_config(max_new_tokens, ctx),
    )?;
    let mut session_budget = MemoryBudget::unbounded();
    let adapter =
        LlamaRamaSessionAdapter::new(&mut session_model, &session_prepared, &mut session_budget)?;
    let mut session = RamaChatSession::new(adapter);

    let mut baseline_visible_history = Vec::new();
    let mut rows = Vec::new();
    let mut valid = true;

    for (idx, user_tokens) in token_turns.iter().enumerate() {
        let mut baseline_input = baseline_visible_history.clone();
        baseline_input.extend_from_slice(user_tokens);
        let (baseline_result, baseline_timing, baseline_peak) = run_baseline_turn(
            &mut baseline_model,
            &baseline_prepared,
            &baseline_input,
        )?;
        baseline_visible_history = baseline_result.token_ids.clone();

        let session_result = session.generate_turn(
            user_tokens,
            max_new_tokens,
            &mut session_budget,
            |_| true,
        )?;
        let generated_match =
            token_match_summary(&baseline_result.generated_token_ids, &session_result.generated_token_ids);
        let history_match = token_match_summary(&baseline_visible_history, session.token_history());
        valid &= generated_match.matched && history_match.matched;

        rows.push(TokenBenchmarkRow {
            turn_index: idx + 1,
            baseline_input_tokens: baseline_input.len(),
            session_input_tokens: user_tokens.len(),
            baseline_timing,
            baseline_peak_transient_bytes: baseline_peak,
            session_result,
            generated_match,
            history_match,
        });
    }

    write_report(out, file, max_new_tokens, ctx, valid, &rows)?;
    if !valid {
        anyhow::bail!("chat-session-token generated token/history mismatch; report written to {out}");
    }
    Ok(())
}
```

- [ ] **Step 4: Implement `run_baseline_turn`**

Use `Instant` around `rama_layer_decoded_llama_transformer_generate_from_model`, with the callback setting first-token time:

```rust
fn run_baseline_turn(
    model: &mut LazyRllmModel,
    prepared: &rllm_runtime::models::llama::LayerDecodedLlamaRamaTransformer,
    input_tokens: &[usize],
) -> Result<(rllm_runtime::models::llama::LlamaTextGenerationResult, GenerationTiming, usize)> {
    let mut budget = MemoryBudget::unbounded();
    let start = Instant::now();
    let mut first_token_time = None;
    let mut on_token = |_: usize| {
        if first_token_time.is_none() {
            first_token_time = Some(Instant::now());
        }
        true
    };
    let result = rama_layer_decoded_llama_transformer_generate_from_model(
        model,
        prepared,
        input_tokens,
        &mut budget,
        LlamaRamaGenerationOptions {
            collect_logits: false,
            ..Default::default()
        },
        &mut on_token,
    )?;
    let end = Instant::now();
    let first = first_token_time.unwrap_or(end);
    let ttft_ms = first.duration_since(start).as_secs_f64() * 1000.0;
    let decode_ms = end.duration_since(first).as_secs_f64() * 1000.0;
    let end_to_end_ms = end.duration_since(start).as_secs_f64() * 1000.0;
    let decode_steps = result.generated_token_ids.len().saturating_sub(1);
    let timing = GenerationTiming {
        generated_tokens: result.generated_token_ids.len(),
        ttft_ms,
        decode_ms,
        end_to_end_ms,
        decode_tok_s: if decode_steps == 0 {
            0.0
        } else {
            decode_steps as f64 / (decode_ms / 1000.0).max(f64::EPSILON)
        },
        end_to_end_tok_s: result.generated_token_ids.len() as f64
            / (end_to_end_ms / 1000.0).max(f64::EPSILON),
    };
    Ok((result, timing, budget.peak_bytes()))
}
```

- [ ] **Step 5: Run focused tests and check**

```bash
cargo test -p rllm-cli chat_session_token -- --nocapture
cargo check -p rllm-cli
```

- [ ] **Step 6: Commit**

```bash
git add crates/rllm-cli/src/commands/chat_session_token.rs
git commit -m "feat(cli): benchmark token-native chat sessions"
```

## Task 4: Report Writer And First Mini R2 Trial

**Files:**
- Modify: `crates/rllm-cli/src/commands/chat_session_token.rs`
- Create: `docs/benchmarks/trials/active/2026-06-14-r2-token-native-smollm2.md`
- Modify: `docs/benchmarks/trials/index.md`

- [ ] **Step 1: Add Markdown report writer**

The report must include:

- status `running`
- folder `active`
- command replay
- one row per turn with baseline/session metrics
- generated/history match notes
- decision `needs follow-up`

- [ ] **Step 2: Run the first mini R2 benchmark**

Use a small deterministic run first:

```bash
cargo run --release -p rllm-cli -- chat-session-token models/SmolLM2-135M-raw.rllm \
  --turn-ids 1 \
  --turn-ids 2 \
  --max-new-tokens 16 \
  --ctx 2048 \
  --out docs/benchmarks/trials/active/2026-06-14-r2-token-native-smollm2.md
```

Expected: command writes a report. If generated tokens/history mismatch, the command exits nonzero and the report must classify the run as incomplete/inconclusive evidence.

- [ ] **Step 3: Update the benchmark index**

Add a row to `docs/benchmarks/trials/index.md` using the measured outcome.

- [ ] **Step 4: Verify R2 scoped commands**

```bash
cargo fmt --check
cargo check -p rllm-cli
cargo test -p rllm-cli chat_session_token -- --nocapture
cargo clippy -p rllm-cli --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add crates/rllm-cli/src/commands/chat_session_token.rs docs/benchmarks/trials/active/2026-06-14-r2-token-native-smollm2.md docs/benchmarks/trials/index.md
git commit -m "docs: record r2 token-native benchmark"
```

## Final Verification

Run:

```bash
cargo fmt --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run --quiet -- doctor
git status --short --branch
```

Expected: all commands pass and working tree is clean.
