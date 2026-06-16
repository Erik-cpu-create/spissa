use anyhow::{Context, Result};
use rllm_runtime::{
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata, LlamaRamaGenerationConfig,
        LlamaRamaSessionAdapter,
    },
    LazyRllmModel, MemoryBudget, RamaChatSession, RamaIntegrityMode, RllmTokenizer,
    StreamingSamplingConfig,
};
use std::fs;
use std::path::Path;

pub fn run(
    file: &str,
    turns: &[String],
    max_new_tokens: usize,
    ctx: usize,
    out: &str,
) -> Result<()> {
    validate_turns(turns)?;
    validate_report_output_path(out)?;
    if max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    if ctx == 0 {
        anyhow::bail!("--ctx must be greater than zero");
    }

    let mut model = LazyRllmModel::open(file).with_context(|| format!("failed to open {file}"))?;
    model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
    let tokenizer_meta = model
        .metadata()
        .tokenizer
        .as_ref()
        .context("model metadata does not include tokenizer metadata")?;
    let tokenizer = RllmTokenizer::from_metadata(tokenizer_meta)?;
    let eos_token_id = tokenizer_meta.eos_token_id;
    let prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(
        &mut model,
        LlamaRamaGenerationConfig {
            max_new_tokens,
            max_seq_len: Some(ctx),
            causal: true,
            sampling: StreamingSamplingConfig::Argmax,
        },
    )?;
    let mut budget = MemoryBudget::unbounded();
    let adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget)?;
    let mut session = RamaChatSession::new(adapter);
    let mut transcript = TextTranscript::default();

    let mut report_turns = Vec::new();
    for (idx, turn) in turns.iter().enumerate() {
        let input_token_ids =
            transcript.append_user_turn(&tokenizer, session.token_history(), turn)?;
        let result =
            session.generate_turn(&input_token_ids, max_new_tokens, &mut budget, |token| {
                Some(token as u64) != eos_token_id
            })?;
        transcript.append_assistant_tokens(
            &tokenizer,
            session.token_history(),
            &result.generated_token_ids,
        )?;
        println!(
            "turn {}: input={} generated={} replayed={} ttft_ms={:.2} decode_tok_s={:.2}",
            idx + 1,
            result.metrics.input_tokens,
            result.metrics.generated_tokens,
            result.metrics.replayed_tokens,
            result.metrics.ttft_ms,
            result.metrics.decode_tok_s
        );
        report_turns.push((idx + 1, turn.clone(), result));
    }

    write_report(out, file, max_new_tokens, ctx, &report_turns)?;
    println!("Benchmark report: {out}");
    Ok(())
}

fn write_report(
    out: &str,
    file: &str,
    max_new_tokens: usize,
    ctx: usize,
    turns: &[(usize, String, rllm_runtime::RamaSessionTurnResult)],
) -> Result<()> {
    if let Some(parent) = Path::new(out).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut body = String::new();
    body.push_str("# Trial: R1 Persistent Chat Session SmolLM2\n\n");
    body.push_str("Date: 2026-06-14\n");
    body.push_str("Owner: RLLM\n");
    body.push_str("Status: running\n");
    body.push_str("Folder: active\n\n");
    body.push_str("## Hypothesis\n\n");
    body.push_str("Keeping KV-cache alive across turns reduces turn 2 prefill latency because only the new transcript suffix is appended.\n\n");
    body.push_str("## Scope\n\n");
    body.push_str("- Mode: exact-lowram\n");
    body.push_str(&format!("- Model/artifact: {}\n", markdown_code_span(file)));
    body.push_str("- Architecture: llama\n");
    body.push_str("- Target device/profile: single CPU, low RAM\n");
    body.push_str("- Expected bottleneck: full-history replay and memory bandwidth\n");
    body.push_str("- Bottleneck tag: cache locality\n\n");
    body.push_str("## Setup\n\n");
    body.push_str("Commands:\n\n");
    let replay_command = format!(
        "cargo run -p rllm-cli -- chat-session {}{} --max-new-tokens {max_new_tokens} --ctx {ctx} --out {}",
        shell_quote(file),
        format_turn_args(turns.iter().map(|(_, text, _)| text.as_str())),
        shell_quote(out)
    );
    body.push_str(&markdown_code_fence(&replay_command));
    body.push('\n');
    body.push_str("## Results\n\n");
    body.push_str("| run | transcript suffix tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |\n");
    body.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for (idx, _, result) in turns {
        body.push_str(&format!(
            "| turn {idx} | {} | {} | {:.2} ms | {:.2} | {:.2} | not captured | {} bytes | replayed_tokens={} flushed_pending_tokens={} context_bytes={} |\n",
            result.metrics.input_tokens,
            result.metrics.generated_tokens,
            result.metrics.ttft_ms,
            result.metrics.decode_tok_s,
            result.metrics.end_to_end_tok_s,
            result.metrics.peak_transient_bytes,
            result.metrics.replayed_tokens,
            result.metrics.flushed_pending_tokens,
            result.metrics.context_memory_bytes
        ));
    }
    body.push_str("\n## Analysis\n\n");
    body.push_str("Turn 2 is valid only if `replayed_tokens` remains zero and `flushed_pending_tokens` is one when turn 1 generated at least one assistant token. The text transcript is validated against full transcript tokenization before each turn.\n\n");
    body.push_str("## Decision\n\n");
    body.push_str("needs follow-up\n\n");
    body.push_str("Reason: compare this report against the existing `llama-test` full-replay baseline before moving it to success or failed.\n\n");
    body.push_str("Paper value:\n\n- use as limitation\n\n");
    body.push_str("## Next Experiment\n\n");
    body.push_str("Run the same turns through the old full-replay chat path and compare turn 2 TTFT, decode tok/s, and memory.\n");
    fs::write(out, body)?;
    Ok(())
}

#[derive(Debug, Clone, Default)]
struct TextTranscript {
    text: String,
    pending_turn_separator: bool,
}

impl TextTranscript {
    fn append_user_turn(
        &mut self,
        tokenizer: &RllmTokenizer,
        cached_token_history: &[usize],
        turn: &str,
    ) -> Result<Vec<usize>> {
        let mut next_text = self.text.clone();
        if self.pending_turn_separator && !next_text.is_empty() {
            next_text.push('\n');
        }
        next_text.push_str(turn);

        let full_tokens = tokenizer.encode(&next_text)?;
        let suffix = transcript_token_suffix(cached_token_history, &full_tokens)?;
        if suffix.is_empty() {
            anyhow::bail!("chat-session turn did not add any transcript tokens");
        }

        self.text = next_text;
        self.pending_turn_separator = false;
        Ok(suffix)
    }

    fn append_assistant_tokens(
        &mut self,
        tokenizer: &RllmTokenizer,
        session_token_history: &[usize],
        generated_token_ids: &[usize],
    ) -> Result<()> {
        let assistant_text = tokenizer.decode(generated_token_ids)?;
        let mut next_text = self.text.clone();
        next_text.push_str(&assistant_text);
        let full_tokens = tokenizer.encode(&next_text)?;
        if full_tokens != session_token_history {
            anyhow::bail!(
                "chat-session token history does not match full transcript tokenization; \
                 choose a transcript boundary that tokenizes incrementally"
            );
        }

        self.text = next_text;
        self.pending_turn_separator = true;
        Ok(())
    }
}

fn transcript_token_suffix(
    cached_token_history: &[usize],
    full_tokens: &[usize],
) -> Result<Vec<usize>> {
    if !full_tokens.starts_with(cached_token_history) {
        anyhow::bail!(
            "chat-session cached token history does not match full transcript tokenization"
        );
    }
    Ok(full_tokens[cached_token_history.len()..].to_vec())
}

fn validate_turns(turns: &[String]) -> Result<()> {
    if turns.is_empty() {
        anyhow::bail!("chat-session requires at least one --turn");
    }
    for (idx, turn) in turns.iter().enumerate() {
        if turn.trim().is_empty() {
            anyhow::bail!(
                "chat-session turn {} must not be empty or whitespace-only",
                idx + 1
            );
        }
    }
    Ok(())
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
                "chat-session writes active reports; use docs/benchmarks/trials/active/ and move the report after review"
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

fn format_turn_args<I, S>(turns: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    turns
        .into_iter()
        .map(|text| format!(" --turn={}", shell_quote(text.as_ref())))
        .collect::<String>()
}

fn markdown_code_span(text: &str) -> String {
    if !text.contains('`') {
        return format!("`{text}`");
    }

    let delimiter_len = text.split(|ch| ch != '`').map(str::len).max().unwrap_or(0) + 1;
    let delimiter = "`".repeat(delimiter_len);
    format!("{delimiter} {text} {delimiter}")
}

fn markdown_code_fence(contents: &str) -> String {
    let longest_backtick_run = contents
        .split(|ch| ch != '`')
        .map(str::len)
        .max()
        .unwrap_or(0);
    let delimiter = "`".repeat(3.max(longest_backtick_run + 1));
    format!("{delimiter}bash\n{contents}\n{delimiter}\n")
}

fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::{
        format_turn_args, markdown_code_fence, shell_quote, validate_report_output_path,
        validate_turns, TextTranscript,
    };
    use rllm_container::TokenizerMetadata;
    use rllm_runtime::RllmTokenizer;

    fn tokenizer(tokens: &[&str]) -> RllmTokenizer {
        RllmTokenizer::from_metadata(&TokenizerMetadata {
            tokenizer_type: Some("test".to_string()),
            id_to_token: tokens.iter().map(|token| (*token).to_string()).collect(),
            bpe_merges: Vec::new(),
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
        })
        .unwrap()
    }

    #[test]
    fn format_turn_args_uses_equals_form_for_hyphen_leading_turns() {
        assert_eq!(format_turn_args(["-x"]), " --turn='-x'");
    }

    #[test]
    fn shell_quote_escapes_single_quotes() {
        assert_eq!(shell_quote("can't stop"), "'can'\\''t stop'");
    }

    #[test]
    fn markdown_code_fence_extends_past_backticks_in_contents() {
        let fenced = markdown_code_fence("cargo run -- --turn='contains ``` fence'");

        assert!(fenced.starts_with("````bash\n"));
        assert!(fenced.contains("cargo run -- --turn='contains ``` fence'"));
        assert!(fenced.ends_with("\n````\n"));
    }

    #[test]
    fn validate_turns_rejects_empty_and_whitespace_turns() {
        let no_turns = validate_turns(&[]);
        assert!(no_turns
            .unwrap_err()
            .to_string()
            .contains("requires at least one --turn"));

        let blank_turn = validate_turns(&["ok".to_string(), "  ".to_string()]);
        assert!(blank_turn
            .unwrap_err()
            .to_string()
            .contains("turn 2 must not be empty"));
    }

    #[test]
    fn report_output_rejects_reviewed_trial_folders() {
        validate_report_output_path("docs/benchmarks/trials/active/run.md").unwrap();
        validate_report_output_path("/tmp/run.md").unwrap();

        for folder in ["success", "failed", "inconclusive"] {
            let result =
                validate_report_output_path(&format!("docs/benchmarks/trials/{folder}/run.md"));
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("writes active reports"));
        }

        let traversal =
            validate_report_output_path("docs/benchmarks/trials/active/../success/replay.md");
        assert!(traversal
            .unwrap_err()
            .to_string()
            .contains("writes active reports"));

        let case_variant = validate_report_output_path("docs/benchmarks/trials/Success/replay.md");
        assert!(case_variant
            .unwrap_err()
            .to_string()
            .contains("writes active reports"));
    }

    #[test]
    fn transcript_turn_input_uses_full_history_suffix() {
        let tokenizer = tokenizer(&["Hello", " reply", "ĊContinue", "Continue"]);
        let mut transcript = TextTranscript::default();

        let first = transcript
            .append_user_turn(&tokenizer, &[], "Hello")
            .unwrap();
        assert_eq!(first, [0]);

        transcript
            .append_assistant_tokens(&tokenizer, &[0, 1], &[1])
            .unwrap();
        let second = transcript
            .append_user_turn(&tokenizer, &[0, 1], "Continue")
            .unwrap();

        assert_eq!(second, [2]);
    }

    #[test]
    fn transcript_rejects_non_incremental_tokenization() {
        let tokenizer = tokenizer(&["A", "B", "AB"]);
        let mut transcript = TextTranscript::default();

        let first = transcript.append_user_turn(&tokenizer, &[], "A").unwrap();
        assert_eq!(first, [0]);

        let result = transcript.append_assistant_tokens(&tokenizer, &[0, 1], &[1]);

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("does not match full transcript tokenization"));
    }
}
