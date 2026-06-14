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

    let mut report_turns = Vec::new();
    for (idx, turn) in turns.iter().enumerate() {
        let input_token_ids = tokenizer.encode(turn)?;
        let result =
            session.generate_turn(&input_token_ids, max_new_tokens, &mut budget, |_| true)?;
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
    body.push_str("Keeping KV-cache alive across turns reduces turn 2 prefill latency because only new user tokens are appended.\n\n");
    body.push_str("## Scope\n\n");
    body.push_str("- Mode: exact-lowram\n");
    body.push_str(&format!("- Model/artifact: {}\n", markdown_code_span(file)));
    body.push_str("- Architecture: llama\n");
    body.push_str("- Target device/profile: single CPU, low RAM\n");
    body.push_str("- Expected bottleneck: full-history replay and memory bandwidth\n");
    body.push_str("- Bottleneck tag: cache locality\n\n");
    body.push_str("## Setup\n\n");
    body.push_str("Commands:\n\n```bash\n");
    body.push_str(&format!(
        "cargo run -p rllm-cli -- chat-session {}{} --max-new-tokens {max_new_tokens} --ctx {ctx} --out {}\n",
        shell_quote(file),
        format_turn_args(turns),
        shell_quote(out)
    ));
    body.push_str("```\n\n");
    body.push_str("## Results\n\n");
    body.push_str("| run | prompt/input tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |\n");
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
    body.push_str("Turn 2 is valid only if `replayed_tokens` remains zero and `flushed_pending_tokens` is one when turn 1 generated at least one assistant token.\n\n");
    body.push_str("## Decision\n\n");
    body.push_str("needs follow-up\n\n");
    body.push_str("Reason: compare this report against the existing `llama-test` full-replay baseline before moving it to success or failed.\n\n");
    body.push_str("Paper value:\n\n- use as limitation\n\n");
    body.push_str("## Next Experiment\n\n");
    body.push_str("Run the same turns through the old full-replay chat path and compare turn 2 TTFT, decode tok/s, and memory.\n");
    fs::write(out, body)?;
    Ok(())
}

fn format_turn_args(turns: &[(usize, String, rllm_runtime::RamaSessionTurnResult)]) -> String {
    turns
        .iter()
        .map(|(_, text, _)| format!(" --turn {}", shell_quote(text)))
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

fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}
