use anyhow::{Context, Result};
use clap::Parser;
use std::io::{self, Write};

use rllm_runtime::{
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata, LlamaRamaGenerationConfig,
        LlamaRamaSessionAdapter,
    },
    LazyRllmModel, MemoryBudget, RamaChatSession, RamaIntegrityMode, RllmTokenizer,
    StreamingSamplingConfig,
};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: String,

    /// Maximum context length for the persistent token session.
    #[arg(long, default_value_t = 2048)]
    ctx: usize,

    /// Maximum assistant tokens to generate per turn.
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
}

fn interactive_turn_text(has_context: bool, text: &str) -> String {
    if has_context {
        format!("\n{text}")
    } else {
        text.to_string()
    }
}

fn format_rolling_suffix(stats: rllm_runtime::RamaRollingStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        format!(
            " | Rolling: tasks={} wakeups={} fallbacks={} scratch={} bytes",
            stats.submitted_tasks,
            stats.worker_wakeups,
            stats.sequential_fallbacks,
            stats.peak_scratch_bytes
        )
    }
}

fn format_aip_suffix(stats: rllm_runtime::RamaExperimentalSpeedStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        let policy_str = stats.aip_policy.map(|p| p.as_str()).unwrap_or("none");
        let lm_head_note = if stats.lm_head_prefix_rows > 0 {
            format!(
                " lm_head_rows={}/{}",
                stats.lm_head_prefix_rows, stats.lm_head_vocab_rows
            )
        } else {
            String::new()
        };
        format!(
            " | AIP: policy={} calls={} fallbacks={} max_topk={} skipped_madds={} scratch={} bytes",
            policy_str,
            stats.sparse_projection_calls,
            stats.exact_fallbacks,
            stats.max_selected_topk,
            stats.estimated_skipped_madds,
            stats.peak_scratch_bytes
        ) + &lm_head_note
    }
}

fn format_repetition_suffix(stats: rllm_runtime::RamaRepetitionStats) -> String {
    if stats.generated_tokens == 0 {
        String::new()
    } else {
        format!(
            " | Repetition: ratio={:.2} max_run={} unique={}/{}",
            stats.repeated_token_ratio,
            stats.max_repeated_token_run,
            stats.unique_generated_tokens,
            stats.generated_tokens
        )
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.ctx == 0 {
        anyhow::bail!("--ctx must be greater than zero");
    }
    if args.max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    let mut model = LazyRllmModel::open(&args.model)?;

    let tokenizer_meta = model
        .metadata()
        .tokenizer
        .as_ref()
        .context("Model does not have tokenizer metadata packed inside")?;

    let tokenizer = RllmTokenizer::from_metadata(tokenizer_meta)?;
    let eos_token_id = tokenizer_meta.eos_token_id;

    let config = LlamaRamaGenerationConfig {
        max_new_tokens: args.max_new_tokens,
        max_seq_len: Some(args.ctx),
        causal: true,
        sampling: StreamingSamplingConfig::Argmax,
    };

    // VerifyOnce: verify each chunk SHA-256 only on first access, then trust it.
    // This eliminates ~420 redundant SHA-256 computations per generated token.
    model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);

    let prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(&mut model, config)?;
    let mut budget = MemoryBudget::unbounded();
    let adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget)?;
    let mut session = RamaChatSession::new(adapter);

    println!("===================================================");
    println!("RLLM Interactive Chat (Llama Architecture, token-native session)");
    println!("Type 'quit' or 'exit' to end.");
    println!("===================================================");

    let mut has_context = false;

    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            // EOF: stdin pipe was closed.
            break;
        }
        let text = input.trim();
        if text.is_empty() {
            continue;
        }
        if text == "exit" || text == "quit" {
            break;
        }
        let turn_text = interactive_turn_text(has_context, text);
        let input_tokens = tokenizer.encode(&turn_text)?;

        let mut on_token = |token: usize| -> bool {
            if let Ok(word) = tokenizer.decode(&[token]) {
                print!("{}", word);
                io::stdout().flush().unwrap();
            }

            Some(token as u64) != eos_token_id
        };

        let result = session.generate_turn(
            &input_tokens,
            config.max_new_tokens,
            &mut budget,
            &mut on_token,
        )?;

        println!();
        let rolling_suffix = format_rolling_suffix(result.metrics.rolling_stats);
        let aip_suffix = format_aip_suffix(result.metrics.experimental_speed_stats);
        let repetition_suffix = format_repetition_suffix(result.metrics.repetition_stats);
        println!(
            "\n[TTFT/Prefill: {:.2}s | Decode: {:.2} tok/s | E2E: {:.2} tok/s | Total: {} tokens | Context: {} tokens | Peak: {} bytes{}{}{}]",
            result.metrics.ttft_ms / 1000.0,
            result.metrics.decode_tok_s,
            result.metrics.end_to_end_tok_s,
            result.metrics.generated_tokens,
            session.token_history().len(),
            result.metrics.peak_transient_bytes,
            rolling_suffix,
            aip_suffix,
            repetition_suffix
        );
        has_context = true;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interactive_turn_text_uses_only_current_turn_with_separator() {
        assert_eq!(interactive_turn_text(false, "good morning"), "good morning");
        assert_eq!(interactive_turn_text(true, "halo"), "\nhalo");
    }

    #[test]
    fn rolling_suffix_is_empty_without_activity() {
        assert_eq!(
            format_rolling_suffix(rllm_runtime::RamaRollingStats::default()),
            ""
        );
    }

    #[test]
    fn rolling_suffix_reports_nonzero_activity() {
        let suffix = format_rolling_suffix(rllm_runtime::RamaRollingStats {
            submitted_tasks: 8,
            worker_wakeups: 8,
            sequential_fallbacks: 1,
            peak_scratch_bytes: 64,
        });

        assert!(suffix.contains("tasks=8"));
        assert!(suffix.contains("fallbacks=1"));
    }

    #[test]
    fn aip_suffix_is_empty_without_activity() {
        assert_eq!(
            format_aip_suffix(rllm_runtime::RamaExperimentalSpeedStats::default()),
            ""
        );
    }

    #[test]
    fn aip_suffix_reports_nonzero_activity() {
        let suffix = format_aip_suffix(rllm_runtime::RamaExperimentalSpeedStats {
            aip_policy: Some(rllm_runtime::RamaAipPolicyKind::Quality),
            sparse_projection_calls: 4,
            exact_fallbacks: 1,
            selected_topk_sum: 256,
            max_selected_topk: 128,
            estimated_skipped_madds: 2048,
            peak_scratch_bytes: 512,
            lm_head_prefix_rows: 512,
            lm_head_vocab_rows: 128_256,
        });

        assert!(suffix.contains("AIP: policy=quality"));
        assert!(suffix.contains("policy=quality"));
        assert!(suffix.contains("calls=4"));
        assert!(suffix.contains("fallbacks=1"));
        assert!(suffix.contains("max_topk=128"));
        assert!(suffix.contains("skipped_madds=2048"));
        assert!(suffix.contains("lm_head_rows=512/128256"));
    }

    #[test]
    fn repetition_suffix_is_empty_without_activity() {
        assert_eq!(
            format_repetition_suffix(rllm_runtime::RamaRepetitionStats::default()),
            ""
        );
    }

    #[test]
    fn repetition_suffix_reports_nonzero_activity() {
        let suffix = format_repetition_suffix(rllm_runtime::RamaRepetitionStats {
            generated_tokens: 10,
            unique_generated_tokens: 5,
            max_repeated_token_run: 3,
            repeated_token_ratio: 0.25,
        });

        assert!(suffix.contains("ratio=0.25"));
        assert!(suffix.contains("max_run=3"));
        assert!(suffix.contains("unique=5/10"));
    }

    #[test]
    fn args_default_to_2k_context_and_accept_override() {
        let default_args = Args::parse_from(["llama-test", "--model", "model.rllm"]);
        assert_eq!(default_args.ctx, 2048);

        let overridden_args =
            Args::parse_from(["llama-test", "--model", "model.rllm", "--ctx", "4096"]);
        assert_eq!(overridden_args.ctx, 4096);
    }

    #[test]
    fn args_default_to_64_new_tokens_and_accept_override() {
        let default_args = Args::parse_from(["llama-test", "--model", "model.rllm"]);
        assert_eq!(default_args.max_new_tokens, 64);

        let overridden_args = Args::parse_from([
            "llama-test",
            "--model",
            "model.rllm",
            "--max-new-tokens",
            "1",
        ]);
        assert_eq!(overridden_args.max_new_tokens, 1);
    }
}
