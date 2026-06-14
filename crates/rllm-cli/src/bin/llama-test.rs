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
}

fn interactive_turn_text(has_context: bool, text: &str) -> String {
    if has_context {
        format!("\n{text}")
    } else {
        text.to_string()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut model = LazyRllmModel::open(&args.model)?;

    let tokenizer_meta = model
        .metadata()
        .tokenizer
        .as_ref()
        .context("Model does not have tokenizer metadata packed inside")?;

    let tokenizer = RllmTokenizer::from_metadata(tokenizer_meta)?;
    let eos_token_id = tokenizer_meta.eos_token_id;

    let config = LlamaRamaGenerationConfig {
        max_new_tokens: 64,
        max_seq_len: Some(2048),
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
        println!(
            "\n[TTFT/Prefill: {:.2}s | Decode: {:.2} tok/s | E2E: {:.2} tok/s | Total: {} tokens | Context: {} tokens | Peak: {} bytes]",
            result.metrics.ttft_ms / 1000.0,
            result.metrics.decode_tok_s,
            result.metrics.end_to_end_tok_s,
            result.metrics.generated_tokens,
            session.token_history().len(),
            result.metrics.peak_transient_bytes
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
}
