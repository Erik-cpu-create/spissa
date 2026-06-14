use anyhow::{Context, Result};
use clap::Parser;
use std::io::{self, Write};
use std::time::Instant;

use rllm_runtime::{
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata,
        rama_layer_decoded_llama_transformer_generate_from_model, LlamaRamaGenerationConfig,
        LlamaRamaGenerationOptions,
    },
    LazyRllmModel, MemoryBudget, RamaIntegrityMode, RllmTokenizer,
};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: String,
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
        sampling: rllm_runtime::StreamingSamplingConfig::Argmax,
    };

    // VerifyOnce: verify each chunk SHA-256 only on first access, then trust it.
    // This eliminates ~420 redundant SHA-256 computations per generated token.
    model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);

    let prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(&mut model, config)?;

    println!("===================================================");
    println!("RLLM Interactive Chat (Llama Architecture)");
    println!("Type 'quit' or 'exit' to end.");
    println!("===================================================");

    let mut conversation_history = String::new();

    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            // EOF — stdin pipe was closed
            break;
        }
        let text = input.trim();
        if text.is_empty() {
            continue;
        }
        if text == "exit" || text == "quit" {
            break;
        }
        // Use raw text as prompt (base model completion, not chat-instruct)
        conversation_history.push_str(text);

        let prompt_tokens = tokenizer.encode(&conversation_history)?;

        let mut generated_tokens = Vec::new();
        let options = LlamaRamaGenerationOptions {
            collect_logits: false,
            ..Default::default()
        };

        let mut budget = MemoryBudget::unbounded();

        let start_time = Instant::now();
        let mut first_token_time = None;

        let mut on_token = |token: usize| -> bool {
            if first_token_time.is_none() {
                first_token_time = Some(Instant::now());
            }
            generated_tokens.push(token);
            if let Ok(word) = tokenizer.decode(&[token]) {
                print!("{}", word);
                io::stdout().flush().unwrap();
            }

            Some(token as u64) != eos_token_id
        };

        rama_layer_decoded_llama_transformer_generate_from_model(
            &mut model,
            &prepared,
            &prompt_tokens,
            &mut budget,
            options,
            &mut on_token,
        )?;

        let end_time = Instant::now();
        println!();

        if let Some(first_time) = first_token_time {
            let prefill_duration = first_time.duration_since(start_time);
            let decode_duration = end_time.duration_since(first_time);
            let decode_speed = if generated_tokens.len() > 1 {
                (generated_tokens.len() - 1) as f64 / decode_duration.as_secs_f64()
            } else {
                0.0
            };
            println!(
                "\n[Prefill: {:.2}s | Decode: {:.2} tok/s | Total: {} tokens]",
                prefill_duration.as_secs_f64(),
                decode_speed,
                generated_tokens.len()
            );
        }

        let reply = tokenizer.decode(&generated_tokens).unwrap_or_default();
        conversation_history.push_str(&reply);
        conversation_history.push('\n');
    }

    Ok(())
}
