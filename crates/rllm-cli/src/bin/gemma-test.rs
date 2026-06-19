use anyhow::{Context, Result};
use clap::Parser;
use std::io::{self, Write};
use std::time::Instant;

use rllm_runtime::{
    models::gemma::{
        gemma_generate_from_model, prepare_gemma_transformer_from_metadata, GemmaGenerationConfig,
        GemmaGenerationOptions,
    },
    LazyRllmModel, MemoryBudget, RamaIntegrityMode, RllmTokenizer, StreamingSamplingConfig,
};

/// Single-shot Gemma text generation harness (Phase 2 adapter bring-up).
///
/// Runs prefill + greedy decode through the `models::gemma` adapter and prints
/// the continuation. `--logits-out` dumps the first-token logits for HF parity.
#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: String,

    /// Prompt text (mutually exclusive with --token-ids).
    #[arg(long)]
    prompt: Option<String>,

    /// Comma-separated prompt token ids (bypasses the tokenizer).
    #[arg(long)]
    token_ids: Option<String>,

    /// Wrap the prompt in Gemma's chat template
    /// (`<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`).
    #[arg(long, default_value_t = false)]
    chat: bool,

    #[arg(long, default_value_t = 2048)]
    ctx: usize,

    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,

    /// Optional JSON output path for the first decode step's logits.
    #[arg(long)]
    logits_out: Option<String>,

    /// Pin the whole model mapping in RAM (mlock) so the OS cannot evict it.
    /// On a machine where the model fits available RAM this keeps the weights
    /// resident across decode steps instead of re-faulting them from disk every
    /// token — a large decode speedup (matches llama.cpp's --mlock). Opt-in
    /// because it risks OOM when the working set exceeds physical RAM.
    #[arg(long, default_value_t = false)]
    mlock: bool,
}

fn parse_token_ids(raw: &str) -> Result<Vec<usize>> {
    let mut ids = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        ids.push(part.parse::<usize>().with_context(|| format!("invalid token id: {part}"))?);
    }
    if ids.is_empty() {
        anyhow::bail!("--token-ids must contain at least one token id");
    }
    Ok(ids)
}

fn write_first_step_logits(path: &str, logits: &[f32], first_token: usize) -> Result<()> {
    let mut body = String::with_capacity(logits.len() * 8 + 64);
    body.push_str(&format!("{{\"first_token\":{first_token},\"logits\":["));
    for (i, value) in logits.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        body.push_str(&format!("{value}"));
    }
    body.push_str("]}");
    std::fs::write(path, body).with_context(|| format!("failed to write logits JSON {path}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use rllm_import::tokenizer_metadata_from_json_str;
    use rllm_runtime::RllmTokenizer;

    /// Validate the metaspace tokenizer path against the REAL Gemma 3
    /// tokenizer.json (needs the local download). Run:
    /// `cargo test -p rllm-cli --bin gemma-test -- --ignored --nocapture real_gemma_tokenizer`
    #[test]
    #[ignore]
    fn real_gemma_tokenizer_encodes_with_metaspace_and_bos() {
        let json =
            std::fs::read_to_string("../../models/gemma-3-4b-it/tokenizer.json").unwrap();
        let meta = tokenizer_metadata_from_json_str(&json).unwrap();
        assert_eq!(meta.pre_tokenizer.as_deref(), Some("metaspace"));
        assert_eq!(meta.add_bos_token, Some(true));
        assert_eq!(meta.bos_token_id, Some(2));
        let tok = RllmTokenizer::from_metadata(&meta).unwrap();

        let ids = tok.encode("The capital of France is").unwrap();
        eprintln!("encode(\"The capital of France is\") = {ids:?}");
        // BOS(2) + faithful SentencePiece (no dummy prefix): leading word "The"
        // has no metaspace; subsequent words carry ▁.
        assert_eq!(ids, vec![2, 818, 5279, 529, 7001, 563]);
        assert_eq!(tok.decode(&ids[1..]).unwrap(), "The capital of France is");
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.ctx == 0 || args.max_new_tokens == 0 {
        anyhow::bail!("--ctx and --max-new-tokens must be greater than zero");
    }
    if args.prompt.is_some() == args.token_ids.is_some() {
        anyhow::bail!("provide exactly one of --prompt or --token-ids");
    }

    // Residency: --mlock pins the model mapping in RAM. The reader reads this
    // via RLLM_MLOCK (the same env-gated knob other runtime experiments use),
    // so translate the flag before opening the model. The flag takes precedence
    // only to enable; an externally-set RLLM_MLOCK=1 still works without --mlock.
    if args.mlock {
        std::env::set_var("RLLM_MLOCK", "1");
        eprintln!("[gemma-test] --mlock: pinning model in RAM (mlock)");
    }

    let mut model = LazyRllmModel::open(&args.model)?;
    model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);

    let tokenizer = model
        .metadata()
        .tokenizer
        .clone()
        .map(|meta| RllmTokenizer::from_metadata(&meta))
        .transpose()?;
    let eos_token_id = model
        .metadata()
        .tokenizer
        .as_ref()
        .and_then(|meta| meta.eos_token_id);

    // Gemma instruction-tuned models end a turn with `<end_of_turn>` as well as
    // `<eos>` (HF generation_config lists eos_token_id = [1, 106]). The packed
    // metadata only carries one eos id, so also resolve `<end_of_turn>` from the
    // tokenizer and treat both as stop tokens; without this the decode loop runs
    // to --max-new-tokens and pads the output with repeated `<end_of_turn>`.
    let mut stop_token_ids: Vec<usize> = Vec::new();
    if let Some(eos) = eos_token_id {
        stop_token_ids.push(eos as usize);
    }
    if let Some(eot) =
        tokenizer.as_ref().and_then(|t| t.token_id_for_raw_token("<end_of_turn>"))
    {
        if !stop_token_ids.contains(&eot) {
            stop_token_ids.push(eot);
        }
    }

    let prompt_token_ids = if let Some(prompt) = args.prompt.as_deref() {
        let tokenizer = tokenizer
            .as_ref()
            .context("model has no tokenizer metadata; use --token-ids")?;
        let text = if args.chat {
            format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")
        } else {
            prompt.to_string()
        };
        tokenizer.encode(&text)?
    } else {
        parse_token_ids(args.token_ids.as_deref().unwrap())?
    };

    let prepared = prepare_gemma_transformer_from_metadata(
        &mut model,
        GemmaGenerationConfig {
            max_new_tokens: args.max_new_tokens,
            max_seq_len: Some(args.ctx),
            causal: true,
            sampling: StreamingSamplingConfig::Argmax,
        },
    )?;

    println!("===================================================");
    println!("RLLM Gemma single-shot generation (Phase 2 adapter)");
    println!("Model: {}", model.metadata().model_name);
    println!("Architecture: {}", model.metadata().architecture);
    println!(
        "layers={} hidden={} heads={}/{} head_dim={} attn_scale={:.5} embed_scale={:.4} pattern={}",
        prepared.layers.len(),
        prepared.config.hidden_size,
        prepared.config.num_heads,
        prepared.config.num_key_value_heads,
        prepared.config.head_dim,
        prepared.config.attn_scale,
        prepared.config.embed_scale,
        prepared.config.sliding_window_pattern,
    );
    println!("Prompt token ids: {prompt_token_ids:?}");
    println!("===================================================");

    let mut budget = MemoryBudget::unbounded();
    let started = Instant::now();
    let mut decoded_so_far = String::new();
    let mut on_token = |token: usize| -> bool {
        if stop_token_ids.contains(&token) {
            return false;
        }
        if let Some(tokenizer) = tokenizer.as_ref() {
            // Re-decode the running sequence so multi-token glyphs render.
            decoded_so_far.push_str(&tokenizer.decode(&[token]).unwrap_or_default());
            print!("\r{decoded_so_far}");
            let _ = io::stdout().flush();
        }
        true
    };

    let result = gemma_generate_from_model(
        &mut model,
        &prepared,
        &prompt_token_ids,
        &mut budget,
        GemmaGenerationOptions {
            collect_logits: args.logits_out.is_some(),
        },
        &mut on_token,
    )?;
    let elapsed = started.elapsed().as_secs_f64();

    println!();
    if let Some(tokenizer) = tokenizer.as_ref() {
        // Hide stop tokens (e.g. `<end_of_turn>`) from the rendered text; the raw
        // ids below still show them so the stop reason stays visible for debugging.
        let visible: Vec<usize> = result
            .generated_token_ids
            .iter()
            .copied()
            .filter(|id| !stop_token_ids.contains(id))
            .collect();
        let text = tokenizer.decode(&visible)?;
        println!("\n--- generated ({} tokens) ---", result.generated_token_ids.len());
        println!("{text}");
    }
    println!("Generated token ids: {:?}", result.generated_token_ids);
    println!(
        "Tokens: {} in {:.2}s ({:.2} tok/s) | peak transient {} bytes | kv {} bytes",
        result.generated_token_ids.len(),
        elapsed,
        result.generated_token_ids.len() as f64 / elapsed.max(1e-9),
        budget.peak_bytes(),
        result.context_echo_bytes,
    );

    if let Some(path) = args.logits_out.as_deref() {
        match result.logits.as_ref() {
            Some(logits) => {
                let first = result.generated_token_ids.first().copied().unwrap_or(0);
                write_first_step_logits(path, logits, first)?;
                eprintln!("first-step logits ({} values) written to {path}", logits.len());
            }
            None => eprintln!("warning: --logits-out set but no logits were collected"),
        }
    }
    Ok(())
}
