use anyhow::{Context, Result};
use clap::Parser;
use std::io::{self, BufRead, IsTerminal, Write};
use std::time::Instant;

use rllm_runtime::{
    models::gemma::{
        gemma_generate_from_model, prepare_gemma_transformer_from_metadata, GemmaChatSession,
        GemmaGenerationConfig, GemmaGenerationOptions, PreparedGemmaTransformer,
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

    /// Turbo mode: enable BOTH residency (mlock) and the int8-activation
    /// sdot/i8mm kernels at once. These two levers only pay off together —
    /// residency keeps the weights in RAM so the int8 kernels run at full
    /// speed instead of stalling on page faults (~10x steady-state decode in
    /// testing). Uses near-exact int8 activation (quant-only diff vs the exact
    /// scalar path, same approach as llama.cpp q8). Opt-in (implies --mlock's
    /// OOM caveat).
    #[arg(long, default_value_t = false)]
    fast: bool,

    /// Interactive multi-turn chat (REPL). Loads the model once and keeps the KV
    /// cache resident across turns, so each message only prefills the new text.
    /// `--ctx` sets the conversation cap (default 2048). Commands: /reset /exit.
    #[arg(short, long, default_value_t = false)]
    interactive: bool,
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

/// Build the token ids for one user turn in Gemma's chat template. The first
/// turn keeps the BOS that `encode` prepends; later turns drop it and prepend the
/// `<end_of_turn>` that closes the previous (still-uncached) model reply.
fn build_user_turn(
    tokenizer: &RllmTokenizer,
    msg: &str,
    is_first: bool,
    bos: Option<u64>,
) -> Result<Vec<usize>> {
    let text = if is_first {
        format!("<start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n")
    } else {
        format!("<end_of_turn>\n<start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n")
    };
    let mut ids = tokenizer.encode(&text)?;
    if !is_first {
        if let Some(b) = bos {
            if ids.first() == Some(&(b as usize)) {
                ids.remove(0);
            }
        }
    }
    Ok(ids)
}

/// Interactive multi-turn chat REPL over a resident `GemmaChatSession`.
fn run_interactive(
    model: &mut LazyRllmModel,
    prepared: &PreparedGemmaTransformer,
    tokenizer: &RllmTokenizer,
    stop_token_ids: &[usize],
    bos: Option<u64>,
    max_context: usize,
) -> Result<()> {
    const PER_TURN_MAX: usize = 512;
    let mut budget = MemoryBudget::unbounded();
    let mut session = GemmaChatSession::new(model, prepared, &mut budget, max_context)?;
    let stdout_is_tty = io::stdout().is_terminal();

    println!("\nRLLM Gemma chat — model loaded, KV cache resident (cap {max_context} tokens).");
    println!("Type a message. Commands: /reset (new conversation), /exit.\n");

    let stdin = io::stdin();
    let mut handle = stdin.lock();
    loop {
        print!("you> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if handle.read_line(&mut line)? == 0 {
            println!();
            break; // EOF (Ctrl-D)
        }
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }
        match msg {
            "/exit" | "/quit" => break,
            "/help" => {
                println!("commands: /reset (new conversation), /exit");
                continue;
            }
            "/reset" => {
                session.reset(prepared)?;
                println!("[new conversation]");
                continue;
            }
            _ => {}
        }

        let is_first = session.total_tokens() == 0;
        let turn_tokens = match build_user_turn(tokenizer, msg, is_first, bos) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("encode error: {e}");
                continue;
            }
        };

        print!("bot> ");
        io::stdout().flush().ok();
        let started = Instant::now();
        let mut acc = String::new();
        let mut on_token = |token: usize| -> bool {
            let piece = tokenizer.decode(&[token]).unwrap_or_default();
            acc.push_str(&piece);
            if stdout_is_tty {
                print!("\rbot> {acc}");
            } else {
                print!("{piece}");
            }
            io::stdout().flush().ok();
            true
        };
        let result = session.feed_and_decode(
            model,
            prepared,
            &mut budget,
            &turn_tokens,
            PER_TURN_MAX,
            stop_token_ids,
            &mut on_token,
        );
        println!();
        match result {
            Ok(gen) => {
                let secs = started.elapsed().as_secs_f64();
                eprintln!(
                    "  [{} tok, {:.1} tok/s, ctx {}/{}]",
                    gen.len(),
                    gen.len() as f64 / secs.max(1e-6),
                    session.total_tokens(),
                    max_context
                );
            }
            Err(e) => eprintln!("  [{e}] — use /reset to start over"),
        }
    }
    println!("bye!");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.ctx == 0 || args.max_new_tokens == 0 {
        anyhow::bail!("--ctx and --max-new-tokens must be greater than zero");
    }
    if !args.interactive && args.prompt.is_some() == args.token_ids.is_some() {
        anyhow::bail!("provide exactly one of --prompt or --token-ids");
    }

    // Residency + turbo. --fast bundles both levers (residency + int8 kernels),
    // which only pay off together. --mlock enables residency alone. The reader
    // and kernels read these via RLLM_MLOCK / RLLM_Q8_ACTIVATION (the env-gated
    // knobs other runtime experiments use), so translate the flags before
    // opening the model. Externally-set env vars still work without the flags.
    if args.fast {
        std::env::set_var("RLLM_MLOCK", "1");
        std::env::set_var("RLLM_Q8_ACTIVATION", "1");
        eprintln!(
            "[gemma-test] --fast: residency (mlock) + int8-activation kernels (near-exact, quant-only diff)"
        );
    } else if args.mlock {
        std::env::set_var("RLLM_MLOCK", "1");
        eprintln!("[gemma-test] --mlock: pinning model in RAM (mlock)");
    }

    let mut model = LazyRllmModel::open(&args.model)?;
    // Integrity mode defaults to VerifyOnce (SHA-256 each tensor's bytes once
    // per session). RLLM_INTEGRITY={unchecked,verifyonce,strict} overrides it —
    // a diagnostic/operational knob to measure or skip the verification cost.
    let integrity_mode = match std::env::var("RLLM_INTEGRITY").ok().as_deref() {
        Some("unchecked") => RamaIntegrityMode::Unchecked,
        Some("strict") => RamaIntegrityMode::Strict,
        Some("verifyonce") | None => RamaIntegrityMode::VerifyOnce,
        Some(other) => anyhow::bail!(
            "invalid RLLM_INTEGRITY={other:?} (expected unchecked|verifyonce|strict)"
        ),
    };
    model.set_rama_integrity_mode(integrity_mode);

    // Front-load the per-chunk SHA-256 integrity pass across cores. In VerifyOnce
    // this moves the multi-second verification out of the first prefill (where it
    // would run serially inline) into a brief parallel startup step, and lets the
    // decode fast-path skip its whole-tensor hash. No-op for Strict/Unchecked.
    let prewarm_start = Instant::now();
    let verified_chunks = model.prewarm_chunk_integrity()?;
    if verified_chunks > 0 {
        eprintln!(
            "[gemma-test] integrity prewarm: verified {verified_chunks} chunks in {:.2}s",
            prewarm_start.elapsed().as_secs_f64()
        );
    }

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

    let prompt_token_ids = if args.interactive {
        Vec::new()
    } else if let Some(prompt) = args.prompt.as_deref() {
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

    if args.interactive {
        let tokenizer = tokenizer
            .as_ref()
            .context("interactive mode needs tokenizer metadata")?;
        let bos = model.metadata().tokenizer.as_ref().and_then(|m| m.bos_token_id);
        return run_interactive(&mut model, &prepared, tokenizer, &stop_token_ids, bos, args.ctx);
    }

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
    // On a real terminal, redraw the whole running line with `\r` so multi-token
    // glyphs render smoothly. When stdout is piped/captured the `\r` isn't
    // honored and the redraws pile up (looking like duplicated output), so there
    // just append each newly decoded piece once.
    let stdout_is_tty = io::stdout().is_terminal();
    let mut decoded_so_far = String::new();
    let mut on_token = |token: usize| -> bool {
        if stop_token_ids.contains(&token) {
            return false;
        }
        if let Some(tokenizer) = tokenizer.as_ref() {
            let piece = tokenizer.decode(&[token]).unwrap_or_default();
            decoded_so_far.push_str(&piece);
            if stdout_is_tty {
                print!("\r{decoded_so_far}");
            } else {
                print!("{piece}");
            }
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
