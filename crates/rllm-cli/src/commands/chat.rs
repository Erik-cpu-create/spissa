//! `rllm chat` — interactive multi-turn REPL over any packed model, codec-agnostic.
//!
//! Dispatches on the packed architecture (gemma3 / llama) to a resident KV-cache
//! chat session. Works with ANY codec — rANS / bit-plane (lossless), q8 (lossy), or
//! raw bf16 — because the forward pass decodes per codec on access, so no codec is
//! special-cased here. Mode flags pick the rANS RAM/speed trade:
//!   * default       → decode-once (RLLM_DECODE_RESIDENT): cache decoded weights for
//!                     bf16-class steady speed at a higher resident footprint.
//!   * `--low-ram`   → stream the embedding (RLLM_STREAM_EMBEDDING): resident ≈ the
//!                     compressed size; slower (re-decode per token) but fits the
//!                     >RAM regime where the bf16 table would not.
//!   * `--fast`      → q8 turbo (mlock + int8-activation kernels) for q8 models.

use anyhow::{Context, Result};
use std::io::{self, BufRead, Write};
use std::time::Instant;

use rllm_runtime::{
    models::gemma::{
        prepare_gemma_transformer_from_metadata, GemmaChatSession, GemmaGenerationConfig,
        PreparedGemmaTransformer,
    },
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata, LlamaRamaGenerationConfig,
        LlamaRamaSessionAdapter,
    },
    LazyRllmModel, MemoryBudget, RamaChatSession, RamaIntegrityMode, RllmTokenizer,
    StreamingSamplingConfig,
};

use crate::chat_template::{render_interactive_user_turn, stop_token_ids, ChatTemplateKind};

/// Entry point for `rllm chat`.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model_path: &str,
    ctx: usize,
    max_new_tokens: usize,
    low_ram: bool,
    fast: bool,
    chat_template: &str,
    system_prompt: Option<&str>,
) -> Result<()> {
    if ctx == 0 || max_new_tokens == 0 {
        anyhow::bail!("--ctx and --max-new-tokens must be greater than zero");
    }
    // Mode levers — set before the model opens and the kernels read them.
    if low_ram {
        std::env::set_var("RLLM_STREAM_EMBEDDING", "1");
    } else {
        std::env::set_var("RLLM_DECODE_RESIDENT", "1");
    }
    if fast {
        std::env::set_var("RLLM_MLOCK", "1");
        std::env::set_var("RLLM_Q8_ACTIVATION", "1");
    }

    let mut model = LazyRllmModel::open(model_path)?;
    let integrity = match std::env::var("RLLM_INTEGRITY").ok().as_deref() {
        Some("unchecked") => RamaIntegrityMode::Unchecked,
        Some("strict") => RamaIntegrityMode::Strict,
        _ => RamaIntegrityMode::VerifyOnce,
    };
    model.set_rama_integrity_mode(integrity);
    let verified = model.prewarm_chunk_integrity()?;
    if verified > 0 {
        eprintln!("[chat] integrity prewarm: verified {verified} chunks");
    }
    // R174: decode-once in parallel at load (not serially on the first token).
    let predecoded = model.prewarm_decode_resident()?;
    if predecoded > 0 {
        eprintln!("[chat] decode prewarm: {predecoded} chunks decoded in parallel");
    }

    let architecture = model.metadata().architecture.clone();
    println!(
        "RLLM chat — {} | arch={architecture} | ctx={ctx} | mode={}",
        model.metadata().model_name,
        if low_ram { "low-ram (stream-embedding)" } else { "decode-once" },
    );
    match architecture.as_str() {
        "gemma3" | "gemma" => gemma_chat(&mut model, ctx, max_new_tokens),
        "llama" => llama_chat(&mut model, ctx, max_new_tokens, chat_template, system_prompt),
        other => {
            anyhow::bail!("rllm chat: unsupported architecture {other:?} (supported: gemma3, llama)")
        }
    }
}

/// Print the streaming reply suffix: re-decode the whole reply each token (so
/// multi-token glyphs render) but print only the NEW tail. Shared by both REPLs.
fn print_reply_suffix(tokenizer: &RllmTokenizer, reply: &[usize], shown: &mut String) {
    let full = tokenizer.decode(reply).unwrap_or_default();
    let full = full.trim_end_matches('\u{FFFD}');
    if let Some(rest) = full.strip_prefix(shown.as_str()) {
        print!("{rest}");
    } else {
        print!("\rbot> {full}");
    }
    *shown = full.to_string();
    io::stdout().flush().ok();
}

/// Build one user turn in Gemma's chat template (first turn keeps the BOS that
/// `encode` prepends; later turns drop it and close the prior model reply).
fn gemma_user_turn(
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

fn gemma_chat(model: &mut LazyRllmModel, ctx: usize, max_new_tokens: usize) -> Result<()> {
    let tokenizer = model
        .metadata()
        .tokenizer
        .clone()
        .map(|m| RllmTokenizer::from_metadata(&m))
        .transpose()?
        .context("model has no packed tokenizer metadata")?;
    let bos = model.metadata().tokenizer.as_ref().and_then(|m| m.bos_token_id);
    let eos = model.metadata().tokenizer.as_ref().and_then(|m| m.eos_token_id);

    // Gemma IT ends a turn with <end_of_turn> as well as <eos>; treat both as stops.
    let mut stop: Vec<usize> = Vec::new();
    if let Some(e) = eos {
        stop.push(e as usize);
    }
    if let Some(eot) = tokenizer.token_id_for_raw_token("<end_of_turn>") {
        if !stop.contains(&eot) {
            stop.push(eot);
        }
    }

    let prepared = prepare_gemma_transformer_from_metadata(
        model,
        GemmaGenerationConfig {
            max_new_tokens,
            max_seq_len: Some(ctx),
            causal: true,
            sampling: StreamingSamplingConfig::Argmax,
        },
    )?;
    let mut budget = MemoryBudget::unbounded();
    let mut session = GemmaChatSession::new(model, &prepared, &mut budget, ctx)?;

    println!("KV cache resident (cap {ctx}). Commands: /reset, /exit.\n");
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    loop {
        print!("you> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if handle.read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }
        match msg {
            "/exit" | "/quit" | "exit" | "quit" => break,
            "/reset" => {
                session.reset(&prepared)?;
                println!("[new conversation]");
                continue;
            }
            _ => {}
        }

        let is_first = session.total_tokens() == 0;
        let turn = gemma_user_turn(&tokenizer, msg, is_first, bos)?;
        gemma_run_turn(
            &mut session,
            model,
            &prepared,
            &mut budget,
            &tokenizer,
            &turn,
            max_new_tokens,
            &stop,
            ctx,
        )?;
    }
    println!("bye!");
    Ok(())
}

/// Decode + stream one Gemma reply turn, printing the running suffix and a
/// per-turn summary. Flags an empty reply (the model emitted a stop token first —
/// common for tiny lossy-quantized models in long, low-content conversations).
#[allow(clippy::too_many_arguments)]
fn gemma_run_turn(
    session: &mut GemmaChatSession,
    model: &mut LazyRllmModel,
    prepared: &PreparedGemmaTransformer,
    budget: &mut MemoryBudget,
    tokenizer: &RllmTokenizer,
    turn: &[usize],
    max_new_tokens: usize,
    stop: &[usize],
    ctx: usize,
) -> Result<()> {
    print!("bot> ");
    io::stdout().flush().ok();
    let started = Instant::now();
    let mut reply: Vec<usize> = Vec::new();
    let mut shown = String::new();
    let mut on_token = |token: usize| -> bool {
        reply.push(token);
        print_reply_suffix(tokenizer, &reply, &mut shown);
        true
    };
    let result =
        session.feed_and_decode(model, prepared, budget, turn, max_new_tokens, stop, &mut on_token);
    println!();
    match result {
        Ok(gen) => {
            if gen.is_empty() {
                println!("[the model ended the turn with no output — try a fuller prompt or /reset]");
            }
            eprintln!(
                "  [{} tok, {:.1} tok/s, ctx {}/{ctx}]",
                gen.len(),
                gen.len() as f64 / started.elapsed().as_secs_f64().max(1e-6),
                session.total_tokens(),
            );
        }
        Err(e) => eprintln!("  [{e}] — /reset to recover"),
    }
    Ok(())
}

fn llama_chat(
    model: &mut LazyRllmModel,
    ctx: usize,
    max_new_tokens: usize,
    chat_template: &str,
    system_prompt: Option<&str>,
) -> Result<()> {
    let tokenizer_meta = model
        .metadata()
        .tokenizer
        .as_ref()
        .context("model has no packed tokenizer metadata")?
        .clone();
    let tokenizer = RllmTokenizer::from_metadata(&tokenizer_meta)?;
    let template: ChatTemplateKind = chat_template.parse()?;
    let stop = stop_token_ids(template, &tokenizer, tokenizer_meta.eos_token_id);

    let prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(
        model,
        LlamaRamaGenerationConfig {
            max_new_tokens,
            max_seq_len: Some(ctx),
            causal: true,
            sampling: StreamingSamplingConfig::Argmax,
        },
    )?;
    let mut budget = MemoryBudget::unbounded();
    let adapter = LlamaRamaSessionAdapter::new(model, &prepared, &mut budget)?;
    let mut session = RamaChatSession::new(adapter);

    println!("Chat template: {chat_template}. Commands: exit / quit.\n");
    let mut has_context = false;
    let mut previous_assistant_ended = true;
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    loop {
        print!("you> ");
        io::stdout().flush().ok();
        let mut input = String::new();
        if handle.read_line(&mut input)? == 0 {
            println!();
            break;
        }
        let text = input.trim();
        if text.is_empty() {
            continue;
        }
        if matches!(text, "exit" | "quit" | "/exit" | "/quit") {
            break;
        }
        let turn_text = render_interactive_user_turn(
            template,
            has_context,
            previous_assistant_ended,
            system_prompt,
            text,
        );
        let input_tokens = tokenizer.encode(&turn_text)?;

        print!("bot> ");
        io::stdout().flush().ok();
        let mut reply: Vec<usize> = Vec::new();
        let mut shown = String::new();
        let mut assistant_ended = false;
        let mut on_token = |token: usize| -> bool {
            if stop.contains(&token) {
                assistant_ended = true;
                return false;
            }
            reply.push(token);
            print_reply_suffix(&tokenizer, &reply, &mut shown);
            true
        };
        let result = session.generate_turn(&input_tokens, max_new_tokens, &mut budget, &mut on_token);
        println!();
        match result {
            Ok(r) => {
                has_context = true;
                previous_assistant_ended = assistant_ended;
                if r.metrics.generated_tokens == 0 {
                    println!("[the model ended the turn with no output — try a fuller prompt]");
                }
                eprintln!(
                    "  [{} tok, {:.1} tok/s, ctx {}]",
                    r.metrics.generated_tokens,
                    r.metrics.decode_tok_s,
                    session.token_history().len(),
                );
            }
            Err(e) => eprintln!("  [{e}]"),
        }
    }
    println!("bye!");
    Ok(())
}
