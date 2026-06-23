// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! `rllm chat` — interactive multi-turn REPL over any packed model, codec-agnostic.
//!
//! Dispatches on the packed architecture (gemma3 / llama) to a resident KV-cache
//! chat session. Works with ANY codec — rANS / bit-plane (lossless), q8 (lossy), or
//! raw bf16 — because the forward pass decodes per codec on access, so no codec is
//! special-cased here. Mode flags pick the rANS RAM/speed trade:
//!   * default       → decode-once (SPISSA_DECODE_RESIDENT): cache decoded weights for
//!                     bf16-class steady speed at a higher resident footprint.
//!   * `--low-ram`   → stream the embedding (SPISSA_STREAM_EMBEDDING): resident ≈ the
//!                     compressed size; slower (re-decode per token) but fits the
//!                     >RAM regime where the bf16 table would not.
//!   * `--fast`      → mlock: page-lock the model in RAM to avoid swap. The fast SIMD decode path
//!                     is already on by default for every dtype, so this only adds the page lock.

use anyhow::{Context, Result};
use std::io::{self, BufRead, Write};
use std::time::Instant;

use spissa_runtime::{
    models::gemma::{
        prepare_gemma_transformer_from_metadata, GemmaChatSession, GemmaGenerationConfig,
        PreparedGemmaTransformer,
    },
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata, LlamaRamaGenerationConfig,
        LlamaRamaSessionAdapter,
    },
    models::qwen::{
        prepare_qwen_transformer_from_metadata, QwenGenerationConfig, QwenSession, SamplingParams,
    },
    LazySpissaModel, MemoryBudget, RamaChatSession, RamaIntegrityMode, SpissaTokenizer,
    StreamingSamplingConfig,
};

use crate::chat_template::{render_interactive_user_turn, stop_token_ids, ChatTemplateKind};

/// Raw sampling flags from the CLI, mapped into a `SamplingParams` once the model opens.
#[derive(Clone, Copy)]
pub struct SamplingArgs {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl SamplingArgs {
    fn to_params(self) -> SamplingParams {
        SamplingParams {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            seed: self.seed,
        }
    }
    /// top-k / repeat-penalty are only wired into the Qwen chat path so far.
    fn has_qwen_only(self) -> bool {
        self.top_k > 0 || self.repeat_penalty != 1.0
    }
}

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
    sampling_args: SamplingArgs,
) -> Result<()> {
    if ctx == 0 || max_new_tokens == 0 {
        anyhow::bail!("--ctx and --max-new-tokens must be greater than zero");
    }
    let params = sampling_args.to_params();
    // Mode levers — set before the model opens and the kernels read them.
    if low_ram {
        std::env::set_var("SPISSA_STREAM_EMBEDDING", "1");
    } else {
        std::env::set_var("SPISSA_DECODE_RESIDENT", "1");
    }
    if fast {
        // The fast SIMD decode path is ON by default for EVERY dtype (q8/bf16/rANS/delta) — it is an
        // opt-out flag — so `--fast`'s only real effect is page-locking the model in RAM (mlock) to
        // avoid swap. (We intentionally do NOT set SPISSA_Q8_ACTIVATION=1: it is a no-op default-on,
        // and setting it here made `--fast` look q8-specific when it isn't.)
        std::env::set_var("SPISSA_MLOCK", "1");
    }

    let mut model = LazySpissaModel::open(model_path)?;
    let integrity = match std::env::var("SPISSA_INTEGRITY").ok().as_deref() {
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
        "spissa chat — {} | arch={architecture} | ctx={ctx} | mode={}",
        model.metadata().model_name,
        if low_ram { "low-ram (stream-embedding)" } else { "decode-once" },
    );
    if sampling_args.has_qwen_only() && !matches!(architecture.as_str(), "qwen3" | "qwen") {
        eprintln!(
            "[chat] note: --top-k / --repeat-penalty are wired for Qwen chat only; \
             ignored for arch={architecture} (temp/top-p still apply)"
        );
    }
    match architecture.as_str() {
        "gemma3" | "gemma" => gemma_chat(&mut model, ctx, max_new_tokens, params.to_streaming()),
        "llama" => llama_chat(
            &mut model,
            ctx,
            max_new_tokens,
            chat_template,
            system_prompt,
            params.to_streaming(),
        ),
        "qwen3" | "qwen" => qwen_chat(&mut model, ctx, max_new_tokens, params, system_prompt),
        // Phi-3 / Phi-4 reuse the LLaMA decode (split fused tensors + partial RoPE + LongRoPE short
        // factor are wired through it), with the Phi `<|user|>…<|assistant|>` chat template.
        "phi3" | "phi" => llama_chat(
            &mut model,
            ctx,
            max_new_tokens,
            "phi",
            system_prompt,
            params.to_streaming(),
        ),
        other => {
            anyhow::bail!(
                "rllm chat: unsupported architecture {other:?} (supported: gemma3, llama, qwen3, phi3)"
            )
        }
    }
}

/// Qwen3.5 chat REPL (ChatML) over a persistent KV / Gated-DeltaNet session: each turn
/// only prefills the NEW user message onto the existing context (no O(n²) re-prefill of
/// the whole conversation). Qwen3.5 is a reasoning model, so replies open with a
/// `<think>…</think>` block before the answer.
fn qwen_chat(
    model: &mut LazySpissaModel,
    ctx: usize,
    max_new_tokens: usize,
    params: SamplingParams,
    system_prompt: Option<&str>,
) -> Result<()> {
    let tok_meta = model
        .metadata()
        .tokenizer
        .clone()
        .context("model has no packed tokenizer metadata")?;
    let tokenizer = SpissaTokenizer::from_metadata(&tok_meta)?;
    let bos = tok_meta.bos_token_id;
    let mut stop: Vec<usize> = Vec::new();
    if let Some(e) = tok_meta.eos_token_id {
        stop.push(e as usize);
    }
    for t in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_id_for_raw_token(t) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
    }

    let prepared = prepare_qwen_transformer_from_metadata(
        model,
        QwenGenerationConfig {
            max_new_tokens,
            max_seq_len: Some(ctx),
            causal: true,
            sampling: params.to_streaming(),
        },
    )?;
    let mut budget = MemoryBudget::unbounded();
    let mut session = QwenSession::new(model, prepared)?;

    let mode = if params.temperature <= 0.0 {
        "greedy".to_string()
    } else {
        format!(
            "temp={} top_p={} top_k={}",
            params.temperature, params.top_p, params.top_k
        )
    };
    let rep = if params.repeat_penalty != 1.0 {
        format!(", repeat_penalty={}", params.repeat_penalty)
    } else {
        String::new()
    };
    if let Some(sys) = system_prompt {
        println!("[system prompt set: {sys:?}]");
    }
    println!("Qwen3.5 chat (KV-session, {mode}{rep}). Commands: /reset, /exit.\n");
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
                session.reset()?;
                println!("[new conversation]");
                continue;
            }
            _ => {}
        }

        // Only the NEW turn's tokens. The leading `<|im_end|>\n` (turns after the first)
        // closes the previous assistant reply, which the session leaves uncommitted. The
        // optional system block is emitted once, before the first user turn.
        let is_first = session.context_len() == 0;
        let turn = if is_first {
            let sys = match system_prompt {
                Some(s) => format!("<|im_start|>system\n{s}<|im_end|>\n"),
                None => String::new(),
            };
            format!("{sys}<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n")
        } else {
            format!("<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n")
        };
        let mut ids = tokenizer.encode(&turn)?;
        if !is_first {
            if let Some(b) = bos {
                if ids.first() == Some(&(b as usize)) {
                    ids.remove(0);
                }
            }
        }

        stream_qwen_turn(
            &mut session,
            model,
            &ids,
            max_new_tokens,
            params,
            &stop,
            &mut budget,
            &tokenizer,
        )?;
    }
    println!("bye!");
    Ok(())
}

/// Decode one Qwen turn: prefill `ids`, stream the reply suffix to stdout, then print the
/// `[N tokens, X tok/s, ctx Y]` line. Re-decodes the running reply each token so multi-token
/// glyphs render, printing only the new tail.
#[allow(clippy::too_many_arguments)]
fn stream_qwen_turn(
    session: &mut QwenSession,
    model: &mut LazySpissaModel,
    ids: &[usize],
    max_new_tokens: usize,
    params: SamplingParams,
    stop: &[usize],
    budget: &mut MemoryBudget,
    tokenizer: &SpissaTokenizer,
) -> Result<()> {
    print!("bot> ");
    io::stdout().flush().ok();
    let started = Instant::now();
    let mut shown = String::new();
    let mut acc: Vec<usize> = Vec::new();
    let generated = session.generate(
        model,
        ids,
        max_new_tokens,
        params,
        stop,
        budget,
        &mut |tok| {
            acc.push(tok);
            print_reply_suffix(tokenizer, &acc, &mut shown, true);
            true
        },
    )?;
    let dt = started.elapsed().as_secs_f64();
    println!(
        "\n[{} tokens, {:.2} tok/s, ctx {}]",
        generated.len(),
        generated.len() as f64 / dt.max(1e-9),
        session.context_len()
    );
    Ok(())
}

/// Drop Qwen reasoning from the visible stream: remove complete `<think>...</think>` blocks, and
/// while a `<think>` is still open (no `</think>` yet) hold back everything after it until the
/// closing tag arrives. Operates on decoded text, so it works even if `<think>` isn't one token.
fn strip_think_spans(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "<think>".len()..];
        match after.find("</think>") {
            Some(end) => rest = &after[end + "</think>".len()..],
            None => return out, // open think → suppress the tail until it closes
        }
    }
    out.push_str(rest);
    out
}

/// Print the streaming reply suffix: re-decode the whole reply each token (so multi-token glyphs
/// render) but print only the NEW tail. Trims a trailing U+FFFD so a multi-byte glyph (e.g. an
/// emoji) split across tokens is held back until its bytes complete instead of flashing a
/// replacement char. `strip_think` additionally hides Qwen `<think>…</think>` reasoning. Shared by
/// every REPL.
fn print_reply_suffix(
    tokenizer: &SpissaTokenizer,
    reply: &[usize],
    shown: &mut String,
    strip_think: bool,
) {
    let decoded = tokenizer.decode(reply).unwrap_or_default();
    let trimmed = decoded.trim_end_matches('\u{FFFD}');
    let full = if strip_think {
        strip_think_spans(trimmed)
    } else {
        trimmed.to_string()
    };
    let full = full.as_str();
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
    tokenizer: &SpissaTokenizer,
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

fn gemma_chat(
    model: &mut LazySpissaModel,
    ctx: usize,
    max_new_tokens: usize,
    sampling: StreamingSamplingConfig,
) -> Result<()> {
    let tokenizer = model
        .metadata()
        .tokenizer
        .clone()
        .map(|m| SpissaTokenizer::from_metadata(&m))
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
            sampling,
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
    model: &mut LazySpissaModel,
    prepared: &PreparedGemmaTransformer,
    budget: &mut MemoryBudget,
    tokenizer: &SpissaTokenizer,
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
        print_reply_suffix(tokenizer, &reply, &mut shown, false);
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
    model: &mut LazySpissaModel,
    ctx: usize,
    max_new_tokens: usize,
    chat_template: &str,
    system_prompt: Option<&str>,
    sampling: StreamingSamplingConfig,
) -> Result<()> {
    let tokenizer_meta = model
        .metadata()
        .tokenizer
        .as_ref()
        .context("model has no packed tokenizer metadata")?
        .clone();
    let tokenizer = SpissaTokenizer::from_metadata(&tokenizer_meta)?;
    let template: ChatTemplateKind = chat_template.parse()?;
    let stop = stop_token_ids(template, &tokenizer, tokenizer_meta.eos_token_id);

    let prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(
        model,
        LlamaRamaGenerationConfig {
            max_new_tokens,
            max_seq_len: Some(ctx),
            causal: true,
            sampling,
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
            print_reply_suffix(&tokenizer, &reply, &mut shown, false);
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

#[cfg(test)]
mod tests {
    use super::strip_think_spans;

    #[test]
    fn strip_think_spans_hides_reasoning() {
        // Complete block removed; the answer remains.
        assert_eq!(strip_think_spans("<think>reasoning here</think>The answer"), "The answer");
        // Block in the middle.
        assert_eq!(strip_think_spans("Sure!<think>x</think> done"), "Sure! done");
        // Streaming: an open <think> with no closing tag yet → suppress the tail.
        assert_eq!(strip_think_spans("ok <think>still going"), "ok ");
        // No think + an emoji → byte-for-byte unchanged.
        assert_eq!(strip_think_spans("just text 😀✨"), "just text 😀✨");
    }
}
