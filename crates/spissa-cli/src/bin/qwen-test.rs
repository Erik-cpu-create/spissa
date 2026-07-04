// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

//! Standalone smoke/bring-up harness for the Qwen3.5 text adapter.
//!
//! Usage: qwen-test [model.spsa] [prompt] [max_new_tokens]
//! Loads a packed Qwen3.5 `.spsa`, encodes a ChatML prompt, runs greedy generation,
//! and streams the decoded reply + a tok/s line. This is a validation tool, not the
//! production chat path (that wiring lands in a later phase).

use anyhow::Context;
use spissa_runtime::models::qwen::{
    prepare_qwen_transformer_from_metadata, qwen_generate_from_model, QwenGenerationConfig,
};
use spissa_runtime::{LazySpissaModel, MemoryBudget, SpissaTokenizer, StreamingSamplingConfig};
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .unwrap_or_else(|| "models/qwen3.5-2b-textonly-raw.spsa".to_string());
    let prompt = args
        .next()
        .unwrap_or_else(|| "What is the capital of France? Answer in one word.".to_string());
    let max_new: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(48);

    eprintln!("[qwen-test] opening {model_path}");
    let mut model = LazySpissaModel::open(&model_path)?;
    model.set_rama_integrity_mode(spissa_runtime::RamaIntegrityMode::Unchecked);
    // Match the chat path's residency: prewarm decode-resident (SPISSA_DECODE_RESIDENT=1).
    let warmed = model.prewarm_decode_resident()?;
    if warmed > 0 {
        eprintln!("[qwen-test] prewarm: {warmed} chunks decoded resident");
    }
    let tok_meta = model
        .metadata()
        .tokenizer
        .clone()
        .context("model has no packed tokenizer metadata")?;
    let tokenizer = SpissaTokenizer::from_metadata(&tok_meta)?;
    let eos = tok_meta.eos_token_id.map(|e| e as usize);

    // "RAW:<text>" feeds the text verbatim (base-style continuation); otherwise wrap
    // in the Qwen ChatML user/assistant template.
    let text = if let Some(raw) = prompt.strip_prefix("RAW:") {
        raw.to_string()
    } else {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    };
    let ids = tokenizer.encode(&text)?;
    if std::env::var("QWEN_DEBUG").is_ok() {
        eprintln!(
            "[qwen-test] token 220 decodes to {:?}",
            tokenizer.decode(&[220])
        );
        eprintln!("[qwen-test] prompt ids = {ids:?}");
    }
    eprintln!(
        "[qwen-test] prompt = {prompt:?} -> {} tokens; generating up to {max_new} (greedy)",
        ids.len()
    );

    let prepared = prepare_qwen_transformer_from_metadata(
        &mut model,
        QwenGenerationConfig {
            max_new_tokens: max_new,
            max_seq_len: Some(4096),
            causal: true,
            sampling: StreamingSamplingConfig::Argmax,
        },
    )?;
    eprintln!(
        "[qwen-test] layers={} (full-attn at idx%4==3), hidden={}, heads={}q/{}kv hd={}, linear {}h k{}/v{} conv{}",
        prepared.layers.len(),
        prepared.config.hidden_size,
        prepared.config.num_heads,
        prepared.config.num_kv_heads,
        prepared.config.head_dim,
        prepared.config.linear_num_heads,
        prepared.config.linear_key_dim,
        prepared.config.linear_value_dim,
        prepared.config.conv_kernel,
    );

    if std::env::var("QWEN_PROFILE").is_ok() {
        spissa_runtime::models::qwen::generate::profile::enable();
        spissa_runtime::models::qwen::generate::profile::reset();
    }
    let mut budget = MemoryBudget::unbounded();
    let t0 = std::time::Instant::now();
    let mut first_token_at: Option<f64> = None;
    let mut generated: Vec<usize> = Vec::new();
    let mut shown = String::new();
    let mut count = 0usize;

    print!("assistant> ");
    std::io::stdout().flush().ok();
    qwen_generate_from_model(&mut model, &prepared, &ids, &mut budget, &mut |tok| {
        if first_token_at.is_none() {
            first_token_at = Some(t0.elapsed().as_secs_f64());
        }
        count += 1;
        if Some(tok) == eos {
            return false;
        }
        generated.push(tok);
        if let Ok(full) = tokenizer.decode(&generated) {
            if full.starts_with(&shown) && full.len() > shown.len() {
                print!("{}", &full[shown.len()..]);
                std::io::stdout().flush().ok();
                shown = full;
            }
        }
        true
    })?;

    let dt = t0.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "[qwen-test] {count} tokens in {dt:.2}s = {:.2} tok/s (prefill+first {:.2}s)",
        count as f64 / dt.max(1e-9),
        first_token_at.unwrap_or(dt)
    );
    if std::env::var("QWEN_PROFILE").is_ok() {
        eprintln!(
            "[qwen-test] profile (cumulative): {}",
            spissa_runtime::models::qwen::generate::profile::report()
        );
    }
    Ok(())
}
