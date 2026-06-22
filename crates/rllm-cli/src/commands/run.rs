// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use crate::commands::common::parse_size;
use anyhow::{Context, Result};
use rllm_runtime::{
    build_runtime_plan, prepare_gpt_neox_rama_layer_decode_transformer_from_metadata,
    recommend_rama_prefill_chunk_tokens, FullDecodeModel, GptNeoxRamaGenerationConfig,
    GptNeoxRamaGenerationOptions, LazyRllmModel, MemoryBudget, PlanStatus, RamaGenerationTiming,
    RamaIntegrityMode, RamaPrefillPolicy, RamaTrace, RllmTokenizer, RuntimeMode, RuntimePlanConfig,
    StreamingEchoTransformerConfig, StreamingSamplingConfig,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::str::FromStr;

pub fn run(
    file: &str,
    mode: &str,
    ctx: usize,
    memory_budget: Option<&str>,
    dry_run: bool,
    prompt: Option<&str>,
    token_ids: Option<&str>,
    max_new_tokens: usize,
    logits_out: Option<&str>,
    rama_trace: Option<&str>,
    rama_timing: Option<&str>,
    rama_prefill_chunk_tokens: Option<usize>,
    rama_prefill_policy: &str,
    no_rama_prefill_chunking: bool,
    rama_integrity: &str,
) -> Result<()> {
    let path = Path::new(file);
    if !path.exists() {
        anyhow::bail!("File does not exist: {}", file);
    }
    if ctx == 0 {
        anyhow::bail!("--ctx must be greater than zero");
    }

    let runtime_mode = RuntimeMode::from_str(mode)?;
    let memory_budget_bytes = memory_budget
        .map(parse_size)
        .transpose()
        .context("failed to parse --memory-budget")?;
    let rama_prefill_policy = parse_rama_prefill_policy(rama_prefill_policy)?;

    if prompt.is_some() || token_ids.is_some() {
        if dry_run {
            anyhow::bail!("--prompt/--token-ids cannot be combined with --dry-run");
        }
        return run_generation(
            path,
            file,
            ctx,
            max_new_tokens,
            memory_budget_bytes,
            prompt,
            token_ids,
            logits_out,
            rama_trace,
            rama_timing,
            rama_prefill_chunk_tokens,
            rama_prefill_policy,
            no_rama_prefill_chunking,
            rama_integrity,
        );
    }

    if logits_out.is_some() {
        anyhow::bail!("--logits-out requires --prompt or --token-ids");
    }
    if rama_trace.is_some() {
        anyhow::bail!("--rama-trace requires --prompt or --token-ids");
    }
    if rama_timing.is_some() {
        anyhow::bail!("--rama-timing requires --prompt or --token-ids");
    }
    if rama_prefill_chunk_tokens.is_some() {
        anyhow::bail!("--rama-prefill-chunk-tokens requires --prompt or --token-ids");
    }
    if rama_prefill_policy != RamaPrefillPolicy::LowRam {
        anyhow::bail!("--rama-prefill-policy requires --prompt or --token-ids");
    }
    if no_rama_prefill_chunking {
        anyhow::bail!("--no-rama-prefill-chunking requires --prompt or --token-ids");
    }
    if rama_integrity != "strict" {
        anyhow::bail!("--rama-integrity requires --prompt or --token-ids");
    }

    match runtime_mode {
        RuntimeMode::FullDecode if !dry_run && memory_budget_bytes.is_none() => {
            run_full_decode(path, file)
        }
        _ => run_planned(path, file, runtime_mode, ctx, memory_budget_bytes, dry_run),
    }
}

fn run_generation(
    path: &Path,
    file: &str,
    ctx: usize,
    max_new_tokens: usize,
    memory_budget_bytes: Option<usize>,
    prompt: Option<&str>,
    token_ids: Option<&str>,
    logits_out: Option<&str>,
    rama_trace: Option<&str>,
    rama_timing: Option<&str>,
    rama_prefill_chunk_tokens: Option<usize>,
    rama_prefill_policy: RamaPrefillPolicy,
    no_rama_prefill_chunking: bool,
    rama_integrity: &str,
) -> Result<()> {
    if max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    if prompt.is_some() == token_ids.is_some() {
        anyhow::bail!("provide exactly one of --prompt or --token-ids");
    }
    if no_rama_prefill_chunking && rama_prefill_chunk_tokens.is_some() {
        anyhow::bail!(
            "--no-rama-prefill-chunking cannot be combined with --rama-prefill-chunk-tokens"
        );
    }
    println!("spissa runtime — Phase 7 tiled RAMA layer-decode text generation");
    println!("====================================================\n");
    println!("Loading metadata and streamed tensors: {}", file);

    let mut model =
        LazyRllmModel::open(path).with_context(|| format!("failed to open {}", file))?;
    let rama_integrity = parse_rama_integrity_mode(rama_integrity)?;
    model.set_rama_integrity_mode(rama_integrity);
    if rama_trace.is_some() {
        model.enable_rama_trace();
    }
    let tokenizer = model
        .metadata()
        .tokenizer
        .clone()
        .map(|metadata| RllmTokenizer::from_metadata(&metadata))
        .transpose()?;
    let prompt_token_ids = if let Some(prompt) = prompt {
        let tokenizer = tokenizer.as_ref().ok_or_else(|| anyhow::anyhow!("model metadata does not include tokenizer metadata; repack with --tokenizer <tokenizer.json> or a sibling tokenizer.json"))?;
        tokenizer.encode(prompt)?
    } else {
        parse_token_ids(token_ids.expect("checked above"))?
    };
    let mut prepared = prepare_gpt_neox_rama_layer_decode_transformer_from_metadata(
        &mut model,
        GptNeoxRamaGenerationConfig {
            max_new_tokens,
            max_seq_len: Some(ctx),
            causal: true,
            sampling: StreamingSamplingConfig::Argmax,
        },
    )?;
    let mut budget = memory_budget_bytes
        .map(MemoryBudget::new)
        .unwrap_or_else(MemoryBudget::unbounded);
    prepared.pin_lm_head(&mut model, &mut budget);
    let effective_prefill_chunk_tokens = effective_rama_prefill_chunk_tokens(
        rama_prefill_chunk_tokens,
        no_rama_prefill_chunking,
        rama_prefill_policy,
        prompt_token_ids.len(),
        prepared.config,
        memory_budget_bytes,
    )?;
    match effective_prefill_chunk_tokens {
        Some(tokens) if rama_prefill_chunk_tokens.is_some() => {
            println!("RAMA prefill window: {tokens} token(s) (fixed override)");
        }
        Some(tokens) => {
            println!(
                "RAMA prefill window: {tokens} token(s) (auto {} policy)",
                rama_prefill_policy.as_str()
            );
        }
        None => println!("RAMA prefill window: disabled; full prompt prefill"),
    }
    let result = prepared.generate_from_model_with_options(
        &mut model,
        &prompt_token_ids,
        &mut budget,
        GptNeoxRamaGenerationOptions {
            timing: rama_timing.is_some(),
            prefill_chunk_tokens: effective_prefill_chunk_tokens,
            collect_logits: logits_out.is_some(),
        },
    )?;
    let generated_text = tokenizer
        .as_ref()
        .map(|tokenizer| tokenizer.decode(&result.generated_token_ids))
        .transpose()?;
    let full_text = tokenizer
        .as_ref()
        .map(|tokenizer| tokenizer.decode(&result.token_ids))
        .transpose()?;

    println!("Model: {}", model.metadata().model_name);
    println!("Architecture: {}", model.metadata().architecture);
    if let Some(tokenizer) = &tokenizer {
        println!("Tokenizer vocab size: {}", tokenizer.vocab_size());
    }
    if let Some(prompt) = prompt {
        println!("Prompt: {}", prompt);
    }
    println!("Prompt token IDs: {:?}", prompt_token_ids);
    println!("Generated token IDs: {:?}", result.generated_token_ids);
    if let Some(generated_text) = &generated_text {
        println!("Generated text: {}", generated_text);
    }
    if let Some(full_text) = &full_text {
        println!("Full text: {}", full_text);
    }
    println!(
        "Resident non-layer params: {}",
        format_bytes(prepared.resident_parameter_bytes)
    );
    println!(
        "Max active layer params: {}",
        format_bytes(prepared.max_layer_parameter_bytes)
    );
    println!(
        "Context memory bytes: {}",
        format_bytes(result.context_memory_bytes())
    );
    println!(
        "Peak transient budget: {}",
        format_bytes(budget.peak_bytes())
    );
    println!(
        "Current transient budget: {}",
        format_bytes(budget.current_bytes())
    );

    if let Some(logits_out) = logits_out {
        write_logits_json(
            logits_out,
            &prompt_token_ids,
            &result.generated_token_ids,
            result
                .step_logits
                .first()
                .ok_or_else(|| anyhow::anyhow!("generation produced no logits"))?,
        )?;
        println!("Logits JSON: {}", logits_out);
    }
    if let Some(rama_trace_out) = rama_trace {
        let trace = model
            .take_rama_trace()
            .ok_or_else(|| anyhow::anyhow!("RAMA trace was requested but not initialized"))?;
        write_rama_trace_json(rama_trace_out, &trace)?;
        println!("RAMA trace JSON: {}", rama_trace_out);
    }
    if let Some(rama_timing_out) = rama_timing {
        let timing = result
            .timing()
            .ok_or_else(|| anyhow::anyhow!("RAMA timing was requested but not collected"))?;
        write_rama_timing_json(rama_timing_out, timing)?;
        println!("RAMA timing JSON: {}", rama_timing_out);
    }
    println!("\n[phase-7.12C] Tokenizer-backed tiled RAMA generation completed.");

    Ok(())
}

fn parse_rama_integrity_mode(raw: &str) -> Result<RamaIntegrityMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "strict" => Ok(RamaIntegrityMode::Strict),
        "verify-once" | "verify_once" | "once" => Ok(RamaIntegrityMode::VerifyOnce),
        other => {
            anyhow::bail!("unsupported --rama-integrity {other:?}; expected strict or verify-once")
        }
    }
}

fn parse_rama_prefill_policy(raw: &str) -> Result<RamaPrefillPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "low-ram" | "low_ram" | "lowram" => Ok(RamaPrefillPolicy::LowRam),
        "speed" | "fast" => Ok(RamaPrefillPolicy::Speed),
        other => {
            anyhow::bail!("unsupported --rama-prefill-policy {other:?}; expected low-ram or speed")
        }
    }
}

fn effective_rama_prefill_chunk_tokens(
    requested: Option<usize>,
    disabled: bool,
    policy: RamaPrefillPolicy,
    prompt_len: usize,
    config: StreamingEchoTransformerConfig,
    memory_budget_bytes: Option<usize>,
) -> Result<Option<usize>> {
    if disabled {
        Ok(None)
    } else if let Some(requested) = requested {
        Ok(Some(requested))
    } else {
        Ok(Some(recommend_rama_prefill_chunk_tokens(
            config,
            policy,
            prompt_len,
            memory_budget_bytes,
        )?))
    }
}

fn parse_token_ids(raw: &str) -> Result<Vec<usize>> {
    let mut ids = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        ids.push(
            part.parse::<usize>()
                .with_context(|| format!("invalid token id: {part}"))?,
        );
    }
    if ids.is_empty() {
        anyhow::bail!("--token-ids must contain at least one token id");
    }
    Ok(ids)
}

fn write_logits_json(
    path: &str,
    prompt_token_ids: &[usize],
    generated_token_ids: &[usize],
    first_step_logits: &[f32],
) -> Result<()> {
    let output = Path::new(path);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create logits output directory {}",
                parent.display()
            )
        })?;
    }
    let payload = json!({
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "logit_step_index": 0,
        "logits": first_step_logits,
    });
    fs::write(output, serde_json::to_vec(&payload)?)
        .with_context(|| format!("failed to write logits JSON to {}", output.display()))?;
    Ok(())
}

fn write_rama_trace_json(path: &str, trace: &RamaTrace) -> Result<()> {
    let output = Path::new(path);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create RAMA trace output directory {}",
                parent.display()
            )
        })?;
    }

    let mut phase_totals: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for event in &trace.events {
        let entry = phase_totals.entry(event.phase.as_str()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(event.duration_ns);
    }
    let duration_by_phase: Vec<_> = phase_totals
        .into_iter()
        .map(|(phase, (event_count, total_ns))| {
            json!({
                "phase": phase,
                "event_count": event_count,
                "total_ns": total_ns,
                "total_ms": (total_ns as f64) / 1_000_000.0,
            })
        })
        .collect();
    let total_ns = trace
        .events
        .iter()
        .fold(0u64, |acc, event| acc.saturating_add(event.duration_ns));
    let payload = json!({
        "trace": trace,
        "summary": {
            "event_count": trace.events.len(),
            "total_recorded_ns": total_ns,
            "total_recorded_ms": (total_ns as f64) / 1_000_000.0,
            "duration_by_phase": duration_by_phase,
        }
    });
    fs::write(output, serde_json::to_vec_pretty(&payload)?)
        .with_context(|| format!("failed to write RAMA trace JSON to {}", output.display()))?;
    Ok(())
}

fn write_rama_timing_json(path: &str, timing: &RamaGenerationTiming) -> Result<()> {
    let output = Path::new(path);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create RAMA timing output directory {}",
                parent.display()
            )
        })?;
    }
    let payload = json!({
        "timing": timing,
        "summary": {
            "prefill_ms": (timing.prefill_ns as f64) / 1_000_000.0,
            "decode_ms": (timing.decode_ns as f64) / 1_000_000.0,
            "final_norm_ms": (timing.final_norm_ns as f64) / 1_000_000.0,
            "lm_head_ms": (timing.lm_head_ns as f64) / 1_000_000.0,
            "sampling_ms": (timing.sampling_ns as f64) / 1_000_000.0,
            "prefill_embedding_ms": (timing.prefill_embedding_ns as f64) / 1_000_000.0,
            "prefill_layer_params_ms": (timing.prefill_layer_params_ns as f64) / 1_000_000.0,
            "prefill_attention_norm_ms": (timing.prefill_attention_norm_ns as f64) / 1_000_000.0,
            "prefill_attention_ms": (timing.prefill_attention_ns as f64) / 1_000_000.0,
            "prefill_attention_qkv_projection_ms": (timing.prefill_attention_qkv_projection_ns as f64) / 1_000_000.0,
            "prefill_attention_qkv_split_ms": (timing.prefill_attention_qkv_split_ns as f64) / 1_000_000.0,
            "prefill_attention_rotary_ms": (timing.prefill_attention_rotary_ns as f64) / 1_000_000.0,
            "prefill_attention_score_context_ms": (timing.prefill_attention_score_context_ns as f64) / 1_000_000.0,
            "prefill_attention_output_projection_ms": (timing.prefill_attention_output_projection_ns as f64) / 1_000_000.0,
            "prefill_attention_kv_append_ms": (timing.prefill_attention_kv_append_ns as f64) / 1_000_000.0,
            "prefill_attention_residual_ms": (timing.prefill_attention_residual_ns as f64) / 1_000_000.0,
            "prefill_mlp_norm_ms": (timing.prefill_mlp_norm_ns as f64) / 1_000_000.0,
            "prefill_mlp_ms": (timing.prefill_mlp_ns as f64) / 1_000_000.0,
            "prefill_mlp_input_projection_ms": (timing.prefill_mlp_input_projection_ns as f64) / 1_000_000.0,
            "prefill_mlp_activation_ms": (timing.prefill_mlp_activation_ns as f64) / 1_000_000.0,
            "prefill_mlp_output_projection_ms": (timing.prefill_mlp_output_projection_ns as f64) / 1_000_000.0,
            "prefill_mlp_residual_ms": (timing.prefill_mlp_residual_ns as f64) / 1_000_000.0,
            "prefill_timed_blocks": timing.prefill_timed_blocks,
            "prefill_chunks": timing.prefill_chunks,
            "decode_steps": timing.decode_steps,
            "max_prefill_chunk_tokens": timing.max_prefill_chunk_tokens,
        }
    });
    fs::write(output, serde_json::to_vec_pretty(&payload)?)
        .with_context(|| format!("failed to write RAMA timing JSON to {}", output.display()))?;
    Ok(())
}

fn run_full_decode(path: &Path, file: &str) -> Result<()> {
    println!("spissa runtime — Phase 5A full-decode loader");
    println!("==========================================\n");
    println!("Loading: {}", file);
    println!("Mode: full-decode\n");

    let model =
        FullDecodeModel::load(path).with_context(|| format!("failed to full-decode {}", file))?;

    println!("[OK] Loaded model: {}", model.metadata.model_name);
    println!("Architecture: {}", model.metadata.architecture);
    println!("Source format: {}", model.metadata.source_format);
    println!("Tensors: {}", model.stats.tensor_count);
    println!(
        "Original tensor bytes: {} ({})",
        model.stats.total_original_bytes,
        format_bytes(model.stats.total_original_bytes as usize)
    );
    println!(
        "Runtime f32 bytes: {} ({})",
        model.stats.total_runtime_bytes,
        format_bytes(model.stats.total_runtime_bytes)
    );

    let mut names = model.tensor_names();
    let shown = names.len().min(12);
    if shown > 0 {
        println!("\nFirst {} tensors:", shown);
        for name in names.drain(..shown) {
            let tensor = model.get(name)?;
            println!(
                "  - {} {:?} {:?} ({} values)",
                tensor.name,
                tensor.dtype,
                tensor.shape,
                tensor.element_count()
            );
        }
    }

    println!("\n[phase-5a] Full-decode runtime load succeeded.");
    println!("[phase-5b] For low-RAM planning, use --mode tile-stream --memory-budget <size>.");

    Ok(())
}

fn run_planned(
    path: &Path,
    file: &str,
    mode: RuntimeMode,
    ctx: usize,
    memory_budget_bytes: Option<usize>,
    dry_run: bool,
) -> Result<()> {
    println!("spissa runtime — Phase 5B low-memory planner");
    println!("===========================================\n");
    println!("Loading metadata only: {}", file);

    let model = LazyRllmModel::open(path).with_context(|| format!("failed to open {}", file))?;
    let plan = build_runtime_plan(
        &model,
        RuntimePlanConfig {
            mode,
            context_length: ctx,
            memory_budget_bytes,
        },
    )?;

    println!("\nModel: {}", plan.model_name);
    println!("Architecture: {}", plan.architecture);
    println!(
        "Compressed size: {} ({})",
        plan.file_size_bytes,
        format_bytes(plan.file_size_bytes as usize)
    );
    println!("Runtime mode: {}", plan.mode.as_str());
    println!("Context length: {}", plan.context_length);
    match plan.memory_budget_bytes {
        Some(bytes) => println!("Memory budget: {}", format_bytes(bytes)),
        None => println!("Memory budget: unlimited"),
    }
    println!("Tensors: {}", plan.tensor_count);
    println!("Chunks: {}", plan.chunk_count);
    println!(
        "Original tensor bytes: {}",
        format_bytes(plan.total_original_bytes as usize)
    );
    println!(
        "Full-decode baseline: {}",
        format_bytes(plan.full_decode_runtime_bytes)
    );
    println!(
        "Metadata/index estimate: {}",
        format_bytes(plan.metadata_index_bytes_estimate)
    );
    println!(
        "Activation window estimate: {}",
        format_bytes(plan.activation_window_bytes)
    );
    println!(
        "KV cache estimate: {}",
        format_bytes(plan.kv_cache_bytes_estimate)
    );
    println!(
        "Planned peak RAM: {}",
        format_bytes(plan.planned_peak_bytes)
    );
    println!("Largest step: {}", plan.largest_step.label);
    if let Some(name) = &plan.largest_step.tensor_name {
        println!("Largest tensor window: {}", name);
    }
    if let Some(chunk_id) = plan.largest_step.chunk_id {
        println!("Largest chunk id: {}", chunk_id);
    }

    if let Some(hidden) = plan.shape_hints.hidden_size {
        println!("Hidden size: {}", hidden);
    }
    if plan.shape_hints.num_layers > 0 {
        println!("Detected layers: {}", plan.shape_hints.num_layers);
    }
    if let Some(vocab) = plan.shape_hints.vocab_size {
        println!("Vocab size: {}", vocab);
    }

    match plan.status {
        PlanStatus::Ok => println!("Status: OK"),
        PlanStatus::OverBudget { over_by_bytes } => {
            println!("Status: OVER_BUDGET by {}", format_bytes(over_by_bytes));
        }
    }

    if !dry_run && mode != RuntimeMode::FullDecode {
        println!("\n[phase-5b] This mode currently performs a memory plan/dry-run only.");
        println!(
            "[phase-5b] Token execution will be added after chunk/tile matmul is implemented."
        );
    } else if dry_run {
        println!("\n[phase-5b] Dry-run complete; no tensor payloads were decoded.");
    }

    Ok(())
}

fn format_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let value = bytes as f64;
    if value >= GIB {
        format!("{:.2} GiB", value / GIB)
    } else if value >= MIB {
        format!("{:.2} MiB", value / MIB)
    } else if value >= KIB {
        format!("{:.2} KiB", value / KIB)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        effective_rama_prefill_chunk_tokens, parse_rama_integrity_mode, parse_rama_prefill_policy,
        parse_token_ids,
    };
    use rllm_runtime::{
        RamaIntegrityMode, RamaPrefillPolicy, StreamingEchoTransformerConfig,
        StreamingSamplingConfig, StreamingTinyRotaryConfig,
    };

    fn policy_config(
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> StreamingEchoTransformerConfig {
        StreamingEchoTransformerConfig {
            num_layers,
            max_new_tokens: 16,
            max_seq_len: 2048,
            vocab_size: 50_304,
            num_heads,
            head_dim: hidden_size / num_heads,
            intermediate_size,
            causal: true,
            layer_norm_eps: 1e-5,
            use_parallel_residual: true,
            sampling: StreamingSamplingConfig::Argmax,
            rotary: Some(StreamingTinyRotaryConfig {
                rotary_dim: hidden_size / num_heads,
                base: 10_000.0,
            }),
        }
    }

    #[test]
    fn parse_token_ids_accepts_comma_separated_values() {
        assert_eq!(parse_token_ids("12092, 39091").unwrap(), vec![12092, 39091]);
    }

    #[test]
    fn parse_token_ids_rejects_empty_values() {
        assert!(parse_token_ids(" , ").is_err());
    }

    #[test]
    fn parse_rama_integrity_mode_accepts_strict_and_verify_once() {
        assert_eq!(
            parse_rama_integrity_mode("strict").unwrap(),
            RamaIntegrityMode::Strict
        );
        assert_eq!(
            parse_rama_integrity_mode("verify-once").unwrap(),
            RamaIntegrityMode::VerifyOnce
        );
        assert_eq!(
            parse_rama_integrity_mode("verify_once").unwrap(),
            RamaIntegrityMode::VerifyOnce
        );
    }

    #[test]
    fn parse_rama_integrity_mode_rejects_unknown_values() {
        assert!(parse_rama_integrity_mode("skip").is_err());
    }

    #[test]
    fn parse_rama_prefill_policy_accepts_low_ram_and_speed() {
        assert_eq!(
            parse_rama_prefill_policy("low-ram").unwrap(),
            RamaPrefillPolicy::LowRam
        );
        assert_eq!(
            parse_rama_prefill_policy("low_ram").unwrap(),
            RamaPrefillPolicy::LowRam
        );
        assert_eq!(
            parse_rama_prefill_policy("speed").unwrap(),
            RamaPrefillPolicy::Speed
        );
        assert_eq!(
            parse_rama_prefill_policy("fast").unwrap(),
            RamaPrefillPolicy::Speed
        );
    }

    #[test]
    fn parse_rama_prefill_policy_rejects_unknown_values() {
        assert!(parse_rama_prefill_policy("largest").is_err());
    }

    #[test]
    fn effective_rama_prefill_chunk_tokens_uses_auto_policy_by_shape() {
        let pythia_70m_like = policy_config(6, 512, 8, 2048);
        let pythia_160m_like = policy_config(12, 768, 12, 3072);
        let budget_100mb = Some(100 * 1024 * 1024);

        assert_eq!(
            effective_rama_prefill_chunk_tokens(
                None,
                false,
                RamaPrefillPolicy::LowRam,
                1024,
                pythia_70m_like,
                budget_100mb,
            )
            .unwrap(),
            Some(32)
        );
        assert_eq!(
            effective_rama_prefill_chunk_tokens(
                None,
                false,
                RamaPrefillPolicy::LowRam,
                1024,
                pythia_160m_like,
                budget_100mb,
            )
            .unwrap(),
            Some(64)
        );
        assert_eq!(
            effective_rama_prefill_chunk_tokens(
                None,
                false,
                RamaPrefillPolicy::Speed,
                1024,
                pythia_160m_like,
                budget_100mb,
            )
            .unwrap(),
            Some(128)
        );
    }

    #[test]
    fn effective_rama_prefill_chunk_tokens_honors_fixed_and_disabled_modes() {
        let pythia_160m_like = policy_config(12, 768, 12, 3072);

        assert_eq!(
            effective_rama_prefill_chunk_tokens(
                Some(64),
                false,
                RamaPrefillPolicy::Speed,
                1024,
                pythia_160m_like,
                Some(100 * 1024 * 1024),
            )
            .unwrap(),
            Some(64)
        );
        assert_eq!(
            effective_rama_prefill_chunk_tokens(
                None,
                true,
                RamaPrefillPolicy::LowRam,
                1024,
                pythia_160m_like,
                Some(100 * 1024 * 1024),
            )
            .unwrap(),
            None
        );
        assert_eq!(
            effective_rama_prefill_chunk_tokens(
                Some(128),
                true,
                RamaPrefillPolicy::Speed,
                1024,
                pythia_160m_like,
                Some(100 * 1024 * 1024),
            )
            .unwrap(),
            None
        );
    }
}
