use crate::commands::common::parse_size;
use anyhow::{Context, Result};
use rllm_runtime::{
    build_runtime_plan, prepare_gpt_neox_rama_layer_decode_transformer_from_metadata,
    FullDecodeModel, GptNeoxRamaGenerationConfig, LazyRllmModel, MemoryBudget, PlanStatus,
    RllmTokenizer, RuntimeMode, RuntimePlanConfig, StreamingSamplingConfig,
};
use std::path::Path;
use std::str::FromStr;

pub fn run(
    file: &str,
    mode: &str,
    ctx: usize,
    memory_budget: Option<&str>,
    dry_run: bool,
    prompt: Option<&str>,
    max_new_tokens: usize,
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

    if let Some(prompt) = prompt {
        if dry_run {
            anyhow::bail!("--prompt cannot be combined with --dry-run");
        }
        return run_text_generation(path, file, ctx, max_new_tokens, memory_budget_bytes, prompt);
    }

    match runtime_mode {
        RuntimeMode::FullDecode if !dry_run && memory_budget_bytes.is_none() => {
            run_full_decode(path, file)
        }
        _ => run_planned(path, file, runtime_mode, ctx, memory_budget_bytes, dry_run),
    }
}

fn run_text_generation(
    path: &Path,
    file: &str,
    ctx: usize,
    max_new_tokens: usize,
    memory_budget_bytes: Option<usize>,
    prompt: &str,
) -> Result<()> {
    if max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    println!("RLLM Runtime — Phase 6 RAMA layer-decode text generation");
    println!("=============================================\n");
    println!("Loading metadata and streamed tensors: {}", file);

    let mut model =
        LazyRllmModel::open(path).with_context(|| format!("failed to open {}", file))?;
    let tokenizer_metadata = model
        .metadata()
        .tokenizer
        .clone()
        .ok_or_else(|| anyhow::anyhow!("model metadata does not include tokenizer metadata; repack with --tokenizer <tokenizer.json> or a sibling tokenizer.json"))?;
    let tokenizer = RllmTokenizer::from_metadata(&tokenizer_metadata)?;
    let prepared = prepare_gpt_neox_rama_layer_decode_transformer_from_metadata(
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
    let result = prepared.generate_text_from_model(&mut model, &tokenizer, prompt, &mut budget)?;

    println!("Model: {}", model.metadata().model_name);
    println!("Architecture: {}", model.metadata().architecture);
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());
    println!("Prompt: {}", prompt);
    println!("Prompt token IDs: {:?}", result.prompt_token_ids);
    println!("Generated token IDs: {:?}", result.generated_token_ids);
    println!("Generated text: {}", result.generated_text);
    println!("Full text: {}", result.text);
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
    println!("\n[phase-6] Tokenizer-backed RAMA layer-decode text generation completed.");

    Ok(())
}

fn run_full_decode(path: &Path, file: &str) -> Result<()> {
    println!("RLLM Runtime — Phase 5A full-decode loader");
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
    println!("RLLM Runtime — Phase 5B low-memory planner");
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
