use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

#[path = "../chat_template.rs"]
mod chat_template;

use chat_template::{render_interactive_user_turn, stop_token_ids, ChatTemplateKind};
use rllm_runtime::{
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata, LlamaRamaGenerationConfig,
        LlamaRamaSessionAdapter,
    },
    LazyRllmModel, MemoryBudget, RamaChatSession, RamaIntegrityMode, RamaSessionPhaseTimings,
    RamaTrace, RllmTokenizer, StreamingSamplingConfig,
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

    /// Print decode-phase timing breakdown for profiler runs.
    #[arg(long, default_value_t = false)]
    profile_phases: bool,

    /// Chat template used to format interactive turns: raw, llama3, or chatml.
    #[arg(long, default_value = "raw")]
    chat_template: String,

    /// Optional system prompt for chat-template modes.
    #[arg(long)]
    system_prompt: Option<String>,

    /// Optional path for RAMA chunk trace JSON output.
    #[arg(long)]
    rama_trace: Option<String>,

    /// Runtime chunk integrity mode: strict, verify-once, or unchecked.
    #[arg(long, default_value = "verify-once")]
    rama_integrity: String,
}

fn parse_rama_integrity_mode(raw: &str) -> Result<RamaIntegrityMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "strict" => Ok(RamaIntegrityMode::Strict),
        "verify-once" | "verify_once" | "once" => Ok(RamaIntegrityMode::VerifyOnce),
        "unchecked" | "none" | "trusted" => Ok(RamaIntegrityMode::Unchecked),
        other => {
            anyhow::bail!(
                "unsupported --rama-integrity {other:?}; expected strict, verify-once, or unchecked"
            )
        }
    }
}

fn tensor_bucket(tensor_name: Option<&str>) -> &'static str {
    let Some(name) = tensor_name else {
        return "other";
    };

    if name.contains(".mlp.gate_proj.weight") {
        "mlp.gate_proj"
    } else if name.contains(".mlp.up_proj.weight") {
        "mlp.up_proj"
    } else if name.contains(".mlp.down_proj.weight") {
        "mlp.down_proj"
    } else if name.contains(".self_attn.q_proj.weight") {
        "attention.q_proj"
    } else if name.contains(".self_attn.k_proj.weight") {
        "attention.k_proj"
    } else if name.contains(".self_attn.v_proj.weight") {
        "attention.v_proj"
    } else if name.contains(".self_attn.o_proj.weight") {
        "attention.o_proj"
    } else if name.contains("lm_head.weight") {
        "lm_head"
    } else {
        "other"
    }
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
    let mut tensor_bucket_totals: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for event in &trace.events {
        let phase_entry = phase_totals.entry(event.phase.as_str()).or_insert((0, 0));
        phase_entry.0 += 1;
        phase_entry.1 = phase_entry.1.saturating_add(event.duration_ns);

        if event.phase == "chunk_compute" || event.phase == "chunk_compute_closure" {
            let bucket = tensor_bucket(event.tensor_name.as_deref());
            let bucket_entry = tensor_bucket_totals.entry(bucket).or_insert((0, 0));
            bucket_entry.0 += 1;
            bucket_entry.1 = bucket_entry.1.saturating_add(event.duration_ns);
        }
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
    let duration_by_tensor_bucket: Vec<_> = tensor_bucket_totals
        .into_iter()
        .map(|(bucket, (event_count, total_ns))| {
            json!({
                "bucket": bucket,
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
            "duration_by_tensor_bucket": duration_by_tensor_bucket,
        }
    });
    fs::write(output, serde_json::to_vec_pretty(&payload)?)
        .with_context(|| format!("failed to write RAMA trace JSON to {}", output.display()))?;
    Ok(())
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
        let lm_head_rescore_note = if stats.lm_head_rescore_checks > 0 {
            format!(
                " lm_head_rescore={}/{} gap_skips={} max_gap_milli={}",
                stats.lm_head_rescore_uses,
                stats.lm_head_rescore_checks,
                stats.lm_head_rescore_gap_skips,
                stats.lm_head_rescore_max_gap_milli
            )
        } else {
            String::new()
        };
        let column_cache_note = if stats.column_cache_hits > 0 || stats.column_cache_misses > 0 {
            format!(
                " column_cache_hits={} column_cache_misses={} column_cache_resident={}/{} bytes",
                stats.column_cache_hits,
                stats.column_cache_misses,
                stats.column_cache_resident_columns,
                stats.column_cache_resident_bytes
            )
        } else {
            String::new()
        };
        let input_tile_note = if stats.input_tile_range_reads > 0 {
            format!(
                " input_tile_reads={} input_tile_bytes={}",
                stats.input_tile_range_reads, stats.input_tile_range_bytes
            )
        } else {
            String::new()
        };
        let attention_locality_note = if stats.attention_locality_uses > 0 {
            format!(
                " attention_locality={}/{} max_selected={}",
                stats.attention_locality_added_indices,
                stats.attention_locality_uses,
                stats.attention_locality_max_selected
            )
        } else {
            String::new()
        };
        let lm_head_agreement_note = if stats.lm_head_agreement_samples > 0 {
            format!(
                " lm_head_agreement=selected:{}/{} raw:{}/{} exact_in_topk:{}/{} topk={}",
                stats.lm_head_agreement_selected_matches,
                stats.lm_head_agreement_samples,
                stats.lm_head_agreement_sparse_argmax_matches,
                stats.lm_head_agreement_samples,
                stats.lm_head_agreement_exact_in_sparse_topk,
                stats.lm_head_agreement_samples,
                stats.lm_head_agreement_max_topk
            )
        } else {
            String::new()
        };
        let lm_head_exact_note = if stats.lm_head_exact_checks > 0 {
            format!(
                " lm_head_exact={}/{}",
                stats.lm_head_exact_switches, stats.lm_head_exact_checks
            )
        } else {
            String::new()
        };
        let lm_head_repeat_margin_note = if stats.lm_head_repeat_margin_checks > 0 {
            let adaptive_note = if stats.lm_head_repeat_margin_adaptive_throttles > 0 {
                format!(
                    " adaptive_throttles={} min_margin_milli={}",
                    stats.lm_head_repeat_margin_adaptive_throttles,
                    stats.lm_head_repeat_margin_min_effective_milli
                )
            } else {
                String::new()
            };
            format!(
                " lm_head_repeat_margin={}/{} max_gap_milli={}",
                stats.lm_head_repeat_margin_switches,
                stats.lm_head_repeat_margin_checks,
                stats.lm_head_repeat_margin_max_gap_milli
            ) + &adaptive_note
        } else {
            String::new()
        };
        let lm_head_phrase_novelty_note = if stats.lm_head_phrase_novelty_checks > 0 {
            let gap_note = if stats.lm_head_phrase_novelty_gap_skips > 0 {
                format!(
                    " gap_skips={} max_gap_milli={}",
                    stats.lm_head_phrase_novelty_gap_skips,
                    stats.lm_head_phrase_novelty_max_gap_milli
                )
            } else {
                String::new()
            };
            let soft_note = if stats.lm_head_phrase_novelty_soft_choices > 0 {
                format!(
                    " soft_choices={}",
                    stats.lm_head_phrase_novelty_soft_choices
                )
            } else {
                String::new()
            };
            let retention_note = if stats.lm_head_phrase_novelty_retentions > 0 {
                format!(" retentions={}", stats.lm_head_phrase_novelty_retentions)
            } else {
                String::new()
            };
            format!(
                " phrase_novelty={}/{} max_ngram={}",
                stats.lm_head_phrase_novelty_switches,
                stats.lm_head_phrase_novelty_checks,
                stats.lm_head_phrase_novelty_max_ngram
            ) + &gap_note
                + &soft_note
                + &retention_note
        } else {
            String::new()
        };
        let layer_drift_note = if stats.layer_drift_probe.samples > 0 {
            format!(
                " layer_drift_probe={} layers={} mismatch_layers={} first_mismatch_layer={} pre_mismatch_max_l2_milli={} pre_mismatch_max_cosine_gap_milli={} max_l2_milli={} max_cosine_gap_milli={} max_exact_margin_milli={}",
                stats.layer_drift_probe.samples,
                stats.layer_drift_probe.layers,
                stats.layer_drift_probe.mismatch_layers,
                stats.layer_drift_probe.first_mismatch_layer,
                stats.layer_drift_probe.pre_mismatch_max_l2_milli,
                stats.layer_drift_probe.pre_mismatch_max_cosine_gap_milli,
                stats.layer_drift_probe.max_l2_milli,
                stats.layer_drift_probe.max_cosine_gap_milli,
                stats.layer_drift_probe.max_exact_margin_milli
            )
        } else {
            String::new()
        };
        let layer_attribution_note = if stats.layer_attribution_probe.samples > 0 {
            format!(
                " layer_attribution_probe={} attribution_layer={} attention_l2_milli={} attention_cosine_gap_milli={} gate_up_l2_milli={} gate_up_cosine_gap_milli={} down_l2_milli={} down_cosine_gap_milli={}",
                stats.layer_attribution_probe.samples,
                stats.layer_attribution_probe.layer,
                stats.layer_attribution_probe.attention_l2_milli,
                stats.layer_attribution_probe.attention_cosine_gap_milli,
                stats.layer_attribution_probe.gate_up_l2_milli,
                stats.layer_attribution_probe.gate_up_cosine_gap_milli,
                stats.layer_attribution_probe.down_l2_milli,
                stats.layer_attribution_probe.down_cosine_gap_milli
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
            + &lm_head_rescore_note
            + &column_cache_note
            + &input_tile_note
            + &attention_locality_note
            + &lm_head_agreement_note
            + &lm_head_exact_note
            + &lm_head_repeat_margin_note
            + &lm_head_phrase_novelty_note
            + &layer_drift_note
            + &layer_attribution_note
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

fn format_phase_profile_suffix(
    prefill_timings: RamaSessionPhaseTimings,
    prefill_wall_ms: f64,
    decode_timings: RamaSessionPhaseTimings,
    decode_wall_ms: f64,
) -> String {
    if prefill_timings.total_ms() == 0.0 && decode_timings.total_ms() == 0.0 {
        return String::new();
    }

    format!(
        "{}{}",
        format_phase_profile_segment(
            "PrefillProfile",
            "prefill",
            prefill_timings,
            prefill_wall_ms
        ),
        format_phase_profile_segment("DecodeProfile", "decode", decode_timings, decode_wall_ms)
    )
}

fn format_phase_profile_segment(
    label: &str,
    total_label: &str,
    timings: RamaSessionPhaseTimings,
    wall_ms: f64,
) -> String {
    let detail = timings.transformer_detail;
    let profiled_total_ms = timings.total_ms();
    let overhead_ms = (wall_ms - profiled_total_ms).max(0.0);
    format!(
        " | {label}: {total_label}_total={:.2}ms profiled={:.2}ms overhead={:.2}ms embedding={:.2}ms transformer={:.2}ms attention_total={:.2}ms mlp_total={:.2}ms final_norm={:.2}ms lm_head={:.2}ms layers={} q={:.2}ms k={:.2}ms v={:.2}ms attn={:.2}ms gate={:.2}ms up={:.2}ms down={:.2}ms",
        wall_ms,
        profiled_total_ms,
        overhead_ms,
        timings.embedding_ms,
        timings.transformer_ms,
        detail.attention_total_ms(),
        detail.mlp_total_ms(),
        timings.final_norm_ms,
        timings.lm_head_ms,
        detail.profiled_layers,
        detail.q_projection_ms,
        detail.k_projection_ms,
        detail.v_projection_ms,
        detail.attention_ms,
        detail.gate_projection_ms,
        detail.up_projection_ms,
        detail.down_projection_ms
    )
}

fn format_q8_kernel_profile_suffix() -> String {
    let Some(snapshot) = rllm_runtime::q8_kernel_profile_snapshot_and_reset() else {
        return String::new();
    };
    if snapshot.rows.is_empty() {
        return String::new();
    }

    let rows = snapshot
        .rows
        .iter()
        .take(4)
        .map(|row| {
            format!(
                "{} calls={} blocks={} rows={} batch_items={} elapsed={:.2}ms",
                row.path,
                row.calls,
                row.blocks,
                row.rows,
                row.batch_items,
                row.elapsed_ns as f64 / 1_000_000.0
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        " | Q8KernelProfile: kernel={} top={}",
        snapshot.ree_kernel, rows
    )
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
    if args.rama_trace.is_some() {
        model.enable_rama_trace();
    }
    let rama_integrity = parse_rama_integrity_mode(&args.rama_integrity)?;

    let tokenizer_meta = model
        .metadata()
        .tokenizer
        .as_ref()
        .context("Model does not have tokenizer metadata packed inside")?;

    let tokenizer = RllmTokenizer::from_metadata(tokenizer_meta)?;
    let chat_template: ChatTemplateKind = args.chat_template.parse()?;
    let stop_token_ids = stop_token_ids(chat_template, &tokenizer, tokenizer_meta.eos_token_id);

    let config = LlamaRamaGenerationConfig {
        max_new_tokens: args.max_new_tokens,
        max_seq_len: Some(args.ctx),
        causal: true,
        sampling: StreamingSamplingConfig::Argmax,
    };

    model.set_rama_integrity_mode(rama_integrity);

    let prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(&mut model, config)?;
    let mut budget = MemoryBudget::unbounded();
    let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget)?;
    adapter.set_transformer_detail_timing(args.profile_phases);
    let mut session = RamaChatSession::new(adapter);

    println!("===================================================");
    println!("RLLM Interactive Chat (Llama Architecture, token-native session)");
    println!("Chat template: {}", args.chat_template);
    println!("Type 'quit' or 'exit' to end.");
    println!("===================================================");

    let mut has_context = false;
    let mut previous_assistant_ended = true;

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
        let turn_text = render_interactive_user_turn(
            chat_template,
            has_context,
            previous_assistant_ended,
            args.system_prompt.as_deref(),
            text,
        );
        let input_tokens = tokenizer.encode(&turn_text)?;

        let mut assistant_ended = false;
        let mut on_token = |token: usize| -> bool {
            if stop_token_ids.contains(&token) {
                assistant_ended = true;
                return false;
            }
            if let Ok(word) = tokenizer.decode(&[token]) {
                print!("{}", word);
                io::stdout().flush().unwrap();
            }

            true
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
        let phase_profile_suffix = if args.profile_phases {
            format_phase_profile_suffix(
                result.metrics.prefill_phase_timings,
                result.metrics.prefill_ms,
                result.metrics.decode_phase_timings,
                result.metrics.decode_ms,
            )
        } else {
            String::new()
        };
        let q8_kernel_profile_suffix = format_q8_kernel_profile_suffix();
        println!(
            "\n[TTFT/Prefill: {:.2}s | Decode: {:.2} tok/s | E2E: {:.2} tok/s | Total: {} tokens | Context: {} tokens | Peak: {} bytes{}{}{}{}{}]",
            result.metrics.ttft_ms / 1000.0,
            result.metrics.decode_tok_s,
            result.metrics.end_to_end_tok_s,
            result.metrics.generated_tokens,
            session.token_history().len(),
            result.metrics.peak_transient_bytes,
            rolling_suffix,
            aip_suffix,
            repetition_suffix,
            phase_profile_suffix,
            q8_kernel_profile_suffix
        );
        has_context = true;
        previous_assistant_ended = assistant_ended;
    }

    drop(session);
    if let Some(rama_trace_out) = args.rama_trace.as_deref() {
        let trace = model
            .take_rama_trace()
            .context("RAMA trace was requested but was not enabled")?;
        write_rama_trace_json(rama_trace_out, &trace)?;
        println!("RAMA trace JSON: {}", rama_trace_out);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            attention_locality_uses: 6,
            attention_locality_added_indices: 18,
            attention_locality_max_selected: 8,
            lm_head_prefix_rows: 512,
            lm_head_vocab_rows: 128_256,
            lm_head_rescore_checks: 9,
            lm_head_rescore_uses: 6,
            lm_head_rescore_gap_skips: 3,
            lm_head_rescore_max_gap_milli: 450,
            column_cache_hits: 8,
            column_cache_misses: 4,
            column_cache_resident_columns: 12,
            column_cache_resident_bytes: 49_152,
            input_tile_range_reads: 5,
            input_tile_range_bytes: 256,
            lm_head_agreement_samples: 10,
            lm_head_agreement_sparse_argmax_matches: 3,
            lm_head_agreement_selected_matches: 4,
            lm_head_agreement_exact_in_sparse_topk: 6,
            lm_head_agreement_max_topk: 8,
            lm_head_exact_checks: 5,
            lm_head_exact_switches: 2,
            lm_head_repeat_margin_checks: 7,
            lm_head_repeat_margin_switches: 5,
            lm_head_repeat_margin_max_gap_milli: 125,
            lm_head_repeat_margin_adaptive_throttles: 2,
            lm_head_repeat_margin_min_effective_milli: 125,
            lm_head_phrase_novelty_checks: 12,
            lm_head_phrase_novelty_switches: 9,
            lm_head_phrase_novelty_max_ngram: 3,
            lm_head_phrase_novelty_gap_skips: 4,
            lm_head_phrase_novelty_max_gap_milli: 900,
            lm_head_phrase_novelty_soft_choices: 6,
            lm_head_phrase_novelty_retentions: 3,
            layer_drift_probe: rllm_runtime::RamaLayerDriftProbeStats {
                samples: 2,
                layers: 64,
                mismatch_layers: 3,
                first_mismatch_layer: 2,
                pre_mismatch_max_l2_milli: 100,
                pre_mismatch_max_cosine_gap_milli: 5,
                max_l2_milli: 1_250,
                max_cosine_gap_milli: 15,
                max_exact_margin_milli: 900,
            },
            layer_attribution_probe: rllm_runtime::RamaLayerAttributionProbeStats {
                samples: 1,
                layer: 2,
                attention_l2_milli: 111,
                attention_cosine_gap_milli: 11,
                gate_up_l2_milli: 222,
                gate_up_cosine_gap_milli: 22,
                down_l2_milli: 333,
                down_cosine_gap_milli: 33,
            },
        });

        assert!(suffix.contains("AIP: policy=quality"));
        assert!(suffix.contains("policy=quality"));
        assert!(suffix.contains("calls=4"));
        assert!(suffix.contains("fallbacks=1"));
        assert!(suffix.contains("max_topk=128"));
        assert!(suffix.contains("skipped_madds=2048"));
        assert!(suffix.contains("lm_head_rows=512/128256"));
        assert!(suffix.contains("lm_head_rescore=6/9 gap_skips=3 max_gap_milli=450"));
        assert!(suffix.contains("column_cache_hits=8"));
        assert!(suffix.contains("column_cache_misses=4"));
        assert!(suffix.contains("column_cache_resident=12/49152 bytes"));
        assert!(suffix.contains("input_tile_reads=5"));
        assert!(suffix.contains("input_tile_bytes=256"));
        assert!(suffix.contains("attention_locality=18/6 max_selected=8"));
        assert!(
            suffix.contains("lm_head_agreement=selected:4/10 raw:3/10 exact_in_topk:6/10 topk=8")
        );
        assert!(suffix.contains("lm_head_exact=2/5"));
        assert!(suffix.contains("lm_head_repeat_margin=5/7 max_gap_milli=125"));
        assert!(suffix.contains("adaptive_throttles=2 min_margin_milli=125"));
        assert!(suffix.contains("phrase_novelty=9/12 max_ngram=3"));
        assert!(suffix.contains("gap_skips=4 max_gap_milli=900"));
        assert!(suffix.contains("soft_choices=6"));
        assert!(suffix.contains("retentions=3"));
        assert!(suffix.contains("layer_drift_probe=2"));
        assert!(suffix.contains("layers=64"));
        assert!(suffix.contains("mismatch_layers=3"));
        assert!(suffix.contains("first_mismatch_layer=2"));
        assert!(suffix.contains("pre_mismatch_max_l2_milli=100"));
        assert!(suffix.contains("pre_mismatch_max_cosine_gap_milli=5"));
        assert!(suffix.contains("max_l2_milli=1250"));
        assert!(suffix.contains("max_cosine_gap_milli=15"));
        assert!(suffix.contains("max_exact_margin_milli=900"));
        assert!(suffix.contains("layer_attribution_probe=1"));
        assert!(suffix.contains("attribution_layer=2"));
        assert!(suffix.contains("attention_l2_milli=111"));
        assert!(suffix.contains("attention_cosine_gap_milli=11"));
        assert!(suffix.contains("gate_up_l2_milli=222"));
        assert!(suffix.contains("gate_up_cosine_gap_milli=22"));
        assert!(suffix.contains("down_l2_milli=333"));
        assert!(suffix.contains("down_cosine_gap_milli=33"));
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

    #[test]
    fn args_disable_phase_profile_by_default_and_accept_override() {
        let default_args = Args::parse_from(["llama-test", "--model", "model.rllm"]);
        assert!(!default_args.profile_phases);

        let profiled_args =
            Args::parse_from(["llama-test", "--model", "model.rllm", "--profile-phases"]);
        assert!(profiled_args.profile_phases);
    }

    #[test]
    fn args_default_to_raw_chat_template_and_accept_llama3() {
        let default_args = Args::parse_from(["llama-test", "--model", "model.rllm"]);
        assert_eq!(default_args.chat_template, "raw");
        assert_eq!(default_args.system_prompt, None);

        let templated_args = Args::parse_from([
            "llama-test",
            "--model",
            "model.rllm",
            "--chat-template",
            "llama3",
            "--system-prompt",
            "You are concise.",
        ]);
        assert_eq!(templated_args.chat_template, "llama3");
        assert_eq!(
            templated_args.system_prompt.as_deref(),
            Some("You are concise.")
        );
    }

    #[test]
    fn args_default_to_no_rama_trace_and_accept_path() {
        let default_args = Args::parse_from(["llama-test", "--model", "model.rllm"]);
        assert_eq!(default_args.rama_trace, None);

        let traced_args = Args::parse_from([
            "llama-test",
            "--model",
            "model.rllm",
            "--rama-trace",
            "/tmp/rama-trace.json",
        ]);
        assert_eq!(
            traced_args.rama_trace.as_deref(),
            Some("/tmp/rama-trace.json")
        );
    }

    #[test]
    fn args_default_to_verify_once_integrity_and_accept_unchecked() {
        let default_args = Args::parse_from(["llama-test", "--model", "model.rllm"]);
        assert_eq!(default_args.rama_integrity, "verify-once");

        let unchecked_args = Args::parse_from([
            "llama-test",
            "--model",
            "model.rllm",
            "--rama-integrity",
            "unchecked",
        ]);
        assert_eq!(unchecked_args.rama_integrity, "unchecked");
    }

    #[test]
    fn tensor_bucket_groups_llama_projection_names() {
        assert_eq!(
            tensor_bucket(Some("model.layers.0.mlp.gate_proj.weight")),
            "mlp.gate_proj"
        );
        assert_eq!(
            tensor_bucket(Some("model.layers.0.mlp.up_proj.weight")),
            "mlp.up_proj"
        );
        assert_eq!(
            tensor_bucket(Some("model.layers.0.mlp.down_proj.weight")),
            "mlp.down_proj"
        );
        assert_eq!(
            tensor_bucket(Some("model.layers.0.self_attn.q_proj.weight")),
            "attention.q_proj"
        );
        assert_eq!(tensor_bucket(Some("lm_head.weight")), "lm_head");
        assert_eq!(tensor_bucket(None), "other");
    }

    #[test]
    fn rama_trace_json_reports_phase_and_tensor_bucket_totals() {
        let mut trace = RamaTrace::new("test-model", "llama");
        trace.record(rllm_runtime::RamaTraceEventInput {
            phase: "chunk_read".to_string(),
            label: "read".to_string(),
            tensor_name: Some("model.layers.0.mlp.gate_proj.weight".to_string()),
            tensor_id: Some(1),
            chunk_id: Some(0),
            codec_id: Some("q8_0".to_string()),
            compressed_bytes: Some(10),
            decoded_bytes: Some(20),
            start_ns: 0,
            duration_ns: 1_000,
            budget_current_bytes: 20,
            budget_peak_bytes: 20,
        });
        trace.record(rllm_runtime::RamaTraceEventInput {
            phase: "chunk_compute_closure".to_string(),
            label: "compute".to_string(),
            tensor_name: Some("model.layers.0.mlp.gate_proj.weight".to_string()),
            tensor_id: Some(1),
            chunk_id: Some(0),
            codec_id: Some("q8_0".to_string()),
            compressed_bytes: Some(10),
            decoded_bytes: Some(20),
            start_ns: 1_000,
            duration_ns: 2_000,
            budget_current_bytes: 20,
            budget_peak_bytes: 20,
        });

        let path =
            std::env::temp_dir().join(format!("rllm-llama-test-trace-{}.json", std::process::id()));
        write_rama_trace_json(path.to_str().unwrap(), &trace).unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(payload["summary"]["event_count"], 2);
        assert_eq!(
            payload["summary"]["duration_by_tensor_bucket"][0]["bucket"],
            "mlp.gate_proj"
        );
        assert_eq!(
            payload["summary"]["duration_by_tensor_bucket"][0]["total_ns"],
            2_000
        );
        assert!(payload["summary"]["duration_by_phase"]
            .as_array()
            .unwrap()
            .iter()
            .any(|phase| phase["phase"] == "chunk_read" && phase["total_ns"] == 1_000));
    }

    #[test]
    fn phase_profile_suffix_reports_prefill_and_decode_subphases() {
        let prefill = rllm_runtime::RamaSessionPhaseTimings {
            embedding_ms: 2.0,
            transformer_ms: 50.0,
            transformer_detail: rllm_runtime::RamaTransformerPhaseTimings {
                q_projection_ms: 10.0,
                k_projection_ms: 11.0,
                v_projection_ms: 12.0,
                attention_ms: 13.0,
                gate_projection_ms: 14.0,
                up_projection_ms: 15.0,
                down_projection_ms: 16.0,
                profiled_layers: 16,
                ..Default::default()
            },
            final_norm_ms: 17.0,
            lm_head_ms: 18.0,
        };
        let decode = rllm_runtime::RamaSessionPhaseTimings {
            embedding_ms: 1.0,
            transformer_ms: 20.0,
            transformer_detail: rllm_runtime::RamaTransformerPhaseTimings {
                q_projection_ms: 2.0,
                k_projection_ms: 3.0,
                v_projection_ms: 4.0,
                attention_ms: 5.0,
                gate_projection_ms: 6.0,
                up_projection_ms: 7.0,
                down_projection_ms: 8.0,
                profiled_layers: 16,
                ..Default::default()
            },
            final_norm_ms: 9.0,
            lm_head_ms: 10.0,
        };
        let suffix = format_phase_profile_suffix(prefill, 60.0, decode, 44.0);

        assert!(suffix.contains("PrefillProfile: prefill_total=60.00ms"));
        assert!(suffix.contains("profiled=87.00ms"));
        assert!(suffix.contains("attention_total=46.00ms"));
        assert!(suffix.contains("mlp_total=45.00ms"));
        assert!(suffix.contains("lm_head=18.00ms"));
        assert!(suffix.contains("DecodeProfile: decode_total=44.00ms"));
        assert!(suffix.contains("profiled=40.00ms"));
        assert!(suffix.contains("overhead=4.00ms"));
        assert!(suffix.contains("attention_total=14.00ms"));
        assert!(suffix.contains("mlp_total=21.00ms"));
        assert!(suffix.contains("lm_head=10.00ms"));
        assert!(suffix.contains("layers=16"));
    }

    #[test]
    fn q8_kernel_profile_suffix_is_empty_without_profile_rows() {
        let _ = rllm_runtime::q8_kernel_profile_snapshot_and_reset();

        assert_eq!(format_q8_kernel_profile_suffix(), "");
    }

    #[test]
    fn q8_kernel_profile_suffix_reports_recorded_rows() {
        let _ = rllm_runtime::q8_kernel_profile_snapshot_and_reset();
        rllm_runtime::record_q8_kernel_path(
            rllm_runtime::Q8KernelPath::ContiguousI8Dot,
            2,
            4,
            0,
            2,
            std::time::Duration::from_micros(1500),
        );
        rllm_runtime::record_q8_kernel_path(
            rllm_runtime::Q8KernelPath::Batch1CompleteMultiply,
            1,
            8,
            1,
            1,
            std::time::Duration::from_micros(500),
        );

        let suffix = format_q8_kernel_profile_suffix();

        assert!(suffix.contains("Q8KernelProfile: kernel=REETHINK-Q8-SHAPE-PROFILER"));
        assert!(suffix.contains("contiguous_i8_dot calls=2 blocks=4"));
        assert!(suffix.contains("batch1_complete_multiply calls=1 blocks=8 rows=1"));
        assert!(suffix.contains("elapsed=1.50ms"));
        assert_eq!(format_q8_kernel_profile_suffix(), "");
    }
}
