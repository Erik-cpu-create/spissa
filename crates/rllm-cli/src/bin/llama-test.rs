use anyhow::{Context, Result};
use clap::Parser;
use std::io::{self, Write};

use rllm_runtime::{
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata, LlamaRamaGenerationConfig,
        LlamaRamaSessionAdapter,
    },
    LazyRllmModel, MemoryBudget, RamaChatSession, RamaIntegrityMode, RamaSessionPhaseTimings,
    RllmTokenizer, StreamingSamplingConfig,
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
}

fn interactive_turn_text(has_context: bool, text: &str) -> String {
    if has_context {
        format!("\n{text}")
    } else {
        text.to_string()
    }
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
        format!(
            " | AIP: policy={} calls={} fallbacks={} max_topk={} skipped_madds={} scratch={} bytes",
            policy_str,
            stats.sparse_projection_calls,
            stats.exact_fallbacks,
            stats.max_selected_topk,
            stats.estimated_skipped_madds,
            stats.peak_scratch_bytes
        ) + &lm_head_note
            + &column_cache_note
            + &input_tile_note
            + &lm_head_agreement_note
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

fn format_phase_profile_suffix(timings: RamaSessionPhaseTimings, decode_wall_ms: f64) -> String {
    if timings.total_ms() == 0.0 {
        return String::new();
    }

    let detail = timings.transformer_detail;
    let profiled_total_ms = timings.total_ms();
    let overhead_ms = (decode_wall_ms - profiled_total_ms).max(0.0);
    format!(
        " | Profile: decode_total={:.2}ms profiled={:.2}ms overhead={:.2}ms embedding={:.2}ms transformer={:.2}ms attention_total={:.2}ms mlp_total={:.2}ms final_norm={:.2}ms lm_head={:.2}ms layers={} q={:.2}ms k={:.2}ms v={:.2}ms attn={:.2}ms gate={:.2}ms up={:.2}ms down={:.2}ms",
        decode_wall_ms,
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.ctx == 0 {
        anyhow::bail!("--ctx must be greater than zero");
    }
    if args.max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    let mut model = LazyRllmModel::open(&args.model)?;

    let tokenizer_meta = model
        .metadata()
        .tokenizer
        .as_ref()
        .context("Model does not have tokenizer metadata packed inside")?;

    let tokenizer = RllmTokenizer::from_metadata(tokenizer_meta)?;
    let eos_token_id = tokenizer_meta.eos_token_id;

    let config = LlamaRamaGenerationConfig {
        max_new_tokens: args.max_new_tokens,
        max_seq_len: Some(args.ctx),
        causal: true,
        sampling: StreamingSamplingConfig::Argmax,
    };

    // VerifyOnce: verify each chunk SHA-256 only on first access, then trust it.
    // This eliminates ~420 redundant SHA-256 computations per generated token.
    model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);

    let prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(&mut model, config)?;
    let mut budget = MemoryBudget::unbounded();
    let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget)?;
    adapter.set_transformer_detail_timing(args.profile_phases);
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
        let rolling_suffix = format_rolling_suffix(result.metrics.rolling_stats);
        let aip_suffix = format_aip_suffix(result.metrics.experimental_speed_stats);
        let repetition_suffix = format_repetition_suffix(result.metrics.repetition_stats);
        let phase_profile_suffix = if args.profile_phases {
            format_phase_profile_suffix(
                result.metrics.decode_phase_timings,
                result.metrics.decode_ms,
            )
        } else {
            String::new()
        };
        println!(
            "\n[TTFT/Prefill: {:.2}s | Decode: {:.2} tok/s | E2E: {:.2} tok/s | Total: {} tokens | Context: {} tokens | Peak: {} bytes{}{}{}{}]",
            result.metrics.ttft_ms / 1000.0,
            result.metrics.decode_tok_s,
            result.metrics.end_to_end_tok_s,
            result.metrics.generated_tokens,
            session.token_history().len(),
            result.metrics.peak_transient_bytes,
            rolling_suffix,
            aip_suffix,
            repetition_suffix,
            phase_profile_suffix
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
            lm_head_prefix_rows: 512,
            lm_head_vocab_rows: 128_256,
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
        });

        assert!(suffix.contains("AIP: policy=quality"));
        assert!(suffix.contains("policy=quality"));
        assert!(suffix.contains("calls=4"));
        assert!(suffix.contains("fallbacks=1"));
        assert!(suffix.contains("max_topk=128"));
        assert!(suffix.contains("skipped_madds=2048"));
        assert!(suffix.contains("lm_head_rows=512/128256"));
        assert!(suffix.contains("column_cache_hits=8"));
        assert!(suffix.contains("column_cache_misses=4"));
        assert!(suffix.contains("column_cache_resident=12/49152 bytes"));
        assert!(suffix.contains("input_tile_reads=5"));
        assert!(suffix.contains("input_tile_bytes=256"));
        assert!(
            suffix.contains("lm_head_agreement=selected:4/10 raw:3/10 exact_in_topk:6/10 topk=8")
        );
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
    fn phase_profile_suffix_reports_decode_subphases_and_overhead() {
        let suffix = format_phase_profile_suffix(
            rllm_runtime::RamaSessionPhaseTimings {
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
            },
            44.0,
        );

        assert!(suffix.contains("Profile: decode_total=44.00ms"));
        assert!(suffix.contains("profiled=40.00ms"));
        assert!(suffix.contains("overhead=4.00ms"));
        assert!(suffix.contains("attention_total=14.00ms"));
        assert!(suffix.contains("mlp_total=21.00ms"));
        assert!(suffix.contains("lm_head=10.00ms"));
        assert!(suffix.contains("layers=16"));
    }
}
