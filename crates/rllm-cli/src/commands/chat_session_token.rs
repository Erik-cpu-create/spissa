use anyhow::{Context, Result};
use rllm_runtime::{
    models::llama::{
        prepare_llama_rama_layer_decode_transformer_from_metadata,
        rama_layer_decoded_llama_transformer_generate_from_model, LayerDecodedLlamaRamaTransformer,
        LlamaRamaGenerationConfig, LlamaRamaGenerationOptions, LlamaRamaSessionAdapter,
        LlamaTextGenerationResult,
    },
    LazyRllmModel, MemoryBudget, RamaChatSession, RamaIntegrityMode, RamaSessionPhaseTimings,
    RamaSessionTurnResult, StreamingSamplingConfig,
};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone)]
struct GenerationTiming {
    generated_tokens: usize,
    ttft_ms: f64,
    decode_ms: f64,
    end_to_end_ms: f64,
    decode_tok_s: f64,
    end_to_end_tok_s: f64,
}

#[derive(Debug, Clone)]
struct TokenMatchSummary {
    matched: bool,
    note: String,
}

#[derive(Debug, Clone)]
struct TokenBenchmarkRow {
    turn_index: usize,
    baseline_input_tokens: usize,
    session_input_tokens: usize,
    baseline_timing: GenerationTiming,
    baseline_peak_transient_bytes: usize,
    session_result: RamaSessionTurnResult,
    generated_match: TokenMatchSummary,
    history_match: TokenMatchSummary,
}

pub fn run(
    file: &str,
    turns: &[String],
    max_new_tokens: usize,
    ctx: usize,
    out: &str,
) -> Result<()> {
    validate_report_output_path(out)?;
    if max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    if ctx == 0 {
        anyhow::bail!("--ctx must be greater than zero");
    }
    let token_turns = parse_token_turns(turns)?;

    let mut baseline_model =
        LazyRllmModel::open(file).with_context(|| format!("failed to open {file}"))?;
    baseline_model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
    let baseline_prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(
        &mut baseline_model,
        generation_config(max_new_tokens, ctx),
    )?;

    let mut session_model =
        LazyRllmModel::open(file).with_context(|| format!("failed to open {file}"))?;
    session_model.set_rama_integrity_mode(RamaIntegrityMode::VerifyOnce);
    let session_prepared = prepare_llama_rama_layer_decode_transformer_from_metadata(
        &mut session_model,
        generation_config(max_new_tokens, ctx),
    )?;
    let mut session_budget = MemoryBudget::unbounded();
    let mut adapter =
        LlamaRamaSessionAdapter::new(&mut session_model, &session_prepared, &mut session_budget)?;
    adapter.set_transformer_detail_timing(true);
    let mut session = RamaChatSession::new(adapter);

    let mut baseline_visible_history = Vec::new();
    let mut rows = Vec::new();
    let mut valid = true;

    for (idx, user_tokens) in token_turns.iter().enumerate() {
        let mut baseline_input = baseline_visible_history.clone();
        baseline_input.extend_from_slice(user_tokens);
        let (baseline_result, baseline_timing, baseline_peak_transient_bytes) =
            run_baseline_turn(&mut baseline_model, &baseline_prepared, &baseline_input)?;
        baseline_visible_history = baseline_result.token_ids.clone();

        let session_result =
            session.generate_turn(user_tokens, max_new_tokens, &mut session_budget, |_| true)?;
        let generated_match = token_match_summary(
            &baseline_result.generated_token_ids,
            &session_result.generated_token_ids,
        );
        let history_match = token_match_summary(&baseline_visible_history, session.token_history());
        valid &= generated_match.matched && history_match.matched;

        println!(
            "turn {}: baseline_input={} session_input={} generated={} match={} session_replayed={} session_decode_tok_s={:.2}",
            idx + 1,
            baseline_input.len(),
            user_tokens.len(),
            session_result.metrics.generated_tokens,
            generated_match.matched && history_match.matched,
            session_result.metrics.replayed_tokens,
            session_result.metrics.decode_tok_s,
        );

        rows.push(TokenBenchmarkRow {
            turn_index: idx + 1,
            baseline_input_tokens: baseline_input.len(),
            session_input_tokens: user_tokens.len(),
            baseline_timing,
            baseline_peak_transient_bytes,
            session_result,
            generated_match,
            history_match,
        });
    }

    write_report(out, file, &token_turns, max_new_tokens, ctx, valid, &rows)?;
    println!("Benchmark report: {out}");
    if !valid {
        anyhow::bail!(
            "chat-session-token generated token/history mismatch; report written to {out}"
        );
    }
    Ok(())
}

fn generation_config(max_new_tokens: usize, ctx: usize) -> LlamaRamaGenerationConfig {
    LlamaRamaGenerationConfig {
        max_new_tokens,
        max_seq_len: Some(ctx),
        causal: true,
        sampling: StreamingSamplingConfig::Argmax,
    }
}

fn run_baseline_turn(
    model: &mut LazyRllmModel,
    prepared: &LayerDecodedLlamaRamaTransformer,
    input_tokens: &[usize],
) -> Result<(LlamaTextGenerationResult, GenerationTiming, usize)> {
    let mut budget = MemoryBudget::unbounded();
    let start = Instant::now();
    let mut first_token_time = None;
    let mut on_token = |_: usize| {
        if first_token_time.is_none() {
            first_token_time = Some(Instant::now());
        }
        true
    };
    let result = rama_layer_decoded_llama_transformer_generate_from_model(
        model,
        prepared,
        input_tokens,
        &mut budget,
        LlamaRamaGenerationOptions {
            collect_logits: false,
            ..Default::default()
        },
        &mut on_token,
    )?;
    let end = Instant::now();
    let first = first_token_time.unwrap_or(end);
    let ttft_ms = first.duration_since(start).as_secs_f64() * 1000.0;
    let decode_ms = end.duration_since(first).as_secs_f64() * 1000.0;
    let end_to_end_ms = end.duration_since(start).as_secs_f64() * 1000.0;
    let decode_steps = result.generated_token_ids.len().saturating_sub(1);
    let timing = GenerationTiming {
        generated_tokens: result.generated_token_ids.len(),
        ttft_ms,
        decode_ms,
        end_to_end_ms,
        decode_tok_s: if decode_steps == 0 {
            0.0
        } else {
            decode_steps as f64 / (decode_ms / 1000.0).max(f64::EPSILON)
        },
        end_to_end_tok_s: result.generated_token_ids.len() as f64
            / (end_to_end_ms / 1000.0).max(f64::EPSILON),
    };
    Ok((result, timing, budget.peak_bytes()))
}

fn token_match_summary(baseline: &[usize], session: &[usize]) -> TokenMatchSummary {
    if baseline == session {
        TokenMatchSummary {
            matched: true,
            note: "match".to_string(),
        }
    } else {
        TokenMatchSummary {
            matched: false,
            note: format!("mismatch baseline={baseline:?} session={session:?}"),
        }
    }
}

fn format_phase_timing_note(timings: RamaSessionPhaseTimings) -> String {
    let detail = timings.transformer_detail;
    format!(
        "embedding={:.2}ms transformer={:.2}ms final_norm={:.2}ms lm_head={:.2}ms profiled_total={:.2}ms layers={} attention_total={:.2}ms attention_norm={:.2}ms q={:.2}ms k={:.2}ms v={:.2}ms rotary={:.2}ms attention={:.2}ms kv_append={:.2}ms o={:.2}ms attention_residual={:.2}ms mlp_total={:.2}ms mlp_norm={:.2}ms gate={:.2}ms up={:.2}ms activation_multiply={:.2}ms down={:.2}ms mlp_residual={:.2}ms",
        timings.embedding_ms,
        timings.transformer_ms,
        timings.final_norm_ms,
        timings.lm_head_ms,
        timings.total_ms(),
        detail.profiled_layers,
        detail.attention_total_ms(),
        detail.attention_norm_ms,
        detail.q_projection_ms,
        detail.k_projection_ms,
        detail.v_projection_ms,
        detail.rotary_ms,
        detail.attention_ms,
        detail.kv_append_ms,
        detail.o_projection_ms,
        detail.attention_residual_ms,
        detail.mlp_total_ms(),
        detail.mlp_norm_ms,
        detail.gate_projection_ms,
        detail.up_projection_ms,
        detail.activation_multiply_ms,
        detail.down_projection_ms,
        detail.mlp_residual_ms
    )
}

fn format_rolling_note(stats: rllm_runtime::RamaRollingStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        format!(
            " rolling_tasks={} rolling_wakeups={} rolling_fallbacks={} rolling_scratch_bytes={}",
            stats.submitted_tasks,
            stats.worker_wakeups,
            stats.sequential_fallbacks,
            stats.peak_scratch_bytes
        )
    }
}

fn format_aip_note(stats: rllm_runtime::RamaExperimentalSpeedStats) -> String {
    if stats.is_empty() {
        String::new()
    } else {
        let policy_str = stats.aip_policy.map(|p| p.as_str()).unwrap_or("none");
        let lm_head_note = if stats.lm_head_prefix_rows > 0 {
            format!(
                " aip_lm_head_rows={}/{}",
                stats.lm_head_prefix_rows, stats.lm_head_vocab_rows
            )
        } else {
            String::new()
        };
        let column_cache_note = if stats.column_cache_hits > 0 || stats.column_cache_misses > 0 {
            format!(
                " aip_column_cache_hits={} aip_column_cache_misses={} aip_column_cache_resident={}/{}",
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
                " aip_input_tile_reads={} aip_input_tile_bytes={}",
                stats.input_tile_range_reads, stats.input_tile_range_bytes
            )
        } else {
            String::new()
        };
        let lm_head_agreement_note = if stats.lm_head_agreement_samples > 0 {
            format!(
                " aip_lm_head_agreement_selected={}/{} aip_lm_head_agreement_raw={}/{} aip_lm_head_agreement_exact_in_topk={}/{} aip_lm_head_agreement_topk={}",
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
        let lm_head_repeat_margin_note = if stats.lm_head_repeat_margin_checks > 0 {
            let adaptive_note = if stats.lm_head_repeat_margin_adaptive_throttles > 0 {
                format!(
                    " aip_lm_head_repeat_margin_adaptive_throttles={} aip_lm_head_repeat_margin_min_effective_milli={}",
                    stats.lm_head_repeat_margin_adaptive_throttles,
                    stats.lm_head_repeat_margin_min_effective_milli
                )
            } else {
                String::new()
            };
            format!(
                " aip_lm_head_repeat_margin={}/{} aip_lm_head_repeat_margin_max_gap_milli={}",
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
                    " aip_lm_head_phrase_novelty_gap_skips={} aip_lm_head_phrase_novelty_max_gap_milli={}",
                    stats.lm_head_phrase_novelty_gap_skips,
                    stats.lm_head_phrase_novelty_max_gap_milli
                )
            } else {
                String::new()
            };
            let soft_note = if stats.lm_head_phrase_novelty_soft_choices > 0 {
                format!(
                    " aip_lm_head_phrase_novelty_soft_choices={}",
                    stats.lm_head_phrase_novelty_soft_choices
                )
            } else {
                String::new()
            };
            format!(
                " aip_lm_head_phrase_novelty={}/{} aip_lm_head_phrase_novelty_max_ngram={}",
                stats.lm_head_phrase_novelty_switches,
                stats.lm_head_phrase_novelty_checks,
                stats.lm_head_phrase_novelty_max_ngram
            ) + &gap_note
                + &soft_note
        } else {
            String::new()
        };
        format!(
            " aip_policy={} aip_calls={} aip_fallbacks={} aip_max_topk={} aip_skipped_madds={} aip_scratch_bytes={}",
            policy_str,
            stats.sparse_projection_calls,
            stats.exact_fallbacks,
            stats.max_selected_topk,
            stats.estimated_skipped_madds,
            stats.peak_scratch_bytes
        )
        + &lm_head_note
            + &column_cache_note
            + &input_tile_note
            + &lm_head_agreement_note
            + &lm_head_repeat_margin_note
            + &lm_head_phrase_novelty_note
    }
}

fn format_repetition_note(stats: rllm_runtime::RamaRepetitionStats) -> String {
    if stats.generated_tokens == 0 {
        String::new()
    } else {
        format!(
            " repeated_ratio={:.2} max_run={} unique={}/{}",
            stats.repeated_token_ratio,
            stats.max_repeated_token_run,
            stats.unique_generated_tokens,
            stats.generated_tokens
        )
    }
}

fn write_report(
    out: &str,
    file: &str,
    token_turns: &[Vec<usize>],
    max_new_tokens: usize,
    ctx: usize,
    valid: bool,
    rows: &[TokenBenchmarkRow],
) -> Result<()> {
    if let Some(parent) = Path::new(out).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut body = String::new();
    body.push_str("# Trial: R2 Token-Native Chat Session\n\n");
    body.push_str("Date: 2026-06-14\n");
    body.push_str("Owner: RLLM\n");
    body.push_str("Status: running\n");
    body.push_str("Folder: active\n\n");
    body.push_str("## Hypothesis\n\n");
    body.push_str("A persistent KV-cache session should reduce later-turn TTFT when compared against full token-history replay for the exact same token stream.\n\n");
    body.push_str("## Scope\n\n");
    body.push_str("- Mode: exact-lowram\n");
    body.push_str(&format!("- Model/artifact: `{file}`\n"));
    body.push_str("- Architecture: llama\n");
    body.push_str("- Target device/profile: single CPU, low RAM\n");
    body.push_str("- Expected bottleneck: full-history replay and memory bandwidth\n");
    body.push_str("- Bottleneck tag: cache locality\n\n");
    body.push_str("## Setup\n\n");
    body.push_str("Commands:\n\n");
    body.push_str("```bash\n");
    body.push_str(&replay_command(file, token_turns, max_new_tokens, ctx, out));
    body.push('\n');
    body.push_str("```\n\n");
    body.push_str("## Results\n\n");
    body.push_str("| turn | baseline input tokens | session input tokens | baseline generated | session generated | baseline TTFT | session TTFT | baseline decode ms | session decode ms | baseline e2e ms | session e2e ms | baseline decode tok/s | session decode tok/s | baseline e2e tok/s | session e2e tok/s | token match | history match | session phase timing | notes |\n");
    body.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|\n");
    for row in rows {
        let phase_note = format_phase_timing_note(row.session_result.metrics.phase_timings)
            + &format_rolling_note(row.session_result.metrics.rolling_stats)
            + &format_aip_note(row.session_result.metrics.experimental_speed_stats)
            + &format_repetition_note(row.session_result.metrics.repetition_stats);
        body.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.2} ms | {:.2} ms | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} | {} | {} | session_replayed={} flushed={} baseline_peak={} session_peak={} |\n",
            row.turn_index,
            row.baseline_input_tokens,
            row.session_input_tokens,
            row.baseline_timing.generated_tokens,
            row.session_result.metrics.generated_tokens,
            row.baseline_timing.ttft_ms,
            row.session_result.metrics.ttft_ms,
            row.baseline_timing.decode_ms,
            row.session_result.metrics.decode_ms,
            row.baseline_timing.end_to_end_ms,
            row.session_result.metrics.end_to_end_ms,
            row.baseline_timing.decode_tok_s,
            row.session_result.metrics.decode_tok_s,
            row.baseline_timing.end_to_end_tok_s,
            row.session_result.metrics.end_to_end_tok_s,
            row.generated_match.note,
            row.history_match.note,
            phase_note,
            row.session_result.metrics.replayed_tokens,
            row.session_result.metrics.flushed_pending_tokens,
            row.baseline_peak_transient_bytes,
            row.session_result.metrics.peak_transient_bytes
        ));
    }
    body.push_str("\n## Analysis\n\n");
    if valid {
        body.push_str("Baseline and session token streams matched for every measured turn.\n\n");
    } else {
        body.push_str("Baseline and session token streams diverged. Treat timing as inconclusive until the mismatch is explained.\n\n");
    }
    body.push_str("R3 phase timing is aggregated from LLaMA session adapter append calls for the measured turn. Treat it as coarse wall-clock evidence for choosing the next hot-path target, not cycle-level profiling.\n\n");
    body.push_str("## Decision\n\n");
    body.push_str("needs follow-up\n\n");
    body.push_str("Reason: review the active report and move it to success, failed, or inconclusive after comparing the measured rows.\n\n");
    body.push_str("Paper value:\n\n- not paper-worthy yet\n\n");
    body.push_str("## Next Experiment\n\n");
    body.push_str("Use the R3 phase timing columns to decide whether the next pass should target LM head/sampling, transformer matmul, KV-cache layout, or memory bandwidth.\n");
    fs::write(out, body)?;
    Ok(())
}

fn format_turn_id_args(turns: &[Vec<usize>]) -> String {
    turns
        .iter()
        .map(|turn| {
            let ids = turn
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            format!(" --turn-ids {ids}")
        })
        .collect::<String>()
}

fn replay_command(
    file: &str,
    token_turns: &[Vec<usize>],
    max_new_tokens: usize,
    ctx: usize,
    out: &str,
) -> String {
    format!(
        "cargo run --release -p rllm-cli -- chat-session-token {}{} --max-new-tokens {max_new_tokens} --ctx {ctx} --out {}",
        shell_quote(file),
        format_turn_id_args(token_turns),
        shell_quote(out)
    )
}

fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

fn parse_token_turns(turns: &[String]) -> Result<Vec<Vec<usize>>> {
    if turns.is_empty() {
        anyhow::bail!("chat-session-token requires at least one --turn-ids");
    }
    if turns.len() < 2 {
        anyhow::bail!("chat-session-token requires at least two --turn-ids values");
    }
    turns
        .iter()
        .enumerate()
        .map(|(turn_idx, raw)| parse_token_turn(turn_idx + 1, raw))
        .collect()
}

fn parse_token_turn(turn_idx: usize, raw: &str) -> Result<Vec<usize>> {
    if raw.trim().is_empty() {
        anyhow::bail!("chat-session-token turn {turn_idx} must not be empty");
    }
    let mut ids = Vec::new();
    for (item_idx, part) in raw.split(',').enumerate() {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            anyhow::bail!(
                "chat-session-token turn {turn_idx} has empty token id at position {}",
                item_idx + 1
            );
        }
        ids.push(trimmed.parse::<usize>().with_context(|| {
            format!(
                "invalid token id in chat-session-token turn {turn_idx} at position {}",
                item_idx + 1
            )
        })?);
    }
    Ok(ids)
}

fn validate_report_output_path(out: &str) -> Result<()> {
    let components = normalized_path_components(out);
    for window in components.windows(4) {
        if window[0] == "docs"
            && window[1] == "benchmarks"
            && window[2] == "trials"
            && matches!(window[3].as_str(), "success" | "failed" | "inconclusive")
        {
            anyhow::bail!(
                "chat-session-token writes active reports; use docs/benchmarks/trials/active/ and move the report after review"
            );
        }
    }
    Ok(())
}

fn normalized_path_components(path: &str) -> Vec<String> {
    let mut components = Vec::new();
    let normalized = path.replace('\\', "/");
    for part in normalized.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if components
                    .last()
                    .is_some_and(|component: &String| component != "..")
                {
                    components.pop();
                } else {
                    components.push(part.to_string());
                }
            }
            _ => components.push(part.to_ascii_lowercase()),
        }
    }
    components
}

#[cfg(test)]
mod tests {
    use super::{
        format_aip_note, format_phase_timing_note, format_repetition_note, format_rolling_note,
        parse_token_turns, token_match_summary, validate_report_output_path,
    };
    use rllm_runtime::{RamaSessionPhaseTimings, RamaTransformerPhaseTimings};

    #[test]
    fn parse_token_turns_accepts_comma_separated_turns() {
        let turns = vec!["1, 2,3".to_string(), "4".to_string()];

        assert_eq!(
            parse_token_turns(&turns).unwrap(),
            vec![vec![1, 2, 3], vec![4]]
        );
    }

    #[test]
    fn parse_token_turns_rejects_empty_and_invalid_values() {
        assert!(parse_token_turns(&[])
            .unwrap_err()
            .to_string()
            .contains("at least one"));
        assert!(parse_token_turns(&["1".to_string()])
            .unwrap_err()
            .to_string()
            .contains("at least two"));
        assert!(parse_token_turns(&["".to_string(), "1".to_string()])
            .unwrap_err()
            .to_string()
            .contains("empty"));
        assert!(parse_token_turns(&["1,,2".to_string(), "3".to_string()])
            .unwrap_err()
            .to_string()
            .contains("empty token id"));
        assert!(parse_token_turns(&["1,nope".to_string(), "3".to_string()])
            .unwrap_err()
            .to_string()
            .contains("invalid token id"));
    }

    #[test]
    fn token_report_output_rejects_reviewed_trial_folders() {
        validate_report_output_path("docs/benchmarks/trials/active/r2.md").unwrap();

        for path in [
            "docs/benchmarks/trials/success/r2.md",
            "docs/benchmarks/trials/active/../failed/r2.md",
            "docs/benchmarks/trials/Inconclusive/r2.md",
        ] {
            assert!(validate_report_output_path(path)
                .unwrap_err()
                .to_string()
                .contains("active reports"));
        }
    }

    #[test]
    fn token_match_summary_reports_match_and_mismatch() {
        let matched = token_match_summary(&[1, 2], &[1, 2]);
        assert!(matched.matched);
        assert_eq!(matched.note, "match");

        let mismatched = token_match_summary(&[1, 2], &[1, 3]);
        assert!(!mismatched.matched);
        assert!(mismatched.note.contains("baseline=[1, 2] session=[1, 3]"));
    }

    #[test]
    fn format_phase_timing_note_summarizes_decode_subphases() {
        let note = format_phase_timing_note(RamaSessionPhaseTimings {
            embedding_ms: 1.25,
            transformer_ms: 8.5,
            transformer_detail: RamaTransformerPhaseTimings {
                q_projection_ms: 1.0,
                k_projection_ms: 2.0,
                v_projection_ms: 3.0,
                o_projection_ms: 4.0,
                gate_projection_ms: 5.0,
                up_projection_ms: 6.0,
                down_projection_ms: 7.0,
                profiled_layers: 32,
                ..Default::default()
            },
            final_norm_ms: 0.75,
            lm_head_ms: 2.0,
        });

        assert!(note.contains("embedding=1.25ms"));
        assert!(note.contains("transformer=8.50ms"));
        assert!(note.contains("final_norm=0.75ms"));
        assert!(note.contains("lm_head=2.00ms"));
        assert!(note.contains("profiled_total=12.50ms"));
        assert!(note.contains("layers=32"));
        assert!(note.contains("q=1.00ms"));
        assert!(note.contains("k=2.00ms"));
        assert!(note.contains("v=3.00ms"));
        assert!(note.contains("o=4.00ms"));
        assert!(note.contains("gate=5.00ms"));
        assert!(note.contains("up=6.00ms"));
        assert!(note.contains("down=7.00ms"));
    }

    #[test]
    fn format_rolling_note_reports_nonzero_stats() {
        let note = format_rolling_note(rllm_runtime::RamaRollingStats {
            submitted_tasks: 4,
            worker_wakeups: 4,
            sequential_fallbacks: 2,
            peak_scratch_bytes: 32,
        });

        assert!(note.contains("rolling_tasks=4"));
        assert!(note.contains("rolling_fallbacks=2"));
    }

    #[test]
    fn format_aip_note_reports_nonzero_stats() {
        let note = format_aip_note(rllm_runtime::RamaExperimentalSpeedStats {
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
        });

        assert!(note.contains("aip_policy=quality"));
        assert!(note.contains("aip_calls=4"));
        assert!(note.contains("aip_fallbacks=1"));
        assert!(note.contains("aip_max_topk=128"));
        assert!(note.contains("aip_lm_head_rows=512/128256"));
        assert!(note.contains("aip_column_cache_hits=8"));
        assert!(note.contains("aip_column_cache_misses=4"));
        assert!(note.contains("aip_column_cache_resident=12/49152"));
        assert!(note.contains("aip_input_tile_reads=5"));
        assert!(note.contains("aip_input_tile_bytes=256"));
        assert!(note.contains("aip_lm_head_agreement_selected=4/10"));
        assert!(note.contains("aip_lm_head_agreement_raw=3/10"));
        assert!(note.contains("aip_lm_head_agreement_exact_in_topk=6/10"));
        assert!(note.contains("aip_lm_head_agreement_topk=8"));
        assert!(note.contains("aip_lm_head_repeat_margin=5/7"));
        assert!(note.contains("aip_lm_head_repeat_margin_max_gap_milli=125"));
        assert!(note.contains("aip_lm_head_repeat_margin_adaptive_throttles=2"));
        assert!(note.contains("aip_lm_head_repeat_margin_min_effective_milli=125"));
        assert!(note.contains("aip_lm_head_phrase_novelty=9/12"));
        assert!(note.contains("aip_lm_head_phrase_novelty_max_ngram=3"));
        assert!(note.contains("aip_lm_head_phrase_novelty_gap_skips=4"));
        assert!(note.contains("aip_lm_head_phrase_novelty_max_gap_milli=900"));
        assert!(note.contains("aip_lm_head_phrase_novelty_soft_choices=6"));
    }

    #[test]
    fn format_repetition_note_reports_nonzero_stats() {
        let note = format_repetition_note(rllm_runtime::RamaRepetitionStats {
            generated_tokens: 10,
            unique_generated_tokens: 5,
            max_repeated_token_run: 3,
            repeated_token_ratio: 0.25,
        });

        assert!(note.contains("repeated_ratio=0.25"));
        assert!(note.contains("max_run=3"));
        assert!(note.contains("unique=5/10"));
    }

    #[test]
    fn format_turn_id_args_replays_all_token_turns() {
        let turns = vec![vec![1, 2], vec![3]];

        assert_eq!(
            super::format_turn_id_args(&turns),
            " --turn-ids 1,2 --turn-ids 3"
        );
    }

    #[test]
    fn replay_command_uses_shell_quotes_not_markdown_backticks() {
        let command = super::replay_command(
            "models/space model.rllm",
            &[vec![1]],
            16,
            2048,
            "docs/benchmarks/trials/active/space out.md",
        );

        assert!(command.contains("'models/space model.rllm'"));
        assert!(command.contains("'docs/benchmarks/trials/active/space out.md'"));
        assert!(!command.contains('`'));
    }
}
