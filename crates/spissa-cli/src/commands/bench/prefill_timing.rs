// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! Prefill timing sweep: compare full-prompt prefill against opt-in RAMA
//! chunked prefill windows under real `--token-ids` input lengths. Native port
//! of `scripts/phase79e_prefill_timing_benchmark.py`.

use super::report::{self, fmt_f};
use super::runner::{self, rllm_bin, Outcome, Timing};
use anyhow::Result;
use clap::Args as ClapArgs;
use std::path::{Path, PathBuf};

#[derive(ClapArgs)]
pub struct Args {
    /// Path to the .spsa artifact to benchmark.
    #[arg(
        long,
        default_value = "models/pythia-70m-low-ram-fast-raw-tileblocks.spsa"
    )]
    pub artifact: String,

    /// Comma-separated actual prompt token counts.
    #[arg(long, default_value = "512,1024")]
    pub input_tokens: String,

    /// Comma-separated generation lengths.
    #[arg(long, default_value = "16")]
    pub max_new_tokens: String,

    /// Comma-separated prefill windows; use `full` (or `none`) for one pass.
    #[arg(long, default_value = "full,128")]
    pub prefill_chunks: String,

    /// Context capacity; must exceed input + generated tokens.
    #[arg(long, default_value_t = 2048)]
    pub ctx: usize,

    /// Memory budget for the low-RAM runtime.
    #[arg(long, default_value = "100mb")]
    pub memory_budget: String,

    /// Runtime integrity policy: strict or verify-once.
    #[arg(long, default_value = "verify-once")]
    pub rama_integrity: String,

    /// Output directory for the CSV/Markdown matrices and timing JSON.
    #[arg(long, default_value = "target/prefill-timing")]
    pub out_dir: String,
}

struct Row {
    input_tokens: usize,
    new_tokens: usize,
    /// `None` means full-prompt prefill.
    chunk: Option<usize>,
    outcome: Outcome,
    timing: Timing,
}

impl Row {
    fn chunk_label(&self) -> String {
        self.chunk
            .map(|c| c.to_string())
            .unwrap_or_else(|| "full".to_string())
    }
}

pub fn run(args: Args) -> Result<()> {
    let bin = rllm_bin()?;
    let input_lengths = report::parse_csv_ints(&args.input_tokens, "--input-tokens")?;
    let generation_lengths = report::parse_csv_ints(&args.max_new_tokens, "--max-new-tokens")?;
    let chunks = parse_chunks(&args.prefill_chunks)?;
    let out_dir = PathBuf::from(&args.out_dir);
    let timing_dir = out_dir.join("timing");

    let mut rows: Vec<Row> = Vec::new();
    for input_tokens in &input_lengths {
        for new_tokens in &generation_lengths {
            for chunk in &chunks {
                let label = chunk
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "full".to_string());
                let timing_path = timing_dir.join(format!(
                    "input{input_tokens}_new{new_tokens}_chunk{label}.json"
                ));
                let cmd = build_command(&args, *input_tokens, *new_tokens, *chunk, &timing_path)?;
                let outcome = runner::run_child(&bin, &cmd)?;
                let failed = outcome.exit_code != 0;
                let timing = Timing::read(Some(&timing_path));
                let row = Row {
                    input_tokens: *input_tokens,
                    new_tokens: *new_tokens,
                    chunk: *chunk,
                    outcome,
                    timing,
                };
                println!(
                    "input={} new={} chunk={} exit={} elapsed={:.2}s tok/s={} prefill_ms={}",
                    row.input_tokens,
                    row.new_tokens,
                    row.chunk_label(),
                    row.outcome.exit_code,
                    row.outcome.real_seconds,
                    fmt_f(row.outcome.tokens_per_second(row.new_tokens), 3),
                    fmt_f(row.timing.f("prefill_ms"), 2),
                );
                rows.push(row);
                write_outputs(&out_dir, &rows)?;
                if failed {
                    anyhow::bail!(
                        "benchmark failed for input={input_tokens}, new={new_tokens}, chunk={label}; stopping"
                    );
                }
            }
        }
    }

    println!("Wrote {}", out_dir.join("prefill_timing.csv").display());
    println!("Wrote {}", out_dir.join("prefill_timing.md").display());
    Ok(())
}

fn parse_chunks(raw: &str) -> Result<Vec<Option<usize>>> {
    let mut chunks = Vec::new();
    for item in raw.split(',') {
        let item = item.trim().to_lowercase();
        if item.is_empty() {
            continue;
        }
        if item == "full" || item == "none" {
            chunks.push(None);
        } else {
            let value: usize = item.parse().map_err(|_| {
                anyhow::anyhow!("--prefill-chunks expects integers or full: {raw:?}")
            })?;
            chunks.push(Some(value));
        }
    }
    if chunks.is_empty() {
        anyhow::bail!("--prefill-chunks must contain at least one value");
    }
    Ok(chunks)
}

fn build_command(
    args: &Args,
    input_tokens: usize,
    new_tokens: usize,
    chunk: Option<usize>,
    timing_path: &Path,
) -> Result<Vec<String>> {
    let mut cmd = vec![
        "run".to_string(),
        args.artifact.clone(),
        "--mode".to_string(),
        "tile-stream".to_string(),
        "--ctx".to_string(),
        args.ctx.to_string(),
        "--memory-budget".to_string(),
        args.memory_budget.clone(),
        "--token-ids".to_string(),
        report::token_ids_arg(input_tokens)?,
        "--max-new-tokens".to_string(),
        new_tokens.to_string(),
        "--rama-integrity".to_string(),
        args.rama_integrity.clone(),
        "--rama-timing".to_string(),
        timing_path.to_string_lossy().to_string(),
    ];
    if let Some(chunk) = chunk {
        cmd.push("--rama-prefill-chunk-tokens".to_string());
        cmd.push(chunk.to_string());
    }
    Ok(cmd)
}

fn write_outputs(out_dir: &Path, rows: &[Row]) -> Result<()> {
    let header = [
        "input_tokens",
        "max_new_tokens",
        "prefill_chunk_tokens",
        "exit_code",
        "elapsed_seconds",
        "generated_tokens_per_second",
        "max_rss_mib",
        "peak_transient",
        "context_memory",
        "prefill_ms",
        "decode_ms",
        "lm_head_ms",
        "prefill_chunks",
        "decode_steps",
        "max_prefill_chunk_tokens",
        "generated_token_ids",
    ];
    let csv_rows: Vec<Vec<String>> = rows
        .iter()
        .map(|r| {
            vec![
                r.input_tokens.to_string(),
                r.new_tokens.to_string(),
                r.chunk_label(),
                r.outcome.exit_code.to_string(),
                fmt_f(Some(r.outcome.real_seconds), 2),
                fmt_f(r.outcome.tokens_per_second(r.new_tokens), 4),
                fmt_f(r.outcome.max_rss_mib(), 2),
                r.outcome.peak_transient().unwrap_or_default(),
                r.outcome.context_memory().unwrap_or_default(),
                fmt_f(r.timing.f("prefill_ms"), 2),
                fmt_f(r.timing.f("decode_ms"), 2),
                fmt_f(r.timing.f("lm_head_ms"), 2),
                report::fmt_u(r.timing.u("prefill_chunks")),
                report::fmt_u(r.timing.u("decode_steps")),
                report::fmt_u(r.timing.u("max_prefill_chunk_tokens")),
                r.outcome.generated_token_ids().unwrap_or_default(),
            ]
        })
        .collect();
    report::write_csv(&out_dir.join("prefill_timing.csv"), &header, &csv_rows)?;
    report::write_text(&out_dir.join("prefill_timing.md"), &markdown(rows))
}

fn markdown(rows: &[Row]) -> String {
    let mut md = String::new();
    md.push_str("# Prefill Timing Benchmark\n\n");
    md.push_str("| input | new | chunk | elapsed s | tok/s | RSS MiB | context | transient | prefill ms | decode ms | lm_head ms | prefill chunks |\n");
    md.push_str("|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|\n");
    for r in rows {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.input_tokens,
            r.new_tokens,
            r.chunk_label(),
            fmt_f(Some(r.outcome.real_seconds), 2),
            fmt_f(r.outcome.tokens_per_second(r.new_tokens), 3),
            fmt_f(r.outcome.max_rss_mib(), 2),
            r.outcome.context_memory().unwrap_or_default(),
            r.outcome.peak_transient().unwrap_or_default(),
            fmt_f(r.timing.f("prefill_ms"), 2),
            fmt_f(r.timing.f("decode_ms"), 2),
            fmt_f(r.timing.f("lm_head_ms"), 2),
            report::fmt_u(r.timing.u("prefill_chunks")),
        ));
    }
    md.push('\n');
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_chunks_handles_full_and_ints() {
        let chunks = parse_chunks("full,64, none ,128").unwrap();
        assert_eq!(chunks, vec![None, Some(64), None, Some(128)]);
        assert!(parse_chunks("").is_err());
        assert!(parse_chunks("abc").is_err());
    }
}
