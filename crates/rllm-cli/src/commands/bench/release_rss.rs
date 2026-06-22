// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! Release RSS matrix: vary `--ctx` and `--max-new-tokens` for a text prompt
//! and record per-cell throughput and peak RSS. Native port of the former
//! `scripts/phase76_release_rss_benchmark.py`.

use super::report::{self, fmt_f};
use super::runner::{self, rllm_bin, Outcome};
use anyhow::Result;
use clap::Args as ClapArgs;
use std::path::PathBuf;

#[derive(ClapArgs)]
pub struct Args {
    /// Path to the .spsa artifact to benchmark.
    #[arg(long, default_value = "models/pythia-70m-phase76-16mb.spsa")]
    pub artifact: String,

    /// Prompt text sent to each run.
    #[arg(long, default_value = "Hello")]
    pub prompt: String,

    /// Comma-separated max-new-tokens values.
    #[arg(long, default_value = "1,4,8,16")]
    pub tokens: String,

    /// Comma-separated context lengths.
    #[arg(long, default_value = "128")]
    pub ctx: String,

    /// Memory budget for the low-RAM runtime.
    #[arg(long, default_value = "100mb")]
    pub memory_budget: String,

    /// Output directory for the CSV/Markdown matrices.
    #[arg(long, default_value = "target/phase76-bench")]
    pub out_dir: String,

    /// Extra argument appended to every `rllm run` invocation (repeatable).
    #[arg(long = "run-arg")]
    pub run_arg: Vec<String>,
}

struct Row {
    ctx: usize,
    tokens: usize,
    outcome: Outcome,
}

pub fn run(args: Args) -> Result<()> {
    let bin = rllm_bin()?;
    let ctx_values = report::parse_csv_ints(&args.ctx, "--ctx")?;
    let token_values = report::parse_csv_ints(&args.tokens, "--tokens")?;
    let out_dir = PathBuf::from(&args.out_dir);

    let mut rows: Vec<Row> = Vec::new();
    for ctx in &ctx_values {
        for tokens in &token_values {
            let mut cmd = vec![
                "run".to_string(),
                args.artifact.clone(),
                "--prompt".to_string(),
                args.prompt.clone(),
                "--max-new-tokens".to_string(),
                tokens.to_string(),
                "--ctx".to_string(),
                ctx.to_string(),
                "--memory-budget".to_string(),
                args.memory_budget.clone(),
            ];
            cmd.extend(args.run_arg.iter().cloned());
            let outcome = runner::run_child(&bin, &cmd)?;
            let failed = outcome.exit_code != 0;
            rows.push(Row {
                ctx: *ctx,
                tokens: *tokens,
                outcome,
            });
            write_outputs(&out_dir, &args, &rows)?;
            if failed {
                anyhow::bail!("benchmark failed for ctx={ctx}, tokens={tokens}; stopping");
            }
        }
    }

    println!("Wrote {}", out_dir.join("release_rss.csv").display());
    println!("Wrote {}", out_dir.join("release_rss.md").display());
    Ok(())
}

fn write_outputs(out_dir: &std::path::Path, args: &Args, rows: &[Row]) -> Result<()> {
    let header = [
        "ctx",
        "max_new_tokens",
        "exit_code",
        "real_seconds",
        "seconds_per_token",
        "max_rss_bytes",
        "max_rss_mib",
        "peak_transient",
        "generated_token_ids",
        "generated_text",
    ];
    let csv_rows: Vec<Vec<String>> = rows
        .iter()
        .map(|r| {
            vec![
                r.ctx.to_string(),
                r.tokens.to_string(),
                r.outcome.exit_code.to_string(),
                fmt_f(Some(r.outcome.real_seconds), 2),
                fmt_f(r.outcome.seconds_per_token(r.tokens), 2),
                report::fmt_u(r.outcome.max_rss_bytes),
                fmt_f(r.outcome.max_rss_mib(), 2),
                r.outcome.peak_transient().unwrap_or_default(),
                r.outcome.generated_token_ids().unwrap_or_default(),
                r.outcome.generated_text().unwrap_or_default(),
            ]
        })
        .collect();
    report::write_csv(&out_dir.join("release_rss.csv"), &header, &csv_rows)?;
    report::write_text(&out_dir.join("release_rss.md"), &markdown(args, rows))
}

fn markdown(args: &Args, rows: &[Row]) -> String {
    let mut md = String::new();
    md.push_str("# Release RSS Benchmark\n\n");
    md.push_str(&format!("- Artifact: `{}`\n", args.artifact));
    md.push_str(&format!("- Prompt: `{}`\n", args.prompt));
    md.push_str(&format!("- Memory budget: `{}`\n", args.memory_budget));
    md.push_str("- Measurement: child `rusage` peak RSS (no `/usr/bin/time` dependency).\n\n");
    md.push_str("| ctx | tokens | exit | real s | s/token | max RSS MiB | peak transient | generated token IDs | generated text |\n");
    md.push_str("|---:|---:|---:|---:|---:|---:|---|---|---|\n");
    for r in rows {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.ctx,
            r.tokens,
            r.outcome.exit_code,
            fmt_f(Some(r.outcome.real_seconds), 2),
            fmt_f(r.outcome.seconds_per_token(r.tokens), 2),
            fmt_f(r.outcome.max_rss_mib(), 2),
            r.outcome.peak_transient().unwrap_or_default(),
            r.outcome.generated_token_ids().unwrap_or_default(),
            r.outcome
                .generated_text()
                .unwrap_or_default()
                .replace('|', "\\|"),
        ));
    }
    md.push('\n');
    md
}
