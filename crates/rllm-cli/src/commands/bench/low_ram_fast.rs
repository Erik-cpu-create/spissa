// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! Low-ram-fast matrix: pack a raw tile-block artifact, optionally verify it,
//! then run the release RSS matrix under a runtime integrity policy and emit a
//! throughput/RSS summary. Native port of
//! `scripts/phase79c_low_ram_fast_benchmark.py`.

use super::report::{self, fmt_f};
use super::runner::{self, rllm_bin, Outcome};
use anyhow::Result;
use clap::Args as ClapArgs;
use std::path::{Path, PathBuf};

#[derive(ClapArgs)]
pub struct Args {
    /// Source safetensors model to pack.
    #[arg(long, default_value = "models/pythia-70m/model.safetensors")]
    pub source: String,

    /// HuggingFace config.json to embed.
    #[arg(long, default_value = "models/pythia-70m/config.json")]
    pub config: String,

    /// HuggingFace tokenizer.json to embed.
    #[arg(long, default_value = "models/pythia-70m/tokenizer.json")]
    pub tokenizer: String,

    /// Output/benchmark .spsa artifact.
    #[arg(
        long,
        default_value = "models/pythia-70m-low-ram-fast-raw-tileblocks.spsa"
    )]
    pub artifact: String,

    /// Pack codec policy (raw keeps the compute-ready low-ram-fast layout).
    #[arg(long, default_value = "raw")]
    pub codec: String,

    /// Tensor elements per packed tile/block chunk.
    #[arg(long, default_value_t = 65536)]
    pub tile_block_elements: usize,

    /// Prompt text sent to each run.
    #[arg(long, default_value = "Hello")]
    pub prompt: String,

    /// Comma-separated max-new-tokens values.
    #[arg(long, default_value = "1,4,8,16")]
    pub tokens: String,

    /// Comma-separated context lengths.
    #[arg(long, default_value = "128,512,1024")]
    pub ctx: String,

    /// Memory budget for the low-RAM runtime.
    #[arg(long, default_value = "100mb")]
    pub memory_budget: String,

    /// Runtime integrity policy: strict or verify-once.
    #[arg(long, default_value = "strict")]
    pub rama_integrity: String,

    /// Output directory for the CSV/Markdown matrices.
    #[arg(long, default_value = "target/low-ram-fast")]
    pub out_dir: String,

    /// Skip the pack step (reuse an existing artifact).
    #[arg(long)]
    pub skip_pack: bool,

    /// Skip the lossless verify step.
    #[arg(long)]
    pub skip_verify: bool,
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

    if !args.skip_pack {
        pack(&bin, &args)?;
    }
    if !args.skip_verify {
        runner::run_checked(
            &bin,
            &[
                "verify".to_string(),
                args.source.clone(),
                args.artifact.clone(),
            ],
        )?;
    }

    let mut rows: Vec<Row> = Vec::new();
    for ctx in &ctx_values {
        for tokens in &token_values {
            let cmd = vec![
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
                "--rama-integrity".to_string(),
                args.rama_integrity.clone(),
            ];
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

    println!("Wrote {}", out_dir.join("low_ram_fast.csv").display());
    println!("Wrote {}", out_dir.join("low_ram_fast.md").display());
    Ok(())
}

fn pack(bin: &Path, args: &Args) -> Result<()> {
    runner::run_checked(
        bin,
        &[
            "pack".to_string(),
            args.source.clone(),
            "--out".to_string(),
            args.artifact.clone(),
            "--codec".to_string(),
            args.codec.clone(),
            "--tile-block-elements".to_string(),
            args.tile_block_elements.to_string(),
            "--config".to_string(),
            args.config.clone(),
            "--tokenizer".to_string(),
            args.tokenizer.clone(),
        ],
    )
}

fn write_outputs(out_dir: &Path, args: &Args, rows: &[Row]) -> Result<()> {
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
    report::write_csv(&out_dir.join("low_ram_fast.csv"), &header, &csv_rows)?;
    report::write_text(&out_dir.join("low_ram_fast.md"), &markdown(args, rows))
}

fn markdown(args: &Args, rows: &[Row]) -> String {
    let ok: Vec<&Row> = rows.iter().filter(|r| r.outcome.exit_code == 0).collect();
    let spt: Vec<f64> = ok
        .iter()
        .filter_map(|r| r.outcome.seconds_per_token(r.tokens))
        .collect();
    let tps: Vec<f64> = ok
        .iter()
        .filter_map(|r| r.outcome.tokens_per_second(r.tokens))
        .collect();
    let rss: Vec<f64> = ok.iter().filter_map(|r| r.outcome.max_rss_mib()).collect();

    let mut md = String::new();
    md.push_str("# Low-RAM-Fast Benchmark\n\n");
    md.push_str(&format!("- Artifact: `{}`\n", args.artifact));
    md.push_str(&format!("- Codec policy: `{}`\n", args.codec));
    md.push_str(&format!("- Runtime integrity: `{}`\n", args.rama_integrity));
    md.push_str(&format!(
        "- Tile-block elements: `{}`\n\n",
        args.tile_block_elements
    ));
    md.push_str("## Summary\n\n");
    md.push_str(&format!(
        "- Successful rows: `{}/{}`\n",
        ok.len(),
        rows.len()
    ));
    md.push_str(&summary_line("seconds/token", &spt, 2));
    md.push_str(&summary_line("tokens/second", &tps, 3));
    md.push_str(&summary_line("max RSS MiB", &rss, 2));
    md.push_str("\n## Rows\n\n");
    md.push_str("| ctx | tokens | exit | real s | s/token | tok/s | max RSS MiB | peak transient | generated token IDs |\n");
    md.push_str("|---:|---:|---:|---:|---:|---:|---:|---|---|\n");
    for r in rows {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.ctx,
            r.tokens,
            r.outcome.exit_code,
            fmt_f(Some(r.outcome.real_seconds), 2),
            fmt_f(r.outcome.seconds_per_token(r.tokens), 2),
            fmt_f(r.outcome.tokens_per_second(r.tokens), 3),
            fmt_f(r.outcome.max_rss_mib(), 2),
            r.outcome.peak_transient().unwrap_or_default(),
            r.outcome.generated_token_ids().unwrap_or_default(),
        ));
    }
    md.push('\n');
    md
}

fn summary_line(label: &str, values: &[f64], precision: usize) -> String {
    match (
        report::min(values),
        report::max(values),
        report::mean(values),
    ) {
        (Some(lo), Some(hi), Some(avg)) => format!(
            "- {label}: `{}`–`{}`; avg `{}`\n",
            fmt_f(Some(lo), precision),
            fmt_f(Some(hi), precision),
            fmt_f(Some(avg), precision),
        ),
        _ => format!("- {label}: n/a\n"),
    }
}
