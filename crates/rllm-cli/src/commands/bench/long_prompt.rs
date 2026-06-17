//! Long-prompt matrix: vary the actual `--token-ids` prompt length and the
//! generation length, reporting end-to-end throughput plus the RAMA timing
//! breakdown. Native port of `scripts/phase79d_long_prompt_benchmark.py`.

use super::report::{self, fmt_f};
use super::runner::{self, rllm_bin, Outcome, Timing};
use anyhow::Result;
use clap::Args as ClapArgs;
use std::path::{Path, PathBuf};

/// RAMA timing summary keys surfaced as columns (label, json key).
const TIMING_COLUMNS: &[(&str, &str)] = &[
    ("prefill_ms", "prefill_ms"),
    ("decode_ms", "decode_ms"),
    ("final_norm_ms", "final_norm_ms"),
    ("lm_head_ms", "lm_head_ms"),
    ("sampling_ms", "sampling_ms"),
    ("prefill_embedding_ms", "prefill_embedding_ms"),
    ("prefill_attention_ms", "prefill_attention_ms"),
    (
        "prefill_attention_score_context_ms",
        "prefill_attention_score_context_ms",
    ),
    ("prefill_mlp_ms", "prefill_mlp_ms"),
    (
        "prefill_mlp_output_projection_ms",
        "prefill_mlp_output_projection_ms",
    ),
];

#[derive(ClapArgs)]
pub struct Args {
    /// Path to the .rllm artifact to benchmark.
    #[arg(
        long,
        default_value = "models/pythia-70m-low-ram-fast-raw-tileblocks.rllm"
    )]
    pub artifact: String,

    /// Comma-separated actual prompt token counts.
    #[arg(long, default_value = "1,128,512,1024")]
    pub input_tokens: String,

    /// Comma-separated generation lengths.
    #[arg(long, default_value = "1,4,16")]
    pub max_new_tokens: String,

    /// Context capacity; must exceed input + generated tokens.
    #[arg(long, default_value_t = 2048)]
    pub ctx: usize,

    /// Memory budget for the low-RAM runtime.
    #[arg(long, default_value = "100mb")]
    pub memory_budget: String,

    /// Runtime integrity policy: strict or verify-once.
    #[arg(long, default_value = "verify-once")]
    pub rama_integrity: String,

    /// Optional fixed prefill chunk window (real input tokens).
    #[arg(long)]
    pub rama_prefill_chunk_tokens: Option<usize>,

    /// Output directory for the CSV/Markdown matrices and timing JSON.
    #[arg(long, default_value = "target/long-prompt")]
    pub out_dir: String,
}

struct Row {
    input_tokens: usize,
    new_tokens: usize,
    outcome: Outcome,
    timing: Timing,
}

pub fn run(args: Args) -> Result<()> {
    let bin = rllm_bin()?;
    let input_lengths = report::parse_csv_ints(&args.input_tokens, "--input-tokens")?;
    let generation_lengths = report::parse_csv_ints(&args.max_new_tokens, "--max-new-tokens")?;
    let out_dir = PathBuf::from(&args.out_dir);
    let timing_dir = out_dir.join("timing");

    let mut rows: Vec<Row> = Vec::new();
    for input_tokens in &input_lengths {
        for new_tokens in &generation_lengths {
            if input_tokens + new_tokens > args.ctx {
                anyhow::bail!(
                    "input_tokens + max_new_tokens exceeds ctx: {input_tokens}+{new_tokens}>{}",
                    args.ctx
                );
            }
            let timing_path = timing_dir.join(format!("input{input_tokens}_new{new_tokens}.json"));
            let cmd = build_command(&args, *input_tokens, *new_tokens, &timing_path)?;
            let outcome = runner::run_child(&bin, &cmd)?;
            let failed = outcome.exit_code != 0;
            let timing = Timing::read(Some(&timing_path));
            rows.push(Row {
                input_tokens: *input_tokens,
                new_tokens: *new_tokens,
                outcome,
                timing,
            });
            write_outputs(&out_dir, &args, &rows)?;
            if failed {
                anyhow::bail!(
                    "benchmark failed for input_tokens={input_tokens}, max_new_tokens={new_tokens}; stopping"
                );
            }
        }
    }

    println!("Wrote {}", out_dir.join("long_prompt.csv").display());
    println!("Wrote {}", out_dir.join("long_prompt.md").display());
    Ok(())
}

fn build_command(
    args: &Args,
    input_tokens: usize,
    new_tokens: usize,
    timing_path: &Path,
) -> Result<Vec<String>> {
    let mut cmd = vec![
        "run".to_string(),
        args.artifact.clone(),
        "--token-ids".to_string(),
        report::token_ids_arg(input_tokens)?,
        "--max-new-tokens".to_string(),
        new_tokens.to_string(),
        "--ctx".to_string(),
        args.ctx.to_string(),
        "--memory-budget".to_string(),
        args.memory_budget.clone(),
        "--rama-integrity".to_string(),
        args.rama_integrity.clone(),
        "--rama-timing".to_string(),
        timing_path.to_string_lossy().to_string(),
    ];
    if let Some(chunk) = args.rama_prefill_chunk_tokens {
        cmd.push("--rama-prefill-chunk-tokens".to_string());
        cmd.push(chunk.to_string());
    }
    Ok(cmd)
}

fn write_outputs(out_dir: &Path, args: &Args, rows: &[Row]) -> Result<()> {
    let mut header: Vec<&str> = vec![
        "input_tokens",
        "max_new_tokens",
        "ctx",
        "exit_code",
        "real_seconds",
        "seconds_per_generated_token",
        "generated_tokens_per_second",
        "max_rss_bytes",
        "max_rss_mib",
        "peak_transient",
        "generated_token_ids",
    ];
    header.extend(TIMING_COLUMNS.iter().map(|(label, _)| *label));
    header.push("prefill_chunks");
    header.push("decode_steps");

    let csv_rows: Vec<Vec<String>> = rows.iter().map(|r| csv_row(r, args.ctx)).collect();
    report::write_csv(&out_dir.join("long_prompt.csv"), &header, &csv_rows)?;
    report::write_text(&out_dir.join("long_prompt.md"), &markdown(args, rows))
}

fn csv_row(r: &Row, ctx: usize) -> Vec<String> {
    let mut row = vec![
        r.input_tokens.to_string(),
        r.new_tokens.to_string(),
        ctx.to_string(),
        r.outcome.exit_code.to_string(),
        fmt_f(Some(r.outcome.real_seconds), 2),
        fmt_f(r.outcome.seconds_per_token(r.new_tokens), 2),
        fmt_f(r.outcome.tokens_per_second(r.new_tokens), 4),
        report::fmt_u(r.outcome.max_rss_bytes),
        fmt_f(r.outcome.max_rss_mib(), 2),
        r.outcome.peak_transient().unwrap_or_default(),
        r.outcome.generated_token_ids().unwrap_or_default(),
    ];
    for (_, key) in TIMING_COLUMNS {
        row.push(fmt_f(r.timing.f(key), 2));
    }
    row.push(report::fmt_u(r.timing.u("prefill_chunks")));
    row.push(report::fmt_u(r.timing.u("decode_steps")));
    row
}

fn markdown(args: &Args, rows: &[Row]) -> String {
    let ok: Vec<&Row> = rows.iter().filter(|r| r.outcome.exit_code == 0).collect();
    let tps: Vec<f64> = ok
        .iter()
        .filter_map(|r| r.outcome.tokens_per_second(r.new_tokens))
        .collect();
    let rss: Vec<f64> = ok.iter().filter_map(|r| r.outcome.max_rss_mib()).collect();

    let mut md = String::new();
    md.push_str("# Long-Prompt Benchmark\n\n");
    md.push_str(&format!("- Artifact: `{}`\n", args.artifact));
    md.push_str(&format!("- Runtime integrity: `{}`\n", args.rama_integrity));
    md.push_str(&format!("- Memory budget: `{}`\n", args.memory_budget));
    md.push_str(&format!("- Context capacity: `{}`\n", args.ctx));
    md.push_str("- Input: deterministic fixed token IDs (not tokenizer text).\n\n");
    md.push_str("## Summary\n\n");
    md.push_str(&format!(
        "- Successful rows: `{}/{}`\n",
        ok.len(),
        rows.len()
    ));
    md.push_str(&summary_line("generated tokens/sec", &tps, 3));
    md.push_str(&summary_line("max RSS MiB", &rss, 2));
    md.push_str("\n## Rows\n\n");
    md.push_str("| input | new | real s | end-to-end tok/s | max RSS MiB | peak transient | prefill ms | decode ms | lm_head ms | prefill chunks | exit |\n");
    md.push_str("|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n");
    for r in rows {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.input_tokens,
            r.new_tokens,
            fmt_f(Some(r.outcome.real_seconds), 2),
            fmt_f(r.outcome.tokens_per_second(r.new_tokens), 3),
            fmt_f(r.outcome.max_rss_mib(), 2),
            r.outcome.peak_transient().unwrap_or_default(),
            fmt_f(r.timing.f("prefill_ms"), 2),
            fmt_f(r.timing.f("decode_ms"), 2),
            fmt_f(r.timing.f("lm_head_ms"), 2),
            report::fmt_u(r.timing.u("prefill_chunks")),
            r.outcome.exit_code,
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
