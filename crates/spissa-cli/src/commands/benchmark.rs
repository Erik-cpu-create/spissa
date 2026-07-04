// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

use anyhow::{Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub struct BenchmarkOptions {
    pub file: String,
    pub prompts: Vec<String>,
    pub runs: usize,
    pub ctx: usize,
    pub max_new_tokens: usize,
    pub out: String,
    pub profile_phases: bool,
    pub target_min_tok_s: f64,
    pub target_max_tok_s: f64,
    pub common_env: Vec<String>,
    pub control_env: Vec<String>,
    pub candidate_env: Vec<String>,
    pub control_name: String,
    pub candidate_name: String,
    pub runner: Option<String>,
}

#[derive(Debug, Clone)]
struct BenchmarkMetrics {
    prefill_seconds: f64,
    decode_tok_s: f64,
    end_to_end_tok_s: f64,
    generated_tokens: usize,
    context_tokens: usize,
    peak_transient_bytes: usize,
    repetition_ratio: f64,
    max_run: usize,
    unique_tokens: usize,
    total_tokens_for_repetition: usize,
}

#[derive(Debug, Clone)]
struct BenchmarkRun {
    variant: String,
    run_index: usize,
    prompt_index: usize,
    prompt: String,
    metrics: BenchmarkMetrics,
    stdout: String,
}

#[derive(Debug, Clone)]
struct VariantSummary {
    name: String,
    runs: usize,
    floor_accepted: bool,
    band_accepted: bool,
    min_decode_tok_s: f64,
    max_decode_tok_s: f64,
    avg_decode_tok_s: f64,
    avg_unique_tokens: f64,
    avg_repetition_ratio: f64,
}

pub fn run(options: BenchmarkOptions) -> Result<()> {
    validate_options(&options)?;
    let runner = options
        .runner
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or(resolve_default_llama_test_runner()?);
    let common_env = parse_env_assignments(&options.common_env)?;
    let control_env = parse_env_assignments(&options.control_env)?;
    let candidate_env = parse_env_assignments(&options.candidate_env)?;
    let plan = alternating_prompt_run_plan(options.runs, options.prompts.len());

    let mut runs = Vec::new();
    for (variant_kind, run_index, prompt_index) in plan {
        let (variant_name, variant_env) = if variant_kind == "control" {
            (&options.control_name, &control_env)
        } else {
            (&options.candidate_name, &candidate_env)
        };
        let prompt = &options.prompts[prompt_index];
        println!(
            "run {variant_name} #{run_index} prompt #{}",
            prompt_index + 1
        );
        runs.push(run_llama_test_variant(
            &runner,
            &options,
            variant_name,
            run_index,
            prompt_index,
            prompt,
            &common_env,
            variant_env,
        )?);
    }

    let control_runs = runs_for_variant(&runs, &options.control_name);
    let candidate_runs = runs_for_variant(&runs, &options.candidate_name);
    let control_summary = summarize_variant(
        &options.control_name,
        &control_runs,
        options.target_min_tok_s,
        options.target_max_tok_s,
    )?;
    let candidate_summary = summarize_variant(
        &options.candidate_name,
        &candidate_runs,
        options.target_min_tok_s,
        options.target_max_tok_s,
    )?;
    write_markdown_report(
        &options,
        &runner,
        &runs,
        &[control_summary.clone(), candidate_summary.clone()],
    )?;

    println!("Benchmark report: {}", options.out);
    println!(
        "{}: avg_decode={:.2} min={:.2} max={:.2} floor_accepted={} band_accepted={}",
        control_summary.name,
        control_summary.avg_decode_tok_s,
        control_summary.min_decode_tok_s,
        control_summary.max_decode_tok_s,
        control_summary.floor_accepted,
        control_summary.band_accepted
    );
    println!(
        "{}: avg_decode={:.2} min={:.2} max={:.2} floor_accepted={} band_accepted={}",
        candidate_summary.name,
        candidate_summary.avg_decode_tok_s,
        candidate_summary.min_decode_tok_s,
        candidate_summary.max_decode_tok_s,
        candidate_summary.floor_accepted,
        candidate_summary.band_accepted
    );
    Ok(())
}

fn validate_options(options: &BenchmarkOptions) -> Result<()> {
    if options.runs == 0 {
        anyhow::bail!("--runs must be greater than zero");
    }
    if options.ctx == 0 {
        anyhow::bail!("--ctx must be greater than zero");
    }
    if options.max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    if options.prompts.is_empty() {
        anyhow::bail!("at least one --prompt is required");
    }
    for prompt in &options.prompts {
        if prompt.trim().is_empty() {
            anyhow::bail!("--prompt must not be empty");
        }
    }
    if options.target_min_tok_s <= 0.0 || options.target_max_tok_s < options.target_min_tok_s {
        anyhow::bail!("target token/s band is invalid");
    }
    Ok(())
}

fn resolve_default_llama_test_runner() -> Result<PathBuf> {
    let current_exe = std::env::current_exe().context("failed to resolve current executable")?;
    let Some(parent) = current_exe.parent() else {
        anyhow::bail!("failed to resolve executable directory");
    };
    Ok(parent.join("llama-test"))
}

fn parse_env_assignments(assignments: &[String]) -> Result<Vec<(String, String)>> {
    assignments
        .iter()
        .map(|assignment| {
            let Some((key, value)) = assignment.split_once('=') else {
                anyhow::bail!("env assignment must use KEY=VALUE: {assignment}");
            };
            let key = key.trim();
            if key.is_empty() {
                anyhow::bail!("env assignment key must not be empty: {assignment}");
            }
            Ok((key.to_string(), value.to_string()))
        })
        .collect()
}

#[cfg(test)]
fn alternating_run_plan(runs: usize) -> Vec<(String, usize)> {
    alternating_prompt_run_plan(runs, 1)
        .into_iter()
        .map(|(variant, run_index, _)| (variant, run_index))
        .collect()
}

fn alternating_prompt_run_plan(runs: usize, prompt_count: usize) -> Vec<(String, usize, usize)> {
    let mut plan = Vec::with_capacity(runs.saturating_mul(2));
    for run_index in 1..=runs {
        for prompt_index in 0..prompt_count {
            plan.push(("control".to_string(), run_index, prompt_index));
            plan.push(("candidate".to_string(), run_index, prompt_index));
        }
    }
    plan
}

fn run_llama_test_variant(
    runner: &Path,
    options: &BenchmarkOptions,
    variant: &str,
    run_index: usize,
    prompt_index: usize,
    prompt: &str,
    common_env: &[(String, String)],
    variant_env: &[(String, String)],
) -> Result<BenchmarkRun> {
    let mut command = Command::new(runner);
    command
        .arg("--model")
        .arg(&options.file)
        .arg("--ctx")
        .arg(options.ctx.to_string())
        .arg("--max-new-tokens")
        .arg(options.max_new_tokens.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if options.profile_phases {
        command.arg("--profile-phases");
    }
    for (key, value) in common_env.iter().chain(variant_env.iter()) {
        command.env(key, value);
    }

    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to start benchmark runner {}",
            runner.to_string_lossy()
        )
    })?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .context("failed to open runner stdin")?;
        writeln!(stdin, "{prompt}")?;
        writeln!(stdin, "exit")?;
    }
    let output = child
        .wait_with_output()
        .context("failed to collect runner output")?;
    if !output.status.success() {
        anyhow::bail!(
            "benchmark runner failed for {variant} #{run_index}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let metrics = parse_llama_test_output(&stdout)
        .with_context(|| format!("failed to parse runner output for {variant} #{run_index}"))?;
    Ok(BenchmarkRun {
        variant: variant.to_string(),
        run_index,
        prompt_index,
        prompt: prompt.to_string(),
        metrics,
        stdout,
    })
}

fn parse_llama_test_output(output: &str) -> Result<BenchmarkMetrics> {
    let metrics_line = output
        .lines()
        .find(|line| line.contains("[TTFT/Prefill:") && line.contains("Decode:"))
        .context("runner output did not contain llama-test metrics line")?;
    let line = metrics_line
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');
    Ok(BenchmarkMetrics {
        prefill_seconds: parse_float_metric(line, "TTFT/Prefill:", "s")?,
        decode_tok_s: parse_float_metric(line, "Decode:", "tok/s")?,
        end_to_end_tok_s: parse_float_metric(line, "E2E:", "tok/s")?,
        generated_tokens: parse_usize_metric(line, "Total:", "tokens")?,
        context_tokens: parse_usize_metric(line, "Context:", "tokens")?,
        peak_transient_bytes: parse_usize_metric(line, "Peak:", "bytes")?,
        repetition_ratio: parse_float_after(line, "ratio=")?,
        max_run: parse_usize_after(line, "max_run=")?,
        unique_tokens: parse_unique_tokens(line)?.0,
        total_tokens_for_repetition: parse_unique_tokens(line)?.1,
    })
}

fn parse_float_metric(line: &str, label: &str, suffix: &str) -> Result<f64> {
    parse_metric_token(line, label, suffix)?
        .parse::<f64>()
        .with_context(|| format!("failed to parse metric {label}"))
}

fn parse_usize_metric(line: &str, label: &str, suffix: &str) -> Result<usize> {
    parse_metric_token(line, label, suffix)?
        .parse::<usize>()
        .with_context(|| format!("failed to parse metric {label}"))
}

fn parse_metric_token<'a>(line: &'a str, label: &str, suffix: &str) -> Result<&'a str> {
    let start = line
        .find(label)
        .with_context(|| format!("missing metric label {label}"))?
        + label.len();
    let rest = line[start..].trim_start();
    let end = rest
        .find(suffix)
        .with_context(|| format!("missing metric suffix {suffix} for {label}"))?;
    Ok(rest[..end].trim())
}

fn parse_float_after(line: &str, label: &str) -> Result<f64> {
    parse_word_after(line, label)?
        .parse::<f64>()
        .with_context(|| format!("failed to parse {label}"))
}

fn parse_usize_after(line: &str, label: &str) -> Result<usize> {
    parse_word_after(line, label)?
        .parse::<usize>()
        .with_context(|| format!("failed to parse {label}"))
}

fn parse_word_after<'a>(line: &'a str, label: &str) -> Result<&'a str> {
    let start = line
        .find(label)
        .with_context(|| format!("missing metric label {label}"))?
        + label.len();
    let rest = &line[start..];
    let end = rest
        .find(|ch: char| ch.is_whitespace() || ch == '|')
        .unwrap_or(rest.len());
    Ok(rest[..end].trim())
}

fn parse_unique_tokens(line: &str) -> Result<(usize, usize)> {
    let value = parse_word_after(line, "unique=")?;
    let Some((unique, total)) = value.split_once('/') else {
        anyhow::bail!("unique metric must use used/total format");
    };
    Ok((unique.parse()?, total.parse()?))
}

fn runs_for_variant(runs: &[BenchmarkRun], variant: &str) -> Vec<BenchmarkRun> {
    runs.iter()
        .filter(|run| run.variant == variant)
        .cloned()
        .collect()
}

fn summarize_variant(
    name: &str,
    runs: &[BenchmarkRun],
    target_min_tok_s: f64,
    target_max_tok_s: f64,
) -> Result<VariantSummary> {
    if runs.is_empty() {
        anyhow::bail!("no benchmark runs found for {name}");
    }
    let mut min_decode_tok_s = f64::INFINITY;
    let mut max_decode_tok_s = f64::NEG_INFINITY;
    let mut sum_decode_tok_s = 0.0;
    let mut sum_unique_tokens = 0.0;
    let mut sum_repetition_ratio = 0.0;
    let mut floor_accepted = true;
    let mut band_accepted = true;
    for run in runs {
        let decode = run.metrics.decode_tok_s;
        min_decode_tok_s = min_decode_tok_s.min(decode);
        max_decode_tok_s = max_decode_tok_s.max(decode);
        sum_decode_tok_s += decode;
        sum_unique_tokens += run.metrics.unique_tokens as f64;
        sum_repetition_ratio += run.metrics.repetition_ratio;
        floor_accepted &= decode >= target_min_tok_s;
        band_accepted &= decode >= target_min_tok_s && decode <= target_max_tok_s;
    }
    let run_count = runs.len() as f64;
    Ok(VariantSummary {
        name: name.to_string(),
        runs: runs.len(),
        floor_accepted,
        band_accepted,
        min_decode_tok_s,
        max_decode_tok_s,
        avg_decode_tok_s: sum_decode_tok_s / run_count,
        avg_unique_tokens: sum_unique_tokens / run_count,
        avg_repetition_ratio: sum_repetition_ratio / run_count,
    })
}

fn write_markdown_report(
    options: &BenchmarkOptions,
    runner: &Path,
    runs: &[BenchmarkRun],
    summaries: &[VariantSummary],
) -> Result<()> {
    if let Some(parent) = Path::new(&options.out).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut body = String::new();
    body.push_str("# Alternating Benchmark Harness\n\n");
    body.push_str("## Setup\n\n");
    body.push_str(&format!("- Model: `{}`\n", options.file));
    body.push_str(&format!("- Runner: `{}`\n", runner.to_string_lossy()));
    if options.prompts.len() == 1 {
        body.push_str(&format!("- Prompt: `{}`\n", options.prompts[0]));
    } else {
        body.push_str(&format!("- Prompts: {}\n", options.prompts.len()));
        for (index, prompt) in options.prompts.iter().enumerate() {
            body.push_str(&format!("  - {}: `{}`\n", index + 1, prompt));
        }
    }
    body.push_str(&format!(
        "- Runs: {} alternating control/candidate pairs per prompt\n",
        options.runs
    ));
    body.push_str(&format!(
        "- Target decode band: {:.2}-{:.2} tok/s\n",
        options.target_min_tok_s, options.target_max_tok_s
    ));
    body.push_str(&format!("- Profile phases: {}\n\n", options.profile_phases));
    body.push_str("## Summary\n\n");
    body.push_str("| variant | runs | floor accepted | band accepted | min decode tok/s | max decode tok/s | avg decode tok/s | avg unique tokens | avg repetition ratio |\n");
    body.push_str("|---|---:|---|---|---:|---:|---:|---:|---:|\n");
    for summary in summaries {
        body.push_str(&format!(
            "| {} | {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            summary.name,
            summary.runs,
            summary.floor_accepted,
            summary.band_accepted,
            summary.min_decode_tok_s,
            summary.max_decode_tok_s,
            summary.avg_decode_tok_s,
            summary.avg_unique_tokens,
            summary.avg_repetition_ratio
        ));
    }
    body.push_str("\n## Runs\n\n");
    body.push_str("| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |\n");
    body.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for run in runs {
        body.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {} | {} | {} | {:.2} | {} | {}/{} |\n",
            run.variant,
            run.run_index,
            run.prompt_index + 1,
            run.metrics.prefill_seconds,
            run.metrics.decode_tok_s,
            run.metrics.end_to_end_tok_s,
            run.metrics.generated_tokens,
            run.metrics.context_tokens,
            run.metrics.peak_transient_bytes,
            run.metrics.repetition_ratio,
            run.metrics.max_run,
            run.metrics.unique_tokens,
            run.metrics.total_tokens_for_repetition
        ));
    }
    body.push_str("\n## Raw Output\n\n");
    for run in runs {
        body.push_str(&format!(
            "### {} run {} prompt {}\n\n",
            run.variant,
            run.run_index,
            run.prompt_index + 1
        ));
        body.push_str(&format!("Prompt: `{}`\n\n", run.prompt));
        body.push_str("```text\n");
        body.push_str(&run.stdout);
        if !run.stdout.ends_with('\n') {
            body.push('\n');
        }
        body.push_str("```\n\n");
    }
    fs::write(&options.out, body)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama_test_output_extracts_speed_and_repetition_metrics() {
        let output = "
[TTFT/Prefill: 12.58s | Decode: 30.26 tok/s | E2E: 4.36 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
";

        let metrics = parse_llama_test_output(output).unwrap();

        assert_eq!(metrics.generated_tokens, 64);
        assert_eq!(metrics.context_tokens, 66);
        assert_eq!(metrics.peak_transient_bytes, 1_050_689_536);
        assert_eq!(metrics.unique_tokens, 15);
        assert_eq!(metrics.total_tokens_for_repetition, 64);
        assert_eq!(metrics.max_run, 2);
        assert!((metrics.prefill_seconds - 12.58).abs() < 0.001);
        assert!((metrics.decode_tok_s - 30.26).abs() < 0.001);
        assert!((metrics.end_to_end_tok_s - 4.36).abs() < 0.001);
        assert!((metrics.repetition_ratio - 0.13).abs() < 0.001);
    }

    #[test]
    fn summarize_variant_requires_all_runs_inside_target_band() {
        let runs = vec![
            sample_run("candidate", 1, 30.26),
            sample_run("candidate", 2, 38.77),
            sample_run("candidate", 3, 29.94),
        ];

        let summary = summarize_variant("candidate", &runs, 30.0, 40.0).unwrap();

        assert!(!summary.band_accepted);
        assert!(!summary.floor_accepted);
        assert_eq!(summary.runs, 3);
        assert!((summary.min_decode_tok_s - 29.94).abs() < 0.001);
        assert!((summary.max_decode_tok_s - 38.77).abs() < 0.001);
    }

    #[test]
    fn summarize_variant_separates_speed_floor_from_strict_band() {
        let runs = vec![
            sample_run("candidate", 1, 30.43),
            sample_run("candidate", 2, 54.75),
        ];

        let summary = summarize_variant("candidate", &runs, 30.0, 40.0).unwrap();

        assert!(summary.floor_accepted);
        assert!(!summary.band_accepted);
        assert!((summary.min_decode_tok_s - 30.43).abs() < 0.001);
        assert!((summary.max_decode_tok_s - 54.75).abs() < 0.001);
    }

    #[test]
    fn run_plan_alternates_control_and_candidate() {
        let plan = alternating_run_plan(3);

        assert_eq!(
            plan,
            vec![
                ("control".to_string(), 1),
                ("candidate".to_string(), 1),
                ("control".to_string(), 2),
                ("candidate".to_string(), 2),
                ("control".to_string(), 3),
                ("candidate".to_string(), 3),
            ]
        );
    }

    #[test]
    fn run_plan_covers_each_prompt_for_each_alternating_pair() {
        let plan = alternating_prompt_run_plan(2, 2);

        assert_eq!(
            plan,
            vec![
                ("control".to_string(), 1, 0),
                ("candidate".to_string(), 1, 0),
                ("control".to_string(), 1, 1),
                ("candidate".to_string(), 1, 1),
                ("control".to_string(), 2, 0),
                ("candidate".to_string(), 2, 0),
                ("control".to_string(), 2, 1),
                ("candidate".to_string(), 2, 1),
            ]
        );
    }

    #[test]
    fn report_includes_prompt_matrix_metadata() {
        let out =
            std::env::temp_dir().join(format!("rllm-benchmark-report-{}.md", std::process::id()));
        let options = BenchmarkOptions {
            file: "model.spsa".to_string(),
            prompts: vec!["good morning".to_string(), "who are you?".to_string()],
            runs: 1,
            ctx: 2048,
            max_new_tokens: 64,
            out: out.to_string_lossy().to_string(),
            profile_phases: false,
            target_min_tok_s: 30.0,
            target_max_tok_s: 40.0,
            common_env: Vec::new(),
            control_env: Vec::new(),
            candidate_env: Vec::new(),
            control_name: "control".to_string(),
            candidate_name: "candidate".to_string(),
            runner: None,
        };
        let runs = vec![
            sample_run_with_prompt("control", 1, 0, "good morning", 31.0),
            sample_run_with_prompt("candidate", 1, 1, "who are you?", 32.0),
        ];
        let summaries = vec![
            summarize_variant("control", &runs_for_variant(&runs, "control"), 30.0, 40.0).unwrap(),
            summarize_variant(
                "candidate",
                &runs_for_variant(&runs, "candidate"),
                30.0,
                40.0,
            )
            .unwrap(),
        ];

        write_markdown_report(&options, Path::new("llama-test"), &runs, &summaries).unwrap();

        let report = fs::read_to_string(&out).unwrap();
        let _ = fs::remove_file(&out);
        assert!(report.contains("- Prompts: 2"));
        assert!(report.contains("| variant | run | prompt | prefill s | decode tok/s |"));
        assert!(report.contains("| candidate | 1 | 2 |"));
        assert!(report.contains("Prompt: `who are you?`"));
    }

    fn sample_run(variant: &str, run_index: usize, decode_tok_s: f64) -> BenchmarkRun {
        sample_run_with_prompt(variant, run_index, 0, "good morning", decode_tok_s)
    }

    fn sample_run_with_prompt(
        variant: &str,
        run_index: usize,
        prompt_index: usize,
        prompt: &str,
        decode_tok_s: f64,
    ) -> BenchmarkRun {
        BenchmarkRun {
            variant: variant.to_string(),
            run_index,
            prompt_index,
            prompt: prompt.to_string(),
            metrics: BenchmarkMetrics {
                prefill_seconds: 12.0,
                decode_tok_s,
                end_to_end_tok_s: 4.0,
                generated_tokens: 64,
                context_tokens: 66,
                peak_transient_bytes: 1_050_689_536,
                repetition_ratio: 0.11,
                max_run: 2,
                unique_tokens: 17,
                total_tokens_for_repetition: 64,
            },
            stdout: String::new(),
        }
    }
}
