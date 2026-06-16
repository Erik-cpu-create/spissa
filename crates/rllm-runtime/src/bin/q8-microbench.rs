use rllm_runtime::q8_kernel_lab::{run_suite, Q8KernelBenchConfig};
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    json: PathBuf,
    markdown: PathBuf,
    config: Q8KernelBenchConfig,
}

fn main() {
    let args = match parse_args(env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    };

    let report = run_suite(args.config);
    if let Err(err) = write_output(&args.json, serde_json::to_string_pretty(&report).unwrap()) {
        eprintln!("failed to write JSON {}: {err}", args.json.display());
        std::process::exit(1);
    }
    if let Err(err) = write_output(&args.markdown, markdown_report(&report)) {
        eprintln!(
            "failed to write Markdown {}: {err}",
            args.markdown.display()
        );
        std::process::exit(1);
    }

    println!("REE kernel: {}", report.ree_kernel);
    println!("JSON: {}", args.json.display());
    println!("Markdown: {}", args.markdown.display());
    for result in &report.results {
        println!(
            "{} elapsed_ns={} speedup={:.3}x max_abs_diff={:.8}",
            result.variant, result.elapsed_ns, result.speedup_vs_baseline, result.max_abs_diff
        );
    }
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut json = None;
    let mut markdown = None;
    let mut config = Q8KernelBenchConfig::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = Some(parse_path(args.next(), "--json")?),
            "--markdown" => markdown = Some(parse_path(args.next(), "--markdown")?),
            "--iters" => config.iters = parse_usize(args.next(), "--iters")?,
            "--batch" => config.batch = parse_usize(args.next(), "--batch")?,
            "--in-features" => {
                config.in_features = parse_usize(args.next(), "--in-features")?;
                config.blocks_per_row = config.in_features / 32;
            }
            "--out-features" => config.out_features = parse_usize(args.next(), "--out-features")?,
            "--help" | "-h" => return Err("q8-microbench help".to_string()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        json: json.ok_or_else(|| "missing required --json".to_string())?,
        markdown: markdown.ok_or_else(|| "missing required --markdown".to_string())?,
        config,
    })
}

fn parse_path(value: Option<String>, flag: &str) -> Result<PathBuf, String> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_usize(value: Option<String>, flag: &str) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    raw.parse::<usize>()
        .map_err(|err| format!("invalid value for {flag}: {raw} ({err})"))
}

fn write_output(path: &PathBuf, content: String) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)
}

fn markdown_report(report: &rllm_runtime::q8_kernel_lab::Q8KernelBenchReport) -> String {
    let mut out = String::new();
    out.push_str("# R91 Q8 Kernel Microbench\n\n");
    out.push_str(&format!("- REE kernel: `{}`\n", report.ree_kernel));
    out.push_str(&format!("- batch: `{}`\n", report.batch));
    out.push_str(&format!("- in_features: `{}`\n", report.in_features));
    out.push_str(&format!(
        "- out_features context: `{}`\n",
        report.out_features
    ));
    out.push_str(&format!("- iters: `{}`\n\n", report.iters));
    out.push_str("| variant | elapsed ns | speedup vs baseline | max abs diff | checksum |\n");
    out.push_str("|---|---:|---:|---:|---:|\n");
    for result in &report.results {
        out.push_str(&format!(
            "| `{}` | {} | {:.3}x | {:.8} | {:.6} |\n",
            result.variant,
            result.elapsed_ns,
            result.speedup_vs_baseline,
            result.max_abs_diff,
            result.checksum
        ));
    }
    out
}

fn usage() -> &'static str {
    "usage: q8-microbench --json <path> --markdown <path> [--iters <n>] [--batch <n>] [--in-features <n>] [--out-features <n>]"
}
