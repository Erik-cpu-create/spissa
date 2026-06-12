//! RLLM CLI - Command-line interface for Runtime-compressed Local LLM

use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;

#[derive(Parser)]
#[command(name = "rllm")]
#[command(about = "RLLM - Runtime-compressed Local LLM")]
#[command(version = "0.1.0")]
#[command(
    long_about = "RLLM is an experimental local LLM runtime built around lossless compressed model storage.\n\nIt stores model tensors in a chunked compressed container (.rllm) and aims to run inference by decoding only the tensor blocks needed at runtime."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Pack a model file into a .rllm container
    Pack {
        /// Input model file
        input: String,

        /// Output .rllm file path
        #[arg(short, long)]
        out: String,

        /// Chunk size (e.g., "1mb", "256kb", "4mb")
        #[arg(long, default_value = "1mb")]
        chunk_size: String,

        /// Optional HuggingFace config.json path. Defaults to sibling config.json when present.
        #[arg(long)]
        config: Option<String>,

        /// Optional HuggingFace tokenizer.json path. Defaults to sibling tokenizer.json when present.
        #[arg(long)]
        tokenizer: Option<String>,

        /// Do not auto-embed sibling tokenizer.json metadata.
        #[arg(long)]
        no_tokenizer: bool,
    },

    /// Inspect a .rllm file
    Inspect {
        /// Path to .rllm file
        file: String,
    },

    /// Verify that a .rllm file matches the original model
    Verify {
        /// Original model file
        original: String,

        /// Path to .rllm file
        compressed: String,
    },

    /// Unpack a .rllm file back to original data
    Unpack {
        /// Path to .rllm file
        file: String,

        /// Output file path
        #[arg(short, long)]
        out: String,
    },

    /// Run inference or low-memory runtime planning from a .rllm file
    Run {
        /// Path to .rllm file
        file: String,

        /// Runtime mode: full-decode, layer-stream, tile-stream
        #[arg(long, default_value = "full-decode")]
        mode: String,

        /// Context length used for runtime memory planning
        #[arg(long, default_value_t = 1024)]
        ctx: usize,

        /// Memory budget for low-RAM modes (e.g., "100mb", "512mb")
        #[arg(long)]
        memory_budget: Option<String>,

        /// Only plan/check memory usage; do not execute token generation
        #[arg(long)]
        dry_run: bool,

        /// Prompt text for Phase 5E tokenizer-backed RAMA generation
        #[arg(long)]
        prompt: Option<String>,

        /// Number of new tokens to generate when --prompt is provided
        #[arg(long, default_value_t = 8)]
        max_new_tokens: usize,
    },

    /// Import a model from external format (not yet implemented)
    Import {
        /// Input model directory
        input: String,
    },

    /// Benchmark a .rllm file (not yet implemented)
    Benchmark {
        /// Path to .rllm file
        file: String,
    },

    /// Check system dependencies and configuration
    Doctor,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logger
    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    match cli.command {
        Commands::Pack {
            input,
            out,
            chunk_size,
            config,
            tokenizer,
            no_tokenizer,
        } => commands::pack::run(
            &input,
            &out,
            &chunk_size,
            config.as_deref(),
            tokenizer.as_deref(),
            no_tokenizer,
        ),
        Commands::Inspect { file } => commands::inspect::run(&file),
        Commands::Verify {
            original,
            compressed,
        } => commands::verify::run(&original, &compressed),
        Commands::Unpack { file, out } => commands::unpack::run(&file, &out),
        Commands::Run {
            file,
            mode,
            ctx,
            memory_budget,
            dry_run,
            prompt,
            max_new_tokens,
        } => commands::run::run(
            &file,
            &mode,
            ctx,
            memory_budget.as_deref(),
            dry_run,
            prompt.as_deref(),
            max_new_tokens,
        ),
        Commands::Import { input } => commands::import::run(&input),
        Commands::Benchmark { file } => commands::benchmark::run(&file),
        Commands::Doctor => commands::doctor::run(),
    }
}
