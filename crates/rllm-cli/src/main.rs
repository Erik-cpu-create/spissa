//! RLLM CLI - Command-line interface for Runtime-compressed Local LLM
#![allow(clippy::too_many_arguments)]

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

        /// Codec policy for packed chunks: auto chooses the smallest lossless codec; raw/rle/huff force one codec.
        #[arg(long, default_value = "auto")]
        codec: String,

        /// Optional fixed decoded-byte range size for per-range checksums.
        /// Currently emitted only for identity-mapped raw chunks.
        #[arg(long)]
        range_checksum_size: Option<String>,

        /// Optional number of tensor elements per packed chunk/block.
        /// Overrides --chunk-size per tensor after multiplying by dtype size.
        #[arg(long)]
        tile_block_elements: Option<usize>,

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

        /// Prompt text for Phase 7 tiled RAMA generation
        #[arg(long)]
        prompt: Option<String>,

        /// Comma-separated input token IDs for fixed-token generation/comparison; bypasses tokenizer
        #[arg(long)]
        token_ids: Option<String>,

        /// Number of new tokens to generate when --prompt or --token-ids is provided
        #[arg(long, default_value_t = 8)]
        max_new_tokens: usize,

        /// Optional JSON output path for first-step logits from --prompt/--token-ids generation
        #[arg(long)]
        logits_out: Option<String>,

        /// Optional JSON output path for RAMA chunk recall timing trace
        #[arg(long)]
        rama_trace: Option<String>,

        /// Optional JSON output path for low-overhead aggregate RAMA generation timings.
        #[arg(long)]
        rama_timing: Option<String>,

        /// Optional prompt prefill chunk size in real input tokens.
        ///
        /// Generation defaults to the generic RAMA shape/budget-aware low-RAM policy
        /// unless --no-rama-prefill-chunking is set.
        #[arg(long)]
        rama_prefill_chunk_tokens: Option<usize>,

        /// RAMA automatic prefill policy when --rama-prefill-chunk-tokens is not set: low-ram or speed.
        #[arg(long, default_value = "low-ram")]
        rama_prefill_policy: String,

        /// Disable the default RAMA prompt prefill chunking window and process prefill in one full prompt pass.
        #[arg(long)]
        no_rama_prefill_chunking: bool,

        /// Runtime integrity policy: strict verifies every chunk recall; verify-once verifies each chunk once per process.
        #[arg(long, default_value = "strict")]
        rama_integrity: String,
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

    /// Run a scripted persistent chat-session benchmark
    ChatSession {
        /// Path to .rllm file
        file: String,

        /// Conversation turn text; pass this flag more than once
        #[arg(long = "turn", required = true, allow_hyphen_values = true)]
        turns: Vec<String>,

        /// Maximum assistant tokens per turn
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,

        /// Maximum context length
        #[arg(long, default_value_t = 2048)]
        ctx: usize,

        /// Markdown report output path
        #[arg(long)]
        out: String,
    },

    /// Run a token-native full-replay vs persistent chat-session benchmark
    ChatSessionToken {
        /// Path to .rllm file
        file: String,

        /// Comma-separated token IDs for one user turn; pass this flag more than once
        #[arg(long = "turn-ids", required = true)]
        turns: Vec<String>,

        /// Maximum assistant tokens per turn
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,

        /// Maximum context length
        #[arg(long, default_value_t = 2048)]
        ctx: usize,

        /// Markdown report output path
        #[arg(long)]
        out: String,
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
            codec,
            range_checksum_size,
            tile_block_elements,
            config,
            tokenizer,
            no_tokenizer,
        } => commands::pack::run(
            &input,
            &out,
            &chunk_size,
            &codec,
            range_checksum_size.as_deref(),
            tile_block_elements,
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
            token_ids,
            max_new_tokens,
            logits_out,
            rama_trace,
            rama_timing,
            rama_prefill_chunk_tokens,
            rama_prefill_policy,
            no_rama_prefill_chunking,
            rama_integrity,
        } => commands::run::run(
            &file,
            &mode,
            ctx,
            memory_budget.as_deref(),
            dry_run,
            prompt.as_deref(),
            token_ids.as_deref(),
            max_new_tokens,
            logits_out.as_deref(),
            rama_trace.as_deref(),
            rama_timing.as_deref(),
            rama_prefill_chunk_tokens,
            &rama_prefill_policy,
            no_rama_prefill_chunking,
            &rama_integrity,
        ),
        Commands::Import { input } => commands::import::run(&input),
        Commands::Benchmark { file } => commands::benchmark::run(&file),
        Commands::ChatSession {
            file,
            turns,
            max_new_tokens,
            ctx,
            out,
        } => commands::chat_session::run(&file, &turns, max_new_tokens, ctx, &out),
        Commands::ChatSessionToken {
            file,
            turns,
            max_new_tokens,
            ctx,
            out,
        } => commands::chat_session_token::run(&file, &turns, max_new_tokens, ctx, &out),
        Commands::Doctor => commands::doctor::run(),
    }
}
