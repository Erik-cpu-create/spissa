//! `rllm bench` — native benchmark matrices for `rllm run`.
//!
//! These subcommands spawn the running `rllm` binary, measure wall-clock time
//! and peak RSS from the child's `rusage`, parse the metrics RLLM prints, and
//! emit CSV/Markdown matrices. They replace the former Python harnesses under
//! `scripts/` so the whole toolchain stays Rust-only.

use anyhow::Result;
use clap::Subcommand;

mod long_prompt;
mod low_ram_fast;
mod prefill_timing;
mod release_rss;
mod report;
mod runner;

#[derive(Subcommand)]
pub enum BenchCommand {
    /// Release RSS matrix over ctx × max-new-tokens for a text prompt.
    ReleaseRss(release_rss::Args),

    /// Pack/verify a raw tile-block artifact then run the low-ram-fast RSS matrix.
    LowRamFast(low_ram_fast::Args),

    /// Long-prompt matrix over fixed token-id lengths with RAMA timing.
    LongPrompt(long_prompt::Args),

    /// Prefill-window timing sweep with RAMA timing.
    PrefillTiming(prefill_timing::Args),
}

pub fn run(command: BenchCommand) -> Result<()> {
    match command {
        BenchCommand::ReleaseRss(args) => release_rss::run(args),
        BenchCommand::LowRamFast(args) => low_ram_fast::run(args),
        BenchCommand::LongPrompt(args) => long_prompt::run(args),
        BenchCommand::PrefillTiming(args) => prefill_timing::run(args),
    }
}
