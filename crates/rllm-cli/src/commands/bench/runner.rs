//! Spawn `rllm` subcommands as child processes, measure wall-clock time and
//! peak resident set size (RSS), and parse the metrics RLLM prints on stdout.
//!
//! RSS is captured from the child's `rusage` (via `wait4` on Unix) so the
//! harness no longer depends on the macOS-only `/usr/bin/time -l` wrapper.

use anyhow::{anyhow, Context, Result};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

/// Result of one child `rllm run` invocation.
pub struct Outcome {
    pub exit_code: i32,
    pub real_seconds: f64,
    pub max_rss_bytes: Option<u64>,
    pub stdout: String,
}

impl Outcome {
    pub fn max_rss_mib(&self) -> Option<f64> {
        self.max_rss_bytes.map(|b| b as f64 / 1024.0 / 1024.0)
    }

    /// Return the trimmed text following a `Label:` line in stdout, if present.
    pub fn line_after(&self, label: &str) -> Option<String> {
        let needle = format!("{label}:");
        self.stdout.lines().find_map(|line| {
            line.trim_start()
                .strip_prefix(&needle)
                .map(|rest| rest.trim().to_string())
        })
    }

    pub fn generated_token_ids(&self) -> Option<String> {
        self.line_after("Generated token IDs")
    }

    pub fn generated_text(&self) -> Option<String> {
        self.line_after("Generated text")
    }

    pub fn peak_transient(&self) -> Option<String> {
        self.line_after("Peak transient budget")
    }

    pub fn context_memory(&self) -> Option<String> {
        self.line_after("Context memory bytes")
    }

    pub fn seconds_per_token(&self, max_new_tokens: usize) -> Option<f64> {
        if max_new_tokens == 0 {
            return None;
        }
        Some(self.real_seconds / max_new_tokens as f64)
    }

    pub fn tokens_per_second(&self, max_new_tokens: usize) -> Option<f64> {
        self.seconds_per_token(max_new_tokens)
            .filter(|spt| *spt > 0.0)
            .map(|spt| 1.0 / spt)
    }
}

/// Resolve the running `rllm` binary so the harness benchmarks itself.
pub fn rllm_bin() -> Result<PathBuf> {
    std::env::current_exe().context("failed to resolve current rllm executable")
}

/// Spawn `rllm <args>`, stream its output, and return the captured outcome.
pub fn run_child(bin: &Path, args: &[String]) -> Result<Outcome> {
    println!("$ {} {}", bin.display(), args.join(" "));
    let started = Instant::now();
    let mut child = Command::new(bin)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("failed to spawn {}", bin.display()))?;

    let mut stdout = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_string(&mut stdout)
            .context("failed to read child stdout")?;
    }
    let (exit_code, max_rss_bytes) = reap(child);
    let real_seconds = started.elapsed().as_secs_f64();

    print!("{stdout}");
    Ok(Outcome {
        exit_code,
        real_seconds,
        max_rss_bytes,
        stdout,
    })
}

/// Spawn `rllm <args>` and fail if it exits non-zero (used for pack/verify).
pub fn run_checked(bin: &Path, args: &[String]) -> Result<()> {
    let outcome = run_child(bin, args)?;
    if outcome.exit_code != 0 {
        return Err(anyhow!(
            "command failed (exit {}): {} {}",
            outcome.exit_code,
            bin.display(),
            args.join(" ")
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn reap(child: std::process::Child) -> (i32, Option<u64>) {
    let pid = child.id() as libc::pid_t;
    let mut status: libc::c_int = 0;
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::wait4(pid, &mut status, 0, &mut usage) };
    // We reaped the child ourselves; prevent std from touching the dead pid.
    std::mem::forget(child);
    if rc < 0 {
        return (-1, None);
    }
    let exited = (status & 0x7f) == 0;
    let code = if exited { (status >> 8) & 0xff } else { -1 };
    let maxrss = (usage.ru_maxrss as i64).max(0) as u64;
    // ru_maxrss is bytes on macOS but kilobytes on Linux.
    let bytes = if cfg!(target_os = "macos") {
        maxrss
    } else {
        maxrss.saturating_mul(1024)
    };
    (code, Some(bytes))
}

#[cfg(not(unix))]
fn reap(mut child: std::process::Child) -> (i32, Option<u64>) {
    let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
    (code, None)
}

/// Parsed `summary` object from a `--rama-timing` JSON file.
pub struct Timing(serde_json::Value);

impl Timing {
    /// Read the timing summary, returning an empty object when the path is
    /// missing or unreadable.
    pub fn read(path: Option<&Path>) -> Timing {
        let empty = serde_json::Value::Object(Default::default());
        let Some(path) = path else {
            return Timing(empty);
        };
        let Ok(text) = std::fs::read_to_string(path) else {
            return Timing(empty);
        };
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) else {
            return Timing(empty);
        };
        match payload.get("summary") {
            Some(summary) if summary.is_object() => Timing(summary.clone()),
            _ => Timing(payload),
        }
    }

    pub fn f(&self, key: &str) -> Option<f64> {
        self.0.get(key).and_then(|v| v.as_f64())
    }

    pub fn u(&self, key: &str) -> Option<u64> {
        self.0.get(key).and_then(|v| v.as_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(stdout: &str) -> Outcome {
        Outcome {
            exit_code: 0,
            real_seconds: 4.0,
            max_rss_bytes: Some(20 * 1024 * 1024),
            stdout: stdout.to_string(),
        }
    }

    #[test]
    fn parses_labeled_lines() {
        let o = outcome(
            "Generated token IDs: [13, 187]\nGenerated text: Hello,\nPeak transient budget: 291.68 KiB\nContext memory bytes: 1.50 MiB\n",
        );
        assert_eq!(o.generated_token_ids().as_deref(), Some("[13, 187]"));
        assert_eq!(o.generated_text().as_deref(), Some("Hello,"));
        assert_eq!(o.peak_transient().as_deref(), Some("291.68 KiB"));
        assert_eq!(o.context_memory().as_deref(), Some("1.50 MiB"));
    }

    #[test]
    fn throughput_math() {
        let o = outcome("");
        assert_eq!(o.seconds_per_token(4), Some(1.0));
        assert_eq!(o.tokens_per_second(4), Some(1.0));
        assert_eq!(o.seconds_per_token(0), None);
        assert!((o.max_rss_mib().unwrap() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn timing_reads_summary_object() {
        let timing = Timing(serde_json::json!({"prefill_ms": 12.5, "prefill_chunks": 3}));
        assert_eq!(timing.f("prefill_ms"), Some(12.5));
        assert_eq!(timing.u("prefill_chunks"), Some(3));
        assert_eq!(timing.f("missing"), None);
    }
}
