// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! `rllm fetch <repo>` — download a Hugging Face model into `models/<category>/<name>/`,
//! auto-categorized by modality (text / vision / audio / embedding). Rust-native (ureq +
//! rustls, no Python `hf` CLI), with skip-existing, byte-range resume, and a progress line.
//!
//! Auth: set `HF_TOKEN` for gated/private repos. Revision/branch via `--revision`.

use anyhow::{bail, Context, Result};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

const HF: &str = "https://huggingface.co";
const UA: &str = concat!("rllm-fetch/", env!("CARGO_PKG_VERSION"));

#[derive(serde::Deserialize)]
struct ModelInfo {
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(serde::Deserialize)]
struct Sibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

/// Map a Hugging Face `pipeline_tag` to a modality folder.
fn category_for(pipeline: Option<&str>) -> &'static str {
    match pipeline {
        Some(
            "text-generation" | "text2text-generation" | "fill-mask" | "text-classification"
            | "token-classification" | "question-answering" | "summarization" | "translation"
            | "zero-shot-classification" | "table-question-answering",
        ) => "text-models",
        Some(
            "image-text-to-text" | "image-to-text" | "visual-question-answering"
            | "image-classification" | "object-detection" | "image-segmentation"
            | "depth-estimation" | "image-to-image" | "zero-shot-image-classification"
            | "video-text-to-text" | "text-to-image" | "image-feature-extraction" | "any-to-any",
        ) => "vision-models",
        Some(
            "automatic-speech-recognition" | "audio-classification" | "text-to-speech"
            | "text-to-audio" | "audio-to-audio" | "voice-activity-detection",
        ) => "audio-models",
        Some("feature-extraction" | "sentence-similarity") => "embedding-models",
        _ => "other-models",
    }
}

/// Normalize a user `--category` override into a `<x>-models` folder name.
fn normalize_category(raw: &str) -> String {
    let base = raw.trim().to_lowercase().replace(' ', "-");
    if base.ends_with("-models") {
        base
    } else {
        format!("{base}-models")
    }
}

fn get_with_auth(url: &str, token: Option<&str>) -> ureq::Request {
    let mut req = ureq::get(url).set("User-Agent", UA);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    req
}

fn fetch_model_info(repo: &str, token: Option<&str>) -> Result<ModelInfo> {
    let url = format!("{HF}/api/models/{repo}?blobs=true");
    let body = get_with_auth(&url, token)
        .call()
        .map_err(|e| anyhow::anyhow!(describe_ureq_err("model info", repo, e)))?
        .into_string()
        .context("reading model-info response body")?;
    serde_json::from_str(&body).context("parsing Hugging Face model-info JSON")
}

fn describe_ureq_err(what: &str, repo: &str, e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(401, _) | ureq::Error::Status(403, _) => format!(
            "{what} for {repo}: unauthorized — set HF_TOKEN for a gated/private repo"
        ),
        ureq::Error::Status(404, _) => format!("{what} for {repo}: not found (check the repo id)"),
        ureq::Error::Status(code, _) => format!("{what} for {repo}: HTTP {code}"),
        ureq::Error::Transport(t) => format!("{what} for {repo}: transport error: {t}"),
    }
}

fn human(bytes: u64) -> String {
    const U: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1}{}", U[i])
}

/// Stream-download one file with resume + a single-line progress indicator.
fn download_file(
    repo: &str,
    revision: &str,
    rfilename: &str,
    expected: Option<u64>,
    dest_dir: &Path,
    token: Option<&str>,
    idx: usize,
    total: usize,
) -> Result<()> {
    let final_path = dest_dir.join(rfilename);
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Skip if already complete (size matches, or just exists when size is unknown).
    if let Ok(meta) = fs::metadata(&final_path) {
        if expected.map(|e| e == meta.len()).unwrap_or(true) {
            println!("[{idx}/{total}] {rfilename} — already present, skip");
            return Ok(());
        }
    }

    let part_path = final_path.with_extension(format!(
        "{}.part",
        final_path.extension().and_then(|e| e.to_str()).unwrap_or("")
    ));
    let resume_from = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);

    let url = format!("{HF}/{repo}/resolve/{revision}/{rfilename}");
    let mut req = get_with_auth(&url, token);
    if resume_from > 0 {
        req = req.set("Range", &format!("bytes={resume_from}-"));
    }
    let resp = req
        .call()
        .map_err(|e| anyhow::anyhow!(describe_ureq_err(rfilename, repo, e)))?;

    // 206 = server honored the Range (append); otherwise (200) restart from scratch.
    let resuming = resp.status() == 206 && resume_from > 0;
    let body_len: Option<u64> = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());
    let total_bytes = expected.or(body_len.map(|l| l + if resuming { resume_from } else { 0 }));

    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resuming)
        .truncate(!resuming)
        .open(&part_path)
        .with_context(|| format!("opening {}", part_path.display()))?;

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20]; // 1 MiB
    let mut done = resume_from;
    let mut last = Instant::now();
    let started = Instant::now();
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        if last.elapsed().as_millis() >= 250 {
            print_progress(idx, total, rfilename, done, total_bytes, started, resume_from);
            last = Instant::now();
        }
    }
    file.flush()?;
    print_progress(idx, total, rfilename, done, total_bytes, started, resume_from);
    println!();

    if let Some(exp) = expected {
        if done != exp {
            bail!("{rfilename}: downloaded {done} bytes but expected {exp}");
        }
    }
    fs::rename(&part_path, &final_path)
        .with_context(|| format!("finalizing {}", final_path.display()))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_progress(
    idx: usize,
    total: usize,
    name: &str,
    done: u64,
    total_bytes: Option<u64>,
    started: Instant,
    resume_from: u64,
) {
    let secs = started.elapsed().as_secs_f64().max(1e-3);
    let speed = (done.saturating_sub(resume_from)) as f64 / secs;
    let pct = total_bytes
        .map(|t| format!("{:5.1}%", done as f64 / t.max(1) as f64 * 100.0))
        .unwrap_or_else(|| "   ?%".to_string());
    print!(
        "\r[{idx}/{total}] {name}  {pct}  {} / {}  {}/s   ",
        human(done),
        total_bytes.map(human).unwrap_or_else(|| "?".to_string()),
        human(speed as u64)
    );
    let _ = std::io::stdout().flush();
}

pub fn run(
    repo: &str,
    category: Option<&str>,
    name: Option<&str>,
    revision: &str,
    models_dir: &str,
) -> Result<()> {
    if !repo.contains('/') {
        bail!("repo must be '<org>/<model>', e.g. Qwen/Qwen3.5-2B (got {repo:?})");
    }
    let token = std::env::var("HF_TOKEN").ok();

    println!("Resolving {repo} (rev {revision}) …");
    let info = fetch_model_info(repo, token.as_deref())?;
    if info.siblings.is_empty() {
        bail!("{repo}: no files listed (private repo without HF_TOKEN, or empty repo?)");
    }

    let cat = category
        .map(normalize_category)
        .unwrap_or_else(|| category_for(info.pipeline_tag.as_deref()).to_string());
    let folder = name
        .map(str::to_string)
        .unwrap_or_else(|| repo.rsplit('/').next().unwrap_or(repo).to_string());
    let dest = Path::new(models_dir).join(&cat).join(&folder);
    fs::create_dir_all(&dest)?;

    let total_bytes: u64 = info.siblings.iter().filter_map(|s| s.size).sum();
    println!(
        "pipeline = {}  ->  category = {cat}",
        info.pipeline_tag.as_deref().unwrap_or("(none)")
    );
    println!(
        "{} files ({}) -> {}",
        info.siblings.len(),
        human(total_bytes),
        dest.display()
    );

    let n = info.siblings.len();
    for (i, s) in info.siblings.iter().enumerate() {
        download_file(
            repo,
            revision,
            &s.rfilename,
            s.size,
            &dest,
            token.as_deref(),
            i + 1,
            n,
        )?;
    }

    println!("\n✓ {repo} -> {}", dest.display());
    println!("  pack it with:  rllm pack {} --out <name>.rllm", dest.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{category_for, normalize_category};

    #[test]
    fn categories_map_by_modality() {
        assert_eq!(category_for(Some("text-generation")), "text-models");
        assert_eq!(category_for(Some("image-text-to-text")), "vision-models");
        assert_eq!(category_for(Some("automatic-speech-recognition")), "audio-models");
        assert_eq!(category_for(Some("sentence-similarity")), "embedding-models");
        assert_eq!(category_for(None), "other-models");
        assert_eq!(category_for(Some("some-future-task")), "other-models");
    }

    #[test]
    fn category_override_normalizes() {
        assert_eq!(normalize_category("vision"), "vision-models");
        assert_eq!(normalize_category("Vision Models"), "vision-models");
        assert_eq!(normalize_category("text-models"), "text-models");
    }
}
