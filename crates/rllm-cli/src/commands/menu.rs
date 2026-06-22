// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! `rllm menu` — interactive arrow-key launcher with a text logo. A friendly layer
//! over the flag-based subcommands (fetch / pack / chat / inspect): pick an action,
//! pick a model/folder from `models/` (auto-discovered), and it dispatches. Power users
//! keep using `rllm <subcommand> --flags` directly.

use anyhow::Result;
use inquire::{Confirm, Select, Text};
use std::fs;
use std::io::Write;
use std::path::Path;

// Succulent rosette — leaves radiating densely into a tight core (`spissa` = dense/packed).
const LOGO: &str = r"          \    \    |    /    /
        \    \    \  |  /    /    /
      \    \    \   \|/   /    /    /
    ──────────────( ❋ )──────────────
      /    /    /   /|\   \    \    \
        /    /    /  |  \    \    \
          /    /     |     \    \
                   |||
                ___|||___
               |_________|";

const FETCH: &str = "📥  Fetch model dari Hugging Face";
const PACK: &str = "📦  Pack model → .rllm";
const CHAT: &str = "💬  Chat sama model";
const INSPECT: &str = "🔍  Inspect / list model";
const EXIT: &str = "🚪  Keluar";

fn print_logo() {
    print!("\x1b[2J\x1b[H"); // clear screen + cursor home
    println!("\x1b[92m{LOGO}\x1b[0m"); // bright green (succulent)
    println!("\x1b[1;92m                  s p i s s a\x1b[0m");
    println!(
        "\x1b[2m             compressed · local · yours · v{}\x1b[0m\n",
        env!("CARGO_PKG_VERSION")
    );
}

pub fn run() -> Result<()> {
    loop {
        print_logo();
        let action = match Select::new("Mau ngapain?", vec![FETCH, PACK, CHAT, INSPECT, EXIT])
            .with_page_size(6)
            .prompt()
        {
            Ok(a) => a,
            Err(_) => break, // Esc / Ctrl-C
        };
        if action == EXIT {
            break;
        }
        let result = match action {
            FETCH => menu_fetch(),
            PACK => menu_pack(),
            CHAT => menu_chat(),
            INSPECT => menu_inspect(),
            _ => Ok(()),
        };
        if let Err(e) = result {
            eprintln!("\n  ⚠ {e}");
        }
        pause();
    }
    println!("\n  bye! 👋");
    Ok(())
}

/// Wait for Enter so command output isn't wiped before the user reads it.
fn pause() {
    print!("\n  \x1b[2m↵ Enter buat balik ke menu…\x1b[0m ");
    let _ = std::io::stdout().flush();
    let mut s = String::new();
    let _ = std::io::stdin().read_line(&mut s);
}

fn menu_fetch() -> Result<()> {
    let repo = match Text::new("HF repo (org/model):")
        .with_help_message("contoh: Qwen/Qwen3.5-2B — kosongin buat batal")
        .prompt()
    {
        Ok(r) if !r.trim().is_empty() => r.trim().to_string(),
        _ => return Ok(()),
    };
    let cat = Text::new("Kategori folder (kosong = auto-detect):")
        .prompt()
        .unwrap_or_default();
    let cat = cat.trim();
    super::fetch::run(
        &repo,
        if cat.is_empty() { None } else { Some(cat) },
        None,
        "main",
        "models",
    )
}

fn menu_pack() -> Result<()> {
    let dirs = discover_model_dirs("models");
    if dirs.is_empty() {
        println!("  (belum ada folder model di models/ — Fetch dulu)");
        return Ok(());
    }
    let dir = match Select::new("Pilih folder model:", dirs).prompt() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let mode = match Select::new(
        "Mode pack:",
        vec![
            "q8 — cepat (lossy ~0.5%)",
            "rANS — lossless (bit-exact)",
            "q4 — kecil (lossy ~10%)",
            "raw bf16 — lossless besar",
        ],
    )
    .prompt()
    {
        Ok(m) => m,
        Err(_) => return Ok(()),
    };
    let (codec, quant): (&str, Option<&str>) = if mode.starts_with("q8") {
        ("raw", Some("q8_transformer_keep_io"))
    } else if mode.starts_with("rANS") {
        ("rans", None)
    } else if mode.starts_with("q4") {
        ("raw", Some("q4_0_keep_io"))
    } else {
        ("raw", None)
    };
    let base = Path::new(&dir)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model");
    let default_out = format!("models/{base}.rllm");
    let out = match Text::new("Output .rllm:").with_default(&default_out).prompt() {
        Ok(o) => o,
        Err(_) => return Ok(()),
    };
    super::pack::run(
        &dir, &out, "1mb", codec, None, None, false, false, false, 16, None, None, false, quant,
    )
}

const STYLE_GREEDY: &str = "🎯  Presisi (greedy — deterministik)";
const STYLE_CREATIVE: &str = "🎨  Kreatif (sampling — variatif)";

fn menu_chat() -> Result<()> {
    let model = match pick_rllm("Chat — pilih model:") {
        Some(m) => m,
        None => return Ok(()),
    };
    let fast = Confirm::new("Mode --fast (q8 turbo)?")
        .with_default(true)
        .prompt()
        .unwrap_or(true);

    // Max tokens per reply (Enter = default). Falls back to 512 on bad/empty input.
    let max_new_tokens = Text::new("Max token per balasan:")
        .with_default("512")
        .prompt()
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(512);

    // Optional system prompt (Qwen ChatML / Llama). Enter = skip.
    let system = Text::new("System prompt (Enter = lewati):")
        .with_help_message("kasih persona/aturan, mis. 'jawab singkat pakai bahasa gaul'")
        .prompt()
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    // Generation style → a friendly preset over the raw sampling flags.
    let style = Select::new("Gaya jawaban:", vec![STYLE_GREEDY, STYLE_CREATIVE])
        .prompt()
        .unwrap_or(STYLE_GREEDY);
    let sampling = if style == STYLE_CREATIVE {
        let temp = Text::new("Temperature (0.1–1.5):")
            .with_default("0.7")
            .prompt()
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|t| *t > 0.0)
            .unwrap_or(0.7);
        super::chat::SamplingArgs {
            temp,
            top_p: 0.95,
            top_k: 40,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 0,
        }
    } else {
        super::chat::SamplingArgs {
            temp: 0.0,
            top_p: 0.95,
            top_k: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    };

    super::chat::run(
        &model,
        2048,
        max_new_tokens,
        false,
        fast,
        "llama3",
        system.as_deref(),
        sampling,
    )
}

fn menu_inspect() -> Result<()> {
    match pick_rllm("Inspect — pilih model:") {
        Some(m) => super::inspect::run(&m),
        None => Ok(()),
    }
}

/// Prompt the user to pick a `.rllm` from `models/`; `None` = none found or cancelled.
fn pick_rllm(prompt: &str) -> Option<String> {
    let models = discover_rllm("models");
    if models.is_empty() {
        println!("  (belum ada .rllm di models/ — Pack dulu)");
        return None;
    }
    Select::new(prompt, models).prompt().ok()
}

// ---- model discovery under models/ ----

fn discover_rllm(root: &str) -> Vec<String> {
    let mut out = Vec::new();
    walk(Path::new(root), 4, &mut |p, is_dir| {
        if !is_dir && p.extension().and_then(|e| e.to_str()) == Some("rllm") {
            out.push(p.display().to_string());
        }
    });
    out.sort();
    out
}

fn discover_model_dirs(root: &str) -> Vec<String> {
    let mut out = Vec::new();
    walk(Path::new(root), 4, &mut |p, is_dir| {
        if is_dir && p.join("config.json").exists() {
            out.push(p.display().to_string());
        }
    });
    out.sort();
    out
}

fn walk(dir: &Path, depth: usize, f: &mut impl FnMut(&Path, bool)) {
    if depth == 0 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        let is_dir = p.is_dir();
        f(&p, is_dir);
        if is_dir {
            walk(&p, depth - 1, f);
        }
    }
}
