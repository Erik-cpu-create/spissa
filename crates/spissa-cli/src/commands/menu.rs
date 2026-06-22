// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! `spissa menu` — interactive arrow-key launcher with a text logo. A friendly layer
//! over the flag-based subcommands (fetch / pack / chat / inspect): pick an action,
//! pick a model/folder from `models/` (auto-discovered), and it dispatches. Power users
//! keep using `spissa <subcommand> --flags` directly.
//!
//! UI strings are i18n'd (English default, Indonesian optional) via a static string table;
//! the choice persists in `~/.config/spissa/settings.json`. No new dependency — `serde_json`
//! is already linked and the home dir comes from `$HOME`.

use anyhow::Result;
use inquire::{Confirm, Select, Text};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

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

fn print_logo() {
    print!("\x1b[2J\x1b[H"); // clear screen + cursor home
    println!("\x1b[92m{LOGO}\x1b[0m"); // bright green (succulent)
    println!("\x1b[1;92m                  s p i s s a\x1b[0m");
    println!(
        "\x1b[2m             compressed · local · yours · v{}\x1b[0m\n",
        env!("CARGO_PKG_VERSION")
    );
}

// ----- language + persisted settings -----

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Lang {
    #[default]
    En,
    Id,
}

fn lang_name(l: Lang) -> &'static str {
    match l {
        Lang::En => "English",
        Lang::Id => "Indonesia",
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Settings {
    #[serde(default)]
    language: Lang,
}

fn settings_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(Path::new(&home).join(".config/spissa/settings.json"))
}

fn load_settings() -> Settings {
    settings_path()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_settings(s: &Settings) {
    if let Some(p) = settings_path() {
        if let Some(dir) = p.parent() {
            fs::create_dir_all(dir).ok();
        }
        if let Ok(json) = serde_json::to_string_pretty(s) {
            fs::write(p, json).ok();
        }
    }
}

// ----- string table (English default, Indonesian optional) -----

struct Strings {
    menu_prompt: &'static str,
    act_fetch: &'static str,
    act_pack: &'static str,
    act_chat: &'static str,
    act_inspect: &'static str,
    act_settings: &'static str,
    act_quit: &'static str,
    pause: &'static str,
    bye: &'static str,
    fetch_repo: &'static str,
    fetch_repo_help: &'static str,
    fetch_category: &'static str,
    pack_no_dirs: &'static str,
    pack_pick_dir: &'static str,
    pack_mode: &'static str,
    pack_mode_q8: &'static str,
    pack_mode_rans: &'static str,
    pack_mode_q4: &'static str,
    pack_mode_raw: &'static str,
    pack_output: &'static str,
    chat_pick: &'static str,
    chat_fast: &'static str,
    chat_maxtok: &'static str,
    chat_system: &'static str,
    chat_system_help: &'static str,
    chat_style: &'static str,
    style_greedy: &'static str,
    style_creative: &'static str,
    chat_temp: &'static str,
    inspect_pick: &'static str,
    no_spsa: &'static str,
    settings_title: &'static str,
    settings_language: &'static str,
    settings_back: &'static str,
}

fn strings(lang: Lang) -> &'static Strings {
    match lang {
        Lang::En => &EN,
        Lang::Id => &ID,
    }
}

static EN: Strings = Strings {
    menu_prompt: "What would you like to do?",
    act_fetch: "📥  Fetch a model from Hugging Face",
    act_pack: "📦  Pack a model → .spsa",
    act_chat: "💬  Chat with a model",
    act_inspect: "🔍  Inspect / list models",
    act_settings: "⚙️   Settings",
    act_quit: "🚪  Quit",
    pause: "↵ Press Enter to return to the menu… ",
    bye: "bye! 👋",
    fetch_repo: "HF repo (org/model):",
    fetch_repo_help: "e.g. Qwen/Qwen3.5-2B — leave empty to cancel",
    fetch_category: "Folder category (empty = auto-detect):",
    pack_no_dirs: "(no model folders in models/ — Fetch one first)",
    pack_pick_dir: "Pick a model folder:",
    pack_mode: "Pack mode:",
    pack_mode_q8: "q8 — fast (lossy ~0.5%)",
    pack_mode_rans: "rANS — lossless (bit-exact)",
    pack_mode_q4: "q4 — small (lossy ~10%)",
    pack_mode_raw: "raw bf16 — lossless, large",
    pack_output: "Output .spsa:",
    chat_pick: "Chat — pick a model:",
    chat_fast: "Fast mode (q8 turbo)?",
    chat_maxtok: "Max tokens per reply:",
    chat_system: "System prompt (Enter = skip):",
    chat_system_help: "give a persona/rule, e.g. 'answer briefly'",
    chat_style: "Answer style:",
    style_greedy: "🎯  Precise (greedy — deterministic)",
    style_creative: "🎨  Creative (sampling — varied)",
    chat_temp: "Temperature (0.1–1.5):",
    inspect_pick: "Inspect — pick a model:",
    no_spsa: "(no .spsa in models/ — Pack one first)",
    settings_title: "⚙️  Settings",
    settings_language: "🌐  Language",
    settings_back: "←  Back",
};

static ID: Strings = Strings {
    menu_prompt: "Mau ngapain?",
    act_fetch: "📥  Fetch model dari Hugging Face",
    act_pack: "📦  Pack model → .spsa",
    act_chat: "💬  Chat sama model",
    act_inspect: "🔍  Inspect / list model",
    act_settings: "⚙️   Pengaturan",
    act_quit: "🚪  Keluar",
    pause: "↵ Enter buat balik ke menu… ",
    bye: "bye! 👋",
    fetch_repo: "HF repo (org/model):",
    fetch_repo_help: "contoh: Qwen/Qwen3.5-2B — kosongin buat batal",
    fetch_category: "Kategori folder (kosong = auto-detect):",
    pack_no_dirs: "(belum ada folder model di models/ — Fetch dulu)",
    pack_pick_dir: "Pilih folder model:",
    pack_mode: "Mode pack:",
    pack_mode_q8: "q8 — cepat (lossy ~0.5%)",
    pack_mode_rans: "rANS — lossless (bit-exact)",
    pack_mode_q4: "q4 — kecil (lossy ~10%)",
    pack_mode_raw: "raw bf16 — lossless besar",
    pack_output: "Output .spsa:",
    chat_pick: "Chat — pilih model:",
    chat_fast: "Mode --fast (q8 turbo)?",
    chat_maxtok: "Max token per balasan:",
    chat_system: "System prompt (Enter = lewati):",
    chat_system_help: "kasih persona/aturan, mis. 'jawab singkat pakai bahasa gaul'",
    chat_style: "Gaya jawaban:",
    style_greedy: "🎯  Presisi (greedy — deterministik)",
    style_creative: "🎨  Kreatif (sampling — variatif)",
    chat_temp: "Temperature (0.1–1.5):",
    inspect_pick: "Inspect — pilih model:",
    no_spsa: "(belum ada .spsa di models/ — Pack dulu)",
    settings_title: "⚙️  Pengaturan",
    settings_language: "🌐  Bahasa",
    settings_back: "←  Kembali",
};

pub fn run() -> Result<()> {
    let mut settings = load_settings();
    loop {
        let s = strings(settings.language);
        print_logo();
        let opts = vec![
            s.act_fetch,
            s.act_pack,
            s.act_chat,
            s.act_inspect,
            s.act_settings,
            s.act_quit,
        ];
        let action = match Select::new(s.menu_prompt, opts).with_page_size(7).prompt() {
            Ok(a) => a,
            Err(_) => break, // Esc / Ctrl-C
        };
        if action == s.act_quit {
            break;
        }
        if action == s.act_settings {
            menu_settings(&mut settings);
            continue; // no pause; re-render in (possibly) new language
        }
        let result = if action == s.act_fetch {
            menu_fetch(s)
        } else if action == s.act_pack {
            menu_pack(s)
        } else if action == s.act_chat {
            menu_chat(s)
        } else if action == s.act_inspect {
            menu_inspect(s)
        } else {
            Ok(())
        };
        if let Err(e) = result {
            eprintln!("\n  ⚠ {e}");
        }
        pause(s);
    }
    println!("\n  {}", strings(settings.language).bye);
    Ok(())
}

/// Settings submenu — currently just the language toggle (extensible later).
fn menu_settings(settings: &mut Settings) {
    let s = strings(settings.language);
    print_logo();
    let lang_row = format!("{}  ›  {}", s.settings_language, lang_name(settings.language));
    let choice = Select::new(s.settings_title, vec![lang_row.clone(), s.settings_back.to_string()])
        .prompt();
    if let Ok(c) = choice {
        if c == lang_row {
            let pick = Select::new(s.settings_language, vec!["English", "Indonesia"]).prompt();
            if let Ok(p) = pick {
                settings.language = if p == "Indonesia" { Lang::Id } else { Lang::En };
                save_settings(settings);
            }
        }
    }
}

/// Wait for Enter so command output isn't wiped before the user reads it.
fn pause(s: &Strings) {
    print!("\n  \x1b[2m{}\x1b[0m", s.pause);
    let _ = std::io::stdout().flush();
    let mut buf = String::new();
    let _ = std::io::stdin().read_line(&mut buf);
}

fn menu_fetch(s: &Strings) -> Result<()> {
    let repo = match Text::new(s.fetch_repo)
        .with_help_message(s.fetch_repo_help)
        .prompt()
    {
        Ok(r) if !r.trim().is_empty() => r.trim().to_string(),
        _ => return Ok(()),
    };
    let cat = Text::new(s.fetch_category).prompt().unwrap_or_default();
    let cat = cat.trim();
    super::fetch::run(
        &repo,
        if cat.is_empty() { None } else { Some(cat) },
        None,
        "main",
        "models",
    )
}

fn menu_pack(s: &Strings) -> Result<()> {
    let dirs = discover_model_dirs("models");
    if dirs.is_empty() {
        println!("  {}", s.pack_no_dirs);
        return Ok(());
    }
    let dir = match Select::new(s.pack_pick_dir, dirs).prompt() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let modes = vec![
        s.pack_mode_q8,
        s.pack_mode_rans,
        s.pack_mode_q4,
        s.pack_mode_raw,
    ];
    let mode = match Select::new(s.pack_mode, modes).prompt() {
        Ok(m) => m,
        Err(_) => return Ok(()),
    };
    let (codec, quant): (&str, Option<&str>) = if mode == s.pack_mode_q8 {
        ("raw", Some("q8_transformer_keep_io"))
    } else if mode == s.pack_mode_rans {
        ("rans", None)
    } else if mode == s.pack_mode_q4 {
        ("raw", Some("q4_0_keep_io"))
    } else {
        ("raw", None)
    };
    let base = Path::new(&dir)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model");
    let default_out = format!("models/{base}.spsa");
    // `with_initial_value` pre-fills an EDITABLE buffer; `with_default` only applies on empty
    // input — that quirk is why a typed name used to land as a bare file in the CWD.
    let raw = match Text::new(s.pack_output)
        .with_initial_value(&default_out)
        .prompt()
    {
        Ok(o) => o,
        Err(_) => return Ok(()),
    };
    let out = normalize_pack_output(&raw, &default_out);
    super::pack::run(
        &dir, &out, "1mb", codec, None, None, false, false, false, 16, None, None, false, quant,
        false, // clean animated UX (not verbose)
    )
}

fn menu_chat(s: &Strings) -> Result<()> {
    let model = match pick_rllm(s.chat_pick, s) {
        Some(m) => m,
        None => return Ok(()),
    };
    let fast = Confirm::new(s.chat_fast)
        .with_default(true)
        .prompt()
        .unwrap_or(true);

    // Max tokens per reply (Enter = default). Falls back to 512 on bad/empty input.
    let max_new_tokens = Text::new(s.chat_maxtok)
        .with_default("512")
        .prompt()
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(512);

    // Optional system prompt (Qwen ChatML / Llama). Enter = skip.
    let system = Text::new(s.chat_system)
        .with_help_message(s.chat_system_help)
        .prompt()
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());

    // Generation style → a friendly preset over the raw sampling flags.
    let style = Select::new(s.chat_style, vec![s.style_greedy, s.style_creative])
        .prompt()
        .unwrap_or(s.style_greedy);
    let sampling = if style == s.style_creative {
        let temp = Text::new(s.chat_temp)
            .with_default("0.7")
            .prompt()
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
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

fn menu_inspect(s: &Strings) -> Result<()> {
    match pick_rllm(s.inspect_pick, s) {
        Some(m) => super::inspect::run(&m),
        None => Ok(()),
    }
}

/// Prompt the user to pick a `.spsa` from `models/`; `None` = none found or cancelled.
fn pick_rllm(prompt: &str, s: &Strings) -> Option<String> {
    let models = discover_rllm("models");
    if models.is_empty() {
        println!("  {}", s.no_spsa);
        return None;
    }
    Select::new(prompt, models).prompt().ok()
}

/// Make the packed output path forgiving: empty → `default`; a bare name (no `/`) → placed
/// under `models/`; the `.spsa` extension is always ensured. So typing `my-model` yields
/// `models/my-model.spsa` instead of a bare `my-model` file in the working directory.
fn normalize_pack_output(input: &str, default: &str) -> String {
    let t = input.trim();
    if t.is_empty() {
        return default.to_string();
    }
    let mut out = if t.contains('/') || t.contains('\\') {
        t.to_string()
    } else {
        format!("models/{t}")
    };
    if !out.ends_with(".spsa") {
        out.push_str(".spsa");
    }
    out
}

// ---- model discovery under models/ ----

fn discover_rllm(root: &str) -> Vec<String> {
    let mut out = Vec::new();
    walk(Path::new(root), 4, &mut |p, is_dir| {
        if !is_dir && p.extension().and_then(|e| e.to_str()) == Some("spsa") {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i18n_table_and_settings_roundtrip() {
        // Both languages exist and actually differ.
        assert_ne!(strings(Lang::En).menu_prompt, strings(Lang::Id).menu_prompt);
        assert_eq!(strings(Lang::En).act_quit, "🚪  Quit");
        assert_eq!(strings(Lang::Id).act_quit, "🚪  Keluar");

        // Default language is English (no settings file → English).
        assert_eq!(Settings::default().language, Lang::En);

        // Persistence format round-trips and stores the short code.
        let json = serde_json::to_string(&Settings {
            language: Lang::Id,
        })
        .unwrap();
        assert!(json.contains("\"id\""), "unexpected json: {json}");
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.language, Lang::Id);

        // A missing/blank field falls back to the default (English).
        let back: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(back.language, Lang::En);
    }

    #[test]
    fn pack_output_is_normalized() {
        let def = "models/qwen.spsa";
        // empty → default
        assert_eq!(normalize_pack_output("  ", def), def);
        // bare name → under models/ + .spsa
        assert_eq!(normalize_pack_output("my-model", def), "models/my-model.spsa");
        // bare name already with ext
        assert_eq!(normalize_pack_output("my-model.spsa", def), "models/my-model.spsa");
        // explicit path is respected, ext ensured
        assert_eq!(normalize_pack_output("out/x", def), "out/x.spsa");
        assert_eq!(normalize_pack_output("models/text/x.spsa", def), "models/text/x.spsa");
    }
}
