//! Tiny no-dependency terminal progress: a braille spinner animated on a background thread
//! (its message is updated live from the main thread), plus helpers for a clean pack result
//! box. Hand-rolled on purpose — keeps the single self-contained binary dependency-free.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// A single-line spinner that animates while work runs. The main thread calls [`set`] to
/// change the trailing message; the background thread owns the line, so the caller must not
/// print to stdout while the spinner is live. End with [`clear`].
pub struct Spinner {
    stop: Arc<AtomicBool>,
    msg: Arc<Mutex<String>>,
    handle: Option<JoinHandle<()>>,
}

impl Spinner {
    pub fn start(initial: impl Into<String>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let msg = Arc::new(Mutex::new(initial.into()));
        let (st, ms) = (stop.clone(), msg.clone());
        let handle = thread::spawn(move || {
            let mut i = 0usize;
            while !st.load(Ordering::Relaxed) {
                let text = ms.lock().map(|m| m.clone()).unwrap_or_default();
                // \r → line start, \x1b[2K → clear line, green frame, message.
                print!(
                    "\r\x1b[2K\x1b[92m{}\x1b[0m  {text}",
                    FRAMES[i % FRAMES.len()]
                );
                io::stdout().flush().ok();
                i += 1;
                thread::sleep(Duration::from_millis(80));
            }
        });
        Self {
            stop,
            msg,
            handle: Some(handle),
        }
    }

    pub fn set(&self, m: impl Into<String>) {
        if let Ok(mut g) = self.msg.lock() {
            *g = m.into();
        }
    }

    /// Stop animating and wipe the line, leaving the cursor at column 0 for the next print.
    pub fn clear(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }
        print!("\r\x1b[2K");
        io::stdout().flush().ok();
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        // Safety net if `clear` was never called (e.g. early return / panic).
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }
    }
}

/// A `▕████░░░▏`-style bar for `frac` in `[0,1]` over `width` cells.
pub fn bar(frac: f64, width: usize) -> String {
    let filled = ((frac.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    format!("▕{}{}▏", "█".repeat(filled), "░".repeat(width - filled))
}

/// One green braille spinner frame for a synchronous (no background thread) progress line —
/// pass a counter or `elapsed_ms/80` so it cycles. Used by the download checklist.
pub fn spinner_frame(i: usize) -> String {
    format!("\x1b[92m{}\x1b[0m", FRAMES[i % FRAMES.len()])
}

/// Human-readable byte size, e.g. `3.76 GB`.
pub fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

/// Draw the green result panel: a `✓ headline` line above a titled box of rows.
fn render_box(headline: &str, title_name: &str, rows: &[String]) {
    let title = format!(" {title_name} ");
    let content = rows
        .iter()
        .map(|r| r.chars().count())
        .max()
        .unwrap_or(0)
        .max(title_name.chars().count())
        .max(40);
    let pad = 2usize;
    let w = content + pad * 2; // inner cells between the borders

    println!();
    println!("  \x1b[1;92m✓\x1b[0m  {headline}");
    println!();
    let top_fill = w.saturating_sub(1 + title.chars().count());
    println!("  \x1b[92m┌─{title}{}┐\x1b[0m", "─".repeat(top_fill));
    for r in rows {
        let right = w.saturating_sub(pad + r.chars().count());
        println!(
            "  \x1b[92m│\x1b[0m{}{r}{}\x1b[92m│\x1b[0m",
            " ".repeat(pad),
            " ".repeat(right)
        );
    }
    println!("  \x1b[92m└{}┘\x1b[0m", "─".repeat(w));
}

/// Print the post-pack summary box (the green-check result panel).
#[allow(clippy::too_many_arguments)]
pub fn print_pack_result(
    filename: &str,
    codec: &str,
    tensors: usize,
    chunks: usize,
    original: u64,
    compressed: u64,
    output_path: &str,
    elapsed_secs: f64,
) {
    let saved = original.saturating_sub(compressed);
    // Reduction vs original, e.g. -24% (clearer than the raw compressed/original ratio).
    let reduction = if original > 0 {
        (1.0 - compressed as f64 / original as f64) * 100.0
    } else {
        0.0
    };
    let rows = [
        format!("codec     {codec}"),
        format!("tensors   {tensors}   ·   chunks {chunks}"),
        format!(
            "size      {} → {}    −{reduction:.0}%  (saved {})",
            human_size(original),
            human_size(compressed),
            human_size(saved)
        ),
        format!("output    {output_path}"),
    ];
    render_box(&format!("Packed in {elapsed_secs:.1}s"), filename, &rows);
}

/// Print the post-download summary box (mirrors the pack result panel).
pub fn print_fetch_result(
    name: &str,
    files: usize,
    total_bytes: u64,
    downloaded_bytes: u64,
    elapsed_secs: f64,
    dest: &str,
) {
    let avg = if elapsed_secs > 0.0 {
        (downloaded_bytes as f64 / elapsed_secs) as u64
    } else {
        0
    };
    let rows = [
        format!("files     {files}   ·   {}", human_size(total_bytes)),
        format!("avg       {}/s", human_size(avg)),
        format!("path      {dest}"),
    ];
    render_box(&format!("Downloaded in {elapsed_secs:.1}s"), name, &rows);
}
