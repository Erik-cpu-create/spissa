// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! Shared helpers for benchmark matrices: deterministic token patterns, CSV
//! list parsing, numeric formatting, and CSV/Markdown writers.

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::Path;

/// Deterministic, known-valid Pythia token IDs reused to exercise real input
/// token counts without tokenizer effects. All IDs are below the Pythia-70M
/// vocab size (50304).
pub const TOKEN_PATTERN: [u32; 16] = [
    12092, 13, 187, 309, 352, 359, 42, 253, 849, 619, 627, 368, 198, 318, 262, 257,
];

/// Build a deterministic token-id vector of the requested length by repeating
/// [`TOKEN_PATTERN`].
pub fn make_token_ids(length: usize) -> Result<Vec<u32>> {
    if length == 0 {
        return Err(anyhow!("token length must be positive"));
    }
    Ok((0..length)
        .map(|i| TOKEN_PATTERN[i % TOKEN_PATTERN.len()])
        .collect())
}

/// Comma-separated `--token-ids` argument of the requested length.
pub fn token_ids_arg(length: usize) -> Result<String> {
    let ids = make_token_ids(length)?;
    Ok(ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(","))
}

/// Parse a comma-separated list of positive integers (e.g. "1,4,8,16").
pub fn parse_csv_ints(raw: &str, flag_name: &str) -> Result<Vec<usize>> {
    let mut values = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let value: usize = part
            .parse()
            .with_context(|| format!("{flag_name} must be comma-separated integers: {raw:?}"))?;
        if value == 0 {
            return Err(anyhow!("{flag_name} values must be > 0: {value}"));
        }
        values.push(value);
    }
    if values.is_empty() {
        return Err(anyhow!("{flag_name} must contain at least one value"));
    }
    Ok(values)
}

/// Format an optional float with fixed precision; `None` becomes empty string.
pub fn fmt_f(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(v) => format!("{v:.precision$}"),
        None => String::new(),
    }
}

/// Format an optional unsigned integer; `None` becomes empty string.
pub fn fmt_u(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

/// Mean of a slice, or `None` when empty.
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Minimum of a slice, or `None` when empty.
pub fn min(values: &[f64]) -> Option<f64> {
    values.iter().copied().fold(None, |acc, v| match acc {
        Some(a) if a <= v => Some(a),
        _ => Some(v),
    })
}

/// Maximum of a slice, or `None` when empty.
pub fn max(values: &[f64]) -> Option<f64> {
    values.iter().copied().fold(None, |acc, v| match acc {
        Some(a) if a >= v => Some(a),
        _ => Some(v),
    })
}

/// Write a CSV file from a header and string rows.
pub fn write_csv(path: &Path, header: &[&str], rows: &[Vec<String>]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    let mut body = String::new();
    body.push_str(
        &header
            .iter()
            .map(|h| csv_escape(h))
            .collect::<Vec<_>>()
            .join(","),
    );
    body.push('\n');
    for row in rows {
        body.push_str(
            &row.iter()
                .map(|f| csv_escape(f))
                .collect::<Vec<_>>()
                .join(","),
        );
        body.push('\n');
    }
    fs::write(path, body).with_context(|| format!("failed to write CSV {}", path.display()))
}

/// Write a UTF-8 text file, creating parent directories.
pub fn write_text(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))
}

fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_ids_repeat_pattern() {
        assert_eq!(make_token_ids(3).unwrap(), vec![12092, 13, 187]);
        assert_eq!(make_token_ids(18).unwrap()[16], 12092);
        assert_eq!(token_ids_arg(2).unwrap(), "12092,13");
    }

    #[test]
    fn parse_csv_ints_rejects_zero_and_empty() {
        assert_eq!(
            parse_csv_ints("1,4, 8 ,16", "--tokens").unwrap(),
            vec![1, 4, 8, 16]
        );
        assert!(parse_csv_ints("0", "--tokens").is_err());
        assert!(parse_csv_ints(" , ", "--tokens").is_err());
    }

    #[test]
    fn csv_escapes_separators_and_quotes() {
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_escape("plain"), "plain");
    }

    #[test]
    fn stats_handle_empty_and_values() {
        assert_eq!(mean(&[]), None);
        assert_eq!(min(&[3.0, 1.0, 2.0]), Some(1.0));
        assert_eq!(max(&[3.0, 1.0, 2.0]), Some(3.0));
        assert_eq!(mean(&[2.0, 4.0]), Some(3.0));
    }
}
