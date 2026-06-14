use anyhow::{Context, Result};

fn parse_token_turns(turns: &[String]) -> Result<Vec<Vec<usize>>> {
    if turns.is_empty() {
        anyhow::bail!("chat-session-token requires at least one --turn-ids");
    }
    turns
        .iter()
        .enumerate()
        .map(|(turn_idx, raw)| parse_token_turn(turn_idx + 1, raw))
        .collect()
}

fn parse_token_turn(turn_idx: usize, raw: &str) -> Result<Vec<usize>> {
    if raw.trim().is_empty() {
        anyhow::bail!("chat-session-token turn {turn_idx} must not be empty");
    }
    let mut ids = Vec::new();
    for (item_idx, part) in raw.split(',').enumerate() {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            anyhow::bail!(
                "chat-session-token turn {turn_idx} has empty token id at position {}",
                item_idx + 1
            );
        }
        ids.push(trimmed.parse::<usize>().with_context(|| {
            format!(
                "invalid token id in chat-session-token turn {turn_idx} at position {}",
                item_idx + 1
            )
        })?);
    }
    Ok(ids)
}

fn validate_report_output_path(out: &str) -> Result<()> {
    let components = normalized_path_components(out);
    for window in components.windows(4) {
        if window[0] == "docs"
            && window[1] == "benchmarks"
            && window[2] == "trials"
            && matches!(window[3].as_str(), "success" | "failed" | "inconclusive")
        {
            anyhow::bail!(
                "chat-session-token writes active reports; use docs/benchmarks/trials/active/ and move the report after review"
            );
        }
    }
    Ok(())
}

fn normalized_path_components(path: &str) -> Vec<String> {
    let mut components = Vec::new();
    let normalized = path.replace('\\', "/");
    for part in normalized.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if components
                    .last()
                    .is_some_and(|component: &String| component != "..")
                {
                    components.pop();
                } else {
                    components.push(part.to_string());
                }
            }
            _ => components.push(part.to_ascii_lowercase()),
        }
    }
    components
}

#[cfg(test)]
mod tests {
    use super::{parse_token_turns, validate_report_output_path};

    #[test]
    fn parse_token_turns_accepts_comma_separated_turns() {
        let turns = vec!["1, 2,3".to_string(), "4".to_string()];

        assert_eq!(
            parse_token_turns(&turns).unwrap(),
            vec![vec![1, 2, 3], vec![4]]
        );
    }

    #[test]
    fn parse_token_turns_rejects_empty_and_invalid_values() {
        assert!(parse_token_turns(&[])
            .unwrap_err()
            .to_string()
            .contains("at least one"));
        assert!(parse_token_turns(&["".to_string()])
            .unwrap_err()
            .to_string()
            .contains("empty"));
        assert!(parse_token_turns(&["1,,2".to_string()])
            .unwrap_err()
            .to_string()
            .contains("empty token id"));
        assert!(parse_token_turns(&["1,nope".to_string()])
            .unwrap_err()
            .to_string()
            .contains("invalid token id"));
    }

    #[test]
    fn token_report_output_rejects_reviewed_trial_folders() {
        validate_report_output_path("docs/benchmarks/trials/active/r2.md").unwrap();

        for path in [
            "docs/benchmarks/trials/success/r2.md",
            "docs/benchmarks/trials/active/../failed/r2.md",
            "docs/benchmarks/trials/Inconclusive/r2.md",
        ] {
            assert!(validate_report_output_path(path)
                .unwrap_err()
                .to_string()
                .contains("active reports"));
        }
    }
}
