use crate::{Result, RuntimeError};
use rllm_container::TokenizerMetadata;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RllmTokenizer {
    id_to_token: Vec<String>,
    id_to_text: Vec<String>,
    token_to_id: HashMap<String, usize>,
    bpe_ranks: HashMap<(String, String), usize>,
    special_token_candidates: Vec<(String, usize)>,
    encode_candidates: Vec<(String, usize)>,
    unk_token_id: Option<usize>,
}

impl RllmTokenizer {
    pub fn from_metadata(metadata: &TokenizerMetadata) -> Result<Self> {
        if metadata.id_to_token.is_empty() {
            return Err(RuntimeError::InvalidTensorData(
                "tokenizer metadata must contain at least one token".to_string(),
            ));
        }
        let unk_token_id = metadata
            .unk_token_id
            .map(usize::try_from)
            .transpose()
            .map_err(|_| {
                RuntimeError::Shape("tokenizer unk_token_id overflows usize".to_string())
            })?;
        if let Some(id) = unk_token_id {
            if id >= metadata.id_to_token.len() {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "tokenizer unk_token_id {id} is outside vocab size {}",
                    metadata.id_to_token.len()
                )));
            }
        }

        let id_to_text: Vec<String> = metadata
            .id_to_token
            .iter()
            .map(|token| token_to_text_surface(token))
            .collect();
        let token_to_id: HashMap<String, usize> = metadata
            .id_to_token
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id))
            .collect();
        let bpe_ranks: HashMap<(String, String), usize> = metadata
            .bpe_merges
            .iter()
            .enumerate()
            .map(|(rank, (left, right))| ((left.clone(), right.clone()), rank))
            .collect();
        let mut special_token_candidates: Vec<(String, usize)> = metadata
            .id_to_token
            .iter()
            .enumerate()
            .filter(|(_, token)| is_raw_special_token(token))
            .map(|(id, token)| (token.clone(), id))
            .collect();
        special_token_candidates.sort_by(|(left_text, left_id), (right_text, right_id)| {
            right_text
                .len()
                .cmp(&left_text.len())
                .then_with(|| left_id.cmp(right_id))
        });
        let mut encode_candidates: Vec<(String, usize)> = id_to_text
            .iter()
            .enumerate()
            .filter(|(_, text)| !text.is_empty())
            .map(|(id, text)| (text.clone(), id))
            .collect();
        encode_candidates.sort_by(|(left_text, left_id), (right_text, right_id)| {
            right_text
                .len()
                .cmp(&left_text.len())
                .then_with(|| left_id.cmp(right_id))
        });

        Ok(Self {
            id_to_token: metadata.id_to_token.clone(),
            id_to_text,
            token_to_id,
            bpe_ranks,
            special_token_candidates,
            encode_candidates,
            unk_token_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn token_id_for_raw_token(&self, token: &str) -> Option<usize> {
        self.id_to_token
            .iter()
            .position(|candidate| candidate == token)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        if text.is_empty() {
            return Err(RuntimeError::InvalidTensorData(
                "prompt text must not be empty".to_string(),
            ));
        }
        if !self.bpe_ranks.is_empty() {
            return self.encode_bpe(text);
        }
        self.encode_greedy(text)
    }

    fn encode_greedy(&self, text: &str) -> Result<Vec<usize>> {
        let mut token_ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let remaining = &text[offset..];
            if let Some((surface, token_id)) = self
                .special_token_candidates
                .iter()
                .find(|(surface, _)| remaining.starts_with(surface))
            {
                token_ids.push(*token_id);
                offset += surface.len();
                continue;
            }

            let next_special_offset = self
                .special_token_candidates
                .iter()
                .filter_map(|(surface, _)| remaining.find(surface))
                .filter(|position| *position > 0)
                .min();
            if let Some((surface, token_id)) = self.encode_candidates.iter().find(|(surface, _)| {
                remaining.starts_with(surface)
                    && next_special_offset.is_none_or(|limit| surface.len() <= limit)
            }) {
                token_ids.push(*token_id);
                offset += surface.len();
                continue;
            }

            if let Some(unk_token_id) = self.unk_token_id {
                let next_char = remaining.chars().next().ok_or_else(|| {
                    RuntimeError::InvalidTensorData("invalid empty tokenizer remainder".to_string())
                })?;
                token_ids.push(unk_token_id);
                offset += next_char.len_utf8();
                continue;
            }

            let preview: String = remaining.chars().take(16).collect();
            return Err(RuntimeError::InvalidTensorData(format!(
                "tokenizer could not encode text starting at byte {offset}: {preview:?}"
            )));
        }
        Ok(token_ids)
    }

    fn encode_bpe(&self, text: &str) -> Result<Vec<usize>> {
        let mut token_ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let remaining = &text[offset..];
            if let Some((surface, token_id)) = self
                .special_token_candidates
                .iter()
                .find(|(surface, _)| remaining.starts_with(surface))
            {
                token_ids.push(*token_id);
                offset += surface.len();
                continue;
            }

            let next_special_offset = self
                .special_token_candidates
                .iter()
                .filter_map(|(surface, _)| remaining.find(surface))
                .filter(|position| *position > 0)
                .min()
                .unwrap_or(remaining.len());
            let segment = &remaining[..next_special_offset];
            if segment.is_empty() {
                return Err(RuntimeError::InvalidTensorData(
                    "invalid empty BPE tokenizer segment".to_string(),
                ));
            }
            for pretoken in byte_level_pretokens(segment) {
                token_ids.extend(self.bpe_piece_ids(&pretoken)?);
            }
            offset += segment.len();
        }
        Ok(token_ids)
    }

    fn bpe_piece_ids(&self, encoded: &str) -> Result<Vec<usize>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }
        let mut parts: Vec<String> = encoded.chars().map(|ch| ch.to_string()).collect();
        loop {
            let Some((merge_index, _rank)) = parts
                .windows(2)
                .enumerate()
                .filter_map(|(idx, pair)| {
                    self.bpe_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (idx, *rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };
            let merged = format!("{}{}", parts[merge_index], parts[merge_index + 1]);
            parts.splice(merge_index..=merge_index + 1, [merged]);
        }

        let mut ids = Vec::with_capacity(parts.len());
        for part in parts {
            if let Some(id) = self.token_to_id.get(&part) {
                ids.push(*id);
            } else if let Some(unk_token_id) = self.unk_token_id {
                ids.push(unk_token_id);
            } else {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "BPE token {part:?} is not present in tokenizer vocab"
                )));
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let mut text = String::new();
        for &token_id in token_ids {
            let token = self.id_to_text.get(token_id).ok_or_else(|| {
                RuntimeError::InvalidTensorData(format!(
                    "token id {token_id} is outside tokenizer vocab size {}",
                    self.id_to_text.len()
                ))
            })?;
            text.push_str(token);
        }
        Ok(text)
    }
}

fn token_to_text_surface(token: &str) -> String {
    token.replace('Ġ', " ").replace('Ċ', "\n")
}

fn is_raw_special_token(token: &str) -> bool {
    token.len() >= 4 && token.starts_with('<') && token.ends_with('>')
}

fn byte_level_pretokens(segment: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut pending_spaces = 0usize;
    let chars: Vec<char> = segment.chars().collect();
    let mut idx = 0usize;
    while idx < chars.len() {
        let ch = chars[idx];
        if ch == ' ' {
            pending_spaces += 1;
            idx += 1;
            continue;
        }
        if ch == '\n' {
            flush_pending_spaces(&mut tokens, &mut pending_spaces);
            tokens.push("Ċ".to_string());
            idx += 1;
            continue;
        }

        if ch.is_ascii_digit() {
            flush_pending_spaces(&mut tokens, &mut pending_spaces);
            tokens.push(ch.to_string());
            idx += 1;
            continue;
        }

        let mut token = String::new();
        if pending_spaces > 0 {
            token.extend(std::iter::repeat_n('Ġ', pending_spaces));
            pending_spaces = 0;
        }
        let class = byte_level_char_class(ch);
        while idx < chars.len() {
            let current = chars[idx];
            if current == ' ' || current == '\n' || current.is_ascii_digit() {
                break;
            }
            if byte_level_char_class(current) != class {
                break;
            }
            token.push(current);
            idx += 1;
        }
        tokens.push(token);
    }
    flush_pending_spaces(&mut tokens, &mut pending_spaces);
    tokens
}

fn flush_pending_spaces(tokens: &mut Vec<String>, pending_spaces: &mut usize) {
    if *pending_spaces > 0 {
        tokens.push("Ġ".repeat(*pending_spaces));
        *pending_spaces = 0;
    }
}

fn byte_level_char_class(ch: char) -> u8 {
    if ch.is_alphabetic() {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_encode_and_decode_use_literal_token_surfaces() {
        let tokenizer = RllmTokenizer::from_metadata(&TokenizerMetadata {
            tokenizer_type: Some("hf-wordlevel".to_string()),
            id_to_token: vec![
                "Hello".to_string(),
                " world".to_string(),
                "<unk>".to_string(),
            ],
            bpe_merges: Vec::new(),
            unk_token_id: Some(2),
            bos_token_id: None,
            eos_token_id: None,
        })
        .unwrap();

        assert_eq!(tokenizer.encode("Hello world").unwrap(), [0, 1]);
        assert_eq!(tokenizer.decode(&[0, 1]).unwrap(), "Hello world");
        assert_eq!(tokenizer.encode("Hello!").unwrap(), [0, 2]);
    }

    #[test]
    fn decode_maps_common_byte_level_space_and_newline_markers() {
        let tokenizer = RllmTokenizer::from_metadata(&TokenizerMetadata {
            tokenizer_type: Some("hf-bpe".to_string()),
            id_to_token: vec!["Hello".to_string(), "Ġworld".to_string(), "Ċ".to_string()],
            bpe_merges: Vec::new(),
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
        })
        .unwrap();

        assert_eq!(tokenizer.encode("Hello world\n").unwrap(), [0, 1, 2]);
        assert_eq!(tokenizer.decode(&[0, 1, 2]).unwrap(), "Hello world\n");
    }

    #[test]
    fn token_id_for_raw_token_finds_special_added_tokens() {
        let tokenizer = RllmTokenizer::from_metadata(&TokenizerMetadata {
            tokenizer_type: Some("hf-bpe".to_string()),
            id_to_token: vec![
                "Hello".to_string(),
                "<|begin_of_text|>".to_string(),
                "<|eot_id|>".to_string(),
            ],
            bpe_merges: Vec::new(),
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
        })
        .unwrap();

        assert_eq!(
            tokenizer.token_id_for_raw_token("<|begin_of_text|>"),
            Some(1)
        );
        assert_eq!(tokenizer.token_id_for_raw_token("<|eot_id|>"), Some(2));
        assert_eq!(tokenizer.token_id_for_raw_token("<missing>"), None);
    }

    #[test]
    fn encode_preserves_raw_special_token_boundaries() {
        let tokenizer = RllmTokenizer::from_metadata(&TokenizerMetadata {
            tokenizer_type: Some("hf-bpe".to_string()),
            id_to_token: vec![
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
                "?".to_string(),
                "?<".to_string(),
                "assistant".to_string(),
                "Ċ".to_string(),
            ],
            bpe_merges: Vec::new(),
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
        })
        .unwrap();

        assert_eq!(
            tokenizer
                .encode("?<|im_end|>\n<|im_start|>assistant\n")
                .unwrap(),
            vec![2, 1, 5, 0, 4, 5]
        );
    }

    #[test]
    fn bpe_encode_uses_merges_without_crossing_special_boundaries() {
        let tokenizer = RllmTokenizer::from_metadata(&TokenizerMetadata {
            tokenizer_type: Some("hf-bpe".to_string()),
            id_to_token: vec![
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
                "?".to_string(),
                "?<".to_string(),
                "ass".to_string(),
                "ist".to_string(),
                "ant".to_string(),
                "istant".to_string(),
                "Ċ".to_string(),
                "i".to_string(),
                "s".to_string(),
                "t".to_string(),
                "a".to_string(),
                "n".to_string(),
            ],
            bpe_merges: vec![
                ("a".to_string(), "s".to_string()),
                ("as".to_string(), "s".to_string()),
                ("i".to_string(), "s".to_string()),
                ("is".to_string(), "t".to_string()),
                ("a".to_string(), "n".to_string()),
                ("an".to_string(), "t".to_string()),
                ("ist".to_string(), "ant".to_string()),
            ],
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
        })
        .unwrap();

        assert_eq!(
            tokenizer
                .encode("?<|im_end|>\n<|im_start|>assistant\n")
                .unwrap(),
            vec![2, 1, 8, 0, 4, 7, 8]
        );
    }
}
