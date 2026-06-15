use crate::{Result, RuntimeError};
use rllm_container::TokenizerMetadata;

#[derive(Debug, Clone)]
pub struct RllmTokenizer {
    id_to_token: Vec<String>,
    id_to_text: Vec<String>,
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
        let mut token_ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let remaining = &text[offset..];
            if let Some((surface, token_id)) = self
                .encode_candidates
                .iter()
                .find(|(surface, _)| remaining.starts_with(surface))
            {
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
}
