use crate::{Result, RuntimeError};
use rllm_container::TokenizerMetadata;
use std::collections::HashMap;

/// SentencePiece metaspace marker (U+2581) used by Gemma for spaces.
const METASPACE: char = '▁';

/// Pre-tokenization / surface scheme. `ByteLevel` is GPT-2 style (`Ġ` spaces,
/// `Ċ` newlines); `Metaspace` is SentencePiece style (`▁` spaces, `<0xNN>`
/// byte fallback), used by Gemma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizerScheme {
    ByteLevel,
    Metaspace,
}

#[derive(Debug, Clone)]
pub struct RllmTokenizer {
    id_to_token: Vec<String>,
    id_to_text: Vec<String>,
    token_to_id: HashMap<String, usize>,
    bpe_ranks: HashMap<(String, String), usize>,
    special_token_candidates: Vec<(String, usize)>,
    encode_candidates: Vec<(String, usize)>,
    unk_token_id: Option<usize>,
    scheme: PreTokenizerScheme,
    add_bos_id: Option<usize>,
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

        let scheme = match metadata.pre_tokenizer.as_deref() {
            Some("metaspace") => PreTokenizerScheme::Metaspace,
            _ => PreTokenizerScheme::ByteLevel,
        };
        let add_bos_id = if metadata.add_bos_token == Some(true) {
            metadata.bos_token_id.and_then(|id| usize::try_from(id).ok())
        } else {
            None
        };

        let id_to_text: Vec<String> = metadata
            .id_to_token
            .iter()
            .map(|token| token_to_text_surface(token, scheme))
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
        // Byte-fallback tokens (`<0xNN>`) syntactically look like specials but
        // must never be matched against raw input text — exclude them.
        let mut special_token_candidates: Vec<(String, usize)> = metadata
            .id_to_token
            .iter()
            .enumerate()
            .filter(|(_, token)| is_raw_special_token(token) && byte_fallback_value(token).is_none())
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
            scheme,
            add_bos_id,
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
        if let Some(bos) = self.add_bos_id {
            token_ids.push(bos);
        }
        if self.bpe_ranks.is_empty() {
            self.encode_greedy(text, &mut token_ids)?;
        } else {
            self.encode_bpe(text, &mut token_ids)?;
        }
        Ok(token_ids)
    }

    fn encode_greedy(&self, text: &str, token_ids: &mut Vec<usize>) -> Result<()> {
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
        Ok(())
    }

    fn encode_bpe(&self, text: &str, token_ids: &mut Vec<usize>) -> Result<()> {
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
            match self.scheme {
                PreTokenizerScheme::ByteLevel => {
                    for pretoken in byte_level_pretokens(segment) {
                        self.bpe_piece_ids(&pretoken, token_ids)?;
                    }
                }
                PreTokenizerScheme::Metaspace => {
                    let encoded = segment.replace(' ', "▁");
                    self.bpe_piece_ids(&encoded, token_ids)?;
                }
            }
            offset += segment.len();
        }
        Ok(())
    }

    fn bpe_piece_ids(&self, encoded: &str, token_ids: &mut Vec<usize>) -> Result<()> {
        if encoded.is_empty() {
            return Ok(());
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

        for part in parts {
            if let Some(id) = self.token_to_id.get(&part) {
                token_ids.push(*id);
            } else if self.scheme == PreTokenizerScheme::Metaspace
                && self.push_byte_fallback(&part, token_ids)
            {
                // handled: decomposed into `<0xNN>` byte tokens
            } else if let Some(unk_token_id) = self.unk_token_id {
                token_ids.push(unk_token_id);
            } else {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "BPE token {part:?} is not present in tokenizer vocab"
                )));
            }
        }
        Ok(())
    }

    /// SentencePiece byte fallback: decompose an out-of-vocab piece into its
    /// UTF-8 bytes and map each to a `<0xNN>` token. Returns false (no tokens
    /// pushed) if any byte token is missing from the vocab.
    fn push_byte_fallback(&self, part: &str, token_ids: &mut Vec<usize>) -> bool {
        let mut byte_ids = Vec::with_capacity(part.len());
        for byte in part.bytes() {
            match self.token_to_id.get(&format!("<0x{byte:02X}>")) {
                Some(id) => byte_ids.push(*id),
                None => return false,
            }
        }
        token_ids.extend(byte_ids);
        true
    }

    pub fn decode(&self, token_ids: &[usize]) -> Result<String> {
        match self.scheme {
            PreTokenizerScheme::ByteLevel => self.decode_byte_level(token_ids),
            PreTokenizerScheme::Metaspace => self.decode_metaspace(token_ids),
        }
    }

    fn decode_byte_level(&self, token_ids: &[usize]) -> Result<String> {
        // GPT-2/tiktoken byte-level decode: each char in the raw token maps back
        // to one byte (the inverse of the bytes->unicode alphabet); reassemble the
        // bytes and interpret as UTF-8. Without this, multi-byte sequences (emoji,
        // accents) leak their raw byte glyphs (e.g. 🤔 -> "ðŁ¤Ķ"). ASCII bytes map
        // to themselves, so plain text is unchanged.
        let map = byte_level_char_to_byte();
        let mut bytes: Vec<u8> = Vec::new();
        for &token_id in token_ids {
            let raw = self.id_to_token.get(token_id).ok_or_else(|| {
                RuntimeError::InvalidTensorData(format!(
                    "token id {token_id} is outside tokenizer vocab size {}",
                    self.id_to_token.len()
                ))
            })?;
            for ch in raw.chars() {
                if let Some(&b) = map.get(&ch) {
                    bytes.push(b);
                } else {
                    // Not a byte-level glyph (e.g. a raw special token): keep as UTF-8.
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Decode SentencePiece output: reassemble runs of `<0xNN>` byte-fallback
    /// tokens into UTF-8 before rendering, and map `▁` back to spaces.
    fn decode_metaspace(&self, token_ids: &[usize]) -> Result<String> {
        let mut text = String::new();
        let mut byte_run: Vec<u8> = Vec::new();
        for &token_id in token_ids {
            let raw = self.id_to_token.get(token_id).ok_or_else(|| {
                RuntimeError::InvalidTensorData(format!(
                    "token id {token_id} is outside tokenizer vocab size {}",
                    self.id_to_token.len()
                ))
            })?;
            if let Some(byte) = byte_fallback_value(raw) {
                byte_run.push(byte);
                continue;
            }
            if !byte_run.is_empty() {
                text.push_str(&String::from_utf8_lossy(&byte_run));
                byte_run.clear();
            }
            text.push_str(&self.surface(token_id)?);
        }
        if !byte_run.is_empty() {
            text.push_str(&String::from_utf8_lossy(&byte_run));
        }
        Ok(text)
    }

    fn surface(&self, token_id: usize) -> Result<&str> {
        self.id_to_text.get(token_id).map(String::as_str).ok_or_else(|| {
            RuntimeError::InvalidTensorData(format!(
                "token id {token_id} is outside tokenizer vocab size {}",
                self.id_to_text.len()
            ))
        })
    }
}

fn token_to_text_surface(token: &str, scheme: PreTokenizerScheme) -> String {
    match scheme {
        PreTokenizerScheme::ByteLevel => token.replace('Ġ', " ").replace('Ċ', "\n"),
        PreTokenizerScheme::Metaspace => token.replace(METASPACE, " "),
    }
}

/// True for the bytes GPT-2's `bytes_to_unicode` maps to themselves; the rest map
/// to `U+0100 + n` in ascending order.
fn byte_level_byte_is_printable(b: u8) -> bool {
    matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)
}

/// Inverse of GPT-2's byte-level alphabet: maps each visible char back to its
/// byte. Built once; used to reassemble UTF-8 when decoding byte-level tokens.
fn byte_level_char_to_byte() -> &'static HashMap<char, u8> {
    static MAP: std::sync::OnceLock<HashMap<char, u8>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = HashMap::with_capacity(256);
        for b in 0u16..256 {
            if byte_level_byte_is_printable(b as u8) {
                map.insert(b as u8 as char, b as u8);
            }
        }
        let mut n = 0u32;
        for b in 0u16..256 {
            if !byte_level_byte_is_printable(b as u8) {
                map.insert(char::from_u32(0x100 + n).unwrap(), b as u8);
                n += 1;
            }
        }
        map
    })
}

fn is_raw_special_token(token: &str) -> bool {
    token.len() >= 4 && token.starts_with('<') && token.ends_with('>')
}

/// Parse a SentencePiece byte-fallback token `<0xNN>` into its byte value.
fn byte_fallback_value(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
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

    fn meta(
        tokenizer_type: &str,
        id_to_token: Vec<String>,
        bpe_merges: Vec<(String, String)>,
        unk_token_id: Option<u64>,
    ) -> TokenizerMetadata {
        TokenizerMetadata {
            tokenizer_type: Some(tokenizer_type.to_string()),
            id_to_token,
            bpe_merges,
            unk_token_id,
            ..Default::default()
        }
    }

    #[test]
    fn greedy_encode_and_decode_use_literal_token_surfaces() {
        let tokenizer = RllmTokenizer::from_metadata(&meta(
            "hf-wordlevel",
            vec!["Hello".to_string(), " world".to_string(), "<unk>".to_string()],
            Vec::new(),
            Some(2),
        ))
        .unwrap();

        assert_eq!(tokenizer.encode("Hello world").unwrap(), [0, 1]);
        assert_eq!(tokenizer.decode(&[0, 1]).unwrap(), "Hello world");
        assert_eq!(tokenizer.encode("Hello!").unwrap(), [0, 2]);
    }

    #[test]
    fn decode_maps_common_byte_level_space_and_newline_markers() {
        let tokenizer = RllmTokenizer::from_metadata(&meta(
            "hf-bpe",
            vec!["Hello".to_string(), "Ġworld".to_string(), "Ċ".to_string()],
            Vec::new(),
            None,
        ))
        .unwrap();

        assert_eq!(tokenizer.encode("Hello world\n").unwrap(), [0, 1, 2]);
        assert_eq!(tokenizer.decode(&[0, 1, 2]).unwrap(), "Hello world\n");
    }

    #[test]
    fn decode_byte_level_reassembles_multibyte_utf8() {
        // Tokens are stored in GPT-2's byte-level alphabet: "Ã©" is the encoding of
        // the two UTF-8 bytes of "é", and "ðŁ¤Ķ" is the four bytes of 🤔. Decoding
        // must reverse the byte map and reassemble UTF-8, not leak the byte glyphs.
        let tokenizer = RllmTokenizer::from_metadata(&meta(
            "hf-bpe",
            vec!["caf".to_string(), "Ã©".to_string(), "ðŁ¤Ķ".to_string()],
            Vec::new(),
            None,
        ))
        .unwrap();

        assert_eq!(tokenizer.decode(&[0, 1]).unwrap(), "café");
        assert_eq!(tokenizer.decode(&[2]).unwrap(), "🤔");
    }

    #[test]
    fn token_id_for_raw_token_finds_special_added_tokens() {
        let tokenizer = RllmTokenizer::from_metadata(&meta(
            "hf-bpe",
            vec![
                "Hello".to_string(),
                "<|begin_of_text|>".to_string(),
                "<|eot_id|>".to_string(),
            ],
            Vec::new(),
            None,
        ))
        .unwrap();

        assert_eq!(tokenizer.token_id_for_raw_token("<|begin_of_text|>"), Some(1));
        assert_eq!(tokenizer.token_id_for_raw_token("<|eot_id|>"), Some(2));
        assert_eq!(tokenizer.token_id_for_raw_token("<missing>"), None);
    }

    #[test]
    fn encode_preserves_raw_special_token_boundaries() {
        let tokenizer = RllmTokenizer::from_metadata(&meta(
            "hf-bpe",
            vec![
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
                "?".to_string(),
                "?<".to_string(),
                "assistant".to_string(),
                "Ċ".to_string(),
            ],
            Vec::new(),
            None,
        ))
        .unwrap();

        assert_eq!(
            tokenizer.encode("?<|im_end|>\n<|im_start|>assistant\n").unwrap(),
            vec![2, 1, 5, 0, 4, 5]
        );
    }

    #[test]
    fn bpe_encode_uses_merges_without_crossing_special_boundaries() {
        let tokenizer = RllmTokenizer::from_metadata(&meta(
            "hf-bpe",
            vec![
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
            vec![
                ("a".to_string(), "s".to_string()),
                ("as".to_string(), "s".to_string()),
                ("i".to_string(), "s".to_string()),
                ("is".to_string(), "t".to_string()),
                ("a".to_string(), "n".to_string()),
                ("an".to_string(), "t".to_string()),
                ("ist".to_string(), "ant".to_string()),
            ],
            None,
        ))
        .unwrap();

        assert_eq!(
            tokenizer.encode("?<|im_end|>\n<|im_start|>assistant\n").unwrap(),
            vec![2, 1, 8, 0, 4, 7, 8]
        );
    }

    fn metaspace_tokenizer() -> RllmTokenizer {
        // Vocab with metaspace word pieces, a non-metaspace leading-word piece,
        // single chars, and byte-fallback tokens.
        let id_to_token = vec![
            "<unk>".to_string(),  // 0
            "<bos>".to_string(),  // 1
            "▁the".to_string(),   // 2
            "▁cat".to_string(),   // 3
            "▁".to_string(),      // 4
            "t".to_string(),      // 5
            "h".to_string(),      // 6
            "e".to_string(),      // 7
            "c".to_string(),      // 8
            "a".to_string(),      // 9
            "the".to_string(),    // 10 (leading word, no metaspace)
            "<0xC3>".to_string(), // 11
            "<0xA9>".to_string(), // 12 (é = C3 A9)
        ];
        let bpe_merges = vec![
            ("t".to_string(), "h".to_string()),   // th
            ("th".to_string(), "e".to_string()),  // the
            ("▁".to_string(), "c".to_string()),   // ▁c
            ("▁c".to_string(), "a".to_string()),  // ▁ca
            ("▁ca".to_string(), "t".to_string()), // ▁cat
        ];
        let metadata = TokenizerMetadata {
            tokenizer_type: Some("hf-bpe".to_string()),
            id_to_token,
            bpe_merges,
            unk_token_id: Some(0),
            bos_token_id: Some(1),
            pre_tokenizer: Some("metaspace".to_string()),
            add_bos_token: Some(true),
            ..Default::default()
        };
        RllmTokenizer::from_metadata(&metadata).unwrap()
    }

    #[test]
    fn metaspace_encode_prepends_bos_and_attaches_space_to_following_word() {
        let tokenizer = metaspace_tokenizer();
        // Faithful SentencePiece (no dummy prefix): "the cat" normalizes to
        // "the▁cat" → <bos> the ▁cat. The leading word has NO metaspace; the
        // space attaches to the *following* word as ▁cat.
        assert_eq!(tokenizer.encode("the cat").unwrap(), vec![1, 10, 3]);
        assert_eq!(tokenizer.decode(&[10, 3]).unwrap(), "the cat");
    }

    #[test]
    fn metaspace_byte_fallback_decomposes_oov_chars_and_decodes_utf8() {
        let tokenizer = metaspace_tokenizer();
        // 'é' (U+00E9) is not a vocab piece → bytes C3 A9 → tokens 11, 12.
        let ids = tokenizer.encode("é").unwrap();
        assert_eq!(ids, vec![1, 11, 12]); // bos + byte fallback
        // Decode reassembles the byte run into UTF-8.
        assert_eq!(tokenizer.decode(&[11, 12]).unwrap(), "é");
    }

    #[test]
    fn byte_fallback_value_parses_only_well_formed_byte_tokens() {
        assert_eq!(byte_fallback_value("<0xC3>"), Some(0xC3));
        assert_eq!(byte_fallback_value("<0x00>"), Some(0x00));
        assert_eq!(byte_fallback_value("<bos>"), None);
        assert_eq!(byte_fallback_value("<0xZZ>"), None);
        assert_eq!(byte_fallback_value("<0x1>"), None);
    }
}
