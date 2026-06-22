// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use crate::safetensors::{Result, SafetensorsError};
use spissa_container::TokenizerMetadata;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub fn read_tokenizer_metadata(path: impl AsRef<Path>) -> Result<TokenizerMetadata> {
    let json = fs::read_to_string(path)?;
    tokenizer_metadata_from_json_str(&json)
}

pub fn tokenizer_metadata_from_json_str(json: &str) -> Result<TokenizerMetadata> {
    let value: Value = serde_json::from_str(json)?;
    let model = value
        .get("model")
        .ok_or_else(|| SafetensorsError::InvalidTokenizer("missing tokenizer model".to_string()))?;
    let model_type = model
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| SafetensorsError::InvalidTokenizer("missing model.vocab".to_string()))?;

    let mut by_id = BTreeMap::new();
    for (token, id_value) in vocab {
        let id = id_value.as_u64().ok_or_else(|| {
            SafetensorsError::InvalidTokenizer(format!("token {token:?} has non-u64 id"))
        })?;
        by_id.insert(id, token.clone());
    }

    if let Some(added_tokens) = value.get("added_tokens").and_then(Value::as_array) {
        for token in added_tokens {
            if let (Some(id), Some(content)) = (
                token.get("id").and_then(Value::as_u64),
                token.get("content").and_then(Value::as_str),
            ) {
                by_id.entry(id).or_insert_with(|| content.to_string());
            }
        }
    }

    let id_to_token = contiguous_id_to_token(by_id)?;
    let bpe_merges = bpe_merges_from_model(model)?;
    let tokenizer_type = Some(format!("hf-{}", model_type.to_ascii_lowercase()));
    let unk_token_id = special_token_id(
        model.get("unk_token").or_else(|| value.get("unk_token")),
        &id_to_token,
    )?;
    let eos_token_id = special_token_id(value.get("eos_token"), &id_to_token)?;
    let pre_tokenizer = Some(detect_pre_tokenizer(&value).to_string());
    // A `TemplateProcessing` post-processor that opens with a SpecialToken means
    // that token is prepended on every encode (Gemma prepends `<bos>`). Use it as
    // both the add-BOS signal and a fallback source for `bos_token_id` (Gemma's
    // tokenizer.json has no top-level `bos_token` field).
    let template_bos = template_prepended_special_id(&value, &id_to_token);
    let bos_token_id = special_token_id(value.get("bos_token"), &id_to_token)?.or(template_bos);
    let add_bos_token = Some(template_bos.is_some());

    Ok(TokenizerMetadata {
        tokenizer_type,
        id_to_token,
        bpe_merges,
        unk_token_id,
        bos_token_id,
        eos_token_id,
        pre_tokenizer,
        add_bos_token,
    })
}

/// Classify the pre-tokenization scheme from the tokenizer.json normalizer /
/// pre_tokenizer. SentencePiece tokenizers (Gemma) replace spaces with the `▁`
/// metaspace marker (a `Replace " " → "▁"` normalizer or a `Metaspace`
/// pre_tokenizer); GPT-2-style tokenizers use a `ByteLevel` pre_tokenizer.
fn detect_pre_tokenizer(value: &Value) -> &'static str {
    const METASPACE: &str = "▁";
    let mentions_metaspace = |node: Option<&Value>| -> bool {
        node.map(|n| {
            let text = n.to_string();
            text.contains("Metaspace") || text.contains(METASPACE)
        })
        .unwrap_or(false)
    };
    if mentions_metaspace(value.get("normalizer")) || mentions_metaspace(value.get("pre_tokenizer"))
    {
        return "metaspace";
    }
    "byte_level"
}

/// If the tokenizer's `TemplateProcessing` post-processor opens a single
/// sequence with a `SpecialToken`, return that token's vocab id (it is prepended
/// on every encode — Gemma prepends `<bos>`). Returns `None` otherwise.
fn template_prepended_special_id(value: &Value, id_to_token: &[String]) -> Option<u64> {
    let surface = value
        .get("post_processor")
        .filter(|pp| pp.get("type").and_then(Value::as_str) == Some("TemplateProcessing"))
        .and_then(|pp| pp.get("single"))
        .and_then(Value::as_array)
        .and_then(|seq| seq.first())
        .and_then(|first| first.get("SpecialToken"))
        .and_then(|tok| tok.get("id"))
        .and_then(Value::as_str)?;
    id_to_token
        .iter()
        .position(|candidate| candidate == surface)
        .map(|id| id as u64)
}

fn bpe_merges_from_model(model: &Value) -> Result<Vec<(String, String)>> {
    let Some(merges) = model.get("merges").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut parsed = Vec::with_capacity(merges.len());
    for merge in merges {
        match merge {
            Value::String(raw) => {
                let mut parts = raw.splitn(2, ' ');
                let left = parts.next().unwrap_or_default();
                let right = parts.next().ok_or_else(|| {
                    SafetensorsError::InvalidTokenizer(format!(
                        "BPE merge {raw:?} must contain two tokens"
                    ))
                })?;
                parsed.push((left.to_string(), right.to_string()));
            }
            Value::Array(parts) if parts.len() == 2 => {
                let left = parts[0].as_str().ok_or_else(|| {
                    SafetensorsError::InvalidTokenizer(
                        "BPE merge array left value must be a string".to_string(),
                    )
                })?;
                let right = parts[1].as_str().ok_or_else(|| {
                    SafetensorsError::InvalidTokenizer(
                        "BPE merge array right value must be a string".to_string(),
                    )
                })?;
                parsed.push((left.to_string(), right.to_string()));
            }
            other => {
                return Err(SafetensorsError::InvalidTokenizer(format!(
                    "unsupported BPE merge entry {other:?}"
                )));
            }
        }
    }
    Ok(parsed)
}

fn contiguous_id_to_token(by_id: BTreeMap<u64, String>) -> Result<Vec<String>> {
    if by_id.is_empty() {
        return Err(SafetensorsError::InvalidTokenizer(
            "tokenizer vocab is empty".to_string(),
        ));
    }
    let max_id = *by_id.keys().next_back().expect("non-empty map has max key");
    let mut id_to_token = Vec::with_capacity(max_id as usize + 1);
    for id in 0..=max_id {
        let token = by_id.get(&id).ok_or_else(|| {
            SafetensorsError::InvalidTokenizer(format!(
                "tokenizer vocab is not contiguous; missing id {id}"
            ))
        })?;
        id_to_token.push(token.clone());
    }
    Ok(id_to_token)
}

fn special_token_id(value: Option<&Value>, id_to_token: &[String]) -> Result<Option<u64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let token = match value {
        Value::String(token) => Some(token.as_str()),
        Value::Object(object) => object.get("content").and_then(Value::as_str),
        _ => None,
    };
    let Some(token) = token else {
        return Ok(None);
    };
    id_to_token
        .iter()
        .position(|candidate| candidate == token)
        .map(|id| id as u64)
        .map(Some)
        .ok_or_else(|| {
            SafetensorsError::InvalidTokenizer(format!(
                "special token {token:?} is not present in tokenizer vocab"
            ))
        })
}
