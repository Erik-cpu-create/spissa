use crate::safetensors::{Result, SafetensorsError};
use rllm_container::TokenizerMetadata;
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
    let tokenizer_type = Some(format!("hf-{}", model_type.to_ascii_lowercase()));
    let unk_token_id = special_token_id(
        model.get("unk_token").or_else(|| value.get("unk_token")),
        &id_to_token,
    )?;
    let bos_token_id = special_token_id(value.get("bos_token"), &id_to_token)?;
    let eos_token_id = special_token_id(value.get("eos_token"), &id_to_token)?;

    Ok(TokenizerMetadata {
        tokenizer_type,
        id_to_token,
        unk_token_id,
        bos_token_id,
        eos_token_id,
    })
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
