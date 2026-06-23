// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use anyhow::{anyhow, Result};
use spissa_runtime::SpissaTokenizer;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateKind {
    Raw,
    Llama3,
    ChatMl,
    Phi,
}

impl FromStr for ChatTemplateKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "raw" | "none" => Ok(Self::Raw),
            "llama3" | "llama-3" | "llama3-instruct" | "llama-3-instruct" => Ok(Self::Llama3),
            "chatml" | "chat-ml" | "smollm" | "smollm2" => Ok(Self::ChatMl),
            "phi" | "phi3" | "phi4" => Ok(Self::Phi),
            other => Err(anyhow!(
                "unknown chat template {other:?}; expected raw, llama3, chatml, or phi"
            )),
        }
    }
}

pub fn render_interactive_user_turn(
    kind: ChatTemplateKind,
    has_context: bool,
    previous_assistant_ended: bool,
    system_prompt: Option<&str>,
    user_text: &str,
) -> String {
    match kind {
        ChatTemplateKind::Raw => {
            if has_context {
                format!("\n{user_text}")
            } else {
                user_text.to_string()
            }
        }
        ChatTemplateKind::Llama3 => render_llama3_user_turn(
            has_context,
            previous_assistant_ended,
            system_prompt,
            user_text,
        ),
        ChatTemplateKind::ChatMl => render_chatml_user_turn(
            has_context,
            previous_assistant_ended,
            system_prompt,
            user_text,
        ),
        ChatTemplateKind::Phi => render_phi_user_turn(
            has_context,
            previous_assistant_ended,
            system_prompt,
            user_text,
        ),
    }
}

pub fn stop_token_ids(
    kind: ChatTemplateKind,
    tokenizer: &SpissaTokenizer,
    metadata_eos_token_id: Option<u64>,
) -> Vec<usize> {
    let mut ids = Vec::new();
    if let Some(id) = metadata_eos_token_id.and_then(|id| usize::try_from(id).ok()) {
        ids.push(id);
    }
    if kind == ChatTemplateKind::Llama3 {
        push_unique_token_id(&mut ids, tokenizer.token_id_for_raw_token("<|eot_id|>"));
        push_unique_token_id(
            &mut ids,
            tokenizer.token_id_for_raw_token("<|end_of_text|>"),
        );
    } else if kind == ChatTemplateKind::ChatMl {
        push_unique_token_id(&mut ids, tokenizer.token_id_for_raw_token("<|im_end|>"));
        push_unique_token_id(&mut ids, tokenizer.token_id_for_raw_token("<|endoftext|>"));
    } else if kind == ChatTemplateKind::Phi {
        push_unique_token_id(&mut ids, tokenizer.token_id_for_raw_token("<|end|>"));
        push_unique_token_id(&mut ids, tokenizer.token_id_for_raw_token("<|endoftext|>"));
    }
    ids
}

/// Phi-3 / Phi-4 chat template: `<|system|>…<|end|><|user|>…<|end|><|assistant|>`.
fn render_phi_user_turn(
    has_context: bool,
    previous_assistant_ended: bool,
    system_prompt: Option<&str>,
    user_text: &str,
) -> String {
    let mut rendered = String::new();
    if !has_context {
        if let Some(sys) = system_prompt.map(str::trim).filter(|t| !t.is_empty()) {
            rendered.push_str("<|system|>\n");
            rendered.push_str(sys);
            rendered.push_str("<|end|>\n");
        }
    } else if !previous_assistant_ended {
        rendered.push_str("<|end|>\n");
    }
    rendered.push_str("<|user|>\n");
    rendered.push_str(user_text.trim());
    rendered.push_str("<|end|>\n");
    rendered.push_str("<|assistant|>\n");
    rendered
}

fn render_llama3_user_turn(
    has_context: bool,
    previous_assistant_ended: bool,
    system_prompt: Option<&str>,
    user_text: &str,
) -> String {
    let mut rendered = String::new();
    if !has_context {
        rendered.push_str("<|begin_of_text|>");
        rendered.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        rendered.push_str("Cutting Knowledge Date: December 2023\n");
        rendered.push_str("Today Date: 26 Jul 2024\n\n");
        if let Some(system_prompt) = system_prompt.map(str::trim).filter(|text| !text.is_empty()) {
            rendered.push_str(system_prompt);
        }
        rendered.push_str("<|eot_id|>");
    } else if !previous_assistant_ended {
        rendered.push_str("<|eot_id|>");
    }

    rendered.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
    rendered.push_str(user_text.trim());
    rendered.push_str("<|eot_id|>");
    rendered.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    rendered
}

fn render_chatml_user_turn(
    has_context: bool,
    previous_assistant_ended: bool,
    system_prompt: Option<&str>,
    user_text: &str,
) -> String {
    let mut rendered = String::new();
    if !has_context {
        rendered.push_str("<|im_start|>system\n");
        let system_prompt = system_prompt
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or("You are a helpful AI assistant named SmolLM, trained by Hugging Face");
        rendered.push_str(system_prompt);
        rendered.push_str("<|im_end|>\n");
    } else if !previous_assistant_ended {
        rendered.push_str("<|im_end|>\n");
    }

    rendered.push_str("<|im_start|>user\n");
    rendered.push_str(user_text.trim());
    rendered.push_str("<|im_end|>\n");
    rendered.push_str("<|im_start|>assistant\n");
    rendered
}

fn push_unique_token_id(ids: &mut Vec<usize>, candidate: Option<usize>) {
    let Some(candidate) = candidate else {
        return;
    };
    if !ids.contains(&candidate) {
        ids.push(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spissa_container::TokenizerMetadata;
    use spissa_runtime::SpissaTokenizer;

    #[test]
    fn raw_template_preserves_existing_turn_separator_behavior() {
        assert_eq!(
            render_interactive_user_turn(ChatTemplateKind::Raw, false, true, None, "good morning"),
            "good morning"
        );
        assert_eq!(
            render_interactive_user_turn(ChatTemplateKind::Raw, true, true, None, "halo"),
            "\nhalo"
        );
    }

    #[test]
    fn llama3_template_renders_first_turn_with_generation_prompt() {
        let rendered = render_interactive_user_turn(
            ChatTemplateKind::Llama3,
            false,
            true,
            Some("You are concise."),
            "good morning",
        );

        assert!(
            rendered.starts_with("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
        );
        assert!(rendered.contains("Cutting Knowledge Date: December 2023\n"));
        assert!(rendered.contains("Today Date: 26 Jul 2024\n\n"));
        assert!(rendered.contains("You are concise.<|eot_id|>"));
        assert!(
            rendered.contains("<|start_header_id|>user<|end_header_id|>\n\ngood morning<|eot_id|>")
        );
        assert!(rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn llama3_template_forces_eot_before_next_user_when_generation_hit_limit() {
        let rendered =
            render_interactive_user_turn(ChatTemplateKind::Llama3, true, false, None, "next");

        assert_eq!(
            rendered,
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nnext<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn llama3_template_uses_raw_special_token_stop_fallbacks() {
        let tokenizer = SpissaTokenizer::from_metadata(&TokenizerMetadata {
            tokenizer_type: Some("hf-bpe".to_string()),
            id_to_token: vec![
                "x".to_string(),
                "<|end_of_text|>".to_string(),
                "<|eot_id|>".to_string(),
            ],
            bpe_merges: Vec::new(),
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(
            stop_token_ids(ChatTemplateKind::Llama3, &tokenizer, None),
            vec![2, 1]
        );
        assert_eq!(
            stop_token_ids(ChatTemplateKind::Raw, &tokenizer, Some(1)),
            vec![1]
        );
    }

    #[test]
    fn chatml_template_renders_default_system_and_generation_prompt() {
        let rendered = render_interactive_user_turn(
            ChatTemplateKind::ChatMl,
            false,
            true,
            None,
            "what is 2+2?",
        );

        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nwhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_template_forces_im_end_before_next_user_when_generation_hit_limit() {
        let rendered =
            render_interactive_user_turn(ChatTemplateKind::ChatMl, true, false, None, "next");

        assert_eq!(
            rendered,
            "<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_template_uses_im_end_stop_fallback() {
        let tokenizer = SpissaTokenizer::from_metadata(&TokenizerMetadata {
            tokenizer_type: Some("hf-bpe".to_string()),
            id_to_token: vec![
                "<|endoftext|>".to_string(),
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
            ],
            bpe_merges: Vec::new(),
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(
            stop_token_ids(ChatTemplateKind::ChatMl, &tokenizer, None),
            vec![2, 0]
        );
    }
}
