import sys

def main():
    with open("crates/rllm-runtime/src/gpt_neox.rs", "r") as f:
        content = f.read()
    
    # We will build a simplified Llama api that mirrors the structure.
    # To save tokens, I'll write a Python script that outputs Rust code.

    code = """use crate::llama::model::*;
use crate::llama::generate::{streaming_llama_transformer_block, LlamaStreamingBlockConfig};
use crate::rotary::KvCache;
use crate::{
    embedding_lookup, rms_norm, sample_argmax, sample_top_p, streaming_tile_linear_from_model,
    LazyRllmModel, MemoryBudget, Result, RuntimeError, StreamingTileLinearConfig,
};
use rllm_container::GlobalMetadata;
use std::time::Instant;

pub fn prepare_llama_rama_layer_decode_transformer_from_metadata(
    model: &mut LazyRllmModel,
    generation: LlamaRamaGenerationConfig,
) -> Result<LayerDecodedLlamaRamaTransformer> {
    let metadata = model.metadata();
    let model_config = metadata.model_config.as_ref().unwrap();
    let num_heads = model_config.num_attention_heads.unwrap() as usize;
    let num_key_value_heads = model_config.num_key_value_heads.unwrap_or(num_heads as u64) as usize;
    let hidden_size = model_config.hidden_size.unwrap() as usize;
    let intermediate_size = model_config.intermediate_size.unwrap() as usize;
    let head_dim = hidden_size / num_heads;
    let max_seq_len = generation.max_seq_len.unwrap_or(2048);

    let build = LlamaEchoBuildConfig {
        max_new_tokens: generation.max_new_tokens,
        max_seq_len: Some(max_seq_len),
        num_heads,
        num_key_value_heads,
        causal: generation.causal,
        rms_norm_eps: model_config.rms_norm_eps.unwrap_or(1e-5),
        rope_theta: model_config.rope_theta.unwrap_or(10000.0),
        sampling: generation.sampling,
    };

    let embedding_weight = "model.embed_tokens.weight".to_string();
    let lm_head_weight = "lm_head.weight".to_string();
    
    // final layer norm
    let final_layernorm_weight = model.load_layer_tensor("model.norm.weight")?.to_f32_vec()?;
    
    // Count layers
    let mut num_layers = 0;
    while model.has_tensor(&format!("model.layers.{num_layers}.input_layernorm.weight")) {
        num_layers += 1;
    }

    let mut layers = Vec::new();
    for i in 0..num_layers {
        layers.push(OwnedLlamaStreamingBlockTensorNames {
            q_weight: format!("model.layers.{i}.self_attn.q_proj.weight"),
            k_weight: format!("model.layers.{i}.self_attn.k_proj.weight"),
            v_weight: format!("model.layers.{i}.self_attn.v_proj.weight"),
            o_weight: format!("model.layers.{i}.self_attn.o_proj.weight"),
            gate_weight: format!("model.layers.{i}.mlp.gate_proj.weight"),
            up_weight: format!("model.layers.{i}.mlp.up_proj.weight"),
            down_weight: format!("model.layers.{i}.mlp.down_proj.weight"),
        });
    }

    Ok(LayerDecodedLlamaRamaTransformer {
        config: build,
        embedding_weight,
        layers,
        lm_head_weight,
        final_layernorm_weight,
        pinned_lm_head_weight: None,
        resident_parameter_bytes: 0,
        max_layer_parameter_bytes: 0,
    })
}

pub fn rama_layer_decoded_llama_transformer_generate_from_model(
    model: &mut LazyRllmModel,
    prepared: &LayerDecodedLlamaRamaTransformer,
    prompt_token_ids: &[usize],
    budget: &mut MemoryBudget,
    options: LlamaRamaGenerationOptions,
) -> Result<LlamaTextGenerationResult> {
    let mut token_ids = prompt_token_ids.to_vec();
    let mut generated_token_ids = Vec::new();

    let hidden_size = model.metadata().model_config.as_ref().unwrap().hidden_size.unwrap() as usize;
    let intermediate_size = model.metadata().model_config.as_ref().unwrap().intermediate_size.unwrap() as usize;
    let head_dim = hidden_size / prepared.config.num_heads;
    
    let mut caches = Vec::new();
    for _ in 0..prepared.layers.len() {
        caches.push(KvCache::new(prepared.config.num_key_value_heads, head_dim, prepared.config.max_seq_len.unwrap())?);
    }

    for step in 0..prepared.config.max_new_tokens {
        let current_tokens = if step == 0 { prompt_token_ids } else { &generated_token_ids[generated_token_ids.len() - 1..] };
        let seq_len = current_tokens.len();
        
        let position_offset = token_ids.len() - seq_len;
        
        // 1. Embedding
        let embedding_data = model.load_layer_tensor(&prepared.embedding_weight)?.to_f32_vec()?;
        let vocab_size = embedding_data.len() / hidden_size;
        let mut hidden = embedding_lookup(&embedding_data, vocab_size, hidden_size, current_tokens)?;
        
        // 2. Layers
        for (i, layer_names) in prepared.layers.iter().enumerate() {
            let input_layernorm_weight = model.load_layer_tensor(&format!("model.layers.{i}.input_layernorm.weight"))?.to_f32_vec()?;
            let post_attention_layernorm_weight = model.load_layer_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight"))?.to_f32_vec()?;
            let params = OwnedLlamaStreamingBlockParameters {
                input_layernorm_weight,
                post_attention_layernorm_weight,
            };
            let config = LlamaStreamingBlockConfig {
                seq_len,
                hidden_size,
                q_heads: prepared.config.num_heads,
                kv_heads: prepared.config.num_key_value_heads,
                head_dim,
                intermediate_size,
                rms_norm_eps: prepared.config.rms_norm_eps,
                rope_theta: prepared.config.rope_theta,
                causal: prepared.config.causal,
                position_offset,
            };
            hidden = streaming_llama_transformer_block(model, &hidden, layer_names, &params, config, budget, Some(&mut caches[i]))?;
        }

        // 3. Final norm
        hidden = rms_norm(&hidden, &prepared.final_layernorm_weight, seq_len, hidden_size, prepared.config.rms_norm_eps)?;

        // 4. LM Head (only on last token for generation)
        let last_hidden = &hidden[(seq_len - 1) * hidden_size..];
        let lm_config = StreamingTileLinearConfig {
            batch: 1,
            in_features: hidden_size,
            out_features: vocab_size,
        };
        let logits = streaming_tile_linear_from_model(model, last_hidden, &prepared.lm_head_weight, lm_config, budget)?;

        // 5. Sample
        let next_token = match prepared.config.sampling {
            crate::StreamingSamplingConfig::Argmax => sample_argmax(&logits)?,
            crate::StreamingSamplingConfig::TopP { temperature, top_p, seed } => sample_top_p(&logits, temperature, top_p, seed)?,
        };

        token_ids.push(next_token);
        generated_token_ids.push(next_token);
        println!("Generated token {}", next_token);
    }

    Ok(LlamaTextGenerationResult {
        prompt_token_ids: prompt_token_ids.to_vec(),
        generated_token_ids,
        token_ids,
        text: String::new(),
        generated_text: String::new(),
        context_echo_bytes: 0,
    })
}
"""
    with open("crates/rllm-runtime/src/llama/api.rs", "w") as f:
        f.write(code)

if __name__ == "__main__":
    main()
