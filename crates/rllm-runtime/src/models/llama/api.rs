use crate::models::llama::generate::{
    streaming_llama_transformer_block, LlamaStreamingBlockConfig,
};
use crate::models::llama::model::*;
use crate::rotary::KvCache;
use crate::{
    ops::{embedding_lookup, rms_norm, sample_argmax, sample_top_p},
    LazyRllmModel, MemoryBudget, Result, RuntimeError,
};
use rllm_container::ModelConfigMetadata;

pub(crate) fn require_model_config<'a>(
    model: &'a LazyRllmModel,
    architecture: &str,
) -> Result<&'a ModelConfigMetadata> {
    model.metadata().model_config.as_ref().ok_or_else(|| {
        RuntimeError::InvalidTensorData(format!(
            "{architecture} generation requires persisted model_config metadata; repack with --config <config.json>"
        ))
    })
}

pub(crate) fn require_config_usize(field_name: &str, value: Option<u64>) -> Result<usize> {
    let value = value.ok_or_else(|| {
        RuntimeError::InvalidTensorData(format!(
            "llama model_config is missing required field {field_name}"
        ))
    })?;
    usize::try_from(value).map_err(|_| {
        RuntimeError::Shape(format!(
            "llama model_config field {field_name}={value} overflows usize"
        ))
    })
}

pub(crate) fn validate_llama_shape(
    hidden_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
) -> Result<usize> {
    if num_heads == 0 || num_key_value_heads == 0 {
        return Err(RuntimeError::Shape(format!(
            "llama attention heads must be non-zero, got num_heads={num_heads}, num_key_value_heads={num_key_value_heads}"
        )));
    }
    if hidden_size == 0 {
        return Err(RuntimeError::Shape(
            "llama hidden_size must be non-zero".to_string(),
        ));
    }
    if !hidden_size.is_multiple_of(num_heads) {
        return Err(RuntimeError::Shape(format!(
            "llama hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )));
    }
    if !num_heads.is_multiple_of(num_key_value_heads) {
        return Err(RuntimeError::Shape(format!(
            "llama num_heads {num_heads} must be a multiple of num_key_value_heads {num_key_value_heads}"
        )));
    }
    Ok(hidden_size / num_heads)
}

pub(crate) fn decode_vector_tensor(
    model: &mut LazyRllmModel,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>> {
    let mut budget = MemoryBudget::unbounded();
    let tensor = model.decode_tensor(name, &mut budget)?;
    if tensor.shape != [expected_len] {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected [{expected_len}]",
            tensor.shape
        )));
    }
    Ok(tensor.data)
}

pub fn prepare_llama_rama_layer_decode_transformer_from_metadata(
    model: &mut LazyRllmModel,
    generation: LlamaRamaGenerationConfig,
) -> Result<LayerDecodedLlamaRamaTransformer> {
    let model_config = require_model_config(model, "llama")?;
    let num_heads = require_config_usize("num_attention_heads", model_config.num_attention_heads)?;
    let num_key_value_heads = model_config
        .num_key_value_heads
        .map(|value| require_config_usize("num_key_value_heads", Some(value)))
        .transpose()?
        .unwrap_or(num_heads);
    let hidden_size = require_config_usize("hidden_size", model_config.hidden_size)?;
    validate_llama_shape(hidden_size, num_heads, num_key_value_heads)?;
    let max_seq_len = generation
        .max_seq_len
        .or_else(|| {
            model_config
                .max_position_embeddings
                .and_then(|value| usize::try_from(value).ok())
        })
        .unwrap_or(2048);

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
    let lm_head_weight = if model.tensor("lm_head.weight").is_ok() {
        "lm_head.weight".to_string()
    } else {
        "model.embed_tokens.weight".to_string()
    };

    let final_layernorm_weight = decode_vector_tensor(model, "model.norm.weight", hidden_size)?;

    let mut num_layers = 0;
    while model
        .tensor(&format!("model.layers.{num_layers}.input_layernorm.weight"))
        .is_ok()
    {
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
    _options: LlamaRamaGenerationOptions,
    on_token: &mut dyn FnMut(usize) -> bool,
) -> Result<LlamaTextGenerationResult> {
    let mut token_ids = prompt_token_ids.to_vec();
    let mut generated_token_ids = Vec::new();

    let model_config = require_model_config(model, "llama")?;
    let hidden_size = require_config_usize("hidden_size", model_config.hidden_size)?;
    let intermediate_size =
        require_config_usize("intermediate_size", model_config.intermediate_size)?;
    let head_dim = validate_llama_shape(
        hidden_size,
        prepared.config.num_heads,
        prepared.config.num_key_value_heads,
    )?;
    let max_seq_len = prepared.config.max_seq_len.ok_or_else(|| {
        RuntimeError::InvalidTensorData("llama generation config requires max_seq_len".to_string())
    })?;

    let mut caches = Vec::new();
    for _ in 0..prepared.layers.len() {
        caches.push(KvCache::new(
            prepared.config.num_key_value_heads,
            head_dim,
            max_seq_len,
        )?);
    }

    // F32 HEAD PINNING: Decode the embedding, layernorms, and LM Head weights ONCE outside the hot generation loop
    let embedding_data = model
        .decode_tensor(&prepared.embedding_weight, budget)?
        .data;
    let vocab_size = embedding_data.len() / hidden_size;

    let mut layer_norms = Vec::new();
    for i in 0..prepared.layers.len() {
        let input_layernorm_weight = decode_vector_tensor(
            model,
            &format!("model.layers.{i}.input_layernorm.weight"),
            hidden_size,
        )?;
        let post_attention_layernorm_weight = decode_vector_tensor(
            model,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
            hidden_size,
        )?;
        layer_norms.push((input_layernorm_weight, post_attention_layernorm_weight));
    }

    // Decode LM head weight ONCE
    let lm_head_weight_data = model.decode_tensor(&prepared.lm_head_weight, budget)?.data;

    let mut final_logits = None;

    for step in 0..prepared.config.max_new_tokens {
        let current_tokens = if step == 0 {
            prompt_token_ids
        } else {
            &generated_token_ids[generated_token_ids.len() - 1..]
        };
        let seq_len = current_tokens.len();

        let position_offset = token_ids.len() - seq_len;

        // 1. Embedding
        let mut hidden =
            embedding_lookup(&embedding_data, vocab_size, hidden_size, current_tokens)?;

        // 2. Layers
        for (i, layer_names) in prepared.layers.iter().enumerate() {
            let (input_ln_weight, post_attn_ln_weight) = &layer_norms[i];
            let params = OwnedLlamaStreamingBlockParameters {
                input_layernorm_weight: input_ln_weight.clone(),
                post_attention_layernorm_weight: post_attn_ln_weight.clone(),
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
            hidden = streaming_llama_transformer_block(
                model,
                &hidden,
                layer_names,
                &params,
                config,
                budget,
                Some(&mut caches[i]),
            )?;
        }

        // 3. Final norm
        hidden = rms_norm(
            &hidden,
            &prepared.final_layernorm_weight,
            seq_len,
            hidden_size,
            prepared.config.rms_norm_eps,
        )?;

        // 4. LM Head (only on last token for generation)
        let last_hidden = &hidden[(seq_len - 1) * hidden_size..];

        // Instead of streaming from model, we compute the linear locally since we already pinned the LM head data
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0;
            for h in 0..hidden_size {
                sum += last_hidden[h] * lm_head_weight_data[v * hidden_size + h];
            }
            logits[v] = sum;
        }

        // 5. Sample
        let next_token = match prepared.config.sampling {
            crate::StreamingSamplingConfig::Argmax => sample_argmax(&logits)?,
            crate::StreamingSamplingConfig::TopP {
                temperature,
                top_p,
                seed,
            } => sample_top_p(&logits, temperature, top_p, seed)?,
        };

        token_ids.push(next_token);
        generated_token_ids.push(next_token);
        if step == prepared.config.max_new_tokens - 1 {
            final_logits = Some(logits);
        }

        if !on_token(next_token) {
            break;
        }
    }

    let context_echo_bytes = caches
        .iter()
        .map(|cache| {
            cache
                .len()
                .saturating_mul(cache.num_heads())
                .saturating_mul(cache.head_dim())
                .saturating_mul(2)
                .saturating_mul(std::mem::size_of::<f32>())
        })
        .sum();

    Ok(LlamaTextGenerationResult {
        prompt_token_ids: prompt_token_ids.to_vec(),
        generated_token_ids,
        token_ids,
        text: String::new(),
        generated_text: String::new(),
        context_echo_bytes,
        logits: if _options.collect_logits {
            final_logits
        } else {
            None
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rllm_container::{GlobalMetadata, ModelConfigMetadata, RllmWriter};

    fn write_empty_model(metadata: GlobalMetadata, name: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("rllm-llama-api-{name}-{}.rllm", std::process::id()));
        let writer = RllmWriter::new(&path, metadata).unwrap();
        writer.finalize().unwrap();
        path
    }

    fn generation_config() -> LlamaRamaGenerationConfig {
        LlamaRamaGenerationConfig {
            max_new_tokens: 1,
            max_seq_len: Some(8),
            causal: true,
            sampling: crate::StreamingSamplingConfig::Argmax,
        }
    }

    #[test]
    fn llama_prepare_rejects_missing_model_config_without_panic() {
        let path = write_empty_model(GlobalMetadata::new_test(), "missing-config");
        let mut model = LazyRllmModel::open(&path).unwrap();

        let result = prepare_llama_rama_layer_decode_transformer_from_metadata(
            &mut model,
            generation_config(),
        );

        assert!(result.is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_prepare_rejects_missing_required_config_fields_without_panic() {
        let mut metadata = GlobalMetadata::new_test();
        metadata.model_config = Some(ModelConfigMetadata {
            architecture_type: Some("llama".to_string()),
            ..Default::default()
        });
        let path = write_empty_model(metadata, "missing-fields");
        let mut model = LazyRllmModel::open(&path).unwrap();

        let result = prepare_llama_rama_layer_decode_transformer_from_metadata(
            &mut model,
            generation_config(),
        );

        assert!(result.is_err());
        std::fs::remove_file(path).ok();
    }
}
