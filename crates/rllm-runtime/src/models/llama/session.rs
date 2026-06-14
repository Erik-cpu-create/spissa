use crate::models::llama::api::{
    decode_vector_tensor, require_config_usize, require_model_config, validate_llama_shape,
};
use crate::models::llama::generate::{
    streaming_llama_transformer_block, LlamaStreamingBlockConfig,
};
use crate::models::llama::model::{
    LayerDecodedLlamaRamaTransformer, OwnedLlamaStreamingBlockParameters,
};
use crate::rotary::KvCache;
use crate::{
    embedding_lookup, rms_norm, sample_argmax, sample_top_p, LazyRllmModel, MemoryBudget, Result,
    RuntimeError,
};
use crate::{RamaSessionAdapter, RamaSessionStep};

pub struct LlamaRamaSessionAdapter<'a> {
    model: &'a mut LazyRllmModel,
    prepared: LayerDecodedLlamaRamaTransformer,
    hidden_size: usize,
    intermediate_size: usize,
    head_dim: usize,
    vocab_size: usize,
    embedding_data: Vec<f32>,
    layer_norms: Vec<OwnedLlamaStreamingBlockParameters>,
    lm_head_weight_data: Vec<f32>,
    caches: Vec<KvCache>,
}

fn tensor_shape_usize(model: &LazyRllmModel, name: &str) -> Result<Vec<usize>> {
    model
        .tensor(name)?
        .shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                RuntimeError::Shape(format!("tensor {name} dimension {dim} overflows usize"))
            })
        })
        .collect()
}

fn validate_matrix_with_columns(
    model: &LazyRllmModel,
    name: &str,
    expected_cols: usize,
) -> Result<usize> {
    let shape = tensor_shape_usize(model, name)?;
    if shape.len() != 2 {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} must be rank-2 [rows, {expected_cols}], got {:?}",
            shape
        )));
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows == 0 {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} must have non-zero row count, got {:?}",
            shape
        )));
    }
    if cols != expected_cols {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected [rows, {expected_cols}]",
            shape
        )));
    }
    Ok(rows)
}

fn validate_matrix_shape(
    model: &LazyRllmModel,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<()> {
    let shape = tensor_shape_usize(model, name)?;
    if shape != [expected_rows, expected_cols] {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected [{expected_rows}, {expected_cols}]",
            shape
        )));
    }
    Ok(())
}

impl<'a> LlamaRamaSessionAdapter<'a> {
    pub fn new(
        model: &'a mut LazyRllmModel,
        prepared: &LayerDecodedLlamaRamaTransformer,
        budget: &mut MemoryBudget,
    ) -> Result<Self> {
        if prepared.layers.is_empty() {
            return Err(RuntimeError::Shape(
                "llama session requires at least one layer".to_string(),
            ));
        }

        let model_config = require_model_config(model, "llama")?;
        let hidden_size = require_config_usize("hidden_size", model_config.hidden_size)?;
        let intermediate_size =
            require_config_usize("intermediate_size", model_config.intermediate_size)?;
        if intermediate_size == 0 {
            return Err(RuntimeError::Shape(
                "llama session intermediate_size must be non-zero".to_string(),
            ));
        }
        if prepared.final_layernorm_weight.len() != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "llama session final_layernorm_weight len {} does not match hidden_size {hidden_size}",
                prepared.final_layernorm_weight.len()
            )));
        }
        let head_dim = validate_llama_shape(
            hidden_size,
            prepared.config.num_heads,
            prepared.config.num_key_value_heads,
        )?;
        let max_seq_len = prepared.config.max_seq_len.ok_or_else(|| {
            RuntimeError::InvalidTensorData("llama session config requires max_seq_len".to_string())
        })?;

        let vocab_size =
            validate_matrix_with_columns(model, &prepared.embedding_weight, hidden_size)?;
        validate_matrix_shape(model, &prepared.lm_head_weight, vocab_size, hidden_size)?;

        let embedding_data = model
            .decode_tensor(&prepared.embedding_weight, budget)?
            .data;
        let lm_head_weight_data = model.decode_tensor(&prepared.lm_head_weight, budget)?.data;

        let mut layer_norms = Vec::with_capacity(prepared.layers.len());
        for i in 0..prepared.layers.len() {
            layer_norms.push(OwnedLlamaStreamingBlockParameters {
                input_layernorm_weight: decode_vector_tensor(
                    model,
                    &format!("model.layers.{i}.input_layernorm.weight"),
                    hidden_size,
                )?,
                post_attention_layernorm_weight: decode_vector_tensor(
                    model,
                    &format!("model.layers.{i}.post_attention_layernorm.weight"),
                    hidden_size,
                )?,
            });
        }

        let mut caches = Vec::with_capacity(prepared.layers.len());
        for _ in 0..prepared.layers.len() {
            caches.push(KvCache::new(
                prepared.config.num_key_value_heads,
                head_dim,
                max_seq_len,
            )?);
        }

        Ok(Self {
            model,
            prepared: prepared.clone(),
            hidden_size,
            intermediate_size,
            head_dim,
            vocab_size,
            embedding_data,
            layer_norms,
            lm_head_weight_data,
            caches,
        })
    }

    fn append_tokens_inner(
        &mut self,
        tokens: &[usize],
        budget: &mut MemoryBudget,
        emit_logits: bool,
    ) -> Result<Option<RamaSessionStep>> {
        if tokens.is_empty() {
            return Err(RuntimeError::InvalidTensorData(
                "llama session append requires at least one token".to_string(),
            ));
        }
        let seq_len = tokens.len();
        let position_offset = self.context_len();
        let projected_len = position_offset.checked_add(seq_len).ok_or_else(|| {
            RuntimeError::Shape("llama session context length overflow".to_string())
        })?;
        if projected_len > self.max_seq_len() {
            return Err(RuntimeError::Shape(format!(
                "llama session context would reach {projected_len}, max_seq_len {}",
                self.max_seq_len()
            )));
        }

        let mut hidden = embedding_lookup(
            &self.embedding_data,
            self.vocab_size,
            self.hidden_size,
            tokens,
        )?;
        for (i, layer_names) in self.prepared.layers.iter().enumerate() {
            let config = LlamaStreamingBlockConfig {
                seq_len,
                hidden_size: self.hidden_size,
                q_heads: self.prepared.config.num_heads,
                kv_heads: self.prepared.config.num_key_value_heads,
                head_dim: self.head_dim,
                intermediate_size: self.intermediate_size,
                rms_norm_eps: self.prepared.config.rms_norm_eps,
                rope_theta: self.prepared.config.rope_theta,
                causal: self.prepared.config.causal,
                position_offset,
            };
            hidden = streaming_llama_transformer_block(
                self.model,
                &hidden,
                layer_names,
                &self.layer_norms[i],
                config,
                budget,
                Some(&mut self.caches[i]),
            )?;
        }

        if !emit_logits {
            return Ok(None);
        }

        let hidden = rms_norm(
            &hidden,
            &self.prepared.final_layernorm_weight,
            seq_len,
            self.hidden_size,
            self.prepared.config.rms_norm_eps,
        )?;
        let last_hidden = &hidden[(seq_len - 1) * self.hidden_size..];
        let mut logits = vec![0.0f32; self.vocab_size];
        for (v, logit) in logits.iter_mut().enumerate() {
            let row_start = v * self.hidden_size;
            let row = &self.lm_head_weight_data[row_start..row_start + self.hidden_size];
            let mut sum = 0.0f32;
            for (&hidden, &weight) in last_hidden.iter().zip(row.iter()) {
                sum += hidden * weight;
            }
            *logit = sum;
        }
        let token_id = match self.prepared.config.sampling {
            crate::StreamingSamplingConfig::Argmax => sample_argmax(&logits)?,
            crate::StreamingSamplingConfig::TopP {
                temperature,
                top_p,
                seed,
            } => sample_top_p(&logits, temperature, top_p, seed)?,
        };
        Ok(Some(RamaSessionStep {
            token_id,
            logits: Some(logits),
            cached_context_len_after: self.context_len(),
        }))
    }
}

impl RamaSessionAdapter for LlamaRamaSessionAdapter<'_> {
    fn context_len(&self) -> usize {
        self.caches.first().map(KvCache::len).unwrap_or(0)
    }

    fn max_seq_len(&self) -> usize {
        self.prepared.config.max_seq_len.unwrap_or(0)
    }

    fn context_memory_bytes(&self) -> usize {
        self.caches.iter().map(KvCache::resident_bytes).sum()
    }

    fn append_tokens(
        &mut self,
        tokens: &[usize],
        budget: &mut MemoryBudget,
        emit_logits: bool,
    ) -> Result<Option<RamaSessionStep>> {
        let old_lens: Vec<usize> = self.caches.iter().map(KvCache::len).collect();
        match self.append_tokens_inner(tokens, budget, emit_logits) {
            Ok(step) => Ok(step),
            Err(error) => {
                for (cache, len) in self.caches.iter_mut().zip(old_lens) {
                    let _ = cache.truncate(len);
                }
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::llama::model::{LlamaRamaBuildConfig, OwnedLlamaStreamingBlockTensorNames};
    use crate::{RamaSessionAdapter, StreamingSamplingConfig};
    use rllm_container::{DType, GlobalMetadata, ModelConfigMetadata, RllmWriter, TensorMeta};
    use sha2::{Digest, Sha256};

    const VOCAB_SIZE: usize = 3;
    const HIDDEN_SIZE: usize = 2;
    const INTERMEDIATE_SIZE: usize = 3;

    fn sha256_array(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rllm-llama-session-{name}-{}.rllm",
            std::process::id()
        ))
    }

    fn add_f32_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        shape: Vec<u64>,
        values: &[f32],
    ) {
        let bytes = f32_bytes(values);
        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape,
            dtype: DType::Fp32,
            original_size_bytes: bytes.len() as u64,
            compressed_size_bytes: bytes.len() as u64,
            original_sha256: sha256_array(&bytes),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(tensor_id, "rtc-raw-v1", &bytes, &bytes, 0)
            .unwrap();
    }

    fn llama_metadata() -> GlobalMetadata {
        let mut metadata = GlobalMetadata::new_test();
        metadata.model_config = Some(ModelConfigMetadata {
            architecture_type: Some("llama".to_string()),
            hidden_size: Some(HIDDEN_SIZE as u64),
            intermediate_size: Some(INTERMEDIATE_SIZE as u64),
            num_attention_heads: Some(1),
            num_key_value_heads: Some(1),
            max_position_embeddings: Some(8),
            rms_norm_eps: Some(1e-5),
            rope_theta: Some(10_000.0),
            vocab_size: Some(VOCAB_SIZE as u64),
            ..Default::default()
        });
        metadata
    }

    fn layer_names(layer_idx: usize) -> OwnedLlamaStreamingBlockTensorNames {
        OwnedLlamaStreamingBlockTensorNames {
            q_weight: format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
            k_weight: format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
            v_weight: format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
            o_weight: format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
            gate_weight: format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
            up_weight: format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
            down_weight: format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
        }
    }

    fn prepared_with_layers(layer_count: usize) -> LayerDecodedLlamaRamaTransformer {
        LayerDecodedLlamaRamaTransformer {
            config: LlamaRamaBuildConfig {
                max_new_tokens: 1,
                max_seq_len: Some(8),
                num_heads: 1,
                num_key_value_heads: 1,
                causal: true,
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
                sampling: StreamingSamplingConfig::Argmax,
            },
            embedding_weight: "model.embed_tokens.weight".to_string(),
            layers: (0..layer_count).map(layer_names).collect(),
            lm_head_weight: "lm_head.weight".to_string(),
            final_layernorm_weight: vec![1.0, 1.0],
            pinned_lm_head_weight: None,
            resident_parameter_bytes: 0,
            max_layer_parameter_bytes: 0,
        }
    }

    fn add_base_tensors(writer: &mut RllmWriter, tensor_id: &mut u64, lm_head_shape: Vec<u64>) {
        add_f32_tensor(
            writer,
            *tensor_id,
            "model.embed_tokens.weight",
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
            &[0.5, -1.0, 1.25, 0.75, -0.5, 0.25],
        );
        *tensor_id += 1;

        let lm_head_values = vec![0.25; lm_head_shape.iter().product::<u64>() as usize];
        add_f32_tensor(
            writer,
            *tensor_id,
            "lm_head.weight",
            lm_head_shape,
            &lm_head_values,
        );
        *tensor_id += 1;
    }

    fn add_layer_norms(writer: &mut RllmWriter, tensor_id: &mut u64, layer_idx: usize) {
        add_f32_tensor(
            writer,
            *tensor_id,
            &format!("model.layers.{layer_idx}.input_layernorm.weight"),
            vec![HIDDEN_SIZE as u64],
            &[1.0, 1.0],
        );
        *tensor_id += 1;
        add_f32_tensor(
            writer,
            *tensor_id,
            &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
            vec![HIDDEN_SIZE as u64],
            &[1.0, 1.0],
        );
        *tensor_id += 1;
    }

    fn add_complete_layer_zero(writer: &mut RllmWriter, tensor_id: &mut u64) {
        add_layer_norms(writer, tensor_id, 0);
        add_f32_tensor(
            writer,
            *tensor_id,
            "model.layers.0.self_attn.q_proj.weight",
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            &[0.0; HIDDEN_SIZE * HIDDEN_SIZE],
        );
        *tensor_id += 1;
        add_f32_tensor(
            writer,
            *tensor_id,
            "model.layers.0.self_attn.k_proj.weight",
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            &[0.0; HIDDEN_SIZE * HIDDEN_SIZE],
        );
        *tensor_id += 1;
        add_f32_tensor(
            writer,
            *tensor_id,
            "model.layers.0.self_attn.v_proj.weight",
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            &[0.0; HIDDEN_SIZE * HIDDEN_SIZE],
        );
        *tensor_id += 1;
        add_f32_tensor(
            writer,
            *tensor_id,
            "model.layers.0.self_attn.o_proj.weight",
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            &[0.0; HIDDEN_SIZE * HIDDEN_SIZE],
        );
        *tensor_id += 1;
        add_f32_tensor(
            writer,
            *tensor_id,
            "model.layers.0.mlp.gate_proj.weight",
            vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            &[0.0; INTERMEDIATE_SIZE * HIDDEN_SIZE],
        );
        *tensor_id += 1;
        add_f32_tensor(
            writer,
            *tensor_id,
            "model.layers.0.mlp.up_proj.weight",
            vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            &[0.0; INTERMEDIATE_SIZE * HIDDEN_SIZE],
        );
        *tensor_id += 1;
        add_f32_tensor(
            writer,
            *tensor_id,
            "model.layers.0.mlp.down_proj.weight",
            vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
            &[0.0; HIDDEN_SIZE * INTERMEDIATE_SIZE],
        );
        *tensor_id += 1;
    }

    fn write_constructor_model(path: &std::path::Path, lm_head_shape: Vec<u64>) {
        let mut writer = RllmWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(&mut writer, &mut tensor_id, lm_head_shape);
        add_layer_norms(&mut writer, &mut tensor_id, 0);
        writer.finalize().unwrap();
    }

    fn write_post_cache_failure_model(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(
            &mut writer,
            &mut tensor_id,
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_complete_layer_zero(&mut writer, &mut tensor_id);
        add_layer_norms(&mut writer, &mut tensor_id, 1);
        writer.finalize().unwrap();
    }

    #[test]
    fn llama_session_new_rejects_empty_prepared_layers() {
        let path = temp_path("empty-prepared-layers");
        write_constructor_model(&path, vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64]);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(0);
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("at least one layer")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_new_rejects_malformed_lm_head_shape() {
        let path = temp_path("malformed-lm-head");
        write_constructor_model(&path, vec![(VOCAB_SIZE - 1) as u64, HIDDEN_SIZE as u64]);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("lm_head.weight")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_new_rejects_malformed_final_layernorm_shape() {
        let path = temp_path("malformed-final-layernorm");
        write_constructor_model(&path, vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64]);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut prepared = prepared_with_layers(1);
        prepared.final_layernorm_weight = vec![1.0];
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("final_layernorm_weight")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_append_rolls_back_all_caches_after_post_cache_layer_failure() {
        let path = temp_path("rollback-post-cache-failure");
        write_post_cache_failure_model(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let prepared = prepared_with_layers(2);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();

        let result = adapter.append_tokens(&[0], &mut budget, false);

        assert!(result.is_err());
        assert_eq!(adapter.context_len(), 0);
        assert_eq!(adapter.context_memory_bytes(), 0);
        std::fs::remove_file(path).ok();
    }
}
