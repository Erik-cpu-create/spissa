// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

    use super::*;
    use crate::models::llama::model::{LlamaRamaBuildConfig, OwnedLlamaStreamingBlockTensorNames};
    use crate::{RamaSessionAdapter, StreamingSamplingConfig};
    use spissa_container::{DType, GlobalMetadata, ModelConfigMetadata, SpissaWriter, TensorMeta};
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

    fn bf16_bytes(values: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rllm-llama-session-{name}-{}.spsa",
            std::process::id()
        ))
    }

    fn add_f32_tensor(
        writer: &mut SpissaWriter,
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

    fn add_bf16_tensor(
        writer: &mut SpissaWriter,
        tensor_id: u64,
        name: &str,
        shape: Vec<u64>,
        values: &[u16],
    ) {
        let bytes = bf16_bytes(values);
        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape,
            dtype: DType::Bf16,
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

    fn llama_metadata_with_vocab(vocab_size: usize) -> GlobalMetadata {
        let mut metadata = llama_metadata();
        metadata.model_config.as_mut().unwrap().vocab_size = Some(vocab_size as u64);
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
                rotary_dim: 2,
                sampling: StreamingSamplingConfig::Argmax,
            },
            embedding_weight: "model.embed_tokens.weight".to_string(),
            layers: (0..layer_count).map(layer_names).collect(),
            lm_head_weight: "lm_head.weight".to_string(),
            rope_freq_scale: None,
            final_layernorm_weight: vec![1.0, 1.0],
            pinned_lm_head_weight: None,
            resident_parameter_bytes: 0,
            max_layer_parameter_bytes: 0,
        }
    }

    fn add_base_tensors(writer: &mut SpissaWriter, tensor_id: &mut u64, lm_head_shape: Vec<u64>) {
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

    fn add_layer_norms(writer: &mut SpissaWriter, tensor_id: &mut u64, layer_idx: usize) {
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

    fn zero_values(shape: &[u64]) -> Vec<f32> {
        vec![0.0; shape.iter().product::<u64>() as usize]
    }

    fn add_zero_f32_tensor(
        writer: &mut SpissaWriter,
        tensor_id: &mut u64,
        name: &str,
        shape: Vec<u64>,
    ) {
        let values = zero_values(&shape);
        add_f32_tensor(writer, *tensor_id, name, shape, &values);
        *tensor_id += 1;
    }

    fn add_layer_projection_tensors(
        writer: &mut SpissaWriter,
        tensor_id: &mut u64,
        layer_idx: usize,
        o_shape: Vec<u64>,
        down_shape: Vec<u64>,
        short_q_data: bool,
    ) {
        let prefix = format!("model.layers.{layer_idx}");
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        let hidden_square = vec![hidden, hidden];

        if short_q_data {
            add_f32_tensor(
                writer,
                *tensor_id,
                &format!("{prefix}.self_attn.q_proj.weight"),
                hidden_square.clone(),
                &[0.0],
            );
            *tensor_id += 1;
        } else {
            add_zero_f32_tensor(
                writer,
                tensor_id,
                &format!("{prefix}.self_attn.q_proj.weight"),
                hidden_square.clone(),
            );
        }
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.self_attn.k_proj.weight"),
            hidden_square.clone(),
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.self_attn.v_proj.weight"),
            hidden_square.clone(),
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.self_attn.o_proj.weight"),
            o_shape,
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        add_zero_f32_tensor(
            writer,
            tensor_id,
            &format!("{prefix}.mlp.down_proj.weight"),
            down_shape,
        );
    }

    fn add_complete_layer(writer: &mut SpissaWriter, tensor_id: &mut u64, layer_idx: usize) {
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        add_layer_norms(writer, tensor_id, layer_idx);
        add_layer_projection_tensors(
            writer,
            tensor_id,
            layer_idx,
            vec![hidden, hidden],
            vec![hidden, intermediate],
            false,
        );
    }

    fn add_layer_with_bad_o_projection(
        writer: &mut SpissaWriter,
        tensor_id: &mut u64,
        layer_idx: usize,
    ) {
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        add_layer_norms(writer, tensor_id, layer_idx);
        add_layer_projection_tensors(
            writer,
            tensor_id,
            layer_idx,
            vec![hidden - 1, hidden],
            vec![hidden, intermediate],
            false,
        );
    }

    fn add_layer_with_bad_down_projection(
        writer: &mut SpissaWriter,
        tensor_id: &mut u64,
        layer_idx: usize,
    ) {
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        add_layer_norms(writer, tensor_id, layer_idx);
        add_layer_projection_tensors(
            writer,
            tensor_id,
            layer_idx,
            vec![hidden, hidden],
            vec![hidden, intermediate - 1],
            false,
        );
    }

    fn add_layer_with_runtime_q_failure(
        writer: &mut SpissaWriter,
        tensor_id: &mut u64,
        layer_idx: usize,
    ) {
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        add_layer_norms(writer, tensor_id, layer_idx);
        add_layer_projection_tensors(
            writer,
            tensor_id,
            layer_idx,
            vec![hidden, hidden],
            vec![hidden, intermediate],
            true,
        );
    }

    fn write_bad_attention_projection_model(path: &std::path::Path) {
        let mut writer = SpissaWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(
            &mut writer,
            &mut tensor_id,
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_layer_with_bad_o_projection(&mut writer, &mut tensor_id, 0);
        writer.finalize().unwrap();
    }

    fn write_bad_mlp_projection_model(path: &std::path::Path) {
        let mut writer = SpissaWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(
            &mut writer,
            &mut tensor_id,
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_layer_with_bad_down_projection(&mut writer, &mut tensor_id, 0);
        writer.finalize().unwrap();
    }

    fn add_complete_layer_zero(writer: &mut SpissaWriter, tensor_id: &mut u64) {
        add_complete_layer(writer, tensor_id, 0);
    }

    fn write_constructor_model(path: &std::path::Path, lm_head_shape: Vec<u64>) {
        let mut writer = SpissaWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(&mut writer, &mut tensor_id, lm_head_shape);
        add_layer_norms(&mut writer, &mut tensor_id, 0);
        writer.finalize().unwrap();
    }

    fn write_bf16_lm_head_model(path: &std::path::Path, vocab_size: usize) {
        let mut writer = SpissaWriter::new(path, llama_metadata_with_vocab(vocab_size)).unwrap();
        let mut tensor_id = 0u64;
        add_f32_tensor(
            &mut writer,
            tensor_id,
            "model.embed_tokens.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0.0; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            "lm_head.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0x0000; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_complete_layer_zero(&mut writer, &mut tensor_id);
        writer.finalize().unwrap();
    }

    fn write_bf16_mlp_speed_model(path: &std::path::Path, vocab_size: usize) {
        let mut writer = SpissaWriter::new(path, llama_metadata_with_vocab(vocab_size)).unwrap();
        let mut tensor_id = 0u64;
        add_f32_tensor(
            &mut writer,
            tensor_id,
            "model.embed_tokens.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0.0; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            "lm_head.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0x0000; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_layer_norms(&mut writer, &mut tensor_id, 0);
        let prefix = "model.layers.0";
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.q_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.k_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.v_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_zero_f32_tensor(
            &mut writer,
            &mut tensor_id,
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
            &[0x0000; HIDDEN_SIZE * INTERMEDIATE_SIZE],
        );
        writer.finalize().unwrap();
    }

    fn write_bf16_mlp_speed_model_with_layers(
        path: &std::path::Path,
        vocab_size: usize,
        layer_count: usize,
    ) {
        let mut writer = SpissaWriter::new(path, llama_metadata_with_vocab(vocab_size)).unwrap();
        let mut tensor_id = 0u64;
        add_f32_tensor(
            &mut writer,
            tensor_id,
            "model.embed_tokens.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0.0; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        add_bf16_tensor(
            &mut writer,
            tensor_id,
            "lm_head.weight",
            vec![vocab_size as u64, HIDDEN_SIZE as u64],
            &vec![0x0000; vocab_size * HIDDEN_SIZE],
        );
        tensor_id += 1;
        for layer_idx in 0..layer_count {
            add_layer_norms(&mut writer, &mut tensor_id, layer_idx);
            let prefix = format!("model.layers.{layer_idx}");
            add_zero_f32_tensor(
                &mut writer,
                &mut tensor_id,
                &format!("{prefix}.self_attn.q_proj.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            add_zero_f32_tensor(
                &mut writer,
                &mut tensor_id,
                &format!("{prefix}.self_attn.k_proj.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            add_zero_f32_tensor(
                &mut writer,
                &mut tensor_id,
                &format!("{prefix}.self_attn.v_proj.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            add_zero_f32_tensor(
                &mut writer,
                &mut tensor_id,
                &format!("{prefix}.self_attn.o_proj.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            add_bf16_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.gate_proj.weight"),
                vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
                &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
            );
            tensor_id += 1;
            add_bf16_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.up_proj.weight"),
                vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
                &[0x0000; INTERMEDIATE_SIZE * HIDDEN_SIZE],
            );
            tensor_id += 1;
            add_bf16_tensor(
                &mut writer,
                tensor_id,
                &format!("{prefix}.mlp.down_proj.weight"),
                vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
                &[0x0000; HIDDEN_SIZE * INTERMEDIATE_SIZE],
            );
            tensor_id += 1;
        }
        writer.finalize().unwrap();
    }

    fn write_post_cache_failure_model(path: &std::path::Path) {
        let mut writer = SpissaWriter::new(path, llama_metadata()).unwrap();
        let mut tensor_id = 0u64;
        add_base_tensors(
            &mut writer,
            &mut tensor_id,
            vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
        );
        add_complete_layer_zero(&mut writer, &mut tensor_id);
        add_layer_with_runtime_q_failure(&mut writer, &mut tensor_id, 1);
        writer.finalize().unwrap();
    }

    #[test]
    fn llama_session_new_rejects_empty_prepared_layers() {
        let path = temp_path("empty-prepared-layers");
        write_constructor_model(&path, vec![VOCAB_SIZE as u64, HIDDEN_SIZE as u64]);
        let mut model = LazySpissaModel::open(&path).unwrap();
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
        let mut model = LazySpissaModel::open(&path).unwrap();
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
        let mut model = LazySpissaModel::open(&path).unwrap();
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
    fn llama_session_new_rejects_malformed_attention_projection_shape() {
        let path = temp_path("malformed-attention-projection");
        write_bad_attention_projection_model(&path);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("self_attn.o_proj.weight")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_new_rejects_malformed_mlp_projection_shape() {
        let path = temp_path("malformed-mlp-projection");
        write_bad_mlp_projection_model(&path);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();

        let result = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget);

        assert!(matches!(
            result,
            Err(RuntimeError::Shape(message)) if message.contains("mlp.down_proj.weight")
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_append_rolls_back_all_caches_after_post_cache_layer_failure() {
        let path = temp_path("rollback-post-cache-failure");
        write_post_cache_failure_model(&path);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let prepared = prepared_with_layers(2);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();

        let result = adapter.append_tokens(&[0], &mut budget, false);

        assert!(result.is_err());
        assert_eq!(adapter.context_len(), 0);
        assert_eq!(adapter.context_memory_bytes(), 0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_records_phase_timings_for_logits_append() {
        let path = temp_path("phase-timing-logits");
        write_post_cache_failure_model(&path);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
        adapter.set_transformer_detail_timing(true);

        let step = adapter.append_tokens(&[0], &mut budget, true).unwrap();
        let timings = adapter.take_last_phase_timings().unwrap();

        assert!(step.is_some());
        assert!(timings.embedding_ms >= 0.0);
        assert!(timings.transformer_ms >= 0.0);
        assert_eq!(timings.transformer_detail.profiled_layers, 1);
        assert!(timings.transformer_detail.attention_total_ms() >= 0.0);
        assert!(timings.transformer_detail.mlp_total_ms() >= 0.0);
        assert!(timings.final_norm_ms >= 0.0);
        assert!(timings.lm_head_ms >= 0.0);
        assert!(timings.total_ms() >= 0.0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_reports_rolling_stats_when_executor_is_enabled() {
        let path = temp_path("llama-session-rolling");
        write_bf16_lm_head_model(&path, 8);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
        adapter.enable_rolling_executor_for_test(4, 1);

        let _ = adapter.append_tokens(&[1], &mut budget, true).unwrap();
        let stats = adapter.take_last_rolling_stats().unwrap();

        assert!(stats.submitted_tasks > 0);
        assert!(stats.worker_wakeups > 0);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_reports_experimental_speed_stats_when_enabled_for_test() {
        let path = temp_path("experimental-speed-stats");
        write_bf16_mlp_speed_model(&path, 8);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
        adapter.enable_experimental_speed_for_test(crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
            aip_topk: Some(1),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        });

        adapter.append_tokens(&[0], &mut budget, true).unwrap();
        let stats = adapter.take_last_experimental_speed_stats().unwrap();

        assert!(stats.sparse_projection_calls > 0);
        assert!(stats.max_selected_topk <= 1);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_layer_drift_probe_records_decode_shadow_pass() {
        let path = temp_path("layer-drift-probe");
        write_bf16_mlp_speed_model(&path, 8);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
        adapter.enable_experimental_speed_for_test(crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
            aip_topk: Some(1),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        });
        adapter.enable_layer_drift_probe_for_test(true);

        let first = adapter
            .append_tokens(&[0], &mut budget, true)
            .unwrap()
            .unwrap();
        let prompt_stats = adapter.take_last_experimental_speed_stats().unwrap();
        let _ = adapter
            .append_tokens(&[first.token_id], &mut budget, true)
            .unwrap();
        let decode_stats = adapter.take_last_experimental_speed_stats().unwrap();

        assert_eq!(prompt_stats.layer_drift_probe.samples, 0);
        assert_eq!(decode_stats.layer_drift_probe.samples, 1);
        assert_eq!(decode_stats.layer_drift_probe.layers, 1);
        assert_eq!(
            decode_stats.aip_policy,
            Some(crate::RamaAipPolicyKind::Speed)
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn llama_session_exact_prefill_keeps_decode_aip_enabled() {
        let path = temp_path("exact-prefill-speed-decode");
        write_bf16_mlp_speed_model(&path, 8);
        let mut model = LazySpissaModel::open(&path).unwrap();
        let prepared = prepared_with_layers(1);
        let mut budget = MemoryBudget::unbounded();
        let mut adapter = LlamaRamaSessionAdapter::new(&mut model, &prepared, &mut budget).unwrap();
        adapter.enable_experimental_speed_for_test(crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
            aip_topk: Some(1),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: false,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        });
        adapter.enable_exact_prefill_for_test(true);

        let first_step = adapter
            .append_tokens(&[0, 1], &mut budget, true)
            .unwrap()
            .unwrap();
        let prompt_stats = adapter.take_last_experimental_speed_stats().unwrap();
        assert_eq!(prompt_stats.sparse_projection_calls, 0);

        adapter
            .append_tokens(&[first_step.token_id], &mut budget, true)
            .unwrap();
        let decode_stats = adapter.take_last_experimental_speed_stats().unwrap();
        assert!(decode_stats.sparse_projection_calls > 0);
        assert_eq!(
            decode_stats.aip_policy,
            Some(crate::RamaAipPolicyKind::Speed)
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn lm_head_exact_periodic_check_targets_decode_token_indices() {
        assert!(!lm_head_exact_check_due(None, true, 3));
        assert!(!lm_head_exact_check_due(Some(4), false, 3));
        assert!(!lm_head_exact_check_due(Some(4), true, 2));
        assert!(lm_head_exact_check_due(Some(4), true, 3));
        assert!(lm_head_exact_check_due(Some(1), true, 0));
    }

