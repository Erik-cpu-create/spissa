// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

impl<'a> LlamaRamaSessionAdapter<'a> {
    fn collect_layer_drift_probe_outputs(
        &mut self,
        tokens: &[usize],
        position_offset: usize,
        experimental_speed_config: RamaExperimentalSpeedConfig,
        budget: &mut MemoryBudget,
    ) -> Result<Vec<LlamaLayerDriftProbeOutput>> {
        let seq_len = tokens.len();
        let mut hidden = embedding_lookup(
            &self.embedding_data,
            self.vocab_size,
            self.hidden_size,
            tokens,
        )?;
        let mut caches = self.caches.clone();
        let mut sparse_column_cache = SparseColumnCache::from_env();
        let mut attention_locality_caches = self.attention_locality_caches.clone();
        let mut shadow_speed_stats = RamaExperimentalSpeedStats::default();
        let mut outputs = Vec::with_capacity(self.prepared.layers.len());
        let lm_head_config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: self.hidden_size,
                out_features: self.vocab_size,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        };

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
                layer_index: i,
                total_layers: self.prepared.layers.len(),
                experimental_speed: experimental_speed_config,
            };
            let experimental_stats_ref = if experimental_speed_config.enabled {
                Some(&mut shadow_speed_stats)
            } else {
                None
            };
            let sparse_column_cache_ref = if experimental_speed_config.aip_column_cache {
                Some(&mut sparse_column_cache)
            } else {
                None
            };
            let attention_locality_cache_ref = if experimental_speed_config
                .attention_locality_enabled_for_layer(i, self.prepared.layers.len())
            {
                Some(&mut attention_locality_caches[i])
            } else {
                None
            };

            hidden = streaming_llama_transformer_block_with_timing(
                self.model,
                &hidden,
                layer_names,
                &self.layer_norms[i],
                config,
                budget,
                Some(&mut caches[i]),
                None,
                experimental_stats_ref,
                sparse_column_cache_ref,
                attention_locality_cache_ref,
                None,
            )?;

            let normalized = rms_norm(
                &hidden,
                &self.prepared.final_layernorm_weight,
                seq_len,
                self.hidden_size,
                self.prepared.config.rms_norm_eps,
            )?;
            let last_hidden_start = (seq_len - 1) * self.hidden_size;
            let last_hidden = &normalized[last_hidden_start..last_hidden_start + self.hidden_size];
            let logits = streaming_tile_linear_from_model(
                self.model,
                &self.prepared.lm_head_weight,
                last_hidden,
                None,
                lm_head_config,
                budget,
            )?;
            let (token_id, top_value, second) = top_two_sparse_logits(&logits)?;
            let exact_margin_milli = second
                .map(|(_, second_value)| gap_to_milli(top_value - second_value))
                .unwrap_or(0);
            let hidden_start = (seq_len - 1) * self.hidden_size;
            outputs.push(LlamaLayerDriftProbeOutput {
                hidden: hidden[hidden_start..hidden_start + self.hidden_size].to_vec(),
                token_id,
                exact_margin_milli,
            });
        }

        Ok(outputs)
    }

    fn collect_layer_attribution_probe(
        &mut self,
        tokens: &[usize],
        position_offset: usize,
        experimental_speed_config: RamaExperimentalSpeedConfig,
        target_layer: usize,
        budget: &mut MemoryBudget,
    ) -> Result<LlamaStreamingBlockProbe> {
        let Some(target_index) = target_layer.checked_sub(1) else {
            return Err(RuntimeError::Shape(
                "layer attribution target must be 1-based".to_string(),
            ));
        };
        let seq_len = tokens.len();
        let mut hidden = embedding_lookup(
            &self.embedding_data,
            self.vocab_size,
            self.hidden_size,
            tokens,
        )?;
        let mut caches = self.caches.clone();
        let mut sparse_column_cache = SparseColumnCache::from_env();
        let mut attention_locality_caches = self.attention_locality_caches.clone();
        let mut shadow_speed_stats = RamaExperimentalSpeedStats::default();

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
                layer_index: i,
                total_layers: self.prepared.layers.len(),
                experimental_speed: experimental_speed_config,
            };
            let experimental_stats_ref = if experimental_speed_config.enabled {
                Some(&mut shadow_speed_stats)
            } else {
                None
            };
            let sparse_column_cache_ref = if experimental_speed_config.aip_column_cache {
                Some(&mut sparse_column_cache)
            } else {
                None
            };
            let attention_locality_cache_ref = if experimental_speed_config
                .attention_locality_enabled_for_layer(i, self.prepared.layers.len())
            {
                Some(&mut attention_locality_caches[i])
            } else {
                None
            };
            let mut block_probe = if i == target_index {
                Some(LlamaStreamingBlockProbe::default())
            } else {
                None
            };

            hidden = streaming_llama_transformer_block_with_timing(
                self.model,
                &hidden,
                layer_names,
                &self.layer_norms[i],
                config,
                budget,
                Some(&mut caches[i]),
                None,
                experimental_stats_ref,
                sparse_column_cache_ref,
                attention_locality_cache_ref,
                block_probe.as_mut(),
            )?;

            if i == target_index {
                return Ok(block_probe.unwrap_or_default());
            }
        }

        Err(RuntimeError::Shape(format!(
            "layer attribution target {target_layer} exceeds layer count {}",
            self.prepared.layers.len()
        )))
    }

    fn record_layer_drift_probe(
        &mut self,
        tokens: &[usize],
        position_offset: usize,
        experimental_speed_config: RamaExperimentalSpeedConfig,
        budget: &MemoryBudget,
        stats: &mut RamaExperimentalSpeedStats,
    ) -> Result<()> {
        let remaining_bytes = budget.limit_bytes().saturating_sub(budget.current_bytes());
        let mut probe_budget = MemoryBudget::new(remaining_bytes);
        let exact_outputs = self.collect_layer_drift_probe_outputs(
            tokens,
            position_offset,
            RamaExperimentalSpeedConfig::disabled(),
            &mut probe_budget,
        )?;
        let sparse_outputs = self.collect_layer_drift_probe_outputs(
            tokens,
            position_offset,
            experimental_speed_config,
            &mut probe_budget,
        )?;
        let layers = exact_outputs.len().min(sparse_outputs.len());
        if layers == 0 {
            return Ok(());
        }

        let mut mismatch_layers = 0usize;
        let mut first_mismatch_layer = None;
        let mut pre_mismatch_max_l2_milli = 0usize;
        let mut pre_mismatch_max_cosine_gap_milli = 0usize;
        let mut max_l2_milli = 0usize;
        let mut max_cosine_gap_milli = 0usize;
        let mut max_exact_margin_milli = 0usize;
        for (layer_idx, (exact, sparse)) in exact_outputs
            .iter()
            .zip(sparse_outputs.iter())
            .take(layers)
            .enumerate()
        {
            if exact.token_id != sparse.token_id {
                mismatch_layers = mismatch_layers.saturating_add(1);
                first_mismatch_layer.get_or_insert(layer_idx + 1);
            }
            let l2 = hidden_l2_milli(&exact.hidden, &sparse.hidden);
            let cosine = hidden_cosine_gap_milli(&exact.hidden, &sparse.hidden);
            max_l2_milli = max_l2_milli.max(l2);
            max_cosine_gap_milli = max_cosine_gap_milli.max(cosine);
            max_exact_margin_milli = max_exact_margin_milli.max(exact.exact_margin_milli);

            if first_mismatch_layer.is_none() && exact.token_id == sparse.token_id {
                pre_mismatch_max_l2_milli = pre_mismatch_max_l2_milli.max(l2);
                pre_mismatch_max_cosine_gap_milli = pre_mismatch_max_cosine_gap_milli.max(cosine);
            }
        }

        stats.record_aip_policy(experimental_speed_config.aip_policy);
        stats.record_layer_drift_probe(
            layers,
            mismatch_layers,
            first_mismatch_layer,
            pre_mismatch_max_l2_milli,
            pre_mismatch_max_cosine_gap_milli,
            max_l2_milli,
            max_cosine_gap_milli,
            max_exact_margin_milli,
        );
        if let Some(layer) = first_mismatch_layer {
            let exact_probe = self.collect_layer_attribution_probe(
                tokens,
                position_offset,
                RamaExperimentalSpeedConfig::disabled(),
                layer,
                &mut probe_budget,
            )?;
            let sparse_probe = self.collect_layer_attribution_probe(
                tokens,
                position_offset,
                experimental_speed_config,
                layer,
                &mut probe_budget,
            )?;
            let (attention_l2_milli, attention_cosine_gap_milli) = optional_vector_metrics(
                exact_probe.attention_output.as_deref(),
                sparse_probe.attention_output.as_deref(),
            );
            let (gate_up_l2_milli, gate_up_cosine_gap_milli) = optional_vector_metrics(
                exact_probe.gate_up_output.as_deref(),
                sparse_probe.gate_up_output.as_deref(),
            );
            let (down_l2_milli, down_cosine_gap_milli) = optional_vector_metrics(
                exact_probe.down_output.as_deref(),
                sparse_probe.down_output.as_deref(),
            );
            stats.record_layer_attribution_probe(
                layer,
                attention_l2_milli,
                attention_cosine_gap_milli,
                gate_up_l2_milli,
                gate_up_cosine_gap_milli,
                down_l2_milli,
                down_cosine_gap_milli,
            );
        }
        Ok(())
    }
}
