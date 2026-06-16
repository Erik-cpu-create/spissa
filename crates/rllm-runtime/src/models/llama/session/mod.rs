use crate::models::llama::api::{
    decode_vector_tensor, require_config_usize, require_model_config, validate_llama_shape,
};
use crate::models::llama::generate::{
    streaming_llama_transformer_block_with_timing, LlamaStreamingBlockConfig,
    LlamaStreamingBlockProbe,
};
use crate::models::llama::model::{
    LayerDecodedLlamaRamaTransformer, OwnedLlamaStreamingBlockParameters,
    OwnedLlamaStreamingBlockTensorNames,
};
use crate::rolling::RollingExecutor;
#[cfg(test)]
use crate::rolling::RollingExecutorConfig;
use crate::rotary::KvCache;
use crate::speed::{
    parse_aip_exact_prefill_enabled, parse_aip_layer_drift_probe_enabled,
    parse_aip_lm_head_exact_every, RLLM_AIP_EXACT_PREFILL_ENV, RLLM_AIP_LAYER_DRIFT_PROBE_ENV,
    RLLM_AIP_LM_HEAD_EXACT_EVERY_ENV,
};
use crate::streaming::{
    streaming_tile_linear_argmax_candidate_rows_range_from_model,
    streaming_tile_linear_argmax_prefix_from_model,
    streaming_tile_linear_argmax_with_rolling_from_model,
};
use crate::{
    embedding_lookup, rms_norm, sample_argmax_excluding, sample_top_p, select_top_indices_by_value,
    streaming_input_tiled_sparse_tile_linear_from_model, streaming_tile_linear_argmax_from_model,
    streaming_tile_linear_from_model, LazyRllmModel, MemoryBudget, Result, RuntimeError,
};
use crate::{RamaAttentionLocalityCache, RamaExperimentalSpeedConfig, RamaExperimentalSpeedStats};
use crate::{
    RamaRollingStats, RamaSessionAdapter, RamaSessionPhaseTimings, RamaSessionStep,
    RamaTransformerPhaseTimings, SparseColumnCache, StreamingLinearConfig,
    StreamingTileLinearConfig, DEFAULT_STREAMING_TILE_ELEMENTS,
};
use std::time::Instant;

pub struct LlamaRamaSessionAdapter<'a> {
    model: &'a mut LazyRllmModel,
    prepared: LayerDecodedLlamaRamaTransformer,
    hidden_size: usize,
    intermediate_size: usize,
    head_dim: usize,
    vocab_size: usize,
    embedding_data: Vec<f32>,
    layer_norms: Vec<OwnedLlamaStreamingBlockParameters>,
    caches: Vec<KvCache>,
    last_phase_timings: Option<RamaSessionPhaseTimings>,
    rolling_executor: Option<RollingExecutor>,
    last_rolling_stats: Option<RamaRollingStats>,
    experimental_speed_config: RamaExperimentalSpeedConfig,
    exact_prefill: bool,
    lm_head_exact_every: Option<usize>,
    layer_drift_probe: bool,
    last_experimental_speed_stats: Option<RamaExperimentalSpeedStats>,
    sparse_column_cache: SparseColumnCache,
    attention_locality_caches: Vec<RamaAttentionLocalityCache>,
    collect_transformer_detail_timing: bool,
    last_generated_token: Option<usize>,
    last_generated_token_run: usize,
    generated_token_count_in_turn: usize,
    lm_head_repeat_margin_state: LmHeadRepeatMarginState,
    lm_head_phrase_novelty_state: LmHeadPhraseNoveltyState,
}

include!("validation.rs");
include!("lm_head.rs");

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
        for layer_names in &prepared.layers {
            validate_layer_tensor_shapes(
                model,
                layer_names,
                hidden_size,
                prepared.config.num_heads,
                prepared.config.num_key_value_heads,
                head_dim,
                intermediate_size,
            )?;
        }

        let embedding_data = model
            .decode_tensor(&prepared.embedding_weight, budget)?
            .data;

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
        let attention_locality_caches =
            vec![RamaAttentionLocalityCache::default(); prepared.layers.len()];

        Ok(Self {
            model,
            prepared: prepared.clone(),
            hidden_size,
            intermediate_size,
            head_dim,
            vocab_size,
            embedding_data,
            layer_norms,
            caches,
            last_phase_timings: None,
            rolling_executor: RollingExecutor::from_env(
                crate::streaming::streaming_available_threads(),
            ),
            last_rolling_stats: None,
            experimental_speed_config: RamaExperimentalSpeedConfig::from_env(),
            exact_prefill: parse_aip_exact_prefill_enabled(
                std::env::var(RLLM_AIP_EXACT_PREFILL_ENV).ok().as_deref(),
            ),
            lm_head_exact_every: parse_aip_lm_head_exact_every(
                std::env::var(RLLM_AIP_LM_HEAD_EXACT_EVERY_ENV)
                    .ok()
                    .as_deref(),
            ),
            layer_drift_probe: parse_aip_layer_drift_probe_enabled(
                std::env::var(RLLM_AIP_LAYER_DRIFT_PROBE_ENV)
                    .ok()
                    .as_deref(),
            ),
            last_experimental_speed_stats: None,
            sparse_column_cache: SparseColumnCache::from_env(),
            attention_locality_caches,
            collect_transformer_detail_timing: false,
            last_generated_token: None,
            last_generated_token_run: 0,
            generated_token_count_in_turn: 0,
            lm_head_repeat_margin_state: LmHeadRepeatMarginState::default(),
            lm_head_phrase_novelty_state: LmHeadPhraseNoveltyState::default(),
        })
    }

    pub fn set_transformer_detail_timing(&mut self, enabled: bool) {
        self.collect_transformer_detail_timing = enabled;
    }

    fn record_generated_token(&mut self, token_id: usize, reset_run: bool) {
        if reset_run {
            self.lm_head_phrase_novelty_state.reset();
            self.generated_token_count_in_turn = 0;
        }
        if !reset_run && self.last_generated_token == Some(token_id) {
            self.last_generated_token_run = self.last_generated_token_run.saturating_add(1);
        } else {
            self.last_generated_token = Some(token_id);
            self.last_generated_token_run = 1;
        }
        self.generated_token_count_in_turn = self.generated_token_count_in_turn.saturating_add(1);
        self.lm_head_phrase_novelty_state.push(token_id);
    }

    #[cfg(test)]
    pub(crate) fn enable_rolling_executor_for_test(
        &mut self,
        worker_count: usize,
        min_rows_per_worker: usize,
    ) {
        self.rolling_executor = Some(RollingExecutor::new(RollingExecutorConfig {
            enabled: true,
            worker_count,
            min_rows_per_worker,
        }));
    }

    #[cfg(test)]
    pub(crate) fn enable_experimental_speed_for_test(
        &mut self,
        config: RamaExperimentalSpeedConfig,
    ) {
        self.experimental_speed_config = config;
    }

    #[cfg(test)]
    pub(crate) fn enable_exact_prefill_for_test(&mut self, enabled: bool) {
        self.exact_prefill = enabled;
    }

    #[cfg(test)]
    pub(crate) fn enable_layer_drift_probe_for_test(&mut self, enabled: bool) {
        self.layer_drift_probe = enabled;
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
        let is_decode_step =
            tokens.len() == 1 && self.last_generated_token == tokens.first().copied();
        let experimental_speed_config = if self.exact_prefill && emit_logits && !is_decode_step {
            RamaExperimentalSpeedConfig::disabled()
        } else {
            self.experimental_speed_config
        };
        if !is_decode_step {
            for cache in &mut self.attention_locality_caches {
                cache.clear();
            }
        }

        let mut phase_timings = RamaSessionPhaseTimings::default();
        let mut experimental_speed_stats = RamaExperimentalSpeedStats::default();
        if self.layer_drift_probe
            && emit_logits
            && is_decode_step
            && experimental_speed_config.enabled
        {
            self.record_layer_drift_probe(
                tokens,
                position_offset,
                experimental_speed_config,
                budget,
                &mut experimental_speed_stats,
            )?;
        }
        let phase_start = Instant::now();
        let mut hidden = embedding_lookup(
            &self.embedding_data,
            self.vocab_size,
            self.hidden_size,
            tokens,
        )?;
        phase_timings.embedding_ms += phase_start.elapsed().as_secs_f64() * 1000.0;

        let phase_start = Instant::now();
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
            let mut transformer_detail = RamaTransformerPhaseTimings::default();
            let transformer_detail_timing = if self.collect_transformer_detail_timing {
                Some(&mut transformer_detail)
            } else {
                None
            };
            let experimental_stats_ref = if experimental_speed_config.enabled {
                Some(&mut experimental_speed_stats)
            } else {
                None
            };
            let sparse_column_cache = if experimental_speed_config.aip_column_cache {
                Some(&mut self.sparse_column_cache)
            } else {
                None
            };
            let attention_locality_cache = if experimental_speed_config
                .attention_locality_enabled_for_layer(i, self.prepared.layers.len())
            {
                Some(&mut self.attention_locality_caches[i])
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
                Some(&mut self.caches[i]),
                transformer_detail_timing,
                experimental_stats_ref,
                sparse_column_cache,
                attention_locality_cache,
                None,
            )?;
            if self.collect_transformer_detail_timing {
                phase_timings
                    .transformer_detail
                    .add_assign(transformer_detail);
            }
        }
        phase_timings.transformer_ms += phase_start.elapsed().as_secs_f64() * 1000.0;

        if !emit_logits {
            self.last_phase_timings = Some(phase_timings);
            self.last_experimental_speed_stats = Some(experimental_speed_stats);
            return Ok(None);
        }

        let phase_start = Instant::now();
        let hidden = rms_norm(
            &hidden,
            &self.prepared.final_layernorm_weight,
            seq_len,
            self.hidden_size,
            self.prepared.config.rms_norm_eps,
        )?;
        phase_timings.final_norm_ms += phase_start.elapsed().as_secs_f64() * 1000.0;

        let phase_start = Instant::now();
        let last_hidden = &hidden[(seq_len - 1) * self.hidden_size..];
        let previous_token_run = if is_decode_step {
            self.last_generated_token_run
        } else {
            self.lm_head_repeat_margin_state.reset();
            self.lm_head_phrase_novelty_state.reset();
            0
        };
        let lm_head_config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: self.hidden_size,
                out_features: self.vocab_size,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        };
        let (token_id, logits) = match self.prepared.config.sampling {
            crate::StreamingSamplingConfig::Argmax => {
                let token_id = if let Some(prefix_rows) =
                    experimental_speed_config.lm_head_prefix_rows(self.vocab_size)
                {
                    experimental_speed_stats
                        .record_aip_policy(experimental_speed_config.aip_policy);
                    experimental_speed_stats.record_lm_head_prefix(prefix_rows, self.vocab_size);
                    streaming_tile_linear_argmax_prefix_from_model(
                        self.model,
                        &self.prepared.lm_head_weight,
                        last_hidden,
                        None,
                        lm_head_config,
                        prefix_rows,
                        budget,
                    )?
                } else if experimental_speed_config.enabled
                    && experimental_speed_config.aip_input_tiles
                {
                    experimental_speed_stats
                        .record_aip_policy(experimental_speed_config.aip_policy);
                    let sparse_config = RamaExperimentalSpeedConfig {
                        enabled: true,
                        aip_policy: experimental_speed_config.aip_policy,
                        aip_topk: Some(
                            experimental_speed_config.lm_head_topk_for_input(self.hidden_size, 128),
                        ),
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
                        aip_input_tiles: true,
                        aip_no_repeat_last: false,
                        aip_repeat_run_limit: None,
                    };
                    match streaming_input_tiled_sparse_tile_linear_from_model(
                        self.model,
                        &self.prepared.lm_head_weight,
                        last_hidden,
                        None,
                        lm_head_config,
                        sparse_config,
                        &mut experimental_speed_stats,
                        budget,
                    )? {
                        Some(logits) => {
                            let sparse_lm_head_topk = sparse_config.aip_topk.unwrap_or(0);
                            let selected_token_id = if let Some(candidates) =
                                sparse_lm_head_rescore_candidates(
                                    &logits,
                                    tokens,
                                    experimental_speed_config,
                                    &mut experimental_speed_stats,
                                )? {
                                match streaming_tile_linear_argmax_candidate_rows_range_from_model(
                                    self.model,
                                    &self.prepared.lm_head_weight,
                                    last_hidden,
                                    None,
                                    lm_head_config,
                                    &candidates,
                                    budget,
                                )? {
                                    Some(token_id) => apply_rescored_lm_head_controllers(
                                        &logits,
                                        token_id,
                                        tokens,
                                        previous_token_run,
                                        experimental_speed_config,
                                        &mut experimental_speed_stats,
                                        &mut self.lm_head_repeat_margin_state,
                                        &mut self.lm_head_phrase_novelty_state,
                                    )?,
                                    None => sample_sparse_lm_head_argmax_with_controller_state(
                                        &logits,
                                        tokens,
                                        previous_token_run,
                                        experimental_speed_config,
                                        &mut experimental_speed_stats,
                                        &mut self.lm_head_repeat_margin_state,
                                        &mut self.lm_head_phrase_novelty_state,
                                    )?,
                                }
                            } else {
                                sample_sparse_lm_head_argmax_with_controller_state(
                                    &logits,
                                    tokens,
                                    previous_token_run,
                                    experimental_speed_config,
                                    &mut experimental_speed_stats,
                                    &mut self.lm_head_repeat_margin_state,
                                    &mut self.lm_head_phrase_novelty_state,
                                )?
                            };
                            let exact_check_due = lm_head_exact_check_due(
                                self.lm_head_exact_every,
                                is_decode_step,
                                self.generated_token_count_in_turn,
                            );
                            if exact_check_due || experimental_speed_config.aip_lm_head_agreement {
                                let exact_token_id = streaming_tile_linear_argmax_from_model(
                                    self.model,
                                    &self.prepared.lm_head_weight,
                                    last_hidden,
                                    None,
                                    lm_head_config,
                                    budget,
                                )?;
                                if exact_check_due {
                                    experimental_speed_stats.record_lm_head_exact_check(
                                        selected_token_id != exact_token_id,
                                    );
                                    if experimental_speed_config.aip_lm_head_agreement {
                                        record_sparse_lm_head_agreement_sample(
                                            &mut experimental_speed_stats,
                                            &logits,
                                            selected_token_id,
                                            exact_token_id,
                                            sparse_lm_head_topk,
                                        )?;
                                    }
                                    exact_token_id
                                } else {
                                    record_sparse_lm_head_agreement_sample(
                                        &mut experimental_speed_stats,
                                        &logits,
                                        selected_token_id,
                                        exact_token_id,
                                        sparse_lm_head_topk,
                                    )?;
                                    selected_token_id
                                }
                            } else {
                                selected_token_id
                            }
                        }
                        None => {
                            if let Some(executor) = self.rolling_executor.as_mut() {
                                let token = streaming_tile_linear_argmax_with_rolling_from_model(
                                    self.model,
                                    &self.prepared.lm_head_weight,
                                    last_hidden,
                                    None,
                                    lm_head_config,
                                    budget,
                                    Some(executor),
                                )?;
                                self.last_rolling_stats = Some(executor.take_stats());
                                token
                            } else {
                                streaming_tile_linear_argmax_from_model(
                                    self.model,
                                    &self.prepared.lm_head_weight,
                                    last_hidden,
                                    None,
                                    lm_head_config,
                                    budget,
                                )?
                            }
                        }
                    }
                } else if let Some(executor) = self.rolling_executor.as_mut() {
                    let token = streaming_tile_linear_argmax_with_rolling_from_model(
                        self.model,
                        &self.prepared.lm_head_weight,
                        last_hidden,
                        None,
                        lm_head_config,
                        budget,
                        Some(executor),
                    )?;
                    self.last_rolling_stats = Some(executor.take_stats());
                    token
                } else {
                    streaming_tile_linear_argmax_from_model(
                        self.model,
                        &self.prepared.lm_head_weight,
                        last_hidden,
                        None,
                        lm_head_config,
                        budget,
                    )?
                };
                (token_id, None)
            }
            crate::StreamingSamplingConfig::TopP {
                temperature,
                top_p,
                seed,
            } => {
                let logits = streaming_tile_linear_from_model(
                    self.model,
                    &self.prepared.lm_head_weight,
                    last_hidden,
                    None,
                    lm_head_config,
                    budget,
                )?;
                let token_id = sample_top_p(&logits, temperature, top_p, seed)?;
                (token_id, Some(logits))
            }
        };
        phase_timings.lm_head_ms += phase_start.elapsed().as_secs_f64() * 1000.0;
        self.last_phase_timings = Some(phase_timings);
        self.last_experimental_speed_stats = Some(experimental_speed_stats);
        self.record_generated_token(token_id, !is_decode_step);
        Ok(Some(RamaSessionStep {
            token_id,
            logits,
            cached_context_len_after: self.context_len(),
        }))
    }
}

include!("drift_probe.rs");

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
        self.last_phase_timings = None;
        self.last_experimental_speed_stats = None;
        let old_lens: Vec<usize> = self.caches.iter().map(KvCache::len).collect();
        let old_last_generated_token = self.last_generated_token;
        let old_last_generated_token_run = self.last_generated_token_run;
        let old_generated_token_count_in_turn = self.generated_token_count_in_turn;
        match self.append_tokens_inner(tokens, budget, emit_logits) {
            Ok(step) => Ok(step),
            Err(error) => {
                for (cache, len) in self.caches.iter_mut().zip(old_lens) {
                    let _ = cache.truncate(len);
                }
                self.last_phase_timings = None;
                self.last_experimental_speed_stats = None;
                self.last_generated_token = old_last_generated_token;
                self.last_generated_token_run = old_last_generated_token_run;
                self.generated_token_count_in_turn = old_generated_token_count_in_turn;
                Err(error)
            }
        }
    }

    fn take_last_phase_timings(&mut self) -> Option<RamaSessionPhaseTimings> {
        self.last_phase_timings.take()
    }

    fn take_last_rolling_stats(&mut self) -> Option<RamaRollingStats> {
        self.last_rolling_stats.take()
    }

    fn take_last_experimental_speed_stats(&mut self) -> Option<RamaExperimentalSpeedStats> {
        self.last_experimental_speed_stats.take()
    }
}

#[cfg(test)]
mod tests {

    include!("tests_core.rs");
    include!("tests_speed.rs");
}
