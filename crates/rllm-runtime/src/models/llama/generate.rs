use crate::models::llama::model::{
    OwnedLlamaStreamingBlockParameters, OwnedLlamaStreamingBlockTensorNames,
};
use crate::rotary::{
    apply_llama_rotary_inplace, KvAttentionConfig, KvCache, RotaryEmbeddingConfig,
};
use crate::{
    ops::{add_inplace, rms_norm, silu_inplace},
    scaled_dot_product_attention_with_cache,
    streaming_column_cached_sparse_silu_gate_up_from_model,
    streaming_column_cached_sparse_tile_linear_from_model,
    streaming_input_tiled_sparse_silu_gate_up_from_model,
    streaming_input_tiled_sparse_tile_linear_from_model, streaming_silu_gate_up_from_model,
    streaming_sparse_silu_gate_up_from_model, streaming_sparse_tile_linear_from_model,
    streaming_tile_linear_from_model, streaming_tile_linear_multiply_into_from_model,
    LazyRllmModel, MemoryBudget, RamaAipProjectionKind, RamaExperimentalSpeedConfig,
    RamaExperimentalSpeedStats, RamaTransformerPhaseTimings, Result, SparseColumnCache,
    StreamingLinearConfig, StreamingTileLinearConfig, DEFAULT_STREAMING_TILE_ELEMENTS,
};
use std::time::Instant;

pub struct LlamaStreamingBlockConfig {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub causal: bool,
    pub position_offset: usize,
    pub layer_index: usize,
    pub total_layers: usize,
    pub experimental_speed: RamaExperimentalSpeedConfig,
}

fn optional_input_tiled_sparse_linear(
    model: &mut LazyRllmModel,
    weight_name: &str,
    input: &[f32],
    linear_config: StreamingTileLinearConfig,
    speed_config: RamaExperimentalSpeedConfig,
    projection: RamaAipProjectionKind,
    layer_index: usize,
    total_layers: usize,
    default_topk: usize,
    stats: Option<&mut RamaExperimentalSpeedStats>,
    budget: &mut MemoryBudget,
) -> Result<Option<Vec<f32>>> {
    if !speed_config.enabled || !speed_config.aip_input_tiles {
        return Ok(None);
    }
    let Some(stats) = stats else {
        return Ok(None);
    };
    let decision = speed_config.aip_decision_for_projection(
        layer_index,
        total_layers,
        projection,
        linear_config.linear.in_features,
        default_topk,
    );
    if !decision.enabled {
        return Ok(None);
    }
    stats.record_aip_policy(speed_config.aip_policy);
    let sparse_config = RamaExperimentalSpeedConfig {
        enabled: true,
        aip_policy: speed_config.aip_policy,
        aip_topk: Some(decision.topk),
        aip_attention_topk: None,
        aip_mlp_topk: None,
        aip_down_topk: None,
        aip_edge_layers: None,
        aip_edge_topk: None,
        aip_lm_head_topk: None,
        aip_lm_head_rescore: None,
        aip_lm_head_agreement: false,
        aip_lm_head_rows: None,
        aip_lm_head_repeat_margin_milli: None,
        aip_column_cache: false,
        aip_input_tiles: true,
        aip_no_repeat_last: false,
        aip_repeat_run_limit: None,
    };
    streaming_input_tiled_sparse_tile_linear_from_model(
        model,
        weight_name,
        input,
        None,
        linear_config,
        sparse_config,
        stats,
        budget,
    )
}

pub fn streaming_llama_transformer_block(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: &OwnedLlamaStreamingBlockTensorNames,
    params: &OwnedLlamaStreamingBlockParameters,
    config: LlamaStreamingBlockConfig,
    budget: &mut MemoryBudget,
    cache: Option<&mut KvCache>,
) -> Result<Vec<f32>> {
    streaming_llama_transformer_block_with_timing(
        model, input, names, params, config, budget, cache, None, None, None,
    )
}

pub fn streaming_llama_transformer_block_with_timing(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: &OwnedLlamaStreamingBlockTensorNames,
    params: &OwnedLlamaStreamingBlockParameters,
    config: LlamaStreamingBlockConfig,
    budget: &mut MemoryBudget,
    cache: Option<&mut KvCache>,
    mut timing: Option<&mut RamaTransformerPhaseTimings>,
    mut experimental_speed_stats: Option<&mut RamaExperimentalSpeedStats>,
    mut sparse_column_cache: Option<&mut SparseColumnCache>,
) -> Result<Vec<f32>> {
    if let Some(timing) = timing.as_deref_mut() {
        timing.profiled_layers = timing.profiled_layers.saturating_add(1);
    }
    let mut residual = input.to_vec();

    let started = Instant::now();
    let attention_input = rms_norm(
        input,
        &params.input_layernorm_weight,
        config.seq_len,
        config.hidden_size,
        config.rms_norm_eps,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.attention_norm_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let q_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.hidden_size,
            out_features: config.q_heads * config.head_dim,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let kv_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.hidden_size,
            out_features: config.kv_heads * config.head_dim,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };

    let started = Instant::now();
    let mut q = match optional_input_tiled_sparse_linear(
        model,
        &names.q_weight,
        &attention_input,
        q_config,
        config.experimental_speed,
        RamaAipProjectionKind::Attention,
        config.layer_index,
        config.total_layers,
        128,
        experimental_speed_stats.as_deref_mut(),
        budget,
    )? {
        Some(output) => output,
        None => streaming_tile_linear_from_model(
            model,
            &names.q_weight,
            &attention_input,
            None,
            q_config,
            budget,
        )?,
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.q_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    let mut k = match optional_input_tiled_sparse_linear(
        model,
        &names.k_weight,
        &attention_input,
        kv_config,
        config.experimental_speed,
        RamaAipProjectionKind::Attention,
        config.layer_index,
        config.total_layers,
        128,
        experimental_speed_stats.as_deref_mut(),
        budget,
    )? {
        Some(output) => output,
        None => streaming_tile_linear_from_model(
            model,
            &names.k_weight,
            &attention_input,
            None,
            kv_config,
            budget,
        )?,
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.k_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    let v = match optional_input_tiled_sparse_linear(
        model,
        &names.v_weight,
        &attention_input,
        kv_config,
        config.experimental_speed,
        RamaAipProjectionKind::Attention,
        config.layer_index,
        config.total_layers,
        128,
        experimental_speed_stats.as_deref_mut(),
        budget,
    )? {
        Some(output) => output,
        None => streaming_tile_linear_from_model(
            model,
            &names.v_weight,
            &attention_input,
            None,
            kv_config,
            budget,
        )?,
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.v_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let rope_config = RotaryEmbeddingConfig {
        seq_len: config.seq_len,
        num_heads: config.q_heads,
        head_dim: config.head_dim,
        rotary_dim: config.head_dim,
        base: config.rope_theta,
        position_offset: config.position_offset,
    };
    let started = Instant::now();
    apply_llama_rotary_inplace(&mut q, &mut k, config.q_heads, config.kv_heads, rope_config)?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.rotary_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let attn_config = KvAttentionConfig {
        query_len: config.seq_len,
        num_heads: config.q_heads,
        kv_heads: config.kv_heads,
        head_dim: config.head_dim,
        causal: config.causal,
    };

    let started = Instant::now();
    let attn_out =
        scaled_dot_product_attention_with_cache(&q, &k, &v, cache.as_deref(), attn_config)?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.attention_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    if let Some(c) = cache {
        let started = Instant::now();
        c.append(&k, &v, config.seq_len)?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.kv_append_ms += started.elapsed().as_secs_f64() * 1000.0;
        }
    }

    let o_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.q_heads * config.head_dim,
            out_features: config.hidden_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let started = Instant::now();
    let o = match optional_input_tiled_sparse_linear(
        model,
        &names.o_weight,
        &attn_out,
        o_config,
        config.experimental_speed,
        RamaAipProjectionKind::Attention,
        config.layer_index,
        config.total_layers,
        128,
        experimental_speed_stats.as_deref_mut(),
        budget,
    )? {
        Some(output) => output,
        None => streaming_tile_linear_from_model(
            model,
            &names.o_weight,
            &attn_out,
            None,
            o_config,
            budget,
        )?,
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.o_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    add_inplace(&mut residual, &o)?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.attention_residual_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    let mlp_input = rms_norm(
        &residual,
        &params.post_attention_layernorm_weight,
        config.seq_len,
        config.hidden_size,
        config.rms_norm_eps,
    )?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.mlp_norm_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let mlp_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.hidden_size,
            out_features: config.intermediate_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let started = Instant::now();
    let gate_up_aip_decision = config.experimental_speed.aip_decision_for_projection(
        config.layer_index,
        config.total_layers,
        RamaAipProjectionKind::MlpGateUp,
        config.hidden_size,
        128,
    );
    let sparse_gate_up = if gate_up_aip_decision.enabled {
        if let Some(stats) = &mut experimental_speed_stats {
            stats.record_aip_policy(config.experimental_speed.aip_policy);
            let sparse_config = crate::RamaExperimentalSpeedConfig {
                enabled: true,
                aip_policy: config.experimental_speed.aip_policy,
                aip_topk: Some(gate_up_aip_decision.topk),
                aip_attention_topk: None,
                aip_mlp_topk: None,
                aip_down_topk: None,
                aip_edge_layers: None,
                aip_edge_topk: None,
                aip_lm_head_topk: None,
                aip_lm_head_rescore: None,
                aip_lm_head_agreement: false,
                aip_lm_head_rows: None,
                aip_lm_head_repeat_margin_milli: None,
                aip_column_cache: config.experimental_speed.aip_column_cache,
                aip_input_tiles: config.experimental_speed.aip_input_tiles,
                aip_no_repeat_last: false,
                aip_repeat_run_limit: None,
            };
            let input_tiled = if sparse_config.aip_input_tiles {
                streaming_input_tiled_sparse_silu_gate_up_from_model(
                    model,
                    &names.gate_weight,
                    &names.up_weight,
                    &mlp_input,
                    mlp_config,
                    sparse_config,
                    stats,
                    budget,
                )?
            } else {
                None
            };
            if input_tiled.is_some() {
                input_tiled
            } else if let Some(cache) = sparse_column_cache.as_mut() {
                match streaming_column_cached_sparse_silu_gate_up_from_model(
                    model,
                    &names.gate_weight,
                    &names.up_weight,
                    &mlp_input,
                    mlp_config,
                    sparse_config,
                    stats,
                    cache,
                    budget,
                )? {
                    Some(output) => Some(output),
                    None => streaming_sparse_silu_gate_up_from_model(
                        model,
                        &names.gate_weight,
                        &names.up_weight,
                        &mlp_input,
                        mlp_config,
                        sparse_config,
                        stats,
                        budget,
                    )?,
                }
            } else {
                streaming_sparse_silu_gate_up_from_model(
                    model,
                    &names.gate_weight,
                    &names.up_weight,
                    &mlp_input,
                    mlp_config,
                    sparse_config,
                    stats,
                    budget,
                )?
            }
        } else {
            None
        }
    } else {
        None
    };
    let fused_gate_up = if sparse_gate_up.is_some() {
        sparse_gate_up
    } else {
        streaming_silu_gate_up_from_model(
            model,
            &names.gate_weight,
            &names.up_weight,
            &mlp_input,
            mlp_config,
            budget,
        )?
    };
    let gate = if let Some(fused_gate_up) = fused_gate_up {
        if let Some(timing) = timing.as_deref_mut() {
            timing.gate_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
        }
        fused_gate_up
    } else {
        let mut gate = streaming_tile_linear_from_model(
            model,
            &names.gate_weight,
            &mlp_input,
            None,
            mlp_config,
            budget,
        )?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.gate_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
        }

        let started = Instant::now();
        silu_inplace(&mut gate);
        if let Some(timing) = timing.as_deref_mut() {
            timing.activation_multiply_ms += started.elapsed().as_secs_f64() * 1000.0;
        }

        let started = Instant::now();
        streaming_tile_linear_multiply_into_from_model(
            model,
            &names.up_weight,
            &mlp_input,
            None,
            &mut gate,
            mlp_config,
            budget,
        )?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.up_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
        }
        gate
    };

    let down_config = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: config.seq_len,
            in_features: config.intermediate_size,
            out_features: config.hidden_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    let started = Instant::now();
    let down_aip_decision = config.experimental_speed.aip_decision_for_projection(
        config.layer_index,
        config.total_layers,
        RamaAipProjectionKind::MlpDown,
        config.intermediate_size,
        512,
    );
    let sparse_down = if down_aip_decision.enabled {
        if let Some(stats) = &mut experimental_speed_stats {
            stats.record_aip_policy(config.experimental_speed.aip_policy);
            let sparse_config = crate::RamaExperimentalSpeedConfig {
                enabled: true,
                aip_policy: config.experimental_speed.aip_policy,
                aip_topk: Some(down_aip_decision.topk),
                aip_attention_topk: None,
                aip_mlp_topk: None,
                aip_down_topk: None,
                aip_edge_layers: None,
                aip_edge_topk: None,
                aip_lm_head_topk: None,
                aip_lm_head_rescore: None,
                aip_lm_head_agreement: false,
                aip_lm_head_rows: None,
                aip_lm_head_repeat_margin_milli: None,
                aip_column_cache: config.experimental_speed.aip_column_cache,
                aip_input_tiles: config.experimental_speed.aip_input_tiles,
                aip_no_repeat_last: false,
                aip_repeat_run_limit: None,
            };
            let input_tiled = if sparse_config.aip_input_tiles {
                streaming_input_tiled_sparse_tile_linear_from_model(
                    model,
                    &names.down_weight,
                    &gate,
                    None,
                    down_config,
                    sparse_config,
                    stats,
                    budget,
                )?
            } else {
                None
            };
            if input_tiled.is_some() {
                input_tiled
            } else if let Some(cache) = sparse_column_cache.as_mut() {
                match streaming_column_cached_sparse_tile_linear_from_model(
                    model,
                    &names.down_weight,
                    &gate,
                    None,
                    down_config,
                    sparse_config,
                    stats,
                    cache,
                    budget,
                )? {
                    Some(output) => Some(output),
                    None => streaming_sparse_tile_linear_from_model(
                        model,
                        &names.down_weight,
                        &gate,
                        None,
                        down_config,
                        sparse_config,
                        stats,
                        budget,
                    )?,
                }
            } else {
                streaming_sparse_tile_linear_from_model(
                    model,
                    &names.down_weight,
                    &gate,
                    None,
                    down_config,
                    sparse_config,
                    stats,
                    budget,
                )?
            }
        } else {
            None
        }
    } else {
        None
    };
    let down = if let Some(sparse_down) = sparse_down {
        sparse_down
    } else {
        streaming_tile_linear_from_model(
            model,
            &names.down_weight,
            &gate,
            None,
            down_config,
            budget,
        )?
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.down_projection_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    let started = Instant::now();
    add_inplace(&mut residual, &down)?;
    if let Some(timing) = timing {
        timing.mlp_residual_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    Ok(residual)
}
