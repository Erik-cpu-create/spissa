// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

/// Low-RAM attention sub-block over chunked `.rllm` QKV and output weights.
///
/// This implements the non-rotary toy baseline used by Phase 5 tests:
/// `tiled QKV linear -> split Q/K/V -> scaled dot-product attention ->
/// tiled output projection`. Real GPT-NeoX rotary embeddings are layered on
/// through `StreamingAttentionRuntime`.
pub fn streaming_attention_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    qkv_weight_name: &str,
    qkv_bias: Option<&[f32]>,
    out_weight_name: &str,
    out_bias: Option<&[f32]>,
    config: StreamingAttentionConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_attention_with_runtime_from_model(
        model,
        input,
        qkv_weight_name,
        qkv_bias,
        out_weight_name,
        out_bias,
        config,
        StreamingAttentionRuntime::default(),
        budget,
    )
}

pub fn streaming_attention_with_runtime_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    qkv_weight_name: &str,
    qkv_bias: Option<&[f32]>,
    out_weight_name: &str,
    out_bias: Option<&[f32]>,
    config: StreamingAttentionConfig,
    runtime: StreamingAttentionRuntime<'_>,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_attention_with_runtime_and_timing_from_model(
        model,
        input,
        qkv_weight_name,
        qkv_bias,
        out_weight_name,
        out_bias,
        config,
        runtime,
        budget,
        None,
    )
}

fn streaming_attention_with_runtime_and_timing_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    qkv_weight_name: &str,
    qkv_bias: Option<&[f32]>,
    out_weight_name: &str,
    out_bias: Option<&[f32]>,
    config: StreamingAttentionConfig,
    mut runtime: StreamingAttentionRuntime<'_>,
    budget: &mut MemoryBudget,
    mut timing: Option<&mut StreamingBlockTiming>,
) -> Result<Vec<f32>> {
    validate_attention_shapes(input, qkv_bias, out_bias, config)?;
    let hidden_size = config.hidden_size()?;
    let qkv_features = hidden_size
        .checked_mul(3)
        .ok_or_else(|| RuntimeError::Shape("QKV feature count overflow".to_string()))?;

    let qkv_bytes = activation_bytes(config.seq_len, qkv_features, "streaming attention QKV")?;
    let qkv_label = "streaming attention fused QKV activation".to_string();
    budget.reserve(qkv_bytes, qkv_label.clone())?;

    let qkv_projection_started = Instant::now();
    let fused_qkv = match streaming_default_tile_linear_from_model(
        model,
        qkv_weight_name,
        input,
        qkv_bias,
        StreamingLinearConfig {
            batch: config.seq_len,
            in_features: hidden_size,
            out_features: qkv_features,
        },
        budget,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(qkv_bytes, qkv_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_qkv_projection(qkv_projection_started.elapsed());
    }

    let split_label = "streaming attention split QKV activation".to_string();
    if let Err(err) = budget.reserve(qkv_bytes, split_label.clone()) {
        budget.release(qkv_bytes, qkv_label)?;
        return Err(err);
    }
    let qkv_split_started = Instant::now();
    let (mut q, mut k, v) = match split_fused_qkv(&fused_qkv, config) {
        Ok(split) => split,
        Err(err) => {
            budget.release(qkv_bytes, split_label)?;
            budget.release(qkv_bytes, qkv_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_qkv_split(qkv_split_started.elapsed());
    }
    drop(fused_qkv);
    budget.release(qkv_bytes, qkv_label)?;

    if let Some(rotary) = runtime.rotary {
        let rotary_started = Instant::now();
        if let Err(err) = apply_gpt_neox_rotary_inplace(&mut q, &mut k, rotary) {
            budget.release(qkv_bytes, split_label)?;
            return Err(err);
        }
        if let Some(timing) = timing.as_deref_mut() {
            timing.record_attention_rotary(rotary_started.elapsed());
        }
    }

    let cache_is_active = runtime.kv_cache.is_some();

    let attention_bytes = activation_bytes(
        config.seq_len,
        hidden_size,
        "streaming attention output activation",
    )?;
    let attention_label = "streaming attention output activation".to_string();
    if let Err(err) = budget.reserve(attention_bytes, attention_label.clone()) {
        budget.release(qkv_bytes, split_label)?;
        return Err(err);
    }
    let cache_view = runtime.kv_cache.as_deref();
    let score_context_started = Instant::now();
    let attended = match scaled_dot_product_attention_with_cache(
        &q,
        &k,
        &v,
        cache_view,
        KvAttentionConfig {
            query_len: config.seq_len,
            num_heads: config.num_heads,
            kv_heads: config.num_heads,
            head_dim: config.head_dim,
            causal: config.causal,
        },
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(attention_bytes, attention_label)?;
            budget.release(qkv_bytes, split_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_score_context(score_context_started.elapsed());
    }
    if !cache_is_active {
        budget.release(qkv_bytes, split_label.clone())?;
    }

    let output_projection_started = Instant::now();
    let output = match streaming_default_tile_linear_from_model(
        model,
        out_weight_name,
        &attended,
        out_bias,
        StreamingLinearConfig {
            batch: config.seq_len,
            in_features: hidden_size,
            out_features: hidden_size,
        },
        budget,
    ) {
        Ok(output) => output,
        Err(err) => {
            budget.release(attention_bytes, attention_label)?;
            if cache_is_active {
                budget.release(qkv_bytes, split_label)?;
            }
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_output_projection(output_projection_started.elapsed());
    }

    if let Some(cache) = runtime.kv_cache.as_deref_mut() {
        let kv_append_started = Instant::now();
        if let Err(err) = cache.append(&k, &v, config.seq_len) {
            budget.release(attention_bytes, attention_label)?;
            budget.release(qkv_bytes, split_label)?;
            return Err(err);
        }
        if let Some(timing) = timing {
            timing.record_attention_kv_append(kv_append_started.elapsed());
        }
        budget.release(qkv_bytes, split_label)?;
    }

    drop((q, k, v));
    drop(attended);
    budget.release(attention_bytes, attention_label)?;
    Ok(output)
}

