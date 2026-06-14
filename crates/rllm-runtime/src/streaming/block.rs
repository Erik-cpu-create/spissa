/// Low-RAM pre-norm transformer block skeleton over chunked `.rllm` weights.
///
/// Computes the Phase 5 toy GPT-NeoX/Pythia-shaped block:
/// `LN -> streaming attention -> residual -> LN -> streaming MLP -> residual`.
/// Rotary embeddings, KV-cache reuse, and tokenizer/generation wiring are
/// intentionally out of scope for this primitive.
pub fn streaming_transformer_block_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: StreamingBlockTensorNames<'_>,
    params: StreamingBlockParameters<'_>,
    config: StreamingBlockConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_transformer_block_with_runtime_from_model(
        model,
        input,
        names,
        params,
        config,
        StreamingBlockRuntime::default(),
        budget,
    )
}

pub fn streaming_transformer_block_with_runtime_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: StreamingBlockTensorNames<'_>,
    params: StreamingBlockParameters<'_>,
    config: StreamingBlockConfig,
    runtime: StreamingBlockRuntime<'_>,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_transformer_block_with_runtime_and_timing_from_model(
        model, input, names, params, config, runtime, budget, None,
    )
}

pub fn streaming_transformer_block_with_runtime_and_timing_from_model(
    model: &mut LazyRllmModel,
    input: &[f32],
    names: StreamingBlockTensorNames<'_>,
    params: StreamingBlockParameters<'_>,
    config: StreamingBlockConfig,
    runtime: StreamingBlockRuntime<'_>,
    budget: &mut MemoryBudget,
    mut timing: Option<&mut StreamingBlockTiming>,
) -> Result<Vec<f32>> {
    validate_block_shapes(input, params, config)?;
    let hidden_size = config.hidden_size()?;
    let hidden_bytes = activation_bytes(
        config.seq_len,
        hidden_size,
        "streaming block hidden activation",
    )?;

    let mut residual = input.to_vec();

    let attention_input_label = "streaming block input layernorm activation".to_string();
    budget.reserve(hidden_bytes, attention_input_label.clone())?;
    let attention_norm_started = Instant::now();
    let attention_input = match crate::ops::layer_norm(
        input,
        params.input_layernorm_weight,
        params.input_layernorm_bias,
        config.seq_len,
        hidden_size,
        config.layer_norm_eps,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, attention_input_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_norm(attention_norm_started.elapsed());
    }

    let attention_output_label = "streaming block attention output activation".to_string();
    if let Err(err) = budget.reserve(hidden_bytes, attention_output_label.clone()) {
        budget.release(hidden_bytes, attention_input_label)?;
        return Err(err);
    }
    let attention_started = Instant::now();
    let attention_output = match streaming_attention_with_runtime_and_timing_from_model(
        model,
        &attention_input,
        names.qkv_weight,
        params.qkv_bias,
        names.attention_out_weight,
        params.attention_out_bias,
        config.attention_config(),
        runtime.attention,
        budget,
        timing.as_deref_mut(),
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, attention_output_label)?;
            budget.release(hidden_bytes, attention_input_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention(attention_started.elapsed());
    }
    let attention_residual_started = Instant::now();
    drop(attention_input);
    budget.release(hidden_bytes, attention_input_label)?;
    if let Err(err) = crate::ops::add_inplace(&mut residual, &attention_output) {
        budget.release(hidden_bytes, attention_output_label)?;
        return Err(err);
    }
    drop(attention_output);
    budget.release(hidden_bytes, attention_output_label)?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_attention_residual(attention_residual_started.elapsed());
    }

    let mlp_input_label = "streaming block post-attention layernorm activation".to_string();
    budget.reserve(hidden_bytes, mlp_input_label.clone())?;
    let mlp_input_source = if runtime.parallel_residual {
        input
    } else {
        residual.as_slice()
    };
    let mlp_norm_started = Instant::now();
    let mlp_input = match crate::ops::layer_norm(
        mlp_input_source,
        params.post_attention_layernorm_weight,
        params.post_attention_layernorm_bias,
        config.seq_len,
        hidden_size,
        config.layer_norm_eps,
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, mlp_input_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp_norm(mlp_norm_started.elapsed());
    }

    let mlp_output_label = "streaming block MLP output activation".to_string();
    if let Err(err) = budget.reserve(hidden_bytes, mlp_output_label.clone()) {
        budget.release(hidden_bytes, mlp_input_label)?;
        return Err(err);
    }
    let mlp_started = Instant::now();
    let mlp_output = match streaming_mlp_with_timing_from_model(
        model,
        &mlp_input,
        names.mlp_in_weight,
        params.mlp_in_bias,
        names.mlp_out_weight,
        params.mlp_out_bias,
        config.mlp_config()?,
        budget,
        timing.as_deref_mut(),
    ) {
        Ok(values) => values,
        Err(err) => {
            budget.release(hidden_bytes, mlp_output_label)?;
            budget.release(hidden_bytes, mlp_input_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp(mlp_started.elapsed());
    }
    let mlp_residual_started = Instant::now();
    drop(mlp_input);
    budget.release(hidden_bytes, mlp_input_label)?;
    if let Err(err) = crate::ops::add_inplace(&mut residual, &mlp_output) {
        budget.release(hidden_bytes, mlp_output_label)?;
        return Err(err);
    }
    drop(mlp_output);
    budget.release(hidden_bytes, mlp_output_label)?;
    if let Some(timing) = timing {
        timing.record_mlp_residual(mlp_residual_started.elapsed());
    }

    Ok(residual)
}

