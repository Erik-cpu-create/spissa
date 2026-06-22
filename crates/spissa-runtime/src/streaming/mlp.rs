// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

fn streaming_default_tile_linear_from_model(
    model: &mut LazySpissaModel,
    weight_name: &str,
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingLinearConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_tile_linear_from_model(
        model,
        weight_name,
        input,
        bias,
        StreamingTileLinearConfig {
            linear: config,
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        },
        budget,
    )
}

/// Low-RAM two-layer MLP block over chunked `.spsa` weight tensors.
///
/// Computes `Linear(input, w_in, b_in) -> GELU -> Linear(hidden, w_out, b_out)`.
/// The intermediate activation is reserved in `budget` for the duration of the
/// second linear pass, while each weight chunk is still decoded/released one at
/// a time through the default Phase 7 tiled linear path.
pub fn streaming_mlp_from_model(
    model: &mut LazySpissaModel,
    input: &[f32],
    w_in_name: &str,
    b_in: Option<&[f32]>,
    w_out_name: &str,
    b_out: Option<&[f32]>,
    config: StreamingMlpConfig,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    streaming_mlp_with_timing_from_model(
        model, input, w_in_name, b_in, w_out_name, b_out, config, budget, None,
    )
}

fn streaming_mlp_with_timing_from_model(
    model: &mut LazySpissaModel,
    input: &[f32],
    w_in_name: &str,
    b_in: Option<&[f32]>,
    w_out_name: &str,
    b_out: Option<&[f32]>,
    config: StreamingMlpConfig,
    budget: &mut MemoryBudget,
    mut timing: Option<&mut StreamingBlockTiming>,
) -> Result<Vec<f32>> {
    validate_mlp_shapes(input, b_in, b_out, config)?;

    let intermediate_bytes = activation_bytes(
        config.batch,
        config.intermediate_size,
        "streaming MLP intermediate",
    )?;
    let intermediate_label = "streaming MLP intermediate activation".to_string();
    budget.reserve(intermediate_bytes, intermediate_label.clone())?;

    let input_projection_started = Instant::now();
    let mut hidden = match streaming_default_tile_linear_from_model(
        model,
        w_in_name,
        input,
        b_in,
        StreamingLinearConfig {
            batch: config.batch,
            in_features: config.hidden_size,
            out_features: config.intermediate_size,
        },
        budget,
    ) {
        Ok(hidden) => hidden,
        Err(err) => {
            budget.release(intermediate_bytes, intermediate_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp_input_projection(input_projection_started.elapsed());
    }

    let activation_started = Instant::now();
    crate::ops::gelu_inplace(&mut hidden);
    if let Some(timing) = timing.as_deref_mut() {
        timing.record_mlp_activation(activation_started.elapsed());
    }

    let output_projection_started = Instant::now();
    let output = match streaming_default_tile_linear_from_model(
        model,
        w_out_name,
        &hidden,
        b_out,
        StreamingLinearConfig {
            batch: config.batch,
            in_features: config.intermediate_size,
            out_features: config.hidden_size,
        },
        budget,
    ) {
        Ok(output) => output,
        Err(err) => {
            budget.release(intermediate_bytes, intermediate_label)?;
            return Err(err);
        }
    };
    if let Some(timing) = timing {
        timing.record_mlp_output_projection(output_projection_started.elapsed());
    }

    drop(hidden);
    budget.release(intermediate_bytes, intermediate_label)?;
    Ok(output)
}

