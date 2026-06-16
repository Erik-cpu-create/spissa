fn validate_block_shapes(
    input: &[f32],
    params: StreamingBlockParameters<'_>,
    config: StreamingBlockConfig,
) -> Result<()> {
    let hidden_size = config.hidden_size()?;
    let input_len = config
        .seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| RuntimeError::Shape("block input length overflow".to_string()))?;
    if input.len() != input_len {
        return Err(RuntimeError::Shape(format!(
            "block input len {} does not match seq_len*hidden_size = {}",
            input.len(),
            input_len
        )));
    }
    validate_norm_params(
        "input layernorm",
        params.input_layernorm_weight,
        params.input_layernorm_bias,
        hidden_size,
    )?;
    validate_norm_params(
        "post-attention layernorm",
        params.post_attention_layernorm_weight,
        params.post_attention_layernorm_bias,
        hidden_size,
    )?;

    let qkv_features = hidden_size
        .checked_mul(3)
        .ok_or_else(|| RuntimeError::Shape("block QKV feature overflow".to_string()))?;
    if let Some(bias) = params.qkv_bias {
        if bias.len() != qkv_features {
            return Err(RuntimeError::Shape(format!(
                "block QKV bias len {} does not match 3*hidden_size {}",
                bias.len(),
                qkv_features
            )));
        }
    }
    if let Some(bias) = params.attention_out_bias {
        if bias.len() != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "block attention output bias len {} does not match hidden_size {}",
                bias.len(),
                hidden_size
            )));
        }
    }
    if let Some(bias) = params.mlp_in_bias {
        if bias.len() != config.intermediate_size {
            return Err(RuntimeError::Shape(format!(
                "block MLP input bias len {} does not match intermediate_size {}",
                bias.len(),
                config.intermediate_size
            )));
        }
    }
    if let Some(bias) = params.mlp_out_bias {
        if bias.len() != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "block MLP output bias len {} does not match hidden_size {}",
                bias.len(),
                hidden_size
            )));
        }
    }
    if !config.layer_norm_eps.is_finite() || config.layer_norm_eps < 0.0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "layer_norm_eps must be finite and non-negative, got {}",
            config.layer_norm_eps
        )));
    }
    Ok(())
}

fn validate_norm_params(
    name: &str,
    weight: &[f32],
    bias: &[f32],
    hidden_size: usize,
) -> Result<()> {
    if weight.len() != hidden_size || bias.len() != hidden_size {
        return Err(RuntimeError::Shape(format!(
            "{name} params must match hidden_size {hidden_size}: weight={}, bias={}",
            weight.len(),
            bias.len()
        )));
    }
    Ok(())
}

fn validate_attention_shapes(
    input: &[f32],
    qkv_bias: Option<&[f32]>,
    out_bias: Option<&[f32]>,
    config: StreamingAttentionConfig,
) -> Result<()> {
    let hidden_size = config.hidden_size()?;
    let input_len = config
        .seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| RuntimeError::Shape("attention input length overflow".to_string()))?;
    if input.len() != input_len {
        return Err(RuntimeError::Shape(format!(
            "attention input len {} does not match seq_len*hidden_size = {}",
            input.len(),
            input_len
        )));
    }
    let qkv_features = hidden_size
        .checked_mul(3)
        .ok_or_else(|| RuntimeError::Shape("QKV feature count overflow".to_string()))?;
    if let Some(bias) = qkv_bias {
        if bias.len() != qkv_features {
            return Err(RuntimeError::Shape(format!(
                "QKV bias len {} does not match 3*hidden_size {}",
                bias.len(),
                qkv_features
            )));
        }
    }
    if let Some(bias) = out_bias {
        if bias.len() != hidden_size {
            return Err(RuntimeError::Shape(format!(
                "attention output bias len {} does not match hidden_size {}",
                bias.len(),
                hidden_size
            )));
        }
    }
    Ok(())
}

fn split_fused_qkv(
    fused: &[f32],
    config: StreamingAttentionConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let hidden_size = config.hidden_size()?;
    let expected = config
        .seq_len
        .checked_mul(hidden_size)
        .and_then(|values| values.checked_mul(3))
        .ok_or_else(|| RuntimeError::Shape("fused QKV length overflow".to_string()))?;
    if fused.len() != expected {
        return Err(RuntimeError::Shape(format!(
            "fused QKV len {} does not match seq_len*3*hidden_size = {}",
            fused.len(),
            expected
        )));
    }

    let values_per_stream = config
        .seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| RuntimeError::Shape("QKV split length overflow".to_string()))?;
    let mut q = vec![0.0f32; values_per_stream];
    let mut k = vec![0.0f32; values_per_stream];
    let mut v = vec![0.0f32; values_per_stream];
    for pos in 0..config.seq_len {
        let fused_row = pos * hidden_size * 3;
        let out_row = pos * hidden_size;
        for head in 0..config.num_heads {
            let fused_head = fused_row + head * config.head_dim * 3;
            let out_head = out_row + head * config.head_dim;
            q[out_head..out_head + config.head_dim]
                .copy_from_slice(&fused[fused_head..fused_head + config.head_dim]);
            k[out_head..out_head + config.head_dim].copy_from_slice(
                &fused[fused_head + config.head_dim..fused_head + 2 * config.head_dim],
            );
            v[out_head..out_head + config.head_dim].copy_from_slice(
                &fused[fused_head + 2 * config.head_dim..fused_head + 3 * config.head_dim],
            );
        }
    }
    Ok((q, k, v))
}

fn validate_mlp_shapes(
    input: &[f32],
    b_in: Option<&[f32]>,
    b_out: Option<&[f32]>,
    config: StreamingMlpConfig,
) -> Result<()> {
    if input.len() != config.batch * config.hidden_size {
        return Err(RuntimeError::Shape(format!(
            "MLP input len {} does not match batch*hidden_size = {}",
            input.len(),
            config.batch * config.hidden_size
        )));
    }
    if let Some(bias) = b_in {
        if bias.len() != config.intermediate_size {
            return Err(RuntimeError::Shape(format!(
                "MLP input bias len {} does not match intermediate_size {}",
                bias.len(),
                config.intermediate_size
            )));
        }
    }
    if let Some(bias) = b_out {
        if bias.len() != config.hidden_size {
            return Err(RuntimeError::Shape(format!(
                "MLP output bias len {} does not match hidden_size {}",
                bias.len(),
                config.hidden_size
            )));
        }
    }
    Ok(())
}

fn activation_bytes(batch: usize, features: usize, label: &str) -> Result<usize> {
    batch
        .checked_mul(features)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| RuntimeError::Shape(format!("{label} byte size overflow")))
}

fn validate_linear_shapes(
    input: &[f32],
    bias: Option<&[f32]>,
    config: StreamingLinearConfig,
) -> Result<()> {
    if input.len() != config.batch * config.in_features {
        return Err(RuntimeError::Shape(format!(
            "input len {} does not match batch*in_features = {}",
            input.len(),
            config.batch * config.in_features
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != config.out_features {
            return Err(RuntimeError::Shape(format!(
                "bias len {} does not match out_features {}",
                bias.len(),
                config.out_features
            )));
        }
    }
    Ok(())
}

fn validate_tile_linear_config(config: StreamingTileLinearConfig) -> Result<()> {
    if config.tile_elements == 0 {
        return Err(RuntimeError::Shape(
            "tile_elements must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn validate_weight_tensor(tensor: &TensorMeta, config: StreamingLinearConfig) -> Result<()> {
    if tensor.shape.len() != 2 {
        return Err(RuntimeError::Shape(format!(
            "weight tensor {} must be rank-2 [out,in], got {:?}",
            tensor.name, tensor.shape
        )));
    }
    let out = usize::try_from(tensor.shape[0])
        .map_err(|_| RuntimeError::Shape("weight out_features overflows usize".to_string()))?;
    let input = usize::try_from(tensor.shape[1])
        .map_err(|_| RuntimeError::Shape("weight in_features overflows usize".to_string()))?;
    if out != config.out_features || input != config.in_features {
        return Err(RuntimeError::Shape(format!(
            "weight tensor {} shape {:?} does not match requested [{}, {}]",
            tensor.name, tensor.shape, config.out_features, config.in_features
        )));
    }

    let expected_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    let expected_bytes = tensor.dtype.byte_size_for_elements(expected_elements);
    if tensor.original_size_bytes != expected_bytes as u64 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {} original_size_bytes={} does not match shape/dtype bytes {}",
            tensor.name, tensor.original_size_bytes, expected_bytes
        )));
    }
    Ok(())
}
