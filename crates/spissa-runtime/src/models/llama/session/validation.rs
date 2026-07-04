// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

fn tensor_shape_usize(model: &LazySpissaModel, name: &str) -> Result<Vec<usize>> {
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
    model: &LazySpissaModel,
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
    model: &LazySpissaModel,
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

fn checked_projection_rows(label: &str, heads: usize, head_dim: usize) -> Result<usize> {
    heads.checked_mul(head_dim).ok_or_else(|| {
        RuntimeError::Shape(format!(
            "llama session {label} projection row count overflow"
        ))
    })
}


fn validate_layer_tensor_shapes(
    model: &LazySpissaModel,
    layer_names: &OwnedLlamaStreamingBlockTensorNames,
    hidden_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
) -> Result<()> {
    let q_rows = checked_projection_rows("q", q_heads, head_dim)?;
    let kv_rows = checked_projection_rows("kv", kv_heads, head_dim)?;

    validate_matrix_shape(model, &layer_names.q_weight, q_rows, hidden_size)?;
    validate_matrix_shape(model, &layer_names.k_weight, kv_rows, hidden_size)?;
    validate_matrix_shape(model, &layer_names.v_weight, kv_rows, hidden_size)?;
    validate_matrix_shape(model, &layer_names.o_weight, hidden_size, q_rows)?;
    validate_matrix_shape(
        model,
        &layer_names.gate_weight,
        intermediate_size,
        hidden_size,
    )?;
    validate_matrix_shape(
        model,
        &layer_names.up_weight,
        intermediate_size,
        hidden_size,
    )?;
    validate_matrix_shape(
        model,
        &layer_names.down_weight,
        hidden_size,
        intermediate_size,
    )?;
    Ok(())
}

