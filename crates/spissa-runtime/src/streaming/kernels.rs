// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

// Generic f32 weight-chunk accumulation (shared base). Q8 kernels -> kernels_q8.rs,
// raw 16-bit kernels -> kernels_raw16.rs (R166 split). All include!d into one module.

fn accumulate_weight_chunk_multiply_into(
    input: &[f32],
    weights: &[f32],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<()> {
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    let element_end = element_start
        .checked_add(weights.len())
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;
    while local_idx < weights.len() {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let row_len = (config.in_features - in_feature).min(weights.len() - local_idx);
        let weight_row = &weights[local_idx..local_idx + row_len];
        for batch_idx in 0..config.batch {
            let input_start = batch_idx * config.in_features + in_feature;
            let input_row = &input[input_start..input_start + row_len];
            let mut acc = state.current_acc[batch_idx];
            let mut idx = 0usize;
            while idx + 4 <= row_len {
                acc += weight_row[idx] * input_row[idx]
                    + weight_row[idx + 1] * input_row[idx + 1]
                    + weight_row[idx + 2] * input_row[idx + 2]
                    + weight_row[idx + 3] * input_row[idx + 3];
                idx += 4;
            }
            while idx < row_len {
                acc += weight_row[idx] * input_row[idx];
                idx += 1;
            }
            state.current_acc[batch_idx] = acc;
        }

        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }
    Ok(())
}

fn accumulate_weight_chunk(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    element_start: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    let weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;

    let element_end = element_start
        .checked_add(weights.len())
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;
    while local_idx < weights.len() {
        let out_feature = global_idx / config.in_features;
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weights.len() - local_idx);
        let weight_row = &weights[local_idx..local_idx + row_len];

        let mut batch_idx = 0usize;
        while batch_idx + 8 <= config.batch {
            let output_idx0 = batch_idx * config.out_features + out_feature;
            let output_idx1 = (batch_idx + 1) * config.out_features + out_feature;
            let output_idx2 = (batch_idx + 2) * config.out_features + out_feature;
            let output_idx3 = (batch_idx + 3) * config.out_features + out_feature;
            let output_idx4 = (batch_idx + 4) * config.out_features + out_feature;
            let output_idx5 = (batch_idx + 5) * config.out_features + out_feature;
            let output_idx6 = (batch_idx + 6) * config.out_features + out_feature;
            let output_idx7 = (batch_idx + 7) * config.out_features + out_feature;
            let mut acc0 = output[output_idx0];
            let mut acc1 = output[output_idx1];
            let mut acc2 = output[output_idx2];
            let mut acc3 = output[output_idx3];
            let mut acc4 = output[output_idx4];
            let mut acc5 = output[output_idx5];
            let mut acc6 = output[output_idx6];
            let mut acc7 = output[output_idx7];
            let input_start0 = batch_idx * config.in_features + in_feature;
            let input_start1 = (batch_idx + 1) * config.in_features + in_feature;
            let input_start2 = (batch_idx + 2) * config.in_features + in_feature;
            let input_start3 = (batch_idx + 3) * config.in_features + in_feature;
            let input_start4 = (batch_idx + 4) * config.in_features + in_feature;
            let input_start5 = (batch_idx + 5) * config.in_features + in_feature;
            let input_start6 = (batch_idx + 6) * config.in_features + in_feature;
            let input_start7 = (batch_idx + 7) * config.in_features + in_feature;
            let mut idx = 0;
            while idx + 4 <= row_len {
                let w = &weight_row[idx..idx + 4];
                let i0 = &input[input_start0 + idx..input_start0 + idx + 4];
                let i1 = &input[input_start1 + idx..input_start1 + idx + 4];
                let i2 = &input[input_start2 + idx..input_start2 + idx + 4];
                let i3 = &input[input_start3 + idx..input_start3 + idx + 4];
                let i4 = &input[input_start4 + idx..input_start4 + idx + 4];
                let i5 = &input[input_start5 + idx..input_start5 + idx + 4];
                let i6 = &input[input_start6 + idx..input_start6 + idx + 4];
                let i7 = &input[input_start7 + idx..input_start7 + idx + 4];

                acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                acc4 += w[0] * i4[0] + w[1] * i4[1] + w[2] * i4[2] + w[3] * i4[3];
                acc5 += w[0] * i5[0] + w[1] * i5[1] + w[2] * i5[2] + w[3] * i5[3];
                acc6 += w[0] * i6[0] + w[1] * i6[1] + w[2] * i6[2] + w[3] * i6[3];
                acc7 += w[0] * i7[0] + w[1] * i7[1] + w[2] * i7[2] + w[3] * i7[3];

                idx += 4;
            }
            while idx < row_len {
                let weight = weight_row[idx];
                acc0 += input[input_start0 + idx] * weight;
                acc1 += input[input_start1 + idx] * weight;
                acc2 += input[input_start2 + idx] * weight;
                acc3 += input[input_start3 + idx] * weight;
                acc4 += input[input_start4 + idx] * weight;
                acc5 += input[input_start5 + idx] * weight;
                acc6 += input[input_start6 + idx] * weight;
                acc7 += input[input_start7 + idx] * weight;
                idx += 1;
            }
            output[output_idx0] = acc0;
            output[output_idx1] = acc1;
            output[output_idx2] = acc2;
            output[output_idx3] = acc3;
            output[output_idx4] = acc4;
            output[output_idx5] = acc5;
            output[output_idx6] = acc6;
            output[output_idx7] = acc7;
            batch_idx += 8;
        }
        while batch_idx + 4 <= config.batch {
            let output_idx0 = batch_idx * config.out_features + out_feature;
            let output_idx1 = (batch_idx + 1) * config.out_features + out_feature;
            let output_idx2 = (batch_idx + 2) * config.out_features + out_feature;
            let output_idx3 = (batch_idx + 3) * config.out_features + out_feature;
            let mut acc0 = output[output_idx0];
            let mut acc1 = output[output_idx1];
            let mut acc2 = output[output_idx2];
            let mut acc3 = output[output_idx3];
            let input_start0 = batch_idx * config.in_features + in_feature;
            let input_start1 = (batch_idx + 1) * config.in_features + in_feature;
            let input_start2 = (batch_idx + 2) * config.in_features + in_feature;
            let input_start3 = (batch_idx + 3) * config.in_features + in_feature;
            let mut idx = 0;
            while idx + 4 <= row_len {
                let w = &weight_row[idx..idx + 4];
                let i0 = &input[input_start0 + idx..input_start0 + idx + 4];
                let i1 = &input[input_start1 + idx..input_start1 + idx + 4];
                let i2 = &input[input_start2 + idx..input_start2 + idx + 4];
                let i3 = &input[input_start3 + idx..input_start3 + idx + 4];

                acc0 += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                acc1 += w[0] * i1[0] + w[1] * i1[1] + w[2] * i1[2] + w[3] * i1[3];
                acc2 += w[0] * i2[0] + w[1] * i2[1] + w[2] * i2[2] + w[3] * i2[3];
                acc3 += w[0] * i3[0] + w[1] * i3[1] + w[2] * i3[2] + w[3] * i3[3];
                idx += 4;
            }
            while idx < row_len {
                let weight = weight_row[idx];
                acc0 += input[input_start0 + idx] * weight;
                acc1 += input[input_start1 + idx] * weight;
                acc2 += input[input_start2 + idx] * weight;
                acc3 += input[input_start3 + idx] * weight;
                idx += 1;
            }
            output[output_idx0] = acc0;
            output[output_idx1] = acc1;
            output[output_idx2] = acc2;
            output[output_idx3] = acc3;
            batch_idx += 4;
        }
        while batch_idx < config.batch {
            let input_start = batch_idx * config.in_features + in_feature;
            let input_row = &input[input_start..input_start + row_len];
            let output_idx = batch_idx * config.out_features + out_feature;
            let mut acc = output[output_idx];
            let mut idx = 0;
            while idx + 4 <= row_len {
                let w = &weight_row[idx..idx + 4];
                let i0 = &input_row[idx..idx + 4];
                acc += w[0] * i0[0] + w[1] * i0[1] + w[2] * i0[2] + w[3] * i0[3];
                idx += 4;
            }
            while idx < row_len {
                acc += input_row[idx] * weight_row[idx];
                idx += 1;
            }
            output[output_idx] = acc;
            batch_idx += 1;
        }

        local_idx += row_len;
        global_idx += row_len;
    }
    Ok(())
}

const Q8_ACTIVATION_ENV: &str = "SPISSA_Q8_ACTIVATION";

