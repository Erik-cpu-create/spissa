// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

// Q8 block-scale / validate / dot helpers + the f32-fallback dot kernels.
// Split out of kernels_q8.rs (R168); include!d into streaming/mod.rs.

fn validate_q8_0_chunk(
    q8_bytes: &[u8],
    element_start: usize,
    weight_elements: usize,
    weight_name: &str,
) -> Result<()> {
    if !q8_bytes.len().is_multiple_of(34) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Q8_0 stream for {weight_name} has byte len {} not aligned to 34-byte blocks",
            q8_bytes.len()
        )));
    }
    if element_start > weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} Q8_0 chunk starts at element {element_start}, beyond expected {weight_elements}"
        )));
    }
    Ok(())
}

fn q8_0_complete_row_span(
    q8_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
) -> Result<Option<(usize, usize, usize)>> {
    if config.batch != 1
        || config.in_features == 0
        || !config.in_features.is_multiple_of(32)
        || !element_start.is_multiple_of(config.in_features)
    {
        return Ok(None);
    }
    let chunk_elements = quantized_elements_for_bytes(rllm_container::DType::Q8_0, q8_bytes.len())?;
    if chunk_elements == 0 || !chunk_elements.is_multiple_of(config.in_features) {
        return Ok(None);
    }
    Ok(Some((
        element_start / config.in_features,
        chunk_elements / config.in_features,
        config.in_features / 32,
    )))
}

fn q8_0_block_scale(q8_bytes: &[u8], block_offset: usize) -> f32 {
    let scale_bits = u16::from_le_bytes([q8_bytes[block_offset], q8_bytes[block_offset + 1]]);
    crate::tensor::fp16_to_f32(scale_bits)
}

fn q8_0_dot_i8_f32(qs: &[u8], input: &[f32], len: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut idx = 0usize;
    while idx + 4 <= len {
        acc += (qs[idx] as i8) as f32 * input[idx]
            + (qs[idx + 1] as i8) as f32 * input[idx + 1]
            + (qs[idx + 2] as i8) as f32 * input[idx + 2]
            + (qs[idx + 3] as i8) as f32 * input[idx + 3];
        idx += 4;
    }
    while idx < len {
        acc += (qs[idx] as i8) as f32 * input[idx];
        idx += 1;
    }
    acc
}

#[allow(dead_code)]
fn q8_0_scaled_block(qs: &[u8], scale: f32) -> [f32; 32] {
    let mut scaled = [0.0f32; 32];
    for idx in 0..32 {
        scaled[idx] = scale * (qs[idx] as i8) as f32;
    }
    scaled
}

fn q8_0_scaled_block_reecast(qs: &[u8], scale: f32) -> [f32; 32] {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return q8_0_scaled_block_neon(qs, scale);
    }

    #[cfg(not(target_arch = "aarch64"))]
    q8_0_scaled_block(qs, scale)
}

#[cfg(target_arch = "aarch64")]
unsafe fn q8_0_scaled_block_neon(qs: &[u8], scale: f32) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    let scale_vec = vdupq_n_f32(scale);
    let mut offset = 0usize;
    while offset < 32 {
        let q_i8 = vld1q_s8(qs.as_ptr().add(offset) as *const i8);
        let low_i16 = vmovl_s8(vget_low_s8(q_i8));
        let high_i16 = vmovl_s8(vget_high_s8(q_i8));

        let low_low_i32 = vmovl_s16(vget_low_s16(low_i16));
        let low_high_i32 = vmovl_s16(vget_high_s16(low_i16));
        let high_low_i32 = vmovl_s16(vget_low_s16(high_i16));
        let high_high_i32 = vmovl_s16(vget_high_s16(high_i16));

        vst1q_f32(
            out.as_mut_ptr().add(offset),
            vmulq_f32(vcvtq_f32_s32(low_low_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 4),
            vmulq_f32(vcvtq_f32_s32(low_high_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 8),
            vmulq_f32(vcvtq_f32_s32(high_low_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 12),
            vmulq_f32(vcvtq_f32_s32(high_high_i32), scale_vec),
        );
        offset += 16;
    }
    out
}

fn f32_dot_32(weights: &[f32; 32], input: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut idx = 0usize;
    while idx < 32 {
        acc0 += weights[idx] * input[idx];
        acc1 += weights[idx + 1] * input[idx + 1];
        acc2 += weights[idx + 2] * input[idx + 2];
        acc3 += weights[idx + 3] * input[idx + 3];
        idx += 4;
    }
    (acc0 + acc1) + (acc2 + acc3)
}

#[allow(dead_code)]
fn accumulate_f32_dot_32_batch4(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    let mut acc0 = output[out_feature];
    let mut acc1 = output[output_stride + out_feature];
    let mut acc2 = output[output_stride * 2 + out_feature];
    let mut acc3 = output[output_stride * 3 + out_feature];
    let mut idx = 0usize;
    while idx < 32 {
        let weight = weights[idx];
        acc0 += weight * input[idx];
        acc1 += weight * input[input_stride + idx];
        acc2 += weight * input[input_stride * 2 + idx];
        acc3 += weight * input[input_stride * 3 + idx];
        idx += 1;
    }
    output[out_feature] = acc0;
    output[output_stride + out_feature] = acc1;
    output[output_stride * 2 + out_feature] = acc2;
    output[output_stride * 3 + out_feature] = acc3;
}

fn accumulate_f32_dot_32_batch4_reevec(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return accumulate_f32_dot_32_batch4_neon(
            weights,
            input,
            input_stride,
            output,
            output_stride,
            out_feature,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_f32_dot_32_batch4(
        weights,
        input,
        input_stride,
        output,
        output_stride,
        out_feature,
    );
}

fn accumulate_f32_dot_32_output2_batch4_reebundle(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return accumulate_f32_dot_32_output2_batch4_neon(
            first,
            second,
            input,
            input_stride,
            output,
            output_stride,
            first_out_feature,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_f32_dot_32_output2_batch4_scalar(
        first,
        second,
        input,
        input_stride,
        output,
        output_stride,
        first_out_feature,
    );
}

#[cfg(not(target_arch = "aarch64"))]
fn accumulate_f32_dot_32_output2_batch4_scalar(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
) {
    let mut first0 = output[first_out_feature];
    let mut first1 = output[output_stride + first_out_feature];
    let mut first2 = output[output_stride * 2 + first_out_feature];
    let mut first3 = output[output_stride * 3 + first_out_feature];
    let second_out_feature = first_out_feature + 1;
    let mut second0 = output[second_out_feature];
    let mut second1 = output[output_stride + second_out_feature];
    let mut second2 = output[output_stride * 2 + second_out_feature];
    let mut second3 = output[output_stride * 3 + second_out_feature];
    let mut idx = 0usize;
    while idx < 32 {
        let x0 = input[idx];
        let x1 = input[input_stride + idx];
        let x2 = input[input_stride * 2 + idx];
        let x3 = input[input_stride * 3 + idx];
        let fw = first[idx];
        let sw = second[idx];
        first0 += fw * x0;
        first1 += fw * x1;
        first2 += fw * x2;
        first3 += fw * x3;
        second0 += sw * x0;
        second1 += sw * x1;
        second2 += sw * x2;
        second3 += sw * x3;
        idx += 1;
    }
    output[first_out_feature] = first0;
    output[output_stride + first_out_feature] = first1;
    output[output_stride * 2 + first_out_feature] = first2;
    output[output_stride * 3 + first_out_feature] = first3;
    output[second_out_feature] = second0;
    output[output_stride + second_out_feature] = second1;
    output[output_stride * 2 + second_out_feature] = second2;
    output[output_stride * 3 + second_out_feature] = second3;
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_f32_dot_32_output2_batch4_neon(
    first: &[f32; 32],
    second: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    first_out_feature: usize,
) {
    let mut first0 = vdupq_n_f32(0.0);
    let mut first1 = vdupq_n_f32(0.0);
    let mut first2 = vdupq_n_f32(0.0);
    let mut first3 = vdupq_n_f32(0.0);
    let mut second0 = vdupq_n_f32(0.0);
    let mut second1 = vdupq_n_f32(0.0);
    let mut second2 = vdupq_n_f32(0.0);
    let mut second3 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let x0 = vld1q_f32(input.as_ptr().add(idx));
        let x1 = vld1q_f32(input.as_ptr().add(input_stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(input_stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(input_stride * 3 + idx));
        let first_weights = vld1q_f32(first.as_ptr().add(idx));
        let second_weights = vld1q_f32(second.as_ptr().add(idx));
        first0 = vfmaq_f32(first0, first_weights, x0);
        first1 = vfmaq_f32(first1, first_weights, x1);
        first2 = vfmaq_f32(first2, first_weights, x2);
        first3 = vfmaq_f32(first3, first_weights, x3);
        second0 = vfmaq_f32(second0, second_weights, x0);
        second1 = vfmaq_f32(second1, second_weights, x1);
        second2 = vfmaq_f32(second2, second_weights, x2);
        second3 = vfmaq_f32(second3, second_weights, x3);
        idx += 4;
    }
    let second_out_feature = first_out_feature + 1;
    output[first_out_feature] += vaddvq_f32(first0);
    output[output_stride + first_out_feature] += vaddvq_f32(first1);
    output[output_stride * 2 + first_out_feature] += vaddvq_f32(first2);
    output[output_stride * 3 + first_out_feature] += vaddvq_f32(first3);
    output[second_out_feature] += vaddvq_f32(second0);
    output[output_stride + second_out_feature] += vaddvq_f32(second1);
    output[output_stride * 2 + second_out_feature] += vaddvq_f32(second2);
    output[output_stride * 3 + second_out_feature] += vaddvq_f32(second3);
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_f32_dot_32_batch4_neon(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    out_feature: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let w = vld1q_f32(weights.as_ptr().add(idx));
        let x0 = vld1q_f32(input.as_ptr().add(idx));
        let x1 = vld1q_f32(input.as_ptr().add(input_stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(input_stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(input_stride * 3 + idx));
        acc0 = vfmaq_f32(acc0, w, x0);
        acc1 = vfmaq_f32(acc1, w, x1);
        acc2 = vfmaq_f32(acc2, w, x2);
        acc3 = vfmaq_f32(acc3, w, x3);
        idx += 4;
    }
    output[out_feature] += vaddvq_f32(acc0);
    output[output_stride + out_feature] += vaddvq_f32(acc1);
    output[output_stride * 2 + out_feature] += vaddvq_f32(acc2);
    output[output_stride * 3 + out_feature] += vaddvq_f32(acc3);
}

#[allow(dead_code)]
fn accumulate_f32_dot_32_batch4_into(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    accumulators: &mut [f32],
    accumulator_start: usize,
) {
    let mut acc0 = accumulators[accumulator_start];
    let mut acc1 = accumulators[accumulator_start + 1];
    let mut acc2 = accumulators[accumulator_start + 2];
    let mut acc3 = accumulators[accumulator_start + 3];
    let mut idx = 0usize;
    while idx < 32 {
        let weight = weights[idx];
        acc0 += weight * input[idx];
        acc1 += weight * input[input_stride + idx];
        acc2 += weight * input[input_stride * 2 + idx];
        acc3 += weight * input[input_stride * 3 + idx];
        idx += 1;
    }
    accumulators[accumulator_start] = acc0;
    accumulators[accumulator_start + 1] = acc1;
    accumulators[accumulator_start + 2] = acc2;
    accumulators[accumulator_start + 3] = acc3;
}

fn accumulate_f32_dot_32_batch4_into_reevec(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    accumulators: &mut [f32],
    accumulator_start: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return accumulate_f32_dot_32_batch4_into_neon(
            weights,
            input,
            input_stride,
            accumulators,
            accumulator_start,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    accumulate_f32_dot_32_batch4_into(
        weights,
        input,
        input_stride,
        accumulators,
        accumulator_start,
    );
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_f32_dot_32_batch4_into_neon(
    weights: &[f32; 32],
    input: &[f32],
    input_stride: usize,
    accumulators: &mut [f32],
    accumulator_start: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let w = vld1q_f32(weights.as_ptr().add(idx));
        let x0 = vld1q_f32(input.as_ptr().add(idx));
        let x1 = vld1q_f32(input.as_ptr().add(input_stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(input_stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(input_stride * 3 + idx));
        acc0 = vfmaq_f32(acc0, w, x0);
        acc1 = vfmaq_f32(acc1, w, x1);
        acc2 = vfmaq_f32(acc2, w, x2);
        acc3 = vfmaq_f32(acc3, w, x3);
        idx += 4;
    }
    accumulators[accumulator_start] += vaddvq_f32(acc0);
    accumulators[accumulator_start + 1] += vaddvq_f32(acc1);
    accumulators[accumulator_start + 2] += vaddvq_f32(acc2);
    accumulators[accumulator_start + 3] += vaddvq_f32(acc3);
}

fn advance_multiply_state_to_row(
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    out_feature: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    while state.current_out_feature < out_feature {
        state.finish_current(config, weight_name)?;
    }
    if state.current_out_feature != out_feature {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
            out_feature, state.current_out_feature
        )));
    }
    Ok(())
}

fn advance_argmax_state_to_row(
    state: &mut StreamingLinearArgmaxState<'_>,
    out_feature: usize,
    config: StreamingLinearConfig,
    weight_name: &str,
) -> Result<()> {
    while state.current_out_feature < out_feature {
        state.finish_current(config, weight_name)?;
    }
    if state.current_out_feature != out_feature {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
            out_feature, state.current_out_feature
        )));
    }
    Ok(())
}

