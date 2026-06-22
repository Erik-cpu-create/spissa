// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! Dequantization and quantization kernels for block-quantized types (e.g. Q4_0).

use crate::Result;
use spissa_container::DType;

/// Quantize a slice of floats (represented as raw bytes of dtype) to Q4_0 block bytes.
pub fn quantize_to_q4_0(raw_data: &[u8], dtype: DType, shape: &[u64]) -> Result<Vec<u8>> {
    let f32_data = raw_to_f32_data(raw_data, dtype, shape, "Q4_0")?;
    quantize_f32_to_q4_0(&f32_data)
}

/// Quantize a slice of floats (represented as raw bytes of dtype) to Q8_0 block bytes.
pub fn quantize_to_q8_0(raw_data: &[u8], dtype: DType, shape: &[u64]) -> Result<Vec<u8>> {
    let f32_data = raw_to_f32_data(raw_data, dtype, shape, "Q8_0")?;
    quantize_f32_to_q8_0(&f32_data)
}

fn raw_to_f32_data(raw_data: &[u8], dtype: DType, shape: &[u64], label: &str) -> Result<Vec<f32>> {
    let elements = shape.iter().product::<u64>() as usize;
    let element_size = dtype.size_bytes();
    let expected_bytes = elements.checked_mul(element_size).ok_or_else(|| {
        crate::RuntimeError::Shape("quantization source byte size overflow".to_string())
    })?;
    if raw_data.len() != expected_bytes {
        return Err(crate::RuntimeError::InvalidTensorData(format!(
            "{label} quantization expected byte length {expected_bytes} for {:?} shape {:?}, got {}",
            dtype,
            shape,
            raw_data.len()
        )));
    }

    let mut f32_data = vec![0.0f32; elements];

    for i in 0..elements {
        let offset = i * element_size;
        let val = match dtype {
            DType::Fp16 => {
                let bits = u16::from_le_bytes([raw_data[offset], raw_data[offset + 1]]);
                crate::tensor::fp16_to_f32(bits)
            }
            DType::Bf16 => {
                let bits = u16::from_le_bytes([raw_data[offset], raw_data[offset + 1]]);
                crate::tensor::bf16_to_f32(bits)
            }
            DType::Fp32 => f32::from_le_bytes([
                raw_data[offset],
                raw_data[offset + 1],
                raw_data[offset + 2],
                raw_data[offset + 3],
            ]),
            _ => {
                return Err(crate::RuntimeError::InvalidTensorData(format!(
                    "Unsupported dtype for quantization: {:?}",
                    dtype
                )))
            }
        };
        f32_data[i] = val;
    }

    Ok(f32_data)
}

fn quantize_f32_to_q4_0(f32_data: &[f32]) -> Result<Vec<u8>> {
    let elements = f32_data.len();
    let num_blocks = (elements + 31) / 32;
    let mut quantized_data = vec![0u8; num_blocks * 18];

    for b in 0..num_blocks {
        let block_start = b * 32;
        let block_len = (elements - block_start).min(32);
        let block_slice = &f32_data[block_start..block_start + block_len];

        let mut max_val = 0.0f32;
        for &x in block_slice {
            let abs_x = x.abs();
            if abs_x > max_val {
                max_val = abs_x;
            }
        }

        let scale = max_val / 8.0;
        let scale_bits = if scale == 0.0 {
            0u16
        } else {
            crate::tensor::f32_to_fp16(scale)
        };
        let scale_f32 = crate::tensor::fp16_to_f32(scale_bits);

        let out_offset = b * 18;
        quantized_data[out_offset..out_offset + 2].copy_from_slice(&scale_bits.to_le_bytes());

        let mut block_qs = [0i8; 32];
        for i in 0..block_len {
            let x = block_slice[i];
            let q = if scale_f32 == 0.0 {
                0
            } else {
                let qi = (x / scale_f32).round() as i32;
                qi.clamp(-8, 7) as i8
            };
            block_qs[i] = q;
        }

        for i in 0..16 {
            let q0 = (block_qs[i * 2] + 8) as u8;
            let q1 = (block_qs[i * 2 + 1] + 8) as u8;
            quantized_data[out_offset + 2 + i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }

    Ok(quantized_data)
}

fn quantize_f32_to_q8_0(f32_data: &[f32]) -> Result<Vec<u8>> {
    let elements = f32_data.len();
    let num_blocks = (elements + 31) / 32;
    let mut quantized_data = vec![0u8; num_blocks * 34];

    for b in 0..num_blocks {
        let block_start = b * 32;
        let block_len = (elements - block_start).min(32);
        let block_slice = &f32_data[block_start..block_start + block_len];

        let mut max_val = 0.0f32;
        for &x in block_slice {
            max_val = max_val.max(x.abs());
        }

        let scale = max_val / 127.0;
        let scale_bits = if scale == 0.0 {
            0u16
        } else {
            crate::tensor::f32_to_fp16(scale)
        };
        let scale_f32 = crate::tensor::fp16_to_f32(scale_bits);

        let out_offset = b * 34;
        quantized_data[out_offset..out_offset + 2].copy_from_slice(&scale_bits.to_le_bytes());

        for i in 0..block_len {
            let q = if scale_f32 == 0.0 {
                0
            } else {
                (block_slice[i] / scale_f32).round().clamp(-127.0, 127.0) as i8
            };
            quantized_data[out_offset + 2 + i] = q as u8;
        }
    }

    Ok(quantized_data)
}

/// Dequantize a Q4_0 byte slice into an f32 slice.
///
/// Input is a Q4_0 slice where each block of 32 elements takes 18 bytes:
/// - scale: fp16 (2 bytes)
/// - qs: 32 4-bit elements packed into 16 bytes.
pub fn dequantize_q4_0(input: &[u8], output: &mut [f32]) {
    let num_blocks = input.len() / 18;
    let limit = output.len().min(num_blocks * 32);

    for b in 0..num_blocks {
        let offset = b * 18;
        if offset + 18 > input.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([input[offset], input[offset + 1]]);
        let scale = crate::tensor::fp16_to_f32(scale_bits);

        let block_out_start = b * 32;
        for i in 0..16 {
            let out_idx = block_out_start + i * 2;
            if out_idx >= limit {
                break;
            }

            let byte = input[offset + 2 + i];
            let q0 = ((byte & 0x0F) as i8 - 8) as f32;
            let q1 = (((byte >> 4) & 0x0F) as i8 - 8) as f32;

            output[out_idx] = scale * q0;
            if out_idx + 1 < limit {
                output[out_idx + 1] = scale * q1;
            }
        }
    }
}

/// Dequantize a Q8_0 byte slice into an f32 slice.
///
/// Input is a Q8_0 slice where each block of 32 elements takes 34 bytes:
/// - scale: fp16 (2 bytes)
/// - qs: 32 signed int8 elements.
pub fn dequantize_q8_0(input: &[u8], output: &mut [f32]) {
    let num_blocks = input.len() / 34;
    let limit = output.len().min(num_blocks * 32);

    for b in 0..num_blocks {
        let offset = b * 34;
        if offset + 34 > input.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([input[offset], input[offset + 1]]);
        let scale = crate::tensor::fp16_to_f32(scale_bits);

        let block_out_start = b * 32;
        for i in 0..32 {
            let out_idx = block_out_start + i;
            if out_idx >= limit {
                break;
            }
            let q = input[offset + 2 + i] as i8;
            output[out_idx] = scale * q as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_quantize_dequantize_roundtrip() {
        // Generate a test float array representing FP32 values
        let original_floats: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.125).collect();

        let mut raw_bytes = Vec::new();
        for &f in &original_floats {
            raw_bytes.extend_from_slice(&f.to_le_bytes());
        }

        // Quantize
        let quantized =
            quantize_to_q4_0(&raw_data_helper(&original_floats), DType::Fp32, &[64]).unwrap();
        assert_eq!(quantized.len(), 36); // 2 blocks of 32 = 2 * 18 = 36 bytes

        // Dequantize
        let mut dequantized = vec![0.0f32; 64];
        dequantize_q4_0(&quantized, &mut dequantized);

        // Check accuracy (reconstruction error should be small)
        for i in 0..64 {
            let diff = (dequantized[i] - original_floats[i]).abs();
            // Q4_0 uses the asymmetric signed range [-8, 7]. With scale=max/8,
            // the negative endpoint is exact while a positive block maximum can
            // saturate by up to one scale step before normal rounding error.
            let block = i / 32;
            let block_max = if block == 0 { 4.0f32 } else { 4.0f32 }; // max values in blocks
            let expected_scale = block_max / 8.0;
            assert!(
                diff <= expected_scale + 1e-5,
                "At index {}, diff={}, expected_scale={}",
                i,
                diff,
                expected_scale
            );
        }
    }

    #[test]
    fn q4_0_quantize_uses_max_abs_over_eight_scale() {
        let mut original_floats = vec![0.0f32; 32];
        original_floats[0] = -8.0;
        original_floats[1] = 7.0;

        let quantized =
            quantize_to_q4_0(&raw_data_helper(&original_floats), DType::Fp32, &[32]).unwrap();

        assert_eq!(
            u16::from_le_bytes([quantized[0], quantized[1]]),
            crate::tensor::f32_to_fp16(1.0)
        );
        assert_eq!(quantized[2], 0xf0);
    }

    #[test]
    fn q4_0_quantize_rejects_source_byte_len_mismatch() {
        let err = quantize_to_q4_0(&[0, 0, 0], DType::Fp32, &[1]).unwrap_err();
        assert!(err.to_string().contains("byte length"));
    }

    #[test]
    fn q8_0_quantize_dequantize_roundtrip() {
        let original_floats: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.125).collect();
        let quantized =
            quantize_to_q8_0(&raw_data_helper(&original_floats), DType::Fp32, &[64]).unwrap();
        assert_eq!(quantized.len(), 68);

        let mut dequantized = vec![0.0f32; 64];
        dequantize_q8_0(&quantized, &mut dequantized);

        for i in 0..64 {
            let diff = (dequantized[i] - original_floats[i]).abs();
            assert!(diff <= (4.0 / 127.0) + 1e-5, "index {i} diff={diff}");
        }
    }

    fn raw_data_helper(floats: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &f in floats {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        bytes
    }
}
