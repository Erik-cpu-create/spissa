use crate::{Result, RuntimeError};
use rllm_container::DType;

/// Runtime tensor representation.
///
/// Phase 5 stores all runtime values as `f32` for correctness and simplicity.
/// Later phases can add dtype-preserving kernels once the full-decode baseline is proven.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn from_bytes(
        name: impl Into<String>,
        shape: Vec<u64>,
        dtype: DType,
        bytes: &[u8],
    ) -> Result<Self> {
        let name = name.into();
        let shape_usize: Vec<usize> = shape
            .iter()
            .map(|&dim| {
                usize::try_from(dim)
                    .map_err(|_| RuntimeError::Shape(format!("dimension {dim} does not fit usize")))
            })
            .collect::<Result<_>>()?;
        let expected_elements = shape_usize.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| RuntimeError::Shape("element count overflow".to_string()))
        })?;
        let expected_bytes = dtype.byte_size_for_elements(expected_elements);

        if expected_bytes != bytes.len() {
            return Err(RuntimeError::InvalidTensorData(format!(
                "{} expects {} bytes from shape {:?} and dtype {:?}, got {}",
                &name,
                expected_bytes,
                shape_usize,
                dtype,
                bytes.len()
            )));
        }

        let mut data = decode_to_f32(dtype, bytes)?;
        if dtype.is_quantized() {
            if data.len() < expected_elements {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "{} decoded {} values from quantized bytes, expected {}",
                    &name,
                    data.len(),
                    expected_elements
                )));
            }
            data.truncate(expected_elements);
        }
        Ok(Self {
            name,
            shape: shape_usize,
            dtype,
            data,
        })
    }

    pub fn element_count(&self) -> usize {
        self.data.len()
    }

    pub fn runtime_size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

pub fn decode_to_f32(dtype: DType, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        DType::Fp16 => decode_chunks_2(bytes, fp16_to_f32),
        DType::Bf16 => decode_chunks_2(bytes, bf16_to_f32),
        DType::Fp32 => decode_chunks_4(bytes, f32::from_bits),
        DType::Fp64 => decode_chunks_8(bytes, |bits| f64::from_bits(bits) as f32),
        DType::I8 => Ok(bytes.iter().map(|&b| (b as i8) as f32).collect()),
        DType::U8 => Ok(bytes.iter().map(|&b| b as f32).collect()),
        DType::I16 => decode_chunks_2(bytes, |bits| i16::from_le_bytes(bits.to_le_bytes()) as f32),
        DType::U16 => decode_chunks_2(bytes, |bits| bits as f32),
        DType::I32 => decode_chunks_4(bytes, |bits| i32::from_le_bytes(bits.to_le_bytes()) as f32),
        DType::U32 => decode_chunks_4(bytes, |bits| bits as f32),
        DType::I64 => decode_chunks_8(bytes, |bits| i64::from_le_bytes(bits.to_le_bytes()) as f32),
        DType::U64 => decode_chunks_8(bytes, |bits| bits as f32),
        DType::Q4_0 => {
            if !bytes.len().is_multiple_of(18) {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "expected Q4_0 byte length divisible by 18, got {}",
                    bytes.len()
                )));
            }
            let num_blocks = bytes.len() / 18;
            let mut out = vec![0.0f32; num_blocks * 32];
            crate::dequantize::dequantize_q4_0(bytes, &mut out);
            Ok(out)
        }
        DType::Q8_0 => {
            if !bytes.len().is_multiple_of(34) {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "expected Q8_0 byte length divisible by 34, got {}",
                    bytes.len()
                )));
            }
            let num_blocks = bytes.len() / 34;
            let mut out = vec![0.0f32; num_blocks * 32];
            crate::dequantize::dequantize_q8_0(bytes, &mut out);
            Ok(out)
        }
    }
}

fn decode_chunks_2(bytes: &[u8], f: impl Fn(u16) -> f32) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "expected even byte length for 16-bit dtype, got {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn decode_chunks_4(bytes: &[u8], f: impl Fn(u32) -> f32) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "expected byte length divisible by 4, got {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect())
}

fn decode_chunks_8(bytes: &[u8], f: impl Fn(u64) -> f32) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "expected byte length divisible by 8, got {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            f(u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]))
        })
        .collect())
}

const fn fp16_to_f32_const(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;

    let f32_bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                // Subnormal half: normalize significand then rebias exponent.
                let mut frac_norm = frac;
                let mut exp_norm = -14i32;
                while (frac_norm & 0x0400) == 0 {
                    frac_norm <<= 1;
                    exp_norm -= 1;
                }
                frac_norm &= 0x03ff;
                sign | (((exp_norm + 127) as u32) << 23) | (frac_norm << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = (exp as u32) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };

    f32::from_bits(f32_bits)
}

const fn generate_fp16_lut() -> [f32; 65536] {
    let mut lut = [0.0; 65536];
    let mut i = 0;
    while i < 65536 {
        lut[i] = fp16_to_f32_const(i as u16);
        i += 1;
    }
    lut
}

pub static FP16_TO_F32_LUT: [f32; 65536] = generate_fp16_lut();

/// Convert IEEE-754 binary16 bits to `f32` using a fast 256KB Lookup Table.
#[inline(always)]
pub fn fp16_to_f32(bits: u16) -> f32 {
    FP16_TO_F32_LUT[bits as usize]
}

/// Convert bfloat16 bits to `f32`.
#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert `f32` to IEEE-754 binary16 bits.
pub fn f32_to_fp16(val: f32) -> u16 {
    let f32_bits = val.to_bits();
    let sign = ((f32_bits >> 16) & 0x8000) as u16;
    let exp_bits = (f32_bits >> 23) & 0xFF;
    let frac_bits = f32_bits & 0x007F_FFFF;

    if exp_bits == 0 {
        // Zero or subnormal f32
        sign
    } else if exp_bits == 0xFF {
        // NaN or Inf
        if frac_bits != 0 {
            // NaN
            sign | 0x7E00
        } else {
            // Inf
            sign | 0x7C00
        }
    } else {
        // Normal f32
        let exp = exp_bits as i32 - 127;
        let exp_norm = exp + 15;
        if exp_norm <= 0 {
            // Underflow to zero or subnormal
            if exp_norm < -10 {
                sign // Underflow to zero
            } else {
                // Representable as subnormal f16
                let frac = frac_bits | 0x0080_0000;
                let shift = (14 - exp) as u32;
                let half_frac = (frac >> shift) as u16;
                sign | half_frac
            }
        } else if exp_norm >= 31 {
            // Overflow to Inf
            sign | 0x7C00
        } else {
            // Normal f16
            let half_frac = (frac_bits >> 13) as u16;
            sign | ((exp_norm as u16) << 10) | half_frac
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn fp16_conversion_handles_common_values() {
        assert_close(fp16_to_f32(0x0000), 0.0);
        assert_close(fp16_to_f32(0x3c00), 1.0);
        assert_close(fp16_to_f32(0xc000), -2.0);
        assert_close(fp16_to_f32(0x3800), 0.5);
        assert!(fp16_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn f32_to_fp16_roundtrip() {
        for val in [0.0f32, 1.0, -2.0, 0.5, 3.5, 4.25, 0.125] {
            let bits = f32_to_fp16(val);
            let decoded = fp16_to_f32(bits);
            assert_close(decoded, val);
        }
    }

    #[test]
    fn bf16_conversion_handles_common_values() {
        assert_close(bf16_to_f32(0x3f80), 1.0);
        assert_close(bf16_to_f32(0xc000), -2.0);
    }

    #[test]
    fn tensor_from_f32_bytes_preserves_values() {
        let mut bytes = Vec::new();
        for value in [1.0f32, -2.0, 3.5, 4.25] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let tensor = Tensor::from_bytes("x", vec![2, 2], DType::Fp32, &bytes).unwrap();
        assert_eq!(tensor.name, "x");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, -2.0, 3.5, 4.25]);
    }

    #[test]
    fn tensor_from_fp16_bytes_decodes_to_f32() {
        let mut bytes = Vec::new();
        for bits in [0x3c00u16, 0xc000, 0x3800] {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }

        let tensor = Tensor::from_bytes("half", vec![3], DType::Fp16, &bytes).unwrap();
        assert_eq!(tensor.data, vec![1.0, -2.0, 0.5]);
    }

    #[test]
    fn tensor_from_q4_0_bytes_uses_block_byte_size() {
        let mut bytes = vec![0u8; 18];
        bytes[0..2].copy_from_slice(&f32_to_fp16(1.0).to_le_bytes());
        bytes[2] = 0xf0;

        let tensor = Tensor::from_bytes("q4.weight", vec![32], DType::Q4_0, &bytes).unwrap();

        assert_eq!(tensor.element_count(), 32);
        assert_eq!(tensor.data[0], -8.0);
        assert_eq!(tensor.data[1], 7.0);
    }

    #[test]
    fn tensor_rejects_unaligned_q4_0_bytes() {
        let err = Tensor::from_bytes("bad-q4", vec![32], DType::Q4_0, &[0u8; 17]).unwrap_err();
        assert!(err.to_string().contains("expects 18 bytes"));

        let err = decode_to_f32(DType::Q4_0, &[0u8; 17]).unwrap_err();
        assert!(err.to_string().contains("Q4_0"));
    }

    #[test]
    fn tensor_rejects_shape_byte_mismatch() {
        let err = Tensor::from_bytes("bad", vec![2], DType::Fp32, &[0, 1, 2]).unwrap_err();
        assert!(err.to_string().contains("expects 8 bytes"));
    }
}
