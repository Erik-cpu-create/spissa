#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linear, sample_argmax};
    use rllm_container::{ChunkRangeSpec, DType, GlobalMetadata, RllmWriter, TensorMeta};
    use rtc_codec::{EncodeMeta, RleCodec, TensorCodec};
    use sha2::{Digest, Sha256};

    fn sha256_array(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn fp16_bytes(values: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn bf16_bytes(values: &[u16]) -> Vec<u8> {
        fp16_bytes(values)
    }

    fn q4_0_block_bytes(scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut bytes = vec![0u8; 18];
        bytes[0..2].copy_from_slice(&crate::tensor::f32_to_fp16(scale).to_le_bytes());
        for i in 0..16 {
            let q0 = (values[i * 2] + 8) as u8;
            let q1 = (values[i * 2 + 1] + 8) as u8;
            bytes[2 + i] = (q0 & 0x0f) | ((q1 & 0x0f) << 4);
        }
        bytes
    }

    fn q8_0_block_bytes(scale: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut bytes = vec![0u8; 34];
        bytes[0..2].copy_from_slice(&crate::tensor::f32_to_fp16(scale).to_le_bytes());
        for (idx, value) in values.iter().enumerate() {
            bytes[2 + idx] = *value as u8;
        }
        bytes
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rllm-streaming-{name}-{}.rllm", std::process::id()))
    }

    fn write_chunked_weight(path: &std::path::Path) {
        let weight = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // [out=2, in=3]
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "linear.weight".to_string(),
            shape: vec![2, 3],
            dtype: DType::Fp32,
            original_size_bytes: weight.len() as u64,
            compressed_size_bytes: weight.len() as u64,
            original_sha256: sha256_array(&weight),
            chunk_count: 2,
            chunk_start_index: 0,
        });

        // Split in the middle of row 1. Streaming must reconstruct global element
        // positions from cumulative decoded size, not from chunk_offset_in_tensor.
        writer
            .write_chunk(0, "rtc-raw-v1", &weight[..16], &weight[..16], 0)
            .unwrap();
        writer
            .write_chunk(0, "rtc-raw-v1", &weight[16..], &weight[16..], 1)
            .unwrap();
        writer.finalize().unwrap();
    }

    fn add_rle_zero_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        out_features: usize,
        in_features: usize,
    ) {
        let values = vec![0.0f32; out_features * in_features];
        let bytes = f32_bytes(&values);
        let encoded = RleCodec
            .encode(
                &bytes,
                &EncodeMeta {
                    name: name.to_string(),
                    shape: vec![out_features as u64, in_features as u64],
                    dtype: "F32".to_string(),
                },
            )
            .unwrap();
        assert!(encoded.data.len() < bytes.len() / 8);

        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape: vec![out_features as u64, in_features as u64],
            dtype: DType::Fp32,
            original_size_bytes: bytes.len() as u64,
            compressed_size_bytes: encoded.data.len() as u64,
            original_sha256: sha256_array(&bytes),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(tensor_id, "rtc-rle-v1", &encoded.data, &bytes, 0)
            .unwrap();
    }

    fn write_rle_zero_weight(path: &std::path::Path, out_features: usize, in_features: usize) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_rle_zero_tensor(
            &mut writer,
            0,
            "linear.zero.weight",
            out_features,
            in_features,
        );
        writer.finalize().unwrap();
    }

    #[test]
    fn streaming_linear_matches_full_decode_linear_across_chunk_boundary() {
        let path = temp_path("linear");
        write_chunked_weight(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![10.0, 20.0, 30.0, -1.0, 2.0, -3.0]; // [batch=2, in=3]
        let bias = vec![1.0, -1.0];
        let expected = linear(
            &input,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(&bias),
            2,
            3,
            2,
        )
        .unwrap();
        let mut budget = MemoryBudget::new(256);

        let actual = streaming_linear_from_model(
            &mut model,
            "linear.weight",
            &input,
            Some(&bias),
            StreamingLinearConfig {
                batch: 2,
                in_features: 3,
                out_features: 2,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 64,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_linear_matches_full_decode_with_eight_and_four_batch_fast_paths_and_tail() {
        let path = temp_path("linear-batch-fast-path-tail");
        write_chunked_weight(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input: Vec<f32> = (0..39).map(|idx| idx as f32 * 0.25 - 1.0).collect(); // [batch=13, in=3]
        let bias = vec![1.0, -1.0];
        let expected = linear(
            &input,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(&bias),
            13,
            3,
            2,
        )
        .unwrap();
        let mut budget = MemoryBudget::new(512);

        let actual = streaming_linear_from_model(
            &mut model,
            "linear.weight",
            &input,
            Some(&bias),
            StreamingLinearConfig {
                batch: 13,
                in_features: 3,
                out_features: 2,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_linear_rejects_too_small_transient_budget_without_leaking() {
        let path = temp_path("budget");
        write_chunked_weight(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0, 2.0, 3.0];
        let mut budget = MemoryBudget::new(31);

        let err = streaming_linear_from_model(
            &mut model,
            "linear.weight",
            &input,
            None,
            StreamingLinearConfig {
                batch: 1,
                in_features: 3,
                out_features: 2,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_matches_full_decode_with_smaller_scratch_budget() {
        let path = temp_path("tile-linear");
        write_rle_zero_weight(&path, 32, 32);
        let mut standard_model = LazyRllmModel::open(&path).unwrap();
        let mut tile_model = LazyRllmModel::open(&path).unwrap();
        let input: Vec<f32> = (0..64).map(|idx| (idx as f32) * 0.01 - 0.25).collect();
        let bias: Vec<f32> = (0..32).map(|idx| idx as f32 * 0.125).collect();
        let config = StreamingLinearConfig {
            batch: 2,
            in_features: 32,
            out_features: 32,
        };

        let mut standard_budget = MemoryBudget::new(5_000);
        let standard_err = streaming_linear_from_model(
            &mut standard_model,
            "linear.zero.weight",
            &input,
            Some(&bias),
            config,
            &mut standard_budget,
        )
        .unwrap_err();
        assert!(matches!(
            standard_err,
            RuntimeError::MemoryBudgetExceeded { .. }
        ));
        assert_eq!(standard_budget.current_bytes(), 0);

        let mut tile_budget = MemoryBudget::new(5_000);
        let actual = streaming_tile_linear_from_model(
            &mut tile_model,
            "linear.zero.weight",
            &input,
            Some(&bias),
            StreamingTileLinearConfig {
                linear: config,
                tile_elements: 16,
            },
            &mut tile_budget,
        )
        .unwrap();

        let expected = linear(&input, &vec![0.0; 32 * 32], Some(&bias), 2, 32, 32).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(tile_budget.current_bytes(), 0);
        assert!(
            tile_budget.peak_bytes() < 5_000,
            "peak was {}",
            tile_budget.peak_bytes()
        );
        assert!(standard_budget.peak_bytes() < 5_000);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_rejects_too_small_tile_scratch_without_leaking() {
        let path = temp_path("tile-linear-budget");
        write_rle_zero_weight(&path, 32, 32);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 32];
        let mut budget = MemoryBudget::new(4_140);

        let err = streaming_tile_linear_from_model(
            &mut model,
            "linear.zero.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 32,
                    out_features: 32,
                },
                tile_elements: 16,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_dequantizes_q4_0_weight_block() {
        let path = temp_path("tile-linear-q4-0");
        let mut q = [0i8; 32];
        q[0..16].fill(1);
        q[16..32].fill(2);
        let weight = q4_0_block_bytes(1.0, &q);

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "linear.q4.weight".to_string(),
            shape: vec![2, 16],
            dtype: DType::Q4_0,
            original_size_bytes: weight.len() as u64,
            compressed_size_bytes: weight.len() as u64,
            original_sha256: sha256_array(&weight),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(0, "rtc-raw-v1", &weight, &weight, 0)
            .unwrap();
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 16];
        let mut budget = MemoryBudget::new(512);

        let actual = streaming_tile_linear_from_model(
            &mut model,
            "linear.q4.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 16,
                    out_features: 2,
                },
                tile_elements: 8,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, vec![16.0, 32.0]);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_argmax_dequantizes_q4_0_weight_block() {
        let path = temp_path("tile-linear-argmax-q4-0");
        let mut q = [0i8; 32];
        q[0..16].fill(1);
        q[16..32].fill(2);
        let weight = q4_0_block_bytes(1.0, &q);

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "linear.q4.argmax.weight".to_string(),
            shape: vec![2, 16],
            dtype: DType::Q4_0,
            original_size_bytes: weight.len() as u64,
            compressed_size_bytes: weight.len() as u64,
            original_sha256: sha256_array(&weight),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(0, "rtc-raw-v1", &weight, &weight, 0)
            .unwrap();
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 16];
        let mut budget = MemoryBudget::new(512);

        let actual = streaming_tile_linear_argmax_from_model(
            &mut model,
            "linear.q4.argmax.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 16,
                    out_features: 2,
                },
                tile_elements: 8,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, 1);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_accumulates_q8_0_without_f32_chunk_scratch() {
        let path = temp_path("tile-linear-q8-0-direct");
        let mut q = [0i8; 32];
        q[0..16].fill(1);
        q[16..32].fill(2);
        let weight = q8_0_block_bytes(1.0, &q);

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "linear.q8.weight".to_string(),
            shape: vec![2, 16],
            dtype: DType::Q8_0,
            original_size_bytes: weight.len() as u64,
            compressed_size_bytes: weight.len() as u64,
            original_sha256: sha256_array(&weight),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(0, "rtc-raw-v1", &weight, &weight, 0)
            .unwrap();
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 16];
        let mut budget = MemoryBudget::new(96);

        let actual = streaming_tile_linear_from_model(
            &mut model,
            "linear.q8.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 16,
                    out_features: 2,
                },
                tile_elements: 8,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, vec![16.0, 32.0]);
        assert_eq!(budget.current_bytes(), 0);
        assert!(budget.peak_bytes() <= 96);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_multiply_into_accumulates_q8_0_without_f32_chunk_scratch() {
        let path = temp_path("tile-linear-q8-0-multiply-direct");
        let mut q = [0i8; 32];
        q[0..16].fill(1);
        q[16..32].fill(2);
        let weight = q8_0_block_bytes(1.0, &q);

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "linear.q8.multiply.weight".to_string(),
            shape: vec![2, 16],
            dtype: DType::Q8_0,
            original_size_bytes: weight.len() as u64,
            compressed_size_bytes: weight.len() as u64,
            original_sha256: sha256_array(&weight),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(0, "rtc-raw-v1", &weight, &weight, 0)
            .unwrap();
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 16];
        let mut target = vec![2.0f32, 3.0];
        let mut budget = MemoryBudget::new(96);

        streaming_tile_linear_multiply_into_from_model(
            &mut model,
            "linear.q8.multiply.weight",
            &input,
            None,
            &mut target,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 16,
                    out_features: 2,
                },
                tile_elements: 8,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(target, vec![32.0, 96.0]);
        assert_eq!(budget.current_bytes(), 0);
        assert!(budget.peak_bytes() <= 96);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_argmax_accumulates_q8_0_without_f32_chunk_scratch() {
        let path = temp_path("tile-linear-q8-0-argmax-direct");
        let mut q = [0i8; 32];
        q[0..16].fill(1);
        q[16..32].fill(2);
        let weight = q8_0_block_bytes(1.0, &q);

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        writer.add_tensor(TensorMeta {
            tensor_id: 0,
            name: "linear.q8.argmax.weight".to_string(),
            shape: vec![2, 16],
            dtype: DType::Q8_0,
            original_size_bytes: weight.len() as u64,
            compressed_size_bytes: weight.len() as u64,
            original_sha256: sha256_array(&weight),
            chunk_count: 1,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(0, "rtc-raw-v1", &weight, &weight, 0)
            .unwrap();
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 16];
        let mut budget = MemoryBudget::new(96);

        let actual = streaming_tile_linear_argmax_from_model(
            &mut model,
            "linear.q8.argmax.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 16,
                    out_features: 2,
                },
                tile_elements: 8,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, 1);
        assert_eq!(budget.current_bytes(), 0);
        assert!(budget.peak_bytes() <= 96);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn q8_0_batch1_row_fast_path_accumulates_complete_rows() {
        let mut row0 = [0i8; 32];
        let mut row1 = [0i8; 32];
        row0.fill(1);
        row1.fill(2);
        let mut q8 = q8_0_block_bytes(1.0, &row0);
        q8.extend_from_slice(&q8_0_block_bytes(1.0, &row1));

        let input = vec![1.0f32; 32];
        let mut output = vec![0.5f32, 1.5];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 32,
            out_features: 2,
        };

        let used_fast_path = accumulate_q8_0_chunk_batch1_complete_rows(
            &input,
            &mut output,
            &q8,
            0,
            config,
            "linear.q8.rows.weight",
        )
        .unwrap();

        assert!(used_fast_path);
        assert_eq!(output, vec![32.5, 65.5]);
    }

    #[test]
    fn q8_0_batch1_row_fast_path_declines_partial_rows() {
        let mut q = [0i8; 32];
        q.fill(1);
        let q8 = q8_0_block_bytes(1.0, &q);
        let input = vec![1.0f32; 48];
        let mut output = vec![0.0f32; 2];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 48,
            out_features: 2,
        };

        let used_fast_path = accumulate_q8_0_chunk_batch1_complete_rows(
            &input,
            &mut output,
            &q8,
            0,
            config,
            "linear.q8.partial.weight",
        )
        .unwrap();

        assert!(!used_fast_path);
        assert_eq!(output, vec![0.0, 0.0]);
    }

    #[test]
    fn q8_0_output2_runtime_path_accumulates_adjacent_rows_exactly() {
        let mut row0_block0 = [0i8; 32];
        let mut row0_block1 = [0i8; 32];
        let mut row1_block0 = [0i8; 32];
        let mut row1_block1 = [0i8; 32];
        row0_block0.fill(1);
        row0_block1.fill(2);
        row1_block0.fill(3);
        row1_block1.fill(4);

        let mut q8 = Vec::new();
        q8.extend_from_slice(&q8_0_block_bytes(0.5, &row0_block0));
        q8.extend_from_slice(&q8_0_block_bytes(0.5, &row0_block1));
        q8.extend_from_slice(&q8_0_block_bytes(0.5, &row1_block0));
        q8.extend_from_slice(&q8_0_block_bytes(0.5, &row1_block1));

        let mut input = Vec::new();
        for batch in 0..4 {
            input.extend(std::iter::repeat_n((batch + 1) as f32, 32));
            input.extend(std::iter::repeat_n((batch + 2) as f32, 32));
        }
        let mut output = vec![0.0f32; 8];
        let config = StreamingLinearConfig {
            batch: 4,
            in_features: 64,
            out_features: 2,
        };

        accumulate_q8_0_chunk(
            &input,
            &mut output,
            &q8,
            0,
            config,
            "linear.q8.output2.weight",
        )
        .unwrap();

        assert_eq!(
            output,
            vec![80.0, 176.0, 128.0, 288.0, 176.0, 400.0, 224.0, 512.0,]
        );
    }

    #[test]
    fn q8_0_output2_runtime_path_declines_non_matching_input_blocks() {
        let mut row0_block1 = [0i8; 32];
        let mut row1_block0 = [0i8; 32];
        row0_block1.fill(2);
        row1_block0.fill(3);

        let mut q8 = Vec::new();
        q8.extend_from_slice(&q8_0_block_bytes(0.5, &row0_block1));
        q8.extend_from_slice(&q8_0_block_bytes(0.5, &row1_block0));

        let input = vec![1.0f32; 4 * 64];
        let mut output = vec![0.0f32; 8];
        let config = StreamingLinearConfig {
            batch: 4,
            in_features: 64,
            out_features: 2,
        };

        accumulate_q8_0_chunk(
            &input,
            &mut output,
            &q8,
            32,
            config,
            "linear.q8.decline.weight",
        )
        .unwrap();

        assert_eq!(
            output,
            vec![32.0, 48.0, 32.0, 48.0, 32.0, 48.0, 32.0, 48.0]
        );
    }

    #[test]
    fn q8_0_scaled_block_applies_scale_once() {
        let mut q = [0i8; 32];
        for (idx, value) in q.iter_mut().enumerate() {
            *value = idx as i8 - 16;
        }
        let q8 = q8_0_block_bytes(0.5, &q);

        let scaled = q8_0_scaled_block(&q8[2..34], 0.5);

        assert_eq!(scaled[0], -8.0);
        assert_eq!(scaled[16], 0.0);
        assert_eq!(scaled[31], 7.5);
    }

    #[test]
    fn f32_dot_32_batch4_accumulates_four_outputs() {
        let mut weights = [0.0f32; 32];
        weights[0] = 1.0;
        weights[1] = 2.0;

        let mut input = vec![0.0f32; 4 * 32];
        input[0] = 1.0;
        input[1] = 10.0;
        input[32] = 2.0;
        input[33] = 20.0;
        input[64] = 3.0;
        input[65] = 30.0;
        input[96] = 4.0;
        input[97] = 40.0;

        let mut output = vec![0.5f32, 1.5, 2.5, 3.5];

        accumulate_f32_dot_32_batch4(&weights, &input, 32, &mut output, 1, 0);

        assert_eq!(output, vec![21.5, 43.5, 65.5, 87.5]);
    }

    #[test]
    fn output2_batch4_helper_accumulates_two_adjacent_features() {
        let mut first = [0.0f32; 32];
        let mut second = [0.0f32; 32];
        first[0] = 1.0;
        first[1] = 2.0;
        second[0] = 3.0;
        second[1] = 4.0;

        let mut input = vec![0.0f32; 4 * 32];
        input[0] = 1.0;
        input[1] = 10.0;
        input[32] = 2.0;
        input[33] = 20.0;
        input[64] = 3.0;
        input[65] = 30.0;
        input[96] = 4.0;
        input[97] = 40.0;

        let mut output = vec![0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

        accumulate_f32_dot_32_output2_batch4_reebundle(
            &first,
            &second,
            &input,
            32,
            &mut output,
            2,
            0,
        );

        assert_eq!(
            output,
            vec![21.5, 44.5, 44.5, 89.5, 67.5, 134.5, 90.5, 179.5]
        );
    }

    #[test]
    fn f32_dot_32_batch4_into_accumulates_existing_values() {
        let mut weights = [0.0f32; 32];
        weights[0] = 1.0;
        weights[1] = 2.0;

        let mut input = vec![0.0f32; 4 * 32];
        input[0] = 1.0;
        input[1] = 10.0;
        input[32] = 2.0;
        input[33] = 20.0;
        input[64] = 3.0;
        input[65] = 30.0;
        input[96] = 4.0;
        input[97] = 40.0;

        let mut accumulators = vec![0.5f32, 1.5, 2.5, 3.5];

        accumulate_f32_dot_32_batch4_into(&weights, &input, 32, &mut accumulators, 0);

        assert_eq!(accumulators, vec![21.5, 43.5, 65.5, 87.5]);
    }

    #[test]
    fn q8_0_batch1_multiply_row_fast_path_accumulates_complete_rows() {
        let mut row0 = [0i8; 32];
        let mut row1 = [0i8; 32];
        row0.fill(1);
        row1.fill(2);
        let mut q8 = q8_0_block_bytes(1.0, &row0);
        q8.extend_from_slice(&q8_0_block_bytes(1.0, &row1));

        let input = vec![1.0f32; 32];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 32,
            out_features: 2,
        };
        let mut target = vec![2.0f32, 3.0];
        let mut state = StreamingLinearMultiplyIntoState::new(&mut target, None, config);

        let used_fast_path = accumulate_q8_0_chunk_multiply_into_batch1_complete_rows(
            &input,
            &q8,
            0,
            config,
            &mut state,
            "linear.q8.multiply.rows.weight",
        )
        .unwrap();
        state
            .finish(config, "linear.q8.multiply.rows.weight")
            .unwrap();

        assert!(used_fast_path);
        assert_eq!(target, vec![64.0, 192.0]);
    }

    #[test]
    fn q8_0_batch1_argmax_row_fast_path_accumulates_complete_rows() {
        let mut row0 = [0i8; 32];
        let mut row1 = [0i8; 32];
        row0.fill(1);
        row1.fill(2);
        let mut q8 = q8_0_block_bytes(1.0, &row0);
        q8.extend_from_slice(&q8_0_block_bytes(1.0, &row1));

        let input = vec![1.0f32; 32];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 32,
            out_features: 2,
        };
        let mut state = StreamingLinearArgmaxState::new(None);

        let used_fast_path = accumulate_q8_0_chunk_argmax_batch1_complete_rows(
            &input,
            &q8,
            0,
            config,
            &mut state,
            "linear.q8.argmax.rows.weight",
        )
        .unwrap();
        let best = state.finish(config, "linear.q8.argmax.rows.weight").unwrap();

        assert!(used_fast_path);
        assert_eq!(best, 1);
    }

    fn add_f32_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        shape: Vec<u64>,
        values: &[f32],
        split_at: usize,
    ) {
        let bytes = f32_bytes(values);
        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape,
            dtype: DType::Fp32,
            original_size_bytes: bytes.len() as u64,
            compressed_size_bytes: bytes.len() as u64,
            original_sha256: sha256_array(&bytes),
            chunk_count: 2,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(
                tensor_id,
                "rtc-raw-v1",
                &bytes[..split_at],
                &bytes[..split_at],
                0,
            )
            .unwrap();
        writer
            .write_chunk(
                tensor_id,
                "rtc-raw-v1",
                &bytes[split_at..],
                &bytes[split_at..],
                1,
            )
            .unwrap();
    }

    fn add_fp16_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        shape: Vec<u64>,
        values: &[u16],
        split_at: usize,
    ) {
        let bytes = fp16_bytes(values);
        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape,
            dtype: DType::Fp16,
            original_size_bytes: bytes.len() as u64,
            compressed_size_bytes: bytes.len() as u64,
            original_sha256: sha256_array(&bytes),
            chunk_count: 2,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(
                tensor_id,
                "rtc-raw-v1",
                &bytes[..split_at],
                &bytes[..split_at],
                0,
            )
            .unwrap();
        writer
            .write_chunk(
                tensor_id,
                "rtc-raw-v1",
                &bytes[split_at..],
                &bytes[split_at..],
                1,
            )
            .unwrap();
    }

    fn add_bf16_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        name: &str,
        shape: Vec<u64>,
        values: &[u16],
        split_at: usize,
    ) {
        let bytes = bf16_bytes(values);
        writer.add_tensor(TensorMeta {
            tensor_id,
            name: name.to_string(),
            shape,
            dtype: DType::Bf16,
            original_size_bytes: bytes.len() as u64,
            compressed_size_bytes: bytes.len() as u64,
            original_sha256: sha256_array(&bytes),
            chunk_count: 2,
            chunk_start_index: 0,
        });
        writer
            .write_chunk(
                tensor_id,
                "rtc-raw-v1",
                &bytes[..split_at],
                &bytes[..split_at],
                0,
            )
            .unwrap();
        writer
            .write_chunk(
                tensor_id,
                "rtc-raw-v1",
                &bytes[split_at..],
                &bytes[split_at..],
                1,
            )
            .unwrap();
    }

    fn add_bf16_input_tile_sidecar_tensor(
        writer: &mut RllmWriter,
        tensor_id: u64,
        source_name: &str,
        out_features: usize,
        in_features: usize,
        row_major_values: &[u16],
        tile_features: usize,
    ) {
        assert_eq!(row_major_values.len(), out_features * in_features);
        let sidecar_name = crate::input_tile_sidecar_weight_name(source_name);
        let mut input_major = vec![0u8; row_major_values.len() * 2];
        for in_feature in 0..in_features {
            for out_feature in 0..out_features {
                let source_idx = out_feature * in_features + in_feature;
                let dst_idx = in_feature * out_features + out_feature;
                input_major[dst_idx * 2..dst_idx * 2 + 2]
                    .copy_from_slice(&row_major_values[source_idx].to_le_bytes());
            }
        }

        writer.add_tensor(TensorMeta {
            tensor_id,
            name: sidecar_name,
            shape: vec![in_features as u64, out_features as u64],
            dtype: DType::Bf16,
            original_size_bytes: input_major.len() as u64,
            compressed_size_bytes: input_major.len() as u64,
            original_sha256: sha256_array(&input_major),
            chunk_count: 0,
            chunk_start_index: 0,
        });

        let column_bytes = (out_features * 2) as u64;
        let chunk_columns = tile_features.max(1);
        for feature_start in (0..in_features).step_by(chunk_columns) {
            let feature_end = (feature_start + chunk_columns).min(in_features);
            let byte_start = feature_start * out_features * 2;
            let byte_end = feature_end * out_features * 2;
            let chunk = &input_major[byte_start..byte_end];
            let mut ranges = Vec::new();
            for feature in feature_start..feature_end {
                let offset = ((feature - feature_start) as u64) * column_bytes;
                ranges.push(ChunkRangeSpec {
                    original_offset: offset,
                    original_size: column_bytes,
                    compressed_offset: offset,
                    compressed_size: column_bytes,
                });
            }
            writer
                .write_chunk_with_range_specs(
                    tensor_id,
                    "rtc-raw-v1",
                    chunk,
                    chunk,
                    (feature_start * out_features) as u64,
                    &ranges,
                )
                .unwrap();
        }
    }

    #[test]
    fn streaming_tile_linear_argmax_matches_full_logits_across_split_rows() {
        let path = temp_path("tile-linear-argmax");
        let weight = vec![
            0.5, -1.0, 2.0, -2.0, 0.25, 0.5, 1.0, 1.0, -1.0, 0.0, -0.5, 0.75,
        ];
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "linear.argmax.weight",
            vec![4, 3],
            &weight,
            20,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let bias = vec![0.0, 0.5, -1.0, 4.0];
        let config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: 3,
                out_features: 4,
            },
            tile_elements: 2,
        };
        let mut logits_model = LazyRllmModel::open(&path).unwrap();
        let mut argmax_model = LazyRllmModel::open(&path).unwrap();
        let mut logits_budget = MemoryBudget::new(256);
        let mut argmax_budget = MemoryBudget::new(256);

        let logits = streaming_tile_linear_from_model(
            &mut logits_model,
            "linear.argmax.weight",
            &input,
            Some(&bias),
            config,
            &mut logits_budget,
        )
        .unwrap();
        let expected = sample_argmax(&logits).unwrap();
        let actual = streaming_tile_linear_argmax_from_model(
            &mut argmax_model,
            "linear.argmax.weight",
            &input,
            Some(&bias),
            config,
            &mut argmax_budget,
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(logits_budget.current_bytes(), 0);
        assert_eq!(argmax_budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_argmax_prefix_scans_only_requested_rows() {
        let path = temp_path("tile-linear-argmax-prefix");
        let weight = vec![
            0.5, -1.0, 2.0, -2.0, 0.25, 0.5, 1.0, 1.0, -1.0, 0.0, -0.5, 0.75,
        ];
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "linear.argmax.prefix.weight",
            vec![4, 3],
            &weight,
            20,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let full_bias = vec![0.0, 0.5, -1.0, 4.0];
        let prefix_bias = vec![0.0, 0.5, -1.0];
        let full_logits = linear(&input, &weight, Some(&full_bias), 1, 3, 4).unwrap();
        assert_eq!(sample_argmax(&full_logits).unwrap(), 3);
        let expected_prefix = sample_argmax(&full_logits[..3]).unwrap();
        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(256);

        let actual = streaming_tile_linear_argmax_prefix_from_model(
            &mut model,
            "linear.argmax.prefix.weight",
            &input,
            Some(&prefix_bias),
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 3,
                    out_features: 4,
                },
                tile_elements: 2,
            },
            3,
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, expected_prefix);
        assert_ne!(actual, 3);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_argmax_candidate_rows_scores_only_candidates() {
        let path = temp_path("tile-linear-argmax-candidates-bf16");
        let weight_bf16 = vec![
            0x3f80, 0x0000, 0x0000, 0x0000, 0x4000, 0x0000, 0x3f80, 0x3f80, 0x3f80, 0x4040, 0x0000,
            0x3f80,
        ];
        let weight_f32: Vec<f32> = weight_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "linear.argmax.candidates.bf16.weight",
            vec![4, 3],
            &weight_bf16,
            12,
        );
        writer.finalize().unwrap();

        let input = vec![1.0, 2.0, 3.0];
        let logits = linear(&input, &weight_f32, None, 1, 3, 4).unwrap();
        assert_eq!(sample_argmax(&logits).unwrap(), 2);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(64);

        let actual = streaming_tile_linear_argmax_candidate_rows_from_model(
            &mut model,
            "linear.argmax.candidates.bf16.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 3,
                    out_features: 4,
                },
                tile_elements: 2,
            },
            &[1, 3],
            &mut budget,
        )
        .unwrap()
        .unwrap();

        assert_eq!(actual, 3);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_argmax_candidate_rows_range_scores_only_candidates() {
        let path = temp_path("tile-linear-argmax-candidate-ranges-bf16");
        let weight_bf16 = vec![
            0x3f80, 0x0000, 0x0000, 0x0000, 0x4000, 0x0000, 0x3f80, 0x3f80, 0x3f80, 0x4040, 0x0000,
            0x3f80,
        ];
        let weight_f32: Vec<f32> = weight_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "linear.argmax.candidate-ranges.bf16.weight",
            vec![4, 3],
            &weight_bf16,
            10,
        );
        writer.finalize().unwrap();

        let input = vec![1.0, 2.0, 3.0];
        let logits = linear(&input, &weight_f32, None, 1, 3, 4).unwrap();
        assert_eq!(sample_argmax(&logits).unwrap(), 2);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(64);

        let actual = streaming_tile_linear_argmax_candidate_rows_range_from_model(
            &mut model,
            "linear.argmax.candidate-ranges.bf16.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 3,
                    out_features: 4,
                },
                tile_elements: 2,
            },
            &[1, 3],
            &mut budget,
        )
        .unwrap()
        .unwrap();

        assert_eq!(actual, 3);
        assert_eq!(budget.current_bytes(), 0);
        assert!(budget.peak_bytes() < 64);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_argmax_uses_raw_bf16_batch1_path() {
        let path = temp_path("tile-linear-argmax-bf16");
        let weight_bf16 = vec![
            0x3f00, 0xbf80, 0x4000, 0xc000, 0x3e80, 0x3f00, 0x3f80, 0x3f80, 0xbf80, 0x0000, 0xbf00,
            0x3f40,
        ];
        let weight_f32: Vec<f32> = weight_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "linear.argmax.bf16.weight",
            vec![4, 3],
            &weight_bf16,
            14,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let bias = vec![0.0, 0.5, -1.0, 4.0];
        let logits = linear(&input, &weight_f32, Some(&bias), 1, 3, 4).unwrap();
        let expected = sample_argmax(&logits).unwrap();
        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(16);
        let actual = streaming_tile_linear_argmax_from_model(
            &mut model,
            "linear.argmax.bf16.weight",
            &input,
            Some(&bias),
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 3,
                    out_features: 4,
                },
                tile_elements: 2,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 16,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_argmax_with_rolling_records_stats() {
        let path = temp_path("tile-linear-argmax-bf16-rolling");
        let weight_bf16 = vec![
            0x3f00, 0xbf80, 0x4000, 0xc000, 0x3e80, 0x3f00, 0x3f80, 0x3f80, 0xbf80, 0x0000, 0xbf00,
            0x3f40, 0x3f80, 0x4000, 0x4040, 0xc040, 0x3f00, 0x3e80, 0x3f00, 0x3f80, 0x4000, 0x4040,
            0x4080, 0x40a0,
        ];
        let weight_f32: Vec<f32> = weight_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "linear.argmax.bf16.rolling.weight",
            vec![8, 3],
            &weight_bf16,
            weight_bf16.len() * 2,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let bias = vec![0.0, 0.5, -1.0, 4.0, 0.25, -0.25, 0.75, 1.25];
        let logits = linear(&input, &weight_f32, Some(&bias), 1, 3, 8).unwrap();
        let expected = sample_argmax(&logits).unwrap();
        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(64);
        let mut executor =
            crate::rolling::RollingExecutor::new(crate::rolling::RollingExecutorConfig {
                enabled: true,
                worker_count: 4,
                min_rows_per_worker: 1,
            });

        let actual = streaming_tile_linear_argmax_with_rolling_from_model(
            &mut model,
            "linear.argmax.bf16.rolling.weight",
            &input,
            Some(&bias),
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 3,
                    out_features: 8,
                },
                tile_elements: 2,
            },
            &mut budget,
            Some(&mut executor),
        )
        .unwrap();
        let stats = executor.take_stats();

        assert_eq!(actual, expected);
        assert!(stats.submitted_tasks > 0);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn raw_bf16_argmax_row_block_kernel_matches_materialized_partial_chunk() {
        let weight_bf16 = vec![
            0x3f00, 0xbf80, 0x4000, 0xc000, 0x3e80, 0x3f00, 0x3f80, 0x3f80, 0xbf80, 0x0000, 0xbf00,
            0x3f40, 0x3f80, 0x4000, 0x4040, 0xc040, 0x3f00, 0x3e80,
        ];
        let raw = bf16_bytes(&weight_bf16);
        let weight_f32: Vec<f32> = weight_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let input = vec![1.5, -2.0, 0.25];
        let bias = vec![0.25, -0.5, 0.75, 1.0, -1.25, 1.5];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 3,
            out_features: 6,
        };
        let expected_logits = linear(&input, &weight_f32, Some(&bias), 1, 3, 6).unwrap();
        let expected = sample_argmax(&expected_logits).unwrap();
        let mut state = StreamingLinearArgmaxState::new(Some(&bias));

        accumulate_raw_16bit_chunk_argmax_row_blocked(
            &input,
            &raw[2..],
            1,
            config,
            DType::Bf16,
            &mut state,
            "linear.argmax.bf16.row-block.weight",
            None,
        )
        .unwrap();
        let actual = state
            .finish(config, "linear.argmax.bf16.row-block.weight")
            .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn runtime_thread_policy_uses_auto_and_numeric_override() {
        assert_eq!(effective_runtime_threads(Some("1"), 8), 1);
        assert_eq!(effective_runtime_threads(Some("4"), 8), 4);
        assert_eq!(effective_runtime_threads(Some("32"), 8), 8);
        assert_eq!(effective_runtime_threads(Some("0"), 8), 8);
        assert_eq!(effective_runtime_threads(Some("bad"), 8), 8);
        assert_eq!(effective_runtime_threads(None, 8), 8);
    }

    #[test]
    fn sparse_parallel_env_parser_requires_explicit_truthy_value() {
        assert!(parse_sparse_parallel_enabled(Some("1")));
        assert!(parse_sparse_parallel_enabled(Some("true")));
        assert!(parse_sparse_parallel_enabled(Some("yes")));
        assert!(parse_sparse_parallel_enabled(Some("on")));
        assert!(!parse_sparse_parallel_enabled(Some("0")));
        assert!(!parse_sparse_parallel_enabled(Some("false")));
        assert!(!parse_sparse_parallel_enabled(Some("")));
        assert!(!parse_sparse_parallel_enabled(None));
    }

    #[test]
    fn auto_argmax_threads_caps_large_vocab_without_overriding_manual_setting() {
        assert_eq!(effective_argmax_runtime_threads(None, 6, 49_152), 6);
        assert_eq!(effective_argmax_runtime_threads(None, 6, 128_256), 2);
        assert_eq!(effective_argmax_runtime_threads(Some("6"), 6, 128_256), 6);
        assert_eq!(effective_argmax_runtime_threads(Some("1"), 6, 128_256), 1);
    }

    #[test]
    fn row_parallelism_is_capped_by_rows_and_minimum_work() {
        assert_eq!(effective_row_block_threads(0, 8), 1);
        assert_eq!(effective_row_block_threads(1, 8), 1);
        assert_eq!(effective_row_block_threads(3, 8), 1);
        assert_eq!(effective_row_block_threads(4, 8), 4);
        assert_eq!(effective_row_block_threads(32, 8), 8);
    }

    #[test]
    fn parallel_raw_bf16_argmax_rows_match_materialized_logits() {
        let weight_bf16 = vec![
            0x3f80, 0x0000, 0x0000, 0x0000, 0x4000, 0x0000, 0x0000, 0x0000, 0x4040, 0xbf80, 0x3f80,
            0x0000, 0x3f00, 0x3f00, 0x3f00, 0xc000, 0x0000, 0x3f80, 0x0000, 0xc040, 0x3f80, 0x3f80,
            0x3f80, 0x3f80,
        ];
        let raw = bf16_bytes(&weight_bf16);
        let weight_f32: Vec<f32> = weight_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let input = vec![1.0, -2.0, 0.5];
        let bias = vec![0.0, 0.25, -0.25, 0.5, 0.0, 1.0, -1.0, 0.75];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 3,
            out_features: 8,
        };
        let expected_logits = linear(&input, &weight_f32, Some(&bias), 1, 3, 8).unwrap();
        let expected = sample_argmax(&expected_logits).unwrap();

        let actual = parallel_raw_16bit_argmax_rows(
            &input,
            &raw,
            0,
            0,
            8,
            config,
            DType::Bf16,
            Some(&bias),
            4,
        );

        assert_eq!(actual.best_index, expected);
    }

    #[test]
    fn rolling_raw_bf16_argmax_rows_match_materialized_logits_and_record_stats() {
        let weight_bf16 = vec![
            0x3f80, 0x0000, 0x0000, 0x0000, 0x4000, 0x0000, 0x0000, 0x0000, 0x4040, 0xbf80, 0x3f80,
            0x0000, 0x3f00, 0x3f00, 0x3f00, 0xc000, 0x0000, 0x3f80, 0x0000, 0xc040, 0x3f80, 0x3f80,
            0x3f80, 0x3f80,
        ];
        let raw = bf16_bytes(&weight_bf16);
        let weight_f32: Vec<f32> = weight_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let input = vec![1.0, -2.0, 0.5];
        let bias = vec![0.0, 0.25, -0.25, 0.5, 0.0, 1.0, -1.0, 0.75];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 3,
            out_features: 8,
        };
        let expected_logits = linear(&input, &weight_f32, Some(&bias), 1, 3, 8).unwrap();
        let expected = sample_argmax(&expected_logits).unwrap();
        let mut executor =
            crate::rolling::RollingExecutor::new(crate::rolling::RollingExecutorConfig {
                enabled: true,
                worker_count: 4,
                min_rows_per_worker: 1,
            });

        let actual = rolling_raw_16bit_argmax_rows(
            &input,
            &raw,
            0,
            0,
            8,
            config,
            DType::Bf16,
            Some(&bias),
            &mut executor,
        );
        let stats = executor.take_stats();

        assert_eq!(actual.best_index, expected);
        assert_eq!(stats.submitted_tasks, 4);
        assert_eq!(stats.worker_wakeups, 4);
        assert_eq!(stats.sequential_fallbacks, 0);
    }

    #[test]
    fn streaming_tile_linear_multiply_into_matches_materialized_linear() {
        let path = temp_path("tile-linear-multiply-into");
        let weight = vec![
            0.5, -1.0, 2.0, -2.0, 0.25, 0.5, 1.0, 1.0, -1.0, 0.0, -0.5, 0.75,
        ];
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "linear.multiply.weight",
            vec![4, 3],
            &weight,
            20,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25, -0.5, 0.75, 2.0]; // [batch=2, in=3]
        let mut target = vec![0.2, -0.3, 0.4, 0.5, -1.0, 1.25, -1.5, 2.0];
        let bias = vec![0.0, 0.5, -1.0, 4.0];
        let config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 2,
                in_features: 3,
                out_features: 4,
            },
            tile_elements: 2,
        };
        let mut materialized_model = LazyRllmModel::open(&path).unwrap();
        let mut fused_model = LazyRllmModel::open(&path).unwrap();
        let mut materialized_budget = MemoryBudget::new(512);
        let mut fused_budget = MemoryBudget::new(512);

        let materialized = streaming_tile_linear_from_model(
            &mut materialized_model,
            "linear.multiply.weight",
            &input,
            Some(&bias),
            config,
            &mut materialized_budget,
        )
        .unwrap();
        let mut expected = target.clone();
        for (dst, value) in expected.iter_mut().zip(materialized.iter()) {
            *dst *= *value;
        }

        streaming_tile_linear_multiply_into_from_model(
            &mut fused_model,
            "linear.multiply.weight",
            &input,
            Some(&bias),
            &mut target,
            config,
            &mut fused_budget,
        )
        .unwrap();

        assert_close_vec(&target, &expected, 1e-6);
        assert_eq!(materialized_budget.current_bytes(), 0);
        assert_eq!(fused_budget.current_bytes(), 0);
        assert!(
            fused_budget.peak_bytes() <= materialized_budget.peak_bytes(),
            "fused peak {} exceeded materialized peak {}",
            fused_budget.peak_bytes(),
            materialized_budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_matches_raw_fp16_batch1_linear() {
        let path = temp_path("tile-linear-fp16-batch1");
        let weight_fp16 = vec![
            0x3800, 0xbc00, 0x4000, 0xc000, 0x3400, 0x3800, 0x3c00, 0x3c00, 0xbc00, 0x0000, 0xb800,
            0x3a00,
        ];
        let weight_f32: Vec<f32> = weight_fp16
            .iter()
            .map(|bits| crate::tensor::fp16_to_f32(*bits))
            .collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_fp16_tensor(
            &mut writer,
            0,
            "linear.fp16.batch1.weight",
            vec![4, 3],
            &weight_fp16,
            14,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let bias = vec![0.0, 0.5, -1.0, 4.0];
        let expected = linear(&input, &weight_f32, Some(&bias), 1, 3, 4).unwrap();
        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(512);
        let actual = streaming_tile_linear_from_model(
            &mut model,
            "linear.fp16.batch1.weight",
            &input,
            Some(&bias),
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 3,
                    out_features: 4,
                },
                tile_elements: 2,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_tile_linear_uses_raw_bf16_batch1_path() {
        let path = temp_path("tile-linear-bf16-batch1");
        let weight_bf16 = vec![
            0x3f00, 0xbf80, 0x4000, 0xc000, 0x3e80, 0x3f00, 0x3f80, 0x3f80, 0xbf80, 0x0000, 0xbf00,
            0x3f40,
        ];
        let weight_f32: Vec<f32> = weight_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "linear.bf16.batch1.weight",
            vec![4, 3],
            &weight_bf16,
            14,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let bias = vec![0.0, 0.5, -1.0, 4.0];
        let expected = linear(&input, &weight_f32, Some(&bias), 1, 3, 4).unwrap();
        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(16);
        let actual = streaming_tile_linear_from_model(
            &mut model,
            "linear.bf16.batch1.weight",
            &input,
            Some(&bias),
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 3,
                    out_features: 4,
                },
                tile_elements: 2,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 16,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn raw_fp16_batch1_row_block_kernel_matches_rowwise_linear_across_partial_rows() {
        let weight_fp16 = vec![
            0x3800, 0xbc00, 0x4000, 0xc000, 0x3400, 0x3800, 0x3c00, 0x3c00, 0xbc00, 0x0000, 0xb800,
            0x3a00, 0x3c00, 0x4000, 0x4200, 0xc200, 0x3800, 0x3400,
        ];
        let raw = fp16_bytes(&weight_fp16);
        let weight_f32: Vec<f32> = weight_fp16
            .iter()
            .map(|bits| crate::tensor::fp16_to_f32(*bits))
            .collect();
        let input = vec![1.5, -2.0, 0.25];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 3,
            out_features: 6,
        };
        let mut expected = vec![0.25, -0.5, 0.75, 1.0, -1.25, 1.5];
        let mut actual = expected.clone();
        for (global_idx, weight) in (1usize..).zip(weight_f32.iter().skip(1)) {
            let out_feature = global_idx / config.in_features;
            let in_feature = global_idx % config.in_features;
            expected[out_feature] += input[in_feature] * weight;
        }

        accumulate_fused_raw_fp16_chunk_batch1_row_blocked(
            &input,
            &mut actual,
            &raw[2..],
            1,
            config,
            "linear.fp16.row-block.weight",
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
    }

    #[test]
    fn raw_fp16_batch1_multiply_row_block_kernel_matches_materialized_partial_chunk() {
        let weight_fp16 = vec![
            0x3800, 0xbc00, 0x4000, 0xc000, 0x3400, 0x3800, 0x3c00, 0x3c00, 0xbc00, 0x0000, 0xb800,
            0x3a00, 0x3c00, 0x4000, 0x4200, 0xc200, 0x3800, 0x3400,
        ];
        let raw = fp16_bytes(&weight_fp16);
        let weight_f32: Vec<f32> = weight_fp16
            .iter()
            .map(|bits| crate::tensor::fp16_to_f32(*bits))
            .collect();
        let input = vec![1.5, -2.0, 0.25];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 3,
            out_features: 6,
        };
        let initial = vec![0.25, -0.5, 0.75, 1.0, -1.25, 1.5];
        let mut expected_linear = [0.0; 6];
        for (global_idx, weight) in (1usize..).zip(weight_f32.iter().skip(1)) {
            let out_feature = global_idx / config.in_features;
            let in_feature = global_idx % config.in_features;
            expected_linear[out_feature] += input[in_feature] * weight;
        }
        let mut expected = initial.clone();
        for (dst, value) in expected.iter_mut().zip(expected_linear.iter()) {
            *dst *= *value;
        }
        let mut actual = initial;
        let mut state = StreamingLinearMultiplyIntoState::new(&mut actual, None, config);

        accumulate_multiply_raw_fp16_chunk_batch1_row_blocked(
            &input,
            &raw[2..],
            1,
            config,
            &mut state,
            "linear.fp16.multiply-row-block.weight",
        )
        .unwrap();
        state
            .finish(config, "linear.fp16.multiply-row-block.weight")
            .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
    }

    #[test]
    fn streaming_tile_linear_multiply_into_uses_raw_fp16_path() {
        let path = temp_path("tile-linear-multiply-into-fp16");
        let weight_fp16 = vec![
            0x3800, 0xbc00, 0x4000, 0xc000, 0x3400, 0x3800, 0x3c00, 0x3c00, 0xbc00, 0x0000, 0xb800,
            0x3a00,
        ];
        let weight_f32: Vec<f32> = weight_fp16
            .iter()
            .map(|bits| crate::tensor::fp16_to_f32(*bits))
            .collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_fp16_tensor(
            &mut writer,
            0,
            "linear.multiply.fp16.weight",
            vec![4, 3],
            &weight_fp16,
            14,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let mut target = vec![0.2, -0.3, 0.4, 0.5];
        let bias = vec![0.0, 0.5, -1.0, 4.0];
        let config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: 3,
                out_features: 4,
            },
            tile_elements: 2,
        };
        let materialized = linear(&input, &weight_f32, Some(&bias), 1, 3, 4).unwrap();
        let mut expected = target.clone();
        for (dst, value) in expected.iter_mut().zip(materialized.iter()) {
            *dst *= *value;
        }

        let mut fused_model = LazyRllmModel::open(&path).unwrap();
        let mut fused_budget = MemoryBudget::new(512);
        streaming_tile_linear_multiply_into_from_model(
            &mut fused_model,
            "linear.multiply.fp16.weight",
            &input,
            Some(&bias),
            &mut target,
            config,
            &mut fused_budget,
        )
        .unwrap();

        assert_close_vec(&target, &expected, 1e-6);
        assert_eq!(fused_budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_silu_gate_up_matches_materialized_raw_fp16_batch1() {
        let path = temp_path("silu-gate-up-fp16-batch1");
        let gate_fp16 = vec![
            0x3800, 0xbc00, 0x4000, 0xc000, 0x3400, 0x3800, 0x3c00, 0x3c00, 0xbc00, 0x0000, 0xb800,
            0x3a00,
        ];
        let up_fp16 = vec![
            0x3c00, 0x3800, 0xb800, 0x3400, 0x4000, 0xbc00, 0xc000, 0x3a00, 0x3c00, 0x3800, 0x0000,
            0xb800,
        ];
        let gate_f32: Vec<f32> = gate_fp16
            .iter()
            .map(|bits| crate::tensor::fp16_to_f32(*bits))
            .collect();
        let up_f32: Vec<f32> = up_fp16
            .iter()
            .map(|bits| crate::tensor::fp16_to_f32(*bits))
            .collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_fp16_tensor(
            &mut writer,
            0,
            "mlp.gate.weight",
            vec![4, 3],
            &gate_fp16,
            14,
        );
        add_fp16_tensor(&mut writer, 1, "mlp.up.weight", vec![4, 3], &up_fp16, 14);
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let mut expected = linear(&input, &gate_f32, None, 1, 3, 4).unwrap();
        for value in &mut expected {
            *value = crate::silu(*value);
        }
        let up = linear(&input, &up_f32, None, 1, 3, 4).unwrap();
        for (gate, up) in expected.iter_mut().zip(up.iter()) {
            *gate *= *up;
        }

        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(512);
        let actual = streaming_silu_gate_up_from_model(
            &mut model,
            "mlp.gate.weight",
            "mlp.up.weight",
            &input,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 3,
                    out_features: 4,
                },
                tile_elements: 2,
            },
            &mut budget,
        )
        .unwrap()
        .expect("raw FP16 aligned gate/up should use fused path");

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_silu_gate_up_matches_materialized_raw_bf16_batch1() {
        let path = temp_path("silu-gate-up-bf16-batch1");
        let gate_bf16 = vec![
            0x3f00, 0xbf80, 0x4000, 0xc000, 0x3e80, 0x3f00, 0x3f80, 0x3f80, 0xbf80, 0x0000, 0xbf00,
            0x3f40,
        ];
        let up_bf16 = vec![
            0x3f80, 0x3f00, 0xbf00, 0x3e80, 0x4000, 0xbf80, 0xc000, 0x3f40, 0x3f80, 0x3f00, 0x0000,
            0xbf00,
        ];
        let gate_f32: Vec<f32> = gate_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let up_f32: Vec<f32> = up_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "mlp.gate.bf16.weight",
            vec![4, 3],
            &gate_bf16,
            14,
        );
        add_bf16_tensor(
            &mut writer,
            1,
            "mlp.up.bf16.weight",
            vec![4, 3],
            &up_bf16,
            14,
        );
        writer.finalize().unwrap();

        let input = vec![1.5, -2.0, 0.25];
        let mut expected = linear(&input, &gate_f32, None, 1, 3, 4).unwrap();
        for value in &mut expected {
            *value = crate::silu(*value);
        }
        let up = linear(&input, &up_f32, None, 1, 3, 4).unwrap();
        for (gate, up) in expected.iter_mut().zip(up.iter()) {
            *gate *= *up;
        }

        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::new(512);
        let actual = streaming_silu_gate_up_from_model(
            &mut model,
            "mlp.gate.bf16.weight",
            "mlp.up.bf16.weight",
            &input,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 3,
                    out_features: 4,
                },
                tile_elements: 2,
            },
            &mut budget,
        )
        .unwrap()
        .expect("raw BF16 aligned gate/up should use fused path");

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn sparse_raw_bf16_linear_matches_manual_topk_projection() {
        let path = temp_path("sparse-linear-bf16");
        let weight_bf16 = vec![
            0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0xc040, 0xc080, 0x3f00, 0x3f80, 0x4000,
            0x4040,
        ];
        let input = vec![1.0, -8.0, 2.0, 7.0];
        let selected = crate::select_top_abs_indices(&input, 2);
        assert_eq!(selected, vec![1, 3]);

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "linear.sparse.bf16.weight",
            vec![3, 4],
            &weight_bf16,
            14,
        );
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::unbounded();
        let mut stats = crate::RamaExperimentalSpeedStats::default();
        let output = streaming_sparse_tile_linear_from_model(
            &mut model,
            "linear.sparse.bf16.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 4,
                    out_features: 3,
                },
                tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
            },
            crate::RamaExperimentalSpeedConfig {
                enabled: true,
                aip_policy: crate::RamaAipPolicyKind::Speed,
                aip_topk: Some(2),
                aip_attention_topk: None,
                aip_attention_locality_window: None,
                aip_attention_locality_extra: None,
                aip_mlp_topk: None,
                aip_down_topk: None,
                aip_edge_layers: None,
                aip_edge_topk: None,
                aip_exact_edge_layers: None,
                aip_exact_prefix_layers: None,
                aip_exact_periodic_layers: None,
                aip_layer_topk_overrides: [0; 128],
                aip_exact_edge_projection: None,
                aip_exact_layer: None,
                aip_exact_layer_projection: None,
                aip_lm_head_topk: None,
                aip_lm_head_rescore: None,
                aip_lm_head_rescore_gap_milli: None,
                aip_lm_head_agreement: false,
                aip_lm_head_rows: None,
                aip_lm_head_repeat_margin_milli: None,
                aip_lm_head_repeat_margin_adaptive: false,
                aip_lm_head_novelty_window: None,
                aip_lm_head_novelty_gap_milli: None,
                aip_lm_head_novelty_repeat_penalty_milli: None,
                aip_lm_head_novelty_retention_milli: None,
                aip_column_cache: false,
                aip_input_tiles: false,
                aip_no_repeat_last: false,
                aip_repeat_run_limit: None,
            },
            &mut stats,
            &mut budget,
        )
        .unwrap()
        .unwrap();

        assert_eq!(output.len(), 3);
        assert!((output[0] - 12.0).abs() < 1e-4);
        assert!((output[1] + 12.0).abs() < 1e-4);
        assert!((output[2] - 13.0).abs() < 1e-4);
        assert_eq!(stats.sparse_projection_calls, 1);
        assert_eq!(stats.max_selected_topk, 2);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn sparse_silu_gate_up_matches_manual_topk_projection() {
        let path = temp_path("sparse-gate-up-bf16");
        let gate_bf16 = vec![
            0x3f80, 0x4000, 0x4040, 0x4080, 0x4000, 0x4040, 0x4080, 0x40a0,
        ];
        let up_bf16 = vec![
            0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x4000, 0x4000, 0x4000, 0x4000,
        ];
        let input = vec![1.0, -8.0, 2.0, 7.0];

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "mlp.gate.sparse.bf16.weight",
            vec![2, 4],
            &gate_bf16,
            10,
        );
        add_bf16_tensor(
            &mut writer,
            1,
            "mlp.up.sparse.bf16.weight",
            vec![2, 4],
            &up_bf16,
            10,
        );
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::unbounded();
        let mut stats = crate::RamaExperimentalSpeedStats::default();
        let output = streaming_sparse_silu_gate_up_from_model(
            &mut model,
            "mlp.gate.sparse.bf16.weight",
            "mlp.up.sparse.bf16.weight",
            &input,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 4,
                    out_features: 2,
                },
                tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
            },
            crate::RamaExperimentalSpeedConfig {
                enabled: true,
                aip_policy: crate::RamaAipPolicyKind::Speed,
                aip_topk: Some(2),
                aip_attention_topk: None,
                aip_attention_locality_window: None,
                aip_attention_locality_extra: None,
                aip_mlp_topk: None,
                aip_down_topk: None,
                aip_edge_layers: None,
                aip_edge_topk: None,
                aip_exact_edge_layers: None,
                aip_exact_prefix_layers: None,
                aip_exact_periodic_layers: None,
                aip_layer_topk_overrides: [0; 128],
                aip_exact_edge_projection: None,
                aip_exact_layer: None,
                aip_exact_layer_projection: None,
                aip_lm_head_topk: None,
                aip_lm_head_rescore: None,
                aip_lm_head_rescore_gap_milli: None,
                aip_lm_head_agreement: false,
                aip_lm_head_rows: None,
                aip_lm_head_repeat_margin_milli: None,
                aip_lm_head_repeat_margin_adaptive: false,
                aip_lm_head_novelty_window: None,
                aip_lm_head_novelty_gap_milli: None,
                aip_lm_head_novelty_repeat_penalty_milli: None,
                aip_lm_head_novelty_retention_milli: None,
                aip_column_cache: false,
                aip_input_tiles: false,
                aip_no_repeat_last: false,
                aip_repeat_run_limit: None,
            },
            &mut stats,
            &mut budget,
        )
        .unwrap()
        .unwrap();

        let selected = crate::select_top_abs_indices(&input, 2);
        let gate_f32: Vec<f32> = gate_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let up_f32: Vec<f32> = up_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let mut expected = Vec::new();
        for row in 0..2 {
            let mut gate_acc = 0.0;
            let mut up_acc = 0.0;
            for &idx in &selected {
                gate_acc += input[idx] * gate_f32[row * 4 + idx];
                up_acc += input[idx] * up_f32[row * 4 + idx];
            }
            expected.push(crate::silu(gate_acc) * up_acc);
        }

        assert_close_vec(&output, &expected, 1e-4);
        assert_eq!(stats.sparse_projection_calls, 1);
        assert_eq!(stats.estimated_skipped_madds, 8);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn input_tiled_sparse_raw_bf16_linear_matches_manual_topk_projection() {
        let path = temp_path("input-tiled-sparse-linear-bf16");
        let weight_bf16 = vec![
            0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0xc040, 0xc080, 0x3f00, 0x3f80, 0x4000,
            0x4040,
        ];
        let input = vec![1.0, -8.0, 2.0, 7.0];

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "linear.input-tile.bf16.weight",
            vec![3, 4],
            &weight_bf16,
            14,
        );
        add_bf16_input_tile_sidecar_tensor(
            &mut writer,
            1,
            "linear.input-tile.bf16.weight",
            3,
            4,
            &weight_bf16,
            2,
        );
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        model.set_rama_integrity_mode(crate::RamaIntegrityMode::VerifyOnce);
        let mut budget = MemoryBudget::new(128);
        let mut stats = crate::RamaExperimentalSpeedStats::default();
        let output = streaming_input_tiled_sparse_tile_linear_from_model(
            &mut model,
            "linear.input-tile.bf16.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 4,
                    out_features: 3,
                },
                tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
            },
            crate::RamaExperimentalSpeedConfig {
                enabled: true,
                aip_policy: crate::RamaAipPolicyKind::Speed,
                aip_topk: Some(2),
                aip_attention_topk: None,
                aip_attention_locality_window: None,
                aip_attention_locality_extra: None,
                aip_mlp_topk: None,
                aip_down_topk: None,
                aip_edge_layers: None,
                aip_edge_topk: None,
                aip_exact_edge_layers: None,
                aip_exact_prefix_layers: None,
                aip_exact_periodic_layers: None,
                aip_layer_topk_overrides: [0; 128],
                aip_exact_edge_projection: None,
                aip_exact_layer: None,
                aip_exact_layer_projection: None,
                aip_lm_head_topk: None,
                aip_lm_head_rescore: None,
                aip_lm_head_rescore_gap_milli: None,
                aip_lm_head_agreement: false,
                aip_lm_head_rows: None,
                aip_lm_head_repeat_margin_milli: None,
                aip_lm_head_repeat_margin_adaptive: false,
                aip_lm_head_novelty_window: None,
                aip_lm_head_novelty_gap_milli: None,
                aip_lm_head_novelty_repeat_penalty_milli: None,
                aip_lm_head_novelty_retention_milli: None,
                aip_column_cache: false,
                aip_input_tiles: true,
                aip_no_repeat_last: false,
                aip_repeat_run_limit: None,
            },
            &mut stats,
            &mut budget,
        )
        .unwrap()
        .unwrap();

        assert_eq!(output.len(), 3);
        assert!((output[0] - 12.0).abs() < 1e-4);
        assert!((output[1] + 12.0).abs() < 1e-4);
        assert!((output[2] - 13.0).abs() < 1e-4);
        assert_eq!(stats.sparse_projection_calls, 1);
        assert_eq!(stats.input_tile_range_reads, 2);
        assert_eq!(stats.input_tile_range_bytes, 12);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn input_tiled_sparse_linear_uses_explicit_selected_indices() {
        let path = temp_path("input-tiled-sparse-linear-selected-bf16");
        let weight_bf16 = vec![
            0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0xc040, 0xc080, 0x3f00, 0x3f80, 0x4000,
            0x4040,
        ];
        let input = vec![1.0, -8.0, 2.0, 7.0];

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "linear.input-tile.selected.bf16.weight",
            vec![3, 4],
            &weight_bf16,
            14,
        );
        add_bf16_input_tile_sidecar_tensor(
            &mut writer,
            1,
            "linear.input-tile.selected.bf16.weight",
            3,
            4,
            &weight_bf16,
            2,
        );
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        model.set_rama_integrity_mode(crate::RamaIntegrityMode::VerifyOnce);
        let mut budget = MemoryBudget::new(128);
        let mut stats = crate::RamaExperimentalSpeedStats::default();
        let output = streaming_input_tiled_sparse_tile_linear_selected_from_model(
            &mut model,
            "linear.input-tile.selected.bf16.weight",
            &input,
            None,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 4,
                    out_features: 3,
                },
                tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
            },
            &[0, 3],
            &mut stats,
            &mut budget,
        )
        .unwrap()
        .unwrap();

        assert_eq!(output.len(), 3);
        assert!((output[0] - 29.0).abs() < 1e-4);
        assert!((output[1] + 29.0).abs() < 1e-4);
        assert!((output[2] - 21.5).abs() < 1e-4);
        assert_eq!(stats.sparse_projection_calls, 1);
        assert_eq!(stats.input_tile_range_reads, 2);
        assert_eq!(stats.input_tile_range_bytes, 12);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn input_tiled_sparse_silu_gate_up_matches_manual_topk_projection() {
        let path = temp_path("input-tiled-sparse-gate-up-bf16");
        let gate_bf16 = vec![
            0x3f80, 0x4000, 0x4040, 0x4080, 0x4000, 0x4040, 0x4080, 0x40a0,
        ];
        let up_bf16 = vec![
            0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x4000, 0x4000, 0x4000, 0x4000,
        ];
        let input = vec![1.0, -8.0, 2.0, 7.0];

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "mlp.gate.input-tile.bf16.weight",
            vec![2, 4],
            &gate_bf16,
            10,
        );
        add_bf16_tensor(
            &mut writer,
            1,
            "mlp.up.input-tile.bf16.weight",
            vec![2, 4],
            &up_bf16,
            10,
        );
        add_bf16_input_tile_sidecar_tensor(
            &mut writer,
            2,
            "mlp.gate.input-tile.bf16.weight",
            2,
            4,
            &gate_bf16,
            2,
        );
        add_bf16_input_tile_sidecar_tensor(
            &mut writer,
            3,
            "mlp.up.input-tile.bf16.weight",
            2,
            4,
            &up_bf16,
            2,
        );
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        model.set_rama_integrity_mode(crate::RamaIntegrityMode::VerifyOnce);
        let mut budget = MemoryBudget::new(128);
        let mut stats = crate::RamaExperimentalSpeedStats::default();
        let output = streaming_input_tiled_sparse_silu_gate_up_from_model(
            &mut model,
            "mlp.gate.input-tile.bf16.weight",
            "mlp.up.input-tile.bf16.weight",
            &input,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 4,
                    out_features: 2,
                },
                tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
            },
            crate::RamaExperimentalSpeedConfig {
                enabled: true,
                aip_policy: crate::RamaAipPolicyKind::Speed,
                aip_topk: Some(2),
                aip_attention_topk: None,
                aip_attention_locality_window: None,
                aip_attention_locality_extra: None,
                aip_mlp_topk: None,
                aip_down_topk: None,
                aip_edge_layers: None,
                aip_edge_topk: None,
                aip_exact_edge_layers: None,
                aip_exact_prefix_layers: None,
                aip_exact_periodic_layers: None,
                aip_layer_topk_overrides: [0; 128],
                aip_exact_edge_projection: None,
                aip_exact_layer: None,
                aip_exact_layer_projection: None,
                aip_lm_head_topk: None,
                aip_lm_head_rescore: None,
                aip_lm_head_rescore_gap_milli: None,
                aip_lm_head_agreement: false,
                aip_lm_head_rows: None,
                aip_lm_head_repeat_margin_milli: None,
                aip_lm_head_repeat_margin_adaptive: false,
                aip_lm_head_novelty_window: None,
                aip_lm_head_novelty_gap_milli: None,
                aip_lm_head_novelty_repeat_penalty_milli: None,
                aip_lm_head_novelty_retention_milli: None,
                aip_column_cache: false,
                aip_input_tiles: true,
                aip_no_repeat_last: false,
                aip_repeat_run_limit: None,
            },
            &mut stats,
            &mut budget,
        )
        .unwrap()
        .unwrap();

        let selected = crate::select_top_abs_indices(&input, 2);
        let gate_f32: Vec<f32> = gate_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let up_f32: Vec<f32> = up_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let mut expected = Vec::new();
        for row in 0..2 {
            let mut gate_acc = 0.0;
            let mut up_acc = 0.0;
            for &idx in &selected {
                gate_acc += input[idx] * gate_f32[row * 4 + idx];
                up_acc += input[idx] * up_f32[row * 4 + idx];
            }
            expected.push(crate::silu(gate_acc) * up_acc);
        }

        assert_close_vec(&output, &expected, 1e-4);
        assert_eq!(stats.sparse_projection_calls, 1);
        assert_eq!(stats.input_tile_range_reads, 4);
        assert_eq!(stats.input_tile_range_bytes, 16);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn column_cached_sparse_raw_bf16_linear_matches_manual_topk_and_reuses_columns() {
        let path = temp_path("column-cache-sparse-linear-bf16");
        let weight_bf16 = vec![
            0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0xc040, 0xc080, 0x3f00, 0x3f80, 0x4000,
            0x4040,
        ];
        let input = vec![1.0, -8.0, 2.0, 7.0];

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "linear.column-cache.bf16.weight",
            vec![3, 4],
            &weight_bf16,
            14,
        );
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::unbounded();
        let mut cache = SparseColumnCache::with_max_columns(8);
        let mut stats = crate::RamaExperimentalSpeedStats::default();
        let speed_config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
            aip_topk: Some(2),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: true,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };
        let config = StreamingTileLinearConfig {
            linear: StreamingLinearConfig {
                batch: 1,
                in_features: 4,
                out_features: 3,
            },
            tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
        };

        let output = streaming_column_cached_sparse_tile_linear_from_model(
            &mut model,
            "linear.column-cache.bf16.weight",
            &input,
            None,
            config,
            speed_config,
            &mut stats,
            &mut cache,
            &mut budget,
        )
        .unwrap()
        .unwrap();

        assert_eq!(output.len(), 3);
        assert!((output[0] - 12.0).abs() < 1e-4);
        assert!((output[1] + 12.0).abs() < 1e-4);
        assert!((output[2] - 13.0).abs() < 1e-4);
        assert_eq!(cache.stats().resident_columns, 2);
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(budget.current_bytes(), 0);

        let second = streaming_column_cached_sparse_tile_linear_from_model(
            &mut model,
            "linear.column-cache.bf16.weight",
            &input,
            None,
            config,
            speed_config,
            &mut stats,
            &mut cache,
            &mut budget,
        )
        .unwrap()
        .unwrap();

        assert_close_vec(&second, &output, 1e-6);
        assert_eq!(cache.stats().resident_columns, 2);
        assert_eq!(cache.stats().hits, 2);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn column_cached_sparse_silu_gate_up_matches_manual_topk_projection() {
        let path = temp_path("column-cache-sparse-gate-up-bf16");
        let gate_bf16 = vec![
            0x3f80, 0x4000, 0x4040, 0x4080, 0x4000, 0x4040, 0x4080, 0x40a0,
        ];
        let up_bf16 = vec![
            0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x4000, 0x4000, 0x4000, 0x4000,
        ];
        let input = vec![1.0, -8.0, 2.0, 7.0];

        let mut writer = RllmWriter::new(&path, GlobalMetadata::new_test()).unwrap();
        add_bf16_tensor(
            &mut writer,
            0,
            "mlp.gate.column-cache.bf16.weight",
            vec![2, 4],
            &gate_bf16,
            10,
        );
        add_bf16_tensor(
            &mut writer,
            1,
            "mlp.up.column-cache.bf16.weight",
            vec![2, 4],
            &up_bf16,
            10,
        );
        writer.finalize().unwrap();

        let mut model = LazyRllmModel::open(&path).unwrap();
        let mut budget = MemoryBudget::unbounded();
        let mut cache = SparseColumnCache::with_max_columns(8);
        let mut stats = crate::RamaExperimentalSpeedStats::default();
        let speed_config = crate::RamaExperimentalSpeedConfig {
            enabled: true,
            aip_policy: crate::RamaAipPolicyKind::Speed,
            aip_topk: Some(2),
            aip_attention_topk: None,
            aip_attention_locality_window: None,
            aip_attention_locality_extra: None,
            aip_mlp_topk: None,
            aip_down_topk: None,
            aip_edge_layers: None,
            aip_edge_topk: None,
            aip_exact_edge_layers: None,
            aip_exact_prefix_layers: None,
            aip_exact_periodic_layers: None,
            aip_layer_topk_overrides: [0; 128],
            aip_exact_edge_projection: None,
            aip_exact_layer: None,
            aip_exact_layer_projection: None,
            aip_lm_head_topk: None,
            aip_lm_head_rescore: None,
            aip_lm_head_rescore_gap_milli: None,
            aip_lm_head_agreement: false,
            aip_lm_head_rows: None,
            aip_lm_head_repeat_margin_milli: None,
            aip_lm_head_repeat_margin_adaptive: false,
            aip_lm_head_novelty_window: None,
            aip_lm_head_novelty_gap_milli: None,
            aip_lm_head_novelty_repeat_penalty_milli: None,
            aip_lm_head_novelty_retention_milli: None,
            aip_column_cache: true,
            aip_input_tiles: false,
            aip_no_repeat_last: false,
            aip_repeat_run_limit: None,
        };
        let output = streaming_column_cached_sparse_silu_gate_up_from_model(
            &mut model,
            "mlp.gate.column-cache.bf16.weight",
            "mlp.up.column-cache.bf16.weight",
            &input,
            StreamingTileLinearConfig {
                linear: StreamingLinearConfig {
                    batch: 1,
                    in_features: 4,
                    out_features: 2,
                },
                tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
            },
            speed_config,
            &mut stats,
            &mut cache,
            &mut budget,
        )
        .unwrap()
        .unwrap();

        let selected = crate::select_top_abs_indices(&input, 2);
        let gate_f32: Vec<f32> = gate_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let up_f32: Vec<f32> = up_bf16
            .iter()
            .map(|bits| crate::tensor::bf16_to_f32(*bits))
            .collect();
        let mut expected = Vec::new();
        for row in 0..2 {
            let mut gate_acc = 0.0;
            let mut up_acc = 0.0;
            for &idx in &selected {
                gate_acc += input[idx] * gate_f32[row * 4 + idx];
                up_acc += input[idx] * up_f32[row * 4 + idx];
            }
            expected.push(crate::silu(gate_acc) * up_acc);
        }

        assert_close_vec(&output, &expected, 1e-4);
        assert_eq!(cache.stats().resident_columns, 4);
        assert_eq!(cache.stats().misses, 4);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn parallel_sparse_raw_bf16_linear_matches_sequential_kernel() {
        let input = vec![1.0, -8.0, 2.0, 7.0];
        let selected = vec![1, 3];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 4,
            out_features: 8,
        };
        let weight_bf16: Vec<u16> = (0..config.out_features)
            .flat_map(|row| [0x3f80, 0x4000 + row as u16, 0x4040, 0x4080 + row as u16])
            .collect();
        let raw_bytes = bf16_bytes(&weight_bf16);
        let mut sequential = vec![0.0; config.out_features];
        let mut parallel = vec![0.0; config.out_features];

        accumulate_sparse_raw_16bit_linear_chunk_batch1(
            &input,
            &selected,
            &mut sequential,
            &raw_bytes,
            0,
            config,
            DType::Bf16,
            "linear.parallel.sparse.bf16.weight",
        )
        .unwrap();
        parallel_sparse_raw_16bit_linear_chunk_batch1(
            &input,
            &selected,
            &mut parallel,
            &raw_bytes,
            0,
            config,
            DType::Bf16,
            "linear.parallel.sparse.bf16.weight",
            4,
        )
        .unwrap();

        assert_close_vec(&parallel, &sequential, 1e-6);
    }

    #[test]
    fn parallel_sparse_silu_gate_up_bf16_matches_sequential_kernel() {
        let input = vec![1.0, -8.0, 2.0, 7.0];
        let selected = vec![1, 3];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 4,
            out_features: 8,
        };
        let gate_bf16: Vec<u16> = (0..config.out_features)
            .flat_map(|row| [0x3f80, 0x4000 + row as u16, 0x4040, 0x4080 + row as u16])
            .collect();
        let up_bf16: Vec<u16> = (0..config.out_features)
            .flat_map(|row| [0x3f80, 0x3f80, 0x4000 + row as u16, 0x4000 + row as u16])
            .collect();
        let gate_bytes = bf16_bytes(&gate_bf16);
        let up_bytes = bf16_bytes(&up_bf16);
        let mut sequential = vec![0.0; config.out_features];
        let mut parallel = vec![0.0; config.out_features];

        {
            let mut state = SiluGateUpState::new(&mut sequential);
            accumulate_sparse_silu_gate_up_raw_16bit_chunk_batch1(
                &input,
                &selected,
                &gate_bytes,
                &up_bytes,
                0,
                config,
                DType::Bf16,
                &mut state,
                "gate.parallel.sparse.bf16.weight",
            )
            .unwrap();
            state
                .finish(config, "gate.parallel.sparse.bf16.weight")
                .unwrap();
        }

        parallel_sparse_silu_gate_up_raw_16bit_chunk_batch1(
            &input,
            &selected,
            &gate_bytes,
            &up_bytes,
            0,
            config,
            DType::Bf16,
            &mut parallel,
            "gate.parallel.sparse.bf16.weight",
            4,
        )
        .unwrap();

        assert_close_vec(&parallel, &sequential, 1e-6);
    }

    #[test]
    fn parallel_sparse_silu_gate_up_single_worker_preserves_prior_output_rows() {
        let input = vec![1.0, -8.0, 2.0, 7.0];
        let selected = vec![1, 3];
        let config = StreamingLinearConfig {
            batch: 1,
            in_features: 4,
            out_features: 8,
        };
        let gate_bf16: Vec<u16> = (4..8)
            .flat_map(|row| [0x3f80, 0x4000 + row as u16, 0x4040, 0x4080 + row as u16])
            .collect();
        let up_bf16: Vec<u16> = (4..8)
            .flat_map(|row| [0x3f80, 0x3f80, 0x4000 + row as u16, 0x4000 + row as u16])
            .collect();
        let gate_bytes = bf16_bytes(&gate_bf16);
        let up_bytes = bf16_bytes(&up_bf16);
        let mut output = vec![99.0; config.out_features];

        parallel_sparse_silu_gate_up_raw_16bit_chunk_batch1(
            &input,
            &selected,
            &gate_bytes,
            &up_bytes,
            4 * config.in_features,
            config,
            DType::Bf16,
            &mut output,
            "gate.parallel.offset.sparse.bf16.weight",
            1,
        )
        .unwrap();

        assert_eq!(&output[..4], &[99.0, 99.0, 99.0, 99.0]);
        assert!(output[4..].iter().all(|value| *value != 99.0));
    }

    fn write_chunked_mlp(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "mlp.dense_h_to_4h.weight",
            vec![3, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, -1.0],
            12,
        );
        add_f32_tensor(
            &mut writer,
            1,
            "mlp.dense_4h_to_h.weight",
            vec![2, 3],
            &[1.0, 2.0, 3.0, -1.0, 0.5, 0.25],
            16,
        );
        writer.finalize().unwrap();
    }

    fn write_rle_zero_mlp(path: &std::path::Path, hidden_size: usize, intermediate_size: usize) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_rle_zero_tensor(
            &mut writer,
            0,
            "mlp.zero.dense_h_to_4h.weight",
            intermediate_size,
            hidden_size,
        );
        add_rle_zero_tensor(
            &mut writer,
            1,
            "mlp.zero.dense_4h_to_h.weight",
            hidden_size,
            intermediate_size,
        );
        writer.finalize().unwrap();
    }

    fn assert_close_vec(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= eps,
                "idx={idx}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn streaming_mlp_matches_full_decode_mlp_and_releases_intermediate_budget() {
        let path = temp_path("mlp");
        write_chunked_mlp(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 2.0, 0.25]; // [batch=2, hidden=2]
        let b_in = vec![0.1, -0.2, 0.3];
        let b_out = vec![0.05, -0.05];
        let expected = crate::mlp(
            &input,
            &[1.0, 0.0, 0.0, 1.0, 1.0, -1.0],
            Some(&b_in),
            &[1.0, 2.0, 3.0, -1.0, 0.5, 0.25],
            Some(&b_out),
            2,
            2,
            3,
        )
        .unwrap();
        let mut budget = MemoryBudget::new(160);

        let actual = streaming_mlp_from_model(
            &mut model,
            &input,
            "mlp.dense_h_to_4h.weight",
            Some(&b_in),
            "mlp.dense_4h_to_h.weight",
            Some(&b_out),
            StreamingMlpConfig {
                batch: 2,
                hidden_size: 2,
                intermediate_size: 3,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 80,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_mlp_uses_tiled_linear_to_fit_below_full_chunk_scratch_budget() {
        let path = temp_path("mlp-tiled-budget");
        write_rle_zero_mlp(&path, 128, 128);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 128];
        let mut budget = MemoryBudget::new(100_000);

        let actual = streaming_mlp_from_model(
            &mut model,
            &input,
            "mlp.zero.dense_h_to_4h.weight",
            None,
            "mlp.zero.dense_4h_to_h.weight",
            None,
            StreamingMlpConfig {
                batch: 1,
                hidden_size: 128,
                intermediate_size: 128,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, vec![0.0; 128]);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() < 100_000,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_mlp_rejects_too_small_intermediate_budget_without_leaking() {
        let path = temp_path("mlp-budget");
        write_chunked_mlp(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0];
        let mut budget = MemoryBudget::new(11);

        let err = streaming_mlp_from_model(
            &mut model,
            &input,
            "mlp.dense_h_to_4h.weight",
            None,
            "mlp.dense_4h_to_h.weight",
            None,
            StreamingMlpConfig {
                batch: 1,
                hidden_size: 2,
                intermediate_size: 3,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    fn write_chunked_attention(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "attention.query_key_value.weight",
            vec![6, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0],
            20,
        );
        add_f32_tensor(
            &mut writer,
            1,
            "attention.dense.weight",
            vec![2, 2],
            &[1.0, 0.5, -0.25, 1.0],
            8,
        );
        writer.finalize().unwrap();
    }

    fn write_rle_zero_attention(path: &std::path::Path, hidden_size: usize) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_rle_zero_tensor(
            &mut writer,
            0,
            "attention.zero.query_key_value.weight",
            3 * hidden_size,
            hidden_size,
        );
        add_rle_zero_tensor(
            &mut writer,
            1,
            "attention.zero.dense.weight",
            hidden_size,
            hidden_size,
        );
        writer.finalize().unwrap();
    }

    fn identity_qkv_weight(hidden_size: usize) -> Vec<f32> {
        let mut weight = vec![0.0f32; 3 * hidden_size * hidden_size];
        for block in 0..3 {
            for dim in 0..hidden_size {
                weight[(block * hidden_size + dim) * hidden_size + dim] = 1.0;
            }
        }
        weight
    }

    fn identity_weight(hidden_size: usize) -> Vec<f32> {
        let mut weight = vec![0.0f32; hidden_size * hidden_size];
        for dim in 0..hidden_size {
            weight[dim * hidden_size + dim] = 1.0;
        }
        weight
    }

    fn write_chunked_rotary_attention(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        let qkv_weight = identity_qkv_weight(4);
        let out_weight = identity_weight(4);
        add_f32_tensor(
            &mut writer,
            0,
            "attention.rotary_qkv.weight",
            vec![12, 4],
            &qkv_weight,
            96,
        );
        add_f32_tensor(
            &mut writer,
            1,
            "attention.rotary_dense.weight",
            vec![4, 4],
            &out_weight,
            32,
        );
        writer.finalize().unwrap();
    }

    fn split_fused_qkv_for_test(
        fused: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hidden = num_heads * head_dim;
        let mut q = vec![0.0f32; seq_len * hidden];
        let mut k = vec![0.0f32; seq_len * hidden];
        let mut v = vec![0.0f32; seq_len * hidden];
        for pos in 0..seq_len {
            let fused_row = pos * hidden * 3;
            let out_row = pos * hidden;
            for head in 0..num_heads {
                let fused_head = fused_row + head * head_dim * 3;
                let out_head = out_row + head * head_dim;
                q[out_head..out_head + head_dim]
                    .copy_from_slice(&fused[fused_head..fused_head + head_dim]);
                k[out_head..out_head + head_dim]
                    .copy_from_slice(&fused[fused_head + head_dim..fused_head + 2 * head_dim]);
                v[out_head..out_head + head_dim]
                    .copy_from_slice(&fused[fused_head + 2 * head_dim..fused_head + 3 * head_dim]);
            }
        }
        (q, k, v)
    }

    #[test]
    fn split_fused_qkv_uses_gpt_neox_per_head_layout() {
        let fused: Vec<f32> = (1..=24).map(|value| value as f32).collect();
        let (q, k, v) = split_fused_qkv(
            &fused,
            StreamingAttentionConfig {
                seq_len: 2,
                num_heads: 2,
                head_dim: 2,
                causal: true,
            },
        )
        .unwrap();

        assert_eq!(q, vec![1.0, 2.0, 7.0, 8.0, 13.0, 14.0, 19.0, 20.0]);
        assert_eq!(k, vec![3.0, 4.0, 9.0, 10.0, 15.0, 16.0, 21.0, 22.0]);
        assert_eq!(v, vec![5.0, 6.0, 11.0, 12.0, 17.0, 18.0, 23.0, 24.0]);
    }

    #[test]
    fn streaming_attention_matches_full_decode_qkv_attention_and_releases_budget() {
        let path = temp_path("attention");
        write_chunked_attention(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 1.25, 0.75]; // [seq=2, hidden=2]
        let qkv_bias = vec![0.1, -0.2, 0.0, 0.3, -0.1, 0.2];
        let out_bias = vec![0.05, -0.05];
        let qkv_weight = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
        let out_weight = [1.0, 0.5, -0.25, 1.0];
        let fused = linear(&input, &qkv_weight, Some(&qkv_bias), 2, 2, 6).unwrap();
        let (q, k, v) = split_fused_qkv_for_test(&fused, 2, 1, 2);
        let attended = crate::scaled_dot_product_attention(&q, &k, &v, 2, 1, 2, true).unwrap();
        let expected = linear(&attended, &out_weight, Some(&out_bias), 2, 2, 2).unwrap();
        let mut budget = MemoryBudget::new(256);

        let actual = streaming_attention_from_model(
            &mut model,
            &input,
            "attention.query_key_value.weight",
            Some(&qkv_bias),
            "attention.dense.weight",
            Some(&out_bias),
            StreamingAttentionConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 2,
                causal: true,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 128,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_attention_uses_tiled_linear_to_fit_below_full_chunk_scratch_budget() {
        let path = temp_path("attention-tiled-budget");
        write_rle_zero_attention(&path, 80);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![1.0f32; 80];
        let mut budget = MemoryBudget::new(112_000);

        let actual = streaming_attention_from_model(
            &mut model,
            &input,
            "attention.zero.query_key_value.weight",
            None,
            "attention.zero.dense.weight",
            None,
            StreamingAttentionConfig {
                seq_len: 1,
                num_heads: 1,
                head_dim: 80,
                causal: true,
            },
            &mut budget,
        )
        .unwrap();

        assert_eq!(actual, vec![0.0; 80]);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() < 112_000,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_attention_rejects_too_small_qkv_budget_without_leaking() {
        let path = temp_path("attention-budget");
        write_chunked_attention(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0];
        let mut budget = MemoryBudget::new(23);

        let err = streaming_attention_from_model(
            &mut model,
            &input,
            "attention.query_key_value.weight",
            None,
            "attention.dense.weight",
            None,
            StreamingAttentionConfig {
                seq_len: 1,
                num_heads: 1,
                head_dim: 2,
                causal: true,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_attention_with_rotary_and_kv_cache_matches_full_decode_last_token() {
        let path = temp_path("rotary-attention");
        write_chunked_rotary_attention(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let qkv_weight = identity_qkv_weight(4);
        let out_weight = identity_weight(4);
        let previous_input = [0.4, -0.2, 0.1, 0.3];
        let current_input = [0.7, 0.5, -0.4, 0.2];
        let mut full_input = Vec::new();
        full_input.extend_from_slice(&previous_input);
        full_input.extend_from_slice(&current_input);
        let fused = linear(&full_input, &qkv_weight, None, 2, 4, 12).unwrap();
        let (mut q, mut k, v) = split_fused_qkv_for_test(&fused, 2, 1, 4);
        crate::apply_gpt_neox_rotary_inplace(
            &mut q,
            &mut k,
            crate::RotaryEmbeddingConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 4,
                rotary_dim: 4,
                base: 10_000.0,
                position_offset: 0,
            },
        )
        .unwrap();
        let full_attended = crate::scaled_dot_product_attention(&q, &k, &v, 2, 1, 4, true).unwrap();
        let expected = linear(&full_attended[4..8], &out_weight, None, 1, 4, 4).unwrap();

        let mut cache = crate::KvCache::new(1, 4, 4).unwrap();
        cache.append(&k[..4], &v[..4], 1).unwrap();
        let mut budget = MemoryBudget::new(1024);
        let mut timing = StreamingBlockTiming::default();

        let actual = streaming_attention_with_runtime_and_timing_from_model(
            &mut model,
            &current_input,
            "attention.rotary_qkv.weight",
            None,
            "attention.rotary_dense.weight",
            None,
            StreamingAttentionConfig {
                seq_len: 1,
                num_heads: 1,
                head_dim: 4,
                causal: true,
            },
            StreamingAttentionRuntime {
                rotary: Some(crate::RotaryEmbeddingConfig {
                    seq_len: 1,
                    num_heads: 1,
                    head_dim: 4,
                    rotary_dim: 4,
                    base: 10_000.0,
                    position_offset: 1,
                }),
                kv_cache: Some(&mut cache),
            },
            &mut budget,
            Some(&mut timing),
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-6);
        assert_eq!(cache.len(), 2);
        assert_close_vec(cache.keys(), &k[..8], 1e-6);
        assert_close_vec(cache.values(), &v[..8], 1e-6);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(timing.attention_qkv_projection_calls, 1);
        assert_eq!(timing.attention_qkv_split_calls, 1);
        assert_eq!(timing.attention_rotary_calls, 1);
        assert_eq!(timing.attention_score_context_calls, 1);
        assert_eq!(timing.attention_output_projection_calls, 1);
        assert_eq!(timing.attention_kv_append_calls, 1);
        // NOTE: the per-phase `*_ns` durations are intentionally not asserted to be
        // > 0. On a fast machine a tiny phase (e.g. the kv-cache append or a small
        // memcpy) can complete below the timer's resolution and record 0 ns, which
        // made these assertions intermittently flaky. The `*_calls` counts above
        // already prove each phase actually ran.

        std::fs::remove_file(&path).ok();
    }

    fn write_chunked_block(path: &std::path::Path) {
        let mut writer = RllmWriter::new(path, GlobalMetadata::new_test()).unwrap();
        add_f32_tensor(
            &mut writer,
            0,
            "gpt_neox.layers.0.attention.query_key_value.weight",
            vec![6, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0],
            20,
        );
        add_f32_tensor(
            &mut writer,
            1,
            "gpt_neox.layers.0.attention.dense.weight",
            vec![2, 2],
            &[1.0, 0.5, -0.25, 1.0],
            8,
        );
        add_f32_tensor(
            &mut writer,
            2,
            "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
            vec![3, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, -1.0],
            12,
        );
        add_f32_tensor(
            &mut writer,
            3,
            "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
            vec![2, 3],
            &[1.0, 2.0, 3.0, -1.0, 0.5, 0.25],
            16,
        );
        writer.finalize().unwrap();
    }

    fn full_decode_block_baseline(input: &[f32], parallel_residual: bool) -> Vec<f32> {
        let ln1_weight = [1.1, 0.9];
        let ln1_bias = [0.05, -0.05];
        let qkv_bias = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2];
        let attention_out_bias = [0.05, -0.05];
        let ln2_weight = [0.8, 1.2];
        let ln2_bias = [-0.02, 0.04];
        let mlp_in_bias = [0.1, -0.2, 0.3];
        let mlp_out_bias = [0.05, -0.05];
        let qkv_weight = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
        let attention_out_weight = [1.0, 0.5, -0.25, 1.0];
        let mlp_in_weight = [1.0, 0.0, 0.0, 1.0, 1.0, -1.0];
        let mlp_out_weight = [1.0, 2.0, 3.0, -1.0, 0.5, 0.25];

        let attention_input = crate::layer_norm(input, &ln1_weight, &ln1_bias, 2, 2, 1e-5).unwrap();
        let fused = linear(&attention_input, &qkv_weight, Some(&qkv_bias), 2, 2, 6).unwrap();
        let (q, k, v) = split_fused_qkv_for_test(&fused, 2, 1, 2);
        let attended = crate::scaled_dot_product_attention(&q, &k, &v, 2, 1, 2, true).unwrap();
        let attention_out = linear(
            &attended,
            &attention_out_weight,
            Some(&attention_out_bias),
            2,
            2,
            2,
        )
        .unwrap();
        let mut residual = input.to_vec();
        crate::add_inplace(&mut residual, &attention_out).unwrap();

        let mlp_input_source = if parallel_residual {
            input
        } else {
            residual.as_slice()
        };
        let mlp_input =
            crate::layer_norm(mlp_input_source, &ln2_weight, &ln2_bias, 2, 2, 1e-5).unwrap();
        let mlp_out = crate::mlp(
            &mlp_input,
            &mlp_in_weight,
            Some(&mlp_in_bias),
            &mlp_out_weight,
            Some(&mlp_out_bias),
            2,
            2,
            3,
        )
        .unwrap();
        crate::add_inplace(&mut residual, &mlp_out).unwrap();
        residual
    }

    fn block_params_for_test<'a>() -> StreamingBlockParameters<'a> {
        static LN1_WEIGHT: [f32; 2] = [1.1, 0.9];
        static LN1_BIAS: [f32; 2] = [0.05, -0.05];
        static QKV_BIAS: [f32; 6] = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2];
        static ATTENTION_OUT_BIAS: [f32; 2] = [0.05, -0.05];
        static LN2_WEIGHT: [f32; 2] = [0.8, 1.2];
        static LN2_BIAS: [f32; 2] = [-0.02, 0.04];
        static MLP_IN_BIAS: [f32; 3] = [0.1, -0.2, 0.3];
        static MLP_OUT_BIAS: [f32; 2] = [0.05, -0.05];

        StreamingBlockParameters {
            input_layernorm_weight: &LN1_WEIGHT,
            input_layernorm_bias: &LN1_BIAS,
            qkv_bias: Some(&QKV_BIAS),
            attention_out_bias: Some(&ATTENTION_OUT_BIAS),
            post_attention_layernorm_weight: &LN2_WEIGHT,
            post_attention_layernorm_bias: &LN2_BIAS,
            mlp_in_bias: Some(&MLP_IN_BIAS),
            mlp_out_bias: Some(&MLP_OUT_BIAS),
        }
    }

    fn block_names_for_test<'a>() -> StreamingBlockTensorNames<'a> {
        StreamingBlockTensorNames {
            qkv_weight: "gpt_neox.layers.0.attention.query_key_value.weight",
            attention_out_weight: "gpt_neox.layers.0.attention.dense.weight",
            mlp_in_weight: "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
            mlp_out_weight: "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
        }
    }

    #[test]
    fn streaming_transformer_block_matches_full_decode_baseline_and_releases_budget() {
        let path = temp_path("block");
        write_chunked_block(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 1.25, 0.75];
        let expected = full_decode_block_baseline(&input, false);
        let mut budget = MemoryBudget::new(512);

        let actual = streaming_transformer_block_from_model(
            &mut model,
            &input,
            block_names_for_test(),
            block_params_for_test(),
            StreamingBlockConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 2,
                intermediate_size: 3,
                causal: true,
                layer_norm_eps: 1e-5,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-5);
        assert_eq!(budget.current_bytes(), 0);
        assert!(
            budget.peak_bytes() <= 256,
            "peak was {}",
            budget.peak_bytes()
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_transformer_block_supports_parallel_residual_baseline() {
        let path = temp_path("block_parallel_residual");
        write_chunked_block(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 1.25, 0.75];
        let expected = full_decode_block_baseline(&input, true);
        let mut budget = MemoryBudget::new(512);

        let actual = streaming_transformer_block_with_runtime_from_model(
            &mut model,
            &input,
            block_names_for_test(),
            block_params_for_test(),
            StreamingBlockConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 2,
                intermediate_size: 3,
                causal: true,
                layer_norm_eps: 1e-5,
            },
            StreamingBlockRuntime {
                attention: StreamingAttentionRuntime::default(),
                parallel_residual: true,
            },
            &mut budget,
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-5);
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_transformer_block_timing_records_each_subphase_once() {
        let path = temp_path("block-timing");
        write_chunked_block(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0, 1.25, 0.75];
        let expected = full_decode_block_baseline(&input, false);
        let mut budget = MemoryBudget::new(512);
        let mut timing = StreamingBlockTiming::default();

        let actual = streaming_transformer_block_with_runtime_and_timing_from_model(
            &mut model,
            &input,
            block_names_for_test(),
            block_params_for_test(),
            StreamingBlockConfig {
                seq_len: 2,
                num_heads: 1,
                head_dim: 2,
                intermediate_size: 3,
                causal: true,
                layer_norm_eps: 1e-5,
            },
            StreamingBlockRuntime::default(),
            &mut budget,
            Some(&mut timing),
        )
        .unwrap();

        assert_close_vec(&actual, &expected, 1e-5);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(timing.attention_norm_calls, 1);
        assert_eq!(timing.attention_calls, 1);
        assert_eq!(timing.attention_qkv_projection_calls, 1);
        assert_eq!(timing.attention_qkv_split_calls, 1);
        assert_eq!(timing.attention_score_context_calls, 1);
        assert_eq!(timing.attention_output_projection_calls, 1);
        assert_eq!(timing.attention_rotary_calls, 0);
        assert_eq!(timing.attention_kv_append_calls, 0);
        assert_eq!(timing.attention_residual_calls, 1);
        assert_eq!(timing.mlp_norm_calls, 1);
        assert_eq!(timing.mlp_calls, 1);
        assert_eq!(timing.mlp_input_projection_calls, 1);
        assert_eq!(timing.mlp_activation_calls, 1);
        assert_eq!(timing.mlp_output_projection_calls, 1);
        assert_eq!(timing.mlp_residual_calls, 1);
        // NOTE: the per-phase `*_ns` durations are intentionally not asserted to be
        // > 0. On a fast machine a tiny phase can complete below the timer's
        // resolution and record 0 ns, which made these assertions intermittently
        // flaky. The `*_calls` counts above already prove each phase actually ran.

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn streaming_transformer_block_rejects_too_small_attention_budget_without_leaking() {
        let path = temp_path("block-budget");
        write_chunked_block(&path);
        let mut model = LazyRllmModel::open(&path).unwrap();
        let input = vec![0.5, -1.0];
        let mut budget = MemoryBudget::new(31);

        let err = streaming_transformer_block_from_model(
            &mut model,
            &input,
            block_names_for_test(),
            block_params_for_test(),
            StreamingBlockConfig {
                seq_len: 1,
                num_heads: 1,
                head_dim: 2,
                intermediate_size: 3,
                causal: true,
                layer_norm_eps: 1e-5,
            },
            &mut budget,
        )
        .unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 0);

        std::fs::remove_file(&path).ok();
    }
}
