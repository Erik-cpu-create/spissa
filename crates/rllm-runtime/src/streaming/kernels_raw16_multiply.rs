// Multiply-into raw fp16 kernels + raw-16bit->f32 / fp16 weight-at helpers.
// Split out of kernels_raw16.rs (R169); include!d into streaming/mod.rs.

#[inline(always)]
fn raw_16bit_to_f32(bits: u16, dtype: rllm_container::DType) -> f32 {
    match dtype {
        rllm_container::DType::Fp16 => crate::tensor::fp16_to_f32(bits),
        rllm_container::DType::Bf16 => crate::tensor::bf16_to_f32(bits),
        _ => unreachable!("raw 16-bit kernel only supports FP16/BF16"),
    }
}

#[inline(always)]
fn fp16_weight_at(raw_bytes: &[u8], element_idx: usize) -> f32 {
    let byte_idx = element_idx * 2;
    crate::tensor::fp16_to_f32(u16::from_le_bytes([
        raw_bytes[byte_idx],
        raw_bytes[byte_idx + 1],
    ]))
}

fn accumulate_multiply_raw_fp16_chunk(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw FP16 stream for {weight_name} has odd length"
        )));
    }

    if config.batch == 1 {
        return accumulate_multiply_raw_fp16_chunk_batch1(
            input,
            raw_bytes,
            element_start,
            config,
            state,
            weight_name,
        );
    }

    let weight_elements = raw_bytes.len() / 2;
    let mut local_idx = 0usize;
    let mut global_idx = element_start;

    const BLOCK_SIZE: usize = 128;
    let mut w_block = [0.0f32; BLOCK_SIZE];

    while local_idx < weight_elements {
        let in_feature = global_idx % config.in_features;
        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);

        let mut row_idx = 0usize;
        while row_idx < row_len {
            let block_len = (row_len - row_idx).min(BLOCK_SIZE);
            let byte_start = (local_idx + row_idx) * 2;
            let block_bytes = &raw_bytes[byte_start..byte_start + block_len * 2];

            let mut idx = 0usize;
            while idx + 4 <= block_len {
                let bytes = &block_bytes[idx * 2..idx * 2 + 8];
                w_block[idx] = crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                w_block[idx + 1] =
                    crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
                w_block[idx + 2] =
                    crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[4], bytes[5]]));
                w_block[idx + 3] =
                    crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[6], bytes[7]]));
                idx += 4;
            }
            while idx < block_len {
                let bytes = &block_bytes[idx * 2..idx * 2 + 2];
                w_block[idx] = crate::tensor::fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                idx += 1;
            }

            accumulate_weight_chunk_multiply_into(
                input,
                &w_block[..block_len],
                global_idx + row_idx,
                config,
                state,
                weight_name,
            )?;

            row_idx += block_len;
        }

        local_idx += row_len;
        global_idx += row_len;
    }

    Ok(())
}

fn accumulate_multiply_raw_fp16_chunk_batch1(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<()> {
    accumulate_multiply_raw_fp16_chunk_batch1_row_blocked(
        input,
        raw_bytes,
        element_start,
        config,
        state,
        weight_name,
    )
}

fn accumulate_multiply_raw_fp16_chunk_batch1_row_blocked(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearMultiplyIntoState<'_>,
    weight_name: &str,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw FP16 stream for {weight_name} has odd length"
        )));
    }

    if config.batch != 1 {
        return Err(RuntimeError::Shape(format!(
            "raw FP16 batch1 multiply row-block kernel requires batch=1, got {}",
            config.batch
        )));
    }

    let expected_weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
    let weight_elements = raw_bytes.len() / 2;
    let element_end = element_start
        .checked_add(weight_elements)
        .ok_or_else(|| RuntimeError::Shape("weight chunk element range overflow".to_string()))?;
    if element_end > expected_weight_elements {
        return Err(RuntimeError::InvalidTensorData(format!(
            "weight tensor {weight_name} chunk elements {element_start}..{element_end} exceed expected {expected_weight_elements}"
        )));
    }

    let mut local_idx = 0usize;
    let mut global_idx = element_start;

    while local_idx < weight_elements && !global_idx.is_multiple_of(config.in_features) {
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

        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);
        let mut acc = state.current_acc[0];
        acc += raw_fp16_dot_segment(input, raw_bytes, local_idx, in_feature, row_len)?;
        state.current_acc[0] = acc;
        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }

    let row_block_elements = config
        .in_features
        .checked_mul(4)
        .ok_or_else(|| RuntimeError::Shape("row block element count overflow".to_string()))?;
    while local_idx + row_block_elements <= weight_elements {
        let out_feature = global_idx / config.in_features;
        if out_feature + 3 >= config.out_features {
            break;
        }
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let mut acc0 = state.current_acc[0];
        let mut acc1 = state
            .bias
            .and_then(|values| values.get(out_feature + 1))
            .copied()
            .unwrap_or(0.0);
        let mut acc2 = state
            .bias
            .and_then(|values| values.get(out_feature + 2))
            .copied()
            .unwrap_or(0.0);
        let mut acc3 = state
            .bias
            .and_then(|values| values.get(out_feature + 3))
            .copied()
            .unwrap_or(0.0);
        let row0_start = local_idx;
        let row1_start = local_idx + config.in_features;
        let row2_start = row1_start + config.in_features;
        let row3_start = row2_start + config.in_features;

        let mut idx = 0usize;
        while idx < config.in_features {
            let x = input[idx];
            acc0 += x * fp16_weight_at(raw_bytes, row0_start + idx);
            acc1 += x * fp16_weight_at(raw_bytes, row1_start + idx);
            acc2 += x * fp16_weight_at(raw_bytes, row2_start + idx);
            acc3 += x * fp16_weight_at(raw_bytes, row3_start + idx);
            idx += 1;
        }

        state.target[out_feature] *= acc0;
        state.target[out_feature + 1] *= acc1;
        state.target[out_feature + 2] *= acc2;
        state.target[out_feature + 3] *= acc3;
        state.current_out_feature += 4;
        if state.current_out_feature < config.out_features {
            let next = state
                .bias
                .and_then(|values| values.get(state.current_out_feature))
                .copied()
                .unwrap_or(0.0);
            state.current_acc[0] = next;
        }
        local_idx += row_block_elements;
        global_idx += row_block_elements;
    }

    while local_idx < weight_elements {
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

        let row_len = (config.in_features - in_feature).min(weight_elements - local_idx);
        let mut acc = state.current_acc[0];
        acc += raw_fp16_dot_segment(input, raw_bytes, local_idx, in_feature, row_len)?;
        state.current_acc[0] = acc;
        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }

    Ok(())
}

#[cfg(all(test, target_arch = "aarch64"))]
mod r119_panel_tests {
    use super::*;

    // Build a deterministic Q8_0 weight chunk: out_rows × in_features, each
    // 32-block = 2-byte fp16 scale (0.125) + 32 i8.
    pub fn make_q8_pub(out_rows: usize, in_features: usize) -> Vec<u8> { make_q8(out_rows, in_features) }
    fn make_q8(out_rows: usize, in_features: usize) -> Vec<u8> {
        let bpr = in_features / 32;
        let mut bytes = Vec::new();
        for o in 0..out_rows {
            for b in 0..bpr {
                bytes.extend_from_slice(&crate::tensor::f32_to_fp16(0.125).to_le_bytes());
                for k in 0..32 {
                    let q = (((o * 7 + b * 5 + k * 3) as i16 % 17) - 8) as i8;
                    bytes.push(q as u8);
                }
            }
        }
        bytes
    }

    pub fn make_input_pub(batch: usize, in_features: usize) -> Vec<f32> { make_input(batch, in_features) }
    fn make_input(batch: usize, in_features: usize) -> Vec<f32> {
        (0..batch * in_features)
            .map(|i| (i as f32 % 91.0) * 0.00390625 - 0.17)
            .collect()
    }

    fn run_panel_vs_r111(batch: usize, in_features: usize, out_features: usize) {
        if !q8_i8mm_available() {
            return;
        }
        let q8 = make_q8(out_features, in_features);
        let input = make_input(batch, in_features);
        let config = StreamingLinearConfig {
            batch,
            in_features,
            out_features,
        };
        let we = out_features * in_features;

        let mut out_ref = vec![0.0f32; batch * out_features];
        accumulate_q8_0_chunk_int8_activation(&input, &mut out_ref, &q8, 0, config, we).unwrap();

        let mut out_panel = vec![0.0f32; batch * out_features];
        let used =
            accumulate_q8_0_chunk_panel_smmla(&input, &mut out_panel, &q8, 0, config).unwrap();
        assert!(used, "panel path should engage for batch={batch}");

        let mut max_diff = 0.0f32;
        let mut worst = (0, 0);
        for t in 0..batch {
            for o in 0..out_features {
                let d = (out_ref[t * out_features + o] - out_panel[t * out_features + o]).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (t, o);
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "panel vs r111 mismatch batch={batch} out={out_features}: max_diff={max_diff} at row {} col {} (ref={} panel={})",
            worst.0,
            worst.1,
            out_ref[worst.0 * out_features + worst.1],
            out_panel[worst.0 * out_features + worst.1],
        );
    }

    #[test]
    fn panel_matches_r111_even_batch_even_out() {
        run_panel_vs_r111(4, 64, 4);
    }

    #[test]
    fn panel_matches_r111_odd_batch() {
        run_panel_vs_r111(3, 64, 4);
    }

    #[test]
    fn panel_matches_r111_odd_out() {
        run_panel_vs_r111(4, 64, 3);
    }

    #[test]
    fn panel_matches_r111_odd_both() {
        run_panel_vs_r111(5, 64, 5);
    }

    #[test]
    fn panel_matches_r111_realistic_shape() {
        run_panel_vs_r111(55, 2048, 8);
    }

    // R124 octet boundaries: exercise output-octet + pair-remainder + odd-row
    // tails for both even and odd batch. out_features chosen to hit each split:
    // 8=1 octet; 10=octet+pair; 11=octet+pair+odd; 17=2 octets+odd; 22=2oct+3pair.
    #[test]
    fn octet_even_batch_boundaries() {
        for out in [8, 9, 10, 11, 12, 15, 16, 17, 22, 24] {
            run_panel_vs_r111(54, 256, out);
        }
    }

    #[test]
    fn octet_odd_batch_boundaries() {
        for out in [8, 9, 10, 11, 12, 15, 16, 17, 22, 24] {
            run_panel_vs_r111(53, 256, out);
        }
    }

    #[test]
    fn octet_realistic_multi_octet() {
        run_panel_vs_r111(53, 2048, 64);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod r119_panel_multichunk_tests {
    use super::*;
    use super::r119_panel_tests::*;

    // Process a full matmul as several row-chunks (each chunk = a sub-range of
    // output rows at its own element_start), comparing panel vs R111.
    fn run_multichunk(batch: usize, in_features: usize, out_features: usize, chunk_rows: usize) {
        if !q8_i8mm_available() {
            return;
        }
        let bpr = in_features / 32;
        let q8_full = make_q8_pub(out_features, in_features);
        let input = make_input_pub(batch, in_features);
        let config = StreamingLinearConfig { batch, in_features, out_features };

        let mut out_ref = vec![0.0f32; batch * out_features];
        let mut out_panel = vec![0.0f32; batch * out_features];

        let we = out_features * in_features;
        let mut row = 0;
        while row < out_features {
            let rows = chunk_rows.min(out_features - row);
            let elem_start = row * in_features;
            let byte_start = row * bpr * 34;
            let byte_end = (row + rows) * bpr * 34;
            let chunk = &q8_full[byte_start..byte_end];

            accumulate_q8_0_chunk_int8_activation(&input, &mut out_ref, chunk, elem_start, config, we).unwrap();
            let used = accumulate_q8_0_chunk_panel_smmla(&input, &mut out_panel, chunk, elem_start, config).unwrap();
            assert!(used, "panel should engage chunk at row {row}");
            row += rows;
        }

        let mut max_diff = 0.0f32;
        let mut worst = (0, 0);
        for t in 0..batch {
            for o in 0..out_features {
                let d = (out_ref[t * out_features + o] - out_panel[t * out_features + o]).abs();
                if d > max_diff { max_diff = d; worst = (t, o); }
            }
        }
        assert!(max_diff < 1e-3,
            "multichunk panel vs r111 mismatch b={batch} out={out_features} chunk_rows={chunk_rows}: max_diff={max_diff} at row {} col {} (ref={} panel={})",
            worst.0, worst.1, out_ref[worst.0*out_features+worst.1], out_panel[worst.0*out_features+worst.1]);
    }

    #[test]
    fn multichunk_even_chunk_rows() { run_multichunk(55, 2048, 8, 4); }
    #[test]
    fn multichunk_odd_chunk_rows() { run_multichunk(55, 2048, 8, 3); }
    #[test]
    fn multichunk_single_row_chunks() { run_multichunk(55, 2048, 6, 1); }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod r119_panel_realchunk_tests {
    use super::*;
    use super::r119_panel_tests::*;

    // Replicate the real q_proj chunk pattern: batch=53, in=2048, out=2048,
    // chunks of 481,481,481,481,124 output rows.
    fn run_pattern(batch: usize, in_features: usize, out_features: usize, chunk_pattern: &[usize]) {
        if !q8_i8mm_available() { return; }
        assert_eq!(chunk_pattern.iter().sum::<usize>(), out_features);
        let bpr = in_features / 32;
        let q8_full = make_q8_pub(out_features, in_features);
        let input = make_input_pub(batch, in_features);
        let config = StreamingLinearConfig { batch, in_features, out_features };
        let we = out_features * in_features;

        let mut out_ref = vec![0.0f32; batch * out_features];
        let mut out_panel = vec![0.0f32; batch * out_features];
        let mut row = 0;
        for &rows in chunk_pattern {
            let elem_start = row * in_features;
            let chunk = &q8_full[row * bpr * 34..(row + rows) * bpr * 34];
            accumulate_q8_0_chunk_int8_activation(&input, &mut out_ref, chunk, elem_start, config, we).unwrap();
            let used = accumulate_q8_0_chunk_panel_smmla(&input, &mut out_panel, chunk, elem_start, config).unwrap();
            assert!(used, "panel should engage chunk at row {row} rows {rows}");
            row += rows;
        }
        let mut max_diff = 0.0f32; let mut worst = (0usize, 0usize);
        for t in 0..batch { for o in 0..out_features {
            let d = (out_ref[t*out_features+o]-out_panel[t*out_features+o]).abs();
            if d > max_diff { max_diff = d; worst = (t,o); }
        }}
        assert!(max_diff < 1e-3,
            "REAL pattern mismatch b={batch} out={out_features}: max_diff={max_diff} at row {} col {} (ref={} panel={})",
            worst.0, worst.1, out_ref[worst.0*out_features+worst.1], out_panel[worst.0*out_features+worst.1]);
    }

    #[test]
    fn real_qproj_b53() { run_pattern(53, 2048, 2048, &[481,481,481,481,124]); }
    #[test]
    fn real_kvproj_b53() { run_pattern(53, 2048, 512, &[481, 31]); }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod r119_panel_in8192_tests {
    use super::*;
    use super::r119_panel_tests::*;

    fn run(batch: usize, in_features: usize, out_features: usize) {
        if !q8_i8mm_available() { return; }
        let q8 = make_q8_pub(out_features, in_features);
        let input = make_input_pub(batch, in_features);
        let config = StreamingLinearConfig { batch, in_features, out_features };
        let we = out_features * in_features;
        let mut out_ref = vec![0.0f32; batch * out_features];
        accumulate_q8_0_chunk_int8_activation(&input, &mut out_ref, &q8, 0, config, we).unwrap();
        let mut out_panel = vec![0.0f32; batch * out_features];
        let used = accumulate_q8_0_chunk_panel_smmla(&input, &mut out_panel, &q8, 0, config).unwrap();
        assert!(used);
        let mut md = 0.0f32; let mut worst=(0,0);
        for t in 0..batch { for o in 0..out_features {
            let d=(out_ref[t*out_features+o]-out_panel[t*out_features+o]).abs();
            if d>md { md=d; worst=(t,o); }
        }}
        assert!(md < 1e-3, "in={in_features} mismatch: max_diff={md} at row {} col {} (ref={} panel={})",
            worst.0, worst.1, out_ref[worst.0*out_features+worst.1], out_panel[worst.0*out_features+worst.1]);
    }

    #[test] fn down_in8192() { run(53, 8192, 4); }
    #[test] fn down_in8192_realistic() { run(53, 8192, 64); }
}

/// R121: the multiply-into fast path (`try_panel_multiply_into_up` +
/// `target *= up + bias`). Validates the two pieces R121 actually adds on top of
/// the already-tested panel kernel: (1) accumulating the panel into a scratch
/// buffer chunk-by-chunk with a threaded `element_start`, and (2) the caller's
/// `target *= up + bias` arithmetic — both against the whole-weight int8
/// reference so the comparison is tight (same int8 dot on both sides), not a
/// quant-tolerance check.
#[cfg(all(test, target_arch = "aarch64"))]
mod r121_multiply_into_tests {
    use super::*;
    use super::r119_panel_tests::*;

    fn run(batch: usize, in_features: usize, out_features: usize, chunk_rows: usize) {
        if !q8_i8mm_available() {
            return;
        }
        let q8 = make_q8_pub(out_features, in_features);
        let input = make_input_pub(batch, in_features);
        let bias: Vec<f32> = (0..out_features).map(|f| (f as f32) * 0.013 - 0.07).collect();
        let config = StreamingLinearConfig { batch, in_features, out_features };

        // Reference up over the full weight via the int8-activation path.
        let we = out_features * in_features;
        let mut up_ref = vec![0.0f32; batch * out_features];
        accumulate_q8_0_chunk_int8_activation(&input, &mut up_ref, &q8, 0, config, we).unwrap();

        // Panel up accumulated chunk-by-chunk, mirroring try_panel_multiply_into_up:
        // each chunk covers `chunk_rows` output rows, with element_start derived
        // from the running byte offset exactly like chunk_element_start_for_dtype.
        let blocks_per_row = in_features / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut up_panel = vec![0.0f32; batch * out_features];
        let mut row = 0usize;
        let mut byte_offset = 0usize;
        while row < out_features {
            let rows = chunk_rows.min(out_features - row);
            let start = row * bytes_per_row;
            let end = start + rows * bytes_per_row;
            let element_start = (byte_offset / 34) * 32;
            let used = accumulate_q8_0_chunk_panel_smmla(
                &input,
                &mut up_panel,
                &q8[start..end],
                element_start,
                config,
            )
            .unwrap();
            assert!(used, "panel should engage for chunk at row {row}");
            byte_offset += end - start;
            row += rows;
        }

        // Apply the caller's multiply-into on both, then compare end results.
        let init: Vec<f32> = (0..batch * out_features)
            .map(|i| (i as f32 % 13.0) * 0.1 + 0.3)
            .collect();
        let mut max = 0.0f32;
        let mut worst = (0, 0);
        for b in 0..batch {
            for f in 0..out_features {
                let idx = b * out_features + f;
                let tgt_ref = init[idx] * (up_ref[idx] + bias[f]);
                let tgt_panel = init[idx] * (up_panel[idx] + bias[f]);
                let d = (tgt_ref - tgt_panel).abs();
                if d > max {
                    max = d;
                    worst = (b, f);
                }
            }
        }
        assert!(
            max < 1e-2,
            "multiply-into panel vs ref max_diff={max} at row {} col {}",
            worst.0,
            worst.1
        );
    }

    #[test] fn multiply_into_even() { run(4, 64, 8, 4); }
    #[test] fn multiply_into_odd_out_and_chunks() { run(5, 64, 7, 3); }
    #[test] fn multiply_into_realistic_up() { run(53, 2048, 16, 8); }
    #[test] fn multiply_into_single_chunk_full() { run(53, 2048, 32, 32); }
}
