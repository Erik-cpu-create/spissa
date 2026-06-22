// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use crate::rolling::RollingExecutor;

struct StreamingLinearArgmaxState<'a> {
    bias: Option<&'a [f32]>,
    current_out_feature: usize,
    current_acc: f32,
    best_index: usize,
    best_value: f32,
    seen: bool,
}

impl<'a> StreamingLinearArgmaxState<'a> {
    fn new(bias: Option<&'a [f32]>) -> Self {
        Self {
            bias,
            current_out_feature: 0,
            current_acc: bias
                .and_then(|values| values.first())
                .copied()
                .unwrap_or(0.0),
            best_index: 0,
            best_value: f32::NEG_INFINITY,
            seen: false,
        }
    }

    fn finish_current(&mut self, config: StreamingLinearConfig, weight_name: &str) -> Result<()> {
        if self.current_out_feature >= config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed more rows than expected {}",
                config.out_features
            )));
        }
        if !self.seen || self.current_acc > self.best_value {
            self.best_index = self.current_out_feature;
            self.best_value = self.current_acc;
            self.seen = true;
        }
        self.current_out_feature += 1;
        if self.current_out_feature < config.out_features {
            self.current_acc = self
                .bias
                .and_then(|values| values.get(self.current_out_feature))
                .copied()
                .unwrap_or(0.0);
        }
        Ok(())
    }

    fn finish(self, config: StreamingLinearConfig, weight_name: &str) -> Result<usize> {
        if self.current_out_feature != config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed {} complete rows, expected {}",
                self.current_out_feature, config.out_features
            )));
        }
        if !self.seen {
            return Err(RuntimeError::InvalidTensorData(
                "cannot argmax empty streaming linear output".to_string(),
            ));
        }
        Ok(self.best_index)
    }
}

#[derive(Debug, Clone, Copy)]
struct ArgmaxCandidate {
    best_index: usize,
    best_value: f32,
    seen: bool,
}

impl ArgmaxCandidate {
    fn empty() -> Self {
        Self {
            best_index: 0,
            best_value: f32::NEG_INFINITY,
            seen: false,
        }
    }

    fn observe(&mut self, index: usize, value: f32) {
        if !self.seen || value > self.best_value {
            self.best_index = index;
            self.best_value = value;
            self.seen = true;
        }
    }

    fn merge(&mut self, other: Self) {
        if other.seen {
            self.observe(other.best_index, other.best_value);
        }
    }
}

fn parallel_raw_16bit_argmax_rows(
    input: &[f32],
    raw_bytes: &[u8],
    local_row_start: usize,
    out_feature_start: usize,
    rows: usize,
    config: StreamingLinearConfig,
    dtype: spissa_container::DType,
    bias: Option<&[f32]>,
    threads: usize,
) -> ArgmaxCandidate {
    let threads = effective_row_block_threads(rows, threads);
    if threads == 1 {
        return raw_16bit_argmax_rows_range(
            input,
            raw_bytes,
            local_row_start,
            out_feature_start,
            rows,
            config,
            dtype,
            bias,
        );
    }

    let rows_per_thread = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for thread_idx in 0..threads {
            let row_offset = thread_idx * rows_per_thread;
            if row_offset >= rows {
                break;
            }
            let row_count = rows_per_thread.min(rows - row_offset);
            handles.push(scope.spawn(move || {
                raw_16bit_argmax_rows_range(
                    input,
                    raw_bytes,
                    local_row_start + row_offset * config.in_features,
                    out_feature_start + row_offset,
                    row_count,
                    config,
                    dtype,
                    bias,
                )
            }));
        }

        let mut best = ArgmaxCandidate::empty();
        for handle in handles {
            best.merge(handle.join().expect("argmax worker panicked"));
        }
        best
    })
}

fn rolling_raw_16bit_argmax_rows(
    input: &[f32],
    raw_bytes: &[u8],
    local_row_start: usize,
    out_feature_start: usize,
    rows: usize,
    config: StreamingLinearConfig,
    dtype: spissa_container::DType,
    bias: Option<&[f32]>,
    executor: &mut RollingExecutor,
) -> ArgmaxCandidate {
    let workers = executor.effective_workers_for_rows(rows);
    if workers == 1 {
        executor.record_sequential_fallback();
        return raw_16bit_argmax_rows_range(
            input,
            raw_bytes,
            local_row_start,
            out_feature_start,
            rows,
            config,
            dtype,
            bias,
        );
    }

    executor.record_parallel_batch(workers, std::mem::size_of::<ArgmaxCandidate>() * workers);
    parallel_raw_16bit_argmax_rows(
        input,
        raw_bytes,
        local_row_start,
        out_feature_start,
        rows,
        config,
        dtype,
        bias,
        workers,
    )
}

fn raw_16bit_argmax_rows_range(
    input: &[f32],
    raw_bytes: &[u8],
    local_row_start: usize,
    out_feature_start: usize,
    rows: usize,
    config: StreamingLinearConfig,
    dtype: spissa_container::DType,
    bias: Option<&[f32]>,
) -> ArgmaxCandidate {
    // In --fast (int8-activation) mode, bf16 rows use the NEON dot kernel (exact
    // (bits<<16) upcast + vectorized FMA — the R137 lm_head kernel). This is the
    // dominant decode cost for tied-bf16-embedding models (LLaMA/Gemma). The exact
    // default keeps the scalar accumulation so its bit-for-bit tests hold.
    let fast_bf16 = q8_activation_path_enabled() && matches!(dtype, spissa_container::DType::Bf16);
    let n = config.in_features;
    let bf16_act = fast_bf16.then(|| Bf16DotActivation::new(input));
    let mut best = ArgmaxCandidate::empty();
    for row_idx in 0..rows {
        let out_feature = out_feature_start + row_idx;
        let row_start = local_row_start + row_idx * config.in_features;
        let mut acc = bias
            .and_then(|values| values.get(out_feature))
            .copied()
            .unwrap_or(0.0);
        if let Some(act) = &bf16_act {
            acc += act.row_dot(&raw_bytes[row_start * 2..(row_start + n) * 2], n);
        } else {
            let mut input_idx = 0usize;
            while input_idx < n {
                acc += input[input_idx] * raw_16bit_weight_at(raw_bytes, row_start + input_idx, dtype);
                input_idx += 1;
            }
        }
        best.observe(out_feature, acc);
    }
    best
}

struct StreamingLinearMultiplyIntoState<'a> {
    target: &'a mut [f32],
    bias: Option<&'a [f32]>,
    current_out_feature: usize,
    current_acc: Vec<f32>,
}

impl<'a> StreamingLinearMultiplyIntoState<'a> {
    fn new(target: &'a mut [f32], bias: Option<&'a [f32]>, config: StreamingLinearConfig) -> Self {
        let initial = bias
            .and_then(|values| values.first())
            .copied()
            .unwrap_or(0.0);
        Self {
            target,
            bias,
            current_out_feature: 0,
            current_acc: vec![initial; config.batch],
        }
    }

    fn finish_current(&mut self, config: StreamingLinearConfig, weight_name: &str) -> Result<()> {
        if self.current_out_feature >= config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed more rows than expected {}",
                config.out_features
            )));
        }
        for batch_idx in 0..config.batch {
            let target_idx = batch_idx * config.out_features + self.current_out_feature;
            self.target[target_idx] *= self.current_acc[batch_idx];
        }
        self.current_out_feature += 1;
        if self.current_out_feature < config.out_features {
            let next = self
                .bias
                .and_then(|values| values.get(self.current_out_feature))
                .copied()
                .unwrap_or(0.0);
            self.current_acc.fill(next);
        }
        Ok(())
    }

    fn finish(self, config: StreamingLinearConfig, weight_name: &str) -> Result<()> {
        if self.current_out_feature != config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed {} complete rows, expected {}",
                self.current_out_feature, config.out_features
            )));
        }
        Ok(())
    }
}

struct SiluGateUpState<'a> {
    output: &'a mut [f32],
    current_out_feature: usize,
    gate_acc: f32,
    up_acc: f32,
}

impl<'a> SiluGateUpState<'a> {
    fn new(output: &'a mut [f32]) -> Self {
        Self {
            output,
            current_out_feature: 0,
            gate_acc: 0.0,
            up_acc: 0.0,
        }
    }

    fn finish_current(&mut self, config: StreamingLinearConfig, weight_name: &str) -> Result<()> {
        if self.current_out_feature >= config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed more rows than expected {}",
                config.out_features
            )));
        }
        self.output[self.current_out_feature] = crate::ops::silu(self.gate_acc) * self.up_acc;
        self.current_out_feature += 1;
        self.gate_acc = 0.0;
        self.up_acc = 0.0;
        Ok(())
    }

    fn finish(self, config: StreamingLinearConfig, weight_name: &str) -> Result<()> {
        if self.current_out_feature != config.out_features {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed {} complete rows, expected {}",
                self.current_out_feature, config.out_features
            )));
        }
        Ok(())
    }
}

fn accumulate_weight_chunk_argmax(
    input: &[f32],
    weights: &[f32],
    element_start: usize,
    config: StreamingLinearConfig,
    state: &mut StreamingLinearArgmaxState<'_>,
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
        let input_row = &input[in_feature..in_feature + row_len];
        let mut idx = 0;
        while idx + 4 <= row_len {
            let w = &weight_row[idx..idx + 4];
            let i_row = &input_row[idx..idx + 4];
            state.current_acc +=
                w[0] * i_row[0] + w[1] * i_row[1] + w[2] * i_row[2] + w[3] * i_row[3];
            idx += 4;
        }
        while idx < row_len {
            state.current_acc += input_row[idx] * weight_row[idx];
            idx += 1;
        }

        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }
    Ok(())
}

fn accumulate_raw_16bit_chunk_argmax(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: spissa_container::DType,
    state: &mut StreamingLinearArgmaxState<'_>,
    weight_name: &str,
    rolling: Option<&mut RollingExecutor>,
) -> Result<()> {
    accumulate_raw_16bit_chunk_argmax_row_blocked(
        input,
        raw_bytes,
        element_start,
        config,
        dtype,
        state,
        weight_name,
        rolling,
    )
}

fn accumulate_raw_16bit_chunk_argmax_row_blocked(
    input: &[f32],
    raw_bytes: &[u8],
    element_start: usize,
    config: StreamingLinearConfig,
    dtype: spissa_container::DType,
    state: &mut StreamingLinearArgmaxState<'_>,
    weight_name: &str,
    rolling: Option<&mut RollingExecutor>,
) -> Result<()> {
    if !raw_bytes.len().is_multiple_of(2) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw 16-bit argmax stream for {weight_name} has odd length"
        )));
    }
    if !matches!(
        dtype,
        spissa_container::DType::Fp16 | spissa_container::DType::Bf16
    ) {
        return Err(RuntimeError::InvalidTensorData(format!(
            "Raw 16-bit argmax stream for {weight_name} has unsupported dtype {dtype:?}"
        )));
    }

    let weight_elements = raw_bytes.len() / 2;
    let expected_weight_elements = config
        .out_features
        .checked_mul(config.in_features)
        .ok_or_else(|| RuntimeError::Shape("weight element count overflow".to_string()))?;
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
        state.current_acc +=
            raw_16bit_dot_segment(input, raw_bytes, local_idx, in_feature, row_len, dtype)?;

        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }

    let full_rows = ((weight_elements - local_idx) / config.in_features)
        .min(config.out_features - (global_idx / config.in_features));
    let worker_count =
        effective_row_block_threads(full_rows, argmax_runtime_thread_count(config.out_features));
    if worker_count > 1 {
        let out_feature = global_idx / config.in_features;
        while state.current_out_feature < out_feature {
            state.finish_current(config, weight_name)?;
        }
        if state.current_out_feature != out_feature {
            return Err(RuntimeError::InvalidTensorData(format!(
                "weight tensor {weight_name} streamed non-monotonic row {}, current {}",
                out_feature, state.current_out_feature
            )));
        }

        let candidate = if let Some(executor) = rolling {
            rolling_raw_16bit_argmax_rows(
                input,
                raw_bytes,
                local_idx,
                out_feature,
                full_rows,
                config,
                dtype,
                state.bias,
                executor,
            )
        } else {
            parallel_raw_16bit_argmax_rows(
                input,
                raw_bytes,
                local_idx,
                out_feature,
                full_rows,
                config,
                dtype,
                state.bias,
                worker_count,
            )
        };
        if candidate.seen && (!state.seen || candidate.best_value > state.best_value) {
            state.best_index = candidate.best_index;
            state.best_value = candidate.best_value;
            state.seen = true;
        }

        let consumed = full_rows.checked_mul(config.in_features).ok_or_else(|| {
            RuntimeError::Shape("parallel argmax consumed rows overflow".to_string())
        })?;
        local_idx += consumed;
        global_idx += consumed;
        state.current_out_feature += full_rows;
        if state.current_out_feature < config.out_features {
            state.current_acc = state
                .bias
                .and_then(|values| values.get(state.current_out_feature))
                .copied()
                .unwrap_or(0.0);
        }
    }

    let row_block_elements = config.in_features.checked_mul(4).ok_or_else(|| {
        RuntimeError::Shape("argmax row block element count overflow".to_string())
    })?;
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

        let row0_start = local_idx;
        let row1_start = row0_start + config.in_features;
        let row2_start = row1_start + config.in_features;
        let row3_start = row2_start + config.in_features;
        let mut acc0 = state.current_acc;
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

        let mut idx = 0usize;
        while idx < config.in_features {
            let x = input[idx];
            acc0 += x * raw_16bit_weight_at(raw_bytes, row0_start + idx, dtype);
            acc1 += x * raw_16bit_weight_at(raw_bytes, row1_start + idx, dtype);
            acc2 += x * raw_16bit_weight_at(raw_bytes, row2_start + idx, dtype);
            acc3 += x * raw_16bit_weight_at(raw_bytes, row3_start + idx, dtype);
            idx += 1;
        }

        state.current_acc = acc0;
        state.finish_current(config, weight_name)?;
        state.current_acc = acc1;
        state.finish_current(config, weight_name)?;
        state.current_acc = acc2;
        state.finish_current(config, weight_name)?;
        state.current_acc = acc3;
        state.finish_current(config, weight_name)?;

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
        state.current_acc +=
            raw_16bit_dot_segment(input, raw_bytes, local_idx, in_feature, row_len, dtype)?;

        local_idx += row_len;
        global_idx += row_len;
        if global_idx.is_multiple_of(config.in_features) {
            state.finish_current(config, weight_name)?;
        }
    }
    Ok(())
}
