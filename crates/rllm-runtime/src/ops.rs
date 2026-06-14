use crate::{Result, RuntimeError};

pub fn embedding_lookup(
    embedding: &[f32],
    vocab_size: usize,
    hidden_size: usize,
    token_ids: &[usize],
) -> Result<Vec<f32>> {
    if embedding.len() != vocab_size * hidden_size {
        return Err(RuntimeError::Shape(format!(
            "embedding len {} does not match vocab_size {vocab_size} * hidden_size {hidden_size}",
            embedding.len()
        )));
    }

    let mut out = Vec::with_capacity(token_ids.len() * hidden_size);
    for &token_id in token_ids {
        if token_id >= vocab_size {
            return Err(RuntimeError::Shape(format!(
                "token id {token_id} out of range for vocab size {vocab_size}"
            )));
        }
        let start = token_id * hidden_size;
        out.extend_from_slice(&embedding[start..start + hidden_size]);
    }
    Ok(out)
}

/// Row-major matrix multiply: A[m,k] × B[k,n] = C[m,n].
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    if a.len() != m * k {
        return Err(RuntimeError::Shape(format!(
            "left matrix len {} does not match m*k = {}",
            a.len(),
            m * k
        )));
    }
    if b.len() != k * n {
        return Err(RuntimeError::Shape(format!(
            "right matrix len {} does not match k*n = {}",
            b.len(),
            k * n
        )));
    }

    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    Ok(out)
}

/// PyTorch-style linear layer: input[batch,in] × weight[out,in]^T + bias[out].
pub fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    if input.len() != batch * in_features {
        return Err(RuntimeError::Shape(format!(
            "input len {} does not match batch*in_features = {}",
            input.len(),
            batch * in_features
        )));
    }
    if weight.len() != out_features * in_features {
        return Err(RuntimeError::Shape(format!(
            "weight len {} does not match out_features*in_features = {}",
            weight.len(),
            out_features * in_features
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != out_features {
            return Err(RuntimeError::Shape(format!(
                "bias len {} does not match out_features {out_features}",
                bias.len()
            )));
        }
    }

    let mut out = vec![0.0f32; batch * out_features];
    for row in 0..batch {
        for out_col in 0..out_features {
            let mut sum = bias.map(|b| b[out_col]).unwrap_or(0.0);
            for in_col in 0..in_features {
                sum += input[row * in_features + in_col] * weight[out_col * in_features + in_col];
            }
            out[row * out_features + out_col] = sum;
        }
    }
    Ok(out)
}

pub fn add_inplace(dst: &mut [f32], src: &[f32]) -> Result<()> {
    if dst.len() != src.len() {
        return Err(RuntimeError::Shape(format!(
            "add len mismatch: dst={}, src={}",
            dst.len(),
            src.len()
        )));
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
    Ok(())
}

pub fn layer_norm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if x.len() != rows * cols || gamma.len() != cols || beta.len() != cols {
        return Err(RuntimeError::Shape(format!(
            "layer_norm shape mismatch: x={}, rows={rows}, cols={cols}, gamma={}, beta={}",
            x.len(),
            gamma.len(),
            beta.len()
        )));
    }

    let mut out = vec![0.0f32; x.len()];
    for row in 0..rows {
        let start = row * cols;
        let row_slice = &x[start..start + cols];
        let mean = row_slice.iter().sum::<f32>() / cols as f32;
        let variance = row_slice
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / cols as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for col in 0..cols {
            out[start + col] = (x[start + col] - mean) * inv_std * gamma[col] + beta[col];
        }
    }
    Ok(out)
}

pub fn rms_norm(x: &[f32], weight: &[f32], rows: usize, cols: usize, eps: f32) -> Result<Vec<f32>> {
    if x.len() != rows * cols || weight.len() != cols {
        return Err(RuntimeError::Shape(format!(
            "rms_norm shape mismatch: x={}, rows={rows}, cols={cols}, weight={}",
            x.len(),
            weight.len()
        )));
    }

    let mut out = vec![0.0f32; x.len()];
    for row in 0..rows {
        let start = row * cols;
        let row_slice = &x[start..start + cols];
        let mean_square = row_slice.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (mean_square + eps).sqrt();
        for col in 0..cols {
            out[start + col] = x[start + col] * inv_rms * weight[col];
        }
    }
    Ok(out)
}

pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

pub fn gelu_inplace(values: &mut [f32]) {
    for value in values {
        *value = gelu(*value);
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn silu_inplace(values: &mut [f32]) {
    for value in values {
        *value = silu(*value);
    }
}

pub fn softmax_rows(logits: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    if logits.len() != rows * cols {
        return Err(RuntimeError::Shape(format!(
            "softmax len {} does not match rows*cols = {}",
            logits.len(),
            rows * cols
        )));
    }

    let mut out = vec![0.0f32; logits.len()];
    for row in 0..rows {
        let start = row * cols;
        let row_slice = &logits[start..start + cols];
        let max = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for col in 0..cols {
            let exp = (logits[start + col] - max).exp();
            out[start + col] = exp;
            sum += exp;
        }
        if sum == 0.0 || !sum.is_finite() {
            return Err(RuntimeError::InvalidTensorData(
                "softmax produced invalid denominator".to_string(),
            ));
        }
        for col in 0..cols {
            out[start + col] /= sum;
        }
    }
    Ok(out)
}

/// Scaled dot-product attention.
///
/// Layout for `q`, `k`, `v`, and output is `[seq_len, num_heads, head_dim]`.
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    let expected = seq_len * num_heads * head_dim;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(RuntimeError::Shape(format!(
            "attention shape mismatch: expected {expected}, got q={}, k={}, v={}",
            q.len(),
            k.len(),
            v.len()
        )));
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; expected];
    let mut scores = vec![0.0f32; seq_len];

    for pos in 0..seq_len {
        for head in 0..num_heads {
            for (key_pos, score) in scores.iter_mut().enumerate().take(seq_len) {
                if causal && key_pos > pos {
                    *score = f32::NEG_INFINITY;
                    continue;
                }

                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    let q_idx = ((pos * num_heads + head) * head_dim) + dim;
                    let k_idx = ((key_pos * num_heads + head) * head_dim) + dim;
                    dot += q[q_idx] * k[k_idx];
                }
                *score = dot * scale;
            }

            let probs = softmax_rows(&scores, 1, seq_len)?;
            for dim in 0..head_dim {
                let out_idx = ((pos * num_heads + head) * head_dim) + dim;
                let mut value = 0.0f32;
                for (key_pos, prob) in probs.iter().copied().enumerate().take(seq_len) {
                    let v_idx = ((key_pos * num_heads + head) * head_dim) + dim;
                    value += prob * v[v_idx];
                }
                out[out_idx] = value;
            }
        }
    }

    Ok(out)
}

/// Two-layer feed-forward network: Linear → GELU → Linear.
pub fn mlp(
    input: &[f32],
    w_in: &[f32],
    b_in: Option<&[f32]>,
    w_out: &[f32],
    b_out: Option<&[f32]>,
    batch: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<Vec<f32>> {
    let mut hidden = linear(input, w_in, b_in, batch, hidden_size, intermediate_size)?;
    gelu_inplace(&mut hidden);
    linear(&hidden, w_out, b_out, batch, intermediate_size, hidden_size)
}

pub fn sample_argmax(logits: &[f32]) -> Result<usize> {
    if logits.is_empty() {
        return Err(RuntimeError::InvalidTensorData(
            "cannot sample from empty logits".to_string(),
        ));
    }
    let mut best_index = 0usize;
    let mut best_value = logits[0];
    for (idx, &value) in logits.iter().enumerate().skip(1) {
        if value > best_value {
            best_index = idx;
            best_value = value;
        }
    }
    Ok(best_index)
}

pub fn sample_argmax_excluding(logits: &[f32], excluded_token: Option<usize>) -> Result<usize> {
    let Some(excluded_token) = excluded_token else {
        return sample_argmax(logits);
    };
    if excluded_token >= logits.len() || logits.len() <= 1 {
        return sample_argmax(logits);
    }

    let mut best: Option<(usize, f32)> = None;
    for (idx, &value) in logits.iter().enumerate() {
        if idx == excluded_token {
            continue;
        }
        match best {
            Some((_, best_value)) if value <= best_value => {}
            _ => best = Some((idx, value)),
        }
    }

    best.map(|(idx, _)| idx).ok_or_else(|| {
        RuntimeError::InvalidTensorData("cannot sample from empty logits".to_string())
    })
}

/// Deterministic temperature + top-p sampling.
///
/// Uses a tiny in-house xorshift PRNG so tests and prompts can be reproduced
/// without adding a random-number dependency.
pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32, seed: u64) -> Result<usize> {
    if logits.is_empty() {
        return Err(RuntimeError::InvalidTensorData(
            "cannot sample from empty logits".to_string(),
        ));
    }
    if temperature <= 0.0 {
        return sample_argmax(logits);
    }
    if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "top_p must be in (0, 1], got {top_p}"
        )));
    }

    let scaled: Vec<f32> = logits.iter().map(|v| *v / temperature).collect();
    let probs = softmax_rows(&scaled, 1, logits.len())?;
    let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut nucleus = Vec::new();
    let mut cumulative = 0.0f32;
    for item in indexed {
        cumulative += item.1;
        nucleus.push(item);
        if cumulative >= top_p {
            break;
        }
    }

    let total = nucleus.iter().map(|(_, p)| *p).sum::<f32>();
    let mut threshold = seeded_unit_f32(seed) * total;
    for (idx, prob) in nucleus {
        if threshold <= prob {
            return Ok(idx);
        }
        threshold -= prob;
    }

    sample_argmax(logits)
}

fn seeded_unit_f32(seed: u64) -> f32 {
    let mut x = if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    };
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let x = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
    ((x >> 40) as u32) as f32 / ((1u32 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_vec(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*a - *e).abs() <= eps,
                "idx={idx}: actual={a}, expected={e}"
            );
        }
    }

    #[test]
    fn embedding_lookup_returns_rows() {
        let embedding = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = embedding_lookup(&embedding, 3, 2, &[2, 0]).unwrap();
        assert_eq!(out, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn matmul_computes_row_major_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let out = matmul(&a, &b, 2, 2, 2).unwrap();
        assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn linear_uses_transposed_weight_layout() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 0.0, 1.0, 0.5, 0.5, 0.5]; // 2x3
        let bias = vec![1.0, -1.0];
        let out = linear(&input, &weight, Some(&bias), 1, 3, 2).unwrap();
        assert_eq!(out, vec![5.0, 2.0]);
    }

    #[test]
    fn layer_norm_normalizes_each_row() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let out = layer_norm(&x, &gamma, &beta, 2, 2, 1e-5).unwrap();
        assert_close_vec(&out, &[-0.99998, 0.99998, -0.99998, 0.99998], 1e-4);
    }

    #[test]
    fn rms_norm_scales_by_root_mean_square() {
        let x = vec![3.0, 4.0];
        let weight = vec![1.0, 1.0];
        let out = rms_norm(&x, &weight, 1, 2, 0.0).unwrap();
        let denom = ((9.0f32 + 16.0) / 2.0).sqrt();
        assert_close_vec(&out, &[3.0 / denom, 4.0 / denom], 1e-6);
    }

    #[test]
    fn softmax_rows_is_stable_and_sums_to_one() {
        let logits = vec![1000.0, 1001.0, 1.0, 1.0];
        let out = softmax_rows(&logits, 2, 2).unwrap();
        assert_close_vec(&out[0..2], &[0.26894143, 0.7310586], 1e-6);
        assert_close_vec(&out[2..4], &[0.5, 0.5], 1e-6);
    }

    #[test]
    fn gelu_is_reasonable_for_known_points() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu(1.0) - 0.841_191_96).abs() < 1e-5);
    }

    #[test]
    fn silu_is_reasonable_for_known_points() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // sigmoid(1) = 0.731058
        assert!((silu(1.0) - 0.731058).abs() < 1e-5);
    }

    #[test]
    fn scaled_dot_product_attention_applies_causal_mask() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = q.clone();
        let v = vec![10.0, 0.0, 0.0, 20.0];
        let out = scaled_dot_product_attention(&q, &k, &v, 2, 1, 2, true).unwrap();

        assert_close_vec(&out[0..2], &[10.0, 0.0], 1e-6);
        assert!(
            out[2] > 3.0 && out[2] < 3.4,
            "unexpected second-token first dim: {}",
            out[2]
        );
        assert!(
            out[3] > 13.0 && out[3] < 13.7,
            "unexpected second-token second dim: {}",
            out[3]
        );
    }

    #[test]
    fn mlp_runs_linear_gelu_linear() {
        let input = vec![1.0, -1.0];
        let w_in = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let b_in = vec![0.0, 0.0];
        let w_out = vec![1.0, 1.0, 1.0, -1.0]; // 2x2
        let b_out = vec![0.0, 0.0];
        let out = mlp(&input, &w_in, Some(&b_in), &w_out, Some(&b_out), 1, 2, 2).unwrap();

        let g0 = gelu(1.0);
        let g1 = gelu(-1.0);
        assert_close_vec(&out, &[g0 + g1, g0 - g1], 1e-6);
    }

    #[test]
    fn sample_argmax_is_deterministic() {
        assert_eq!(sample_argmax(&[0.1, 2.0, 1.9]).unwrap(), 1);
    }

    #[test]
    fn sample_argmax_excluding_uses_next_best_token() {
        assert_eq!(
            sample_argmax_excluding(&[0.1, 2.0, 1.9], Some(1)).unwrap(),
            2
        );
        assert_eq!(
            sample_argmax_excluding(&[0.1, 2.0, 1.9], Some(99)).unwrap(),
            1
        );
        assert_eq!(sample_argmax_excluding(&[2.0], Some(0)).unwrap(), 0);
    }

    #[test]
    fn sample_top_p_is_deterministic_for_same_seed() {
        let logits = [0.1, 0.2, 2.0, 1.0];
        let first = sample_top_p(&logits, 0.8, 0.95, 1234).unwrap();
        let second = sample_top_p(&logits, 0.8, 0.95, 1234).unwrap();
        assert_eq!(first, second);
        assert!(first < logits.len());
    }
}
