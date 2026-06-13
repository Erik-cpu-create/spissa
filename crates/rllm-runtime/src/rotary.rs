use crate::{softmax_rows, Result, RuntimeError};

#[derive(Debug, Clone, Copy)]
pub struct RotaryEmbeddingConfig {
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub base: f32,
    pub position_offset: usize,
}

/// Convert GPT-NeoX/Pythia `rotary_pct` into the even rotary dimension used per head.
pub fn gpt_neox_rotary_dim(head_dim: usize, rotary_pct: f32) -> Result<usize> {
    if head_dim == 0 {
        return Err(RuntimeError::Shape(
            "head_dim must be greater than zero".to_string(),
        ));
    }
    if !rotary_pct.is_finite() || rotary_pct <= 0.0 || rotary_pct > 1.0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "rotary_pct must be finite and in (0, 1], got {rotary_pct}"
        )));
    }
    let mut dim = (head_dim as f32 * rotary_pct) as usize;
    if dim % 2 != 0 {
        dim -= 1;
    }
    if dim == 0 {
        return Err(RuntimeError::Shape(format!(
            "rotary_pct {rotary_pct} produces zero even rotary dim for head_dim {head_dim}"
        )));
    }
    Ok(dim)
}

/// Apply GPT-NeoX/Pythia rotary position embeddings in-place to Q and K.
///
/// Layout is `[seq_len, num_heads, head_dim]`. GPT-NeoX style rotates the first
/// half of the rotary slice against the second half, leaving dimensions after
/// `rotary_dim` untouched.
pub fn apply_gpt_neox_rotary_inplace(
    q: &mut [f32],
    k: &mut [f32],
    config: RotaryEmbeddingConfig,
) -> Result<()> {
    validate_rotary_inputs(q, k, config)?;
    let half_rotary = config.rotary_dim / 2;
    for pos in 0..config.seq_len {
        let absolute_pos = config
            .position_offset
            .checked_add(pos)
            .ok_or_else(|| RuntimeError::Shape("rotary absolute position overflow".to_string()))?;
        for head in 0..config.num_heads {
            let row_start = (pos * config.num_heads + head) * config.head_dim;
            for pair in 0..half_rotary {
                let angle = rotary_angle(absolute_pos, pair, config)?;
                let cos = angle.cos();
                let sin = angle.sin();
                rotate_neox_pair(q, row_start, pair, half_rotary, cos, sin);
                rotate_neox_pair(k, row_start, pair, half_rotary, cos, sin);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct KvCache {
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    len: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvCache {
    pub fn new(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Result<Self> {
        if num_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(RuntimeError::Shape(format!(
                "KV cache dimensions must be non-zero, got num_heads={num_heads}, head_dim={head_dim}, max_seq_len={max_seq_len}"
            )));
        }
        let capacity = max_seq_len
            .checked_mul(num_heads)
            .and_then(|values| values.checked_mul(head_dim))
            .ok_or_else(|| RuntimeError::Shape("KV cache capacity overflow".to_string()))?;
        Ok(Self {
            num_heads,
            head_dim,
            max_seq_len,
            len: 0,
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        })
    }

    pub fn append(&mut self, keys: &[f32], values: &[f32], token_count: usize) -> Result<()> {
        if token_count == 0 {
            return Err(RuntimeError::Shape(
                "KV cache append token_count must be greater than zero".to_string(),
            ));
        }
        let append_values = token_count
            .checked_mul(self.token_width())
            .ok_or_else(|| RuntimeError::Shape("KV append length overflow".to_string()))?;
        if keys.len() != append_values || values.len() != append_values {
            return Err(RuntimeError::Shape(format!(
                "KV append shape mismatch: expected {append_values}, got keys={}, values={}",
                keys.len(),
                values.len()
            )));
        }
        let new_len = self
            .len
            .checked_add(token_count)
            .ok_or_else(|| RuntimeError::Shape("KV cache length overflow".to_string()))?;
        if new_len > self.max_seq_len {
            return Err(RuntimeError::Shape(format!(
                "KV cache capacity exceeded: append would set len {new_len}, max_seq_len {}",
                self.max_seq_len
            )));
        }
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
        self.len = new_len;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    fn token_width(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KvAttentionConfig {
    pub query_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
}

/// Scaled dot-product attention where keys/values may be prefixed by a KV cache.
///
/// `q`, `current_k`, and `current_v` use `[query_len, num_heads, head_dim]`.
/// Cached tensors, when present, use `[past_len, num_heads, head_dim]` and are
/// treated as absolute positions before the current query window.
pub fn scaled_dot_product_attention_with_cache(
    q: &[f32],
    current_k: &[f32],
    current_v: &[f32],
    cache: Option<&KvCache>,
    config: KvAttentionConfig,
) -> Result<Vec<f32>> {
    validate_kv_attention_inputs(q, current_k, current_v, cache, config)?;
    let expected_current = config.query_len * config.num_heads * config.head_dim;
    let past_len = cache.map(KvCache::len).unwrap_or(0);
    let key_len = past_len
        .checked_add(config.query_len)
        .ok_or_else(|| RuntimeError::Shape("KV attention key length overflow".to_string()))?;
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let mut out = vec![0.0f32; expected_current];
    let mut scores = vec![0.0f32; key_len];

    for query_pos in 0..config.query_len {
        let query_abs_pos = past_len + query_pos;
        for head in 0..config.num_heads {
            let query_start = (query_pos * config.num_heads + head) * config.head_dim;
            let query_row = &q[query_start..query_start + config.head_dim];
            for key_pos in 0..key_len {
                if config.causal && key_pos > query_abs_pos {
                    scores[key_pos] = f32::NEG_INFINITY;
                    continue;
                }
                let key_row = kv_row(
                    cache,
                    current_k,
                    past_len,
                    key_pos,
                    head,
                    config,
                    KvTensorKind::Key,
                );
                let mut dot = 0.0f32;
                for dim in 0..config.head_dim {
                    dot += query_row[dim] * key_row[dim];
                }
                scores[key_pos] = dot * scale;
            }

            let probs = softmax_rows(&scores, 1, key_len)?;
            let out_start = (query_pos * config.num_heads + head) * config.head_dim;
            let out_row = &mut out[out_start..out_start + config.head_dim];
            out_row.fill(0.0);
            for key_pos in 0..key_len {
                let value_row = kv_row(
                    cache,
                    current_v,
                    past_len,
                    key_pos,
                    head,
                    config,
                    KvTensorKind::Value,
                );
                let prob = probs[key_pos];
                for dim in 0..config.head_dim {
                    out_row[dim] += prob * value_row[dim];
                }
            }
        }
    }

    Ok(out)
}

fn validate_rotary_inputs(q: &[f32], k: &[f32], config: RotaryEmbeddingConfig) -> Result<()> {
    let expected = config
        .seq_len
        .checked_mul(config.num_heads)
        .and_then(|values| values.checked_mul(config.head_dim))
        .ok_or_else(|| RuntimeError::Shape("rotary input length overflow".to_string()))?;
    if q.len() != expected || k.len() != expected {
        return Err(RuntimeError::Shape(format!(
            "rotary Q/K shape mismatch: expected {expected}, got q={}, k={}",
            q.len(),
            k.len()
        )));
    }
    if config.seq_len == 0 || config.num_heads == 0 || config.head_dim == 0 {
        return Err(RuntimeError::Shape(format!(
            "rotary dimensions must be non-zero, got seq_len={}, num_heads={}, head_dim={}",
            config.seq_len, config.num_heads, config.head_dim
        )));
    }
    if config.rotary_dim == 0 || config.rotary_dim > config.head_dim || config.rotary_dim % 2 != 0 {
        return Err(RuntimeError::Shape(format!(
            "rotary_dim must be even and in 1..=head_dim, got rotary_dim={}, head_dim={}",
            config.rotary_dim, config.head_dim
        )));
    }
    if !config.base.is_finite() || config.base <= 0.0 {
        return Err(RuntimeError::InvalidTensorData(format!(
            "rotary base must be finite and positive, got {}",
            config.base
        )));
    }
    Ok(())
}

fn rotary_angle(absolute_pos: usize, pair: usize, config: RotaryEmbeddingConfig) -> Result<f32> {
    let exponent = (2 * pair) as f32 / config.rotary_dim as f32;
    let inv_freq = 1.0 / config.base.powf(exponent);
    Ok(absolute_pos as f32 * inv_freq)
}

fn rotate_neox_pair(
    values: &mut [f32],
    row_start: usize,
    pair: usize,
    half_rotary: usize,
    cos: f32,
    sin: f32,
) {
    let first_idx = row_start + pair;
    let second_idx = row_start + pair + half_rotary;
    let first = values[first_idx];
    let second = values[second_idx];
    values[first_idx] = first * cos - second * sin;
    values[second_idx] = second * cos + first * sin;
}

fn validate_kv_attention_inputs(
    q: &[f32],
    current_k: &[f32],
    current_v: &[f32],
    cache: Option<&KvCache>,
    config: KvAttentionConfig,
) -> Result<()> {
    if config.query_len == 0 || config.num_heads == 0 || config.head_dim == 0 {
        return Err(RuntimeError::Shape(format!(
            "KV attention dimensions must be non-zero, got query_len={}, num_heads={}, head_dim={}",
            config.query_len, config.num_heads, config.head_dim
        )));
    }
    let expected = config
        .query_len
        .checked_mul(config.num_heads)
        .and_then(|values| values.checked_mul(config.head_dim))
        .ok_or_else(|| RuntimeError::Shape("KV attention current length overflow".to_string()))?;
    if q.len() != expected || current_k.len() != expected || current_v.len() != expected {
        return Err(RuntimeError::Shape(format!(
            "KV attention shape mismatch: expected {expected}, got q={}, k={}, v={}",
            q.len(),
            current_k.len(),
            current_v.len()
        )));
    }
    if let Some(cache) = cache {
        if cache.num_heads != config.num_heads || cache.head_dim != config.head_dim {
            return Err(RuntimeError::Shape(format!(
                "KV cache shape mismatch: cache heads/dim={}/{}, attention heads/dim={}/{}",
                cache.num_heads, cache.head_dim, config.num_heads, config.head_dim
            )));
        }
        let expected_cache_values = cache
            .len
            .checked_mul(cache.token_width())
            .ok_or_else(|| RuntimeError::Shape("KV cache stored length overflow".to_string()))?;
        if cache.keys.len() != expected_cache_values || cache.values.len() != expected_cache_values
        {
            return Err(RuntimeError::InvalidTensorData(format!(
                "KV cache internal length mismatch: expected {expected_cache_values}, keys={}, values={}",
                cache.keys.len(),
                cache.values.len()
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum KvTensorKind {
    Key,
    Value,
}

fn kv_row<'a>(
    cache: Option<&'a KvCache>,
    current: &'a [f32],
    past_len: usize,
    key_pos: usize,
    head: usize,
    config: KvAttentionConfig,
    kind: KvTensorKind,
) -> &'a [f32] {
    if key_pos < past_len {
        let cache = cache.expect("past_len is non-zero only when cache is present");
        let start = (key_pos * config.num_heads + head) * config.head_dim;
        match kind {
            KvTensorKind::Key => &cache.keys[start..start + config.head_dim],
            KvTensorKind::Value => &cache.values[start..start + config.head_dim],
        }
    } else {
        let current_pos = key_pos - past_len;
        let start = (current_pos * config.num_heads + head) * config.head_dim;
        &current[start..start + config.head_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{scaled_dot_product_attention, RuntimeError};

    fn assert_close_vec(actual: &[f32], expected: &[f32], eps: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= eps,
                "idx={idx}: actual={actual}, expected={expected}"
            );
        }
    }

    fn manual_gpt_neox_rotary(values: &mut [f32], config: RotaryEmbeddingConfig) {
        let half_rotary = config.rotary_dim / 2;
        for pos in 0..config.seq_len {
            let absolute_pos = config.position_offset + pos;
            for head in 0..config.num_heads {
                let row_start = (pos * config.num_heads + head) * config.head_dim;
                for pair in 0..half_rotary {
                    let first_idx = row_start + pair;
                    let second_idx = row_start + pair + half_rotary;
                    let first = values[first_idx];
                    let second = values[second_idx];
                    let inv_freq = 1.0
                        / config
                            .base
                            .powf((2 * pair) as f32 / config.rotary_dim as f32);
                    let angle = absolute_pos as f32 * inv_freq;
                    let cos = angle.cos();
                    let sin = angle.sin();
                    values[first_idx] = first * cos - second * sin;
                    values[second_idx] = second * cos + first * sin;
                }
            }
        }
    }

    #[test]
    fn gpt_neox_rotary_rotates_first_rotary_dims_and_preserves_tail() {
        let config = RotaryEmbeddingConfig {
            seq_len: 2,
            num_heads: 1,
            head_dim: 6,
            rotary_dim: 4,
            base: 10_000.0,
            position_offset: 0,
        };
        let mut q = vec![
            1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 0.5, -0.25, 1.5, -2.0, 11.0, 12.0,
        ];
        let mut k = vec![
            -1.0, 0.75, 2.0, 0.25, 13.0, 14.0, 2.5, -1.5, 0.5, 3.0, 15.0, 16.0,
        ];
        let mut expected_q = q.clone();
        let mut expected_k = k.clone();
        manual_gpt_neox_rotary(&mut expected_q, config);
        manual_gpt_neox_rotary(&mut expected_k, config);

        apply_gpt_neox_rotary_inplace(&mut q, &mut k, config).unwrap();

        assert_close_vec(&q, &expected_q, 1e-6);
        assert_close_vec(&k, &expected_k, 1e-6);
        assert_eq!(q[4], 9.0);
        assert_eq!(q[5], 10.0);
        assert_eq!(k[10], 15.0);
        assert_eq!(k[11], 16.0);
    }

    #[test]
    fn kv_cache_appends_tokens_and_rejects_capacity_overflow_without_mutating() {
        let mut cache = KvCache::new(1, 2, 3).unwrap();
        cache
            .append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2)
            .unwrap();

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.keys(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.values(), &[5.0, 6.0, 7.0, 8.0]);

        let err = cache
            .append(&[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0], 2)
            .unwrap_err();
        assert!(matches!(err, RuntimeError::Shape(_)));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.keys(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cached_attention_for_next_token_matches_full_causal_attention_last_row() {
        let q_all = vec![0.2, 0.7, -0.1, 0.4, 0.9, -0.3];
        let k_all = vec![0.5, -0.2, 0.1, 0.8, -0.4, 0.3];
        let v_all = vec![1.0, 0.0, -0.5, 2.0, 0.25, -1.0];
        let full = scaled_dot_product_attention(&q_all, &k_all, &v_all, 3, 1, 2, true).unwrap();

        let mut cache = KvCache::new(1, 2, 3).unwrap();
        cache.append(&k_all[..4], &v_all[..4], 2).unwrap();
        let incremental = scaled_dot_product_attention_with_cache(
            &q_all[4..],
            &k_all[4..],
            &v_all[4..],
            Some(&cache),
            KvAttentionConfig {
                query_len: 1,
                num_heads: 1,
                head_dim: 2,
                causal: true,
            },
        )
        .unwrap();

        assert_close_vec(&incremental, &full[4..], 1e-6);
    }
}
