use crate::tensor::decode_to_f32;
use crate::{
    apply_gpt_neox_rotary_inplace, q8_kernel_profile_enabled, record_q8_kernel_path,
    scaled_dot_product_attention_with_cache, KvAttentionConfig, KvCache, LazyRllmModel,
    MemoryBudget, Q8KernelPath, Result, RotaryEmbeddingConfig, RuntimeError,
};
use rllm_container::{ChunkMeta, TensorMeta};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::time::{Duration, Instant};

pub const DEFAULT_STREAMING_TILE_ELEMENTS: usize = 4096;
const RLLM_THREADS_ENV: &str = "RLLM_THREADS";
const RLLM_SPARSE_PARALLEL_ENV: &str = "RLLM_SPARSE_PARALLEL";
const MIN_ROWS_PER_PARALLEL_ARGMAX: usize = 4;
const LARGE_VOCAB_ARGMAX_THRESHOLD: usize = 65_536;
const LARGE_VOCAB_AUTO_THREAD_CAP: usize = 2;

fn available_runtime_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

pub(crate) fn streaming_available_threads() -> usize {
    available_runtime_threads()
}

fn argmax_runtime_thread_count(out_features: usize) -> usize {
    effective_argmax_runtime_threads(
        std::env::var(RLLM_THREADS_ENV).ok().as_deref(),
        available_runtime_threads(),
        out_features,
    )
}

fn effective_argmax_runtime_threads(
    override_value: Option<&str>,
    available: usize,
    out_features: usize,
) -> usize {
    if override_value.is_some() {
        effective_runtime_threads(override_value, available)
    } else if out_features > LARGE_VOCAB_ARGMAX_THRESHOLD {
        available.clamp(1, LARGE_VOCAB_AUTO_THREAD_CAP)
    } else {
        effective_runtime_threads(None, available)
    }
}

fn effective_runtime_threads(override_value: Option<&str>, available: usize) -> usize {
    let available = available.max(1);
    match override_value.and_then(|value| value.trim().parse::<usize>().ok()) {
        Some(value) if value > 0 => value.min(available),
        _ => available,
    }
}

fn effective_row_block_threads(rows: usize, available_threads: usize) -> usize {
    if rows < MIN_ROWS_PER_PARALLEL_ARGMAX {
        1
    } else {
        available_threads.max(1).min(rows)
    }
}

fn sparse_runtime_thread_count() -> usize {
    if !parse_sparse_parallel_enabled(std::env::var(RLLM_SPARSE_PARALLEL_ENV).ok().as_deref()) {
        return 1;
    }
    effective_runtime_threads(
        std::env::var(RLLM_THREADS_ENV).ok().as_deref(),
        available_runtime_threads(),
    )
}

fn parse_sparse_parallel_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// R131: LM-head logits GEMV `logits[v] = Σ_h last_hidden[h] · weight[v, h]`,
/// parallelized over the vocabulary rows.
///
/// The LM head over a large vocabulary (Gemma 3 = 262k) is a dense matrix–vector
/// product recomputed every decode step; the previous adapter loop ran it on a
/// single thread while the q8 transformer kernels already use all cores. Each
/// logit is an independent dot product, so splitting the vocab rows across
/// workers is embarrassingly parallel and **bit-identical** to the serial path:
/// every output keeps the same scalar accumulation order, only the row range a
/// thread owns changes. Honors `RLLM_THREADS`; falls back to serial for tiny
/// vocabularies or a single available core.
pub fn lm_head_logits_parallel(
    last_hidden: &[f32],
    weight: &[f32],
    vocab_size: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab_size];
    let threads = effective_runtime_threads(
        std::env::var(RLLM_THREADS_ENV).ok().as_deref(),
        available_runtime_threads(),
    );
    if threads <= 1 || vocab_size < 2 * MIN_ROWS_PER_PARALLEL_ARGMAX {
        lm_head_logits_rows(last_hidden, weight, hidden, &mut logits);
        return logits;
    }
    let workers = threads.min(vocab_size / MIN_ROWS_PER_PARALLEL_ARGMAX).max(1);
    let rows_per_worker = vocab_size.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut out_rest = &mut logits[..];
        let mut row_start = 0usize;
        while row_start < vocab_size {
            let rows = rows_per_worker.min(vocab_size - row_start);
            let (out_slice, rest) = out_rest.split_at_mut(rows);
            out_rest = rest;
            let weight_slice = &weight[row_start * hidden..(row_start + rows) * hidden];
            scope.spawn(move || lm_head_logits_rows(last_hidden, weight_slice, hidden, out_slice));
            row_start += rows;
        }
    });
    logits
}

/// Serial reference for one contiguous block of LM-head rows. Kept as the single
/// source of the per-logit accumulation order so the parallel split stays
/// bit-identical to it (see the R131 parity test).
fn lm_head_logits_rows(last_hidden: &[f32], weight: &[f32], hidden: usize, out: &mut [f32]) {
    for (v, logit) in out.iter_mut().enumerate() {
        let row = &weight[v * hidden..v * hidden + hidden];
        let mut sum = 0.0f32;
        for (h, w) in row.iter().enumerate() {
            sum += last_hidden[h] * *w;
        }
        *logit = sum;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingLinearConfig {
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingTileLinearConfig {
    pub linear: StreamingLinearConfig,
    /// Maximum number of weight elements converted into f32 scratch at once.
    ///
    /// Current RTC codecs still decode one compressed chunk to original bytes;
    /// Phase 7 starts by fusing f32 conversion and matmul accumulation over
    /// bounded tiles instead of materializing a full f32 chunk.
    pub tile_elements: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingMlpConfig {
    pub batch: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingAttentionConfig {
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
}

#[derive(Debug, Default)]
pub struct StreamingAttentionRuntime<'a> {
    pub rotary: Option<RotaryEmbeddingConfig>,
    pub kv_cache: Option<&'a mut KvCache>,
}

impl StreamingAttentionConfig {
    fn hidden_size(self) -> Result<usize> {
        self.num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| RuntimeError::Shape("attention hidden_size overflow".to_string()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingBlockConfig {
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub causal: bool,
    pub layer_norm_eps: f32,
}

#[derive(Debug, Default)]
pub struct StreamingBlockRuntime<'a> {
    pub attention: StreamingAttentionRuntime<'a>,
    /// GPT-NeoX/Pythia can use parallel residual blocks:
    /// `x + attention(LN1(x)) + mlp(LN2(x))`.
    ///
    /// The default remains the older sequential pre-norm toy block:
    /// `x + attention(LN1(x)) -> LN2(residual) -> + mlp(...)`.
    pub parallel_residual: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamingBlockTiming {
    pub attention_norm_ns: u64,
    pub attention_ns: u64,
    pub attention_qkv_projection_ns: u64,
    pub attention_qkv_split_ns: u64,
    pub attention_rotary_ns: u64,
    pub attention_score_context_ns: u64,
    pub attention_output_projection_ns: u64,
    pub attention_kv_append_ns: u64,
    pub attention_residual_ns: u64,
    pub mlp_norm_ns: u64,
    pub mlp_ns: u64,
    pub mlp_input_projection_ns: u64,
    pub mlp_activation_ns: u64,
    pub mlp_output_projection_ns: u64,
    pub mlp_residual_ns: u64,
    pub attention_norm_calls: usize,
    pub attention_calls: usize,
    pub attention_qkv_projection_calls: usize,
    pub attention_qkv_split_calls: usize,
    pub attention_rotary_calls: usize,
    pub attention_score_context_calls: usize,
    pub attention_output_projection_calls: usize,
    pub attention_kv_append_calls: usize,
    pub attention_residual_calls: usize,
    pub mlp_norm_calls: usize,
    pub mlp_calls: usize,
    pub mlp_input_projection_calls: usize,
    pub mlp_activation_calls: usize,
    pub mlp_output_projection_calls: usize,
    pub mlp_residual_calls: usize,
}

impl StreamingBlockTiming {
    fn record_attention_norm(&mut self, elapsed: Duration) {
        self.attention_norm_ns = self
            .attention_norm_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_norm_calls = self.attention_norm_calls.saturating_add(1);
    }

    fn record_attention(&mut self, elapsed: Duration) {
        self.attention_ns = self.attention_ns.saturating_add(elapsed_ns_u64(elapsed));
        self.attention_calls = self.attention_calls.saturating_add(1);
    }

    fn record_attention_qkv_projection(&mut self, elapsed: Duration) {
        self.attention_qkv_projection_ns = self
            .attention_qkv_projection_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_qkv_projection_calls = self.attention_qkv_projection_calls.saturating_add(1);
    }

    fn record_attention_qkv_split(&mut self, elapsed: Duration) {
        self.attention_qkv_split_ns = self
            .attention_qkv_split_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_qkv_split_calls = self.attention_qkv_split_calls.saturating_add(1);
    }

    fn record_attention_rotary(&mut self, elapsed: Duration) {
        self.attention_rotary_ns = self
            .attention_rotary_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_rotary_calls = self.attention_rotary_calls.saturating_add(1);
    }

    fn record_attention_score_context(&mut self, elapsed: Duration) {
        self.attention_score_context_ns = self
            .attention_score_context_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_score_context_calls = self.attention_score_context_calls.saturating_add(1);
    }

    fn record_attention_output_projection(&mut self, elapsed: Duration) {
        self.attention_output_projection_ns = self
            .attention_output_projection_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_output_projection_calls =
            self.attention_output_projection_calls.saturating_add(1);
    }

    fn record_attention_kv_append(&mut self, elapsed: Duration) {
        self.attention_kv_append_ns = self
            .attention_kv_append_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_kv_append_calls = self.attention_kv_append_calls.saturating_add(1);
    }

    fn record_attention_residual(&mut self, elapsed: Duration) {
        self.attention_residual_ns = self
            .attention_residual_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.attention_residual_calls = self.attention_residual_calls.saturating_add(1);
    }

    fn record_mlp_norm(&mut self, elapsed: Duration) {
        self.mlp_norm_ns = self.mlp_norm_ns.saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_norm_calls = self.mlp_norm_calls.saturating_add(1);
    }

    fn record_mlp(&mut self, elapsed: Duration) {
        self.mlp_ns = self.mlp_ns.saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_calls = self.mlp_calls.saturating_add(1);
    }

    fn record_mlp_input_projection(&mut self, elapsed: Duration) {
        self.mlp_input_projection_ns = self
            .mlp_input_projection_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_input_projection_calls = self.mlp_input_projection_calls.saturating_add(1);
    }

    fn record_mlp_activation(&mut self, elapsed: Duration) {
        self.mlp_activation_ns = self
            .mlp_activation_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_activation_calls = self.mlp_activation_calls.saturating_add(1);
    }

    fn record_mlp_output_projection(&mut self, elapsed: Duration) {
        self.mlp_output_projection_ns = self
            .mlp_output_projection_ns
            .saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_output_projection_calls = self.mlp_output_projection_calls.saturating_add(1);
    }

    fn record_mlp_residual(&mut self, elapsed: Duration) {
        self.mlp_residual_ns = self.mlp_residual_ns.saturating_add(elapsed_ns_u64(elapsed));
        self.mlp_residual_calls = self.mlp_residual_calls.saturating_add(1);
    }
}

fn elapsed_ns_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl StreamingBlockConfig {
    fn hidden_size(self) -> Result<usize> {
        self.num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| RuntimeError::Shape("block hidden_size overflow".to_string()))
    }

    fn attention_config(self) -> StreamingAttentionConfig {
        StreamingAttentionConfig {
            seq_len: self.seq_len,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            causal: self.causal,
        }
    }

    fn mlp_config(self) -> Result<StreamingMlpConfig> {
        Ok(StreamingMlpConfig {
            batch: self.seq_len,
            hidden_size: self.hidden_size()?,
            intermediate_size: self.intermediate_size,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingBlockTensorNames<'a> {
    pub qkv_weight: &'a str,
    pub attention_out_weight: &'a str,
    pub mlp_in_weight: &'a str,
    pub mlp_out_weight: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingBlockParameters<'a> {
    pub input_layernorm_weight: &'a [f32],
    pub input_layernorm_bias: &'a [f32],
    pub qkv_bias: Option<&'a [f32]>,
    pub attention_out_bias: Option<&'a [f32]>,
    pub post_attention_layernorm_weight: &'a [f32],
    pub post_attention_layernorm_bias: &'a [f32],
    pub mlp_in_bias: Option<&'a [f32]>,
    pub mlp_out_bias: Option<&'a [f32]>,
}

include!("linear.rs");
include!("mlp.rs");
include!("attention.rs");
include!("block.rs");
include!("validation.rs");
include!("argmax.rs");
include!("kernels.rs");
include!("tests.rs");
