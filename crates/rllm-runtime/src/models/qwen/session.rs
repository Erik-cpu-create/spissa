//! Stateful Qwen3.5 chat session: persistent per-layer KV / Gated-DeltaNet state across
//! turns so each turn only prefills the NEW tokens (no O(n²) re-prefill of the whole
//! conversation). Also the single forward used by one-shot generation.

use crate::models::qwen::api::gather_embedding_rows;
use crate::models::qwen::generate::{qwen_gated_attention_block, qwen_gated_deltanet_block};
use crate::models::qwen::model::{
    GatedDeltaNetState, PreparedQwenTransformer, QwenLayerCache, QwenLayerKind,
};
use crate::ops::{rms_norm, sample_argmax, sample_top_p};
use crate::rotary::KvCache;
use crate::{
    streaming_tile_linear_from_model, LazyRllmModel, MemoryBudget, Result, RuntimeError,
    StreamingLinearConfig, StreamingSamplingConfig, StreamingTileLinearConfig,
    DEFAULT_STREAMING_TILE_ELEMENTS,
};

/// Allocate fresh per-layer mixing state (growing KV for full-attn layers, constant-size
/// recurrent state for linear-attn layers).
pub(crate) fn alloc_caches(prepared: &PreparedQwenTransformer) -> Result<Vec<QwenLayerCache>> {
    let cfg = &prepared.config;
    prepared
        .layer_params
        .iter()
        .map(|p| match p.kind() {
            QwenLayerKind::FullAttention => Ok(QwenLayerCache::Attn(KvCache::new(
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.max_seq_len,
            )?)),
            QwenLayerKind::LinearAttention => Ok(QwenLayerCache::Linear(GatedDeltaNetState::new(
                cfg.linear_num_heads,
                cfg.linear_key_dim,
                cfg.linear_value_dim,
                cfg.conv_kernel,
                cfg.linear_conv_channels(),
            ))),
        })
        .collect()
}

/// Vocabulary size from the embedding tensor metadata (`[vocab, hidden]`).
pub(crate) fn vocab_from_meta(model: &LazyRllmModel, prepared: &PreparedQwenTransformer) -> usize {
    model
        .tensor(&prepared.embedding_weight)
        .ok()
        .and_then(|t| t.shape.first().copied())
        .unwrap_or(0) as usize
}

/// One forward pass over `tokens` (prefill `seq_len>1` or decode `seq_len==1`), appending
/// to the persistent `caches`; returns logits for the LAST token. `position_offset` is the
/// number of tokens already in the caches (for RoPE positions).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_forward(
    model: &mut LazyRllmModel,
    prepared: &PreparedQwenTransformer,
    caches: &mut [QwenLayerCache],
    vocab_size: usize,
    tokens: &[usize],
    position_offset: usize,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let cfg = prepared.config;
    let hidden = cfg.hidden_size;
    let seq_len = tokens.len();

    let mut h = gather_embedding_rows(model, &prepared.embedding_weight, hidden, tokens, budget)?;
    for (i, tensors) in prepared.layers.iter().enumerate() {
        let params = &prepared.layer_params[i];
        h = match &mut caches[i] {
            QwenLayerCache::Attn(cache) => qwen_gated_attention_block(
                model,
                &h,
                tensors,
                params,
                &cfg,
                seq_len,
                position_offset,
                budget,
                cache,
            )?,
            QwenLayerCache::Linear(state) => qwen_gated_deltanet_block(
                model, &h, tensors, params, &cfg, seq_len, budget, state,
            )?,
        };
    }
    h = rms_norm(&h, &prepared.final_norm, seq_len, hidden, cfg.rms_norm_eps)?;

    let last = &h[(seq_len - 1) * hidden..];
    let lm_cfg = StreamingTileLinearConfig {
        linear: StreamingLinearConfig {
            batch: 1,
            in_features: hidden,
            out_features: vocab_size,
        },
        tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS,
    };
    streaming_tile_linear_from_model(model, &prepared.lm_head_weight, last, None, lm_cfg, budget)
}

pub(crate) fn sample_next(logits: &[f32], sampling: StreamingSamplingConfig) -> Result<usize> {
    match sampling {
        StreamingSamplingConfig::Argmax => sample_argmax(logits),
        StreamingSamplingConfig::TopP {
            temperature,
            top_p,
            seed,
        } => sample_top_p(logits, temperature, top_p, seed),
    }
}

/// A persistent Qwen chat session. Owns the prepared transformer + per-layer caches; each
/// `generate` call PREFILLS the new tokens onto the existing context and decodes from there.
pub struct QwenSession {
    prepared: PreparedQwenTransformer,
    caches: Vec<QwenLayerCache>,
    vocab_size: usize,
    /// Tokens already committed to the caches (for the next RoPE position offset).
    pos: usize,
}

impl QwenSession {
    pub fn new(model: &LazyRllmModel, prepared: PreparedQwenTransformer) -> Result<Self> {
        let caches = alloc_caches(&prepared)?;
        let vocab_size = vocab_from_meta(model, &prepared);
        if vocab_size == 0 {
            return Err(RuntimeError::Shape(
                "qwen embedding tensor has no vocab dimension".to_string(),
            ));
        }
        Ok(Self {
            prepared,
            caches,
            vocab_size,
            pos: 0,
        })
    }

    pub fn prepared(&self) -> &PreparedQwenTransformer {
        &self.prepared
    }

    /// Tokens currently committed to the context.
    pub fn context_len(&self) -> usize {
        self.pos
    }

    /// Drop all context (new conversation) without re-decoding the prepared weights.
    pub fn reset(&mut self) -> Result<()> {
        self.caches = alloc_caches(&self.prepared)?;
        self.pos = 0;
        Ok(())
    }

    /// Prefill `prompt` onto the existing context, then decode up to `max_new` tokens.
    /// Stops early on any id in `stop` (the stop token is NOT emitted or committed — the
    /// caller appends the turn's closing tokens next turn). Calls `on_token` per emitted
    /// token; returning false stops generation.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        model: &mut LazyRllmModel,
        prompt: &[usize],
        max_new: usize,
        sampling: StreamingSamplingConfig,
        stop: &[usize],
        budget: &mut MemoryBudget,
        on_token: &mut dyn FnMut(usize) -> bool,
    ) -> Result<Vec<usize>> {
        let mut generated = Vec::new();
        let mut feed: Vec<usize> = prompt.to_vec();
        for _ in 0..max_new {
            if feed.is_empty() {
                break;
            }
            let logits = qwen_forward(
                model,
                &self.prepared,
                &mut self.caches,
                self.vocab_size,
                &feed,
                self.pos,
                budget,
            )?;
            self.pos += feed.len();
            let next = sample_next(&logits, sampling)?;
            if stop.contains(&next) {
                break;
            }
            generated.push(next);
            if !on_token(next) {
                break;
            }
            feed = vec![next];
        }
        Ok(generated)
    }
}
