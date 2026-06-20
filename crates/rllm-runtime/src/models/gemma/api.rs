use crate::models::gemma::generate::{
    gemma_embed_input_streaming, gemma_lm_head_streaming, streaming_gemma_transformer_block,
    GemmaBlockRuntime,
};
use crate::models::gemma::model::*;
use crate::rotary::KvCache;
use crate::{
    bf16_to_f32, lm_head_logits_parallel, lm_head_logits_parallel_bf16,
    ops::{embedding_lookup, rms_norm, sample_argmax, sample_top_p},
    LazyRllmModel, MemoryBudget, Result, RuntimeError, StreamingSamplingConfig,
};
use rllm_container::{DType, ModelConfigMetadata};

/// Gemma 3 fixed family default for the global-layer linear RoPE divisor
/// (`rope_scaling.factor`). Used only when the packed metadata predates the
/// `rope_scaling_factor` field; current packs carry the value explicitly.
const GEMMA3_DEFAULT_ROPE_SCALING_FACTOR: f32 = 8.0;

#[derive(Debug, Clone, Copy)]
pub struct GemmaGenerationConfig {
    pub max_new_tokens: usize,
    pub max_seq_len: Option<usize>,
    pub causal: bool,
    pub sampling: StreamingSamplingConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct GemmaGenerationOptions {
    pub collect_logits: bool,
}

impl Default for GemmaGenerationOptions {
    fn default() -> Self {
        Self {
            collect_logits: true,
        }
    }
}

fn require_config<'a>(model: &'a LazyRllmModel) -> Result<&'a ModelConfigMetadata> {
    model.metadata().model_config.as_ref().ok_or_else(|| {
        RuntimeError::InvalidTensorData(
            "gemma generation requires persisted model_config metadata; repack with --config <config.json>"
                .to_string(),
        )
    })
}

fn require_usize(field: &str, value: Option<u64>) -> Result<usize> {
    let value = value.ok_or_else(|| {
        RuntimeError::InvalidTensorData(format!("gemma model_config is missing required field {field}"))
    })?;
    usize::try_from(value)
        .map_err(|_| RuntimeError::Shape(format!("gemma model_config field {field}={value} overflows usize")))
}

/// Decode an RMSNorm weight and pre-bake Gemma's `(1 + weight)` convention so it
/// can drive the standard `rms_norm` directly.
fn decode_norm_1plus(
    model: &mut LazyRllmModel,
    name: &str,
    expected_len: usize,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let tensor = model.decode_tensor(name, budget)?;
    if tensor.shape != [expected_len] {
        return Err(RuntimeError::Shape(format!(
            "gemma norm {name} shape {:?} does not match expected [{expected_len}]",
            tensor.shape
        )));
    }
    let mut data = tensor.data;
    for value in data.iter_mut() {
        *value += 1.0;
    }
    Ok(data)
}

pub fn prepare_gemma_transformer_from_metadata(
    model: &mut LazyRllmModel,
    generation: GemmaGenerationConfig,
) -> Result<PreparedGemmaTransformer> {
    let (build, num_layers) = {
        let config = require_config(model)?;
        let hidden_size = require_usize("hidden_size", config.hidden_size)?;
        let num_heads = require_usize("num_attention_heads", config.num_attention_heads)?;
        let num_key_value_heads = config
            .num_key_value_heads
            .map(|v| require_usize("num_key_value_heads", Some(v)))
            .transpose()?
            .unwrap_or(num_heads);
        // Gemma sets head_dim explicitly (256), independent of hidden/heads.
        let head_dim = config
            .head_dim
            .map(|v| require_usize("head_dim", Some(v)))
            .transpose()?
            .unwrap_or(hidden_size / num_heads);
        let query_pre_attn_scalar = config.query_pre_attn_scalar.unwrap_or(head_dim as f32);
        let max_seq_len = generation
            .max_seq_len
            .or_else(|| config.max_position_embeddings.and_then(|v| usize::try_from(v).ok()))
            .unwrap_or(2048);
        let num_layers = require_usize("num_hidden_layers", config.num_hidden_layers)?;
        let build = GemmaBuildConfig {
            max_new_tokens: generation.max_new_tokens,
            max_seq_len,
            num_heads,
            num_key_value_heads,
            head_dim,
            hidden_size,
            intermediate_size: require_usize("intermediate_size", config.intermediate_size)?,
            rms_norm_eps: config.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: config.rope_theta.unwrap_or(1_000_000.0),
            rope_local_base_freq: config.rope_local_base_freq.unwrap_or(10_000.0),
            rope_scaling_factor: config
                .rope_scaling_factor
                .unwrap_or(GEMMA3_DEFAULT_ROPE_SCALING_FACTOR),
            sliding_window_pattern: config
                .sliding_window_pattern
                .map(|v| v as usize)
                .unwrap_or(1),
            attn_scale: 1.0 / query_pre_attn_scalar.sqrt(),
            embed_scale: (hidden_size as f32).sqrt(),
            causal: generation.causal,
            sampling: generation.sampling,
        };
        (build, num_layers)
    };

    let embedding_weight = "model.embed_tokens.weight".to_string();
    let lm_head_weight = if model.tensor("lm_head.weight").is_ok() {
        "lm_head.weight".to_string()
    } else {
        embedding_weight.clone()
    };

    let final_layernorm =
        decode_norm_1plus(model, "model.norm.weight", build.hidden_size, &mut MemoryBudget::unbounded())?;

    let mut layers = Vec::with_capacity(num_layers);
    let mut layer_norms = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        layers.push(GemmaBlockTensorNames {
            q_weight: format!("model.layers.{i}.self_attn.q_proj.weight"),
            k_weight: format!("model.layers.{i}.self_attn.k_proj.weight"),
            v_weight: format!("model.layers.{i}.self_attn.v_proj.weight"),
            o_weight: format!("model.layers.{i}.self_attn.o_proj.weight"),
            gate_weight: format!("model.layers.{i}.mlp.gate_proj.weight"),
            up_weight: format!("model.layers.{i}.mlp.up_proj.weight"),
            down_weight: format!("model.layers.{i}.mlp.down_proj.weight"),
        });
        layer_norms.push(decode_layer_norms(model, i, &build)?);
    }
    if layers.is_empty() {
        return Err(RuntimeError::Shape("gemma model requires at least one layer".to_string()));
    }

    Ok(PreparedGemmaTransformer {
        config: build,
        embedding_weight,
        lm_head_weight,
        layers,
        layer_norms,
        final_layernorm,
    })
}

fn decode_layer_norms(
    model: &mut LazyRllmModel,
    layer: usize,
    build: &GemmaBuildConfig,
) -> Result<GemmaLayerNorms> {
    let hidden = build.hidden_size;
    let head_dim = build.head_dim;
    let mut budget = MemoryBudget::unbounded();
    let mut norm = |model: &mut LazyRllmModel, suffix: &str, len: usize| -> Result<Vec<f32>> {
        decode_norm_1plus(model, &format!("model.layers.{layer}.{suffix}"), len, &mut budget)
    };
    Ok(GemmaLayerNorms {
        input_layernorm: norm(model, "input_layernorm.weight", hidden)?,
        post_attention_layernorm: norm(model, "post_attention_layernorm.weight", hidden)?,
        pre_feedforward_layernorm: norm(model, "pre_feedforward_layernorm.weight", hidden)?,
        post_feedforward_layernorm: norm(model, "post_feedforward_layernorm.weight", hidden)?,
        q_norm: norm(model, "self_attn.q_norm.weight", head_dim)?,
        k_norm: norm(model, "self_attn.k_norm.weight", head_dim)?,
    })
}

/// Tied-embedding context resolved once per generation/session: the embedding
/// tensor id, vocab size, and an optional resident f32 table (None when the
/// bf16-direct mmap path is used).
struct GemmaEmbedCtx<'a> {
    embed_id: u64,
    vocab_size: usize,
    embedding_f32: Option<&'a [f32]>,
    // R158b: a resident bf16 embedding (604 MB) for a non-raw bf16 tensor (e.g. rANS),
    // instead of the 1.2 GB f32 table. Tried before f32 / with_raw_tensor.
    embedding_bf16: Option<&'a [u8]>,
}

/// One transformer forward over `current_tokens` (prefill chunk or a single
/// decode token): input embed → blocks (appending to `caches`) → final norm →
/// LM head, returning the next-token logits. Shared by the single-shot
/// generator and the interactive `GemmaChatSession` so both stay identical.
#[allow(clippy::too_many_arguments)]
fn gemma_forward_logits(
    model: &mut LazyRllmModel,
    prepared: &PreparedGemmaTransformer,
    embed: &GemmaEmbedCtx,
    current_tokens: &[usize],
    position_offset: usize,
    caches: &mut [KvCache],
    budget: &mut MemoryBudget,
    profile_step: usize,
) -> Result<Vec<f32>> {
    let build = &prepared.config;
    let hidden = build.hidden_size;
    let seq_len = current_tokens.len();

    let mut hidden_states = gemma_embed_input(
        model,
        embed.embedding_f32,
        embed.embedding_bf16,
        embed.embed_id,
        &prepared.embedding_weight,
        current_tokens,
        embed.vocab_size,
        hidden,
        build.embed_scale,
        budget,
    )?;

    let layers_started = crate::q8_kernel_profile_enabled().then(std::time::Instant::now);
    for (i, names) in prepared.layers.iter().enumerate() {
        hidden_states = streaming_gemma_transformer_block(
            model,
            &hidden_states,
            names,
            &prepared.layer_norms[i],
            build,
            GemmaBlockRuntime { seq_len, position_offset, layer_index: i },
            budget,
            Some(&mut caches[i]),
        )?;
    }
    if let Some(t) = layers_started {
        eprintln!(
            "[gemma-profile] step {profile_step} seq_len={seq_len}: {} layers {:.0}ms",
            prepared.layers.len(),
            t.elapsed().as_secs_f64() * 1000.0
        );
    }

    hidden_states = rms_norm(&hidden_states, &prepared.final_layernorm, seq_len, hidden, build.rms_norm_eps)?;

    let last_hidden = &hidden_states[(seq_len - 1) * hidden..];
    // R131: parallel LM-head GEMV over the 262k-row vocabulary (bf16-direct from
    // mmap, or f32 fallback).
    let lm_head_started = crate::q8_kernel_profile_enabled().then(std::time::Instant::now);
    let logits = gemma_lm_head(model, embed.embedding_f32, embed.embedding_bf16, embed.embed_id, &prepared.embedding_weight, last_hidden, embed.vocab_size, hidden, budget)?;
    if let Some(t) = lm_head_started {
        eprintln!(
            "[gemma-profile] step {profile_step} lm_head {:.0}ms (vocab={})",
            t.elapsed().as_secs_f64() * 1000.0,
            embed.vocab_size
        );
    }

    Ok(logits)
}

pub fn gemma_generate_from_model(
    model: &mut LazyRllmModel,
    prepared: &PreparedGemmaTransformer,
    prompt_token_ids: &[usize],
    budget: &mut MemoryBudget,
    options: GemmaGenerationOptions,
    on_token: &mut dyn FnMut(usize) -> bool,
) -> Result<GemmaTextGenerationResult> {
    let build = &prepared.config;
    let mut token_ids = prompt_token_ids.to_vec();
    let mut generated_token_ids = Vec::new();

    let mut caches = Vec::with_capacity(prepared.layers.len());
    for _ in 0..prepared.layers.len() {
        caches.push(KvCache::new(build.num_key_value_heads, build.head_dim, build.max_seq_len)?);
    }

    // Tied embeddings. Materializing the 262208×2560 table as f32 costs 2.68 GB
    // resident — which evicts the q8 weights on an 8 GB device (the decode/RAM
    // bottleneck). Instead read the bf16 table DIRECTLY from the mmap (zero-copy)
    // and dequant-on-the-fly for both the input lookup and the LM head. Fall back
    // to a one-time f32 decode only if the embedding isn't a contiguous bf16 raw
    // tensor (e.g. a different codec/dtype).
    let (embed_id, vocab_size, embedding_f32, embedding_bf16) = resolve_gemma_embedding(model, prepared, budget)?;
    let embed_ctx = GemmaEmbedCtx {
        embed_id,
        vocab_size,
        embedding_f32: embedding_f32.as_deref(),
        embedding_bf16: embedding_bf16.as_deref(),
    };

    // For parity we keep the first decode step's logits (the prefill → first
    // predicted token over the raw prompt), which is the most directly
    // comparable to a single HF forward.
    let mut first_step_logits = None;
    for step in 0..build.max_new_tokens {
        let current_tokens: &[usize] = if step == 0 {
            prompt_token_ids
        } else {
            &generated_token_ids[generated_token_ids.len() - 1..]
        };
        let position_offset = token_ids.len() - current_tokens.len();

        let logits = gemma_forward_logits(
            model,
            prepared,
            &embed_ctx,
            current_tokens,
            position_offset,
            &mut caches,
            budget,
            step,
        )?;

        let next_token = match build.sampling {
            StreamingSamplingConfig::Argmax => sample_argmax(&logits)?,
            StreamingSamplingConfig::TopP { temperature, top_p, seed } => {
                sample_top_p(&logits, temperature, top_p, seed)?
            }
        };
        token_ids.push(next_token);
        generated_token_ids.push(next_token);
        if step == 0 && options.collect_logits {
            first_step_logits = Some(logits);
        }
        if !on_token(next_token) {
            break;
        }
    }

    let context_echo_bytes = caches.iter().map(KvCache::resident_bytes).sum();
    Ok(GemmaTextGenerationResult {
        prompt_token_ids: prompt_token_ids.to_vec(),
        generated_token_ids,
        token_ids,
        context_echo_bytes,
        logits: first_step_logits,
    })
}

/// Capacity-bound mode (RLLM_STREAM_EMBEDDING): stream the tied embedding from the
/// container instead of holding it resident. Trades decode-per-token speed for a
/// resident footprint near the compressed size — for the >RAM regime where the bf16
/// table would not fit. Only affects non-raw bf16 embeddings (e.g. rANS/bit-plane).
fn stream_embedding_enabled() -> bool {
    std::env::var("RLLM_STREAM_EMBEDDING")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// Resolve the tied embedding once: prefer the bf16-direct mmap path (no 2.68 GB
/// f32 materialization); otherwise decode the table to a resident f32 vector.
fn resolve_gemma_embedding(
    model: &mut LazyRllmModel,
    prepared: &PreparedGemmaTransformer,
    budget: &mut MemoryBudget,
) -> Result<(u64, usize, Option<Vec<f32>>, Option<Vec<u8>>)> {
    let embed_meta = model.tensor(&prepared.embedding_weight)?.clone();
    let vocab_size = embed_meta.shape.first().copied().unwrap_or(0) as usize;
    let embed_id = embed_meta.tensor_id;
    let is_bf16 = embed_meta.dtype == DType::Bf16;
    let raw_bf16 = is_bf16 && model.with_raw_tensor(embed_id, |_| Ok::<(), RuntimeError>(()))?.is_some();
    if raw_bf16 {
        // Zero-copy bf16 from the mmap — no resident table at all.
        return Ok((embed_id, vocab_size, None, None));
    }
    if is_bf16 {
        if stream_embedding_enabled() {
            // R162 capacity-bound: do NOT hold the 604 MB bf16 table resident. Both the
            // input lookup and the LM head stream the embedding from the container per
            // use (decode per chunk, never materialize) — so peak resident stays near
            // the compressed size. The trade is speed (re-decode per token) for RAM, the
            // >RAM regime's whole point.
            return Ok((embed_id, vocab_size, None, None));
        }
        // R158b: non-raw bf16 (e.g. rANS-compressed) — decode ONCE to resident bf16
        // bytes (604 MB) instead of the f32 table (1.2 GB).
        let bf16 = model.decode_tensor_raw_bytes(&prepared.embedding_weight)?;
        return Ok((embed_id, vocab_size, None, Some(bf16)));
    }
    // Non-bf16 embedding (e.g. q8) — keep the f32 table.
    let f32 = model.decode_tensor(&prepared.embedding_weight, budget)?.data;
    Ok((embed_id, vocab_size, Some(f32), None))
}

/// A multi-turn Gemma chat with a RESIDENT KV cache. Built once (model load +
/// embedding resolve), then each turn prefills only the new tokens into the
/// existing caches and decodes the reply — so per-turn latency is independent of
/// the conversation length (unlike re-prefilling the whole history every turn).
pub struct GemmaChatSession {
    caches: Vec<KvCache>,
    total_tokens: usize,
    max_context: usize,
    embed_id: u64,
    vocab_size: usize,
    embedding_f32: Option<Vec<f32>>,
    embedding_bf16: Option<Vec<u8>>,
}

impl GemmaChatSession {
    /// Allocate the per-layer KV caches (sized to `max_context`) and resolve the
    /// tied embedding once.
    pub fn new(
        model: &mut LazyRllmModel,
        prepared: &PreparedGemmaTransformer,
        budget: &mut MemoryBudget,
        max_context: usize,
    ) -> Result<Self> {
        let build = &prepared.config;
        let mut caches = Vec::with_capacity(prepared.layers.len());
        for _ in 0..prepared.layers.len() {
            caches.push(KvCache::new(build.num_key_value_heads, build.head_dim, max_context)?);
        }
        let (embed_id, vocab_size, embedding_f32, embedding_bf16) = resolve_gemma_embedding(model, prepared, budget)?;
        Ok(Self { caches, total_tokens: 0, max_context, embed_id, vocab_size, embedding_f32, embedding_bf16 })
    }

    /// Tokens currently held in the KV cache (the running conversation length).
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Drop the conversation: fresh KV caches, position back to zero.
    pub fn reset(&mut self, prepared: &PreparedGemmaTransformer) -> Result<()> {
        let build = &prepared.config;
        for cache in &mut self.caches {
            *cache = KvCache::new(build.num_key_value_heads, build.head_dim, self.max_context)?;
        }
        self.total_tokens = 0;
        Ok(())
    }

    /// Prefill `new_tokens` into the resident caches, then greedily/sampled-decode
    /// the reply, streaming each token through `on_token` and stopping at a
    /// `stop_ids` token (which is NOT fed to the cache, so the next turn supplies
    /// the turn-closing marker). Returns the generated tokens (excluding stop).
    #[allow(clippy::too_many_arguments)]
    pub fn feed_and_decode(
        &mut self,
        model: &mut LazyRllmModel,
        prepared: &PreparedGemmaTransformer,
        budget: &mut MemoryBudget,
        new_tokens: &[usize],
        max_new: usize,
        stop_ids: &[usize],
        on_token: &mut dyn FnMut(usize) -> bool,
    ) -> Result<Vec<usize>> {
        if new_tokens.is_empty() {
            return Ok(Vec::new());
        }
        if self.total_tokens + new_tokens.len() > self.max_context {
            return Err(RuntimeError::Shape(format!(
                "gemma chat context full ({} + {} > {})",
                self.total_tokens,
                new_tokens.len(),
                self.max_context
            )));
        }
        let build = &prepared.config;
        let embed = GemmaEmbedCtx {
            embed_id: self.embed_id,
            vocab_size: self.vocab_size,
            embedding_f32: self.embedding_f32.as_deref(),
            embedding_bf16: self.embedding_bf16.as_deref(),
        };
        let mut generated = Vec::new();
        let mut current: Vec<usize> = new_tokens.to_vec();
        for step in 0..max_new {
            let position_offset = self.total_tokens;
            let logits = gemma_forward_logits(
                model, prepared, &embed, &current, position_offset, &mut self.caches, budget, step,
            )?;
            // `current` is now resident in the caches.
            self.total_tokens += current.len();

            let next = match build.sampling {
                StreamingSamplingConfig::Argmax => sample_argmax(&logits)?,
                StreamingSamplingConfig::TopP { temperature, top_p, seed } => {
                    sample_top_p(&logits, temperature, top_p, seed)?
                }
            };
            if stop_ids.contains(&next) {
                break;
            }
            generated.push(next);
            if !on_token(next) || self.total_tokens >= self.max_context {
                break;
            }
            current = vec![next];
        }
        Ok(generated)
    }
}

/// Input embedding lookup scaled by `embed_scale` — from a resident f32 table
/// (fallback) or bf16-direct from the mmap (default; no 2.68 GB f32 alloc).
#[allow(clippy::too_many_arguments)]
fn gemma_embed_input(
    model: &mut LazyRllmModel,
    embedding_f32: Option<&[f32]>,
    embedding_bf16: Option<&[u8]>,
    embed_id: u64,
    embedding_weight: &str,
    token_ids: &[usize],
    vocab_size: usize,
    hidden: usize,
    embed_scale: f32,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    // R158b: resident bf16 table (non-raw bf16, e.g. rANS) — half the f32 footprint.
    if let Some(bf16) = embedding_bf16 {
        return gemma_embed_lookup_bf16(bf16, token_ids, hidden, vocab_size, embed_scale);
    }
    if let Some(emb) = embedding_f32 {
        let mut h = embedding_lookup(emb, vocab_size, hidden, token_ids)?;
        for value in h.iter_mut() {
            *value *= embed_scale;
        }
        return Ok(h);
    }
    // Raw bf16 → zero-copy gather straight from the mmap.
    if let Some(h) = model.with_raw_tensor(embed_id, |bf16| {
        gemma_embed_lookup_bf16(bf16, token_ids, hidden, vocab_size, embed_scale)
    })? {
        return Ok(h);
    }
    // R162: non-raw bf16 with no resident table (RLLM_STREAM_EMBEDDING) — stream only
    // the chunk(s) holding the requested rows.
    gemma_embed_input_streaming(model, embedding_weight, token_ids, hidden, embed_scale, budget)
}

/// Tied LM head — f32 table (fallback) or bf16-direct from the mmap (default).
#[allow(clippy::too_many_arguments)]
fn gemma_lm_head(
    model: &mut LazyRllmModel,
    embedding_f32: Option<&[f32]>,
    embedding_bf16: Option<&[u8]>,
    embed_id: u64,
    embedding_weight: &str,
    last_hidden: &[f32],
    vocab_size: usize,
    hidden: usize,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    // R149a: opt-in streaming lm-head from a bit-plane sidecar (capacity-bound mode).
    #[cfg(target_arch = "aarch64")]
    if let Ok(sidecar) = std::env::var("RLLM_STREAM_LMHEAD") {
        if !sidecar.is_empty() {
            let _ = (embedding_f32, embedding_bf16, embed_id); // resident inputs unused here
            return crate::streaming::stream_lmhead_from_sidecar(&sidecar, last_hidden);
        }
    }
    // R158b: resident bf16 table (non-raw bf16, e.g. rANS).
    if let Some(bf16) = embedding_bf16 {
        return Ok(lm_head_logits_parallel_bf16(last_hidden, bf16, vocab_size, hidden));
    }
    if let Some(emb) = embedding_f32 {
        return Ok(lm_head_logits_parallel(last_hidden, emb, vocab_size, hidden));
    }
    // Raw bf16 → zero-copy parallel GEMV straight from the mmap.
    if let Some(logits) = model.with_raw_tensor(embed_id, |bf16| {
        Ok::<_, RuntimeError>(lm_head_logits_parallel_bf16(
            last_hidden,
            bf16,
            vocab_size,
            hidden,
        ))
    })? {
        return Ok(logits);
    }
    // R162: non-raw bf16 with no resident table (RLLM_STREAM_EMBEDDING) — stream the
    // embedding as an output projection (decode per chunk, never materialize 604 MB).
    gemma_lm_head_streaming(model, embedding_weight, last_hidden, vocab_size, hidden, budget)
}

/// Gather `token_ids` rows from the bf16 embedding table (mmap bytes), dequant
/// bf16→f32, scaled by `embed_scale`. Row-major, 2 bytes/element.
fn gemma_embed_lookup_bf16(
    bf16: &[u8],
    token_ids: &[usize],
    hidden: usize,
    vocab_size: usize,
    embed_scale: f32,
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(token_ids.len() * hidden);
    for &token in token_ids {
        if token >= vocab_size {
            return Err(RuntimeError::Shape(format!(
                "token id {token} out of range for vocab {vocab_size}"
            )));
        }
        let row_base = token * hidden * 2;
        for h in 0..hidden {
            let off = row_base + h * 2;
            let bits = u16::from_le_bytes([bf16[off], bf16[off + 1]]);
            out.push(bf16_to_f32(bits) * embed_scale);
        }
    }
    Ok(out)
}

