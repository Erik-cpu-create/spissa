//! RLLM runtime foundation.
//!
//! Phase 5 starts with full-decode loading: read a `.rllm` container,
//! decode every tensor into memory, convert supported dtypes to `f32`, and
//! expose small tensor operations needed by a toy transformer runtime.
#![allow(clippy::too_many_arguments)]

#[path = "session.rs"]
mod chat_session;
mod echo;
mod error;
mod lazy;
mod loader;
mod memory;
pub mod models;
mod ops;
mod planner;
mod rolling;
mod rotary;
mod speed;
mod streaming;
mod tensor;
mod tiny;
mod tokenizer;
mod trace;

pub use chat_session::{
    RamaChatSession, RamaRepetitionStats, RamaRollingStats, RamaSessionAdapter,
    RamaSessionPhaseTimings, RamaSessionStep, RamaSessionTurnMetrics, RamaSessionTurnResult,
    RamaTransformerPhaseTimings,
};
pub use echo::{
    streaming_echo_transformer_decode_step_from_model,
    streaming_echo_transformer_generate_from_model, streaming_echo_transformer_prefill_from_model,
    streaming_rama_transformer_decode_step_from_model,
    streaming_rama_transformer_generate_from_model, streaming_rama_transformer_prefill_from_model,
    RamaContextState, RamaGenerationTiming, StreamingEchoGenerationResult,
    StreamingEchoTransformerConfig, StreamingEchoTransformerParameters,
    StreamingEchoTransformerTensorNames, StreamingRamaGenerationResult,
    StreamingRamaTransformerConfig, StreamingRamaTransformerParameters,
    StreamingRamaTransformerTensorNames,
};
pub use error::{Result, RuntimeError};
pub use lazy::{LazyModelStats, LazyRllmModel, RamaIntegrityMode};
pub use loader::{FullDecodeModel, FullDecodeStats};
pub use memory::MemoryBudget;
pub use models::gpt_neox::*;
pub use models::llama::*;
pub use ops::*;
pub use planner::{
    build_runtime_plan, ModelShapeHints, PlanStatus, PlanStep, RuntimeMode, RuntimePlan,
    RuntimePlanConfig,
};
pub use rotary::{
    apply_gpt_neox_rotary_inplace, gpt_neox_rotary_dim, scaled_dot_product_attention_with_cache,
    KvAttentionConfig, KvCache, RotaryEmbeddingConfig,
};
pub use speed::{
    parse_aip_column_cache_enabled, parse_aip_edge_layers, parse_aip_edge_topk,
    parse_aip_input_tiles_enabled, parse_aip_lm_head_agreement_enabled,
    parse_aip_lm_head_novelty_gap_milli, parse_aip_lm_head_novelty_window,
    parse_aip_lm_head_repeat_margin_adaptive_enabled, parse_aip_lm_head_repeat_margin_milli,
    parse_aip_lm_head_rescore, parse_aip_lm_head_rows, parse_aip_no_repeat_last_enabled,
    parse_aip_policy, parse_aip_repeat_run_limit, parse_aip_topk, parse_experimental_speed_enabled,
    parse_turbo_topk, select_top_abs_indices, RamaAipPolicyKind, RamaAipProjectionDecision,
    RamaAipProjectionKind, RamaExperimentalSpeedConfig, RamaExperimentalSpeedStats,
};
pub use streaming::{
    input_tile_sidecar_weight_name, streaming_attention_from_model,
    streaming_attention_with_runtime_from_model,
    streaming_column_cached_sparse_silu_gate_up_from_model,
    streaming_column_cached_sparse_tile_linear_from_model,
    streaming_input_tiled_sparse_silu_gate_up_from_model,
    streaming_input_tiled_sparse_tile_linear_from_model, streaming_linear_from_model,
    streaming_mlp_from_model, streaming_silu_gate_up_from_model,
    streaming_sparse_silu_gate_up_from_model, streaming_sparse_tile_linear_from_model,
    streaming_tile_linear_argmax_candidate_rows_from_model,
    streaming_tile_linear_argmax_candidate_rows_range_from_model,
    streaming_tile_linear_argmax_from_model, streaming_tile_linear_argmax_prefix_from_model,
    streaming_tile_linear_from_model, streaming_tile_linear_multiply_into_from_model,
    streaming_transformer_block_from_model,
    streaming_transformer_block_with_runtime_and_timing_from_model,
    streaming_transformer_block_with_runtime_from_model, SparseColumnCache, SparseColumnCacheStats,
    StreamingAttentionConfig, StreamingAttentionRuntime, StreamingBlockConfig,
    StreamingBlockParameters, StreamingBlockRuntime, StreamingBlockTensorNames,
    StreamingBlockTiming, StreamingLinearConfig, StreamingMlpConfig, StreamingTileLinearConfig,
    DEFAULT_STREAMING_TILE_ELEMENTS,
};
pub use tensor::{bf16_to_f32, fp16_to_f32, Tensor};
pub use tiny::{
    streaming_embedding_lookup_from_model, streaming_tiny_transformer_decode_step_from_model,
    streaming_tiny_transformer_generate_from_model,
    streaming_tiny_transformer_next_token_from_model,
    streaming_tiny_transformer_next_token_with_runtime_from_model,
    streaming_tiny_transformer_prefill_from_model, ContextEchoState, StreamingEmbeddingConfig,
    StreamingNextTokenResult, StreamingSamplingConfig, StreamingTinyGenerationConfig,
    StreamingTinyGenerationResult, StreamingTinyRotaryConfig, StreamingTinyTransformerConfig,
    StreamingTinyTransformerParameters, StreamingTinyTransformerTensorNames,
};
pub use tokenizer::RllmTokenizer;
pub use trace::{RamaTrace, RamaTraceEvent, RamaTraceEventInput};
