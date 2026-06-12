//! RLLM runtime foundation.
//!
//! Phase 5 starts with full-decode loading: read a `.rllm` container,
//! decode every tensor into memory, convert supported dtypes to `f32`, and
//! expose small tensor operations needed by a toy transformer runtime.

mod echo;
mod error;
mod gpt_neox;
mod lazy;
mod loader;
mod memory;
mod ops;
mod planner;
mod rotary;
mod streaming;
mod tensor;
mod tiny;
mod tokenizer;

pub use echo::{
    streaming_echo_transformer_decode_step_from_model,
    streaming_echo_transformer_generate_from_model, streaming_echo_transformer_prefill_from_model,
    streaming_rama_transformer_decode_step_from_model,
    streaming_rama_transformer_generate_from_model, streaming_rama_transformer_prefill_from_model,
    RamaContextState, StreamingEchoGenerationResult, StreamingEchoTransformerConfig,
    StreamingEchoTransformerParameters, StreamingEchoTransformerTensorNames,
    StreamingRamaGenerationResult, StreamingRamaTransformerConfig,
    StreamingRamaTransformerParameters, StreamingRamaTransformerTensorNames,
};
pub use error::{Result, RuntimeError};
pub use gpt_neox::{
    prepare_gpt_neox_echo_transformer_from_metadata, prepare_gpt_neox_echo_transformer_from_model,
    prepare_gpt_neox_rama_layer_decode_transformer_from_metadata,
    prepare_gpt_neox_rama_layer_decode_transformer_from_model,
    prepare_gpt_neox_rama_transformer_from_metadata, prepare_gpt_neox_rama_transformer_from_model,
    GptNeoxEchoBuildConfig, GptNeoxEchoGenerationConfig, GptNeoxRamaBuildConfig,
    GptNeoxRamaGenerationConfig, GptNeoxTextGenerationResult, LayerDecodedGptNeoxRamaTransformer,
    OwnedStreamingBlockParameters, OwnedStreamingBlockTensorNames, PreparedGptNeoxEchoTransformer,
    PreparedGptNeoxRamaTransformer,
};
pub use lazy::{LazyModelStats, LazyRllmModel};
pub use loader::{FullDecodeModel, FullDecodeStats};
pub use memory::MemoryBudget;
pub use ops::*;
pub use planner::{
    build_runtime_plan, ModelShapeHints, PlanStatus, PlanStep, RuntimeMode, RuntimePlan,
    RuntimePlanConfig,
};
pub use rotary::{
    apply_gpt_neox_rotary_inplace, gpt_neox_rotary_dim, scaled_dot_product_attention_with_cache,
    KvAttentionConfig, KvCache, RotaryEmbeddingConfig,
};
pub use streaming::{
    streaming_attention_from_model, streaming_attention_with_runtime_from_model,
    streaming_linear_from_model, streaming_mlp_from_model, streaming_transformer_block_from_model,
    streaming_transformer_block_with_runtime_from_model, StreamingAttentionConfig,
    StreamingAttentionRuntime, StreamingBlockConfig, StreamingBlockParameters,
    StreamingBlockRuntime, StreamingBlockTensorNames, StreamingLinearConfig, StreamingMlpConfig,
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
