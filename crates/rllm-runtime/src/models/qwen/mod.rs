// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! Qwen3.5-2B (qwen3_5 / Qwen3-Next-style) text-only adapter: hybrid Gated-DeltaNet
//! linear attention + gated full attention. See `docs/qwen3_5-adapter-design.md`.

pub mod api;
pub mod generate;
pub mod model;
pub mod session;

pub use api::{
    prepare_qwen_transformer_from_metadata, qwen_generate_from_model, QwenGenerationConfig,
};
pub use model::{PreparedQwenTransformer, QwenBuildConfig, QwenLayerKind};
pub use session::{QwenSession, SamplingParams};
