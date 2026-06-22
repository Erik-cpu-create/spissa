// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! Error types for the RLLM container crate

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ContainerError {
    #[error("Invalid magic header: expected 'SPSA'")]
    InvalidMagic,

    #[error("Unsupported format version: {0}")]
    UnsupportedVersion(u32),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Checksum mismatch for {context}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        context: String,
        expected: String,
        actual: String,
    },

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Chunk not found: {0}")]
    ChunkNotFound(u64),

    #[error("Invalid byte range for {context}: offset={offset}, len={len}, size={size}")]
    InvalidRange {
        context: String,
        offset: u64,
        len: u64,
        size: u64,
    },

    #[error("Truncated file: expected {expected} bytes, got {actual}")]
    TruncatedFile { expected: u64, actual: u64 },
}

pub type Result<T> = std::result::Result<T, ContainerError>;
