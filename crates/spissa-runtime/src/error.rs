// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("unsupported dtype for runtime tensor: {0}")]
    UnsupportedDType(String),

    #[error("shape mismatch: {0}")]
    Shape(String),

    #[error("missing tensor: {0}")]
    MissingTensor(String),

    #[error("unknown codec: {0}")]
    UnknownCodec(String),

    #[error("checksum mismatch: {0}")]
    ChecksumMismatch(String),

    #[error("invalid tensor data: {0}")]
    InvalidTensorData(String),

    #[error("memory budget exceeded while reserving {requested} bytes for {label}: current={current}, limit={limit}")]
    MemoryBudgetExceeded {
        requested: usize,
        current: usize,
        limit: usize,
        label: String,
    },

    #[error(
        "memory budget underflow while releasing {released} bytes for {label}: current={current}"
    )]
    MemoryBudgetUnderflow {
        released: usize,
        current: usize,
        label: String,
    },

    #[error("invalid runtime mode: {0}")]
    InvalidRuntimeMode(String),

    #[error(transparent)]
    Container(#[from] spissa_container::ContainerError),

    #[error(transparent)]
    Codec(#[from] rtc_codec::CodecError),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;
