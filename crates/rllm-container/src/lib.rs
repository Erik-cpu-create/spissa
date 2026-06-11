//! RLLM Container - Binary container format for compressed LLM models
//!
//! This crate handles the .rllm file format: parsing, writing, and managing
//! the binary layout of compressed model tensors.

mod error;
mod header;
mod metadata;
mod reader;
mod writer;

pub use error::*;
pub use header::*;
pub use metadata::*;
pub use reader::*;
pub use writer::*;

/// Magic bytes for .rllm files: "RLLM"
pub const RLLM_MAGIC: &[u8; 4] = b"RLLM";

/// Current format version
pub const RLLM_VERSION: u32 = 1;
