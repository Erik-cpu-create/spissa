// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

//! RLLM Container - Binary container format for compressed LLM models
//!
//! This crate handles the .spsa file format: parsing, writing, and managing
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

/// Magic bytes for `.spsa` files: "SPSA". (Pre-rebrand files used "RLLM"; existing models
/// were patched in place — first 4 bytes only — so no legacy magic is accepted on read.)
pub const SPSA_MAGIC: &[u8; 4] = b"SPSA";

/// Current format version
pub const RLLM_VERSION: u32 = 1;
