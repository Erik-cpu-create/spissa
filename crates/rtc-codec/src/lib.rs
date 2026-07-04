// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

//! RTC - Rama Tensor Codec
//!
//! Lossless tensor compression codecs for Spissa.
//! Every codec must satisfy: decode(encode(input)) == input (bit-identical).

mod bitplane;
mod bitreader_fast;
mod codec;
pub mod delta;
mod dfloat;
mod error;
mod forcodec;
mod huff;
mod rans;
mod raw;
mod rle;

pub use bitplane::*;
pub use codec::*;
pub use delta::*;
pub use dfloat::*;
pub use error::*;
pub use forcodec::*;
pub use huff::*;
pub use rans::*;
pub use raw::*;
pub use rle::*;

/// Codec ID for the raw (no compression) codec
pub const CODEC_RAW_V1: &str = "rtc-raw-v1";

/// Codec ID for the RLE codec
pub const CODEC_RLE_V1: &str = "rtc-rle-v1";

/// Codec ID for the byte-level Huffman codec
pub const CODEC_HUFF_V1: &str = "rtc-huff-v1";

/// Codec ID for the lossless bf16 codec
pub const CODEC_DFLOAT_V1: &str = "rtc-dfloat-v1";

/// Codec ID for the SIMD-decodable bit-plane bf16 codec
pub const CODEC_BITPLANE_V1: &str = "rtc-bitplane-v1";

/// Codec ID for the rANS lossless bf16 codec (at the entropy floor, ~10.5 bits/weight)
pub const CODEC_RANS_V1: &str = "rtc-rans-v1";

/// Codec ID for the REEBORN coderless FOR bf16 codec — raw significand + per-tensor
/// fixed-width exponent. Larger (~13 bits/weight) but branch-free, ~6× faster decode;
/// wins in the model>RAM streaming regime where decode bandwidth is the wall.
pub const CODEC_REEBORN_FOR_V1: &str = "rtc-reeborn-for-v1";
