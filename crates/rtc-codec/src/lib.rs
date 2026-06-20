//! RTC - Rama Tensor Codec
//!
//! Lossless tensor compression codecs for RLLM.
//! Every codec must satisfy: decode(encode(input)) == input (bit-identical).

mod bitplane;
mod bitreader_fast;
mod codec;
mod dfloat;
mod error;
mod huff;
mod rans;
mod raw;
mod rle;

pub use bitplane::*;
pub use codec::*;
pub use dfloat::*;
pub use error::*;
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
