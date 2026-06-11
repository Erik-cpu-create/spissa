//! Error types for the RTC codec crate

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodecError {
    #[error("Codec error: {0}")]
    General(String),

    #[error("Round-trip verification failed for codec {codec_id}")]
    RoundTripFailed { codec_id: String },

    #[error("Unknown codec: {0}")]
    UnknownCodec(String),

    #[error("Invalid compressed data: {0}")]
    InvalidData(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CodecError>;
