// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

use serde::Serialize;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
pub struct RamaTrace {
    pub schema_version: u32,
    pub model_name: String,
    pub architecture: String,
    pub started_at_unix_ms: u128,
    pub events: Vec<RamaTraceEvent>,
    #[serde(skip)]
    start_instant: Instant,
}

#[derive(Debug, Clone, Serialize)]
pub struct RamaTraceEvent {
    pub event_index: usize,
    pub phase: String,
    pub label: String,
    pub tensor_name: Option<String>,
    pub tensor_id: Option<u64>,
    pub chunk_id: Option<u64>,
    pub codec_id: Option<String>,
    pub compressed_bytes: Option<u64>,
    pub decoded_bytes: Option<u64>,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub budget_current_bytes: usize,
    pub budget_peak_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct RamaTraceEventInput {
    pub phase: String,
    pub label: String,
    pub tensor_name: Option<String>,
    pub tensor_id: Option<u64>,
    pub chunk_id: Option<u64>,
    pub codec_id: Option<String>,
    pub compressed_bytes: Option<u64>,
    pub decoded_bytes: Option<u64>,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub budget_current_bytes: usize,
    pub budget_peak_bytes: usize,
}

impl RamaTrace {
    pub fn new(model_name: impl Into<String>, architecture: impl Into<String>) -> Self {
        let started_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or_default();
        Self {
            schema_version: 1,
            model_name: model_name.into(),
            architecture: architecture.into(),
            started_at_unix_ms,
            events: Vec::new(),
            start_instant: Instant::now(),
        }
    }

    pub fn elapsed_ns_since_start(&self, instant: Instant) -> u64 {
        instant
            .checked_duration_since(self.start_instant)
            .map(saturating_duration_nanos)
            .unwrap_or_default()
    }

    pub fn record(&mut self, input: RamaTraceEventInput) {
        let event_index = self.events.len();
        self.events.push(RamaTraceEvent {
            event_index,
            phase: input.phase,
            label: input.label,
            tensor_name: input.tensor_name,
            tensor_id: input.tensor_id,
            chunk_id: input.chunk_id,
            codec_id: input.codec_id,
            compressed_bytes: input.compressed_bytes,
            decoded_bytes: input.decoded_bytes,
            start_ns: input.start_ns,
            duration_ns: input.duration_ns,
            budget_current_bytes: input.budget_current_bytes,
            budget_peak_bytes: input.budget_peak_bytes,
        });
    }
}

pub fn saturating_duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
