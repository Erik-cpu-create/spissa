use std::sync::{Mutex, OnceLock};
use std::time::Duration;

const PROFILE_ENV: &str = "RLLM_Q8_KERNEL_PROFILE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q8KernelPath {
    BatchGt1Scaled,
    BatchGt1NormalScale,
    BatchGt1NormalBatch4,
    BatchGt1NormalBatch4Setup,
    BatchGt1NormalBatch4Kernel,
    BatchGt1NormalTail,
    BatchGt1MultiplyAdvance,
    BatchGt1MultiplyScale,
    BatchGt1MultiplyBatch4,
    BatchGt1MultiplyTail,
    BatchGt1MultiplyFinish,
    Batch1CompleteLinear,
    Batch1CompleteMultiply,
    Batch1CompleteArgmax,
    ContiguousI8Dot,
    SplitRowScalar,
}

impl Q8KernelPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BatchGt1Scaled => "batch_gt1_scaled",
            Self::BatchGt1NormalScale => "batch_gt1_normal_scale",
            Self::BatchGt1NormalBatch4 => "batch_gt1_normal_batch4",
            Self::BatchGt1NormalBatch4Setup => "batch_gt1_normal_batch4_setup",
            Self::BatchGt1NormalBatch4Kernel => "batch_gt1_normal_batch4_kernel",
            Self::BatchGt1NormalTail => "batch_gt1_normal_tail",
            Self::BatchGt1MultiplyAdvance => "batch_gt1_multiply_advance",
            Self::BatchGt1MultiplyScale => "batch_gt1_multiply_scale",
            Self::BatchGt1MultiplyBatch4 => "batch_gt1_multiply_batch4",
            Self::BatchGt1MultiplyTail => "batch_gt1_multiply_tail",
            Self::BatchGt1MultiplyFinish => "batch_gt1_multiply_finish",
            Self::Batch1CompleteLinear => "batch1_complete_linear",
            Self::Batch1CompleteMultiply => "batch1_complete_multiply",
            Self::Batch1CompleteArgmax => "batch1_complete_argmax",
            Self::ContiguousI8Dot => "contiguous_i8_dot",
            Self::SplitRowScalar => "split_row_scalar",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8KernelProfileRow {
    pub path: &'static str,
    pub calls: u64,
    pub blocks: u64,
    pub rows: u64,
    pub batch_items: u64,
    pub elapsed_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8KernelProfileSnapshot {
    pub ree_kernel: &'static str,
    pub rows: Vec<Q8KernelProfileRow>,
}

#[derive(Debug, Default)]
struct Q8KernelProfileState {
    rows: Vec<Q8KernelProfileRow>,
}

static PROFILE: OnceLock<Mutex<Q8KernelProfileState>> = OnceLock::new();

pub fn q8_kernel_profile_enabled() -> bool {
    matches!(
        std::env::var(PROFILE_ENV)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
    )
}

pub fn record_q8_kernel_path(
    path: Q8KernelPath,
    calls: u64,
    blocks: u64,
    rows: u64,
    batch_items: u64,
    elapsed: Duration,
) {
    if calls == 0 && blocks == 0 && rows == 0 && batch_items == 0 {
        return;
    }

    let mutex = PROFILE.get_or_init(|| Mutex::new(Q8KernelProfileState::default()));
    let mut state = mutex.lock().expect("Q8 profile mutex poisoned");
    let key = path.as_str();
    if let Some(row) = state.rows.iter_mut().find(|row| row.path == key) {
        row.calls += calls;
        row.blocks += blocks;
        row.rows += rows;
        row.batch_items += batch_items;
        row.elapsed_ns += elapsed.as_nanos();
        return;
    }

    state.rows.push(Q8KernelProfileRow {
        path: key,
        calls,
        blocks,
        rows,
        batch_items,
        elapsed_ns: elapsed.as_nanos(),
    });
}

pub fn q8_kernel_profile_snapshot_and_reset() -> Option<Q8KernelProfileSnapshot> {
    let mutex = PROFILE.get()?;
    let mut state = mutex.lock().expect("Q8 profile mutex poisoned");
    if state.rows.is_empty() {
        return None;
    }
    let mut rows = std::mem::take(&mut state.rows);
    rows.sort_by(|left, right| right.elapsed_ns.cmp(&left.elapsed_ns));
    Some(Q8KernelProfileSnapshot {
        ree_kernel: "REEGLASS-Q8-HOTLOOP-PROFILER",
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_profile_records_sorts_and_resets_rows() {
        let _ = q8_kernel_profile_snapshot_and_reset();

        record_q8_kernel_path(
            Q8KernelPath::ContiguousI8Dot,
            1,
            2,
            3,
            4,
            Duration::from_nanos(10),
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1Scaled,
            1,
            2,
            3,
            4,
            Duration::from_nanos(30),
        );
        record_q8_kernel_path(
            Q8KernelPath::ContiguousI8Dot,
            2,
            3,
            4,
            5,
            Duration::from_nanos(20),
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1NormalBatch4,
            4,
            8,
            0,
            16,
            Duration::from_nanos(40),
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1NormalBatch4Setup,
            4,
            4,
            0,
            16,
            Duration::from_nanos(15),
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1NormalBatch4Kernel,
            4,
            4,
            0,
            16,
            Duration::from_nanos(45),
        );
        record_q8_kernel_path(
            Q8KernelPath::BatchGt1MultiplyFinish,
            2,
            2,
            2,
            0,
            Duration::from_nanos(5),
        );

        let snapshot = q8_kernel_profile_snapshot_and_reset().unwrap();
        assert_eq!(snapshot.ree_kernel, "REEGLASS-Q8-HOTLOOP-PROFILER");
        assert_eq!(snapshot.rows[0].path, "batch_gt1_normal_batch4_kernel");
        assert_eq!(snapshot.rows[0].calls, 4);
        assert_eq!(snapshot.rows[0].elapsed_ns, 45);
        assert!(snapshot
            .rows
            .iter()
            .any(|row| row.path == "batch_gt1_normal_batch4"));
        assert!(snapshot
            .rows
            .iter()
            .any(|row| row.path == "batch_gt1_multiply_finish"));
        assert!(snapshot
            .rows
            .iter()
            .any(|row| row.path == "batch_gt1_normal_batch4_setup"));
        assert!(snapshot
            .rows
            .iter()
            .any(|row| row.path == "batch_gt1_normal_batch4_kernel"));
        assert!(q8_kernel_profile_snapshot_and_reset().is_none());
    }
}
