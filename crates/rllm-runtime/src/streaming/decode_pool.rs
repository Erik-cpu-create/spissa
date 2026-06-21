// REEPOOL — persistent decode worker pool (R172).
//
// The decode GEMVs were row-parallelized with a fresh `std::thread::scope` PER
// projection (~182 spawn+join+barrier cycles/token). On slow mobile schedulers
// that per-call OS-thread spawn dominates: measured 1B q8 phone decode was FASTER
// single-threaded (1-thread 1.2 > 2-thread 0.7 tok/s) because spawning cost more
// than the parallelism bought. But with the spawn amortized over a big GEMV, 4-6
// cores genuinely help (microbench 1.8 -> 4.7 GB/s). This pool keeps the workers
// alive so a dispatch is a condvar wake + an atomic work-claim + a barrier — no
// OS thread creation on the hot path — letting decode actually use the cores.
//
// SAFETY MODEL: `parallel_for` blocks until every worker has finished the current
// job before returning, so the borrowed closure (and everything it captures)
// strictly outlives all worker calls. The lifetime erasure to `'static` is sound
// ONLY because of that barrier; do not return early.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

/// A raw `*mut T` carried across pool tasks. SAFETY: every task must write only
/// its OWN disjoint sub-range through this pointer — no two tasks may alias the
/// same element. Used so a `parallel_for` closure (which is `Fn`, not `FnMut`) can
/// scatter into disjoint output rows.
#[derive(Clone, Copy)]
pub(crate) struct DisjointMut<T>(pub *mut T);
// SAFETY: soundness is the caller's disjointness guarantee (see above).
unsafe impl<T> Send for DisjointMut<T> {}
unsafe impl<T> Sync for DisjointMut<T> {}

impl<T> DisjointMut<T> {
    /// Pointer to element `offset`. Taking `self` by value makes a capturing
    /// closure hold the whole (Send+Sync) wrapper rather than the bare `*mut T`
    /// (Rust 2021 disjoint captures would otherwise grab the non-Sync field).
    /// SAFETY: caller must only access its own disjoint range; `offset` in bounds.
    #[inline]
    pub(crate) unsafe fn at(self, offset: usize) -> *mut T {
        self.0.add(offset)
    }
}

/// Erased pointer to the current `&(dyn Fn(usize) + Sync)` job, plus bookkeeping.
struct JobState {
    /// `*const (dyn Fn(usize) + Sync)` fattened into two usizes (data, vtable),
    /// or `(0, 0)` when idle. Valid only between dispatch set/clear (the barrier).
    func: (usize, usize),
    n_tasks: usize,
    generation: u64,
    active: usize, // workers still running the current generation
    shutdown: bool,
}

struct Shared {
    state: Mutex<JobState>,
    cv_work: Condvar, // workers wait here for a new generation
    cv_done: Condvar, // dispatcher waits here for active == 0
    next: AtomicUsize, // shared work-claim counter for the current job
}

pub(crate) struct DecodePool {
    shared: &'static Shared,
    workers: Vec<JoinHandle<()>>,
    size: usize,
}

#[inline]
fn run_claimed(shared: &Shared, func: (usize, usize), n_tasks: usize) {
    // Reconstruct the fat pointer and claim task indices until exhausted.
    let f: *const (dyn Fn(usize) + Sync) =
        unsafe { std::mem::transmute::<(usize, usize), *const (dyn Fn(usize) + Sync)>(func) };
    let f: &(dyn Fn(usize) + Sync) = unsafe { &*f };
    loop {
        let idx = shared.next.fetch_add(1, Ordering::Relaxed);
        if idx >= n_tasks {
            break;
        }
        f(idx);
    }
}

impl DecodePool {
    fn new(size: usize) -> Self {
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            state: Mutex::new(JobState {
                func: (0, 0),
                n_tasks: 0,
                generation: 0,
                active: 0,
                shutdown: false,
            }),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
            next: AtomicUsize::new(0),
        }));
        // Spawn `size - 1` workers; the dispatching thread is the `size`-th worker.
        let mut workers = Vec::with_capacity(size.saturating_sub(1));
        for _ in 1..size {
            workers.push(std::thread::spawn(move || worker_loop(shared)));
        }
        DecodePool { shared, workers, size }
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Run `f(i)` for every `i` in `0..n_tasks` across the pool, returning only
    /// after all calls complete. `f` runs on worker threads AND the caller.
    pub(crate) fn parallel_for<F: Fn(usize) + Sync>(&self, n_tasks: usize, f: F) {
        if n_tasks == 0 {
            return;
        }
        if self.size <= 1 || n_tasks == 1 {
            for i in 0..n_tasks {
                f(i);
            }
            return;
        }
        let f_ref: &(dyn Fn(usize) + Sync) = &f;
        let fat: (usize, usize) = unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync), (usize, usize)>(
                f_ref as *const (dyn Fn(usize) + Sync),
            )
        };
        let workers_to_wake = self.size - 1; // the caller is the size-th worker
        {
            let mut st = self.shared.state.lock().unwrap();
            self.shared.next.store(0, Ordering::Relaxed);
            st.func = fat;
            st.n_tasks = n_tasks;
            st.active = workers_to_wake;
            st.generation = st.generation.wrapping_add(1);
            self.shared.cv_work.notify_all();
        }
        // The caller participates as a worker (uses the dispatching core).
        run_claimed(self.shared, fat, n_tasks);
        // Wait for the spawned workers to drain the remaining tasks.
        let mut st = self.shared.state.lock().unwrap();
        while st.active != 0 {
            st = self.shared.cv_done.wait(st).unwrap();
        }
        // Clear the job pointer BEFORE returning so `f` can be dropped safely.
        st.func = (0, 0);
    }
}

fn worker_loop(shared: &'static Shared) {
    let mut last_gen = 0u64;
    loop {
        let (func, n_tasks) = {
            let mut st = shared.state.lock().unwrap();
            while !st.shutdown && st.generation == last_gen {
                st = shared.cv_work.wait(st).unwrap();
            }
            if st.shutdown {
                return;
            }
            last_gen = st.generation;
            (st.func, st.n_tasks)
        };
        run_claimed(shared, func, n_tasks);
        let mut st = shared.state.lock().unwrap();
        st.active -= 1;
        if st.active == 0 {
            shared.cv_done.notify_one();
        }
    }
}

/// Pool worker count. Unlike per-call threading (which paid a spawn per
/// projection, so fewer cores won), the pool amortizes spawn — the microbench
/// peaks at 4-6 cores and work-stealing self-balances heterogeneous big.LITTLE
/// cores (fast P-cores claim more tasks), so default to all logical cores.
/// `RLLM_THREADS` overrides (RLLM_THREADS=1 forces serial inline).
fn decode_pool_threads() -> usize {
    match std::env::var(RLLM_THREADS_ENV).ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(v) if v > 0 => v,
        _ => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
    }
}

/// Process-wide decode pool, sized to the decode runtime thread count.
pub(crate) fn decode_pool() -> &'static DecodePool {
    static POOL: OnceLock<DecodePool> = OnceLock::new();
    POOL.get_or_init(|| DecodePool::new(decode_pool_threads().max(1)))
}

#[cfg(test)]
mod decode_pool_tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn parallel_for_runs_every_task_once() {
        let pool = DecodePool::new(4);
        let n = 10_000usize;
        let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        pool.parallel_for(n, |i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        for h in &hits {
            assert_eq!(h.load(Ordering::Relaxed), 1, "each task must run exactly once");
        }
    }

    #[test]
    fn parallel_for_disjoint_writes_match_serial() {
        let pool = DecodePool::new(4);
        let rows = 1000usize;
        let cols = 37usize;
        let input: Vec<f32> = (0..rows * cols).map(|i| (i % 13) as f32).collect();
        let mut out = vec![0f32; rows];
        // Disjoint per-row writes via a raw pointer (rows never overlap).
        let out_ptr = out.as_mut_ptr() as usize;
        pool.parallel_for(rows, |r| {
            let mut acc = 0f32;
            for c in 0..cols {
                acc += input[r * cols + c];
            }
            unsafe { *(out_ptr as *mut f32).add(r) = acc };
        });
        for r in 0..rows {
            let expected: f32 = (0..cols).map(|c| input[r * cols + c]).sum();
            assert_eq!(out[r], expected);
        }
    }

    #[test]
    fn parallel_for_repeated_dispatch_is_stable() {
        let pool = DecodePool::new(3);
        let counter = AtomicU64::new(0);
        for _ in 0..200 {
            pool.parallel_for(64, |_| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(counter.load(Ordering::Relaxed), 200 * 64);
    }
}
