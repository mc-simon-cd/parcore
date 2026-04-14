// Copyright 2026 mcsimon
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Runtime
//!
//! The `Runtime` struct is the central entry point for ParCore.
//! It owns the `ThreadPool` and provides:
//!
//! - `parallel_for` — parallel iteration over a range
//! - `spawn`        — fire-and-forget task submission
//! - `barrier`      — wait for all pending tasks
//! - `shutdown`     — graceful thread pool teardown

use crate::scheduler::ThreadPool;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The ParCore runtime. Create one per application with [`Runtime::new`].
///
/// # Example
/// ```rust,no_run
/// use parcore::Runtime;
/// let rt = Runtime::new(4);
/// rt.parallel_for(0..256, |i| println!("{i}"));
/// ```
pub struct Runtime {
    /// The underlying work-stealing thread pool.
    pub pool: ThreadPool,
    /// Logical number of compute units (mirrors thread count).
    pub num_units: usize,
    /// Registered compute units (CPU, GPU, NPU).
    pub units: Vec<crate::kernel::ComputeUnit>,
}

impl Runtime {
    /// Create a runtime with `num_threads` worker threads.
    /// Pass `0` to auto-detect logical CPU count.
    pub fn new(num_threads: usize) -> Self {
        let n = if num_threads == 0 {
            num_cpus::get()
        } else {
            num_threads
        };
        Self {
            pool: ThreadPool::new(n),
            num_units: n,
            units: vec![crate::kernel::ComputeUnit::new(0, crate::kernel::UnitKind::Cpu)],
        }
    }

    /// Execute `f(i)` for every index in `range` in parallel.
    ///
    /// The range is split into `num_units` chunks; each chunk is submitted as
    /// a separate task so the work-stealer can redistribute imbalanced loads.
    pub fn parallel_for<F>(&self, range: std::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        let len = range.end.saturating_sub(range.start);
        if len == 0 {
            return;
        }

        // Wrap the closure in an Arc so it can be shared across tasks.
        let f = Arc::new(f);

        // Each task processes one index (fine-grained) up to a threshold,
        // then switches to chunk-based dispatch for large ranges.
        let chunk_size = ((len + self.num_units - 1) / self.num_units).max(1);
        let start = range.start;

        let total_chunks = (len + chunk_size - 1) / chunk_size;
        let remaining = Arc::new(AtomicUsize::new(total_chunks));

        for chunk_idx in 0..total_chunks {
            let f = Arc::clone(&f);
            let remaining = Arc::clone(&remaining);
            let chunk_start = start + chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(range.end);

            self.pool.submit(move || {
                for i in chunk_start..chunk_end {
                    f(i);
                }
                remaining.fetch_sub(1, Ordering::Release);
            });
        }

        // Wait for all chunks to complete before returning.
        while remaining.load(Ordering::Acquire) > 0 {
            std::thread::yield_now();
        }
    }

    /// Submit a one-shot task to run asynchronously on the thread pool.
    pub fn spawn<F: FnOnce() + Send + 'static>(&self, f: F) {
        self.pool.submit(f);
    }

    /// Block until all submitted tasks have finished.
    pub fn barrier(&self) {
        self.pool.barrier();
    }

    /// Gracefully shut down the runtime.
    /// After calling this the runtime must not be used again.
    pub fn shutdown(&mut self) {
        self.pool.shutdown();
    }

    /// Access the compute units registered with this runtime.
    pub fn get_compute_units(&self) -> &[crate::kernel::ComputeUnit] {
        &self.units
    }

    /// Register a new compute unit with this runtime.
    pub fn add_compute_unit(&mut self, unit: crate::kernel::ComputeUnit) {
        self.units.push(unit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn parallel_for_counts_correctly() {
        let rt = Runtime::new(4);
        let sum = Arc::new(AtomicUsize::new(0));
        let sum2 = Arc::clone(&sum);
        rt.parallel_for(0..1000, move |i| {
            sum2.fetch_add(i, Ordering::Relaxed);
        });
        // 0+1+…+999 = 499500
        assert_eq!(sum.load(Ordering::SeqCst), 499500);
    }

    #[test]
    fn spawn_and_barrier() {
        let rt = Runtime::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..200 {
            let c = Arc::clone(&counter);
            rt.spawn(move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }
        rt.barrier();
        assert_eq!(counter.load(Ordering::SeqCst), 200);
    }
}
