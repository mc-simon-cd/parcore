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

//! Thread pool — the heart of the work-stealing scheduler.
//!
//! `ThreadPool` owns N OS threads. Tasks submitted via `submit()` are pushed
//! into the global `Injector` queue. The workers drain from their local deques
//! and steal from peers when idle.

use super::worker::{TaskFn, Worker};
use crossbeam_deque::{Injector, Stealer};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};

/// A multi-threaded work-stealing thread pool.
pub struct ThreadPool {
    /// Global injector — any thread can push tasks here.
    injector: Arc<Injector<TaskFn>>,
    /// OS thread handles (joined on drop).
    handles: Vec<JoinHandle<()>>,
    /// Shared stop flag — set to `true` during shutdown.
    stop: Arc<AtomicBool>,
    /// Number of live worker threads.
    pub num_threads: usize,
    /// Count of pending (submitted but not yet completed) tasks.
    pending: Arc<AtomicUsize>,
}

impl ThreadPool {
    /// Create a pool with `num_threads` worker threads.
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0, "ThreadPool requires at least one thread");

        let injector: Arc<Injector<TaskFn>> = Arc::new(Injector::new());
        let stop = Arc::new(AtomicBool::new(false));
        let pending = Arc::new(AtomicUsize::new(0));

        // First pass: create all local deques so we can harvest stealers.
        let workers_and_stealers: Vec<(crossbeam_deque::Worker<TaskFn>, Stealer<TaskFn>)> =
            (0..num_threads)
                .map(|_| {
                    let w = crossbeam_deque::Worker::new_fifo();
                    let s = w.stealer();
                    (w, s)
                })
                .collect();

        let stealers: Arc<Vec<Stealer<TaskFn>>> = Arc::new(
            workers_and_stealers.iter().map(|(_, s)| s.clone()).collect(),
        );

        let mut handles = Vec::with_capacity(num_threads);

        for (i, (local, _)) in workers_and_stealers.into_iter().enumerate() {
            // Each worker gets all stealers except its own.
            let peer_stealers: Vec<Stealer<TaskFn>> = stealers
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, s)| s.clone())
                .collect();

            let worker = Worker {
                local,
                injector: Arc::clone(&injector),
                stealers: Arc::new(peer_stealers),
            };

            let stop_clone = Arc::clone(&stop);
            let pending_clone = Arc::clone(&pending);

            let handle = thread::Builder::new()
                .name(format!("parcore-worker-{i}"))
                .spawn(move || {
                    worker.run_with_completion(stop_clone, pending_clone);
                })
                .expect("failed to spawn worker thread");

            handles.push(handle);
        }

        Self {
            injector,
            handles,
            stop,
            num_threads,
            pending,
        }
    }

    /// Submit a task to the global injector queue.
    pub fn submit<F: FnOnce() + Send + 'static>(&self, f: F) {
        self.pending.fetch_add(1, Ordering::Release);
        self.injector.push(Box::new(f));
    }

    /// Block the calling thread until all submitted tasks have finished.
    ///
    /// Uses a **hybrid back-off** strategy:
    /// - Phase 1 (≤64 iters): `spin_loop` hint — keeps the core warm for
    ///   micro-tasks that finish in nanoseconds (avoids OS scheduler round-trip).
    /// - Phase 2 (>64 iters): `yield_now` — relinquishes the OS timeslice so
    ///   other threads can be scheduled, reducing CPU waste for long barriers.
    pub fn barrier(&self) {
        let mut spins: u32 = 0;
        while self.pending.load(Ordering::Acquire) > 0 {
            if spins < 64 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Gracefully shut down all worker threads (called by `Runtime::shutdown`).
    pub fn shutdown(&mut self) {
        self.barrier();
        self.stop.store(true, Ordering::SeqCst);
        // Drain any lingering handles.
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        if !self.stop.load(Ordering::SeqCst) {
            self.shutdown();
        }
    }
}

// ---------------------------------------------------------------------------
// Worker extension: completion-aware run loop
// ---------------------------------------------------------------------------

use super::worker::Worker as W;

impl W {
    /// Like `run` but decrements the shared `pending` counter after each task.
    pub fn run_with_completion(
        &self,
        stop: Arc<AtomicBool>,
        pending: Arc<AtomicUsize>,
    ) {
        use crossbeam_deque::Steal;

        let mut rng: u64 = (self as *const _ as u64).wrapping_add(0xcafe_babe);
        // Idle back-off counter — resets on every task execution.
        let mut idle_iters: u32 = 0;

        loop {
            // ── 1. Local deque (fastest path) ────────────────────────────
            if let Some(task) = self.local.pop() {
                task();
                pending.fetch_sub(1, Ordering::Release);
                idle_iters = 0;
                continue;
            }

            // ── 2. Global injector ───────────────────────────────────────
            let mut got = false;
            loop {
                match self.injector.steal_batch_and_pop(&self.local) {
                    Steal::Success(task) => {
                        task();
                        pending.fetch_sub(1, Ordering::Release);
                        idle_iters = 0;
                        got = true;
                        break;
                    }
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
            if got {
                continue;
            }

            // ── 3. Peer stealing (xorshift64 for zero-dep RNG) ───────────
            if !self.stealers.is_empty() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let idx = (rng as usize) % self.stealers.len();
                loop {
                    match self.stealers[idx].steal() {
                        Steal::Success(task) => {
                            task();
                            pending.fetch_sub(1, Ordering::Release);
                            idle_iters = 0;
                            break;
                        }
                        Steal::Retry => continue,
                        Steal::Empty => break,
                    }
                }
            }

            // ── 4. Stop check ────────────────────────────────────────────
            if stop.load(Ordering::Relaxed) && self.injector.is_empty() {
                break;
            }

            // ── 5. Hybrid idle back-off ──────────────────────────────────
            // Phase 1 (≤64 idle iters): spin_loop keeps the core ready for
            //   new tasks that arrive in nanoseconds (avoids OS wakeup latency).
            // Phase 2 (>64 idle iters): yield_now releases the timeslice to
            //   avoid burning power when the pool is genuinely idle.
            if idle_iters < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
            idle_iters = idle_iters.saturating_add(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_all_tasks_run() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..1000 {
            let c = Arc::clone(&counter);
            pool.submit(move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }

        pool.barrier();
        assert_eq!(counter.load(Ordering::SeqCst), 1000);
    }

    #[test]
    fn test_single_thread_pool() {
        let pool = ThreadPool::new(1);
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..100 {
            let c = Arc::clone(&counter);
            pool.submit(move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }

        pool.barrier();
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }
}
