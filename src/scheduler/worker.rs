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

//! Task type alias used by the scheduler.

use std::sync::Arc;

/// A boxed, send-able, single-call closure representing one unit of work.
pub type TaskFn = Box<dyn FnOnce() + Send + 'static>;

/// Worker represents one thread in the pool.
/// It holds a reference to the shared injector and the stealers of all peers.
pub struct Worker {
    /// This worker's FIFO local deque — producer side.
    pub local: crossbeam_deque::Worker<TaskFn>,
    /// Global injector shared by all workers.
    pub injector: Arc<crossbeam_deque::Injector<TaskFn>>,
    /// Stealers from every other worker in the pool.
    pub stealers: Arc<Vec<crossbeam_deque::Stealer<TaskFn>>>,
}

impl Worker {
    /// Run the worker event loop until `stop` becomes `true`.
    ///
    /// Strategy:
    /// 1. Pop from local deque (LIFO — best cache behaviour).
    /// 2. Steal from global injector.
    /// 3. Steal from a random peer.
    /// 4. Yield/sleep briefly to avoid busy-spinning.
    pub fn run(&self, stop: Arc<std::sync::atomic::AtomicBool>) {
        use crossbeam_deque::Steal;
        use std::sync::atomic::Ordering;

        let mut rng_state: u64 = (self as *const _ as u64).wrapping_add(0xdeadbeef);

        loop {
            // -- Local deque --------------------------------------------------
            if let Some(task) = self.local.pop() {
                task();
                continue;
            }

            // -- Global injector ----------------------------------------------
            loop {
                match self.injector.steal_batch_and_pop(&self.local) {
                    Steal::Success(task) => {
                        task();
                        break;
                    }
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }

            // -- Peer stealing (xorshift RNG — no std dep) --------------------
            if !self.stealers.is_empty() {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let idx = (rng_state as usize) % self.stealers.len();
                loop {
                    match self.stealers[idx].steal() {
                        Steal::Success(task) => {
                            task();
                            break;
                        }
                        Steal::Retry => continue,
                        Steal::Empty => break,
                    }
                }
            }

            // -- Check stop ---------------------------------------------------
            if stop.load(Ordering::Relaxed) && self.injector.is_empty() {
                break;
            }

            // -- Brief back-off to avoid spinning 100% ------------------------
            std::thread::yield_now();
        }
    }
}
