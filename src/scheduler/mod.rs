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

//! # Scheduler Module
//!
//! Implements a work-stealing thread pool.
//!
//! ## Design
//!
//! Each worker thread owns a local `crossbeam_deque::Worker` deque (LIFO for
//! cache locality). New tasks are either pushed to a worker's local deque or
//! injected into the global `Injector` queue.
//!
//! When a worker runs out of local work it:
//! 1. Drains the global `Injector`.
//! 2. Randomly attempts to steal from a peer's deque.
//! 3. Parks (busy-yields briefly then sleeps) to avoid spinning.

pub mod pool;
pub mod worker;

pub use pool::ThreadPool;
pub use worker::TaskFn;
