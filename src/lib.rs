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

//! ParCore – Parallel Runtime System
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │         Application Layer           │
//! │       (user code / main.rs)         │
//! ├─────────────────────────────────────┤
//! │       ParCore Runtime API           │
//! │  parallel_for · spawn · barrier     │
//! ├──────────────┬──────────────────────┤
//! │  Scheduler   │   Kernel System      │
//! │ (work-steal) │  (compute units)     │
//! ├──────────────┴──────────────────────┤
//! │          Memory Layer               │
//! │   SharedBuffer<T>  (Arc+RwLock)     │
//! └─────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//! ```rust,no_run
//! use parcore::Runtime;
//!
//! let rt = Runtime::new(4);
//! parcore::parallel_for(&rt, 0..1024, |i| {
//!     println!("task {i}");
//! });
//! ```

pub mod kernel;
pub mod matmul;
pub mod memory;
pub mod runtime;
pub mod scheduler;
/// Domain-Specific Language for defining and launching compute kernels.
pub mod dsl;

pub use runtime::Runtime;

/// Execute `f(i)` for every `i` in `range` using the runtime's work-stealing
/// thread pool. Returns only after all iterations are complete.
///
/// # Example
/// ```rust,no_run
/// let rt = parcore::Runtime::new(num_cpus::get());
/// parcore::parallel_for(&rt, 0..1024, |i| {
///     // per-element computation
///     let _ = i * 2;
/// });
/// ```
pub fn parallel_for<F>(rt: &Runtime, range: std::ops::Range<usize>, f: F)
where
    F: Fn(usize) + Send + Sync + 'static,
{
    rt.parallel_for(range, f);
}

/// Spawn a fire-and-forget async task onto the runtime's thread pool.
/// The task starts immediately and runs concurrently with the caller.
/// Use [`Runtime::barrier`] to wait for all outstanding tasks.
pub fn spawn<F>(rt: &Runtime, task: F)
where
    F: FnOnce() + Send + 'static,
{
    rt.spawn(task);
}
