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

//! # ClosureKernel — Zero-Cost Closure-Based Kernel
//!
//! `ClosureKernel<F>` wraps any `Fn(&KernelCtx) + Send + Sync` closure and
//! implements the `Kernel` trait, plugging directly into the existing dispatch
//! infrastructure with **zero overhead** — the closure is called exactly once
//! per work-item, inlined by LLVM at release builds.
//!
//! ## Why closures instead of proc-macros?
//!
//! Procedural macros in Rust require a **separate crate** with `proc-macro = true`
//! in its `Cargo.toml`. For a single-crate project this would require converting
//! to a Cargo workspace. Closures give us:
//!
//! - Equal expressiveness (closures capture any Rust value)
//! - Full type safety (the Rust compiler checks everything)
//! - Zero-cost (closure call inlined by LLVM)
//! - No build-time proc-macro overhead
//!
//! The `parcore_kernel!` declarative macro (in `dsl/mod.rs`) provides the
//! CUDA-like syntax sugar on top of `ClosureKernel`.

use super::context::KernelCtx;
use crate::kernel::Kernel;
use crate::kernel::KernelContext; // existing trait's context (bridged below)
use std::sync::Arc;

// ---------------------------------------------------------------------------
// ClosureKernel
// ---------------------------------------------------------------------------

/// A kernel defined by a closure `Fn(&KernelCtx) + Send + Sync`.
///
/// Created by the [`parcore_kernel!`] macro — never instantiated directly
/// in user code.
///
/// # Type Parameter
/// `F` — any closure matching `Fn(&KernelCtx) + Send + Sync`.
///
/// # Zero-cost guarantee
/// At release optimisation level, the monomorphised `execute` method is a
/// single inlined call — no vtable, no heap allocation.
pub struct ClosureKernel<F>
where
    F: Fn(&KernelCtx) + Send + Sync,
{
    /// Human-readable name passed to `parcore_kernel!`.
    name: &'static str,
    /// The user-supplied closure representing the kernel body.
    f: F,
}

impl<F: Fn(&KernelCtx) + Send + Sync> ClosureKernel<F> {
    /// Create a named `ClosureKernel`.
    ///
    /// Prefer [`parcore_kernel!`] over calling this directly.
    #[inline]
    pub fn named(name: &'static str, f: F) -> Self {
        Self { name, f }
    }
}

/// Bridge between `ClosureKernel`'s rich `KernelCtx` and the existing
/// `KernelContext` used by `ComputeUnit::dispatch`.
///
/// `ClosureKernel` re-constructs the full `KernelCtx` from the flat
/// `KernelContext.global_id` supplied by the existing dispatch machinery.
impl<F: Fn(&KernelCtx) + Send + Sync> Kernel for ClosureKernel<F> {
    fn name(&self) -> &str {
        self.name
    }

    /// Called once per work-item by `ComputeUnit::dispatch`.
    ///
    /// Bridges `KernelContext` → `KernelCtx` and invokes the closure.
    fn execute(&self, kc: &KernelContext) {
        use super::context::{Dim3, KernelCtx};

        // Re-construct rich context from the flat global_id.
        // global_size and local_size are stored in the existing context.
        let global_size = Dim3::new1(kc.global_size);
        let local_size  = Dim3::new1(kc.local_size);
        let ctx = KernelCtx::from_flat(kc.global_id, global_size, local_size);
        (self.f)(&ctx);
    }
}

// ---------------------------------------------------------------------------
// 2-D variant
// ---------------------------------------------------------------------------

/// A 2-D closure kernel where work-items are indexed by `(x, y)`.
///
/// Used for 2-D grids such as matrix operations where `row × col` indexing
/// is more natural than a flat linear index.
pub struct ClosureKernel2D<F>
where
    F: Fn(&KernelCtx) + Send + Sync,
{
    name: &'static str,
    /// Number of columns in the 2-D grid (x dimension).
    pub width: usize,
    /// Number of rows in the 2-D grid (y dimension).
    pub height: usize,
    /// Optional WGSL source for GPU execution.
    pub wgsl: Option<String>,
    /// Associated GPU buffers.
    pub buffers: Vec<Arc<wgpu::Buffer>>,
    f: F,
}

impl<F: Fn(&KernelCtx) + Send + Sync> ClosureKernel2D<F> {
    /// Create a 2-D closure kernel with given grid dimensions.
    #[inline]
    pub fn named(name: &'static str, width: usize, height: usize, f: F) -> Self {
        Self { name, width, height, wgsl: None, buffers: Vec::new(), f }
    }

    /// Attach WGSL source to this kernel.
    pub fn with_wgsl(mut self, wgsl: impl Into<String>) -> Self {
        self.wgsl = Some(wgsl.into());
        self
    }

    /// Attach GPU buffers to this kernel.
    pub fn with_gpu_buffers(mut self, buffers: Vec<Arc<wgpu::Buffer>>) -> Self {
        self.buffers = buffers;
        self
    }
}

impl<F: Fn(&KernelCtx) + Send + Sync> Kernel for ClosureKernel2D<F> {
    fn name(&self) -> &str { self.name }

    fn wgsl_code(&self) -> String {
        self.wgsl.clone().unwrap_or_default()
    }

    fn gpu_buffers(&self) -> Vec<Arc<wgpu::Buffer>> {
        self.buffers.clone()
    }

    fn execute(&self, kc: &KernelContext) {
        use super::context::{Dim3, KernelCtx};
        // Decode flat global_id back to (x, y).
        let global_size = Dim3::new2(self.width, self.height);
        let local_size  = Dim3::new2(kc.local_size, 1);
        let ctx = KernelCtx::from_flat(kc.global_id, global_size, local_size);
        (self.f)(&ctx);
    }

    fn mark_outputs_dirty(&self) {
        // In a real DSL, we'd know which SharedBuffers were passed.
        // For our matmul_dsl bridge, we handle this in the bridge logic.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

    #[test]
    fn closure_kernel_has_correct_name() {
        let k = ClosureKernel::named("my_kernel", |_ctx| {});
        assert_eq!(k.name(), "my_kernel");
    }

    /// Verify that a ClosureKernel dispatched through ComputeUnit runs
    /// every work-item exactly once.
    #[test]
    fn closure_kernel_runs_all_items() {
        use crate::kernel::{ComputeUnit, UnitKind};
        use crate::Runtime;

        let rt = Runtime::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let kernel = Arc::new(ClosureKernel::named("counter", move |_ctx| {
            c.fetch_add(1, Ordering::Relaxed);
        }));
        let cu = ComputeUnit::new(0, UnitKind::SimGpu);
        cu.dispatch(&rt, kernel, 512, 64);
        assert_eq!(counter.load(Ordering::SeqCst), 512);
    }
}
