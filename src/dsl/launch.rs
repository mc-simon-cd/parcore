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

//! # Kernel Launch System
//!
//! `KernelLaunchConfig` is the builder-pattern API for launching DSL kernels.
//! It mirrors CUDA's `<<<grid, block>>>` launch syntax in a type-safe,
//! ergonomic Rust style.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use parcore::dsl::{KernelLaunchConfig, parcore_kernel};
//! use parcore::Runtime;
//! use std::sync::Arc;
//!
//! let rt = Runtime::new(0); // auto-detect CPUs
//!
//! let kernel = parcore_kernel!(name: "noop", |_ctx| {});
//!
//! KernelLaunchConfig::new()
//!     .grid1d(1024)
//!     .block1d(64)
//!     .dispatch(&rt, Arc::new(kernel));
//! ```
//!
//! ## Design
//!
//! `KernelLaunchConfig` stores grid + block dimensions and an `OptHints`
//! struct. When `dispatch` is called it:
//!
//! 1. Builds a `KernelIR` (materialises the structured description).
//! 2. Logs the IR summary at debug level.
//! 3. Dispatches to the appropriate backend:
//!    - `Backend::Cpu`  → existing `ComputeUnit::dispatch`
//!    - Others          → stub (future GPU/NPU paths)

use super::{
    context::Dim3,
    ir::{Backend, KernelIR, OptHints},
};
use crate::{
    kernel::{ComputeUnit, Kernel, UnitKind},
    Runtime,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// LaunchResult
// ---------------------------------------------------------------------------

/// Outcome of a kernel launch.
#[derive(Debug, Clone)]
pub struct LaunchResult {
    /// Human-readable name of the launched kernel.
    pub kernel_name: String,
    /// Number of work-items dispatched.
    pub work_items: usize,
    /// Number of parallel chunks submitted to the thread pool.
    pub chunks: usize,
    /// The IR produced for this launch (useful for inspection/logging).
    pub ir: KernelIR,
}

// ---------------------------------------------------------------------------
// KernelLaunchConfig
// ---------------------------------------------------------------------------

/// Builder for kernel launch parameters — analogous to CUDA's `<<<G, B>>>`.
///
/// # Example
/// ```rust,no_run
/// use parcore::dsl::KernelLaunchConfig;
///
/// // 1-D: 1024 work-items in groups of 64
/// let cfg = KernelLaunchConfig::new().grid1d(1024).block1d(64);
///
/// // 2-D: 32×32 grid with 8×8 blocks
/// let cfg = KernelLaunchConfig::new().grid2d(32, 32).block2d(8, 8);
/// ```
#[derive(Debug, Clone)]
pub struct KernelLaunchConfig {
    /// Total grid dimensions (work-items per axis).
    grid:  Dim3,
    /// Block / workgroup size.
    block: Dim3,
    /// Optimisation hints forwarded to the IR.
    hints: OptHints,
    /// Target backend.
    backend: Backend,
}

impl Default for KernelLaunchConfig {
    fn default() -> Self { Self::new() }
}

impl KernelLaunchConfig {
    /// Create a config with no grid/block set (must call grid*/block* before dispatch).
    pub fn new() -> Self {
        Self {
            grid:    Dim3::new1(1),
            block:   Dim3::new1(1),
            hints:   OptHints::default(),
            backend: Backend::Auto,
        }
    }

    // ── Grid setters ──────────────────────────────────────────────────────

    /// Set a 1-D grid of `x` total work-items.
    pub fn grid1d(mut self, x: usize) -> Self { self.grid = Dim3::new1(x); self }

    /// Set a 2-D grid of `x × y` total work-items.
    pub fn grid2d(mut self, x: usize, y: usize) -> Self { self.grid = Dim3::new2(x, y); self }

    /// Set a 3-D grid.
    pub fn grid3d(mut self, x: usize, y: usize, z: usize) -> Self {
        self.grid = Dim3::new3(x, y, z); self
    }

    // ── Block/workgroup setters ────────────────────────────────────────────

    /// Set a 1-D block of `x` work-items.
    pub fn block1d(mut self, x: usize) -> Self { self.block = Dim3::new1(x); self }

    /// Set a 2-D block.
    pub fn block2d(mut self, x: usize, y: usize) -> Self { self.block = Dim3::new2(x, y); self }

    /// Set a 3-D block.
    pub fn block3d(mut self, x: usize, y: usize, z: usize) -> Self {
        self.block = Dim3::new3(x, y, z); self
    }

    // ── Optimisation hint setters ─────────────────────────────────────────

    /// Override the default adaptive tile size.
    pub fn tile(mut self, t: usize) -> Self { self.hints.tile_size = t; self }

    /// Disable auto-vectorisation hint.
    pub fn no_vectorize(mut self) -> Self { self.hints.vectorize = false; self }

    /// Enable B-transpose optimisation for matmul-like kernels.
    pub fn transpose_input(mut self) -> Self { self.hints.transpose_input = true; self }

    /// Request auto-tuning of chunk/tile parameters at runtime.
    pub fn auto_tune(mut self) -> Self { self.hints.auto_tune = true; self }

    /// Assign this kernel to a fusion group (kernels in the same group
    /// may be merged by a future fusion pass).
    pub fn fusion_group(mut self, id: usize) -> Self { self.hints.fusion_group = id; self }

    // ── Backend ───────────────────────────────────────────────────────────

    /// Force a specific backend (default: `Backend::Cpu`).
    pub fn backend(mut self, b: Backend) -> Self { self.backend = b; self }

    // ── IR construction ───────────────────────────────────────────────────

    /// Materialise the `KernelIR` for this configuration.
    ///
    /// Useful for inspection, logging, or passing to a future compiler
    /// pass before actual dispatch.
    pub fn build_ir(&self, name: impl Into<String>) -> KernelIR {
        KernelIR {
            name: name.into(),
            params:    vec![],    // populated by higher-level DSL wrappers
            grid:      self.grid,
            block:     self.block,
            opt_hints: self.hints.clone(),
            backend:   self.backend,
        }
    }

    // ── Dispatch ──────────────────────────────────────────────────────────

    /// Compile the IR and dispatch `kernel` across the runtime.
    ///
    /// Returns a [`LaunchResult`] containing the IR and dispatch statistics.
    ///
    /// ## Backend routing
    ///
    /// Currently only `Backend::Cpu` and `Backend::Auto` are active.
    /// `WgpuGpu` and `Npu` stubs are present for future extension.
    pub fn dispatch<K: Kernel + 'static>(
        &self,
        rt: &Runtime,
        kernel: Arc<K>,
    ) -> LaunchResult {
        // Extract the name FIRST as an owned String so we don't borrow `kernel`
        // for 'static — the Kernel::name() return value is tied to &self, and
        // we need to move `kernel` into the backend dispatch below.
        let name: String = kernel.name().to_owned();
        let ir = self.build_ir(name.clone());

        // Log the IR (debug builds only — zero cost in release).
        if cfg!(debug_assertions) {
            eprintln!("[ParCore DSL] {}", ir.summary());
        }

        let total = ir.total_work_items();
        let block_vol = self.block.volume().max(1);

        let chunks = match ir.backend {
            Backend::Cpu => {
                let cu = ComputeUnit::new(0, UnitKind::Cpu);
                cu.dispatch(rt, kernel, total, block_vol);
                (total + block_vol - 1) / block_vol
            }
            Backend::Auto | Backend::WgpuGpu => {
                let wgsl = kernel.wgsl_code();
                if !wgsl.is_empty() {
                    if let Some(cu) = rt.get_compute_units().iter().find(|u| u.kind == UnitKind::WgpuGpu) {
                        cu.dispatch(rt, kernel, total, block_vol);
                        1 // GPU is considered 1 chunk for IR purposes
                    } else {
                        // Fallback to CPU if no GPU unit registered
                        let cu = ComputeUnit::new(0, UnitKind::Cpu);
                        cu.dispatch(rt, kernel, total, block_vol);
                        (total + block_vol - 1) / block_vol
                    }
                } else {
                    let cu = ComputeUnit::new(0, UnitKind::Cpu);
                    cu.dispatch(rt, kernel, total, block_vol);
                    (total + block_vol - 1) / block_vol
                }
            }
            Backend::Npu => {
                eprintln!("[ParCore DSL] Npu backend not yet implemented — falling back to CPU");
                let cu = ComputeUnit::new(0, UnitKind::SimNpu);
                cu.dispatch(rt, kernel, total, block_vol);
                (total + block_vol - 1) / block_vol
            }
        };

        LaunchResult { kernel_name: name, work_items: total, chunks, ir }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::closure_kernel::ClosureKernel;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn launch_1d_all_items_run() {
        let rt = Runtime::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let kernel = Arc::new(ClosureKernel::named("count", move |_ctx| {
            c.fetch_add(1, Ordering::Relaxed);
        }));
        let result = KernelLaunchConfig::new()
            .grid1d(512)
            .block1d(64)
            .dispatch(&rt, kernel);
        assert_eq!(counter.load(Ordering::SeqCst), 512);
        assert_eq!(result.work_items, 512);
    }

    #[test]
    fn launch_2d_items_run() {
        let rt = Runtime::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let kernel = Arc::new(ClosureKernel::named("count2d", move |_ctx| {
            c.fetch_add(1, Ordering::Relaxed);
        }));
        let result = KernelLaunchConfig::new()
            .grid2d(16, 16)
            .block2d(4, 4)
            .dispatch(&rt, kernel);
        assert_eq!(counter.load(Ordering::SeqCst), 256);
        assert_eq!(result.work_items, 256);
    }

    #[test]
    fn launch_ir_has_correct_grid() {
        let cfg = KernelLaunchConfig::new().grid2d(32, 16).block2d(8, 4);
        let ir = cfg.build_ir("test");
        assert_eq!(ir.grid, Dim3::new2(32, 16));
        assert_eq!(ir.block, Dim3::new2(8, 4));
        assert_eq!(ir.total_work_items(), 512);
    }

    #[test]
    fn launch_global_id_correctness() {
        // Verify that global_id.x covers all indices 0..N exactly once.
        use std::sync::Mutex;
        let rt = Runtime::new(4);
        const N: usize = 128;
        let seen = Arc::new(Mutex::new(vec![0u32; N]));
        let seen2 = Arc::clone(&seen);
        let kernel = Arc::new(ClosureKernel::named("index_check", move |ctx| {
            let i = ctx.global_id.x;
            if i < N { seen2.lock().unwrap()[i] += 1; }
        }));
        KernelLaunchConfig::new().grid1d(N).block1d(32).dispatch(&rt, kernel);
        let s = seen.lock().unwrap();
        assert!(s.iter().all(|&c| c == 1), "each index must be visited exactly once");
    }
}
