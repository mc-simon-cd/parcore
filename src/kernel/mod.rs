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

//! # Kernel System
//!
//! Provides a `Kernel` trait for defining compute kernels, a `KernelContext`
//! carrying per-work-item identifiers (analogous to CUDA thread/block IDs),
//! and a `ComputeUnit` abstraction that dispatches kernels via the scheduler.

pub mod compute_unit;
pub use compute_unit::{ComputeUnit, UnitKind};

/// Per-work-item execution context — mirrors CUDA's `threadIdx` / `blockIdx`.
///
/// When a kernel is dispatched over a grid of `(global_size)` items split into
/// blocks of `local_size`, each invocation receives one `KernelContext`.
#[derive(Debug, Clone, Copy)]
pub struct KernelContext {
    /// Global work-item ID (0 … global_size-1)
    pub global_id: usize,
    /// Local ID within its workgroup (0 … local_size-1)
    pub local_id: usize,
    /// Index of the workgroup / block
    pub group_id: usize,
    /// Total global work items
    pub global_size: usize,
    /// Workgroup size
    pub local_size: usize,
}

/// A compute kernel — any struct that implements this trait can be dispatched
/// onto a `ComputeUnit`.
///
/// # Example
/// ```rust
/// use parcore::kernel::{Kernel, KernelContext};
///
/// struct ScaleKernel { factor: f64 }
///
/// impl Kernel for ScaleKernel {
///     fn name(&self) -> &str { "scale" }
///     fn execute(&self, ctx: &KernelContext) {
///         println!("item {} × {}", ctx.global_id, self.factor);
///     }
/// }
/// ```
pub trait Kernel: Send + Sync {
    /// Human-readable name (used in tracing/debugging).
    fn name(&self) -> &str;

    /// Called once per work-item with the item's context for CPU execution.
    fn execute(&self, ctx: &KernelContext);

    /// WGSL source code for GPU execution.
    /// Returns an empty string by default (GPU not supported).
    fn wgsl_code(&self) -> String {
        String::new()
    }

    /// Workgroup size (local_size) for GPU dispatch.
    /// Default is (64, 1, 1).
    fn workgroup_size(&self) -> (u32, u32, u32) {
        (64, 1, 1)
    }

    /// Number of workgroups to dispatch (grid_size).
    /// If returning None, a 1D grid is automatically calculated from global_size.
    fn grid_size(&self) -> Option<(u32, u32, u32)> {
        None
    }

    /// Associated GPU buffers for binding.
    /// Returns an empty vector by default.
    fn gpu_buffers(&self) -> Vec<std::sync::Arc<wgpu::Buffer>> {
        Vec::new()
    }

    /// Mark output buffers as DeviceDirty after GPU execution.
    fn mark_outputs_dirty(&self) {}
}
