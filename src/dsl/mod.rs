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

//! # ParCore DSL — Domain-Specific Language for Parallel Kernels
//!
//! This module provides a CUDA/OpenCL-inspired DSL for defining and launching
//! compute kernels in pure, safe Rust.
//!
//! ## Quick Reference
//!
//! ```rust,no_run
//! use parcore::dsl::{KernelLaunchConfig, parcore_kernel};
//! use parcore::Runtime;
//! use std::sync::Arc;
//!
//! let rt = Runtime::new(0); // 0 = auto-detect CPUs
//!
//! // 1. Define a kernel with parcore_kernel!
//! let data = vec![1.0_f64; 1024];
//! let out  = parcore::memory::SharedBuffer::new(1024, 0.0_f64);
//!
//! let out_handle = out.clone_handle();
//! let kernel = parcore_kernel! {
//!     name: "scale",
//!     |ctx: &KernelCtx| {
//!         let i = ctx.global_id.x;
//!         out_handle.write()[i] = data[i] * 2.0;
//!     }
//! };
//!
//! // 2. Launch
//! KernelLaunchConfig::new()
//!     .grid1d(1024)
//!     .block1d(64)
//!     .dispatch(&rt, Arc::new(kernel));
//! ```
//!
//! ## DSL Compilation Pipeline
//!
//! ```text
//! parcore_kernel! { name: "...", |ctx| { ... } }
//!        │
//!        ▼  macro expansion (this file)
//! ClosureKernel<F>        ← zero-cost struct, implements Kernel
//!        │
//!        ▼  KernelLaunchConfig::dispatch()
//! KernelIR { name, grid, block, opt_hints, backend }
//!        │
//!        ▼  Backend::Cpu
//! ComputeUnit::dispatch() ← existing work-stealing pool
//!        │
//!        ▼  per work-item
//! KernelCtx { global_id, local_id, group_id, … }
//!        │
//!        ▼
//! user closure executes
//! ```

pub mod closure_kernel;
pub mod context;
pub mod ir;
pub mod kernels;
pub mod launch;

pub use closure_kernel::{ClosureKernel, ClosureKernel2D};
pub use context::{Dim3, KernelCtx};
pub use ir::{
    AccessMode, Backend, ExecutionGraph, GraphNode, KernelIR,
    MemorySpace, OptHints, ParamDecl,
};
pub use launch::{KernelLaunchConfig, LaunchResult};

// ---------------------------------------------------------------------------
// The parcore_kernel! macro
// ---------------------------------------------------------------------------

/// Define a compute kernel from a closure — the core DSL primitive.
///
/// # Syntax
///
/// ```rust,no_run
/// use parcore::dsl::{parcore_kernel, KernelCtx};
///
/// let kernel = parcore_kernel! {
///     name: "my_kernel",
///     |ctx: &KernelCtx| {
///         let i = ctx.global_id.x;
///         // kernel body — runs once per work-item
///     }
/// };
/// ```
///
/// # What the macro generates
///
/// ```rust,ignore
/// // Expands to:
/// ClosureKernel::named("my_kernel", move |ctx: &KernelCtx| {
///     let i = ctx.global_id.x;
///     // body
/// })
/// ```
///
/// ## Why a macro instead of a function?
///
/// The `move` keyword is inserted automatically, capturing all referenced
/// variables from the surrounding scope without explicit `move` boilerplate.
/// This mirrors the CUDA mental model where the kernel "captures" its data.
///
/// ## CUDA analogy
///
/// | parcore_kernel! | CUDA |
/// |-----------------|------|
/// | `name:`         | `__global__ void name(…)` |
/// | `ctx.global_id` | `blockIdx * blockDim + threadIdx` |
/// | `ctx.local_id`  | `threadIdx` |
/// | `ctx.group_id`  | `blockIdx` |
#[macro_export]
macro_rules! parcore_kernel {
    // Full syntax: name + typed ctx pattern
    (name: $name:literal, |$ctx:ident: &$ctx_ty:ty| $body:block) => {
        $crate::dsl::ClosureKernel::named(
            $name,
            move |$ctx: &$crate::dsl::KernelCtx| $body,
        )
    };
    // Short syntax: name + untyped ctx
    (name: $name:literal, |$ctx:ident| $body:block) => {
        $crate::dsl::ClosureKernel::named(
            $name,
            move |$ctx: &$crate::dsl::KernelCtx| $body,
        )
    };
}

/// Define a 2-D compute kernel indexed by `(x, y)`.
///
/// # Syntax
///
/// ```rust,no_run
/// use parcore::dsl::{parcore_kernel_2d, KernelCtx};
///
/// let (width, height) = (32_usize, 32_usize);
/// let kernel = parcore_kernel_2d! {
///     name: "matrix_scale",
///     width: width,
///     height: height,
///     |ctx: &KernelCtx| {
///         let col = ctx.global_id.x;
///         let row = ctx.global_id.y;
///         // operate on element (row, col)
///     }
/// };
/// ```
#[macro_export]
macro_rules! parcore_kernel_2d {
    (name: $name:literal, width: $w:expr, height: $h:expr, |$ctx:ident: &$ctx_ty:ty| $body:block) => {
        $crate::dsl::ClosureKernel2D::named(
            $name,
            $w,
            $h,
            move |$ctx: &$crate::dsl::KernelCtx| $body,
        )
    };
    (name: $name:literal, width: $w:expr, height: $h:expr, |$ctx:ident| $body:block) => {
        $crate::dsl::ClosureKernel2D::named(
            $name,
            $w,
            $h,
            move |$ctx: &$crate::dsl::KernelCtx| $body,
        )
    };
}

// Re-export macros publicly so they appear as `parcore::parcore_kernel!`.
pub use parcore_kernel;
pub use parcore_kernel_2d;
