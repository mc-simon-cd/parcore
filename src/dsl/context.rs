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

//! # Kernel Execution Context
//!
//! Provides GPU-style index types injected into every kernel invocation.
//!
//! ## Analogy to CUDA
//!
//! | ParCore              | CUDA                         |
//! |----------------------|------------------------------|
//! | `ctx.global_id.x`   | `blockIdx.x*blockDim.x + threadIdx.x` |
//! | `ctx.local_id.x`    | `threadIdx.x`                |
//! | `ctx.group_id.x`    | `blockIdx.x`                 |
//! | `ctx.local_size.x`  | `blockDim.x`                 |
//! | `ctx.global_size.x` | `gridDim.x * blockDim.x`    |
//!
//! ## 3-D Grid Model
//!
//! ```text
//! Grid  ─────────────────────────────────────────
//!   Block(0,0)   Block(1,0)   Block(2,0)   …
//!     Thread(0,0) Thread(1,0) …
//!     Thread(0,1) Thread(1,1) …
//!     …
//! ```
//!
//! All three axes are optional — 1-D and 2-D kernels just leave `y`/`z` = 0.

/// A three-dimensional index vector used throughout the DSL.
///
/// For 1-D kernels only `x` is meaningful; `y` and `z` are always 0.
/// For 2-D kernels `x` and `y` are used; `z` is always 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Dim3 {
    /// First dimension index (columns, or the only index for 1-D).
    pub x: usize,
    /// Second dimension index (rows for 2-D).
    pub y: usize,
    /// Third dimension index (depth for 3-D, rarely used on CPU).
    pub z: usize,
}

impl Dim3 {
    /// Create a 1-D `Dim3` (y = z = 0).
    #[inline]
    pub const fn new1(x: usize) -> Self { Self { x, y: 0, z: 0 } }

    /// Create a 2-D `Dim3` (z = 0).
    #[inline]
    pub const fn new2(x: usize, y: usize) -> Self { Self { x, y, z: 0 } }

    /// Create a 3-D `Dim3`.
    #[inline]
    pub const fn new3(x: usize, y: usize, z: usize) -> Self { Self { x, y, z } }

    /// Total number of elements this Dim3 represents (x * y * z, treating 0 as 1).
    #[inline]
    pub fn volume(&self) -> usize {
        self.x.max(1) * self.y.max(1) * self.z.max(1)
    }

    /// Flatten a 3-D coordinate to a linear index given grid dimensions.
    ///
    /// `linear = z * (size_y * size_x) + y * size_x + x`
    #[inline]
    pub fn to_linear(&self, size: &Dim3) -> usize {
        let sx = size.x.max(1);
        let sy = size.y.max(1);
        self.z * (sy * sx) + self.y * sx + self.x
    }

    /// Reconstruct a 3-D `Dim3` from a linear index and grid size.
    ///
    /// Inverse of [`to_linear`].
    #[inline]
    pub fn from_linear(linear: usize, size: &Dim3) -> Self {
        let sx = size.x.max(1);
        let sy = size.y.max(1);
        let x = linear % sx;
        let y = (linear / sx) % sy;
        let z = linear / (sx * sy);
        Self { x, y, z }
    }
}

impl std::fmt::Display for Dim3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

/// Per-work-item execution context injected into every kernel invocation.
///
/// Mirrors the built-in variables available in CUDA/OpenCL/GLSL:
///
/// ```text
/// Grid of groups (blocks):
///   ┌──────────┬──────────┬──────────┐
///   │ group(0) │ group(1) │ group(2) │
///   │ ┌──┬──┐  │ ┌──┬──┐  │          │
///   │ │t0│t1│  │ │t0│t1│  │  …       │
///   │ └──┴──┘  │ └──┴──┘  │          │
///   └──────────┴──────────┴──────────┘
/// ```
///
/// Fields are separated so the kernel can choose its granularity:
/// - Use only `global_id` for simple element-parallel kernels.
/// - Use `local_id` + `group_id` for scratchpad/shared-memory patterns.
#[derive(Debug, Clone, Copy)]
pub struct KernelCtx {
    /// Unique flat work-item index across the entire grid.
    /// Equivalent to `blockIdx * blockDim + threadIdx` in CUDA.
    pub global_id: Dim3,

    /// Index of this work-item within its group (workgroup / block).
    /// Equivalent to `threadIdx` in CUDA.
    pub local_id: Dim3,

    /// Index of the group (workgroup / block) this work-item belongs to.
    /// Equivalent to `blockIdx` in CUDA.
    pub group_id: Dim3,

    /// Total global work items in each dimension (grid_dim × block_dim).
    pub global_size: Dim3,

    /// Work-items per group (block / workgroup size).
    pub local_size: Dim3,
}

impl KernelCtx {
    /// Construct a `KernelCtx` from a flat linear work-item index.
    ///
    /// Used by the dispatch engine to cheaply compute all derived indices.
    ///
    /// # Parameters
    /// - `flat`        — the linear work-item index `[0, global_volume)`
    /// - `global_size` — total grid dimensions
    /// - `local_size`  — block/workgroup dimensions
    pub fn from_flat(flat: usize, global_size: Dim3, local_size: Dim3) -> Self {
        let global_id = Dim3::from_linear(flat, &global_size);
        let lsx = local_size.x.max(1);
        let lsy = local_size.y.max(1);
        let local_id = Dim3 {
            x: global_id.x % lsx,
            y: global_id.y % lsy,
            z: global_id.z % local_size.z.max(1),
        };
        let group_id = Dim3 {
            x: global_id.x / lsx,
            y: global_id.y / lsy,
            z: global_id.z / local_size.z.max(1),
        };
        Self { global_id, local_id, group_id, global_size, local_size }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim3_volume() {
        assert_eq!(Dim3::new1(10).volume(), 10);
        assert_eq!(Dim3::new2(4, 8).volume(), 32);
        assert_eq!(Dim3::new3(2, 3, 4).volume(), 24);
        assert_eq!(Dim3::default().volume(), 1); // (0,0,0) → 1×1×1
    }

    #[test]
    fn dim3_round_trip_1d() {
        let size = Dim3::new1(1024);
        for i in 0..1024 {
            let d = Dim3::from_linear(i, &size);
            assert_eq!(d.to_linear(&size), i);
        }
    }

    #[test]
    fn dim3_round_trip_2d() {
        let size = Dim3::new2(16, 8);
        for i in 0..128 {
            let d = Dim3::from_linear(i, &size);
            assert_eq!(d.to_linear(&size), i);
        }
    }

    #[test]
    fn kernel_ctx_from_flat_1d() {
        let gs = Dim3::new1(256);
        let ls = Dim3::new1(32);
        let ctx = KernelCtx::from_flat(33, gs, ls);
        assert_eq!(ctx.global_id.x, 33);
        assert_eq!(ctx.local_id.x,  1);   // 33 % 32
        assert_eq!(ctx.group_id.x,  1);   // 33 / 32
    }

    #[test]
    fn kernel_ctx_from_flat_2d() {
        let gs = Dim3::new2(4, 4);
        let ls = Dim3::new2(2, 2);
        // flat=5 → x=5%4=1, y=5/4=1
        let ctx = KernelCtx::from_flat(5, gs, ls);
        assert_eq!(ctx.global_id, Dim3::new2(1, 1));
        assert_eq!(ctx.local_id,  Dim3::new2(1, 1)); // x:1%2=1, y:1%2=1
        assert_eq!(ctx.group_id,  Dim3::new2(0, 0)); // x:1/2=0, y:1/2=0
    }
}
