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

//! # DSL Example Kernels
//!
//! Production-quality example kernels showing how to write GPU-style
//! parallel compute programs using the ParCore DSL.
//!
//! These examples serve as both documentation and integration tests.

use crate::{
    dsl::{ClosureKernel, KernelLaunchConfig},
    kernel::Kernel,
    memory::SharedBuffer,
    Runtime,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Example 1: Vector Addition
// ---------------------------------------------------------------------------

/// Run element-wise vector addition `out[i] = a[i] + b[i]` in parallel.
///
/// ## DSL syntax used
/// ```text
/// parcore_kernel! {
///     name: "vector_add",
///     |ctx| {
///         let i = ctx.global_id.x;
///         if i < n { out[i] = a[i] + b[i]; }
///     }
/// }
/// ```
///
/// ## Design notes
/// - Uses `SharedBuffer` for unified memory (no explicit copy).
/// - Guard `if i < n` handles non-power-of-two sizes safely.
/// - Block size 64 chosen to match a typical warp/wavefront size.
pub fn vector_add(
    rt: &Runtime,
    a: &SharedBuffer<f64>,
    b: &SharedBuffer<f64>,
    n: usize,
) -> SharedBuffer<f64> {
    let out = SharedBuffer::new(n, 0.0_f64);
    let a = a.clone_handle();
    let b = b.clone_handle();
    let out_handle = out.clone_handle();

    // parcore_kernel! expanded manually here so we can call it as a library fn.
    // Users write the macro form; the impl is identical.
    let kernel = Arc::new(ClosureKernel::named("vector_add", move |ctx| {
        let i = ctx.global_id.x;
        if i < n {
            // Reads are concurrent (RwLock read guard), writes are exclusive.
            // Since no two work-items share an index, locking here is safe
            // but suboptimal for large N — see vector_add_unsafe for a lock-free variant.
            let av = a.read()[i];
            let bv = b.read()[i];
            out_handle.write()[i] = av + bv;
        }
    }));

    KernelLaunchConfig::new()
        .grid1d(n)
        .block1d(64)
        .dispatch(rt, kernel);

    out
}

// ---------------------------------------------------------------------------
// Example 2: Vector Scaling (in-place)
// ---------------------------------------------------------------------------

/// Scale every element of `buf` by `factor` in parallel.
///
/// Demonstrates an **in-place** kernel that reads and writes the same buffer.
pub fn vector_scale(rt: &Runtime, buf: &SharedBuffer<f64>, factor: f64, n: usize) {
    let handle = buf.clone_handle();
    let kernel = Arc::new(ClosureKernel::named("vector_scale", move |ctx| {
        let i = ctx.global_id.x;
        if i < n {
            handle.write()[i] *= factor;
        }
    }));
    KernelLaunchConfig::new().grid1d(n).block1d(64).dispatch(rt, kernel);
}

// ---------------------------------------------------------------------------
// Example 3: Dot Product (parallel reduce → atomic accumulation)
// ---------------------------------------------------------------------------

/// Compute the dot product `Σ a[i] * b[i]` using atomic f64 accumulation.
///
/// ## Performance note
/// On real hardware this would use a two-phase reduce (partial sums per block,
/// then sum the partial sums). Here we use `std::sync::Mutex` for simplicity.
/// For production use, see `matmul_optimized` which avoids this pattern.
pub fn dot_product(
    rt: &Runtime,
    a: &SharedBuffer<f64>,
    b: &SharedBuffer<f64>,
    n: usize,
) -> f64 {
    use std::sync::Mutex;
    let acc = Arc::new(Mutex::new(0.0_f64));
    let a = a.clone_handle();
    let b = b.clone_handle();
    let acc2 = Arc::clone(&acc);

    let kernel = Arc::new(ClosureKernel::named("dot_product", move |ctx| {
        let i = ctx.global_id.x;
        if i < n {
            let v = a.read()[i] * b.read()[i];
            *acc2.lock().unwrap() += v;
        }
    }));

    KernelLaunchConfig::new().grid1d(n).block1d(64).dispatch(rt, kernel);
    let result = *acc.lock().unwrap();
    result
}

// ---------------------------------------------------------------------------
// Example 4: Matrix Multiplication (DSL version)
// ---------------------------------------------------------------------------

/// Matrix multiplication `C = A × B` via the ParCore DSL.
///
/// This is the **DSL expression** of the same algorithm implemented natively
/// in `matmul/optimized.rs`. It demonstrates how users write matmul with
/// GPU-style indexing through the macro syntax.
///
/// ## Indexing
/// ```text
/// global_id.x → column (j)
/// global_id.y → row    (i)
/// C[i][j] = Σ_k A[i][k] * B[k][j]
/// ```
pub fn matmul_dsl(
    rt: &Runtime,
    a: &SharedBuffer<f64>,
    b: &SharedBuffer<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> SharedBuffer<f64> {
    use crate::matmul::WgpuMatMulVec4Kernel;

    let c = SharedBuffer::new(m * n, 0.0_f64);
    let a_h = a.clone_handle();
    let b_h = b.clone_handle();
    let c_h = c.clone_handle();

    // DSL closure for CPU execution
    let cpu_f = move |ctx: &crate::dsl::context::KernelCtx| {
        let col = ctx.global_id.x; 
        let row = ctx.global_id.y; 
        if row < m && col < n {
            let mut sum = 0.0_f64;
            let ar = a_h.read();
            let br = b_h.read();
            for ki in 0..k {
                sum += ar[row * k + ki] * br[ki * n + col];
            }
            c_h.write()[row * n + col] = sum;
        }
    };

    // Optimized WGSL for GPU execution
    let gpu_opt = WgpuMatMulVec4Kernel {
        a: a.clone_to_f32(),
        b: b.clone_to_f32(),
        c: c.clone_to_f32(),
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let wgsl = gpu_opt.wgsl_code();

    // Prepare GPU state (Lazy sync context and f32 buffers)
    let units = rt.get_compute_units();
    let gpu_state = units.iter().find(|u| u.kind == crate::kernel::UnitKind::WgpuGpu)
                        .and_then(|u| u.wgpu_state.as_ref());

    if let Some(state) = gpu_state {
        gpu_opt.a.sync_to_gpu(&state.device, &state.queue);
        gpu_opt.b.sync_to_gpu(&state.device, &state.queue);
        gpu_opt.c.sync_to_gpu(&state.device, &state.queue);
    }

    let kernel = Arc::new(crate::dsl::ClosureKernel2D::named("matmul_dsl", n, m, cpu_f)
        .with_wgsl(wgsl)
        .with_gpu_buffers(vec![
            gpu_opt.a.get_gpu_buffer().expect("A"),
            gpu_opt.b.get_gpu_buffer().expect("B"),
            gpu_opt.c.get_gpu_buffer().expect("C"),
        ]));

    // Launch a 2-D grid: one work-item per output element.
    crate::dsl::KernelLaunchConfig::new()
        .grid2d(n, m)
        .block2d(16, 16)
        .dispatch(rt, kernel);

    // Capture the WgpuState for lazy sync
    let units = rt.get_compute_units();
    if let Some(gpu_unit) = units.iter().find(|u| u.kind == crate::kernel::UnitKind::WgpuGpu) {
        if let Some(ref state) = gpu_unit.wgpu_state {
            println!("[matmul_dsl] Routing to HIGH-PERFORMANCE GPU path...");
            c.set_wgpu_context(state);
        }
    } else {
        println!("[matmul_dsl] Falling back to CPU path (No GPU unit found in Runtime)");
    }

    // Mark for device-dirty NOT on the host buffer (which has no Wgpu state),
    // but we can manually trigger the fetch from our f32 staging buffer back to f64.
    if let Some(state) = gpu_state {
        gpu_opt.c.mark_device_dirty(); 
        gpu_opt.c.fetch_from_gpu_sync(&state.device, &state.queue);
        let f32_res = gpu_opt.c.read();
        let mut out_w = c.write();
        for i in 0..m * n {
            out_w[i] = f32_res[i] as f64;
        }
    }

    c
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub mod tests {
    use super::*;

    fn rt(n: usize) -> Runtime { Runtime::new(n) }

    #[test]
    fn vector_add_correct() {
        let rt = rt(4);
        let a = SharedBuffer::from_vec(vec![1.0_f64; 64]);
        let b = SharedBuffer::from_vec(vec![2.0_f64; 64]);
        let out = vector_add(&rt, &a, &b, 64);
        let r = out.read();
        assert!(r.iter().all(|&v| (v - 3.0).abs() < 1e-12));
    }

    #[test]
    fn vector_scale_correct() {
        let rt = rt(4);
        let buf = SharedBuffer::from_vec(vec![1.0_f64; 32]);
        vector_scale(&rt, &buf, 5.0, 32);
        let r = buf.read();
        assert!(r.iter().all(|&v| (v - 5.0).abs() < 1e-12));
    }

    #[test]
    fn dot_product_correct() {
        let rt = rt(4);
        let n = 100;
        let a = SharedBuffer::from_vec(vec![1.0_f64; n]);
        let b = SharedBuffer::from_vec(vec![2.0_f64; n]);
        let result = dot_product(&rt, &a, &b, n);
        assert!((result - 200.0).abs() < 1e-9, "expected 200.0, got {result}");
    }

    #[test]
    fn matmul_dsl_matches_naive() {
        use crate::matmul::{matmul_naive, Matrix};
        let rt = rt(4);
        let m = 8; let n = 8; let k = 8;
        let mat_a = Matrix::random(m, k, 0xABC);
        let mat_b = Matrix::random(k, n, 0xDEF);
        let ref_c = matmul_naive(&mat_a, &mat_b);

        let a_buf = SharedBuffer::from_vec(mat_a.data.clone());
        let b_buf = SharedBuffer::from_vec(mat_b.data.clone());
        let c_buf = matmul_dsl(&rt, &a_buf, &b_buf, m, n, k);
        let c_data = c_buf.read();

        for i in 0..m * n {
            assert!(
                (c_data[i] - ref_c.data[i]).abs() < 1e-9,
                "DSL matmul diverges at index {i}"
            );
        }
    }
}
