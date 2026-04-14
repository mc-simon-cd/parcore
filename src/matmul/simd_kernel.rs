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

//! # HPC SIMD Micro-kernels
//!
//! Highly optimized inner loops for matrix multiplication.
//! Designed for transposed B access patterns.

use wide::f64x4;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

// RISC-V specific: Zicbop extension provides prefetch instructions.
// Currently (as of 1.70+) these are often handled by the compiler or via asm!.
// We'll use a generic approach or keep it as a no-op for now.

/// Unroll factor for SIMD loops. 
/// 4x unrolling allows the CPU to overlap memory loads and FMA instructions.
const UNROLL: usize = 4;
const LANES: usize = 4; // f64x4

/// Core Dot-Product Kernel (SIMD + Unroll + Prefetch)
///
/// Computes sum(a[0..len] * b[0..len]) using SIMD.
///
/// # Requirements
/// - `len` must be >= 0.
/// - `a` and `b` must have at least `len` elements.
/// - B is assumed to be a row in a transposed matrix (contiguous access).
#[inline(always)]
pub unsafe fn dot_simd_unroll_prefetch(a: *const f64, b: *const f64, len: usize) -> f64 {
    let mut sum0 = f64x4::ZERO;
    let mut sum1 = f64x4::ZERO;
    let mut sum2 = f64x4::ZERO;
    let mut sum3 = f64x4::ZERO;

    let mut k = 0;
    let step = LANES * UNROLL;

    // Main loop: 4 lanes * 4 unroll = 16 elements per iteration
    while k + step <= len {
        // Prefetch next cache lines (64-128 bytes ahead)
        #[cfg(target_arch = "x86_64")]
        {
            _mm_prefetch(a.add(k + 64) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.add(k + 64) as *const i8, _MM_HINT_T0);
        }
        
        #[cfg(target_arch = "riscv64")]
        {
            // Future: Use core::arch::riscv64::prefetch_r when stabilized.
            // For now, the hardware prefetcher or compiler will do the heavy lifting.
        }

        // Load 4 vectors (unrolled)
        let a0 = f64x4::new(unsafe { std::ptr::read_unaligned(a.add(k + LANES * 0) as *const [f64; 4]) });
        let b0 = f64x4::new(unsafe { std::ptr::read_unaligned(b.add(k + LANES * 0) as *const [f64; 4]) });
        let a1 = f64x4::new(unsafe { std::ptr::read_unaligned(a.add(k + LANES * 1) as *const [f64; 4]) });
        let b1 = f64x4::new(unsafe { std::ptr::read_unaligned(b.add(k + LANES * 1) as *const [f64; 4]) });
        let a2 = f64x4::new(unsafe { std::ptr::read_unaligned(a.add(k + LANES * 2) as *const [f64; 4]) });
        let b2 = f64x4::new(unsafe { std::ptr::read_unaligned(b.add(k + LANES * 2) as *const [f64; 4]) });
        let a3 = f64x4::new(unsafe { std::ptr::read_unaligned(a.add(k + LANES * 3) as *const [f64; 4]) });
        let b3 = f64x4::new(unsafe { std::ptr::read_unaligned(b.add(k + LANES * 3) as *const [f64; 4]) });

        // Fused Multiply-Add (simulated if not supported by hardware)
        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;

        k += step;
    }

    // Handle SIMD tail (one vector at a time)
    while k + LANES <= len {
        let a_vec = f64x4::new(unsafe { std::ptr::read_unaligned(a.add(k) as *const [f64; 4]) });
        let b_vec = f64x4::new(unsafe { std::ptr::read_unaligned(b.add(k) as *const [f64; 4]) });
        sum0 += a_vec * b_vec;
        k += LANES;
    }

    // Horizontal reduction
    let final_vec = sum0 + sum1 + sum2 + sum3;
    let mut total = final_vec.reduce_add();

    // Handle scalar tail
    while k < len {
        total += *a.add(k) * *b.add(k);
        k += 1;
    }

    total
}

/// Nightly implementation using `std::simd`.
/// Only enabled if requested and on nightly.
#[cfg(feature = "nightly")]
pub mod nightly {
    use std::simd::f64x4;
    use std::simd::num::SimdFloat;

    #[inline(always)]
    pub unsafe fn dot_std_simd(a: *const f64, b: *const f64, len: usize) -> f64 {
        let mut sum = f64x4::splat(0.0);
        let mut k = 0;
        
        while k + 4 <= len {
            let va = f64x4::from_slice(std::slice::from_raw_parts(a.add(k), 4));
            let vb = f64x4::from_slice(std::slice::from_raw_parts(b.add(k), 4));
            sum += va * vb;
            k += 4;
        }
        
        let mut total = sum.reduce_sum();
        while k < len {
            total += *a.add(k) * *b.add(k);
            k += 1;
        }
        total
    }
}
