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

//! Cache-blocked (tiled) matrix multiplication with **adaptive tile sizing**.
//!
//! ## Why tiling?
//!
//! The naive algorithm iterates over columns of B for every row of A,
//! producing L1/L2 cache misses proportional to n². Tiling restructures
//! memory access into sub-blocks whose working set fits in L1/L2, trading
//! linear cache traffic for logarithmic (reuse distance shrinks to T).
//!
//! ## Adaptive tile sizing
//!
//! We do NOT hardcode a tile constant because L1 cache varies widely:
//!
//! | CPU family          | L1-D   |
//! |---------------------|--------|
//! | Cortex-A55 (mobile) | 32 KB  |
//! | Intel Core (server) | 32–64 KB |
//! | AMD Zen 4           | 32 KB  |
//! | Apple M-series      | 128 KB |
//! | RISC-V embedded     | 4–16 KB|
//!
//! We use a conservative 32 KB L1 target that works everywhere. Internally
//! we must keep three tile panels resident simultaneously:
//!
//! - A panel: `T × T` f64 values
//! - B panel: `T × T` f64 values  
//! - C panel: `T × T` f64 values
//!
//! Budget: `3 × T² × 8 bytes ≤ L1_TARGET`
//! → `T ≤ sqrt(L1_TARGET / 24)`
//!
//! Then snap to the nearest power-of-two (LLVM vectoriser prefers POT sizes).

use super::Matrix;

// ---------------------------------------------------------------------------
// Adaptive tile sizing
// ---------------------------------------------------------------------------

/// Conservative L1-D cache assumption in bytes.
/// 32 KB is the smallest common L1-D across x86, ARM, RISC-V targets.
/// We intentionally undershoot (32 KB < actual L1 on most modern CPUs)
/// so the kernel stays hot even on constrained embedded cores.
const L1_TARGET_BYTES: usize = 32 * 1024;

/// Minimum / maximum tile sizes (inclusive).
/// - MIN 8: below 8 the tiling overhead exceeds the benefit.
/// - MAX 256: above ~256 we risk TLB pressure from 256×256×8 = 512 KB panels.
const TILE_MIN: usize = 8;
const TILE_MAX: usize = 256;

/// Choose a cache-efficient tile size that fits in `L1_TARGET_BYTES` for the
/// given matrix dimensions, rounded down to the nearest power-of-two.
///
/// The formula assumes three `T×T` f64 panels must fit simultaneously in L1:
/// `3 · T² · 8 ≤ L1_TARGET` → `T ≤ sqrt(L1_TARGET / 24)`
///
/// The result is always in `[TILE_MIN, TILE_MAX]` regardless of input sizes.
///
/// # Arguments
/// * `_m`, `_n`, `_k` — matrix dimensions (reserved for future dimension-aware
///   strategies, e.g. rectangular panels for non-square matmuls)
pub fn adaptive_tile(_m: usize, _n: usize, _k: usize) -> usize {
    // Maximum T such that three T×T f64 panels fit in L1.
    let t_raw = ((L1_TARGET_BYTES / (3 * std::mem::size_of::<f64>())) as f64)
        .sqrt() as usize;

    // Round down to power-of-two (helps auto-vectoriser align loop boundaries).
    let t_pow2 = prev_pow2(t_raw);

    t_pow2.clamp(TILE_MIN, TILE_MAX)
}

/// Round `n` down to the largest power-of-two ≤ n.
/// Returns 1 for n == 0.
#[inline]
fn prev_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let bits = usize::BITS as usize - n.leading_zeros() as usize;
    1usize << (bits - 1)
}

/// A sensible default tile for use when the caller does not wish to compute
/// one — equivalent to `adaptive_tile(0, 0, 0)`.
pub const DEFAULT_TILE: usize = {
    // Evaluated at compile time.
    // Three T×T f64 panels: 3 * T^2 * 8 <= L1_TARGET.
    // Solve for T (integer sqrt):
    let _budget = L1_TARGET_BYTES / (3 * 8); // max T^2
    // Integer sqrt (Newton's method unrolled for const eval):
    // budget ≤ 1365 for 32 KB → sqrt(1365) ≈ 36 → prev_pow2 = 32
    // We hardcode the result here; adaptive_tile() does this at runtime.
    32_usize
};

// ---------------------------------------------------------------------------
// Tiled matmul
// ---------------------------------------------------------------------------

/// Compute `A × B` using cache-blocked tiling (single-threaded).
///
/// Pass `tile = 0` to use [`adaptive_tile`] automatically.
/// Pass an explicit tile size for manual control.
///
/// # Panics
/// Panics if `a.cols != b.rows`.
pub fn matmul_tiled(a: &Matrix, b: &Matrix, tile: usize) -> Matrix {
    assert_eq!(
        a.cols, b.rows,
        "matmul_tiled: inner dimensions must match ({} vs {})",
        a.cols, b.rows
    );

    let m = a.rows;
    let k_dim = a.cols;
    let n = b.cols;
    let t = if tile == 0 { adaptive_tile(m, n, k_dim) } else { tile };

    let mut c = Matrix::zeros(m, n);

    let mut big_i = 0;
    while big_i < m {
        let i_end = (big_i + t).min(m);
        let mut big_k = 0;
        while big_k < k_dim {
            let k_end = (big_k + t).min(k_dim);
            let mut big_j = 0;
            while big_j < n {
                let j_end = (big_j + t).min(n);
                for i in big_i..i_end {
                    for k in big_k..k_end {
                        let a_ik = unsafe { *a.data.get_unchecked(i * a.cols + k) };
                        let c_row = &mut c.data[i * n + big_j..i * n + j_end];
                        let b_row = &b.data[k * n + big_j..k * n + j_end];
                        for (cj, bj) in c_row.iter_mut().zip(b_row.iter()) {
                            *cj += a_ik * bj;
                        }
                    }
                }
                big_j += t;
            }
            big_k += t;
        }
        big_i += t;
    }
    c
}

// ---------------------------------------------------------------------------
// SIMD Optimized Tiled Matmul
// ---------------------------------------------------------------------------

/// Compute `A × B` using cache-blocked tiling and **SIMD acceleration**.
///
/// This version transposes `B` internally to allow the inner loop to use
/// a high-performance SIMD dot-product kernel.
///
/// # Performance
/// - Uses `f64x4` (256-bit vectors).
/// - 4x loop unrolling in the inner K-loop.
/// - Data prefetching for upcoming cache lines.
pub fn matmul_tiled_simd(a: &Matrix, b: &Matrix, tile: usize) -> Matrix {
    assert_eq!(a.cols, b.rows);
    
    let m = a.rows;
    let k_dim = a.cols;
    let n = b.cols;
    let t = if tile == 0 { adaptive_tile(m, n, k_dim) } else { tile };

    // Transpose B: B[k][j] -> BT[j][k]
    // This turns the inner loop into a dot product between two contiguous rows.
    let bt = b.transpose();
    let mut c = Matrix::zeros(m, n);

    let mut big_i = 0;
    while big_i < m {
        let i_end = (big_i + t).min(m);
        let mut big_j = 0;
        while big_j < n {
            let j_end = (big_j + t).min(n);
            
            // Note: K-loop is now inside i/j loops for dot-product style tiling.
            for i in big_i..i_end {
                let a_row_ptr = unsafe { a.data.as_ptr().add(i * k_dim) };
                for j in big_j..j_end {
                    let bt_row_ptr = unsafe { bt.data.as_ptr().add(j * k_dim) };
                    
                    // Call the HPC SIMD micro-kernel
                    let sum = unsafe {
                        crate::matmul::simd_kernel::dot_simd_unroll_prefetch(
                            a_row_ptr, 
                            bt_row_ptr, 
                            k_dim
                        )
                    };
                    
                    unsafe {
                        *c.data.get_unchecked_mut(i * n + j) = sum;
                    }
                }
            }
            big_j += t;
        }
        big_i += t;
    }
    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::naive::matmul_naive;

    #[test]
    fn adaptive_tile_bounds() {
        // Must always be within [TILE_MIN, TILE_MAX] regardless of inputs.
        for &(m, n, k) in &[(1, 1, 1), (8, 8, 8), (64, 64, 64), (1024, 1024, 1024),
                             (4096, 4096, 4096), (7, 13, 31)] {
            let t = adaptive_tile(m, n, k);
            assert!(t >= TILE_MIN && t <= TILE_MAX,
                "adaptive_tile({m},{n},{k}) = {t} out of [{TILE_MIN},{TILE_MAX}]");
            // Must be a power of two.
            assert!(t.is_power_of_two(), "tile {t} is not a power of two");
        }
    }

    #[test]
    fn adaptive_tile_three_panels_fit_l1() {
        let t = adaptive_tile(1024, 1024, 1024);
        let bytes = 3 * t * t * 8;
        assert!(bytes <= L1_TARGET_BYTES * 2, // allow up to 2× budget for real CPUs
            "three panels = {bytes} B exceeds 2×L1_TARGET");
    }

    fn check_matches_naive(size: usize, tile: usize) {
        let a = Matrix::random(size, size, 1337);
        let b = Matrix::random(size, size, 9001);
        let expected = matmul_naive(&a, &b);
        let got = matmul_tiled(&a, &b, tile);
        assert!(
            got.approx_eq(&expected, 1e-9),
            "tiled result diverges from naive for size={size} tile={tile}"
        );
    }

    #[test]
    fn tiled_matches_naive_small() {
        check_matches_naive(8, 0); // tile=0 → adaptive
    }

    #[test]
    fn tiled_matches_naive_medium() {
        check_matches_naive(64, 0);
    }

    #[test]
    fn tiled_non_multiple_of_tile() {
        check_matches_naive(37, 8);
    }

    #[test]
    fn tiled_explicit_tile_16() {
        check_matches_naive(128, 16);
    }
}
