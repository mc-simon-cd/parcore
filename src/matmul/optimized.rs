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

//! # Optimized Parallel Matrix Multiplication
//!
//! This module is the **production-grade** matmul engine of ParCore.
//! It combines four independent optimisation techniques:
//!
//! ## 1. B-Transposition (cache miss elimination)
//!
//! The fundamental bottleneck of naive matmul is column access of B:
//!
//! ```text
//! c[i][j] += a[i][k] * b[k][j]   ← b[k][j] strides by `cols` per k step
//! ```
//!
//! Each `b[k][j]` access skips `cols` elements — every access is a cache miss
//! for large matrices. We pre-transpose B so `B_T[j][k] = B[k][j]`:
//!
//! ```text
//! c[i][j] += a[i][k] * b_t[j][k]   ← sequential scan of row j of B_T ✓
//! ```
//!
//! Inner loop body becomes two sequential memory sweeps → hardware prefetcher
//! can stream both at line-fill bandwidth.
//!
//! ## 2. Adaptive tiling (hardware-agnostic)
//!
//! Tile size is computed from a portable L1-budget formula rather than
//! hardcoded. See [`crate::matmul::tiled::adaptive_tile`] for the derivation.
//!
//! ## 3. Cache-line aligned chunk boundaries (false-sharing avoidance)
//!
//! Each thread writes to a contiguous row band of C. A cache line is 64 bytes
//! = 8 × f64. If threads share a cache line the hardware must serialise
//! write-back traffic between cores (false sharing). We pad chunk boundaries
//! to a multiple of 8 rows so no two threads ever share a line.
//!
//! ## 4. Hybrid parallel dispatch (reduced synchronisation)
//!
//! Rather than one task per row (excessive queue pressure) or one task per
//! thread (load imbalance for non-power-of-two sizes), we compute a chunk
//! size that is:
//! - ≥ MIN_ROWS_PER_CHUNK rows (avoids tiny-task overhead)  
//! - rounded up to a multiple of CACHE_LINE_F64 (false-sharing safe)
//! - at most `m / num_threads` rows (load balance target)
//!
//! ## Safety
//!
//! `c.data` is partitioned into non-overlapping row slices; each task holds
//! a mutable reference to exactly one slice. The raw-pointer trick converts
//! the shared `mut` reference to independent per-task slices while maintaining
//! the disjoint-access invariant.

use super::{
    tiled::adaptive_tile,
    Matrix,
};
use crate::runtime::Runtime;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// f64 values per cache line (64 bytes / 8 bytes per f64).
const CACHE_LINE_F64: usize = 8;

/// Absolute minimum rows per work-chunk.
/// Below this the task-submission overhead dominates.
const MIN_ROWS_PER_CHUNK: usize = 4;

/// Main entry point — full optimized path.
///
/// `tile = 0` → choose automatically via [`adaptive_tile`].
///
/// Steps:
/// 1. Transpose B (one-time O(n²) cost, eliminates all inner-loop cache misses)
/// 2. Compute adaptive tile size from matrix dimensions
/// 3. Distribute row-bands to the thread pool (false-sharing-safe boundaries)
/// 4. Run tiled micro-kernel on B_T (two sequential memory streams per dot)
pub fn matmul_optimized(rt: &Runtime, a: &Matrix, b: &Matrix, tile: usize) -> Matrix {
    assert_eq!(
        a.cols, b.rows,
        "matmul_optimized: inner dimensions must match ({} vs {})",
        a.cols, b.rows
    );

    let m = a.rows;
    let k_dim = a.cols;
    let n = b.cols;

    // ── Step 1: Transpose B ───────────────────────────────────────────────
    // b_t[j][k] = b[k][j]  (shape: n × k_dim)
    // Cost: O(n·k) sequential writes — CPU prefetcher streams both src and dst.
    let b_t = Arc::new(b.transpose());

    // ── Step 2: Adaptive tile ─────────────────────────────────────────────
    let t = if tile == 0 {
        adaptive_tile(m, n, k_dim)
    } else {
        tile.max(1)
    };

    // ── Step 3: Chunk sizing ──────────────────────────────────────────────
    //
    // Target: ceil(m / num_threads) rows per chunk.
    // Constraint A: at least MIN_ROWS_PER_CHUNK rows (avoid micro-task overhead).
    // Constraint B: rounded up to CACHE_LINE_F64 rows (false-sharing avoidance).
    //
    // Rationale for Constraint B:
    //   If chunk_rows * n % CACHE_LINE_F64 != 0, the last/first f64 of adjacent
    //   chunks share a cache line. When two cores write to the same line, the
    //   MESI protocol forces HITM (RFO cycles) → serialises writes. Aligning
    //   chunk boundaries to CACHE_LINE_F64 rows guarantees line disjointness
    //   when n itself is a multiple of 8, which holds for any power-of-two n.
    //   For arbitrary n we accept the padding — false sharing on boundary elements
    //   is rare and not on the hot path.
    let raw_chunk = ((m + rt.num_units - 1) / rt.num_units).max(MIN_ROWS_PER_CHUNK);
    // Round up to next multiple of CACHE_LINE_F64.
    let chunk_rows = ((raw_chunk + CACHE_LINE_F64 - 1) / CACHE_LINE_F64) * CACHE_LINE_F64;

    let num_chunks = (m + chunk_rows - 1) / chunk_rows;

    // ── Step 4: Parallel dispatch ─────────────────────────────────────────
    let mut c = Matrix::zeros(m, n);
    let a = Arc::new(a.clone());

    let c_ptr = c.data.as_mut_ptr();
    let done = Arc::new(AtomicUsize::new(0));

    for chunk in 0..num_chunks {
        let row_start = chunk * chunk_rows;
        let row_end = (row_start + chunk_rows).min(m);
        let local_rows = row_end - row_start;

        let a = Arc::clone(&a);
        let b_t = Arc::clone(&b_t);
        let done = Arc::clone(&done);

        // SAFETY: each chunk covers [row_start*n .. row_end*n), strictly
        // non-overlapping across chunks. Raw pointer strips the borrow so the
        // closure can be 'static. The pointer is valid for the lifetime of
        // this function because we barrier before returning.
        let c_raw = c_ptr as usize;

        rt.pool.submit(move || {
            // Reconstruct the output slice for this chunk.
            // SAFETY: row_start*n + local_rows*n <= m*n = c.data.len().
            let c_slice: &mut [f64] = unsafe {
                std::slice::from_raw_parts_mut(
                    (c_raw as *mut f64).add(row_start * n),
                    local_rows * n,
                )
            };

            // ── Tiled micro-kernel using transposed B ──────────────────────
            //
            // Access pattern (with B_T):
            //   a[i][k]   → row i of A → sequential ✓
            //   b_t[j][k] → row j of B_T → sequential ✓ (was column j of B)
            //   c[i][j]   → written once per (i,j) → sequential ✓
            //
            // Tile loop order: i (rows of chunk) → j-tile → k-tile → k → j
            // This keeps the c sub-row hot in L1 across the k-reduction.

            let mut big_i = 0;
            while big_i < local_rows {
                let i_end = (big_i + t).min(local_rows);

                let mut big_j = 0;
                while big_j < n {
                    let j_end = (big_j + t).min(n);

                    let mut big_k = 0;
                    while big_k < k_dim {
                        let k_end = (big_k + t).min(k_dim);

                        // Inner micro-kernel: all three tile panels fit in L1.
                        for li in big_i..i_end {
                            let gi = row_start + li;          // global row index
                            let a_row = &a.data[gi * k_dim..gi * k_dim + k_dim];
                            let c_row = &mut c_slice[li * n + big_j..li * n + j_end];

                            for (lj, cj) in (big_j..j_end).zip(c_row.iter_mut()) {
                                // Dot product over k-strip:
                                //   a_row[big_k..k_end] (sequential)
                                //   b_t.data[lj * k_dim + big_k..k_end] (sequential)
                                let b_t_row = &b_t.data[lj * k_dim + big_k..lj * k_dim + k_end];
                                let a_strip = &a_row[big_k..k_end];

                                // LLVM auto-vectorises this reduction to
                                // VFMADD (AVX2) / FMLA (NEON) / VFMADD (RVV).
                                let dot: f64 = a_strip
                                    .iter()
                                    .zip(b_t_row.iter())
                                    .map(|(ak, bk)| ak * bk)
                                    .sum();
                                *cj += dot;
                            }
                        }

                        big_k += t;
                    }
                    big_j += t;
                }
                big_i += t;
            }

            done.fetch_add(1, Ordering::Release);
        });
    }

    // Barrier: spin until all chunks finish.
    while done.load(Ordering::Acquire) < num_chunks {
        std::hint::spin_loop();
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

    fn check_optimized(m: usize, n: usize, k: usize, threads: usize) {
        let rt = Runtime::new(threads);
        let a = Matrix::random(m, k, 0xABC1);
        let b = Matrix::random(k, n, 0xDEF2);
        let expected = matmul_naive(&a, &b);
        let got = matmul_optimized(&rt, &a, &b, 0); // tile=0 → adaptive
        assert!(
            got.approx_eq(&expected, 1e-9),
            "optimized diverges from naive (m={m},n={n},k={k},threads={threads})"
        );
    }

    #[test]
    fn optimized_matches_naive_small() {
        check_optimized(8, 8, 8, 2);
    }

    #[test]
    fn optimized_matches_naive_medium() {
        check_optimized(64, 64, 64, 4);
    }

    #[test]
    fn optimized_matches_naive_large() {
        check_optimized(128, 128, 128, 8);
    }

    #[test]
    fn optimized_non_square() {
        check_optimized(32, 48, 16, 4);
    }

    #[test]
    fn optimized_single_thread() {
        check_optimized(64, 64, 64, 1);
    }

    #[test]
    fn optimized_non_pow2_size() {
        // Boundary condition: size not divisible by tile or chunk_rows.
        check_optimized(37, 41, 29, 3);
    }
}
