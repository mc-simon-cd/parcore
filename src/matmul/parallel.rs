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

//! Parallel matrix multiplication using the ParCore work-stealing scheduler.
//!
//! ## Strategy
//!
//! The output matrix C has `M` rows. We partition rows into fixed-size chunks
//! (one chunk per task), submitting each chunk as an independent task to the
//! thread pool. Within each chunk we use the tiled inner kernel for cache
//! efficiency.
//!
//! This gives O(M / num_threads) latency (assuming no contention) and writes
//! to non-overlapping rows — no locks needed on the output buffer.
//!
//! ## Safety note
//!
//! `Matrix.data` is accessed at non-overlapping row ranges across threads.
//! We use `unsafe` pointer arithmetic to satisfy the borrow checker while
//! keeping the invariant that no two tasks touch the same elements.

use super::Matrix;
use crate::runtime::Runtime;
use std::sync::Arc;

/// Compute `A × B` in parallel, distributing row-blocks across the thread pool.
///
/// `tile` is the inner cache-blocking size (default: [`DEFAULT_TILE`]).
/// Returns only after all row-blocks have been computed.
///
/// # Panics
/// Panics if `a.cols != b.rows`.
pub fn matmul_parallel(rt: &Runtime, a: &Matrix, b: &Matrix, tile: usize) -> Matrix {
    assert_eq!(
        a.cols, b.rows,
        "matmul_parallel: inner dimensions must match ({} vs {})",
        a.cols, b.rows
    );

    let m = a.rows;
    let k_dim = a.cols;
    let n = b.cols;
    let t = tile.max(1);

    // Allocate output — all zeros.
    let mut c = Matrix::zeros(m, n);

    // Share A and B as Arcs (read-only — safe).
    let a = Arc::new(a.clone());
    let b = Arc::new(b.clone());

    // Number of row-chunks — one per thread is a good starting point.
    let chunk_rows = ((m + rt.num_units - 1) / rt.num_units).max(1);
    let num_chunks = (m + chunk_rows - 1) / chunk_rows;

    // We need mutable, non-overlapping slices of `c.data` across threads.
    // Safety: each chunk writes to a distinct, non-overlapping row range.
    let c_ptr = c.data.as_mut_ptr();
    let c_len = c.data.len();

    // Use a completion counter identical in style to Runtime::parallel_for.
    use std::sync::atomic::{AtomicUsize, Ordering};
    let done = Arc::new(AtomicUsize::new(0));

    for chunk in 0..num_chunks {
        let row_start = chunk * chunk_rows;
        let row_end = (row_start + chunk_rows).min(m);

        let a = Arc::clone(&a);
        let b = Arc::clone(&b);
        let done = Arc::clone(&done);

        // SAFETY: row ranges [row_start * n .. row_end * n) are non-overlapping
        // across chunks, so no two tasks alias the same memory.
        let c_raw = c_ptr as usize; // strip lifetime to send across threads
        let _ = c_len;              // suppress unused warning in release

        rt.pool.submit(move || {
            // Reconstruct mutable slice for this chunk.
            // SAFETY: bounds are [row_start*n .. row_end*n) < c_len, non-aliasing.
            let slice: &mut [f64] = unsafe {
                std::slice::from_raw_parts_mut(
                    (c_raw as *mut f64).add(row_start * n),
                    (row_end - row_start) * n,
                )
            };

            // Local row offset for indexing `slice`.
            let local_m = row_end - row_start;

            // Tiled inner kernel over this row slice.
            let mut big_j = 0;
            while big_j < n {
                let j_end = (big_j + t).min(n);

                let mut big_k = 0;
                while big_k < k_dim {
                    let k_end = (big_k + t).min(k_dim);

                    for li in 0..local_m {
                        let gi = row_start + li;
                        for k in big_k..k_end {
                            let a_ik = a.get(gi, k);
                            for j in big_j..j_end {
                                slice[li * n + j] += a_ik * b.get(k, j);
                            }
                        }
                    }

                    big_k += t;
                }
                big_j += t;
            }

            done.fetch_add(1, Ordering::Release);
        });
    }

    // Wait for all chunks to complete (barrier).
    while done.load(std::sync::atomic::Ordering::Acquire) < num_chunks {
        std::thread::yield_now();
    }

    // Decrement the pool's pending counter for each task we submitted manually.
    // Because we used pool.submit directly (bypassing Runtime::spawn's own
    // pending tracking), we owe the pool num_chunks decrements.
    // Actually pool.submit already incremented pending; tasks decremented it
    // themselves — this is correct as-is (pool.submit → pending++; task done → pending--).

    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::naive::matmul_naive;
    use crate::matmul::tiled::DEFAULT_TILE;

    fn check_parallel(size: usize, threads: usize) {
        let rt = Runtime::new(threads);
        let a = Matrix::random(size, size, 777);
        let b = Matrix::random(size, size, 888);
        let expected = matmul_naive(&a, &b);
        let got = matmul_parallel(&rt, &a, &b, DEFAULT_TILE);
        assert!(
            got.approx_eq(&expected, 1e-9),
            "parallel result diverges from naive (size={size}, threads={threads})"
        );
    }

    #[test]
    fn parallel_matches_naive_small() {
        check_parallel(16, 4);
    }

    #[test]
    fn parallel_matches_naive_medium() {
        check_parallel(128, 8);
    }

    #[test]
    fn parallel_single_thread() {
        check_parallel(64, 1);
    }
}
