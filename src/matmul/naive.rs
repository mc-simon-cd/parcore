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

//! Naive matrix multiplication — O(n³) triple loop.
//!
//! This is the straightforward textbook algorithm with no optimisations.
//! Used as a correctness baseline and performance lower bound in benchmarks.
//!
//! C[i][j] = Σₖ A[i][k] · B[k][j]

use super::Matrix;

/// Compute `A × B` using the naive triple-loop algorithm (single-threaded).
///
/// # Panics
/// Panics if `a.cols != b.rows`.
pub fn matmul_naive(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(
        a.cols, b.rows,
        "matmul_naive: inner dimensions must match ({} vs {})",
        a.cols, b.rows
    );

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let mut c = Matrix::zeros(m, n);

    for i in 0..m {
        for p in 0..k {
            // Hoist A[i][p] out of the inner loop — avoids repeated indexing.
            let a_ip = a.get(i, p);
            for j in 0..n {
                // c[i][j] += a[i][p] * b[p][j]
                let val = a_ip * b.get(p, j);
                c.add(i, j, val);
            }
        }
    }

    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_multiplication() {
        // A × I = A
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let eye = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let c = matmul_naive(&a, &eye);
        assert!(c.approx_eq(&a, 1e-12));
    }

    #[test]
    fn known_result_2x2() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = matmul_naive(&a, &b);
        let expected = Matrix::from_vec(2, 2, vec![19.0, 22.0, 43.0, 50.0]);
        assert!(c.approx_eq(&expected, 1e-12));
    }

    #[test]
    fn non_square() {
        // 2×3 × 3×4 → 2×4
        let a = Matrix::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let b = Matrix::random(3, 4, 42);
        let c = matmul_naive(&a, &b);
        assert_eq!((c.rows, c.cols), (2, 4));
        // Row 0 of C = row 0 of B; row 1 of C = row 1 of B
        for j in 0..4 {
            assert!((c.get(0, j) - b.get(0, j)).abs() < 1e-12);
            assert!((c.get(1, j) - b.get(1, j)).abs() < 1e-12);
        }
    }
}
