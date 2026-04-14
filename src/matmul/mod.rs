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

//! # Matrix Multiplication Engine
//!
//! Provides four implementations of matrix multiplication, ordered from
//! slowest/simplest to fastest/most complex:
//!
//! | Module       | Algorithm                           | Parallelism |
//! |--------------|--------------------------------------|-------------|
//! | `naive`      | Triple nested loop (O(n³))           | None        |
//! | `tiled`      | Adaptive cache-blocked (i-k-j order) | None        |
//! | `parallel`   | Tiled + row-chunk dispatch           | Pool        |
//! | `optimized`  | Transpose-B + adaptive tile + chunks | Pool + SIMD |
//!
//! All four share the `Matrix` type defined in this module.

pub mod naive;
pub mod optimized;
pub mod parallel;
pub mod simd_kernel;
pub mod tiled;
pub mod wgpu_kernel;
pub mod wgpu_opt_kernel;

pub use naive::matmul_naive;
pub use optimized::matmul_optimized;
pub use parallel::matmul_parallel;
pub use tiled::{adaptive_tile, matmul_tiled, matmul_tiled_simd, DEFAULT_TILE};
pub use wgpu_kernel::WgpuMatMulKernel;
pub use wgpu_opt_kernel::WgpuMatMulVec4Kernel;

/// A row-major dense matrix of `f64`.
///
/// Element `(r, c)` is stored at offset `r * cols + c` in `data`.
/// Row-major layout makes row-iteration cache-friendly; transposing B before
/// column iteration converts the otherwise-strided B access to sequential.
#[derive(Debug, Clone)]
pub struct Matrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Flat row-major storage.  Length == rows × cols.
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create an all-zero matrix.
    #[inline]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Wrap an existing `Vec<f64>` (row-major) without copying.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Matrix::from_vec: length {}, expected {}",
            data.len(), rows * cols
        );
        Self { rows, cols, data }
    }

    /// Create a pseudo-random matrix using a fast LCG (no external dep).
    ///
    /// Values are in `[0, 1)`. Seeded purely from `seed` — deterministic
    /// across platforms (no PRNG state from the OS).
    pub fn random(rows: usize, cols: usize, seed: u64) -> Self {
        let n = rows * cols;
        let mut data = Vec::with_capacity(n);
        let mut s = seed;
        for _ in 0..n {
            // Knuth multiplicative hash (LCG constants from Numerical Recipes).
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            data.push((s >> 33) as f64 / u32::MAX as f64);
        }
        Self { rows, cols, data }
    }

    /// Return the transpose: a new matrix where `T[j][i] = self[i][j]`.
    ///
    /// ## Cache behaviour
    ///
    /// Transposing reads A in row-major (sequential) and writes A_T in
    /// column-major of A, which is row-major of A_T. The one-time O(n²)
    /// transpose cost is recovered immediately in matrix-multiply inner loops
    /// because `B_T[j][:]` is a sequential row rather than a strided column.
    ///
    /// For matmul: without transpose, reading column `j` of B requires
    /// strides of `cols` elements — every access is a cache miss.
    /// After transposing B, reading `B_T[j][:]` is sequential → hardware
    /// prefetcher can stream it optimally.
    pub fn transpose(&self) -> Self {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                unsafe {
                    // SAFETY: i < self.rows, j < self.cols → both indices in bounds.
                    *out.data.get_unchecked_mut(j * self.rows + i) =
                        *self.data.get_unchecked(i * self.cols + j);
                }
            }
        }
        out
    }

    /// Get element at `(row, col)`.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// Set element at `(row, col)`.
    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.cols + col] = val;
    }

    /// Add `val` to element at `(row, col)`.
    #[inline(always)]
    pub fn add(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.cols + col] += val;
    }

    /// Return `true` if every element is within `eps` of `other` element-wise.
    pub fn approx_eq(&self, other: &Matrix, eps: f64) -> bool {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "approx_eq: shape mismatch ({},{}) vs ({},{})",
            self.rows, self.cols, other.rows, other.cols
        );
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| (a - b).abs() <= eps)
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let show_r = self.rows.min(6);
        let show_c = self.cols.min(6);
        for r in 0..show_r {
            for c in 0..show_c {
                write!(f, "{:8.3}", self.get(r, c))?;
            }
            if self.cols > show_c {
                write!(f, " …")?;
            }
            writeln!(f)?;
        }
        if self.rows > show_r {
            writeln!(f, "  … ({} total rows)", self.rows)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_roundtrip() {
        let a = Matrix::random(7, 13, 42);
        let tt = a.transpose().transpose();
        assert!(a.approx_eq(&tt, 1e-15), "transpose roundtrip failed");
    }

    #[test]
    fn transpose_shape() {
        let a = Matrix::zeros(3, 5);
        let t = a.transpose();
        assert_eq!((t.rows, t.cols), (5, 3));
    }

    #[test]
    fn transpose_values() {
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        // T[j][i] = A[i][j]
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(2, 0), 3.0);
        assert_eq!(t.get(0, 1), 4.0);
    }
}
