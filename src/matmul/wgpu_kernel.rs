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

use crate::kernel::{Kernel, KernelContext};
use crate::memory::SharedBuffer;
use std::sync::Arc;

/// An ADVANCED Tiled Matrix Multiplication kernel for WGPU.
/// 
/// Uses workgroup (shared) memory to cache A and B tiles, drastically
/// reducing storage buffer bandwidth requirements for large matrices.
pub struct WgpuMatMulKernel {
    pub a: SharedBuffer<f32>,
    pub b: SharedBuffer<f32>,
    pub c: SharedBuffer<f32>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl Kernel for WgpuMatMulKernel {
    fn name(&self) -> &str { "matmul_tiled_wgpu" }

    fn execute(&self, _ctx: &KernelContext) {
        // CPU fallback could be implemented here using matmul_tiled_simd
    }

    fn wgsl_code(&self) -> String {
        format!(r#"
            @group(0) @binding(0) var<storage, read_write> A: array<f32>;
            @group(0) @binding(1) var<storage, read_write> B: array<f32>;
            @group(0) @binding(2) var<storage, read_write> C: array<f32>;

            const TILE_SIZE: u32 = 16u;

            var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
            var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

            @compute @workgroup_size(16, 16)
            fn main(
                @builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) group_id: vec3<u32>
            ) {{
                let col = global_id.x;
                let row = global_id.y;
                let tx = local_id.x;
                let ty = local_id.y;

                let M = {m}u;
                let N = {n}u;
                let K = {k}u;

                var sum = 0.0f;

                // Loop over tiles
                let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;
                for (var t = 0u; t < num_tiles; t = t + 1u) {{
                    
                    // Cooperative Load: Each thread loads one element per tile
                    // Load A tile
                    let a_col = t * TILE_SIZE + tx;
                    if (row < M && a_col < K) {{
                        tile_a[ty][tx] = A[row * K + a_col];
                    }} else {{
                        tile_a[ty][tx] = 0.0f;
                    }}

                    // Load B tile
                    let b_row = t * TILE_SIZE + ty;
                    if (b_row < K && col < N) {{
                        tile_b[ty][tx] = B[b_row * N + col];
                    }} else {{
                        tile_b[ty][tx] = 0.0f;
                    }}

                    // Synchronize to ensure all threads finished loading
                    workgroupBarrier();

                    // Compute partial dot product within the tile (unrolled for performance)
                    sum = sum + tile_a[ty][0] * tile_b[0][tx];
                    sum = sum + tile_a[ty][1] * tile_b[1][tx];
                    sum = sum + tile_a[ty][2] * tile_b[2][tx];
                    sum = sum + tile_a[ty][3] * tile_b[3][tx];
                    sum = sum + tile_a[ty][4] * tile_b[4][tx];
                    sum = sum + tile_a[ty][5] * tile_b[5][tx];
                    sum = sum + tile_a[ty][6] * tile_b[6][tx];
                    sum = sum + tile_a[ty][7] * tile_b[7][tx];
                    sum = sum + tile_a[ty][8] * tile_b[8][tx];
                    sum = sum + tile_a[ty][9] * tile_b[9][tx];
                    sum = sum + tile_a[ty][10] * tile_b[10][tx];
                    sum = sum + tile_a[ty][11] * tile_b[11][tx];
                    sum = sum + tile_a[ty][12] * tile_b[12][tx];
                    sum = sum + tile_a[ty][13] * tile_b[13][tx];
                    sum = sum + tile_a[ty][14] * tile_b[14][tx];
                    sum = sum + tile_a[ty][15] * tile_b[15][tx];

                    // Synchronize before loading next tile
                    workgroupBarrier();
                }}

                if (row < M && col < N) {{
                    C[row * N + col] = sum;
                }}
            }}
        "#, m = self.m, n = self.n, k = self.k)
    }

    fn workgroup_size(&self) -> (u32, u32, u32) {
        (16, 16, 1)
    }

    fn grid_size(&self) -> Option<(u32, u32, u32)> {
        Some((
            (self.n + 15) / 16,
            (self.m + 15) / 16,
            1
        ))
    }

    fn gpu_buffers(&self) -> Vec<Arc<wgpu::Buffer>> {
        vec![
            self.a.get_gpu_buffer().expect("A buffer not synced to GPU"),
            self.b.get_gpu_buffer().expect("B buffer not synced to GPU"),
            self.c.get_gpu_buffer().expect("C buffer not synced to GPU"),
        ]
    }
}
