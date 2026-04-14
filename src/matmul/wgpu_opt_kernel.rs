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

/// A HIGH-PERFORMANCE Vectorized Matrix Multiplication kernel for WGPU.
/// 
/// Leverages vec4 storage loads and 1x4 micro-tiling:
/// - Each thread computes 4 horizontal elements of C.
/// - Uses workgroup memory to cache A and B tiles.
pub struct WgpuMatMulVec4Kernel {
    pub a: SharedBuffer<f32>,
    pub b: SharedBuffer<f32>,
    pub c: SharedBuffer<f32>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl Kernel for WgpuMatMulVec4Kernel {
    fn name(&self) -> &str { "matmul_vec4_wgpu" }

    fn execute(&self, _ctx: &KernelContext) {
        // CPU fallback could be added here
    }

    fn wgsl_code(&self) -> String {
        format!(r#"
            @group(0) @binding(0) var<storage, read_write> A: array<f32>;
            @group(0) @binding(1) var<storage, read_write> B: array<vec4<f32>>;
            @group(0) @binding(2) var<storage, read_write> C: array<vec4<f32>>;

            const TS: u32 = 16u; // Tile size

            var<workgroup> tile_a: array<array<f32, TS>, TS>;
            var<workgroup> tile_b: array<array<vec4<f32>, TS>, TS>;

            @compute @workgroup_size(16, 16)
            fn main(
                @builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
            ) {{
                let col = global_id.x; // thread 0..N/4
                let row = global_id.y; // thread 0..M
                let tx = local_id.x;
                let ty = local_id.y;

                let M = {m}u;
                let N = {n}u;
                let K = {k}u;

                var sum = vec4<f32>(0.0);

                let num_tiles = (K + TS - 1u) / TS;
                for (var t = 0u; t < num_tiles; t = t + 1u) {{
                    
                    // Load A into shared memory (one f32 per thread)
                    let a_col = t * TS + tx;
                    if (row < M && a_col < K) {{
                        tile_a[ty][tx] = A[row * K + a_col];
                    }} else {{
                        tile_a[ty][tx] = 0.0;
                    }}

                    // Load B into shared memory (one vec4 per thread)
                    // Covers 4 columns of B per load
                    let b_row = t * TS + ty;
                    if (b_row < K && col < (N / 4u)) {{
                        tile_b[ty][tx] = B[b_row * (N / 4u) + col];
                    }} else {{
                        tile_b[ty][tx] = vec4<f32>(0.0);
                    }}

                    workgroupBarrier();

                    // Accumulate 1x4 block
                    for (var i = 0u; i < TS; i = i + 1u) {{
                        sum = sum + tile_a[ty][i] * tile_b[i][tx];
                    }}

                    workgroupBarrier();
                }}

                if (row < M && col < (N / 4u)) {{
                    C[row * (N / 4u) + col] = sum;
                }}
            }}
        "#, m = self.m, n = self.n, k = self.k)
    }

    fn workgroup_size(&self) -> (u32, u32, u32) {
        (16, 16, 1)
    }

    fn grid_size(&self) -> Option<(u32, u32, u32)> {
        Some((
            (self.n / 4 + 15) / 16, // N is divided by 4 due to vec4
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

    fn mark_outputs_dirty(&self) {
        self.c.mark_device_dirty();
    }
}
