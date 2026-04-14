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

//! # ComputeUnit
//!
//! Simulates heterogeneous compute units — CPU cores, GPU SMs, or NPU engines —
//! by mapping each to one or more threads in the ParCore thread pool.
//!
//! `ComputeUnit::dispatch` takes a kernel + grid params, constructs a
//! `KernelContext` for every work item, and submits them to the scheduler.

use super::{Kernel, KernelContext};
use crate::runtime::Runtime;
use std::sync::Arc;

/// Simulated compute unit type.
///
/// In a real heterogeneous runtime, these would correspond to actual hardware.
/// Here each variant is a label; all dispatch to the CPU thread pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitKind {
    /// Standard CPU core(s).
    Cpu,
    /// Simulated GPU streaming multiprocessor.
    SimGpu,
    /// Simulated Neural Processing Unit.
    SimNpu,
    /// Real GPU hardware via WebGPU.
    WgpuGpu,
}

/// WebGPU state container.
pub struct WgpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
    /// Cache for compiled shader modules (keyed by WGSL source hash).
    pub shader_cache: dashmap::DashMap<String, Arc<wgpu::ShaderModule>>,
    /// Cache for compute pipelines (keyed by kernel name).
    pub pipeline_cache: dashmap::DashMap<String, Arc<wgpu::ComputePipeline>>,
    /// Cache for bind group layouts (keyed by number of bindings).
    pub layout_cache: dashmap::DashMap<usize, Arc<wgpu::BindGroupLayout>>,
    /// Cache for bind groups (keyed by buffer address list).
    pub bind_group_cache: dashmap::DashMap<u64, Arc<wgpu::BindGroup>>,
}

/// A compute unit that can dispatch kernels onto the ParCore runtime.
pub struct ComputeUnit {
    /// Unit identity/index (e.g., device 0, 1, …)
    pub id: usize,
    /// Hardware kind.
    pub kind: UnitKind,
    /// WebGPU state (only present for WgpuGpu).
    pub wgpu_state: Option<Arc<WgpuState>>,
}

impl ComputeUnit {
    /// Create a new simulated or CPU compute unit.
    pub fn new(id: usize, kind: UnitKind) -> Self {
        Self { id, kind, wgpu_state: None }
    }

    /// Create a new handle to the same compute unit.
    pub fn clone_handle(&self) -> Self {
        Self {
            id: self.id,
            kind: self.kind,
            wgpu_state: self.wgpu_state.clone(),
        }
    }

    /// Create a new real GPU compute unit.
    pub fn new_wgpu(id: usize) -> Result<Self, String> {
        // Use pollster to block on the async initialization since ParCore is sync.
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .ok_or_else(|| "Failed to find a suitable GPU adapter".to_string())?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("ParCore High-Perf Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
                .map_err(|e| format!("Failed to create wgpu device: {}", e))?;

            Ok(Self {
                id,
                kind: UnitKind::WgpuGpu,
                wgpu_state: Some(Arc::new(WgpuState {
                    device,
                    queue,
                    adapter,
                    shader_cache: dashmap::DashMap::new(),
                    pipeline_cache: dashmap::DashMap::new(),
                    layout_cache: dashmap::DashMap::new(),
                    bind_group_cache: dashmap::DashMap::new(),
                })),
            })
        })
    }

    /// Dispatch `kernel` over `global_size` work items using `local_size`-sized
    /// workgroups. Blocks until all work items complete.
    ///
    /// # Parameters
    /// - `rt`            – the ParCore runtime (owns the thread pool)
    /// - `kernel`        – the kernel to execute; shared across work items
    /// - `global_size`   – total number of work items (analogous to CUDA grid × block)
    /// - `local_size`    – workgroup / block size (must evenly divide global_size,
    ///                     or the remainder is handled automatically)
    ///
    /// # Example
    /// ```rust,no_run
    /// use parcore::kernel::{ComputeUnit, UnitKind, Kernel, KernelContext};
    /// use parcore::Runtime;
    /// use std::sync::Arc;
    ///
    /// struct PrintKernel;
    /// impl Kernel for PrintKernel {
    ///     fn name(&self) -> &str { "print" }
    ///     fn execute(&self, ctx: &KernelContext) { println!("{}", ctx.global_id); }
    /// }
    ///
    /// let rt = Runtime::new(4);
    /// let cu = ComputeUnit::new(0, UnitKind::SimGpu);
    /// cu.dispatch(&rt, Arc::new(PrintKernel), 64, 8);
    /// ```
    pub fn dispatch<K: Kernel + 'static>(
        &self,
        rt: &Runtime,
        kernel: Arc<K>,
        global_size: usize,
        local_size: usize,
    ) {
        if self.kind == UnitKind::WgpuGpu {
            self.dispatch_wgpu(rt, kernel, global_size);
        } else {
            self.dispatch_cpu(rt, kernel, global_size, local_size);
        }
    }

    fn dispatch_cpu<K: Kernel + 'static>(
        &self,
        rt: &Runtime,
        kernel: Arc<K>,
        global_size: usize,
        local_size: usize,
    ) {
        let local_size = local_size.max(1);
        rt.parallel_for(0..global_size, move |gid| {
            let group_id = gid / local_size;
            let local_id = gid % local_size;
            let ctx = KernelContext {
                global_id: gid,
                local_id,
                group_id,
                global_size,
                local_size,
            };
            kernel.execute(&ctx);
        });
    }

    fn dispatch_wgpu<K: Kernel + 'static>(
        &self,
        _rt: &Runtime,
        kernel: Arc<K>,
        global_size: usize,
    ) {
        let state = self.wgpu_state.as_ref().expect("WgpuGpu must have WgpuState");
        let wgsl = kernel.wgsl_code();
        if wgsl.is_empty() {
            panic!("Kernel '{}' does not support GPU execution (empty WGSL)", kernel.name());
        }

        // 1. Calculate WGSL hash (used for both shader and pipeline keys)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        wgsl.hash(&mut hasher);
        let wgsl_hash = hasher.finish();

        // 2. Get or Create Shader Module (Cache by WGSL source)
        let shader_key = wgsl.clone();
        let module = state.shader_cache.entry(shader_key).or_insert_with(|| {
            Arc::new(state.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", kernel.name())),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            }))
        }).value().clone();

        let buffers = kernel.gpu_buffers();
        let num_bindings = buffers.len();

        // 3. Get or Create Bind Group Layout (Cache by number of bindings)
        let bind_group_layout = state.layout_cache.entry(num_bindings).or_insert_with(|| {
            Arc::new(state.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} BGL ({} bindings)", kernel.name(), num_bindings)),
                entries: &(0..num_bindings).map(|i| {
                    wgpu::BindGroupLayoutEntry {
                        binding: i as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                }).collect::<Vec<_>>(),
            }))
        }).value().clone();

        // 4. Get or Create Pipeline (Cache by kernel name + shader hash)
        let pipeline_key = format!("{}:{:x}", kernel.name(), wgsl_hash);
        let pipeline = state.pipeline_cache.entry(pipeline_key).or_insert_with(|| {
            let pipeline_layout = state.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} Pipeline Layout", kernel.name())),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            Arc::new(state.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Pipeline", kernel.name())),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            }))
        }).value().clone();

        // 5. Get or Create Bind Group (Cache by buffer address hash)
        // We hash the raw pointers of the wgpu::Buffer objects to detect reuse.
        let mut bg_hasher = DefaultHasher::new();
        for b in &buffers { Arc::as_ptr(b).hash(&mut bg_hasher); }
        let bg_key = bg_hasher.finish();

        let bind_group = state.bind_group_cache.entry(bg_key).or_insert_with(|| {
            Arc::new(state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Bind Group", kernel.name())),
                layout: &bind_group_layout,
                entries: &buffers.iter().enumerate().map(|(i, b)| {
                    wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    }
                }).collect::<Vec<_>>(),
            }))
        }).value().clone();

        // 5. Encode and Submit
        let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{} Encoder", kernel.name())),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Compute Pass", kernel.name())),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            
            let (ws_x, _ws_y, _ws_z) = kernel.workgroup_size();
            let (gx, gy, gz) = kernel.grid_size().unwrap_or_else(|| {
                ((global_size as u32 + ws_x - 1) / ws_x, 1, 1)
            });
            cpass.dispatch_workgroups(gx, gy, gz);
        }

        state.queue.submit(Some(encoder.finish()));
        state.device.poll(wgpu::Maintain::Wait);
        println!("[dispatch_wgpu] GPU Submission complete for '{}'", kernel.name());

        // Mark output buffers as modified by GPU to trigger lazy fetch
        kernel.mark_outputs_dirty();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountKernel(Arc<AtomicUsize>);
    impl Kernel for CountKernel {
        fn name(&self) -> &str { "count" }
        fn execute(&self, _ctx: &KernelContext) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn dispatch_counts_all_items() {
        let rt = Runtime::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let cu = ComputeUnit::new(0, UnitKind::SimGpu);
        cu.dispatch(&rt, Arc::new(CountKernel(Arc::clone(&counter))), 256, 32);
        assert_eq!(counter.load(Ordering::SeqCst), 256);
    }
}
