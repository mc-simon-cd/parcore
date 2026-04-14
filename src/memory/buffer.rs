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

//! `SharedBuffer<T>` — thread-safe unified memory abstraction.
//!
//! Backed by `Arc<parking_lot::RwLock<Vec<T>>>`, this allows:
//! - Many concurrent readers (`.read()`)
//! - One exclusive writer (`.write()`)
//! - Cheap handle cloning (`.clone_handle()`) — no data copy.

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::Arc;

/// A clone-able, thread-safe handle to a heap-allocated data buffer.
///
/// All clones share the same underlying allocation, making this the
/// "unified memory" primitive: no explicit host↔device copying is needed.
///
/// # Example
/// ```rust
/// use parcore::memory::SharedBuffer;
///
/// let buf = SharedBuffer::from_vec(vec![1.0_f64; 1024]);
/// let handle = buf.clone_handle();
///
/// // Thread 1
/// let r = buf.read();
/// assert_eq!(r[0], 1.0);
/// drop(r);
///
/// // Thread 2 (could be in a closure sent to the pool)
/// buf.write()[0] = 42.0;
/// assert_eq!(handle.read()[0], 42.0);
/// ```
/// Tracks where the most up-to-date data resides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirtyState {
    /// Data is identical on Host (CPU) and Device (GPU).
    Synced,
    /// Data was modified on Host, Device version is stale.
    HostDirty,
    /// Data was modified on Device, Host version is stale.
    DeviceDirty,
}

/// A clone-able, thread-safe handle to a heap-allocated data buffer.
pub struct SharedBuffer<T> {
    inner: Arc<RwLock<Vec<T>>>,
    /// Logical size (number of elements).
    pub len: usize,
    /// Optional GPU buffer associated with this handle.
    wgpu_buf: Arc<RwLock<Option<Arc<wgpu::Buffer>>>>,
    /// Current synchronization state.
    pub state: Arc<RwLock<DirtyState>>,
    /// Cached WgpuState for lazy synchronization.
    wgpu_context: Arc<RwLock<Option<std::sync::Weak<crate::kernel::compute_unit::WgpuState>>>>,
}

impl<T: Send + Sync + 'static> SharedBuffer<T> {
    /// Create a new buffer filled with `len` copies of `value`.
    pub fn new(len: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self {
            inner: Arc::new(RwLock::new(vec![value; len])),
            len,
            wgpu_buf: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(DirtyState::HostDirty)),
            wgpu_context: Arc::new(RwLock::new(None)),
        }
    }

    /// Wrap an existing `Vec<T>` without copying.
    pub fn from_vec(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            inner: Arc::new(RwLock::new(data)),
            len,
            wgpu_buf: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(DirtyState::HostDirty)),
            wgpu_context: Arc::new(RwLock::new(None)),
        }
    }

    /// Acquire a shared read guard. Triggers Lazy Sync if DeviceDirty.
    pub fn read(&self) -> RwLockReadGuard<'_, Vec<T>> 
    where T: Copy + 'static
    {
        let s = *self.state.read();
        if s == DirtyState::DeviceDirty {
            let ctx_opt = self.wgpu_context.read().clone();
            if let Some(weak_ctx) = ctx_opt {
                if let Some(ctx) = weak_ctx.upgrade() {
                    self.fetch_from_gpu_sync(&ctx.device, &ctx.queue);
                }
            }
        }
        self.inner.read()
    }

    /// Acquire an exclusive write guard. Marks as HostDirty.
    pub fn write(&self) -> RwLockWriteGuard<'_, Vec<T>> {
        let mut s = self.state.write();
        let guard = self.inner.write();
        *s = DirtyState::HostDirty;
        guard
    }

    /// Mark the buffer as modified by the GPU.
    pub fn mark_device_dirty(&self) {
        *self.state.write() = DirtyState::DeviceDirty;
    }

    /// Set the GPU context for lazy synchronization.
    pub fn set_wgpu_context(&self, state: &Arc<crate::kernel::compute_unit::WgpuState>) {
        *self.wgpu_context.write() = Some(Arc::downgrade(state));
    }

    /// Create a new handle pointing to the same allocation.
    pub fn clone_handle(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            len: self.len,
            wgpu_buf: Arc::clone(&self.wgpu_buf),
            state: Arc::clone(&self.state),
            wgpu_context: Arc::clone(&self.wgpu_context),
        }
    }

    /// Create an f32-backed SharedBuffer with the same content (converted).
    /// Used for GPU kernels requiring f32.
    pub fn clone_to_f32(&self) -> SharedBuffer<f32> 
    where T: Copy + Into<f64> + 'static {
        let data = self.inner.read();
        let f32_data: Vec<f32> = data.iter().map(|&v| {
            let v_f64: f64 = v.into();
            v_f64 as f32
        }).collect();
        SharedBuffer::from_vec(f32_data)
    }

    /// Get the underlying wgpu::Buffer handle if it exists.
    pub fn get_gpu_buffer(&self) -> Option<Arc<wgpu::Buffer>> {
        let g = self.wgpu_buf.read();
        (*g).clone()
    }
}

impl<T: Send + Sync + Copy + 'static> SharedBuffer<T> {
    /// Sync data from Host (CPU) to Device (GPU).
    pub fn sync_to_gpu(&self, device: &wgpu::Device, queue: &wgpu::Queue)
    where
        T: Into<f64> + 'static,
    {
        // Capture context for lazy sync (assumes this is the primary device for this buffer)
        // Note: In refined impl, we'd need the Arc<WgpuState> passed here.
        // For now, we'll continue with existing signature and update in the bridge.
        
        let mut s = self.state.write();
        if *s == DirtyState::Synced {
            return; // No-op if already clean
        }

        let mut g = self.wgpu_buf.write();
        let data = self.inner.read();

        let element_size = 4; // f32
        let raw_size = self.len * element_size;
        let padded_size = (raw_size + 255) & !255;

        if g.is_none() {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ParCore SharedBuffer GPU"),
                size: padded_size as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            *g = Some(Arc::new(buffer));
        }

        if let Some(ref buffer) = *g {
            let f32_data: Vec<f32> = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                data.iter().map(|&v| {
                    let v_f64: f64 = unsafe { *(&v as *const T as *const f64) };
                    v_f64 as f32
                }).collect()
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                data.iter().map(|&v| {
                    unsafe { *(&v as *const T as *const f32) }
                }).collect()
            } else {
                panic!("SharedBuffer synchronization only supported for f32 and f64");
            };

            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&f32_data));
            *s = DirtyState::Synced;
        }
    }

    /// Fetch data from GPU back to Host. Blocking wrapper for DSL safety.
    pub fn fetch_from_gpu_sync(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        pollster::block_on(self.fetch_from_gpu(device, queue));
    }

    /// Fetch data from Device (GPU) back to Host (CPU) (Async).
    pub async fn fetch_from_gpu(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut s = self.state.write();
        if *s == DirtyState::Synced || *s == DirtyState::HostDirty {
            return;
        }

        let g = self.wgpu_buf.read();
        if let Some(ref buffer) = *g {
            let element_size = 4;
            let raw_size = self.len * element_size;
            let padded_size = (raw_size + 255) & !255;

            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ParCore Staging Buffer"),
                size: padded_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, padded_size as u64);
            queue.submit(Some(encoder.finish()));

            let (tx, rx) = futures::channel::oneshot::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
            device.poll(wgpu::Maintain::Wait);
            
            if let Ok(Ok(_)) = rx.await {
                let view = staging.slice(..raw_size as u64).get_mapped_range();
                let f32_data: &[f32] = bytemuck::cast_slice(&view);
                
                let mut host_data = self.inner.write();
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    for (i, &v) in f32_data.iter().enumerate() {
                        if i < self.len {
                            let v_64 = v as f64;
                            host_data[i] = unsafe { *(&v_64 as *const f64 as *const T) };
                        }
                    }
                } else {
                    for (i, &v) in f32_data.iter().enumerate() {
                        if i < self.len {
                            host_data[i] = unsafe { *(&v as *const f32 as *const T) };
                        }
                    }
                }
                drop(view);
                staging.unmap();
                *s = DirtyState::Synced;
            }
        }
    }
}

impl<T: Send + Sync + 'static> Clone for SharedBuffer<T> {
    fn clone(&self) -> Self {
        self.clone_handle()
    }
}

unsafe impl<T: Send> Send for SharedBuffer<T> {}
unsafe impl<T: Sync> Sync for SharedBuffer<T> {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn basic_read_write() {
        let buf = SharedBuffer::new(8, 0_u64);
        buf.write()[3] = 99;
        assert_eq!(buf.read()[3], 99);
    }

    #[test]
    fn shared_across_threads() {
        let buf = SharedBuffer::from_vec(vec![0_i32; 16]);
        let handle = buf.clone_handle();

        let t = thread::spawn(move || {
            for i in 0..16 {
                handle.write()[i] = i as i32;
            }
        });

        t.join().unwrap();

        let r = buf.read();
        for i in 0..16 {
            assert_eq!(r[i], i as i32);
        }
    }
}
