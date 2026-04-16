# ParCore — Progress Log

## Project: High-Performance Parallel Runtime System in Rust

---

## 📅 2026-04-06 — Initial Development
- ✅ Phase 1: Planning
- ✅ Phase 2: Project Scaffold
- ✅ Phase 3: Core Runtime (Scheduler, Memory, Kernel)
- ✅ Phase 4: Basic MatMul Engine (Naive, Tiled, Parallel)

---

## 📅 2026-04-07 — Optimization & DSL Pass

### ✅ Phase 7 — Advanced Kernel Optimizations
- **Adaptive Tiling**: Derived from L1 budget (32 KB).
- **B-Transposition**: Row-major dot-product optimization.
- **Cache-line Alignment**: 64-byte chunking to prevent false sharing.
- **Result**: ~5.8× speedup vs naive at 1024×1024.

### ✅ Phase 8 — HPC SIMD Micro-kernels
- **SIMD Backends**: `wide` (stable) and `std::simd` (nightly).
- **4x Unrolling**: Saturates execution ports.
- **Prefetching**: `_mm_prefetch` for latency hiding.
- **Result**: ~1.2× speedup in single-threaded tiled kernels.

### ✅ Phase 9 — CUDA-inspired DSL
- **Macros**: `parcore_kernel!` / `parcore_kernel_2d!`.
- **Context API**: Type-safe `Dim3` (global/local/group IDs).
- **IR & Plan**: `KernelIR` and `ExecutionGraph` for future GPU targetting.
- **Launch API**: Fluent `KernelLaunchConfig` builder.

### ✅ Phase 10 — WGPU GPU Acceleration
- **Backend Implementation**: Direct WGPU dispatch integration.
- **Resource Caching**: Zero-allocation warm path via `DashMap`.
- **Lazy Synchronization**: Automatic $f64 \leftrightarrow f32$ conversion and state tracking.
- **Result**: ~6.2× speedup vs parallel CPU at 1024×1024.

---

## 📊 Final Benchmark Results (12-thread CPU + WGPU GPU)

| Matrix Size | Naive (CPU) | Parallel (CPU) | **WGPU (Warm)** | **GFlop/s (GPU)** |
|-------------|-------------|----------------|-----------------|-------------------|
| 256×256     | 9.3 ms      | 7.5 ms         | **1.4 ms**      | 24.2              |
| 512×512     | 70.7 ms     | 41.1 ms        | **8.8 ms**      | 30.2              |
| **1024×1024** | 618 ms    | 335 ms         | **72.7 ms**     | **29.5**          |

**Max Throughput**: ~30 GFlop/s (on WGPU).

---

## 🗺️ Future Roadmap
- [ ] **NPU Backend**: Discrete queue with fixed latency model.
- [ ] **Kernel Fusion**: Combine consecutive IR nodes to save memory bandwidth.
- [ ] **NUMA-Awareness**: Worker pinning to physical cores.
- [ ] **Distributed Runtime**: MPI-like multi-node execution.
