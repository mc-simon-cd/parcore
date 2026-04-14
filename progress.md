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

---

## 📊 Final Benchmark Results (12-thread CPU, release build)

| Matrix Size | Naive | Tiled (Scalar) | **SIMD** | **Parallel** | **Optimized** |
|-------------|-------|----------------|----------|--------------|---------------|
| 256×256 | 9.3 ms | 10.0 ms | 7.8 ms | 7.5 ms | **2.8 ms** |
| 512×512 | 70.7 ms | 83.1 ms | 62.2 ms | 41.1 ms | **24.0 ms** |
| **1024×1024** | 618 ms | 662 ms | 549 ms | 335 ms | **171 ms** |

**Max Throughput**: ~12.5 GFlop/s (on 1024x1024 optimized).

---

## 🗺️ Future Roadmap
- [ ] **wgpu Backend**: Real GPU dispatch in `ComputeUnit`.
- [ ] **NPU Backend**: Discrete queue with fixed latency model.
- [ ] **Kernel Fusion**: Combine consecutive IR nodes to save memory bandwidth.
- [ ] **NUMA-Awareness**: Worker pinning to physical cores.
