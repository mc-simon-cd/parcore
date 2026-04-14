# ParCore — Claude / AI Agent Context

> This file provides context for AI coding assistants (Claude, Gemini, etc.)
> working on the **ParCore** project.

---

## Project Summary

**ParCore** is a high-performance parallel runtime system built in Rust.  
It is inspired by CUDA and OpenCL and features a **CUDA-like DSL** and **SIMD-optimized** compute kernels.

- **Language**: Rust (edition 2021, stable toolchain)
- **Build**: `cargo build --release` (LTO enabled, `opt-level=3`)
- **Test**: `cargo test --release`
- **Run**: `cargo run --release`

---

## Architecture

```
src/
├── lib.rs           → public API: parallel_for, spawn
├── runtime.rs       → Runtime struct (owns ThreadPool)
├── main.rs          → demo + benchmark
├── dsl/             → [NEW] DSL System (Macros, IR, Launch API)
├── scheduler/
│   ├── worker.rs    → work-stealing loop
│   └── pool.rs      → ThreadPool, hybrid spin/yield barrier
├── memory/
│   └── buffer.rs    → SharedBuffer<T>: Arc handles, zero-copy
├── kernel/
│   ├── mod.rs       → Kernel trait, KernelContext
│   └── compute_unit.rs → ComputeUnit: Cpu|SimGpu|SimNpu
└── matmul/
    ├── mod.rs       → Matrix (row-major f64)
    ├── optimized.rs → [NEW] Multi-threaded + SIMD + Adaptive Tile
    ├── simd_kernel.rs → [NEW] Manual SIMD micro-kernel
    ├── tiled.rs     → Adaptive cache-blocked (tile=32)
    └── parallel.rs  → parallel tiled
```

---

## Key Design Decisions

### DSL System (`dsl/`)
- `parcore_kernel!` macros generate `ClosureKernel` with static closure captures.
- `KernelCtx` provides CUDA-like `global_id`, `local_id`, etc.
- `KernelIR` allows future backend-agnostic compilation.

### SIMD Optimizations (`simd_kernel.rs`)
- Uses `wide` crate for stable SIMD (f64x4).
- 4x unrolling and x86 prefetching (`_mm_prefetch`).

### Work-Stealing Scheduler (`pool.rs`, `worker.rs`)
- Hybrid spin/yield back-off for barrier synchronization.
- Workers: pop local → drain injector → steal peer.

---

## Dependencies

```toml
crossbeam = "0.8"
parking_lot = "0.12"
num_cpus = "1"
wide = "0.7"
```

---

## Unsafe Code Locations

| File | Reason / Invariant |
|------|--------------------|
| `src/matmul/parallel.rs` | Non-overlapping row slices across thread tasks. |
| `src/matmul/optimized.rs` | Row-slice partitioning for chunked parallel work. |
| `src/matmul/simd_kernel.rs` | Raw pointer loads/unaligned loading for SIMD vectors. |
| `src/matmul/tiled.rs` | `get_unchecked` for performance in tight loops. |

---

## Extension Points

- **wgpu backend**: Implement a real GPU target in `ComputeUnit`.
- **Kernel Fusion**: Combine consecutive DSL kernels into a single pass.
- **NPU Backend**: Fixed-latency model for AI-specific accelerators.
- **NUMA-Awareness**: Worker pinning to physical cores.
