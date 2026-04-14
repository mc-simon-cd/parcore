# ParCore 🦀⚡

> **High-performance parallel runtime system in Rust — inspired by CUDA & OpenCL**

---

## Overview

**ParCore** is a modular, heterogeneous parallel computing runtime built for speed. It features a **top-down architecture** from high-level macros down to SIMD-optimized assembly. 

It targets **CPU/NPU** execution (fully **RISC-V compatible**) while offering a **CUDA-inspired DSL** for high-level parallel programming.

```text
┌───────────────────────────────────────────┐
│              Application Layer            │
│       (parcore_kernel! · parcore_dsl!)    │
├──────────────────┬────────────────────────┤
│   Runtime API    │     Kernel System      │
│ (parallel_for)   │   (IR · DSL · Launch)  │
├──────────────────┴────────────────────────┤
│             Scheduler Layer               │
│     (Hybrid Spin/Yield · Work-Stealing)   │
├───────────────────────────────────────────┤
│              Hardware Layer               │
│        (AVX2 · NEON · RVV · SVV)          │
└───────────────────────────────────────────┘
```

---

## 🚀 Key Features

*   📜 **CUDA-like DSL**: `parcore_kernel!` macros with type-safe indexing (`Dim3`, `KernelCtx`).
*   ⚡ **HPC SIMD Engine**: Manual vectorization via `wide` crate with **4x unrolling** and **x86 prefetching**.
*   🔀 **Smart Scheduler**: Hybrid spin/yield back-off for ultra-low latency barriers.
*   🧠 **Unified Memory**: `SharedBuffer<T>` for zero-copy data sharing across compute units.
*   🔧 **Hardware-Agnostic**: **Adaptive tiling** automatically tuned to the CPU's L1-D cache budget.
*   🛡️ **Safe by Design**: Strictly audited `unsafe` usage, fully compliant with **Apache 2.0**.

---

## 🏎️ Benchmark Results

> Machine: 12-thread CPU, release build (`opt-level=3, lto=true`)
> Hardware-adaptive tile: **32 × 32**

| Matrix Size | Naive | Tiled (Scalar) | **SIMD (Unroll)** | **Optimized (Par)** | GFlop/s |
|-------------|-------|----------------|-------------------|---------------------|---------|
| 256×256     | 9.3 ms| 10.0 ms         | 7.8 ms            | **2.8 ms**          | 11.7    |
| 512×512     | 70.7 ms| 83.1 ms        | 62.2 ms           | **24.0 ms**         | 11.1    |
| **1024×1024** | 618 ms| 662 ms         | 549 ms            | **171 ms**          | **12.5**|

*Note: `Optimized` combines B-transpose, manual SIMD, and multi-threading.*

---

## 🛠️ Usage Example

A complete vector addition kernel using the ParCore DSL:

```rust
use parcore::{Runtime, dsl::*};
use parcore::memory::SharedBuffer;

fn main() {
    let rt = Runtime::new(0); // Auto-detect CPUs
    let n = 1024;
    
    // Allocate unified memory
    let a = SharedBuffer::new(n, 1.0f64);
    let b = SharedBuffer::new(n, 2.0f64);
    let out = SharedBuffer::new(n, 0.0f64);

    // Define the kernel
    let kernel = parcore_kernel! {
        name: "vector_add",
        |ctx| {
            let i = ctx.global_id.x;
            if i < n { out[i] = a[i] + b[i]; }
        }
    };

    // Dispatch: 1024 work-items in groups of 32
    KernelLaunchConfig::new_1d(n, 32).dispatch(&rt, kernel);
    
    rt.barrier();
    assert_eq!(out.read()[42], 3.0);
}
```

---

## 📦 Installation & Build

Requires Rust 1.70+ (Stable for basic use, Nightly only if `portable_simd` enabled).

```bash
# Clone and run the comprehensive benchmark suite
git clone https://github.com/mc-simon-cd/parcore.git
cd parcore
cargo run --release
```

---

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
# parcore

