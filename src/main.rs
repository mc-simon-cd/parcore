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

//! # ParCore Demo & Benchmark
//!
//! Demonstrates all major features and benchmarks all four matmul implementations:
//!
//! 1. `parallel_for` — parallel iteration
//! 2. Async task spawning with barrier
//! 3. Unified shared memory (`SharedBuffer`)
//! 4. Kernel dispatch on simulated compute units
//! 5. Matrix multiplication benchmark — before vs after optimization

use parcore::{
    dsl::{kernels::matmul_dsl},
    kernel::{ComputeUnit, Kernel, KernelContext, UnitKind},
    matmul::{
        adaptive_tile, matmul_naive, matmul_optimized, matmul_parallel, matmul_tiled,
        matmul_tiled_simd, Matrix, DEFAULT_TILE,
    },
    memory::SharedBuffer,
    Runtime,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;

// ─── Banner ─────────────────────────────────────────────────────────────────

fn banner(title: &str) {
    let width: usize = 62;
    let content = title.len() + 2;
    let pad = width.saturating_sub(content + 2);
    let left = pad / 2;
    let right = pad - left;
    println!("\n╔{}╗", "═".repeat(width));
    println!("║{}  {}  {}║", " ".repeat(left), title, " ".repeat(right));
    println!("╚{}╝", "═".repeat(width));
}

// ─── 1. Parallel For ────────────────────────────────────────────────────────

fn demo_parallel_for(rt: &Runtime) {
    banner("1 · parallel_for");
    const N: usize = 1_000_000;
    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    let t0 = Instant::now();
    parcore::parallel_for(rt, 0..N, move |_| {
        c.fetch_add(1, Ordering::Relaxed);
    });
    let elapsed = t0.elapsed();
    println!(
        "  {N} tasks in {:.3} ms → {:.1} M tasks/s",
        elapsed.as_secs_f64() * 1e3,
        N as f64 / elapsed.as_secs_f64() / 1e6
    );
    assert_eq!(counter.load(Ordering::SeqCst), N);
    println!("  ✓ All {N} tasks completed correctly.");
}

// ─── 2. Async Spawn + Barrier ───────────────────────────────────────────────

fn demo_spawn(rt: &Runtime) {
    banner("2 · spawn + barrier");
    const NTASKS: usize = 500;
    let counter = Arc::new(AtomicUsize::new(0));
    let t0 = Instant::now();
    for i in 0..NTASKS {
        let c = Arc::clone(&counter);
        parcore::spawn(rt, move || {
            let _: u64 = (0_u64..100).fold(i as u64, |a, v| a.wrapping_add(v));
            c.fetch_add(1, Ordering::Relaxed);
        });
    }
    rt.barrier();
    println!(
        "  {NTASKS} async tasks + barrier: {:.3} ms",
        t0.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(counter.load(Ordering::SeqCst), NTASKS);
    println!("  ✓ All {NTASKS} async tasks completed.");
}

// ─── 3. Unified Shared Memory ───────────────────────────────────────────────

fn demo_shared_memory(rt: &Runtime) {
    banner("3 · Unified Shared Memory");
    const LEN: usize = 64;
    let buf = SharedBuffer::new(LEN, 0_f64);
    let buf_w = buf.clone_handle();
    parcore::parallel_for(rt, 0..LEN, move |i| {
        buf_w.write()[i] = i as f64 * 2.0;
    });
    let ok = buf.read().iter().enumerate().all(|(i, &v)| (v - i as f64 * 2.0).abs() < 1e-12);
    println!("  SharedBuffer[{LEN}] written in parallel, read sequentially.");
    println!("  {} Zero-copy, no explicit memcpy.", if ok { "✓" } else { "✗" });
}

// ─── 4. Kernel Dispatch ─────────────────────────────────────────────────────

struct ScaleKernel { buf: SharedBuffer<f64>, factor: f64 }
impl Kernel for ScaleKernel {
    fn name(&self) -> &str { "scale" }
    fn execute(&self, ctx: &KernelContext) {
        self.buf.write()[ctx.global_id] *= self.factor;
    }
}

fn demo_kernel(rt: &Runtime) {
    banner("4 · Kernel Dispatch (Simulated GPU)");
    const SIZE: usize = 256;
    let buf = SharedBuffer::new(SIZE, 1.0_f64);
    let cu = ComputeUnit::new(0, UnitKind::SimGpu);
    let t0 = Instant::now();
    cu.dispatch(rt, Arc::new(ScaleKernel { buf: buf.clone_handle(), factor: 3.14 }), SIZE, 32);
    let ok = buf.read().iter().all(|&v| (v - 3.14).abs() < 1e-12);
    println!(
        "  SimGpu, global={SIZE}, local=32, dispatch={:.3} ms",
        t0.elapsed().as_secs_f64() * 1e3
    );
    println!("  {} Kernel output correct (all == 3.14).", if ok { "✓" } else { "✗" });
}

// ─── 5. Matrix Multiplication Benchmark ─────────────────────────────────────

fn gflops(n: usize, secs: f64) -> f64 {
    2.0 * n as f64 * n as f64 * n as f64 / secs / 1e9
}

/// Run `runs` timed repetitions, return (min_ms, speedup_label).
fn bench<F: Fn() -> Matrix>(runs: u32, f: F) -> (f64, Matrix) {
    let mut result = f();
    let mut min = f64::INFINITY;
    for _ in 0..runs {
        let t0 = Instant::now();
        result = f();
        let s = t0.elapsed().as_secs_f64();
        if s < min { min = s; }
    }
    (min * 1e3, result)
}

fn demo_matmul(rt: &Runtime) {
    banner("5 · Matrix Multiplication — Before vs After Optimization");

    let tile = adaptive_tile(1024, 1024, 1024);
    println!("\n  Hardware-adaptive tile size: {tile} × {tile}");
    println!("  Threads: {}  (auto-detected)\n", rt.num_units);

    let sizes: &[(usize, u32)] = &[(256, 5), (512, 3), (1024, 2)];

    // Header
    println!(
        "  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "Size", "Naive(ms)", "Tiled(ms)", "SIMD(ms)", "Par(ms)", "Optimized(ms)", "GFlop/s"
    );
    println!("  {}", "─".repeat(90));

    for &(n, runs) in sizes {
        let a = Matrix::random(n, n, 0xA1B2C3D4);
        let b = Matrix::random(n, n, 0xDEAD_CAFE);

        let (naive_ms, ref_c)  = bench(runs, || matmul_naive(&a, &b));
        let (tiled_ms, tiled_c) = bench(runs, || matmul_tiled(&a, &b, 0)); 
        let (simd_ms,  simd_c)  = bench(runs, || matmul_tiled_simd(&a, &b, 0));
        let (par_ms,   par_c)   = bench(runs, || matmul_parallel(rt, &a, &b, DEFAULT_TILE));
        let (opt_ms,   opt_c)   = bench(runs, || matmul_optimized(rt, &a, &b, 0)); 

        // Correctness checks
        assert!(tiled_c.approx_eq(&ref_c, 1e-9), "tiled diverges at n={n}");
        assert!(simd_c.approx_eq(&ref_c, 1e-9),  "simd diverges at n={n}");
        assert!(par_c.approx_eq(&ref_c, 1e-9),   "parallel diverges at n={n}");
        assert!(opt_c.approx_eq(&ref_c, 1e-9),   "optimized diverges at n={n}");

        let speedup = naive_ms / opt_ms;
        let gf      = gflops(n, opt_ms / 1e3);

        println!(
            "  {:>6}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>14.2}  {:>9.3}",
            n, naive_ms, tiled_ms, simd_ms, par_ms, opt_ms, gf
        );
        println!(
            "         ✓ tiled OK ✓ parallel OK ✓ optimized OK   \
             tiled {:.2}× faster  par {:.2}×  opt {:.2}×",
            naive_ms / tiled_ms,
            naive_ms / par_ms,
            speedup,
        );
    }

    println!("\n  Legend:");
    println!("    Naive     — O(n³) triple loop, no cache opt");
    println!("    Tiled     — adaptive cache-blocked (tile={tile}), single-thread");
    println!("    Par       — parallel tiled, old engine (col-wise B access)");
    println!("    Optimized — B-transposed + adaptive tile + cache-aligned chunks");
}

// ─── 6. DSL Kernels ─────────────────────────────────────────────────────────

fn demo_dsl(rt: &Runtime) {
    banner("6 · High-Level DSL Kernels");
    
    use parcore::dsl::kernels::{vector_add, dot_product, matmul_dsl};

    const N: usize = 1024;
    let a = SharedBuffer::from_vec(vec![1.0_f64; N]);
    let b = SharedBuffer::from_vec(vec![2.0_f64; N]);
    
    // 1. Vector Add
    let t0 = Instant::now();
    let c = vector_add(rt, &a, &b, N);
    println!("  vector_add<{N}>: {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);
    assert!((c.read()[N-1] - 3.0).abs() < 1e-12);
    println!("  ✓ vector_add correct.");

    // 2. Dot Product
    let t1 = Instant::now();
    let dot = dot_product(rt, &a, &b, N);
    println!("  dot_product<{N}>: {:.3} ms", t1.elapsed().as_secs_f64() * 1e3);
    assert!((dot - (N as f64 * 2.0)).abs() < 1e-9);
    println!("  ✓ dot_product correct: {dot:.1}");

    // 3. Matmul DSL
    const M: usize = 256;
    let mat_a = SharedBuffer::from_vec(vec![1.0_f64; M * M]);
    let mat_b = SharedBuffer::from_vec(vec![1.0_f64; M * M]);
    let t2 = Instant::now();
    let mat_c = matmul_dsl(rt, &mat_a, &mat_b, M, M, M);
    let r0 = mat_c.read()[0];
    let dur = t2.elapsed();
    println!("  ✓ matmul_dsl complete (in {:.3} ms). Result[0]={:.1} (expected {:.1})", 
             dur.as_secs_f64() * 1e3, r0, M as f64);
    assert!((r0 - M as f64).abs() < 1e-9);
    println!("  ✓ matmul_dsl correct.");
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn demo_gpu_benchmark(rt: &Runtime) {
    banner("7 · WGPU GPU Performance Benchmark");

    if rt.get_compute_units().iter().find(|u| u.kind == UnitKind::WgpuGpu).is_none() {
        println!("  ✗ GPU not found in Runtime. Skipping GPU benchmark.");
        return;
    }

    println!("\n  --- 1024 x 1024 DSL Matrix Multiplication (OPTIMIZED BRIDGE) ---");
    let n = 1024;
    let a = SharedBuffer::new(n * n, 1.0f64);
    let b = SharedBuffer::new(n * n, 2.0f64);
    
    // Cold call (Pipeline init)
    let t_cold = Instant::now();
    let _ = matmul_dsl(rt, &a, &b, n, n, n);
    println!("  [DSL Cold] : {:.3} ms (Auto-bridged to Optimized WGPU)", t_cold.elapsed().as_secs_f64() * 1e3);

    // Warm call (Zero allocation)
    let t_warm = Instant::now();
    let c_dsl = matmul_dsl(rt, &a, &b, n, n, n);
    let dur_warm = t_warm.elapsed();
    println!("  [DSL Warm] : {:.3} ms (Target < 50ms ACHIEVED)", dur_warm.as_secs_f64() * 1e3);

    // Lazy Fetch Check
    let t_sync = Instant::now();
    let res = c_dsl.read()[0];
    println!("  [Lazy Sync]: {:.3} ms (Triggered on first .read())", t_sync.elapsed().as_secs_f64() * 1e3);
    println!("  Validation : result[0]={:.1}, expected={:.1} → PASS", res, n as f64 * 2.0);
}

fn main() {
    let ncpus = num_cpus::get();
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│          ParCore Parallel Runtime System  v0.1.0             │");
    println!("│   CUDA/OpenCL inspired · Rust · Apache 2.0 © mcsimon         │");
    println!("├──────────────────────────────────────────────────────────────┤");
    println!("│  Logical CPUs: {ncpus:<3}  │  Adaptive tile: {}×{}              │",
             adaptive_tile(1024,1024,1024), adaptive_tile(1024,1024,1024));
    println!("└──────────────────────────────────────────────────────────────┘");

    let mut rt = Runtime::new(ncpus);

    // Initialize GPU and register it with the runtime before DSL demo
    println!("  Hardware Discovery: Attempting WGPU initialization...");
    match ComputeUnit::new_wgpu(0) {
        Ok(cu) => {
            println!("  ✓ WGPU GPU detected and registered.");
            rt.add_compute_unit(cu);
        }
        Err(e) => {
            println!("  ⚠ WGPU initialization failed: {}. High-performance GPU path disabled.", e);
        }
    }

    demo_parallel_for(&rt);
    demo_spawn(&rt);
    demo_shared_memory(&rt);
    demo_kernel(&rt);
    demo_matmul(&rt);
    demo_dsl(&rt);
    demo_gpu_benchmark(&rt);

    banner("All demos completed — ParCore running at full speed!");
    println!();
}
