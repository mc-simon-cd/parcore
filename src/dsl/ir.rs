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

//! # Kernel Intermediate Representation (IR)
//!
//! Every kernel compiled by the ParCore DSL is first converted into a
//! `KernelIR` — a structured, backend-agnostic description of the computation.
//!
//! ## Why an IR?
//!
//! A textual macro expansion alone is fragile and not extensible. The IR:
//!
//! - Separates *what* the kernel does from *how* it runs
//! - Enables future backends (GPU via wgpu, NPU, etc.) to compile from
//!   the same structure without touching user code
//! - Enables optimisation passes (kernel fusion, loop tiling detection,
//!   auto-tuning) to query and transform metadata before dispatch
//!
//! ## Compilation Pipeline
//!
//! ```text
//! parcore_kernel! { … }         ← Developer writes this
//!        │
//!        ▼ macro expansion
//! ClosureKernel<F>               ← Zero-cost Rust struct, implements Kernel
//!        │
//!        ▼ KernelLaunchConfig::build()
//! KernelIR {                     ← Structured IR
//!     name, params, grid, block,
//!     opt_hints, backend
//! }
//!        │
//!        ▼ Backend::compile(ir)
//! Scheduled tasks                ← e.g. pool.submit(…) per chunk
//!        │
//!        ▼
//! Runtime execution              ← thread pool, future: GPU queue
//! ```
//!
//! ## Future Backend Hooks
//!
//! ```text
//! Box<dyn BackendCompiler>
//! ├── CpuBackend   → parallel_for (current)
//! ├── WgpuBackend  → wgpu compute shader (future)
//! └── NpuBackend   → custom NPU queue (future)
//! ```

use super::context::Dim3;

// ---------------------------------------------------------------------------
// Memory space
// ---------------------------------------------------------------------------

/// Where a kernel parameter lives in the memory hierarchy.
///
/// | Variant    | Analogy             | Scope        | Cached? |
/// |------------|---------------------|--------------|---------|
/// | `Global`   | CUDA global mem     | All threads  | L2/L3   |
/// | `Local`    | CUDA shared mem     | Block only   | SRAM    |
/// | `Register` | CUDA register       | Single thread| —       |
/// | `Constant` | CUDA constant mem   | All threads  | L1      |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    /// Visible to all work-items; backed by `SharedBuffer<T>`.
    Global,
    /// Shared within a workgroup (block); backed by thread-local `Vec<T>`.
    Local,
    /// Per-work-item variable; lives in a CPU register / stack slot.
    Register,
    /// Read-only shared constant; never written by kernels.
    Constant,
}

// ---------------------------------------------------------------------------
// Parameter descriptor
// ---------------------------------------------------------------------------

/// Describes one parameter of a kernel in the IR.
///
/// This is used by backends and the auto-tuner to reason about access
/// patterns, memory pressure, and data dependencies.
#[derive(Debug, Clone)]
pub struct ParamDecl {
    /// Human-readable name (from the kernel definition).
    pub name: &'static str,
    /// Memory space this parameter lives in.
    pub mem_space: MemorySpace,
    /// Optional hint about element type (for future code-gen).
    pub type_hint: &'static str,
    /// Access mode.
    pub access: AccessMode,
}

/// How a kernel parameter is accessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Read-only input.
    ReadOnly,
    /// Write-only output.
    WriteOnly,
    /// Read-write (both input and output).
    ReadWrite,
}

// ---------------------------------------------------------------------------
// Optimisation hints
// ---------------------------------------------------------------------------

/// Hints the DSL compiler / auto-tuner can derive from kernel structure.
///
/// These travel with the IR and inform the backend scheduler.
#[derive(Debug, Clone)]
pub struct OptHints {
    /// Suggested cache-blocking tile size (0 = use adaptive_tile).
    pub tile_size: usize,
    /// Whether the inner reduction loop is vectorisable.
    pub vectorize: bool,
    /// Whether `B` should be transposed before launch (matmul pattern).
    pub transpose_input: bool,
    /// Kernel fusion group ID (0 = standalone, ≥1 = fuse with same ID).
    pub fusion_group: usize,
    /// If `true`, the scheduler will auto-tune chunk/tile params.
    pub auto_tune: bool,
    /// Minimum rows per parallel chunk (0 = runtime default).
    pub min_chunk_rows: usize,
}

impl Default for OptHints {
    fn default() -> Self {
        Self {
            tile_size: 0,        // 0 → adaptive_tile() at runtime
            vectorize: true,     // optimistic: assume vectorisable
            transpose_input: false,
            fusion_group: 0,
            auto_tune: false,
            min_chunk_rows: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Backend abstraction
// ---------------------------------------------------------------------------

/// Tag identifying which hardware backend will execute this kernel.
///
/// Future backends only need to implement their own dispatch path;
/// the IR and DSL layer stay unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CPU thread pool (current — always available).
    Cpu,
    /// wgpu-based GPU compute shader (future).
    WgpuGpu,
    /// Custom NPU queue with batch accumulation (future).
    Npu,
    /// Automatic selection based on workload size.
    Auto,
}

impl Default for Backend { fn default() -> Self { Backend::Cpu } }

// ---------------------------------------------------------------------------
// Kernel IR
// ---------------------------------------------------------------------------

/// The structured intermediate representation of a ParCore kernel.
///
/// Created by [`KernelLaunchConfig::build_ir`] and consumed by dispatch.
///
/// ## Extending with new backends
///
/// 1. Add a variant to [`Backend`].
/// 2. Implement dispatch in `launch.rs` by matching on `ir.backend`.
/// 3. No changes to user-facing DSL or macros required.
#[derive(Debug, Clone)]
pub struct KernelIR {
    /// Human-readable kernel name (used in traces and error messages).
    pub name: String,
    /// Declared parameters.
    pub params: Vec<ParamDecl>,
    /// Grid dimensions (total work-item count per axis).
    pub grid: Dim3,
    /// Block / workgroup dimensions.
    pub block: Dim3,
    /// Optimisation hints derived by the DSL compiler.
    pub opt_hints: OptHints,
    /// Target execution backend.
    pub backend: Backend,
}

impl KernelIR {
    /// Total number of work-items to execute.
    pub fn total_work_items(&self) -> usize {
        self.grid.volume()
    }

    /// Derive the number of groups (blocks) in this launch.
    pub fn num_groups(&self) -> usize {
        let bv = self.block.volume().max(1);
        (self.total_work_items() + bv - 1) / bv
    }

    /// Return a human-readable summary for logging/debug.
    pub fn summary(&self) -> String {
        format!(
            "KernelIR[{}] grid={} block={} items={} backend={:?} tile={} vectorize={}",
            self.name,
            self.grid,
            self.block,
            self.total_work_items(),
            self.backend,
            if self.opt_hints.tile_size == 0 { "adaptive".to_string() }
            else { self.opt_hints.tile_size.to_string() },
            self.opt_hints.vectorize,
        )
    }
}

// ---------------------------------------------------------------------------
// Lazy execution graph node (future)
// ---------------------------------------------------------------------------

/// A node in the lazy kernel execution graph.
///
/// When `lazy_graph` feature is enabled, kernels are not dispatched
/// immediately but queued here. The graph is then analysed for fusion,
/// reordering, and dead-kernel elimination before execution.
///
/// Currently a placeholder — not connected to the runtime.
#[derive(Debug)]
pub struct GraphNode {
    /// The IR of the kernel this node represents.
    pub ir: KernelIR,
    /// Indices of graph nodes this node depends on (data dependencies).
    pub depends_on: Vec<usize>,
    /// Whether this node has been dispatched to the runtime.
    pub executed: bool,
}

impl GraphNode {
    pub fn new(ir: KernelIR) -> Self {
        Self { ir, depends_on: vec![], executed: false }
    }
}

/// A lazy execution graph — collect kernels, then flush() to run all.
///
/// This enables:
/// - **Kernel fusion**: consecutive kernels with no data dependency
///   may be merged into a single pass (halves memory bandwidth cost).
/// - **Dead kernel elimination**: if a kernel's output is unused,
///   skip it entirely.
/// - **Reordering**: schedule independent kernels for better data locality.
#[derive(Debug, Default)]
pub struct ExecutionGraph {
    pub nodes: Vec<GraphNode>,
}

impl ExecutionGraph {
    pub fn push(&mut self, ir: KernelIR) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::new(ir));
        idx
    }

    pub fn depends(&mut self, consumer: usize, producer: usize) {
        self.nodes[consumer].depends_on.push(producer);
    }

    /// Analyse the graph and return a topologically sorted execution order.
    /// (Simple Kahn's algorithm — O(V+E))
    pub fn topological_order(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_deg = vec![0usize; n];
        let mut successors = vec![Vec::new(); n];
        
        for (i, node) in self.nodes.iter().enumerate() {
            in_deg[i] = node.depends_on.len();
            for &dep in &node.depends_on {
                // dep is the producer (predecessor), i is the consumer (successor)
                successors[dep].push(i);
            }
        }
        
        let mut queue: std::collections::VecDeque<usize> =
            (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        
        while let Some(i) = queue.pop_front() {
            order.push(i);
            for &succ in &successors[i] {
                in_deg[succ] -= 1;
                if in_deg[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }
        order
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_total_work_items() {
        let ir = KernelIR {
            name: "test".to_owned(),
            params: vec![],
            grid: Dim3::new2(16, 8),
            block: Dim3::new2(4, 4),
            opt_hints: OptHints::default(),
            backend: Backend::Cpu,
        };
        assert_eq!(ir.total_work_items(), 128);
        assert_eq!(ir.num_groups(), 8);
    }

    #[test]
    fn execution_graph_topo_order() {
        let mut g = ExecutionGraph::default();
        let a = g.push(KernelIR { name: "a".to_owned(), params: vec![], grid: Dim3::new1(1),
            block: Dim3::new1(1), opt_hints: OptHints::default(), backend: Backend::Cpu });
        let b = g.push(KernelIR { name: "b".to_owned(), params: vec![], grid: Dim3::new1(1),
            block: Dim3::new1(1), opt_hints: OptHints::default(), backend: Backend::Cpu });
        g.depends(b, a); // b depends on a
        let order = g.topological_order();
        // a must come before b
        let pa = order.iter().position(|&x| x == a).unwrap();
        let pb = order.iter().position(|&x| x == b).unwrap();
        assert!(pa < pb, "a must precede b in topo order");
    }
}
