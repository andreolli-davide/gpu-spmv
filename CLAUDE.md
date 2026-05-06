# GPU-SPMV-2

Sparse Matrix-Vector Multiplication (SpMV) on GPU — research project.

## Overview

This project implements and benchmarks GPU kernels for sparse matrix-vector multiplication,
focused on CSR and related formats with CUDA.

## Project Structure

```
gpu-spmv-2/
├── src/                  # Source code
├── include/              # Headers
├── data/                 # Matrix market files, test data
├── scripts/              # Build and run scripts
├── build/                # CMake build output (gitignored)
├── outputs/              # Test outputs and logs (gitignored)
├── docs/                 # Project documentation
├── CLAUDE.md             # This file
└── AGENTS.md             # Agent instructions
```

## Building

```bash
# Create build directory and configure
mkdir -p build && cd build
cmake ..

# Compile
cmake --build .
```

## Cache Behavior Verification

Use **Valgrind/Cachegrind** to analyze CPU cache performance of host code (memory accesses, cache misses, branch prediction):

```bash
# Run cachegrind on a CPU-bound program
valgrind --tool=cachegrind ./build/src/<target> [args]

# View results (cached from previous run)
cg_annotate cachegrind.out.<pid>

# Compare two runs
cg_diff vgout1 vgout2
```

**What to check:**
- **D1 miss rate**: Data cache L1 misses — high misses indicate poor spatial locality
- **LL miss rate**: Last-level cache misses — indicates working set too large for L3
- **I1 miss rate**: Instruction cache — usually a problem with tight loops

**Typical SpMV concerns:**
- Row-index traversal: sequential → good L1 cache utilization
- Value access: strided → depends on row width
- Re-using row offsets within a thread block

**Note:** Cachegrind simulates cache, not GPU. GPU memory access patterns are analyzed with `cuda-memcheck` or `compute-sanitizer`.

See AGENTS.md for remote execution instructions.