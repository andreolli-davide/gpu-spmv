# SpMV GPU Optimization - GPU Computing Course Deliverable 1

## Project Overview

This repository contains a comprehensive investigation of **Sparse Matrix-Vector Multiplication (SpMV)** implementations on **NVIDIA GPU Ampere architecture**. The goal is to understand performance characteristics across different sparse matrix formats, develop optimized GPU kernels, and analyze how matrix structure influences computational efficiency.

### Course Context
- **Course**: GPU Computing 2025-2026
- **Deliverable**: Deliverable 1
- **Focus**: Single-GPU SpMV research with performance interpretation and algorithmic analysis

## Objectives

1. **Develop SpMV implementations** across multiple sparse formats (CSR, COO, ELL, HYB, JDS, etc.)
2. **Compare performance** using a diverse dataset of 10 SuiteSparse matrices
3. **Analyze results** to explain why certain formats outperform others under specific conditions
4. **Validate correctness** against a CPU reference implementation
5. **Interpret findings** in context of GPU memory hierarchy, load balancing, and matrix properties

## Key Components

### Code Artifacts
- **CPU Baseline**: Sequential/OpenMP reference implementation with Matrix Market (.mtx) file parsing
- **GPU Implementations**:
  - Version 1: Straightforward parallelization (basic CSR-based row-level kernel)
  - Version 2: Optimized kernel with shared memory, load balancing improvements
  - (Optional) Multiple sparse format implementations for comparison

### Experimental Methodology
- **Dataset**: 10 SuiteSparse matrices with documented diversity across:
  - Matrix dimensions (rows × cols)
  - Nonzero count (nnz)
  - Row-length regularity and variability
  - Structural patterns (dense, structured, irregular)
  
- **Measurements**:
  - Kernel execution time (excluding setup/I/O overhead)
  - Throughput (GFLOP/s) using formula: 2·nnz operations per SpMV
  - Multiple runs with statistics (average, variability, median)
  - Optional memory profiling (cache misses, global memory access patterns)

- **Validation**:
  - GPU numerical correctness verified against CPU baseline
  - Tolerance-based comparison for floating-point arithmetic
  - Edge case testing (dense rows, irregular sparsity patterns)

### Analysis & Insights
The investigation answers key questions:
- **Why do certain formats outperform others?** Connection to load balancing, memory coalescing, and matrix structure
- **How does row-length regularity affect performance?** Impact on SM utilization and occupancy
- **What are the memory-bound characteristics?** Global memory access patterns, cache behavior, storage overhead
- **Which optimizations matter most on Ampere?** Shared memory usage, warp-level scheduling, occupancy tuning

## Key References

### Required
- Gao, J., Liu, B., Ji, W. and Huang, H., 2024. *A systematic literature survey of sparse matrix-vector multiplication*. arXiv:2404.06047.
- Chu, G., et al. *Efficient Algorithm Design of Optimizing SpMV on GPU*. HPDC '23, 2023.

### Classical Papers
- Bell & Garland (SC '09): Efficient Sparse-Matrix–Vector Multiplication on GPUs
- Greathouse & Daga (SC14): CSR-Adaptive and Structure-Aware GPU SpMV
- Ashari et al. (SC14): Irregular Matrices and Graph-Oriented SpMV

### Modern References
- Liu & Vinter (ICS '15): CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication
- Merrill & Garland (SC16): Merge-based Parallel Sparse Matrix-Vector Multiplication
- Niu et al. (IPDPS 2021): Tiled SpMV for Local Structure Exploitation
