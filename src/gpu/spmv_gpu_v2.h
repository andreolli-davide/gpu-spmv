// =============================================================================
// spmv_gpu_v2.h
// =============================================================================
// GPU SpMV v2 — Shared memory tiling for input vector x (Bell & Garland '09)
//
// This is the second GPU implementation of Sparse Matrix-Vector Multiplication
// (SpMV). It improves upon v1 by caching the input vector x in shared memory.
//
// Key Optimizations over v1
// -----------------------
//   - Shared memory tiling: each block loads contiguous x chunk into shared mem
//   - Lower latency: shared memory ~1-10ns vs global memory ~100-400ns
//   - x accessed from shared memory when in loaded range, __ldg fallback otherwise
//
// Bell & Garland '09 Reference:
//   "Hot data is cached in shared memory to reduce global memory traffic:
//    shared_mem[i] = values[i] (loaded per warp)"
//   "Shared memory latency: ~1–10 ns vs. global memory: ~100–400 ns"
//
// Shared Memory Configuration:
//   - Default: 32 KB (4096 double elements)
//   - Configurable via SHARED_MEM_SIZE compile-time macro
//   - Ampere limit: 48 KB per block (49152 bytes)
//
// Thread Mapping:
//   thread_id = block_idx * block_dim + thread_idx (one thread per row)
//
// Performance Expectations:
//   Most benefit when column indices have locality (contiguous access patterns).
//   Falls back to __ldg for x elements outside shared memory range.
//
// Verification:
//   Same as v1 — use --verify flag and infinity-norm < 1e-10 tolerance.
//
// Persistent Buffer Support:
//   For repeated SpMV calls with the same matrix, see gpu_persistent_buffers.h
//   and spmv_gpu_v2_persistent() to eliminate per-call cudaMalloc/cudaFree overhead.
//
// =============================================================================

#ifndef SPMV_GPU_V2_H
#define SPMV_GPU_V2_H

#include <cstdint>  // int64_t

#include "sparse_matrix.h"  // SparseMatrix, DenseVector
#include "timer.h"          // GPUTimer

namespace spmv {

// =============================================================================
// spmv_gpu_v2_kernel — Shared memory tiled CSR SpMV kernel (exposed for testing)
// =============================================================================
// This kernel is exposed so benchmarks can measure kernel time directly,
// excluding memory allocation and transfer overhead.
//
// Template parameter SHARED_ELEMENTS: number of double elements in shared memory
// Must be consistent with shared memory size passed to kernel launch.
//
// Kernel configuration:
//   - BLOCK_DIM = 256 threads per block
//   - One thread per matrix row
//   - Shared memory size: SHARED_ELEMENTS * sizeof(double)
//
// @param d_values    CSR values array (size nnz)
// @param d_col_index CSR column index array (size nnz)
// @param d_row_ptr   CSR row pointer array (size rows+1)
// @param d_x         Input vector (size cols)
// @param d_y         Output vector (size rows)
// @param rows        Number of matrix rows
//
template <int SHARED_ELEMENTS>
__global__ void spmv_gpu_v2_kernel(const double* __restrict__ d_values,
                                   const int64_t* __restrict__ d_col_index,
                                   const int64_t* __restrict__ d_row_ptr,
                                   const double* __restrict__ d_x,
                                   double* __restrict__ d_y,
                                   int64_t rows);

// =============================================================================
// spmv_gpu_v2 — GPU SpMV wrapper with shared memory tiling for x
// =============================================================================
// Top-level wrapper for GPU SpMV v2 with shared memory tiling of input vector.
//
// This function handles:
//   1. GPU memory allocation (cudaMalloc)
//   2. Host-to-device transfer (cudaMemcpy H2D)
//   3. Kernel launch with shared memory configuration
//   4. Device-to-host transfer (cudaMemcpy D2H)
//   5. GPU memory deallocation
//
// Shared Memory Details:
//   - Each block loads x[block_id * SHARED_ELEMENTS : (block_id+1) * SHARED_ELEMENTS)
//   - SHARED_ELEMENTS = SHARED_MEM_SIZE / sizeof(double) = 4096 (for 32KB default)
//   - Threads access x from shared memory when col_index in range, else __ldg
//
// @param A   Input matrix in CSR format, size rows × cols
// @param x   Input dense vector, size cols
// @param[out] y Output dense vector, size rows. Resized automatically.
//
// =============================================================================
void spmv_gpu_v2(const SparseMatrix& A, const DenseVector& x, DenseVector& y);

// =============================================================================
// spmv_gpu_v2_custom_smem — Wrapper with custom shared memory size
// =============================================================================
// Variant that allows specifying shared memory size at runtime.
//
// @param A             Input matrix in CSR format
// @param x             Input dense vector
// @param[out] y       Output dense vector (resized automatically)
// @param shared_kb    Shared memory size in KB (16-48, clamped to Ampere limit)
//
// Note: Actual shared memory size is instantiation-based (16KB, 32KB, or 48KB).
//       The closest upper bound is used if shared_kb doesn't exactly match.
//
// =============================================================================
void spmv_gpu_v2_custom_smem(const SparseMatrix& A, const DenseVector& x,
                             DenseVector& y, int shared_kb);

// =============================================================================
// spmv_gpu_v2_autotuned — Auto-tuned block size SpMV
// =============================================================================
// Wrapper that automatically selects optimal block size based on matrix sparsity.
//
// Uses auto_select_block_size() to choose between 128/256/512 threads per block,
// then delegates to spmv_gpu_v2_custom_smem() for the actual computation.
//
// @param A   Input matrix in CSR format
// @param x   Input dense vector
// @param y   Output dense vector (resized automatically)
//
// =============================================================================
void spmv_gpu_v2_autotuned(const SparseMatrix& A, const DenseVector& x, DenseVector& y);

} // namespace spmv

#endif // SPMV_GPU_V2_H