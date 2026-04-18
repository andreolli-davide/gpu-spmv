// =============================================================================
// spmv_gpu_v1.h
// =============================================================================
// GPU SpMV v1 — straightforward row-parallel CSR kernel.
//
// This is the first GPU implementation of Sparse Matrix-Vector Multiplication
// (SpMV).  It uses a naively parallel approach: one GPU thread per matrix row,
// reading the corresponding slice of the CSR data structure.
//
// Performance Expectations
// -----------------------
// This version is intentionally simple to serve as a baseline.  It will likely
// underperform for matrices with highly variable row lengths (load imbalance)
// and for matrices with very short rows (low arithmetic intensity).  Later
// versions (v2, v3) address these shortcomings with shared memory, tiling,
// and format changes (ELL, HYB).
//
// Memory Access Pattern
// --------------------
// Each thread reads x[col_index[j]] for every non-zero in its row.  With CSR's
// row-contiguous layout, accesses to the values and col_index arrays are
// sequential and fully coalesced.  Accesses to x are gather operations (indirect)
// and depend on the column index distribution.
//
// Verification
// ----------
// The --verify flag (when implemented in the driver) compares GPU output
// against the CPU serial implementation using infinity-norm < 1e-10 tolerance.
// Floating-point rounding differences between CPU and GPU may cause small
// deviations; the tolerance accounts for these.
//
// =============================================================================

#ifndef SPMV_GPU_V1_H
#define SPMV_GPU_V1_H

#include <cstdint>  // int64_t

#include "sparse_matrix.h"  // SparseMatrix, DenseVector
#include "timer.h"          // GPUTimer

namespace spmv {

// =============================================================================
// spmv_gpu_v1 — GPU SpMV wrapper (Phase 1 placeholder)
// =============================================================================
// Top-level wrapper for GPU SpMV v1.  This function handles:
//
//   1. GPU memory allocation (cudaMalloc)
//   2. Host-to-device transfer (cudaMemcpy H2D)
//   3. Kernel launch (calls the actual GPU kernel)
//   4. Device-to-host transfer (cudaMemcpy D2H)
//   5. GPU memory deallocation
//   6. Timing instrumentation via GPUTimer
//
// Usage with --verify:
//   When --verify is enabled, the caller is responsible for comparing the
//   resulting y vector against a CPU reference (e.g., spmv_cpu_serial) and
//   checking that |y_gpu - y_cpu|_inf < 1e-10.
//
// @param A   Input matrix in CSR format, size rows × cols
// @param x   Input dense vector, size cols
// @param[out] y  Output dense vector, size rows.  Resized automatically.
//
// @note The actual GPU kernel is implemented in spmv_gpu_v1.cu.
//       This header only provides the host-side wrapper interface.
//
// =============================================================================
void spmv_gpu_v1(const SparseMatrix& A, const DenseVector& x, DenseVector& y);

} // namespace spmv

#endif // SPMV_GPU_V1_H
