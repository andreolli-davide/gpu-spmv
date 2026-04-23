// =============================================================================
// spmv_gpu.h
// =============================================================================
// GPU SpMV kernel declarations — basic CSR parallelization.
//
// SpMV Overview: y = A · x
// -------------------------------
// Sparse Matrix-Vector Multiplication computes y_i = Σ_j A_ij · x_j
// over only the non-zero entries of A.  In CSR format this becomes:
//
//   for each row i:
//       y[i] = sum_{j=row_ptr[i]}^{row_ptr[i+1]-1} values[j] * x[col_index[j]]
//
// This is the fundamental operation in iterative linear solvers (Conjugate
// Gradient, GMRES) and eigenvalue methods (Power Iteration).
//
// GPU Implementation Strategy
// ---------------------------
// spmv_gpu_v1() is the baseline GPU implementation — a straightforward
// parallelization that assigns one thread per row.  Each thread iterates
// over the non-zero entries in its row and accumulates the dot product.
//
// This kernel serves as the foundation for more sophisticated GPU
// implementations that explore shared memory, texture caching, and
// custom sparse formats.
// =============================================================================

#ifndef SPMV_GPU_H
#define SPMV_GPU_H

#include "sparse_matrix.h"

namespace spmv {

// =============================================================================
// spmv_gpu_v1 — Baseline GPU CSR SpMV
// =============================================================================
// Straightforward GPU parallelization: one thread per row, each thread
// iterates over its row's non-zero entries and accumulates into y[i].
//
// Memory Access Pattern:
//   • Each thread reads its row's column indices and values coalesced
//   • Random access into x (input vector) is scattered across threads
//   • Results are written back to y with some write-combining benefit
//
// @param A   Input matrix in CSR format, size rows × cols
// @param x   Input dense vector, size cols
// @param[out] y  Output dense vector, size rows.  Resized automatically.
// @return  true on success, false on error (e.g., CUDA runtime failure)
// =============================================================================
bool spmv_gpu_v1(const SparseMatrix& A, const DenseVector& x, DenseVector& y);

} // namespace spmv

#endif // SPMV_GPU_H