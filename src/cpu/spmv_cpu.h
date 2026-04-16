// =============================================================================
// spmv_cpu.h
// =============================================================================
// CPU SpMV baseline implementations — sequential and OpenMP parallel.
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
// Why Two Implementations?
// ------------------------
// spmv_cpu_serial() is the reference — single-threaded, easy to verify,
// and free of any parallelization artifacts.  All correctness tests use it
// as the "golden" output.
//
// spmv_cpu_omp() is the CPU performance baseline — identical arithmetic,
// same number of memory loads and floating-point operations, but running
// in parallel over rows.  It exists so that we can measure the overhead of
// GPU kernel launch and PCIe transfer against a well-tuned CPU implementation.
//
// IMPORTANT: both implementations produce BITWISE identical results when run
// on the same input (no non-determinism from thread scheduling).  This
// is critical for the correctness test.
//
// Numerical Considerations
// ------------------------
// IEEE-754 guarantees that floating-point addition is deterministic and
// associative enough for this use case — both implementations accumulate
// in the same order (row-major, left-to-right within a row) so their
// outputs are guaranteed to be bitwise identical.  This is NOT true for
// the OpenMP parallel version if you used a reduction clause (the order
// of accumulation across threads is undefined), which is why we use a
// threadprivate scalar accumulator per row instead.
// =============================================================================

#ifndef SPMV_CPU_H
#define SPMV_CPU_H

#include "sparse_matrix.h"

namespace spmv {

// =============================================================================
// spmv_cpu_serial — Sequential (single-threaded) CSR SpMV
// =============================================================================
// The canonical reference implementation.  Runs on one CPU core.
// Used as the golden reference for all correctness tests.
//
// Performance: O(nnz) memory bandwidth-bound operations.
// Typical throughput: 5–15 GB/s on a modern CPU (DDR4 or DDR5).
//
// @param A   Input matrix in CSR format, size rows × cols
// @param x   Input dense vector, size cols
// @param[out] y  Output dense vector, size rows.  Resized automatically.
// =============================================================================
void spmv_cpu_serial(const SparseMatrix& A, const DenseVector& x, DenseVector& y);

// =============================================================================
// spmv_cpu_omp — OpenMP-parallel CSR SpMV
// =============================================================================
// Parallel row-wise SpMV using OpenMP.  One thread per row (or a contiguous
// chunk of rows for very large matrices).
//
// Why static schedule?
//   The amount of work per row (row_ptr[i+1] - row_ptr[i]) varies — but
//   for the row-length distributions typical of sparse matrices, this
//   variation averages out over many rows.  Static scheduling avoids the
//   overhead of dynamic load balancing (which can dominate for simple kernels).
//
// Why no reduction clause?
//   The reduction clause accumulates into an unspecified order across threads,
//   which WOULD produce different bit-patterns than the serial version due
//   to IEEE-754 rounding differences.  By using a private scalar accumulator
//   per thread and writing the result atomically only at the end of each row,
//   we guarantee bitwise equality with the serial version.
//
// @param A   Input matrix in CSR format, size rows × cols
// @param x   Input dense vector, size cols
// @param[out] y  Output dense vector, size rows.  Resized automatically.
// =============================================================================
void spmv_cpu_omp(const SparseMatrix& A, const DenseVector& x, DenseVector& y);

// =============================================================================
// fill_zero — sets all entries of a vector to 0.0
// =============================================================================
void fill_zero(DenseVector& v);

// =============================================================================
// fill_constant — sets all entries of a vector to a given value
// =============================================================================
// @param v    Vector to fill
// @param val  Value to write into every element
// =============================================================================
void fill_constant(DenseVector& v, double val);

// =============================================================================
// infnorm — L-infinity norm of (a - b)
// =============================================================================
// Returns max_i |a_i - b_i|.  Used to compare two SpMV outputs for correctness.
//
// Tolerance: we consider two results "equal" if the L-inf error is below
// 1e-15 for double-precision arithmetic.  This accounts for rounding
// differences in accumulation order while catching real bugs.
//
// @param a  First vector (size N)
// @param b  Second vector (size N); must have same size as a
// @return   max |a_i - b_i| over all i
// =============================================================================
double infnorm(const DenseVector& a, const DenseVector& b);

} // namespace spmv

#endif // SPMV_CPU_H
