#pragma once

#include "sparse_matrix.h"
#include <vector>

namespace spmv {

// --------------------------------------------------------------------------
// Compute y = A*x using CSR format
// Sequential (single-threaded) baseline for correctness validation
// --------------------------------------------------------------------------
void spmv_cpu_serial(const SparseMatrix& A, const DenseVector& x, DenseVector& y);

// --------------------------------------------------------------------------
// Compute y = A*x using CSR format
// Parallel (OpenMP) baseline — same algorithmic result as serial
// --------------------------------------------------------------------------
void spmv_cpu_omp(const SparseMatrix& A, const DenseVector& x, DenseVector& y);

// --------------------------------------------------------------------------
// Fill a vector with zeros
// --------------------------------------------------------------------------
void fill_zero(DenseVector& v);

// --------------------------------------------------------------------------
// Fill a vector with a constant value
// --------------------------------------------------------------------------
void fill_constant(DenseVector& v, double val);

// --------------------------------------------------------------------------
// Compute L-infinity norm of (a - b)
// --------------------------------------------------------------------------
double infnorm(const DenseVector& a, const DenseVector& b);

} // namespace spmv
