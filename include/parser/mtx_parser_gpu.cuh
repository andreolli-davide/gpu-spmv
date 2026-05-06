#pragma once

#include "parser/mtx_parser.h"

/// GPU SpMV kernel — one thread per matrix row.
///
/// Launch via spmv_gpu_kernel() which selects block_size=256 and computes
/// the block count needed to cover all rows.  The kernel is a simple CSR
/// SpMV: each thread processes one row by iterating over its nonzero range
/// and accumulating y[row] = sum(values[idx] * x[col_indices[idx]]).
///
/// \note Results may differ from the CPU reference by ~1e-6 due to
///       floating-point non-associativity; a tolerance of 1e-6 is used
///       in correctness tests.
void spmv_gpu_kernel(const DeviceMatrix& csr, const double* d_x, double* d_y);
