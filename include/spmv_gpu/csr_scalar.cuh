// include/spmv_gpu/csr_scalar.cuh
#pragma once
#include "parser/mtx_parser.h"

/// GPU SpMV: y = A*x using CSR-Scalar (one CUDA thread per matrix row).
/// d_x must be a device pointer of size A.num_cols.
/// d_y must be a device pointer of size A.num_rows; it is overwritten
/// (not accumulated) — no pre-zeroing required.
/// Caller is responsible for allocation, H2D transfer of d_x, and D2H
/// transfer of d_y after the call.
void spmv_csr_scalar(const DeviceMatrix& A, const float* d_x, float* d_y);
