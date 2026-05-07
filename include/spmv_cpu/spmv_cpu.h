#pragma once
#include "parser/mtx_parser.h"

/// y = A * x using CSR layout.
/// Caller allocates x (size A.num_cols) and y (size A.num_rows).
void spmv_csr_cpu(const MtxCsr& A, const float* x, float* y);

/// Compare two length-n vectors element-wise using absolute difference.
/// Prints each element whose |a[i] - b[i]| exceeds tolerance.
/// Returns the maximum absolute difference found.
float compare_vectors(const float* a, const float* b, int n, float tolerance = 1e-5f);
