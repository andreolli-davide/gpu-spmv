// =============================================================================
// spmv_ell.h
// =============================================================================
// GPU SpMV using ELLPACK format — fixed-length row representation
//
// ELLPACK pads all rows to the same length (max_row_length), enabling:
//   - Coalesced memory access: consecutive threads access consecutive memory
//   - No row-boundary checks: fixed-length inner loop
//   - Simple indexing: row_offset = row * max_row_length
//
// Thread Mapping:
//   One thread per matrix row. Each thread iterates exactly max_row_length
//   times (skipping padding entries where col_index == -1).
//
// Performance Characteristics:
//   - Best for: Regular matrices with similar row lengths (FEM, structured grids)
//   - Warning: Can be wasteful for highly irregular matrices (e.g., webbase)
//   - Template unrolling (#pragma unroll) enables efficient fixed-length loops
//
// Reference: Bell & Garland '09 "Efficient Sparse Matrix-Vector Multiplication on GPUs"
//
// =============================================================================

#ifndef SPMV_ELL_H
#define SPMV_ELL_H

#include <cstdint>

#include "../common/sparse_matrix.h"

namespace spmv {

template <int MAX_ROW_LENGTH>
__global__ void spmv_ell_kernel(const double* __restrict__ d_values,
                                 const int64_t* __restrict__ d_col_index,
                                 const double* __restrict__ d_x,
                                 double* __restrict__ d_y,
                                 int64_t rows);

void spmv_ell(const ELL_SparseMatrix& A, const DenseVector& x, DenseVector& y);

} // namespace spmv

#endif // SPMV_ELL_H