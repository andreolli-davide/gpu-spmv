// =============================================================================
// sparse_matrix.cpp
// =============================================================================
// Implementations of SparseMatrix, COO_SparseMatrix, and DenseVector.
//
// See sparse_matrix.h for the design rationale behind each structure.
// =============================================================================

#include "sparse_matrix.h"

namespace spmv {

// =============================================================================
// SparseMatrix (CSR)
// =============================================================================

SparseMatrix::SparseMatrix(int64_t r, int64_t c, int64_t n)
    : rows(r), cols(c), nnz(n) {}

void SparseMatrix::allocate() {
    // CSR layout:
    //   values   — one entry per non-zero
    //   col_index — one column index per non-zero
    //   row_ptr  — one entry per row PLUS one sentinel at the end (rows + 1)
    values.resize(nnz);
    col_index.resize(nnz);
    row_ptr.resize(rows + 1);
}

int64_t SparseMatrix::memory_bytes() const {
    // clang-format off
    return nnz        * sizeof(double)       // values
         + nnz        * sizeof(int64_t)     // col_index
         + (rows + 1) * sizeof(int64_t)    // row_ptr (rows + 1 entries)
         + 3          * sizeof(int64_t);    // rows, cols, nnz metadata
    // clang-format on
}

// =============================================================================
// COO_SparseMatrix (COO)
// =============================================================================

COO_SparseMatrix::COO_SparseMatrix(int64_t r, int64_t c, int64_t n)
    : rows(r), cols(c), nnz(n) {}

void COO_SparseMatrix::allocate() {
    // COO layout — three parallel arrays, each of length nnz:
    values.resize(nnz);
    row.resize(nnz);
    col.resize(nnz);
}

int64_t COO_SparseMatrix::memory_bytes() const {
    // clang-format off
    return nnz * sizeof(double)     // values
         + nnz * sizeof(int64_t)    // row
         + nnz * sizeof(int64_t)    // col
         + 3  * sizeof(int64_t);    // rows, cols, nnz metadata
    // clang-format on
}

// =============================================================================
// DenseVector
// =============================================================================

DenseVector::DenseVector(int64_t s) : size(s) {
    data.resize(size);
}

void DenseVector::resize(int64_t s) {
    size = s;
    data.resize(size);
}

int64_t DenseVector::memory_bytes() const {
    return size * sizeof(double) + sizeof(int64_t);
}

} // namespace spmv
