// =============================================================================
// sparse_matrix.cpp
// =============================================================================
// Implementations of SparseMatrix, COO_SparseMatrix, and DenseVector.
//
// See sparse_matrix.h for the design rationale behind each structure.
// =============================================================================

#include "sparse_matrix.h"
#include <algorithm>

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

// =============================================================================
// ELL_SparseMatrix (ELLPACK)
// =============================================================================

ELL_SparseMatrix::ELL_SparseMatrix(int64_t r, int64_t c, int64_t max_len)
    : rows(r), cols(c), max_row_length(max_len) {}

void ELL_SparseMatrix::allocate() {
    const int64_t total = rows * max_row_length;
    values.resize(total);
    col_index.resize(total, -1);
}

int64_t ELL_SparseMatrix::memory_bytes() const {
    return (values.size() * sizeof(double)) + (col_index.size() * sizeof(int64_t));
}

ELL_SparseMatrix csr_to_ell(const SparseMatrix& csr) {
    int64_t max_len = 0;
    for (int64_t i = 0; i < csr.rows; ++i) {
        const int64_t len = csr.row_ptr[i + 1] - csr.row_ptr[i];
        max_len = std::max(max_len, len);
    }

    ELL_SparseMatrix ell(csr.rows, csr.cols, max_len);
    ell.nnz = csr.nnz;
    ell.allocate();

    for (int64_t i = 0; i < csr.rows; ++i) {
        const int64_t row_start = csr.row_ptr[i];
        const int64_t row_len = csr.row_ptr[i + 1] - row_start;
        const int64_t ell_row_start = i * max_len;

        for (int64_t j = 0; j < max_len; ++j) {
            if (j < row_len) {
                ell.values[ell_row_start + j] = csr.values[row_start + j];
                ell.col_index[ell_row_start + j] = csr.col_index[row_start + j];
            } else {
                ell.values[ell_row_start + j] = 0.0;
                ell.col_index[ell_row_start + j] = -1;
            }
        }
    }

    return ell;
}

} // namespace spmv
