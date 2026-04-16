#include "sparse_matrix.h"

namespace spmv {

// --------------------------------------------------------------------------
// SparseMatrix (CSR)
// --------------------------------------------------------------------------
SparseMatrix::SparseMatrix(int64_t r, int64_t c, int64_t n)
    : rows(r), cols(c), nnz(n) {}

void SparseMatrix::allocate() {
    values.resize(nnz);
    col_index.resize(nnz);
    row_ptr.resize(rows + 1);
}

int64_t SparseMatrix::memory_bytes() const {
    return nnz * sizeof(double)
         + nnz * sizeof(int64_t)
         + (rows + 1) * sizeof(int64_t)
         + 3 * sizeof(int64_t);
}

// --------------------------------------------------------------------------
// COO_SparseMatrix (COO)
// --------------------------------------------------------------------------
COO_SparseMatrix::COO_SparseMatrix(int64_t r, int64_t c, int64_t n)
    : rows(r), cols(c), nnz(n) {}

void COO_SparseMatrix::allocate() {
    values.resize(nnz);
    row.resize(nnz);
    col.resize(nnz);
}

int64_t COO_SparseMatrix::memory_bytes() const {
    return nnz * sizeof(double)
         + nnz * sizeof(int64_t)
         + nnz * sizeof(int64_t)
         + 3 * sizeof(int64_t);
}

// --------------------------------------------------------------------------
// DenseVector
// --------------------------------------------------------------------------
DenseVector::DenseVector(int64_t s) : size(s) { data.resize(size); }

void DenseVector::resize(int64_t s) {
    size = s;
    data.resize(size);
}

int64_t DenseVector::memory_bytes() const {
    return size * sizeof(double) + sizeof(int64_t);
}

} // namespace spmv
