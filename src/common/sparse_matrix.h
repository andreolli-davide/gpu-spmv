#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace spmv {

// --------------------------------------------------------------------------
// Sparse matrix stored in CSR (Compressed Sparse Row) format
// --------------------------------------------------------------------------
struct SparseMatrix {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;

    std::vector<double> values;
    std::vector<int64_t> col_index;
    std::vector<int64_t> row_ptr;

    SparseMatrix() = default;
    explicit SparseMatrix(int64_t rows, int64_t cols, int64_t nnz);

    void allocate();
    int64_t memory_bytes() const;
};

// --------------------------------------------------------------------------
// Sparse matrix stored in COO (Coordinate) format
// --------------------------------------------------------------------------
struct COO_SparseMatrix {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;

    std::vector<double> values;
    std::vector<int64_t> row;
    std::vector<int64_t> col;

    COO_SparseMatrix() = default;
    explicit COO_SparseMatrix(int64_t rows, int64_t cols, int64_t nnz);

    void allocate();
    int64_t memory_bytes() const;
};

// --------------------------------------------------------------------------
// Dense vector
// --------------------------------------------------------------------------
struct DenseVector {
    int64_t size = 0;
    std::vector<double> data;

    DenseVector() = default;
    explicit DenseVector(int64_t size);
    void resize(int64_t size);
    int64_t memory_bytes() const;
};

} // namespace spmv
