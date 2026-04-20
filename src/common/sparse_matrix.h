// =============================================================================
// sparse_matrix.h
// =============================================================================
// Data structures for sparse matrices in CSR and COO format, and dense vectors.
//
// Sparse Matrix Storage Overview
// --------------------------------
// A sparse matrix stores only the non-zero elements to avoid the O(rows × cols)
// memory cost of a dense matrix.  This is essential for scientific matrices
// where nnz << rows × cols (e.g. a 1M × 1M matrix with 10M non-zeros).
//
// Two formats are provided:
//
//   COO (Coordinate) — the simplest format.  Three parallel arrays hold every
//   non-zero as a triple (row, col, value).  Easy to build, easy to read/write,
//   and the natural output of the Matrix Market parser.  Not ideal for
//   arithmetic because row operations require scanning the entire values array.
//
//   CSR (Compressed Sparse Row) — the workhorse of sparse linear algebra.
//   Values and column indices are laid out row-by-row in a single flat array.
//   `row_ptr[i]` encodes the index range [row_ptr[i], row_ptr[i+1]) of the
//   non-zeros belonging to row `i`.  This enables O(1) access to a given row
//   and makes CSR the standard format for SpMV on both CPU and GPU.
//
// Memory layout for a 3×4 matrix with nnz=5:
//
//   row 0: [a   b]   values  = [a, b, c, d, e]
//   row 1: [  c   ]   col_idx = [0, 3, 1, 2, 3]   (0-based column of each value)
//   row 2: [d e  ]   row_ptr = [0, 2, 3, 5]       (prefix sum of nnz per row)
//
//   row_ptr has length rows+1; row_ptr[rows] == nnz by convention.
//
// =============================================================================

#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <cstdint>   // int64_t — 64-bit for matrix dimensions ( SuiteSparse matrices
                     // regularly exceed 2³¹−1 rows )
#include <vector>    // std::vector — storage for all arrays

namespace spmv {

// =============================================================================
// SparseMatrix — CSR (Compressed Sparse Row) format
// =============================================================================
// Primary format used throughout the project.  All GPU kernels (Phase 2+) expect
// data laid out exactly as described here.
//
// Invariants:
//   • values.size()  == nnz
//   • col_index.size() == nnz
//   • row_ptr.size() == rows + 1
//   • row_ptr[0] == 0
//   • row_ptr[rows] == nnz  (always true after coo_to_csr())
//   • 0 ≤ col_index[i] < cols  for all i
//   • row_ptr is strictly non-decreasing
//
struct SparseMatrix {
    int64_t rows = 0;      // number of matrix rows
    int64_t cols = 0;      // number of matrix columns
    int64_t nnz  = 0;      // number of non-zero elements

    std::vector<double>  values;     // non-zero values, size nnz
    std::vector<int64_t> col_index; // column index of each value, size nnz
    std::vector<int64_t> row_ptr;    // start offset of each row, size rows+1

    SparseMatrix() = default;

    // Constructs a matrix with the given dimensions and non-zero count.
    // All storage arrays are sized but not filled — call allocate() next.
    explicit SparseMatrix(int64_t rows, int64_t cols, int64_t nnz);

    // Allocates/resizes all three storage arrays to their required sizes.
    // Call this after construction or when resizing.
    void allocate();

    // Returns the total memory footprint on the host in bytes.
    // Useful for verifying that GPU copies will fit in device memory.
    int64_t memory_bytes() const;
};

// =============================================================================
// COO_SparseMatrix — COO (Coordinate / "ijv") format
// =============================================================================
// Simpler format used as the intermediate representation during parsing.
// All three arrays are parallel: entry i is at (row[i], col[i]) with value[i].
//
// Invariants:
//   • values.size() == row.size() == col.size() == nnz
//   • 0 ≤ row[i] < rows  for all i
//   • 0 ≤ col[i] < cols  for all i
//   • Entries need NOT be sorted (parse_matrix_market_coo returns them in file order)
//
struct COO_SparseMatrix {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz  = 0;

    std::vector<double>  values; // non-zero values, size nnz
    std::vector<int64_t> row;    // row index of each entry, size nnz (0-based)
    std::vector<int64_t> col;    // column index of each entry, size nnz (0-based)

    COO_SparseMatrix() = default;

    explicit COO_SparseMatrix(int64_t rows, int64_t cols, int64_t nnz);
    void allocate();
    int64_t memory_bytes() const;
};

// =============================================================================
// ELL_SparseMatrix — ELLPACK (Equal-Length List) format
// =============================================================================
// Pads all rows to the same length (max_row_length) with zeros.
// Enables coalesced memory access and eliminates row-boundary checks.
//
// Memory layout: values and col_index are stored row-by-row, all rows padded
// to max_row_length. Entries with col_index = -1 are padding (no value).
//
// Example 3×4 matrix with max_row_length=2:
//   row 0: [a b]   values   = [a, b, c, d, 0, 0]
//   row 1: [c  ]   col_idx  = [0, 3, 1, -1, -1, -1]
//   row 2: [d  ]
//
// Best for: Regular matrices with similar row lengths (FEM, structured grids).
// Warning: Can be wasteful for highly irregular matrices (e.g., webbase).
//
// Reference: Bell & Garland '09 "Efficient Sparse Matrix-Vector Multiplication on GPUs"
//
struct ELL_SparseMatrix {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    int64_t max_row_length = 0;  // Padded row length (all rows same)

    std::vector<double> values;      // size: rows × max_row_length
    std::vector<int64_t> col_index; // size: rows × max_row_length (-1 for padding)

    ELL_SparseMatrix() = default;
    explicit ELL_SparseMatrix(int64_t rows, int64_t cols, int64_t max_row_length);
    void allocate();
    int64_t memory_bytes() const;
};

// Converts a CSR matrix into ELLPACK format.
ELL_SparseMatrix csr_to_ell(const SparseMatrix& csr);

// =============================================================================
// DenseVector — flat double array with size tracking
// =============================================================================
// Used for the input vector x and output vector y in SpMV (y = A·x).
// Storing size explicitly lets us validate dimensions without a separate
// parameter or an out-of-bounds sentinel.
//
struct DenseVector {
    int64_t size = 0;             // number of elements
    std::vector<double> data;     // element storage, size elements

    DenseVector() = default;

    // Constructs and allocates a vector of the given size, filled with zeros.
    explicit DenseVector(int64_t size);

    // Resizes the vector, preserving existing data where possible.
    void resize(int64_t size);

    int64_t memory_bytes() const;
};

} // namespace spmv

#endif // SPARSE_MATRIX_H
