#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "sparse_matrix.h"

namespace spmv {

// --------------------------------------------------------------------------
// Matrix Market banner field identifiers
// --------------------------------------------------------------------------
enum class MatrixFormat { COO, CSR };

enum class ScalarField { REAL, INTEGER, COMPLEX, PATTERN };
enum class SymmetryField { GENERAL, SYMMETRIC, SKEW_SYMMETRIC, HERMITIAN };

struct MatrixMarketHeader {
    MatrixFormat format;
    ScalarField  scalar;
    SymmetryField symmetry;
};

// --------------------------------------------------------------------------
// Parse a Matrix Market .mtx file into a SparseMatrix (COO, then convert to CSR)
// --------------------------------------------------------------------------
SparseMatrix parse_matrix_market(const std::string& filepath);

// Also support reading into COO directly (useful for some GPU formats later)
COO_SparseMatrix parse_matrix_market_coo(const std::string& filepath);

// --------------------------------------------------------------------------
// Convert COO to CSR format
// --------------------------------------------------------------------------
SparseMatrix coo_to_csr(const COO_SparseMatrix& coo);

} // namespace spmv
