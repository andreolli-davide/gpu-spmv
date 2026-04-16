// =============================================================================
// matrix_market.h
// =============================================================================
// Parsing of Matrix Market files (.mtx) into sparse matrix data structures.
//
// Matrix Market Format Overview
// ------------------------------
// The Matrix Market Exchange Formats (Boeing & Nastase 2017) is a de facto
// standard for distributing sparse matrices.  A .mtx file looks like this:
//
//   %%MatrixMarket matrix coordinate real general
//   %----------------------------------------------------------------------------|
//   % This is a comment block describing the matrix                                 |
//   %----------------------------------------------------------------------------|
//        5     5    13
//     1  1  2.0
//     1  2  1.0
//     2  1  1.0
//     ...
//
// The banner (line 1) has four space-separated fields:
//
//   1. "%%MatrixMarket"         — magic header
//   2. "matrix"                 — object type (matrix, vector, stiffness)
//   3. "coordinate" | "array"   — storage format
//        coordinate → COO format  (what we support)
//        array     → dense format  (not supported in Phase 1)
//   4. "real" | "integer" | "complex" | "pattern"
//        real      → floating-point values  (supported)
//        integer   → integer values, stored as double (supported)
//        complex   → not supported (Phase 1)
//        pattern   → values omitted, assumed 1.0 (supported)
//
// Optional symmetry field (line 1, 4th token):
//   "general"            — no symmetry, all entries stored explicitly
//   "symmetric"          — only the lower or upper triangle is stored;
//                           the other half is the same.  This effectively
//                           doubles nnz when converting to a full matrix.
//   "skew-symmetric"     — A[i,j] = -A[j,i]; not yet handled
//   "hermitian"          — complex conjugate symmetry; not yet handled
//
// Coordinate format lines (one per non-zero):
//   row  col  [value]
//   — all 1-based indices (converted to 0-based on read)
//   — may appear in any order (parse does NOT sort them)
//
// References
// ----------
//   Boisvert, R., Pozo, R., Remming, K. & Suzuki, J. (1997).
//   The Matrix Market Exchange Formats.
//   NIST Report NISTIR-6025.  https://math.nist.gov/MatrixMarket/
//
// =============================================================================

#ifndef MATRIX_MARKET_H
#define MATRIX_MARKET_H

#include <string>
#include "sparse_matrix.h"

namespace spmv {

// =============================================================================
// Banner field identifiers
// =============================================================================

// Storage layout of the matrix as declared in the banner.
enum class MatrixFormat {
    COO,    // coordinate — row/col/index triples (what we read)
    CSR     // array — dense row-major (not yet supported)
};

// Data type of each matrix entry.
enum class ScalarField {
    REAL,     // IEEE-754 double (what we use internally)
    INTEGER,  // integer values, promoted to double on read
    COMPLEX,  // not supported in Phase 1
    PATTERN   // value is 1.0 (common in graph adjacency matrices)
};

// Symmetry of the matrix — currently only GENERAL is fully handled.
enum class SymmetryField {
    GENERAL,         // all entries stored explicitly
    SYMMETRIC,       // only one triangle stored; would double on expansion
    SKEW_SYMMETRIC,  // A[i,j] = -A[j,i]; not yet handled
    HERMITIAN        // complex conjugate; not yet handled
};

// Parsed header — the four tokens of the %%MatrixMarket banner.
struct MatrixMarketHeader {
    MatrixFormat  format;
    ScalarField   scalar;
    SymmetryField symmetry;
};

// =============================================================================
// High-level API
// =============================================================================

// Parse a .mtx file and return a CSR matrix.
//
// This is the primary entry point.  Internally it:
//   1. Reads into COO_SparseMatrix (parse_matrix_market_coo)
//   2. Converts to CSR (coo_to_csr)
//
// Throws std::runtime_error on malformed input (missing banner, wrong format,
// I/O errors, etc.).  The error message includes the problematic line.
//
// @param filepath  Absolute or relative path to the .mtx file
// @return SparseMatrix in CSR format, ready for SpMV
SparseMatrix parse_matrix_market(const std::string& filepath);

// Parse a .mtx file and return a COO matrix.
//
// Use this when you need the raw (row, col, value) triples — for example
// to build other formats (ELL, HYB) or to inspect the original ordering.
//
// @param filepath  Absolute or relative path to the .mtx file
// @return COO_SparseMatrix; entries are in file order (not sorted)
COO_SparseMatrix parse_matrix_market_coo(const std::string& filepath);

// =============================================================================
// Format conversion
// =============================================================================

// Convert a COO matrix to CSR format.
//
// Algorithm: counting sort (O(nnz) time, O(rows) auxiliary space).
//
//   1. Count the number of non-zeros in each row  → row_count[rows]
//   2. Prefix-sum row_count to obtain row_ptr        → row_ptr[rows+1]
//   3. Scan COO entries once; for each (row[i], col[i], val[i])
//      write it to values[write_offset[row[i]]++]
//
// The prefix-sum at step 2 is the critical path:
//   row_ptr[i+1] = row_ptr[i] + row_count[i]
// so row_ptr[i] = total non-zeros in rows [0, i).
//
// This algorithm is branch-free in the hot loop (step 3) and accesses all
// arrays sequentially — ideal for cache-friendly execution on modern CPUs.
//
// @param coo  Input matrix in COO format; entries need NOT be sorted
// @return SparseMatrix in CSR format
SparseMatrix coo_to_csr(const COO_SparseMatrix& coo);

} // namespace spmv

#endif // MATRIX_MARKET_H
