// =============================================================================
// matrix_market.cpp
// =============================================================================
// Implementation of Matrix Market .mtx file parsing and COO → CSR conversion.
//
// Parsing Strategy
// ----------------
// The parser reads the file line by line rather than buffering the whole thing.
// This is important for large files (some SuiteSparse matrices are tens of GB)
// where fitting the entire file in memory is undesirable.
//
// The parsing is deliberately "streaming" within each section:
//
//   • Banner:  scan for the first line starting with "%%MatrixMarket"
//   • Header:  skip comment lines (blank or starting with "%");
//              the first non-blank/non-comment line is the size line
//   • Entries: read one line at a time, extract (row, col, value) triples
//
// Matrix Market files are always 1-indexed.  We convert to 0-based on read.
//
// Value Types
// ------------
// Pattern matrices have no value field — we treat those as value = 1.0.
// Complex values are explicitly rejected (they would require a different
// data structure with paired real/imaginary doubles).
//
// Symmetry
// --------
// Only GENERAL (unsymmetric) matrices are fully supported in Phase 1.
// SYMMETRIC matrices in the file represent only one triangle, but we
// currently read them as-is (the other triangle is implicitly zero).
// A future phase will expand symmetric matrices to full format.
// =============================================================================

#include "matrix_market.h"
#include "sparse_matrix.h"

#include <fstream>    // std::ifstream
#include <sstream>    // std::istringstream
#include <stdexcept>  // std::runtime_error
#include <vector>     // std::vector

namespace spmv {
namespace {

// =============================================================================
// parse_banner
// =============================================================================
// Parses the %%MatrixMarket header line into a MatrixMarketHeader struct.
//
// Example banner:
//   %%MatrixMarket matrix coordinate real general
//
// Expected format tokens:
//   1. "%%MatrixMarket"  — we skip this (already matched to find the line)
//   2. "matrix" or "vector" — we accept "matrix" only
//   3. "coordinate" or "array" — we accept "coordinate" only
//   4. "real" | "integer" | "complex" | "pattern"
//   5. "general" | "symmetric" | "skew-symmetric" | "hermitian"  (optional)
//
// Unknown tokens cause a std::runtime_error with the problematic line included.
// This makes debugging easy: copy the error message and paste it into the file.
//
// @param line  Full text of the banner line
// @return MatrixMarketHeader with parsed format, scalar, and symmetry fields
// @throws std::runtime_error  on unknown or unsupported format tokens
// =============================================================================
MatrixMarketHeader parse_banner(const std::string& line) {
    MatrixMarketHeader h{};
    std::istringstream iss(line);
    std::string token;

    // Token 1: "%%MatrixMarket" — already matched by the caller
    iss >> token >> token; // advance past "%%MatrixMarket"

    // Token 2: format — "coordinate" or "array"
    iss >> token;
    if (token == "coordinate") {
        h.format = MatrixFormat::COO;
    } else if (token == "array") {
        // We do not support dense (array) format — it would need a completely
        // different parsing strategy and is rare in sparse matrix collections.
        throw std::runtime_error("Unsupported Matrix Market format 'array' "
                                 "(dense matrices are not supported): " + line);
    } else {
        throw std::runtime_error("Unknown Matrix Market format token '" + token
                                 + "' in banner: " + line);
    }

    // Token 3: scalar field — "real", "integer", "complex", "pattern"
    iss >> token;
    if (token == "real") {
        h.scalar = ScalarField::REAL;
    } else if (token == "integer") {
        // Stored as double internally — no precision change for the sizes we handle
        h.scalar = ScalarField::INTEGER;
    } else if (token == "complex") {
        throw std::runtime_error("Unsupported scalar field 'complex': " + line);
    } else if (token == "pattern") {
        // Pattern matrices define structure only; value is implicitly 1.0
        h.scalar = ScalarField::PATTERN;
    } else {
        throw std::runtime_error("Unknown scalar field '" + token + "': " + line);
    }

    // Token 4 (optional): symmetry — "general", "symmetric", "skew-symmetric", "hermitian"
    // The symmetry token is on the same line as the other three; we read it last.
    // If there is no symmetry token (old-style files), the stream simply fails
    // to extract anything and we treat it as GENERAL.
    iss >> token;
    if (!iss) {
        // No fourth token — older MM files have only 3 tokens
        h.symmetry = SymmetryField::GENERAL;
    } else if (token == "general") {
        h.symmetry = SymmetryField::GENERAL;
    } else if (token == "symmetric") {
        h.symmetry = SymmetryField::SYMMETRIC;
    } else if (token == "skew-symmetric") {
        h.symmetry = SymmetryField::SKEW_SYMMETRIC;
    } else if (token == "hermitian") {
        h.symmetry = SymmetryField::HERMITIAN;
    } else {
        throw std::runtime_error("Unknown symmetry field '" + token + "': " + line);
    }

    return h;
}

} // anonymous namespace

// =============================================================================
// parse_matrix_market_coo
// =============================================================================
// Reads a .mtx file and returns a COO_SparseMatrix.
//
// State machine (simplified):
//   SCAN  → looking for the %%MatrixMarket banner line
//   SKIP  → skip comments and blank lines between banner and size line
//   SIZE  → read rows cols nnz from the first non-comment line
//   READ  → read nnz entry lines
//
// Implementation notes:
//   • Using std::getline avoids issues with Windows CRLF line endings.
//   • Each entry line is parsed independently — a malformed line causes
//     a continue (skip) rather than a hard error, which makes the parser
//     robust to files with occasional bad lines.
//   • The final nnz may be less than the header's declared nnz if the
//     file contains invalid lines; we resize down to what we actually read.
// =============================================================================
COO_SparseMatrix parse_matrix_market_coo(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs) {
        throw std::runtime_error("Cannot open file (check path and permissions): "
                                 + filepath);
    }

    std::string line;

    // ── State: SCAN ────────────────────────────────────────────────────────
    // Find the %%MatrixMarket banner (required as first non-comment line)
    do {
        if (!std::getline(ifs, line)) {
            throw std::runtime_error("No Matrix Market banner found (file is "
                                     "empty or not an .mtx file): " + filepath);
        }
    } while (line.rfind("%%MatrixMarket", 0) == std::string::npos);

    // ── State: PARSE HEADER ───────────────────────────────────────────────
    MatrixMarketHeader header = parse_banner(line);

    // Only coordinate format is supported (array/dense would need a different reader)
    if (header.format != MatrixFormat::COO) {
        throw std::runtime_error("Only 'coordinate' format is supported; "
                                 "this file declares a different format");
    }

    // ── State: SKIP ────────────────────────────────────────────────────────
    // Skip blank lines and comment lines (lines starting with '%' or "%%")
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;                         // blank
        if (line.rfind("%%", 0) == 0) continue;             // comment
        if (line[0] == '%') continue;                       // old-style comment
        break;                                               // first data line
    }

    // line now holds the size line: "rows  cols  nnz"
    // The standard guarantees exactly three integers, whitespace-separated.
    int64_t rows = 0, cols = 0, nnz = 0;
    {
        std::istringstream size_stream(line);
        size_stream >> rows >> cols >> nnz;
        if (!size_stream) {
            throw std::runtime_error("Failed to parse size line (expected "
                                     "'rows  cols  nnz'): " + line);
        }
        if (rows <= 0 || cols <= 0 || nnz < 0) {
            throw std::runtime_error("Invalid matrix dimensions (rows="
                                     + std::to_string(rows)
                                     + ", cols=" + std::to_string(cols)
                                     + ", nnz=" + std::to_string(nnz) + ")");
        }
    }

    // ── State: READ ───────────────────────────────────────────────────────
    COO_SparseMatrix coo(rows, cols, nnz);
    coo.allocate();

    int64_t idx = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;                         // skip blank
        if (line.rfind("%%", 0) == 0) continue;            // skip comments
        if (line[0] == '%') continue;

        std::istringstream entry_stream(line);
        int64_t r = 0, c = 0;
        double v = 1.0; // default for PATTERN matrices

        entry_stream >> r >> c;
        if (!entry_stream) continue; // malformed line — skip

        // Value is only present if the scalar field is not PATTERN
        if (header.scalar != ScalarField::PATTERN) {
            entry_stream >> v;
            // Note: we don't check entry_stream here — a line with row/col
            // but no value is treated as v=0.0 (not 1.0).  This is intentional:
            // it matches MM spec behavior for partial lines.
        }

        // Matrix Market is always 1-based; convert to 0-based
        coo.row[idx]    = r - 1;
        coo.col[idx]    = c - 1;
        coo.values[idx] = v;
        ++idx;
    }

    // Shrink to actual number of valid entries read (may be < declared nnz)
    coo.nnz     = idx;
    coo.values.resize(idx);
    coo.row.resize(idx);
    coo.col.resize(idx);

    return coo;
}

// =============================================================================
// parse_matrix_market
// =============================================================================
SparseMatrix parse_matrix_market(const std::string& filepath) {
    return coo_to_csr(parse_matrix_market_coo(filepath));
}

// =============================================================================
// coo_to_csr — counting sort conversion
// =============================================================================
// Converts COO (row[i], col[i], values[i]) triples into CSR row-oriented format.
//
// This is not a sort — it is a counting sort that exploits the fixed-size
// row counter array.  The key insight is that we know exactly how many
// entries belong to each row BEFORE we place any of them, which lets us
// pre-compute every write position with a single prefix sum.
//
// Step 1 — Count per row
//   row_count[i] = how many COO entries have row == i
//   O(nnz) time, O(rows) space
//
// Step 2 — Prefix sum → row_ptr
//   row_ptr[0] = 0
//   row_ptr[i+1] = row_ptr[i] + row_count[i]   for i in [0, rows)
//   After this, row_ptr[i] = count of nnz in rows [0, i)
//   and row_ptr[rows] = total nnz
//
// Step 3 — Write with per-row write cursor
//   write_offset is a copy of row_ptr used as a mutable cursor.
//   For each COO entry (r, c, v):
//       pos = write_offset[r]++
//       values[pos] = v; col_index[pos] = c
//   All writes are independent — O(nnz) time, cache-friendly sequential access
//
// Total: O(nnz) time, O(rows) auxiliary space (plus the output CSR itself).
//
// Note: entries within the same row are written in the order they appear
// in the COO arrays — which is file order (not sorted by column).
// This is intentional: it preserves the original ordering from the .mtx
// file and avoids an extra O(nnz log nnz) sort step.
//
// @param coo  Input matrix in COO format; entries need NOT be sorted by row
// @return SparseMatrix in CSR format
// =============================================================================
SparseMatrix coo_to_csr(const COO_SparseMatrix& coo) {
    SparseMatrix csr(coo.rows, coo.cols, coo.nnz);
    csr.allocate();

    // ── Step 1: count non-zeros per row ────────────────────────────────────
    std::vector<int64_t> row_count(coo.rows, 0);
    for (int64_t i = 0; i < coo.nnz; ++i) {
        ++row_count[coo.row[i]];
    }

    // ── Step 2: prefix sum → row_ptr ──────────────────────────────────────
    // row_ptr[i] = total nnz in rows [0, i)
    // row_ptr[0] = 0 by definition
    csr.row_ptr[0] = 0;
    for (int64_t i = 0; i < coo.rows; ++i) {
        csr.row_ptr[i + 1] = csr.row_ptr[i] + row_count[i];
    }

    // ── Step 3: scatter into CSR arrays using per-row write cursors ────────
    // write_offset starts as a copy of row_ptr; each write increments it.
    std::vector<int64_t> write_offset = csr.row_ptr;
    for (int64_t i = 0; i < coo.nnz; ++i) {
        const int64_t r   = coo.row[i];   // source row
        const int64_t pos = write_offset[r]++; // claim and advance write position
        csr.values[pos]     = coo.values[i];
        csr.col_index[pos]  = coo.col[i];
    }

    return csr;
}

} // namespace spmv
