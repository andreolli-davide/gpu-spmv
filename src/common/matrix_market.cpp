#include "matrix_market.h"
#include "sparse_matrix.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace spmv {

namespace {

// --------------------------------------------------------------------------
// Parse the %%MatrixMarket banner line
// --------------------------------------------------------------------------
MatrixMarketHeader parse_banner(const std::string& line) {
    MatrixMarketHeader h{};
    std::istringstream iss(line);
    std::string token;

    // Skip "%%MatrixMarket"
    iss >> token >> token; // now token is "matrix"

    iss >> token; // format: coordinate or array
    if (token == "coordinate") {
        h.format = MatrixFormat::COO;
    } else if (token == "array") {
        throw std::runtime_error("Array format not supported: " + line);
    } else {
        throw std::runtime_error("Unknown format: " + line);
    }

    iss >> token; // scalar field: real, integer, complex, pattern
    if (token == "real") {
        h.scalar = ScalarField::REAL;
    } else if (token == "integer") {
        h.scalar = ScalarField::INTEGER;
    } else if (token == "complex") {
        throw std::runtime_error("Complex not supported: " + line);
    } else if (token == "pattern") {
        h.scalar = ScalarField::PATTERN;
    } else {
        throw std::runtime_error("Unknown scalar field: " + line);
    }

    iss >> token; // symmetry: general, symmetric, skew-symmetric, hermitian
    if (token == "general") {
        h.symmetry = SymmetryField::GENERAL;
    } else if (token == "symmetric") {
        h.symmetry = SymmetryField::SYMMETRIC;
    } else if (token == "skew-symmetric") {
        h.symmetry = SymmetryField::SKEW_SYMMETRIC;
    } else if (token == "hermitian") {
        h.symmetry = SymmetryField::HERMITIAN;
    } else {
        throw std::runtime_error("Unknown symmetry: " + line);
    }

    return h;
}

// --------------------------------------------------------------------------
// Parse .mtx into COO_SparseMatrix
// --------------------------------------------------------------------------
COO_SparseMatrix parse_matrix_market_coo(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs) throw std::runtime_error("Cannot open file: " + filepath);

    std::string line;

    // Read banner line
    do {
        if (!std::getline(ifs, line))
            throw std::runtime_error("No MatrixMarket banner found");
    } while (line.rfind("%%MatrixMarket", 0) == std::string::npos);

    auto header = parse_banner(line);
    if (header.format != MatrixFormat::COO)
        throw std::runtime_error("Only coordinate format supported");

    // Skip comment lines
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (line.rfind("%%", 0) == 0) continue;
        break;
    }

    // Read size line: rows cols nnz
    int64_t rows = 0, cols = 0, nnz = 0;
    {
        std::istringstream iss(line);
        iss >> rows >> cols >> nnz;
        if (!iss) throw std::runtime_error("Failed to parse size line: " + line);
    }

    COO_SparseMatrix coo(rows, cols, nnz);
    coo.allocate();

    int64_t idx = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (line.rfind("%%", 0) == 0) continue;

        std::istringstream iss(line);
        int64_t r, c;
        double v = 1.0;
        iss >> r >> c;
        if (header.scalar != ScalarField::PATTERN) iss >> v;
        if (!iss) continue;

        coo.row[idx] = r - 1; // 1-based to 0-based
        coo.col[idx] = c - 1;
        coo.values[idx] = v;
        ++idx;
    }

    coo.nnz = idx;
    coo.values.resize(idx);
    coo.row.resize(idx);
    coo.col.resize(idx);

    return coo;
}

// --------------------------------------------------------------------------
// Parse .mtx into CSR SparseMatrix
// --------------------------------------------------------------------------
SparseMatrix parse_matrix_market(const std::string& filepath) {
    return coo_to_csr(parse_matrix_market_coo(filepath));
}

// --------------------------------------------------------------------------
// Convert COO to CSR
// --------------------------------------------------------------------------
SparseMatrix coo_to_csr(const COO_SparseMatrix& coo) {
    SparseMatrix csr(coo.rows, coo.cols, coo.nnz);
    csr.allocate();

    // Count nnz per row
    std::vector<int64_t> row_count(coo.rows, 0);
    for (int64_t i = 0; i < coo.nnz; ++i) {
        ++row_count[coo.row[i]];
    }

    // row_ptr[i] = number of non-zeros in rows [0, i)
    csr.row_ptr[0] = 0;
    for (int64_t i = 0; i < coo.rows; ++i) {
        csr.row_ptr[i + 1] = csr.row_ptr[i] + row_count[i];
    }

    // Fill values and col_index using per-row write offsets
    std::vector<int64_t> write_offset = csr.row_ptr;
    for (int64_t i = 0; i < coo.nnz; ++i) {
        int64_t r = coo.row[i];
        int64_t pos = write_offset[r]++;
        csr.values[pos] = coo.values[i];
        csr.col_index[pos] = coo.col[i];
    }

    return csr;
}

} // namespace spmv
