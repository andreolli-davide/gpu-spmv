#include "parser/mtx_parser.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace {

// ── Small string utilities (file-private) ────────────────────────────────────

/// Remove leading and trailing whitespace from \p s.
/// Returns an empty string if \p s is all whitespace.
std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

/// True if \p line is empty or starts with '%' (MTX comment / header guard).
bool is_comment_or_empty(const std::string& line) {
    auto trimmed = trim(line);
    return trimmed.empty() || trimmed.front() == '%';
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// MTX file parsing — to COO
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a Matrix Market file into an MtxCoo structure.
///
/// The expected MTX format (coordinate variant) is:
///
///   %%MatrixMarket matrix coordinate real/general
///   % comments ...
///   M  N  nz        ← dimensions + nnz (1-based)
///   r  c  v         ← one nonzero per line, 1-based indices
///   ...
///
/// Blank lines and all comment lines (starting with '%') are skipped.
/// The first non-comment, non-blank line must contain three integers:
/// num_rows, num_cols, and num_nonzeros.  Subsequent lines are the nonzero
/// entries.  All indices are converted from 1-based (MTX spec) to 0-based
/// (our internal convention).
///
/// \throws std::runtime_error  if the file cannot be opened, the header is
/// missing, or a data line cannot be parsed.
MtxCoo parse_mtx(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MTX file: " + filepath);
    }

    // ── Step 0: parse the %%MatrixMarket banner ─────────────────────────────
    // Format: %%MatrixMarket matrix coordinate|array real|complex|integer|pattern general|symmetric|...
    bool is_pattern = false;
    std::string line;
    if (std::getline(file, line)) {
        auto trimmed = trim(line);
        if (trimmed.size() >= 2 && trimmed[0] == '%' && trimmed[1] == '%') {
            std::istringstream iss(trimmed);
            std::string token, object, format, field;
            iss >> token >> object >> format >> field; // %%MatrixMarket matrix coordinate real
            // Convert to lowercase for case-insensitive comparison
            std::transform(format.begin(), format.end(), format.begin(), ::tolower);
            std::transform(field.begin(), field.end(), field.begin(), ::tolower);
            if (format == "array") {
                throw std::runtime_error(
                    "Dense (array) MTX format is not supported — only coordinate format: "
                    + filepath);
            }
            is_pattern = (field == "pattern");
        }
    }

    // ── Step 1: find the size/nnz header line ───────────────────────────────
    int32_t num_rows = 0, num_cols = 0, num_nonzeros = 0;
    bool header_found = false;

    while (std::getline(file, line)) {
        if (is_comment_or_empty(line)) continue;
        if (!header_found) {
            std::istringstream iss(line);
            if (!(iss >> num_rows >> num_cols >> num_nonzeros)) {
                throw std::runtime_error("Invalid MTX header: " + line);
            }
            header_found = true;
        } else {
            break;
        }
    }

    if (!header_found) {
        throw std::runtime_error("MTX header not found in file: " + filepath);
    }

    // ── Step 2: parse nonzero entries ───────────────────────────────────────
    MtxCoo coo;
    coo.num_rows = num_rows;
    coo.num_cols = num_cols;
    coo.num_nonzeros = num_nonzeros;
    coo.row_indices.reserve(num_nonzeros);
    coo.col_indices.reserve(num_nonzeros);
    coo.values.reserve(num_nonzeros);

    auto parse_data_line = [&](const std::string& l) {
        std::istringstream iss(l);
        int32_t r, c;
        float v = 1.0f; // default for pattern matrices
        if (!(iss >> r >> c)) {
            throw std::runtime_error("Invalid MTX data line: " + l);
        }
        if (!is_pattern && !(iss >> v)) {
            throw std::runtime_error("Invalid MTX data line (missing value): " + l);
        }
        if (r - 1 < 0 || r - 1 >= num_rows || c - 1 < 0 || c - 1 >= num_cols) {
            throw std::runtime_error("MTX index out of range: row=" +
                std::to_string(r) + " col=" + std::to_string(c));
        }
        coo.row_indices.push_back(r - 1);
        coo.col_indices.push_back(c - 1);
        coo.values.push_back(v);
    };

    // The header loop already consumed the first data line into `line` —
    // parse it here so it isn't lost.
    if (!is_comment_or_empty(line)) {
        parse_data_line(line);
    }

    while (std::getline(file, line)) {
        if (is_comment_or_empty(line)) continue;
        parse_data_line(line);
    }

    // Reconcile: use actual parsed count, not the header-declared value.
    coo.num_nonzeros = static_cast<int32_t>(coo.row_indices.size());

    return coo;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: parse directly to CSR
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an MTX file and immediately convert to CSR (parse_mtx + coo_to_csr).
MtxCsr parse_mtx_csr(const std::string& filepath) {
    return coo_to_csr(parse_mtx(filepath));
}

// ─────────────────────────────────────────────────────────────────────────────
// COO → CSR conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a COO matrix to CSR format.
///
/// Algorithm: counting sort with prefix sum.
///
///   1. Count nonzeros per row   → row_ptr[i] = count of nz in row i
///   2. Prefix sum               → row_ptr[i] = start offset of row i
///   3. Walk nonzeros again, placing each at row_ptr_copy[row]++
///
/// A working copy of row_ptr is kept during fill (step 3) so that the
/// prefix-sum result is not clobbered before it is fully consumed.
///
/// Runtime: O(num_nonzeros + num_rows)
/// Space:   O(num_nonzeros) for col_indices + values output;
///         O(num_rows) temporary for the row_ptr copy.
MtxCsr coo_to_csr(const MtxCoo& coo) {
    MtxCsr csr;
    csr.num_rows = coo.num_rows;
    csr.num_cols = coo.num_cols;

    // Use actual vector size — not the header-declared num_nonzeros —
    // to guard against truncated files or header/data mismatches.
    const int32_t nnz = static_cast<int32_t>(coo.row_indices.size());
    csr.num_nonzeros = nnz;
    csr.row_ptr.resize(coo.num_rows + 1, 0);

    // Step 1 — count nonzeros per row
    for (int32_t i = 0; i < nnz; ++i) {
        ++csr.row_ptr[coo.row_indices[i]];
    }

    // Step 2 — prefix sum: row_ptr[i] becomes the start offset of row i
    int32_t sum = 0;
    for (int32_t i = 0; i < coo.num_rows + 1; ++i) {
        int32_t val = csr.row_ptr[i];
        csr.row_ptr[i] = sum;
        sum += val;
    }

    // Step 3 — scatter nonzeros into col_indices and values using the
    //         working copy to track the next free position per row
    std::vector<int32_t> row_ptr_copy = csr.row_ptr;
    csr.col_indices.resize(nnz);
    csr.values.resize(nnz);

    for (int32_t i = 0; i < nnz; ++i) {
        int32_t row = coo.row_indices[i];
        int32_t pos = row_ptr_copy[row]++;
        csr.col_indices[pos] = coo.col_indices[i];
        csr.values[pos] = coo.values[i];
    }

    return csr;
}

// ─────────────────────────────────────────────────────────────────────────────
// CSR → COO conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a CSR matrix back to COO by expanding each row's run of entries
/// into individual (row, col, value) triples.
///
/// This is O(num_nonzeros) and is primarily useful for validating that
/// parse_mtx and coo_to_csr are inverses of each other.
MtxCoo csr_to_coo(const MtxCsr& csr) {
    MtxCoo coo;
    coo.num_rows = csr.num_rows;
    coo.num_cols = csr.num_cols;
    coo.num_nonzeros = csr.num_nonzeros;
    coo.row_indices.reserve(csr.num_nonzeros);
    coo.col_indices.reserve(csr.num_nonzeros);
    coo.values.reserve(csr.num_nonzeros);

    for (int32_t row = 0; row < csr.num_rows; ++row) {
        for (int32_t idx = csr.row_ptr[row]; idx < csr.row_ptr[row + 1]; ++idx) {
            coo.row_indices.push_back(row);
            coo.col_indices.push_back(csr.col_indices[idx]);
            coo.values.push_back(csr.values[idx]);
        }
    }

    return coo;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU SpMV reference implementations
// ─────────────────────────────────────────────────────────────────────────────

/// CPU SpMV: y = A*x  using CSR layout.
///
/// One thread (here, one loop iteration) processes one matrix row:
///   sum = 0
///   for idx in row_ptr[row] .. row_ptr[row+1]-1:
///       sum += values[idx] * x[col_indices[idx]]
///   y[row] = sum
///
/// This is the reference implementation used to verify the GPU kernel
/// produces numerically correct results.  Floating-point accumulation order
/// is deterministic (row-major, stride-1 on values and col_indices).
void spmv_cpu(const MtxCsr& csr, const float* x, float* y) {
    for (int32_t row = 0; row < csr.num_rows; ++row) {
        float sum = 0.0f;
        for (int32_t idx = csr.row_ptr[row]; idx < csr.row_ptr[row + 1]; ++idx) {
            sum += csr.values[idx] * x[csr.col_indices[idx]];
        }
        y[row] = sum;
    }
}

/// CPU SpMV: y = A*x  using COO layout.
///
/// Accumulators are zeroed upfront (one pass over all rows), then each
/// nonzero updates its target row.  Because nonzeros arrive in file order
/// rather than sorted by row, the accumulation order differs from CSR —
/// this is deliberate, as it exposes floating-point associativity differences
/// between GPU and CPU when the kernel is validated against this path.
void spmv_cpu(const MtxCoo& coo, const float* x, float* y) {
    for (int32_t row = 0; row < coo.num_rows; ++row) {
        y[row] = 0.0f;
    }
    for (int32_t i = 0; i < coo.num_nonzeros; ++i) {
        int32_t row = coo.row_indices[i];
        int32_t col = coo.col_indices[i];
        y[row] += coo.values[i] * x[col];
    }
}
