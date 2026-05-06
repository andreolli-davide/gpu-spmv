#include "parser/mtx_parser.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace {

std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool is_comment_or_empty(const std::string& line) {
    auto trimmed = trim(line);
    return trimmed.empty() || trimmed.front() == '%';
}

} // anonymous namespace

MtxCoo parse_mtx(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MTX file: " + filepath);
    }

    int32_t num_rows = 0, num_cols = 0, num_nonzeros = 0;
    bool header_found = false;

    std::string line;
    while (std::getline(file, line)) {
        if (is_comment_or_empty(line)) continue;
        if (!header_found) {
            std::istringstream iss(line);
            if (!(iss >> num_rows >> num_cols >> num_nonzeros)) {
                throw std::runtime_error("Invalid MTX header: " + line);
            }
            header_found = true;
        } else {
            throw std::runtime_error("Unexpected content after MTX header");
        }
    }

    if (!header_found) {
        throw std::runtime_error("MTX header not found in file: " + filepath);
    }

    MtxCoo coo;
    coo.num_rows = num_rows;
    coo.num_cols = num_cols;
    coo.num_nonzeros = num_nonzeros;
    coo.row_indices.reserve(num_nonzeros);
    coo.col_indices.reserve(num_nonzeros);
    coo.values.reserve(num_nonzeros);

    // Rewind and read data
    file.clear();
    file.seekg(0);
    while (std::getline(file, line)) {
        if (is_comment_or_empty(line)) continue;
        std::istringstream iss(line);
        int32_t r, c;
        float v;
        if (!(iss >> r >> c >> v)) {
            throw std::runtime_error("Invalid MTX data line: " + line);
        }
        // Convert from 1-indexed to 0-indexed
        coo.row_indices.push_back(r - 1);
        coo.col_indices.push_back(c - 1);
        coo.values.push_back(v);
    }

    return coo;
}

MtxCsr parse_mtx_csr(const std::string& filepath) {
    return coo_to_csr(parse_mtx(filepath));
}

MtxCsr coo_to_csr(const MtxCoo& coo) {
    MtxCsr csr;
    csr.num_rows = coo.num_rows;
    csr.num_cols = coo.num_cols;
    csr.num_nonzeros = coo.num_nonzeros;

    csr.row_ptr.resize(coo.num_rows + 1, 0);

    // Count non-zeros per row
    for (int32_t i = 0; i < coo.num_nonzeros; ++i) {
        ++csr.row_ptr[coo.row_indices[i]];
    }

    // Prefix sum to get row_ptr
    int32_t sum = 0;
    for (int32_t i = 0; i < coo.num_rows + 1; ++i) {
        int32_t val = csr.row_ptr[i];
        csr.row_ptr[i] = sum;
        sum += val;
    }

    // Fill col_indices and values using a working copy of row_ptr
    std::vector<int32_t> row_ptr_copy = csr.row_ptr;
    csr.col_indices.resize(coo.num_nonzeros);
    csr.values.resize(coo.num_nonzeros);

    for (int32_t i = 0; i < coo.num_nonzeros; ++i) {
        int32_t row = coo.row_indices[i];
        int32_t pos = row_ptr_copy[row]++;
        csr.col_indices[pos] = coo.col_indices[i];
        csr.values[pos] = coo.values[i];
    }

    return csr;
}

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

void spmv_cpu(const MtxCsr& csr, const float* x, float* y) {
    for (int32_t row = 0; row < csr.num_rows; ++row) {
        float sum = 0.0f;
        for (int32_t idx = csr.row_ptr[row]; idx < csr.row_ptr[row + 1]; ++idx) {
            sum += csr.values[idx] * x[csr.col_indices[idx]];
        }
        y[row] = sum;
    }
}

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