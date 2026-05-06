#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

/// Host-side sparse matrix in Coordinate (COO) format
struct MtxCoo {
    int32_t num_rows;
    int32_t num_cols;
    int32_t num_nonzeros;
    std::vector<int32_t> row_indices;  // 0-indexed
    std::vector<int32_t> col_indices;  // 0-indexed
    std::vector<float> values;
};

/// Host-side sparse matrix in Compressed Sparse Row (CSR) format
struct MtxCsr {
    int32_t num_rows;
    int32_t num_cols;
    int32_t num_nonzeros;
    std::vector<int32_t> row_ptr;       // size = num_rows + 1
    std::vector<int32_t> col_indices;   // 0-indexed
    std::vector<float> values;
};

/// GPU-side sparse matrix in CSR format
struct DeviceMatrix {
    int32_t num_rows;
    int32_t num_cols;
    int32_t num_nonzeros;
    int32_t* d_row_ptr;     // device memory
    int32_t* d_col_indices; // device memory
    float* d_values;        // device memory
};

/// Parse MTX file into COO format (host)
MtxCoo parse_mtx(const std::string& filepath);

/// Parse MTX file directly into CSR format (host)
MtxCsr parse_mtx_csr(const std::string& filepath);

/// Explicit COO -> CSR conversion
MtxCsr coo_to_csr(const MtxCoo& coo);

/// Explicit CSR -> COO conversion
MtxCoo csr_to_coo(const MtxCsr& csr);

/// CPU SpMV reference implementations
void spmv_cpu(const MtxCsr& csr, const float* x, float* y);
void spmv_cpu(const MtxCoo& coo, const float* x, float* y);

/// Parse MTX file directly into GPU device memory
void parse_mtx_gpu(const std::string& filepath, DeviceMatrix* result);

/// Free GPU device memory
void free_gpu(DeviceMatrix* mat);