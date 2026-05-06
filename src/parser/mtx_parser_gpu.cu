#include "parser/mtx_parser.h"
#include "parser/mtx_parser_gpu.cuh"
#include <cuda_runtime.h>
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

void parse_mtx_gpu(const std::string& filepath, DeviceMatrix* result) {
    // Parse to host COO first
    MtxCoo coo = parse_mtx(filepath);
    // Convert to CSR
    MtxCsr csr = coo_to_csr(coo);
    // Allocate GPU memory
    cudaMalloc(&result->d_row_ptr,     (csr.num_rows + 1) * sizeof(int32_t));
    cudaMalloc(&result->d_col_indices, csr.num_nonzeros * sizeof(int32_t));
    cudaMalloc(&result->d_values,     csr.num_nonzeros * sizeof(float));
    // Copy to device
    cudaMemcpy(result->d_row_ptr,     csr.row_ptr.data(),     (csr.num_rows + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result->d_col_indices, csr.col_indices.data(), csr.num_nonzeros * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result->d_values,      csr.values.data(),      csr.num_nonzeros * sizeof(float),     cudaMemcpyHostToDevice);
    result->num_rows     = csr.num_rows;
    result->num_cols     = csr.num_cols;
    result->num_nonzeros = csr.num_nonzeros;
}

void free_gpu(DeviceMatrix* mat) {
    if (mat->d_row_ptr)     cudaFree(mat->d_row_ptr);
    if (mat->d_col_indices) cudaFree(mat->d_col_indices);
    if (mat->d_values)     cudaFree(mat->d_values);
    mat->d_row_ptr     = nullptr;
    mat->d_col_indices = nullptr;
    mat->d_values     = nullptr;
}

__global__
void spmv_gpu_kernel_impl(const int32_t* row_ptr,
                          const int32_t* col_indices,
                          const float*   values,
                          int32_t        num_rows,
                          const float*   d_x,
                          float*         d_y) {
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float sum = 0.0f;
    for (int32_t idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        sum += values[idx] * d_x[col_indices[idx]];
    }
    d_y[row] = sum;
}

void spmv_gpu_kernel(const DeviceMatrix& csr, const float* d_x, float* d_y) {
    int32_t num_threads = csr.num_rows;
    int32_t block_size = 256;
    int32_t num_blocks = (num_threads + block_size - 1) / block_size;
    spmv_gpu_kernel_impl<<<num_blocks, block_size>>>(
        csr.d_row_ptr, csr.d_col_indices, csr.d_values,
        csr.num_rows, d_x, d_y);
}
