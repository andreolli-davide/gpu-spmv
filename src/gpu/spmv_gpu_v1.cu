// =============================================================================
// spmv_gpu_v1.cu
// =============================================================================
// Baseline GPU SpMV kernel — straightforward CSR row-parallel implementation.
//
// Each thread handles one row: iterates over its non-zero entries and
// accumulates the dot product.  This is the foundation for more sophisticated
// GPU kernels that explore shared memory, load balancing, and custom formats.
// =============================================================================

#include "spmv_gpu.h"
#include <cuda_runtime.h>

namespace spmv {

namespace {

// -----------------------------------------------------------------------------
// spmv_kernel — one thread per row
// -----------------------------------------------------------------------------
// Each thread computes y[row] = sum(values[j] * x[col_index[j]]) for all
// non-zero entries j in row.
//
// @param values    CSR values array, size nnz
// @param col_index CSR column index array, size nnz
// @param row_ptr   CSR row pointer array, size rows+1
// @param x         Input dense vector, size cols
// @param y         Output dense vector, size rows
// @param rows      Number of rows in the matrix
// -----------------------------------------------------------------------------
__global__ void spmv_kernel(const double* __restrict__ values,
                            const int64_t* __restrict__ col_index,
                            const int64_t* __restrict__ row_ptr,
                            const double* __restrict__ x,
                            double* __restrict__ y,
                            int64_t rows) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        for (int64_t j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            sum += values[j] * x[col_index[j]];
        }
        y[row] = sum;
    }
}

} // anonymous namespace

// =============================================================================
// spmv_gpu_v1 — public API
// =============================================================================
bool spmv_gpu_v1(const SparseMatrix& A, const DenseVector& x, DenseVector& y) {
    if (A.rows == 0) {
        y.resize(0);
        return true;
    }

    y.resize(A.rows);

    if (A.nnz == 0) {
        return true;
    }

    double* d_values     = nullptr;
    int64_t* d_col_index = nullptr;
    int64_t* d_row_ptr   = nullptr;
    double* d_x         = nullptr;
    double* d_y         = nullptr;
    cudaError_t err = cudaSuccess;

    const int threads_per_block = 256;
    const int blocks = static_cast<int>((A.rows + threads_per_block - 1) / threads_per_block);

    err = cudaMalloc(&d_values,     static_cast<size_t>(A.nnz)  * sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_col_index,  static_cast<size_t>(A.nnz)  * sizeof(int64_t));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_row_ptr,    static_cast<size_t>(A.rows + 1) * sizeof(int64_t));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_x,          static_cast<size_t>(A.cols) * sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_y,          static_cast<size_t>(A.rows) * sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_values,    A.values.data(),     static_cast<size_t>(A.nnz)  * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_col_index, A.col_index.data(),  static_cast<size_t>(A.nnz)  * sizeof(int64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_row_ptr,  A.row_ptr.data(),    static_cast<size_t>(A.rows + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_x,        x.data.data(),       static_cast<size_t>(A.cols) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    spmv_kernel<<<blocks, threads_per_block>>>(d_values, d_col_index, d_row_ptr, d_x, d_y, A.rows);

    // Synchronize to ensure kernel completes before checking errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;

    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(y.data.data(), d_y, static_cast<size_t>(A.rows) * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;

cleanup:
    if (d_values)     cudaFree(d_values);
    if (d_col_index)  cudaFree(d_col_index);
    if (d_row_ptr)    cudaFree(d_row_ptr);
    if (d_x)          cudaFree(d_x);
    if (d_y)          cudaFree(d_y);

    return (err == cudaSuccess);
}

} // namespace spmv